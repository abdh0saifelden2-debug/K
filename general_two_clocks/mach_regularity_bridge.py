r"""Mach -> 0 singular limit: the *installation* of the nonlocal pressure Hessian
(the bridge between REPORT_NS.md and REPORT_REGULARITY.md).

No external data; CPU only.  This module closes the loop the regularity report
(`REPORT_REGULARITY.md`, sec.5) explicitly leaves open: how the **nonlocal,
anisotropic pressure Hessian** -- the object that carries regularity for the
velocity-gradient tensor -- *emerges* from a genuinely compressible flow as the
Mach number M = U/c -> 0.

The two existing pieces
-----------------------
* `REPORT_REGULARITY.md` (VGT / restricted Euler).  Discarding the nonlocal
  (anisotropic) part of the pressure Hessian and keeping only the LOCAL isotropic
  part P = (1/d) tr(P) I = -(1/d) tr(A^2) I is the restricted-Euler model, and it
  blows up in finite time.  The regularizing structure is exactly the anisotropic
  part the elliptic Poisson solve supplies nonlocally.

* `REPORT_NS.md` (nonlinear compressible DNS).  In the compressible equations the
  pressure is a LOCAL algebraic function of density, p = c^2 rho (isothermal EOS):
  its Hessian P_ij = c^2 d_i d_j rho is whatever the density field happens to be --
  there is NO elliptic constraint tying it to the velocity field.  As M -> 0 the
  acoustic clock (tau_p ~ L/c) collapses onto the advective clock (tau_adv ~ L/U)
  and the pressure becomes the instantaneous global Poisson field.

The bridge measured here
------------------------
Run the fully nonlinear 2D isothermal compressible Navier-Stokes solver
(`compressible/ns.py`) over a Mach sweep and measure, for the acoustic-averaged
pressure of each run, how the *pressure Hessian* approaches the incompressible
(elliptic) one:

  P_comp  = Hess( <p_comp> ),     p_comp = c^2 (rho - <rho>)   (LOCAL EOS Hessian)
  P_inc   = Hess( <p_inc> ),      lap p_inc = -rho0 tr(A^2)    (NONLOCAL elliptic)

Three residuals, all -> 0 as M -> 0 (with a power-law rate, ~M^2 for well-prepared
acoustic-averaged data):

  1. field      ||<p_comp> - <p_inc>|| / ||<p_inc>||           (reproduces REPORT_NS)
  2. Hessian    ||P_comp - P_inc||_F / ||P_inc||_F             (the full
                  regularity-bearing tensor).  By Parseval this L2 norm equals the
                  Poisson *constraint* residual ||lap<p_comp> + rho0<tr A^2>|| /
                  ||rho0<tr A^2>|| -- i.e. "the constraint tr(P) = -tr(A^2) emerges"
                  and "the pressure Hessian converges" are one and the same
                  statement (verified to round-off in the test suite).

and the anisotropy fraction of P_comp converges to that of the incompressible
limit P_inc.  That incompressible anisotropy fraction is *exactly* the part the
restricted-Euler isotropic truncation discards (sec.2 of REPORT_REGULARITY): so
the M -> 0 singular limit installs precisely the nonlocal/anisotropic structure
whose absence causes the restricted-Euler blowup.

Honest scope (carried from the two parent reports)
--------------------------------------------------
This is a 2D *demonstration* solver.  It shows the MECHANISM -- the local EOS
pressure Hessian becoming the nonlocal elliptic one as M -> 0 -- not a proof that
3D Navier-Stokes regularity survives the singular limit at long times (the genuinely
open problem named in REPORT_REGULARITY sec.5).
"""
from __future__ import annotations

import sys

import numpy as np

from compressible.ns import (
    IsothermalCompressibleNS,
    NSState,
    helmholtz,
    incompressible_pressure,
    taylor_green,
)

U0 = 1.0            # characteristic velocity
RHO0 = 1.0          # background density
MU = 0.01           # viscosity (Re = U0 * 2pi / mu ~ 600)
T_ADV = 2.0 * np.pi / U0   # advective time L/U (box is [0,2pi)^2)


# --------------------------------------------------------------------------- #
# spectral Hessian + anisotropy (matched to restricted_euler_regularity.py)
# --------------------------------------------------------------------------- #
def spectral_hessian(sp, f):
    """Pressure Hessian components (P_xx, P_xy, P_yy) of a scalar field f, via
    P_ij = d_i d_j f  ->  -k_i k_j f_hat (full spectral accuracy)."""
    fh = sp.fft(f)
    pxx = sp.ifft(-sp.kx * sp.kx * fh)
    pxy = sp.ifft(-sp.kx * sp.ky * fh)
    pyy = sp.ifft(-sp.ky * sp.ky * fh)
    return pxx, pxy, pyy


def _fro(pxx, pxy, pyy):
    """Pointwise Frobenius norm of the symmetric 2x2 Hessian (P_xy = P_yx)."""
    return np.sqrt(pxx * pxx + 2.0 * pxy * pxy + pyy * pyy)


def anisotropy_fraction(pxx, pxy, pyy):
    """Fraction of the pressure Hessian carried by its anisotropic (deviatoric) part
    -- the part restricted Euler discards.  Same construction as
    restricted_euler_regularity.nonlocal_hessian_anisotropy, in d=2:
        iso = (tr P / 2) I,  aniso = P - iso,
        frac = <||aniso||_F> / <||P||_F>,   aniso/iso = <||aniso||> / <||iso||>.
    """
    tr = pxx + pyy
    iso = 0.5 * tr                                   # diagonal value of the iso part
    axx, ayy, axy = pxx - iso, pyy - iso, pxy
    fro_aniso = np.sqrt(axx * axx + 2.0 * axy * axy + ayy * ayy)
    fro_iso = np.sqrt(2.0) * np.abs(iso)             # ||(iso) I||_F = sqrt(2)|iso|
    fro_P = _fro(pxx, pxy, pyy)
    frac = float(np.mean(fro_aniso) / (np.mean(fro_P) + 1e-30))
    aoi = float(np.mean(fro_aniso) / (np.mean(fro_iso) + 1e-30))
    return frac, aoi


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# --------------------------------------------------------------------------- #
# one compressible DNS run at a given Mach number; acoustic-averaged diagnostics
# --------------------------------------------------------------------------- #
def run_mach(n, M, t_end_factor=1.0, cfl=0.3, avg_from=0.5):
    r"""Run the isothermal compressible NS solver at Mach M = U0/c from a Taylor-Green
    start, acoustic-average pressure / tr(A^2) / Helmholtz energies over the
    quasi-steady second half, and return the pressure-Hessian diagnostics.

    The uniform-density start launches a persistent standing acoustic wave, so
    instantaneous snapshots are polluted; averaging over the fast acoustic clock
    (many acoustic periods fit in one advective time at low M) recovers the slow,
    balanced field -- the same methodology as run_compressible_ns.run_one.
    """
    c = U0 / M
    solver = IsothermalCompressibleNS(n, c, MU, RHO0, cfl=cfl)
    sp = solver.sp
    rho, mx, my = taylor_green(sp, U0, RHO0)
    st = NSState(rho, mx, my, 0.0)
    t_end = t_end_factor * T_ADV

    pcomp_sum = np.zeros((n, n))
    pinc_sum = np.zeros((n, n))
    trA2_sum = np.zeros((n, n))
    keS_sum = keD_sum = dt_sum = 0.0

    while st.t < t_end:
        dt = solver.dt_cfl(st)
        if st.t >= avg_from * t_end:
            u = st.mx / st.rho
            v = st.my / st.rho
            p_comp = c * c * st.rho
            pcomp_sum += (p_comp - p_comp.mean()) * dt
            p_inc = incompressible_pressure(sp, u, v, RHO0)
            pinc_sum += (p_inc - p_inc.mean()) * dt
            ux, uy = sp.ddx(u), sp.ddy(u)
            vx, vy = sp.ddx(v), sp.ddy(v)
            trA2_sum += (ux * ux + 2.0 * uy * vx + vy * vy) * dt   # tr(A^2)
            us, vs, ud, vd = helmholtz(sp, u, v)
            keS_sum += 0.5 * float(np.mean(st.rho * (us * us + vs * vs))) * dt
            keD_sum += 0.5 * float(np.mean(st.rho * (ud * ud + vd * vd))) * dt
            dt_sum += dt
        solver.step(st, dt)
        if not np.isfinite(st.rho).all() or st.rho.min() <= 0:
            raise RuntimeError(f"compressible solver diverged at t={st.t} (M={M})")

    dt_sum = max(dt_sum, 1e-30)
    p_comp = pcomp_sum / dt_sum
    p_inc = pinc_sum / dt_sum
    trA2 = trA2_sum / dt_sum
    trA2 = trA2 - trA2.mean()                        # zero-mean (matches lap of p)

    # pressure Hessians (regularity-bearing tensor)
    Cxx, Cxy, Cyy = spectral_hessian(sp, p_comp)     # LOCAL EOS Hessian, c^2 Hess(rho)
    Ixx, Ixy, Iyy = spectral_hessian(sp, p_inc)      # NONLOCAL elliptic Hessian

    # residuals (all -> 0 as M -> 0)
    field_resid = _rel(p_comp, p_inc)
    trP_comp = Cxx + Cyy                             # = lap<p_comp>
    trP_inc_target = -RHO0 * trA2                    # = lap<p_inc> = -rho0 tr(A^2)
    constraint_resid = _rel(trP_comp, trP_inc_target)
    # internal check: the incompressible reference satisfies the Poisson constraint
    inc_constraint_check = _rel(Ixx + Iyy, trP_inc_target)
    hess_resid = float(np.linalg.norm(_fro(Cxx - Ixx, Cxy - Ixy, Cyy - Iyy))
                       / (np.linalg.norm(_fro(Ixx, Ixy, Iyy)) + 1e-30))

    aniso_comp, aoi_comp = anisotropy_fraction(Cxx, Cxy, Cyy)
    aniso_inc, aoi_inc = anisotropy_fraction(Ixx, Ixy, Iyy)
    ke_ratio = float(keD_sum / (keS_sum + 1e-30))    # acoustic / vortical energy

    return dict(
        M=float(M), c=float(c), n=int(n),
        field_resid=field_resid,
        constraint_resid=constraint_resid,
        hess_resid=hess_resid,
        inc_constraint_check=inc_constraint_check,
        aniso_comp=aniso_comp, aniso_inc=aniso_inc,
        aniso_over_iso_inc=aoi_inc,
        aniso_gap=float(abs(aniso_comp - aniso_inc)),
        ke_ratio=ke_ratio,
    )


def _powerlaw(machs, vals):
    """Least-squares log-log slope (exponent p in resid ~ M^p) over positive data."""
    m = np.asarray(machs, float)
    y = np.asarray(vals, float)
    good = (m > 0) & (y > 0)
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(m[good]), np.log(y[good]), 1)[0])


# --------------------------------------------------------------------------- #
# Mach sweep
# --------------------------------------------------------------------------- #
def mach_sweep(n=64, machs=(0.4, 0.2, 0.1, 0.05), t_end_factor=1.0, cfl=0.3):
    rows = []
    for M in machs:
        try:
            rows.append(run_mach(n, M, t_end_factor=t_end_factor, cfl=cfl))
        except RuntimeError as e:                    # skip a diverged (too-stiff) point
            print(f"  (skipped M={M}: {e})")
    rows.sort(key=lambda r: -r["M"])                 # high -> low Mach
    machs_ok = [r["M"] for r in rows]
    field = [r["field_resid"] for r in rows]
    constraint = [r["constraint_resid"] for r in rows]
    hess = [r["hess_resid"] for r in rows]
    aniso_gap = [r["aniso_gap"] for r in rows]
    p_field = _powerlaw(machs_ok, field)
    p_constraint = _powerlaw(machs_ok, constraint)
    p_hess = _powerlaw(machs_ok, hess)
    # monotone decrease toward the incompressible limit (robust: ends, not strict)
    decreasing = bool(rows[-1]["field_resid"] < rows[0]["field_resid"]
                      and rows[-1]["constraint_resid"] < rows[0]["constraint_resid"]
                      and rows[-1]["hess_resid"] < rows[0]["hess_resid"])
    aniso_converges = bool(rows[-1]["aniso_gap"] < rows[0]["aniso_gap"])
    inc_ok = bool(max(r["inc_constraint_check"] for r in rows) < 1e-6)
    ok = bool(decreasing and aniso_converges and inc_ok and len(rows) >= 3)
    return dict(rows=rows, machs=machs_ok,
                field=field, constraint=constraint, hess=hess, aniso_gap=aniso_gap,
                p_field=p_field, p_constraint=p_constraint, p_hess=p_hess,
                aniso_inc=rows[-1]["aniso_inc"],
                aniso_over_iso_inc=rows[-1]["aniso_over_iso_inc"],
                decreasing=decreasing, aniso_converges=aniso_converges,
                inc_constraint_ok=inc_ok, ok=ok)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(path, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.asarray(sweep["machs"], float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    ax.loglog(m, sweep["field"], "o-", color="teal", ms=7,
              label=f"pressure field  (~M^{sweep['p_field']:.2f})")
    ax.loglog(m, sweep["hess"], "^-", color="#c0392b", ms=7,
              label=("pressure Hessian ||P_comp-P_inc|| = Poisson constraint\n"
                     f"tr(P)=-tr(A^2)  (Parseval;  ~M^{sweep['p_hess']:.2f})"))
    ref = sweep["hess"][0] * (m / m[0]) ** 2
    ax.loglog(m, ref, "k--", lw=1.2, label="~M^2 reference")
    ax.set_xlabel("Mach number  M = U / c")
    ax.set_ylabel("relative residual vs incompressible (elliptic) limit")
    ax.set_title("The nonlocal pressure Hessian emerges as M -> 0\n"
                 "(local EOS pressure -> nonlocal elliptic Poisson pressure)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    aniso_comp = [r["aniso_comp"] for r in sweep["rows"]]
    ax.semilogx(m, aniso_comp, "o-", color="darkorange", ms=8,
                label="compressible Hessian anisotropy (measured)")
    ax.axhline(sweep["aniso_inc"], color="navy", lw=1.6, ls="--",
               label=f"incompressible (elliptic) limit = {sweep['aniso_inc']:.2f}")
    ax.fill_between([m.min() * 0.8, m.max() * 1.2], 0, sweep["aniso_inc"],
                    color="navy", alpha=0.06)
    ax.set_xlim(m.min() * 0.8, m.max() * 1.2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mach number  M = U / c")
    ax.set_ylabel("anisotropic fraction of the pressure Hessian")
    ax.set_title("Anisotropy converges to the incompressible value\n"
                 "(= the part restricted Euler discards -> blowup)")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Mach -> 0 singular limit installs the regularity-bearing nonlocal "
                 "pressure Hessian", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run():
    return dict(sweep=mach_sweep())


def summary():
    r = run()
    r["all_ok"] = bool(r["sweep"]["ok"])
    return r


if __name__ == "__main__":
    import os

    r = summary()
    s = r["sweep"]
    print("Mach -> 0 singular limit: emergence of the nonlocal pressure Hessian")
    print("=" * 70)
    print(f"{'Mach':>6} | {'field':>10} | {'constraint':>11} | {'Hessian':>10} | "
          f"{'aniso_comp':>10} | {'KE_dil/sol':>10}")
    print("-" * 70)
    for row in s["rows"]:
        print(f"{row['M']:6.3f} | {row['field_resid']:10.3e} | "
              f"{row['constraint_resid']:11.3e} | {row['hess_resid']:10.3e} | "
              f"{row['aniso_comp']:10.3f} | {row['ke_ratio']:10.3e}")
    print("-" * 70)
    print(f"power-law exponents (resid ~ M^p):  field={s['p_field']:.2f}  "
          f"Hessian={s['p_hess']:.2f}")
    dmax = max(abs(rr['hess_resid'] - rr['constraint_resid'])
              / (rr['constraint_resid'] + 1e-30) for rr in s['rows'])
    print(f"Parseval identity  ||Hess||_2 == ||Poisson constraint||_2  to "
          f"{dmax:.1e} (relative)")
    print(f"incompressible-limit anisotropy fraction = {s['aniso_inc']:.3f} "
          f"(aniso/iso = {s['aniso_over_iso_inc']:.2f})  "
          f"-- the part restricted Euler discards")
    print(f"max incompressible Poisson-constraint check = "
          f"{max(rr['inc_constraint_check'] for rr in s['rows']):.2e}  (should be ~0)")

    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "figures", "66_mach_regularity_bridge.png")
    try:
        make_figure(fig_path, s)
        print(f"\nfigure -> {fig_path}")
    except Exception as e:
        print(f"\n(figure skipped: {e})")

    print("\n" + "=" * 70)
    print("ALL VERIFIED" if r["all_ok"] else "SOME FAILED")
    sys.exit(0 if r["all_ok"] else 1)
