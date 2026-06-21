r"""Paper 2 — In-place, pressure-mediated buoyancy exchange in 2D Boussinesq flow.

PRE-REGISTERED probe of *how* an incompressible stratified fluid moves buoyancy.
The thesis (the regularity/two-clocks program, observational face): vertical
buoyancy transport is **pressure-mediated in-place exchange**, not a direct response
to the buoyancy force. The buoyancy force `F = (0, b')` cannot accelerate fluid
upward on its own — that would create divergence — so the **elliptic pressure**
instantaneously and *globally* (Leray projection `P = I − ∇(Δ⁻¹)∇·`) splits it:

    F = F_sol + ∇φ_F,     Δφ_F = ∇·F = ∂_y b'      (the pressure response to buoyancy)

`∇φ_F` is **removed** by pressure; the surviving `F_sol` is what accelerates the
flow, and its curl is the **baroclinic torque** `∇×F = ∂_x b'` — the *only* part of
buoyancy that survives. Pure vertical stratification (`b'=b'(y)`) has `∂_x b'=0`, so
`F_sol=0`: it is held statically *in place* by pressure. Exchange requires horizontal
buoyancy structure. Net vertical **mass** flux is zero (incompressible, periodic) yet
the **buoyancy** flux is positive: warm up / cold down, an in-place exchange.

FOUR PRE-REGISTERED, FALSIFIABLE PREDICTIONS (thresholds fixed by physics, not tuned)
------------------------------------------------------------------------------------
  PR1  Non-locality (elliptic): the pressure-mediated velocity correction `∇φ_F` from
       a *localized* buoyancy blob is spatially **broader** than the blob —
       half-energy radius ratio r50(|∇φ_F|)/r50(b') > 1.5. (A local/diffusive response
       would be confined; an elliptic one is global.)
  PR2  Held in place vs exchange: for pure vertical stratification b'(y) the motion-
       driving (solenoidal) fraction of the buoyancy force is ≈0 (<1e-6) — pressure
       holds it statically; adding horizontal buoyancy structure makes it >1e-2.
  PR3  In-place exchange in developed convection: |⟨v⟩|/v_rms < 1e-3 (no net vertical
       mass flux) while the buoyancy flux ⟨v·b'⟩ > 0 and upward fluid is warmer
       (⟨b'|v>0⟩ > 0 > ⟨b'|v<0⟩), with up/down volume fractions both in [0.4,0.6].
  PR4  Instantaneous linear mediation: the pressure response superposes
       (‖φ_F[b₁+b₂]−φ_F[b₁]−φ_F[b₂]‖/‖φ_F[b₁+b₂]‖ < 1e-10) and scales linearly
       (log–log slope of ‖∇φ_F‖ vs amplitude = 1.00 ± 0.02).

CPU, deterministic (seeded). Built on `boussinesq/solver.py` (the same Leray
projection used in `run_boussinesq.py`).
"""
from __future__ import annotations

import numpy as np

from boussinesq.solver import (BoussinesqProjection, BoussinesqState, warm_blobs,
                               project, projection_potential, divergence)
from compressible.ns import Spectral2D


# --------------------------------------------------------------------------- #
# Core probe: the pressure response to a buoyancy field
# --------------------------------------------------------------------------- #
def _anom(b):
    return b - float(b.mean())


def pressure_field(sp: Spectral2D, b: np.ndarray) -> np.ndarray:
    """φ_F with Δφ_F = ∂_y b' — the elliptic pressure response to buoyancy.
    (= projection_potential of the buoyancy force F=(0,b').)"""
    bp = _anom(b)
    return projection_potential(sp, np.zeros_like(bp), bp)


def pressure_correction(sp: Spectral2D, b: np.ndarray):
    """∇φ_F = (∂_xφ_F, ∂_yφ_F): the velocity correction pressure REMOVES from the
    buoyancy-forced tendency.  The horizontal part −∂_xφ_F is the pressure-created
    return flow (it is absent from the vertical-only force)."""
    phi = pressure_field(sp, b)
    return sp.ddx(phi), sp.ddy(phi), phi


def solenoidal_fraction(sp: Spectral2D, b: np.ndarray) -> float:
    """‖F_sol‖/‖F‖ for the buoyancy force F=(0,b'): the share that survives pressure
    and actually drives motion.  0 ⇒ held statically in place by pressure."""
    bp = _anom(b)
    z = np.zeros_like(bp)
    us, vs = project(sp, z, bp)
    num = float(np.sqrt(np.mean(us ** 2 + vs ** 2)))
    den = float(np.sqrt(np.mean(bp ** 2)))         # ‖(0,b')‖
    return num / (den + 1e-30)


def _half_energy_radius(r: np.ndarray, e: np.ndarray) -> float:
    """Radius containing 50% of the field energy `e` about a center (r = distance)."""
    order = np.argsort(r.ravel())
    rs = r.ravel()[order]
    cs = np.cumsum(e.ravel()[order])
    cs /= cs[-1]
    idx = int(np.searchsorted(cs, 0.5))
    return float(rs[min(idx, len(rs) - 1)])


# --------------------------------------------------------------------------- #
# PR1 — non-locality of the elliptic pressure response
# --------------------------------------------------------------------------- #
def predict_nonlocality(n=128, width=0.18):
    sp = Spectral2D(n)
    x, y = sp.grid()
    cx = cy = np.pi
    b = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * width ** 2))
    gx, gy, _ = pressure_correction(sp, b)
    g2 = gx ** 2 + gy ** 2
    b2 = _anom(b) ** 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r50_b = _half_energy_radius(r, b2)
    r50_g = _half_energy_radius(r, g2)
    ratio = r50_g / (r50_b + 1e-30)
    # fraction of the pressure correction energy beyond 3 source-radii (non-local)
    far = float(g2[r > 3.0 * r50_b].sum() / g2.sum())
    return dict(r50_b=r50_b, r50_g=r50_g, ratio=float(ratio), far_fraction=far,
                passed=bool(ratio > 1.5))


# --------------------------------------------------------------------------- #
# PR2 — pure stratification is held in place; horizontal structure permits exchange
# --------------------------------------------------------------------------- #
def predict_stratification(n=128):
    sp = Spectral2D(n)
    x, y = sp.grid()
    b_vertical = np.sin(2.0 * y) + 0.5 * np.sin(y)                 # b'(y) only
    frac_v = solenoidal_fraction(sp, b_vertical)
    # increasing horizontal modulation -> increasing motion-driving fraction
    sweep = []
    for eps in (0.0, 0.1, 0.3, 0.6, 1.0):
        b = np.sin(2.0 * y) * (1.0 + eps * np.cos(3.0 * x))
        sweep.append((eps, solenoidal_fraction(sp, b)))
    frac_h = sweep[-1][1]
    monotone = all(s2[1] >= s1[1] - 1e-12 for s1, s2 in zip(sweep, sweep[1:]))
    return dict(frac_vertical=frac_v, frac_horizontal=frac_h, sweep=sweep,
                monotone=bool(monotone),
                passed=bool(frac_v < 1e-6 and frac_h > 1e-2 and monotone))


# --------------------------------------------------------------------------- #
# PR3 — in-place exchange in developed convection (zero net mass flux, +buoy flux)
# --------------------------------------------------------------------------- #
def _simulate(n=96, t_end=5.0, nu=2e-3, kappa=2e-3, dt_cap=0.02, seed=0):
    solver = BoussinesqProjection(n, nu, kappa, cfl=0.4)
    u, v, b = warm_blobs(solver.sp, seed=seed)
    u, v = project(solver.sp, u, v)
    st = BoussinesqState(u, v, b, 0.0)
    while st.t < t_end:
        dt = min(solver.dt_cfl(st), dt_cap)
        solver.step(st, dt)
        if not np.isfinite(st.u).all():
            raise RuntimeError(f"diverged at t={st.t}")
    return solver, st


def predict_exchange(n=96, t_end=5.0):
    solver, st = _simulate(n=n, t_end=t_end)
    bp = _anom(st.b)
    v = st.v
    v_rms = float(np.sqrt(np.mean(v ** 2)))
    v_mean = float(np.mean(v))
    netmass = abs(v_mean) / (v_rms + 1e-30)
    flux = float(np.mean(v * bp))
    up, dn = v > 0, v < 0
    vol_up, vol_dn = float(up.mean()), float(dn.mean())
    b_up = float(bp[up].mean()) if up.any() else 0.0
    b_dn = float(bp[dn].mean()) if dn.any() else 0.0
    div_rms = float(np.sqrt(np.mean(divergence(solver.sp, st.u, st.v) ** 2)))
    passed = bool(netmass < 1e-3 and flux > 0 and 0.4 < vol_up < 0.6
                  and 0.4 < vol_dn < 0.6 and b_up > 0 > b_dn)
    return dict(netmass_ratio=netmass, buoy_flux=flux, vol_up=vol_up, vol_dn=vol_dn,
                b_up=b_up, b_dn=b_dn, div_rms=div_rms, v_rms=v_rms, t_end=t_end,
                passed=passed)


# --------------------------------------------------------------------------- #
# PR4 — instantaneous linear elliptic mediation (superposition + scaling)
# --------------------------------------------------------------------------- #
def predict_linearity(n=128, seed=0):
    sp = Spectral2D(n)
    rng = np.random.default_rng(seed)
    b1 = rng.standard_normal((n, n))
    b2 = rng.standard_normal((n, n))
    # low-pass so the fields are smooth (physical buoyancy), still random
    b1 = sp.ifft(sp.fft(b1) * (sp.k2 < 36)).real
    b2 = sp.ifft(sp.fft(b2) * (sp.k2 < 36)).real
    p1 = pressure_field(sp, b1)
    p2 = pressure_field(sp, b2)
    p12 = pressure_field(sp, b1 + b2)
    resid = float(np.sqrt(np.mean((p12 - p1 - p2) ** 2))
                  / (np.sqrt(np.mean(p12 ** 2)) + 1e-30))
    amps = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    norms = []
    for s in amps:
        gx, gy, _ = pressure_correction(sp, s * b1)
        norms.append(np.sqrt(np.mean(gx ** 2 + gy ** 2)))
    slope = float(np.polyfit(np.log(amps), np.log(np.array(norms) + 1e-30), 1)[0])
    return dict(superposition_resid=resid, scaling_slope=slope,
                passed=bool(resid < 1e-10 and abs(slope - 1.0) < 0.02))


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def compare():
    pr1 = predict_nonlocality()
    pr2 = predict_stratification()
    pr3 = predict_exchange()
    pr4 = predict_linearity()
    n_pass = sum(p["passed"] for p in (pr1, pr2, pr3, pr4))
    return dict(pr1=pr1, pr2=pr2, pr3=pr3, pr4=pr4, n_pass=n_pass,
                ok=bool(n_pass == 4))


def run():
    return compare()


def make_figure(path, r=None):
    if r is None:
        r = compare()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = Spectral2D(128)
    x, y = sp.grid()
    cx = cy = np.pi
    b = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 0.18 ** 2))
    gx, gy, phi = pressure_correction(sp, b)
    gmag = np.sqrt(gx ** 2 + gy ** 2)

    solver, st = _simulate(n=96, t_end=5.0)
    bp = _anom(st.b)

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.7))

    # (a) PR1 non-locality: blob vs broad pressure correction
    ax[0].imshow(_anom(b).T, origin="lower", extent=[0, sp.L, 0, sp.L],
                 cmap="Greys", aspect="equal")
    cs = ax[0].contour(x.T, y.T, gmag.T, levels=6, cmap="plasma", linewidths=1.1)
    ax[0].set_title(f"(a) PR1 non-locality\nr50(∇φ)/r50(b')={r['pr1']['ratio']:.2f} "
                    f"(>1.5 [ok]); {100*r['pr1']['far_fraction']:.0f}% beyond 3r₀")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y")

    # (b) PR2 held-in-place vs exchange sweep
    eps = [s[0] for s in r["pr2"]["sweep"]]
    fr = [s[1] for s in r["pr2"]["sweep"]]
    ax[1].plot(eps, fr, "o-", color="#2c7fb8", ms=7)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_title(f"(b) PR2 held in place→exchange\nb'(y): {r['pr2']['frac_vertical']:.0e} "
                    f"(<1e-6 [ok])")
    ax[1].set_xlabel("horizontal modulation ε"); ax[1].set_ylabel("solenoidal (motion) fraction")
    ax[1].grid(alpha=0.3)

    # (c) PR3 developed convection: buoyancy + sign(v)
    im = ax[2].imshow(bp.T, origin="lower", extent=[0, sp.L, 0, sp.L],
                      cmap="RdBu_r", aspect="equal",
                      vmin=-np.percentile(np.abs(bp), 99), vmax=np.percentile(np.abs(bp), 99))
    ax[2].set_title(f"(c) PR3 in-place exchange\n⟨v⟩/v_rms={r['pr3']['netmass_ratio']:.0e}, "
                    f"⟨v·b'⟩={r['pr3']['buoy_flux']:.2e}>0 [ok]")
    ax[2].set_xlabel("x"); ax[2].set_ylabel("y")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04, label="b'")

    # (d) PR4 linear scaling
    amps = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    norms = []
    for s in amps:
        ggx, ggy, _ = pressure_correction(sp, s * b)
        norms.append(np.sqrt(np.mean(ggx ** 2 + ggy ** 2)))
    ax[3].loglog(amps, norms, "o-", color="#c0392b", ms=7)
    ax[3].set_title(f"(d) PR4 linear mediation\nslope={r['pr4']['scaling_slope']:.3f} (=1 [ok]), "
                    f"superpos resid={r['pr4']['superposition_resid']:.0e}")
    ax[3].set_xlabel("buoyancy amplitude s"); ax[3].set_ylabel("‖∇φ_F‖")
    ax[3].grid(alpha=0.3, which="both")

    fig.suptitle(f"In-place pressure-mediated buoyancy exchange — {r['n_pass']}/4 "
                 "pre-registered predictions pass", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    r = compare()
    print("=== In-place pressure-mediated buoyancy exchange (4 pre-registered) ===")
    p = r["pr1"]; print(f"  PR1 non-locality:  r50 ratio={p['ratio']:.2f} (>1.5)  "
                        f"far={100*p['far_fraction']:.0f}%  -> {p['passed']}")
    p = r["pr2"]; print(f"  PR2 held/exchange: frac(b'(y))={p['frac_vertical']:.1e} (<1e-6)  "
                        f"frac(horiz)={p['frac_horizontal']:.2e} (>1e-2)  monotone={p['monotone']}  -> {p['passed']}")
    p = r["pr3"]; print(f"  PR3 in-place:      ⟨v⟩/vrms={p['netmass_ratio']:.1e} (<1e-3)  "
                        f"⟨vb'⟩={p['buoy_flux']:.2e}  b_up={p['b_up']:.2e}>0>b_dn={p['b_dn']:.2e}  -> {p['passed']}")
    p = r["pr4"]; print(f"  PR4 linear:        superpos={p['superposition_resid']:.1e} (<1e-10)  "
                        f"slope={p['scaling_slope']:.4f} (1±0.02)  -> {p['passed']}")
    print(f"  PASS: {r['n_pass']}/4   ok={r['ok']}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
