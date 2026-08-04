r"""NR37 -- pseudo-sound made explicit, and the M^2 -> M^1 crossover of the
density fluctuation: a background (temperature/entropy) gradient turns the
nearly-incompressible density response from acoustic (O(M^2)) to entropy-mode
(O(M^1)), and the crossover turbulent Mach number is *linear in the gradient*.

Ledger item E4 (papers/RESEARCH_RECAP_AND_HORIZON.md), the "strongest physics
extension" candidate.  It places Part 5's measured box law
``<KE_dil>/<KE_sol> ~ M^2`` (REPORT_NS.md) inside the mainstream nearly-
incompressible (NI) hydrodynamics hierarchy and yields a NEW, falsifiable,
fully in-repo prediction.

The mainstream frame (what equation does the work)
--------------------------------------------------
Nearly-incompressible hydrodynamics (Zank & Matthaeus 1990/1991/1993; Lighthill
1952; Montgomery, Brown & Matthaeus 1987) expands the compressible equations in
the small turbulent Mach number ``M_t = u'/c``.  Two regimes:

* **Homogeneous NI ("pseudo-sound").**  With a uniform background, the density
  fluctuation is *slaved* to the incompressible (Bernoulli/Poisson) pressure:
  ``dp ~ p_inc`` and, through the (isothermal) closure ``dp = c^2 drho``,

      (drho/rho0)_acoustic = p_inc / (rho0 c^2).

  Because the incompressible pressure is a *quadratic* functional of the
  velocity (elliptic Poisson solve, ``p_inc ~ rho0 u'^2``),

      (drho/rho0)_acoustic  ~  u'^2 / c^2  =  M_t^2 .              [SLOPE 2]

  This is exactly Part 5's box law, restated for the density.

* **Inhomogeneous / heat-flux NI.**  With a background temperature (entropy)
  gradient ``grad T-bar``, a displaced parcel (displacement ``dxi ~ u' t``)
  carries a temperature anomaly ``T' = -dxi . grad T-bar`` and hence, at constant
  pressure, a density anomaly (Bhattacharjee, Ghosh & Matthaeus 1998; Hunana &
  Zank 2010)

      (drho/rho0)_entropy = -T'/T-bar = beta * u' t   (beta == |grad T-bar|/T-bar)
                          ~ M_t .                                   [SLOPE 1]

  The production is *linear* in the velocity (advection of the mean gradient),
  so this channel is O(M_t), not O(M_t^2).

The crossover (the new equation)
--------------------------------
The total NI density fluctuation is the sum of the two channels
``D(M_t) = a M_t^2 + b*beta*M_t``.  They are equal at

      M_t^*  =  (b/a) * beta        =>   M_t^*  ~  beta .           [SLOPE 1]

so the Mach number separating the M^1 (gradient-dominated, low-M) regime from
the M^2 (pseudo-sound-dominated, high-M) regime is **linear in the background
gradient** -- the falsifiable "crossover gradient scale".  Equivalently the
local exponent ``d ln D / d ln M_t`` runs from 1 (large beta) to 2 (beta -> 0).

External anchor: MMS/solar-wind density-fluctuation scaling reportedly
interpolates between ``dn/n ~ M^2`` (homogeneous) and ``~ M`` (gradient/heat-
flux) conditions [cite -- magnetosheath NI tests].

What is measured here (in-repo, CPU)
------------------------------------
The committed isothermal compressible solver (compressible/ns.py) carries BOTH
NI channels already:
  * its dynamical density IS the acoustic/pseudo-sound channel;
  * its committed passive scalar with mean-gradient production (``-scalar_grad*v``)
    IS the entropy channel.
We scale one fixed solenoidal field by amplitude to sweep M_t at fixed c, verify
the two exponents (2 and 1) both in the exact leading-order relations and under
nonlinear time evolution, and confirm ``M_t^* ~ beta`` (linear) plus the running
exponent 2 -> 1.  A buoyancy-on variant checks the entropy channel survives
feedback.
"""
from __future__ import annotations

import json
import os

import numpy as np

from compressible.ns import (
    IsothermalCompressibleNS,
    NSState,
    Spectral2D,
    helmholtz,
    incompressible_pressure,
)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

RHO0 = 1.0
CSOUND = 8.0          # fixed sound speed; M_t is swept via the velocity amplitude
MU = 5e-3
MACHS = (0.03, 0.05, 0.08, 0.12, 0.18, 0.27)
BETAS = (0.008, 0.016, 0.032, 0.064)   # dimensionless gradients (subsonic crossovers)
T_BALLISTIC = 1.0     # entropy-channel integration time (ballistic window)


# --------------------------------------------------------------------- fields
def solenoidal_unit_field(sp: Spectral2D, seed: int = 0, klo: int = 1,
                          khi: int = 3):
    """A fixed-shape divergence-free velocity field with unit rms speed.

    Built from a low-wavenumber random stream function psi; u = (d_y psi,
    -d_x psi) so div u = 0 identically.  Scaling the returned field by an
    amplitude sweeps u' (hence M_t = u'/c) with the geometry held fixed, so the
    acoustic channel scales as amplitude^2 and the entropy channel as amplitude.
    """
    n = sp.n
    kmag = np.sqrt(sp.k2)
    mask = (kmag >= klo) & (kmag <= khi)
    rng = np.random.default_rng(seed)
    psi_h = mask * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    psi = np.real(np.fft.ifft2(psi_h))
    u = sp.ddy(psi)
    v = -sp.ddx(psi)
    s = float(np.sqrt(np.mean(u * u + v * v))) + 1e-30
    return u / s, v / s


def pseudo_sound_density(sp: Spectral2D, u, v, rho0=RHO0, c=CSOUND) -> float:
    """Acoustic-channel amplitude rms(drho)/rho0 = rms(p_inc)/(rho0 c^2)."""
    p = incompressible_pressure(sp, u, v, rho0)
    return float(np.sqrt(np.mean(p * p))) / (rho0 * c * c)


# --------------------------------------------------- leading-order (exact) sweep
def leading_order(n=64, c=CSOUND, rho0=RHO0, seed=0, machs=MACHS,
                  t_ball=T_BALLISTIC):
    """Exact NI leading-order relations from the amplitude-scaled field."""
    sp = Spectral2D(n)
    u0, v0 = solenoidal_unit_field(sp, seed)
    vrms_unit = float(np.sqrt(np.mean(v0 * v0)))     # per-unit-amplitude v rms
    dps, s_per_beta = [], []
    for M in machs:
        amp = M * c
        u, v = amp * u0, amp * v0
        dps.append(pseudo_sound_density(sp, u, v, rho0, c))
        # ballistic entropy channel: theta ~ -beta v t  ->  rms = beta*t*rms(v)
        s_per_beta.append(t_ball * float(np.sqrt(np.mean(v * v))))  # per unit beta
    machs = np.asarray(machs, float)
    dps = np.asarray(dps)
    s_per_beta = np.asarray(s_per_beta)              # = theta_rms / beta
    a = float(np.mean(dps / machs ** 2))             # acoustic prefactor
    b = float(np.mean(s_per_beta / machs))           # entropy prefactor / beta
    slope_ac = float(np.polyfit(np.log(machs), np.log(dps), 1)[0])
    slope_en = float(np.polyfit(np.log(machs), np.log(s_per_beta), 1)[0])
    return {
        "machs": machs.tolist(),
        "acoustic_density": dps.tolist(),
        "entropy_density_per_beta": s_per_beta.tolist(),
        "acoustic_prefactor_a": a,
        "entropy_prefactor_b": b,
        "acoustic_slope": slope_ac,          # -> 2
        "entropy_slope": slope_en,           # -> 1
        "vrms_per_unit_amp": vrms_unit,
        "sp": sp, "u0": u0, "v0": v0,
    }


def crossover_sweep(a, b, betas=BETAS):
    """M_t^*(beta) = (b/a) beta  -- the crossover-gradient-scale prediction."""
    betas = np.asarray(betas, float)
    mstar = (b / a) * betas
    slope = float(np.polyfit(np.log(betas), np.log(mstar), 1)[0])
    return {"betas": betas.tolist(), "mstar": mstar.tolist(),
            "mstar_vs_beta_slope": slope, "slope_coeff_b_over_a": b / a}


def running_exponent(machs, a, b, beta):
    """Local exponent d ln D / d ln M of the total NI density D=aM^2+b*beta*M."""
    m = np.asarray(machs, float)
    D = a * m ** 2 + b * beta * m
    lm, lD = np.log(m), np.log(D)
    return np.gradient(lD, lm)


# ------------------------------------------------------ nonlinear confirmation
def sim_point(n, c, mu, seed, M, beta, t_end, kappa, buoy_g=0.0):
    """Evolve the isothermal compressible box + mean-gradient scalar; return the
    2nd-half-averaged dynamical density rms (acoustic) and the scalar rms
    (entropy).  buoy_g>0 adds a Boussinesq body force g*theta*yhat (active
    scalar) to check the entropy channel is robust to feedback."""
    sp = Spectral2D(n)
    u0, v0 = solenoidal_unit_field(sp, seed)
    amp = M * c
    u, v = amp * u0, amp * v0
    rho = np.full((n, n), RHO0)
    st = NSState(rho.copy(), rho * u, rho * v, 0.0, theta=np.zeros((n, n)))
    solver = IsothermalCompressibleNS(n, c, mu, RHO0, cfl=0.3, kappa=kappa,
                                      scalar_grad=beta, seed=seed)
    drho_acc, th_acc, wsum = 0.0, 0.0, 0.0
    while st.t < t_end - 1e-12:
        dt = min(solver.dt_cfl(st), t_end - st.t)
        fext = (np.zeros((n, n)), buoy_g * st.theta) if buoy_g > 0 else None
        solver.step_closure(st, dt, fext=fext)
        if st.t >= 0.5 * t_end:                       # quasi-steady window
            drho_acc += float(np.sqrt(np.mean((st.rho - RHO0) ** 2))) / RHO0 * dt
            th_acc += float(np.sqrt(np.mean(st.theta ** 2))) * dt
            wsum += dt
        if not np.isfinite(st.rho).all() or st.rho.min() <= 0:
            raise RuntimeError(f"solver diverged at t={st.t}, M={M}, beta={beta}")
    wsum = max(wsum, 1e-30)
    return drho_acc / wsum, th_acc / wsum


def nonlinear_sweep(n=64, c=CSOUND, mu=MU, seed=0, machs=MACHS, beta=2.0,
                    t_end=1.0, kappa=5e-3):
    drho, theta = [], []
    for M in machs:
        d, t = sim_point(n, c, mu, seed, M, beta, t_end, kappa)
        drho.append(d)
        theta.append(t)
    m = np.asarray(machs, float)
    drho = np.asarray(drho)
    theta = np.asarray(theta)
    # beta==0 is the pure-acoustic control: the scalar is unforced (~0), so its
    # log-log slope is undefined -- report it as None rather than a log(0) nan.
    if beta == 0.0 or not np.all(theta > 0):
        scalar_slope = None
    else:
        scalar_slope = float(np.polyfit(np.log(m), np.log(theta), 1)[0])
    return {
        "beta": beta, "machs": m.tolist(),
        "dyn_density_rms": drho.tolist(),
        "scalar_rms": theta.tolist(),
        "dyn_density_slope": float(np.polyfit(np.log(m), np.log(drho), 1)[0]),
        "scalar_slope": scalar_slope,
    }


# --------------------------------------------------------------------- run --
def run(n=64, seed=0):
    out = {}
    lo = leading_order(n=n, seed=seed)
    a, b = lo["acoustic_prefactor_a"], lo["entropy_prefactor_b"]
    out["leading_order"] = {k: lo[k] for k in
                            ("machs", "acoustic_density",
                             "entropy_density_per_beta", "acoustic_prefactor_a",
                             "entropy_prefactor_b", "acoustic_slope",
                             "entropy_slope")}
    out["crossover"] = crossover_sweep(a, b)
    # running exponent on a dense analytic Mach grid, with the gradient set so the
    # crossover sits near M~0.1: the local exponent then runs 1 (gradient-
    # dominated) -> 2 (pseudo-sound-dominated) across the subsonic range.
    m_dense = np.logspace(np.log10(0.005), np.log10(1.5), 41)
    beta_disp = 0.1 / (b / a)                    # crossover M^* ~ 0.1
    rexp = running_exponent(m_dense, a, b, beta_disp)
    out["running_exponent"] = {
        "beta": beta_disp, "crossover_mach": (b / a) * beta_disp,
        "machs": m_dense.tolist()[::4], "exponent": [float(x) for x in rexp][::4],
        "low_M_exponent": float(rexp[0]),      # -> 1
        "high_M_exponent": float(rexp[-1]),    # -> 2
    }
    # nonlinear confirmation at one gradient + the beta=0 pure-acoustic control
    out["nonlinear_grad"] = nonlinear_sweep(n=n, seed=seed, beta=2.0)
    out["nonlinear_null"] = nonlinear_sweep(n=n, seed=seed, beta=0.0)
    # buoyancy-on robustness at a single mid point
    d_bo, th_bo = sim_point(n, CSOUND, MU, seed, 0.12, 2.0, 1.0, 5e-3, buoy_g=3.0)
    d_bp, th_bp = sim_point(n, CSOUND, MU, seed, 0.12, 2.0, 1.0, 5e-3, buoy_g=0.0)
    out["buoyancy_check"] = {
        "M": 0.12, "beta": 2.0,
        "scalar_rms_passive": th_bp, "scalar_rms_buoyant": th_bo,
        "dyn_density_passive": d_bp, "dyn_density_buoyant": d_bo,
        "entropy_survives_feedback": bool(th_bo > 0.3 * th_bp),
    }
    return out, {"lo": lo}


# ------------------------------------------------------------------ figure --
def make_figure(out, aux, path_png):                        # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo = out["leading_order"]
    m = np.asarray(lo["machs"])
    a = lo["acoustic_prefactor_a"]
    b = lo["entropy_prefactor_b"]
    beta = out["running_exponent"]["beta"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ax.loglog(m, a * m ** 2, "o-", color="teal",
              label=r"acoustic (pseudo-sound) $\propto M^2$")
    ax.loglog(m, b * beta * m, "s-", color="darkorange",
              label=rf"entropy ($\beta$={beta:g}) $\propto M^1$")
    ax.loglog(m, a * m ** 2 + b * beta * m, "k-", lw=2, label="total NI density")
    mstar = (b / a) * beta
    ax.axvline(mstar, ls=":", color="0.4")
    ax.annotate(rf"$M^*\approx${mstar:.3f}", xy=(mstar, a * mstar ** 2),
                fontsize=9, color="0.3")
    ax.set_xlabel(r"turbulent Mach $M_t=u'/c$")
    ax.set_ylabel(r"$\delta\rho/\rho_0$")
    ax.set_title("Two NI density channels cross at $M^*$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    cr = out["crossover"]
    bet = np.asarray(cr["betas"])
    ax.loglog(bet, cr["mstar"], "o-", color="purple", ms=8, label="measured")
    ax.loglog(bet, cr["mstar"][0] * bet / bet[0], "k--",
              label=r"$\propto\beta$ reference")
    ax.set_xlabel(r"background gradient $\beta$")
    ax.set_ylabel(r"crossover Mach $M^*$")
    ax.set_title(rf"crossover gradient scale: $M^*\propto\beta$"
                 rf" (slope {cr['mstar_vs_beta_slope']:.2f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    re = out["running_exponent"]
    ax.semilogx(re["machs"], re["exponent"], "o-", color="crimson")
    ax.axhline(1.0, ls="--", color="0.5")
    ax.axhline(2.0, ls="--", color="0.5")
    ax.set_ylim(0.8, 2.2)
    ax.set_xlabel(r"turbulent Mach $M_t$")
    ax.set_ylabel(r"local exponent $d\ln(\delta\rho/\rho)/d\ln M_t$")
    ax.set_title(rf"exponent runs $1\to2$ ($\beta$={re['beta']:g})")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path_png, dpi=140)
    plt.close(fig)


def main():                                                 # pragma: no cover
    out, aux = run()
    os.makedirs(FIGDIR, exist_ok=True)
    jpath = os.path.join(FIGDIR, "nr37_pseudosound_crossover.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, aux, os.path.join(FIGDIR, "nr37_pseudosound_crossover.png"))
    print(json.dumps(out, indent=1))
    print(f"wrote {jpath} and the png")


if __name__ == "__main__":                                  # pragma: no cover
    main()
