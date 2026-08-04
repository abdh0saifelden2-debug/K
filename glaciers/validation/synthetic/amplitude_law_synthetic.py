r"""§V.8 synthetic unit test for the scallop amplitude-law coefficients (§G.2).

No external data.  §G.2 reduces the audited scallop-creep law to the monostable
two-term balance (Glen creep sub-dominant, §D.6)

    rho L * da/dt = alpha * a^{1/2} - beta * a ,     single stable a* = (alpha/beta)^2 .

This module derives and checks the parts that close from the *structure* of the
melt-opening vs. conduction balance.  The numerical *values* of alpha, beta stay
[HYP] (one (a, lambda) point; Caveat D) -- only their forms, signs, units, and the
fixed-point/stability algebra are derived here.

Derived results (all checked below):

  1. **Fixed point & stability [DERIVED].** ``a* = (alpha/beta)^2`` is the unique
     positive root; ``a = 0`` is unstable (the concave ``a^{1/2}`` growth dominates
     near zero) and ``a*`` is stable.  The linearised relaxation rate onto ``a*`` is
     exactly ``beta/(2 rho L)`` -- so ``beta`` doubles as the amplitude
     relaxation-rate coefficient.  (Eigenvalue ``d(da/dt)/da |_{a*} = -beta/(2 rho L) < 0``.)

  2. **Scaling [DERIVED].** ``a* ∝ (alpha/beta)^2``: doubling ``alpha`` quadruples
     ``a*``; doubling ``beta`` quarters it.

  3. **Dimensional forms [DERIVED].** ``rho L da/dt`` is a *latent-heat flux*
     [W m^-2].  Hence ``alpha`` has units [W m^{-5/2}] and ``beta`` [W m^-3].  The
     conduction-smoothing coefficient has the closed curvature form

        beta = c_beta * k_ice * dT * (2 pi / lambda)^2          [W m^-3]   (checks out)

     (curvature-driven conduction melts crests faster than troughs -> linear
     ``-beta a`` smoothing, ``beta > 0``).  With ``beta ∝ lambda^{-2}`` this predicts
     ``a* ∝ lambda^4`` at fixed ``alpha``.

  4. **Flow-dependence is in beta, not alpha [DERIVED constraint / falsification].**
     The naive "melt-opening grows with drive" ansatz ``alpha ∝ u_*`` predicts
     ``a* ∝ u_*^2`` -- *increasing* with drive.  The solver forcing probe
     (`scallop_forcing_probe.py`) instead [VERIFIED] that **more drive -> smaller
     a***.  So the net flow-dependence cannot live in the growth coefficient
     ``alpha``; it must enter the smoothing coefficient ``beta`` (stronger turbulent
     drive -> more uniform basal melt -> larger ``beta`` -> smaller ``a*``).  This
     falsifies the simplest ``alpha ∝ u_*`` closure and is why ``alpha``'s magnitude
     stays [HYP].
"""
from __future__ import annotations

import numpy as np


def a_dot(a, alpha, beta, rhoL):
    """rho L da/dt = alpha a^{1/2} - beta a  ->  da/dt."""
    a = np.maximum(a, 0.0)
    return (alpha * np.sqrt(a) - beta * a) / rhoL


def integrate(alpha, beta, rhoL, a0=1e-4, dt=1e-3, t_end=200.0):
    n = int(round(t_end / dt))
    a = a0
    traj = np.empty(n + 1)
    traj[0] = a
    for i in range(1, n + 1):
        a = a + dt * a_dot(a, alpha, beta, rhoL)  # explicit Euler (stable, dt small)
        traj[i] = a
    return traj


def relaxation_rate(traj, dt, a_star):
    """Late-time linear relaxation rate r in  a(t) - a* ~ e^{-r t}."""
    d = np.abs(traj - a_star)
    # use the last decade where d is small but resolvable
    m = (d > a_star * 1e-6) & (d < a_star * 1e-1)
    idx = np.where(m)[0]
    if len(idx) < 50:
        return np.nan
    i0, i1 = idx[0], idx[-1]
    return float(-(np.log(d[i1]) - np.log(d[i0])) / ((i1 - i0) * dt))


def beta_conduction(c_beta, k_ice, dT, lam):
    """Curvature-conduction smoothing coefficient beta = c_beta k_ice dT (2pi/lam)^2."""
    return c_beta * k_ice * dT * (2.0 * np.pi / lam) ** 2


def run():
    rhoL = 1.0  # reduced units: coefficients alpha,beta are [HYP] anyway, so only
    #             the dimensionless fixed-point/stability structure is tested here.
    #             (Physical rho_i L ~ 3e8 J m^-3 sets the absolute timescale only.)

    # (1) fixed point & stability
    alpha, beta = 2.0, 0.5
    a_star = (alpha / beta) ** 2
    traj = integrate(alpha, beta, rhoL)
    converged = bool(abs(traj[-1] - a_star) / a_star < 1e-3)
    grows_from_zero = bool(traj[10] > traj[0])           # a=0 unstable
    monotone = bool(np.all(np.diff(traj) > -1e-12))      # no overshoot
    # derived eigenvalue: relaxation rate = beta/(2 rho L)
    dt = 1e-3
    r_meas = relaxation_rate(traj, dt, a_star)
    r_theory = beta / (2.0 * rhoL)
    rate_ok = bool(abs(r_meas - r_theory) / r_theory < 0.05)

    # (2) scaling a* ∝ (alpha/beta)^2
    def astar_num(al, be):
        return integrate(al, be, rhoL)[-1]
    s_alpha = astar_num(2 * alpha, beta) / astar_num(alpha, beta)   # expect 4
    s_beta = astar_num(alpha, 2 * beta) / astar_num(alpha, beta)    # expect 1/4
    scaling_ok = bool(abs(s_alpha - 4.0) < 0.05 and abs(s_beta - 0.25) < 0.02)

    # (3) dimensional forms: rho L da/dt [W/m^2]; alpha [W m^-5/2]; beta [W m^-3];
    #     conduction beta = c_beta k_ice dT k^2 has units [W m^-3].
    # units as (kg, m, s, K) exponent vectors; W = kg m^2 s^-3
    W = np.array([1, 2, -3, 0])
    flux = W - np.array([0, 2, 0, 0])                 # W/m^2
    # rho L da/dt : (kg m^-3)(J kg^-1)(m s^-1) = (kg m^-3)(m^2 s^-2)(m s^-1)
    rhoL_adot = (np.array([1, -3, 0, 0]) + np.array([0, 2, -2, 0])
                 + np.array([0, 1, -1, 0]))
    lhs_is_flux = bool(np.array_equal(rhoL_adot, flux))
    # beta from k_ice [W m^-1 K^-1] * dT [K] * k^2 [m^-2]
    k_ice_u = W - np.array([0, 1, 0, 1])
    beta_u = k_ice_u + np.array([0, 0, 0, 1]) + np.array([0, -2, 0, 0])
    beta_units_ok = bool(np.array_equal(beta_u, np.array([1, -1, -3, 0])))  # W m^-3
    # beta * a must equal the flux:  [W m^-3][m] = [W m^-2]
    beta_a_is_flux = bool(np.array_equal(beta_u + np.array([0, 1, 0, 0]), flux))
    dims_ok = bool(lhs_is_flux and beta_units_ok and beta_a_is_flux)

    # (4) beta ∝ lambda^-2  ->  a* ∝ lambda^4 at fixed alpha
    k_ice, dT, c_beta = 2.1, 1.0, 1.0e-3
    lam1, lam2 = 1.0, 2.0
    b1 = beta_conduction(c_beta, k_ice, dT, lam1)
    b2 = beta_conduction(c_beta, k_ice, dT, lam2)
    astar_ratio = ((alpha / b2) ** 2) / ((alpha / b1) ** 2)
    lam4_ratio = (lam2 / lam1) ** 4
    lambda_scaling_ok = bool(abs(astar_ratio / lam4_ratio - 1.0) < 1e-6)

    # (5) flow-dependence falsification.  This module can only verify the *algebraic*
    #     half: under the naive closure alpha ∝ u_*, the fixed point a*=(alpha/beta)^2
    #     should INCREASE with drive.  We compute that prediction from the integrator
    #     (not a hardcoded closed form) so the check is a genuine test of the solved
    #     fixed point rather than a tautology.
    u1, u2 = 1.5, 3.0
    astar_u1 = astar_num(alpha * (u1 / u1), beta)   # reference drive
    astar_u2 = astar_num(alpha * (u2 / u1), beta)   # naive closure: alpha scales ∝ u_*
    naive_predicts_increasing = bool(astar_u2 > astar_u1 * (1.0 + 1e-6))
    # The independently-[VERIFIED] solver result (scallop_forcing_probe.py) is that
    # MORE drive gives a SMALLER a*.  That empirical sign is an external premise, not
    # recomputed here; this module asserts only that the naive prediction's sign
    # disagrees with it, i.e. the naive alpha∝u_* closure is falsified.
    OBSERVED_TREND = "decreasing"                   # external: scallop_forcing_probe.py
    naive_alpha_falsified = bool(naive_predicts_increasing
                                 and OBSERVED_TREND == "decreasing")

    ok = bool(converged and grows_from_zero and monotone and rate_ok
              and scaling_ok and dims_ok and lambda_scaling_ok
              and naive_alpha_falsified)
    return {
        "a_star_theory": a_star,
        "a_star_num": float(traj[-1]),
        "converged": converged,
        "zero_unstable": grows_from_zero,
        "monotone_no_overshoot": monotone,
        "relax_rate_meas": r_meas,
        "relax_rate_theory": r_theory,
        "relax_rate_ok": rate_ok,
        "scale_alpha_x2": float(s_alpha),
        "scale_beta_x2": float(s_beta),
        "scaling_ok": scaling_ok,
        "dims_ok": dims_ok,
        "lambda4_scaling_ok": lambda_scaling_ok,
        "naive_alpha_u_star_falsified": naive_alpha_falsified,
        "pass": ok,
    }


def main():
    r = run()
    print("=== §V.8 scallop amplitude-law test (§G.2) ===")
    print(f"  fixed point a*=(a/b)^2     : num {r['a_star_num']:.4f} vs theory {r['a_star_theory']:.4f}"
          f"  (converged={r['converged']}, a=0 unstable={r['zero_unstable']})")
    print(f"  stability eigen-rate b/2rhoL: meas {r['relax_rate_meas']:.3e} vs theory"
          f" {r['relax_rate_theory']:.3e}  ({r['relax_rate_ok']})")
    print(f"  scaling a*~(a/b)^2         : x2 alpha -> {r['scale_alpha_x2']:.2f} (=4),"
          f" x2 beta -> {r['scale_beta_x2']:.2f} (=0.25)  ({r['scaling_ok']})")
    print(f"  dimensional forms          : {r['dims_ok']}"
          f"  (rhoL a' & beta*a are W/m^2; beta=c k_ice dT k^2 is W/m^3)")
    print(f"  beta~lam^-2 => a*~lam^4    : {r['lambda4_scaling_ok']}")
    print(f"  naive alpha~u_* falsified  : {r['naive_alpha_u_star_falsified']}"
          f"  (would give a* INCREASING with drive; solver [VERIFIED] decreasing)")
    print(f"  PASS                       : {r['pass']}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
