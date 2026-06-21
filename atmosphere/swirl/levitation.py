r"""Levitation diagnostics for the Goldshtik-Sorokin swirl (Part 10).

The Goldshtik-Sorokin effect: a heavy particle is held up at a turbulent vortex
core by the radial pressure-gradient force of the core's low-pressure well.  Here
we turn the a-priori closure scorecard (the same Part-8b machinery) into a
**suspension margin** -- a force-balance proxy for whether the particle stays up.

Construction (all quantities are diagnosed from the frozen, developed truth field;
nothing is asserted):

  * The levitation force scales with the depth of the core pressure well, and the
    well depth scales with the resolved swirl kinetic energy E_res, because the
    incompressible pressure obeys p ~ rho |u|^2 (cyclostrophic balance
    rho u_theta^2 / r = dp/dr at the core).  So F_lev ~ E_res.

  * A closure changes the resolved swirl energy at the net rate P = <ubar . m>
    (the resolved-scale SGS power; P < 0 drains the swirl).  Over one core
    turnover tau = 2*pi*r_core / U_swirl the LES swirl energy becomes
    E = E_res + tau * P.

  * The suspension margin is that predicted swirl energy normalised by the truth's
    (so the truth field marginally suspends the particle, margin_truth := 1):

        margin(P) = max(E_res + tau*P, 0) / max(E_res + tau*P_truth, eps).

    margin >= 1  -> particle stays up;  margin < 1 -> particle falls.

This is a 2D, frozen-field, a-priori *mechanism* proxy, not an integrated
Lagrangian particle trajectory -- see the report's scope section.
"""

from __future__ import annotations

import numpy as np


def resolved_sgs_power(ub, vb, mx, my) -> float:
    """Net resolved-scale SGS power P = <ubar . m>.  P < 0 drains the resolved
    swirl; P > 0 feeds it (backscatter)."""
    return float(np.mean(ub * mx + vb * my))


def swirl_turnover(r_core: float, U_swirl: float) -> float:
    """Core eddy-turnover time tau = 2*pi*r_core / U_swirl."""
    return 2.0 * np.pi * r_core / U_swirl


def suspension_margin(e_res: float, tau: float, p_model: float,
                      p_truth: float) -> float:
    """Suspension margin for a closure with resolved SGS power ``p_model``,
    normalised so the truth field gives exactly 1 (marginal suspension)."""
    e_model = max(e_res + tau * p_model, 0.0)
    e_truth = max(e_res + tau * p_truth, 1e-12)
    return e_model / e_truth


def radial_pressure_profile(p, r, rmax: float = 2.6, nbin: int = 26):
    """Azimuthally-averaged pressure as a function of radius from the core,
    referenced to the far-field mean.  Returns (r_centres, p_profile)."""
    far = float(np.mean(p[r > 2.2]))
    bins = np.linspace(0.0, rmax, nbin)
    idx = np.digitize(r.ravel(), bins)
    pr = (p - far).ravel()
    prof = np.array([pr[idx == i].mean() if np.any(idx == i) else np.nan
                     for i in range(1, len(bins))])
    rc = 0.5 * (bins[1:] + bins[:-1])
    return rc, prof


def radial_speed_profile(u, v, r, rmax: float = 2.6, nbin: int = 26):
    """Azimuthally-averaged speed |u| as a function of radius from the core."""
    speed = np.sqrt(u ** 2 + v ** 2)
    bins = np.linspace(0.0, rmax, nbin)
    idx = np.digitize(r.ravel(), bins)
    sr = speed.ravel()
    prof = np.array([sr[idx == i].mean() if np.any(idx == i) else np.nan
                     for i in range(1, len(bins))])
    rc = 0.5 * (bins[1:] + bins[:-1])
    return rc, prof
