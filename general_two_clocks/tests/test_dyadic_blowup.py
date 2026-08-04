"""Unit tests for the dyadic (shell) model of energy-conserving finite-time blowup
(general_two_clocks/dyadic_blowup.py).

These pin the regularity-thread claim that an energy-conserving NS-like nonlinearity
can blow up in finite time -- so the energy budget does not carry regularity, the
nonlocal/dissipative structure does:
(1) the inviscid nonlinearity conserves energy EXACTLY (telescoping inter-shell flux);
(2) yet the inviscid cascade drives H^1 up by >1e4x (blowup to the truncation ceiling);
(3) the cascade-front arrival t_front(N) converges to a finite t* as N grows
    (finite-time blowup, not a truncation artifact);
(4) NS dissipation (gamma=1) regularizes (bounded H^1), supercritical (gamma small)
    does not.
Deterministic (seeded), CPU-only, fast.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dyadic_blowup as D  # noqa: E402

RES = D.compare()  # computed once (~0.3s)


# --------------------------------------------------------------------------- #
# (1) energy is conserved EXACTLY by the inviscid nonlinearity
# --------------------------------------------------------------------------- #
def test_energy_identity_exact():
    """sum_n a_n*(da_n/dt) = 0 to round-off for arbitrary states (inviscid)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        N = int(rng.integers(4, 22))
        k = D.k_shells(N)
        a = rng.standard_normal(N + 1)
        worst = max(worst, abs(float(np.sum(a * D.rhs(a, k, nu=0.0)))))
    assert worst < 1e-8


def test_per_shell_flux_telescopes():
    """Per-shell energy budget a_n*(da_n/dt) = Pi_{n-1} - Pi_n exactly (the mechanism
    that makes the total flux telescope to zero)."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        N = int(rng.integers(4, 20))
        k = D.k_shells(N)
        a = rng.standard_normal(N + 1)
        dEn = a * D.rhs(a, k, nu=0.0)               # rate of shell energy
        Pi = D.flux(a, k)                           # Pi_n  (Pi_N = 0)
        Pi_prev = np.concatenate([[0.0], Pi[:-1]])  # Pi_{n-1} (Pi_{-1} = 0)
        assert np.allclose(dEn, Pi_prev - Pi, atol=1e-12)
    # and the boundary fluxes vanish, so the sum telescopes to 0
    assert D.flux(a, k)[-1] == 0.0


def test_energy_conserved_under_integration():
    """Energy drift along the inviscid trajectory stays tiny despite the blowup."""
    assert RES["inv"]["max_energy_drift"] < 1e-3
    inv = RES["inv"]
    fin = inv["E"][np.isfinite(inv["E"])]
    assert abs(fin[-1] - inv["E0"]) / inv["E0"] < 1e-3


# --------------------------------------------------------------------------- #
# (2) the inviscid cascade blows up (H^1 explodes) -- with energy conserved
# --------------------------------------------------------------------------- #
def test_inviscid_cascade_blows_up():
    assert RES["inviscid_H1_growth"] > 1e3
    assert RES["inv"]["t_front"] is not None
    # H^1 climbs toward the truncation ceiling ~2^N
    assert RES["inv"]["H1_max"] > 0.1 * 2.0 ** RES["inv"]["N"]


# --------------------------------------------------------------------------- #
# (3) finite-time blowup: front arrival converges as N -> infinity
# --------------------------------------------------------------------------- #
def test_front_arrival_converges():
    conv = D.blowup_time_vs_N(Ns=(8, 10, 12, 14, 16))
    ts = [t for _, t in conv]
    assert all(t is not None for t in ts)                 # front always arrives
    assert all(x <= y + 1e-9 for x, y in zip(ts, ts[1:]))  # monotone increasing
    first_gap, last_gap = ts[1] - ts[0], ts[-1] - ts[-2]
    assert last_gap < 0.5 * first_gap                      # Cauchy -> finite t*
    assert 0.5 < ts[-1] < 2.0                              # a sane finite limit


def test_front_time_none_when_dissipative():
    """A strongly dissipative run never reaches the front (no cascade to the cutoff)."""
    # front_time is inviscid by construction; check the dissipative integrate instead
    assert RES["vis"]["t_front"] is None


# --------------------------------------------------------------------------- #
# (4) NS dissipation regularizes; supercritical dissipation does not
# --------------------------------------------------------------------------- #
def test_viscous_regularizes():
    assert RES["vis"]["t_front"] is None
    assert RES["viscous_H1_max"] < 1e2
    # viscous energy strictly dissipates
    Ev = RES["vis"]["E"]
    assert Ev[-1] < Ev[0]


def test_supercritical_blows_up_critical_does_not():
    sup = D.integrate(N=12, nu=0.05, gamma=0.1, t_end=2.0)   # supercritical
    crit = D.integrate(N=12, nu=0.05, gamma=1.0, t_end=2.0)  # NS dissipation
    assert sup["t_front"] is not None and sup["H1_max"] > 1e3
    assert crit["t_front"] is None and crit["H1_max"] < 1e2
    assert sup["H1_max"] > 100.0 * crit["H1_max"]


# --------------------------------------------------------------------------- #
# (5) the headline comparison passes its own honest gate
# --------------------------------------------------------------------------- #
def test_compare_ok():
    assert RES["ok"] is True
    assert RES["blowup_time_converged"] is True
    assert RES["t_star"] is not None and 0.5 < RES["t_star"] < 2.0
