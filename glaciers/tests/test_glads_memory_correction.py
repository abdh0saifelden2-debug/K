"""Unit tests for the GlaDS-type memory-kernel correction
(subglacial/glads_memory_correction.py).

These guard the claim that eliminating the channel degree of freedom from a
linearised GlaDS cavity<->channel system is an *exact* Mori-Zwanzig projection,
so the mainstream memoryless (local) closure is a fast-channel approximation that
drops the post-drainage surge lag.  All checks are deterministic and CPU-only.
"""
import os
import sys

import numpy as np

_SUBGLACIAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "subglacial")
if _SUBGLACIAL not in sys.path:
    sys.path.insert(0, _SUBGLACIAL)

import glads_memory_correction as G  # noqa: E402

RES = G.compare()  # computed once


# --------------------------------------------------------------------------- #
# the MZ memory kernel itself
# --------------------------------------------------------------------------- #
def test_kernel_value_and_causality():
    a, b, tau2 = 0.15, 0.30, 0.5
    # K(0) = -a b (the eliminated channel's Green's function at zero lag)
    assert np.isclose(G.memory_kernel(0.0, tau2, a, b), -a * b)
    # the kernel is causal: K(tau<0) = 0
    assert G.memory_kernel(-1.0, tau2, a, b) == 0.0
    # exponential channel relaxation: K(tau)/K(0) = e^{-tau/tau2}
    tau = 0.37
    assert np.isclose(G.memory_kernel(tau, tau2, a, b) / (-a * b),
                      np.exp(-tau / tau2))


def test_kernel_integral_is_dc_gain():
    # int_0^inf K dtau = -a b tau2  (the DC gain the local closure keeps)
    a, b, tau2 = 0.15, 0.30, 0.5
    tt = np.linspace(0.0, 40.0 * tau2, 40001)
    integral = G._trapz(G.memory_kernel(tt, tau2, a, b), tt)
    assert np.isclose(integral, -a * b * tau2, rtol=1e-4)


# --------------------------------------------------------------------------- #
# the three closures
# --------------------------------------------------------------------------- #
def test_full_coupled_initial_condition():
    s, q = G.full_coupled([0.0], 2.0, 0.5, 0.15, 0.30, s0=1.0, q0=0.0)
    assert np.isclose(s[0], 1.0) and np.isclose(q[0], 0.0)


def test_projection_is_exact():
    # the memory-corrected closure reproduces the resolved cavity<->channel truth
    assert RES["cavity_relerr_corrected_vs_full"] < 1e-4
    assert RES["surge_relerr_corrected_vs_full"] < 1e-4


def test_corrected_has_interior_surge_lag_local_does_not():
    # the resolved truth and the corrected closure both rise to an INTERIOR peak
    assert RES["full_peak_interior"] is True
    assert RES["corrected_peak_interior"] is True
    # the corrected peak time tracks the resolved peak time (the surge lag)
    assert np.isclose(RES["corrected_peak_time"], RES["full_peak_time"], rtol=5e-2)
    assert RES["corrected_peak_time"] > 0.0
    # the mainstream LOCAL closure is monotone from t=0 -> no interior lag peak
    assert RES["local_peak_interior"] is False
    assert RES["ok"] is True


def test_local_closure_surge_is_monotone():
    t = np.linspace(0.0, 30.0, 3001)
    _, q_loc = G.local_closure(t, 2.0, 0.5, 0.15, 0.30)
    # memoryless slaved channel decays monotonically (no surge)
    assert np.all(np.diff(q_loc) <= 1e-12)


# --------------------------------------------------------------------------- #
# the Mori residual force (non-zero eliminated-variable initial state)
# --------------------------------------------------------------------------- #
def test_mori_residual_keeps_projection_exact():
    # with q0 != 0 the eliminated channel injects a residual force R(t); the
    # corrected cavity must still reproduce the resolved truth
    t = np.linspace(0.0, 30.0, 6001)
    s_full, q_full = G.full_coupled(t, 2.0, 0.5, 0.15, 0.30, s0=1.0, q0=0.4)
    s_cor, q_cor = G.memory_corrected(t, 2.0, 0.5, 0.15, 0.30, s0=1.0, q0=0.4)
    assert np.linalg.norm(s_cor - s_full) / np.linalg.norm(s_full) < 1e-4
    assert np.linalg.norm(q_cor - q_full) / np.linalg.norm(q_full) < 1e-4


# --------------------------------------------------------------------------- #
# the fast-channel (DC / mainstream-local) limit
# --------------------------------------------------------------------------- #
def test_retrofit_recovers_local_limit_as_tau2_shrinks():
    # apply_memory_kernel -> b tau2 s (the mainstream local response) as the
    # channel time tau2 -> 0, provided the grid resolves tau2 (dt = tau2/100).
    a, b = 0.15, 0.30
    errs = []
    for tau2 in (0.2, 0.1, 0.05):
        dt = tau2 / 100.0
        t = np.arange(0.0, 20.0 + dt, dt)
        s_slow = np.exp(-t / 8.0)                       # slow vs channel memory
        q = G.apply_memory_kernel(s_slow, dt, 2.0, tau2, a, b)
        ref = b * tau2 * s_slow
        mask = t > 2.0 * tau2                           # drop startup transient
        errs.append(np.linalg.norm(q[mask] - ref[mask]) / np.linalg.norm(ref[mask]))
    # small everywhere and monotonically vanishing as tau2 -> 0
    assert errs[0] < 5e-2
    assert errs[-1] < errs[0]
