"""Tests for the §A.3 scallop -> Röthlisberger channel feedback probe
(:mod:`scallop_channel_feedback`).

CPU-only, DNS-free: the reattachment-source / phase-locking diagnostics and the
channel-network ODE are exercised on synthetic per-column flux fields so the
three structural claims (sign, phase-locking, preferred-site selection) are
checked without running the spectral solver.
"""

import json
import math

import numpy as np

from scallop_channel_feedback import (
    channel_steady_size,
    integrate_channel_network,
    reattachment_source,
    rothlisberger_rhs,
)
from scallop_probe import _json_safe

LX = 4.0 * 2.0 * np.pi


def _synthetic_phase_locked_flux(n_waves=4, nx=256, base=1.0, amp=3.0,
                                 width=0.6, center=np.pi):
    """A flat-control mean of ``base`` and a bump field whose normal flux has a
    narrow reattachment spike (flux >> base) at wall phase ``center`` and is
    suppressed (flux < base) elsewhere -- a clean phase-locked reattachment."""
    x = np.arange(nx) * (LX / nx)
    phase = (2.0 * np.pi * n_waves * x / LX) % (2.0 * np.pi)
    d = np.abs((phase - center + np.pi) % (2.0 * np.pi) - np.pi)
    bump = 0.4 * base + amp * np.exp(-(d / width) ** 2)   # spike at `center`
    flat = np.full(nx, base)
    return bump, flat, phase


# --------------------------------------------------------------------------- #
# reattachment_source -- sign + phase-locking
# --------------------------------------------------------------------------- #
def test_reattachment_source_positive_sign_and_phase_locked():
    """The reattachment population carries a net positive normalised excess
    (``v_scallop > 0`` -> opening source) and is sharply phase-locked
    (``R_phase`` near 1) to the injected phase center."""
    m_bump, m_flat, _ = _synthetic_phase_locked_flux(center=np.pi)
    src = reattachment_source(m_bump, m_flat, n_waves=4, Lx=LX)

    assert src["v_scallop"] > 0.0                  # opening source (sign result)
    assert 0.0 < src["f_reatt"] < 0.5              # reattachment is a minority
    assert src["R_phase"] > 0.8                    # strongly phase-locked
    assert math.isclose(src["phase_pref"], np.pi, abs_tol=0.4)
    # nu_ratio - 1 is the signed net excess
    assert np.isclose(src["v_net"], src["nu_ratio"] - 1.0)


def test_reattachment_source_uniform_has_low_resultant():
    """A spatially uniform bump excess (no phase preference) gives a small
    Rayleigh resultant -- the control that ``R_phase`` actually measures
    locking, not just the presence of reattachment."""
    nx = 256
    m_flat = np.ones(nx)
    m_bump = np.full(nx, 1.5)                       # every column reattaches equally
    src = reattachment_source(m_bump, m_flat, n_waves=4, Lx=LX)

    assert np.isclose(src["f_reatt"], 1.0)
    assert src["v_scallop"] > 0.0
    assert src["R_phase"] < 0.2                     # no preferred phase


def test_reattachment_source_degenerate_guards_return_nan():
    """All-nonfinite bump (n==0) and a non-positive flat control both return a
    NaN dict instead of dividing by ~0, and stay JSON-safe."""
    nx = 16
    # (a) every bump column non-finite
    d = reattachment_source(np.full(nx, np.nan), np.ones(nx), n_waves=4, Lx=LX)
    assert d["n_valid"] == 0
    for k in ("nu_ratio", "v_scallop", "R_phase", "phase_pref"):
        assert math.isnan(d[k]), k
    # (b) non-positive flat control
    d2 = reattachment_source(np.ones(nx), np.zeros(nx), n_waves=4, Lx=LX)
    assert math.isnan(d2["v_scallop"])
    # JSON-safe (NaN -> null)
    reloaded = json.loads(json.dumps(_json_safe(d), allow_nan=False))
    assert reloaded["nu_ratio"] is None


# --------------------------------------------------------------------------- #
# Röthlisberger channel ODE
# --------------------------------------------------------------------------- #
def test_rothlisberger_steady_state_zero_rhs():
    """``channel_steady_size`` is the exact fixed point of ``rothlisberger_rhs``
    (creep closure linear in S), and a larger opening gives a larger channel."""
    k_creep = 0.75
    V_src = np.array([1.0, 2.0, 4.0])
    S_star = channel_steady_size(V_src, k_creep)
    rhs = rothlisberger_rhs(S_star, V_src, k_creep)
    assert np.max(np.abs(rhs)) < 1e-12 * float(V_src.max())
    assert S_star[2] > S_star[1] > S_star[0]       # monotone in opening


def test_integrate_channel_network_relaxes_to_steady_when_uncoupled():
    """With no concentration gain (g=0) each channel relaxes to its own
    closed-form steady size regardless of the initial noise."""
    k_creep = 1.0
    V_o = np.full(32, 1.0)
    V_sc = np.zeros(32)
    S, _ = integrate_channel_network(V_o, V_sc, k_creep,
                                     conc_gain=0.0, dt=0.05, n_steps=6000, seed=1)
    S_star = channel_steady_size(V_o, k_creep)
    assert np.allclose(S, S_star, rtol=1e-3)


# --------------------------------------------------------------------------- #
# preferred-site selection (the §A.3 structural claim)
# --------------------------------------------------------------------------- #
def test_phase_locked_source_selects_deterministic_site():
    """With the concentration loop active, a phase-locked scallop source makes
    the winning (largest) channel form at the source maximum for *every* noise
    seed, whereas a uniform source + noise picks scattered winners."""
    k_creep = 1.0
    m = 48
    phase = (2.0 * np.pi * 6 * np.arange(m) / m) % (2.0 * np.pi)
    # phase-locked source peaked at index where phase ~ pi
    V_sc = 0.3 * np.exp(-(((phase - np.pi + np.pi) % (2 * np.pi) - np.pi) / 0.5) ** 2)
    V_o = np.full(m, 1.0)
    site = int(np.argmax(V_sc))

    winners = []
    for sd in range(16):
        _, w = integrate_channel_network(V_o, V_sc, k_creep,
                                         conc_gain=0.6, dt=0.05, n_steps=4000,
                                         seed=sd)
        winners.append(w)
    # every seed converges on the source-max segment (deterministic selection)
    assert all(w == site for w in winners)

    # control: uniform source -> winners scattered (noise-selected)
    winners_noise = []
    for sd in range(16):
        _, w = integrate_channel_network(V_o, np.zeros(m), k_creep,
                                         conc_gain=0.6, dt=0.05, n_steps=4000,
                                         seed=sd)
        winners_noise.append(w)
    assert len(set(winners_noise)) > 1             # not a single deterministic site
