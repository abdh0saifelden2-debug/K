"""DNS-free tests for the §D.2 scallop -> double-diffusion probe
(:mod:`scallop_doublediff`).

The per-column flux/Nusselt reductions, the phase-locking index and the regime
split are pure array operations, so they are exercised on small synthetic fields
without running any solver.  Two short integration checks confirm the new
backward-compatible scalloped-wall knob on the Candidate-2 solver behaves.
"""

import numpy as np

from scallop_doublediff import (
    column_turb_fluxes,
    harmonic_fractions,
    local_nusselt,
    phase_lock_fraction,
    phase_profile,
    regime_split,
    wall_coherent_fraction,
    wall_phase,
)
from subglacial.candidate2_doublediff import DoubleDiffConfig, DoubleDiffFlow


# --------------------------------------------------------------------------- #
# per-column turbulent fluxes
# --------------------------------------------------------------------------- #
def _global_turb_flux(v, c, fluid):
    f = fluid.astype(float)
    fvol = f.sum()
    vbar = (v * f).sum() / fvol
    cbar = (c * f).sum() / fvol
    return ((v - vbar) * (c - cbar) * f).sum() / fvol


def test_column_flux_fluid_weighted_mean_reproduces_global():
    rng = np.random.default_rng(0)
    nx, ny = 20, 16
    v = rng.standard_normal((nx, ny))
    th = rng.standard_normal((nx, ny))
    S = rng.standard_normal((nx, ny))
    fluid = rng.random((nx, ny)) > 0.3
    F_T, F_S = column_turb_fluxes(v, th, S, fluid)
    # weight each column by its fluid-cell count -> domain-mean flux
    ncol = fluid.sum(axis=1).astype(float)
    wT = (ncol * F_T).sum() / ncol.sum()
    wS = (ncol * F_S).sum() / ncol.sum()
    assert np.isclose(wT, _global_turb_flux(v, th, fluid))
    assert np.isclose(wS, _global_turb_flux(v, S, fluid))


def test_column_flux_empty_column_is_zero_not_nan():
    nx, ny = 5, 4
    v = np.ones((nx, ny))
    th = np.ones((nx, ny))
    S = np.ones((nx, ny))
    fluid = np.ones((nx, ny), bool)
    fluid[2, :] = False  # column 2 has no fluid
    F_T, F_S = column_turb_fluxes(v, th, S, fluid)
    assert np.isfinite(F_T).all() and np.isfinite(F_S).all()
    assert F_T[2] == 0.0 and F_S[2] == 0.0


def test_local_nusselt_formula():
    F_T = np.array([0.0, 2.0])
    F_S = np.array([0.0, 2.0])
    Nu_T, Nu_S = local_nusselt(F_T, F_S, kappa_T=1.0, kappa_S=0.01, grad=2.0)
    assert np.allclose(Nu_T, [1.0, 2.0])           # 1 + 2/(1*2)
    assert np.allclose(Nu_S, [1.0, 101.0])         # 1 + 2/(0.01*2)


# --------------------------------------------------------------------------- #
# phase-locking index
# --------------------------------------------------------------------------- #
def test_phase_lock_uniform_is_zero():
    phi = wall_phase(64, 12)
    f, _ = phase_lock_fraction(np.full(64, 3.3), phi)
    assert f < 1e-9   # constant field: no variance to lock (float residual only)


def test_phase_lock_pure_harmonic_is_one():
    nx, n = 240, 12
    phi = wall_phase(nx, n)
    a = 5.0 + 0.7 * np.cos(phi)            # pure fundamental, zero phase
    f, pref = phase_lock_fraction(a, phi)
    assert f > 0.99
    assert np.isclose(pref, 0.0, atol=1e-6) or np.isclose(pref, 2 * np.pi, atol=1e-6)


def test_phase_lock_harmonic_beats_noise():
    nx, n = 240, 12
    phi = wall_phase(nx, n)
    rng = np.random.default_rng(3)
    locked = 0.5 * np.cos(phi) + 0.05 * rng.standard_normal(nx)
    noise = rng.standard_normal(nx)
    f_lock, _ = phase_lock_fraction(locked, phi)
    f_noise, _ = phase_lock_fraction(noise, phi)
    assert f_lock > 0.8
    assert f_noise < 0.2
    assert f_lock > 3.0 * f_noise


# --------------------------------------------------------------------------- #
# total wall-coherent fraction (eta^2) and harmonic decomposition
# --------------------------------------------------------------------------- #
def test_wall_coherent_captures_second_harmonic_lock():
    # A response locked at *2x* the wall wavenumber (symmetric: peaks at every
    # crest AND trough).  The fundamental-only index sees ~0, but the total
    # wall-coherent fraction sees a near-perfect lock.  This is the exact
    # false-negative the metric is designed to fix.
    nx, n = 256, 12
    phi = wall_phase(nx, n)
    a = 3.0 + 0.8 * np.cos(2.0 * phi)
    f_fund, _ = phase_lock_fraction(a, phi)
    eta2 = wall_coherent_fraction(a, phi)
    assert f_fund < 0.05            # fundamental-only is blind to the 2nd harmonic
    # 12 phase bins smear a pure 2nd harmonic slightly (within-bin averaging),
    # so eta^2 ~ 0.9 rather than exactly 1 -- still an unambiguous lock.
    assert eta2 > 0.85              # total wall-coherent fraction sees the lock


def test_wall_coherent_uniform_is_zero():
    phi = wall_phase(128, 12)
    assert wall_coherent_fraction(np.full(128, 2.2), phi) < 1e-9


def test_wall_coherent_noise_is_small_and_beaten_by_lock():
    nx, n = 256, 12
    phi = wall_phase(nx, n)
    rng = np.random.default_rng(11)
    locked = 0.6 * np.cos(2.0 * phi) + 0.05 * rng.standard_normal(nx)
    noise = rng.standard_normal(nx)
    eta_lock = wall_coherent_fraction(locked, phi)
    eta_noise = wall_coherent_fraction(noise, phi)
    assert eta_lock > 0.8
    assert eta_noise < 0.2
    assert eta_lock > 3.0 * eta_noise


def test_harmonic_fractions_isolate_fundamental_and_second():
    nx, n = 256, 12
    phi = wall_phase(nx, n)
    fund = 5.0 + 0.7 * np.cos(phi)
    sec = 5.0 + 0.7 * np.cos(2.0 * phi)
    hf_fund = harmonic_fractions(fund, phi)
    hf_sec = harmonic_fractions(sec, phi)
    assert hf_fund[0] > 0.95 and hf_fund[1] < 0.05
    assert hf_sec[1] > 0.95 and hf_sec[0] < 0.05
    # frac[0] equals the fundamental-only f_lock by construction
    f_fund, _ = phase_lock_fraction(fund, phi)
    assert np.isclose(hf_fund[0], f_fund, atol=1e-9)


def test_harmonic_fractions_uniform_is_zero():
    phi = wall_phase(64, 12)
    assert harmonic_fractions(np.full(64, 1.0), phi) == [0.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# regime split
# --------------------------------------------------------------------------- #
def test_regime_split_detects_both_regimes():
    # reference |gamma_ref| = 1.0; some columns above, some below in magnitude
    gamma = np.array([0.2, 0.5, 1.4, 2.0, -1.8, -0.3])
    out = regime_split(gamma, gamma_ref=1.0)
    assert out["both_regimes"] is True
    assert out["frac_enhanced"] > 0.0 and out["frac_suppressed"] > 0.0
    assert np.isclose(out["gamma_range"], 2.0 - (-1.8))


def test_regime_split_one_sided_is_not_both():
    gamma = np.array([1.5, 2.0, 3.0, 1.2])     # all |gamma| > 1.0
    out = regime_split(gamma, gamma_ref=1.0)
    assert out["both_regimes"] is False
    assert out["frac_suppressed"] == 0.0


def test_regime_split_handles_nonfinite():
    gamma = np.array([np.nan, np.inf, 0.5, 2.0])
    out = regime_split(gamma, gamma_ref=1.0)
    assert np.isfinite(out["gamma_range"])
    assert out["both_regimes"] is True


def test_phase_profile_recovers_harmonic_and_flattens_noise():
    nx, n, nb = 600, 12, 12
    phi = wall_phase(nx, n)
    rng = np.random.default_rng(7)
    # a wall-locked cosine buried in turbulence: binning recovers a clean
    # peak-to-trough ~ 2*amplitude; an unlocked noise field bins to ~flat.
    locked = 4.0 + 1.0 * np.cos(phi) + 0.6 * rng.standard_normal(nx)
    noise = 4.0 + 0.6 * rng.standard_normal(nx)
    _, p2p_locked = phase_profile(locked, phi, nbins=nb)
    _, p2p_noise = phase_profile(noise, phi, nbins=nb)
    assert p2p_locked > 1.5            # ~2*amplitude survives the averaging
    assert p2p_noise < 0.5             # incoherent -> small binned spread
    assert p2p_locked > 3.0 * p2p_noise


def test_phase_profile_empty_bins_dropped():
    # only two distinct phases populated; remaining bins are NaN and ignored
    phi = np.array([0.1, 0.1, 3.2, 3.2])
    prof, p2p = phase_profile(np.array([1.0, 3.0, 5.0, 7.0]), phi, nbins=12)
    assert np.isfinite(p2p)
    assert np.isclose(p2p, 6.0 - 2.0)   # bin means 2.0 and 6.0


def test_wall_phase_in_range():
    phi = wall_phase(128, 12)
    assert phi.shape == (128,)
    assert phi.min() >= 0.0 and phi.max() < 2.0 * np.pi


# --------------------------------------------------------------------------- #
# backward-compatible scalloped wall on the Candidate-2 solver
# --------------------------------------------------------------------------- #
def test_flat_wall_is_unchanged_default():
    s = DoubleDiffFlow(DoubleDiffConfig(nx=48, ny=40), xp=np)
    assert np.unique(np.asarray(s.y_ice_x)).size == 1
    Y = np.asarray(s.Y)
    ref = np.clip((5.50 - Y) / (5.50 - 0.30), 0.0, 1.0)
    assert np.allclose(np.asarray(s.ramp), ref)


def test_scalloped_wall_varies_with_zero_mean_offset():
    cfg = DoubleDiffConfig(nx=64, ny=40, wall_amp=0.30, wall_nwaves=12)
    s = DoubleDiffFlow(cfg, xp=np)
    yx = np.asarray(s.y_ice_x)
    assert yx.max() - yx.min() > 0.5          # genuine corrugation
    assert np.isclose(yx.mean(), cfg.y_ice, atol=1e-6)   # sin averages to 0
    # a few steps stay finite and bounded
    s.run(20)
    for c in (np.asarray(s.theta), np.asarray(s.S)):
        assert np.isfinite(c).all()
        assert c.min() > -0.5 and c.max() < 1.5
