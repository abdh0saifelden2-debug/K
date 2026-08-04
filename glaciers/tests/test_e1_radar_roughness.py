"""Unit proofs for §V.1e ``e1_radar_roughness`` (synthetic only, no downloads).

Each test proves one load-bearing claim of the Kirchhoff specularity->metres
gauge: the power-vs-field convention (Monte Carlo from first principles), the
closed-form inversion and its corollaries (window, sweet spot, Schroeder-2015
threshold), the two bias directions (Jensen), and the rank-exact transfer of
the §V.1d lake anchor.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "validation"))
from external.e1_radar_roughness import (  # noqa: E402
    LAMBDA_ICE, LAMBDA_VAC, N_ICE, SCHROEDER15_SIGMA_MAX, censored_percentile,
    fresnel_radius, gauge_constant, gauge_window, rayleigh_g, sigma_from_spec,
    spec_from_sigma, spec_water_threshold, sweet_spot)
from external.rtn_spec_sign_pin import stratified_mw  # noqa: E402


# ------------------------------------------------------------- instrument --
def test_wavelength_in_ice():
    """lambda_ice = c/(f n) with n = sqrt(3.17): ~2.81 m (NOT the 5 m vacuum
    value -- the wave strikes the bed while propagating in ice)."""
    assert abs(LAMBDA_VAC - 4.9965) < 1e-3
    assert abs(N_ICE - np.sqrt(3.17)) < 1e-12
    assert abs(LAMBDA_ICE - LAMBDA_VAC / N_ICE) < 1e-12
    assert 2.79 < LAMBDA_ICE < 2.83
    assert abs(gauge_constant() - LAMBDA_ICE / (4 * np.pi)) < 1e-15


# ---------------------------------------------------- first-principles MC --
def test_power_factor_convention_monte_carlo():
    """The exp(-g) POWER law with g=(4 pi sigma cos(theta)/lambda)^2 follows
    from <e^{i phi}>, phi = 2 k sigma cos(theta) xi, xi~N(0,1): coherent
    field e^{-g/2}, power its square.  Monte Carlo, no shortcuts."""
    rng = np.random.default_rng(7)
    lam, theta = 2.81, 0.3
    k = 2 * np.pi / lam
    for sigma in (0.05, 0.15, 0.30):
        phase = 2 * k * np.cos(theta) * rng.normal(0.0, sigma, 400_000)
        coh_power = abs(np.exp(1j * phase).mean()) ** 2
        assert abs(coh_power - spec_from_sigma(sigma, lam, theta)) < 5e-3


# ----------------------------------------------------------- closed forms --
def test_round_trip_exactness():
    sig = np.linspace(0.01, 0.6, 200)
    s = spec_from_sigma(sig)
    back = sigma_from_spec(s)
    assert np.allclose(back, sig, rtol=1e-12, atol=1e-14)


def test_wavelength_and_angle_scaling():
    """sigma(s) is proportional to lambda and to 1/cos(theta) -- the Rayleigh
    parameter's whole geometry dependence."""
    s = 0.37
    assert np.isclose(sigma_from_spec(s, lam=2 * LAMBDA_ICE),
                      2 * sigma_from_spec(s, lam=LAMBDA_ICE))
    th = 0.7
    assert np.isclose(sigma_from_spec(s, theta=th),
                      sigma_from_spec(s, theta=0.0) / np.cos(th))
    assert np.isclose(rayleigh_g(0.2, theta=th),
                      rayleigh_g(0.2 * np.cos(th), theta=0.0))


def test_censoring_and_limits():
    assert np.isnan(sigma_from_spec(0.0))
    assert np.isnan(sigma_from_spec(-0.1))
    assert sigma_from_spec(1.0) == 0.0
    assert sigma_from_spec(1.3) == 0.0
    a = sigma_from_spec(np.array([0.0, 0.2, 0.8, 1.0]))
    assert np.isnan(a[0]) and a[1] > a[2] > a[3] == 0.0  # strictly decreasing


def test_gauge_window_closed_form():
    """spec at the window edges is exactly (1-eps, eps): the window IS the
    trusted-specularity interval mapped through the inversion."""
    eps = 0.05
    lo, hi = gauge_window(eps)
    assert np.isclose(spec_from_sigma(lo), 1 - eps, rtol=1e-12)
    assert np.isclose(spec_from_sigma(hi), eps, rtol=1e-12)
    assert 0.045 < lo < 0.055 and 0.37 < hi < 0.40  # 5.1 cm .. 38.7 cm


def test_sweet_spot_minimises_noise_amplification():
    """|d sigma/ds| is minimised at s* = e^{-1/2} exactly (analytic), and
    numerically on a dense grid."""
    s_star, sig_star = sweet_spot()
    assert np.isclose(s_star, np.exp(-0.5), rtol=1e-12)
    assert np.isclose(sig_star, gauge_constant() / np.sqrt(2), rtol=1e-12)
    s = np.linspace(0.02, 0.98, 20001)
    dsig = np.abs(np.gradient(sigma_from_spec(s), s))
    assert abs(s[np.argmin(dsig)] - s_star) < 2e-3


def test_schroeder2015_threshold():
    """sigma <= 15 cm  <=>  s >= exp(-(0.15/K)^2) ~ 0.64: the independent
    scattering-model water-roughness bound becomes a specularity threshold."""
    s_thr = spec_water_threshold()
    assert np.isclose(sigma_from_spec(s_thr), SCHROEDER15_SIGMA_MAX,
                      rtol=1e-12)
    assert 0.60 < s_thr < 0.68
    # above the threshold the gauge reads smoother than 15 cm
    assert sigma_from_spec(0.9) < SCHROEDER15_SIGMA_MAX < sigma_from_spec(0.3)


def test_fresnel_radius():
    """sqrt(lambda h/2): 50-80 m for 2-4 km ice -- the horizontal scale the
    sigma estimate lives at."""
    assert 50 < fresnel_radius(2000.0) < 80
    assert 60 < fresnel_radius(4000.0) < 90
    assert np.isclose(fresnel_radius(3000.0),
                      np.sqrt(LAMBDA_ICE * 3000.0 / 2.0), rtol=1e-12)


# -------------------------------------------------------------- biases ----
def test_jensen_binning_direction():
    """For a heterogeneous cell, sigma(mean spec) <= sqrt(mean sigma^2):
    binned-mean inversion under-estimates the cell RMS.  Proof by random
    two-and-many-patch cells."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        sig_patches = rng.uniform(0.02, 0.45, rng.integers(2, 12))
        s_patches = spec_from_sigma(sig_patches)
        sig_of_mean = sigma_from_spec(s_patches.mean())
        rms = np.sqrt((sig_patches ** 2).mean())
        assert sig_of_mean <= rms + 1e-12


def test_extraneous_decoherence_overestimates():
    """Multiplying spec by any factor < 1 (englacial scattering, clutter)
    only RAISES the inverted sigma: extraneous decoherence over-estimates
    roughness -- opposite sign to Jensen, as documented."""
    s = np.array([0.9, 0.6, 0.3, 0.1])
    for f in (0.9, 0.7, 0.5):
        assert np.all(sigma_from_spec(f * s) > sigma_from_spec(s))


def test_censored_percentiles_are_exact_not_biased_smooth():
    """Censored (below-floor) samples enter as +inf ranks: a percentile
    inside the finite mass is EXACT (equals the percentile of the full true
    sample for any true censored values above the bound); one inside the
    censored mass returns None.  Dropping censored samples instead would
    bias every quantile smooth -- the bug this guards against."""
    rng = np.random.default_rng(9)
    true = np.sort(rng.uniform(0.05, 1.0, 1000))
    bound = np.quantile(true, 0.7)          # top 30% censored
    cens = true > bound
    reported = np.where(cens, np.inf, true)
    # exact where finite mass covers the quantile
    assert np.isclose(censored_percentile(reported, 50),
                      float(np.percentile(true, 50)), rtol=1e-12)
    # inside censored mass: only a bound -> None
    assert censored_percentile(reported, 90) is None
    # the naive (dropped-censored) median IS biased smooth
    naive = float(np.median(true[~cens]))
    assert naive < float(np.percentile(true, 50))


# ------------------------------------------------- rank-exact lake anchor --
def test_lake_anchor_rank_exact_transfer():
    """P[sigma_lake < sigma_ctrl] computed on inverted sigma equals
    P[spec_lake > spec_ctrl] computed on raw specularity: the monotone
    inversion transfers the §V.1d Mann-Whitney verbatim (metres for free)."""
    rng = np.random.default_rng(11)
    n = 400
    spec = np.clip(rng.beta(1.2, 6.0, n), 1e-4, 1 - 1e-4)
    lake = np.zeros(n, bool)
    lake[rng.choice(n, 40, replace=False)] = True
    spec[lake] = np.clip(spec[lake] + 0.15, 1e-4, 1 - 1e-4)  # lakes shinier
    f_af = rng.uniform(0.2, 1.0, n)
    ctrl = ~lake

    mw_spec = stratified_mw(spec, lake, ctrl, f_af,
                            np.random.default_rng(0), n_perm=200)
    sig = sigma_from_spec(spec)
    mw_sig = stratified_mw(-sig, lake, ctrl, f_af,
                           np.random.default_rng(0), n_perm=200)
    assert np.isclose(mw_spec["prob_lake_gt_ctrl"],
                      mw_sig["prob_lake_gt_ctrl"], atol=1e-12)
    # and the medians land in the physically sensible order, in metres
    assert (np.median(sig[lake]) < np.median(sig[ctrl]))


def test_water_like_cells_are_smooth_in_metres():
    """End-to-end synthetic: cells drawn smooth (sigma 5-12 cm) all clear the
    Schroeder threshold; cells drawn rough (sigma 0.3-0.5 m) all fail it."""
    rng = np.random.default_rng(5)
    smooth = rng.uniform(0.05, 0.12, 100)
    rough = rng.uniform(0.30, 0.50, 100)
    s_thr = spec_water_threshold()
    assert np.all(spec_from_sigma(smooth) >= s_thr)
    assert np.all(spec_from_sigma(rough) < s_thr)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
