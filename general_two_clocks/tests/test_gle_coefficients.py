"""Regression tests for the §D.4 / §G.5 coefficient closure (gle_coefficients.py).

These pin the *derivable* properties (not the measured magnitudes, which are
solver-run dependent):

  * §G.5 correction is a pure time-lag (Taylor/lag equivalence) and is net-zero
    over a statistically-steady K_u cycle  -> PART B;
  * §B.2 ice-kernel coefficients are positive, finite, and obey the fast/slow
    scale separation tau_d >> tau_sgs       -> PART C;
  * the slow:fast bath weight is the Stefan number St = c_i*theta_far/L (the
    §B.2 ice DC gain, fast bath unit-normalised) and matches the §G.4 thermal-
    tail weight                              -> PART C bath weights;
  * the tau_c sign helper returns + for any decaying autocorrelation -> PART A math.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gle_coefficients as gle


def test_g5_correction_is_pure_time_lag():
    """K_u - tau_c d_tK == K_u(t - tau_c) to O(tau_c^2): the correction lags."""
    r = gle.lag_and_netzero(n=48, tau_c=0.02, omega=0.9, eps=0.4)
    assert r["is_pure_lag"]
    assert r["lag_equivalence_rel_err"] < 1e-2
    # the correction is a non-trivial perturbation (not identically zero)
    assert r["correction_rel_amplitude"] > 1e-4


def test_g5_correction_net_zero_in_steady_state():
    """Averaged over a full K_u cycle (<d_tK>=0) the mean correction vanishes."""
    r = gle.lag_and_netzero(n=48, tau_c=0.02, omega=0.9, eps=0.4)
    assert r["is_net_zero_in_steady_state"]
    assert r["net_over_rms"] < 1e-2
    # but the instantaneous correction is genuinely non-zero (it is a phase lag)
    assert r["cycle_typical_rms"] > 0.0


def test_g5_lag_error_shrinks_with_tau_c():
    """The first-order lag identity is O(tau_c^2): halving tau_c cuts the error
    by ~4x (a clean signature that it is a Taylor lag, not a coincidence)."""
    big = gle.lag_and_netzero(n=48, tau_c=0.04, omega=0.9, eps=0.4)
    small = gle.lag_and_netzero(n=48, tau_c=0.02, omega=0.9, eps=0.4)
    ratio = big["lag_equivalence_rel_err"] / (small["lag_equivalence_rel_err"] + 1e-30)
    assert ratio > 2.5  # ~4 for exact quadratic scaling


def test_ice_kernel_coefficients_positive_and_separated():
    """§B.2 coefficients are positive/finite and the bath times are well
    separated (tau_d >> tau_sgs => §D.4 scale-selectivity)."""
    c = gle.ice_kernel_coefficients(theta_far=2.0, Vbar=3.2e-8)
    assert c["tau_d_s"] > 0 and np.isfinite(c["tau_d_s"])
    assert c["B_s_minus_half"] > 0 and np.isfinite(c["B_s_minus_half"])
    assert c["fast_slow_separation"] > 1e3          # many orders of separation
    # tau_d = kappa / Vbar^2 scales as Vbar^-2: 10x faster melt -> 100x shorter tau_d
    c10 = gle.ice_kernel_coefficients(theta_far=2.0, Vbar=3.2e-7)
    assert c["tau_d_s"] / c10["tau_d_s"] == pytest.approx(100.0, rel=1e-6)


def test_ice_bath_weight_is_stefan_number():
    """The §D.4 slow:fast bath weight is the Stefan number St = c_i*theta_far/L
    (the §B.2 ice DC gain with the fast bath unit-normalised, int K_SGS dtau=1):
    closed form, subdominant (<0.1 up to 10 K), Vbar-independent, linear in theta."""
    cp, L = 2100.0, 3.34e5                      # gle.ice_kernel_coefficients constants
    for th in (2.0, 10.0):
        c = gle.ice_kernel_coefficients(theta_far=th, Vbar=3.2e-8)
        St = cp * th / L
        assert c["bath_weight_slow_over_fast"] == pytest.approx(St, rel=1e-12)
        assert c["stefan_weight"] == pytest.approx(St, rel=1e-12)
        assert c["fast_bath_dc_gain_normalized"] == 1.0
        assert c["bath_weight_slow_over_fast"] < 0.1     # slow ice bath subdominant
    # weight = St depends only on theta_far (not the melt speed Vbar)
    c_slow = gle.ice_kernel_coefficients(theta_far=2.0, Vbar=3.2e-8)
    c_fast = gle.ice_kernel_coefficients(theta_far=2.0, Vbar=3.2e-7)
    assert c_slow["bath_weight_slow_over_fast"] == pytest.approx(
        c_fast["bath_weight_slow_over_fast"], rel=1e-12)
    # St is linear in theta_far: doubling theta doubles the slow-bath weight
    c2 = gle.ice_kernel_coefficients(theta_far=2.0, Vbar=3.2e-8)
    c4 = gle.ice_kernel_coefficients(theta_far=4.0, Vbar=3.2e-8)
    assert (c4["bath_weight_slow_over_fast"] / c2["bath_weight_slow_over_fast"]
            == pytest.approx(2.0, rel=1e-12))


def test_ice_bath_weight_matches_thermal_tail_stefan_number():
    """Cross-module: the §D.4 bath weight is the *same* Stefan number as the §G.4
    thermal-tail weight (thermal_tail_amplitude: W_thermal/W_hydraulic = St, with
    the hydraulic kernel unit-gain). The two modules pick slightly different
    representative c_i (2100 vs 2009 J/kg/K ~ -10 C), so the formula St/c_i =
    theta_far/L matches exactly and the values agree within that ~4.5% choice."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "glaciers", "validation", "synthetic"))
    import thermal_tail_amplitude as tt
    th = 10.0
    c = gle.ice_kernel_coefficients(theta_far=th, Vbar=3.2e-8)
    St_gle = c["bath_weight_slow_over_fast"]
    St_tt = tt.stefan_weight(-th)               # c_i|theta|/L with C_ICE=2009
    assert tt.hydraulic_kernel_dc_gain() == 1.0  # fast bath unit-gain in both
    # identical formula: St / c_i == theta_far / L (exact)
    assert St_gle / 2100.0 == pytest.approx(St_tt / tt.C_ICE, rel=1e-12)
    assert St_gle / 2100.0 == pytest.approx(th / 3.34e5, rel=1e-12)
    # same physical Stefan number to within the c_i reference-temperature choice
    assert St_gle == pytest.approx(St_tt, rel=0.05)


def test_tau_c_sign_helper_positive_for_decaying_acf():
    """An autocorrelation time is non-negative by construction; the e-folding
    estimator returns a strictly positive time for a decaying (OU-like) series."""
    dt = 0.01
    t = np.arange(2000) * dt
    rng = np.random.default_rng(0)
    # synthesize an OU process with known correlation time tau=0.2
    tau = 0.2
    x = np.zeros_like(t)
    a = np.exp(-dt / tau)
    s = np.sqrt(1 - a * a)
    for i in range(1, len(t)):
        x[i] = a * x[i - 1] + s * rng.standard_normal()
    est = gle._autocorr_efold(x, dt)
    assert est > 0.0
    assert est == pytest.approx(tau, rel=0.5)   # recovers the set time within 50%
