"""Unit tests for the nonlocal eddy-diffusivity correction
(general_two_clocks/nonlocal_flux_correction.py).

These pin the structural claim that the mainstream gradient-diffusion (K-theory)
closure cannot represent counter-gradient transport, and that the scale-dependent
(nonlocal) diffusivity correction can -- while reducing to the local Fickian closure
exactly when the diffusivity is constant.  Deterministic, CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nonlocal_flux_correction as NF  # noqa: E402

RES = NF.compare()  # computed once


# --------------------------------------------------------------------------- #
# the structural theorem: a local down-gradient closure is pinned at C_G = -1
# --------------------------------------------------------------------------- #
def test_local_scalar_closure_is_pure_downgradient():
    x, C = NF.make_field(seed=1)
    dx = x[1] - x[0]
    gradC = NF.grad_spectral(C, dx)
    for kappa in (1e-3, 0.05, 1.0, 10.0):
        F = NF.local_flux(C, dx, kappa)        # F = -kappa grad C, kappa > 0
        # flux is down-gradient at every point
        assert np.all(F * gradC <= 1e-12)
        # ...so the magnitude-weighted alignment is exactly -1
        assert np.isclose(NF.counter_gradient(F, gradC), -1.0, atol=1e-9)


def test_local_variable_nonnegative_kappa_also_pinned():
    # even a spatially varying, non-negative diffusivity kappa(x) >= 0 is down-gradient
    x, C = NF.make_field(seed=2)
    dx = x[1] - x[0]
    gradC = NF.grad_spectral(C, dx)
    rng = np.random.default_rng(7)
    kappa_field = np.abs(rng.normal(size=len(C))) + 0.01   # kappa(x) >= 0
    F = -kappa_field * gradC                                # local closure, pointwise
    assert np.all(F * gradC <= 1e-12)
    assert np.isclose(NF.counter_gradient(F, gradC), -1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# the truth is counter-gradient; the correction recovers it
# --------------------------------------------------------------------------- #
def test_truth_is_counter_gradient_in_result11_regime():
    # counter-gradient (C_G > -1) but still net down-gradient (C_G < 0): RESULT-11 band
    assert -1.0 < RES["cg_true"] < 0.0


def test_mainstream_local_cannot_fit_counter_gradient():
    # the best *admissible* (kappa >= 0) local closure is non-degenerate ...
    assert RES["kappa_t_admissible"] > 0.0
    # ... yet still pinned at C_G = -1 and far from the truth (large residual)
    assert np.isclose(RES["cg_local"], -1.0, atol=1e-9)
    assert RES["err_local"] > 0.5
    assert RES["err_local"] > 10.0 * RES["err_nonlocal"]


def test_nonlocal_correction_is_exact():
    # the scale-dependent closure reproduces the truth and its alignment
    assert RES["err_nonlocal"] < 1e-10
    assert np.isclose(RES["cg_nonlocal"], RES["cg_true"], atol=1e-9)


def test_compare_ok():
    assert RES["ok"] is True


# --------------------------------------------------------------------------- #
# the local (Fickian) limit is recovered exactly -- the safe default
# --------------------------------------------------------------------------- #
def test_constant_diffusivity_equals_local_fick():
    x, C = NF.make_field(seed=3)
    dx = x[1] - x[0]
    k = NF.wavenumbers(len(C), dx)
    kappa0 = 0.073
    F_const = NF.apply_spectral_diffusivity(C, dx, NF.khat_constant(k, kappa0))
    F_local = NF.local_flux(C, dx, kappa0)
    assert np.allclose(F_const, F_local, atol=1e-12)
    assert np.isclose(NF.counter_gradient(F_const, NF.grad_spectral(C, dx)), -1.0, atol=1e-9)


def test_dc_gain_is_large_scale_diffusivity():
    k = NF.wavenumbers(512, (2 * np.pi) / 512)
    # constant kernel: DC gain == kappa0
    assert np.isclose(NF.dc_gain(NF.khat_constant(k, 0.05)), 0.05)
    # backscatter kernel: the dip is centred at kc >> 0, so kappa_hat(0) ~ kappa0
    khat = NF.khat_backscatter(k, 0.05, kc=6.0, width=1.0, beta=1.5)
    assert np.isclose(NF.dc_gain(khat), 0.05, atol=1e-3)


def test_backscatter_kernel_goes_negative_in_band():
    k = NF.wavenumbers(512, (2 * np.pi) / 512)
    khat = NF.khat_backscatter(k, 0.05, kc=6.0, width=1.0, beta=1.5)
    # beta > 1 => kappa_hat dips below zero somewhere (true backscatter / up-gradient)
    assert khat.min() < 0.0
    # ...while remaining positive (forward-scatter) at the smallest scales
    assert khat[np.argmax(np.abs(k))] > 0.0
