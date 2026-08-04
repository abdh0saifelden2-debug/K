"""Unit tests for the Theorem 3 counter-gradient diagnostic (THEORY_CAVITY §5).

GPU-free: the alignment math is exercised on synthetic flux/gradient fields,
and the accumulator is exercised on a tiny n=16 CPU run. The full Ri sweep that
produces figures/52 runs on a GPU (Kaggle P100).
"""
from __future__ import annotations

import numpy as np
import pytest

from theorem3_cg_gpu_probe import (
    CavityConfig,
    CavityFlow,
    CGAccumulator,
    counter_gradient_alignment,
)


def _const_field(value, shape=(4, 4, 4)):
    return np.full(shape, value, dtype=float)


def test_pure_down_gradient_gives_minus_one():
    """K-theory limit F = -c*grad(theta) (c>0) must give C_G = -1 exactly."""
    gx, gy, gz = _const_field(1.0), _const_field(-0.5), _const_field(0.3)
    fx, fy, fz = -2.0 * gx, -2.0 * gy, -2.0 * gz
    mask = np.ones((4, 4, 4), dtype=bool)
    assert counter_gradient_alignment((fx, fy, fz), (gx, gy, gz), mask) == pytest.approx(-1.0)


def test_pure_counter_gradient_gives_plus_one():
    """Pure counter-gradient flux F = +c*grad must give C_G = +1."""
    gx, gy, gz = _const_field(0.7), _const_field(0.2), _const_field(-0.9)
    fx, fy, fz = 3.0 * gx, 3.0 * gy, 3.0 * gz
    mask = np.ones((4, 4, 4), dtype=bool)
    assert counter_gradient_alignment((fx, fy, fz), (gx, gy, gz), mask) == pytest.approx(1.0)


def test_orthogonal_flux_gives_zero():
    """Flux perpendicular to the gradient everywhere -> C_G = 0."""
    gx, gy, gz = _const_field(1.0), _const_field(0.0), _const_field(0.0)
    fx, fy, fz = _const_field(0.0), _const_field(1.0), _const_field(0.0)
    mask = np.ones((4, 4, 4), dtype=bool)
    assert counter_gradient_alignment((fx, fy, fz), (gx, gy, gz), mask) == pytest.approx(0.0)


def test_alignment_is_magnitude_weighted():
    """The doc's definition <F.g>/<|F||g|> is magnitude-weighted: a high-|F|,
    high-|g| down-gradient cell must dominate a unit counter-gradient cell, so
    the result is strictly below the +1/-1 midpoint (i.e. negative)."""
    z = np.zeros(2)
    # cell 0: large down-gradient (|F|=|g|=10), cell 1: small counter-gradient
    gx = np.array([10.0, 1.0]); fx = np.array([-10.0, 1.0])
    cg = counter_gradient_alignment((fx, z, z), (gx, z, z), np.ones(2, dtype=bool))
    # dot = (-100 + 1)/2 = -49.5 ; |F||g| mean = (100+1)/2 = 50.5 -> -0.980
    assert cg == pytest.approx(-99.0 / 101.0)
    assert cg < 0.0


def test_empty_mask_and_zero_field_return_nan():
    z = np.zeros((3, 3, 3))
    g = _const_field(1.0, (3, 3, 3))
    assert np.isnan(counter_gradient_alignment((z, z, z), (g, g, g),
                                               np.zeros((3, 3, 3), dtype=bool)))
    # nonempty mask but identically zero flux -> denominator zero -> nan
    assert np.isnan(counter_gradient_alignment((z, z, z), (g, g, g),
                                               np.ones((3, 3, 3), dtype=bool)))


def test_accumulator_integration_small_cpu_run():
    """End-to-end on a tiny n=16 grid: the accumulator yields finite C_G values
    in [-1, 1] for both the fluid and the ice-base band."""
    cfg = CavityConfig(n=16, Ri=0.5, sgs="backscatter", backscatter=0.7,
                       bs_tau=0.05, seed=1)
    flow = CavityFlow(cfg)
    flow.run(20, ramp=7)
    acc = CGAccumulator(flow)
    for s in range(30):
        flow.step()
        if s % 3 == 0:
            acc.sample()
    cg_fluid, cg_band, fmag = acc.finalize()
    assert np.isfinite(cg_fluid) and -1.0 <= cg_fluid <= 1.0
    assert np.isfinite(cg_band) and -1.0 <= cg_band <= 1.0
    assert np.isfinite(fmag) and fmag >= 0.0
