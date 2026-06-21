"""Tests for the Option-3 minimal Stefan (moving-boundary) prototype.

These exercise the *coupled 2-D* prototype in :mod:`subglacial.stefan_prototype`.
The rigorous <2% Neumann validation of the underlying 1-D enthalpy core lives in
``tests/test_moving_boundary.py``; here we check that the 2-D penalized prototype

  * recovers the Stefan ``s(t) ~ sqrt(t)`` similarity law in the no-flow limit
    (Test A), to within the accuracy of its diffuse (Brinkman) interface,
  * keeps the explicit moving mask numerically stable and the flow properly
    killed inside the ice (penalty consistency), and
  * produces a *differential* melt rate over a wavy ice base whose spatial
    pattern is set by the geometry -- the melt -> geometry feedback leg.
"""

import numpy as np
import pytest

from subglacial.stefan_prototype import StefanPrototype, StefanConfig
from subglacial.moving_boundary import neumann_lambda


def _test_a_solver(steps=3000, block=300):
    """Test A: flat base, no forcing, warm cavity -> conduction-limited melt."""
    St, kappa, ybed = 0.5, 3.0e-3, 0.30
    cfg = StefanConfig(n=96, y_bed=ybed, H0=ybed + 0.18, eps=0.0,
                       f0=0.0, df=0.0, beta=0.0, St=St, kappa=kappa,
                       nu=3.0e-3, dt=5.0e-4, n_mask=5, interface=2.0)
    s = StefanPrototype(cfg).init_warm_cavity()
    ts, ss = [], []
    for _ in range(steps // block):
        s.run(block)
        ts.append(s.t)
        ss.append(s.Hc.mean() - ybed)
    return s, np.array(ts), np.array(ss), St, kappa, ybed


def test_stefan_recovers_sqrt_t_similarity():
    """Test A: the melt front follows s^2 ~ t (i.e. s ~ sqrt(t)) and the
    similarity slope matches the analytic Neumann value to within the
    diffuse-interface tolerance (~30%)."""
    s, ts, ss, St, kappa, ybed = _test_a_solver()

    # s^2 must be linear in t (signature of sqrt(t) front growth)
    A, B = np.polyfit(ts, ss ** 2, 1)
    pred = A * ts + B
    ss2 = ss ** 2
    r2 = 1.0 - np.sum((ss2 - pred) ** 2) / np.sum((ss2 - ss2.mean()) ** 2)
    assert r2 > 0.99, f"s^2 not linear in t (R^2={r2:.4f}) -> not sqrt(t)"

    # similarity slope vs analytic Neumann: (2 lambda)^2 * alpha
    lam = neumann_lambda(1.0 / St)          # St_core = 1 / St_proto
    slope_exp = (2.0 * lam) ** 2 * kappa
    ratio = A / slope_exp
    assert 0.6 < ratio < 1.2, f"Stefan slope ratio {ratio:.3f} outside [0.6, 1.2]"

    # the front must advance monotonically (melting only)
    assert np.all(np.diff(ss) >= -1e-9), "ice base did not recede monotonically"


def test_moving_boundary_stability_and_penalty():
    """Mask stability (no blow-up / sub-cell motion per update) and penalty
    consistency (flow is killed inside the ice)."""
    cfg = StefanConfig(n=96, eps=0.05, f0=0.1, df=0.0, beta=0.0, St=2.0,
                       dt=1.0e-3, n_mask=10, interface=2.0)
    s = StefanPrototype(cfg).init_warm_cavity()
    Hprev = s.Hc.copy()
    max_jump = 0.0
    for _ in range(30):
        s.run(s.cfg.n_mask)
        max_jump = max(max_jump, float(np.abs(s.Hc - Hprev).max()))
        Hprev = s.Hc.copy()
    assert np.all(np.isfinite(s.Hc)), "H_c blew up (non-finite)"
    # per-mask-update motion stays well below a grid cell -> explicit update stable
    assert max_jump < 0.5 * s.dx, f"mask motion {max_jump:.3e} >= 0.5 dy ({0.5*s.dx:.3e})"
    # flow inside the ice is negligible vs the cavity
    assert s.penalty_consistency() < 1e-3, "penalty failed to kill flow in the ice"


def test_differential_melt_over_wavy_base():
    """Melt -> geometry leg: over a wavy ice base the melt rate is spatially
    differential and its pattern correlates with the imposed geometry (k0)."""
    ybed = 0.30
    cfg = StefanConfig(n=128, y_bed=ybed, H0=ybed + 0.25, eps=0.05, k0=2,
                       f0=0.0, df=0.0, beta=0.0, St=1.0, kappa=3.0e-3,
                       nu=3.0e-3, dt=5.0e-4, n_mask=5, interface=2.0)
    s = StefanPrototype(cfg).init_warm_cavity()
    sig0 = s.roughness()
    s.run(3000)
    m = s.melt_rate()
    x = s.xcol
    # the melt rate is not uniform ...
    assert m.std() / abs(m.mean()) > 0.1, "melt rate is spatially uniform (no feedback)"
    # ... and its spatial pattern tracks the imposed ice-base geometry
    corr = abs(np.corrcoef(m, np.sin(cfg.k0 * x))[0, 1])
    assert corr > 0.5, f"melt pattern uncorrelated with geometry (|corr|={corr:.2f})"
    # the roughness evolves (differential melt reshapes the base)
    assert abs(s.roughness() - sig0) > 1e-4, "roughness did not evolve"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
