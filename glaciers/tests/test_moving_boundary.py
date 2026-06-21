"""Validation tests for the moving-boundary (Stefan) machinery.

The one-phase Stefan solver must reproduce the analytic Neumann similarity
solution: a melt front advancing as ``X(t) = 2 lambda sqrt(alpha t)`` with the
Stefan-number-dependent prefactor ``lambda``.
"""

import numpy as np
import pytest

from subglacial.moving_boundary import (
    Stefan1D,
    Stefan1DConfig,
    neumann_front,
    neumann_lambda,
    neumann_temperature,
)


def test_neumann_lambda_known_values():
    # lambda solves lam e^{lam^2} erf(lam) = St / sqrt(pi); monotone in St.
    lam_small = neumann_lambda(0.2)
    lam_mid = neumann_lambda(1.0)
    lam_big = neumann_lambda(4.0)
    assert lam_small < lam_mid < lam_big
    # residual of the transcendental equation is ~0
    for St in (0.2, 1.0, 4.0):
        lam = neumann_lambda(St)
        from scipy.special import erf
        res = lam * np.exp(lam * lam) * erf(lam) - St / np.sqrt(np.pi)
        assert abs(res) < 1e-9


def test_neumann_lambda_requires_positive_stefan():
    with pytest.raises(ValueError):
        neumann_lambda(0.0)


@pytest.mark.parametrize("St", [0.2, 1.0, 4.0])
def test_front_matches_neumann_similarity(St):
    cfg = Stefan1DConfig(nx=800, Lx=1.0, k=1.0, c=1.0, Lf=1.0, T_w=St)
    alpha = cfg.k / cfg.c
    lam = neumann_lambda(St)
    # integrate until the front reaches ~0.55 Lx (stays clear of the far wall)
    t_end = (0.55 * cfg.Lx / (2.0 * lam)) ** 2 / alpha
    s = Stefan1D(cfg)
    s.run(t_end)
    X_num = s.front_position()
    X_ana = float(neumann_front(s.t, St, alpha))
    assert abs(X_num - X_ana) / X_ana < 0.01  # < 1 %


@pytest.mark.parametrize("St", [0.2, 1.0, 4.0])
def test_temperature_profile_matches_neumann(St):
    cfg = Stefan1DConfig(nx=800, Lx=1.0, k=1.0, c=1.0, Lf=1.0, T_w=St)
    alpha = cfg.k / cfg.c
    lam = neumann_lambda(St)
    t_end = (0.55 * cfg.Lx / (2.0 * lam)) ** 2 / alpha
    s = Stefan1D(cfg)
    s.run(t_end)
    T_num = s.temperature()
    T_ana = neumann_temperature(s.x, s.t, cfg.T_w, St, alpha)
    l2 = np.sqrt(np.mean((T_num - T_ana) ** 2)) / cfg.T_w
    assert l2 < 0.02  # < 2 % L2


def test_front_grows_like_sqrt_t():
    # X(t)^2 should be linear in t (similarity scaling).
    St = 1.0
    cfg = Stefan1DConfig(nx=800, Lx=1.0, T_w=St)
    alpha = cfg.k / cfg.c
    lam = neumann_lambda(St)
    t_end = (0.55 * cfg.Lx / (2.0 * lam)) ** 2 / alpha
    s = Stefan1D(cfg)
    ts, xs = [], []
    for frac in np.linspace(0.2, 1.0, 8):
        s.run(frac * t_end)
        ts.append(s.t)
        xs.append(s.front_position())
    ts = np.array(ts)
    xs = np.array(xs)
    # fit X^2 = m t + b; expect tight linear fit and slope = (2 lambda)^2 alpha
    m, b = np.polyfit(ts, xs**2, 1)
    resid = xs**2 - (m * ts + b)
    rel_resid = np.sqrt(np.mean(resid**2)) / np.mean(xs**2)
    assert rel_resid < 0.02
    slope_expected = (2.0 * lam) ** 2 * alpha
    assert abs(m - slope_expected) / slope_expected < 0.03


def test_dt_stability_guard():
    # explicit scheme must reject an over-large dt
    dx = 1.0 / (200 - 1)
    alpha = 1.0
    with pytest.raises(ValueError):
        Stefan1D(Stefan1DConfig(nx=200, Lx=1.0, k=1.0, c=1.0, dt=dx * dx / alpha))
