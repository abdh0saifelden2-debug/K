r"""Deterministic tests for the temporal two-clocks analysis (temporal_clocks.py).

No network: the autocorrelation / decorrelation-time machinery is tested on synthetic
AR(1) processes with known decorrelation. (The real-NCEP verdict lives in
run_temporal_clocks.py / REPORT_TEMPORAL_CLOCKS.md.)
"""
from __future__ import annotations

import numpy as np

from reanalysis import temporal_clocks as tc


def _ar1_block(rho, nt=4000, npts=200, seed=0):
    """An AR(1) field block x[t] = rho x[t-1] + sqrt(1-rho^2) eps; ACF(k)=rho^k."""
    rng = np.random.default_rng(seed)
    x = np.zeros((nt, npts))
    s = np.sqrt(1.0 - rho * rho)
    for t in range(1, nt):
        x[t] = rho * x[t - 1] + s * rng.standard_normal(npts)
    return x


def test_acf_recovers_ar1():
    """temporal_acf recovers the analytic AR(1) autocorrelation ACF(k)=rho^k."""
    rho = 0.8
    x = _ar1_block(rho)
    acf = tc.temporal_acf(x, max_lag=10)
    assert abs(acf[0] - 1.0) < 1e-12
    for k in range(1, 6):
        assert abs(acf[k] - rho ** k) < 0.05


def test_fast_decorrelates_faster_than_slow():
    """A fast (small-rho) process has a shorter e-folding and integral time than a
    slow (large-rho) one — the fast-clock/slow-clock separation."""
    fast = tc.temporal_acf(_ar1_block(0.3, seed=1), max_lag=40)
    slow = tc.temporal_acf(_ar1_block(0.9, seed=2), max_lag=40)
    dt = 6.0
    assert tc.efold_hours(fast, dt) < tc.efold_hours(slow, dt)
    assert tc.integral_hours(fast, dt) < tc.integral_hours(slow, dt)


def test_efold_matches_ar1_theory():
    """The e-folding lag matches the AR(1) prediction k* = -1/ln(rho)."""
    rho = 0.85
    acf = tc.temporal_acf(_ar1_block(rho, seed=3), max_lag=60)
    dt = 6.0
    expected_lag = -1.0 / np.log(rho)              # ~6.15 steps
    got_lag = tc.efold_hours(acf, dt) / dt
    assert abs(got_lag - np.ceil(expected_lag)) <= 1.0
