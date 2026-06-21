"""Unit tests for the new derived relationship NR29
(general_two_clocks/new_relationships6.py). Deterministic, CPU-only.
"""
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import new_relationships6 as NR6  # noqa: E402

_RES = None


def _res():
    global _RES
    if _RES is None:
        _RES = NR6.nr29(n=32, nu=4.0e-3, steps=400)   # fast, deterministic; ok=True
    return _RES


# --- analytic closed forms ------------------------------------------------- #
def test_gaussian_law_identity():
    # P(Pi<0) = Phi(-mu/sigma): exact CDF identity
    assert abs(NR6.gaussian_backscatter(0.05, 1.0) - float(norm.cdf(-0.05))) < 1e-12
    # symmetric flux (mu=0) -> exactly 1/2 backscatter
    assert abs(NR6.gaussian_backscatter(0.0, 2.3) - 0.5) < 1e-12
    # monotonically decreasing in mu/sigma, and -> 0 as the net dissipation dominates
    assert NR6.gaussian_backscatter(0.1, 1.0) < NR6.gaussian_backscatter(0.02, 1.0)
    assert NR6.gaussian_backscatter(10.0, 1.0) < 1e-6


def test_edgeworth_positive_skew_adds_backscatter():
    # positive flux skewness (forward-dissipation tail) ADDS to P(Pi<0) above the Gaussian law
    mu, sigma = 0.05, 1.0
    assert NR6.edgeworth_backscatter(mu, sigma, 0.4) > NR6.gaussian_backscatter(mu, sigma)
    # zero skew recovers the Gaussian law
    assert abs(NR6.edgeworth_backscatter(mu, sigma, 0.0)
               - NR6.gaussian_backscatter(mu, sigma)) < 1e-12


# --- NR29 simulated verification ------------------------------------------- #
def test_gaussian_law_matches_measured_backscatter():
    bs = _res()["backscatter"]
    # the Gaussian flux law reproduces the measured backscatter volume fraction to ~1%
    assert bs["gaussian_err"] < 0.02
    assert 0.45 < bs["gaussian_law"] < 0.50      # ~1/2, just below


def test_residual_is_positive_flux_skewness():
    r = _res()
    # the few-x10^-3 residual (measured above Gaussian) is the positive flux skewness
    assert r["flux"]["skewness"] > 0.0
    assert r["backscatter"]["measured"] > r["backscatter"]["gaussian_law"]


def test_smagorinsky_has_zero_backscatter():
    bs = _res()["backscatter"]
    # one-sided eddy-viscosity flux Pi = 2 nu_t |S|^2 >= 0 -> EXACTLY zero backscatter volume
    assert bs["smag"] == 0.0
    assert bs["smag_min_flux"] >= 0.0


def test_forward_cascade_pulls_backscatter_below_half():
    r = _res()
    assert r["flux"]["mean"] > 0.0                 # net forward dissipation
    assert r["backscatter"]["measured"] < 0.5


def test_gpu_closed_form_crosscheck():
    # the closed form holds across the six P100 GPU runs (n=128-192, k_c=16-32)
    assert _res()["gpu_crosscheck"]["max_abs_residual"] < 0.012


def test_nr29_overall_ok():
    assert _res()["ok"] is True
