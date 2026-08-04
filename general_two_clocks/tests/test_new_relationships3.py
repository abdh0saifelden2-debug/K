"""Unit tests for the new derived relationship NR26
(general_two_clocks/new_relationships3.py). Deterministic, CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import new_relationships3 as NR3  # noqa: E402

# Compute the full NR26 result once at the production sample size and reuse it
# (the blind de-bias fit's precision is sample-size dependent; n=4e6 is the default).
_RES = None


def _res():
    global _RES
    if _RES is None:
        _RES = NR3.nr26(n=4_000_000)
    return _RES


def test_analytic_white_noise_limit():
    # tau_c -> 0 recovers the white-noise OU: Var -> D/lambda, ACF -> exp(-lam t)
    lam = 0.2
    assert abs(NR3.var_analytic(lam, 1e-6) - 1.0 / lam) / (1.0 / lam) < 1e-4
    t = np.array([0.5, 1.0, 2.0])
    assert np.max(np.abs(NR3.acf_analytic(t, lam, 1e-9) - np.exp(-lam * t))) < 1e-6


def test_simulation_matches_analytic_variance_and_acf():
    # the OU-driven slow mode reproduces Var=D/(lam(1+De)) and the bi-exponential ACF
    lam, tau_c = 0.2, 1.0
    s = NR3.simulate(lam, tau_c, n=2_000_000, seed=1)
    assert abs(np.var(s) - NR3.var_analytic(lam, tau_c)) / NR3.var_analytic(lam, tau_c) < 0.05
    lags = [50, 100, 150, 200]                      # dt=0.02 -> t = 1,2,3,4
    ac = NR3.ac_at(s, lags)
    ac_an = NR3.acf_analytic(np.array(lags) * 0.02, lam, tau_c)
    assert np.max(np.abs(ac - ac_an)) < 0.02


# NR26 — memory biases the critical-slowing-down early warning by De ---------- #
def test_nr26_variance_bias_is_one_plus_De():
    r = _res()
    # the variance EWS reads lambda_V = lambda (1 + De): false-safety bias = 1 + De
    for row in r["rows"]:
        assert abs(row["bias_V_measured"] - (1.0 + row["De"])) < 0.03
    assert r["checks"]["bias_ok"] and r["checks"]["var_ok"]


def test_nr26_precursors_bracket_the_truth():
    r = _res()
    # AC1 rate reads LOW (false alarm), variance rate reads HIGH (false safety):
    # lambda_A < lambda_true < lambda_V at every distance from the fold
    for row in r["rows"]:
        assert row["lam_A"] < row["lam_true"] < row["lam_V"]
    assert r["checks"]["bracket_ok"]


def test_nr26_lagslope_is_a_Dfree_memory_gauge():
    r = _res()
    # apparent rate grows with lag (bi-exponential ACF); slope > 0 and monotone in De
    slopes = [row["apprate_lagslope"] for row in r["rows"]]
    assert all(g > 0 for g in slopes)
    assert all(slopes[i] < slopes[i + 1] for i in range(len(slopes) - 1))


def test_nr26_debias_recovers_both_timescales():
    d = _res()["debias"]
    # a blind 2-exp fit recovers the true proximity lambda AND the bath memory tau_c
    assert d["lam_rel_err"] < 0.10 and d["tau_c_rel_err"] < 0.12


def test_nr26_overall_ok():
    assert _res()["ok"]
