"""Offline unit-proofs for the §I.6 spatial-EWS FIELD test (spatial_ews_field.py).

Replays the committed derived cache (no network); plants synthetic signals to prove
the detectors fire when the effect is present and stay quiet on noise.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = os.path.normpath(os.path.join(_HERE, "..", "validation", "external"))
sys.path.insert(0, _EXT)

import spatial_ews_field as F  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE_CACHE = os.path.exists(F._CACHE)
needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="committed cache missing")


@pytest.fixture(scope="module")
def cache():
    with open(F._CACHE) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ cache integrity
@needs_cache
def test_cache_has_all_streams(cache):
    assert set(cache["streams"]) >= {"Thwaites", "Pine_Island", "Smith", "Rutford"}
    for nm, s in cache["streams"].items():
        assert len(s["points"]) >= 50, nm


@needs_cache
def test_cache_points_aligned_and_grounded(cache):
    for nm, s in cache["streams"].items():
        for p in s["points"]:
            assert len(p["years"]) == len(p["v_annual"]) == len(p["se_annual"])
            assert not p["floating"]
            assert p["d_gl_km"] >= 0


@needs_cache
def test_cache_speeds_physical(cache):
    for nm, s in cache["streams"].items():
        v = [np.median(p["v_annual"]) for p in s["points"] if p["v_annual"]]
        assert 30 <= np.nanmax(v) <= 6000, nm


# ------------------------------------------------------------------ estimator proofs
def test_point_stats_noise_correction():
    rng = np.random.default_rng(0)
    yrs = list(range(2014, 2026))
    v = list(1000.0 + rng.normal(0, 5.0, len(yrs)))
    se = [5.0] * len(yrs)
    st = F.point_stats(dict(years=yrs, v_annual=v, se_annual=se))
    # all variance is noise -> corrected signal ~ 0 (well below raw)
    assert st["var_sig"] < st["var_raw"]
    assert st["sigma_rel"] < math.sqrt(st["var_raw"]) / st["vbar"]


def test_point_stats_detrends_acceleration():
    yrs = np.arange(2014, 2026)
    v = 1000.0 + 50.0 * (yrs - 2014)          # pure secular speed-up, no fluctuation
    st = F.point_stats(dict(years=yrs.tolist(), v_annual=v.tolist(),
                            se_annual=[1e-6] * len(yrs)))
    assert st["var_sig"] < 1e-6                # trend removed, not counted as variance


def test_point_stats_requires_min_years():
    assert F.point_stats(dict(years=[2014, 2015], v_annual=[1, 2],
                              se_annual=[0.1, 0.1])) is None


def test_kendall_shift_null_fires_on_planted_monotone():
    rng = np.random.default_rng(1)
    d = np.arange(10.0, 210.0, 2.0)
    y = 1.0 / np.sqrt(d) + 0.002 * rng.normal(size=d.size)   # rises toward GL (d->0)
    tau, p, n, K = F._kendall_shift_p(d, y)
    assert tau < -0.5 and p < 0.05 and n == d.size and K > 50


def test_kendall_shift_null_quiet_on_smooth_bump():
    # smooth NON-monotone structure with the same autocorrelation scale:
    # a naive iid-permutation test would over-reject; the shift null must not.
    d = np.arange(10.0, 210.0, 2.0)
    y = np.sin(2 * np.pi * d / 90.0)
    tau, p, _, _ = F._kendall_shift_p(d, y)
    assert p > 0.05 or abs(tau) < 0.3


def test_shift_null_p_floor_is_explicit():
    # tiny profile: support is only a handful of shifts -> p cannot be tiny
    d = np.arange(0.0, 30.0, 2.0)
    y = -d + 0.01 * np.sin(d)
    tau, p, n, K = F._kendall_shift_p(d, y)
    assert K <= 10 and p >= 1.0 / (K + 1) - 1e-12


def test_pooled_trunk_test_fires_and_reports_support():
    rng = np.random.default_rng(5)
    profs = []
    for _ in range(2):
        d = np.arange(4.0, 60.0, 2.0)
        y = 1.0 / np.sqrt(d + 5) + 0.004 * rng.normal(size=d.size)
        profs.append((d, y))
    res = F.pooled_trunk_test(profs)
    assert res["tau"] < -0.4 and res["p_one_sided"] < 0.05
    assert res["null_support"] > 50 and res["p_floor"] <= res["p_one_sided"]


def test_pooled_trunk_test_quiet_on_shuffled():
    rng = np.random.default_rng(6)
    profs = []
    for _ in range(2):
        d = np.arange(4.0, 60.0, 2.0)
        y = rng.permutation(1.0 / np.sqrt(d + 5))
        profs.append((d, y))
    res = F.pooled_trunk_test(profs)
    assert res["p_one_sided"] > 0.05 or res["tau"] > -0.2


def test_corr_length_orders_smooth_vs_rough():
    rng = np.random.default_rng(4)
    d = np.arange(0.0, 120.0, 2.0)
    years = list(range(2014, 2026))

    def make_R(l_corr):
        cov = np.exp(-np.abs(d[:, None] - d[None, :]) / l_corr)
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(d.size))
        fields = {y: L @ rng.normal(size=d.size) for y in years}
        return {i: {y: float(fields[y][i]) for y in years} for i in range(d.size)}

    c_s, xi_s = F.corr_length_profile(d, make_R(15.0), None)
    c_r, xi_r = F.corr_length_profile(d, make_R(3.0), None)
    assert np.nanmedian(xi_s) > np.nanmedian(xi_r)


def test_point_stats_quadratic_detrend_removes_curvature():
    yrs = np.arange(2014, 2026)
    v = 1000.0 + 30.0 * (yrs - 2014) + 4.0 * (yrs - 2019.5) ** 2   # curved speed-up
    st1 = F.point_stats(dict(years=yrs.tolist(), v_annual=v.tolist(),
                             se_annual=[1e-6] * len(yrs)), detrend_order=1)
    st2 = F.point_stats(dict(years=yrs.tolist(), v_annual=v.tolist(),
                             se_annual=[1e-6] * len(yrs)), detrend_order=2)
    assert st2["var_sig"] < 1e-6 < st1["var_sig"]   # curvature is NOT a fluctuation


# ------------------------------------------------------------------ analysis replay
@needs_cache
def test_analyze_offline_and_reportable(cache):
    res = F.analyze(cache)
    assert "verdict" in res and res["streams"]
    assert "pooled_trunk_snr10" in res and "pooled_trunk_snr10_quad_detrend" in res
    for nm, s in res["streams"].items():
        assert s["n_points"] >= 40, nm
        assert np.isfinite(s["kendall_tau_sigma_vs_dist"])
        assert 0 < s["p_shift_sigma"] <= 1
        assert s["d_gl_range_km"][0] < 25 and s["d_gl_range_km"][1] > 80
        assert np.isfinite(s["noise_control_tau"])


@needs_cache
def test_committed_report_matches_cache(cache):
    """The committed report JSON must replay from the committed cache."""
    rep_path = os.path.join(F._REPORTS, "spatial_ews_field.json")
    if not os.path.exists(rep_path):
        pytest.skip("report not committed yet")
    rep = json.load(open(rep_path))
    res = F.analyze(cache)
    for nm in rep["streams"]:
        a = rep["streams"][nm]["kendall_tau_sigma_vs_dist"]
        b = res["streams"][nm]["kendall_tau_sigma_vs_dist"]
        assert abs(a - b) < 1e-9, nm
