"""Offline unit-proofs for the §I.3 temporal-CSD FIELD test (temporal_csd_field.py).

Replays the committed cache; plants CSD signals to prove the detectors fire, and
stationary/noise signals to prove they stay quiet.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = os.path.normpath(os.path.join(_HERE, "..", "validation", "external"))
sys.path.insert(0, _EXT)

import temporal_csd_field as T  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE_CACHE = os.path.exists(T._CACHE)
needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="committed cache missing")


@pytest.fixture(scope="module")
def cache():
    with open(T._CACHE) as fh:
        return json.load(fh)


def _fake_point(v_series, se=2.0, d_gl=10.0):
    med = [float(x) if np.isfinite(x) else None for x in v_series]
    return dict(d_gl_km=d_gl, q_median=med,
                q_se=[se if m is not None else None for m in med],
                q_n=[10 if m is not None else 0 for m in med])


def test_prep_series_removes_season_and_trend():
    n = 48
    t = np.arange(n)
    v = 1000 + 5 * t + 30 * np.sin(2 * np.pi * t / 4)
    pr = T.prep_series(_fake_point(v))
    assert pr is not None
    assert np.nanstd(pr["resid"]) < 3.0     # season+trend nearly fully removed


def test_rolling_ews_detects_planted_csd_population():
    """CSD detection on 48 quarterly samples is noisy per-realization (that is why the
    module reads populations); the detector must fire at the POPULATION level."""
    tv, ta, joint = [], [], []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n = 48
        lam = np.linspace(0.9, 0.05, n)      # restoring rate collapsing
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] * (1 - lam[i]) + rng.normal(0, 1.0)
        st = T.point_ews(_fake_point(1000 + 25 * x, se=0.5), seed=100 + seed)
        assert st is not None
        tv.append(st["tau_var"]); ta.append(st["tau_ac1"])
        joint.append(st["tau_var"] > 0 and st["tau_ac1"] > 0)
    assert np.median(tv) > 0.1 and np.median(ta) > 0.15
    assert np.mean(joint) >= 0.5


def test_rolling_ews_quiet_on_stationary():
    rng = np.random.default_rng(2)
    taus = []
    for k in range(6):
        x = rng.normal(0, 10.0, 48)
        st = T.point_ews(_fake_point(1000 + x, se=0.5), seed=3 + k)
        if st is not None:
            taus.append((st["tau_var"], st["p_var"]))
    assert taus, "stationary points must be analyzable"
    # stationary noise: surrogate p should not be systematically tiny
    assert np.median([p for _, p in taus]) > 0.1


def test_noise_correction_absorbs_se_inflation():
    """A pure measurement-noise blow-up (SE² tracked) must not fake rising variance."""
    rng = np.random.default_rng(4)
    n = 48
    se_t = np.linspace(1.0, 6.0, n)                    # noise grows 6x
    v = 1000 + rng.normal(0, 1.0, n) * se_t
    med = [float(x) for x in v]
    p = dict(d_gl_km=10.0, q_median=med, q_se=list(se_t), q_n=[10] * n)
    st = T.point_ews(p, seed=5)
    assert st is not None
    # the SE² control flags the drift...
    assert st["tau_se2"] > 0.5
    # ...and the corrected variance trend is much weaker than the raw drift implies
    assert st["tau_var"] < 0.5


def test_phase_surrogates_preserve_gaps_and_power():
    rng = np.random.default_rng(6)
    r = rng.normal(0, 1, 48)
    r[[3, 9, 20, 33]] = np.nan
    surr = T.phase_surrogates(r, n_surr=8, seed=7)
    for s in surr:
        assert np.isnan(s[3]) and np.isnan(s[20])
        assert abs(np.nanstd(s) - np.nanstd(r)) < 0.75


# ------------------------------------------------------------------ cache replay
@needs_cache
def test_cache_streams_and_alignment(cache):
    assert set(cache["streams"]) >= {"Thwaites", "Pine_Island", "Smith", "Rutford"}
    nq = len(cache["_quarters"])
    for nm, s in cache["streams"].items():
        assert len(s["points"]) >= 20, nm
        for p in s["points"]:
            assert len(p["q_median"]) == len(p["q_se"]) == len(p["q_n"]) == nq


@needs_cache
def test_analyze_offline(cache):
    res = T.analyze(cache)
    assert res["groups"]["trunk_nearGL"]["n"] >= 10
    assert res["groups"]["control_nearGL"]["n"] >= 5
    assert "verdict" in res and res["verdict"]


@needs_cache
def test_committed_report_replays(cache):
    rep_path = os.path.join(T._REPORTS, "temporal_csd_field.json")
    if not os.path.exists(rep_path):
        pytest.skip("report not committed yet")
    rep = json.load(open(rep_path))
    res = T.analyze(cache)
    a = rep["groups"]["trunk_nearGL"]["median_tau_var"]
    b = res["groups"]["trunk_nearGL"]["median_tau_var"]
    assert abs(a - b) < 1e-9
