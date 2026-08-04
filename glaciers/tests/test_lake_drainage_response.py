"""Offline unit-proofs for the §G.4/§H.2 ATL15-dated drainage-response population test
(lake_drainage_response_atl15.py). Replays committed caches; plants signals."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = os.path.normpath(os.path.join(_HERE, "..", "validation", "external"))
sys.path.insert(0, _EXT)

import lake_drainage_response_atl15 as D  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(D._CACHE) and os.path.exists(D._ATL15)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed caches missing")


# ------------------------------------------------------------------ event detection
def test_detect_events_finds_planted_drop():
    atl15 = dict(per_lake=[dict(name="X", cx=0.0, cy=0.0, ns_std_quad=0.05,
                                t=[2019 + 0.25 * i for i in range(29)],
                                dh=[0.0] * 12 + [-1.0] * 17)])
    ev, quiet = D.detect_events(atl15)
    assert len(ev) == 1 and not quiet
    assert abs(ev[0]["events"][0]["t"] - (2019 + 0.25 * 11)) < 0.3
    assert ev[0]["events"][0]["drop_m"] < -0.9


def test_detect_events_quiet_on_noise():
    rng = np.random.default_rng(0)
    atl15 = dict(per_lake=[dict(name=f"q{i}", cx=0.0, cy=0.0, ns_std_quad=0.1,
                                t=[2019 + 0.25 * k for k in range(29)],
                                dh=list(rng.normal(0, 0.1, 29)))
                           for i in range(10)])
    ev, quiet = D.detect_events(atl15)
    assert len(ev) <= 1 and len(quiet) >= 9      # ~3-sigma: rare false alarms allowed


@needs_cache
def test_detect_events_on_committed_cache():
    ev, quiet = D.detect_events()
    n_ev = sum(len(L["events"]) for L in ev)
    assert len(ev) >= 15 and n_ev >= 20
    assert len(ev) + len(quiet) == 131


# ------------------------------------------------------------------ response detector
def _months():
    return D._mgrid()


def _series(vfunc, se=1.0, n=6):
    ms = _months()
    med = [float(vfunc(m)) for m in ms]
    return ms, med, [se] * len(ms), [n] * len(ms)


def test_event_response_detects_planted_step():
    t0 = 2022.0
    ms, med, se, n = _series(lambda m: 300.0 + (30.0 if m > t0 else 0.0), se=1.0)
    resp, why = D.event_response(ms, med, se, n, t0)
    assert resp is not None and resp["detected"] and resp["det_sign"] == +1
    assert abs(resp["r_at_max"] - 0.1) < 0.02


def test_event_response_quiet_on_flat():
    ms, med, se, n = _series(lambda m: 300.0, se=1.0)
    resp, why = D.event_response(ms, med, se, n, 2022.0)
    assert resp is not None and not resp["detected"]
    assert resp["max_abs_r"] < 0.01


def test_event_response_untestable_when_slow():
    ms, med, se, n = _series(lambda m: 10.0, se=0.5)
    resp, why = D.event_response(ms, med, se, n, 2022.0)
    assert resp is None and "slow" in why


def test_event_response_untestable_without_post_coverage():
    ms = _months()
    med = [300.0 if m < 2022.0 else None for m in ms]
    resp, why = D.event_response(ms, med, [1.0] * len(ms), [6] * len(ms), 2022.0)
    assert resp is None and "post" in why


def test_pool_weighted_median():
    ms = [2020.0, 2020.1, 2020.2]
    out = D._pool(ms, [100.0, 200.0, None], [1.0, 1.0, None], [1, 10, 0], 2019.9, 2020.3)
    assert out["v"] == 200.0 and out["n_months"] == 2


# ------------------------------------------------------------------ cache replay
@needs_cache
def test_cache_alignment():
    cache = json.load(open(D._CACHE))
    nm = len(cache["_months"])
    assert len(cache["event_lakes"]) >= 15
    assert len(cache["control_lakes"]) >= 20
    for L in cache["event_lakes"] + cache["control_lakes"]:
        assert len(L["m_median"]) == len(L["m_se"]) == len(L["m_n"]) == nm


@needs_cache
def test_analyze_population_offline():
    res = D.analyze()
    assert res["n_events_dated"] >= 20
    assert res["n_events_testable"] >= 5
    assert res["n_pseudo"] >= 50
    assert res["untestable_reasons"]
    assert "verdict" in res


@needs_cache
def test_committed_report_replays():
    rep_path = os.path.join(D._REPORTS, "lake_drainage_response.json")
    if not os.path.exists(rep_path):
        pytest.skip("report not committed yet")
    rep = json.load(open(rep_path))
    res = D.analyze()
    assert rep["n_events_testable"] == res["n_events_testable"]
    assert rep["n_detections"] == res["n_detections"]
