"""Offline unit-proofs for the §I.5 tidal-admittance FIELD test (tidal_admittance_field.py).

Replays the committed per-station harmonic cache; proves the harmonic-fit machinery on
planted signals (gaps, trends, relocations) before trusting field numbers.
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

import tidal_admittance_field as A  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(A._CACHE)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed cache missing")


# ------------------------------------------------------------------ harmonic core
def _make_series(days=60.0, dt_s=300.0, gaps=((20.0, 22.0), (40.0, 41.5))):
    t = np.arange(0.0, days, dt_s / 86400.0)
    keep = np.ones(t.size, bool)
    for g0, g1 in gaps:
        keep &= ~((t >= g0) & (t < g1))
    return t[keep]


def test_harmonic_fit_recovers_planted_constituents():
    rng = np.random.default_rng(0)
    t = _make_series()
    w_m2 = 2 * math.pi / (A.CONSTITUENTS["M2"] / 24.0)
    w_msf = 2 * math.pi / (A.CONSTITUENTS["MSf"] / 24.0)
    y = (0.004 * np.cos(w_m2 * t + 0.3) + 0.150 * np.cos(w_msf * t - 1.0)
         + 0.5 * t + 0.001 * t ** 2 + 0.002 * rng.normal(size=t.size))
    segs = A._segments(t)
    cons = A.pick_constituents(t[-1] - t[0])
    fit, rms = A.harmonic_fit(t, y, segs, cons)
    assert abs(fit["M2"][0] - 0.004) < 0.001
    assert abs(fit["MSf"][0] - 0.150) < 0.01
    assert rms < 0.004


def test_harmonic_fit_separates_m2_s2():
    t = _make_series(days=40.0, gaps=())
    w_m2 = 2 * math.pi / (A.CONSTITUENTS["M2"] / 24.0)
    w_s2 = 2 * math.pi / (A.CONSTITUENTS["S2"] / 24.0)
    y = 0.010 * np.cos(w_m2 * t) + 0.004 * np.cos(w_s2 * t + 0.7)
    fit, _ = A.harmonic_fit(t, y, A._segments(t), ["M2", "S2", "O1"])
    assert abs(fit["M2"][0] - 0.010) < 0.001
    assert abs(fit["S2"][0] - 0.004) < 0.001
    assert fit["O1"][0] < 0.001


def test_gapped_multiseason_span_separates_msf_mf():
    """Two 60-day seasons 200 days apart: single segments cannot separate MSf/Mf
    (Rayleigh 183 d) but the joint fit across the gap can."""
    t1 = _make_series(days=60.0, gaps=())
    t2 = t1 + 260.0
    t = np.r_[t1, t2]
    w_msf = 2 * math.pi / (A.CONSTITUENTS["MSf"] / 24.0)
    w_mf = 2 * math.pi / (A.CONSTITUENTS["Mf"] / 24.0)
    y = 0.100 * np.cos(w_msf * t) + 0.040 * np.cos(w_mf * t + 0.5)
    cons = A.pick_constituents(t[-1] - t[0])
    assert "Mf" in cons
    fit, _ = A.harmonic_fit(t, y, A._segments(t), cons)
    assert abs(fit["MSf"][0] - 0.100) < 0.015
    assert abs(fit["Mf"][0] - 0.040) < 0.015


def test_pick_constituents_respects_rayleigh():
    short = A.pick_constituents(20.0)
    assert "M2" in short and "S2" in short and "MSf" not in short
    med = A.pick_constituents(60.0)
    assert "MSf" in med and "Mf" not in med
    long_ = A.pick_constituents(200.0)
    assert "Mf" in long_


def test_kendall_exact_small_n():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    tau, p, n = A._kendall_exact(x, y)
    assert tau < -0.999 and p < 0.01 and n == 6


# ------------------------------------------------------------------ cache replay
@pytest.fixture(scope="module")
def cache():
    with open(A._CACHE) as fh:
        return json.load(fh)


@needs_cache
def test_cache_stations_structure(cache):
    sts = cache["stations"]
    assert len(sts) >= 20
    for st in sts:
        assert st["span_days"] >= 10
        assert st["vbar_m_yr"] > 5
        assert st["constituents"]
        for c, d in st["constituents"].items():
            assert d["A_disp_m"] >= 0 and d["sig_disp_m"] >= 0


@needs_cache
def test_grounded_floating_vertical_separation(cache):
    """Vertical semidiurnal amplitude must separate shelf from grounded sites."""
    g = [A.vertical_admittance(st) for st in cache["stations"]
         if st.get("grounded_measures")]
    f = [A.vertical_admittance(st) for st in cache["stations"]
         if st.get("grounded_measures") is False]
    assert np.median(g) < 0.1 < np.median(f)


@needs_cache
def test_grounded_stations_are_fortnightly_dominated(cache):
    """The Gudmundsson phenomenon on the committed cache: grounded near-GL stations
    carry MSf displacement >> M2 displacement."""
    n_dom = 0; n_tot = 0
    for st in cache["stations"]:
        if not st.get("grounded_measures"):
            continue
        c = st["constituents"]
        if "MSf" in c and "M2" in c and c["MSf"]["A_disp_m"] > 3 * c["MSf"]["sig_disp_m"]:
            n_tot += 1
            if c["MSf"]["A_disp_m"] > c["M2"]["A_disp_m"]:
                n_dom += 1
    assert n_tot >= 10 and n_dom / n_tot > 0.8


@needs_cache
def test_analyze_replay_and_registered_orderings(cache):
    res = A.analyze(cache)
    assert res["n_grounded"] >= 12
    t = res["ordering_tests"]["eps_v_MSf_vs_distance"]
    assert np.isfinite(t["tau"]) and 0 < t["p_exact"] <= 1
    ws = res["within_stream_concordance"]
    assert ws["pairs"] >= 15
    assert "verdict" in res


@needs_cache
def test_committed_report_replays(cache):
    rep_path = os.path.join(A._REPORTS, "tidal_admittance_field.json")
    if not os.path.exists(rep_path):
        pytest.skip("report not committed yet")
    rep = json.load(open(rep_path))
    res = A.analyze(cache)
    a = rep["ordering_tests"]["eps_v_MSf_vs_distance"]["tau"]
    b = res["ordering_tests"]["eps_v_MSf_vs_distance"]["tau"]
    assert abs(a - b) < 1e-9
