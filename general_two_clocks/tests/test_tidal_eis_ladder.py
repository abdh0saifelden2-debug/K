"""Unit-proofs for NR64 (new_relationships41.py) — the tidal-ladder EIS read-out.

Offline: replays the committed §I.5 harmonic cache; plants exponential decays to
prove the attenuation-fit machinery (weights, jackknife, two-point flag, caps).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

import new_relationships41 as R  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(R._CACHE)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed §I.5 cache missing")


# ------------------------------------------------------------------ fit machinery
def test_fit_delta_recovers_planted_decay():
    d = [4.0, 12.0, 25.0, 40.0, 55.0]
    delta_true = 20.0
    A = [0.2 * math.exp(-x / delta_true) for x in d]
    s = [a / 50.0 for a in A]                     # uniform SNR 50
    f = R.fit_delta(d, A, s)
    assert f and not f["two_point"] and f["n"] == 5
    assert abs(f["delta_km"] - delta_true) / delta_true < 0.05
    assert f["err_km"] < 5.0


def test_fit_delta_two_point_flagged():
    f = R.fit_delta([10.0, 30.0], [0.1, 0.02], [0.001, 0.001])
    assert f["two_point"] and f["n"] == 2
    assert abs(f["delta_km"] - 20.0 / math.log(5.0)) < 0.5


def test_fit_delta_nondecaying_gives_inf():
    d = [5.0, 20.0, 40.0]
    A = [0.05, 0.05, 0.06]
    f = R.fit_delta(d, A, [1e-4] * 3)
    assert f["delta_km"] == np.inf


def test_fit_delta_drops_insignificant_stations():
    d = [5.0, 20.0, 40.0, 60.0]
    A = [0.2, 0.05, 0.001, 0.0008]
    s = [0.002, 0.002, 0.002, 0.002]              # last two below MIN_SIG
    f = R.fit_delta(d, A, s)
    assert f["n"] == 2 and f["two_point"]


def test_fit_delta_weight_cap_limits_single_station():
    d = [5.0, 20.0, 40.0, 60.0]
    delta_true = 25.0
    A = [0.2 * math.exp(-x / delta_true) for x in d]
    s = [a / 30.0 for a in A]
    s[0] = A[0] / 3000.0                          # one absurdly precise station
    f = R.fit_delta(d, A, s)
    assert abs(f["delta_km"] - delta_true) / delta_true < 0.10


def test_parabolic_ratios_are_parameter_free():
    assert abs(math.sqrt(R.PERIODS_H["Mm"] / R.PERIODS_H["MSf"]) - 1.366) < 0.01
    assert abs(math.sqrt(R.PERIODS_H["MSf"] / R.PERIODS_H["M2"]) - 5.34) < 0.02


# ------------------------------------------------------------------ cache replay
@needs_cache
def test_analyze_branches_present():
    res = R.analyze()
    assert {"Evans", "Foundation", "Talutis", "Rutford"} <= set(res["branches"])
    f = res["branches"]["Foundation"]["fits"]["MSf"]
    assert f["n"] >= 4 and not f["two_point"] and np.isfinite(f["delta_km"])


@needs_cache
def test_insitu_diffusivities_physical():
    res = R.analyze()
    ks = [e["K_MSf_m2_s"] for e in res["branches"].values() if "K_MSf_m2_s" in e]
    assert ks, "at least one branch must yield an in-situ K"
    for k in ks:
        assert 50.0 < k < 1e6                     # 'highly conductive' regime


@needs_cache
def test_semidiurnal_floor_and_bounds():
    res = R.analyze()
    n_floor = sum(e["semidiurnal"]["at_floor"] for e in res["branches"].values())
    assert n_floor >= 3                            # dead SD band is the rule
    bounds = [e["semidiurnal"]["delta_M2_upper_bound_km"]
              for e in res["branches"].values()
              if e["semidiurnal"].get("delta_M2_upper_bound_km")]
    assert bounds and max(bounds) < 15.0           # far below the MSf lengths


@needs_cache
def test_discriminant_reported_with_predictions():
    res = R.analyze()
    got = [e["ratio_Mm_over_MSf"] for e in res["branches"].values()
           if "ratio_Mm_over_MSf" in e]
    assert len(got) >= 3
    for r in got:
        assert abs(r["parabolic_prediction"] - 1.366) < 0.01
        assert r["maxwell_saturation_prediction"] == 1.0


@needs_cache
def test_committed_figure_json_replays():
    fig_json = os.path.join(R._FIGDIR, "99_tidal_eis_ladder.json")
    if not os.path.exists(fig_json):
        pytest.skip("figure json not committed yet")
    rep = json.load(open(fig_json))
    res = R.analyze()
    a = rep["branches"]["Foundation"]["fits"]["MSf"]["delta_km"]
    b = res["branches"]["Foundation"]["fits"]["MSf"]["delta_km"]
    assert abs(a - b) < 1e-9
