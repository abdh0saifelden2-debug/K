"""Consistency tests for the NEON seasonal cross-check artifact (01b_neon_seasonal.json)."""

import json
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).resolve().parent.parent / "figures" / "01b_neon_seasonal.json"


@pytest.fixture(scope="module")
def months():
    with open(ARTIFACT) as f:
        return json.load(f)


def test_two_months_same_site(months):
    assert [m["month"] for m in months] == ["2020-01", "2020-07"]
    assert all(m["site"] == "WREF" for m in months)


def test_winter_matches_paper_headlines(months):
    w = months[0]
    assert w["n_valid_pressure"] == 1097 and w["n_valid_temperature"] == 1180
    assert w["pearson_r"] == pytest.approx(0.07, abs=0.02)


def test_fingerprint_persists_both_seasons(months):
    for m in months:
        # temperature diurnal-dominated: semidiurnal power well below diurnal
        assert m["temp_semi_over_diurnal_power"] < 0.10
        # pressure's semidiurnal fraction is several times temperature's
        assert m["pres_semi_over_diurnal_power"] > 2 * m["temp_semi_over_diurnal_power"]
        # pressure tide line lands near 12 h
        assert 11.5 < m["pres_semidiurnal_period_h"] < 12.5
        # pressure's dominant 6-240 h variability is synoptic (multi-day)
        assert m["pres_dominant_period_h"] > 96.0
    # summer temperature is the clean daily cycle; winter's weak diurnal line
    # can be out-powered by multi-day cold spells in the broad 6-240 h window
    assert 20.0 < months[1]["temp_dominant_period_h"] < 28.0


def test_coupling_regime_dependent_never_locked(months):
    w, s = months
    assert abs(w["pearson_r"]) < 0.15          # winter: near zero
    assert -0.65 < s["pearson_r"] < -0.30      # summer: thermal-low negative
    for m in months:
        # observed slope nowhere near the constant-density gas-law lock
        assert m["gaslaw_slope_kpa_per_k"] > 0.30
        assert m["slope_kpa_per_k"] < 0.10


def test_transport_ratio_swing_both_seasons(months):
    for m in months:
        assert m["transport_ratio_swing"] > 3.0
