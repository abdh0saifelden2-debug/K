"""Unit proofs for NR46 (`general_two_clocks/new_relationships23.py`): the
solar two clocks -- NR28's coherence-gated cross-spectral protocol on 26
years of SOHO GOLF x VIRGO/SPM with BiSON control.  Offline-safe: all
numbers come from the committed aggregate cache
``data/nr46_solar_cache.json`` (raw FITS archives are public SOHO
mission-long bundles + the BiSON open-data portal; regeneration via
``build_cache()`` with env paths set).

Covered: the circular-statistics helpers; the gap-gated accumulating Welch
machinery on synthetic signals (coherence magnitude, imposed-delay phase
recovery, gap gating); the phase-sign convention; cache schema; and every
committed verdict (cross-channel resonance detection, channel-local slow
clock, cross-channel-only cutoff collapse, nonadiabatic one-sided mode
phase matching Jimenez et al. 1999, literature phase shape, background
second branch with monotone convergence, 26-yr epoch stability, instrument
controls).
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships23 import (  # noqa: E402
    FIG, MODE_G2, analyze, circdist_deg, circmean_deg, coh_phase, load_cache,
    welch_pair,
)


@pytest.fixture(scope="module")
def cache():
    return load_cache()


@pytest.fixture(scope="module")
def verdicts(cache):
    return analyze(cache)


# --------------------------------------------------------------------------- #
# helpers + machinery (synthetic)
# --------------------------------------------------------------------------- #
def test_circular_helpers():
    assert circmean_deg([170.0, -170.0]) == pytest.approx(180.0, abs=1e-9)
    assert circdist_deg(-119.0, 111.0) == pytest.approx(130.0)
    assert circdist_deg(10.0, -10.0) == pytest.approx(20.0)


def test_welch_pair_coherence_and_delay():
    rng = np.random.default_rng(0)
    n, L, k = 1 << 17, 512, 3                    # y = x delayed by 3 samples
    c = rng.standard_normal(n + k)
    a = 0.5
    x = c[k:] + a * rng.standard_normal(n)
    y = c[:-k] + a * rng.standard_normal(n)      # y lags x by k samples
    acc = welch_pair(x, y, np.ones(n, bool), L)
    g2, ph = coh_phase(acc)
    fr = np.fft.rfftfreq(L, 1.0)
    band = (fr > 0.05) & (fr < 0.30)
    # gamma^2 = 1/(1+a^2)^2 for equal independent noise on both channels
    assert np.median(g2[band]) == pytest.approx(1.0 / (1 + a ** 2) ** 2,
                                                abs=0.05)
    # phase of Y relative to X = -360 f k (Y lags)
    slope = np.polyfit(fr[band], np.unwrap(np.radians(ph[band])), 1)[0]
    assert slope / (2 * np.pi) == pytest.approx(-k, rel=0.05)


def test_welch_pair_gap_gating():
    rng = np.random.default_rng(1)
    n, L = 1 << 14, 512
    x = rng.standard_normal(n)
    ok = np.ones(n, bool)
    ok[::100] = False                            # 1 % gaps -> all windows fail
    acc = welch_pair(x, x, ok, L, min_valid=0.995)
    assert acc["nwin"] == 0
    acc2 = welch_pair(x, x, ok, L, min_valid=0.98)
    assert acc2["nwin"] > 0
    g2, ph = coh_phase(acc2)
    assert np.all(g2[1:] > 0.999)                # identical series
    assert np.allclose(ph[1:], 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# committed cache schema
# --------------------------------------------------------------------------- #
def test_cache_schema(cache):
    for pair in ("golf_x_green", "green_x_blue", "bison_x_golf"):
        assert pair in cache
    x = cache["golf_x_green"]
    assert x["nwin"] >= 500
    assert x["n_mode_bins"] >= 300
    assert len(x["mode_rows"]) == x["n_mode_bins"]
    assert all(set(r) == {"uhz", "g2", "ph"} for r in x["mode_rows"][:5])
    assert all(r["g2"] >= MODE_G2 for r in x["mode_rows"])
    assert len(x["epochs"]) >= 12
    assert cache["meta"]["dt_cal_s"] == 55.0
    assert "Jimenez" in cache["meta"]["dt_cal_method"]
    assert "downward positive" in cache["meta"]["phase_convention"]


# --------------------------------------------------------------------------- #
# the committed verdicts
# --------------------------------------------------------------------------- #
def test_fast_clock_cross_channel(cache, verdicts):
    assert verdicts["fast_clock_cross_channel"]
    x = cache["golf_x_green"]
    assert x["g2_mode_median"] >= 0.55           # measured 0.65
    assert x["g2_pband_max"] >= 0.90             # measured 0.96
    assert cache["bison_x_golf"]["g2_top_quintile"] >= 0.70   # measured 0.91


def test_slow_clock_channel_local(cache, verdicts):
    assert verdicts["slow_clock_channel_local"]
    assert cache["golf_x_green"]["g2_gran"] <= 0.10       # 0.065
    assert cache["bison_x_golf"]["g2_gran"] <= 0.06       # 0.035
    assert cache["green_x_blue"]["g2_gran"] >= 0.90       # 0.988


def test_cutoff_collapse_is_cross_channel_only(cache, verdicts):
    assert verdicts["cutoff_cross_channel_collapse"]
    assert cache["golf_x_green"]["g2_cutoff"] <= 0.03     # 0.014
    assert cache["bison_x_golf"]["g2_cutoff"] <= 0.05     # 0.032
    assert cache["green_x_blue"]["g2_cutoff"] >= 0.70     # 0.845 retained


def test_mode_phase_nonadiabatic(cache, verdicts):
    assert verdicts["mode_phase_nonadiabatic_convection_side"]
    lvl = cache["golf_x_green"]["phase_level_2900_3300"]
    assert -135.0 <= lvl <= -105.0               # Jimenez -121.3, IPHIR -119
    assert lvl < -90.0                           # below adiabatic
    for r in cache["golf_x_green"]["mode_curve"]:
        if 2400 <= r["uhz_mid"] <= 3400:
            assert r["ph"] < -90.0               # one-sided in the core
        assert -180.0 < r["ph"] < -60.0          # bounded


def test_phase_shape_matches_literature(cache, verdicts):
    assert verdicts["phase_shape_matches_literature"]
    ctr = [r["ph"] for r in cache["golf_x_green"]["mode_curve"]
           if 2900 <= r["uhz_mid"] <= 3500]
    hi = [r["ph"] for r in cache["golf_x_green"]["mode_curve"]
          if r["uhz_mid"] >= 4000]
    assert np.mean(hi) - np.mean(ctr) >= 20.0    # central dip -> high-nu rise


def test_background_second_branch_converges(verdicts):
    assert verdicts["background_second_branch"]
    seps = [s for _, s in verdicts["branch_separation_deg"]]
    assert seps[0] >= 80.0                       # distinct at the band core
    assert all(a >= b for a, b in zip(seps, seps[1:]))   # monotone blend-in
    assert seps[-1] <= 30.0                      # converged near cutoff


def test_epoch_stability(cache, verdicts):
    assert verdicts["epoch_stable_across_cycles"]
    assert verdicts["epoch_circ_std_deg"] <= 5.0          # measured 2.2
    eph = [e["ph"] for e in cache["golf_x_green"]["epochs"]]
    assert len(eph) >= 12
    assert all(-135.0 < p < -85.0 for p in eph)


def test_instrument_controls(cache, verdicts):
    assert verdicts["instrument_controls_pass"]
    assert abs(cache["green_x_blue"]["mode_phase"]) <= 1.5    # zero-phase
    assert circdist_deg(cache["bison_x_golf"]["top_phase"], 0.0) <= 20.0


def test_committed_figure_artifact(cache):
    with open(FIG) as fh:
        fig = json.load(fh)
    fresh = analyze(cache)
    assert fig["verdicts"] == fresh
    assert "mode_rows" not in fig["golf_x_green"]        # kept in cache only
