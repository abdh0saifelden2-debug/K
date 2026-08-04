"""Tests for the extended ASOS ladder artifact (P0-R2/R3/R4)."""
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "figures", "09b_asos_ci.json")


@pytest.fixture(scope="module")
def art():
    with open(ART) as fh:
        return json.load(fh)


def test_ladder_size_and_span(art):
    rows = art["stations"]
    assert len(rows) >= 8
    lats = [r["lat"] for r in rows]
    assert min(lats) < 26.0 and max(lats) > 46.0


def test_s2_declines_monotonically_with_cis(art):
    rows = sorted(art["stations"], key=lambda r: r["lat"])
    s2 = [r["S2_P"] for r in rows]
    assert all(a > b for a, b in zip(s2, s2[1:]))
    for r in rows:
        lo, hi = r["S2_P_ci95"]
        assert lo < r["S2_P"] < hi


def test_haurwitz_protocol_validation(art):
    t = art["s2_vs_haurwitz"]
    assert t["trend_corr"] > 0.9
    assert 0.8 < t["ratio_min"] and t["ratio_max"] < 1.6


def test_coherence_split(art):
    for r in art["stations"]:
        # at the solar lines: above the surrogate level (shared solar clock)
        assert r["coh_S2"] > r["coh_S2_surr95"]
        # off-line: at the noise floor (within 2x of the surrogate level)
        assert r["coh_offline"] < 2.0 * r["coh_offline_surr95"]


def test_no_tight_positive_daily_lock(art):
    for r in art["stations"]:
        lo, hi = r["r_ci95"]
        assert hi < 0.5                      # nowhere near a gas-law lock
