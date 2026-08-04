"""Unit proofs for NR43 (`general_two_clocks/new_relationships20.py`): the active-lake two
clocks in ICESat-2 ATL15.  Offline-safe -- all numbers come from the committed cache
``data/atl15_lake_dh_cache.json`` (derived from ATL15 v005 + Siegfried & Fricker 2018
outlines; regeneration path documented in the report).

Covered: the localisation discriminator (in-lake episodic dh variance significantly
enriched over matched controls); the top-episodic lakes being canonical active systems;
the fast-drain / slow-fill sawtooth asymmetry; the resolved-vs-subpixel size dependence;
the connected-vs-unconnected lag-coupling contrast (illustrative); and the detrender.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from new_relationships20 import (  # noqa: E402
    CACHE, _detrended, load_cache, nr43,
)


@pytest.fixture(scope="module")
def out():
    return nr43()


def test_cache_present_and_shape(out):
    d = load_cache(CACHE)
    assert d["_meta"]["n_lakes_covered"] == 131
    assert len(d["controls"]["ns_std"]) >= 800
    assert out["discriminator"]["n_lakes"] == 131


def test_localisation_significant(out):
    dsc = out["discriminator"]
    # in-lake episodic dh variance enriched over matched off-lake controls
    assert dsc["detect_binomial_p"] < 1e-3          # 22/131 above control p95
    assert dsc["detect_frac"] > 0.10                # > 2x the 5% chance rate
    assert dsc["detect_enrichment_over_chance"] > 2.0
    assert dsc["mannwhitney_p"] < 0.01
    assert dsc["perm_p_median"] < 0.05
    assert dsc["lake_p90"] > dsc["ctrl_p90"]


def test_top_lakes_are_known_active(out):
    # the strongest-episodic lakes are documented active subglacial lakes
    assert len(out["top_includes_known_active"]) >= 5
    names = {t["name"] for t in out["top_lakes"]}
    assert "MercerSubglacialLake" in names
    assert "EngelhardtSubglacialLake" in names


def test_fill_drain_asymmetry(out):
    fa = out["fill_drain_asymmetry"]
    assert fa["n_active"] >= 15
    assert fa["median_fall_over_rise"] > 1.0        # drainage faster than fill
    assert fa["n_fall_faster"] > fa["n_active"] / 2  # majority
    assert fa["sign_test_p"] < 0.1


def test_size_dependence(out):
    sd = out["size_dependence"]
    # 10 km pixels dilute sub-pixel lakes -> resolved lakes carry more episodic variance
    assert sd["resolved_median"] > sd["subpixel_median"]


def test_connectivity_contrast_suggestive(out):
    cl = out["connectivity_lag"]
    # connected lakes are (at least as) lag-coupled as unconnected controls; strong
    # coupling exists among the connected systems (illustrative -- 29 quarterly epochs)
    assert cl["connected_median_absr"] > 0.5
    assert cl["connected_median_absr"] >= cl["unconnected_median_absr"]


def test_detrender_removes_linear():
    # a pure secular (linear) series has ~zero non-secular residual
    t = np.linspace(2019, 2026, 29)
    r = {"dh_arr": 3.0 + 1.7 * (t - 2019), "t_arr": t}
    assert np.nanstd(_detrended(r, deg=1)) < 1e-9
