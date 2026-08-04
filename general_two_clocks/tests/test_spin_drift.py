"""Unit proofs for NR54 (`general_two_clocks/new_relationships31.py`): the
spin sets the drift compass.  Beta-drift read from single-float
statistics: strong spinners drift westward; among strong spinners,
cyclonic spin drifts poleward and anticyclonic equatorward -- the
altimetric eddy-drift law recovered at 700-1300 dbar with no eddy
detection.  Offline-safe: committed cache
``data/nr54_spin_drift_cache.json``.

Covered: compass helpers (quintiles, two-sample t, hemisphere compass
with the poleward-positive convention); synthetic validation with a
prescribed spin-drift coupling and a shuffled null; cache schema; real
westward monotone shift in both hemispheres; the pooled poleward
splitting; magnitude sanity window; figure write-through.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships31 import (  # noqa: E402
    CACHE, FIG, KM_PER_DAY, analyze, hemisphere_compass, load_cache,
    quintile_bins, run, synthetic_checks, two_sample_t,
)


@pytest.fixture(scope="module")
def syn():
    return synthetic_checks()


@pytest.fixture(scope="module")
def cache():
    return load_cache()


@pytest.fixture(scope="module")
def result(cache):
    return analyze(cache)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_quintile_bins_balanced():
    x = np.random.default_rng(0).random(5000)
    b = quintile_bins(x)
    counts = np.bincount(b, minlength=5)
    assert counts.min() > 900 and counts.max() < 1100


def test_two_sample_t_sign():
    d, t = two_sample_t([1.0] * 50 + [1.2] * 50, [0.0] * 100)
    assert d == pytest.approx(1.1) and t > 10


def test_poleward_convention():
    # SH: dv < 0 is poleward, so pole_sign=-1 must make eta positive
    rng = np.random.default_rng(7)
    spin = np.concatenate([np.full(30, -0.1), np.full(30, 0.1),
                           rng.normal(0, 1e-3, 240)])
    dv = np.concatenate([np.full(30, -0.01), np.full(30, 0.01),
                         np.zeros(240)])
    du = np.zeros(300)
    lat = np.full(300, -30.0)
    h = hemisphere_compass(spin, du, dv, lat, -1.0)
    # cyclonic in SH = spin<0; those floats have dv<0 = poleward
    assert h["poleward_split_km_d"] > 0


# --------------------------------------------------------------------------- #
# synthetic validation
# --------------------------------------------------------------------------- #
def test_synthetic_compass_recovers_pattern(syn):
    for hemi in ("nh", "sh"):
        assert syn[hemi]["westward_t"] < -5
        assert syn[hemi]["poleward_t"] > 5


def test_shuffled_null_is_flat(syn):
    assert abs(syn["null_westward_t"]) < 2.5
    assert abs(syn["null_poleward_t"]) < 2.5


# --------------------------------------------------------------------------- #
# committed cache
# --------------------------------------------------------------------------- #
def test_cache_schema(cache):
    f = cache["floats"]
    n = cache["meta"]["n_floats"]
    assert n > 7000
    for k in ("spin", "du", "dv", "lat", "n"):
        assert len(f[k]) == n
    assert np.isfinite(f["spin"]).all()
    assert np.isfinite(f["du"]).all() and np.isfinite(f["dv"]).all()


# --------------------------------------------------------------------------- #
# real-data verdicts
# --------------------------------------------------------------------------- #
def test_westward_shift_both_hemispheres(result):
    for hemi in ("nh", "sh"):
        h = result["hemispheres"][hemi]
        assert h["westward_t"] < -2.0
        assert h["u_trend_km_d"][4] < h["u_trend_km_d"][0]
    assert result["verdicts"]["westward_drift_increases_with_spin"]


def test_poleward_splitting(result):
    assert result["pooled_poleward_t"] > 3.0
    assert result["hemispheres"]["nh"]["poleward_split_km_d"] > 0
    assert result["hemispheres"]["sh"]["poleward_split_km_d"] > 0
    assert result["verdicts"][
        "cyclones_poleward_anticyclones_equatorward"]


def test_sh_split_is_decisive(result):
    assert result["hemispheres"]["sh"]["poleward_t"] > 4.0


def test_magnitude_window(result):
    for hemi in ("nh", "sh"):
        assert 0.02 < abs(
            result["hemispheres"][hemi]["westward_shift_km_d"]) < 1.0
    assert result["verdicts"]["magnitude_order_consistent"]


def test_all_verdicts_and_figure(result):
    for key in ("compass_estimator_validated",
                "westward_drift_increases_with_spin",
                "cyclones_poleward_anticyclones_equatorward",
                "magnitude_order_consistent"):
        assert result["verdicts"][key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["pooled_poleward_t"] == pytest.approx(
        res["pooled_poleward_t"], rel=1e-12)
