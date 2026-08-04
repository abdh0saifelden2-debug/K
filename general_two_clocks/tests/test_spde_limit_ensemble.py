"""Tests for the projection/SPDE-limit surrogate ensemble (P0-R6).

Validates the committed artifact ``figures/24_spde_limit_ensemble.json``:
the paper's structural-ceiling claim must be ensemble-backed, not a single
favourable seed.
"""
import json
import os

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "figures", "24_spde_limit_ensemble.json")


@pytest.fixture(scope="module")
def art():
    with open(ART) as fh:
        return json.load(fh)


def test_ensemble_size_and_schema(art):
    assert art["n_surrogates"] >= 100
    for k in ("corr_abs_median", "corr_abs_p95", "rms_surr_median",
              "rms_surr_ci95", "rms_proj", "single_realization"):
        assert k in art


def test_correlation_small_but_honestly_scattered(art):
    # the claim is "geometrically unrelated", not "correlation exactly 0.01":
    # a few large-scale modes dominate, so single draws scatter
    assert art["corr_abs_median"] < 0.15
    assert art["corr_abs_p95"] < 0.35
    lo, hi = art["corr_ci95"]
    assert lo < 0.0 < hi                       # centred on zero


def test_divergence_ceiling_is_structural(art):
    # projection: machine zero; surrogate: leaves div O(1e-2) with a tight
    # ensemble interval -- the structural ceiling is not seed luck
    assert art["rms_proj"] < 1e-12
    lo, hi = art["rms_surr_ci95"]
    assert 1e-3 < lo <= art["rms_surr_median"] <= hi < 1.0
    assert (hi - lo) / art["rms_surr_median"] < 0.5
