"""Consistency tests for the P0-R5 Ra sweep artifact (figures/15b_rb_ra_sweep.json)."""

import json
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).resolve().parent.parent / "figures" / "15b_rb_ra_sweep.json"


@pytest.fixture(scope="module")
def sweep():
    with open(ARTIFACT) as f:
        return json.load(f)


def test_three_ra_at_pr1(sweep):
    assert [r["Ra"] for r in sweep] == [1e6, 1e7, 1e8]
    assert all(r["Pr"] == 1.0 for r in sweep)
    assert all(r["n_samples"] == 30 for r in sweep)


def test_taylor_ratio_above_one_and_widening(sweep):
    med = [r["lambda_ratio_median"] for r in sweep]
    assert all(m > 1.5 for m in med)
    assert med[0] < med[1] < med[2], "split should widen with Ra"
    # 5th percentile stays above 1 at every Ra: the split never closes.
    assert all(r["lambda_ratio_p5"] > 1.0 for r in sweep)


def test_paper_headline_anchor(sweep):
    """paper0's 2.40x (Ra=1e6, traj 0, t=100) is reproduced inside the sweep."""
    r6 = sweep[0]
    s = next(x for x in r6["samples"] if x["traj"] == 0 and x["time_idx"] == 100)
    assert s["lambda_ratio"] == pytest.approx(2.40, abs=0.02)
    assert r6["lambda_ratio_p5"] <= 2.40 <= r6["lambda_ratio_p95"]


def test_integral_ratio_stays_order_one(sweep):
    for r in sweep:
        assert 0.8 < r["L_ratio_median"] < 2.0


def test_highk_suppression_grows(sweep):
    sup = [r["highk_suppression_median"] for r in sweep]
    assert all(s > 1e2 for s in sup)
    assert sup[0] < sup[2]
