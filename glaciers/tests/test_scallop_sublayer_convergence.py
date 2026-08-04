"""Tests for scallop_sublayer_convergence.py (Paper 2 convergence study).

The deterministic test pins the seed-ensemble aggregator ``_agg`` (means, std,
and the robustness fractions reported per grid point).  A short CPU smoke
confirms ``one()`` returns the sublayer decomposition with the grid/penalization
tags and a finite suppression ratio.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scallop_sublayer_convergence as cv


def _st(nu, thicken, convex, seed):
    return {"ny": 96, "eta": 5e-5, "dy": 0.065, "seed": seed,
            "nu_ratio": nu, "thicken": thicken, "convex": convex,
            "umax_bump": 2.0}


def test_agg_means_and_fractions():
    stats = [_st(0.80, 1.5, 1.2, 0), _st(0.90, 1.4, 1.3, 1), _st(1.05, 1.1, 1.2, 2)]
    a = cv._agg(stats)
    assert abs(a["nu_mean"] - (0.80 + 0.90 + 1.05) / 3.0) < 1e-12
    assert abs(a["nu_std"] - np.std([0.80, 0.90, 1.05])) < 1e-12
    assert abs(a["nu_min"] - 0.80) < 1e-12 and abs(a["nu_max"] - 1.05) < 1e-12
    # two of three seeds have Nu<1; two of three have thicken>convex
    assert abs(a["frac_nu_lt_1"] - 2.0 / 3.0) < 1e-12
    assert abs(a["frac_thicken_gt_convex"] - 2.0 / 3.0) < 1e-12
    assert a["seeds"] == [0, 1, 2]


def test_one_smoke():
    st = cv.one(48, 32, 5e-5, 12, 1.5, 0.4, 0.20, 120, 60, 0, np)
    for k in ("nu_ratio", "thicken", "convex", "nu_lt_1", "ny", "eta", "dy"):
        assert k in st
    assert np.isfinite(st["nu_ratio"]) and st["ny"] == 32
    assert isinstance(bool(st["nu_lt_1"]), bool)
