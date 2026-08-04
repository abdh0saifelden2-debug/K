"""Tests for the tidal-admittance contamination study (P4b-R2/R3).

Validates (1) the phase-closure principle and inversion behaviour recomputed
from the module, and (2) the committed artifact
``validation/reports/tidal_admittance_contamination.json``.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "validation", "synthetic"))

import tidal_admittance_contamination as tc

ART = os.path.join(HERE, "validation", "reports",
                   "tidal_admittance_contamination.json")


def test_clean_phase_closure_and_recovery():
    r = tc.contaminated_response()
    tr = tc._truth()
    # memoryless response to a cosine tide: Chebyshev expansion, phases in
    # {0, pi} to machine precision
    assert r["closure_residual_1f"] < 1e-6
    assert r["closure_residual_2f"] < 1e-6
    assert abs(r["R_rec"] - tr["R_true"]) / tr["R_true"] < 0.05
    assert abs(r["m_rec"] - tr["m_true"]) / tr["m_true"] < 0.05


def test_lagged_flexure_flagged_at_fundamental():
    r = tc.contaminated_response(flex_amp=0.02)   # phase pi/3 (lagged) default
    assert r["closure_residual_2f"] < 0.05        # 2f stays closed (log-additive)
    assert r["closure_residual_1f"] > 0.05        # flagged at 1f
    tr = tc._truth()
    assert abs(r["R_rec"] - tr["R_true"]) / tr["R_true"] > 0.05  # real bias


def test_inphase_flexure_is_phase_invisible():
    # exactly in-phase flexure cannot be phase-flagged -> placement rule
    r = tc.contaminated_response(flex_amp=0.02, flex_phase=0.0)
    assert r["closure_residual_1f"] < 1e-6
    assert r["closure_residual_2f"] < 1e-6


def test_gl_migration_breaks_phase_closure():
    tr = tc._truth()
    r = tc.contaminated_response(gl_amp=0.02, gl_lag_cycles=0.15)
    assert r["closure_residual_2f"] > 0.20        # screened
    assert abs(r["R_rec"] - tr["R_true"]) / tr["R_true"] > 0.10  # fakes proximity
    for g in (0.005, 0.01):
        assert tc.contaminated_response(
            gl_amp=g, gl_lag_cycles=0.15)["closure_residual_2f"] > 0.20


def test_red_noise_scatter_bounded():
    mc = tc.red_noise_mc(sigma=0.01, n_mc=30)
    tr = tc._truth()
    assert abs(mc["R_median"] - tr["R_true"]) / tr["R_true"] < 0.10
    assert (mc["R_ci68"][1] - mc["R_ci68"][0]) / 2.0 < 0.10


@pytest.fixture(scope="module")
def art():
    with open(ART) as fh:
        return json.load(fh)


def test_artifact_verdict(art):
    v = art["verdict"]
    assert v["clean_phase_closed"] is True
    assert v["flexure_2f_phase_closed"] is True
    assert v["flexure_flagged_at_1f_when_lagged"] is True
    assert v["gl_2f_fakes_proximity"] is True
    assert v["gl_flagged_by_phase"] is True
    assert v["red_noise_R_ci68_halfwidth_1pct"] < 0.10
    assert art["clean"]["closure_residual_2f"] < 1e-9
