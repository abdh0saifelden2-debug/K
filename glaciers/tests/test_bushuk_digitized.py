"""Tests for the Bushuk figure digitization (bushuk_fig3_digitize.py).

The digitizer needs network + poppler at runtime, so these tests validate:

  1. the pure calibration helpers (PiecewiseAxis, match_anchors) on synthetic
     inputs, and
  2. the committed artifact ``figures/58b_bushuk_digitized.json`` -- schema,
     provenance pin, kinematic sanity, internal consistency, and the honest
     verdict (digitized I sits ABOVE the solver band; the old figure-read
     bound's tau/lambda inputs are superseded).
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bushuk_fig3_digitize as bz

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "figures", "58b_bushuk_digitized.json")


def test_piecewise_axis_interpolates_and_extrapolates():
    ax = bz.PiecewiseAxis([(100.0, 10.0), (200.0, 9.0), (400.0, 8.0)])
    assert ax(100.0) == pytest.approx(10.0)
    assert ax(150.0) == pytest.approx(9.5)
    assert ax(300.0) == pytest.approx(8.5)
    # extrapolation continues the edge slopes
    assert ax(50.0) == pytest.approx(10.5)
    assert ax(500.0) == pytest.approx(7.5)
    out = ax(np.array([100.0, 200.0, 400.0]))
    assert np.allclose(out, [10.0, 9.0, 8.0])


def test_match_anchors_requires_geometry_match():
    anchors = [(100.0, 10.0), (200.0, 9.0), (300.0, 8.0),
               (400.0, 7.0), (500.0, 6.0)]
    cents = [101.0, 199.5, 300.2, 401.0, 499.0, 777.0]
    matched = bz.match_anchors(cents, anchors, tol_px=8.0, min_match=5)
    assert len(matched) == 5
    assert matched[0][1] == 10.0
    with pytest.raises(RuntimeError):
        bz.match_anchors([c + 40.0 for c in cents], anchors, min_match=5)


@pytest.fixture(scope="module")
def art():
    with open(ART) as fh:
        return json.load(fh)


def test_artifact_provenance_pinned(art):
    assert art["provenance"]["sha256"] == bz.PDF_SHA256
    assert art["provenance"]["dpi"] == 600
    assert art["provenance"]["amp_calibration"]["n_anchor_matched"] >= 5
    assert art["provenance"]["crest_calibration"]["n_anchor_matched"] >= 5


def test_amplitude_series_sane(art):
    a = art["amplitude"]
    assert a["n"] >= 40
    assert len(a["t_min"]) == len(a["A_mm"]) == a["n"]
    assert a["beta_per_min"] > 0.0                     # damps
    assert a["A_early_mm"] > a["A_late_mm"]            # net decay
    assert 500.0 < a["tau_envelope_min"] < 3000.0
    # oscillatory series: the exponential fit is honest but weak
    assert 0.0 < a["r2"] < 0.8


def test_oscillator_refit_preferred(art):
    a = art["amplitude"]
    o = a["oscillator"]
    # physically-motivated model: exponential envelope x periodic modulation
    assert 1000.0 < o["tau_min"] < 5000.0
    assert 20.0 < o["T_min"] < 300.0                   # interior of the grid
    assert 0.0 < o["b"] < 0.2                          # small modulation depth
    assert o["r2"] > a["r2"]                           # beats single exponential
    lo68, hi68 = o["tau_ci68_min"]
    lo95, hi95 = o["tau_ci95_min"]
    assert lo95 <= lo68 < o["tau_min"] < hi68 <= hi95
    assert o["n_boot_kept"] >= 1900
    # two-exponential control degenerates and fits worse
    te = a["two_exp_control"]
    assert te["failed"] is False
    assert te["degenerate"] is True
    assert te["r2"] < o["r2"]


def test_crest_kinematics_sane(art):
    c = art["crest"]
    assert c["n_crests"] >= 5
    assert 0.05 < c["c_mig_mm_per_min"] < 0.35         # downstream, cm/h scale
    assert 15.0 < c["lam_mm"] < 40.0
    assert all(10.0 < s < 45.0 for s in c["lam_spacings_mm"])


def test_index_consistency_and_verdict(art):
    k = art["kinematics_si"]
    I = art["I"]
    recomputed = k["tau_envelope_s"] * k["c_published_m_per_s"] / k["lam_m"]
    assert I["combos"]["tau_env_c_pub"] == pytest.approx(recomputed, rel=1e-9)
    assert I["point"] == pytest.approx(I["combos"]["tau_env_c_pub"], rel=1e-9)
    pref = k["tau_oscillator_s"] * k["c_published_m_per_s"] / k["lam_m"]
    assert I["point_preferred"] == pytest.approx(pref, rel=1e-9)
    assert I["point_preferred"] == pytest.approx(I["combos"]["tau_osc_c_pub"], rel=1e-9)
    lo, hi = I["estimator_range"]
    assert lo <= I["point"] <= hi
    assert lo <= I["point_preferred"] <= hi
    # the honest verdict: every estimator combination sits ABOVE the solver
    # band -- the old figure-read bound (~0.10, from tau 1-3 h and lam 13 cm)
    # is superseded by regime-matched digitized kinematics
    band_lo, band_hi = I["solver_band"]
    assert lo > band_hi
    assert I["inside_solver_band"] is False
    assert I["downstream"] is True
    assert art["verdict"]["migrates_downstream"] is True
    assert art["verdict"]["damps"] is True
    assert art["verdict"]["oscillator_r2"] > art["verdict"]["exponential_decay_r2"]
