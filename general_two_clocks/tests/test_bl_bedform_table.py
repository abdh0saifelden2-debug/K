"""Tests for NR44 (ledger E8) -- the B_L bedform-drag table
(new_relationships21.py): Carey-1966 field ice ripples vs the Blumberg-Curl
1974 dissolution-scallop log-law constant."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import new_relationships21 as nr


@pytest.fixture(scope="module")
def rows():
    return nr.build_table()


@pytest.fixture(scope="module")
def v(rows):
    return nr.verdicts(rows)


# --------------------------------------------------------------------------- #
# transcription integrity: every derived hydraulic column reproduces Carey
# --------------------------------------------------------------------------- #
def test_hydraulic_radii_consistent(rows):
    """R_I = A_I / P_I must match the published hydraulic radii we
    cross-checked (Jan 5: 0.673 ft; Feb 18: 0.995; Mar 6: 1.44)."""
    by = {r["date"]: r for r in rows}
    assert by["1965-01-05"]["R_I_ft"] == pytest.approx(0.673, abs=0.002)
    assert by["1965-02-18"]["R_I_ft"] == pytest.approx(0.995, abs=0.002)
    assert by["1965-03-06"]["R_I_ft"] == pytest.approx(1.445, abs=0.002)


def test_manning_nI_reproduces_published_captions(rows, v):
    """The recomputed Manning n_I must reproduce every published value
    (table + photo captions) to <2%; the committed max error is ~0.3%."""
    assert v["reconstruction_max_nI_err"] < 0.02
    by = {r["date"]: r for r in rows}
    # photo-caption anchors (figs 1, 5, 7, 8)
    assert by["1965-01-05"]["n_I"] == pytest.approx(0.0151, rel=0.02)
    assert by["1965-02-18"]["n_I"] == pytest.approx(0.0229, rel=0.02)
    assert by["1965-03-06"]["n_I"] == pytest.approx(0.0281, rel=0.02)
    assert by["1965-03-20"]["n_I"] == pytest.approx(0.0244, rel=0.02)


def test_friction_factor_reproduces_published(rows):
    """f_I from 8 g R_I S / V^2 must reproduce the published Darcy-Weisbach
    column (Jan 5: .0304; Feb 3: .0288; Mar 6: .0815)."""
    by = {r["date"]: r for r in rows}
    assert by["1965-01-05"]["f_I"] == pytest.approx(0.0304, rel=0.02)
    assert by["1965-02-03"]["f_I"] == pytest.approx(0.0288, rel=0.02)
    assert by["1965-03-06"]["f_I"] == pytest.approx(0.0815, rel=0.02)


def test_rouse_inversion_roundtrip():
    """k_s -> f -> k_s must round-trip through the Rouse rough-pipe law."""
    R = 1.0
    for ks_true in (0.01, 0.1, 0.5):
        f = (2.0 * np.log10(2.0 * R / ks_true) + 1.74) ** -2
        ks = 2.0 * R / 10.0 ** ((1.0 / np.sqrt(f) - 1.74) / 2.0)
        assert ks == pytest.approx(ks_true, rel=1e-12)


# --------------------------------------------------------------------------- #
# the NR44 verdicts
# --------------------------------------------------------------------------- #
def test_any_state_universality_falsified(v):
    assert v["any_state_universality_falsified"] is True
    assert v["B_L_developing"][0] > nr.BC_ROW["B_L"] + 3.0
    assert v["max_z0_error_if_9p4_applied"] > 10.0


def test_developed_attractor(v):
    """Five successive developed rows cluster tightly (spread <= 2.5) just
    above the Blumberg-Curl constant, well below the developing phase."""
    assert v["mature_attractor_supported"] is True
    lo, hi = v["B_L_developed"]
    assert 9.4 <= lo <= 11.0 and hi <= 12.5
    assert v["B_L_developed_spread"] <= 2.5
    assert v["B_L_developed_median"] < v["B_L_developing"][0]
    assert v["all_developed_fully_rough"] is True


def test_ks_over_h_approaches_scallop_class(v):
    lo_s, hi_s = v["ks_over_h_scallops"]
    lo_d, hi_d = v["ks_over_h_developed"]
    assert hi_d > 1.0                       # roughness exceeds the relief
    assert hi_d < hi_s                      # ...but from below the scallops
    assert hi_d >= 0.5 * lo_s               # within ~2x at closest approach


def test_drag_not_steepness_controlled(v):
    assert v["steepness_controls_drag"] is False
    assert abs(v["rho_steepness_drag"]) < 0.4


def test_developed_Re_star_in_selection_band(v):
    lo, hi = v["Re_star_developed"]
    assert 3100 <= lo and hi <= 6300 * 1.05  # Thorsness-Hanratty band


def test_transitional_flags_direction(rows):
    """Transitional rows (k_s+ < 70) exist only in the developing phase, so
    their fully-rough inversion (k_s overestimated, B_L underestimated) can
    only WIDEN the developing-vs-9.4 gap -- the falsification is one-sided."""
    for r in rows:
        if "B_L" in r and not r["fully_rough"]:
            assert r["date"] < "1965-02-18"


def test_outlier_excluded(rows):
    ge = [r for r in rows if not r["outlier"] and "B_L" in r]
    assert len(ge) == 11
    assert all(r["date"] != "1965-02-24" for r in ge)
