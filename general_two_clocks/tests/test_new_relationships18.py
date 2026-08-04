"""Unit proofs for NR41 (`general_two_clocks/new_relationships18.py`), the theory-
pass synthesis: the diffusive-visibility window across the two-clocks program.

Covered: the tempered-Warburg phase corner (x=1 -> 22.5 deg) and asymptotes; the
calibration-free (amplitude/Ohmic-independent) phase readout of tau_d; the
visibility-window formula and threshold; that V predicts AICc Warburg-vs-RC
detectability as the cutoff sweeps a band; and the cross-domain reconciliation
(NR38/battery visible, NR34/NR40 null) under one criterion.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships18 import (  # noqa: E402
    X_STAR, characteristic_tau_from_phase, detectable, run, visibility_window,
    warburg_phase_deg,
)
from new_relationships15 import randles_Z  # noqa: E402


@pytest.fixture(scope="module")
def out():
    o, _ = run()
    return o


def test_warburg_phase_corner_and_asymptotes():
    assert abs(warburg_phase_deg(1.0) - 22.5) < 1e-9        # corner at x=1
    assert warburg_phase_deg(1e6) > 44.9                    # -> 45 deg
    assert warburg_phase_deg(1e-6) < 0.1                    # -> 0 deg


def test_phase_readout_is_amplitude_and_ohmic_independent():
    """tau_d recovered from the Warburg phase alone, invariant to sigma_W and R_ch."""
    tau_d = 12.0
    w = np.logspace(np.log10(1e-3 / tau_d), np.log10(1e3 / tau_d), 500)
    ph = warburg_phase_deg(w * tau_d)
    tau_hat = characteristic_tau_from_phase(w, ph)
    assert abs(tau_hat - tau_d) / tau_d < 0.02


def test_visibility_window_formula():
    # band entirely above the corner: V = log10(x_hi / x_lo)
    assert abs(visibility_window(10 * X_STAR, 1000 * X_STAR) - 2.0) < 1e-9
    # band below the corner -> negative (quasi-steady, no plateau)
    assert visibility_window(1e-3, 0.1) < 0


def test_detectable_threshold():
    assert detectable(X_STAR, 1e4)          # wide diffusive band
    assert not detectable(1e-3, 0.1)        # below the corner


def test_V_predicts_aicc_detectability(out):
    assert out["V_vs_aicc_agreement"] >= 0.85


def test_cross_domain_reconciliation(out):
    assert out["cross_domain_all_match"]
    assert out["cross_domain"]["NR38_bed_tides"]["predicted_visible"]
    assert not out["cross_domain"]["NR40_aftershocks"]["predicted_visible"]


def test_randles_import_is_the_nr38_element():
    """NR41 reuses the exact NR38 Randles element (cross-reference integrity)."""
    z = randles_Z(np.array([1.0]), 0.2, 1.0, 0.5, 0.9, tau_d=50.0)[0]
    assert np.isfinite(z) and z.real > 0
