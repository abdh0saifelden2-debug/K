"""Unit proofs for NR58 (`general_two_clocks/new_relationships35.py`): the
B.2 tempered-Warburg ice memory kernel (NR34) placed in the NR47-NR57
response anatomy.  It is pure minimum-phase memory -- phase 0 -> 45 deg
with the KK crossover EXACTLY at the tempering rate lam = 1/(4 tau_d) --
with no transport and no handedness, a new corner of the anatomy.
Offline-safe, fully analytic.

Covered: the exact memory-phase formula and its two poles; the crossover
identity delta(lam) = 22.5 deg; minimum-phase reconstruction and the
phase-area theorem; zero transport (flat excess phase) on the full
physical kernel; the tau_d-invariance of the crossover; the Warburg
half-order limit in both fractional-order faces.
"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships35 import (  # noqa: E402
    FIG, analyze, bode_phase_from_logamp, full_transfer, memory_phase,
    run, warburg_chi,
)


@pytest.fixture(scope="module")
def result():
    return analyze()


# --------------------------------------------------------------------------- #
# exact memory phase
# --------------------------------------------------------------------------- #
def test_memory_phase_formula():
    lam = 2.0
    w = np.geomspace(1e-3, 1e3, 500)
    assert np.allclose(memory_phase(w, lam),
                       np.angle(warburg_chi(w, lam)))


def test_two_poles(result):
    assert abs(result["delta_lo_deg"]) < 0.1        # elliptic
    assert abs(result["delta_hi_deg"] - 45.0) < 0.1  # Warburg


def test_crossover_at_tempering_rate(result):
    assert abs(result["delta_at_lam_deg"] - 22.5) < 1e-4
    assert result["verdicts"]["tempering_rate_is_kk_crossover"]


def test_crossover_invariant_in_tau_d():
    for tau_d in (0.1, 1.0, 37.0):
        lam = 1.0 / (4.0 * tau_d)
        assert math.degrees(memory_phase(lam, lam)) == pytest.approx(
            22.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# minimum-phase + area
# --------------------------------------------------------------------------- #
def test_minimum_phase_reconstruction(result):
    assert result["minphase_median_deg"] < 2.5


def test_phase_area_theorem(result):
    assert result["area_rel_err"] < 1e-3
    assert result["verdicts"]["ice_kernel_is_minimum_phase"]


# --------------------------------------------------------------------------- #
# transport = none
# --------------------------------------------------------------------------- #
def test_no_transport_clock(result):
    assert result["transport_delay_over_tau_d"] < 1e-3
    assert result["verdicts"]["ice_kernel_has_no_transport"]


def test_full_transfer_dc_pole_cancels():
    # the A/s and W sqrt(lam)/s poles cancel, leaving a finite real DC gain
    # (magnitude 2 tau_d); the overall sign is convention-dependent and does
    # not enter any anatomy verdict
    G0 = full_transfer(np.array([1e-6]), A=-1.0, tau_d=1.0)[0]
    assert abs(G0.imag) < 1e-4                 # finite: no residual 1/s pole
    assert abs(abs(G0.real) - 2.0) < 1e-3      # |DC| = 2 tau_d


# --------------------------------------------------------------------------- #
# Warburg half-order limit (NR34 / five faces)
# --------------------------------------------------------------------------- #
def test_warburg_half_order(result):
    assert abs(result["alpha_phase_hi"] - 0.5) < 1e-3
    assert abs(result["alpha_slope_hi"] - 0.5) < 1e-3
    assert result["verdicts"]["warburg_half_order_limit"]


def test_amplitude_slope_matches_phase_face():
    # at high omega, (2/pi) delta and d ln|chi|/d ln w both -> 1/2
    lam = 1.0
    w = np.geomspace(1e2, 1e5, 4000)
    chi = warburg_chi(w, lam)
    slope = np.gradient(np.log(np.abs(chi)), np.log(w))
    alpha_phase = 2.0 * np.angle(chi) / np.pi
    assert abs(slope[-1] - 0.5) < 1e-2
    assert abs(alpha_phase[-1] - 0.5) < 1e-2


# --------------------------------------------------------------------------- #
def test_all_verdicts_and_figure(result):
    for key in ("tempering_rate_is_kk_crossover",
                "ice_kernel_is_minimum_phase",
                "ice_kernel_has_no_transport",
                "warburg_half_order_limit"):
        assert result["verdicts"][key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["verdicts"]["tempering_rate_is_kk_crossover"] is True
    assert disk["delta_at_lam_deg"] == pytest.approx(
        res["delta_at_lam_deg"], rel=1e-12)
