"""Unit proofs for NR42 (`general_two_clocks/new_relationships19.py`), the grid-frequency
drainage window: NR32's band-limited two-clocks transmission and its interior-peak
criterion, transferred verbatim to the power-grid System Frequency Response.

Covered: the imported-functional identity (the grid uses the IDENTICAL NR32 transmission,
band integral, channelisation map and interior-peak criterion -- only the constants
differ); closed-form == quadrature in grid units; the swing-equation pole proof that the
two clocks are (T_g, M/beta); the interior-peak criterion predicting the observability
topology across a wide sweep and directly (interior iff the transition removes inertia);
the Markovian (quasi-static control) monotone wrong-ordering null; the RoCoF-vs-window
dichotomy (monotone vs single-peaked); the criterion boundary (the transition must cross
the band); and the calibration-free critical-inertia readout.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships19 import (  # noqa: E402
    BAND_GRID, grid_stiffness, nr42, primary_sfr_poles, rocof_initial,
)
import new_relationships9 as nr9  # noqa: E402
import new_relationships19 as nr19  # noqa: E402


@pytest.fixture(scope="module")
def out():
    return nr42()


def test_imported_functional_identity():
    """The cross-domain claim made concrete: the grid module reuses NR32's exact
    functional, band integral, channelisation map and interior-peak criterion."""
    assert nr19.transmission is nr9.transmission
    assert nr19.band_integral is nr9.band_integral
    assert nr19.interior_peak_criterion is nr9.interior_peak_criterion
    assert nr19.maps is nr9.maps
    assert nr19.t_peak is nr9.t_peak


def test_closed_form_matches_quadrature(out):
    assert out["closed_form_check"]["max_rel_err"] < 1e-6


def test_sfr_poles_are_the_two_clocks(out):
    """Swing-equation grounding: in the overdamped regime the slow SFR pole IS the
    inertial relaxation clock M/beta, and the fast pole is the governor clock T_g."""
    pc = out["sfr_pole_check"]
    assert pc["all_overdamped"] is True
    assert pc["max_rel_err"] < 0.08          # slow pole = M/beta to <8%
    for rec in pc["cases"]:
        assert rec["tau_slow_pole"] > rec["tau_fast_pole"]      # well separated
        assert abs(rec["tau_fast_pole"] - rec["T_g"]) / rec["T_g"] < 0.35


def test_pole_helpers_direct():
    beta = grid_stiffness(1.0, 2.0)
    assert beta == 3.0
    tau_slow, tau_fast, overdamped = primary_sfr_poles(24.0, 1.0, 2.0, 0.6)
    assert overdamped is True
    assert abs(tau_slow - 24.0 / beta) / (24.0 / beta) < 0.08
    assert rocof_initial(1.0, 20.0) == pytest.approx(0.05)     # RoCoF_0 = DP/M


def test_criterion_predicts_topology_across_sweep(out):
    assert out["criterion_predicts_all"] is True
    assert out["n_combos"] == 108
    assert out["interior_cases"]["n"] > 0


def test_interior_peak_iff_removes_storage():
    """Directly: with decades_C>0 the observability peak is interior; with decades_C=0
    (fixed inertia, stiffer control only) it is monotone with argmax at the high-inertia
    end -- the observable difference between removing inertia and merely adding control."""
    cs = np.linspace(0.0, 1.0, 241)
    T_rm = nr19.transmission(cs, tau_in=0.6, band=BAND_GRID, tau_dist=120.0,
                             decades_R=3.0, decades_C=1.0)
    i_rm = int(np.argmax(T_rm))
    assert 0 < i_rm < len(cs) - 1                              # interior
    T_fx = nr19.transmission(cs, tau_in=0.6, band=BAND_GRID, tau_dist=120.0,
                             decades_R=3.0, decades_C=0.0)
    assert int(np.argmax(T_fx)) == 0                           # argmax at highest inertia
    assert nr19.interior_peak_criterion(3.0, 1.0) is True
    assert nr19.interior_peak_criterion(3.0, 0.0) is False


def test_markovian_control_monotone_wrong_ordering(out):
    mk = out["markovian_control"]
    assert mk["monotone_decreasing"] is True
    assert mk["argmax_at_c0"] is True
    assert mk["interior_peak"] is False


def test_rocof_vs_window_dichotomy(out):
    rv = out["rocof_vs_window"]
    assert rv["rocof_monotone_increasing"] is True
    assert rv["rocof_argmax_at_c1"] is True                    # RoCoF worst at lowest inertia
    assert rv["window_interior"] is True                       # observability peaks inside
    assert 0.0 < rv["window_c_peak"] < 1.0


def test_criterion_boundary_needs_band_crossing(out):
    """A grid-exposed refinement: removing storage is necessary but not sufficient -- the
    transition must span enough decades that tau_sys crosses below the band."""
    cb = out["criterion_boundary"]
    assert cb["peak_rides_to_c1"] is True
    assert cb["tau_sys_at_c1_s"] > cb["band_high_tau_s"]       # never crossed the band


def test_critical_inertia_readout_knee_in_band(out):
    ci = out["critical_inertia_readout"]
    assert 0.0 < ci["c_star"] < 1.0
    assert 0.0 < ci["inertia_fraction_at_peak"] < 1.0          # a fraction of original M
    assert ci["knee_in_band"] is True                          # relaxation knee inside band
