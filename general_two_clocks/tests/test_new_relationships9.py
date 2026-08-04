"""Unit tests for the new derived relationship NR32
(general_two_clocks/new_relationships9.py). Deterministic, CPU-only, no data needed
(the optional ICECAP specularity anchor is exercised only when the local download exists).

NR32: the drainage-response window -- band-limited hydraulic transmission is peaked at the
cavity<->channel transition IFF channelization removes storage (decades_C>0); the Markovian
(delta-kernel) collapse destroys the peak and predicts the wrong ordering; the lag-to-peak at
the transmission maximum falls inside the observable 0.02-2 yr window; Thw_142's detection +
lag jointly select the transition neighbourhood.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import new_relationships9 as NR9  # noqa: E402

_RES = None


def _res():
    global _RES
    if _RES is None:
        _RES = NR9.nr32(n_c=121)
    return _RES


# --- closed forms ----------------------------------------------------------- #
def test_band_integral_closed_form_matches_quadrature():
    for ta, tb in ((0.05, 3.0), (0.1, 0.1), (0.02, 40.0), (0.3, 0.005), (2.0, 2.0)):
        exact = NR9.band_integral(ta, tb)
        num = NR9.band_integral_numeric(ta, tb)
        assert abs(exact - num) < 1e-6 * num


def test_t_peak_matches_numeric_argmax():
    t = np.linspace(1e-6, 60.0, 4_000_001)
    for ta, tb in ((0.25, 3.0), (0.1, 0.5), (1.0, 10.0)):
        g = (np.exp(-t / tb) - np.exp(-t / ta)) / (tb - ta)
        assert abs(t[np.argmax(g)] - NR9.t_peak(ta, tb)) < 5e-4


# --- the interior-peak existence criterion (the NR32 dichotomy) -------------- #
def test_criterion_predicts_topology_everywhere():
    r = _res()
    assert r["criterion_predicts_all"] is True
    assert r["n_combos"] == 108


def test_no_storage_removal_means_no_interior_peak():
    cs = np.linspace(0, 1, 121)
    T = NR9.transmission(cs, decades_C=0.0)
    assert int(np.argmax(T)) == 0          # distributed end dominates
    T1 = NR9.transmission(cs, decades_C=1.0)
    i = int(np.argmax(T1))
    assert 0 < i < 120                     # storage removal -> interior peak
    assert T1[i] > 3.0 * max(T1[0], T1[-1])  # and it is prominent on the central map


def test_peak_sits_where_lag_crosses_the_window():
    r = _res()
    assert r["interior_cases"]["all_in_window"] is True
    assert r["interior_cases"]["n"] > 0


# --- the Markovian (adiabatic) control --------------------------------------- #
def test_markovian_collapse_destroys_the_peak():
    r = _res()["markovian_control"]
    assert r["monotone_decreasing"] is True
    assert r["argmax_at_c0"] is True
    assert r["interior_peak"] is False


# --- Thw_142: the one in-band detection, read honestly ------------------------ #
def test_thw142_lag_excludes_mature_channels():
    r = _res()["thw142"]
    # a same-quarter response bound: tau_sys below ~the drawdown quarter is excluded
    assert 0.1 < r["lag_floor_tau_sys_yr"] < 0.5


def test_thw142_joint_interval_is_interior_transition():
    r = _res()["thw142"]
    lo, hi = r["joint_c_interval"]
    dlo, dhi = r["detectable_c_interval_central_map"]
    assert 0.0 < lo < hi < 1.0             # interior neighbourhood, not an endpoint
    assert hi <= dhi + 1e-9                # the lag cuts the channelized side
    assert abs(lo - dlo) < 1e-6            # ... and only that side


# --- optional data anchor ------------------------------------------------------ #
def test_spec_anchor_structure_when_available():
    sa = _res()["spec_anchor"]
    assert isinstance(sa, dict)
    if sa.get("available"):
        assert 0.0 <= sa["coverage_median_spec"] <= 1.0
        for rec in sa["lakes"].values():
            if rec["n_pts"]:
                assert 0.0 <= rec["mean_spec"] <= 1.0
