"""Unit proofs for NR55 (`general_two_clocks/new_relationships32.py`): the
response-anatomy capstone.  Four independent phases (memory, transport,
shape parity, handedness), four guard theorems, and the corpus's real
systems occupying distinct corners of the phase space -- every matrix
entry read from a committed NR47-NR54 artifact and gated.
Offline-safe: reads committed figures only.

Covered: all eight source figures exist and load; the matrix is assembled
from the exact committed values (spot-checked against the raw files);
each corner verdict (EEG elliptic, ocean parabolic+odd, solar pure
transport, scallops transport+weak parity); cross-face consistency
(NR48's tau_d equals NR50's fitted tau within tolerance; NR52's spin
signs match NR54's compass hemispheres); figure write-through.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships32 import (  # noqa: E402
    FIG, FIGDIR, SOURCES, analyze, assemble_matrix, load_sources, run,
)


@pytest.fixture(scope="module")
def src():
    return load_sources()


@pytest.fixture(scope="module")
def result():
    return analyze()


def test_all_sources_exist():
    for name in SOURCES.values():
        assert os.path.exists(os.path.join(FIGDIR, name)), name


def test_matrix_matches_committed_values(src):
    mat = assemble_matrix(src)
    assert mat["eeg"]["memory_deg"] == pytest.approx(
        src["nr47"]["reads"]["eeg"]["delta_deg_near"])
    assert mat["solar"]["tau_d_s"] == pytest.approx(
        src["nr48"]["delay_tau_d_s"])
    assert mat["scallops"]["celerity_mm_hr"] == pytest.approx(
        src["nr49"]["frames"]["lab"]["pattern"]["celerity_mm_hr"])
    assert mat["ocean"]["spin_nh_t"] == pytest.approx(
        src["nr52"]["stats"]["nh_all"]["t_stat"])


def test_eeg_elliptic(result):
    assert abs(result["matrix"]["eeg"]["memory_deg"]) < 5.0
    assert result["verdicts"]["eeg_at_elliptic_pole"]


def test_ocean_parabolic_plus_odd(result):
    o = result["matrix"]["ocean"]
    assert 85.0 < o["memory_deg"] < 95.0
    assert o["minphase_median_deg"] < 6.0
    assert o["spin_nh_t"] < -4.0 < 4.0 < o["spin_sh_t"]
    assert abs(o["spin_eq_t"]) < 2.0
    assert 30.0 < o["eddy_period_nh_d"] < 120.0
    assert result["verdicts"]["ocean_parabolic_plus_odd"]


def test_solar_pure_transport(result):
    s = result["matrix"]["solar"]
    assert 5.0 < abs(s["tau_d_s"]) < 15.0
    assert s["gain_log_std"] < 0.6
    assert s["static_gain_veto_x"] > 2.0
    assert result["verdicts"]["solar_pure_transport"]


def test_scallops_transport_weak_shape(result):
    c = result["matrix"]["scallops"]
    assert 30.0 < c["celerity_mm_hr"] < 70.0
    assert c["boost_removable_frac"] > 0.8
    assert abs(c["t_shape_parity"]) < 0.6 * abs(c["t_flux_parity"])
    assert result["verdicts"]["scallops_transport_weak_shape"]


def test_cross_face_consistency(src):
    # NR48's fitted delay equals NR50's independent fit on the same cache
    assert src["nr50"]["solar"]["tau_d_s"] == pytest.approx(
        src["nr48"]["delay_tau_d_s"], abs=0.5)
    # NR52 spin signs match NR54's compass hemisphere conventions
    assert src["nr52"]["stats"]["nh_all"]["rho1_mean"] < 0
    assert src["nr52"]["stats"]["sh_all"]["rho1_mean"] > 0
    assert src["nr54"]["pooled_poleward_t"] > 3.0
    # NR53's tropical eddy chirality agrees with NR52's hemispheric signs
    assert (src["nr53"]["bands"]["nh_trop"]["fit"]["f_e_rad_day"]
            * src["nr53"]["bands"]["sh_trop"]["fit"]["f_e_rad_day"]) < 0


def test_all_verdicts_and_figure(result):
    for key in ("eeg_at_elliptic_pole", "ocean_parabolic_plus_odd",
                "solar_pure_transport", "scallops_transport_weak_shape",
                "four_corners_distinct"):
        assert result["verdicts"][key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["verdicts"]["four_corners_distinct"] is True
    assert disk["matrix"]["solar"]["tau_d_s"] == pytest.approx(
        res["matrix"]["solar"]["tau_d_s"], rel=1e-12)
