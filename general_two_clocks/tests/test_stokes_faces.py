"""Unit proofs for NR56 (`general_two_clocks/new_relationships33.py`): the
Lagrangian velocity has Stokes parameters -- waves are linearly polarized
(stretch Q,U; zero spin), vortices circularly polarized (spin V; zero
stretch) -- and the deep ocean crosses from wave-polarized at the equator
(deep jets, Q/I = +0.44) to vortex-polarized poleward, with a polar
meridional flip.  Offline-safe: committed cache
``data/nr56_stokes_cache.json``.

Covered: the plane-wave rectilinear theorem (V = 0 to machine precision,
polarization angle exact, 2-theta rotation covariance); the vortex
circular identity (V(lag1) = sin(f dt) exactly, Q,U -> 0); mirror parity
(Q fixed, U and V flip); estimator helpers; cache schema; the equatorial
wave polarization, monotone poleward spin growth, the 20-35 deg
crossover, and the polar flip.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships33 import (  # noqa: E402
    BANDS, CACHE, FIG, analyze, band_stats, load_cache,
    polarization_angle_deg, run, stokes_zero_lag, synthetic_checks,
)


@pytest.fixture(scope="module")
def syn():
    return synthetic_checks()


@pytest.fixture(scope="module")
def cache():
    return load_cache()


@pytest.fixture(scope="module")
def result(cache):
    return analyze(cache)


# --------------------------------------------------------------------------- #
# polarization theorems
# --------------------------------------------------------------------------- #
def test_plane_wave_is_rectilinear(syn):
    assert syn["wave_V"] < 1e-12
    assert syn["wave_angle_err_deg"] < 1e-6


def test_two_theta_rotation_covariance(syn):
    assert syn["rotation_2theta_err_deg"] < 1e-6


def test_vortex_is_circular(syn):
    assert syn["vortex_QU"] < 1e-3
    assert syn["vortex_V_err"] < 1e-6 and syn["vortex_V"] > 0


def test_mirror_parity(syn):
    assert syn["mirror_Q_err"] < 1e-12       # Q parity-even
    assert syn["mirror_U_flip_err"] < 1e-12  # U flips
    assert syn["mirror_V_flip_err"] < 1e-12  # V flips


def test_stokes_helpers():
    # pure zonal oscillation: Q = +1, U = 0
    t = np.linspace(0, 40 * np.pi, 4001)
    Q, U = stokes_zero_lag(np.cos(t), np.zeros_like(t))
    assert Q == pytest.approx(1.0) and U == pytest.approx(0.0)
    assert polarization_angle_deg(1.0, 0.0) == pytest.approx(0.0)
    assert polarization_angle_deg(-1.0, 0.0) == pytest.approx(90.0)


def test_band_stats_basic():
    lat = np.array([1.0, 2.0, -3.0, 40.0])
    st = band_stats(lat, np.array([1.0, 1.0, 1.0, 5.0]), 0, 5)
    assert st["n"] == 3 and st["mean"] == 1.0


# --------------------------------------------------------------------------- #
# committed cache
# --------------------------------------------------------------------------- #
def test_cache_schema(cache):
    f = cache["floats"]
    n = cache["meta"]["n_floats"]
    assert n > 7000
    for k in ("lat", "Q", "U", "V"):
        assert len(f[k]) == n
    assert np.all(np.abs(f["Q"]) <= 1.0 + 1e-9)


# --------------------------------------------------------------------------- #
# real-data verdicts
# --------------------------------------------------------------------------- #
def test_equator_wave_polarized(result):
    b = result["bands"]["0-5"]
    assert b["stretch"]["mean"] > 0.3 and b["stretch"]["t"] > 20.0
    assert result["verdicts"]["equator_is_wave_polarized"]


def test_vortex_polarization_grows_poleward(result):
    spins = [result["bands"][f"{lo}-{hi}"]["spin_abs"]["mean"]
             for lo, hi in BANDS]
    assert spins[0] == min(spins) and spins[-1] == max(spins)
    assert result["verdicts"]["vortex_polarization_grows_poleward"]


def test_crossover_and_polar_flip(result):
    assert result["verdicts"]["wave_vortex_crossover_20_35"]
    b = result["bands"]["65-85"]
    assert b["stretch"]["mean"] < 0 and b["stretch"]["t"] < -2.0
    assert result["verdicts"]["polar_meridional_flip"]


def test_all_verdicts_and_figure(result):
    for key in ("polarization_theorems_exact",
                "equator_is_wave_polarized",
                "vortex_polarization_grows_poleward",
                "wave_vortex_crossover_20_35", "polar_meridional_flip"):
        assert result["verdicts"][key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["verdicts"]["equator_is_wave_polarized"] is True
    assert disk["bands"]["0-5"]["stretch"]["mean"] == pytest.approx(
        res["bands"]["0-5"]["stretch"]["mean"], rel=1e-12)
