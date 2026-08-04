"""Unit proofs for NR57 (`general_two_clocks/new_relationships34.py`): the
spin is the irreversibility clock.  The Lagrangian areal velocity (odd
correlation slope, NR52) is the phase-space probability current, its
square lower-bounds the entropy production rate, and detailed balance
holds iff the spin vanishes; the deep ocean is time-irreversible with
opposite handedness across a reversible equatorial line.  Offline-safe:
reads the committed NR52 spin cache.

Covered: the exact current identity M = 2 Omega C; the isotropic entropy
law sigma = m^2/(2 c nu); the general anisotropic lower bound (no
violations, equality only isotropic); detailed balance <=> zero spin <=>
zero entropy; the lag-covariance asymmetry slope = -M; and the real-ocean
irreversibility (handed, hemisphere-antisymmetric, equatorial reversible
line) read from the committed spin cache.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships34 import (  # noqa: E402
    FIG, analyze, band_spin_t, current_rotation, entropy_production,
    load_spin, ou_solve, run, spin_bound, synthetic_checks,
)


@pytest.fixture(scope="module")
def syn():
    return synthetic_checks()


@pytest.fixture(scope="module")
def cache():
    return load_spin()


@pytest.fixture(scope="module")
def result(cache):
    return analyze(cache)


# --------------------------------------------------------------------------- #
# OU thermodynamics: exact identities
# --------------------------------------------------------------------------- #
def test_lyapunov_solution():
    A = np.array([[1.0, -2.0], [2.0, 1.0]])
    D = np.eye(2)
    C = ou_solve(A, D)
    assert np.allclose(A @ C + C @ A.T, 2 * D)


def test_current_identity_M_equals_2_Omega_C(syn):
    assert syn["m_identity_err"] < 1e-11


def test_isotropic_entropy_law(syn):
    assert syn["iso_identity_err"] < 1e-9


def test_entropy_production_positive_for_rotational_drift():
    A = np.array([[1.0, -2.0], [2.0, 1.0]])
    assert entropy_production(A, np.eye(2)) > 0


def test_detailed_balance_iff_zero_spin(syn):
    assert syn["db_spin"] < 1e-9
    assert syn["db_sigma"] < 1e-9


def test_symmetric_drift_has_no_current():
    As = np.array([[2.0, 0.5], [0.5, 1.0]])
    _, Om, M = current_rotation(As, np.eye(2))
    assert abs(M[1, 0]) < 1e-9
    assert entropy_production(As, np.eye(2)) < 1e-12


def test_lag_covariance_slope_is_minus_M(syn):
    assert syn["lag_slope_err"] < 1e-4


# --------------------------------------------------------------------------- #
# the lower bound
# --------------------------------------------------------------------------- #
def test_spin_squared_lower_bounds_entropy(syn):
    assert syn["bound_violations"] == 0
    assert syn["bound_min_ratio"] >= 1.0 - 1e-6
    assert syn["n_bound_tested"] > 500


def test_bound_is_tight_in_isotropic_limit():
    # isotropic C and D -> sigma equals the bound exactly
    A = np.array([[1.0, -1.7], [1.7, 1.0]])
    D = np.eye(2)
    C, _, M = current_rotation(A, D)
    assert entropy_production(A, D) == pytest.approx(
        spin_bound(M, C, D), rel=1e-9)


# --------------------------------------------------------------------------- #
# real ocean
# --------------------------------------------------------------------------- #
def test_band_spin_helper(cache):
    nh = band_spin_t(cache, "nh")
    assert nh["irreversibility_floor"] == pytest.approx(
        nh["rho1_mean"] ** 2)
    assert nh["cycling_rate_per_day"] == pytest.approx(
        nh["rho1_mean"] / 10.0)


def test_ocean_irreversible_and_handed(result):
    r = result["real"]
    assert r["nh"]["t"] < -4.0 and r["sh"]["t"] > 4.0
    assert r["nh"]["rho1_mean"] * r["sh"]["rho1_mean"] < 0
    assert r["nh"]["irreversibility_floor"] > 0
    assert result["verdicts"]["ocean_is_time_irreversible_handed"]


def test_equatorial_reversible_line(result):
    assert abs(result["real"]["eq"]["t"]) < 2.0
    assert result["verdicts"]["equatorial_reversible_line"]


def test_all_verdicts_and_figure(result):
    for key in ("current_identity_and_isotropic_law_exact",
                "spin_squared_lower_bounds_entropy",
                "detailed_balance_iff_zero_spin",
                "ocean_is_time_irreversible_handed",
                "equatorial_reversible_line"):
        assert result["verdicts"][key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["verdicts"]["ocean_is_time_irreversible_handed"] is True
    assert disk["real"]["nh"]["t"] == pytest.approx(
        res["real"]["nh"]["t"], rel=1e-12)
