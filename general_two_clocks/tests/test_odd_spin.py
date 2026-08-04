"""Unit proofs for NR52 (`general_two_clocks/new_relationships29.py`): the
odd completion of the two-clocks response.  The antisymmetric (Lagrangian
spin) correlation is the parity-odd face that every scalar contraction
(MSD/GSER) projects out; for the damped rotator it is exactly
``sin(f tau) e^{-tau/tau_L}`` with positive spin = counterclockwise; on
the real ANDRO deep floats the chirality flips sign across the equator
with an equatorial null, anticyclonic in both hemispheres.
Offline-safe: reads the committed cache ``data/nr52_odd_spin_cache.json``.

Covered: the exact matrix-exponential factorization; the CCW sign anchor;
mirrored-realization pairing (MSD identical sample-by-sample, spin flipped
exactly); lag reversal = transpose (doubly odd); estimator vs analytic
rho_odd; committed cache schema; hemispheric sign flip, equatorial null,
anticyclonic dominance, and the odd-memory decay scale.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships29 import (  # noqa: E402
    CACHE, EPS, FIG, analyze, load_cache, odd_even_accumulate, odd_part,
    rotator_correlation, run, spin_stats, synthetic_checks,
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
# exact structure of the damped rotator
# --------------------------------------------------------------------------- #
def test_eps_is_ccw_generator():
    # z-cross convention: EPS @ (1,0) = (0,1): x-hat rotates toward y-hat
    assert np.allclose(EPS @ np.array([1.0, 0.0]), [0.0, 1.0])


def test_expm_factorization(syn):
    assert syn["expm_factorization_err"] < 1e-12


def test_ccw_motion_has_positive_spin(syn):
    assert syn["sign_anchor_ccw_positive"] > 0.0


def test_cw_motion_has_negative_spin():
    tt = np.arange(400) * 0.05
    O, E, _ = odd_even_accumulate(np.cos(0.31 * tt), -np.sin(0.31 * tt),
                                  maxlag=4)
    assert O[1] / E[0] < 0.0


def test_mirror_pairing_msd_blind_spin_flips(syn):
    assert syn["mirrored_msd_diff"] < 1e-12       # MSD chirality-blind
    assert syn["mirrored_spin_flip_err"] < 1e-12  # spin exactly flips
    assert syn["mirrored_even_same_err"] < 1e-12  # even part untouched


def test_lag_reversal_is_transpose_and_odd(syn):
    assert syn["lag_reversal_err"] < 1e-12
    C = rotator_correlation(1.3, 0.31, 3.7)
    assert odd_part(C.T) == pytest.approx(-odd_part(C), abs=1e-15)


def test_rho_odd_matches_analytic(syn):
    assert syn["rho_odd_vs_analytic_err"] < 0.05


def test_odd_estimator_zero_lag_vanishes():
    rng = np.random.default_rng(5)
    O, E, C = odd_even_accumulate(rng.standard_normal(500),
                                  rng.standard_normal(500))
    assert O[0] == 0.0                     # u v - v u at equal times
    assert E[0] > 0 and C[0] == 500


# --------------------------------------------------------------------------- #
# committed cache schema
# --------------------------------------------------------------------------- #
def test_cache_schema(cache):
    assert set(cache) >= {"meta", "per_float", "bands"}
    n = cache["meta"]["n_floats"]
    assert n > 5000
    assert len(cache["per_float"]["rho1"]) == n
    assert len(cache["per_float"]["lat"]) == n
    for b in ("nh_all", "sh_all", "eq", "nh_mid", "sh_mid"):
        assert b in cache["bands"]
        assert len(cache["bands"][b]["O"]) == cache["meta"]["maxlag"] + 1


def test_spin_stats_helper():
    st = spin_stats([1.0, 1.0, 1.0, -1.0], np.array([True] * 4))
    assert st["n_floats"] == 4 and st["rho1_mean"] == 0.5


# --------------------------------------------------------------------------- #
# real-data verdicts
# --------------------------------------------------------------------------- #
def test_chirality_flips_at_equator(result):
    st = result["stats"]
    assert st["nh_all"]["t_stat"] < -4.0
    assert st["sh_all"]["t_stat"] > 4.0
    assert abs(st["eq"]["t_stat"]) < 2.0
    assert st["nh_all"]["rho1_mean"] * st["sh_all"]["rho1_mean"] < 0
    assert result["verdicts"]["ocean_chirality_flips_at_equator"]


def test_anticyclonic_dominance(result):
    st = result["stats"]
    assert st["nh_all"]["rho1_mean"] < 0      # NH clockwise
    assert st["sh_all"]["rho1_mean"] > 0      # SH counterclockwise
    assert st["nh_mid"]["rho1_mean"] < 0 and st["sh_mid"]["rho1_mean"] > 0
    assert result["verdicts"]["anticyclonic_dominance_both_hemispheres"]


def test_band_sign_structure_antisymmetric(result):
    st = result["stats"]
    signs = [np.sign(st[b]["rho1_mean"])
             for b in ("nh_mid", "nh_low", "sh_low", "sh_mid")]
    assert signs == [-1.0, -1.0, 1.0, 1.0]


def test_odd_memory_decay_scale(result):
    for hemi in ("nh_all", "sh_all"):
        hl = result["decay"][hemi]["halflife_days"]
        assert 10.0 <= hl <= 60.0


def test_all_verdicts_and_figure(result):
    for key in ("odd_completion_exact", "msd_is_chirality_blind",
                "ocean_chirality_flips_at_equator",
                "anticyclonic_dominance_both_hemispheres"):
        assert result["verdicts"][key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["verdicts"]["ocean_chirality_flips_at_equator"] is True
    assert disk["stats"]["nh_all"]["t_stat"] == pytest.approx(
        res["stats"]["nh_all"]["t_stat"], rel=1e-12)
