"""Unit proofs for NR53 (`general_two_clocks/new_relationships30.py`): the
two clocks of the spin.  The ocean's odd (Lagrangian-spin) correlation
decomposes into an oscillatory EDDY clock (damped rotator: zero-crossing,
finite memory) and a persistent CIRCULATION clock (linear growth, no
crossing); the deep eddy rotation period is measured and
hemisphere-mirrored in the tropics, and the subpolar/polar cyclonic
reversal is a circulation-clock feature.  Offline-safe: committed cache
``data/nr53_spin_bands_cache.json``.

Covered: the spin model and its fingerprints (zero-crossing lag, tail
fraction) on analytic curves; synthetic mixture separation (f_e to ~2 %,
g within a factor 2, drift-only control has no crossing); cache schema
across 11 bands; tropical oscillation with mirrored chirality and
50-90 d fitted periods; the polar persistent cyclonic reversal; the
equatorial null; energy-weighted vs per-float statistics distinction.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships30 import (  # noqa: E402
    CACHE, DT_DAYS, FIG, analyze, band_rho, fit_spin, load_cache, run,
    spin_model, synthetic_checks, tail_fraction, zero_crossing_lag,
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
# model + fingerprints
# --------------------------------------------------------------------------- #
def test_spin_model_components():
    tau = np.arange(1, 13) * 10.0
    eddy = spin_model(tau, 0.05, 2 * np.pi / 60.0, 30.0, 0.0)
    drift = spin_model(tau, 0.0, 0.1, 30.0, 5e-4)
    assert eddy[0] > 0 and np.any(eddy < 0)      # oscillates
    assert np.all(np.diff(drift) > 0)            # grows linearly


def test_zero_crossing_detects_damped_rotator():
    tau = np.arange(0, 13) * 10.0
    rho = spin_model(tau, -0.05, -2 * np.pi / 60.0, 40.0, 0.0)
    rho[0] = 0.0
    lag = zero_crossing_lag(rho)
    assert lag is not None and 2 <= lag <= 5


def test_no_crossing_for_pure_drift():
    tau = np.arange(0, 13) * 10.0
    rho = spin_model(tau, 0.0, 0.1, 30.0, 4e-4)
    assert zero_crossing_lag(rho) is None
    assert tail_fraction(rho) > 1.0              # tail exceeds rho1


def test_fit_recovers_known_curve():
    tau = np.arange(0, 13) * 10.0
    truth = dict(A=0.06, f_e=-2 * np.pi / 55.0, tau_m=35.0, g=2e-4)
    rho = spin_model(tau, truth["A"], truth["f_e"], truth["tau_m"],
                     truth["g"])
    fit = fit_spin(rho)
    assert abs(fit["f_e_rad_day"] - truth["f_e"]) < 0.15 * abs(truth["f_e"])
    assert abs(fit["period_days"] - 55.0) < 8.0


# --------------------------------------------------------------------------- #
# synthetic mixture separation
# --------------------------------------------------------------------------- #
def test_mixture_separation(syn):
    assert syn["f_e_rel_err"] < 0.15
    assert 0.5 < syn["g_ratio"] < 2.0
    assert syn["mixture_has_crossing"]


def test_drift_control(syn):
    assert not syn["drift_has_crossing"]
    assert syn["drift_tail_same_sign"]


# --------------------------------------------------------------------------- #
# committed cache
# --------------------------------------------------------------------------- #
def test_cache_schema(cache):
    expect = {"nh_polar", "nh_subpolar", "nh_mid", "nh_sub", "nh_trop",
              "eq", "sh_trop", "sh_sub", "sh_mid", "sh_subpolar",
              "sh_polar"}
    assert expect <= set(cache["bands"])
    for name in expect:
        b = cache["bands"][name]
        assert len(b["O"]) == cache["meta"]["maxlag"] + 1
        assert b["E"][0] > 0 and b["n_floats"] > 20


def test_band_rho_normalization(cache):
    rho = band_rho(cache, "nh_trop")
    assert rho[0] == pytest.approx(0.0)   # O[0] = 0 identically (u v - v u)
    assert cache["bands"]["nh_trop"]["O"][0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# real-data verdicts
# --------------------------------------------------------------------------- #
def test_tropical_eddy_clock_mirrored(result):
    nh, sh = result["bands"]["nh_trop"], result["bands"]["sh_trop"]
    assert nh["rho1"] < 0 < sh["rho1"]
    assert nh["crossing_lag"] is not None
    assert sh["crossing_lag"] is not None
    assert 30.0 < nh["fit"]["period_days"] < 120.0
    assert 30.0 < sh["fit"]["period_days"] < 120.0
    assert nh["fit"]["f_e_rad_day"] * sh["fit"]["f_e_rad_day"] < 0
    assert result["verdicts"]["deep_eddy_rotation_measured_and_mirrored"]


def test_polar_reversal_is_circulation_clock(result):
    p = result["bands"]["nh_polar"]
    assert p["rho1_float"] > 0 and p["t_stat"] > 4.0
    assert p["crossing_lag"] is None
    assert p["tail_frac"] > 0.25
    assert result["bands"]["nh_mid"]["rho1_float"] < 0
    assert result["verdicts"]["polar_reversal_is_circulation_clock"]


def test_sh_subpolar_flips_against_sh_mid(result):
    assert (result["bands"]["sh_subpolar"]["rho1_float"]
            * result["bands"]["sh_mid"]["rho1_float"] < 0)


def test_equatorial_null(result):
    assert abs(result["bands"]["eq"]["t_stat"]) < 2.0
    assert result["verdicts"]["equatorial_null"]


def test_all_verdicts_and_figure(result):
    for key in ("two_spin_clocks_separable",
                "deep_eddy_rotation_measured_and_mirrored",
                "polar_reversal_is_circulation_clock", "equatorial_null"):
        assert result["verdicts"][key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["verdicts"]["polar_reversal_is_circulation_clock"] is True
    assert disk["bands"]["nh_trop"]["fit"]["period_days"] == pytest.approx(
        res["bands"]["nh_trop"]["fit"]["period_days"], rel=1e-9)
