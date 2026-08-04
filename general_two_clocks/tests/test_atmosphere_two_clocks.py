"""Unit-proofs for NR68 (new_relationships45.py) — the atmosphere's vertical
structure as the two-clocks competition: scale-height formulae, mass-ratio
limits, Jeans-escape ordering, cache schema, and the end-to-end analyze().
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

import new_relationships45 as R  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(R.CACHE)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed NR68 MSIS cache missing")


# ------------------------------------------------------------ exact formulae
def test_measured_scale_height_recovers_pure_exponential():
    # n(z) = n0 exp(-z/H) with H = 30 km must be read back exactly
    alts = np.linspace(100.0, 500.0, 200)
    H_true = 30.0
    n = 1e12 * np.exp(-alts / H_true)
    H = R.measured_scale_height(alts, n)
    assert np.allclose(H[5:-5], H_true, rtol=1e-6)


def test_theoretical_scale_height_value_and_mass_scaling():
    # kT/(mg): N2 at 300 K near the ground -> ~8.8 km (textbook)
    H_N2 = R.theoretical_scale_height(np.array([0.0]), np.array([300.0]),
                                      R.MASS["N2"])[0]
    assert 8.0 < H_N2 < 9.5
    # exact 1/m scaling at fixed T, g
    H_He = R.theoretical_scale_height(np.array([0.0]), np.array([300.0]),
                                      R.MASS["He"])[0]
    assert abs(H_He / H_N2 - R.MASS["N2"] / R.MASS["He"]) < 1e-9


def test_gravity_decreases_with_altitude():
    g0, g400 = R.g_of_z(0.0), R.g_of_z(400e3)
    assert 9.7 < g0 < 9.9
    assert g400 < g0
    assert abs(g0 / g400 - ((R.R_EARTH + 400e3) / R.R_EARTH) ** 2) < 1e-12


# ------------------------------------------------------------ Jeans escape
def test_jeans_parameter_is_linear_in_mass_and_inverse_in_T():
    lam1 = R.jeans_parameter(500.0, 1000.0, 4.0)
    lam2 = R.jeans_parameter(500.0, 1000.0, 8.0)
    lam3 = R.jeans_parameter(500.0, 2000.0, 4.0)
    assert abs(lam2 / lam1 - 2.0) < 1e-12
    assert abs(lam3 / lam1 - 0.5) < 1e-12


def test_jeans_flux_factor_monotone_decreasing_and_astronomical_range():
    lams = np.array([5.0, 10.0, 50.0, 100.0])
    f = R.jeans_flux_factor(lams)
    assert np.all(np.diff(f) < 0)
    # light-vs-heavy contrast is astronomically large
    assert f[0] / f[-1] > 1e30


def test_jeans_parameter_matches_vesc_over_vp_definition():
    # λ = v_esc²/v_p² with v_esc² = 2GM/r, v_p² = 2kT/m
    z_km, T, m_amu = 500.0, 1000.0, 4.0026
    r = R.R_EARTH + z_km * 1e3
    v_esc2 = 2.0 * R.G_GRAV * R.M_EARTH / r
    v_p2 = 2.0 * R.K_B * T / (m_amu * R.M_U)
    lam = R.jeans_parameter(z_km, T, m_amu)
    assert abs(lam / (v_esc2 / v_p2) - 1.0) < 1e-12


# ------------------------------------------------------------ cache schema
@needs_cache
def test_cache_schema_and_monotonic_altitude():
    cache = R.load_cache()
    assert "conditions" in cache and len(cache["conditions"]) >= 3
    alts = np.array(cache["_alts_km"], float)
    assert np.all(np.diff(alts) > 0)
    for name, cond in cache["conditions"].items():
        arr = R._profile(cond)
        assert arr.shape[0] == alts.size
        assert arr.shape[1] == len(R.COL)
        # densities positive, temperature physical
        for s in R.INERT:
            assert np.all(arr[:, R.COL[s]] > 0)
        assert np.all(arr[:, R.COL["T"]] > 100.0)
        assert np.all(arr[:, R.COL["T"]] < 3000.0)


# ------------------------------------------------------------ end-to-end
@needs_cache
def test_analyze_two_clocks_verdict():
    res = R.analyze()
    assert res["conditions"]
    for name, e in res["conditions"].items():
        # homosphere: flow clock -> shared scale height (discriminant near 1)
        assert e["homosphere"]["He_over_N2"] < 1.3
        # heterosphere: molecular clock -> He/N2 hits the mass ratio 28/4 within 5%
        het = e["heterosphere"]
        assert abs(het["He_over_N2"] / het["mass_prediction_He"] - 1.0) < 0.05
        # Ar and O land on their own mass predictions within 5%
        for s in ("Ar", "O"):
            sp = het["species"][s]
            assert abs(sp["measured_ratio_vs_N2"] / sp["mass_prediction"] - 1.0) < 0.05
        # turbopause band: onset in the mainstream departure zone, ordered levels
        tb = e["turbopause"]
        assert 80.0 < tb["onset_km"] < 120.0
        assert tb["onset_km"] < tb["log_mid_km"] < tb["linear_mid_km"] < 200.0
        # Jeans ordering: lighter -> smaller λ -> larger escape factor
        j = e["jeans"]
        assert j["lam"]["H"] < j["lam"]["He"] < j["lam"]["O"] < j["lam"]["N2"]
        assert (j["flux_factor"]["H"] > j["flux_factor"]["He"]
                > j["flux_factor"]["O"] > j["flux_factor"]["N2"])
    assert "turbopause" in res["verdict"] and "Jeans" in res["verdict"]
