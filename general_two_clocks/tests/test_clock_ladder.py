"""Unit-proofs for NR69 (new_relationships46.py) — the clock-crossover ladder:
hard-sphere kinetics identities, the one-curve/two-ceilings structure, exobase
and turbopause rung locations, subsonic forcing, and the Jeans/blow-off rung.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

import new_relationships46 as R  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(R.CACHE)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed NR68 MSIS cache missing")


# ------------------------------------------------------------ kinetic identities
def test_mean_free_path_inverse_in_density_and_sigma():
    l1 = R.mean_free_path(1e20, 3e-19)
    assert abs(l1 * math.sqrt(2.0) * 1e20 * 3e-19 - 1.0) < 1e-12
    assert abs(R.mean_free_path(2e20, 3e-19) / l1 - 0.5) < 1e-12
    assert abs(R.mean_free_path(1e20, 6e-19) / l1 - 0.5) < 1e-12


def test_thermal_speed_scalings():
    v = R.thermal_speed(300.0, 28.0 * R.M_U)
    # sqrt(8kT/pi m): N2 at 300 K ~ 476 m/s (textbook)
    assert 450.0 < v < 500.0
    assert abs(R.thermal_speed(1200.0, 28.0 * R.M_U) / v - 2.0) < 1e-12
    assert abs(R.thermal_speed(300.0, 7.0 * R.M_U) / v - 2.0) < 1e-12


def test_knudsen_is_exactly_tau_ratio_and_3D_over_vH():
    # synthetic profile: exponential density, isothermal
    alts = np.linspace(100.0, 500.0, 100)
    n = 1e20 * np.exp(-alts / 50.0)
    T = np.full_like(alts, 800.0)
    arr = np.zeros((alts.size, len(R.COL)))
    arr[:, R.COL["N2"]] = n
    arr[:, R.COL["T"]] = T
    arr[:, R.COL["rho"]] = n * 28.0 * R.M_U
    p = R.ladder_profiles(alts, arr)
    assert np.allclose(p["Kn"], p["tau_coll"] / p["tau_transit"], rtol=1e-12)
    assert np.allclose(p["Kn"], 3.0 * p["D"] / (p["v_ms"] * p["H_m"]), rtol=1e-12)


def test_first_upcross_log_interpolation():
    alts = np.array([0.0, 10.0, 20.0])
    y = np.array([0.1, 1.0, 10.0])
    # crosses 1.0 exactly at 10 km; crosses sqrt(10) halfway in log space
    assert abs(R._first_upcross(alts, y, 1.0) - 10.0) < 1e-9
    assert abs(R._first_upcross(alts, y, math.sqrt(10.0)) - 15.0) < 1e-9
    assert R._first_upcross(alts, y, 100.0) is None


# ------------------------------------------------------------ the ladder on real data
@needs_cache
def test_rung1_turbopause_band_matches_mainstream_and_nr68():
    res = R.analyze()
    for name, e in res["conditions"].items():
        zt = e["z_turbopause_km"]
        # D = K_zz for the literature eddy band lands in the mainstream
        # turbopause zone, inside NR68's composition band 89-132 km
        assert 95.0 < zt["Kzz=100"] < 125.0
        assert 100.0 < zt["Kzz=400"] < 130.0
        assert zt["Kzz=100"] < zt["Kzz=400"] < zt["Kzz=1000"] < 135.0


@needs_cache
def test_rung2_exobase_in_mainstream_band_and_above_rung1():
    res = R.analyze()
    for name, e in res["conditions"].items():
        # solar-cycle exobase band ~350-700 km; cache epoch is low activity
        assert 300.0 < e["z_exobase_km"] < 700.0
        assert e["z_exobase_km"] > e["z_turbopause_km"]["Kzz=1000"] + 100.0
        assert 500.0 < e["T_exobase_K"] < 1200.0


@needs_cache
def test_subsonic_forcing_margin():
    res = R.analyze()
    for name, e in res["conditions"].items():
        # ballistic ceiling >> eddy ceiling: ladder order forced, > 2 decades
        assert e["ceiling_ratio_at_turbo"] > 100.0


@needs_cache
def test_rung3_every_species_in_jeans_regime_with_H_closest():
    res = R.analyze()
    lc_lo = res["lambda_c"][0]
    for name, e in res["conditions"].items():
        lam = e["jeans_at_exobase"]
        assert lam["H"] < lam["He"] < lam["O"] < lam["N2"]
        assert lam["H"] > lc_lo          # even H is above blow-off
        assert 6.0 < lam["H"] < 14.0     # mainstream Earth-H value ~9
        # blow-off would need a ~3x hotter exosphere
        tb = min(e["T_blowoff_K"]["H"].values())
        assert tb > 1.8 * e["T_exobase_K"]


@needs_cache
def test_exact_identities_hold_on_real_profiles():
    res = R.analyze()
    for name, e in res["conditions"].items():
        ir = e["identity_residuals"]
        assert ir["Kn_vs_3D_over_vH"] < 1e-9
        assert ir["Kn_vs_tau_ratio"] < 1e-9


@needs_cache
def test_verdict_mentions_all_rungs():
    res = R.analyze()
    v = res["verdict"]
    assert "rung1" in v and "rung2" in v and "rung3" in v
    assert "Jeans regime" in v
