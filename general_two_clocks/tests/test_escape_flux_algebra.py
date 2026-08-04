"""Unit-proofs for NR70 (new_relationships47.py) — the escape ladder's clock
algebra: exact series/parallel composition laws, Jeans-flux scalings, and the
measured H (series/diffusion-limited) and He (parallel/helium-problem) verdicts.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

import new_relationships47 as R  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(R.CACHE)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed NR68 MSIS cache missing")


# ------------------------------------------------------------ the algebra, exact
def test_two_slab_exact_equals_series_harmonic():
    phi_exact, phi_series = R.two_slab_steady_flux(2.0, 1.0, 0.1, 3.0, 5.0)
    assert abs(phi_exact - phi_series) < 1e-12
    # and the slow slab dominates: g2 = 0.1/3 << g1 = 2
    assert abs(phi_exact - (0.1 / 3.0) * 5.0) / phi_exact < 0.02


def test_series_slowest_rules_parallel_fastest_rules():
    gs = [10.0, 1.0, 0.01]
    g_ser = R.series_conductance(gs)
    g_par = R.parallel_conductance(gs)
    assert g_ser < min(gs)                       # series below the slowest
    assert abs(g_ser - 0.01) / 0.01 < 0.12       # ...and pinned to it
    assert g_par > max(gs)                       # parallel above the fastest
    assert abs(g_par - 10.0) / 10.0 < 0.11       # ...and pinned to it


def test_series_and_parallel_limits_are_exact_identities():
    gs = np.array([3.0, 7.0])
    assert abs(R.series_conductance(gs) - 21.0 / 10.0) < 1e-12
    assert abs(R.parallel_conductance(gs) - 10.0) < 1e-12


# ------------------------------------------------------------ Jeans flux scalings
def test_jeans_flux_linear_in_density_and_exponential_in_lambda():
    phi1, lam1 = R.jeans_flux(1e11, 800.0, 1.008, 400.0)
    phi2, _ = R.jeans_flux(2e11, 800.0, 1.008, 400.0)
    assert abs(phi2 / phi1 - 2.0) < 1e-12
    # heavier species: lam scales with m, flux collapses much faster than 1/m
    phi_he, lam_he = R.jeans_flux(1e11, 800.0, 4.0026, 400.0)
    assert lam_he > 3.5 * lam1
    assert phi_he < phi1 * 1e-9


@needs_cache
def test_H_valve_in_mainstream_band_and_below_pipe():
    res = R.analyze()
    for name, e in res["conditions"].items():
        s = e["species"]["H"]
        assert 5e6 < s["phi_cm2s"] < 5e7          # thermal valve, low activity
        assert 0.05 < s["global_kg_s"] < 0.5      # ~0.1-0.3 kg/s
        assert s["phi_cm2s"] < R.PHI_DIFF_LIMIT   # valve under the pipe ceiling
    # observed total sits at the pipe scale, not the valve scale
    assert res["series_H"]["observed_over_pipe"] > 0.2
    assert res["series_H"]["valve_over_pipe"] < 0.2


@needs_cache
def test_He_thermal_valve_fails_by_many_decades():
    res = R.analyze()
    assert res["parallel_He"]["thermal_valve_max"] < 1e-2
    assert res["parallel_He"]["decades_short"] > 8.0


@needs_cache
def test_O_is_bound_entirely():
    res = R.analyze()
    for e in res["conditions"].values():
        assert e["species"]["O"]["phi_cm2s"] < 1e-40


@needs_cache
def test_exponential_rectifier_swings():
    res = R.analyze()
    assert res["series_H"]["rectifier_swing"] > 2.0
    assert res["parallel_He"]["rectifier_swing"] > 100.0
    # He swings enormously more than H: d(lam) scales with mass
    assert (res["parallel_He"]["rectifier_swing"]
            > 50.0 * res["series_H"]["rectifier_swing"])


@needs_cache
def test_uses_nr69_exobase_and_verdict_complete():
    res = R.analyze()
    for e in res["conditions"].values():
        assert 300.0 < e["z_exobase_km"] < 700.0
    v = res["verdict"]
    assert "series" in v and "parallel" in v
    assert "diffusion-limited" in v and "helium problem" in v
