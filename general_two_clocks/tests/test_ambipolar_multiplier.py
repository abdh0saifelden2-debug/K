"""Unit-proofs for NR71 (new_relationships48.py) — the ambipolar field as the
quasineutrality constraint's multiplier: exact read-out theorem on synthetic
columns, the doubled plasma clock, and the measured static/excess split.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

import new_relationships48 as R  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(R.CACHE)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed NR71 IRI cache missing")


def _synthetic_column(Te, Ti, m_amu=16.0, alt0=300.0, alt1=600.0, n0=1e12):
    """Exact isothermal static-DE column: n ~ exp(-int m g dz / k(Te+Ti))."""
    alt = np.linspace(alt0, alt1, 400)
    z = alt * 1e3
    g = R.g_of_alt_km(alt)
    m = m_amu * R.M_U
    integ = np.concatenate([[0.0], np.cumsum(
        0.5 * (g[1:] + g[:-1]) * np.diff(z))])
    n = n0 * np.exp(-m * integ / (R.K_B * (Te + Ti)))
    return alt, n


# ------------------------------------------------------- theorem 1: read-out
def test_multiplier_readout_recovers_static_lift_exactly():
    Te, Ti = 2000.0, 1000.0
    alt, n = _synthetic_column(Te, Ti)
    eE = R.multiplier_readout(alt, n, np.full_like(n, Te))
    g = R.g_of_alt_km(alt)
    lift = eE / (16.0 * R.M_U * g)
    # static prediction Te/(Te+Ti) = 2/3, recovered to numerical gradient error
    assert np.allclose(lift[5:-5], Te / (Te + Ti), rtol=2e-3)


def test_multiplier_readout_includes_Te_gradient_term():
    # constant n: eE must equal -k dTe/dz exactly (pure thermo-electric term)
    alt = np.linspace(300.0, 600.0, 301)
    Te = 1000.0 + 2.0 * (alt - 300.0)          # 2 K/km
    ne = np.full_like(alt, 1e11)
    eE = R.multiplier_readout(alt, ne, Te)
    expect = -R.K_B * 2.0 / 1e3
    assert np.allclose(eE[2:-2], expect, rtol=1e-6)


# ------------------------------------------------------- theorem 2: doubled clock
def test_plasma_scale_height_doubling_when_Te_equals_Ti():
    Te = Ti = 1500.0
    alt, n = _synthetic_column(Te, Ti)
    H = R.measured_scale_height_m(alt, n)
    H_single = R.K_B * Ti / (16.0 * R.M_U * R.g_of_alt_km(alt))
    ratio = H[5:-5] / H_single[5:-5]
    assert np.allclose(ratio, 2.0, rtol=2e-3)   # exactly double


def test_plasma_scale_height_general_lift_is_one_plus_Te_over_Ti():
    Te, Ti = 3000.0, 1000.0
    alt, n = _synthetic_column(Te, Ti)
    H = R.measured_scale_height_m(alt, n)
    H_single = R.K_B * Ti / (16.0 * R.M_U * R.g_of_alt_km(alt))
    assert np.allclose(H[5:-5] / H_single[5:-5], 4.0, rtol=2e-3)


def test_static_lift_fraction_and_effective_mass():
    assert abs(R.static_lift_fraction(1500.0, 1500.0) - 0.5) < 1e-12
    assert abs(R.static_lift_fraction(3000.0, 1000.0) - 0.75) < 1e-12
    prof = {"nO+": [9e10], "nH+": [1e10], "nHe+": [0.0],
            "nO2+": [0.0], "nNO+": [0.0]}
    m = R.effective_ion_mass_amu(prof)[0]
    assert abs(m - (0.9 * 15.999 + 0.1 * 1.008)) < 1e-9


# ------------------------------------------------------- measured on the cache
@needs_cache
def test_naive_barometric_clock_excluded_everywhere():
    res = R.analyze()
    for name, e in res["conditions"].items():
        assert 1.3 < e["H_lift_over_neutral"] < 1.9
        assert e["n_window"] >= 10


@needs_cache
def test_multiplier_lift_bands_and_static_prediction():
    res = R.analyze()
    for name, e in res["conditions"].items():
        assert 0.5 < e["lift_static"] < 0.75     # Te/(Te+Ti) in the IRI topside
        assert 0.55 < e["lift_measured"] < 1.15


@needs_cache
def test_high_dip_excess_vs_equatorial_static_support():
    res = R.analyze()
    eq = res["conditions"]["equator_noon"]
    assert eq["lift_excess"] < 0.18              # statically supported column
    for name, e in res["conditions"].items():
        if name.startswith("midlat"):
            assert e["lift_excess"] > 0.2        # outflow-driving excess
            assert e["lift_excess"] > eq["lift_excess"] + 0.1


@needs_cache
def test_equator_near_static_DE_midlat_steepened():
    res = R.analyze()
    assert res["conditions"]["equator_noon"]["H_over_static_DE"] > 0.85
    for name, e in res["conditions"].items():
        if name.startswith("midlat"):
            assert 0.6 < e["H_over_static_DE"] < 0.85


@needs_cache
def test_verdict_reports_contrast():
    res = R.analyze()
    assert "excess" in res["verdict"]
    assert "polar-wind" in res["verdict"]
