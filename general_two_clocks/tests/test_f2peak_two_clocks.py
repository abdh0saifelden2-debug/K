"""Unit-proofs for NR72 (new_relationships49.py) — the F2-peak two-clocks audit:
rate-coefficient and collision-frequency sanity, crossing/e-fold machinery, and
the measured balance-formed vs dynamics-formed classifier gap.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

import new_relationships49 as R  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(R.CACHE)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed NR72 cache missing")


# ------------------------------------------------------------ ingredient physics
def test_o_plus_loss_rate_magnitude_and_linearity():
    # k1(300 K) ~ 1.0e-12 cm^3/s: beta for [N2]=1e9 cm^-3 alone ~ 1e-3 /s
    beta = R.o_plus_loss_rate(1e15, 0.0, 300.0)
    assert 5e-4 < float(beta) < 2e-3
    assert abs(R.o_plus_loss_rate(2e15, 0.0, 300.0) / beta - 2.0) < 1e-12
    # O2 channel is faster per molecule at these temperatures
    b_n2 = R.o_plus_loss_rate(1e15, 0.0, 1000.0)
    b_o2 = R.o_plus_loss_rate(0.0, 1e15, 1000.0)
    assert b_o2 > b_n2


def test_collision_rate_magnitude_and_scaling():
    nu = R.o_plus_o_collision_rate(1e14, 1000.0)   # n(O)=1e8 cm^-3
    assert 0.03 < float(nu) < 0.3
    assert abs(R.o_plus_o_collision_rate(2e14, 1000.0) / nu - 2.0) < 1e-12


def test_crossing_altitude_and_efold_on_synthetic_profile():
    alt = np.linspace(200.0, 400.0, 201)
    Rprof = np.exp(-(alt - 300.0) / 20.0)          # R=1 at 300 km, e-fold 20 km
    assert abs(R.crossing_altitude(alt, Rprof, 1.0) - 300.0) < 1e-6
    assert abs(R.r_efold_km(alt, Rprof, 220.0, 380.0) - 20.0) < 1e-6
    assert R.crossing_altitude(alt, Rprof, 1e9) is None


def test_clock_ratio_dimensional_consistency():
    # R = beta * Hp^2 / Da = tau_diff / tau_chem, verified on synthetic columns
    alt = np.linspace(200.0, 400.0, 51)
    msis = dict(N2=np.full(51, 1e15), O2=np.full(51, 1e14),
                O=np.full(51, 1e15), Tn=np.full(51, 900.0))
    iri = dict(Te=np.full(51, 2000.0), Ti=np.full(51, 1100.0))
    p = R.clock_ratio_profile(alt, msis, iri)
    assert np.allclose(p["R"], p["tau_diff"] / p["tau_chem"], rtol=1e-12)
    assert np.all(p["R"] > 0)


# ------------------------------------------------------------ measured classifier
@needs_cache
def test_daytime_midlat_peaks_are_balance_formed():
    res = R.analyze()
    for name, e in res["conditions"].items():
        if "noon" in name and name.startswith("midlat"):
            assert e["balance_formed"]
            assert 3.0 < e["R_at_peak"] < 50.0
            # peak sits 1-2 N2 scale heights BELOW the R=1 level
            assert -70.0 < e["peak_minus_z1_km"] < -15.0


@needs_cache
def test_midnight_and_equator_peaks_are_dynamics_formed():
    res = R.analyze()
    for name in ("midlat_equinox_midnight", "equator_equinox_noon"):
        e = res["conditions"][name]
        assert not e["balance_formed"]
        assert e["R_at_peak"] < 0.05
        assert e["peak_minus_z1_km"] > 50.0     # held far above the handoff


@needs_cache
def test_classifier_gap_exceeds_two_decades():
    res = R.analyze()
    assert res["classifier_gap_decades"] > 2.0


@needs_cache
def test_R1_levels_and_efolds_are_stable():
    res = R.analyze()
    for e in res["conditions"].values():
        assert 240.0 < e["z_R1_km"] < 290.0     # the handoff level itself
        assert 10.0 < e["efold_km"] < 25.0      # convention robustness


@needs_cache
def test_verdict_names_both_families():
    res = R.analyze()
    assert "balance-formed" in res["verdict"]
    assert "dynamics-formed" in res["verdict"]
    assert "fountain" in res["verdict"]
