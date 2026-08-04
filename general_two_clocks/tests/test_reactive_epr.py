"""Unit-proofs for NR66 (new_relationships43.py) — the reactive/housekeeping EPR.

Closed-form OU proofs (reactive rotator vs gyrator) plus the real ANDRO Stokes
read-out (NR56 committed cache).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

import new_relationships43 as R  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(R.STOKES_CACHE)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed NR56 Stokes cache missing")


# ---------------------------------------------------- reactive rotator (closed form)
@pytest.mark.parametrize("tau,f,D0", [(20.0, 0.10, 0.5), (17.0, 0.23, 0.7),
                                      (40.0, 0.05, 1.1), (5.0, 0.25, 1.3)])
def test_reactive_rotator_isotropic_antisymmetric_heatless(tau, f, D0):
    r = R.reactive_rotator(tau, f, D0)
    # stationary covariance is isotropic: C = D0*tau*I, Q = U = 0
    assert r["anisotropy"] < 1e-9
    assert abs(r["C"][0, 0] - D0 * tau) < 1e-9
    # Omega is purely antisymmetric (no symmetric = no dissipative channel)
    assert r["sym_norm"] < 1e-9
    assert r["asym_norm"] > 1e-6
    # EPR equals the two-clocks closed form 2 f^2 tau = 2 r^2 / tau
    assert abs(r["sigma"] - 2.0 * f * f * tau) < 1e-10
    assert abs(r["sigma"] - 2.0 * r["r_clock"] ** 2 / tau) < 1e-10
    # the antisymmetric (Coriolis) drive does no work
    assert abs(r["coriolis_power"]) < 1e-9


def test_reactive_sigma_is_temperature_independent():
    """Scaling D0 (bath temperature) leaves the reactive EPR unchanged."""
    a = R.reactive_rotator(15.0, 0.2, 0.3)
    b = R.reactive_rotator(15.0, 0.2, 3.0)
    assert abs(a["sigma"] - b["sigma"]) < 1e-10
    assert a["sigma"] > 0


def test_reactive_rate_is_clock_ratio_squared():
    """σ·τ/2 = r² across a sweep — the two-clocks ratio governs the rate."""
    for tau in (8.0, 25.0, 60.0):
        for f in (0.03, 0.12, 0.4):
            r = R.reactive_rotator(tau, f, 1.0)
            assert abs(r["sigma"] * tau / 2.0 - (f * tau) ** 2) < 1e-9


# ---------------------------------------------------- gyrator (dissipative) contrast
def test_gyrator_needs_gradient_and_is_anisotropic():
    hot = R.gyrator(1.0, 0.6, 3.0, 0.5)
    iso = R.gyrator(1.0, 0.6, 1.0, 1.0)
    assert hot["sigma"] > 1e-6            # gradient present -> dissipation
    assert iso["sigma"] < 1e-9            # no gradient -> no EPR despite coupling
    assert hot["anisotropy"] > 1e-6       # gyrator lives in the anisotropy
    assert hot["sym_norm"] > 1e-6         # dissipative channel = symmetric Omega


def test_gyrator_vs_reactive_are_distinct_classes():
    react = R.reactive_rotator(20.0, 0.3, 1.0)
    gyr = R.gyrator(1.0, 0.6, 3.0, 0.5)
    # reactive: isotropic + antisymmetric; gyrator: anisotropic + has symmetric part
    assert react["anisotropy"] < 1e-9 and gyr["anisotropy"] > 1e-6
    assert react["sym_norm"] < 1e-9 and gyr["sym_norm"] > 1e-6


# ---------------------------------------------------- stationary covariance solver
def test_stationary_cov_solves_lyapunov():
    A = np.array([[0.4, -0.7], [0.3, 0.5]])
    D = np.array([[1.2, 0.1], [0.1, 0.8]])
    C = R.stationary_cov(A, D)
    assert np.allclose(A @ C + C @ A.T, 2.0 * D, atol=1e-9)
    assert np.allclose(C, C.T, atol=1e-9)


# ---------------------------------------------------- real ANDRO data
@needs_cache
def test_ocean_spin_tracks_coriolis():
    oc = R.ocean_reactive_test()
    assert oc["n_floats"] > 5000
    # |V| grows with |f|, and signed V is cyclonic (sign follows sign f: NH-, SH+
    # in the O/E convention) — both highly significant
    assert oc["spin_tracks_f"]["rho"] > 0 and oc["spin_tracks_f"]["p"] < 1e-3
    assert oc["signed_spin_vs_f"]["rho"] < 0 and oc["signed_spin_vs_f"]["p"] < 1e-3


@needs_cache
def test_ocean_anisotropy_is_not_gyrator():
    oc = R.ocean_reactive_test()
    # anisotropy does NOT grow with |f| (it actually peaks at the equator), and
    # once |f| is controlled it carries no residual spin coupling -> no gyrator
    assert oc["anisotropy_vs_f"]["rho"] <= 0.05
    assert abs(oc["spin_vs_anisotropy_partial_f"]["rho"]) < 0.10


@needs_cache
def test_verdict_ocean_epr_reactive():
    res = R.analyze()
    v = res["verdict"]
    assert v["reactive_isotropic"] and v["reactive_antisymmetric"]
    assert v["reactive_sigma_closed"] and v["reactive_heatless"]
    assert v["gyrator_needs_gradient"] and v["gyrator_anisotropic"]
    assert v["ocean_epr_is_reactive"]
