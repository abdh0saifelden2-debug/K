"""Unit proofs for NR50 (`general_two_clocks/new_relationships27.py`): the
phase-area budget.  Bode's area theorem makes the integrated memory phase a
conserved quantity fixed by the endpoint gains (waterbed), and the banded
budget classifies gain structure into transport / causal memory / static
weighting.  Offline-safe -- real reads come from the committed NR48 cache
``data/nr48_minphase_cache.json``.

Covered: the exact full-line area theorem on lead-lag (analytic value);
waterbed invariance (same endpoints, different interior => same area,
different pointwise phase); the power-law banded identity (exact); the
transport area formula (analytic); a static (zero-phase) weighting pays
gain but no area; cache-driven ocean closure (~6 %) and the solar
transport-only budget with the x3.5 causal-gain veto.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships27 import (  # noqa: E402
    FIG, analyze, endpoint_area, load_cache, ocean_budget, phase_area,
    robust_endpoints, run, solar_budget, synthetic_checks, transport_area,
)


@pytest.fixture(scope="module")
def cache():
    return load_cache()


@pytest.fixture(scope="module")
def result(cache):
    return analyze(cache)


# --------------------------------------------------------------------------- #
# the exact theorem
# --------------------------------------------------------------------------- #
def test_leadlag_area_matches_endpoint_gains():
    syn = synthetic_checks()
    assert syn["leadlag_err"] < 1e-6
    # analytic value: (pi/2) ln(tau2/tau1) with tau2=0.2, tau1=3.0
    assert syn["leadlag_pred"] == pytest.approx(
        np.pi / 2 * np.log(0.2 / 3.0), rel=1e-12)


def test_waterbed_conserves_area_not_shape():
    syn = synthetic_checks()
    assert syn["waterbed_diff"] < 1e-6          # same area...
    assert syn["shapes_differ"] > 0.1           # ...different phase pointwise
    assert abs(syn["mid_phase_1"] - syn["mid_phase_2"]) > 0.01


def test_power_law_banded_identity_exact():
    # |chi| ~ w^alpha, delta = pi alpha/2: banded area = (pi/2) Delta ln|chi|
    alpha, u1, u2 = 0.6, -2.0, 3.0
    u = np.linspace(u1, u2, 4001)
    area = phase_area(u, np.full_like(u, np.pi * alpha / 2))
    assert area == pytest.approx(endpoint_area(alpha * u1, alpha * u2),
                                 rel=1e-12)


def test_transport_area_analytic():
    f = np.linspace(1.0, 5.0, 400)
    tau, phi0 = 0.03, 0.7
    w = 2 * np.pi * f
    num = phase_area(np.log(w), -w * tau + phi0)
    assert num == pytest.approx(transport_area(f, tau, phi0), rel=1e-3)


def test_static_weighting_pays_no_area():
    # a real positive per-frequency factor multiplies the gain but adds no
    # phase: the budget assigns it zero area by construction
    u = np.linspace(-3, 3, 1001)
    weighting_phase = np.zeros_like(u)
    assert phase_area(u, weighting_phase) == 0.0


def test_robust_endpoints():
    lo, hi = robust_endpoints([1.0, 2.0, 3.0, 10.0, 11.0, 12.0], n=3)
    assert lo == 2.0 and hi == 11.0


# --------------------------------------------------------------------------- #
# real-data budgets
# --------------------------------------------------------------------------- #
def test_ocean_budget_closes(cache, result):
    oc = result["ocean"]
    assert oc["rel_closure"] < 0.10
    assert oc["area_meas"] > 0 and oc["area_pred_minphase"] > 0
    assert result["verdicts"]["ocean_gain_is_causal_memory"]


def test_ocean_budget_matches_direct_computation(cache):
    o = cache["ocean_gser"]
    direct = float(np.trapezoid(np.array(o["delta_meas_rad"]),
                                np.array(o["log_w"])))
    assert ocean_budget(cache)["area_meas"] == pytest.approx(direct,
                                                             rel=1e-12)


def test_solar_transport_budget_and_veto(result):
    so = result["solar"]
    assert so["rel_closure_transport"] < 0.10
    assert so["naive_overshoot"] > 2.0          # causal gain reading vetoed
    assert 5.0 < abs(so["tau_d_s"]) < 15.0      # NR48's delay clock
    assert so["n_coherent"] >= 100
    assert result["verdicts"]["solar_gain_is_static_weighting"]


def test_solar_tau_matches_nr48(cache):
    so = solar_budget(cache)
    assert so["tau_d_s"] == pytest.approx(-10.3, abs=0.5)


def test_all_verdicts_and_figure(result):
    v = result["verdicts"]
    for key in ("area_theorem_and_waterbed_exact",
                "ocean_gain_is_causal_memory",
                "solar_gain_is_static_weighting"):
        assert v[key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["verdicts"]["solar_gain_is_static_weighting"] is True
    assert disk["ocean"]["rel_closure"] == pytest.approx(
        res["ocean"]["rel_closure"], rel=1e-12)
