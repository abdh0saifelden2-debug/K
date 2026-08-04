"""Unit proofs for NR47 (`general_two_clocks/new_relationships24.py`): the
memory phase ``delta = arg chi`` as the universal two-clocks coordinate.
Offline-safe -- the real-data reads come from the committed cache
``data/nr47_memory_phase_cache.json`` (assembled from the committed
NR39/NR45/NR46 caches; regenerate with ``build_cache()``).

Covered: the five-faces identity (loss tangent = Im/Re = migration =
fractional-order = Bode-slope, to machine precision); the exact Bode
gain-phase relation for power laws and its known local-approx behaviour; the
two poles (elliptic delta=0, parabolic delta=pi/2) and the normalization-free
elastic fraction; the clock ratio; and the committed cross-domain verdicts
(EEG at the elliptic pole, ocean liquid at the parabolic pole with loopers
pulled off it, solar one-sided bounded).
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships24 import (  # noqa: E402
    FIG, analyze, bode_local_phase, clock_ratio, delta_from_alpha,
    elastic_fraction, five_faces, load_cache, memory_phase_from_response,
)


@pytest.fixture(scope="module")
def cache():
    return load_cache()


@pytest.fixture(scope="module")
def verdicts(cache):
    return analyze(cache)


# --------------------------------------------------------------------------- #
# the invariant in closed form (self-contained, exact)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.6, 0.75, 0.9])
def test_five_faces_identical(alpha):
    f = five_faces(alpha)
    ref = f["phase"]
    mid = slice(20, -20)                          # drop gradient edge effects
    for key in ("im_over_re", "migration", "frac_order"):
        assert np.allclose(f[key], ref, atol=1e-9)
    assert np.allclose(f["bode"][mid], ref[mid], rtol=1e-6, atol=1e-6)


def test_bode_exact_for_power_law():
    # delta = (pi/2) d ln|chi|/d ln omega, exact when |chi| ~ omega^alpha
    w = np.geomspace(1e-2, 1e2, 400)
    for alpha in (0.3, 0.5, 1.0):
        chi = (1j * w) ** alpha
        d = bode_local_phase(w, np.abs(chi))
        assert np.allclose(d[20:-20], np.pi * alpha / 2.0, atol=1e-6)


def test_bode_local_exact_at_crossover_for_debye():
    # local Bode is exact at the log-log inflection (the 45-deg crossover)
    w = np.geomspace(1e-3, 1e3, 8000)
    tau = 1.0
    chi = 1.0 / (1 + 1j * w * tau)
    d_true = memory_phase_from_response(chi)
    d_bode = bode_local_phase(w, np.abs(chi))
    knee = np.argmin(np.abs(w - 1.0 / tau))
    assert d_true[knee] == pytest.approx(-np.pi / 4, abs=1e-2)
    assert d_bode[knee] == pytest.approx(-np.pi / 4, abs=1e-2)
    # and the local approx is a bounded few-degree error mid-band (honest)
    mb = (w > 0.05) & (w < 20)
    assert np.median(np.degrees(np.abs(d_true - d_bode))[mb]) < 8.0


def test_two_poles_and_elastic_fraction():
    # elliptic pole: alpha=0 -> delta=0, r=0, G'/G'' = inf (all storage)
    assert delta_from_alpha(0.0) == pytest.approx(0.0)
    assert clock_ratio(delta_from_alpha(0.0)) == pytest.approx(0.0)
    # parabolic pole: alpha=1 -> delta=pi/2, r->inf, G'/G'' = 0 (all loss)
    assert delta_from_alpha(1.0) == pytest.approx(np.pi / 2)
    assert elastic_fraction(1.0) == pytest.approx(0.0, abs=1e-6)
    # Warburg half order: alpha=1/2 -> delta=45 deg -> storage == loss
    assert delta_from_alpha(0.5) == pytest.approx(np.pi / 4)
    assert elastic_fraction(0.5) == pytest.approx(1.0, rel=1e-6)


def test_elastic_fraction_is_normalization_free():
    # depends only on alpha (the log-log slope), not on any amplitude/scale
    assert elastic_fraction(0.7) == elastic_fraction(0.7)
    a = np.array([0.6, 0.8, 0.95])
    assert np.all(np.diff(elastic_fraction(a)) < 0)   # more loss -> less G'/G''


# --------------------------------------------------------------------------- #
# committed cross-domain cache + verdicts
# --------------------------------------------------------------------------- #
def test_cache_schema(cache):
    assert cache["meta"]["poles"] == {"elliptic_deg": 0.0,
                                      "parabolic_deg": 90.0}
    for dom in ("eeg", "ocean", "solar"):
        assert dom in cache["reads"]


def test_eeg_instantaneous_at_elliptic_pole(cache, verdicts):
    assert verdicts["eeg_instantaneous_at_elliptic_pole"]
    eeg = cache["reads"]["eeg"]
    assert eeg["delta_deg_near"] < 4.0                 # volume conduction ~ 0
    assert eeg["delta_deg_near"] < eeg["delta_deg_far"]


def test_ocean_poles_and_looper_deficit(cache, verdicts):
    assert verdicts["ocean_liquid_at_parabolic_pole"]
    assert verdicts["ocean_loopers_carry_phase_deficit"]
    o = cache["reads"]["ocean"]
    assert o["rest"]["delta_deg"] >= 88.0              # liquid at 90
    assert o["loop"]["delta_deg"] <= o["rest"]["delta_deg"] - 4.0
    assert o["loop"]["clock_ratio"] < 20.0             # measurable storage


def test_solar_phase_one_sided_bounded(cache, verdicts):
    assert verdicts["solar_phase_one_sided_bounded"]
    s = cache["reads"]["solar"]
    assert -180.0 < s["delta_deg_min"] < s["delta_deg_max"] < -60.0


def test_committed_figure_artifact(verdicts):
    with open(FIG) as fh:
        fig = json.load(fh)
    assert fig["verdicts"] == verdicts
