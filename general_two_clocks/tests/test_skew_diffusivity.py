"""Unit proofs for NR59 (`general_two_clocks/new_relationships36.py`): the
spin is a skew (divergence-free) diffusivity.  The Green-Kubo integral of
the odd correlation is the antisymmetric eddy-transport tensor K_A; its
ratio to the Taylor diffusivity K_S is exactly the clock ratio f tau_L =
tan(delta); the skew flux advects along contours and mixes nothing; and
in the deep ocean the skew transport flips sign across the equator.
Offline-safe: reads the committed NR52 band cache.

Covered: the closed-form K_S, K_A and the K_A/K_S = f tau_L identity to
1e-10; the divergence-free and contour-parallel properties of the skew
flux; the Green-Kubo band estimator; the hemispheric sign flip of the
ocean skew transport, the equatorial minimum, and the Taylor-diffusivity
scale.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships36 import (  # noqa: E402
    FIG, analyze, band_diffusivity, closed_form, load_cache,
    rotator_diffusivity, run, synthetic_checks,
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
# closed-form diffusivity and the clock-ratio identity
# --------------------------------------------------------------------------- #
def test_rotator_matches_closed_form():
    for tau_L in (1.3, 5.0, 9.0):
        for f in (0.15, 0.5, 1.2):
            KS, KA = rotator_diffusivity(tau_L, f)
            pS, pA = closed_form(tau_L, f)
            assert KS == pytest.approx(pS, rel=1e-9)
            assert KA == pytest.approx(pA, rel=1e-9)


def test_skew_ratio_is_clock_ratio(syn):
    assert syn["ratio_max_err"] < 1e-10
    # explicit: K_A/K_S = f tau_L
    KS, KA = rotator_diffusivity(4.0, 0.3)
    assert KA / KS == pytest.approx(0.3 * 4.0, rel=1e-9)


def test_zero_rotation_gives_zero_skew():
    KS, KA = rotator_diffusivity(5.0, 0.0)
    assert abs(KA) < 1e-12 and KS == pytest.approx(25.0, rel=1e-9)


def test_skew_sign_tracks_f():
    _, KA_pos = rotator_diffusivity(5.0, 0.4)
    _, KA_neg = rotator_diffusivity(5.0, -0.4)
    assert KA_pos > 0 > KA_neg
    assert KA_pos == pytest.approx(-KA_neg, rel=1e-9)


# --------------------------------------------------------------------------- #
# divergence-free skew flux
# --------------------------------------------------------------------------- #
def test_skew_flux_divergence_free(syn):
    assert syn["skew_div_max"] < 1e-9


def test_skew_flux_is_contour_parallel(syn):
    assert syn["skew_dot_grad_max"] < 1e-9


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #
def test_band_diffusivity_helper(cache):
    d = band_diffusivity(cache, "nh_all")
    assert d["skew_fraction"] == pytest.approx(d["K_A"] / d["K_S"])
    assert d["K_S"] > 0


def test_ocean_skew_flips_at_equator(result):
    b = result["bands"]
    assert b["nh_all"]["K_A"] < 0 < b["sh_all"]["K_A"]
    assert b["nh_mid"]["K_A"] < 0 < b["sh_mid"]["K_A"]
    assert result["verdicts"]["ocean_skew_transport_flips_at_equator"]


def test_equator_skew_fraction_is_minimal(result):
    b = result["bands"]
    assert abs(b["eq"]["skew_fraction"]) < abs(b["nh_mid"]["skew_fraction"])
    assert abs(b["eq"]["skew_fraction"]) < abs(b["sh_mid"]["skew_fraction"])


def test_taylor_diffusivity_scale(result):
    for hemi in ("nh_all", "sh_all"):
        assert 500.0 < result["bands"][hemi]["K_S"] < 20000.0
    assert result["verdicts"]["taylor_diffusivity_scale_reasonable"]


def test_all_verdicts_and_figure(result):
    for key in ("skew_ratio_is_clock_ratio",
                "skew_flux_divergence_free",
                "ocean_skew_transport_flips_at_equator",
                "taylor_diffusivity_scale_reasonable"):
        assert result["verdicts"][key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["verdicts"]["ocean_skew_transport_flips_at_equator"] is True
    assert disk["bands"]["sh_all"]["K_A"] == pytest.approx(
        res["bands"]["sh_all"]["K_A"], rel=1e-12)
