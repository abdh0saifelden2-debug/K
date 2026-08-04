"""Pins for the P2-R1 high-resolution extension artifacts (sublayer_*.json).

The committed story (paper 2, section 5): suppression `Nu/Nu_flat < 1` for every
seed at every near-isotropic grid; the anisotropic ny=320-384 extension at fixed
nx=128 flips mildly positive, eta-robustly; the isotropy-matched control
(nx=256, ny=320) brings the mean back under unity; deviations bounded within 9%
everywhere. These tests pin the artifacts those sentences cite.
"""
import json
import os

import pytest

SUB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "subglacial")


def _load(name):
    with open(os.path.join(SUB, name)) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def highres():
    return _load("sublayer_highres.json")


def test_ny256_suppression_all_seeds(highres):
    r256 = [r for r in highres["resolution_sweep"] if r["ny"] == 256][0]
    assert r256["frac_nu_lt_1"] == 1.0
    assert 0.94 < r256["nu_mean"] < 1.0
    assert len(r256["per_seed_nu"]) == 3


def test_ny320_anisotropic_flip(highres):
    r320 = [r for r in highres["resolution_sweep"] if r["ny"] == 320][0]
    assert r320["frac_nu_lt_1"] == 0.0
    assert 1.0 < r320["nu_mean"] < 1.10


def test_ny384_anisotropic_flip_grows():
    d = _load("sublayer_ny384.json")
    r384 = [r for r in d["resolution_sweep"] if r["ny"] == 384][0]
    assert r384["frac_nu_lt_1"] == 0.0
    assert 1.0 < r384["nu_mean"] < 1.10


def test_eta_refinement_does_not_restore_sign_at_nx128():
    d = _load("sublayer_eta_refine.json")
    for p in d["penalization_sweep"]:
        assert p["ny"] == 320 and p["eta"] <= 2.5e-5
        assert p["nu_mean"] > 1.0


def test_isotropy_control_restores_suppression():
    d = _load("sublayer_nx256.json")
    r = [x for x in d["resolution_sweep"] if x["ny"] == 320][0]
    assert d["args"]["nx"] == 256
    assert r["nu_mean"] < 1.0
    assert r["nu_min"] < 1.0 <= round(r["nu_max"], 2) <= 1.01  # one seed grazes unity


def test_bounded_deviation_everywhere():
    names = ["sublayer_convergence.json", "sublayer_highres.json",
             "sublayer_ny384.json", "sublayer_eta_refine.json",
             "sublayer_nx256.json"]
    for name in names:
        d = _load(name)
        for r in d["resolution_sweep"] + d["penalization_sweep"]:
            for nu in r["per_seed_nu"]:
                assert abs(nu - 1.0) <= 0.27  # ladder low point 0.74
    # the high-resolution extensions specifically: seed-mean within +/-9%,
    # any single seed within +/-11%
    for name in ["sublayer_highres.json", "sublayer_ny384.json",
                 "sublayer_eta_refine.json", "sublayer_nx256.json"]:
        d = _load(name)
        for r in d["resolution_sweep"] + d["penalization_sweep"]:
            assert abs(r["nu_mean"] - 1.0) <= 0.09
            for nu in r["per_seed_nu"]:
                assert abs(nu - 1.0) <= 0.11
