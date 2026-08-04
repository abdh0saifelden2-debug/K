"""Unit tests for §A.2 scallop roughness z_0(λ,a)
(validation/synthetic/a2_z0_roughness.py)."""
import os
import sys

import numpy as np

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import a2_z0_roughness as A2  # noqa: E402


def test_curl_wavelength_matches_anchor():
    lam = A2.lam_from_ustar(0.05)
    assert abs(lam - 2200 * 1.8e-6 / 0.05) < 1e-12
    assert 0.07 < lam < 0.085           # ~7.9 cm, the repo Curl anchor


def test_z0_and_Cd_identities():
    z0 = A2.z0_from_geom(0.008, 0.05)
    assert np.isclose(z0, 0.05 * 0.008)
    Cd = A2.drag_Cd(z0, H=2.4)
    assert np.isclose(Cd, (0.41 / np.log(2.4 / z0)) ** 2)
    # dφ/ds identity
    assert np.isclose(A2.dphi_ds(Cd, u=1.0, H=2.4),
                      1000.0 * Cd * 1.0 ** 2 / 2.4)


def test_drag_increases_with_roughness():
    a = 0.008
    cz = np.array([0.01, 0.05, 0.1])
    Cd = A2.drag_Cd(A2.z0_from_geom(a, cz), H=2.4)
    assert np.all(np.diff(Cd) > 0)      # rougher -> more drag


def test_log_law_buffers_prefactor_uncertainty():
    res = A2.run()
    b = res["buffering"]
    # the headline: 10x roughness uncertainty compresses to <2x in drag
    assert np.isclose(b["input_cz_ratio"], 10.0)
    assert b["output_Cd_ratio"] < 2.0
    assert b["log_law_compression"] > 4.0
    # validity: fully-rough, large scale separation
    assert min(res["H_over_z0_band"]) > 100


def test_settled_and_open_legs():
    res = A2.run()
    # both other legs settled; only the geometry prefactor open; field point honest
    assert "VERIFIED" in res["settled_legs"]["wavelength"]
    assert "amplitude-independent" in res["settled_legs"]["amplitude"]
    assert res["one_field_point"]["in_repo"] is False
    lo, hi = res["Cd_band"]
    assert 1e-3 < lo < hi < 4e-3
