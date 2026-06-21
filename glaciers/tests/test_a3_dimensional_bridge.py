"""Unit tests for §A.3 dimensional bridge
(validation/synthetic/a3_dimensional_bridge.py)."""
import os
import sys

import numpy as np

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import a3_dimensional_bridge as A3  # noqa: E402


def test_v_o_linear_in_inputs():
    base = A3.v_o_m2_s(10.0, 50.0)
    assert np.isclose(A3.v_o_m2_s(20.0, 50.0), 2 * base)     # ∝ Q
    assert np.isclose(A3.v_o_m2_s(10.0, 100.0), 2 * base)    # ∝ ∂φ/∂s


def test_k_creep_scales_as_N_cubed():
    k1 = A3.k_creep_per_s(2.4e-24, 1.0e6)
    k2 = A3.k_creep_per_s(2.4e-24, 2.0e6)
    assert np.isclose(k2 / k1, 8.0, rtol=1e-6)              # n=3
    # linear in A
    assert np.isclose(A3.k_creep_per_s(4.8e-24, 1.0e6) / k1, 2.0)


def test_steady_area_and_radius_identities():
    vo = A3.v_o_m2_s(10.0, 50.0)
    kc = A3.k_creep_per_s(2.4e-24, 1.0e6)
    S = A3.steady_area_m2(vo, kc)
    assert np.isclose(S, vo / kc)
    assert np.isclose(A3.radius_m(S), np.sqrt(2 * S / np.pi))


def test_central_point_is_metre_scale():
    res = A3.run()
    c = res["central"]
    assert 1.0 < c["R_star_m"] < 5.0          # ~2.4 m
    assert 0.05 < c["tau_yr"] < 1.0           # ~0.18 yr (sub-annual)


def test_scallop_fraction_is_calibration_free():
    res = A3.run()
    sf = res["scallop_fraction"]
    # ρ_iL cancels: ΔS/S = V_scallop/V_o = 0.33 exactly; ΔR/R = sqrt(1.33)-1
    assert np.isclose(sf["dS_over_S"], 0.33)
    assert np.isclose(sf["dR_over_R"], np.sqrt(1.33) - 1.0, atol=1e-6)
    # explicit: the ratio is independent of ρ_iL
    vo_a = A3.v_o_m2_s(10.0, 50.0, rhoL=3.0e8)
    vo_b = A3.v_o_m2_s(10.0, 50.0, rhoL=2.0e8)
    assert not np.isclose(vo_a, vo_b)                 # V_o itself depends on ρ_iL
    # but V_scallop/V_o (= 0.33) does not -> fraction is ρ_iL-free by construction


def test_calibration_isolated_to_gain_g():
    res = A3.run()
    cal = res["calibration"]
    assert "g" in cal["knob"].lower()
    assert cal["derived_does_not_need_g"] is True
    assert res["metre_scale_fraction"] > 0.5         # majority of band is R-channel scale
