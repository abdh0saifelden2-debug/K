"""Unit proofs for the §A.2 field-point closure (`a2_field_point.py`)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "validation"))
from synthetic.a2_field_point import (  # noqa: E402
    B_L_SCALLOP, B_S_SAND, KAPPA, c_z, drag_coefficient, ks_over_L, run,
    z0_over_L)


def test_loglaw_identity_is_exact():
    # (1/k) ln(y/z0) == (1/k) ln(y/L) + B_L  for every y  <=>  z0 = L e^{-kB}
    y = np.array([0.1, 1.0, 10.0])
    L = 0.079
    z0 = z0_over_L() * L
    lhs = (1 / KAPPA) * np.log(y / z0)
    rhs = (1 / KAPPA) * np.log(y / L) + B_L_SCALLOP
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_magnitudes_documented():
    assert abs(z0_over_L(kappa=0.40) - np.exp(-3.76)) < 1e-12
    assert 0.020 < z0_over_L(kappa=0.40) < 0.024
    assert 0.019 < z0_over_L(kappa=0.41) < 0.023
    # sand-grain equivalent ~ 0.70 L32
    assert 0.65 < ks_over_L() < 0.75


def test_nikuradse_consistency():
    # a hypothetical surface with B_L == sand's 8.5 must give k_s == L exactly
    assert abs(ks_over_L(B_L=B_S_SAND) - 30.0 * np.exp(-KAPPA * 8.5)) < 1e-12
    assert abs(30.0 * np.exp(-KAPPA * 8.5) - 1.0) < 0.01   # z0=k_s/30 anchor


def test_rougher_surface_means_lower_B():
    # lower additive constant = rougher wall = larger z0 (monotone decreasing)
    assert z0_over_L(B_L=8.0) > z0_over_L(B_L=9.4) > z0_over_L(B_L=11.0)


def test_cz_exceeds_assumed_band_but_Cd_is_buffered():
    out = run()
    amp = out["amplitude_form"]
    assert amp["exceeds_assumed_band"] is True
    assert min(amp["alpha_s_range"]) > 3.0          # above the old band top
    au = out["anchor_update"]
    # ~20x above the band-bottom prefactor, yet C_d moves <= 2.2x
    assert au["z0_shift_factor_vs_band_top"] > 2.0
    assert max(au["C_d_shift_factor_vs_band"]) < 2.2
    assert au["C_d_new"] == pytest.approx(
        drag_coefficient(au["H_m"], au["z0_new_m"]))


def test_bl_systematic_is_bounded():
    out = run()
    assert out["b_l_systematic"]["B_L_pm1_z0_factor"] == pytest.approx(
        np.exp(KAPPA), rel=1e-12)
    # +/-1 in B_L never takes alpha_s back inside the old [0.3, 3] band
    assert 30.0 * c_z(0.10, B_L=B_L_SCALLOP + 1.0) > 3.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
