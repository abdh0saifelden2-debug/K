"""Unit tests for the Monin-Obukhov stability primitives (synthetic inputs)."""

import numpy as np
import pandas as pd

from neon_pt import stability as S


def test_air_density_ideal_gas():
    p = pd.Series([97.0])  # kPa
    t = pd.Series([283.0])  # K
    rho = S.air_density(p, t).iloc[0]
    # 97000 / (287.052 * 283) ~= 1.194 kg/m^3
    assert abs(rho - 1.194) < 0.01


def test_derive_sign_of_stability():
    # Upward heat flux -> unstable (L<0, zeta<0); downward -> stable.
    df = pd.DataFrame(
        {
            "temp_soni_c": [10.0, 10.0],
            "pres_kpa": [97.0, 97.0],
            "ustar_ms": [0.3, 0.3],
            "H_sensible_wm2": [120.0, -40.0],
            "wind_u_vari": [0.5, 0.5],
            "wind_v_vari": [0.5, 0.5],
            "wind_w_vari": [0.2, 0.2],
            "temp_soni_vari": [0.1, 0.1],
        }
    )
    out = S.derive(df, z_minus_d=40.0)
    assert out["zeta"].iloc[0] < 0  # daytime convective
    assert out["zeta"].iloc[1] > 0  # nighttime stable


def test_transport_efficiency_formula():
    df = pd.DataFrame(
        {
            "temp_soni_c": [10.0],
            "pres_kpa": [97.0],
            "ustar_ms": [0.4],
            "H_sensible_wm2": [100.0],
            "wind_u_vari": [0.6],
            "wind_v_vari": [0.4],  # sigma_hor = 1.0
            "wind_w_vari": [0.25],  # sigma_w = 0.5
            "temp_soni_vari": [0.09],
        }
    )
    out = S.derive(df, z_minus_d=40.0)
    # r_uw = u*^2 / (sigma_hor * sigma_w) = 0.16 / (1.0 * 0.5) = 0.32
    assert abs(out["r_uw"].iloc[0] - 0.32) < 1e-6


def test_summary_ratio_varies_with_made_up_classes():
    n = 400
    rng = np.random.default_rng(0)
    zeta = np.concatenate([rng.uniform(-2, -0.1, n), rng.uniform(0.1, 2, n)])
    df = pd.DataFrame(
        {
            "zeta": zeta,
            "r_uw": np.r_[np.full(n, 0.3), np.full(n, 0.15)],
            "r_wT": np.r_[np.full(n, 0.1), np.full(n, 0.2)],
            "sigma_w": 0.5,
            "ustar_ms": 0.3,
        }
    )
    out = S.stability_summary(df)
    # Ratio in unstable bin should clearly exceed the stable bin.
    assert out.loc["unstable", "transport_ratio"] > out.loc["stable", "transport_ratio"]
