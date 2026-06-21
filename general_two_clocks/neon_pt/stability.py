"""Monin-Obukhov stability and momentum-vs-heat transport decoupling.

This is the empirical, data-appropriate version of the "single-timescale
K-theory fails" claim. A first-order (K-theory) closure assumes momentum and
heat are mixed by the *same* turbulence with a fixed turbulent Prandtl number,
i.e. K_M and K_H stay locked. Eddy-covariance data lets us check that by
comparing how efficiently the turbulence transports momentum vs heat as a
function of atmospheric stability.

We only have a single measurement level, so we cannot form vertical gradients
and therefore cannot compute K_M, K_H directly. What we *can* compute robustly
from the tower-top sonic statistics are the **transport efficiencies** (the
w-u and w-T correlation coefficients) and standard flux-variance similarity
ratios, both as functions of the stability parameter zeta = (z-d)/L.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KARMAN = 0.4
G = 9.81
CP = 1005.0  # J kg^-1 K^-1, dry air
R_SPECIFIC = 287.052  # J kg^-1 K^-1, dry air


def air_density(pres_kpa: pd.Series, temp_k: pd.Series) -> pd.Series:
    """Air density (kg m^-3) from the ideal gas law; P in kPa -> Pa."""
    return (pres_kpa * 1000.0) / (R_SPECIFIC * temp_k)


def derive(df: pd.DataFrame, z_minus_d: float) -> pd.DataFrame:
    """Add Obukhov length, stability zeta and turbulence scales to ``df``.

    Uses the sonic (virtual) temperature for the buoyancy flux, which is what
    Monin-Obukhov theory wants.
    """
    out = df.copy()
    theta_v = out["temp_soni_c"] + 273.15
    rho = air_density(out["pres_kpa"], theta_v)
    out["rho"] = rho

    ustar = out["ustar_ms"]
    # Kinematic buoyancy (heat) flux w'theta_v' = H / (rho * cp).
    wt = out["H_sensible_wm2"] / (rho * CP)
    out["w_theta"] = wt

    # Obukhov length L and stability parameter zeta.
    L = -(ustar**3) * theta_v / (KARMAN * G * wt)
    out["L"] = L
    out["zeta"] = z_minus_d / L

    # Turbulence scales / standard deviations.
    out["sigma_w"] = np.sqrt(out["wind_w_vari"])
    out["sigma_u"] = np.sqrt(out["wind_u_vari"])
    out["sigma_v"] = np.sqrt(out["wind_v_vari"])
    out["sigma_hor"] = np.sqrt(out["wind_u_vari"] + out["wind_v_vari"])
    out["sigma_T"] = np.sqrt(out["temp_soni_vari"])
    out["Tstar"] = -wt / ustar  # temperature scale

    # Transport efficiencies (|correlation coefficient| of the covariance).
    # Momentum: |u'w'| = ustar^2  ->  r_uw = ustar^2 / (sigma_hor * sigma_w).
    out["r_uw"] = (ustar**2) / (out["sigma_hor"] * out["sigma_w"])
    # Heat: r_wT = w'T' / (sigma_w * sigma_T).
    out["r_wT"] = wt / (out["sigma_w"] * out["sigma_T"])
    return out


@dataclass
class StabilityBin:
    label: str
    lo: float
    hi: float


# Standard surface-layer stability classes in zeta = (z-d)/L.
STABILITY_BINS = [
    StabilityBin("strongly unstable", -np.inf, -0.5),
    StabilityBin("unstable", -0.5, -0.1),
    StabilityBin("near-neutral", -0.1, 0.1),
    StabilityBin("stable", 0.1, 0.5),
    StabilityBin("strongly stable", 0.5, np.inf),
]


def stability_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Median transport efficiencies and counts per stability class."""
    rows = []
    for b in STABILITY_BINS:
        m = (df["zeta"] > b.lo) & (df["zeta"] <= b.hi)
        sub = df[m]
        # Keep physically sensible efficiencies only.
        r_uw = sub["r_uw"].where((sub["r_uw"] > 0) & (sub["r_uw"] < 1.0))
        r_wT = sub["r_wT"].abs().where(sub["r_wT"].abs() < 1.0)
        rows.append(
            {
                "class": b.label,
                "n": int(sub["zeta"].notna().sum()),
                "median_r_uw": float(r_uw.median()),
                "median_abs_r_wT": float(r_wT.median()),
            }
        )
    out = pd.DataFrame(rows).set_index("class")
    # Prandtl-like proxy: momentum efficiency relative to heat efficiency.
    out["transport_ratio"] = out["median_r_uw"] / out["median_abs_r_wT"]
    return out


def clean_for_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot yield a meaningful stability/efficiency estimate."""
    needed = ["zeta", "r_uw", "r_wT", "sigma_w", "ustar_ms"]
    out = df.dropna(subset=needed).copy()
    # Remove near-zero friction velocity (decoupled / calm) and absurd zeta.
    out = out[(out["ustar_ms"] > 0.05) & (out["zeta"].abs() < 50)]
    return out
