"""Analyses that probe the user's narrative about temperature and pressure.

The hypothesis (paraphrased): solar heating is uneven, hot air becomes buoyant
and rises, the resulting low pressure draws in air streams, and those streams
shear against each other. Crucially, the claim is that **pressure and temperature
run on two different "clocks"** and that a kinetic-theory style picture is wrong
to treat them as locked together (proportional).

Each function below returns plain numbers / arrays so the results can be both
plotted and written into a Markdown report. We let the data decide; nothing is
tuned to a desired conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal, stats

# ----------------------------------------------------------------------------
# Quality control: gross physical range checks to drop obvious spikes.
# ----------------------------------------------------------------------------
_QC_LIMITS = {
    "pres_kpa": (80.0, 110.0),
    "temp_air_c": (-40.0, 50.0),
    "sw_in_wm2": (-20.0, 1400.0),
    "H_sensible_wm2": (-300.0, 800.0),
    "LE_latent_wm2": (-150.0, 800.0),
    "ustar_ms": (0.0, 3.0),
    "wind_speed_ms": (0.0, 40.0),
    "wind_w_vari": (0.0, 20.0),
}


def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in _QC_LIMITS.items():
        if col in out.columns:
            bad = (out[col] < lo) | (out[col] > hi)
            out.loc[bad, col] = np.nan
    return out


# ----------------------------------------------------------------------------
# Diurnal composites
# ----------------------------------------------------------------------------
def diurnal_composite(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean value of each column as a function of local half-hour of day.

    The ``% 24.0`` folds the rounded bin back into ``[0, 24)`` so that
    sub-hourly data (e.g. ASOS 1-min, where ``hour_local`` is a continuous
    float) near midnight is not split into a spurious ``24.0`` bin separate
    from ``0.0``.
    """
    bins = ((df["hour_local"] * 2).round() / 2.0) % 24.0
    grouped = df.groupby(bins)[cols].mean()
    grouped.index.name = "hour_local"
    return grouped


def peak_hour(series: pd.Series) -> float:
    """Local hour at which a diurnal composite reaches its maximum."""
    return float(series.idxmax())


def trough_hour(series: pd.Series) -> float:
    return float(series.idxmin())


# ----------------------------------------------------------------------------
# Lead / lag cross-correlation
# ----------------------------------------------------------------------------
def lagged_xcorr(a: pd.Series, b: pd.Series, max_lag: int = 12):
    """Cross-correlation of two series across +/- ``max_lag`` samples (30-min each).

    Returns (lags_hours, corr). A positive peak lag means ``b`` lags ``a``
    (i.e. ``a`` leads ``b``).
    """
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    x = (joined["a"] - joined["a"].mean()).to_numpy()
    y = (joined["b"] - joined["b"].mean()).to_numpy()
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []
    for k in lags:
        if k < 0:
            c = np.corrcoef(x[-k:], y[: len(y) + k])[0, 1]
        elif k > 0:
            c = np.corrcoef(x[: len(x) - k], y[k:])[0, 1]
        else:
            c = np.corrcoef(x, y)[0, 1]
        corr.append(c)
    corr = np.asarray(corr)
    lag_hours = lags * 0.5
    best = lag_hours[int(np.nanargmax(corr))]
    return lag_hours, corr, best


# ----------------------------------------------------------------------------
# Spectral analysis ("two clocks")
# ----------------------------------------------------------------------------
@dataclass
class Spectrum:
    periods_h: np.ndarray
    power: np.ndarray
    peaks_h: list = field(default_factory=list)

    def power_at(self, period_h: float) -> float:
        i = int(np.argmin(np.abs(self.periods_h - period_h)))
        return float(self.power[i])


def lomb_scargle_spectrum(series: pd.Series, periods_h: np.ndarray) -> Spectrum:
    """Normalised Lomb-Scargle periodogram, robust to the data's gaps.

    Time is measured in hours from the first sample. The series is mean-removed.
    """
    s = series.dropna()
    t = (s.index - s.index[0]).total_seconds().to_numpy() / 3600.0
    y = s.to_numpy() - s.to_numpy().mean()
    ang = 2.0 * np.pi / periods_h
    power = signal.lombscargle(t, y, ang, normalize=True)
    return Spectrum(periods_h=periods_h, power=power)


def dominant_period(spec: Spectrum, lo_h: float, hi_h: float) -> float:
    """Period (hours) of the single strongest spectral peak in a wide band."""
    return find_band_peak(spec, lo_h, hi_h)[0]


def find_band_peak(spec: Spectrum, lo_h: float, hi_h: float) -> tuple[float, float]:
    """Return (period, power) of the strongest peak within a period band."""
    mask = (spec.periods_h >= lo_h) & (spec.periods_h <= hi_h)
    if not mask.any():
        return float("nan"), float("nan")
    sub_p = spec.periods_h[mask]
    sub_pow = spec.power[mask]
    i = int(np.argmax(sub_pow))
    return float(sub_p[i]), float(sub_pow[i])


# ----------------------------------------------------------------------------
# Pressure / temperature coupling ("is K-theory right to make them equal?")
# ----------------------------------------------------------------------------
@dataclass
class CouplingResult:
    pearson_r: float
    pearson_p: float
    slope_kpa_per_k: float
    r_squared: float
    ideal_gas_slope: float
    density_cv_pct: float
    detrended_r: float


def coupling_tests(df: pd.DataFrame) -> CouplingResult:
    """Quantify how tightly bulk P and T move together.

    Kinetic theory for a fixed parcel gives P = rho * R_specific * T, i.e. at
    fixed density P should be a straight, steep line in T (~0.33 kPa/K near the
    surface). We test the *bulk atmospheric* P and T instead.
    """
    sub = df[["pres_kpa", "temp_air_k"]].dropna()
    p = sub["pres_kpa"].to_numpy()
    tk = sub["temp_air_k"].to_numpy()

    r, pval = stats.pearsonr(tk, p)
    slope, intercept, rval, _, _ = stats.linregress(tk, p)

    # Ideal-gas expectation if density were constant: dP/dT = P/T.
    ideal = float(np.mean(p / tk))

    # Implied air density rho = P/(R T); coefficient of variation shows it is the
    # density (not a fixed P/T lock) that actually varies.
    r_specific = 0.287052  # kJ kg^-1 K^-1 for dry air -> P[kPa]=rho*R*T
    rho = p / (r_specific * tk)
    density_cv = float(np.std(rho) / np.mean(rho) * 100.0)

    # Remove the slow synoptic trend (24h rolling) to compare fast variability.
    pr = (sub["pres_kpa"] - sub["pres_kpa"].rolling(48, center=True, min_periods=12).mean())
    tr = (sub["temp_air_k"] - sub["temp_air_k"].rolling(48, center=True, min_periods=12).mean())
    det = pd.concat([pr, tr], axis=1).dropna()
    det_r = float(np.corrcoef(det.iloc[:, 0], det.iloc[:, 1])[0, 1])

    return CouplingResult(
        pearson_r=float(r),
        pearson_p=float(pval),
        slope_kpa_per_k=float(slope),
        r_squared=float(rval**2),
        ideal_gas_slope=ideal,
        density_cv_pct=density_cv,
        detrended_r=det_r,
    )


# ----------------------------------------------------------------------------
# Shear / momentum ("uneven streams shearing against each other")
# ----------------------------------------------------------------------------
@dataclass
class ShearResult:
    r_ustar_wind: float
    r_turb_wind: float
    r_turb_heat: float
    wind_dir_std_deg: float


def shear_tests(df: pd.DataFrame) -> ShearResult:
    """Relate momentum flux / turbulence to mechanical shear and buoyancy."""

    def safe_r(x, y):
        d = pd.concat([x, y], axis=1).dropna()
        if len(d) < 10:
            return float("nan")
        return float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])

    r_uw = safe_r(df["wind_speed_ms"], df["ustar_ms"])
    r_tw = safe_r(df["wind_speed_ms"], df["wind_w_vari"])
    r_th = safe_r(df["H_sensible_wm2"].clip(lower=0), df["wind_w_vari"])

    # Circular spread of wind direction (degrees) -> "uneven streams".
    ang = np.deg2rad(df["wind_dir_deg"].dropna().to_numpy())
    rbar = np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang)))
    circ_std = float(np.rad2deg(np.sqrt(-2.0 * np.log(max(rbar, 1e-9)))))

    return ShearResult(
        r_ustar_wind=r_uw,
        r_turb_wind=r_tw,
        r_turb_heat=r_th,
        wind_dir_std_deg=circ_std,
    )
