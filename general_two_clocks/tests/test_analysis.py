"""Unit tests for the analysis primitives, using synthetic signals.

These do not require the (large) NEON download: we build signals with known
lags and periods and check the analysis recovers them.
"""

import numpy as np
import pandas as pd

from neon_pt import analysis as A


def _frame(n=24 * 2 * 20):  # ~20 days at 30-min cadence
    idx = pd.date_range("2020-01-01", periods=n, freq="30min")
    hour = idx.hour + idx.minute / 60.0
    return idx, hour


def test_lagged_xcorr_recovers_known_lag():
    idx, _ = _frame()
    t = (idx - idx[0]).total_seconds().to_numpy() / 3600.0
    base = np.sin(2 * np.pi * t / 24.0)
    a = pd.Series(base, index=idx)
    # b lags a by 2 hours (4 samples).
    b = pd.Series(np.roll(base, 4), index=idx)
    _, _, best = A.lagged_xcorr(a, b, max_lag=12)
    assert abs(best - 2.0) <= 0.5


def test_spectrum_finds_diurnal_and_semidiurnal():
    idx, _ = _frame()
    t = (idx - idx[0]).total_seconds().to_numpy() / 3600.0
    sig = np.sin(2 * np.pi * t / 24.0) + 0.5 * np.sin(2 * np.pi * t / 12.0)
    s = pd.Series(sig, index=idx)
    spec = A.lomb_scargle_spectrum(s, np.linspace(6, 48, 2000))
    p24, _ = A.find_band_peak(spec, 20, 28)
    p12, _ = A.find_band_peak(spec, 10.5, 13.5)
    assert abs(p24 - 24.0) < 1.0
    assert abs(p12 - 12.0) < 1.0


def test_coupling_decoupled_when_density_varies():
    idx, hour = _frame()
    # Temperature: clean diurnal swing in Kelvin.
    tk = 280.0 + 5.0 * np.sin(2 * np.pi * hour / 24.0)
    # Pressure: independent slow drift -> should be ~uncorrelated with T.
    t = (idx - idx[0]).total_seconds().to_numpy() / 3600.0
    p = 97.0 + 0.5 * np.sin(2 * np.pi * t / 80.0)
    df = pd.DataFrame({"temp_air_k": tk, "pres_kpa": p}, index=idx)
    res = A.coupling_tests(df)
    assert res.r_squared < 0.2
    # Ideal-gas constant-density slope is steep and positive (~0.33 kPa/K).
    assert 0.2 < res.ideal_gas_slope < 0.45


def test_qc_drops_out_of_range():
    idx, _ = _frame(n=10)
    df = pd.DataFrame(
        {
            "pres_kpa": [97.0] * 9 + [5000.0],
            "temp_air_c": [3.0] * 10,
        },
        index=idx,
    )
    out = A.apply_qc(df)
    assert out["pres_kpa"].isna().sum() == 1
    assert out["temp_air_c"].isna().sum() == 0
