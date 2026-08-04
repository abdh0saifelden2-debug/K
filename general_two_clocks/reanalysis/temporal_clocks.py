r"""The two clocks in TIME: temporal decorrelation of the divergent (fast) vs the
rotational (slow) wind, in real reanalysis.

Companion to `reanalysis/rossby_clocks.py` (which shows the *spatial/energy* two
clocks, `KE_div/KE_rot ~ Ro²`).  Here the **temporal** fingerprint: the repo is named
"two clocks" because the fast (divergent/ageostrophic) flow and the slow
(rotational/balanced) flow evolve on *different time scales*.  Measured directly from
the 6-hourly NCEP/NCAR Reanalysis:

  Prediction:  the divergent wind is set by fast adjustment (gravity waves, convective
            overturning, ageostrophic adjustment) and should decorrelate in TIME much
            faster than the rotational (balanced, advective) wind -- a literal
            fast-clock / slow-clock separation, `tau_div << tau_rot`.

  Validation:  compute the temporal autocorrelation of the vorticity (rotational) and
            divergence (divergent) fields over an extratropical domain and compare
            their e-folding / integral decorrelation times.

Helmholtz / vorticity-divergence machinery reused from `reanalysis/ncep.py`.
"""
from __future__ import annotations

import numpy as np

from reanalysis import ncep


def temporal_acf(block: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized temporal autocorrelation of a (nt, ...) field block, computed from
    per-point time anomalies and averaged over all spatial points.

    ACF(k) = <a(t) a(t+k)> / <a(t)^2>,  a = field - time_mean(field).
    """
    nt = block.shape[0]
    a = block - block.mean(axis=0, keepdims=True)
    denom = float(np.mean(a * a))
    if denom <= 0:
        return np.concatenate([[1.0], np.zeros(max_lag - 1)])
    acf = np.empty(max_lag)
    for k in range(max_lag):
        acf[k] = float(np.mean(a[: nt - k] * a[k:])) / denom
    return acf


def efold_hours(acf: np.ndarray, dt_hours: float) -> float:
    """Lag (in hours) at which the ACF first drops below 1/e."""
    below = np.where(acf < 1.0 / np.e)[0]
    return float((below[0] if below.size else len(acf)) * dt_hours)


def integral_hours(acf: np.ndarray, dt_hours: float) -> float:
    """Integral time scale: dt * (1/2 + sum of ACF up to the first zero crossing)."""
    zero = np.where(acf < 0)[0]
    end = zero[0] if zero.size else len(acf)
    return float((0.5 + np.sum(acf[1:end])) * dt_hours)


def decorrelation_times(u: np.ndarray, v: np.ndarray, lat: np.ndarray,
                        dt_hours: float = 6.0, lat_min: float = 20.0,
                        max_lag: int = 40) -> dict:
    """Temporal decorrelation of the rotational (vorticity) vs divergent (divergence)
    wind over the extratropics (|lat| >= lat_min).  Returns e-folding and integral
    times for each and their ratio (the model predicts tau_rot/tau_div > 1)."""
    nt = u.shape[0]
    ext = np.abs(lat) >= lat_min
    VOR = np.empty((nt, int(ext.sum()), u.shape[2]))
    DIV = np.empty_like(VOR)
    for t in range(nt):
        vor, div = ncep.vorticity_divergence(u[t], v[t], lat)
        VOR[t] = vor[ext]
        DIV[t] = div[ext]
    max_lag = min(max_lag, nt // 2)
    acf_rot = temporal_acf(VOR, max_lag)
    acf_div = temporal_acf(DIV, max_lag)
    tau_rot_e = efold_hours(acf_rot, dt_hours)
    tau_div_e = efold_hours(acf_div, dt_hours)
    return dict(
        dt_hours=dt_hours, max_lag=max_lag,
        acf_rot=acf_rot, acf_div=acf_div,
        tau_rot_efold=tau_rot_e, tau_div_efold=tau_div_e,
        tau_rot_integral=integral_hours(acf_rot, dt_hours),
        tau_div_integral=integral_hours(acf_div, dt_hours),
        ratio_efold=float(tau_rot_e / max(tau_div_e, 1e-9)),
    )
