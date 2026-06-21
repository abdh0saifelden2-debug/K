r"""Paper 4 — Non-Markovian, multiscale timescale memory in per-float Argo records.

PRE-REGISTERED, falsification-driven. This is the **ocean validation** of the
terrestrial two-clocks/Mori–Zwanzig result: a *local Markov* (memoryless) model is
structurally insufficient; the real field carries memory across multiple timescales
(the MZ memory kernel, REPORT_THEORY.md / REPORT_GLE_COEFFICIENTS.md). For a
first-order Markov (AR(1)) process the autocorrelation is a single exponential
C(τ)=ρ^τ and obeys C(2)=C(1)² (Chapman–Kolmogorov). Memory breaks both: C(τ) needs
≥2 timescales and C(2) > C(1)².

Data: per-float core-Argo temperature at a fixed pressure level (~1000 dbar, near
parking depth), one value per ~10-day cycle. Per float we deseasonalize (remove mean
+ annual harmonic), bin to a uniform 10-day grid, and estimate the autocorrelation
C(τ); we aggregate across many long-record floats. Public Argo GDAC (Ifremer), no auth.

PRE-REGISTERED PREDICTIONS (thresholds fixed before the real data)
-----------------------------------------------------------------
  NM1  Chapman-Kolmogorov violation (the exact, parameter-free non-Markov test):
       the excess Δ₂ = C(2) − C(1)² is positive and robust across floats (median > 0,
       one-sided sign test p < 0.01). AR(1)/Markov predicts exactly Δ₂ = 0.
  NM2  Memory persists past lag 1: the lag-3 excess Δ₃ = C(3) − C(1)³ is also positive
       and robust across floats (sign test p < 0.01) — again exactly 0 for AR(1).
  NM3  Two separated timescales: a double-exponential beats a single exponential by
       ΔAIC > 50 (substantial, not sampling-noise overfit) with τ_slow/τ_fast > 3.
  NM4  Genuine ocean memory: the slow timescale τ_slow > 60 days (months), not a
       fast measurement-noise artifact.

`synthetic` AR(1) vs two-timescale generators validate the method (AR(1) → single-exp
preferred, excess ≈ 0; two-timescale → double-exp preferred, excess > 0). Uses scipy.
"""
from __future__ import annotations


import numpy as np
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------- #
# Autocorrelation + heavy-memory model fitting
# --------------------------------------------------------------------------- #
def acf_nan(x, maxlag):
    """NaN-aware (pairwise) autocorrelation up to `maxlag`. C[0]=1."""
    x = np.asarray(x, float)
    xm = x - np.nanmean(x)
    var = np.nanmean(xm * xm)
    out = np.full(maxlag + 1, np.nan)
    out[0] = 1.0
    for k in range(1, maxlag + 1):
        a, b = xm[:-k], xm[k:]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 10 and var > 0:
            out[k] = float(np.mean(a[m] * b[m]) / var)
    return out


def _single(tau, t1):
    return np.exp(-tau / t1)


def _double(tau, a, t1, t2):
    return a * np.exp(-tau / t1) + (1.0 - a) * np.exp(-tau / t2)


def fit_models(C, step=10.0):
    """Fit single- vs double-exponential to C(τ); return timescales + AIC choice."""
    lags = np.arange(len(C))
    m = np.isfinite(C)
    tau = lags[m] * step
    c = C[m]
    n = len(c)

    def aic(res, k):
        return n * np.log(np.sum(res ** 2) / n + 1e-300) + 2 * k

    try:
        p1, _ = curve_fit(_single, tau, c, p0=[100.0], bounds=(1.0, 1e5), maxfev=10000)
        a1 = aic(c - _single(tau, *p1), 1)
        t1 = float(p1[0])
    except Exception:
        t1, a1 = np.nan, np.inf
    try:
        p2, _ = curve_fit(_double, tau, c, p0=[0.5, 30.0, 300.0],
                          bounds=([0.0, 1.0, 1.0], [1.0, 1e4, 1e5]), maxfev=30000)
        a2 = aic(c - _double(tau, *p2), 3)
        tf, ts = sorted([float(p2[1]), float(p2[2])])
        w = float(p2[0])
    except Exception:
        tf, ts, w, a2 = np.nan, np.nan, np.nan, np.inf
    return dict(tau_single=t1, aic_single=a1, tau_fast=tf, tau_slow=ts, weight=w,
                aic_double=a2, d_aic=float(a1 - a2),
                double_preferred=bool(a2 < a1 - 50.0),
                sep=(ts / tf if (tf and np.isfinite(tf) and tf > 0) else np.nan))


def nonmarkov_excess_k(C, k):
    """C(k) − C(1)^k: zero for AR(1)/Markov at every k, positive under memory."""
    if len(C) > k and np.isfinite(C[1]) and np.isfinite(C[k]):
        return float(C[k] - C[1] ** k)
    return np.nan


def nonmarkov_excess(C):
    """C(2) − C(1)²: zero for AR(1)/Markov, positive for memory."""
    if len(C) > 2 and np.isfinite(C[1]) and np.isfinite(C[2]):
        return float(C[2] - C[1] ** 2)
    return np.nan


# --------------------------------------------------------------------------- #
# Per-float loading (core Argo *_prof.nc) and deseasonalized anomaly
# --------------------------------------------------------------------------- #
def load_float_level(path, target_p=1000.0, var="TEMP"):
    """(juld_days, value) at `target_p` dbar for each profile of a float."""
    import xarray as xr
    ds = xr.open_dataset(path, decode_times=False)

    def pick(name):
        adj = name + "_ADJUSTED"
        if adj in ds and np.isfinite(np.asarray(ds[adj].values)).any():
            return np.asarray(ds[adj].values, float)
        return np.asarray(ds[name].values, float)

    P, V = pick("PRES"), pick(var)
    juld = np.asarray(ds["JULD"].values, float)
    ds.close()
    out_t, out_v = [], []
    for i in range(P.shape[0]):
        p, v = P[i], V[i]
        msk = np.isfinite(p) & np.isfinite(v)
        if msk.sum() < 5 or not np.isfinite(juld[i]):
            continue
        p, v = p[msk], v[msk]
        o = np.argsort(p)
        p, v = p[o], v[o]
        if p.min() > target_p + 60 or p.max() < target_p - 60:
            continue
        out_t.append(juld[i])
        out_v.append(float(np.interp(target_p, p, v)))
    return np.array(out_t), np.array(out_v)


def to_uniform_anomaly(t_days, v, step=10.0, min_pts=40):
    """Bin to a uniform step-day grid; remove mean + annual harmonic -> anomaly."""
    if len(t_days) < min_pts:
        return None
    t0 = t_days.min()
    bins = np.floor((t_days - t0) / step).astype(int)
    nb = int(bins.max()) + 1
    grid = np.full(nb, np.nan)
    for b in range(nb):
        sel = bins == b
        if sel.any():
            grid[b] = np.nanmean(v[sel])
    tt = np.arange(nb) * step
    valid = np.isfinite(grid)
    if valid.sum() < min_pts:
        return None
    X = np.column_stack([np.ones(valid.sum()),
                         np.sin(2 * np.pi * tt[valid] / 365.25),
                         np.cos(2 * np.pi * tt[valid] / 365.25)])
    coef, *_ = np.linalg.lstsq(X, grid[valid], rcond=None)
    full = np.column_stack([np.ones(nb),
                            np.sin(2 * np.pi * tt / 365.25),
                            np.cos(2 * np.pi * tt / 365.25)])
    return grid - full @ coef


def analyze_floats(paths, target_p=1000.0, var="TEMP", maxlag=30, step=10.0):
    acfs, excess, excess3, nvalid = [], [], [], []
    for p in paths:
        try:
            t, v = load_float_level(p, target_p, var)
        except Exception:
            continue
        anom = to_uniform_anomaly(t, v, step)
        if anom is None:
            continue
        C = acf_nan(anom, maxlag)
        e2, e3 = nonmarkov_excess(C), nonmarkov_excess_k(C, 3)
        if not np.isfinite(e2):
            continue
        acfs.append(C)
        excess.append(e2)
        excess3.append(e3)
        nvalid.append(int(np.isfinite(anom).sum()))
    if len(acfs) < 5:
        return dict(n_floats=len(acfs), ok=False)
    acfs = np.array(acfs)
    Cmed = np.nanmedian(acfs, axis=0)
    fits = fit_models(Cmed, step)
    excess = np.array(excess)
    excess3 = np.array(excess3)
    e3v = excess3[np.isfinite(excess3)]
    from scipy.stats import binomtest
    n_pos = int((excess > 0).sum())
    p_sign = float(binomtest(n_pos, len(excess), 0.5, alternative="greater").pvalue)
    n_pos3 = int((e3v > 0).sum())
    p_sign3 = float(binomtest(n_pos3, len(e3v), 0.5, alternative="greater").pvalue) if len(e3v) else 1.0

    nm1 = bool(np.median(excess) > 0 and p_sign < 0.01)
    nm2 = bool(len(e3v) and np.median(e3v) > 0 and p_sign3 < 0.01)
    nm3 = bool(fits["double_preferred"] and np.isfinite(fits["sep"]) and fits["sep"] > 3.0)
    nm4 = bool(np.isfinite(fits["tau_slow"]) and fits["tau_slow"] > 60.0)
    n_pass = int(nm1) + int(nm2) + int(nm3) + int(nm4)
    return dict(n_floats=len(acfs), Cmed=Cmed, fits=fits,
                excess_median=float(np.median(excess)), excess3_median=float(np.median(e3v)) if len(e3v) else np.nan,
                frac_pos=float(n_pos / len(excess)), p_sign=p_sign, p_sign3=p_sign3,
                median_nvalid=float(np.median(nvalid)),
                nm1=nm1, nm2=nm2, nm3=nm3, nm4=nm4, n_pass=n_pass,
                ok=bool(n_pass >= 3), maxlag=maxlag, step=step)


# --------------------------------------------------------------------------- #
# Deterministic synthetic validators
# --------------------------------------------------------------------------- #
def ar1(n=2000, phi=0.8, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    return x


def two_timescale(n=2000, phi_fast=0.5, phi_slow=0.97, w=0.5, seed=0):
    """Sum of a fast and a slow AR(1) -> non-Markovian, two-timescale memory."""
    rng = np.random.default_rng(seed)
    a = np.zeros(n)
    b = np.zeros(n)
    for i in range(1, n):
        a[i] = phi_fast * a[i - 1] + rng.standard_normal()
        b[i] = phi_slow * b[i - 1] + rng.standard_normal()
    a /= a.std()
    b /= b.std()
    return w * a + (1 - w) * b


def main():
    print("=== synthetic validation ===")
    C = acf_nan(ar1(phi=0.8, seed=0), 30)
    f = fit_models(C, step=1.0)
    print(f"  AR(1):  ΔAIC={f['d_aic']:+.1f} (double_pref={f['double_preferred']}, want False)  "
          f"Δ₂={nonmarkov_excess(C):+.4f} Δ₃={nonmarkov_excess_k(C,3):+.4f} (want ~0)")
    C2 = acf_nan(two_timescale(seed=0), 30)
    f2 = fit_models(C2, step=1.0)
    print(f"  2-scale: ΔAIC={f2['d_aic']:+.1f} (double_pref={f2['double_preferred']}, want True)  "
          f"sep={f2['sep']:.1f}  Δ₂={nonmarkov_excess(C2):+.4f} Δ₃={nonmarkov_excess_k(C2,3):+.4f} (want >0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
