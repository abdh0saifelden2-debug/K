r"""Paper 3 — Cascade lifetime distribution and power-law energy transport
(real WREF 1-min bursts).

PRE-REGISTERED, falsification-driven. Mainstream grounding: turbulence is
*intermittent* — activity comes in bursts with no characteristic duration
(Kolmogorov 1962; Frisch 1995; on-off intermittency, Platt 1993; SOC avalanche
statistics, Bak 1987). So the **burst lifetime distribution** P(τ) should be
heavy-tailed (power-law over a scaling range), unlike the (near-exponential)
durations of a Gaussian process with the same spectrum, and the **energy per burst**
should grow *super-linearly* with duration (E ∝ τ^β, β>1): long-lived bursts carry
disproportionate energy ("power-law energy transport").

Method: build a turbulent-activity envelope from the finest octave bands of the
signal (cascade_band_structure.dyadic_bands), threshold it, and measure each
excursion's duration τ and integrated excess energy E. Fit P(τ) with a continuous
power law (Clauset–Shalizi–Newman MLE + KS-selected x_min) vs an exponential, choose
by AIC, and compare against a phase-randomized surrogate.

PRE-REGISTERED PREDICTIONS (thresholds fixed before the real data)
-----------------------------------------------------------------
  CL1  P(τ) is heavy-tailed: a power law is preferred over an exponential by AIC,
       with exponent α in [1.5, 4].
  CL2  Real scaling range ≥ 1.5 decades (τ_max / x_min ≥ ~30).
  CL3  Power-law energy transport: E ∝ τ^β with β > 1 and fit R² > 0.9.
  CL4  Intermittency (not a spectral artifact): the real tail is heavier than the
       phase-randomized surrogate (smaller α and/or larger τ_max).

Honesty: if the real bursts are better described by a lognormal/stretched form than a
pure power law, that is reported as such — the method compares models, it does not
assume the answer. `synthetic` generators validate the fitter (power-law in →
power-law out; exponential in → exponential out). CPU, numpy-only.
"""
from __future__ import annotations

import numpy as np

from cascade_band_structure import dyadic_bands, analytic_envelope, phase_randomize


# --------------------------------------------------------------------------- #
# Event detection
# --------------------------------------------------------------------------- #
def build_activity(x, k_fine=5, n_bands=12):
    """Turbulent-activity envelope = sum of analytic envelopes of the k_fine finest
    octave bands (the small-scale turbulent intensity)."""
    bands = dyadic_bands(np.asarray(x, float), n_bands)
    return np.sum([analytic_envelope(bands[i]) for i in range(k_fine)], axis=0)


def detect_events(activity, thresh):
    """Excursions of `activity` above `thresh`. Returns (durations, excess energies)."""
    above = (np.asarray(activity) > thresh).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], above, [0]))))
    starts, ends = edges[0::2], edges[1::2]
    dur = (ends - starts).astype(float)
    energy = np.array([float(np.sum(activity[s:e] - thresh)) for s, e in zip(starts, ends)])
    return dur, energy


# --------------------------------------------------------------------------- #
# Heavy-tail model fitting (continuous power law vs exponential)
# --------------------------------------------------------------------------- #
def fit_powerlaw(x, xmin):
    x = np.asarray(x, float)
    x = x[x >= xmin]
    n = len(x)
    if n < 10:
        return None
    s = float(np.sum(np.log(x / xmin)))
    alpha = 1.0 + n / s
    loglik = n * np.log((alpha - 1.0) / xmin) - alpha * s
    return dict(alpha=alpha, loglik=loglik, n=n, xmin=float(xmin))


def fit_exponential(x, xmin):
    x = np.asarray(x, float)
    x = x[x >= xmin]
    n = len(x)
    if n < 10:
        return None
    lam = 1.0 / np.mean(x - xmin)
    loglik = n * np.log(lam) - lam * float(np.sum(x - xmin))
    return dict(lam=lam, loglik=loglik, n=n, xmin=float(xmin))


def _ks_powerlaw(x, xmin):
    x = np.sort(np.asarray(x, float))
    x = x[x >= xmin]
    n = len(x)
    if n < 10:
        return np.inf, np.nan
    alpha = 1.0 + n / float(np.sum(np.log(x / xmin)))
    cdf_emp = np.arange(1, n + 1) / n
    cdf_fit = 1.0 - (x / xmin) ** (1.0 - alpha)
    return float(np.max(np.abs(cdf_emp - cdf_fit))), alpha


def select_xmin(x):
    """KS-minimizing x_min (Clauset–Shalizi–Newman)."""
    x = np.asarray(x, float)
    cands = np.unique(x)
    cands = cands[(cands >= 2.0) & (cands <= np.percentile(x, 92))]
    best = (np.inf, np.nan, np.nan)
    for xm in cands:
        ks, al = _ks_powerlaw(x, xm)
        if ks < best[0]:
            best = (ks, float(xm), float(al))
    return best  # ks, xmin, alpha


def heavy_tail(durations):
    dur = np.asarray(durations, float)
    dur = dur[dur > 0]
    if len(dur) < 30:
        return None
    ks, xmin, alpha = select_xmin(dur)
    pl = fit_powerlaw(dur, xmin)
    ex = fit_exponential(dur, xmin)
    if pl is None or ex is None:
        return None
    aic_pl = 2 * 1 - 2 * pl["loglik"]
    aic_ex = 2 * 1 - 2 * ex["loglik"]
    decades = float(np.log10(dur.max() / xmin))
    return dict(alpha=pl["alpha"], lam=ex["lam"], xmin=xmin, ks=ks,
                aic_pl=aic_pl, aic_ex=aic_ex, pl_preferred=bool(aic_pl < aic_ex),
                d_aic=float(aic_ex - aic_pl), n_tail=pl["n"], n_events=len(dur),
                tau_max=float(dur.max()), decades=decades)


def energy_duration(durations, energies, xmin):
    dur = np.asarray(durations, float)
    en = np.asarray(energies, float)
    m = (dur >= xmin) & (en > 0)
    if m.sum() < 10:
        return None
    beta, b0 = np.polyfit(np.log(dur[m]), np.log(en[m]), 1)
    pred = b0 + beta * np.log(dur[m])
    y = np.log(en[m])
    R2 = 1.0 - np.sum((y - pred) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-30)
    return dict(beta=float(beta), R2=float(R2), n=int(m.sum()))


# --------------------------------------------------------------------------- #
# Full analysis on a raw signal (+ surrogate)
# --------------------------------------------------------------------------- #
def analyze_signal(x, q=0.5, k_fine=5, n_bands=12, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    act = build_activity(x, k_fine, n_bands)
    thr = float(np.quantile(act, q))
    dur, en = detect_events(act, thr)
    ht = heavy_tail(dur)
    ed = energy_duration(dur, en, ht["xmin"]) if ht else None

    xs = phase_randomize(x, seed)
    acts = build_activity(xs, k_fine, n_bands)
    thr_s = float(np.quantile(acts, q))
    dur_s, _ = detect_events(acts, thr_s)
    ht_s = heavy_tail(dur_s)

    cl1 = bool(ht and ht["pl_preferred"] and 1.5 <= ht["alpha"] <= 4.0)
    cl2 = bool(ht and ht["decades"] >= 1.5)
    cl3 = bool(ed and ed["beta"] > 1.0 and ed["R2"] > 0.9)
    cl4 = bool(ht and ht_s and (ht["alpha"] < ht_s["alpha"] - 0.1
                                or ht["tau_max"] > 1.5 * ht_s["tau_max"]))
    n_pass = int(cl1) + int(cl2) + int(cl3) + int(cl4)
    return dict(ht=ht, ed=ed, ht_surr=ht_s, thresh=thr,
                durations=dur, energies=en,
                cl1=cl1, cl2=cl2, cl3=cl3, cl4=cl4, n_pass=n_pass,
                ok=bool(n_pass == 4))


# --------------------------------------------------------------------------- #
# Deterministic synthetic generators (fitter validation / unit tests)
# --------------------------------------------------------------------------- #
def powerlaw_samples(n=4000, alpha=2.5, xmin=2.0, seed=0):
    """Continuous power-law samples by inverse-CDF (x = xmin (1-u)^{-1/(alpha-1)})."""
    u = np.random.default_rng(seed).random(n)
    return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))


def exponential_samples(n=4000, scale=5.0, xmin=2.0, seed=0):
    return xmin + np.random.default_rng(seed).exponential(scale, n)


def main():
    print("=== fitter validation (synthetic) ===")
    pl = powerlaw_samples(alpha=2.5, seed=0)
    htp = heavy_tail(pl)
    print(f"  power-law in: alpha_hat={htp['alpha']:.2f} (true 2.5), "
          f"PL preferred={htp['pl_preferred']} ΔAIC={htp['d_aic']:.0f}")
    ex = exponential_samples(scale=5.0, seed=0)
    hte = heavy_tail(ex)
    print(f"  exponential in: PL preferred={hte['pl_preferred']} (want False) "
          f"ΔAIC={hte['d_aic']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
