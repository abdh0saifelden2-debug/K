r"""§V.7 synthetic unit test for the unified memory GLE (§D.4).

No external data.  §D.4 asks whether the subgrid (Mori-Zwanzig) memory and the
resolved ice-thermal memory (§B.2) can be folded into a *single* generalized
Langevin equation (GLE)

    dx/dt = - \int_0^t K(t-s) x(s) ds + F(t),     <F(t)F(t')> = k_B T_eff K(t-t')

with **scale-selective** kernels.  Two physically independent baths are eliminated
by the projection:

  * fast subgrid turbulence  -> Ornstein-Uhlenbeck (exponential) kernel
        K_SGS(tau) = (1/tau_c) e^{-tau/tau_c}          (correlation time tau_c)
    whose white-noise limit (tau_c -> 0) is the *instantaneous* eddy diffusivity
        K_SGS(tau) -> delta(tau) * \int K_SGS  = delta(tau)   (Markovian FDT);
  * slow ice-thermal diffusion (§B.2) -> heavy-tailed power-law kernel
        K_ice(tau) = B tau^{-1/2} e^{-tau/tau_d}        (cutoff time tau_d >> tau_c).

This module validates the *structure* that §D.4 claims (all pure math; the
amplitudes B, tau_c, tau_d and the bath weights are pinned separately in RESULT 12
/ ``gle_coefficients.py`` -- here they are free):

  1. **Additivity of independent baths [DERIVED].** The Mori-Zwanzig memory kernel
     is the autocorrelation of the projected (orthogonal) force,
     ``K(t) = <(QLf)(t) (QLf)(0)>``.  For two *independent* sub-baths the orthogonal
     forces are uncorrelated, so the cross-term vanishes and both the kernel and
     (by the second FDT) the noise covariance are **additive**:
     ``K = K_SGS + K_ice``.  Checked by Monte-Carlo: the autocorrelation of a sum of
     two independent colored noises equals the sum of the individual
     autocorrelations (cross-correlation ~ 0).

  2. **Scale selectivity / two relaxation timescales [DERIVED].** The combined
     kernel has a fast exponential head (scale tau_c) and a heavy power-law tail
     (scale tau_d): there is a crossover lag tau* with tau_c << tau* << tau_d, the
     head is dominated by K_SGS and the tail by K_ice.  Solving the GLE relaxation
     ``x(0)=1, F=0`` then shows the effective decay rate ``r(t) = -d ln x/dt`` drops
     by orders of magnitude from the fast (early) to the slow (late) regime -- the
     multi-scale prediction.

  3. **Markovian white-noise limit (FDT) [DERIVED].** ``\int_0^inf K_SGS dtau = 1``
     independent of tau_c, while the peak ``K_SGS(0) = 1/tau_c -> inf`` as
     ``tau_c -> 0``: the OU kernel approaches ``delta(tau)`` -- i.e. the resolved
     subgrid memory collapses to the *instantaneous* eddy-diffusivity closure that
     the model already uses, recovering the local FDT as a special case.
"""
from __future__ import annotations

import numpy as np


# --- kernels ------------------------------------------------------------------
def k_sgs(tau, tau_c):
    """Ornstein-Uhlenbeck (fast) memory kernel, unit integral."""
    return np.exp(-tau / tau_c) / tau_c


def k_ice(tau, B, tau_d):
    """Heavy-tailed (slow) power-law kernel with long-lag cutoff (§B.2 tail)."""
    return B * tau ** (-0.5) * np.exp(-tau / tau_d)


# --- (1) additivity of independent baths via projected-force autocorrelation --
def _ou_path(n, dt, tau_c, sigma, rng):
    """Sample an OU colored-noise path (stationary), correlation time tau_c."""
    x = np.empty(n)
    a = np.exp(-dt / tau_c)
    s = sigma * np.sqrt(1.0 - a * a)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = a * x[i - 1] + s * rng.normal()
    return x


def _autocorr(x, max_lag):
    x = x - x.mean()
    n = len(x)
    out = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        out[k] = np.dot(x[: n - k], x[k:]) / (n - k)
    return out


def additivity_check(n=200_000, dt=0.01, tau_fast=0.2, tau_slow=8.0,
                     max_lag=400, seed=0):
    """Two independent OU baths (fast + slow).  Autocorr of the sum should equal
    the sum of autocorrs (cross-term ~ 0) -> kernel additivity."""
    rng = np.random.default_rng(seed)
    a = _ou_path(n, dt, tau_fast, 1.0, rng)
    b = _ou_path(n, dt, tau_slow, 1.0, rng)
    ca = _autocorr(a, max_lag)
    cb = _autocorr(b, max_lag)
    csum = _autocorr(a + b, max_lag)
    # additive part vs the measured sum-autocorrelation
    resid = np.abs(csum - (ca + cb))
    scale = ca[0] + cb[0]
    add_rel_err = float(resid.max() / scale)
    # cross-correlation of the two independent baths at lag 0, normalised
    cross0 = float(abs(np.dot(a - a.mean(), b - b.mean()) / n) /
                   np.sqrt(ca[0] * cb[0]))
    return add_rel_err, cross0


# --- (2) GLE relaxation with the combined kernel ------------------------------
def gle_relax(t_end, dt, kernel_weights, tau_c, B, tau_d):
    """Solve x'(t) = -∫_0^t K(t-s) x(s) ds, x(0)=1, with K = wS*K_SGS + wI*K_ice.

    Product-integration for the integrable tau^{-1/2} ice singularity: x is taken
    piecewise-constant over each step and the kernel is integrated analytically
    over the cell, so  ∫ tau^{-1/2} dtau = 2(sqrt(b)-sqrt(a)).
    """
    wS, wI = kernel_weights
    n = int(round(t_end / dt)) + 1
    t = np.arange(n) * dt
    x = np.empty(n)
    x[0] = 1.0
    # per-lag convolution weights W[j] = ∫_{(j-1)dt}^{j dt} K dtau, so the cells
    # tile [0, n dt] exactly (no half-cell offset at the origin).
    j = np.arange(1, n + 1)
    edges_lo = (j - 1) * dt
    edges_hi = j * dt
    # SGS: exact integral of (1/tau_c)e^{-tau/tau_c}
    Wsgs = np.exp(-edges_lo / tau_c) - np.exp(-edges_hi / tau_c)
    # ice: 2B(sqrt(hi)-sqrt(lo)) * e^{-tau_mid/tau_d}  (product integration,
    # exponential factor at the cell midpoint)
    mid = (j - 0.5) * dt
    Wice = 2.0 * B * (np.sqrt(edges_hi) - np.sqrt(edges_lo)) * np.exp(-mid / tau_d)
    W = wS * Wsgs + wI * Wice
    for nidx in range(1, n):
        # convolution sum_{m=0}^{nidx-1} W[nidx-1-m] x[m]  (lags 1..nidx)
        conv = np.dot(W[:nidx][::-1], x[:nidx])
        x[nidx] = x[nidx - 1] - dt * conv
    return t, x


# --- (3) Markovian limit of the SGS kernel ------------------------------------
def sgs_integral(tau_c, tau_max=200.0, n=200_000):
    tau = np.linspace(tau_c * 1e-3, tau_max * tau_c, n)
    return float(np.trapezoid(k_sgs(tau, tau_c), tau))


def run():
    # (1) additivity
    add_rel_err, cross0 = additivity_check()
    additive = bool(add_rel_err < 0.05 and cross0 < 0.02)

    # (2) scale-selective response: solve the relaxation with the fast kernel
    #     alone vs the combined kernel.  The fast (SGS) kernel sets the EARLY
    #     relaxation (the two responses agree at t_early); the slow (ice) kernel
    #     adds a LONG-TIME tail that the fast-only response lacks (the responses
    #     differ by many orders at t_late).  => two relaxation timescales, each
    #     carried by its own kernel.
    # Illustrative coefficients chosen only to make the two timescales visible in
    # one plot window; the *physical* values are pinned in RESULT 12
    # (gle_coefficients.py / REPORT_GLE_COEFFICIENTS.md): tau_c is the measured SGS
    # K_u memory time (~0.02-0.03 solver units, sign +) and B, tau_d follow the
    # §B.2 closed form (tau_d = kappa/Vbar^2 ~ 0.3-34 yr).  This test checks only
    # the structural scale-selectivity, which holds for any tau_c << tau_d.
    tau_c, B, tau_d = 0.2, 0.15, 60.0
    t, x_comb = gle_relax(t_end=120.0, dt=0.005, kernel_weights=(1.0, 1.0),
                          tau_c=tau_c, B=B, tau_d=tau_d)
    _, x_fast = gle_relax(t_end=120.0, dt=0.005, kernel_weights=(1.0, 0.0),
                          tau_c=tau_c, B=B, tau_d=tau_d)
    i_early, i_late = int(round(0.2 / 0.005)), int(round(30.0 / 0.005))
    early_gap = float(abs(x_comb[i_early] - x_fast[i_early]))
    late_ratio = float(abs(x_comb[i_late]) / max(abs(x_fast[i_late]), 1e-300))
    early_tracks_fast = bool(early_gap < 0.05)        # fast head shared
    late_tail_from_slow = bool(late_ratio > 1e3)      # slow tail only with ice
    two_timescales = bool(early_tracks_fast and late_tail_from_slow)

    # crossover lag tau*: K_ice == K_SGS, must sit between the two scales
    taus = np.logspace(-3, np.log10(3 * tau_d), 4000)
    diff = k_ice(taus, B, tau_d) - k_sgs(taus, tau_c)
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    tau_star = float(taus[sign_change[-1]]) if len(sign_change) else np.nan
    scale_selective = bool(tau_c < tau_star < tau_d)

    # (3) Markovian FDT limit: unit integral independent of tau_c; peak -> inf
    I1, I2 = sgs_integral(0.5), sgs_integral(0.01)
    integral_invariant = bool(abs(I1 - 1.0) < 1e-3 and abs(I2 - 1.0) < 1e-3)
    peak_grows = bool(k_sgs(0.0, 0.01) > k_sgs(0.0, 0.5) * 10)
    markov_limit = bool(integral_invariant and peak_grows)

    ok = bool(additive and two_timescales and scale_selective and markov_limit)
    return {
        "additivity_rel_err": add_rel_err,
        "independent_cross_corr": cross0,
        "additive": additive,
        "early_gap_fast_vs_combined": early_gap,
        "late_tail_ratio": late_ratio,
        "early_tracks_fast": early_tracks_fast,
        "late_tail_from_slow": late_tail_from_slow,
        "two_timescales": two_timescales,
        "tau_star": tau_star,
        "scale_selective": scale_selective,
        "sgs_integral_invariant": integral_invariant,
        "sgs_peak_grows": peak_grows,
        "markov_limit": markov_limit,
        "pass": ok,
    }


def main():
    r = run()
    print("=== §V.7 unified-memory GLE test (§D.4) ===")
    print(f"  additivity (sum kernel == sum of kernels): {r['additive']}"
          f"  (rel err {r['additivity_rel_err']:.3f}, cross-corr {r['independent_cross_corr']:.4f})")
    print(f"  two relaxation timescales : {r['two_timescales']}"
          f"  (early gap {r['early_gap_fast_vs_combined']:.3f} -> fast head shared;"
          f" late tail ratio {r['late_tail_ratio']:.1e} -> slow tail only with ice)")
    print(f"  scale-selective crossover : {r['scale_selective']}"
          f"  (tau_c < tau*={r['tau_star']:.2f} < tau_d)")
    print(f"  Markovian white-noise limit: {r['markov_limit']}"
          f"  (unit integral inv.; peak ~1/tau_c -> inf)")
    print(f"  PASS                       : {r['pass']}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
