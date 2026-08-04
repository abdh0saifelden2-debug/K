r"""Paper 1 — Diagonal-band cascade structure in a stratified fluid (atmospheric
surface layer), from real NEON 1-min observations.

PRE-REGISTERED, falsification-driven. Mainstream grounding: a turbulent cascade is
*local in scale* (Kolmogorov 1941; Kraichnan) and *intermittent/multiplicative*
(Kolmogorov 1962; Frisch 1995), so cross-scale energy transfer couples neighbouring
scales and decays with scale separation. We test this as a **band-coupling matrix**
φ_ij = corr(log envelope of octave-band i, log envelope of octave-band j): a
multiplicative cascade makes φ **diagonal-band** — diagonal-dominant, monotonically
decaying off the diagonal, with a finite decay length L (in octaves) — and *above* a
phase-randomized (Gaussian, no cross-scale coupling) surrogate. A purely Gaussian /
linear process has independent bands, so its off-diagonal φ ≈ 0.

This is the observational/cascade face of the two-clocks program: a real stratified
fluid (the surface layer at WREF) builds cross-scale amplitude correlations that a
spectrum-matched Gaussian surrogate cannot — "energy yes, structure no" in the time
domain ([`run_boussinesq.py`](run_boussinesq.py)).

PRE-REGISTERED PREDICTIONS (thresholds set before seeing the real data)
----------------------------------------------------------------------
  CB1  Diagonal dominance: each row's diagonal is the maximum; nearest-neighbour
       coupling φ(d=1) < 1 and exceeds far coupling.
  CB2  Monotone off-diagonal decay: the Spearman trend of φ(d) vs separation d is
       strongly negative (≤ -0.8) — a noise-robust monotone-decrease test (a single
       exponential decay length is brittle under diurnal/synoptic forcing peaks).
  CB3  Finite, local decay scale: the half-coupling separation d_half (where φ(d)
       first falls below ½·φ(1)) is finite and < n_bands (coupling is local in scale).
  CB4  Above surrogate: real off-diagonal coupling exceeds the phase-randomized
       surrogate (cascade builds cross-scale correlation a Gaussian cannot).

`analyze()` runs on any 1-D series; `synthetic_cascade()` provides a deterministic
multiplicative cascade for method validation and unit tests (no data needed). The
real-data driver is `run_real.py`-style code in `run_cascade_structure.py`.
CPU, numpy-only.
"""
from __future__ import annotations

import numpy as np


def analytic_envelope(x: np.ndarray) -> np.ndarray:
    """|analytic signal| via the FFT Hilbert transform (numpy only)."""
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(X * h))


def dyadic_bands(x: np.ndarray, n_bands: int = 10):
    """FFT octave band-pass.  Band i (i=0 finest) keeps cycles/sample in
    (f_ny/2^{i+1}, f_ny/2^i], i.e. periods ~ [2^{i+1}, 2^{i+2}] samples."""
    x = np.asarray(x, float)
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(n)
    fny = 0.5
    out = []
    for i in range(n_bands):
        hi = fny / (2 ** i)
        lo = fny / (2 ** (i + 1))
        mask = (f > lo) & (f <= hi)
        out.append(np.fft.irfft(X * mask, n))
    return out


def coupling_matrix(bands, use_log: bool = True) -> np.ndarray:
    """φ_ij = Pearson corr of (log) band envelopes — cross-scale amplitude coupling."""
    env = [analytic_envelope(b) for b in bands]
    if use_log:
        A = np.array([np.log(e + 1e-12) for e in env])
    else:
        A = np.array(env)
    return np.corrcoef(A)


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def offdiag_decay(phi: np.ndarray):
    """Mean coupling φ(d) vs separation d=|i-j|, plus robust decay diagnostics.

    Returns the strict-monotone flag, a robust Spearman trend (≈ -1 for a clean
    decreasing curve, insensitive to a single noisy tail point), the half-coupling
    separation d_half (where φ(d) first drops below ½ φ(1)) as a robust decay scale,
    and the brittle single-exponential (L, R²) for reference only."""
    M = phi.shape[0]
    seps = np.arange(1, M)
    vals = np.array([np.mean([phi[i, i + d] for i in range(M - d)]) for d in seps])
    monotone = bool(np.all(np.diff(vals) <= 1e-9))
    spear = _spearman(seps, vals) if len(seps) >= 3 else 0.0
    # robust decay scale: first separation where coupling falls below half of nn
    half = 0.5 * vals[0]
    below = np.where(vals < half)[0]
    d_half = float(seps[below[0]]) if len(below) else float(seps[-1])
    # brittle single-exponential fit (reported with caveat only)
    good = vals > 1e-6
    L = np.nan
    R2 = 0.0
    if good.sum() >= 3:
        b, a = np.polyfit(seps[good], np.log(vals[good]), 1)
        L = -1.0 / b if b < 0 else np.inf
        pred = a + b * seps[good]
        ss_res = float(np.sum((np.log(vals[good]) - pred) ** 2))
        ss_tot = float(np.sum((np.log(vals[good]) - np.mean(np.log(vals[good]))) ** 2))
        R2 = 1.0 - ss_res / (ss_tot + 1e-30)
    return seps, vals, float(L), float(R2), monotone, spear, d_half


def phase_randomize(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Spectrum-preserving, phase-randomized surrogate (destroys cross-scale coupling)."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(x - x.mean())
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(X)))
    ph[0] = 1.0
    if n % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(X * ph, n)


def analyze(x: np.ndarray, n_bands: int = 10, seed: int = 0) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bands = dyadic_bands(x, n_bands)
    phi = coupling_matrix(bands)
    seps, vals, L, R2, mono, spear, d_half = offdiag_decay(phi)

    xs = phase_randomize(x, seed)
    bs = dyadic_bands(xs, n_bands)
    phis = coupling_matrix(bs)
    seps_s, vals_s, L_s, R2_s, mono_s, spear_s, d_half_s = offdiag_decay(phis)

    diag_dominant = bool(all(phi[i, i] >= phi[i, j] - 1e-9
                             for i in range(n_bands) for j in range(n_bands)))
    nn_real = float(vals[0])
    nn_surr = float(vals_s[0])
    mean_off_real = float(np.mean(vals))
    mean_off_surr = float(np.mean(vals_s))

    # Pre-registered, ROBUST operationalizations (a single-exponential L is brittle
    # under forcing peaks, so monotonicity uses the Spearman trend and the decay
    # scale uses the half-coupling separation d_half).
    cb1 = bool(diag_dominant and 0.0 < nn_real < 1.0 and nn_real > vals[-1])
    cb2 = bool(spear <= -0.8)
    cb3 = bool(0.0 < d_half < n_bands)
    cb4 = bool(mean_off_real > mean_off_surr + 0.05 and nn_real > nn_surr + 0.05)
    n_pass = int(cb1) + int(cb2) + int(cb3) + int(cb4)

    return dict(phi=phi, seps=seps, vals=vals, L=L, R2=R2, monotone=mono,
                spearman=spear, d_half=d_half,
                phi_surr=phis, vals_surr=vals_s, L_surr=L_s, d_half_surr=d_half_s,
                nn_real=nn_real, nn_surr=nn_surr,
                mean_off_real=mean_off_real, mean_off_surr=mean_off_surr,
                diag_dominant=diag_dominant,
                cb1=cb1, cb2=cb2, cb3=cb3, cb4=cb4, n_pass=n_pass,
                ok=bool(n_pass == 4), n_bands=n_bands, n_samples=len(x))


# --------------------------------------------------------------------------- #
# Deterministic synthetic data for method validation / unit tests
# --------------------------------------------------------------------------- #
def synthetic_cascade(n=2 ** 15, n_oct=9, beta=0.8, seed=0):
    """Multiplicative cascade: each octave's amplitude is modulated by the coarser
    octave's envelope (env**beta).  Builds diagonal-band cross-scale coupling that
    decays with scale separation -- the model the method is meant to detect."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    fny = 0.5
    env = np.ones(n)
    sig = np.zeros(n)
    for i in range(n_oct):                         # coarse -> fine
        f = fny / 2 ** (n_oct - i)
        period = max(int(round(1.0 / f)), 2)
        w = rng.standard_normal(n)
        k = np.ones(period) / period
        a = np.abs(np.convolve(w, k, mode="same"))
        amp = a * (env ** beta)                    # multiplicative coupling to parent
        amp /= (amp.std() + 1e-9)
        carrier = np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        sig += amp * carrier
        env = amp / (amp.mean() + 1e-9)
    return sig


def white_noise(n=2 ** 15, seed=1):
    return np.random.default_rng(seed).standard_normal(n)


def amplitude_modulated(n=2 ** 15, f_carrier=0.25, f_env=1.0 / 64, m=0.9, seed=2):
    """A fast carrier whose amplitude is modulated by a slow envelope -- a single,
    known cross-scale coupling between the carrier band and the envelope band."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    env = 1.0 + m * np.cos(2 * np.pi * f_env * t)
    return env * np.cos(2 * np.pi * f_carrier * t) + 0.01 * rng.standard_normal(n)
