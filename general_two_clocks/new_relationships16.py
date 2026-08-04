r"""NR39 -- imaginary coherence is projection invariance: instantaneous linear
(elliptic / volume-conduction / Poisson) mixing CANNOT create imaginary cross-
spectra, so Nolte et al. (2004)'s imaginary-coherency artifact rejection and
P0's phase-surrogate ceiling are two corollaries of one lemma.

Ledger item E7 (papers/RESEARCH_RECAP_AND_HORIZON.md).

The lemma  [DERIVED]
--------------------
Let the observed channels be an *instantaneous, real* linear mix of latent
sources,  x(t) = A s(t),  with A a real (frequency-independent) matrix -- the
defining property of a volume-conduction / elliptic (Poisson) mixing operator:
it has zero phase at every frequency.  The cross-spectral matrix is then

    S_x(f) = A S_s(f) A^T .

If the source cross-spectrum S_s(f) is *real* -- which holds whenever the
sources are mutually uncorrelated (S_s diagonal) OR only zero-lag correlated
(S_s real-symmetric) -- then S_x(f) is real-symmetric, so

    Im S_x(f) = 0    =>    imaginary coherency  Im C_x(f) = 0 .

Only a genuinely *lagged* (dynamical) source interaction makes S_s(f) complex-
Hermitian, and that is the sole route to nonzero imaginary coherency.  Hence:

  * Corollary 1 (Nolte 2004): imaginary coherency rejects volume conduction --
    any nonzero Im C is genuine lagged coupling, immune to instantaneous mixing.
  * Corollary 2 (P0 surrogate ceiling): phase randomization destroys the lagged
    (imaginary) part while preserving the auto-spectra, so the phase-surrogate
    distribution IS the instantaneous-mixing null -- the ceiling a real coupling
    must clear.

Both are the single statement: an instantaneous real operator is a projection
that cannot manufacture cross-spectral phase.

What is proved here (in-repo, CPU, deterministic)
-------------------------------------------------
1. algebra: real A + real (diagonal) S_s  =>  max|Im S_x| = 0 to machine eps;
   a single lagged (complex-Hermitian) source entry breaks it.
2. synthetic array: pure instantaneous mixing of independent band-limited
   sources gives every pair's |Im coherency| at the estimator null; injecting
   ONE lagged path lights up only that pair, well above the analytic null.
3. surrogate ceiling: the injected pair's imaginary coherency sits far above its
   phase-randomized surrogate ceiling; a pure-mixing pair does not.

Real resting EEG (OpenNeuro) is a data-gated consistency demo (Re >> Im), not
required for the proofs.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import signal

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
FS = 128.0
BAND = (8.0, 13.0)          # alpha, for the band-averaged coherency


# --------------------------------------------------------------- the lemma
def max_imag_real_mixing(seed=0, n_ch=6, m=4):
    """Real A, real diagonal source spectrum -> S_x = A S_s A^T is real."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_ch, m))
    Ss = np.diag(rng.uniform(0.5, 2.0, m)).astype(complex)
    Sx = A @ Ss @ A.T
    return float(np.max(np.abs(Sx.imag)))


def max_imag_lagged_source(seed=0, n_ch=6, m=4):
    """One lagged (complex-Hermitian) source coupling breaks the lemma."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_ch, m))
    Ss = np.diag(rng.uniform(0.5, 2.0, m)).astype(complex)
    Ss[0, 1] = 0.4 + 0.3j
    Ss[1, 0] = np.conj(Ss[0, 1])
    Sx = A @ Ss @ A.T
    return float(np.max(np.abs(Sx.imag)))


# --------------------------------------------------- synthetic mixed array
def _bandlimited(n, rng, lo=6.0, hi=15.0, fs=FS):
    x = rng.standard_normal(n)
    b, a = signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, x)


def synth_mixed(seed=0, n=20000, n_ch=6, m=4, lag_pair=None, lag=3, gain=0.9,
                amp=1.8):
    """n_ch channels = instantaneous real mix of m band-limited sources.
    lag_pair=(i,j) adds a FRESH private source seen only by i (t) and j (t-lag),
    i.e. a genuine lagged path on exactly that pair (amp = its strength)."""
    rng = np.random.default_rng(seed)
    S = np.array([_bandlimited(n, rng) for _ in range(m)])
    A = rng.standard_normal((n_ch, m))
    X = A @ S
    if lag_pair is not None:
        i, j = lag_pair
        priv = _bandlimited(n, rng)
        X[i] = X[i] + amp * priv
        X[j] = X[j] + amp * gain * np.roll(priv, lag)
    return X


def coherency_matrix(X, fs=FS, nperseg=256, band=BAND):
    """Band-averaged complex coherency matrix + segment count."""
    n_ch = X.shape[0]
    f, Pxx = signal.welch(X, fs=fs, nperseg=nperseg, axis=-1)
    bm = (f >= band[0]) & (f <= band[1])
    C = np.zeros((n_ch, n_ch), complex)
    for i in range(n_ch):
        for j in range(n_ch):
            _, Sij = signal.csd(X[i], X[j], fs=fs, nperseg=nperseg)
            C[i, j] = np.mean(Sij[bm]) / np.sqrt(np.mean(Pxx[i][bm])
                                                 * np.mean(Pxx[j][bm]))
    n_seg = max(1, 2 * X.shape[-1] // nperseg - 1)     # ~50% overlap Welch
    return C, n_seg


def imag_null_1sigma(n_seg):
    """Under H0 the coherency real/imag parts are ~Gaussian, std ~ 1/sqrt(2 L)."""
    return 1.0 / np.sqrt(2.0 * n_seg)


def phase_randomize(x, rng):
    X = np.fft.rfft(x)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, X.shape))
    ph[0] = 1.0
    if x.size % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(np.abs(X) * ph, x.size)


# ----------------------------------------------------------------------- run
def run(seed=0):
    out = {}
    out["lemma"] = {
        "max_imag_real_mixing": max_imag_real_mixing(seed),
        "max_imag_lagged_source": max_imag_lagged_source(seed),
    }

    # pure instantaneous mixing: no pair should clear the null
    Xn = synth_mixed(seed=seed)
    Cn, nseg = coherency_matrix(Xn)
    iu = np.triu_indices(Cn.shape[0], 1)
    im_null = np.abs(Cn.imag[iu])
    re_null = np.abs(Cn.real[iu])

    # inject ONE lagged path on pair (2,4)
    pair = (2, 4)
    Xl = synth_mixed(seed=seed, lag_pair=pair)
    Cl, _ = coherency_matrix(Xl)
    others = [abs(Cl.imag[i, j]) for i, j in zip(*iu) if (i, j) != pair]
    null1 = imag_null_1sigma(nseg)
    out["synthetic_array"] = {
        "n_seg": nseg, "analytic_null_1sigma": null1,
        "max_abs_imcoh_pure_mixing": float(im_null.max()),
        "median_abs_recoh_pure_mixing": float(np.median(re_null)),
        "imcoh_injected_pair": float(abs(Cl.imag[pair])),
        "max_imcoh_other_pairs": float(np.max(others)),
        "injected_over_null": float(abs(Cl.imag[pair]) / null1),
        "mixing_dominated_re_over_im": float(np.median(re_null)
                                             / (np.median(im_null) + 1e-12)),
    }

    # surrogate ceiling: phase-randomize the injected pair's channels
    rng = np.random.default_rng(seed + 7)
    sur = []
    for _ in range(200):
        xi = phase_randomize(Xl[pair[0]], rng)
        xj = phase_randomize(Xl[pair[1]], rng)
        Cs, _ = coherency_matrix(np.array([xi, xj]))
        sur.append(abs(Cs.imag[0, 1]))
    sur = np.array(sur)
    out["surrogate_ceiling"] = {
        "injected_imcoh": float(abs(Cl.imag[pair])),
        "surrogate_p95": float(np.percentile(sur, 95)),
        "surrogate_median": float(np.median(sur)),
        "clears_ceiling": bool(abs(Cl.imag[pair]) > np.percentile(sur, 95)),
    }

    # real EEG (data-gated)
    edf = os.environ.get("NR39_EDF", "")
    out["real_eeg"] = {"available": bool(edf and os.path.exists(edf))}
    return out, {"Cn": Cn, "Cl": Cl, "pair": pair, "sur": sur, "nseg": nseg}


# --------------------------------------------------------------------- figure
def make_figure(out, aux, path_png):                        # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    sy = out["synthetic_array"]

    ax = axes[0]
    ax.bar(["pure mixing\n(max pair)", "injected lagged\npair (2,4)", "other pairs\n(max)"],
           [sy["max_abs_imcoh_pure_mixing"], sy["imcoh_injected_pair"],
            sy["max_imcoh_other_pairs"]],
           color=["tab:blue", "tab:red", "tab:blue"])
    ax.axhline(3 * sy["analytic_null_1sigma"], ls="--", color="k", lw=1,
               label="3 sigma estimator null")
    ax.set_ylabel("|Im coherency| (alpha)")
    ax.set_title("lemma: mixing -> 0; one lagged path -> detected")
    ax.legend(fontsize=8)

    ax = axes[1]
    Cn = aux["Cn"]
    iu = np.triu_indices(Cn.shape[0], 1)
    ax.scatter(np.abs(Cn.real[iu]), np.abs(Cn.imag[iu]), s=30, color="tab:blue",
               label="pure mixing pairs")
    Cl = aux["Cl"]; p = aux["pair"]
    ax.scatter([abs(Cl.real[p])], [abs(Cl.imag[p])], s=80, color="tab:red",
               marker="*", label="injected lagged pair")
    lim = max(0.05, float(np.abs(Cn.real[iu]).max()))
    ax.plot([0, lim], [0, lim], "k--", lw=0.7)
    ax.set_xlabel("|Re coherency|"); ax.set_ylabel("|Im coherency|")
    ax.set_title("mixing lives on the real axis"); ax.legend(fontsize=8)

    ax = axes[2]
    sur = aux["sur"]
    ax.hist(sur, bins=25, color="0.7", label="phase-surrogate ceiling")
    ax.axvline(out["surrogate_ceiling"]["injected_imcoh"], color="tab:red",
               lw=2, label="injected coupling")
    ax.axvline(np.percentile(sur, 95), color="k", ls="--", lw=1, label="p95")
    ax.set_xlabel("|Im coherency|"); ax.set_ylabel("count")
    ax.set_title("P0 ceiling: real coupling clears the surrogates")
    ax.legend(fontsize=8)

    fig.suptitle("NR39 (E7): imaginary coherence = instantaneous-mixing projection invariance",
                 y=1.03)
    fig.tight_layout()
    fig.savefig(path_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():                                                 # pragma: no cover
    out, aux = run()
    os.makedirs(FIGDIR, exist_ok=True)
    jpath = os.path.join(FIGDIR, "nr39_imaginary_coherence.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, aux, os.path.join(FIGDIR, "nr39_imaginary_coherence.png"))
    print(json.dumps(out, indent=1))
    print(f"wrote {jpath} and the png")


if __name__ == "__main__":                                  # pragma: no cover
    main()
