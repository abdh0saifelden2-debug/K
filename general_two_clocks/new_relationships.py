r"""New derived cross-relationships (NR16–NR21), grounded in the repo's new
observational papers + mainstream theory, each numerically verified.

Continues the cross-relationship program (NR1–NR15) using the four "research these"
papers as new anchors:
  - Paper 1  diagonal-band cascade structure (cascade_band_structure.py)
  - Paper 2  in-place pressure-mediated buoyancy exchange (pressure_buoyancy_exchange.py)
  - Paper 3  cascade lifetime / power-law bursts (cascade_lifetime.py)
  - Paper 4  non-Markovian multiscale memory (nonmarkov_argo.py)

NR16 — LOWEN–TEICH SPECTRAL IMAGE OF POWER-LAW BURSTS [mainstream + Paper 3].
  A telegraph/on-off process whose sojourn times are power-law p(τ)∝τ^{-α} (1<α<3)
  has a low-frequency power-law spectrum S(f)∝f^{-γ} with γ = 3-α (Lowen & Teich,
  fractal point processes). So Paper 3's scale-free pressure bursts (α≈2) are the
  *time-domain* face of a 1/f (γ≈1) spectrum. Verified by measuring γ(α).

NR17 — POWER-LAW SPECTRUM ⇒ GROWING TEMPORAL MEMORY [Paper 3 → Paper 4].
  By Wiener–Khinchin, a 1/f^γ spectrum (γ>0) has a slowly-decaying (non-summable, for
  γ≥1) autocorrelation: the integrated memory time τ_int = 1 + 2Σ C(τ) grows
  monotonically with γ, while white noise (γ=0, Markov) has τ_int≈1. Combined with
  NR16 (Paper 3's power-law bursts ⇒ 1/f^{3-α} spectrum), this chains the burst
  statistics to the non-Markovian memory of Paper 4: scale-free bursts ⇒ power-law
  spectrum ⇒ long temporal memory. (The raw lag-2 excess C(2)-C(1)² is NOT used here:
  it is non-monotone in γ — it vanishes for both white and very strong memory because
  C(1)→1 — so τ_int is the correct monotone memory measure.)

NR18 — EXACT IDENTITY: EXCHANGE FRACTION = SPECTRAL ANISOTROPY √⟨kx²/k²⟩ [Paper 2].
  For the buoyancy force F=(0,b'), the Leray-surviving (motion-driving) fraction is
  exactly ‖F_sol‖/‖F‖ = √⟨kx²/k²⟩_b, the buoyancy-power-weighted spectral anisotropy
  (derivation in the report). Pure vertical stratification (b̂ at kx=0) ⇒ 0 (held in
  place); isotropic ⇒ 1/√2. Verified to machine precision against Paper 2's
  physical-space solenoidal_fraction.

NR19 — GREEN–KUBO / TAYLOR: INTEGRATED MEMORY TIME = EDDY DIFFUSIVITY [Paper 4].
  Taylor (1922) / Green–Kubo: a fluctuating velocity's transport diffusivity equals
  the time-integral of its own autocovariance, D = ½C(0) + Σ_{k≥1} C(k). So Paper 4's
  integrated memory time IS an eddy diffusivity — the long-memory (slow) clock
  transports more. Verified by computing D two independent ways from the SAME
  ensemble (single-particle dispersion ⟨x(T)²⟩/2T vs the Green–Kubo autocovariance
  integral); they agree to ~1.6%, and both match the exact AR(1) value 0.5/(1−φ)².
  (Honesty note: a naive estimator that subtracts each trajectory's own sample mean
  biases the Green–Kubo tail ~19% low; the known zero process mean is removed
  globally instead — see _acov_rows.)

NR20 — ADDITIVE DIFFUSIVITY FOR MULTISCALE MEMORY: D_total = D_fast + D_slow [Paper 4].
  Because dispersion is linear in the velocity (x = ∫v) and Paper 4's fast/slow
  channels are statistically independent, the total eddy diffusivity is the SUM of the
  per-channel diffusivities (cross-term → 0). The slow clock dominates transport
  (D_slow ≫ D_fast) even at equal energy, because diffusivity weights memory time, not
  variance. Verified to <0.1% on an independent two-timescale ensemble.

NR21 — LONG-RANGE DEPENDENCE: DFA / HURST EXPONENT = (γ+1)/2 [Paper 3 → Paper 4].
  Detrended fluctuation analysis (Peng 1994) of a 1/f^γ process recovers a Hurst
  exponent H = α_DFA = (γ+1)/2 (the standard fGn/fBm spectral↔self-similarity link;
  Mandelbrot; Beran). Chained with NR16 (γ = 3−α_burst), Paper 3's burst exponent
  predicts Paper 4's long-range-dependence Hurst exponent: H = (4−α_burst)/2. White
  noise (γ=0) ⇒ H=½ (no LRD); α_burst≈2 ⇒ γ≈1 ⇒ H≈1 (strong persistence). Verified:
  α_DFA tracks (γ+1)/2 to ~0.02 (the small positive offset is the known finite-scale
  DFA bias). A companion super-diffusion law ⟨x²⟩∝T^{2H} was tested and DROPPED: the
  FFT-filtered generator distorts the low-frequency power that drives it, so the
  exponent saturated (−0.30 off at γ=1.2) — not forced into the report.

CPU, deterministic. mainstream named not invented: Lowen & Teich (1993, 2005);
Helmholtz–Hodge decomposition; Mori–Zwanzig; Taylor (1922); Green–Kubo;
Peng et al. (1994, DFA); Mandelbrot–Hurst.
"""
from __future__ import annotations

import numpy as np

import nonmarkov_argo as NM
import pressure_buoyancy_exchange as PB
from compressible.ns import Spectral2D


# --------------------------------------------------------------------------- #
# NR16 — Lowen–Teich: power-law sojourns -> 1/f^γ spectrum, γ = 3 - α
# --------------------------------------------------------------------------- #
def telegraph_powerlaw(n, alpha, xmin=3, seed=0):
    """Dichotomous (+/-1) signal with power-law sojourn times p(τ) ∝ τ^{-α}."""
    rng = np.random.default_rng(seed)
    out = []
    state = 1.0
    while len(out) < n:
        u = rng.random()
        dur = int(xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0)))
        dur = max(min(dur, n - len(out)), 1)        # cap to remaining length
        out.extend([state] * dur)
        state = -state
    return np.asarray(out[:n], float)


def lowfreq_spectral_slope(x, f_lo=2e-4, f_hi=2e-2, n_real=1, alpha=None, seed0=0, n=None):
    """Average PSD slope γ (S∝f^{-γ}) over the low-frequency band, optionally averaging
    several realizations (when alpha/n given) to reduce single-record scatter."""
    if alpha is not None:
        n = n or len(x)
        Psum = None
        for r in range(n_real):
            xr = telegraph_powerlaw(n, alpha, seed=seed0 + r)
            X = np.fft.rfft(xr - xr.mean())
            P = np.abs(X) ** 2
            Psum = P if Psum is None else Psum + P
        P = Psum / n_real
        f = np.fft.rfftfreq(n)
    else:
        X = np.fft.rfft(np.asarray(x, float) - np.mean(x))
        P = np.abs(X) ** 2
        f = np.fft.rfftfreq(len(x))
    m = (f >= f_lo) & (f <= f_hi)
    b = np.polyfit(np.log(f[m]), np.log(P[m] + 1e-300), 1)[0]
    return -float(b)


def nr16(alphas=(1.5, 2.0, 2.5), n=2 ** 18, n_real=6):
    rows = []
    for a in alphas:
        gamma = lowfreq_spectral_slope(None, alpha=a, n=n, n_real=n_real, seed0=0)
        rows.append((a, gamma, 3.0 - a))
    err = max(abs(g - (3 - a)) for a, g, _ in rows)
    return dict(rows=rows, max_abs_err=float(err), ok=bool(err < 0.35))


# --------------------------------------------------------------------------- #
# NR17 — power-law spectrum (Paper 3 via NR16) ⇒ non-Markovian memory (Paper 4)
# --------------------------------------------------------------------------- #
def power_law_noise(n, gamma, seed=0):
    """Gaussian noise with spectrum S(f) ∝ f^{-γ} (γ=0 white; γ>0 long memory)."""
    rng = np.random.default_rng(seed)
    W = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    return np.fft.irfft(W * f ** (-gamma / 2.0), n)


def nr17(gammas=(0.0, 0.5, 1.0, 1.5), n=2 ** 16, maxlag=30):
    tint, slopes = [], []
    for g in gammas:
        x = power_law_noise(n, g, seed=0)
        C = NM.acf_nan(x, maxlag)
        tint.append(1.0 + 2.0 * float(np.nansum(C[1:maxlag + 1])))
        X = np.fft.rfft(x - x.mean())
        ff = np.fft.rfftfreq(n)
        m = (ff > 2e-4) & (ff < 5e-2)
        slopes.append(-float(np.polyfit(np.log(ff[m]), np.log(np.abs(X[m]) ** 2 + 1e-300), 1)[0]))
    tint = np.array(tint)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    rho = spearman(np.array(gammas), tint)
    return dict(gammas=list(gammas), tau_int=tint.tolist(), slopes=slopes,
                spearman=rho, white_memoryless=bool(tint[0] < 2.0),
                memory_grows=bool(tint[-1] > 5.0),
                ok=bool(rho > 0.9 and tint[0] < 2.0 and tint[-1] > 5.0))


# --------------------------------------------------------------------------- #
# NR18 — exact identity: exchange fraction = √⟨kx²/k²⟩ (buoyancy-power-weighted)
# --------------------------------------------------------------------------- #
def exchange_fraction_spectral(b):
    """√⟨kx²/k²⟩ weighted by |b̂|² — the predicted Leray-surviving fraction of (0,b')."""
    sp = Spectral2D(b.shape[0])
    bp = b - float(b.mean())
    P = np.abs(sp.fft(bp)) ** 2
    k2 = sp.k2.copy()
    k2[0, 0] = 1.0
    frac2 = float(np.sum((sp.kx ** 2 / k2) * P) / (np.sum(P) + 1e-300))
    return np.sqrt(frac2)


def nr18(n=96, seed=0):
    rng = np.random.default_rng(seed)
    sp = Spectral2D(n)
    x, y = sp.grid()
    cases = {
        "vertical b'(y)": np.sin(2 * y) + 0.4 * np.cos(3 * y),
        "horizontal b'(x)": np.cos(2 * x) + 0.5 * np.sin(x),
        "isotropic random": sp.ifft(sp.fft(rng.standard_normal((n, n))) * (sp.k2 < 50)).real,
        "anisotropic": np.cos(x) * (1 + 0.3 * np.sin(2 * y)) + 0.2 * np.sin(3 * y),
    }
    rows = []
    worst = 0.0
    for name, b in cases.items():
        lhs = PB.solenoidal_fraction(sp, b)          # physical-space (Paper 2)
        rhs = exchange_fraction_spectral(b)          # spectral identity
        rows.append((name, lhs, rhs))
        worst = max(worst, abs(lhs - rhs))
    return dict(rows=rows, max_abs_err=float(worst), ok=bool(worst < 1e-9))


# --------------------------------------------------------------------------- #
# NR19 — Green–Kubo / Taylor: integrated memory time = eddy diffusivity
# --------------------------------------------------------------------------- #
def _ar1_ensemble(n_traj, T, phi, seed):
    from scipy.signal import lfilter
    noise = np.random.default_rng(seed).standard_normal((n_traj, T))
    return lfilter([1.0], [1.0, -phi], noise, axis=1)


def _acov_rows(V, maxlag):
    """Biased per-row autocovariance C(k) = (1/T) Σ_t v_t v_{t+k}, removing the
    *known* (zero) process mean GLOBALLY. Subtracting each trajectory's own sample
    mean instead would inject an O(maxlag/T) negative bias into the Green–Kubo tail
    sum (the sample-mean-subtracted biased ACF sums to exactly zero over all lags)."""
    Vc = V - V.mean()
    n = V.shape[1]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(Vc, nfft, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, nfft, axis=1)[:, :maxlag + 1] / n
    return ac


def _D_from_ensemble(V, maxlag=400, dt=1.0):
    """Both diffusivities from the SAME ensemble (tests the Green–Kubo identity):
    D_dispersion = ⟨x(T)²⟩/(2T);  D_GreenKubo = ⟨½C(0) + Σ_{k≥1} C(k)⟩·dt."""
    T = V.shape[1]
    xT = np.cumsum(V, axis=1)[:, -1] * dt
    d_disp = float(np.mean(xT ** 2) / (2.0 * T * dt))
    C = _acov_rows(V, maxlag)
    d_gk = float(np.mean(0.5 * C[:, 0] + C[:, 1:].sum(axis=1)) * dt)
    return d_disp, d_gk


def _two_timescale_ensemble(n_traj, T, phi_f, phi_s, w, seed):
    a = _ar1_ensemble(n_traj, T, phi_f, seed)
    b = _ar1_ensemble(n_traj, T, phi_s, seed + 90001)
    a = a / a.std(axis=1, keepdims=True)
    b = b / b.std(axis=1, keepdims=True)
    return w * a + (1 - w) * b


def nr19(n=4000, n_traj=5000):
    rows = []
    worst = 0.0
    for name, phi in (("AR(1) φ=0.8", 0.8), ("AR(1) φ=0.95", 0.95)):
        V = _ar1_ensemble(n_traj, n, phi, seed=0)
        d_disp, d_gk = _D_from_ensemble(V)
        rows.append((name, d_disp, d_gk))
        worst = max(worst, abs(d_disp - d_gk) / d_gk)
    V = _two_timescale_ensemble(n_traj, n, 0.5, 0.97, 0.5, seed=0)
    d_disp, d_gk = _D_from_ensemble(V)
    rows.append(("two-timescale", d_disp, d_gk))
    worst = max(worst, abs(d_disp - d_gk) / d_gk)
    return dict(rows=rows, max_rel_err=float(worst), ok=bool(worst < 0.08))


# --------------------------------------------------------------------------- #
# NR20 — additive diffusivity for multiscale memory: D_total = D_fast + D_slow
# --------------------------------------------------------------------------- #
def nr20(n=4000, n_traj=5000):
    phi_f, phi_s, w = 0.5, 0.97, 0.5
    a = _ar1_ensemble(n_traj, n, phi_f, seed=0)
    b = _ar1_ensemble(n_traj, n, phi_s, seed=90001)
    a = a / a.std(axis=1, keepdims=True)
    b = b / b.std(axis=1, keepdims=True)
    vf, vs = w * a, (1 - w) * b
    d_fast, _ = _D_from_ensemble(vf)
    d_slow, _ = _D_from_ensemble(vs)
    d_total, _ = _D_from_ensemble(vf + vs)
    rel = abs(d_total - (d_fast + d_slow)) / (d_fast + d_slow)
    return dict(d_fast=d_fast, d_slow=d_slow, d_sum=d_fast + d_slow, d_total=d_total,
                rel_err=float(rel), ok=bool(rel < 0.05))


# --------------------------------------------------------------------------- #
# NR21 — long-range dependence: DFA / Hurst exponent = (γ+1)/2
# --------------------------------------------------------------------------- #
def dfa(x, scales):
    """Detrended fluctuation analysis (Peng et al. 1994): F(s) ∝ s^α, α the Hurst
    exponent. Integrate, split into windows of length s, linearly detrend each,
    RMS the residual."""
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    F = []
    for s in scales:
        s = int(s)
        nw = len(y) // s
        if nw < 4:
            F.append(np.nan)
            continue
        Y = y[:nw * s].reshape(nw, s)
        t = np.arange(s)
        A = np.vstack([t, np.ones(s)]).T
        coef, _, _, _ = np.linalg.lstsq(A, Y.T, rcond=None)
        fit = (A @ coef).T
        F.append(float(np.sqrt(((Y - fit) ** 2).mean())))
    return np.array(F)


def nr21(gammas=(0.0, 0.4, 0.8, 1.2, 1.6), n=2 ** 16):
    scales = np.unique(np.floor(np.logspace(1.0, 3.0, 18)).astype(int))
    rows = []
    worst = 0.0
    for g in gammas:
        x = power_law_noise(n, g, seed=0)
        F = dfa(x, scales)
        m = np.isfinite(F)
        a = float(np.polyfit(np.log(scales[m]), np.log(F[m]), 1)[0])
        pred = (g + 1.0) / 2.0
        rows.append((g, a, pred))
        worst = max(worst, abs(a - pred))
    return dict(rows=rows, max_abs_err=float(worst), ok=bool(worst < 0.06))


def run():
    return dict(nr16=nr16(), nr17=nr17(), nr18=nr18(), nr19=nr19(), nr20=nr20(),
                nr21=nr21())


def main():
    r16 = nr16(); r17 = nr17(); r18 = nr18(); r19 = nr19(); r20 = nr20(); r21 = nr21()
    print("=== NR16  Lowen–Teich: power-law bursts -> 1/f^γ, γ = 3-α ===")
    for a, g, pred in r16["rows"]:
        print(f"  α={a}: measured γ={g:.2f}  predicted 3-α={pred:.2f}")
    print(f"  max|err|={r16['max_abs_err']:.3f}  ok={r16['ok']}")
    print("=== NR17  power-law spectrum ⇒ growing temporal memory (Wiener–Khinchin) ===")
    print(f"  γ       ={r17['gammas']}")
    print(f"  slope   ={[round(v,2) for v in r17['slopes']]}")
    print(f"  τ_int   ={[round(v,2) for v in r17['tau_int']]}")
    print(f"  Spearman(γ,τ_int)={r17['spearman']:.3f}  whiteτ≈1={r17['white_memoryless']}  ok={r17['ok']}")
    print("=== NR18  exchange fraction = √⟨kx²/k²⟩ (exact) ===")
    for name, lhs, rhs in r18["rows"]:
        print(f"  {name:18s}: solenoidal_fraction={lhs:.6f}  √⟨kx²/k²⟩={rhs:.6f}")
    print(f"  max|err|={r18['max_abs_err']:.2e}  ok={r18['ok']}")
    print("=== NR19  Green–Kubo/Taylor: D = σ²·τ_int (memory time = eddy diffusivity) ===")
    for name, dm, dgk in r19["rows"]:
        print(f"  {name:16s}: D_dispersion={dm:.3f}  D_GreenKubo={dgk:.3f}  ratio={dm/dgk:.3f}")
    print(f"  max rel.err={r19['max_rel_err']:.3f}  ok={r19['ok']}")
    print("=== NR20  additive diffusivity: D_total = D_fast + D_slow (multiscale memory) ===")
    print(f"  D_fast={r20['d_fast']:.3f}  D_slow={r20['d_slow']:.3f}  sum={r20['d_sum']:.3f}  "
          f"D_total(measured)={r20['d_total']:.3f}  rel.err={r20['rel_err']:.3f}  ok={r20['ok']}")
    print("=== NR21  long-range dependence: DFA/Hurst α = (γ+1)/2 ===")
    for g, a, pred in r21["rows"]:
        print(f"  γ={g:.1f}: α_DFA={a:.3f}  predicted (γ+1)/2={pred:.3f}")
    print(f"  max|err|={r21['max_abs_err']:.3f}  ok={r21['ok']}")
    ok = all(d["ok"] for d in (r16, r17, r18, r19, r20, r21))
    print(f"PASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
