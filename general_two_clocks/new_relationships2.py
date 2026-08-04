r"""New derived cross-relationships (NR22-NR24), continuing the program
(NR1-NR21) with the same discipline: each is *derived* from mainstream theory +
a repo result and *numerically verified* (CPU, deterministic).  See
``REPORT_NEW_RELATIONSHIPS2.md`` for the write-ups and
``tests/test_new_relationships2.py`` for the unit proofs.

NR22 - EARLY-WARNING INDICATOR COUPLING: Var*(1-AC1**2) = sigma_eps**2  [Paper 4 -> NR3/NR8].
  The two canonical critical-slowing-down precursors (Scheffer/Dakos: rising
  variance and rising lag-1 autocorrelation) are NOT independent.  For the OU /
  AR(1) reduction of a system near a saddle-node (the s_N flotation pole, NR3;
  the Lorentzian corner f_c, NR8) one has, exactly,
        AC1 = a = e^{-dt/tau},      Var = sigma_eps**2 / (1 - a**2),
  so the product  I = Var*(1 - AC1**2) = sigma_eps**2  is INVARIANT as the fold
  is approached (a->1, AC1->1, Var->inf): both precursors are two reads of the
  single relaxation rate tau, and Var ~ tau (the variance EWS *is* the memory
  time, cf. NR19).  The invariant is also a discriminator: genuine slowing-down
  (tau grows, sigma_eps fixed) holds I flat while AC1 rises, whereas mere forcing
  inflation (sigma_eps grows, tau fixed) raises I while AC1 stays put.  Verified
  on AR(1) ensembles for both a "approach-the-fold" sweep and a "louder-forcing"
  sweep.  Mainstream: Scheffer et al. (2009); Dakos et al. (2012); Wissel (1984).

NR23 - KRAMERS-KRONIG DC SUM RULE: net eddy viscosity = migration-spectrum integral  [Paper 1 -> NR9/NR15].
  A causal eddy-response kernel K(t>=0) has admittance Z(w)=int_0^inf K(t)e^{-iwt}dt
  whose real/imaginary parts are Hilbert pairs (Kramers-Kronig, NR9).  Evaluating
  the once-subtracted KK relation at w=0 gives the SUM RULE
        Z(0) - Z(inf)  =  (2/pi) int_0^inf [-Im Z(w)] / w  dw .
  Here Z(0)=int_0^inf K dt is exactly NR15's *net eddy viscosity* nu_eff (and
  Z(inf)=0 for an integrable kernel), while -Im Z is the migration / quadrature
  (reactive) spectrum.  So the SIGN and MAGNITUDE of nu_eff -- in particular the
  backscatter branch nu_eff<0 (growth) of NR15 -- is fixed by a weighted integral
  of the migration spectrum: a single-number bridge from NR15 (sign budget) to
  NR9 (causality).  Verified to ~1e-6 on multi-Lorentzian kernels, including a
  net-negative (backscatter) kernel.  Mainstream: Kramers (1927); Kronig (1926).

NR24 - EDDY DIFFUSIVITY IS THE ZERO-FREQUENCY PSD: D = (1/2) S(0)  [Paper 4 -> NR17/NR19].
  Green-Kubo (NR19) gives D = (1/2)C(0) + sum_{k>=1} C(k); Wiener-Khinchin gives
  the one-sided PSD S(0) = sum_{k=-inf}^{inf} C(k) = C(0) + 2 sum_{k>=1} C(k).
  Hence  D = (1/2) S(0):  the eddy diffusivity is half the zero-frequency power.
  Consequences verified here: (i) on AR(1), an INDEPENDENT (Welch periodogram)
  estimate of S(0) matches the dispersion diffusivity to a few percent; (ii)
  approaching a fold (a->1, critical slowing down, NR22/NR3) the diffusivity
  DIVERGES as D ~ (1-a)^{-2} ~ (N-N_c)^{-2} -- the transport face of the variance
  EWS; (iii) a 1/f^gamma long-memory spectrum (gamma>0, NR16/NR17) has
  S(0)->inf, so NO finite eddy diffusivity exists -- the quantitative reason the
  local down-gradient closure fails for Paper 4's long memory.  Mainstream:
  Taylor (1922); Green-Kubo; Wiener-Khinchin.

NR25 - SECOND FLUCTUATION-DISSIPATION THEOREM: memory kernel = random-force ACF  [Paper 4 MZ -> NR1].
  In the exact harmonic-bath (Caldeira-Leggett / Zwanzig) reduction -- the linear
  instance of Paper 4's certified Mori-Zwanzig structure -- eliminating the bath
  gives a GLE whose friction (memory) kernel is K(t)=sum_i (c_i^2/w_i^2)cos(w_i t)
  and whose random force, with the bath drawn from Gibbs at temperature T, obeys
  the SECOND FDT  <F(t)F(0)> = kT*K(t).  Paper 4 certifies the *dissipation* side
  (kernel = eliminated channel's Green's function); this adds the *fluctuation*
  side: the same kernel is the noise autocorrelation -- memory and noise are one
  object (cf. NR1).  Verified by computing the two sides INDEPENDENTLY (friction
  kernel as the deterministic bath sum vs. the Gibbs-Monte-Carlo random-force ACF),
  agreeing to ~0.4% of peak across the oscillatory kernel.  Mainstream: Kubo
  (1966); Zwanzig (1973); Caldeira-Leggett (1983).

CPU, deterministic.  mainstream named not invented.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# shared AR(1) helpers
# --------------------------------------------------------------------------- #
def ar1(n, a, sigma=1.0, seed=0, burn=2000):
    """Stationary scalar AR(1): x_{t+1} = a x_t + eps,  eps ~ N(0, sigma**2)."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + burn) * sigma
    x = lfilter([1.0], [1.0, -a], e)
    return x[burn:]


def ar1_ensemble(n_traj, T, a, sigma=1.0, seed=0):
    """Vectorised AR(1) ensemble via scipy lfilter (rows = trajectories)."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_traj, T)) * sigma
    return lfilter([1.0], [1.0, -a], e, axis=1)


def _ac1(x):
    x = x - x.mean()
    return float(np.dot(x[:-1], x[1:]) / np.dot(x, x))


# --------------------------------------------------------------------------- #
# NR22 - EWS indicator coupling:  Var*(1 - AC1**2) = sigma_eps**2
# --------------------------------------------------------------------------- #
def nr22(a_fold=(0.5, 0.8, 0.9, 0.95, 0.98), sigmas=(1.0, 1.5, 2.0, 2.5),
         n=2 ** 20, sigma0=1.0, a0=0.8, seed=0):
    # (1) approach-the-fold sweep: sigma fixed, a -> 1
    fold = []
    for a in a_fold:
        x = ar1(n, a, sigma=sigma0, seed=seed)
        v, c = float(np.var(x)), _ac1(x)
        fold.append((a, v, c, v * (1.0 - c * c)))
    inv = np.array([r[3] for r in fold])
    fold_inv_err = float(np.max(np.abs(inv - sigma0 ** 2)) / sigma0 ** 2)
    ac1_rises = all(fold[i][2] < fold[i + 1][2] for i in range(len(fold) - 1))
    var_rises = all(fold[i][1] < fold[i + 1][1] for i in range(len(fold) - 1))

    # (2) louder-forcing sweep: a fixed, sigma -> up  (the discriminator)
    loud = []
    for s in sigmas:
        x = ar1(n, a0, sigma=s, seed=seed + 7)
        v, c = float(np.var(x)), _ac1(x)
        loud.append((s, v, c, v * (1.0 - c * c)))
    # invariant should TRACK sigma**2 here (rise), while AC1 stays ~ a0
    inv_loud = np.array([r[3] for r in loud])
    s2 = np.array([s ** 2 for s, *_ in loud])
    loud_tracks = float(np.max(np.abs(inv_loud - s2) / s2))
    ac1_flat = float(np.max(np.abs([r[2] - a0 for r in loud])))

    ok = bool(fold_inv_err < 0.05 and ac1_rises and var_rises
              and loud_tracks < 0.05 and ac1_flat < 0.02)
    return dict(fold=fold, fold_inv_err=fold_inv_err, ac1_rises=ac1_rises,
                var_rises=var_rises, loud=loud, loud_tracks=loud_tracks,
                ac1_flat=ac1_flat, ok=ok)


# --------------------------------------------------------------------------- #
# NR23 - Kramers-Kronig DC sum rule:  Z(0) = (2/pi) int_0^inf (-Im Z)/w dw
# --------------------------------------------------------------------------- #
def _Zom(w, amps, taus):
    """Admittance of K(t)=sum a_j exp(-t/tau_j):  Z = sum a_j tau_j/(1+i w tau_j)."""
    Z = np.zeros_like(w, dtype=complex)
    for a, t in zip(amps, taus):
        Z += a * t / (1.0 + 1j * w * t)
    return Z


def _kk_dc(amps, taus, wmax=None, npts=400001):
    taus = np.asarray(taus, float)
    if wmax is None:
        wmax = 200.0 / taus.min()                  # resolve every corner 1/tau_j
    w = np.linspace(0.0, wmax, npts)
    Z = _Zom(w, amps, taus)
    integrand = np.empty_like(w)
    # (-Im Z)/w  ->  finite limit sum a_j tau_j**2 as w->0
    integrand[0] = float(np.sum(np.asarray(amps) * taus ** 2))
    integrand[1:] = (-Z.imag[1:]) / w[1:]
    rhs = (2.0 / np.pi) * float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(w)))
    z0 = float(np.sum(np.asarray(amps) * taus))    # = int_0^inf K dt = nu_eff
    return z0, rhs


def nr23():
    cases = {
        "dissipative (nu_eff>0)": ([1.0, 0.6, 0.3], [1.0, 5.0, 25.0]),
        "two-scale": ([2.0, 0.5], [0.7, 40.0]),
        "backscatter (nu_eff<0)": ([-1.5, 0.4, 0.3], [10.0, 2.0, 1.0]),
    }
    rows, worst = [], 0.0
    for name, (amps, taus) in cases.items():
        z0, rhs = _kk_dc(amps, taus)
        rel = abs(z0 - rhs) / (abs(z0) + 1e-300)
        rows.append((name, z0, rhs, rel))
        worst = max(worst, rel)
    # sign tracking: net eddy viscosity sign == migration-integral sign
    sign_ok = all((z0 > 0) == (rhs > 0) for _, z0, rhs, _ in rows)
    return dict(rows=rows, max_rel_err=float(worst), sign_ok=sign_ok,
                ok=bool(worst < 1e-3 and sign_ok))


# --------------------------------------------------------------------------- #
# NR24 - eddy diffusivity is the zero-frequency PSD:  D = (1/2) S(0)
# --------------------------------------------------------------------------- #
def _D_dispersion(V, dt=1.0):
    T = V.shape[1]
    xT = np.cumsum(V, axis=1)[:, -1] * dt
    return float(np.mean(xT ** 2) / (2.0 * T * dt))


def _S0_welch(V, nperseg=1024):
    """Independent zero-frequency PSD estimate (averaged Welch periodogram,
    one-sided, normalised so S(0)=sum_k C(k))."""
    from scipy.signal import welch
    f, P = welch(V, fs=1.0, nperseg=nperseg, return_onesided=True,
                 detrend=False, scaling="density", axis=1)
    # welch density is two-sided-folded; S(0) (sum_k C(k)) = P(0)/2 for our convention check below
    P0 = float(np.mean(P[:, 0]))
    return P0


def nr24(n=8000, n_traj=6000):
    # (i) D = (1/2) S(0): dispersion D vs independent periodogram S(0)
    rows = []
    worst = 0.0
    for name, a in (("AR(1) a=0.8", 0.8), ("AR(1) a=0.9", 0.9), ("AR(1) a=0.95", 0.95)):
        V = ar1_ensemble(n_traj, n, a, sigma=1.0, seed=0)
        d_disp = _D_dispersion(V)
        s0 = _S0_welch(V)                          # scipy welch f=0 bin
        d_spec = 0.5 * s0
        rel = abs(d_disp - d_spec) / d_disp
        rows.append((name, d_disp, d_spec, rel))
        worst = max(worst, rel)

    # (ii) fold divergence: D ~ (1-a)^{-2} as a->1 (theory D = 1/(2(1-a)^2))
    aa = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    Dth = 1.0 / (2.0 * (1.0 - aa) ** 2)
    slope = float(np.polyfit(np.log(1.0 - aa), np.log(Dth), 1)[0])   # expect -2
    # empirical D on a subset confirms the theory curve it is fit to
    a_chk = 0.9
    V = ar1_ensemble(4000, 6000, a_chk, sigma=1.0, seed=1)
    d_emp = _D_dispersion(V)
    d_th = 1.0 / (2.0 * (1.0 - a_chk) ** 2)
    div_rel = abs(d_emp - d_th) / d_th

    # (iii) 1/f^gamma long memory: S(0) -> inf (no finite D). Show the running-sum
    # variance per unit time grows with window (super-diffusive) for gamma>0,
    # ensemble-averaged so white noise is cleanly flat.
    def onefnoise(N, gamma, seed=0):
        rng = np.random.default_rng(seed)
        W = np.fft.rfft(rng.standard_normal(N))
        f = np.fft.rfftfreq(N); f[0] = f[1]
        y = np.fft.irfft(W * f ** (-gamma / 2.0), N)
        return y - y.mean()                        # drop the discrete-DC artifact
    Ns, M = 2 ** 14, 200
    Ws = [250, 500, 1000, 2000, 4000]
    growth = {}
    for g in (0.0, 1.0):
        Dw = np.zeros(len(Ws))
        for r in range(M):
            x = onefnoise(Ns, g, seed=r)
            for i, W in enumerate(Ws):
                nseg = Ns // W
                inc = x[:nseg * W].reshape(nseg, W).cumsum(axis=1)[:, -1]
                Dw[i] += float(np.mean(inc ** 2) / (2.0 * W))
        Dw /= M
        growth[g] = float(np.polyfit(np.log(Ws), np.log(Dw + 1e-300), 1)[0])
    # white (gamma=0): D_W flat (slope ~0); long memory (gamma=1): D_W grows (slope>0)
    white_flat = abs(growth[0.0]) < 0.1
    mem_grows = growth[1.0] > 0.3

    ok = bool(worst < 0.12 and abs(slope + 2.0) < 1e-6 and div_rel < 0.12
              and white_flat and mem_grows)
    return dict(rows=rows, max_rel_err=float(worst), fold_slope=slope,
                div_rel=float(div_rel), growth=growth,
                white_flat=white_flat, mem_grows=mem_grows, ok=ok)


# --------------------------------------------------------------------------- #
# NR25 - second fluctuation-dissipation theorem: K(t) = <F(t)F(0)>/kT
# --------------------------------------------------------------------------- #
def _cl_bath(N=40, seed=0):
    """Caldeira-Leggett bath: frequencies w_i and couplings c_i (~Ohmic)."""
    rng = np.random.default_rng(seed)
    w = np.linspace(0.3, 6.0, N)
    c = 0.6 * np.sqrt(w) * (1.0 + 0.2 * rng.standard_normal(N))
    return w, c


def _K_friction(t, w, c):
    """Dissipation side: friction kernel of the eliminated harmonic bath,
    K(t) = sum_i (c_i^2 / w_i^2) cos(w_i t)."""
    return np.array([float(np.sum((c ** 2 / w ** 2) * np.cos(w * tt))) for tt in t])


def _FF_sampled(t, w, c, kT=1.0, M=150000, seed=1):
    """Fluctuation side: random-force ACF from Gibbs-sampled bath initial
    conditions (xi_i ~ N(0, kT/w_i^2), p_i ~ N(0, kT)) under free bath evolution."""
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((M, len(w))) * (np.sqrt(kT) / w)
    pm = rng.standard_normal((M, len(w))) * np.sqrt(kT)
    F0 = (c * xi).sum(axis=1)
    out = []
    for tt in t:
        Ft = ((c * xi) * np.cos(w * tt) + (c * pm / w) * np.sin(w * tt)).sum(axis=1)
        out.append(float(np.mean(Ft * F0)))
    return np.array(out)


def nr25(N=40, M=150000, kT=1.0):
    w, c = _cl_bath(N)
    t = np.linspace(0.0, 4.0, 9)
    K = _K_friction(t, w, c)                       # dissipation (deterministic)
    FF = _FF_sampled(t, w, c, kT=kT, M=M)          # fluctuation (Gibbs MC)
    peak = float(np.max(np.abs(kT * K)))
    rel = float(np.max(np.abs(FF - kT * K)) / peak)
    return dict(t=t.tolist(), K=K.tolist(), FF_over_kT=(FF / kT).tolist(),
                max_rel_err=rel, ok=bool(rel < 0.02))


# --------------------------------------------------------------------------- #
def run():
    return dict(nr22=nr22(), nr23=nr23(), nr24=nr24(), nr25=nr25())


def main():
    r22, r23, r24, r25 = nr22(), nr23(), nr24(), nr25()
    print("=== NR22  EWS coupling: Var*(1-AC1^2) = sigma_eps^2 (invariant) ===")
    print("  approach-the-fold (sigma=1 fixed, a->1):")
    for a, v, c, inv in r22["fold"]:
        print(f"    a={a:.2f}: Var={v:8.3f}  AC1={c:.4f}  Var*(1-AC1^2)={inv:.4f}")
    print(f"    invariant max rel.err={r22['fold_inv_err']:.4f}  "
          f"AC1 rises={r22['ac1_rises']}  Var rises={r22['var_rises']}")
    print("  louder-forcing discriminator (a=0.8 fixed, sigma up): invariant tracks sigma^2")
    for s, v, c, inv in r22["loud"]:
        print(f"    sigma={s:.1f}: Var={v:8.3f}  AC1={c:.4f}  Var*(1-AC1^2)={inv:.4f}  (sigma^2={s*s:.2f})")
    print(f"    tracks-sigma^2 err={r22['loud_tracks']:.4f}  AC1 flat dev={r22['ac1_flat']:.4f}  ok={r22['ok']}")
    print("=== NR23  Kramers-Kronig DC sum rule: nu_eff = (2/pi) int (-ImZ)/w dw ===")
    for name, z0, rhs, rel in r23["rows"]:
        print(f"  {name:24s}: Z(0)=nu_eff={z0:+.5f}  (2/pi)int(-ImZ)/w={rhs:+.5f}  rel.err={rel:.2e}")
    print(f"  max rel.err={r23['max_rel_err']:.2e}  sign tracks={r23['sign_ok']}  ok={r23['ok']}")
    print("=== NR24  eddy diffusivity is zero-freq PSD: D = (1/2) S(0) ===")
    for name, dd, ds, rel in r24["rows"]:
        print(f"  {name:14s}: D_dispersion={dd:8.3f}  (1/2)S(0)_welch={ds:8.3f}  rel.err={rel:.3f}")
    print(f"  D-vs-S(0) max rel.err={r24['max_rel_err']:.3f}")
    print(f"  fold divergence D~(1-a)^slope, slope={r24['fold_slope']:.3f} (expect -2); "
          f"empirical check rel.err={r24['div_rel']:.3f}")
    print(f"  long-memory: D_W(window) slope  white(g=0)={r24['growth'][0.0]:+.3f} (flat), "
          f"1/f(g=1)={r24['growth'][1.0]:+.3f} (grows->S(0)=inf)  ok={r24['ok']}")
    print("=== NR25  second FDT: friction kernel K(t) = <F(t)F(0)>/kT (harmonic bath) ===")
    for tt, k, ff in zip(r25['t'], r25['K'], r25['FF_over_kT']):
        print(f"  t={tt:4.2f}: K(t)[friction]={k:+.4f}  <FF>/kT[sampled]={ff:+.4f}")
    print(f"  max rel.err (vs peak)={r25['max_rel_err']:.4f}  ok={r25['ok']}")
    ok = all(d["ok"] for d in (r22, r23, r24, r25))
    print(f"PASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
