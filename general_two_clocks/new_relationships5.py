r"""New derived cross-relationship NR28, continuing the program (NR1-NR27) with the
same discipline: *derived* from mainstream theory + a repo result and *numerically
verified* (CPU, deterministic).  See ``REPORT_NEW_RELATIONSHIPS5.md`` for the write-up
and ``tests/test_new_relationships5.py`` for the unit proof.

NR28 - THE TWO CLOCKS IN ONE CROSS-SPECTRUM: pressure (elliptic, instantaneous) LEADS
       temperature (parabolic, lagged) by the phase arctan(omega/(kappa k^2)), and the
       p-T COHERENCE drops at high frequency -- the clocks decouple at small scales.
       [the foundational two-clocks thesis as a co-located-sensor measurement]

  Context.  The repo's core thesis (README "Core Thesis"; REPORT_RB, REPORT_THEORY,
  REPORT_TIMESCALE_SEPARATION) is that pressure and temperature run on different clocks
  AND different spatial operators: incompressible pressure is a GLOBAL, ELLIPTIC field
  (the Leray/Poisson solve is an instantaneous diagnostic of the strain, p ~ -Delta^{-1}
  d_i d_j (u_i u_j)), whereas temperature is a LOCAL, PARABOLIC field (the prognostic
  heat equation, carrying memory of past straining through the diffusive integral).  The
  NR program (NR1-NR27) has mined the CLOSURE and FOLD consequences of this split, but
  never written down the split's most direct OBSERVABLE: the cross-spectrum between a
  co-located pressure proxy and temperature proxy.  NR28 supplies it.

  Setup (one representative mode k, both driven by the same turbulent straining d(t)).
  Diagnostic (elliptic) pressure responds INSTANTANEOUSLY to the common driver; prognostic
  (parabolic) temperature responds through a first-order diffusive lag with corner
  omega_c = kappa k^2, and additionally carries its OWN small-scale filamentation that the
  global pressure does not (independent noise n_theta -- "temperature is torn into
  filaments", REPORT_RB):
        p(t)      = d(t) + n_p(t)                         (elliptic: instantaneous)
        theta'(t) = -omega_c theta + d(t),  + n_theta     (parabolic: lagged + filaments)
  so the transfer is H_theta(omega) = 1/(omega_c + i omega).

  Derived consequences (all verified below against the simulated processes).
  (a) THE CROSS-SPECTRAL PHASE IS THE PARABOLIC CLOCK.  Pressure leads temperature by
        phi(omega) = arg S_{p,theta}(omega) = arctan(omega / omega_c),
      rising 0 -> 90deg, with the 45deg crossover EXACTLY at omega = omega_c = kappa k^2.
      A co-located (p, theta) sensor pair therefore MEASURES the parabolic clock from the
      phase crossover alone -- no diffusivity calibration, no amplitude calibration.
  (b) THE COHERENCE DROPS AT HIGH FREQUENCY -> the clocks DECOUPLE at small scales.  The
      common-driver coherence gamma^2(omega) = |S_{p,theta}|^2 / (S_pp S_thetatheta) is ~1
      at low omega (both fields quasi-statically track the driver: clocks coupled) and
      FALLS at high omega, because the parabolic low-pass shrinks temperature's common
      content (|H_theta|^2 ~ omega^-2) until its independent filamentation dominates.  The
      coherence half-fall frequency is a second, independent read of the decoupling scale.
  (c) ONE-SIDED CAUSALITY.  The phase is POSITIVE for all omega (pressure leads, never
      lags): the elliptic field is the instantaneous diagnostic, the parabolic field the
      laggard.  The lead is bounded by 90deg (a pure first-order lag), distinguishing the
      diffusive clock from a pure transport delay (which would give an unbounded linear
      phase omega*tau).

  Net: the two-clocks decoupling -- the repo's foundational claim -- is a single
  cross-spectral signature on co-located pressure + temperature time series (exactly what
  NEON/ASOS/reanalysis provide): a bounded, one-sided phase arctan(omega/omega_c) that
  measures the parabolic clock omega_c=kappa k^2, plus a high-frequency coherence drop that
  is the decoupling itself.  Mainstream: Bendat & Piersol (cross-spectral phase/coherence);
  the heat equation (parabolic) vs the pressure Poisson equation (elliptic, Leray 1934).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.signal import csd, coherence, lfilter, welch


# --------------------------------------------------------------------------- #
# analytic closed forms
# --------------------------------------------------------------------------- #
def phase_analytic(omega, omega_c):
    """Cross-spectral phase: pressure leads temperature by arctan(omega/omega_c)."""
    return np.arctan2(np.asarray(omega, float), omega_c)


def coherence_analytic(omega, omega_c, s_d, n_p, n_th):
    """Common-driver coherence gamma^2 with independent sensor/filamentation noise.

    p = d + n_p ;  theta = H_theta * d + n_th,  |H_theta|^2 = 1/(omega_c^2+omega^2).
    S_pp     = s_d + n_p
    S_thth   = s_d/(omega_c^2+omega^2) + n_th
    |S_p,th| = s_d/sqrt(omega_c^2+omega^2)       (only the common driver correlates)
    gamma^2  = |S_p,th|^2 / (S_pp S_thth).
    """
    w = np.asarray(omega, float)
    Ht2 = 1.0 / (omega_c ** 2 + w ** 2)
    Spp = s_d + n_p
    Stt = s_d * Ht2 + n_th
    Spt2 = (s_d ** 2) * Ht2
    return Spt2 / (Spp * Stt)


# --------------------------------------------------------------------------- #
# simulate one mode: common driver d, elliptic p (instant), parabolic theta (lagged)
# --------------------------------------------------------------------------- #
def simulate(omega_c, dt=0.02, n=4_000_000, s_d=1.0, n_p=0.05, n_th=0.25, seed=0):
    """White common driver d; p = d + sensor noise; theta = first-order lag of d + filaments."""
    rng = np.random.default_rng(seed)
    d = np.sqrt(s_d) * rng.standard_normal(n)            # common turbulent straining (white)
    p = d + np.sqrt(n_p) * rng.standard_normal(n)        # elliptic: instantaneous diagnostic
    a = np.exp(-omega_c * dt)                            # exact first-order (parabolic) lag
    theta_common = lfilter([(1.0 - a) / omega_c], [1.0, -a], d)
    theta = theta_common + np.sqrt(n_th) * rng.standard_normal(n)   # + small-scale filaments
    return p, theta


# --------------------------------------------------------------------------- #
# NR28
# --------------------------------------------------------------------------- #
def nr28(omega_c=0.5, dt=0.02, n=4_000_000, s_d=1.0, n_p=0.05, n_th=0.25,
         nperseg=16384, seed=0):
    p, theta = simulate(omega_c, dt=dt, n=n, s_d=s_d, n_p=n_p, n_th=n_th, seed=seed)
    fs = 1.0 / dt

    # cross-spectrum phase and magnitude-squared coherence
    f, Pxy = csd(p, theta, fs=fs, nperseg=nperseg)
    fc, gamma2 = coherence(p, theta, fs=fs, nperseg=nperseg)
    w = 2.0 * np.pi * f
    # scipy csd(p, theta) uses the conjugate convention -> arg = -arctan(w/w_c); the physical
    # LEAD of (elliptic, instantaneous) pressure over (parabolic, lagged) temperature is the
    # negative of that.  gamma2 shares this same frequency grid (identical nperseg).
    phi = -np.angle(Pxy)                                 # p-leads-theta phase = +arctan(w/w_c)

    # Phase estimates are only meaningful where the coherence is non-negligible (Bendat &
    # Piersol Sec. 9); and the claim arctan(w/w_c) is a *continuous-time* law, so we validate
    # it in the band where the sampled first-order lag still tracks continuous time (w*dt<<1).
    band = (w > 0.1 * omega_c) & (w < 8.0 * omega_c) & (gamma2 > 0.1)
    phi_pred = phase_analytic(w[band], omega_c)
    err = np.abs(phi[band] - phi_pred)
    g_band = gamma2[band]
    # coherence-weighted RMS phase error (robust to noisy low-coherence high-omega bins)
    phase_rms_err = float(np.sqrt(np.sum(g_band * err ** 2) / np.sum(g_band)))
    phase_p95_err = float(np.percentile(err, 95))
    phase_max_err = float(np.max(err))

    # (a) recover omega_c from the PHASE ALONE (no amplitude/diffusivity calibration):
    # (i) interpolated 45deg crossover: omega where phi = pi/4  ->  omega = omega_c
    sel = (w > 0.05 * omega_c) & (w < 5.0 * omega_c)
    o = np.argsort(phi[sel]); ps, ws = phi[sel][o], w[sel][o]
    omega_c_cross = float(np.interp(np.pi / 4, ps, ws))
    crossover_rel_err = abs(omega_c_cross - omega_c) / omega_c
    # (ii) linear fit tan(phi) = w/omega_c on the well-conditioned high-coherence low band
    lin = (w > 0.1 * omega_c) & (np.abs(phi) < 1.0) & (gamma2 > 0.3)
    slope = float(np.sum(w[lin] * np.tan(phi[lin])) / np.sum(w[lin] ** 2))   # = 1/omega_c
    omega_c_fit = 1.0 / slope
    omega_c_fit_err = abs(omega_c_fit - omega_c) / omega_c

    # (b) coherence: high at low omega, drops at high omega (decoupling)
    w_coh = 2.0 * np.pi * fc
    def med_gamma(lo, hi):
        m = (w_coh >= lo) & (w_coh <= hi)
        return float(np.median(gamma2[m])) if np.any(m) else np.nan
    gamma_low = med_gamma(0.05 * omega_c, 0.3 * omega_c)     # slow / large scales
    gamma_high = med_gamma(8.0 * omega_c, 25.0 * omega_c)    # fast / small scales
    gamma_an_low = float(coherence_analytic(0.15 * omega_c, omega_c, s_d, n_p, n_th))
    gamma_an_high = float(coherence_analytic(15.0 * omega_c, omega_c, s_d, n_p, n_th))
    coherence_drops = bool(gamma_high < 0.6 * gamma_low)
    coh_low_ok = abs(gamma_low - gamma_an_low) < 0.08
    coh_high_ok = abs(gamma_high - gamma_an_high) < 0.08

    # (c) one-sided causality: phase positive (pressure leads) and bounded by 90deg, evaluated
    # over the coherent band where the phase estimate is meaningful
    one_sided = bool(np.mean(phi[band] > 0) > 0.98)
    bounded_90 = bool(np.percentile(phi[band], 98) < np.pi / 2)

    # checks
    phase_ok = bool(phase_rms_err < 0.05 and phase_p95_err < 0.15)   # ~2deg RMS over the band
    crossover_ok = bool(crossover_rel_err < 0.10)
    fit_ok = bool(omega_c_fit_err < 0.06)
    coh_ok = bool(coherence_drops and coh_low_ok and coh_high_ok)
    causal_ok = bool(one_sided and bounded_90)

    ok = bool(phase_ok and crossover_ok and fit_ok and coh_ok and causal_ok)

    return dict(
        params=dict(omega_c=omega_c, dt=dt, n=n, s_d=s_d, n_p=n_p, n_th=n_th,
                    nperseg=nperseg),
        phase=dict(rms_err_rad=phase_rms_err, p95_err_rad=phase_p95_err,
                   max_err_rad=phase_max_err, omega_c_true=omega_c,
                   omega_c_from_crossover=omega_c_cross, crossover_rel_err=crossover_rel_err,
                   omega_c_from_fit=omega_c_fit, fit_rel_err=omega_c_fit_err),
        coherence=dict(gamma_low=gamma_low, gamma_high=gamma_high,
                       gamma_an_low=gamma_an_low, gamma_an_high=gamma_an_high,
                       drops=coherence_drops),
        causality=dict(one_sided_pressure_leads=one_sided, bounded_by_90deg=bounded_90),
        checks=dict(phase_ok=phase_ok, crossover_ok=crossover_ok, fit_ok=fit_ok,
                    coh_ok=coh_ok, causal_ok=causal_ok),
        verdict=(
            "Co-located pressure (elliptic, instantaneous) and temperature (parabolic, "
            "lagged) have a cross-spectral phase phi(omega)=arctan(omega/omega_c) with the "
            f"45deg crossover at the parabolic clock omega_c=kappa k^2 (recovered to "
            f"{omega_c_fit_err*100:.1f}% from the phase alone, no calibration). The phase is "
            "POSITIVE for all omega (pressure leads, never lags) and BOUNDED by 90deg (a "
            "first-order diffusive lag, not an unbounded transport delay). The magnitude-"
            f"squared coherence falls from {gamma_low:.2f} at low omega (clocks coupled, "
            f"large scales) to {gamma_high:.2f} at high omega (clocks DECOUPLED, small "
            "scales) as the parabolic low-pass cedes temperature to its own filamentation. "
            "The two-clocks decoupling is thus one cross-spectral signature on the kind of "
            "co-located p+T time series NEON/ASOS/reanalysis already provide."),
        ok=ok)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = res["params"]
    wc = p["omega_c"]
    w = np.logspace(np.log10(0.03 * wc), np.log10(30 * wc), 400)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # (a) phase = arctan(omega/omega_c)
    ax[0].semilogx(w / wc, np.degrees(phase_analytic(w, wc)), "-", color="#1f77b4",
                   label=r"$\phi=\arctan(\omega/\omega_c)$ (derived)")
    ax[0].axhline(45, color="#888", lw=0.8, ls="--")
    ax[0].axvline(1.0, color="#2ca02c", lw=1, ls=":", label=r"$\omega=\omega_c=\kappa k^2$ (45$\degree$)")
    ax[0].set_xlabel(r"$\omega/\omega_c$"); ax[0].set_ylabel("p$\\to$T phase lead (deg)")
    ax[0].set_title("(a) phase = the parabolic clock (one-sided, $<$90$\\degree$)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    # (b) coherence drop = the decoupling
    c = res["params"]
    ax[1].semilogx(w / wc, coherence_analytic(w, wc, c["s_d"], c["n_p"], c["n_th"]),
                   "-", color="#d62728", label=r"$\gamma^2(\omega)$ (derived)")
    ax[1].axvline(1.0, color="#2ca02c", lw=1, ls=":")
    ax[1].set_xlabel(r"$\omega/\omega_c$"); ax[1].set_ylabel(r"coherence $\gamma^2$")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("(b) coherence drops $\\Rightarrow$ clocks decouple at small scales")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("NR28 - the two clocks in one cross-spectrum: elliptic pressure leads parabolic temperature, and they decohere at high frequency", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures", "nr28_two_clocks_cross_spectrum.json"))
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    res = nr28(n=a.n)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("=== NR28 - the two clocks in one cross-spectrum ===")
    ph = res["phase"]
    print(f"  (a) phase=arctan(w/w_c): RMS err {ph['rms_err_rad']:.3f} rad (max {ph['max_err_rad']:.3f}); "
          f"omega_c {ph['omega_c_true']}->{ph['omega_c_from_fit']:.3f} (fit {ph['fit_rel_err']*100:.1f}%), "
          f"crossover {ph['omega_c_from_crossover']:.3f} ({ph['crossover_rel_err']*100:.1f}%)")
    co = res["coherence"]
    print(f"  (b) coherence low {co['gamma_low']:.2f} (an {co['gamma_an_low']:.2f}) -> "
          f"high {co['gamma_high']:.2f} (an {co['gamma_an_high']:.2f}); drops={co['drops']}")
    ca = res["causality"]
    print(f"  (c) one-sided (pressure leads)={ca['one_sided_pressure_leads']}, "
          f"bounded<90deg={ca['bounded_by_90deg']}")
    print(f"  checks={res['checks']}")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  ok={res['ok']}  json -> {a.out}")
    if not a.no_fig:
        make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
