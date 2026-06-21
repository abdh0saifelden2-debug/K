r"""New derived cross-relationship NR27, continuing the program (NR1-NR26) with the
same discipline: *derived* from mainstream theory + a repo result and *numerically
verified* (CPU, deterministic).  See ``REPORT_NEW_RELATIONSHIPS4.md`` for the write-up
and ``tests/test_new_relationships4.py`` for the unit proof.

NR27 - THE FLUCTUATION-RESPONSE (FIRST-FDT) FACE OF THE FOLD: the tidal-admittance
       RESPONSE measures proximity to ungrounding noise-free, and the fold mode's
       FDT-violation is a frequency-resolved effective temperature whose roll-off
       MEASURES the bath memory.  [Paper 4b admittance x Paper 4a hydraulic memory]

  Context.  NR25 certified the *second* fluctuation-dissipation theorem for the
  Mori-Zwanzig reduction (the certified memory kernel IS the random-force
  autocorrelation; <F(t)F(0)> = k_B T K(t)).  NR26 then showed that a slow critical
  fold mode driven by that finite-memory (colored) bath has its critical-slowing-down
  *correlations* biased by the Deborah number De = lambda*tau_c (variance suppressed
  1/(1+De), bi-exponential ACF, the two precursors bracket the truth).  NR26 is a
  statement about CORRELATIONS only.  Its missing dual is the RESPONSE side -- the
  *first* FDT -- which the repo already measures in the field as Paper 4b's tidal
  velocity admittance.  NR27 supplies it.

  Setup (the same minimal, mainstream model as NR26, now also forced).  A slow mode s
  near the fold with vanishing restoring rate lambda (lambda -> 0 at the fold), driven by
  an Ornstein-Uhlenbeck bath of memory tau_c and white-intensity D, AND by a small
  external (tidal) force f(t):
        s' = -lambda s + eta + f(t) ,  eta = OU(tau_c),  <eta(t)eta(0)> = (D/tau_c)e^{-|t|/tau_c}.
  De := lambda*tau_c.  Two observables of the SAME record:
    * RESPONSE  chi(omega) = 1/(lambda + i*omega)  (admittance to the tidal force);
    * FLUCTUATION spectrum  S_s(omega) = 2D / [(lambda^2+omega^2)(1+omega^2 tau_c^2)].

  Derived consequences (all verified below against the simulated process).
  (a) THE RESPONSE IS NOISE-FREE -> a calibration-free proximity gauge.  chi(omega) =
      1/(lambda+i*omega) does NOT depend on the noise intensity D or on the bath memory
      tau_c: it is the bare deterministic relaxation.  Hence the static admittance
      |chi(0)| = 1/lambda DIVERGES at the fold and measures the distance to ungrounding
      with NO knowledge of the noise.  From a measured chi(omega): lambda = Re(1/chi),
      independent of D.  (This is the response face of Paper 4b's |s_N|.)
  (b) THE FLUCTUATION SPECTRUM IS A DOUBLE LORENTZIAN -> a model-free memory signature.
      S_s ~ omega^-2 at high frequency for a WHITE bath (the standard CSD assumption,
      NR8), but ~ omega^-4 once the bath has memory; the SLOPE steepens from -2 to -4
      and the crossover frequency is exactly 1/tau_c.  The high-frequency log-log slope
      of the velocity spectrum is therefore a calibration-free (D-free, lambda-free) test
      of whether the early-warning bath has memory at all.
  (c) THE FDT IS VIOLATED BY A FREQUENCY-DEPENDENT EFFECTIVE TEMPERATURE.  Form the
      first-FDT ratio (Kubo 1966; Cugliandolo-Kurchan-Peliti 1997 effective temperature;
      Harada-Sasa 2005 violation):
            T_eff(omega) := omega * S_s(omega) / (2 |Im chi(omega)|) = D / (1 + omega^2 tau_c^2).
      In equilibrium (white bath, De->0) this is a CONSTANT = D and the FDT holds.  With
      bath memory it ROLLS OFF as a Lorentzian: plateau D at low omega (the equilibrium
      temperature) -> 0 at high omega (FDT maximally violated), half-value at omega=1/tau_c.
      The roll-off MEASURES tau_c and the plateau measures D -> De = lambda*tau_c is
      recovered from RESPONSE + FLUCTUATION (a frequency-domain de-bias of NR26's
      time-domain bi-exponential-ACF fit).

  Net: the fold's correlation face (NR26) and its response face (NR27) are the two
  halves of one fluctuation-dissipation statement.  The response alone gives a
  noise-free proximity (Paper 4b admittance); the FDT violation between response and
  fluctuation gives a signed, frequency-resolved readout of the hydraulic bath memory
  (Paper 4a tau_c) that de-biases the CSD ungrounding warning.  Mainstream: Kubo (1966);
  Cugliandolo, Kurchan & Peliti (1997); Harada & Sasa (2005); Zwanzig (1973); Hanggi &
  Jung (1995); Scheffer et al. (2009); Dakos et al. (2012); Gudmundsson (2011).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lfilter, welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import new_relationships3 as nr3  # noqa: E402  (reuse the verified OU-bath sim + var law)


# --------------------------------------------------------------------------- #
# analytic (continuous-time) closed forms, generic D
# --------------------------------------------------------------------------- #
def chi(omega, lam):
    """Admittance / linear response of s to the external force: chi = 1/(lam + i*omega).

    Independent of the noise (D) and the bath memory (tau_c): the bare relaxation.
    """
    return 1.0 / (lam + 1j * np.asarray(omega, float))


def spectrum(omega, lam, tau_c, D=1.0):
    """Fluctuation PSD (two-sided, angular): 2D / [(lam^2+w^2)(1+w^2 tau_c^2)]."""
    w = np.asarray(omega, float)
    return 2.0 * D / ((lam ** 2 + w ** 2) * (1.0 + w ** 2 * tau_c ** 2))


def teff_analytic(omega, tau_c, D=1.0):
    """First-FDT effective temperature  T_eff(w)=w S(w)/(2|Im chi|) = D/(1+w^2 tau_c^2)."""
    w = np.asarray(omega, float)
    return D / (1.0 + w ** 2 * tau_c ** 2)


# --------------------------------------------------------------------------- #
# forced simulation: s' = -lam s + eta(OU,tau_c,D) + A cos(omega0 t)   (exact ZOH)
# --------------------------------------------------------------------------- #
def simulate_forced(lam, tau_c, omega0, A, D=1.0, dt=0.02, n=1_500_000, seed=0):
    """s(t) for the OU-driven slow mode plus a sinusoidal (tidal) force at omega0.

    Reuses the exact-ZOH machinery: with input g held constant over dt, the s-update is
    s_{k+1} = a_s s_k + (1-a_s) g_k / lam,  a_s = e^{-lam dt}.  Here g = eta + f.
    """
    rng = np.random.default_rng(seed)
    a_e = np.exp(-dt / tau_c)
    var_eta = D / tau_c
    if D <= 0.0:
        eta = np.zeros(n)                               # noiseless: response is deterministic
    else:
        b_e = np.sqrt(var_eta * (1.0 - a_e * a_e))
        eta = lfilter([b_e], [1.0, -a_e], rng.standard_normal(n))
        eta *= np.sqrt(var_eta / np.var(eta))           # pin stationary bath variance
    t = np.arange(n) * dt
    g = eta + A * np.cos(omega0 * t)
    a_s = np.exp(-lam * dt)
    s = lfilter([(1.0 - a_s) / lam], [1.0, -a_s], g)
    return t, s


def measure_response(lam, tau_c, omega0, D=1.0, A=1.0, dt=0.02, n=1_500_000, seed=0):
    """Lock-in (cross-correlation) detection of the complex response chi(omega0).

    Steady response to f = A cos(w0 t) is  <s> = A[Re chi cos(w0 t) - Im chi sin(w0 t)].
    Projecting the (burn-in-trimmed) record over an integer number of forcing periods:
        X = <s cos> * 2 = A Re chi ,   Y = <s sin> * 2 = -A Im chi.
    The noise eta averages out; the result is independent of D (verified).
    """
    t, s = simulate_forced(lam, tau_c, omega0, A, D=D, dt=dt, n=n, seed=seed)
    period = 2.0 * np.pi / omega0
    burn = int(min(n // 5, 10.0 * max(1.0 / lam, period) / dt))     # >> relaxation & a period
    t, s = t[burn:], s[burn:]
    nper = int(((t[-1] - t[0]) / period))                          # integer periods only
    keep = int(round(nper * period / dt))
    t, s = t[:keep], s[:keep]
    c = np.cos(omega0 * t)
    sn = np.sin(omega0 * t)
    X = 2.0 * np.mean(s * c)            # = A Re chi
    Y = 2.0 * np.mean(s * sn)           # = -A Im chi
    re_chi = X / A
    im_chi = -Y / A
    chi_meas = re_chi + 1j * im_chi
    inv = 1.0 / chi_meas
    return dict(omega0=omega0, chi_meas=chi_meas, lam_from_resp=float(inv.real),
                w_from_resp=float(inv.imag))


def loglog_slope(f, P, fmin, fmax):
    """Least-squares log-log slope of P(f) over [fmin, fmax] (f in Hz, P>0)."""
    m = (f >= fmin) & (f <= fmax) & (P > 0)
    return float(np.polyfit(np.log(f[m]), np.log(P[m]), 1)[0])


# --------------------------------------------------------------------------- #
# NR27
# --------------------------------------------------------------------------- #
def nr27(lam=0.10, tau_c=1.0, D=1.0, dt=0.02, n=3_000_000,
         probe_w=(0.05, 0.1, 0.2, 0.4, 0.8), seed=0):
    # ----- (a) response is noise-free: chi from a noiseless run (exact) + a noisy run ---- #
    # The response equation has no D, so chi is deterministic.  The D=0 lock-in recovers
    # 1/(lam+iw) to discretization precision (the rigorous proof); a D>0 record recovers the
    # SAME chi within statistical error (so the proximity read-out needs no noise calibration).
    resp = []
    rdt = 0.01                                          # finer dt -> continuous-limit response
    for w0 in probe_w:
        r0 = measure_response(lam, tau_c, w0, D=0.0, A=1.0, dt=rdt, n=n // 4, seed=seed)      # exact
        rN = measure_response(lam, tau_c, w0, D=D, A=1.0, dt=rdt, n=n // 2, seed=seed + 7)    # noisy
        chi_an = chi(w0, lam)
        resp.append(dict(
            omega0=w0,
            chi_noiseless=[r0["chi_meas"].real, r0["chi_meas"].imag],
            chi_noisy=[rN["chi_meas"].real, rN["chi_meas"].imag],
            chi_analytic=[chi_an.real, chi_an.imag],
            err_noiseless_vs_analytic=float(abs(r0["chi_meas"] - chi_an) / abs(chi_an)),
            noisefree_err=float(abs(rN["chi_meas"] - r0["chi_meas"]) / abs(chi_an)),
            lam_from_resp_noisy=rN["lam_from_resp"]))
    lam_resp = float(np.median([r["lam_from_resp_noisy"] for r in resp]))   # proximity from noisy record
    resp_err = max(r["err_noiseless_vs_analytic"] for r in resp)            # exact: << 1%
    noisefree_err = max(r["noisefree_err"] for r in resp)                   # noisy recovers exact
    lam_resp_err = abs(lam_resp - lam) / lam

    # ----- (b) double-Lorentzian spectrum: high-f slope -4 (memory) vs -2 (white) ---- #
    s_free = nr3.simulate(lam, tau_c, D=D, dt=dt, n=n, seed=seed + 1)
    s_white = nr3.simulate(lam, 1e-3, D=D, dt=dt, n=n, seed=seed + 2)   # tau_c->0 control
    nperseg = 16384
    f, P = welch(s_free, fs=1.0 / dt, nperseg=nperseg)                 # one-sided PSD (per Hz)
    fw, Pw = welch(s_white, fs=1.0 / dt, nperseg=nperseg)
    fc = 1.0 / (2.0 * np.pi * tau_c)                                   # bath corner in Hz
    slope_mem = loglog_slope(f, P, 3.0 * fc, 12.0 * fc)                # above the bath corner
    slope_white = loglog_slope(fw, Pw, 3.0 * fc, 12.0 * fc)            # same band, white bath

    # ----- (c) FDT effective temperature T_eff(w)=w S(w)/(2|Im chi|)=D/(1+w^2 tau_c^2) -- #
    # S_meas from welch (convert one-sided per-Hz PSD to two-sided angular: S(w)=P(f)/(4 pi));
    # |Im chi| analytic at lam_resp (chi validated in (a)).  Fit T_eff -> (D, tau_c).
    w = 2.0 * np.pi * f
    band = (w > 0) & (f <= 0.6 / dt / 2)                               # drop DC + near-Nyquist
    w_b = w[band]
    # one-sided per-Hz welch PSD -> two-sided angular PSD: P(f) = 2 S(w=2 pi f)  =>  S = P/2
    S_meas = P[band] / 2.0
    im_chi = np.abs(w_b / (lam_resp ** 2 + w_b ** 2))                  # |Im chi|, chi=1/(lam+iw)
    T_eff = w_b * S_meas / (2.0 * im_chi)

    def teff_form(ww, Dfit, tcfit):
        return Dfit / (1.0 + ww ** 2 * tcfit ** 2)

    wfit = (w_b >= 0.2 / tau_c) & (w_b <= 8.0 / tau_c)                 # around the roll-off
    popt, _ = curve_fit(teff_form, w_b[wfit], T_eff[wfit],
                        p0=[D * 1.5, tau_c * 0.5], maxfev=40000)
    D_fit, tau_c_fit = float(popt[0]), float(abs(popt[1]))
    # white control: T_eff should be ~flat (no roll-off) over the same window
    Sw_meas = Pw[band] / (4.0 * np.pi)
    Tw_eff = w_b * Sw_meas / (2.0 * im_chi)
    tw_lo = float(np.median(Tw_eff[(w_b >= 0.2 / tau_c) & (w_b <= 0.5 / tau_c)]))
    tw_hi = float(np.median(Tw_eff[(w_b >= 4.0 / tau_c) & (w_b <= 8.0 / tau_c)]))
    white_flat_ratio = tw_hi / tw_lo                                   # ~1 if FDT holds (white)
    # memory case: same low/high comparison should show the predicted 1/(1+(w tau_c)^2) drop
    tm_lo = float(np.median(T_eff[(w_b >= 0.2 / tau_c) & (w_b <= 0.5 / tau_c)]))
    tm_hi = float(np.median(T_eff[(w_b >= 4.0 / tau_c) & (w_b <= 8.0 / tau_c)]))
    mem_drop_ratio = tm_hi / tm_lo                                     # << 1 if memory present

    de_true = lam * tau_c
    de_recovered = lam_resp * tau_c_fit                                # response x FDT roll-off

    # ----- checks ----- #
    resp_ok = resp_err < 0.01                                          # noiseless lock-in is exact
    Dindep_ok = noisefree_err < 0.06                                   # noisy record recovers it
    proximity_ok = lam_resp_err < 0.05
    slope_mem_ok = abs(slope_mem - (-4.0)) < 0.5
    slope_white_ok = abs(slope_white - (-2.0)) < 0.5
    tcfit_ok = abs(tau_c_fit - tau_c) / tau_c < 0.15
    Dfit_ok = abs(D_fit - D) / D < 0.15
    white_flat_ok = abs(white_flat_ratio - 1.0) < 0.15                 # FDT holds for white bath
    mem_drop_ok = mem_drop_ratio < 0.25                                # FDT violated for memory
    debias_ok = abs(de_recovered - de_true) / de_true < 0.15

    ok = bool(resp_ok and Dindep_ok and proximity_ok and slope_mem_ok and slope_white_ok
              and tcfit_ok and Dfit_ok and white_flat_ok and mem_drop_ok and debias_ok)

    return dict(
        params=dict(lam=lam, tau_c=tau_c, D=D, dt=dt, n=n, probe_w=list(probe_w), De=de_true),
        response=dict(rows=resp, lam_from_response=lam_resp, lam_true=lam,
                      lam_resp_rel_err=lam_resp_err, max_err_vs_analytic=resp_err,
                      max_D_independence_err=noisefree_err),
        spectrum=dict(corner_hz=fc, highf_slope_memory=slope_mem,
                      highf_slope_white=slope_white,
                      slope_band_hz=[3.0 * fc, 12.0 * fc]),
        fdt=dict(D_fit=D_fit, D_true=D, tau_c_fit=tau_c_fit, tau_c_true=tau_c,
                 white_flat_ratio=white_flat_ratio, mem_drop_ratio=mem_drop_ratio,
                 mem_drop_predicted=float(teff_analytic(6.0 / tau_c, tau_c)
                                          / teff_analytic(0.35 / tau_c, tau_c))),
        debias=dict(De_true=de_true, De_recovered=de_recovered,
                    De_rel_err=abs(de_recovered - de_true) / de_true),
        checks=dict(resp_ok=resp_ok, Dindep_ok=Dindep_ok, proximity_ok=proximity_ok,
                    slope_mem_ok=slope_mem_ok, slope_white_ok=slope_white_ok,
                    tcfit_ok=tcfit_ok, Dfit_ok=Dfit_ok, white_flat_ok=white_flat_ok,
                    mem_drop_ok=mem_drop_ok, debias_ok=debias_ok),
        verdict=(
            "The fold mode's RESPONSE chi(w)=1/(lam+iw) is noise-free: the tidal admittance "
            "|chi(0)|=1/lam measures proximity to ungrounding with no knowledge of the noise D "
            "(verified D-independent and lam recovered to "
            f"{lam_resp_err*100:.1f}%). Its FLUCTUATION spectrum is a DOUBLE Lorentzian: the "
            f"high-frequency slope steepens from {slope_white:.2f} (white bath, ~-2) to "
            f"{slope_mem:.2f} (memory bath, ~-4), a model-free memory signature. The first-FDT "
            "effective temperature T_eff(w)=w S/(2|Im chi|)=D/(1+w^2 tau_c^2) is FLAT for a white "
            f"bath (ratio {white_flat_ratio:.2f}) but rolls off for a memory bath (ratio "
            f"{mem_drop_ratio:.2f}); fitting the roll-off recovers tau_c to "
            f"{abs(tau_c_fit-tau_c)/tau_c*100:.1f}% and D to {abs(D_fit-D)/D*100:.1f}%, so "
            "De=lam*tau_c is de-biased from RESPONSE + FLUCTUATION. NR27 is the first-FDT / "
            "response dual of NR26's correlation-only result: the fold has one "
            "fluctuation-dissipation structure with two measurable faces."),
        ok=ok)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = res["params"]
    lam, tau_c, D = p["lam"], p["tau_c"], p["D"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    # (a) measured response vs analytic in the complex plane (Nyquist-style)
    rows = res["response"]["rows"]
    w = np.array([r["omega0"] for r in rows])
    ca = chi(np.linspace(min(w), max(w), 200), lam)
    ax[0].plot(ca.real, ca.imag, "-", color="#1f77b4", label=r"$\chi=1/(\lambda+i\omega)$ (derived)")
    ax[0].plot([r["chi_noiseless"][0] for r in rows], [r["chi_noiseless"][1] for r in rows],
               "o", ms=7, color="#d62728", label="measured (lock-in, noise-free)")
    ax[0].set_xlabel(r"Re $\chi$"); ax[0].set_ylabel(r"Im $\chi$")
    ax[0].set_title("(a) tidal admittance = noise-free proximity"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # (b) double-Lorentzian spectrum: slope -2 (white) vs -4 (memory)
    ww = np.logspace(-2, 1.5, 400)
    ax[1].loglog(ww, spectrum(ww, lam, tau_c, D), "-", color="#1f77b4", label=r"memory bath ($\propto\omega^{-4}$)")
    ax[1].loglog(ww, spectrum(ww, lam, 1e-3, D), "--", color="#888", label=r"white bath ($\propto\omega^{-2}$)")
    ax[1].axvline(1.0 / tau_c, color="#2ca02c", lw=1, ls=":", label=r"bath corner $1/\tau_c$")
    ax[1].set_xlabel(r"$\omega$"); ax[1].set_ylabel(r"$S_s(\omega)$")
    ax[1].set_title(f"(b) double Lorentzian: slope {res['spectrum']['highf_slope_white']:.1f}$\\to${res['spectrum']['highf_slope_memory']:.1f}")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    # (c) FDT effective temperature: flat (white, FDT holds) vs roll-off (memory)
    ax[2].semilogx(ww, teff_analytic(ww, tau_c, D), "-", color="#d62728", label=r"$T_{eff}=D/(1+\omega^2\tau_c^2)$")
    ax[2].semilogx(ww, teff_analytic(ww, 1e-3, D), "--", color="#888", label="white: $T_{eff}=D$ (FDT holds)")
    ax[2].axvline(1.0 / tau_c, color="#2ca02c", lw=1, ls=":", label=r"$1/\tau_c$ (recovered)")
    ax[2].set_xlabel(r"$\omega$"); ax[2].set_ylabel(r"$T_{eff}(\omega)$")
    ax[2].set_title("(c) FDT violation roll-off measures the bath memory"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, which="both")
    fig.suptitle("NR27 - fluctuation-response (first-FDT) face of the fold: noise-free proximity + memory-measuring FDT violation", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures", "nr27_fluctuation_response_fdt.json"))
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    res = nr27(n=a.n)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("=== NR27 - fluctuation-response (first-FDT) face of the fold ===")
    r = res["response"]
    print(f"  (a) response noise-free: chi err<= {r['max_err_vs_analytic']*100:.2f}%, "
          f"D-independence<= {r['max_D_independence_err']*100:.2f}%, "
          f"lambda from response {r['lam_true']:.3f}->{r['lam_from_response']:.3f} "
          f"({r['lam_resp_rel_err']*100:.1f}%)")
    s = res["spectrum"]
    print(f"  (b) high-f slope white {s['highf_slope_white']:.2f} (~-2) -> memory "
          f"{s['highf_slope_memory']:.2f} (~-4); bath corner {s['corner_hz']:.4f} Hz")
    fd = res["fdt"]
    print(f"  (c) T_eff white-flat ratio {fd['white_flat_ratio']:.2f} (~1, FDT holds), "
          f"memory drop ratio {fd['mem_drop_ratio']:.2f} (<<1, FDT violated); "
          f"fit tau_c {fd['tau_c_true']}->{fd['tau_c_fit']:.3f}, D {fd['D_true']}->{fd['D_fit']:.3f}")
    d = res["debias"]
    print(f"  de-bias De={d['De_true']:.3f}->{d['De_recovered']:.3f} ({d['De_rel_err']*100:.1f}%)")
    print(f"  checks={res['checks']}")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  ok={res['ok']}  json -> {a.out}")
    if not a.no_fig:
        make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
