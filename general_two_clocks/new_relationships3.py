r"""New derived cross-relationship NR26, continuing the program (NR1-NR25) with the
same discipline: *derived* from mainstream theory + a repo result and *numerically
verified* (CPU, deterministic).  See ``REPORT_NEW_RELATIONSHIPS3.md`` for the
write-up and ``tests/test_new_relationships3.py`` for the unit proof.

NR26 - THE TWO FAILURE MODES ARE COUPLED AT FINITE DISTANCE: bath memory biases the
       critical-slowing-down early warning by the Deborah number  [Paper 1 (De) x Paper 4 (fold)].

  Context.  The cross-relationship index partitions NR1-NR25 into two structures the
  mainstream local/equilibrium description fails by: (1) discarded MEMORY / reactive
  operator structure, organised by the Deborah number De=tau_c/tau_event (NR1, NR13,
  NR15); and (2) CRITICALITY at a saddle-node FOLD - the s_N flotation pole / MISI -
  with critical slowing down (CSD: rising variance + lag-1 autocorrelation, NR3/NR22;
  Lorentzian corner f_c, NR8).  The index argues these are *separate* because
  "De = tau_c/tau_relax VANISHES at a fold (tau_relax -> inf)".  That asymptotic
  statement is true but hides the operationally decisive fact this relationship
  establishes: the LEADING correction to every CSD early-warning estimator at FINITE
  distance from the fold is O(De), it has a DEFINITE SIGN, and a memoryless (white-noise)
  early warning therefore mis-reads the distance to tipping.

  Setup (both structures in one minimal, mainstream model).  A slow mode s near the
  fold has a vanishing restoring rate lambda (lambda -> 0 at the fold; tau_relax=1/lambda
  is the diverging CSD time).  By the exact Mori-Zwanzig / second-FDT reduction the repo
  certifies (NR25; Zwanzig 1973), the fast eliminated bath supplies s with a friction +
  random force that is NOT white but carries the bath's own finite memory time tau_c.
  The minimal realisation is a slow linear mode driven by Ornstein-Uhlenbeck (colored)
  noise (Hanggi & Jung 1995):
        s' = -lambda s + eta ,   eta' = -eta/tau_c + (sqrt(2D)/tau_c) xi(t) ,
  so <eta(t)eta(0)> = (D/tau_c) e^{-|t|/tau_c}  (white-noise intensity D as tau_c->0).
  De := lambda*tau_c = tau_c/tau_relax is exactly the index's memory ratio.

  Derived consequences (all verified below against the simulated process).
  (a) VARIANCE SUPPRESSION + a SIGNED bias.  Var(s) = D / (lambda (1 + De)).  The
      white-noise CSD law is Var=D/lambda; memory SUPPRESSES the variance by 1/(1+De).
      A variance-based EWS that inverts lambda_V = D/Var reports lambda_V = lambda(1+De)
      > lambda -- it reads the system as FARTHER from the fold (falsely SAFER) by 1+De.
  (b) BI-EXPONENTIAL ACF -> the two standard EWS estimators BRACKET the truth.
      C(t)/C(0) = [e^{-lambda t} - De e^{-t/tau_c}] / (1 - De): the autocorrelation is a
      DIFFERENCE OF TWO EXPONENTIALS (rates lambda and 1/tau_c), not the single-exponential
      AR(1)/OU that CSD theory (Wissel 1984; Dakos 2012) assumes.  Consequently the lag-1
      (AC1) rate lambda_A=-ln AC1/dt reads LOW (lambda_A<lambda, false ALARM) while the
      variance rate lambda_V reads HIGH (false safety): lambda_A < lambda < lambda_V.  The
      apparent rate -ln AC(t)/t therefore GROWS with lag; its lag-slope is a
      noise-intensity-(D)-FREE gauge of bath memory (zero only when De=0).
  (c) DE-BIASABLE FROM THE RECORD ALONE.  A blind two-exponential fit of the measured ACF
      recovers BOTH rates (the true proximity lambda AND the bath memory 1/tau_c), removing
      the bias without knowing the noise intensity D.

  Net: the two structures the index separates are coupled at finite distance, the coupling
  is the Deborah number, it is a SIGNED bias in the CSD early warning, and it is removable
  from the same record.  This sharpens Paper 4b's tidal-admittance/CSD ungrounding warning.
  Mainstream: Zwanzig (1973); Hanggi & Jung (1995); Scheffer et al. (2009); Dakos et al.
  (2012); Wissel (1984).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lfilter


# --------------------------------------------------------------------------- #
# analytic (continuous-time) closed forms, D=1
# --------------------------------------------------------------------------- #
def var_analytic(lam, tau_c, D=1.0):
    """Var(s) = D / (lambda (1 + lambda tau_c))."""
    return D / (lam * (1.0 + lam * tau_c))


def acf_analytic(t, lam, tau_c):
    """Normalised ACF: [e^{-lam t} - De e^{-t/tau_c}] / (1 - De),  De = lam tau_c."""
    de = lam * tau_c
    t = np.asarray(t, float)
    return (np.exp(-lam * t) - de * np.exp(-t / tau_c)) / (1.0 - de)


# --------------------------------------------------------------------------- #
# exact-ZOH simulation of the slow mode driven by OU (colored) noise
# --------------------------------------------------------------------------- #
def simulate(lam, tau_c, D=1.0, dt=0.02, n=4_000_000, seed=0):
    """Generate s(t) for  s'=-lam s+eta,  eta=OU(tau_c) of white-intensity D.

    eta: exact OU update  eta_{k+1}=a_e eta_k + b_e w_k ,  Var(eta)=D/tau_c.
    s  : zero-order-hold (eta const over dt) exact update  s_{k+1}=a_s s_k+(1-a_s)eta_k/lam.
    Both are first-order recursions applied with scipy.signal.lfilter (vectorised).
    """
    rng = np.random.default_rng(seed)
    a_e = np.exp(-dt / tau_c)
    var_eta = D / tau_c
    b_e = np.sqrt(var_eta * (1.0 - a_e * a_e))
    w = rng.standard_normal(n)
    eta = lfilter([b_e], [1.0, -a_e], w)
    eta *= np.sqrt(var_eta / np.var(eta))             # pin stationary bath variance
    a_s = np.exp(-lam * dt)
    s = lfilter([(1.0 - a_s) / lam], [1.0, -a_s], eta)
    burn = min(n // 10, 200_000)
    return s[burn:]


def ac_at(s, lags):
    """Normalised autocorrelation of s at integer sample lags (direct estimator)."""
    s = s - s.mean()
    v = float(np.dot(s, s) / s.size)
    out = []
    for L in lags:
        out.append(float(np.dot(s[:-L], s[L:]) / (s.size - L) / v))
    return np.array(out)


# --------------------------------------------------------------------------- #
# NR26
# --------------------------------------------------------------------------- #
def nr26(lams=(0.05, 0.10, 0.20, 0.40), tau_c=1.0, D=1.0,
         dt=0.02, n=4_000_000, ews_dt=2.0, seed=0):
    rows = []
    m = int(round(ews_dt / dt))               # EWS sampling stride (samples per ews_dt)
    for lam in lams:
        de = lam * tau_c
        s = simulate(lam, tau_c, D=D, dt=dt, n=n, seed=seed)

        # (a) variance suppression + white-noise variance-EWS bias (needs D)
        var_sim = float(np.var(s))
        var_an = var_analytic(lam, tau_c, D)
        lam_V = D / var_sim                    # variance inversion assuming white noise
        bias_V = lam_V / lam                   # predicted = 1 + De

        # (b) bi-exponential ACF, bracketing, and the D-free lag-slope gauge
        lags = [m, 2 * m, 3 * m, 4 * m]        # dt..4dt, all robustly estimated
        ac_sim = ac_at(s, lags)
        ac_an = acf_analytic(np.array(lags) * dt, lam, tau_c)
        lam_app = -np.log(np.clip(ac_sim, 1e-9, None)) / (np.array(lags) * dt)
        lam_A = float(lam_app[0])              # AC1 (lag-dt) white-noise rate inversion (D-free)
        gauge = float(lam_app[1] - lam_app[0])  # lambda_app grows with lag for De>0
        brackets = bool(lam_A < lam < lam_V)   # AC1 low (alarm), variance high (safety)

        # (c) NR22 white-noise invariant Var*(1-AC1^2): no longer the innovation variance
        nr22_inv = var_sim * (1.0 - ac_sim[0] ** 2)

        rows.append(dict(
            lam=lam, De=de,
            var_sim=var_sim, var_analytic=var_an,
            var_rel_err=abs(var_sim / var_an - 1.0),
            lam_A=lam_A, lam_true=lam, lam_V=lam_V,
            bias_V_measured=bias_V, bias_V_predicted=1.0 + de, brackets=brackets,
            ac_sim=ac_sim.tolist(), ac_analytic=ac_an.tolist(),
            ac_max_abs_err=float(np.max(np.abs(ac_sim - ac_an))),
            lam_app_lags=lam_app.tolist(), apprate_lagslope=gauge,
            nr22_invariant=nr22_inv))

    # (c) DE-BIAS: blind 2-exponential fit of the measured ACF recovers BOTH timescales
    # (true proximity lambda AND bath memory 1/tau_c) on the strongest-memory case.
    lam_hi = max(lams)
    s = simulate(lam_hi, tau_c, D=D, dt=dt, n=n, seed=seed + 1)
    Lfit = np.arange(1, int(round(8 * tau_c / dt)))
    acf = ac_at(s, Lfit.tolist())
    t = Lfit * dt

    def biexp(tt, c1, r1, r2):                 # c1 e^{-r1 t} + (1-c1) e^{-r2 t}
        return c1 * np.exp(-r1 * tt) + (1.0 - c1) * np.exp(-r2 * tt)

    de_hi = lam_hi * tau_c
    p0 = [1.3, lam_hi * 1.6, (1.0 / tau_c) * 0.6]   # deliberately-off start -> blind recovery
    popt, _ = curve_fit(biexp, t, acf, p0=p0, maxfev=40000)
    c1_fit, r_slow, r_fast = popt
    if r_slow > r_fast:                          # order: slow rate < fast rate
        r_slow, r_fast = r_fast, r_slow
    lam_rec_err = abs(r_slow - lam_hi) / lam_hi
    tau_c_rec_err = abs(1.0 / r_fast - tau_c) / tau_c
    debias_err = float(max(lam_rec_err, tau_c_rec_err))

    var_ok = all(r["var_rel_err"] < 0.05 for r in rows)
    bias_ok = all(abs(r["bias_V_measured"] - r["bias_V_predicted"]) < 0.03 for r in rows)
    acf_ok = all(r["ac_max_abs_err"] < 0.02 for r in rows)
    bias_monotone = all(rows[i]["bias_V_measured"] < rows[i + 1]["bias_V_measured"]
                        for i in range(len(rows) - 1))
    bracket_ok = all(r["brackets"] for r in rows)             # lam_A < lam < lam_V
    gauge_positive = all(r["apprate_lagslope"] > 0 for r in rows)
    gauge_monotone = all(rows[i]["apprate_lagslope"] < rows[i + 1]["apprate_lagslope"]
                         for i in range(len(rows) - 1))
    debias_ok = debias_err < 0.10

    ok = bool(var_ok and bias_ok and acf_ok and bias_monotone and bracket_ok
              and gauge_positive and gauge_monotone and debias_ok)

    return dict(
        params=dict(tau_c=tau_c, D=D, dt=dt, n=n, ews_dt=ews_dt, lams=list(lams)),
        rows=rows,
        debias=dict(lam_true=lam_hi, lam_recovered=float(r_slow),
                    tau_c_true=tau_c, tau_c_recovered=float(1.0 / r_fast),
                    c1_fit=float(c1_fit), c1_predicted=float(1.0 / (1.0 - de_hi)),
                    lam_rel_err=float(lam_rec_err), tau_c_rel_err=float(tau_c_rec_err),
                    max_rel_err=debias_err),
        checks=dict(var_ok=var_ok, bias_ok=bias_ok, acf_ok=acf_ok,
                    bias_monotone=bias_monotone, bracket_ok=bracket_ok,
                    gauge_positive=gauge_positive, gauge_monotone=gauge_monotone,
                    debias_ok=debias_ok),
        verdict=(
            "A slow critical mode (lambda->0 at the fold) driven by a finite-memory bath "
            "(tau_c) has Var=D/(lambda(1+De)), De=lambda*tau_c: memory SUPPRESSES the CSD "
            "variance by 1/(1+De), so the white-noise variance EWS reads lambda_V=lambda(1+De) "
            "(false SAFETY) while the lag-1 AC1 rate reads lambda_A<lambda (false ALARM) -- the "
            "two standard precursors BRACKET the truth and split by ~De. The ACF is "
            "bi-exponential, so the apparent rate grows with lag (a D-free memory gauge) and a "
            "blind 2-exp fit recovers (lambda, tau_c), de-biasing the proximity from the record "
            "alone. The index's two 'separate' failure modes are coupled at finite distance, and "
            "the coupling is exactly the Deborah number."),
        ok=ok)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    De = [r["De"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    # (a) the bracket: AC1 rate (alarm) < truth < variance rate (safety)
    ax[0].plot(De, [r["bias_V_measured"] for r in rows], "o", ms=8, color="#d62728",
               label=r"$\lambda_V/\lambda$ (variance: false safety)")
    ax[0].plot(De, [r["lam_A"] / r["lam"] for r in rows], "s", ms=8, color="#1f77b4",
               label=r"$\lambda_A/\lambda$ (AC1: false alarm)")
    xx = np.linspace(0, max(De) * 1.05, 100)
    ax[0].plot(xx, 1 + xx, "-", color="#d62728", lw=1)
    ax[0].axhline(1.0, color="k", lw=0.8, label=r"truth $\lambda$")
    ax[0].set_xlabel("De = $\\lambda\\,\\tau_c$"); ax[0].set_ylabel("apparent rate / true rate")
    ax[0].set_title("(a) the two CSD precursors bracket the truth"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # (b) bi-exponential ACF for the strongest-memory case
    r = rows[-1]; lam = r["lam"]; tc = res["params"]["tau_c"]
    t = np.linspace(0, 8 * tc, 300)
    ax[1].plot(t, acf_analytic(t, lam, tc), "-", color="#1f77b4", label="bi-exponential (derived)")
    ax[1].plot(t, np.exp(-lam * t), "--", color="#888", label=r"white-noise AR(1) $e^{-\lambda t}$")
    ax[1].set_xlabel("lag $t$"); ax[1].set_ylabel("ACF"); ax[1].set_title(f"(b) ACF is bi-exponential (De={r['De']:.2f})")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    # (c) apparent-rate lag-slope (D-free memory gauge)
    ax[2].plot(De, [r["apprate_lagslope"] for r in rows], "^-", color="#2ca02c")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("De = $\\lambda\\,\\tau_c$")
    ax[2].set_ylabel(r"$\lambda_{app}(2\Delta t)-\lambda_{app}(\Delta t)$")
    ax[2].set_title("(c) apparent-rate lag-slope: a D-free memory gauge"); ax[2].grid(alpha=0.3)
    fig.suptitle("NR26 - bath memory biases the critical-slowing-down early warning by the Deborah number", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures", "nr26_memory_ews_bias.json"))
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    res = nr26(n=a.n)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("=== NR26 - bath memory biases the CSD early warning by De ===")
    for r in res["rows"]:
        print(f"  De={r['De']:.2f}: Var sim/an={r['var_sim']:.3f}/{r['var_analytic']:.3f} "
              f"({r['var_rel_err']*100:.2f}%)  bracket lam_A<{r['lam']}<lam_V: "
              f"{r['lam_A']:.3f}<{r['lam']:.2f}<{r['lam_V']:.3f} [{r['brackets']}]  "
              f"lagslope={r['apprate_lagslope']:+.4f}  ACFerr={r['ac_max_abs_err']:.4f}")
    d = res["debias"]
    print(f"  de-bias (blind 2-exp, De={d['lam_true']*d['tau_c_true']:.2f}): "
          f"lambda {d['lam_true']:.3f}->{d['lam_recovered']:.3f} ({d['lam_rel_err']*100:.1f}%), "
          f"tau_c {d['tau_c_true']:.3f}->{d['tau_c_recovered']:.3f} ({d['tau_c_rel_err']*100:.1f}%)")
    print(f"  checks={res['checks']}")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  ok={res['ok']}  json -> {a.out}")
    if not a.no_fig:
        make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
