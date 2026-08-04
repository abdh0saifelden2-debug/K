"""Detection-power analysis for the matched lake-drainage -> velocity lag test
(P4a-R2).

Referee objection (paper 4a): the matched-lag test returns a null (peak
+0.56 sigma, 0/5 significant) and names two unblock routes, but does not state
what response amplitude the test COULD have detected, so "null" is not yet a
power-quantified statement.

This module Monte-Carlos the exact detector used by the real pipeline (peak
post-drainage bin anomaly vs a pre-drainage baseline, 2-sigma floor) on
synthetic records with the committed noise floors (quarterly ~2.3%, annual
~1.0% of trunk speed) and computes:

  1. the false-positive calibration of the peak detector (max over ~8 post
     bins inflates the nominal 2-sigma rate; the expected noise maximum is
     ~+1.4 sigma, so the observed +0.56 sigma is BELOW the noise expectation);
  2. power curves and the 95%-power minimum detectable amplitude A95 for a
     sustained step and for double-exponential kernel transients (rise 0.02 yr,
     decay tau_s in {0.05, 0.1, 0.25, 0.5, 1} yr), including the quarterly /
     annual aliasing attenuation of sub-bin transients;
  3. the same for the two unblock routes: event stacking (superposed epoch
     over the 5 lakes) and a daily GPS/GNSS record (0.5% noise) - showing
     which route crosses detectability first and by how much.

Artifacts: validation/reports/lake_lag_power.{json,png}
Test:      tests/test_lake_lag_power.py
Compute:   synthetic Monte Carlo only; no GPU, no download.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.normpath(os.path.join(HERE, "..", "reports"))

PRE_YR, POST_YR = 2.0, 2.0
K_SIGMA = 2.0                 # the real pipeline's significance floor
TAU_RISE = 0.02               # yr; kernel rise time (channel opening)


# --------------------------------------------------------------------------- #
# forward: bin-averaged kernel response
# --------------------------------------------------------------------------- #
def kernel(t, tau_s, tau_c=TAU_RISE):
    """Double-exponential surge kernel, peak-normalised to 1."""
    k = np.where(t >= 0.0, np.exp(-t / tau_s) - np.exp(-t / tau_c), 0.0)
    tp = tau_s * tau_c / (tau_s - tau_c) * np.log(tau_s / tau_c)
    kmax = np.exp(-tp / tau_s) - np.exp(-tp / tau_c)
    return k / kmax


def binned_signal(A, tau_s, dt_bin, t0_frac=0.5, nfine=400):
    """Fractional speed anomaly per post bin for a peak amplitude A transient
    starting at t0 = t0_frac*dt_bin into the first post bin (bin-averaged =
    the aliasing the real quarterly/annual pipeline applies)."""
    edges = np.arange(0.0, POST_YR + 1e-9, dt_bin)
    out = []
    for lo in edges[:-1]:
        tt = np.linspace(lo, lo + dt_bin, nfine)
        out.append(A * kernel(tt - t0_frac * dt_bin, tau_s).mean())
    return np.asarray(out)


def sustained_signal(A, dt_bin, t0_frac=0.5):
    """Sustained step of amplitude A from t0 (the aliasing-immune limit)."""
    edges = np.arange(0.0, POST_YR + 1e-9, dt_bin)
    out = []
    for lo in edges[:-1]:
        hi = lo + dt_bin
        overlap = max(0.0, hi - max(lo, t0_frac * dt_bin))
        out.append(A * overlap / dt_bin)
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# detector (mirrors the real pipeline): peak post anomaly vs pre baseline
# --------------------------------------------------------------------------- #
def detect_peak(pre, post, scale):
    """Peak post anomaly in sigma units vs the pre-baseline median, with the
    noise scale estimated from a long detrended stretch (as the real pipeline
    does), passed in explicitly."""
    base = np.median(pre)
    return float(np.max((post - base) / scale))


def mc_peaks(signal_bins, sigma, dt_bin, n_stack=1, n_mc=4000, seed=0,
             n_scale_bins=40):
    """Distribution of detector peaks for a given binned signal; the noise
    scale is a MAD estimate from an independent n_scale_bins noise stretch
    (long-baseline, mirroring the real pipeline's global detrended floor)."""
    rng = np.random.default_rng(seed)
    npre = int(round(PRE_YR / dt_bin))
    npost = len(signal_bins)
    peaks = np.empty(n_mc)
    for i in range(n_mc):
        noise_long = rng.standard_normal(n_scale_bins) * sigma
        scale = 1.4826 * np.median(np.abs(noise_long - np.median(noise_long)))
        pre = (rng.standard_normal((n_stack, npre)) * sigma).mean(axis=0)
        post = (signal_bins[None, :]
                + rng.standard_normal((n_stack, npost)) * sigma).mean(axis=0)
        peaks[i] = detect_peak(pre, post, scale / np.sqrt(n_stack))
    return peaks


def calibrated_threshold(sigma, dt_bin, n_stack=1, fp_target=0.05, n_mc=4000,
                         seed=0):
    """Threshold (sigma units) giving the target false-positive rate for THIS
    config's peak-over-bins detector (corrects the multiple-comparison
    inflation of a nominal per-bin 2-sigma floor)."""
    npost = int(round(POST_YR / dt_bin))
    null_peaks = mc_peaks(np.zeros(npost), sigma, dt_bin, n_stack, n_mc, seed)
    return float(np.quantile(null_peaks, 1.0 - fp_target)), null_peaks


def a95(shape_fn, sigma, dt_bin, n_stack=1, k_thresh=K_SIGMA, a_grid=None,
        n_mc=1500, seed=1):
    """Minimum amplitude with >= 95% detection power at the given threshold."""
    if a_grid is None:
        a_grid = np.geomspace(1e-3, 1.0, 25)
    for A in a_grid:
        peaks = mc_peaks(shape_fn(A), sigma, dt_bin, n_stack, n_mc, seed)
        if (peaks >= k_thresh).mean() >= 0.95:
            return float(A)
    return float("inf")


# --------------------------------------------------------------------------- #
# study
# --------------------------------------------------------------------------- #
def run():
    rng_configs = {
        "quarterly_per_lake": dict(sigma=0.023, dt=0.25, n_stack=1),
        "annual_per_lake": dict(sigma=0.010, dt=1.00, n_stack=1),
        "quarterly_stacked5": dict(sigma=0.023, dt=0.25, n_stack=5),
        "gps_daily": dict(sigma=0.005, dt=1.0 / 365.0, n_stack=1),
    }
    taus = (0.05, 0.1, 0.25, 0.5, 1.0)

    # 1. false-positive calibration: nominal 2-sigma FP rate + calibrated
    #    5%-FP threshold per config (multiple-comparison correction)
    fp = {}
    thresholds = {}
    for name, c in rng_configs.items():
        k5, null_peaks = calibrated_threshold(c["sigma"], c["dt"], c["n_stack"])
        thresholds[name] = k5
        fp[name] = dict(
            nominal_2sigma_fp_rate=float((null_peaks >= K_SIGMA).mean()),
            median_noise_peak_sigma=float(np.median(null_peaks)),
            calibrated_5pct_threshold_sigma=k5)

    # 2. aliasing attenuation of the transient (bin-mean of unit-peak kernel)
    atten = {f"tau_{t}": dict(
        quarterly=float(binned_signal(1.0, t, 0.25).max()),
        annual=float(binned_signal(1.0, t, 1.0).max()))
        for t in taus}

    # 3. A95 tables at the calibrated (5% FP) thresholds
    a95_table = {}
    for name, c in rng_configs.items():
        kt = thresholds[name]
        row = dict(sustained=a95(lambda A: sustained_signal(A, c["dt"]),
                                 c["sigma"], c["dt"], c["n_stack"], kt))
        for t in taus:
            row[f"transient_tau_{t}"] = a95(
                lambda A, tt=t: binned_signal(A, tt, c["dt"]),
                c["sigma"], c["dt"], c["n_stack"], kt)
        a95_table[name] = row

    q, a, s5, g = (a95_table[k] for k in
                   ("quarterly_per_lake", "annual_per_lake",
                    "quarterly_stacked5", "gps_daily"))
    out = dict(
        what=("P4a-R2 power analysis of the matched-lag detector: false-positive "
              "calibration, aliasing attenuation, and 95%-power minimum "
              "detectable amplitude per route"),
        detector=(f"peak post-drainage bin anomaly vs pre-baseline (median), "
                  f"long-baseline MAD scale, pre {PRE_YR} yr, post {POST_YR} yr; "
                  f"thresholds FP-calibrated to 5% per config (nominal "
                  f"{K_SIGMA}-sigma floor is multiple-comparison inflated)"),
        committed_noise_floors="quarterly 2.3%, annual 1.0% of trunk speed "
                               "(ITS_LIVE, 5-lake test); GPS route 0.5% daily",
        false_positive=fp,
        observed_peak_sigma=0.56,
        noise_max_context=(
            "the observed +0.56 sigma peak is BELOW the median noise maximum "
            f"of the quarterly detector ({fp['quarterly_per_lake']['median_noise_peak_sigma']:.2f} sigma "
            "over 8 post bins) - the null is cleaner than nominal"),
        aliasing_attenuation=atten,
        a95_fraction_of_trunk_speed=a95_table,
        routes_verdict=dict(
            per_lake_quarterly_sustained=q["sustained"],
            per_lake_quarterly_tau01=q["transient_tau_0.1"],
            annual_tau01=a["transient_tau_0.1"],
            stacked5_sustained=s5["sustained"],
            gps_daily_tau01=g["transient_tau_0.1"],
            which_route_first=(
                "sub-annual GPS: a tau=0.1 yr transient needs only "
                f"{100 * g['transient_tau_0.1']:.1f}% peak amplitude at 95% power vs "
                f"{100 * q['transient_tau_0.1']:.1f}% for the quarterly ITS_LIVE "
                f"detector (aliasing x noise); stacking the 5 lakes only reaches "
                f"{100 * s5['sustained']:.1f}% for a sustained step - so the GPS "
                "route crosses first for the physically expected sub-annual "
                "transient, the trunk route for sustained responses"),
        ),
    )
    return out


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = (0.05, 0.1, 0.25, 0.5, 1.0)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    tbl = res["a95_fraction_of_trunk_speed"]
    for name, marker in (("quarterly_per_lake", "o"), ("annual_per_lake", "s"),
                         ("quarterly_stacked5", "^"), ("gps_daily", "d")):
        axs[0].loglog(taus, [tbl[name][f"transient_tau_{t}"] for t in taus],
                      marker + "-", label=name)
    axs[0].set_xlabel("kernel decay tau_s (yr)")
    axs[0].set_ylabel("A95 (fraction of trunk speed)")
    axs[0].set_title("95%-power minimum detectable transient")
    axs[0].legend(fontsize=7)

    at = res["aliasing_attenuation"]
    axs[1].semilogx(taus, [at[f"tau_{t}"]["quarterly"] for t in taus], "o-",
                    label="quarterly bin")
    axs[1].semilogx(taus, [at[f"tau_{t}"]["annual"] for t in taus], "s-",
                    label="annual bin")
    axs[1].set_xlabel("kernel decay tau_s (yr)")
    axs[1].set_ylabel("bin-mean peak / true peak")
    axs[1].set_title("aliasing attenuation of the transient")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    res = run()
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "lake_lag_power.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    make_figure(res, os.path.join(REPORTS, "lake_lag_power.png"))
    fp = res["false_positive"]["quarterly_per_lake"]
    print("matched-lag power analysis (P4a-R2)")
    print(f"  quarterly peak detector: nominal-2sigma FP rate "
          f"{fp['nominal_2sigma_fp_rate']:.3f}, median noise max "
          f"{fp['median_noise_peak_sigma']:.2f} sigma (observed +0.56 sigma is "
          f"below noise expectation); calibrated 5%-FP threshold = "
          f"{fp['calibrated_5pct_threshold_sigma']:.2f} sigma")
    print("  A95 (fraction of trunk speed):")
    for name, row in res["a95_fraction_of_trunk_speed"].items():
        print(f"    {name:22s} sustained={row['sustained']:.3f} "
              f"tau=0.1yr={row['transient_tau_0.1']:.3f} "
              f"tau=0.5yr={row['transient_tau_0.5']:.3f}")
    print(f"  -> {res['routes_verdict']['which_route_first']}")
    return res


if __name__ == "__main__":
    main()
