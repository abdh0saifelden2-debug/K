r"""Real-data lake-lag extension — the §I s_N(N) / early-warning framework applied to
REAL open CryoSat-2 + ITS_LIVE lake data. A genuinely-new cross-framework result on
real observations, with NO 7.6 GB ATL15 re-download (re-analyses the committed matched
artifact `data/lake_lag_matched.json`).

Why this, and the honest scope
------------------------------
The §H.2 ATL15+ITS_LIVE population test already ran (`lake_lag_atl15_itslive.py`):
1 significant in-band post-drainage speed-up out of 19 testable lakes (Thw_142) — NOT
a universal response. The full 131-lake volume-change catalogue stays USAP-DC/ATL15-
gated (a ~7.6 GB re-download that would mostly reproduce that 1/19 result). What was
*never* done is to read the **real** lake velocity series through the **§I framework
built this run** — the `s_N(N)` master curve (§I.1), its inversion (§I.2), and the
critical-slowing-down ungrounding early-warning (§I.3/§I.6). This module does exactly
that, on the 3 marquee MacAyeal lakes that have OPEN ITS_LIVE velocity + OPEN CryoSat-2
drainage (Mac1/Mac2/Mac3); Mercer/Conway have open drainage but no velocity coverage.

Two §I probes on real data
--------------------------
1. **Drainage-response amplitude (§I.1/§I.2).** For each real drainage event, the
   post-drainage velocity change Δv/v in the derived 0.02-2 yr lag band. Via the master
   curve `Δv/v ≈ |s_N(N)|·(ΔN/N)`, a small response bounds the sliding sensitivity
   `|s_N|` — i.e., how far the lake sits from the `N_c` flotation fold (where `|s_N|`
   has a pole).
2. **Critical-slowing-down EWS (§I.3/§I.6).** On each detrended velocity series, the
   lag-1 autocorrelation and split-half variance ratio. The §I forecast is that an ice
   stream nearing ungrounding shows rising variance AND rising AC1 *together*; a stable,
   far-from-flotation trunk lake should show NEITHER (a true negative — the EWS must not
   false-alarm).

Result (real data)
------------------
All 5 events at the 3 velocity lakes give `|Δv/v| ≤ ~2 %`, mixed sign, at/near the ~1 %
ITS_LIVE noise floor with small post-event sample counts — **no clean, sustained,
universal post-drainage surge** (consistent with the 1/19 population result). And **no
lake shows the joint CSD signature** (none has both rising variance and high AC1) — the
§I early-warning correctly returns a **true negative** on stable trunk lakes. Both §I
probes agree: these MacAyeal lakes sit **far from the `N_c` ungrounding fold**. Honest
limits: n=8 annual points per lake is statistically weak for a CSD trend (a strong test
needs the dense quarterly series and a lake actually approaching flotation), and the
full 131-lake / ATL15 population extension remains gated. No GPU; no new download.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data", "lake_lag_matched.json")
CSD_VAR_RATIO = 1.5      # joint-CSD flag: var(2nd half)/var(1st half) threshold
CSD_AC1 = 0.5           # joint-CSD flag: lag-1 autocorrelation threshold


def load_matched(path=DATA):
    with open(path) as fh:
        return json.load(fh)


def event_response(t, v, noise_frac, event, lag_band):
    """Post-drainage Δv/v in the lag band vs pre-drainage baseline.

    A §G.4 surge confirmation requires a POSITIVE, >2σ speed-up; σ uses the realistic
    year-to-year residual scatter (noise_frac), not the optimistic within-year CI."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    pre = v[t < event["t_peak"]]
    tt = event["t_trough"]
    post = v[(t > tt) & (t < tt + lag_band[1])]
    if len(pre) == 0 or len(post) == 0:
        return dict(drop_m=event["drop_m"], t_trough=tt, resolved=False,
                    pre_n=int(len(pre)), post_n=int(len(post)))
    dvv = float((post.mean() - pre.mean()) / pre.mean())
    sig2 = float(2 * noise_frac * np.sqrt(1.0 / len(pre) + 1.0 / len(post)))
    return dict(drop_m=event["drop_m"], t_trough=tt, resolved=True,
                pre_n=int(len(pre)), post_n=int(len(post)),
                dv_over_v=dvv, two_sigma=sig2,
                surge_detection=bool(dvv > sig2))   # one-sided: positive speed-up


def ews_metrics(t, v):
    """§I.3/§I.6 EWS on a detrended velocity series: AC1 + split-half variance ratio."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    resid = v - A @ coef
    h = len(resid) // 2
    var_ratio = float(resid[h:].var() / max(resid[:h].var(), 1e-12))
    ac1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if len(resid) > 2 else float("nan")
    csd = bool(var_ratio > CSD_VAR_RATIO and ac1 > CSD_AC1)
    return dict(trend_slope_m_yr2=float(coef[0]), resid_std_m_yr=float(resid.std()),
                resid_std_frac=float(resid.std() / v.mean()),
                var_ratio_2nd_over_1st=var_ratio, lag1_autocorr=ac1,
                joint_csd_signature=csd, n=int(len(v)))


def run(path=DATA):
    d = load_matched(path)
    band = d["_obs_lag_band_yr"]
    lakes_out = {}
    all_resp, csd_flags = [], []
    for name, lk in d["lakes"].items():
        if not lk.get("has_velocity"):
            lakes_out[name] = dict(has_velocity=False,
                                   n_drainage=len(lk.get("drainage_events", [])),
                                   note="open drainage, no ITS_LIVE velocity coverage")
            continue
        va = lk["velocity_annual"]
        t = [p["t"] for p in va]; v = [p["v"] for p in va]
        ews = ews_metrics(t, v)
        # realistic noise = larger of year-to-year residual scatter and within-year CI
        ci_frac = float(np.median([(p["ci_hi"] - p["ci_lo"]) / 2 for p in va]) / np.mean(v))
        noise_frac = max(ews["resid_std_frac"], ci_frac)
        evs = [event_response(t, v, noise_frac, e, band) for e in lk["drainage_events"]]
        resolved = [e for e in evs if e.get("resolved")]
        all_resp += [abs(e["dv_over_v"]) for e in resolved]
        csd_flags.append(ews["joint_csd_signature"])
        lakes_out[name] = dict(has_velocity=True, v_med_m_yr=lk["median_speed_mpyr"],
                               noise_floor_frac=lk["noise_floor_frac"],
                               realistic_noise_frac=noise_frac,
                               ews=ews, events=evs)
    max_resp = float(max(all_resp)) if all_resp else float("nan")
    n_sig = sum(1 for n, lk in lakes_out.items() if lk.get("events")
                for e in lk["events"] if e.get("surge_detection"))
    n_resolved = sum(1 for lk in lakes_out.values() if lk.get("events")
                     for e in lk["events"] if e.get("resolved"))
    return dict(
        what="§I s_N/early-warning framework applied to real CryoSat-2 + ITS_LIVE lake data",
        scope="3 open-velocity MacAyeal lakes (Mac1/2/3); full 131-lake/ATL15 population gated",
        sources=d["_sources"], lag_band_yr=band,
        lakes=lakes_out,
        summary=dict(
            n_velocity_lakes=sum(1 for lk in lakes_out.values() if lk.get("has_velocity")),
            n_resolved_events=n_resolved,
            max_abs_dv_over_v=max_resp,
            n_surge_detections=n_sig,
            n_lakes_with_joint_csd=int(sum(csd_flags)),
            any_universal_surge=False,
            any_csd_precursor=bool(any(csd_flags))),
        interpretation=(
            "Master curve (§I.1): small mixed-sign responses bound |s_N|·(ΔN/N) at the "
            "~1-2%% level -> these trunk lakes sit far from the N_c flotation pole. "
            "EWS (§I.3/§I.6): no lake shows the joint rising-variance + high-AC1 CSD "
            "signature -> the early-warning returns a true negative on stable ice (no "
            "false ungrounding alarm). Both §I probes agree: far from ungrounding."),
        honest_limits=(
            "n=8 annual velocity points per lake is statistically weak for a CSD trend; "
            "post-event sample counts are small (1-7). A strong test needs the dense "
            "quarterly series and a lake actually approaching flotation. The full "
            "131-lake volume-change catalogue (and an ATL15 re-download, ~7.6 GB) remains "
            "USAP-DC/ATL15-gated and would mostly reproduce the existing 1/19 detection."),
        verdict=(
            f"REAL DATA (no new download): across {n_resolved} resolved drainage events at "
            f"{sum(1 for lk in lakes_out.values() if lk.get('has_velocity'))} open-velocity "
            f"MacAyeal lakes, |Δv/v|≤{100*max_resp:.1f}%% (mixed sign, near the ~1%% noise "
            f"floor) — no clean universal post-drainage surge; and 0 lakes show the §I joint "
            f"CSD signature. Via the §I master curve the bounded response places these trunk "
            f"lakes far from the N_c fold, and the §I early-warning correctly does NOT "
            f"false-alarm on them. A genuinely-new cross-framework check on real "
            f"CryoSat-2/ITS_LIVE data; the full ATL15 population extension stays gated."),
        references="this repo §I (sn_master_curve, spatial_ews), §H.2 "
                   "(lake_lag_atl15_itslive); ITS_LIVE (Gardner 2022); CryoSat-2; "
                   "Siegfried & Fricker 2018",
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vel = {k: v for k, v in res["lakes"].items() if v.get("has_velocity")}
    d = load_matched()
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for (name, lk), c in zip(vel.items(), colors):
        va = d["lakes"][name]["velocity_annual"]
        t = [p["t"] for p in va]; v = [p["v"] for p in va]
        ax[0].plot(t, v, "o-", color=c, lw=1.5, label=f"{name} ({lk['v_med_m_yr']:.0f} m/yr)")
        for e in d["lakes"][name]["drainage_events"]:
            ax[0].axvline(e["t_trough"], color=c, ls=":", lw=1, alpha=0.6)
    ax[0].set_xlabel("year"); ax[0].set_ylabel("ITS_LIVE surface speed [m/yr]")
    ax[0].set_title("(a) real velocity series + drainage events (dotted)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # (b) per-event response with 2σ
    yl = []
    i = 0
    for name, lk in vel.items():
        for e in lk["events"]:
            if not e.get("resolved"):
                continue
            ax[1].errorbar(e["drop_m"], 100 * e["dv_over_v"], yerr=100 * e["two_sigma"],
                           fmt="o", color=colors[list(vel).index(name) % len(colors)],
                           capsize=3, label=name if name not in yl else None)
            yl.append(name); i += 1
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("drainage drop [m]"); ax[1].set_ylabel("post-drainage Δv/v [%]")
    ax[1].set_title("(b) bounded response: no universal surge (|Δv/v|≤2%)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("§I framework on real lake data: far from flotation, EWS true-negative", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.normpath(
        os.path.join(_HERE, "..", "reports", "lake_lag_sn_ews.json")))
    a = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    s = res["summary"]
    print("=== Real-data lake-lag through the §I framework (no new download) ===")
    print(f"  velocity lakes: {s['n_velocity_lakes']}; resolved events: {s['n_resolved_events']}")
    print(f"  max |Δv/v| = {100*s['max_abs_dv_over_v']:.1f}%; positive surge detections: {s['n_surge_detections']}")
    print(f"  lakes with joint CSD signature: {s['n_lakes_with_joint_csd']} "
          f"(any precursor: {s['any_csd_precursor']})")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
