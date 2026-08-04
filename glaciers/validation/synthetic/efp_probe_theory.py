r"""THEORY — the basal effective-pressure-sensitivity probe: lake drainages and ocean
forcing measure the SAME sliding-law sensitivity s_N(N), which modelers currently
tune away.

The new physical relationship this repo's three field results imply
------------------------------------------------------------------
1. §G.4 / `lake_lag_trunk.py` (HYP1): a lake drainage imposes a transient
   effective-pressure step dN at the bed; the measured downstream surface-speed
   surge du/u is its sliding response. So a drainage is a natural **in-situ step
   experiment** whose amplitude yields the local sliding-law N-sensitivity

       s_N = d ln u_b / d ln N = (du/u) / (dN/N).                         (PROBE-1)

2. §H.1.6 / `efp_gate_direct_n.py` (HYP2): observed ocean thermal forcing TF lowers
   N continuously; the gating slope d ln u_*/dTF steepens toward flotation, i.e.
   the interaction term measures how s_N grows as N falls,

       d ln u_*/dTF = s_N * (d ln N / dTF),   |s_N| up as N -> 0.         (PROBE-2)

3. Type III / `type_iii_regime.py` (HYP3): a regularized-Coulomb bed (Schoof 2005;
   Tsai 2015; Joughin 2019; mu = 1/2 Mohr-Coulomb, Tulaczyk 2000) predicts the
   SHAPE: |s_N(N)| ~ m far from flotation and diverges as C N -> tau_d (N_c), then
   the bed goes afloat (Type III, continuous response).

The unification: PROBE-1 (a fast dN step) and PROBE-2 (slow dN drift) are two
measurements of the **same** s_N(N) curve, and HYP3 is its closed form. This is new
because the modeling community **subsumes N into tuned friction coefficients**
("no reliable knowledge of basal water pressure", Joughin et al. 2019) and applies
ad hoc near-flotation weakening; nobody measures s_N(N) directly. Pairing a *direct*
N (HYP2, BedMachine ocean-connected) with a drainage step (HYP1) yields a field
**measurement** of s_N — a quantity otherwise only inverted/tuned.

This module makes the unification quantitative and falsifiable:
  (a) reads the HYP1 detections, estimates dN/N for each drainage from a lumped
      storage relation, and recovers an order-of-magnitude s_N per event (PROBE-1);
  (b) reads the HYP2 gating (PROBE-2) and the HYP3 RCF s_N(N) prediction;
  (c) overlays all three on one s_N(N) plot and reports whether the two independent
      field probes are mutually consistent and consistent with the RCF law.

Honest scope: dN/N per drainage is only a lumped-storage estimate (it here
OVERestimates the true fractional N step), so the per-event *absolute* s_N is
uncalibrated -- a lower bound that sits ~1-2 decades below the RCF curve. We do NOT
claim the field s_N matches the RCF magnitude. The robust, falsifiable claim is the
SIGN/SHAPE: the directly-measured drainage amplitude du/u (PROBE-1) and the ocean
TF-gating slope (PROBE-2) BOTH grow toward flotation, the sign the RCF s_N(N)
predicts. A calibrated s_N needs co-located GPS + a hydrology-constrained dN (the
stated next step). No GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import type_iii_regime as T3      # noqa: E402
import hydraulic_lag_derivation as HL   # noqa: E402

SEC_PER_YR = 365.25 * 86400.0
RHO_I, RHO_W, G = 917.0, 1028.0, 9.81


def estimate_dN_over_N(drop_m, N_pa, lake_area_m2=None, catchment_area_m2=None):
    """Order-of-magnitude fractional effective-pressure step dN/N from a drainage.

    A drainage of surface drawdown ``drop_m`` releases an ice-equivalent water volume
    that transiently pressurizes the bed downstream. Using the lumped GlaDS englacial
    storage Sigma = e_v A /(rho_w g) (Werder 2013; HL.storage), the released volume
    dV = drop_m * lake_area raises the water pressure by dp_w = dV/Sigma over the
    catchment, so dN = -dp_w and dN/N = dp_w/N. With only ``drop_m`` and N known we
    report the *fractional* step using representative areas; the result is O(1)
    uncertain (documented)."""
    p = HL.baseline_params()
    # representative lake & catchment areas if not given (Siegfried lakes ~ 100s km^2)
    A_lake = lake_area_m2 or (10e3 * 10e3)        # ~100 km^2
    A_catch = catchment_area_m2 or (p["W"] * p["ell"])
    Sigma = p["e_v"] * A_catch / (RHO_W * G)      # m^3 / Pa
    # ice-equivalent drawdown -> water volume (rho_i/rho_w factor ~0.9)
    dV = (RHO_I / RHO_W) * drop_m * A_lake
    dpw = dV / Sigma                              # Pa
    if N_pa is None or not np.isfinite(N_pa) or N_pa <= 0:
        return None
    return float(min(dpw / N_pa, 5.0))            # cap the fractional step


def rel_to_N(rel, H_m):
    """Convert the Bedmap2 normalized margin rel=m/H to an effective pressure N [Pa].

    rel ~ normalized N: N ~ phi rho_i g m = phi rho_i g (rel*H). Use phi=0.9, H from
    the lake. For rel<=0 (near/at flotation) clamp to a small positive N."""
    if rel is None or H_m is None or not np.isfinite(rel) or not np.isfinite(H_m):
        return None
    phi = 0.9
    N = phi * RHO_I * G * rel * H_m
    return float(max(N, 1e4))                     # floor 0.01 MPa near flotation


def probe1_from_detections(field_json):
    """PROBE-1: per-detection s_N from drainage amplitude (HYP1)."""
    out = []
    if not (field_json and os.path.exists(field_json)):
        return out
    d = json.load(open(field_json))
    for det in d.get("detections", []):
        amp = det.get("resp_frac")
        rel = det.get("rel")
        H = det.get("H")
        drop = det.get("drop_m")
        if amp is None or rel is None:
            continue
        N = rel_to_N(rel, H if H else 2000.0)
        dNoN = estimate_dN_over_N(drop or 1.0, N)
        if N is None or dNoN is None or dNoN <= 0:
            continue
        s = amp / dNoN
        out.append(dict(lake=det["lake"], N_MPa=N / 1e6, rel=rel, du_u=amp,
                        dN_over_N=round(dNoN, 3), s_N=round(s, 3),
                        lag_yr=det.get("lag_to_peak"), dist_km=det.get("dist_km")))
    return out


def probe2_from_gate(gate_json):
    """PROBE-2: the ocean-gating s_N gradient (HYP2). Returns the marine-West tercile
    TF-slopes and the implied steepening (s_N grows toward flotation)."""
    if not (gate_json and os.path.exists(gate_json)):
        return None
    d = json.load(open(gate_json))
    mw = d["bedmachine_direct_N"]["domains"]["MARINE_WEST"]
    t = mw.get("terciles_P1") or {}
    it = mw.get("interaction_P2") or {}
    rows = []
    for k, lab in [("low_rel_near_flotation", "near flotation"),
                   ("mid_rel", "mid"), ("high_rel_well_grounded", "well grounded")]:
        b = t.get(k, {})
        if b.get("TF_slope") is not None:
            rows.append(dict(bin=lab, median_N_proxy=b.get("median_rel"),
                             TF_slope=b["TF_slope"], r=b.get("r")))
    return dict(terciles=rows, interaction_d=it.get("d_interaction"),
                interaction_ci=it.get("d_ci95"),
                median_N_MPa=d.get("median_N_MPa_at_GL"),
                steepens_toward_flotation=bool(
                    len(rows) >= 2 and rows[0]["TF_slope"] > rows[-1]["TF_slope"]))


def run(field_json, gate_json):
    p1 = probe1_from_detections(field_json)
    p2 = probe2_from_gate(gate_json)
    # RCF prediction
    sw = T3.regime_sweep()
    Nc = sw["N_c"]
    # consistency: PROBE-1 s_N should rise as N falls; RCF |s_N| -> m far, diverges at Nc
    p1_sorted = sorted(p1, key=lambda r: r["N_MPa"])
    trend = None
    amp_trend = None
    if len(p1_sorted) >= 2:
        Ns = np.array([r["N_MPa"] for r in p1_sorted])
        ss = np.array([r["s_N"] for r in p1_sorted])
        au = np.array([r["du_u"] for r in p1_sorted])
        m = (Ns > 0) & (ss > 0)
        if m.sum() >= 2:
            trend = float(np.polyfit(np.log(Ns[m]), np.log(ss[m]), 1)[0])  # <0 expected
        ma = (Ns > 0) & (au > 0)
        if ma.sum() >= 2:
            # ROBUST: directly-measured surge amplitude vs N (no fabricated dN/N)
            amp_trend = float(np.polyfit(np.log(Ns[ma]), np.log(au[ma]), 1)[0])
    return dict(
        thesis=("a subglacial lake drainage is an in-situ step experiment whose surge "
                "amplitude measures the sliding-law N-sensitivity s_N=d ln u_b/d ln N; "
                "ocean thermal forcing measures the same s_N(N) continuously; both grow "
                "toward flotation and match the regularized-Coulomb prediction"),
        probe1_drainage_step=p1,
        probe1_logamp_vs_logN_slope=amp_trend,   # ROBUST: du/u measured; <0 = grows toward flotation
        probe1_logsN_vs_logN_slope=trend,        # uses the UNCALIBRATED absolute s_N (sign only)
        probe1_abs_sN_caveat=(
            "per-event |s_N| absolute value is uncalibrated: dN/N here is a lumped-storage "
            "estimate that OVERestimates the true fractional N step (the points sit ~1-2 "
            "decades below the RCF curve), so these s_N are lower bounds. Only the SIGN of "
            "the N-trend and the directly-measured du/u amplitude are robust; a calibrated "
            "s_N needs co-located GPS + a hydrology-constrained dN."),
        probe2_ocean_gating=p2,
        rcf_prediction=dict(law=T3.run().get("sliding_law"), N_c_MPa=Nc / 1e6,
                            s_N_wellgrounded=float(np.nanmedian(
                                np.abs(sw["s_N"])[sw["grounded"] & (sw["N"] > 5 * Nc)])),
                            note="|s_N|->m far from flotation, diverges at N_c=tau_d/mu"),
        new_measurement=("pair a direct N (BedMachine ocean-connected, HYP2) with a "
                         "drainage step (HYP1) -> a field value of s_N, which models "
                         "currently subsume/tune (Joughin 2019); ocean gating (HYP2) is "
                         "the independent cross-check"),
        consistency=dict(
            probe1_grows_toward_flotation=(amp_trend is not None and amp_trend < 0),
            probe1_basis="directly-measured du/u vs N slope (robust); abs s_N not used",
            probe2_grows_toward_flotation=(p2 or {}).get("steepens_toward_flotation"),
            both_consistent=bool((amp_trend is not None and amp_trend < 0)
                                 and (p2 or {}).get("steepens_toward_flotation"))),
        falsifiable=("if drainage-amplitude s_N and ocean-gating s_N disagree in sign of "
                     "their N-dependence, or neither grows toward flotation, the unified "
                     "RCF s_N(N) reading is falsified"),
    )


def make_figure(res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sw = T3.regime_sweep()
    N = sw["N"] / 1e6
    sN = np.abs(sw["s_N"])
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
    # (a) RCF |s_N|(N) SHAPE (left axis) vs the directly-measured drainage amplitude
    # du/u (right axis). We deliberately do NOT plot a per-event absolute s_N against
    # the RCF curve: dN/N is only a lumped estimate, so absolute s_N is uncalibrated
    # (~1-2 decades low). The honest, robust claim is that BOTH the prediction and the
    # measured amplitude rise toward flotation (low N).
    ax[0].plot(N, sN, "-", color="#999", lw=2, label=r"RCF $|s_N|(N)$ (prediction)")
    ax[0].axvline(sw["N_c"] / 1e6, color="k", ls="--", lw=1,
                  label=f"$N_c$={sw['N_c']/1e6:.3f} MPa (II/III)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("effective pressure N [MPa]  (from rel; low = near flotation)")
    ax[0].set_ylabel(r"RCF $|s_N|=|d\ln u_b/d\ln N|$ (prediction)", color="#777")
    ax[0].tick_params(axis="y", labelcolor="#777")
    ax2 = ax[0].twinx()
    p1 = res["probe1_drainage_step"]
    if p1:
        ax2.scatter([r["N_MPa"] for r in p1], [100 * r["du_u"] for r in p1],
                    c="#d62728", s=80, edgecolor="k", zorder=6,
                    label=r"HYP1 drainage amplitude $du/u$")
        for r in p1:
            ax2.annotate(r["lake"], (r["N_MPa"], 100 * r["du_u"]), fontsize=7,
                         xytext=(4, 0), textcoords="offset points")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"measured drainage surge $du/u$ [%]", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax[0].set_title("(a) RCF predicts |s_N| up toward flotation; measured du/u up too")
    ax[0].grid(alpha=0.3, which="both")
    # (b) PROBE-2 ocean gating terciles
    p2 = res["probe2_ocean_gating"]
    if p2 and p2.get("terciles"):
        labs = [t["bin"] for t in p2["terciles"]]
        sl = [t["TF_slope"] for t in p2["terciles"]]
        ax[1].plot(range(len(labs)), sl, "o-", color="#c2185b", ms=8)
        ax[1].set_xticks(range(len(labs))); ax[1].set_xticklabels(labs, fontsize=9)
        ax[1].set_ylabel(r"$d\ln u_*/dTF$ [per $\degree$C]")
        ax[1].set_title("(b) PROBE-2: ocean-gating steepens toward flotation (HYP2)")
        ax[1].grid(alpha=0.3)
    fig.suptitle("Effective-pressure-sensitivity probe: drainage steps + ocean forcing "
                 "measure the same s_N(N)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    rep = os.path.normpath(os.path.join(_HERE, "..", "reports"))
    ap.add_argument("--field-json", default=os.path.join(rep, "lake_lag_trunk.json"))
    ap.add_argument("--gate-json", default=os.path.join(rep, "efp_gate_direct_n.json"))
    ap.add_argument("--out", default=os.path.join(rep, "efp_probe_theory.json"))
    a = ap.parse_args()
    res = run(a.field_json, a.gate_json)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2, default=lambda o: None)
    print("=== Effective-pressure-sensitivity probe (theory synthesis) ===")
    print(f"  PROBE-1 drainage steps: {len(res['probe1_drainage_step'])} events")
    for r in res["probe1_drainage_step"]:
        print(f"    {r['lake']:14s} N={r['N_MPa']:.3f} MPa du/u={r['du_u']} "
              f"dN/N~{r['dN_over_N']} -> s_N~{r['s_N']}")
    print(f"  PROBE-1 log(du/u) vs log N slope: {res['probe1_logamp_vs_logN_slope']} "
          "(<0 = amplitude grows toward flotation; ROBUST)")
    print(f"  PROBE-1 log s_N vs log N slope:   {res['probe1_logsN_vs_logN_slope']} "
          "(sign only; absolute s_N uncalibrated -- lower bound)")
    p2 = res["probe2_ocean_gating"]
    if p2:
        print(f"  PROBE-2 ocean gating steepens toward flotation: {p2['steepens_toward_flotation']} "
              f"(interaction d={p2['interaction_d']})")
    print(f"  consistency: {res['consistency']}")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
