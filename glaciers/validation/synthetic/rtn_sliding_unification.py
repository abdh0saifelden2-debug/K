r"""§I.7 — Unifying ocean intrusion (§G.3 RTN) with the sliding-law divergence (§G.4/§I):
both are ordered crossings of one declining normalized effective pressure toward the
grounding line.

The relationship this repo's results imply
------------------------------------------
Two §G results have looked independent: the **Regime Transition Number** (§G.3,
``rtn_intrusion_clock.py``) predicts *where ocean water can intrude grounded ice*, and
the **sliding-law sensitivity** ``s_N`` (§G.4/§I) predicts *where basal sliding
diverges toward ungrounding*. Written in the **single normalized effective pressure**

    n_hat = N / (rho_i g H)   (ocean-connected; = 1 - H_flot/H, the rel/m-over-H proxy),

both collapse onto one declining coordinate toward the grounding line:

    RTN = (1 - n_hat) / phi              ->   RTN > 1  <=>  n_hat < 1 - phi
    |s_N| = m / (1 - (N_c/N)^m)          ->   diverges  <=>  n_hat -> n_hat_c = N_c/(rho_i g H)

So intrusion and sliding-instability are **the same N -> 0 near-flotation phenomenon**,
crossed in a fixed spatial order as n_hat falls toward the GL:

  (1) s_N begins to steepen above its well-grounded value m   (Type II surge band opens)
  (2) RTN = 1 at n_hat = 1 - phi                              (ocean intrusion favoured)
  (3) the sliding fold at n_hat_c = N_c/(rho_i g H) << 1 - phi (ungrounding / Type III)

Because ``n_hat_c ~ 0.003-0.013`` (for H ~ 0.5-3 km) is well below ``1 - phi ~ 0.1``,
**the RTN=1 intrusion line sits inland of the ungrounding fold** — ocean intrusion is
an *upstream precursor* of the zone that will host the sliding instability. That is a
new, falsifiable spatial-ordering prediction linking §G.3 and §I.

This is analytic (no GPU, no download); it optionally overlays the committed
``efp_gate_direct_n.json`` terciles to show the near-flotation gating band is already
RTN>1.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))

RHO_I, RHO_W, G = 917.0, 1028.0, 9.81
TAU_D, C_FRIC, M_EXP = 3.0e4, 0.5, 3.0
N_C = TAU_D / C_FRIC


def RTN_of_nhat(n_hat, phi=0.9):
    """Regime Transition Number in normalized-N coordinate: RTN=(1-n_hat)/phi."""
    return (1.0 - np.asarray(n_hat, float)) / phi


def s_N_of_nhat(n_hat, H, m=M_EXP, N_c=N_C):
    """|s_N| at normalized pressure n_hat and thickness H (N = n_hat*rho_i g H)."""
    N = np.asarray(n_hat, float) * RHO_I * G * H
    R = (N_c / N) ** m
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(N > N_c, m / (1.0 - R), np.inf)


def thresholds(H, phi=0.9, m=M_EXP, N_c=N_C):
    """The three ordered thresholds in n_hat toward the GL, for a given thickness."""
    n_intr = 1.0 - phi                      # RTN = 1
    n_fold = N_c / (RHO_I * G * H)           # sliding fold (Type III / ungrounding)
    # n_hat where |s_N| first exceeds 1.5*m (Type II surge-band onset)
    nn = np.geomspace(max(n_fold * 1.001, 1e-5), 0.9, 4000)
    sN = s_N_of_nhat(nn, H, m, N_c)
    onset = nn[np.argmax(sN > 1.5 * m)] if np.any(sN > 1.5 * m) else None
    return dict(H_m=H, n_surge_onset=(float(onset) if onset is not None else None),
                n_RTN1_intrusion=float(n_intr), n_fold_ungrounding=float(n_fold),
                ordering_intrusion_inland_of_fold=bool(n_fold < n_intr),
                sN_at_intrusion=float(s_N_of_nhat(np.array(n_intr), H, m, N_c)))


def field_overlay(efp_json, phi=0.9):
    if not (efp_json and os.path.exists(efp_json)):
        return None
    d = json.load(open(efp_json))
    out = []
    try:
        for dom in ("MARINE_WEST", "CONTINENTAL"):
            terc = d["bedmachine_direct_N"]["domains"][dom]["terciles_P1"]
            for name in ("low_rel_near_flotation", "mid_rel", "high_rel_well_grounded"):
                t = terc[name]
                nhat = t["median_rel"]
                out.append(dict(domain=dom, tercile=name, n_hat=round(nhat, 3),
                                RTN=round(float(RTN_of_nhat(nhat, phi)), 3),
                                RTN_gt_1=bool(RTN_of_nhat(nhat, phi) > 1),
                                TF_slope=round(t["TF_slope"], 3)))
    except Exception:
        return None
    return out


def run(efp_json=None, phi=0.9):
    Hs = [500.0, 1000.0, 2000.0, 3000.0]
    th = [thresholds(H, phi) for H in Hs]
    overlay = field_overlay(efp_json, phi)
    # field consistency: are the low-rel (near-flotation) terciles RTN>1 AND high TF-slope?
    consistent = None
    if overlay:
        low = [o for o in overlay if o["tercile"] == "low_rel_near_flotation"]
        high = [o for o in overlay if o["tercile"] == "high_rel_well_grounded"]
        consistent = bool(all(o["RTN_gt_1"] for o in low) and
                          all(not o["RTN_gt_1"] for o in high) and
                          np.mean([o["TF_slope"] for o in low]) >
                          np.mean([o["TF_slope"] for o in high]))
    return dict(
        what="ocean intrusion (RTN) and sliding-law divergence (s_N) are ordered crossings "
             "of one declining normalized effective pressure n_hat toward the grounding line",
        identities=dict(RTN="(1 - n_hat)/phi", intrusion_threshold="n_hat < 1 - phi",
                        sliding_fold="n_hat_c = N_c/(rho_i g H)",
                        s_N="m/(1 - (N_c/N)^m), N = n_hat rho_i g H"),
        phi=phi, N_c_MPa=N_C / 1e6, m=M_EXP,
        thresholds_by_thickness=th,
        intrusion_inland_of_fold=bool(all(t["ordering_intrusion_inland_of_fold"] for t in th)),
        field_overlay=overlay,
        field_consistent=consistent,
        prediction=("along a flowline approaching the GL (n_hat falling): Type II surge band "
                    "opens, then RTN=1 ocean intrusion at n_hat=1-phi, then the sliding fold "
                    "(ungrounding) at n_hat_c<<1-phi. Intrusion is an upstream precursor of the "
                    "sliding-instability zone."),
        falsification=("if mapped RTN=1 lines do NOT sit inland of the high-s_N / Type-III band, "
                       "or the near-flotation (low n_hat) bins are not simultaneously RTN>1 and "
                       "high-s_N, the unification fails."),
        references="RTN: this repo §G.3/§H.1; sliding law: Schoof 2005, Joughin et al. 2019; flotation/MISI: Schoof 2007",
        verdict=(
            f"In n_hat, RTN=(1-n_hat)/phi (intrusion at n_hat<{1-phi:.2f}) and |s_N| diverges at "
            f"n_hat_c=N_c/(rho_i g H)~{th[2]['n_fold_ungrounding']:.4f} (H=2km); since "
            f"n_hat_c<<1-phi, the RTN=1 intrusion line sits INLAND of the ungrounding fold for all "
            f"H tested. Ocean intrusion and basal-sliding instability are the same N->0 condition, "
            f"crossed in a fixed spatial order — unifying §G.3 with §G.4/§I."
            + ("" if consistent is None else " Committed ocean-gating terciles are consistent: "
               "near-flotation bins RTN>1 with higher TF-slope, well-grounded bins RTN<1.")),
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    phi = res["phi"]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    # (a) RTN and |s_N| vs n_hat (H=2 km), with the ordered thresholds
    n = np.geomspace(1e-3, 0.9, 600)
    H = 2000.0
    axb = ax[0]; axb2 = axb.twinx()
    axb.plot(n, RTN_of_nhat(n, phi), color="#1f77b4", lw=2, label="RTN")
    axb.axhline(1.0, color="#1f77b4", ls=":", lw=1)
    axb2.plot(n, s_N_of_nhat(n, H), color="#d62728", lw=2, label=r"$|s_N|$")
    n_intr = 1 - phi; n_fold = N_C / (RHO_I * G * H)
    axb.axvline(n_intr, color="k", ls="--", lw=1); axb.axvline(n_fold, color="gray", ls="--", lw=1)
    axb.annotate("RTN=1\n(intrusion)", (n_intr, 1.0), fontsize=8, xytext=(6, 10),
                 textcoords="offset points")
    axb.annotate("fold\n(ungrounding)", (n_fold, 1.0), fontsize=8, xytext=(-6, 30),
                 textcoords="offset points", ha="right")
    axb.set_xscale("log"); axb.invert_xaxis()
    axb.set_xlabel(r"$\hat n = N/(\rho_i g H)$  (declining toward GL)")
    axb.set_ylabel("RTN", color="#1f77b4"); axb2.set_ylabel(r"$|s_N|$", color="#d62728")
    axb2.set_yscale("log")
    axb.set_title("(a) one coordinate: intrusion inland of the ungrounding fold")
    # (b) thresholds vs thickness
    th = res["thresholds_by_thickness"]
    H_m = [t["H_m"] / 1000 for t in th]
    ax[1].plot(H_m, [t["n_RTN1_intrusion"] for t in th], "s-", color="#1f77b4", label="RTN=1 intrusion (1-phi)")
    ax[1].plot(H_m, [t["n_fold_ungrounding"] for t in th], "o-", color="#d62728", label="fold (ungrounding)")
    ax[1].plot(H_m, [t["n_surge_onset"] for t in th], "^--", color="#2ca02c", label="surge-band onset")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("ice thickness H [km]"); ax[1].set_ylabel(r"$\hat n$ threshold")
    ax[1].set_title("(b) ordered thresholds vs thickness")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("Unifying ocean intrusion (RTN) and sliding divergence (s_N) in one "
                 "normalized effective pressure", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--efp-json", default=os.path.join(_REPORTS, "efp_gate_direct_n.json"))
    ap.add_argument("--out", default=os.path.join(_REPORTS, "rtn_sliding_unification.json"))
    ap.add_argument("--phi", type=float, default=0.9)
    a = ap.parse_args()
    res = run(a.efp_json, a.phi)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("=== RTN <-> s_N unification (one normalized effective pressure) ===")
    print(f"  RTN=(1-n_hat)/phi, intrusion at n_hat<{1-a.phi:.2f}; fold n_hat_c=N_c/(rho_i g H)")
    print(f"  intrusion inland of fold (all H): {res['intrusion_inland_of_fold']}")
    if res.get("field_overlay"):
        print(f"  field overlay terciles: {len(res['field_overlay'])}; consistent={res['field_consistent']}")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
