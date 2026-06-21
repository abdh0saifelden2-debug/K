r"""§H.1 follow-ups on real BAS Bedmap2: the RTN φ-calibration and the
critical-thinning (proximity-to-tipping) map.

Builds directly on the gauge-corrected RTN (§G.3) and the §H.1 directional run.
Two analyses, both constant-free (only ρ_w/ρ_i and the dimensionless φ enter):

**(#2) φ-calibration.**  RTN>1 ⇔ d_base/H > φ·ρ_i/ρ_w, so the *intruded area*
(fraction of grounded ice with RTN>1) is a monotone readout of the
continent-effective basal water fraction φ.  We sweep φ and report the intruded
area fraction + absolute km², and the sensitivity dA/dφ — i.e. how sharply a
mapped intrusion extent would invert for φ.

**(#3) critical-thinning / proximity-to-tipping.**  RTN>1 ⇔ H < H* where
``H* = (ρ_w / (φ ρ_i)) · d_base`` is the *local critical thickness*.  The
margin ``m = (H − H*) / H`` is the fraction of present thickness a grounded cell
must lose before intrusion is favoured (m≤0 ⇒ already intrudable).  Because φ<1,
``H* = H_flot/φ`` with ``H_flot = (ρ_w/ρ_i) d_base`` (the flotation thickness),
so the RTN=1 line sits *inland* of the classical flotation line by the
grounded-but-intrudable band ``H_flot < H < H*`` of width ∝(1−φ).  This is the
hydrology-corrected MISI margin.

Run (data auto-downloaded by the loader if absent, open BAS, no auth)::

    python validation/external/rtn_phi_calibration.py --stride 3

Outputs a 2-panel figure and a JSON summary under ``validation/reports/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from validators.rtn_validator import RHO_I  # noqa: E402
from external.run_rtn_bedmap2 import RHO_W, build_rtn, classify, \
    distance_to_groundingline_km  # noqa: E402


def analyse(bin_dir, stride=3, phis=None, phi_ref=0.9):
    from external.bedmap2_loader import load_fields
    if phis is None:
        phis = np.round(np.arange(0.70, 0.981, 0.02), 2)

    d = load_fields(bin_dir, stride=stride)
    H, bed, mask = d["thickness"], d["bed"], d["icemask_grounded_and_shelves"]
    cellsize_km = d["_meta"]["cellsize"] / 1000.0
    cell_area = cellsize_km ** 2  # km^2 per cell

    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)
    ng = int(grounded.sum())
    dist = distance_to_groundingline_km(grounded, d["_meta"]["cellsize"])

    # (#2) phi-calibration: intruded area vs phi
    curve = []
    for phi in phis:
        pred = classify(build_rtn(H, bed, phi)) & grounded
        n = int(pred.sum())
        curve.append({
            "phi": float(phi),
            "frac": n / ng,
            "area_km2": n * cell_area,
            "median_dist_km": float(np.median(dist[pred])) if n else float("nan"),
        })
    fr = np.array([c["frac"] for c in curve])
    # local sensitivity dA/dphi (per 0.01 in phi) by central difference
    dfdphi = np.gradient(fr, np.array(phis))
    for c, s in zip(curve, dfdphi):
        c["dfrac_dphi_per_0p01"] = float(s * 0.01)

    # (#3) critical-thinning map at phi_ref
    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    H_flot = (RHO_W / RHO_I) * d_base                  # flotation thickness
    H_star = H_flot / phi_ref                          # RTN=1 critical thickness
    with np.errstate(invalid="ignore", divide="ignore"):
        margin = (H - H_star) / H                       # thinning fraction to tip
    margin_g = np.where(grounded, margin, np.nan)

    # grounded-but-intrudable band: grounded, not floating (H>H_flot) but H<H_star
    band = grounded & (H > H_flot) & (H < H_star)
    nb = int(band.sum())
    summary = {
        "stride_km": cellsize_km,
        "n_grounded": ng,
        "description": (
            "Two constant-free consequences of the gauge RTN (§G.3/§H.1.1) on "
            "real Bedmap2: (#2) RTN>1 intruded area is a monotone, sensitive "
            "inverse for the basal water fraction phi; (#3) H*=(rho_w/phi rho_i) "
            "d_base is a local critical thickness -> a hydrology-corrected MISI "
            "margin (grounded-but-intrudable band, inland of the flotation line)."
        ),
        "phi_ref": phi_ref,
        "rho_ratio_w_i": RHO_W / RHO_I,
        "calibration": curve,
        "sensitivity_note": "dfrac_dphi_per_0p01 = change in RTN>1 area fraction per +0.01 in phi",
        "critical_thinning": {
            "band_cells": nb,
            "band_area_km2": nb * cell_area,
            "band_frac_grounded": nb / ng,
            "band_median_dist_km": float(np.median(dist[band])) if nb else float("nan"),
            "band_p90_dist_km": float(np.percentile(dist[band], 90)) if nb else float("nan"),
            "median_margin_grounded": float(np.nanmedian(margin_g)),
            "frac_within_10pct_of_tipping":
                float((grounded & (margin > 0) & (margin < 0.10)).sum() / ng),
        },
    }
    return summary, (H, grounded, dist, margin_g, band, phis, fr, dfdphi, phi_ref)


def make_figure(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, grounded, dist, margin_g, band, phis, fr, dfdphi, phi_ref = payload

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # left: phi-calibration curve + sensitivity
    axL.plot(phis, 100 * fr, "o-", color="#1f77b4", label="RTN>1 area")
    axL.axvline(phi_ref, ls=":", color="grey")
    axL.set_xlabel("subglacial water fraction  φ")
    axL.set_ylabel("RTN>1 area  [% of grounded ice]", color="#1f77b4")
    axL.tick_params(axis="y", labelcolor="#1f77b4")
    axL.invert_xaxis()  # lower phi (wetter bed) -> more intrusion, to the right
    ax2 = axL.twinx()
    ax2.plot(phis, -100 * dfdphi * 0.01, "s--", color="#d62728", alpha=0.7)
    ax2.set_ylabel("sensitivity  |dA/dφ|  [%-area per +0.01 φ]", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    axL.set_title("(#2) φ-calibration: intruded area inverts for φ")

    # right: proximity-to-tipping margin map (grounded ice), band highlighted
    m = np.ma.masked_invalid(margin_g)
    im = axR.imshow(np.clip(m, -0.2, 0.6), cmap="RdYlBu", origin="upper",
                    vmin=-0.2, vmax=0.6)
    axR.contour(band.astype(float), levels=[0.5], colors="k", linewidths=0.3)
    axR.set_title(f"(#3) thinning margin (H−H*)/H  at φ={phi_ref}\n"
                  "blue→red = closer to RTN=1 tipping; black = pre-flotation band")
    axR.set_xticks([]); axR.set_yticks([])
    fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="thinning margin")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin-dir", default="/home/data_bedmap/bedmap2_bin")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--phi-ref", type=float, default=0.9)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    reports = os.path.normpath(os.path.join(here, "..", "reports"))
    figdir = os.path.normpath(os.path.join(here, "..", "..", "figures"))
    fig = args.fig or os.path.join(reports, "rtn_phi_calibration.png")
    js = args.json or os.path.join(figdir, "57_rtn_phi_calibration.json")

    summary, payload = analyse(args.bin_dir, stride=args.stride, phi_ref=args.phi_ref)

    print("=== §H.1 #2 φ-calibration (RTN>1 area vs φ) ===")
    for c in summary["calibration"]:
        print(f"  φ={c['phi']:.2f}: {100*c['frac']:5.2f}% "
              f"({c['area_km2']:8.0f} km²)  med dist-to-GL={c['median_dist_km']:.1f} km")
    ct = summary["critical_thinning"]
    print("\n=== §H.1 #3 critical-thinning / pre-flotation band "
          f"(φ={summary['phi_ref']}) ===")
    print(f"  grounded-but-intrudable band: {ct['band_cells']} cells "
          f"({ct['band_area_km2']:.0f} km², {100*ct['band_frac_grounded']:.2f}% of grounded)")
    print(f"  band distance-to-GL: median={ct['band_median_dist_km']:.1f} km  "
          f"p90={ct['band_p90_dist_km']:.1f} km")
    print(f"  median thinning margin (grounded) = {100*ct['median_margin_grounded']:.0f}%")
    print(f"  grounded ice within 10% thinning of tipping: "
          f"{100*ct['frac_within_10pct_of_tipping']:.2f}%")

    make_figure(payload, fig)
    with open(js, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"json   -> {js}")


if __name__ == "__main__":
    main()
