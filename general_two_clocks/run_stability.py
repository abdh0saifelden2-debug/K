"""Stability-resolved momentum-vs-heat transport analysis.

Tests, empirically, whether momentum and heat are transported by the same
turbulence on a single "clock" (the first-order / K-theory assumption) or
whether they decouple with atmospheric stability.

Usage:
    python run_stability.py --data-dir data --out-dir figures --report REPORT_STABILITY.md
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neon_pt import analysis as A
from neon_pt import stability as S
from neon_pt.loader import load_dataframe


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_stability_dist(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(np.clip(df["zeta"], -5, 5), bins=60, color="#4c72b0")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel(r"stability $\zeta=(z-d)/L$ (clipped to $\pm5$)")
    axes[0].set_ylabel("count (30-min periods)")
    axes[0].set_title("Distribution of atmospheric stability")
    axes[0].grid(alpha=0.3)

    comp = A.diurnal_composite(df.assign(zeta_c=np.clip(df["zeta"], -5, 5)), ["zeta_c"])
    axes[1].plot(comp.index, comp["zeta_c"], "-o", ms=3, color="#dd8452")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("local hour (PST)")
    axes[1].set_ylabel(r"mean $\zeta$")
    axes[1].set_title("Stability is unstable by day, stable by night")
    axes[1].grid(alpha=0.3)
    return _save(fig, out_dir, "06_stability_distribution.png")


def fig_decoupling(df, summary, out_dir):
    classes = list(summary.index)
    xpos = np.arange(len(classes))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: median transport efficiencies per stability class.
    width = 0.38
    axes[0].bar(xpos - width / 2, summary["median_r_uw"], width,
                color="#4c72b0", label=r"momentum  $r_{uw}$")
    axes[0].bar(xpos + width / 2, summary["median_abs_r_wT"], width,
                color="#c44e52", label=r"heat  $|r_{wT}|$")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(classes, rotation=20, ha="right")
    axes[0].set_ylabel("median transport efficiency")
    axes[0].set_title("Momentum and heat efficiencies move oppositely")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    # Right: the momentum/heat ratio per class (constant => single-clock K-theory).
    axes[1].plot(xpos, summary["transport_ratio"], "-o", color="#55a868", lw=2, ms=7)
    axes[1].axhline(summary["transport_ratio"].mean(), color="0.5", ls="--",
                    label="mean (a fixed-Prandtl closure assumes a flat line)")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(classes, rotation=20, ha="right")
    axes[1].set_ylabel(r"momentum/heat efficiency ratio")
    axes[1].set_title("The ratio is far from constant (~%.1fx swing)"
                      % (summary["transport_ratio"].max() / summary["transport_ratio"].min()))
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Momentum and heat do not ride one locked turbulence clock", fontsize=13)
    return _save(fig, out_dir, "07_transport_decoupling.png")


def fig_flux_variance(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    unst = df[df["zeta"] < 0]
    x = (-unst["zeta"]).clip(upper=10)
    axes[0].scatter(x, unst["sigma_w"] / unst["ustar_ms"], s=10, alpha=0.3, color="#55a868")
    xx = np.linspace(0.001, 10, 200)
    axes[0].plot(xx, 1.25 * (1 + 3 * xx) ** (1 / 3), "k-", lw=1.5,
                 label=r"MOST: $1.25(1+3|\zeta|)^{1/3}$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$-\zeta$ (unstable)")
    axes[0].set_ylabel(r"$\sigma_w/u_*$")
    axes[0].set_title("Momentum-side similarity (well behaved)")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    sub = df[(df["zeta"] < 0) & (df["Tstar"].abs() > 1e-3)]
    x2 = (-sub["zeta"]).clip(upper=10)
    axes[1].scatter(x2, sub["sigma_T"] / sub["Tstar"].abs(), s=10, alpha=0.3, color="#c44e52")
    axes[1].plot(xx, 2.0 * (1 + 9.5 * xx) ** (-1 / 3), "k-", lw=1.5,
                 label=r"MOST: $\sim2(1+9.5|\zeta|)^{-1/3}$")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 8)
    axes[1].set_xlabel(r"$-\zeta$ (unstable)")
    axes[1].set_ylabel(r"$\sigma_T/|T_*|$")
    axes[1].set_title("Scalar-side similarity (more scatter)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
    return _save(fig, out_dir, "08_flux_variance_similarity.png")


def write_report(path, meta, df, summary, fig_paths):
    near = summary.loc["near-neutral"]
    s_unstable = summary.loc["strongly unstable"]
    ratio_min = float(summary["transport_ratio"].min())
    ratio_max = float(summary["transport_ratio"].max())
    fold = ratio_max / ratio_min if ratio_min else float("nan")

    lines = []
    w = lines.append
    w("# Do momentum and heat ride the same turbulence? (K-theory, empirically)")
    w("")
    w(f"Site **{meta.site}**, January 2020, tower-top sonic at measurement height "
      f"{meta.meas_height_m:.1f} m, zero-plane displacement d = {meta.displacement_m:.1f} m, "
      f"so z-d = **{meta.z_minus_d:.1f} m**. {int(df['zeta'].notna().sum())} usable "
      "30-min periods after QC.")
    w("")
    w("A first-order / K-theory closure assumes momentum and heat are stirred by "
      "the same eddies with a near-constant turbulent Prandtl number, i.e. the "
      "momentum and heat **transport efficiencies should stay locked together**. "
      "With a single measurement level we cannot form K_M, K_H directly, so we "
      "test the locking via the w-u and w-T correlation coefficients "
      "(transport efficiencies) across stability classes.")
    w("")
    w("## Stability climatology")
    w("")
    w("Stability follows the expected daily rhythm - convective/unstable by day, "
      "stable by night (figure 6).")
    w("")
    w(f"![stability]({os.path.basename(fig_paths['dist'])})")
    w("")
    w("## Transport efficiencies by stability class")
    w("")
    w("| stability class | n | median $r_{uw}$ (momentum) | median $|r_{wT}|$ (heat) | momentum/heat ratio |")
    w("|---|---|---|---|---|")
    for cls, row in summary.iterrows():
        w(f"| {cls} | {int(row['n'])} | {row['median_r_uw']:.3f} | "
          f"{row['median_abs_r_wT']:.3f} | {row['transport_ratio']:.2f} |")
    w("")
    w(f"- Momentum efficiency r_uw peaks where there is shear but not runaway "
      f"convection - highest near-neutral/weakly-unstable "
      f"(~{near['median_r_uw']:.2f}) and lowest under strong free convection "
      f"(~{s_unstable['median_r_uw']:.2f}), where big thermals carry little "
      "shear stress.")
    w(f"- Heat efficiency |r_wT| does the opposite: it is **largest under strong "
      f"convection** (~{s_unstable['median_abs_r_wT']:.2f}) and smallest "
      f"near-neutral (~{near['median_abs_r_wT']:.2f}), where the heat flux itself "
      "passes through zero.")
    w(f"- The momentum/heat transport ratio is therefore **far from constant**: it "
      f"swings from ~{ratio_min:.2f} to ~{ratio_max:.2f} across regimes - a "
      f"**~{fold:.1f}x** range. A single fixed-Prandtl K-closure assumes this "
      "ratio is constant, so it cannot reproduce the observed swing.")
    w("")
    w(f"![decoupling]({os.path.basename(fig_paths['decoupling'])})")
    w("")
    w("## Flux-variance similarity")
    w("")
    w("On the momentum side, sigma_w/u* follows the textbook Monin-Obukhov curve "
      "closely; the scalar side (sigma_T/|T*|) is far more scattered, the usual "
      "fingerprint that scalar (heat) transport is influenced by non-local / "
      "larger-scale motions that momentum is not - i.e. they are not slaved to "
      "one local length scale.")
    w("")
    w(f"![similarity]({os.path.basename(fig_paths['similarity'])})")
    w("")
    w("## Verdict")
    w("")
    w("**Supported, and this is the part of the hypothesis that field data can "
      "actually speak to.** Momentum and heat are *not* transported on a single "
      "locked clock: their transport efficiencies diverge with stability, so a "
      "first-order closure with a constant turbulent Prandtl number is "
      "structurally unable to capture the decoupling. This is the empirically "
      "defensible core of the \"single-timescale K-theory is blind\" claim.")
    w("")
    w("**Out of scope for this data:** none of this tests the projection-method / "
      "wave-radiation-damping / Beale-Kato-Majda regularity arguments. Those are "
      "claims about the PDE structure and 3D fields and would need a simulation "
      "(DNS/LES) or spatially-resolved measurements, not a single-point tower.")
    w("")
    w("_Generated by `run_stability.py`._")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--report", default="REPORT_STABILITY.md")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, meta = load_dataframe(args.data_dir)
    df = A.apply_qc(df)
    df = S.derive(df, meta.z_minus_d)
    clean = S.clean_for_stats(df)
    print(f"{len(clean)} usable periods; z-d = {meta.z_minus_d:.1f} m")

    summary = S.stability_summary(clean)
    print(summary)

    fig_paths = {
        "dist": fig_stability_dist(clean, args.out_dir),
        "decoupling": fig_decoupling(clean, summary, args.out_dir),
        "similarity": fig_flux_variance(clean, args.out_dir),
    }
    write_report(args.report, meta, clean, summary, fig_paths)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
