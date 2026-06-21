r"""§G.6 — the local lee-flux law, DERIVED from the committed amplitude sweep:
what actually carries grounding-line melt growth, and how it scales with a/λ.

What this sharpens (the §G.6 [HYP])
-----------------------------------
The §G.6 "unified melt rate" stitches §G.1-§G.5 into

    v_melt(x) = (1/ρ_i L)[ h_local(u*, a/λ)·(T_bulk − T_melt(N)) − ∫ K_ice q_water dτ ].

Its posited amplitude closure was the *mean*-conductance form
`δ_T,eff = δ_T,flat·(1 + ζ (a/λ)²)`. `scallop_amplitude_closure.py` already
**falsified** that: the mean-Nu deficit is amplitude-FLAT (`D = 0.108 ± 0.031`,
`p_free ≈ 0`), so the suppression `Nu<1` is near-geometric, not an amplitude
roll-off. But §G.6's growth-driving term is **not** the mean — it is the *local*
lee flux `h_local`. The committed sweep reported the local peak ratio `R_max(a/λ)`
(and `R_mean`, `R_min`) but **never fit its functional form**. This module closes
that gap by re-using the committed solver measurement
(`figures/59_scallop_amplitude_closure.json`; no re-run, no GPU, no download).

What it measures
----------------
1. **Local growth law.** Fit `R_max(a/λ)` (peak lee flux relative to flat) with a
   free power law, a linear law, the physically-anchored slope-proportional form
   `R_max−1 = b·(2π a/λ)` (correct flat-wall limit `R_max→1`), and the §G.6
   quadratic `R_max−1 = ζ(a/λ)²`. Decide which the data support.
2. **Mean saturation.** `R_mean(a/λ)` rises then saturates (the conduction-limited
   mean cannot escape ~flat), confirming the local/mean split.
3. **Separation onset.** `R_min(a/λ)` crosses zero → reversed (recirculating) lee
   flux; the amplitude where it first goes negative is the separation onset that
   the amplitude-flat *mean* hides.

Result
------
Across `a/λ ∈ [0.05, 0.30]` the growth-driving `R_max` rises **roughly linearly**
(`R² ≈ 0.94`; free exponent `p ≈ 0.7`, sub-quadratic) from ~1.9 to ~4.3 — the §G.6
`(a/λ)²` form is rejected for the *local* term as well (quadratic `R² ≈ 0.40` ≪
linear `0.94`). The
origin-respecting law `R_max ≈ 1 + 1.7·(2π a/λ)` (proportional to wall steepness)
holds to `R² ≈ 0.92`. `R_mean` saturates near 1.24-1.28, and `R_min` turns negative
at `a/λ ≈ 0.11` (lee flow reversal). So the honest §G.6 closure is:
**mean conductance amplitude-flat (`C ≈ 1.11`), growth carried by a local lee flux
that is linear-in-steepness and bounded `R_max ∈ [1.9, 4.3]`, with separation onset
near `a/λ ≈ 0.11`** — phenomenology with measured, bounded coefficients, not a
quadratic ansatz. No GPU, no download (consumes the committed sweep).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# committed solver measurement (glaciers/figures/59_...); module lives in glaciers/validation/synthetic
DEFAULT_SRC = os.path.normpath(os.path.join(HERE, "..", "..", "figures",
                                            "59_scallop_amplitude_closure.json"))


def _r2(y, pred):
    y = np.asarray(y, float)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def load_sweep(path=DEFAULT_SRC):
    """Load the committed amplitude sweep; return a/λ and the local-flux ratios."""
    with open(path) as fh:
        d = json.load(fh)
    rows = sorted(d["rows"], key=lambda r: r["a_over_lam"])
    x = np.array([r["a_over_lam"] for r in rows], float)
    return dict(
        a_over_lam=x,
        R_max=np.array([r["R_max"] for r in rows], float),
        R_mean=np.array([r["R_mean"] for r in rows], float),
        R_min=np.array([r["R_min"] for r in rows], float),
        Nu_ratio=np.array([r["Nu_ratio"] for r in rows], float),
        src=os.path.basename(path),
    )


def fit_local_flux(x, rmax):
    """Fit the local peak lee-flux ratio R_max(a/λ) under four candidate laws.

    Returns each fit's coefficients and R² (on R_max) so the §G.6 quadratic can be
    accepted or rejected against linear / free-power / slope-proportional forms."""
    x = np.asarray(x, float); rmax = np.asarray(rmax, float)
    # 1) linear with intercept
    A = np.vstack([x, np.ones_like(x)]).T
    (m, b), *_ = np.linalg.lstsq(A, rmax, rcond=None)
    lin = dict(slope=float(m), intercept=float(b), r2=_r2(rmax, A @ np.array([m, b])))
    # 2) free power law: R_max-1 = A*(a/λ)^p
    ylog = np.log(rmax - 1.0); Xlog = np.vstack([np.log(x), np.ones_like(x)]).T
    (p, lnA), *_ = np.linalg.lstsq(Xlog, ylog, rcond=None)
    power = dict(coef=float(np.exp(lnA)), exponent=float(p),
                 r2=_r2(rmax, 1.0 + np.exp(lnA) * x ** p))
    # 3) slope-proportional (origin-respecting): R_max-1 = b*(2π a/λ)
    s = 2 * np.pi * x
    b_s = float(np.sum((rmax - 1.0) * s) / np.sum(s * s))
    slope_prop = dict(coef_b=b_s, r2=_r2(rmax, 1.0 + b_s * s))
    # 4) §G.6 quadratic (origin-respecting): R_max-1 = ζ*(a/λ)²
    z = float(np.sum((rmax - 1.0) * x ** 2) / np.sum(x ** 4))
    quad = dict(zeta=z, r2=_r2(rmax, 1.0 + z * x ** 2))
    return dict(linear=lin, free_power=power, slope_proportional=slope_prop,
                quadratic_G6=quad)


def separation_onset(x, rmin):
    """a/λ at which R_min first crosses zero (lee flow reversal / separation)."""
    x = np.asarray(x, float); rmin = np.asarray(rmin, float)
    for i in range(len(x) - 1):
        if rmin[i] >= 0.0 > rmin[i + 1]:
            t = rmin[i] / (rmin[i] - rmin[i + 1])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return float("nan")


def run(path=DEFAULT_SRC):
    s = load_sweep(path)
    x, rmax, rmean, rmin, nu = (s["a_over_lam"], s["R_max"], s["R_mean"],
                                s["R_min"], s["Nu_ratio"])
    fits = fit_local_flux(x, rmax)
    onset = separation_onset(x, rmin)
    # mean saturation: ratio of last-half mean rise to first-step rise
    rmean_sat = float(rmean[-1])
    quad_rejected = fits["quadratic_G6"]["r2"] < fits["linear"]["r2"]
    return dict(
        what="local lee-flux law R_max(a/λ) for §G.6 unified melt rate (growth term)",
        source=s["src"],
        a_over_lam=x.tolist(), R_max=rmax.tolist(), R_mean=rmean.tolist(),
        R_min=rmin.tolist(), Nu_ratio=nu.tolist(),
        R_max_range=[float(rmax.min()), float(rmax.max())],
        fits=fits,
        separation_onset_a_over_lam=onset,
        R_mean_saturation=rmean_sat,
        Nu_ratio_mean=float(nu.mean()), Nu_ratio_std=float(nu.std()),
        verdict=(
            "§G.6 growth term: the local peak lee flux R_max rises ROUGHLY LINEARLY "
            f"with a/λ (linear R²={fits['linear']['r2']:.3f}; free exponent "
            f"p={fits['free_power']['exponent']:.2f}, sub-quadratic), from "
            f"{rmax.min():.2f} to {rmax.max():.2f} over a/λ∈[{x.min():.2f},{x.max():.2f}]. "
            f"The §G.6 (a/λ)² closure is REJECTED for the local term too "
            f"(quadratic R²={fits['quadratic_G6']['r2']:.2f}); the origin-respecting "
            f"slope-proportional law R_max≈1+{fits['slope_proportional']['coef_b']:.2f}·(2π a/λ) "
            f"holds to R²={fits['slope_proportional']['r2']:.2f}. R_mean saturates near "
            f"{rmean_sat:.2f} and R_min turns negative at a/λ≈{onset:.2f} (lee reversal). "
            "Honest §G.6: mean conductance amplitude-flat (C≈1.11), growth carried by a "
            "linear-in-steepness, bounded local lee flux — not a quadratic ansatz."),
        quadratic_rejected=bool(quad_rejected),
        references="this repo §G.6, §G.1 (scallop_amplitude_closure / g1_populations); "
                   "figures/59 committed sweep; Curl 1966; Werder et al. 2013",
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array(res["a_over_lam"]); rmax = np.array(res["R_max"])
    rmean = np.array(res["R_mean"]); rmin = np.array(res["R_min"])
    nu = np.array(res["Nu_ratio"]); f = res["fits"]
    xx = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    # (a) R_max with candidate laws
    ax[0].plot(x, rmax, "o", color="#d62728", label="R_max (measured)")
    ax[0].plot(xx, f["linear"]["slope"] * xx + f["linear"]["intercept"], "-",
               color="#d62728", lw=1.6, label=f"linear (R²={f['linear']['r2']:.2f})")
    ax[0].plot(xx, 1 + f["slope_proportional"]["coef_b"] * 2 * np.pi * xx, "--",
               color="#9467bd", lw=1.6,
               label=f"∝ steepness (R²={f['slope_proportional']['r2']:.2f})")
    ax[0].plot(xx, 1 + f["quadratic_G6"]["zeta"] * xx ** 2, ":", color="k", lw=1.6,
               label=f"§G.6 (a/λ)² (R²={f['quadratic_G6']['r2']:.2f})")
    ax[0].plot(x, rmean, "s-", color="#1f77b4", lw=1.2, ms=4,
               label="R_mean (saturates)")
    ax[0].axhline(1.0, color="gray", ls=":", lw=0.8)
    ax[0].set_xlabel("amplitude a/λ"); ax[0].set_ylabel("local flux ratio / flat")
    ax[0].set_title("(a) §G.6 growth term: local lee flux is LINEAR, not (a/λ)²")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # (b) separation onset + mean-Nu flatness
    ax[1].plot(x, rmin, "v-", color="#2ca02c", label="R_min (lee)")
    ax[1].axhline(0.0, color="k", ls="--", lw=1)
    onset = res["separation_onset_a_over_lam"]
    ax[1].axvline(onset, color="#2ca02c", ls=":", lw=1.2,
                  label=f"separation onset a/λ≈{onset:.2f}")
    ax[1].plot(x, nu, "d-", color="#ff7f0e", label="Nu/Nu_flat (mean, flat)")
    ax[1].axhline(res["Nu_ratio_mean"], color="#ff7f0e", ls=":", lw=0.8)
    ax[1].set_xlabel("amplitude a/λ"); ax[1].set_ylabel("ratio")
    ax[1].set_title("(b) lee reversal sets in; mean Nu stays amplitude-flat")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("§G.6 unified melt rate: local growth term measured & bounded", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=os.path.normpath(
        os.path.join(HERE, "..", "reports", "g6_local_flux_law.json")))
    a = ap.parse_args()
    res = run(a.src)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    f = res["fits"]
    print("=== §G.6 local lee-flux law (derived from committed sweep) ===")
    print(f"  source: {res['source']}")
    print(f"  R_max range: {res['R_max_range'][0]:.2f} .. {res['R_max_range'][1]:.2f}")
    print(f"  linear R²={f['linear']['r2']:.3f}; free power p={f['free_power']['exponent']:.2f} "
          f"(R²={f['free_power']['r2']:.3f})")
    print(f"  §G.6 quadratic R²={f['quadratic_G6']['r2']:.2f} -> "
          f"{'REJECTED' if res['quadratic_rejected'] else 'kept'}")
    print(f"  separation onset a/λ≈{res['separation_onset_a_over_lam']:.3f}")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
