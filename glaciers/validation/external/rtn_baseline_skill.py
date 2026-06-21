r"""RTN baseline-skill test on open BAS Bedmap2 (no auth).

Question (referee, P4 sec3.4): does the Regime Transition Number ``RTN`` add any
information beyond the trivial predictors it is built from -- thickness-above-
flotation, bed-depth-below-sea-level, distance-to-grounding-line?

Algebra (P4 sec3.1).  With gauge ocean head ``p_ocean = rho_w g d_base`` and
``p_w = phi p_i = phi rho_i g H``,

    RTN = rho_w d_base / (phi rho_i H) .

Thickness above flotation is ``H_af = H - (rho_w/rho_i) d_base`` so the flotation
fraction ``f = H_af / H = 1 - (rho_w d_base)/(rho_i H) = 1 - phi*RTN``.  Hence

    RTN = (1 - f) / phi                          (exact, monotone-decreasing in f)
    RTN > 1  <=>  f < 1 - phi                     (a pure threshold on H_af/H).

So ``RTN`` is a strictly monotone function of the flotation fraction and ``RTN>1``
is *identical* to a thickness-above-flotation threshold.  This script confirms the
identity numerically on real geometry and quantifies the (zero) added skill over
``H_af/H`` and the (partial) overlap with distance-to-GL, so the manuscript can
state the baseline result with measured numbers instead of an assertion.

Run:
    python rtn_baseline_skill.py --bin-dir /home/K/_data_bedmap2/bedmap2_bin --stride 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from validators.rtn_validator import rtn, classify, RHO_I, G  # noqa: E402

RHO_W = 1028.0  # seawater density [kg m^-3] (matches run_rtn_bedmap2)


def build_rtn(H, bed, phi):
    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    p_ocean = RHO_W * G * d_base
    p_i = RHO_I * G * H
    N_eff = (1.0 - phi) * p_i
    return rtn(H, p_ocean=p_ocean, N_eff=N_eff)


def spearman(a, b, rng, cap=500_000):
    """Spearman rank correlation on a (sub)sample of paired finite values."""
    from scipy import stats
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size > cap:
        idx = rng.choice(a.size, size=cap, replace=False)
        a, b = a[idx], b[idx]
    if a.size < 3:
        return float("nan"), int(a.size)
    r = stats.spearmanr(a, b).statistic
    return float(r), int(a.size)


def run(bin_dir, stride=5, phi=0.9, seed=0):
    from external.bedmap2_loader import load_fields
    d = load_fields(bin_dir, stride=stride)
    H = d["thickness"]; bed = d["bed"]
    mask = d["icemask_grounded_and_shelves"]
    cellsize = d["_meta"]["cellsize"]
    rng = np.random.default_rng(seed)

    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0) & np.isfinite(bed)

    d_base = np.maximum(0.0, -bed)
    # thickness above flotation and flotation fraction
    H_af = H - (RHO_W / RHO_I) * d_base
    with np.errstate(invalid="ignore", divide="ignore"):
        f_flot = H_af / H               # = 1 - phi*RTN  (algebraic identity)

    rtn_map = build_rtn(H, bed, phi)
    pred_rtn = classify(rtn_map) & grounded               # {RTN > 1}
    pred_flot = (f_flot < (1.0 - phi)) & grounded          # {H_af/H < 1-phi}

    # distance to grounding line (km)
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(grounded) * (cellsize / 1000.0)

    g = grounded
    n_g = int(g.sum())

    # (1) set identity: RTN>1  ==  flotation threshold
    agree = float((pred_rtn[g] == pred_flot[g]).mean())
    n_rtn = int(pred_rtn[g].sum())
    n_flot = int(pred_flot[g].sum())
    n_xor = int((pred_rtn[g] ^ pred_flot[g]).sum())

    # (2) rank equivalence of RTN with each baseline predictor (grounded cells)
    rtn_g = np.where(g, rtn_map, np.nan)
    sp_flot, n1 = spearman(rtn_g, np.where(g, f_flot, np.nan), rng)
    sp_dist, n2 = spearman(rtn_g, np.where(g, dist, np.nan), rng)
    sp_dbase, n3 = spearman(rtn_g, np.where(g, d_base, np.nan), rng)
    sp_haf, n4 = spearman(rtn_g, np.where(g, H_af, np.nan), rng)

    return {
        "phi": phi,
        "stride_km": cellsize / 1000.0,
        "n_grounded": n_g,
        "n_rtn_gt1": n_rtn,
        "n_flotthresh": n_flot,
        "rtn_vs_flotthresh_agreement": agree,      # expect 1.0 exactly
        "n_disagree_cells": n_xor,
        "spearman_rtn_flotfrac": sp_flot,          # expect -1.0 exactly
        "spearman_rtn_thickaboveflot": sp_haf,     # expect -1.0 (monotone)
        "spearman_rtn_distGL": sp_dist,            # partial (not +/-1)
        "spearman_rtn_beddepth": sp_dbase,         # partial
        "sample_n": n1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="/home/K/_data_bedmap2/bedmap2_bin")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports", "rtn_baseline_skill.json"))
    args = ap.parse_args()

    print("=== RTN baseline-skill test on real BAS Bedmap2 ===")
    results = {}
    for phi in (0.8, 0.9, 0.95):
        s = run(args.bin_dir, stride=args.stride, phi=phi)
        results[f"phi_{phi}"] = s
        print(f"\nphi={phi}: grounded cells={s['n_grounded']} @ {s['stride_km']:g} km")
        print(f"  RTN>1 cells={s['n_rtn_gt1']}  flotation-threshold cells={s['n_flotthresh']}")
        print(f"  {{RTN>1}} == {{H_af/H < 1-phi}} agreement : {100*s['rtn_vs_flotthresh_agreement']:.4f}%"
              f"  (disagree {s['n_disagree_cells']} cells)")
        print(f"  Spearman(RTN, flotation fraction H_af/H) : {s['spearman_rtn_flotfrac']:+.4f}")
        print(f"  Spearman(RTN, thickness-above-flotation) : {s['spearman_rtn_thickaboveflot']:+.4f}")
        print(f"  Spearman(RTN, distance-to-GL)            : {s['spearman_rtn_distGL']:+.4f}")
        print(f"  Spearman(RTN, bed-depth below sea level) : {s['spearman_rtn_beddepth']:+.4f}")

    os.makedirs(os.path.dirname(os.path.normpath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {os.path.normpath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
