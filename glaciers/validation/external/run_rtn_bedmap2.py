r"""Real-data §V.1 RTN run on open BAS Bedmap2 ice thickness (no auth).
 
Pipeline
--------
1. Load Bedmap2 ``thickness``, ``bed`` and the grounded/shelf ``icemask``
   (1 km grid, decimated by ``--stride`` km for tractability).
2. Build the RTN inputs *from real geometry*:
     - overburden       p_i      = rho_i g H
     - ocean head at bed p_ocean = rho_w g d_base ,  d_base = max(0, -bed)
       (depth of the bed below sea level -- the seawater column that can
       intrude at the grounding zone)
     - subglacial water p_w      = phi * p_i      (phi = fraction of overburden;
       N_eff = (1-phi) p_i is the effective pressure handed to the validator)
3. Restrict to **grounded** ice (icemask==0); shelves are already afloat.
4. Score the **falsifiable directional prediction**: does ``RTN>1`` concentrate
   within a few km of the grounding line?  We bin the grounded-ice ``RTN>1``
   fraction by distance to the nearest non-grounded (shelf/ocean) cell.
 
No systematic, gridded ocean-intrusion survey exists (per the §G.3 caveat), so
this is a *directional* spatial test, not a precision/recall score against a
catalogue.  The 1 km posting also cannot resolve the channel scale the
Roethlisberger term needs -- reported honestly, not papered over.
"""
from __future__ import annotations
 
import argparse
import os
import sys
 
import numpy as np
 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from validators.rtn_validator import rtn, classify, RHO_I, G  # noqa: E402
 
RHO_W = 1028.0  # seawater density [kg m^-3]
 
 
def build_rtn(H, bed, phi):
    """RTN over the grid for a given subglacial-water fraction ``phi``."""
    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    p_ocean = RHO_W * G * d_base         # gauge ocean head (relative to atmosphere)
    p_i = RHO_I * G * H
    N_eff = (1.0 - phi) * p_i            # => validator's p_w = p_i - N_eff = phi*p_i
    return rtn(H, p_ocean=p_ocean, N_eff=N_eff)  # gauge inputs => atmosphere cancels (default p_atm=0)
 
 
def distance_to_groundingline_km(grounded, cellsize_m):
    """EDT distance (km) from each grounded cell to the nearest non-grounded cell."""
    from scipy import ndimage
    d = ndimage.distance_transform_edt(grounded)  # cells; 0 outside grounded
    return d * (cellsize_m / 1000.0)


def _median(a):
    """``np.median`` that returns nan (no RuntimeWarning) for an empty selection."""
    return float(np.median(a)) if a.size else float("nan")
 
 
def run(bin_dir, stride=5, phi=0.9):
    from external.bedmap2_loader import load_fields
    d = load_fields(bin_dir, stride=stride)
    H = d["thickness"]; bed = d["bed"]
    mask = d["icemask_grounded_and_shelves"]
    cellsize = d["_meta"]["cellsize"]
 
    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)
    rtn_map = build_rtn(H, bed, phi)
    pred = classify(rtn_map) & grounded
 
    dist = distance_to_groundingline_km(grounded, cellsize)
    edges = [0, 5, 10, 25, 50, 100, 250, np.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = grounded & (dist >= lo) & (dist < hi)
        n = int(sel.sum())
        frac = float(pred[sel].mean()) if n else float("nan")
        rows.append((lo, hi, n, frac))
 
    ng = int(grounded.sum())
    summary = {
        "phi": phi,
        "grid": H.shape,
        "cellsize_km": cellsize / 1000.0,
        "n_grounded": ng,
        "rtn_gt1_frac_grounded": float(pred[grounded].mean()),
        "median_dist_pred": _median(dist[pred]),
        "median_dist_notpred": _median(dist[grounded & ~pred]),
        "bins": rows,
    }
    return summary, rtn_map, pred, grounded, dist
 
 
def make_figure(rtn_map, pred, grounded, summary, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
 
    show = np.full(grounded.shape, np.nan)
    show[grounded] = 0.0
    show[pred] = 1.0
    ax[0].imshow(show, cmap="coolwarm", interpolation="nearest")
    ax[0].set_title(f"Bedmap2 grounded ice: RTN>1 (red), phi={summary['phi']}\n"
                    f"{100*summary['rtn_gt1_frac_grounded']:.1f}% of grounded cells")
    ax[0].set_xticks([]); ax[0].set_yticks([])
 
    fr = [100 * b[3] for b in summary["bins"]]
    xs = np.arange(len(fr))
    ax[1].bar(xs, fr, color="#c0392b")
    labels = [f"{b[0]:g}-{'inf' if b[1] == np.inf else f'{b[1]:g}'}" for b in summary["bins"]]
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labels, rotation=45, ha="right")
    ax[1].set_xlabel("distance to grounding line [km]")
    ax[1].set_ylabel("RTN>1 fraction of grounded ice [%]")
    ax[1].set_title("Falsifiable test: RTN>1 vs distance to GL")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="/home/data_bedmap/bedmap2_bin")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--phi", type=float, default=0.9)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()
 
    print("=== §V.1 RTN on real BAS Bedmap2 ice thickness ===")
    for phi in sorted({0.8, 0.9, 0.95, args.phi}):
        s, *_ = run(args.bin_dir, stride=args.stride, phi=phi)
        print(f"\nphi={phi}: grid={s['grid']} @ {s['cellsize_km']:g} km, "
              f"grounded cells={s['n_grounded']}")
        print(f"  RTN>1 over grounded ice : {100*s['rtn_gt1_frac_grounded']:.1f}%")
        print(f"  median dist-to-GL: RTN>1={s['median_dist_pred']:.0f} km  "
              f"RTN<1={s['median_dist_notpred']:.0f} km")
        print("  RTN>1 fraction vs distance-to-GL:")
        for lo, hi, n, fr in s["bins"]:
            hh = "inf" if hi == np.inf else f"{hi:g}"
            print(f"    {lo:>4g}-{hh:>4} km : {100*fr:5.1f}%  (n={n})")
 
    s, rtn_map, pred, grounded, dist = run(args.bin_dir, stride=args.stride, phi=args.phi)
    fig = args.fig or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "reports", "rtn_bedmap2.png")
    make_figure(rtn_map, pred, grounded, s, fig)
    print(f"\nfigure -> {os.path.normpath(fig)}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
