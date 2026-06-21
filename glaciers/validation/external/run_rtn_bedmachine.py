r"""Real-data §V.1 RTN run on NSIDC BedMachine Antarctica ice thickness.

Independent-dataset companion to ``run_rtn_bedmap2.py``.  Bedmap2 (1 km, BAS,
open) already established the §H.1 directional result; BedMachine (500 m, NSIDC,
Earthdata-gated) is a *separate* thickness/bed product (Morlighem et al. 2020),
so reproducing ``RTN>1 concentrates near the grounding line`` on it is a genuine
robustness cross-check rather than a re-run of the same grid.

Pipeline (identical scoring to the Bedmap2 run)
-----------------------------------------------
1. Load BedMachine ``thickness``, ``bed`` and the integer ``mask``
   (500 m grid, optionally decimated by ``--stride``; stride=2 -> 1 km).
2. Build the RTN inputs from real geometry (same gauge convention as §V.1):
     - overburden       p_i      = rho_i g H
     - ocean head       p_ocean  = rho_w g d_base ,  d_base = max(0, -bed)
     - subglacial water p_w      = phi * p_i  (N_eff = (1-phi) p_i)
3. Restrict to **grounded** ice (BedMachine ``mask == 2``); shelves/ocean/land
   are the non-grounded class.
4. Score the falsifiable directional prediction: bin the grounded-ice ``RTN>1``
   fraction by distance to the nearest non-grounded cell (the grounding line).

Backend
-------
The native 500 m grid (13333x13333) makes the Euclidean distance transform the
memory bottleneck; it OOMs a small-RAM CPU box.  ``--gpu`` runs the RTN map and
the EDT on the GPU via CuPy / ``cupyx.scipy.ndimage`` (a 16 GB card handles full
resolution).  The GPU RTN formula is cross-checked against the canonical
``validators.rtn_validator.rtn`` at load time so the two backends agree.

Provenance (NASA Earthdata login, e.g. via ``earthaccess``)::

    pip install earthaccess netCDF4
    python -c "import earthaccess; earthaccess.login()"   # writes ~/.netrc
    # earthaccess.download(earthaccess.search_data(short_name='NSIDC-0756'), '.')
    #   -> BedMachineAntarctica-v3.nc  (~3.6 GB)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from validators.rtn_validator import rtn, classify, RHO_I, G, RTN_PW_EPS_PA  # noqa: E402

RHO_W = 1028.0  # seawater density [kg m^-3]


def build_rtn(H, bed, phi, xp=np, eps=RTN_PW_EPS_PA):
    """RTN over the grid for water fraction ``phi`` (backend-agnostic in ``xp``).

    Replicates ``validators.rtn_validator.rtn`` with gauge inputs (p_atm=0) so it
    runs on either ``numpy`` or ``cupy`` arrays; the numpy path is checked against
    the canonical validator in ``_assert_matches_validator``.
    """
    d_base = xp.where(xp.isfinite(bed), xp.maximum(0.0, -bed), xp.nan)
    p_ocean = RHO_W * G * d_base
    p_i = RHO_I * G * H
    p_w_raw = p_i - (1.0 - phi) * p_i          # = phi * p_i
    p_w = xp.where(p_w_raw > eps, p_w_raw, xp.nan)
    out = p_ocean / p_w
    return xp.where(xp.isnan(out), xp.inf, out)


def classify_xp(rtn_map, threshold=1.0):
    """Backend-agnostic equivalent of ``validators.rtn_validator.classify``.

    ``r > threshold`` is already ``False`` for ``NaN`` and ``True`` for ``+inf``
    under IEEE 754 on both NumPy and CuPy, so it reproduces ``classify`` (whose
    explicit ``NaN -> False`` ``where`` guard is redundant) without a NumPy-only
    op.  Using one function on both backends removes the CPU/GPU threshold-logic
    divergence; ``_assert_matches_validator`` pins it to the canonical
    ``classify`` at load time.
    """
    return rtn_map > threshold


def _assert_matches_validator(H, bed, phi):
    """Guard: the inline (xp-generic) formula must equal the canonical validator."""
    mine = build_rtn(H, bed, phi, xp=np)
    p_i = RHO_I * G * H
    ref = rtn(H, p_ocean=RHO_W * G * np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan),
              N_eff=(1.0 - phi) * p_i)
    finite = np.isfinite(mine) & np.isfinite(ref)
    assert np.allclose(mine[finite], ref[finite], rtol=1e-12, atol=0.0), \
        "build_rtn diverged from validators.rtn_validator.rtn"
    assert np.array_equal(np.isinf(mine), np.isinf(ref)), "RTN inf-mask mismatch"
    # the backend-agnostic classify must reproduce the canonical NumPy classify
    assert np.array_equal(classify_xp(mine), classify(mine)), \
        "classify_xp diverged from validators.rtn_validator.classify"


def distance_to_groundingline_km(grounded, cellsize_m, xp=np):
    """EDT distance (km) from each grounded cell to the nearest non-grounded cell."""
    if xp is np:
        from scipy import ndimage
        d = ndimage.distance_transform_edt(grounded)
    else:
        from cupyx.scipy import ndimage  # GPU EDT
        d = ndimage.distance_transform_edt(grounded)
    return d * (cellsize_m / 1000.0)


def run(path, stride=1, phi=0.9, use_gpu=False):
    from external.bedmachine_loader import load_fields, MASK_GROUNDED
    d = load_fields(path, fields=("thickness", "bed", "mask"), stride=stride)
    H = d["thickness"]; bed = d["bed"]; mask = d["mask"]
    cellsize = d["_meta"]["cellsize"]

    # Validate the GPU/inline formula against the canonical validator on a small
    # decimated slice (cheap, dataset-independent correctness guard).
    sl = (slice(None, None, max(1, H.shape[0] // 200)),
          slice(None, None, max(1, H.shape[1] // 200)))
    _assert_matches_validator(H[sl], bed[sl], phi)

    if use_gpu:
        import cupy as cp
        xp = cp
        H = cp.asarray(H); bed = cp.asarray(bed); mask = cp.asarray(mask)
    else:
        xp = np

    grounded = (mask == MASK_GROUNDED) & xp.isfinite(H) & (H > 0)
    rtn_map = build_rtn(H, bed, phi, xp=xp)
    # Single backend-agnostic classifier on both CPU and GPU (no path divergence);
    # classify_xp is pinned to the canonical validators.rtn_validator.classify in
    # _assert_matches_validator, so a future threshold change cannot silently make
    # the two backends disagree.
    pred = classify_xp(rtn_map) & grounded
    dist = distance_to_groundingline_km(grounded, cellsize, xp=xp)

    edges = [0, 5, 10, 25, 50, 100, 250, np.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = grounded & (dist >= lo) & (dist < (hi if np.isfinite(hi) else xp.inf))
        n = int(sel.sum())
        frac = float(pred[sel].mean()) if n else float("nan")
        rows.append((lo, hi, n, frac))

    ng = int(grounded.sum())
    summary = {
        "phi": phi,
        "grid": tuple(int(s) for s in H.shape),
        "cellsize_km": cellsize / 1000.0,
        "n_grounded": ng,
        "rtn_gt1_frac_grounded": float(pred[grounded].mean()) if ng else float("nan"),
        "median_dist_pred": float(_median(dist[pred], xp)),
        "median_dist_notpred": float(_median(dist[grounded & ~pred], xp)),
        "bins": rows,
        "backend": "gpu" if use_gpu else "cpu",
    }
    if use_gpu:
        rtn_map = xp.asnumpy(rtn_map); pred = xp.asnumpy(pred)
        grounded = xp.asnumpy(grounded); dist = xp.asnumpy(dist)
    return summary, rtn_map, pred, grounded, dist


def _median(a, xp):
    return xp.median(a) if a.size else xp.nan


def make_figure(rtn_map, pred, grounded, summary, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))

    show = np.full(grounded.shape, np.nan)
    show[grounded] = 0.0
    show[pred] = 1.0
    ax[0].imshow(show, cmap="coolwarm", interpolation="nearest")
    ax[0].set_title(f"BedMachine grounded ice: RTN>1 (red), phi={summary['phi']}\n"
                    f"{100*summary['rtn_gt1_frac_grounded']:.1f}% of grounded cells "
                    f"@ {summary['cellsize_km']:g} km")
    ax[0].set_xticks([]); ax[0].set_yticks([])

    fr = [100 * b[3] for b in summary["bins"]]
    xs = np.arange(len(fr))
    ax[1].bar(xs, fr, color="#c0392b")
    labels = [f"{b[0]:g}-{'inf' if b[1] == np.inf else f'{b[1]:g}'}" for b in summary["bins"]]
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labels, rotation=45, ha="right")
    ax[1].set_xlabel("distance to grounding line [km]")
    ax[1].set_ylabel("RTN>1 fraction of grounded ice [%]")
    ax[1].set_title("Falsifiable test: RTN>1 vs distance to GL (BedMachine)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    import datetime
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.expanduser("~/data_bedmachine/BedMachineAntarctica-v3.nc"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--phi", type=float, default=0.9)
    ap.add_argument("--gpu", action="store_true", help="run RTN map + EDT on GPU (CuPy)")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the phi-sweep summary to this JSON path "
                         "(schema mirrors reports/rtn_bedmap2.json)")
    args = ap.parse_args()

    print("=== §V.1 RTN on NSIDC BedMachine Antarctica ice thickness ===")
    phi_runs = {}
    for phi in sorted({0.8, 0.9, 0.95, args.phi}):
        s, *_ = run(args.path, stride=args.stride, phi=phi, use_gpu=args.gpu)
        phi_runs[f"{phi:g}"] = {
            **{k: v for k, v in s.items() if k != "bins"},
            "bins": [{"lo_km": lo, "hi_km": (None if hi == np.inf else hi),
                      "n": n, "rtn_gt1_frac": fr} for (lo, hi, n, fr) in s["bins"]],
        }
        print(f"\nphi={phi}: grid={s['grid']} @ {s['cellsize_km']:g} km [{s['backend']}], "
              f"grounded cells={s['n_grounded']}")
        print(f"  RTN>1 over grounded ice : {100*s['rtn_gt1_frac_grounded']:.1f}%")
        print(f"  median dist-to-GL: RTN>1={s['median_dist_pred']:.0f} km  "
              f"RTN<1={s['median_dist_notpred']:.0f} km")
        print("  RTN>1 fraction vs distance-to-GL:")
        for lo, hi, n, fr in s["bins"]:
            hh = "inf" if hi == np.inf else f"{hi:g}"
            print(f"    {lo:>4g}-{hh:>4} km : {100*fr:5.1f}%  (n={n})")

    s, rtn_map, pred, grounded, dist = run(args.path, stride=args.stride,
                                           phi=args.phi, use_gpu=args.gpu)
    fig = args.fig or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "reports", "rtn_bedmachine.png")
    make_figure(rtn_map, pred, grounded, s, fig)
    print(f"\nfigure -> {os.path.normpath(fig)}")

    if args.json_out:
        doc = {
            "dataset": "NSIDC-0756 BedMachine Antarctica v4 (thickness+bed+mask)",
            "test": ("\u00a7V.1 / \u00a7H.1 RTN directional: RTN>1 concentration vs "
                     "distance-to-grounding-line (independent-dataset cross-check of "
                     "the open-data Bedmap2 result)"),
            "provenance": ("NSIDC-0756 via NASA Earthdata (earthaccess "
                           "search_data(short_name='NSIDC-0756')); "
                           "BedMachineAntarctica V04.1.nc, 500 m native posting"),
            "stride": args.stride,
            "stride_km": args.stride * 0.5,
            "generated_utc": datetime.datetime.now(datetime.timezone.utc)
            .isoformat().replace("+00:00", "Z"),
            "by": "reproduction run",
            "backend": "gpu" if args.gpu else "cpu",
            "phi_runs": phi_runs,
        }
        with open(args.json_out, "w") as fh:
            json.dump(doc, fh, indent=1, default=float)
        print(f"json -> {os.path.normpath(args.json_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
