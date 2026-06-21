r"""§H.1.1b — does a *data-derived* variable connectivity fraction phi(x,y) make the
Regime Transition Number reorder intrusion-favourability on **real Bedmap2**?

Context
-------
``rtn_baseline_skill.py`` proved (478,767 grounded Bedmap2 cells) that with a
*constant* phi, ``RTN`` is an exact monotone function of the flotation fraction
(Spearman -1.000, ``{RTN>1}`` identical to thickness-above-flotation, 0 disagreeing
cells).  ``synthetic/rtn_variable_phi_skill.py`` shows the *only* way RTN can add
skill is a spatially-varying, data-derived phi(x,y) carrying genuine basal-water
**connectivity** information, and that it then does (modestly, concentrated in the
near-flotation band).

This script supplies a real, calibration-free connectivity field from geometry
alone -- the **subglacial hydraulic-potential drainage network** -- and measures
how much RTN reorders on the actual continent.  No ground-truth intrusion survey
exists, so (as the manuscript states) this is a **reordering + physical-plausibility**
test, not precision/recall.

Method (Shreve 1972 hydraulic routing, no external data beyond Bedmap2)
-----------------------------------------------------------------------
1. Subglacial hydraulic potential (water pressure = ice overburden approximation):
       Phi = rho_w g z_bed + rho_i g H      (z_bed = bed elevation; H = thickness)
   Water flows down-gradient of ``Phi``.
2. Priority-Flood depression fill (Barnes et al. 2014) within the grounded mask,
   with the grounding line / ice margin as outlets; the flood naturally yields a
   flow-receiver for every cell (a drainage forest rooted at the outlets).
3. D8 flow accumulation up that forest -> upstream flux ``A`` (organized drainage
   axes carry high flux).
4. Connectivity -> phi map (decreasing): well-connected/channelized axes hold a
   **low** steady water-pressure fraction (small phi, large effective pressure),
   so phi(x,y) = PHI_HI - (PHI_HI-PHI_LO)*norm(log A).  The *magnitude* of this map
   is a modelling choice; the **degeneracy break** (RTN no longer a function of the
   flotation fraction alone) is sign-independent and is the robust deliverable.
5. Optionally ingest a real radar bed-**specularity** / basal-water field as phi (the
   gold-standard observable; ``--specularity-npy``), raising ``DataUnavailableError``
   with provisioning hints if absent rather than fabricating it.

Outputs (``validation/reports/rtn_variable_phi_real.json`` + figure): the constant-phi
anchor (reproduces -1.000 / 0-disagree), then for phi_routed the Spearman vs the
flotation fraction, the number and location (distance-to-GL) of cells whose
intrusion classification *changes*, and the concentration of those changes.

Run:
    python rtn_variable_phi_real.py --bin-dir /home/K/_data_bedmap2/bedmap2_bin --stride 10
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external import DataUnavailableError  # noqa: E402
from external.run_rtn_bedmap2 import build_rtn, classify, RHO_W, RHO_I  # noqa: E402
from validators.rtn_validator import G  # noqa: E402

PHI_LO = 0.80   # well-connected / channelized axis: low water-pressure fraction
PHI_HI = 0.97   # poorly connected / distributed: high water-pressure fraction


def shreve_potential(bed, H):
    r"""Subglacial hydraulic potential Phi = rho_w g z_bed + rho_i g H  [Pa]."""
    return RHO_W * G * bed + RHO_I * G * H


def priority_flood_route(phi, valid):
    r"""Priority-Flood (Barnes et al. 2014) fill + flow-receiver on potential ``phi``.

    ``valid`` is the boolean domain mask (grounded cells).  Cells outside ``valid``
    (shelf / ocean / NaN / grid edge) are **outlets**: a valid cell adjacent to a
    non-valid cell (or the grid edge) spills there.  Returns ``(filled, receiver)``
    where ``receiver[i]`` is the flat index the cell first drains to (or -1 for an
    outlet-adjacent spill / outside the domain), and a pop-order array for O(n)
    accumulation.
    """
    nr, nc = phi.shape
    n = nr * nc
    flat_phi = phi.ravel()
    valid_flat = valid.ravel()
    filled = flat_phi.copy()
    receiver = np.full(n, -1, dtype=np.int64)
    processed = np.zeros(n, dtype=bool)
    pop_order = np.empty(n, dtype=np.int64)
    npop = 0

    heap: list = []
    # seed: every valid cell that touches the domain boundary (non-valid neighbour
    # or grid edge) is an outlet seed at its own potential.
    def nbrs(idx):
        r, c = divmod(idx, nc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < nr and 0 <= cc < nc:
                    yield rr * nc + cc
                else:
                    yield -1  # off-grid => outlet

    for idx in np.flatnonzero(valid_flat):
        is_edge = False
        r, c = divmod(int(idx), nc)
        if r == 0 or r == nr - 1 or c == 0 or c == nc - 1:
            is_edge = True
        else:
            for nb in nbrs(int(idx)):
                if nb == -1 or not valid_flat[nb]:
                    is_edge = True
                    break
        if is_edge:
            heapq.heappush(heap, (float(filled[idx]), int(idx)))
            processed[idx] = True

    while heap:
        elev, c = heapq.heappop(heap)
        pop_order[npop] = c
        npop += 1
        for nb in nbrs(c):
            if nb == -1 or processed[nb] or not valid_flat[nb]:
                continue
            if filled[nb] < elev:
                filled[nb] = elev          # raise to spill level (fill depression)
            receiver[nb] = c               # first reached-from = downstream receiver
            processed[nb] = True
            heapq.heappush(heap, (float(filled[nb]), nb))

    return filled, receiver, pop_order[:npop]


def flow_accumulation(receiver, pop_order, valid):
    r"""Upstream flux: each valid cell contributes unit area to all downstream cells.

    Accumulate in *reverse* pop order (downstream cells popped first), pushing each
    cell's accumulated area onto its receiver.
    """
    n = receiver.size
    acc = np.where(valid.ravel(), 1.0, 0.0)
    for c in pop_order[::-1]:
        rcv = receiver[c]
        if rcv >= 0:
            acc[rcv] += acc[c]
    return acc


def phi_from_flux(acc, valid):
    r"""Decreasing connectivity->phi map from log upstream flux, normalised over the
    grounded domain to [0,1] then mapped to [PHI_LO, PHI_HI] (high flux -> low phi)."""
    a = np.full(acc.shape, np.nan)
    g = valid.ravel()
    la = np.log10(acc[g] + 1.0)
    lo, hi = np.percentile(la, 2), np.percentile(la, 98)
    norm = np.clip((la - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    a[g] = PHI_HI - (PHI_HI - PHI_LO) * norm
    return a.reshape(valid.shape)


def spearman(a, b, rng, cap=500_000):
    from scipy import stats
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size > cap:
        idx = rng.choice(a.size, size=cap, replace=False)
        a, b = a[idx], b[idx]
    if a.size < 3:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def run(bin_dir, stride=10, phi_const=0.9, specularity_npy=None, seed=0):
    from external.bedmap2_loader import load_fields
    d = load_fields(bin_dir, fields=("thickness", "bed", "surface",
                                     "icemask_grounded_and_shelves"), stride=stride)
    H = d["thickness"]; bed = d["bed"]; mask = d["icemask_grounded_and_shelves"]
    cellsize = d["_meta"]["cellsize"]
    rng = np.random.default_rng(seed)

    grounded = (np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0) & np.isfinite(bed))
    n_g = int(grounded.sum())

    d_base = np.maximum(0.0, -bed)
    H_af = H - (RHO_W / RHO_I) * d_base
    with np.errstate(invalid="ignore", divide="ignore"):
        h_af_frac = H_af / H                      # committed 'flotation fraction'

    from scipy import ndimage
    dist = ndimage.distance_transform_edt(grounded) * (cellsize / 1000.0)  # km

    # ---- constant-phi anchor (reproduce the committed baseline) ----
    rtn_c = build_rtn(H, bed, phi_const)
    pred_c = classify(rtn_c) & grounded
    pred_flot = (h_af_frac < (1.0 - phi_const)) & grounded
    sp_c = spearman(np.where(grounded, rtn_c, np.nan), np.where(grounded, h_af_frac, np.nan), rng)
    disagree_c = int(np.sum((pred_c ^ pred_flot) & grounded))

    # ---- variable phi from hydraulic routing (geometry only) ----
    Phi = shreve_potential(bed, H)
    Phi_masked = np.where(grounded, Phi, np.nan)
    filled, receiver, pop_order = priority_flood_route(
        np.where(grounded, Phi, np.inf), grounded)
    acc = flow_accumulation(receiver, pop_order, grounded)
    phi_routed = phi_from_flux(acc, grounded)

    rtn_v = build_rtn(H, bed, phi_routed)
    pred_v = classify(rtn_v) & grounded
    sp_v = spearman(np.where(grounded, rtn_v, np.nan), np.where(grounded, h_af_frac, np.nan), rng)
    # GLOBAL Spearman is dominated by the huge inland mass of f=0 cells (bed above sea
    # level => d_base=0 => RTN=0 regardless of phi), so it stays near -1. The reordering
    # only *can* happen among cells with appreciable ocean ratio; restrict there.
    band = grounded & (h_af_frac < 0.7)            # f = 1-H_af/H > 0.3: near-flotation
    sp_v_band = spearman(np.where(band, rtn_v, np.nan), np.where(band, h_af_frac, np.nan), rng)
    sp_c_band = spearman(np.where(band, rtn_c, np.nan), np.where(band, h_af_frac, np.nan), rng)
    # collinearity of routing-phi with the flotation fraction (the reason the break is weak):
    # both the drainage flux and the flotation fraction peak at the grounding line.
    corr_phi_f = spearman(np.where(band, phi_routed, np.nan), np.where(band, h_af_frac, np.nan), rng)
    n_band = int(band.sum())
    # cells whose intrusion classification CHANGES vs the flotation baseline
    changed = (pred_v ^ pred_flot) & grounded
    n_changed = int(changed.sum())
    newly_on = int(np.sum(pred_v & ~pred_flot & grounded))   # flagged by phi-RTN, not flotation
    newly_off = int(np.sum(~pred_v & pred_flot & grounded))
    # where do the changes sit? distance-to-GL of changed cells vs all grounded
    med_dist_changed = float(np.median(dist[changed])) if n_changed else float("nan")
    med_dist_all = float(np.median(dist[grounded]))
    frac_changed_within_10km = (float(np.mean(dist[changed] <= 10.0)) if n_changed else float("nan"))

    out_head = {
        "what": "real-Bedmap2 reordering of RTN by a hydraulic-routing connectivity phi(x,y)",
        "stride_km": cellsize / 1000.0, "n_grounded": n_g, "phi_const": phi_const,
        "phi_map": {"PHI_LO": PHI_LO, "PHI_HI": PHI_HI, "source": "Shreve hydraulic-potential D8 flow accumulation"},
        "anchor_constant_phi": {
            "spearman_rtn_flotfrac": sp_c, "rtn_gt1_cells": int((pred_c & grounded).sum()),
            "flotthresh_cells": int((pred_flot & grounded).sum()),
            "disagree_cells": disagree_c,
            "note": "reproduces committed baseline: Spearman -1, identical to thickness-above-flotation"},
        "variable_phi_routed": {
            "spearman_rtn_flotfrac_global": sp_v,
            "spearman_rtn_flotfrac_nearflot_band": sp_v_band,
            "spearman_rtn_flotfrac_nearflot_band_constphi": sp_c_band,
            "n_nearflot_band": n_band,
            "spearman_phiRouted_vs_flotfrac_band": corr_phi_f,
            "degeneracy_broken_global": bool(abs(sp_v) < 0.999),
            "degeneracy_broken_in_band": bool(abs(sp_v_band) < 0.999),
            "rtn_gt1_cells": int((pred_v & grounded).sum()),
            "n_classification_changed": n_changed,
            "newly_flagged_by_phiRTN": newly_on, "dropped_vs_flotation": newly_off,
            "median_distGL_changed_km": med_dist_changed,
            "median_distGL_all_grounded_km": med_dist_all,
            "frac_changed_within_10km_of_GL": frac_changed_within_10km},
    }
    collinear = abs(corr_phi_f) > 0.3
    out_verdict = (
        "constant-phi: Spearman(RTN, H_af/H)=%.4f, %d disagreeing cells (identical to "
        "thickness-above-flotation, as committed). hydraulic-routing phi(x,y): the GLOBAL "
        "Spearman stays %.4f (the continent is dominated by inland f=0 cells where RTN=0 for "
        "any phi), but in the near-flotation band it moves to %.4f (vs %.4f at constant phi); "
        "there phi_routed is %s the flotation fraction (Spearman %.3f), so it %s the flotation "
        "signal -- the intrusion class CHANGES on %d cells (%d newly flagged, %d dropped), "
        "concentrated at the grounding line (median %.0f vs %.0f km). A geometry-only "
        "connectivity field already turns RTN into a different screen from plain flotation in "
        "the band that matters; whether the reordering is *correct* is the open test, needing a "
        "radar bed-specularity / basal-water phi (--specularity-npy) or a gridded intrusion "
        "survey to score against."
        % (sp_c, disagree_c, sp_v, sp_v_band, sp_c_band,
           ("collinear with" if collinear else "nearly independent of"), corr_phi_f,
           ("mostly amplifies" if collinear else "genuinely reorders"),
           n_changed, newly_on, newly_off, med_dist_changed, med_dist_all))
    out = dict(out_head)
    out["verdict"] = out_verdict

    # ---- optional gold-standard: real radar specularity / basal-water phi ----
    if specularity_npy is not None:
        if not os.path.exists(specularity_npy):
            raise DataUnavailableError(
                f"specularity field not found: {specularity_npy}\n"
                "Provision a basal-water/specularity map on the Bedmap2 grid, e.g.:\n"
                "  - Jordan et al. 2018 continental specularity content (USAP-DC / NSIDC)\n"
                "  - MEaSUREs / Schroeder et al. 2013 bed-echo specularity\n"
                "Save as a .npy aligned to the (decimated) Bedmap2 grid in [0,1] "
                "(1=specular/wet/connected). It is mapped phi=PHI_HI-(PHI_HI-PHI_LO)*spec.")
        spec = np.load(specularity_npy)
        if spec.shape != H.shape:
            raise ValueError(f"specularity shape {spec.shape} != Bedmap2 grid {H.shape}")
        phi_spec = PHI_HI - (PHI_HI - PHI_LO) * np.clip(spec, 0, 1)
        rtn_s = build_rtn(H, bed, phi_spec)
        pred_s = classify(rtn_s) & grounded
        sp_s = spearman(np.where(grounded, rtn_s, np.nan), np.where(grounded, h_af_frac, np.nan), rng)
        out["variable_phi_specularity"] = {
            "spearman_rtn_flotfrac": sp_s, "degeneracy_broken": bool(abs(sp_s) < 0.999),
            "n_classification_changed": int(np.sum((pred_s ^ pred_flot) & grounded))}

    out["_arrays_for_fig"] = {"shape": list(H.shape)}
    return out, dict(grounded=grounded, phi_routed=phi_routed, acc=acc, dist=dist,
                     pred_flot=pred_flot, pred_v=pred_v)


def make_figure(arr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = arr["grounded"]
    phi = np.where(g, arr["phi_routed"], np.nan)
    logacc = np.where(g, np.log10(arr["acc"].reshape(g.shape) + 1.0), np.nan)
    changed = (arr["pred_v"] ^ arr["pred_flot"]) & g
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    im0 = ax[0].imshow(logacc, cmap="viridis"); ax[0].set_title("(a) log10 upstream flux\n(hydraulic drainage network)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(phi, cmap="cividis_r"); ax[1].set_title("(b) routing phi(x,y)\n(low=well-connected)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    ov = np.zeros(g.shape); ov[g] = 0.2; ov[changed] = 1.0
    ax[2].imshow(np.where(g, ov, np.nan), cmap="Reds"); ax[2].set_title("(c) cells whose RTN intrusion\nclass changes vs flotation")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("§H.1.1b RTN variable-phi on real Bedmap2 (hydraulic routing)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="/home/K/_data_bedmap2/bedmap2_bin")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--phi-const", type=float, default=0.9)
    ap.add_argument("--specularity-npy", default=None)
    ap.add_argument("--out", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports", "rtn_variable_phi_real.json")))
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()

    print("=== §H.1.1b RTN variable-phi on real Bedmap2 (hydraulic routing) ===")
    out, arr = run(a.bin_dir, stride=a.stride, phi_const=a.phi_const,
                   specularity_npy=a.specularity_npy)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out_to_write = {k: v for k, v in out.items() if k != "_arrays_for_fig"}
    with open(a.out, "w") as fh:
        json.dump(out_to_write, fh, indent=2)
    ac = out["anchor_constant_phi"]; vp = out["variable_phi_routed"]
    print(f"  grounded cells: {out['n_grounded']} @ {out['stride_km']:g} km")
    print(f"  anchor (const phi={a.phi_const}): Spearman(RTN,H_af/H)={ac['spearman_rtn_flotfrac']:+.4f}"
          f"  disagree={ac['disagree_cells']}")
    print(f"  routing phi(x,y): Spearman global={vp['spearman_rtn_flotfrac_global']:+.4f}  "
          f"near-flot band={vp['spearman_rtn_flotfrac_nearflot_band']:+.4f} "
          f"(const-phi band={vp['spearman_rtn_flotfrac_nearflot_band_constphi']:+.4f})")
    print(f"    collinearity Spearman(phi_routed, H_af/H) in-band = "
          f"{vp['spearman_phiRouted_vs_flotfrac_band']:+.3f}  -> break is {'partial' if not vp['degeneracy_broken_in_band'] else 'real'}")
    print(f"    classification changed on {vp['n_classification_changed']} cells "
          f"({vp['newly_flagged_by_phiRTN']} newly flagged, {vp['dropped_vs_flotation']} dropped)")
    print(f"    median dist-to-GL of changed cells {vp['median_distGL_changed_km']:.0f} km "
          f"vs {vp['median_distGL_all_grounded_km']:.0f} km all grounded")
    print(f"  VERDICT: {out['verdict']}")
    print(f"  json -> {a.out}")
    if not a.no_fig:
        make_figure(arr, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
