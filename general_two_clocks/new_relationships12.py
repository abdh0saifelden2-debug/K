r"""NR35 -- the two-sided persistence bracket: repeat radar is a THRESHOLD
GAUGE for effective pressure, and a dated drainage measures its correction.

Ledger item E5a (papers/RESEARCH_RECAP_AND_HORIZON.md).  NR33 proved one
direction: a basal water body specular in two surveys separated by ``Dt``
must have survived Nye creep closure, so ``N <= N*(Dt)``.  This module derives
and applies the OTHER direction, and the dated refinement of both.

The statements
--------------
Let ``c(N) = 2A(N/n)^n`` be the Nye closure rate (survival ``exp(-int c)``)
and let ``m >= 0`` be any melt/refill *opening* rate (relative, per unit
cavity: opening velocity / cavity depth).  Define the e-fold threshold

    N*(Dt) = n (2 A Dt)^{-1/n}          (= 3.5 bar at Dt = 4 yr, n = 3).

(1) **Vanish ceiling (NEW; melt-robust).**  A cell specular at epoch A and
    dark at epoch B closed (net) by at least one e-fold within ``Dt``:
    ``int (c - m) dt >= 1``  =>  ``c Dt >= 1 + int m dt >= 1``  =>

        N >= N*(Dt)              -- a LOWER bound on N, i.e. a pressure
                                    CEILING  phi <= 1 - N*(Dt)/p_i,

    and melt only *strengthens* it (a fortiori: any opening that fought the
    closure means the closure was even faster).  The systematic caveats run
    the other way (roughening/spreading below detection also dim a cell), so
    the ceiling is reported as the creep-attribution reading with the
    alternative dimming channels stated.

(2) **Persist floor with a MEASURED melt correction.**  Survival despite
    closure requires ``int (c - m) dt <= 1``, so with measured mean opening
    ``mbar``:

        N <= n ((1/Dt + mbar) / (2A))^{1/n}   -- NR33's floor, WEAKENED by
                                                 exactly the measured amount
                                                 (reduces to N*(Dt) at mbar=0).

    NR33 had to *assume* melt-opening << closure; a dated, altimetry-tracked
    lake replaces the assumption with a number (below).

(3) **Threshold-gauge corollary.**  (1)+(2) share the same constant: one
    repeat pair at separation ``Dt`` splits every twice-seen specular cell
    into ``N <= N*(Dt)`` (persist) or ``N >= N*(Dt)`` (vanish) -- repeat
    radar is a *binary effective-pressure classifier* at a threshold set
    purely by ice rheology and the revisit time.  Survey-design corollary:
    the threshold is tunable, ``N* ~ Dt^{-1/n}`` (the NR33 archive dividend,
    now two-sided).

Applied to ICECAP 2008/09 x 2011/12 (USAP-DC 601371, Bedmap2 grounded cells)
----------------------------------------------------------------------------
Census with a flicker deadband (specular >= 0.2, dark < 0.1, min 3 pts/cell
per epoch): persist / vanish / appear / dark counts, each cell carrying its
N-side label, the phi bound at local overburden, and the SSV.1e roughness
gauge values sigma(spec) per epoch (the geometric face of the same
transition).  Deadband sensitivity is reported at three threshold pairs.

The dated anchor: Totten_2 (USAP-DC 601439 x both radar epochs)
---------------------------------------------------------------
Exactly one 601439 volume-history lake sits in the twice-seen set: Totten_2
(centroids parsed from the per-lake GeoTIFF z-grids -- PIL tag read, EPSG:3031
verified against the 601470 outline catalogue).  Its record gives two dated
drainages (troughs 2005.4 and 2007.8; 0.60 and 0.32 km^3) with the lake LOW
and slowly refilling at the record end -- and the cell is the strongest
persistent-specular cell in the census (spec 0.49 -> 0.32).  Three
independent instruments interlock:

  * altimetry (601439): water present, roof deflated since t_d = 2007.8,
    refill rate dV/dt measured  -> the opening correction ``mbar`` for (2);
  * radar persistence: survival over the DATED span t_d -> epoch B
    (Dt = 4.2 yr, measured not assumed)  -> corrected floor via (2);
  * the creep clock: turns both into  N <= N*(4.2 yr) x (1 + mbar Dt)^{1/3}
    with every factor measured.

This is the E5a upgrade: NR33's floor becomes a bracket machine whose
assumption (quiescence) is replaced by a measurement at the one cell where
the datasets overlap, and whose ceiling class (11 cells) is new information
no single survey carries.

Artifacts -> figures/nr35_two_sided_bracket.{json,png}
Run:  python general_two_clocks/new_relationships12.py
Test: pytest general_two_clocks/tests/test_new_relationships12.py -v
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "glaciers", "validation"))
sys.path.insert(0, _HERE)

from new_relationships10 import (  # noqa: E402  (NR33 closed forms)
    GLEN_A, GLEN_N, RHO_I, G as G_GRAV, YR, nmax_cylinder)

SPEC_DIR = "/home/K/_data_specularity"
BEDMAP_DIR = "/home/K/_data_bedmap2/bedmap2_bin"
DIR_601439 = os.path.expanduser("~/data_usapdc/601439")
LAKE_STATS_601470 = os.path.expanduser(
    "~/data_usapdc/601470/data/active_lake_statistics.dat")

EPOCH_A_YR = 2009.0            # 2008/09 ICECAP season midpoint
EPOCH_B_YR = 2012.0            # 2011/12 season midpoint
# conservative spans per direction: the floor (persist) uses the SHORTEST
# defensible survival window (midpoint separation, 3 yr); the ceiling
# (vanish) must allow the LONGEST window in which closure could have
# happened (earliest-A to latest-B flights, ~4 yr) -- each choice weakens
# its own bound, never strengthens it
DT_FLOOR_YR = 3.0
DT_CEIL_YR = 4.0


# ------------------------------------------------------------- closed forms --
def n_threshold(dt_s, A=GLEN_A, n=GLEN_N):
    """N*(Dt) = n (2 A Dt)^{-1/n} -- the shared persist/vanish threshold."""
    return nmax_cylinder(dt_s, A=A, n=n)


def n_floor_melt_corrected(dt_s, mbar_per_s=0.0, A=GLEN_A, n=GLEN_N):
    """Persist floor with measured mean relative opening rate mbar:
    N <= n ((1/Dt + mbar)/(2A))^{1/n}; reduces to N*(Dt) at mbar = 0."""
    return n * ((1.0 / dt_s + mbar_per_s) / (2.0 * A)) ** (1.0 / n)


def phi_bound(H_m, N_pa):
    """phi = p_w/p_i bound corresponding to an effective-pressure bound."""
    p_i = RHO_I * G_GRAV * np.asarray(H_m, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p_i > 0, 1.0 - N_pa / p_i, np.nan)


def survival_efolds(N_pa, dt_s, mbar_per_s=0.0, A=GLEN_A, n=GLEN_N):
    """Net closure e-folds int (c - m) dt for constant N and m (the ODE is
    linear in S, so the exponent is exact)."""
    return (2.0 * A * (np.asarray(N_pa, float) / n) ** n - mbar_per_s) * dt_s


# --------------------------------------------------------------- census -----
def epoch_census(spec_dir=SPEC_DIR, bin_dir=BEDMAP_DIR, stride=5, min_pts=3,
                 thr_on=0.2, thr_off=0.1):
    """Grounded twice-seen cells classified persist/vanish/appear/dark."""
    from external.bedmap2_loader import load_fields
    from external.rtn_spec_sign_pin import per_epoch_binning

    d = load_fields(bin_dir, stride=stride,
                    fields=("thickness", "bed", "icemask_grounded_and_shelves"))
    H, mask = d["thickness"], d["icemask_grounded_and_shelves"]
    meta = d["_meta"]
    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)
    (sA, cA), (sB, cB) = per_epoch_binning(spec_dir, meta)
    if sA is None or sB is None:
        raise RuntimeError("per-epoch binning unavailable")
    seen = grounded & (cA >= min_pts) & (cB >= min_pts) \
        & np.isfinite(sA) & np.isfinite(sB)
    classes = {
        "persist": seen & (sA >= thr_on) & (sB >= thr_on),
        "vanish": seen & (sA >= thr_on) & (sB < thr_off),
        "appear": seen & (sA < thr_off) & (sB >= thr_on),
        "dark": seen & (sA < thr_off) & (sB < thr_off),
    }
    return classes, seen, sA, sB, H, meta


def lake_centroids_from_geotiffs(dir_439=DIR_601439):
    """(name, x, y) in EPSG:3031 for every 601439 lake, from the z-grid
    GeoTIFF tags (ModelPixelScale 33550 + ModelTiepoint 33922; the GeoKey
    directory pins EPSG:3031).  No GDAL needed -- PIL reads the tags."""
    from PIL import Image
    rows = []
    for lakedir in sorted(glob.glob(os.path.join(dir_439, "*", ""))):
        name = os.path.basename(lakedir.rstrip("/"))
        tifs = sorted(glob.glob(os.path.join(lakedir, "z_grids", "*.tif")))
        if not tifs:
            continue
        im = Image.open(tifs[0])
        t = im.tag_v2
        px, tp, keys = t.get(33550), t.get(33922), t.get(34735)
        if px is None or tp is None:
            continue
        if keys is not None:
            k = list(keys)
            # GeoKey 3072 (ProjectedCSTypeGeoKey) must be EPSG:3031
            for a in range(4, len(k) - 3, 4):
                if k[a] == 3072:
                    assert k[a + 3] == 3031, f"{name}: unexpected CRS {k[a+3]}"
        w, h = im.size
        rows.append((name, tp[3] + 0.5 * w * px[0], tp[4] - 0.5 * h * px[1]))
    return rows


def grid_index(x, y, meta):
    cs = meta["cellsize"]
    ytop = meta["yll"] + meta["nrows_full"] * meta["cellsize_full"]
    return int((ytop - y) // cs), int((x - meta["xll"]) // cs)


# ------------------------------------------------------------------- run ----
def run(thr_on=0.2, thr_off=0.1):
    sys.path.insert(0, os.path.join(_HERE, "..", "glaciers", "validation",
                                    "external"))
    from e1_radar_roughness import sigma_from_spec
    from external.run_usapdc_lakes import parse_volume_history, detect_drainages

    classes, seen, sA, sB, H, meta = epoch_census(thr_on=thr_on, thr_off=thr_off)
    n_star_floor = n_threshold(DT_FLOOR_YR * YR)
    n_star_ceil = n_threshold(DT_CEIL_YR * YR)

    out = {
        "threshold": {
            "Dt_floor_yr": DT_FLOOR_YR, "Dt_ceiling_yr": DT_CEIL_YR,
            "N_star_floor_bar": n_star_floor / 1e5,
            "N_star_ceiling_bar": n_star_ceil / 1e5,
            "statement": "one repeat pair splits twice-seen specular cells "
                         "into N <= N*(Dt_floor) (persist) vs N >= "
                         "N*(Dt_ceiling) (vanish); each direction uses its "
                         "conservative span, leaving the honest deadband "
                         "[N*_ceil, N*_floor]; the vanish side is "
                         "melt-robust (a fortiori)",
        },
        "census": {}, "deadband_sensitivity": {}, "cells": {},
    }
    for name, m in classes.items():
        out["census"][name] = int(m.sum())
    out["census"]["seen_both"] = int(seen.sum())

    # per-cell ledger for the two informative classes
    for cls in ("persist", "vanish"):
        rows = []
        for i, j in zip(*np.where(classes[cls])):
            h = float(H[i, j])
            bound = (phi_bound(h, n_star_floor) if cls == "persist"
                     else phi_bound(h, n_star_ceil))
            rows.append({
                "spec_A": float(sA[i, j]), "spec_B": float(sB[i, j]),
                "sigma_A_m": float(sigma_from_spec(max(sA[i, j], 1e-6))),
                "sigma_B_m": float(sigma_from_spec(max(sB[i, j], 1e-6))),
                "H_m": h,
                ("phi_floor" if cls == "persist" else "phi_ceiling"):
                    float(bound),
            })
        out["cells"][cls] = rows

    # deadband sensitivity
    for on, off in ((0.15, 0.075), (0.2, 0.1), (0.3, 0.15)):
        cl, sn, *_ = epoch_census(thr_on=on, thr_off=off)
        out["deadband_sensitivity"][f"on={on:g},off={off:g}"] = {
            k: int(v.sum()) for k, v in cl.items()}

    # ---- dated anchor: 601439 centroids x twice-seen set --------------------
    cents = lake_centroids_from_geotiffs()
    anchor = {"n_lakes_with_grids": len(cents), "covered": []}
    # cross-validate centroid parsing against the 601470 outline catalogue
    try:
        from external.rtn_spec_sign_pin import load_lake_stats
        cat = load_lake_stats(LAKE_STATS_601470)
        cx, cy = np.asarray(cat["x"]), np.asarray(cat["y"])
        dmin = [float(np.min(np.hypot(cx - x, cy - y)) / 1e3)
                for _, x, y in cents]
        anchor["centroid_vs_601470_median_nearest_km"] = float(np.median(dmin))
    except Exception as e:                                  # pragma: no cover
        anchor["centroid_vs_601470_median_nearest_km"] = f"unavailable: {e}"

    for name, x, y in cents:
        i, j = grid_index(x, y, meta)
        if not (0 <= i < seen.shape[0] and 0 <= j < seen.shape[1]) \
                or not seen[i, j]:
            continue
        t, v = parse_volume_history(
            os.path.join(DIR_601439, name, "Volume_history.csv"))
        ev = detect_drainages(t, v)
        cell = {"lake": name, "x": x, "y": y,
                "spec_A": float(sA[i, j]), "spec_B": float(sB[i, j]),
                "H_m": float(H[i, j]),
                "events": ev, "record_end_yr": float(t.max()),
                "V_end_rel_km3": float(v[-1]),
                "V_min_rel_km3": float(v.min())}
        if ev:
            t_d = ev[-1]["t_trough"]
            dt_dated = (EPOCH_B_YR - t_d) * YR
            # measured mean relative opening (refill) rate after the trough;
            # the normalisation (what "per unit cavity" means for a lake-mean
            # volume series) is a modelling choice, so BOTH defensible
            # readings are carried as a bracket: refill velocity over the
            # full historical range (optimistic, smallest mbar) vs over the
            # CURRENT deflated depth (conservative, largest mbar)
            after = t >= t_d
            if after.sum() >= 2 and (v.max() - v.min()) > 0:
                dv_dt = max(float(np.polyfit(t[after], v[after], 1)[0]), 0.0)
                mbar_lo = dv_dt / (v.max() - v.min())          # per yr
                depth_now = max(float(v[-1] - v.min()), 1e-9)
                mbar_hi = dv_dt / depth_now
            else:                                           # pragma: no cover
                mbar_lo = mbar_hi = 0.0
            n_fl_lo = n_floor_melt_corrected(dt_dated, mbar_lo / YR)
            n_fl_hi = n_floor_melt_corrected(dt_dated, mbar_hi / YR)
            cell.update({
                "t_drain_yr": t_d,
                "Dt_dated_yr": dt_dated / YR,
                "mbar_per_yr_range_norm": mbar_lo,
                "mbar_per_yr_currentdepth_norm": mbar_hi,
                "N_floor_undated_bar": n_star_floor / 1e5,
                "N_floor_dated_bar": n_threshold(dt_dated) / 1e5,
                "N_floor_melt_corrected_bar_bracket": [n_fl_lo / 1e5,
                                                       n_fl_hi / 1e5],
                "phi_floor_corrected_bracket": [
                    float(phi_bound(H[i, j], n_fl_hi)),
                    float(phi_bound(H[i, j], n_fl_lo))],
                "melt_correction_pct_bracket": [
                    100.0 * (n_fl_lo / n_threshold(dt_dated) - 1.0),
                    100.0 * (n_fl_hi / n_threshold(dt_dated) - 1.0)],
            })
        anchor["covered"].append(cell)
    out["dated_anchor"] = anchor
    return out, (classes, seen, sA, sB, meta)


# ------------------------------------------------------------------ figure --
def make_figure(out, aux, path_png):                        # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes, seen, sA, sB, meta = aux

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    dts = np.linspace(0.5, 20, 200) * YR
    ax.plot(dts / YR, n_threshold(dts) / 1e5, lw=2)
    ax.axvline(DT_FLOOR_YR, color="k", ls=":", lw=1)
    ax.axvline(DT_CEIL_YR, color="k", ls=":", lw=1)
    ax.annotate("ICECAP pair:\nfloor N* = %.2f bar (3 yr)\nceiling N* = %.2f bar (4 yr)\npersist %d | vanish %d"
                % (out["threshold"]["N_star_floor_bar"],
                   out["threshold"]["N_star_ceiling_bar"],
                   out["census"]["persist"], out["census"]["vanish"]),
                xy=(0.45, 0.6), xycoords="axes fraction", fontsize=8)
    ax.set_xlabel("revisit separation Dt [yr]")
    ax.set_ylabel("threshold N* [bar]")
    ax.set_title(r"repeat radar = N-classifier at $n(2A\,\Delta t)^{-1/3}$")

    ax = axes[1]
    cs = meta["cellsize"] / 1e3
    ii, jj = np.where(seen & ~(classes["persist"] | classes["vanish"]
                               | classes["appear"]))
    ax.scatter(jj * cs, -ii * cs, s=4, c="0.8", label="dark/mixed")
    for cls, col in (("persist", "tab:blue"), ("vanish", "tab:red"),
                     ("appear", "tab:green")):
        ii, jj = np.where(classes[cls])
        ax.scatter(jj * cs, -ii * cs, s=22, c=col, label=cls)
    ax.legend(fontsize=7); ax.set_aspect("equal")
    ax.set_title("twice-seen cells: the two-sided census")
    ax.set_xlabel("grid x [km]"); ax.set_ylabel("grid y [km]")

    ax = axes[2]
    cov = out["dated_anchor"]["covered"]
    if cov and "t_drain_yr" in cov[0]:
        c = cov[0]
        import numpy as _np
        t, v = [], []
        from external.run_usapdc_lakes import parse_volume_history
        t, v = parse_volume_history(os.path.join(
            DIR_601439, c["lake"], "Volume_history.csv"))
        ax.plot(t, v, "o-", lw=1.5, label=f"{c['lake']} volume anomaly")
        for e in c["events"]:
            ax.axvspan(e["t_peak"], e["t_trough"], color="tab:red", alpha=0.15)
        for ep, sp, lab in ((EPOCH_A_YR, c["spec_A"], "epoch A"),
                            (EPOCH_B_YR, c["spec_B"], "epoch B")):
            ax.axvline(ep, color="tab:blue", ls="--", lw=1)
            ax.annotate(f"{lab}\nspec={sp:.2f}", xy=(ep, 0.02),
                        fontsize=7, ha="center")
        ax.annotate("dated floor: N <= %.1f-%.1f bar\n(melt-corr. +%.0f-%.0f%%)"
                    % (*c["N_floor_melt_corrected_bar_bracket"],
                       *c["melt_correction_pct_bracket"]),
                    xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8)
        ax.set_xlabel("year"); ax.set_ylabel("V anomaly [km$^3$]")
        ax.set_title("the dated anchor: altimetry x radar x creep clock")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path_png, dpi=140)
    plt.close(fig)


def main():                                                 # pragma: no cover
    out, aux = run()
    os.makedirs(FIGDIR, exist_ok=True)
    jpath = os.path.join(FIGDIR, "nr35_two_sided_bracket.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, aux, os.path.join(FIGDIR, "nr35_two_sided_bracket.png"))
    print(json.dumps({k: out[k] for k in
                      ("threshold", "census", "deadband_sensitivity")},
                     indent=1))
    print(json.dumps(out["dated_anchor"], indent=1)[:2500])
    print(f"wrote {jpath} and the png")


if __name__ == "__main__":                                  # pragma: no cover
    main()
