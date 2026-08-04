r"""§V.1d — pinning the spec→p_w sign convention for variable-φ RTN.

Context (the sharpened open step this addresses)
------------------------------------------------
``rtn_specularity_real.py`` (§V.1c) executed the P4a §6.4 open data step — a
*measured* connectivity observable (ICECAP bed-echo specularity content,
USAP-DC 601371) driving variable-φ RTN on real Bedmap2 — and found that the two
physically-arguable spec→``p_w`` sign conventions move the near-grounding-line
intrusion classification in *opposite directions* (162 newly flagged vs 89
dropped).  The papers therefore state the sharpened open step: "pin the
spec→``p_w`` sign with co-located borehole pressure or an intrusion survey".

No co-located borehole exists in the ICECAP coverage.  This module pins the
sign with three *independent* consistency lines, each of which the two
conventions predict with OPPOSITE signs — so one convention must fail:

1. **The lake anchor (data; non-circular).**  Active subglacial lakes are
   ponded water bodies at the ice-bed interface detected by *surface-height*
   anomalies (ICESat altimetry: Smith et al. 2009; Siegfried & Fricker 2018)
   — an observable completely independent of radar specularity.  By roof force
   balance a persistent ponded water body carries the overburden: ``p_w ≈ p_i``
   (``φ → 1``, the top of the pressure scale; Stubblefield et al. 2021 model
   lakes at ``N ≈ 0``), with transient excursions during fill-drain cycles.
   If lake-footprint cells sit HIGH on the measured specularity scale, a
   monotone *decreasing* spec→``p_w`` map (the "connectivity" reading) would
   assign the *lowest* water pressures to exactly the cells where physics
   requires ``p_w ≈ p_i`` — untenable.  Test: stratified (flotation-matched)
   one-sided Mann-Whitney of lake-cell specularity against far-from-lake
   control cells, with a label-permutation null within flotation strata.
2. **The creep-persistence floor (physics; quantitative).**  A quiescent
   basal water body at effective pressure ``N`` closes by ice creep at the Nye
   rate ``V_c/S = 2A(N/n)^n`` (Nye 1953; Röthlisberger 1972; conservative for
   flat/sheet geometries, which sag *faster* — Walder 1986).  Radar-persistent
   specular water (same cell specular in the 2008/09 AND 2011/12 ICECAP
   seasons, span ``t_obs ≈ 4 yr``) therefore requires
   ``N ≤ N_max = (n^n / (2 A t_obs))^{1/3}`` (≈ 3.5 bar at 4 yr; the closure
   has e-folded once at ``N_max``), i.e. ``φ = p_w/p_i ≥ 1 − N_max/p_i`` ≈
   0.98–0.99 for 2–4 km East Antarctic ice.  The pressure convention puts
   these cells at the top of its φ scale (consistent); the connectivity
   convention maps the *most* specular persistent cells to its *lowest* φ —
   an ordering violation against the physical floor.  (Caveat carried: the
   bound assumes melt-opening ≪ closure at the cell scale, which holds for
   distributed/ponded systems; concentrated R-channels evade it but are
   low-specularity objects at 1-km footprint — Schroeder et al. 2013, 2015.)
3. **The lubrication consistency check (data; weaker, confound-controlled).**
   Hard-bed sliding laws (Lliboutry 1968; Iken 1981; Schoof 2005) have sliding
   speed increasing as effective pressure drops (``u_b ∝ τ_b^m N^{−q}``,
   ``q > 0``).  If high specularity means high ``p_w`` (low ``N``), covered
   cells with higher specularity should slide FASTER *at matched driving
   stress and thickness*.  Test: partial Spearman of MEaSUReS phase-based
   velocity (NSIDC-0754, 450 m; Mouginot et al. 2019) against binned
   specularity, rank-residualised on driving stress ``τ_d = ρ_i g H |∇s|``
   (Bedmap2 surface, 3-cell smoothing), thickness ``H`` and flotation fraction
   ``f``, with a 50-km block bootstrap CI (spatial autocorrelation honesty).
   This line is *consistency evidence, not proof* (velocity and hydrology are
   mutually causal); it is reported with that scope.

Honest scope
------------
This is a *sign* pin, not a calibration: it fixes the monotone orientation of
the spec→φ map (which of the two §V.1c conventions is physically admissible),
not the φ magnitudes (PHI_LO/PHI_HI remain modelling bounds).  Co-located
borehole pressure or a gridded intrusion survey remains the gold standard and
is still stated as such in the papers.  Coverage caveats of §V.1c (East
Antarctic flight lines, 1-km L2 smoothing, 2008-2012 epoch vs Bedmap2) carry
over; the S&F 2018 outlines epoch (2003-2016) vs ICECAP epoch mismatch adds
scatter that *dilutes* the lake anchor (conservative for the one-sided test),
as does 5-km binning of ~10-km lakes.

Run:
    python rtn_spec_sign_pin.py --bin-dir /home/K/_data_bedmap2/bedmap2_bin \
        --spec-dir /home/K/_data_specularity \
        --lake-stats /home/data_usapdc/601470/data/active_lake_statistics.dat \
        --vel-nc /home/data_velocity/antarctic_ice_vel_phase_map_v01.nc
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external import DataUnavailableError  # noqa: E402
from external.run_rtn_bedmap2 import (  # noqa: E402
    RHO_I, RHO_W, distance_to_groundingline_km)
from external.rtn_variable_phi_real import PHI_LO, PHI_HI, spearman  # noqa: E402
from external.rtn_specularity_real import (  # noqa: E402
    bin_to_bedmap2, load_specularity_points)

G = 9.81
GLEN_A = 2.4e-24        # Pa^-3 s^-1, temperate ice (Cuffey & Paterson 2010)
GLEN_N = 3.0
YR = 365.25 * 24 * 3600.0


# ----------------------------------------------------------------- physics --
def creep_nmax(t_obs_s, A=GLEN_A, n=GLEN_N):
    r"""Effective pressure whose Nye closure e-folds a cavity once in ``t_obs``.

    ``V_c/S = 2 A (N/n)^n``  =>  survival ``exp(-2A(N/n)^n t)``; the persistence
    bound ``N_max`` solves ``2A(N_max/n)^n t_obs = 1``:

        N_max = n * (2 A t_obs)^{-1/n}  =  (n^n / (2 A t_obs))^{1/n}.

    Conservative across geometries: wide flat roofs sag faster than the
    cylindrical rate (Walder 1986), so their true ``N_max`` is *smaller*.
    """
    return n * (2.0 * A * t_obs_s) ** (-1.0 / n)


def phi_floor(H, t_obs_s, A=GLEN_A, n=GLEN_N):
    r"""Lower bound on ``φ = p_w/p_i`` for radar-persistent quiescent water."""
    p_i = RHO_I * G * np.asarray(H, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p_i > 0, 1.0 - creep_nmax(t_obs_s, A, n) / p_i, np.nan)


# ------------------------------------------------------------------- lakes --
def load_lake_stats(path):
    """USAP-DC 601470 ``active_lake_statistics.dat`` (Stubblefield et al. 2021;
    outlines after Siegfried & Fricker 2018).

    Columns: PS71 x [m], PS71 y [m], Feret width [km], Feret length [km],
    area-equivalent circle diameter [km], BedMachine thickness [m].
    """
    if path is None or not os.path.exists(path):
        raise DataUnavailableError(
            f"active_lake_statistics.dat not found: {path}\n"
            "Provision (bot-check, no login): https://www.usap-dc.org/view/dataset/601470\n"
            "unzip data.zip -> data/active_lake_statistics.dat")
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    if a.shape[1] < 6:
        raise ValueError(f"expected >=6 columns, got {a.shape[1]}")
    return {"x": a[:, 0], "y": a[:, 1], "feret_w_km": a[:, 2],
            "feret_l_km": a[:, 3], "eqdiam_km": a[:, 4], "H_bm": a[:, 5]}


def lake_cell_masks(lakes, meta, shape, radius_floor_cells=1.0, buffer_km=25.0):
    """Boolean masks (lake footprint, exclusion halo) on the decimated grid.

    A cell belongs to a lake iff its centre lies within
    ``max(eqdiam/2, radius_floor_cells*cellsize)`` of the lake centroid; the
    exclusion halo extends a further ``buffer_km`` (controls are drawn outside
    every halo).  Returns (lake_mask, halo_mask, lake_id_grid, per_lake_cells).
    """
    nr, nc = shape
    cs = meta["cellsize"]
    ytop = meta["yll"] + meta["nrows_full"] * meta["cellsize_full"]
    jj, ii = np.meshgrid(np.arange(nc), np.arange(nr))
    xc = meta["xll"] + (jj + 0.5) * cs
    yc = ytop - (ii + 0.5) * cs
    lake = np.zeros(shape, bool)
    halo = np.zeros(shape, bool)
    lake_id = np.full(shape, -1, np.int32)
    per_lake = []
    for k in range(lakes["x"].size):
        r = max(0.5e3 * lakes["eqdiam_km"][k], radius_floor_cells * cs)
        d2 = (xc - lakes["x"][k]) ** 2 + (yc - lakes["y"][k]) ** 2
        m = d2 <= r * r
        lake |= m
        lake_id[m & (lake_id < 0)] = k
        halo |= d2 <= (r + buffer_km * 1e3) ** 2
        per_lake.append(int(m.sum()))
    return lake, halo, lake_id, np.asarray(per_lake)


# -------------------------------------------------------------- statistics --
def stratified_mw(spec, lake_mask, ctrl_mask, f_af, rng, n_perm=2000,
                  n_strata=10):
    """One-sided Mann-Whitney (lake > control) with a flotation-stratified
    label-permutation null.

    Strata are deciles of ``f_af`` over the pooled sample; labels are permuted
    *within* strata so the null preserves any spec-flotation dependence.
    Returns dict with observed U-statistic z-ish effect (rank-biserial),
    medians, and the permutation p-value.
    """
    a = spec[lake_mask]
    pool = spec[ctrl_mask]
    f_a, f_p = f_af[lake_mask], f_af[ctrl_mask]
    ok_a, ok_p = np.isfinite(a) & np.isfinite(f_a), np.isfinite(pool) & np.isfinite(f_p)
    a, f_a, pool, f_p = a[ok_a], f_a[ok_a], pool[ok_p], f_p[ok_p]
    if a.size == 0 or pool.size == 0:
        return {"n_lake": int(a.size), "n_ctrl": int(pool.size),
                "p_perm": float("nan")}
    allv = np.concatenate([a, pool])
    allf = np.concatenate([f_a, f_p])
    lab = np.concatenate([np.ones(a.size, bool), np.zeros(pool.size, bool)])
    edges = np.nanquantile(allf, np.linspace(0, 1, n_strata + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    stratum = np.clip(np.searchsorted(edges, allf, side="right") - 1, 0,
                      n_strata - 1)

    from scipy import stats
    def _rb(labels):
        u = stats.mannwhitneyu(allv[labels], allv[~labels],
                               alternative="greater")
        n1, n2 = labels.sum(), (~labels).sum()
        return float(u.statistic / (n1 * n2)), u  # rank-biserial-ish in [0,1]

    obs, u_obs = _rb(lab)
    cnt = 0
    for _ in range(n_perm):
        pl = lab.copy()
        for s in range(n_strata):
            m = stratum == s
            if m.sum() > 1:
                pl[m] = rng.permutation(pl[m])
        if _rb(pl)[0] >= obs:
            cnt += 1
    return {
        "n_lake": int(a.size), "n_ctrl": int(pool.size),
        "median_spec_lake": float(np.median(a)),
        "median_spec_ctrl": float(np.median(pool)),
        "lake_median_percentile_in_ctrl": float(
            100.0 * np.mean(pool <= np.median(a))),
        "prob_lake_gt_ctrl": obs,          # P(spec_lake > spec_ctrl), 0.5 = null
        "mw_p_onesided_iid": float(u_obs.pvalue),
        "p_perm_stratified": float((cnt + 1) / (n_perm + 1)),
        "n_perm": n_perm, "n_strata": n_strata,
    }


def rank_partial(y, x, controls, rng, block_ids=None, n_boot=500):
    """Partial Spearman of ``y`` vs ``x`` given ``controls`` (list of arrays),
    via rank-residualisation, with an optional block bootstrap CI."""
    from scipy import stats
    m = np.isfinite(y) & np.isfinite(x)
    for c in controls:
        m &= np.isfinite(c)
    y, x = y[m], x[m]
    C = np.column_stack([c[m] for c in controls])
    if y.size < 30:
        return {"n": int(y.size), "rho_partial": float("nan")}

    def _partial(idx):
        ry = stats.rankdata(y[idx]); rx = stats.rankdata(x[idx])
        D = np.column_stack([np.ones(idx.size)] +
                            [stats.rankdata(C[idx, k]) for k in range(C.shape[1])])
        by, *_ = np.linalg.lstsq(D, ry, rcond=None)
        bx, *_ = np.linalg.lstsq(D, rx, rcond=None)
        ey, ex = ry - D @ by, rx - D @ bx
        sy, sx = ey.std(), ex.std()
        if sy == 0 or sx == 0:
            return float("nan")
        return float(np.mean(ey * ex) / (sy * sx))

    idx_all = np.arange(y.size)
    rho = _partial(idx_all)
    out = {"n": int(y.size), "rho_partial": rho,
           "rho_plain": float(spearman(y, x, rng))}
    if block_ids is not None:
        b = block_ids[m]
        blocks = np.unique(b)
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(blocks, blocks.size, replace=True)
            idx = np.concatenate([idx_all[b == bl] for bl in pick])
            boots.append(_partial(idx))
        boots = np.asarray([v for v in boots if np.isfinite(v)])
        out["rho_partial_ci95"] = [float(np.percentile(boots, 2.5)),
                                   float(np.percentile(boots, 97.5))]
        out["n_blocks"] = int(blocks.size)
        out["n_boot"] = int(boots.size)
    return out


# ------------------------------------------------------- epoch persistence --
def per_epoch_binning(spec_dir, meta, epochs=(("2008", "2009"), ("2011", "2012"))):
    """Bin specularity separately for two season groups (epoch A, epoch B).

    Season is parsed from the profile path (the season directory / file name
    starts with the year, e.g. ``2008_AN_UTIG.IRSPC2/...``).
    """
    files = sorted(glob.glob(os.path.join(spec_dir, "**", "*_spec.txt"),
                             recursive=True))
    if not files:
        raise DataUnavailableError(f"no ICECAP profiles under {spec_dir}")
    out = []
    for group in epochs:
        lons, lats, sps = [], [], []
        for f in files:
            rel = os.path.relpath(f, spec_dir)
            m = re.search(r"(20\d\d)", rel)
            if m and m.group(1) in group:
                a = np.loadtxt(f, comments="#", usecols=(3, 4, 5), ndmin=2)
                if a.size:
                    lons.append(a[:, 0]); lats.append(a[:, 1]); sps.append(a[:, 2])
        if not lons:
            out.append((None, None))
            continue
        lon = np.concatenate(lons); lat = np.concatenate(lats)
        sp = np.concatenate(sps)
        mfin = np.isfinite(sp) & np.isfinite(lon) & np.isfinite(lat)
        out.append(bin_to_bedmap2(lon[mfin], lat[mfin],
                                  np.clip(sp[mfin], 0, 1), meta))
    return out


# ---------------------------------------------------------------- velocity --
def sample_velocity(vel_nc, meta, shape):
    """Nearest-pixel MEaSUReS speed [m/yr] on the decimated Bedmap2 grid.

    NSIDC-0754 (Mouginot et al. 2019): 450 m PS71 grid, variables VX/VY.
    Reads coordinate vectors + strided rows only (h5netcdf lazy access).
    """
    if vel_nc is None or not os.path.exists(vel_nc):
        raise DataUnavailableError(
            f"velocity mosaic not found: {vel_nc}\n"
            "Provision: earthaccess.search_data(short_name='NSIDC-0754') "
            "-> antarctic_ice_vel_phase_map_v01.nc (Earthdata login)")
    import xarray as xr
    nr, nc = shape
    cs = meta["cellsize"]
    ytop = meta["yll"] + meta["nrows_full"] * meta["cellsize_full"]
    xc = meta["xll"] + (np.arange(nc) + 0.5) * cs
    yc = ytop - (np.arange(nr) + 0.5) * cs
    ds = xr.open_dataset(vel_nc)
    vx = ds["VX"]; vy = ds["VY"]
    x = ds["x"].values; y = ds["y"].values
    jx = np.clip(np.searchsorted(x, xc), 1, x.size - 1)
    jx = np.where(np.abs(x[jx] - xc) <= np.abs(x[jx - 1] - xc), jx, jx - 1)
    ydesc = y[0] > y[-1]
    ys = y[::-1] if ydesc else y
    iy = np.clip(np.searchsorted(ys, yc), 1, ys.size - 1)
    iy = np.where(np.abs(ys[iy] - yc) <= np.abs(ys[iy - 1] - yc), iy, iy - 1)
    if ydesc:
        iy = y.size - 1 - iy
    vxs = vx.isel(y=xr.DataArray(iy, dims="i"), x=xr.DataArray(jx, dims="j")).values
    vys = vy.isel(y=xr.DataArray(iy, dims="i"), x=xr.DataArray(jx, dims="j")).values
    ds.close()
    speed = np.hypot(vxs, vys)
    speed[speed <= 0] = np.nan          # no-data is 0/negative in some tiles
    return speed


def driving_stress(H, surface, cellsize, smooth_cells=3):
    """``τ_d = ρ_i g H |∇s|`` with a boxcar-smoothed surface (edge-safe)."""
    s = np.where(np.isfinite(surface), surface, np.nan)
    k = smooth_cells
    pad = np.pad(s, k // 2, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(pad, (k, k))
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ssm = np.nanmean(win, axis=(2, 3))
    gy, gx = np.gradient(ssm, cellsize)
    return RHO_I * G * np.where(H > 0, H, np.nan) * np.hypot(gx, gy)


# --------------------------------------------------------------------- run --
def run(bin_dir, spec_dir, lake_stats, vel_nc=None, stride=5, min_pts=3,
        spec_thresh=0.2, t_obs_yr=4.0, seed=0, n_perm=2000, n_boot=500,
        json_path=None, fig_path=None):
    from external.bedmap2_loader import load_fields
    rng = np.random.default_rng(seed)
    d = load_fields(bin_dir, stride=stride)
    H, bed, mask, surf = (d["thickness"], d["bed"],
                          d["icemask_grounded_and_shelves"], d["surface"])
    meta = d["_meta"]
    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)

    lon, lat, sp, nfiles = load_specularity_points(spec_dir)
    spec_mean, spec_cnt = bin_to_bedmap2(lon, lat, sp, meta)
    cov = grounded & (spec_cnt >= min_pts) & np.isfinite(spec_mean)

    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_af = np.where(H > 0, (H - (RHO_W / RHO_I) * d_base)
                        / np.where(H > 0, H, 1.0), np.nan)
    dist = distance_to_groundingline_km(grounded, meta["cellsize"])

    lakes = load_lake_stats(lake_stats)
    lake_m, halo_m, lake_id, per_lake = lake_cell_masks(lakes, meta, H.shape)
    lake_cov = lake_m & cov
    ctrl_cov = cov & ~halo_m
    covered_lakes = np.unique(lake_id[lake_cov])
    covered_lakes = covered_lakes[covered_lakes >= 0]

    # --- line 1: the lake anchor ------------------------------------------
    anchor = stratified_mw(spec_mean, lake_cov, ctrl_cov, f_af, rng,
                           n_perm=n_perm)
    anchor.update({
        "n_lakes_total": int(lakes["x"].size),
        "n_lakes_covered": int(covered_lakes.size),
        "median_dist_km_lake_cells": float(np.nanmedian(dist[lake_cov]))
        if lake_cov.any() else float("nan"),
        "median_dist_km_ctrl_cells": float(np.nanmedian(dist[ctrl_cov]))
        if ctrl_cov.any() else float("nan"),
        "per_covered_lake": [],
    })
    ctrl_pool = spec_mean[ctrl_cov]
    ctrl_pool = ctrl_pool[np.isfinite(ctrl_pool)]
    for k in covered_lakes:
        sel = lake_cov & (lake_id == k)
        mu = float(np.nanmean(spec_mean[sel]))
        anchor["per_covered_lake"].append({
            "lake_index": int(k),
            "x": float(lakes["x"][k]), "y": float(lakes["y"][k]),
            "eqdiam_km": float(lakes["eqdiam_km"][k]),
            "n_cells": int(sel.sum()),
            "mean_spec": mu,
            "percentile_in_ctrl": float(100.0 * np.mean(ctrl_pool <= mu)),
            "H_bedmap2": float(np.nanmean(H[sel])),
            "H_bedmachine_catalog": float(lakes["H_bm"][k]),
        })
    frac_above = (np.mean([p["percentile_in_ctrl"] > 50.0
                           for p in anchor["per_covered_lake"]])
                  if anchor["per_covered_lake"] else float("nan"))
    anchor["frac_covered_lakes_above_ctrl_median"] = float(frac_above)
    # far-from-GL variant (GL-proximity confound control)
    far = dist > 50.0
    anchor_far = stratified_mw(spec_mean, lake_cov & far, ctrl_cov & far,
                               f_af, rng, n_perm=n_perm)

    # --- line 2: the creep-persistence floor ------------------------------
    (specA, cntA), (specB, cntB) = per_epoch_binning(spec_dir, meta)
    persist = (cov & (cntA >= min_pts) & (cntB >= min_pts)
               & (specA >= spec_thresh) & (specB >= spec_thresh))
    nmax = creep_nmax(t_obs_yr * YR)
    pfl = phi_floor(H, t_obs_yr * YR)
    phi_conn = PHI_HI - (PHI_HI - PHI_LO) * np.clip(spec_mean, 0, 1)
    phi_pres = PHI_LO + (PHI_HI - PHI_LO) * np.clip(spec_mean, 0, 1)
    both_seen = cov & (cntA >= min_pts) & (cntB >= min_pts)
    # The discriminator is the ORDERING, not the absolute level: both maps are
    # capped at PHI_HI=0.97 < floor ~0.98-0.99 (a modelling range, re-calibratable),
    # but the floor grows toward the top of the SPEC scale, so the per-cell
    # violation (floor - phi_map)+ must SHRINK with spec under an admissible
    # monotone map. The connectivity map sends spec up -> phi down, so its
    # violation GROWS with spec; the pressure map's shrinks.
    viol_conn = np.maximum(pfl - phi_conn, 0.0)
    viol_pres = np.maximum(pfl - phi_pres, 0.0)
    # Calibration-independent orientation logic: the floor constrains ONLY the
    # persistent (high-spec) cells, phi(s in persistent set) >= floor. An
    # INCREASING map keeps its low end free (admissible range up to ~1); a
    # DECREASING map is >= its value on the persistent set everywhere below
    # spec_thresh, so its ENTIRE range is squeezed above the floor:
    # max range = 1 - floor (~0.02). The connectivity convention claims range
    # PHI_HI-PHI_LO = 0.17 >> 0.02 -> inadmissible; pressure is unconstrained.
    required_floor = (float(np.nanpercentile(pfl[persist], 10))
                      if persist.any() else float("nan"))
    claimed_range = PHI_HI - PHI_LO
    max_range_decreasing = (max(0.0, 1.0 - required_floor)
                            if np.isfinite(required_floor) else float("nan"))
    creep = {
        "t_obs_yr": t_obs_yr, "glen_A": GLEN_A, "glen_n": GLEN_N,
        "N_max_Pa": float(nmax), "N_max_bar": float(nmax / 1e5),
        "spec_thresh": spec_thresh,
        "n_cells_seen_both_epochs": int(both_seen.sum()),
        "n_persistent_specular": int(persist.sum()),
        "median_phi_floor_persistent": float(np.nanmedian(pfl[persist]))
        if persist.any() else float("nan"),
        "p10_phi_floor_persistent": float(np.nanpercentile(pfl[persist], 10))
        if persist.any() else float("nan"),
        "mean_violation_conn": float(np.nanmean(viol_conn[persist]))
        if persist.any() else float("nan"),
        "mean_violation_pres": float(np.nanmean(viol_pres[persist]))
        if persist.any() else float("nan"),
        "required_floor_p10": required_floor,
        "claimed_range_both_conventions": claimed_range,
        "max_admissible_range_decreasing_map": max_range_decreasing,
        "max_admissible_range_increasing_map": 1.0,
        "decreasing_map_inadmissible":
            (bool(claimed_range > max_range_decreasing)
             if np.isfinite(required_floor) else None),
        "per_persistent_cell": [
            {"spec": float(spec_mean[i, j]), "phi_floor": float(pfl[i, j]),
             "phi_conn": float(phi_conn[i, j]), "phi_pres": float(phi_pres[i, j]),
             "H_m": float(H[i, j])}
            for i, j in zip(*np.where(persist))],
        "thresh_sensitivity": [],
        "note": ("floor conservative for sheets (Walder 1986; a wide thin "
                 "sheet sags ~w/h faster, sharpening N_max by "
                 "(2h/(w n^n))^(1/n) to ~0.1 bar); assumes melt-opening << "
                 "closure (quiescent distributed water); R-channels evade "
                 "the bound but are low-specularity objects at the 1-km "
                 "footprint (Schroeder 2013, 2015). The floor constrains only "
                 "the persistent high-spec cells, so it is an ORIENTATION "
                 "constraint: an increasing map keeps a free low end, while "
                 "a decreasing map is squeezed entirely above the floor "
                 "(max range 1-floor ~0.02) - the connectivity convention's "
                 "claimed 0.17 range is inadmissible. Both conventions' "
                 "absolute values also sit below the floor at PHI_HI=0.97: "
                 "that is a top-of-scale calibration note (PHI_HI should be "
                 ">=0.98 for persistent cells), not a sign discriminator."),
    }
    for th in (0.10, 0.15, 0.20, 0.30):
        p_th = (cov & (cntA >= min_pts) & (cntB >= min_pts)
                & (specA >= th) & (specB >= th))
        creep["thresh_sensitivity"].append({
            "spec_thresh": th, "n_persistent": int(p_th.sum()),
            "median_phi_floor": float(np.nanmedian(pfl[p_th]))
            if p_th.any() else float("nan"),
            "mean_violation_conn": float(np.nanmean(viol_conn[p_th]))
            if p_th.any() else float("nan"),
            "mean_violation_pres": float(np.nanmean(viol_pres[p_th]))
            if p_th.any() else float("nan"),
        })

    # --- line 3: lubrication partial-rank ---------------------------------
    lub = {"skipped": True}
    if vel_nc is not None:
        speed = sample_velocity(vel_nc, meta, H.shape)
        tau_d = driving_stress(H, surf, meta["cellsize"])
        ii = np.arange(H.shape[0])[:, None] * np.ones(H.shape[1], int)
        jj = np.ones(H.shape[0], int)[:, None] * np.arange(H.shape[1])
        blocks = (ii // 10) * 10_000 + (jj // 10)   # 50-km blocks at 5-km cells
        sel = cov & np.isfinite(speed) & np.isfinite(tau_d)
        sel &= tau_d > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            lub = rank_partial(
                np.where(sel, np.log10(np.maximum(speed, 1e-2)), np.nan).ravel(),
                np.where(sel, spec_mean, np.nan).ravel(),
                [np.where(sel, np.log10(np.where(sel, tau_d, 1.0)), np.nan).ravel(),
                 np.where(sel, H, np.nan).ravel(),
                 np.where(sel, f_af, np.nan).ravel()],
                rng, block_ids=blocks.ravel(), n_boot=n_boot)
        lub["skipped"] = False
        lub["velocity_product"] = "NSIDC-0754 phase-based 450m (Mouginot 2019)"
        lub["controls"] = ["log10 tau_d (3-cell-smoothed Bedmap2 surface)",
                           "H", "flotation fraction"]
        lub["scope"] = ("consistency evidence, not proof: sliding and "
                        "hydrology are mutually causal; sign is the claim")

    # --- verdict ------------------------------------------------------------
    lines = {
        "lake_anchor_supports_pressure":
            bool(anchor.get("p_perm_stratified", 1.0) < 0.05
                 and anchor.get("prob_lake_gt_ctrl", 0.5) > 0.5),
        "creep_floor_supports_pressure":
            (None if creep["n_persistent_specular"] == 0 else
             creep["decreasing_map_inadmissible"]),
        "lubrication_supports_pressure":
            (None if lub.get("skipped") else
             bool(lub.get("rho_partial", 0) > 0 and
                  lub.get("rho_partial_ci95", [-1, 1])[0] > 0)),
    }
    n_for = sum(1 for v in lines.values() if v is True)
    n_against = sum(1 for v in lines.values() if v is False)
    verdict = {
        **lines,
        "pinned_convention": ("pressure_highspec_highphi"
                              if n_for >= 2 and n_against == 0 else
                              "connectivity_highspec_lowphi"
                              if n_against >= 2 and n_for == 0 else
                              "unresolved"),
        "remaining_gold_standard": "co-located borehole p_w / gridded "
                                   "intrusion survey (unchanged)",
    }

    out = {
        "inputs": {
            "spec_dataset": "USAP-DC 601371 (Young et al. 2020) ICECAP IRSPC2",
            "lake_dataset": "USAP-DC 601470 (Stubblefield et al. 2021; "
                            "Siegfried & Fricker 2018 outlines)",
            "n_profiles": nfiles, "n_points": int(sp.size),
            "stride_km": meta["cellsize"] / 1000.0,
            "min_pts_per_cell": min_pts, "seed": seed,
            "PHI_LO": PHI_LO, "PHI_HI": PHI_HI,
        },
        "coverage": {
            "n_covered_grounded": int(cov.sum()),
            "n_lake_cells_covered": int(lake_cov.sum()),
            "n_ctrl_cells": int(ctrl_cov.sum()),
        },
        "line1_lake_anchor": anchor,
        "line1b_lake_anchor_far_from_gl": anchor_far,
        "line2_creep_floor": creep,
        "line3_lubrication": lub,
        "verdict": verdict,
    }

    here = os.path.dirname(os.path.abspath(__file__))
    json_path = json_path or os.path.join(here, "..", "reports",
                                          "rtn_spec_sign_pin.json")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"json -> {os.path.abspath(json_path)}")

    fig_path = fig_path or os.path.join(here, "..", "reports",
                                        "rtn_spec_sign_pin.png")
    try:
        _figure(fig_path, spec_mean, cov, lake_cov, ctrl_cov, pfl, persist,
                phi_conn, phi_pres, lub, anchor)
        print(f"figure -> {os.path.abspath(fig_path)}")
    except Exception as ex:                                  # pragma: no cover
        print(f"figure skipped: {ex}")
    return out


def _figure(path, spec_mean, cov, lake_cov, ctrl_cov, pfl, persist,
            phi_conn, phi_pres, lub, anchor):                # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    a = ax[0, 0]
    sc, sl = spec_mean[ctrl_cov], spec_mean[lake_cov]
    bins = np.linspace(0, 1, 41)
    a.hist(sc[np.isfinite(sc)], bins=bins, density=True, alpha=0.6,
           label=f"control cells (n={np.isfinite(sc).sum()})")
    a.hist(sl[np.isfinite(sl)], bins=bins, density=True, alpha=0.6,
           label=f"active-lake cells (n={np.isfinite(sl).sum()})")
    a.axvline(np.nanmedian(sc), color="C0", ls="--")
    a.axvline(np.nanmedian(sl), color="C1", ls="--")
    a.set_xlabel("binned specularity"); a.set_ylabel("pdf")
    a.set_title("line 1 — altimetry lakes sit high on the spec scale\n"
                f"P(lake>ctrl)={anchor.get('prob_lake_gt_ctrl', float('nan')):.2f}, "
                f"p_perm={anchor.get('p_perm_stratified', float('nan')):.4f}")
    a.legend(fontsize=8)

    b = ax[0, 1]
    if persist.any():
        pf = pfl[persist]; pc = phi_conn[persist]; pp = phi_pres[persist]
        s = spec_mean[persist]
        o = np.argsort(s)
        b.plot(s[o], pf[o], "k.", ms=3, label="creep floor φ ≥ 1−N_max/p_i")
        b.plot(s[o], pc[o], "C3.", ms=3, label="connectivity map φ(spec)")
        b.plot(s[o], pp[o], "C2.", ms=3, label="pressure map φ(spec)")
        b.set_xlabel("binned specularity (persistent cells)")
        b.set_ylabel("φ = p_w/p_i")
        b.set_title("line 2 — persistence floor vs the two maps\n"
                    "(connectivity sends spec↑ → φ↓ against the floor)")
        b.legend(fontsize=8)
    else:
        b.text(0.5, 0.5, "no persistent cells", ha="center")

    c = ax[1, 0]
    if not lub.get("skipped"):
        ci = lub.get("rho_partial_ci95", [np.nan, np.nan])
        c.bar([0, 1], [lub.get("rho_plain", np.nan),
                       lub.get("rho_partial", np.nan)],
              yerr=[[0, lub["rho_partial"] - ci[0]],
                    [0, ci[1] - lub["rho_partial"]]] if np.isfinite(ci[0])
              else None,
              tick_label=["plain ρ(spec,log v)", "partial ρ | τ_d,H,f"])
        c.axhline(0, color="k", lw=0.8)
        c.set_title(f"line 3 — lubrication consistency (n={lub.get('n')})\n"
                    "pressure reading predicts > 0")
    else:
        c.text(0.5, 0.5, "velocity skipped", ha="center")

    dpl = ax[1, 1]
    dpl.axis("off")
    rows = [(p["lake_index"], p["n_cells"], f"{p['mean_spec']:.3f}",
             f"{p['percentile_in_ctrl']:.0f}%")
            for p in anchor.get("per_covered_lake", [])[:14]]
    if rows:
        tbl = dpl.table(cellText=rows,
                        colLabels=["lake#", "cells", "mean spec", "pctile"],
                        loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    dpl.set_title("covered active lakes (601470)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():                                                   # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin-dir", default="/home/K/_data_bedmap2/bedmap2_bin")
    ap.add_argument("--spec-dir", default="/home/K/_data_specularity")
    ap.add_argument("--lake-stats",
                    default="/home/data_usapdc/601470/data/active_lake_statistics.dat")
    ap.add_argument("--vel-nc", default=None,
                    help="NSIDC-0754 nc; omit to skip the lubrication line")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-pts", type=int, default=3)
    ap.add_argument("--spec-thresh", type=float, default=0.2)
    ap.add_argument("--t-obs-yr", type=float, default=4.0)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = run(a.bin_dir, a.spec_dir, a.lake_stats, vel_nc=a.vel_nc,
              stride=a.stride, min_pts=a.min_pts, spec_thresh=a.spec_thresh,
              t_obs_yr=a.t_obs_yr, seed=a.seed, n_perm=a.n_perm,
              n_boot=a.n_boot)
    v = out["verdict"]
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()
