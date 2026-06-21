r"""§G.4 — DOWNSTREAM-TRUNK lake-drainage -> ITS_LIVE lag test + amplitude calibration.

Why this module exists (what is new vs ``lake_lag_atl15_itslive.py``)
--------------------------------------------------------------------
The modern co-temporal runner (§V.2d, ``lake_lag_atl15_itslive.py``) samples the
ITS_LIVE velocity *response* in a 2 km box at the **lake centroid** and found a
single in-band detection (Thwaites ``Thw_142``, +8.5 %, lag 1.1 yr) among 19
testable drainages, explicitly noting the reason most lakes look null:

    "the active lakes sit in **slow ice** at their centroids ... the genuinely
     fast outlets ... carry the test."

A subglacial drainage perturbs the effective pressure ``N`` locally, but the
*sliding* response is expressed where the bed is already near the
Coulomb/plastic regime (low ``N``, fast sliding) -- i.e. **downstream on the fast
trunk**, not in the thick, slow, high-``N`` ice over the lake itself.  This module
therefore measures the response **along the ice-flow line downstream of each
lake**, using the static ITS_LIVE 240 m mosaic ``(vx, vy)`` to trace the flowline
and the ITS_LIVE datacubes for the velocity *time series* at the trunk.

For every Siegfried & Fricker (2018) lake that drains in the ICESat-2 ATL15 era
(2019-2026) we:
  1. detect the drainage date from ATL15 ``delta_h`` over the lake outline
     (reused ``detect_drainages``);
  2. trace the flowline downstream from the lake centroid on the ITS_LIVE mosaic;
  3. sample the box-mean speed *time series* (datacube) at the centroid AND at the
     downstream max-speed **trunk** point;
  4. run the same robust, secular-trend-controlled drainage-response test
     (``lake_lag_itslive_match.analyse_event``) at each;
  5. record the in-band detection's **amplitude** ``resp_frac = du/u``, lag, the
     trunk speed and downstream distance, and the lake's **effective-pressure
     proxy** ``rel = m/H`` (Bedmap2 + §H.1.1 ``margin_field``).

Amplitude calibration (the deliverable)
---------------------------------------
With more than one field detection in hand we can finally test the §H.1.6
prediction that the surge **amplitude** grows as the bed approaches flotation
(``|s_N| = |d ln u_b/d ln N|`` increases as ``N -> 0``): we regress
``du/u`` on the effective-pressure proxy ``rel`` (and on trunk speed and drained
volume) across detections.  This converts the "one Thwaites point" of §V.2d into
a *population* with a fitted amplitude law -- the "enough field detections to
calibrate the amplitude" item the project flagged as missing.

Data (ATL15 = Earthdata; ITS_LIVE = anon S3; Bedmap2 = BAS, all already used in
the repo).  No reproduction of prior runs -- this is a new measurement geometry.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))
# allow both `import bedmap2_loader` and `from external.bedmap2_loader import ...`
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ATL = _load("lake_lag_atl15_itslive", "lake_lag_atl15_itslive.py")
_MM = _load("lake_lag_itslive_match", "lake_lag_itslive_match.py")

ATL15_GLOB = _ATL.ATL15_GLOB
SF2018_H5 = _ATL.SF2018_H5
OBS_LAG_YR = _MM.OBS_LAG_YR
K_SIGMA = _MM.K_SIGMA

MOSAIC_URL = ("https://its-live-data.s3.amazonaws.com/velocity_mosaic/landsat/"
              "v00.0/static/ANT_G0240_0000.nc")
STEP_M = 500.0           # flowline integration step [m]
MAX_TRACE_M = 25000.0    # trace at most this far downstream [m] (a drainage signal
#                          propagates coherently only ~a few ice thicknesses /
#                          tens of km; far-downstream fast ice is a different
#                          dynamic unit and gives spurious blips)
MIN_TRUNK_M = 2000.0     # ignore the first ~2 km (still "over" the lake)
TRUNK_MIN_SPEED = 30.0   # don't bother if the whole flowline is < this [m/yr]
TRUNK_DISTS_M = (5000.0, 10000.0, 15000.0, 20000.0, 25000.0)  # systematic samples


# --------------------------------------------------------------------------- #
# ITS_LIVE 240 m static mosaic: on-demand (vx, vy, v) sampling for flowlines
# --------------------------------------------------------------------------- #
class Mosaic:
    """Chunk-cached reader of the ITS_LIVE ANT_G0240 mosaic (vx, vy, v) over HTTPS.

    Reads only the 64x64 HDF5 chunks the flowline touches (block cache), so the
    5.4 GB file is never downloaded whole.  One open handle is reused for a run.
    """

    def __init__(self, url=MOSAIC_URL, block_mb=4):
        import fsspec
        import h5py
        self._f = fsspec.filesystem("https").open(
            url, block_size=block_mb * 1024 * 1024, cache_type="blockcache")
        self._h = h5py.File(self._f, "r")
        self.x = self._h["x"][:]
        self.y = self._h["y"][:]
        self.x0, self.dx = float(self.x[0]), float(self.x[1] - self.x[0])
        self.y0, self.dy = float(self.y[0]), float(self.y[1] - self.y[0])
        self._dv = {k: self._h[k] for k in ("vx", "vy", "v")}
        self._cy, self._cx = self._h["v"].chunks
        self._cache = {}

    def close(self):
        try:
            self._h.close()
        finally:
            self._f.close()

    def _chunk(self, var, cr, cc):
        key = (var, cr, cc)
        bb = self._cache.get(key)
        if bb is None:
            d = self._dv[var]
            r0, c0 = cr * self._cy, cc * self._cx
            bb = np.asarray(d[r0:r0 + self._cy, c0:c0 + self._cx], dtype="f4")
            bb[(bb < -1e5) | (bb > 1e5)] = np.nan
            self._cache[key] = bb
        return bb

    def at(self, px, py):
        """Nearest-cell (vx, vy, v) at projected (px, py) [m]; nan outside grid."""
        ix = int(round((px - self.x0) / self.dx))
        iy = int(round((py - self.y0) / self.dy))
        if not (0 <= ix < self.x.size and 0 <= iy < self.y.size):
            return np.nan, np.nan, np.nan
        cr, cc = iy // self._cy, ix // self._cx
        rr, rc = iy - cr * self._cy, ix - cc * self._cx
        vx = float(self._chunk("vx", cr, cc)[rr, rc])
        vy = float(self._chunk("vy", cr, cc)[rr, rc])
        v = float(self._chunk("v", cr, cc)[rr, rc])
        return vx, vy, v

    def flowline(self, x0, y0, step=STEP_M, max_m=MAX_TRACE_M):
        """Trace downstream (along +(vx,vy)) from (x0,y0). Returns list of
        dicts {x,y,v,dist} (RK2 integration on the mosaic flow field)."""
        pts = []
        x, y, dist = float(x0), float(y0), 0.0
        for _ in range(int(max_m / step) + 1):
            vx, vy, v = self.at(x, y)
            if not np.isfinite(v) or v <= 0:
                break
            pts.append(dict(x=x, y=y, v=v, dist=dist))
            sp = np.hypot(vx, vy)
            if sp <= 0:
                break
            # RK2 (midpoint) step downstream
            ux, uy = vx / sp, vy / sp
            mx, my = x + 0.5 * step * ux, y + 0.5 * step * uy
            vmx, vmy, vm = self.at(mx, my)
            smp = np.hypot(vmx, vmy)
            if np.isfinite(smp) and smp > 0:
                ux, uy = vmx / smp, vmy / smp
            x += step * ux
            y += step * uy
            dist += step
        return pts


# --------------------------------------------------------------------------- #
# Bedmap2 effective-pressure proxy rel = m/H at lake centroids
# --------------------------------------------------------------------------- #
def lake_rel(centroids_xy, bin_dir, phi=0.9, stride=4, search_m=8000.0):
    """rel = m/H (proxy for normalized N) at each lake (cx,cy) [3031], nearest
    grounded cell within search_m. centroids_xy: (n,2) array."""
    from scipy.spatial import cKDTree
    from external.bedmap2_loader import load_fields
    from external.rtn_intrusion_clock import margin_field
    d = load_fields(bin_dir, stride=stride)
    H, bed, mask = d["thickness"], d["bed"], d["icemask_grounded_and_shelves"]
    meta = d["_meta"]
    ny, nx = H.shape
    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)
    margin, Hstar, dbase = margin_field(H, bed, phi)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(grounded & (H > 0), margin / H, np.nan)
    # grid coords: north-anchored y counts down from top row
    cs = meta["cellsize"]
    xll, yll = meta["xll"], meta["yll"]
    nrows_full = meta["nrows_full"]
    xs = xll + cs * (np.arange(nx) + 0.5)
    ytop = yll + meta["cellsize_full"] * nrows_full
    ys = ytop - cs * (np.arange(ny) + 0.5)
    XX, YY = np.meshgrid(xs, ys)
    valid = grounded & np.isfinite(rel)
    tree = cKDTree(np.c_[XX[valid], YY[valid]])
    rel_v, H_v = rel[valid], H[valid]
    cxy = np.asarray(centroids_xy, float)
    out_rel = np.full(len(cxy), np.nan)
    out_H = np.full(len(cxy), np.nan)
    finite = np.isfinite(cxy).all(axis=1)
    if finite.any():
        dist, idx = tree.query(cxy[finite], distance_upper_bound=search_m)
        hit = np.isfinite(dist)
        fi = np.where(finite)[0]
        out_rel[fi[hit]] = rel_v[idx[hit]]
        out_H[fi[hit]] = H_v[idx[hit]]
    return out_rel, out_H


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _sample_event(itl, lon, lat, x, y, t_peak):
    """Quarterly response test at one (x,y) sample point for a drainage at t_peak."""
    try:
        yr, v, dtp = itl.series(lon, lat, x, y)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    modern = yr >= 2018.0
    tq, vq, nq = _MM.bin_series(yr[modern], v[modern], dtp[modern], 0.25)
    nf, med = _MM.noise_floor_frac(tq, vq)
    a = _MM.analyse_event(tq, vq, nq, t_peak)
    a["median_speed"] = round(float(np.median(v[modern])) if modern.any() else float("nan"), 1)
    a["noise_floor_qtr"] = round(nf, 4) if np.isfinite(nf) else None
    return a, None


def run(h5=SF2018_H5, atl15_glob=ATL15_GLOB, bin_dir=None, only=None,
        n_trunk=5, verbose=True):
    bin_dir = bin_dir or os.path.expanduser("~/data_bedmap/bedmap2_bin")
    outlines = _ATL.lake_outlines(h5)
    import glob
    files = glob.glob(atl15_glob)
    if not files:
        raise FileNotFoundError(f"no ATL15 files at {atl15_glob}")
    atl = _ATL.ATL15(files)
    itl = _MM.ITSLive()
    names = sorted(outlines) if only is None else [n for n in sorted(outlines) if n in only]

    # effective-pressure proxy at every lake centroid (one Bedmap2 load)
    cxy = np.array([[float(np.mean(outlines[n]["x"])),
                     float(np.mean(outlines[n]["y"]))] for n in names])
    rel_all, H_all = lake_rel(cxy, bin_dir)
    rel_by = {n: (float(rel_all[i]) if np.isfinite(rel_all[i]) else None,
                  float(H_all[i]) if np.isfinite(H_all[i]) else None)
              for i, n in enumerate(names)}

    mos = Mosaic()
    lakes_out, detections = {}, []
    n_drained = n_testable = 0
    amp_bounds = []
    t_start = time.time()
    try:
        for k, lake in enumerate(names):
            o = outlines[lake]
            cx, cy = float(np.mean(o["x"])), float(np.mean(o["y"]))
            try:
                t, dh, npix = atl.inlake_series(o["x"], o["y"])
            except Exception as exc:
                lakes_out[lake] = dict(error=f"ATL15: {type(exc).__name__}: {exc}")
                continue
            evs, _ = _ATL.detect_drainages(t, dh)
            relv, Hv = rel_by[lake]
            rec = dict(lon=round(o["lon"], 3), lat=round(o["lat"], 3),
                       rel=relv, H=Hv, n_modern_drainages=len(evs))
            if not evs:
                lakes_out[lake] = rec
                continue
            n_drained += 1
            # flowline + downstream trunk sample points
            fl = mos.flowline(cx, cy)
            vmax = max((p["v"] for p in fl), default=0.0)
            sample_pts = [dict(x=cx, y=cy, dist=0.0, v=(fl[0]["v"] if fl else None),
                               label="centroid")]
            if fl and vmax >= TRUNK_MIN_SPEED:
                # systematic samples at fixed downstream distances within the
                # coherent-propagation zone (nearest flowline point to each target)
                for target in TRUNK_DISTS_M[:n_trunk]:
                    cand = [p for p in fl if p["dist"] >= MIN_TRUNK_M]
                    if not cand:
                        break
                    p = min(cand, key=lambda q: abs(q["dist"] - target))
                    if abs(p["dist"] - target) > 3000.0:   # flowline ended early
                        continue
                    if any(abs(sp["dist"] - p["dist"]) < 1500.0 for sp in sample_pts):
                        continue
                    sample_pts.append(dict(x=p["x"], y=p["y"], dist=p["dist"],
                                           v=p["v"], label="trunk"))
            rec["flowline_len_km"] = round(fl[-1]["dist"] / 1000.0, 1) if fl else 0.0
            rec["trunk_speed_mosaic"] = round(vmax, 1)
            # test every drainage at every sample point
            best = None
            ev_recs = []
            for e in evs:
                pt_res = []
                for sp in sample_pts:
                    a, err = _sample_event(itl, o["lon"], o["lat"], sp["x"], sp["y"],
                                           e["t_peak"])
                    if a is None:
                        pt_res.append(dict(label=sp["label"], dist_km=round(sp["dist"]/1000.0,1),
                                           error=err))
                        continue
                    if a.get("testable"):
                        n_testable += 1
                        if a.get("baseline") and a.get("sigma_bin_mpyr"):
                            amp_bounds.append(a["sigma_bin_mpyr"] / a["baseline"])
                    entry = dict(label=sp["label"], dist_km=round(sp["dist"]/1000.0, 1),
                                 mosaic_speed=round(sp["v"], 1) if sp["v"] else None,
                                 **a)
                    pt_res.append(entry)
                    if a.get("in_band"):
                        cand = dict(lake=lake, t_peak=round(e["t_peak"], 2),
                                    drop_m=e["drop_m"], label=sp["label"],
                                    dist_km=round(sp["dist"]/1000.0, 1),
                                    resp_frac=a.get("resp_frac"),
                                    lag_to_peak=a.get("lag_to_peak"),
                                    peak_sigma=a.get("peak_sigma"),
                                    sustained_sigma=a.get("sustained_sigma"),
                                    detrend_sigma=a.get("detrend_sigma"),
                                    sustained_detrend_sigma=a.get("sustained_detrend_sigma"),
                                    trunk_speed=a.get("median_speed"),
                                    noise_floor=a.get("noise_floor"),
                                    rel=relv, H=Hv)
                        # keep the strongest (by detrend_sigma) detection for this lake
                        score = a.get("detrend_sigma") or 0
                        if best is None or score > best[0]:
                            best = (score, cand)
                ev_recs.append(dict(drainage=e, points=pt_res))
            rec["events"] = ev_recs
            if best is not None:
                detections.append(best[1])
                rec["detection"] = best[1]
            lakes_out[lake] = rec
            if verbose:
                d = rec.get("detection")
                tag = (f"DETECT {d['label']}@{d['dist_km']}km du/u={d['resp_frac']} "
                       f"lag={d['lag_to_peak']}yr {d['detrend_sigma']}sd" if d else "-")
                print(f"[{k+1}/{len(names)}] {lake}: drain={len(evs)} "
                      f"trunk={rec.get('trunk_speed_mosaic')} rel={relv} {tag} "
                      f"({time.time()-t_start:.0f}s)", flush=True)
    finally:
        mos.close()

    return _summarise(lakes_out, detections, amp_bounds, n_drained, n_testable, names)


def _fit_amp(detections):
    """Calibrate surge amplitude du/u vs effective-pressure proxy rel, trunk speed,
    and drained depth. Returns OLS log-log / semilog slopes."""
    d = [x for x in detections if x.get("resp_frac") and x["resp_frac"] > 0]
    out = {"n_detections": len(d)}
    if len(d) < 3:
        out["note"] = "need >=3 detections to fit"
        return out
    amp = np.array([x["resp_frac"] for x in d])
    rel = np.array([x["rel"] if x.get("rel") else np.nan for x in d])
    spd = np.array([x["trunk_speed"] if x.get("trunk_speed") else np.nan for x in d])
    drop = np.array([x["drop_m"] if x.get("drop_m") else np.nan for x in d])

    def slope(xv, yv, logx=True):
        m = np.isfinite(xv) & np.isfinite(yv) & (yv > 0) & (xv > 0 if logx else True)
        if m.sum() < 3:
            return None
        X = np.log(xv[m]) if logx else xv[m]
        Y = np.log(yv[m])
        b = np.polyfit(X, Y, 1)
        r = float(np.corrcoef(X, Y)[0, 1])
        return {"slope": float(b[0]), "r": r, "n": int(m.sum())}

    out["logamp_vs_logrel"] = slope(rel, amp, True)   # predict <0 (amp up as N->0)
    out["logamp_vs_logspeed"] = slope(spd, amp, True)
    out["logamp_vs_logdrop"] = slope(drop, amp, True)

    # rel can be <=0 near flotation (grounded-but-intrudable band), so also fit
    # log(amp) on rel LINEARLY (semilog): slope<0 => amplitude rises toward
    # flotation (low N), the §H.1.6 |s_N|-steepening prediction.
    def semilog(xv, yv):
        m = np.isfinite(xv) & np.isfinite(yv) & (yv > 0)
        if m.sum() < 3:
            return None
        b = np.polyfit(xv[m], np.log(yv[m]), 1)
        r = float(np.corrcoef(xv[m], np.log(yv[m]))[0, 1])
        return {"slope_per_unit_rel": float(b[0]), "r": r, "n": int(m.sum())}

    out["logamp_vs_rel_semilog"] = semilog(rel, amp)
    out["amp_median"] = float(np.median(amp))
    out["amp_range"] = [float(amp.min()), float(amp.max())]
    out["rel_range"] = [float(np.nanmin(rel)), float(np.nanmax(rel))]
    return out


def _summarise(lakes_out, detections, amp_bounds, n_drained, n_testable, names):
    median_amp_bound = float(np.median(amp_bounds)) if amp_bounds else float("nan")
    # one detection per lake (strongest), for the population fit
    by_lake = {}
    for d in detections:
        cur = by_lake.get(d["lake"])
        if cur is None or (d.get("detrend_sigma") or 0) > (cur.get("detrend_sigma") or 0):
            by_lake[d["lake"]] = d
    uniq = list(by_lake.values())
    fit = _fit_amp(uniq)
    trunk_only = [d for d in uniq if d["label"] == "trunk"]
    return {
        "n_lakes": len(names),
        "n_lakes_drained": n_drained,
        "n_testable_event_points": n_testable,
        "n_detections_total": len(detections),
        "n_lakes_with_detection": len(uniq),
        "n_detections_on_trunk": len(trunk_only),
        "detections": sorted(uniq, key=lambda d: -(d.get("detrend_sigma") or 0)),
        "median_amplitude_bound_frac": (round(median_amp_bound, 4)
                                        if np.isfinite(median_amp_bound) else None),
        "amplitude_calibration": fit,
        "obs_lag_band_yr": list(OBS_LAG_YR),
        "k_sigma": K_SIGMA,
        "method": ("downstream-trunk flowline sampling of ITS_LIVE velocity response "
                   "to ATL15-dated lake drainages; amplitude du/u regressed on the "
                   "effective-pressure proxy rel=m/H (Bedmap2)"),
        "lakes": lakes_out,
    }


def make_figure(summary, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    det = summary["detections"]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
    if det:
        rel = [d["rel"] for d in det if d.get("rel")]
        amp = [d["resp_frac"] for d in det if d.get("rel")]
        labels = [d["lake"] for d in det if d.get("rel")]
        sc = ax[0].scatter(rel, np.array(amp) * 100, s=60, c="#d62728",
                           edgecolor="k", linewidth=0.4, zorder=3)
        for r, a, l in zip(rel, amp, labels):
            ax[0].annotate(l, (r, a * 100), fontsize=7, xytext=(3, 3),
                           textcoords="offset points")
        fit = summary["amplitude_calibration"].get("logamp_vs_logrel")
        if fit and len(rel) >= 3:
            xs = np.array([min(rel), max(rel)])
            b = np.polyfit(np.log(rel), np.log(amp), 1)
            ax[0].plot(xs, 100 * np.exp(b[1]) * xs ** b[0], "k--",
                       label=f"slope={fit['slope']:+.2f}, r={fit['r']:+.2f}")
            ax[0].legend(fontsize=9)
    ax[0].set_xlabel("effective-pressure proxy  rel = m/H  (low = near flotation)")
    ax[0].set_ylabel("post-drainage surge amplitude  du/u  [%]")
    ax[0].set_title("(a) amplitude calibration: surge grows toward flotation?")
    ax[0].grid(alpha=0.3)
    # lag-vs-distance scatter
    if det:
        dist = [d["dist_km"] for d in det]
        lag = [d["lag_to_peak"] for d in det]
        ax[1].scatter(dist, lag, s=55, c="#1f77b4", edgecolor="k", linewidth=0.4)
        ax[1].axhspan(OBS_LAG_YR[0], OBS_LAG_YR[1], color="#d95f0e", alpha=0.15,
                      label="derived 0.02-2 yr band")
        ax[1].legend(fontsize=9)
    ax[1].set_xlabel("downstream distance of detection [km]")
    ax[1].set_ylabel("lag-to-peak [yr]")
    ax[1].set_title("(b) detections: lag vs downstream distance")
    ax[1].grid(alpha=0.3)
    fig.suptitle("§G.4 downstream-trunk lake-drainage surge detections + amplitude "
                 "calibration", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=SF2018_H5)
    ap.add_argument("--atl15", default=ATL15_GLOB)
    ap.add_argument("--bin-dir", default=os.path.expanduser("~/data_bedmap/bedmap2_bin"))
    ap.add_argument("--only", default=None, help="comma list of lake names (debug)")
    ap.add_argument("--n-trunk", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(_REPORTS, "lake_lag_trunk.json"))
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    summary = run(a.h5, a.atl15, a.bin_dir, only=only, n_trunk=a.n_trunk)
    print("\n=== §G.4 downstream-trunk matched lag + amplitude calibration ===")
    print(f"lakes {summary['n_lakes']}  drained {summary['n_lakes_drained']}  "
          f"testable-pts {summary['n_testable_event_points']}")
    print(f"detections: {summary['n_detections_total']} "
          f"({summary['n_lakes_with_detection']} lakes, "
          f"{summary['n_detections_on_trunk']} on trunk)")
    for d in summary["detections"]:
        print(f"   {d['lake']:14s} {d['label']:8s} @{d['dist_km']:5.1f}km "
              f"du/u={d['resp_frac']} lag={d['lag_to_peak']}yr "
              f"detrendσ={d['detrend_sigma']} rel={d['rel']} spd={d['trunk_speed']}")
    print("amplitude_calibration:", json.dumps(summary["amplitude_calibration"], indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"json -> {a.out}")
    make_figure(summary, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
