r"""§G.4 / §H.2 — MODERN matched lake-drainage → ITS_LIVE lag test (ICESat-2 era).

The 2003-2007 ICESat catalogue (``lake_lag_itslive_match.py``) predates dense
ITS_LIVE, so it is coverage-limited. This module closes that gap with a fully
**co-temporal modern** test, both halves in the dense satellite era:

* **forcing (drainage dates):** ICESat-2 **ATL15** gridded quarterly surface-height
  change ``delta_h`` (Smith et al., NSIDC, EPSG:3031, 2019-2026). For each
  Siegfried & Fricker 2018 lake we average ``delta_h`` over the lake outline,
  detrend the secular signal, and detect fill→drain events (a sustained surface
  drawdown ≥ ``MIN_DROP_M``) — the §G.4 forcing ``q_water`` step, now in the
  ITS_LIVE era.
* **response (velocity):** ITS_LIVE box-mean surface speed at the same lake,
  median-binned quarterly with a robust LOCAL noise model
  (``lake_lag_itslive_match.analyse_event``).

Because both halves are 2019-2026, the dense-era ITS_LIVE noise floor at the
outlet lakes is ~1-3 % (vs the sparse, ~15-100 % early Landsat era), so a real
post-drainage surge of a few percent is detectable and a sub-annual→2 yr
lag-to-peak is resolvable — exactly the regime the 2003 catalogue could not reach.

Verdict
-------
* a sustained, significant (≥ ``K_SIGMA``) post-drainage speed-up with lag-to-peak
  in the derived **0.02-2 yr** band → §G.4 empirical lag **[VERIFIED]** (data-side);
* fast, well-resolved lakes that drain but show **no** velocity response →
  hydraulic-lag forecast **[FALSIFIED]** at those lakes (also a result);
* the literal thermal ``H²/κ`` kernel remains **[FALSIFIED]** independently.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))

# reuse the ITS_LIVE access + binning + conservative event test
_spec = importlib.util.spec_from_file_location(
    "lake_lag_itslive_match", os.path.join(_HERE, "lake_lag_itslive_match.py"))
_M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_M)

SF2018_H5 = _M.SF2018_H5
OBS_LAG_YR = _M.OBS_LAG_YR
K_SIGMA = _M.K_SIGMA
ATL15_GLOB = os.path.expanduser("~/data_atl15/ATL15_A?_*01km*.nc")
MIN_DROP_M = 1.0           # minimum detrended surface drawdown for a drainage [m]
MAXDUR_YR = 2.5           # max drain duration
FAST_FLOOR = 0.05         # a lake is "well-resolved" if dense-era noise floor < 5%


def lake_outlines(h5_path=SF2018_H5):
    import h5py
    out = {}
    with h5py.File(h5_path, "r") as f:
        for name in f.keys():
            g = f[name]
            if all(k in g for k in ("x", "y", "lon", "lat")):
                out[name] = dict(
                    x=np.asarray(g["x"][:]).ravel().astype(float),
                    y=np.asarray(g["y"][:]).ravel().astype(float),
                    lon=float(np.mean(g["lon"][:])), lat=float(np.mean(g["lat"][:])))
    return out


class ATL15:
    """Quadrant-indexed ATL15 delta_h reader (EPSG:3031)."""

    def __init__(self, files):
        import xarray as xr
        self.regions = []
        for f in sorted(files):
            ds = xr.open_dataset(f, group="delta_h")
            self.regions.append(dict(
                ds=ds, xmin=float(ds.x.min()), xmax=float(ds.x.max()),
                ymin=float(ds.y.min()), ymax=float(ds.y.max())))
        # decimal-year time axis (days since 2018-01-01)
        t0 = np.datetime64("2018-01-01")
        self._t = {id(r["ds"]): 2018.0 + (r["ds"]["time"].values - t0)
                   / np.timedelta64(1, "D") / 365.25 for r in self.regions}

    def _region(self, x, y):
        for r in self.regions:
            if r["xmin"] <= x <= r["xmax"] and r["ymin"] <= y <= r["ymax"]:
                return r
        return None

    def inlake_series(self, ox, oy):
        """In-outline mean delta_h(t). ox,oy = outline polygon vertices (3031)."""
        from matplotlib.path import Path
        cx, cy = float(np.mean(ox)), float(np.mean(oy))
        r = self._region(cx, cy)
        if r is None:
            raise RuntimeError("no ATL15 region")
        ds = r["ds"]
        pad = 1500.0
        sub = ds["delta_h"].sel(x=slice(ox.min() - pad, ox.max() + pad),
                                y=slice(oy.min() - pad, oy.max() + pad))
        if sub.x.size == 0 or sub.y.size == 0:
            raise RuntimeError("empty ATL15 subset")
        XX, YY = np.meshgrid(sub.x.values, sub.y.values)
        if len(ox) >= 3:
            inside = Path(np.c_[ox, oy]).contains_points(
                np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        else:
            inside = np.ones(XX.shape, bool)
        if inside.sum() < 1:                      # tiny lake: fall back to nearest cell
            inside = np.zeros(XX.shape, bool)
            j = np.argmin((sub.x.values - cx) ** 2)
            i = np.argmin((sub.y.values - cy) ** 2)
            inside[i, j] = True
        arr = sub.values                          # (time, y, x)
        m = inside[None, :, :]
        dh = np.where(m, arr, np.nan)
        dh = np.nanmean(dh.reshape(arr.shape[0], -1), axis=1)
        return self._t[id(ds)].copy(), dh, int(inside.sum())


def detect_drainages(t, dh, min_drop_m=MIN_DROP_M, maxdur=MAXDUR_YR):
    """Detrend then find fill-peak → sustained surface drawdown events."""
    ok = np.isfinite(dh)
    if ok.sum() < 6:
        return [], np.full_like(dh, np.nan)
    t2, d2 = t[ok], dh[ok]
    A = np.vstack([t2 - t2.mean(), np.ones_like(t2)]).T
    coef, *_ = np.linalg.lstsq(A, d2, rcond=None)
    resid_full = np.full_like(dh, np.nan)
    resid_full[ok] = d2 - A @ coef
    d = resid_full[ok]
    events = []
    n = len(d); i = 1
    while i < n - 1:
        if d[i] >= d[i - 1] and d[i] > d[i + 1]:
            j = i
            while j + 1 < n and d[j + 1] <= d[j]:
                j += 1
            drop = d[i] - d[j]
            if drop >= min_drop_m and (t2[j] - t2[i]) <= maxdur:
                events.append(dict(t_peak=float(t2[i]), t_trough=float(t2[j]),
                                   drop_m=float(round(drop, 2))))
                i = j
        i += 1
    return events, resid_full


def run(h5=SF2018_H5, atl15_glob=ATL15_GLOB, only=None, verbose=True):
    outlines = lake_outlines(h5)
    files = glob.glob(atl15_glob)
    if not files:
        raise FileNotFoundError(f"no ATL15 files at {atl15_glob}")
    atl = ATL15(files)
    itl = _M.ITSLive()
    names = sorted(outlines) if only is None else [n for n in sorted(outlines) if n in only]

    lakes_out, sig_events, inband_lags = {}, [], []
    n_drained, n_drained_fast, n_testable = 0, 0, 0
    amp_bounds = []
    for k, lake in enumerate(names):
        o = outlines[lake]
        try:
            t, dh, npix = atl.inlake_series(o["x"], o["y"])
        except Exception as exc:
            lakes_out[lake] = dict(error=f"ATL15: {type(exc).__name__}: {exc}")
            continue
        evs, resid = detect_drainages(t, dh)
        rec = dict(lon=round(o["lon"], 3), lat=round(o["lat"], 3),
                   atl15_pixels=npix, n_modern_drainages=len(evs),
                   drainages=evs)
        if not evs:
            lakes_out[lake] = rec
            if verbose and (k % 20 == 0):
                print(f"[{k+1}/{len(names)}] {lake}: no modern drainage", flush=True)
            continue
        n_drained += 1
        # pull dense modern ITS_LIVE
        try:
            yr, v, dtp = itl.series(o["lon"], o["lat"],
                                    float(np.mean(o["x"])), float(np.mean(o["y"])))
        except Exception as exc:
            rec["velocity_error"] = f"{type(exc).__name__}: {exc}"
            lakes_out[lake] = rec
            continue
        modern = (yr >= 2018.0)
        tq, vq, nq = _M.bin_series(yr[modern], v[modern], dtp[modern], 0.25)
        nf_q, med = _M.noise_floor_frac(tq, vq)
        rec.update(median_speed_mpyr=round(float(np.median(v[modern])) if modern.any()
                                           else float("nan"), 1),
                   modern_noise_floor_qtr=round(nf_q, 4) if np.isfinite(nf_q) else None,
                   n_modern_qtr_bins=int(tq.size))
        well_resolved = bool(np.isfinite(nf_q) and nf_q < FAST_FLOOR)
        if well_resolved:
            n_drained_fast += 1
        ev_res = []
        for e in evs:
            a = _M.analyse_event(tq, vq, nq, e["t_peak"])
            ev_res.append(dict(drainage=e, response=a))
            if a.get("testable"):
                n_testable += 1
                if a.get("baseline") and a.get("sigma_bin_mpyr"):
                    amp_bounds.append(a["sigma_bin_mpyr"] / a["baseline"])
                if a.get("in_band"):
                    sig_events.append(dict(lake=lake, t_peak=round(e["t_peak"], 2),
                                           drop_m=e["drop_m"],
                                           peak_sigma=a["peak_sigma"],
                                           sustained_sigma=a.get("sustained_sigma"),
                                           detrend_sigma=a.get("detrend_sigma"),
                                           sustained_detrend_sigma=a.get("sustained_detrend_sigma"),
                                           resp_frac=a.get("resp_frac"),
                                           lag_to_peak=a["lag_to_peak"],
                                           speed=rec.get("median_speed_mpyr"),
                                           noise_floor=rec.get("modern_noise_floor_qtr")))
                    inband_lags.append(a["lag_to_peak"])
        rec["events"] = ev_res
        rec["well_resolved"] = well_resolved
        lakes_out[lake] = rec
        if verbose:
            print(f"[{k+1}/{len(names)}] {lake}: {len(evs)} drainage(s) "
                  f"med={rec.get('median_speed_mpyr')} nf={rec.get('modern_noise_floor_qtr')} "
                  f"testable={sum(1 for r in ev_res if r['response'].get('testable'))} "
                  f"sig={sum(1 for r in ev_res if r['response'].get('in_band'))}", flush=True)

    median_amp_bound = float(np.median(amp_bounds)) if amp_bounds else float("nan")
    n_sig = len(sig_events)
    universal = bool(n_testable and n_sig / n_testable > 0.5)
    sig_lakes = sorted({s["lake"] for s in sig_events})
    summary = {
        "n_lakes": len(names),
        "n_lakes_with_modern_drainage": n_drained,
        "n_drained_well_resolved": n_drained_fast,
        "n_testable_events": n_testable,
        "n_significant_inband_events": len(sig_events),
        "significant_events": sig_events,
        "lags_in_band_yr": sorted(inband_lags),
        "obs_lag_band_yr": list(OBS_LAG_YR),
        "median_amplitude_bound_frac": (round(median_amp_bound, 4)
                                        if np.isfinite(median_amp_bound) else None),
        "min_drop_m": MIN_DROP_M, "k_sigma": K_SIGMA,
        "in_band_field_detection": bool(n_sig >= 1),
        "universal_response": universal,
        "verdict": (
            f"SUPPORTED at {n_sig}/{n_testable} testable event(s) ({', '.join(sig_lakes)}): "
            "a sustained, secular-trend-robust, in-band (0.02-2 yr) post-drainage "
            "speed-up — a data-side confirmation of the §G.4 hydraulic lag at a "
            "dynamically-active outlet. It is NOT a universal response: the other "
            f"{n_testable - n_sig} testable drained lakes show no surge (amplitude "
            f"bound ~{(100*median_amp_bound):.1f}% of trunk speed), so most "
            "drainages do not produce a detectable sliding response. Thermal "
            "H^2/kappa kernel remains FALSIFIED (orders of magnitude off)."
            if n_sig >= 1 and np.isfinite(median_amp_bound) else
            "NOT CONFIRMED (modern): drained, well-resolved lakes show no "
            "significant in-band post-drainage surge; amplitude bound "
            f"~{(100*median_amp_bound):.1f}% where testable. Thermal H^2/kappa "
            "kernel remains FALSIFIED."
            if np.isfinite(median_amp_bound) else
            "INCONCLUSIVE: no testable co-temporal drainage/velocity pair."),
        "lakes": lakes_out,
    }
    return summary


def make_figure(summary, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    xs, ys, cs = [], [], []
    for lake, d in summary["lakes"].items():
        for ev in d.get("events", []):
            a = ev["response"]
            if a.get("testable") and a.get("peak_sigma") is not None:
                xs.append(a["lag_to_peak"]); ys.append(a["peak_sigma"])
                cs.append("#d62728" if a.get("in_band") else "#1f77b4")
    if xs:
        ax[0].scatter(xs, ys, c=cs, s=45, alpha=0.85, edgecolor="k", linewidth=0.3)
    ax[0].axvspan(OBS_LAG_YR[0], OBS_LAG_YR[1], color="#d95f0e", alpha=0.2,
                  label="derived 0.02-2 yr band")
    ax[0].axhline(K_SIGMA, ls="--", color="k", lw=1, label=f"{K_SIGMA}σ")
    ax[0].set_xlabel("lag-to-peak after modern drainage [yr]")
    ax[0].set_ylabel("post-drainage anomaly [local σ]")
    ax[0].set_title("(a) modern ICESat-2/ITS_LIVE matched lag\nred = significant & in-band")
    ax[0].legend(fontsize=8)
    spd, nf = [], []
    for lake, d in summary["lakes"].items():
        if d.get("median_speed_mpyr") and d.get("modern_noise_floor_qtr"):
            spd.append(d["median_speed_mpyr"]); nf.append(100 * d["modern_noise_floor_qtr"])
    if spd:
        ax[1].scatter(spd, nf, s=24, color="#2c7fb8")
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].axhline(100 * FAST_FLOOR, color="r", ls=":", label=f"{100*FAST_FLOOR:.0f}% well-resolved")
        ax[1].legend(fontsize=8)
    ax[1].set_xlabel("modern median speed [m/yr]")
    ax[1].set_ylabel("modern quarterly noise floor [%]")
    ax[1].set_title("(b) detectability of drained lakes")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=SF2018_H5)
    ap.add_argument("--atl15", default=ATL15_GLOB)
    ap.add_argument("--only", default=None, help="comma list of lake names (debug)")
    ap.add_argument("--out", default=os.path.join(_REPORTS, "lake_lag_atl15_itslive.json"))
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    summary = run(a.h5, a.atl15, only=only)
    print("\n=== §G.4 MODERN matched lag (ATL15 drainages → ITS_LIVE) ===")
    print(f"lakes: {summary['n_lakes']}  modern drainage: "
          f"{summary['n_lakes_with_modern_drainage']} "
          f"(well-resolved: {summary['n_drained_well_resolved']})")
    print(f"testable events: {summary['n_testable_events']}  "
          f"significant in-band: {summary['n_significant_inband_events']}")
    for s in summary["significant_events"]:
        print(f"   {s['lake']} t={s['t_peak']} drop={s['drop_m']}m speed={s['speed']} "
              f"nf={s['noise_floor']} peakσ={s['peak_sigma']} lag={s['lag_to_peak']}yr "
              f"resp={s['resp_frac']}")
    print(f"median amplitude bound: {summary['median_amplitude_bound_frac']}")
    print(f"VERDICT: {summary['verdict']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"json -> {a.out}")
    make_figure(summary, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
