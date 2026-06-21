r"""§G.4 / §H.2 — matched lake-drainage → ITS_LIVE velocity lag test on the FULL
vetted catalogue, including the **fast outlets** (Byrd, David, Totten, Cook,
Lambert, Rutford, Slessor, Foundation, …).

Why this exists (what is new vs ``lag_fit_real.py``)
---------------------------------------------------
``lag_fit_real.py`` ran the matched §G.4 lag test on the **five open CryoSat-2
lakes** (MacAyeal ``Mac1-3``, ``Mercer``, ``Conway``) and returned a NULL —
*below detection*. It named the gate that would beat the amplitude floor:

    "[VERIFIED] needs … fast-outlet *trunk* velocity (e.g. Byrd, Stearns et al.
     2008) paired with that outlet's drainage dates."

This module runs **exactly that gate**: the Smith et al. (2009) / Siegfried &
Fricker (2018) drainage catalogue (``reports/usapdc_lakes_events.csv``, 58
events, 2003-2007) matched to **ITS_LIVE** surface speed at every catalogued
lake we have a centroid for, with the outlets (where the absolute trunk speed is
large and ITS_LIVE is dense back into the ICESat era) carrying the test.

Method (per lake)
-----------------
Raw ITS_LIVE *image-pair* speeds are very noisy point-wise (per-pair box-mean
scatter ≫ any plausible surge), so a peak-of-raw-pairs test is meaningless. We
therefore **time-bin** the box-mean speed (median of all image pairs with a
sensible separation ``date_dt`` ∈ [30, 400] d) at two resolutions:

* **quarterly (0.25 yr)** — resolves a sub-annual→2 yr lag-to-peak;
* **annual (1.0 yr)** — robust amplitude (averages down per-pair noise).

A global linear detrend of the binned series gives the robust fractional **noise
floor** (1.4826·MAD of residuals / median speed). For each drainage event we
compare post-drainage bins in ``[t0, t0+POST]`` to a local pre-drainage baseline
``median([t0-PRE, t0])``, recording the peak anomaly (in σ) and its lag-to-peak.

Verdict
-------
* significant (≥ ``K_SIGMA``) post-drainage surge with lag-to-peak in the derived
  **0.02-2 yr** band on a testable outlet → §G.4 empirical lag **[VERIFIED]**;
* otherwise a **quantified amplitude bound** (any sustained surge < k·noise-floor
  of the trunk speed) → forecast **not confirmed on this catalogue**; the literal
  thermal ``H²/κ`` kernel stays **[FALSIFIED]** independently. Either way: a
  result, not a "can't test".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))

EVENTS_CSV = os.path.join(_REPORTS, "usapdc_lakes_events.csv")
# Siegfried & Fricker 2018 outlines (open: mrsiegfried/Siegfried2021-GRL).
SF2018_H5 = os.path.expanduser("~/data_lakes/SiegfriedFricker2018-outlines.h5")
CATALOG_URL = "https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json"

OBS_LAG_YR = (0.02, 2.0)        # derived hydraulic band (hydraulic_lag_derivation)
BOX_M = 2000.0                  # ± box half-width for the centroid speed mean [m]
DT_RANGE_D = (30.0, 400.0)     # keep image pairs with sensible separation
PRE_YR = 2.0                    # pre-drainage baseline window
POST_YR = 2.5                   # post-drainage scan (band upper edge 2 yr + tail)
MAXLAG_YR = 2.0                 # lag-to-peak counts as "in band" up to here
MIN_PER_BIN = 6                # ≥ this many image pairs to accept a time bin
MIN_PRE_BINS = 2
MIN_POST_BINS = 2
MIN_PAIRS_WIN = 25            # ≥ this many image pairs in EACH of pre/post windows
K_SIGMA = 2.0
FAST_PREFIXES = ("Byrd", "David", "Totten", "Cook", "Lambert", "Rutford",
                 "Slessor", "Foundation", "Recovery", "Ninnis", "Mertz")


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_lake_coords(h5_path=SF2018_H5):
    import h5py
    out = {}
    with h5py.File(h5_path, "r") as f:
        for name in f.keys():
            g = f[name]
            if all(k in g for k in ("x", "y", "lon", "lat")):
                out[name] = dict(
                    x=float(np.mean(g["x"][:])), y=float(np.mean(g["y"][:])),
                    lon=float(np.mean(g["lon"][:])), lat=float(np.mean(g["lat"][:])),
                )
    return out


def load_events(csv_path=EVENTS_CSV):
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(lake=r["lake"], t=float(r["t"]),
                             drop_km3=float(r["drop_km3"]),
                             dur_yr=float(r["dur_yr"]),
                             rate=float(r["rate_km3_per_yr"])))
    return rows


# --------------------------------------------------------------------------- #
# ITS_LIVE
# --------------------------------------------------------------------------- #
class ITSLive:
    """Cached ITS_LIVE datacube access (anon S3 zarr).

    Antarctic ITS_LIVE v2 datacubes are EPSG:3031. The catalog geometry is in
    lon/lat, which is badly distorted near the pole, so cube selection is done in
    projected 3031 coordinates: each cube's lon/lat ring is transformed to 3031
    once and stored as a bbox + center; a lake's 3031 (x, y) picks the
    bbox-containing cube whose center is nearest (tie-break: most granules).
    """

    def __init__(self):
        self._cubes = None
        self._open = {}

    def _load_catalog(self):
        from pyproj import Transformer
        cat = json.load(urllib.request.urlopen(CATALOG_URL, timeout=90))
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
        cubes = []
        for f in cat["features"]:
            p = f["properties"]
            if not p.get("datacube_exist") or str(p.get("epsg")) != "3031":
                continue
            ring = f["geometry"]["coordinates"][0]
            xs, ys = tr.transform([c[0] for c in ring], [c[1] for c in ring])
            xs, ys = np.asarray(xs), np.asarray(ys)
            cubes.append(dict(xmin=float(xs.min()), xmax=float(xs.max()),
                              ymin=float(ys.min()), ymax=float(ys.max()),
                              cx=float(xs.mean()), cy=float(ys.mean()),
                              gc=int(p.get("granule_count", 0)),
                              url=p["zarr_url"]))
        self._cubes = cubes

    @property
    def cubes(self):
        if self._cubes is None:
            self._load_catalog()
        return self._cubes

    def _cube_url(self, x, y):
        cand = [c for c in self.cubes
                if c["xmin"] <= x <= c["xmax"] and c["ymin"] <= y <= c["ymax"]]
        if not cand:
            return None
        cand.sort(key=lambda c: ((x - c["cx"]) ** 2 + (y - c["cy"]) ** 2, -c["gc"]))
        return cand[0]["url"]

    def series(self, lon, lat, x, y, box=BOX_M):
        """Return (yr, v, dt_days) box-mean speed series, time-sorted."""
        import xarray as xr
        zu = self._cube_url(x, y)
        if zu is None:
            raise RuntimeError("no datacube")
        url = zu.replace(
            "https://its-live-data.s3.amazonaws.com/", "s3://its-live-data/")
        ds = self._open.get(url)
        if ds is None:
            ds = xr.open_dataset(url, engine="zarr",
                                 storage_options={"anon": True}, consolidated=True)
            self._open[url] = ds
        sub = ds["v"].sel(x=slice(x - box, x + box), y=slice(y + box, y - box))
        v = sub.mean(dim=("x", "y")).values.astype(float)
        t = ds["mid_date"].values
        dt = ds["date_dt"].values
        if np.issubdtype(np.asarray(dt).dtype, np.timedelta64):
            dt = dt / np.timedelta64(1, "D")
        dt = np.asarray(dt, float)
        yr = 1970.0 + t.astype("datetime64[D]").astype(float) / 365.25
        m = np.isfinite(v) & np.isfinite(yr)
        yr, v, dt = yr[m], v[m], dt[m]
        if v.size == 0:
            raise RuntimeError("empty box series (centroid outside cube coverage)")
        o = np.argsort(yr)
        return yr[o], v[o], dt[o]


# --------------------------------------------------------------------------- #
# binning + noise floor
# --------------------------------------------------------------------------- #
def bin_series(yr, v, dt, bin_yr, dt_range=DT_RANGE_D, min_per_bin=MIN_PER_BIN):
    """Median-bin the (noisy) image-pair box-mean speed to ``bin_yr`` cadence."""
    sel = (dt >= dt_range[0]) & (dt <= dt_range[1])
    if sel.sum() < 10:
        sel = np.ones_like(dt, bool)
    y, vv = yr[sel], v[sel]
    if y.size == 0:
        return np.array([]), np.array([]), np.array([])
    lo = np.floor(y.min() / bin_yr) * bin_yr
    edges = np.arange(lo, y.max() + bin_yr, bin_yr)
    idx = np.digitize(y, edges)
    tb, vb, nb = [], [], []
    for b in range(1, len(edges) + 1):
        mb = idx == b
        if int(mb.sum()) >= min_per_bin:
            tb.append(edges[b - 1] + bin_yr / 2.0)
            vb.append(float(np.median(vv[mb])))
            nb.append(int(mb.sum()))
    return np.array(tb), np.array(vb), np.array(nb)


def noise_floor_frac(tb, vb):
    """Robust fractional scatter of the globally-detrended binned series."""
    if tb.size < 4:
        return float("nan"), float("nan")
    A = np.vstack([tb - tb.mean(), np.ones_like(tb)]).T
    coef, *_ = np.linalg.lstsq(A, vb, rcond=None)
    resid = vb - A @ coef
    med = float(np.median(vb))
    if med <= 0:
        return float("nan"), med
    nf = float(1.4826 * np.median(np.abs(resid - np.median(resid))) / med)
    return nf, med


def _robust_scatter(x):
    x = np.asarray(x, float)
    if x.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def analyse_event(tb, vb, nb, t0, pre=PRE_YR, post=POST_YR, maxlag=MAXLAG_YR):
    """Robust post-drainage response test using a LOCAL noise model.

    The significance of a surge is judged against the *local* bin-to-bin scatter
    in the pre/post windows (a robust two-sample comparison), NOT a global noise
    floor — early-era (sparse Landsat) data is far noisier than the dense modern
    record, so a global floor would manufacture false detections. Requires dense
    co-temporal coverage (``MIN_PAIRS_WIN`` image pairs in each window).
    """
    pre_m = (tb >= t0 - pre) & (tb < t0)
    post_m = (tb >= t0) & (tb <= t0 + post)
    n_pre, n_post = int(pre_m.sum()), int(post_m.sum())
    pre_pairs = int(nb[pre_m].sum()) if n_pre else 0
    post_pairs = int(nb[post_m].sum()) if n_post else 0
    out = dict(t0=round(float(t0), 3), n_pre_bins=n_pre, n_post_bins=n_post,
               pre_pairs=pre_pairs, post_pairs=post_pairs,
               testable=bool(n_pre >= MIN_PRE_BINS and n_post >= MIN_POST_BINS
                             and pre_pairs >= MIN_PAIRS_WIN
                             and post_pairs >= MIN_PAIRS_WIN))
    if not out["testable"]:
        out.update(baseline=None, peak_sigma=None, resp_frac=None,
                   lag_to_peak=None, sustained_sigma=None, in_band=None)
        return out
    vpre, vpost, tpost = vb[pre_m], vb[post_m], tb[post_m]
    tpre = tb[pre_m]
    base = float(np.median(vpre))
    # local per-bin scatter (conservative: the larger of the two windows)
    s_pre, s_post = _robust_scatter(vpre), _robust_scatter(vpost)
    sigma_bin = max([s for s in (s_pre, s_post) if np.isfinite(s)] or [np.nan])
    if not np.isfinite(sigma_bin) or sigma_bin <= 0:
        sigma_bin = max(0.03 * base, 1.0)
    i = int(np.argmax(vpost - base))
    resp_abs = float(vpost[i] - base); lag = float(tpost[i] - t0)
    peak_sigma = resp_abs / sigma_bin                      # peak bin vs local noise
    # sustained change: robust two-sample (SE of the post-window median)
    se = sigma_bin / np.sqrt(max(n_post, 1))
    sustained_abs = float(np.median(vpost) - base)
    sustained_sigma = sustained_abs / max(se, 1e-6)
    # secular-trend control: judge the response against the pre-drainage linear
    # trend EXTRAPOLATED into the post window, not just the flat pre-median, so a
    # steadily accelerating glacier (e.g. Thwaites trunk) cannot masquerade as a
    # drainage surge. A real drainage response is a step ABOVE the prior trend.
    if n_pre >= 3:
        Apre = np.vstack([tpre - t0, np.ones_like(tpre)]).T
        sl, ic = np.linalg.lstsq(Apre, vpre, rcond=None)[0]
        trend_post = sl * (tpost - t0) + ic
    else:
        trend_post = np.full_like(vpost, base)
    detrend_sigma = float((vpost[i] - trend_post[i]) / sigma_bin)
    sustained_detrend_sigma = float(np.median(vpost - trend_post) / max(se, 1e-6))
    out.update(baseline=round(base, 1), sigma_bin_mpyr=round(sigma_bin, 2),
               resp_frac=round(resp_abs / base, 4) if base > 0 else None,
               peak_sigma=round(peak_sigma, 2),
               sustained_sigma=round(sustained_sigma, 2),
               detrend_sigma=round(detrend_sigma, 2),
               sustained_detrend_sigma=round(sustained_detrend_sigma, 2),
               lag_to_peak=round(lag, 3),
               in_band=bool(OBS_LAG_YR[0] <= lag <= maxlag
                            and peak_sigma >= K_SIGMA and sustained_sigma >= K_SIGMA
                            and detrend_sigma >= K_SIGMA
                            and sustained_detrend_sigma >= K_SIGMA))
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(events_csv=EVENTS_CSV, h5=SF2018_H5, box=BOX_M, limit_lakes=None,
        verbose=True):
    coords = load_lake_coords(h5)
    events = load_events(events_csv)
    by_lake = {}
    for e in events:
        by_lake.setdefault(e["lake"], []).append(e)

    itl = ITSLive()
    lakes_out = {}
    matched = sorted(l for l in by_lake if l in coords)
    if limit_lakes:
        matched = matched[:limit_lakes]
    skipped = sorted(l for l in by_lake if l not in coords)

    for k, lake in enumerate(matched):
        c = coords[lake]
        try:
            yr, v, dt = itl.series(c["lon"], c["lat"], c["x"], c["y"], box=box)
        except Exception as exc:
            if verbose:
                print(f"[{k+1}/{len(matched)}] {lake}: ITS_LIVE FAIL "
                      f"({type(exc).__name__}: {exc})")
            lakes_out[lake] = dict(error=f"{type(exc).__name__}: {exc}")
            continue
        tq, vq, nq = bin_series(yr, v, dt, 0.25)
        ta, va, na = bin_series(yr, v, dt, 1.0)
        nf_q, med_q = noise_floor_frac(tq, vq)
        nf_a, med_a = noise_floor_frac(ta, va)
        evs = []
        for e in by_lake[lake]:
            evs.append(dict(event=e,
                            quarterly=analyse_event(tq, vq, nq, e["t"]),
                            annual=analyse_event(ta, va, na, e["t"])))
        lakes_out[lake] = dict(
            lon=round(c["lon"], 3), lat=round(c["lat"], 3),
            n_pairs=int(v.size), median_speed_mpyr=round(float(np.median(v)), 1),
            n_qtr_bins=int(tq.size), n_ann_bins=int(ta.size),
            noise_floor_qtr=round(nf_q, 4) if np.isfinite(nf_q) else None,
            noise_floor_ann=round(nf_a, 4) if np.isfinite(nf_a) else None,
            yr_range=[round(float(yr.min()), 1), round(float(yr.max()), 1)],
            n_2003_2009_pairs=int(((yr >= 2003) & (yr < 2010)).sum()),
            fast=bool(lake.startswith(FAST_PREFIXES)),
            events=evs)
        if verbose:
            nsig = sum(1 for ev in evs
                       if (ev["annual"].get("peak_sigma") or 0) >= K_SIGMA
                       or (ev["quarterly"].get("peak_sigma") or 0) >= K_SIGMA)
            print(f"[{k+1}/{len(matched)}] {lake}: med={np.median(v):.0f} m/yr "
                  f"nf_a={nf_a:.3f} bins(q/a)={tq.size}/{ta.size} "
                  f"2003-09pairs={lakes_out[lake]['n_2003_2009_pairs']} "
                  f"ev={len(evs)} sig={nsig}")

    return _summarise(lakes_out, skipped, by_lake, coords)


def _summarise(lakes_out, skipped, by_lake, coords):
    sig_events, inband_lags, amp_bounds, testable_fast = [], [], [], []
    n_testable = 0
    for lake, d in lakes_out.items():
        if "events" not in d:
            continue
        for ev in d["events"]:
            t0 = ev["event"]["t"]
            tested = False
            best = None
            for key in ("annual", "quarterly"):
                a = ev[key]
                if not a.get("testable"):
                    continue
                tested = True
                s = a.get("peak_sigma") or 0.0
                if best is None or s > best[0]:
                    best = (s, key, a)
            if not tested:
                continue
            n_testable += 1
            if d.get("fast"):
                testable_fast.append((lake, round(t0, 2)))
            s, key, a = best
            # local fractional amplitude bound for this (testable) event
            if a.get("baseline") and a.get("sigma_bin_mpyr"):
                amp_bounds.append(a["sigma_bin_mpyr"] / a["baseline"])
            if a.get("in_band"):       # in_band already requires peak & sustained >= K_SIGMA
                sig_events.append(dict(lake=lake, t0=round(t0, 2), res=key,
                                       peak_sigma=a["peak_sigma"],
                                       sustained_sigma=a.get("sustained_sigma"),
                                       resp_frac=a.get("resp_frac"),
                                       lag_to_peak=a["lag_to_peak"]))
                inband_lags.append(a["lag_to_peak"])

    median_amp_bound = float(np.median(amp_bounds)) if amp_bounds else float("nan")
    verified = bool(len(sig_events) >= 1)

    return {
        "n_event_lakes_total": len(by_lake),
        "n_lakes_with_coords": len([l for l in by_lake if l in coords]),
        "n_lakes_run": len([d for d in lakes_out.values() if "events" in d]),
        "lakes_skipped_no_coords": skipped,
        "n_events_testable": n_testable,
        "n_fast_outlet_events_testable": len(testable_fast),
        "fast_outlet_events_testable": testable_fast,
        "n_significant_events": len(sig_events),
        "significant_events": sig_events,
        "lags_in_band_yr": sorted(inband_lags),
        "obs_lag_band_yr": list(OBS_LAG_YR),
        "median_amplitude_bound_frac": (round(median_amp_bound, 4)
                                        if np.isfinite(median_amp_bound)
                                        else None),
        "k_sigma": K_SIGMA,
        "verified_lag_value": verified,
        "verdict": (
            "VERIFIED (data-side): a sustained, significant post-drainage surge "
            "with lag-to-peak in the derived 0.02-2 yr band on testable outlet "
            "trunk velocity."
            if verified else
            "NOT CONFIRMED on this catalogue: no sustained, significant, in-band "
            "post-drainage surge on the testable outlet events; the matched test "
            "returns a quantified amplitude bound (median local floor "
            f"~{100*median_amp_bound:.1f}% of trunk speed). The literal thermal "
            "H^2/kappa kernel remains FALSIFIED (orders of magnitude off)."
            if np.isfinite(median_amp_bound) else
            "INCONCLUSIVE: insufficient dense co-temporal coverage on outlet "
            "events (the 2003-2007 ICESat catalogue predates dense ITS_LIVE)."),
        "lakes": lakes_out,
    }


def make_figure(summary, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    xs, ys, cs = [], [], []
    for lake, d in summary["lakes"].items():
        if "events" not in d:
            continue
        for ev in d["events"]:
            a = ev["annual"] if ev["annual"].get("testable") else ev["quarterly"]
            if a.get("testable") and a.get("peak_sigma") is not None:
                xs.append(a["lag_to_peak"]); ys.append(a["peak_sigma"])
                cs.append("#d62728" if a.get("in_band") else "#1f77b4")
    if xs:
        ax[0].scatter(xs, ys, c=cs, s=40, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax[0].axvspan(OBS_LAG_YR[0], OBS_LAG_YR[1], color="#d95f0e", alpha=0.2,
                  label="derived 0.02-2 yr band")
    ax[0].axhline(K_SIGMA, ls="--", color="k", lw=1, label=f"{K_SIGMA}σ detection")
    ax[0].set_xlabel("lag-to-peak after drainage [yr]")
    ax[0].set_ylabel("post-drainage anomaly [noise-floor σ]")
    ax[0].set_title("(a) §G.4 matched lag (binned trunk velocity)\n"
                    "red = significant & in-band")
    ax[0].legend(fontsize=8)
    labels, nf = [], []
    for lake, d in sorted(summary["lakes"].items(),
                          key=lambda kv: kv[1].get("noise_floor_ann") or 9):
        if "events" in d and d.get("noise_floor_ann"):
            labels.append(f"{lake} ({d['median_speed_mpyr']:.0f})")
            nf.append(100 * d["noise_floor_ann"])
    if labels:
        ax[1].barh(range(len(labels)), nf, color="#2c7fb8")
        ax[1].set_yticks(range(len(labels)))
        ax[1].set_yticklabels(labels, fontsize=6)
        ax[1].axvline(100 * (summary["median_amplitude_bound_frac"] or 0),
                      color="r", ls=":", label="median")
    ax[1].set_xlabel("annual velocity noise floor [% of trunk speed]")
    ax[1].set_title("(b) detectability per lake (speed in parens)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=EVENTS_CSV)
    ap.add_argument("--h5", default=SF2018_H5)
    ap.add_argument("--box", type=float, default=BOX_M)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(_REPORTS, "lake_lag_itslive.json"))
    a = ap.parse_args()
    summary = run(a.events, a.h5, box=a.box, limit_lakes=a.limit)
    print("\n=== §G.4 matched lake-drainage → ITS_LIVE lag (outlets) ===")
    print(f"lakes run: {summary['n_lakes_run']} / {summary['n_lakes_with_coords']} "
          f"with coords ({summary['n_event_lakes_total']} event lakes total)")
    print(f"testable events: {summary['n_events_testable']} "
          f"({summary['n_fast_outlet_events_testable']} fast-outlet)")
    print(f"significant: {summary['n_significant_events']}")
    for s in summary["significant_events"]:
        print(f"   {s['lake']} t0={s['t0']} {s['res']} peakσ={s['peak_sigma']} "
              f"sustσ={s['sustained_sigma']} lag={s['lag_to_peak']}yr")
    print(f"median amplitude bound: "
          f"{summary['median_amplitude_bound_frac']}")
    print(f"VERDICT: {summary['verdict']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"json -> {a.out}")
    make_figure(summary, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
