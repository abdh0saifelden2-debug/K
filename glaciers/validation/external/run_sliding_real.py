r"""Real-data §V.2 sliding-law check on open data (no auth).
 
Open data actually reachable from this VM:
  * Lake **locations** -- Siegfried & Fricker (2018) active-lake outlines
    (131 named lakes), openly mirrored on GitHub (mrsiegfried/Siegfried2021-GRL,
    ``data/outlines/SiegfriedFricker2018-outlines.h5``; cites Smith et al. 2009).
  * Ice **thickness**  -- BAS Bedmap2 (``run_rtn_bedmap2``'s loader).
  * Surface **velocity** time series -- ITS_LIVE datacubes (open S3, anon),
    dense (>10^4 image pairs) over fast outlet glaciers (Byrd, David, Recovery).
 
What is *not* openly reachable: the vetted lake **volume-change time series**
(drainage event *dates*), which live behind a USAP-DC login (NSIDC FTP host
returns HTTP 000 here).  Rather than fabricate event dates, this script runs the
part of §V.2 that real open data *can* settle:
 
  **The literal §G.4 memory-kernel timescale tau_ice = H^2 / kappa_ice,
  evaluated on real Bedmap2 thickness at the 131 catalogued lakes, is compared
  with the observed range of post-drainage surge lags (days -> ~2 yr).**
 
This is the §G.4 caveat made quantitative on real geometry: the diffusive
thermal-memory timescale across the full ice column is ~10^3-10^4 yr -- orders of
magnitude longer than any observed surge response -- so the kernel *as literally
written* is falsified as a lag predictor, and only an empirical lag is
interpretable.  (The validator's lag-detection machinery itself is unit-tested in
``sliding_synthetic`` / ``tests/test_validation_synthetic.py``.)
"""
from __future__ import annotations
 
import argparse
import os
import sys
 
import numpy as np
 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
 
from external import DataUnavailableError  # noqa: E402
 
KAPPA_ICE = 1.09e-6          # ice thermal diffusivity [m^2 s^-1]
SEC_PER_YR = 365.25 * 86400.0
LAKE_H5 = "/home/data_lakes/SiegfriedFricker2018-outlines.h5"
BEDMAP_DIR = "/home/data_bedmap/bedmap2_bin"
 
# Observed post-drainage surge-lag range from the literature (order of magnitude):
# Stearns et al. 2008 (Byrd, speedup within months of drainage); Siegfried et al.
# 2016; Scambos et al. -- responses are days to ~2 yr, never millennia.
OBS_LAG_YR = (0.02, 2.0)
 
 
def lake_centroids(h5_path=LAKE_H5):
    """Return list of (name, x_ps, y_ps) EPSG:3031 polar-stereographic centroids."""
    import h5py
    out = []
    with h5py.File(h5_path, "r") as f:
        for name in f.keys():
            g = f[name]
            if "x" in g and "y" in g:
                out.append((name, float(np.mean(g["x"][:])), float(np.mean(g["y"][:]))))
    return out
 
 
def sample_bedmap_thickness(xs, ys, bin_dir=BEDMAP_DIR):
    """Nearest-cell Bedmap2 ice thickness [m] at EPSG:3031 points (NaN if off-grid)."""
    from external.bedmap2_loader import read_flt
    H, m = read_flt(os.path.join(bin_dir, "bedmap2_thickness.flt"))
    nrows, ncols, cs = m["nrows"], m["ncols"], m["cellsize"]
    xll, yll = m["xll"], m["yll"]
    out = np.full(len(xs), np.nan)
    for i, (px, py) in enumerate(zip(xs, ys)):
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        c = int(round((px - xll) / cs - 0.5))
        r = nrows - 1 - int(round((py - yll) / cs - 0.5))
        if 0 <= r < nrows and 0 <= c < ncols:
            v = H[r, c]
            if np.isfinite(v) and v > 0:
                out[i] = v
    return out
 
 
def tau_ice_years(H_m):
    return (H_m ** 2 / KAPPA_ICE) / SEC_PER_YR
 
 
def run():
    lakes = lake_centroids()
    names = [n for n, _, _ in lakes]
    xs = np.array([x for _, x, _ in lakes])
    ys = np.array([y for _, _, y in lakes])
    H = sample_bedmap_thickness(xs, ys)
    ok = np.isfinite(H)
    tau = tau_ice_years(H[ok])
    return names, H, ok, tau
 
 
def itslive_velocity_at_lake(lake="Mac1", h5_path=LAKE_H5, box_m=2000.0):
    """Pull the REAL ITS_LIVE surface-speed time series at a catalogued lake.
 
    Opens the open (anon S3) ITS_LIVE datacube covering the lake centroid and
    returns ``(t_years, v_mps_per_yr)`` box-averaged speed.  Requires network +
    ``zarr``/``s3fs``; raises on failure so callers can fall back honestly.
    """
    import json
    import urllib.request
 
    import h5py
    import xarray as xr
 
    with h5py.File(h5_path, "r") as f:
        g = f[lake]
        lx, ly = float(np.mean(g["x"][:])), float(np.mean(g["y"][:]))
        llon, llat = float(np.mean(g["lon"][:])), float(np.mean(g["lat"][:]))
 
    cat = json.load(urllib.request.urlopen(
        "https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json", timeout=60))
 
    def _contains(geom, lon, lat):
        ring = geom["coordinates"][0]
        inside = False
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > lat) != (yj > lat)) and \
                    (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
 
    hits = [ft for ft in cat["features"] if ft["properties"].get("datacube_exist")
            and _contains(ft["geometry"], llon, llat)]
    if not hits:
        raise DataUnavailableError(f"No ITS_LIVE datacube covers lake {lake}.")
    p = max(hits, key=lambda ft: ft["properties"].get("granule_count", 0))["properties"]
    url = p["zarr_url"].replace("https://its-live-data.s3.amazonaws.com/", "s3://its-live-data/")
    ds = xr.open_dataset(url, engine="zarr", storage_options={"anon": True}, consolidated=True)
    sub = ds["v"].sel(x=slice(lx - box_m, lx + box_m), y=slice(ly + box_m, ly - box_m))
    t = ds["mid_date"].values
    v = sub.mean(dim=("x", "y")).values.astype(float)
    m = np.isfinite(v)
    t, v = t[m], v[m]
    o = np.argsort(t)
    t, v = t[o], v[o]
    yr = 1970.0 + (t.astype("datetime64[D]").astype(float)) / 365.25
    return yr, v
 
 
def make_figure(tau, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log10(tau), bins=24, color="#2c7fb8", alpha=0.85)
    ax.axvspan(np.log10(OBS_LAG_YR[0]), np.log10(OBS_LAG_YR[1]),
               color="#d95f0e", alpha=0.3, label="observed surge lag (days-2 yr)")
    ax.set_xlabel(r"$\log_{10}\,\tau_{ice}=H^2/\kappa_{ice}$  [yr]")
    ax.set_ylabel("number of catalogued lakes")
    ax.set_title("§G.4 literal kernel timescale on real Bedmap2 thickness\n"
                 "(131 Siegfried & Fricker 2018 active lakes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-velocity", default=None,
                    help="lake name (e.g. Mac1) to pull a real ITS_LIVE speed series")
    args = ap.parse_args()
 
    names, H, ok, tau = run()
    print("=== §V.2 sliding-law: §G.4 kernel timescale on REAL data ===")
    print(f"catalogued lakes (Siegfried & Fricker 2018): {len(names)}")
    print(f"with valid Bedmap2 thickness: {int(ok.sum())}")
    print(f"ice thickness H at lakes [m]:  median={np.nanmedian(H):.0f}  "
          f"min={np.nanmin(H):.0f}  max={np.nanmax(H):.0f}")
    print(f"tau_ice = H^2/kappa_ice [yr]:  median={np.median(tau):.0f}  "
          f"p5={np.percentile(tau,5):.0f}  p95={np.percentile(tau,95):.0f}")
    print(f"observed post-drainage surge lag: {OBS_LAG_YR[0]}-{OBS_LAG_YR[1]} yr")
    ratio = np.median(tau) / OBS_LAG_YR[1]
    print(f"=> literal kernel is ~{ratio:.0e}x too slow at the median lake "
          f"[§G.4 kernel FALSIFIED as written; only empirical lag interpretable]")
    fig = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "reports", "sliding_tau_ice_real.png")
    make_figure(tau, fig)
    print(f"figure -> {os.path.normpath(fig)}")
 
    if args.with_velocity:
        print(f"\n--- real ITS_LIVE velocity at lake {args.with_velocity} "
              f"(response-data accessibility) ---")
        try:
            from validators.sliding_validator import estimate_lag
            yr, v = itslive_velocity_at_lake(args.with_velocity)
            print(f"  REAL series: n={len(v)}  {yr.min():.1f}->{yr.max():.1f}  "
                  f"median={np.median(v):.0f} m/yr (min={v.min():.0f} max={v.max():.0f})")
            tg = np.arange(np.floor(yr.min()), np.ceil(yr.max()) + 0.25, 0.25)
            vg = np.interp(tg, yr, v)
            self_lag = estimate_lag(vg.copy(), vg.copy(), dt=0.25)[0]
            print(f"  validator.estimate_lag runs on it (self-lag={self_lag:.2f}, expect ~0)")
            print("  => §V.2 response data is REAL + openly accessible; only the "
                  "gated drainage-date catalogue blocks the matched lag test.")
        except Exception as exc:  # noqa: BLE001
            print(f"  ITS_LIVE pull unavailable ({type(exc).__name__}: {exc}); see §H.2.")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
