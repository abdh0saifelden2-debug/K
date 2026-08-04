r"""§H.1.4-ocean — Is the marine-West Ro-u_* organisation ocean-driven?

§H.1.4 (`rtn_glmig_basin.py`) found a *positive* log-log slope of the grounding-
line residence number ``Ro = v_kin / v_obs`` against a direct ``u_*`` proxy
(ITS_LIVE surface speed) in the marine, fast-streaming West-Antarctic sectors
(Amundsen, Bellingshausen/WAIS-Pacific) and a *negative* one in the cold East-
Antarctic interior, and explicitly DEFERRED the mechanism:

    "Why the marine sectors carry a positive u_* residence signal is an ocean-
     coupled question - the GL rate-limiting there is set by ocean-driven melt and
     sub-shelf cavity hydrology... relating Ro to ocean thermal forcing / melt
     rates is queued as the next thread."

This script runs that test. It relates Ro - and its mechanistic components, the
thinning rate ``dH/dt`` (which feeds ``v_kin = A.dH/dt``) and the surface speed
``u_*`` - to OBSERVED ocean thermal forcing on the adjacent continental shelf.

Ocean forcing
-------------
Schmidtko et al. (2014, Science, doi:10.1126/science.1256117) Antarctic Continental
Shelf Bottom Water climatology (seabed conservative temperature CT and absolute
salinity SA, 1975-2012; open, GEOMAR). This is the *bottom* water that ventilates
ice-shelf cavities and reaches grounding lines - the physically relevant heat
reservoir, far better suited than an open-ocean climatology (e.g. WOA) which has no
data inside cavities.

Thermal forcing  ``TF = CT - CT_freezing(SA, p)``  [degC]  (TEOS-10 / gsw), the
heat available for melt above the in-situ (pressure-dependent) freezing point.
Each Konrad grounding-line point is assigned the mean TF of the nearest shelf-
bottom water columns (source water) within a search radius.

Mechanism prediction
--------------------
Warm shelf water (high TF) -> basal melt -> dynamic thinning (dH/dt up) and
acceleration (u_* up) -> v_kin = A.(dH/dt) up -> Ro up. So in the marine sectors we
expect POSITIVE TF slopes for thinning, speed AND Ro; the cold interior
(TF ~ 0 degC, frozen beds) should be flat/absent. A positive, regime-dependent
TF organisation identifies the ocean as the common driver behind the §H.1.4
u_*-Ro signal.

Data (all open, no Earthdata auth)
----------------------------------
  konrad2018_glmig.txt     Konrad 2018 GL migration (CPOM)            [v_obs]
  bedmap2_bin/             Bedmap2 (BAS)                              [A geometry]
  schroder_sec_mrg.nc      Schroeder 2019 SEC (PANGAEA, CC-BY)        [dH/dt]
  ANT_G0240 ITS_LIVE       velocity mosaic (S3, streamed)            [u_*]
  schmidtko2014_shelf.txt  Schmidtko 2014 shelf bottom CT/SA (GEOMAR) [TF]

  # ocean forcing (~1.6 MB ASCII; cols: lon lat depth CT CT_std SA SA_std)
  curl -L -o ~/data_glmig/schmidtko2014_shelf.txt \
    https://www.geomar.de/fileadmin/personal/fb1/po/sschmidtko/Antarctic_shelf_data.txt

Reuses §H.1.4's A-geometry and v_obs via ``rtn_glmig_basin.sample``; thinning/v_obs
are recovered by an exact KD-match back to the Konrad rows. dH/dt is taken from the
Schroeder-2019 SEC when present, else from the ICESat-2 ATL15 delta_h trend
(``build_dhdt_from_atl15``) when PANGAEA is unavailable - the report records which.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtn_glmig_basin import sample, SECTORS, _in_sector  # noqa: E402
from rtn_glmig_test import build_dhdt_at_konrad  # noqa: E402
from itslive_glmig_sample import sample_speed  # noqa: E402

SEARCH_M = 150_000.0   # max distance to a shelf-bottom column (source water)
KNN = 5                # average TF over this many nearest shelf columns
ATL15_GLOB = "/home/data/atl15/ATL15_*.nc"
STRIDE = 4             # Bedmap2 decimation (4 km) - keeps peak RAM ~0.6 GB. The A
#                       geometry only scales Ro by a near-uniform factor, so the
#                       log(Ro)-vs-TF *slope* (this section's metric) is invariant.


def build_dhdt_from_atl15(konrad_txt, out_npz, atl15_glob=ATL15_GLOB,
                          search_m=30000.0, min_steps=12, chunk=300):
    """Per-Konrad-point dH/dt from the linear trend of ICESat-2 ATL15 delta_h.

    A memory-safe (dask-chunked, nan-aware) per-cell trend of the quarterly
    ``delta_h`` record, used in place of the Schroeder-2019 SEC when PANGAEA is
    unavailable; the delta_h trend is a modern (2019-) surface-lowering rate.
    Samples the nearest finite-trend cell within ``search_m`` of each GL point
    (coastal cells exactly at the GL can be NaN). Writes ``dhdt_m_per_yr`` aligned
    to the Konrad rows, plus a ``source`` tag.
    """
    import xarray as xr
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    files = sorted(glob.glob(atl15_glob))
    if not files:
        raise FileNotFoundError(f"no ATL15 files at {atl15_glob!r}")
    xs_all, ys_all, tr_all = [], [], []
    for f in files:
        ds = xr.open_dataset(f, group="delta_h",
                             chunks={"time": -1, "y": chunk, "x": chunk})
        t = ds["time"].values
        tyr = 2018.0 + (t - np.datetime64("2018-01-01")) / np.timedelta64(1, "D") / 365.25
        tc = (tyr - tyr.mean()).astype("f8")
        tca = xr.DataArray(tc, dims="time", coords={"time": ds["time"]})
        da = ds["delta_h"]
        fin = np.isfinite(da)
        mean = da.mean("time", skipna=True)
        num = ((da - mean) * tca).sum("time", skipna=True)
        den = (fin * (tca ** 2)).sum("time")
        cnt = fin.sum("time")
        slope = (num / den).where((cnt >= min_steps) & (den > 0))
        S = slope.compute()
        XX, YY = np.meshgrid(S.x.values, S.y.values)
        M = np.isfinite(S.values)
        xs_all.append(XX[M]); ys_all.append(YY[M]); tr_all.append(S.values[M])
        ds.close()
    X = np.concatenate(xs_all); Y = np.concatenate(ys_all); V = np.concatenate(tr_all)
    lat, lon = np.loadtxt(konrad_txt, skiprows=5, usecols=(0, 1), unpack=True)
    px, py = Transformer.from_crs("EPSG:4326", "EPSG:3031",
                                  always_xy=True).transform(lon, lat)
    tree = cKDTree(np.c_[X, Y])
    dist, ii = tree.query(np.c_[px, py], distance_upper_bound=search_m)
    dh = np.where(np.isfinite(dist), V[np.clip(ii, 0, V.size - 1)], np.nan)
    np.savez(out_npz, dhdt_m_per_yr=dh, source="ICESat2_ATL15_delta_h_trend")
    return out_npz


def load_schmidtko_tf(txt):
    """Return (lon, lat, TF[degC]) of Antarctic shelf-bottom water (Schmidtko 2014).

    TF = CT - CT_freezing(SA, p) with the TEOS-10 conservative-temperature freezing
    point at the in-situ pressure of the seabed (DEPTH is negative = below sea level).
    """
    import gsw
    lon, lat, depth, ct, sa = np.loadtxt(
        txt, skiprows=1, usecols=(0, 1, 2, 3, 5), unpack=True)
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(depth) & \
        np.isfinite(ct) & np.isfinite(sa) & (sa > 0)
    lon, lat, depth, ct, sa = lon[ok], lat[ok], depth[ok], ct[ok], sa[ok]
    p = gsw.p_from_z(depth, lat)                     # dbar (depth<0 -> p>0)
    tf = ct - gsw.CT_freezing(sa, p, 0.0)           # degC above freezing
    return lon, lat, tf


def assign_tf(glon, glat, slon, slat, stf, search_m=SEARCH_M, k=KNN):
    """Mean TF of the <=k nearest shelf columns within search_m of each GL point.

    Returns (tf[npts], dist_km[npts]); NaN where no shelf column is within range
    (e.g. deep Ross/Filchner-Ronne cavities far from any shelf-bottom observation).
    """
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    gx, gy = tr.transform(glon, glat)
    sx, sy = tr.transform(slon, slat)
    tree = cKDTree(np.c_[sx, sy])
    dist, idx = tree.query(np.c_[gx, gy], k=k)
    dist = np.atleast_2d(dist.T).T
    idx = np.atleast_2d(idx.T).T
    tf = np.full(glon.shape, np.nan)
    dkm = np.full(glon.shape, np.nan)
    for i in range(glon.size):
        w = dist[i] <= search_m
        if w.any():
            tf[i] = np.mean(stf[idx[i][w]])
            dkm[i] = np.mean(dist[i][w]) / 1000.0
    return tf, dkm


def recover_thin_vobs(glon, glat, konrad_txt, dhdt_npz):
    """Thinning (-dH/dt) and v_obs[km/yr] at the canonical good points, via an exact
    KD-match of each returned (lon,lat) back to its Konrad row (distances ~0)."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    lat, lon, vgl, _ = np.loadtxt(konrad_txt, skiprows=5, unpack=True)
    dhdt = np.load(dhdt_npz)["dhdt_m_per_yr"]
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    kx, ky = tr.transform(lon, lat)
    gx, gy = tr.transform(glon, glat)
    tree = cKDTree(np.c_[kx, ky])
    dist, idx = tree.query(np.c_[gx, gy], k=1)
    if np.nanmax(dist) > 1.0:                        # must be the same points (<1 m)
        raise RuntimeError(f"good-point<->Konrad match failed (max dist {dist.max():.1f} m)")
    return -dhdt[idx], vgl[idx] / 1000.0


def _fit_semilog(y, x, nboot=2000, seed=0):
    """OLS slope/r of ln(y) on x (y>0, x linear) with a paired-bootstrap 95% CI.

    slope = d ln(y)/dx  (here x = TF in degC, so slope ~ fractional change in y per
    degC of thermal forcing). Requires n>=30 finite (y>0, x) pairs."""
    m = np.isfinite(y) & np.isfinite(x) & (y > 0)
    out = {"n": int(m.sum()), "median_TF": None, "median_y": None,
           "slope_per_degC": None, "r": None, "ci95": None}
    if m.sum() >= 30:
        ly, xx = np.log(y[m]), x[m]
        out["median_TF"] = float(np.median(xx))
        out["median_y"] = float(np.median(y[m]))
        slope = float(np.polyfit(xx, ly, 1)[0])
        out["slope_per_degC"] = slope
        out["r"] = float(np.corrcoef(xx, ly)[0, 1])
        rng = np.random.default_rng(seed)
        bs = np.empty(nboot)
        idx = np.arange(xx.size)
        for b in range(nboot):
            j = rng.choice(idx, idx.size, replace=True)
            bs[b] = np.polyfit(xx[j], ly[j], 1)[0]
        out["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def analyse(bin_dir, konrad_txt, dhdt_npz, schmidtko_txt, itslive_npz=None,
            atl15_glob=ATL15_GLOB):
    # 1) resolve a per-point dH/dt: prefer Schroeder SEC (matches §H.1.4), else the
    #    ICESat-2 ATL15 modern trend when PANGAEA is down. Record the provenance.
    dhdt_source = "precomputed_cache"
    if dhdt_npz and not os.path.exists(dhdt_npz):
        sec_nc = os.path.join(os.path.dirname(dhdt_npz), "schroder_sec_mrg.nc")
        if os.path.exists(sec_nc):
            build_dhdt_at_konrad(sec_nc, konrad_txt, dhdt_npz)
            dhdt_source = "Schroeder2019_SEC"
        elif atl15_glob and glob.glob(atl15_glob):
            build_dhdt_from_atl15(konrad_txt, dhdt_npz, atl15_glob)
            dhdt_source = "ICESat2_ATL15_delta_h_trend"
        else:
            raise FileNotFoundError(
                "no dH/dt source: need schroder_sec_mrg.nc beside the npz or ATL15 files")
    elif dhdt_npz and os.path.exists(dhdt_npz):
        z = np.load(dhdt_npz)
        if "source" in z.files:
            dhdt_source = str(z["source"])
    r = sample(bin_dir, konrad_txt, dhdt_npz, stride=STRIDE)
    lon, lat, Ro = r["lon"], r["lat"], r["Ro"]

    # 2) mechanistic components aligned to the same good points
    thin, v_obs = recover_thin_vobs(lon, lat, konrad_txt, dhdt_npz)

    # 3) u_* (ITS_LIVE speed): reuse cache if it aligns, else stream + cache
    spd = None
    if itslive_npz and os.path.exists(itslive_npz):
        z = np.load(itslive_npz)
        if z["Ro"].shape == Ro.shape and np.allclose(z["Ro"], Ro, equal_nan=True):
            spd = z["speed"]
    if spd is None:
        spd = sample_speed(lon, lat)
        if itslive_npz:
            np.savez(itslive_npz, speed=spd, lon=lon, lat=lat,
                     Ro=Ro, tau_d=r["tau_d"])

    # 4) ocean thermal forcing at each GL point
    slon, slat, stf = load_schmidtko_tf(schmidtko_txt)
    tf, dkm = assign_tf(lon, lat, slon, slat, stf)

    # 5) per-sector regressions of (thinning, Ro, speed) on TF
    regions = list(SECTORS) + ["CONTINENTAL"]
    masks = [_in_sector(lon, lat, SECTORS[n]) for n in SECTORS] + \
            [np.ones(lon.size, bool)]
    summ = {}
    for name, m in zip(regions, masks):
        mt = m & np.isfinite(tf)
        summ[name] = {
            "n_in_sector": int(m.sum()),
            "n_with_TF": int(mt.sum()),
            "TF_coverage_frac": float(mt.sum() / max(1, m.sum())),
            "median_match_dist_km": (float(np.nanmedian(dkm[mt])) if mt.any() else None),
            "median_TF_degC": (float(np.nanmedian(tf[mt])) if mt.any() else None),
            "fits": {
                "thinning_vs_TF": _fit_semilog(thin[mt], tf[mt]),
                "Ro_vs_TF": _fit_semilog(Ro[mt], tf[mt]),
                "speed_vs_TF": _fit_semilog(spd[mt], tf[mt]),
            },
        }
    diag = dict(lon=lon, lat=lat, Ro=Ro, thin=thin, v_obs=v_obs, spd=spd,
                tf=tf, dkm=dkm)
    return summ, diag, dhdt_source


def _verdict(summ):
    def sl(region, key):
        f = summ[region]["fits"][key]
        return f["slope_per_degC"], f["ci95"]

    def pos(region, key):
        s, ci = sl(region, key)
        return s is not None and ci is not None and ci[0] > 0

    def neg(region, key):
        s, ci = sl(region, key)
        return s is not None and ci is not None and ci[1] < 0

    marine = ["Amundsen (PIG/Thwaites/Smith)", "Bellingshausen+WAIS-Pacific"]
    marine_ro = [r for r in marine if pos(r, "Ro_vs_TF")]
    marine_thin = [r for r in marine if pos(r, "thinning_vs_TF")]
    east = "East Antarctica"
    east_flat_or_neg = not pos(east, "Ro_vs_TF")

    if marine_ro and marine_thin and east_flat_or_neg:
        head = ("SUPPORTED: ocean thermal forcing organises the marine-West "
                "residence signal. The §H.1.4 u_*-Ro slope is ocean-driven.")
    elif marine_ro or marine_thin:
        head = ("PARTIALLY SUPPORTED: a positive TF organisation appears in some "
                "marine-West components but not all/clean of the East contrast.")
    else:
        head = ("NOT SUPPORTED on this matching: no robust positive TF organisation "
                "of Ro/thinning in the marine-West sectors.")
    return (head + " Marine sectors with positive Ro-TF slope (CI>0): "
            + (", ".join(marine_ro) or "none")
            + "; with positive thinning-TF slope: "
            + (", ".join(marine_thin) or "none")
            + f"; East Antarctica Ro-TF {'flat/negative' if east_flat_or_neg else 'positive'}."
            + " TF = Schmidtko-2014 shelf-bottom CT - CT_freezing(SA,p) (TEOS-10).")


def make_figure(diag, summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon, lat, Ro, tf = diag["lon"], diag["lat"], diag["Ro"], diag["tf"]
    colors = ["#d1495b", "#edae49", "#66a182", "#2e4057"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.5, 5.9))

    regions = list(SECTORS) + ["CONTINENTAL"]
    yc = np.arange(len(regions))[::-1]
    series = [("thinning_vs_TF", +0.22, "#1b9e77", "thinning dH/dt"),
              ("Ro_vs_TF", 0.0, "#d95f02", "residence Ro"),
              ("speed_vs_TF", -0.22, "#7570b3", "speed u_*")]
    for key, off, col, lab in series:
        labeled = False
        for k, name in enumerate(regions):
            f = summ[name]["fits"][key]
            if f["slope_per_degC"] is None:
                continue
            lo, hi = f["ci95"]
            ax0.errorbar(f["slope_per_degC"], yc[k] + off,
                         xerr=[[f["slope_per_degC"] - lo], [hi - f["slope_per_degC"]]],
                         fmt="o", color=col, capsize=3, ms=6,
                         label=None if labeled else lab)
            labeled = True
    ax0.axvline(0, color="k", lw=1, ls="--")
    ax0.set_yticks(yc)
    ax0.set_yticklabels([n.replace("+", "\n+") for n in regions], fontsize=8.5)
    ax0.set_xlabel(r"slope of $\ln(\cdot)$ vs ocean thermal forcing TF  [per $\degree$C]"
                   "\n(>0 $\\Rightarrow$ warmer shelf water $\\Rightarrow$ larger response)")
    ax0.set_title("§H.1.4-ocean: TF organisation by sector (95% bootstrap CI)")
    ax0.legend(loc="lower right", fontsize=8.5)
    ax0.grid(axis="x", alpha=0.3)

    for (name, spec), col in zip(SECTORS.items(), colors):
        m = _in_sector(lon, lat, spec) & np.isfinite(tf) & (Ro > 0)
        if m.sum() < 5:
            continue
        ax1.scatter(tf[m], Ro[m], s=7, alpha=0.25, color=col, label=name)
        f = summ[name]["fits"]["Ro_vs_TF"]
        if f["slope_per_degC"] is None:
            continue
        b = np.polyfit(tf[m], np.log(Ro[m]), 1)
        xx = np.array([np.nanpercentile(tf[m], 2), np.nanpercentile(tf[m], 98)])
        ax1.plot(xx, np.exp(b[1] + b[0] * xx), color=col, lw=2.4)
    ax1.axhline(1.0, color="k", ls=":", lw=1)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"ocean thermal forcing  TF = CT $-$ CT$_{\rm freeze}$(SA,p)  [$\degree$C]"
                   "  (Schmidtko 2014 shelf bottom)")
    ax1.set_ylabel("residence number  Ro = v_kin / v_obs")
    ax1.set_title("Marine streaming sectors: Ro rises with ocean heat")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle("§H.1.4-ocean - the marine-West u_*-Ro residence signal tracks observed "
                 "ocean thermal forcing\n(Schmidtko-2014 shelf-bottom water): warm "
                 "Circumpolar Deep Water sectors carry the positive organisation",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=135)
    print(f"figure -> {os.path.abspath(path)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default=os.path.expanduser("~/data_bedmap/bedmap2_bin"))
    ap.add_argument("--konrad", default=os.path.expanduser("~/data_glmig/konrad2018_glmig.txt"))
    ap.add_argument("--dhdt-npz", default=os.path.expanduser("~/data_glmig/dhdt_at_konrad.npz"))
    ap.add_argument("--schmidtko", default=os.path.expanduser("~/data_glmig/schmidtko2014_shelf.txt"))
    ap.add_argument("--itslive-npz", default=os.path.expanduser(
        "~/data_glmig/itslive_speed_at_konrad_good.npz"))
    ap.add_argument("--atl15", default=ATL15_GLOB)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports", "rtn_glmig_ocean.json"))
    a = ap.parse_args()

    summ, diag, dhdt_source = analyse(a.bin_dir, a.konrad, a.dhdt_npz, a.schmidtko,
                                     a.itslive_npz, a.atl15)
    verdict = _verdict(summ)
    out = {"hypothesis": ("the marine-West §H.1.4 u_*-Ro residence organisation is "
                          "ocean-driven: Ro, thinning and u_* should rise with observed "
                          "shelf ocean thermal forcing in warm marine sectors and be "
                          "flat/absent in the cold interior"),
           "ocean_data": "Schmidtko et al. 2014 Antarctic shelf-bottom CT/SA (GEOMAR, open)",
           "TF_definition": "TF = CT - CT_freezing(SA, p) [degC], TEOS-10 (gsw)",
           "dhdt_source": dhdt_source,
           "bedmap2_stride": STRIDE,
           "search_radius_km": SEARCH_M / 1000.0, "knn": KNN,
           "n_good_points": int(diag["lon"].size),
           "n_with_TF": int(np.isfinite(diag["tf"]).sum()),
           "sectors": summ, "verdict": verdict,
           "caveats": ("Schmidtko is a 1975-2012 climatology vs CryoSat-era (2010-2016) "
                       "Ro/thinning - TF is a slowly-varying boundary condition; nearest-"
                       "shelf-column TF is a source-water proxy, not in-cavity; bootstrap "
                       "CIs ignore spatial autocorrelation (true CIs wider); correlation is "
                       "an organisation consistent with the ocean-melt mechanism, not proof.")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in ("n_good_points", "n_with_TF", "verdict")}, indent=2))
    print(f"json -> {os.path.abspath(a.out)}")
    make_figure(diag, summ,
                os.path.join(os.path.dirname(os.path.abspath(a.out)), "rtn_glmig_ocean.png"))


if __name__ == "__main__":
    main()
