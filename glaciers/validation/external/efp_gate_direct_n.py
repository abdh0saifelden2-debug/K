r"""§H.1.6 (upgrade) — effective-pressure gating with a MORE DIRECT, phi-free N.

The committed §H.1.6 test (``rtn_ocean_efp_gate.py``) gates the ocean->velocity
coupling on ``rel = m/H`` with ``m = H - H*`` and ``H* = H_flot/phi`` — a margin
*proxy* for normalized effective pressure that (i) leans on Bedmap2 (2013, 1 km)
and (ii) carries a free, continent-uniform basal-water fraction ``phi = 0.9``.
Its honest caveat: *"rel is a normalized-effective-pressure proxy, not measured N"*.

This module removes both weaknesses by computing the **physical** normalized
effective pressure directly from the ocean-connected flotation balance on the more
accurate, independent **BedMachine Antarctica v3 (500 m)** geometry:

    d_base = max(0, -bed)                      # bed depth below sea level
    H_flot = (rho_w / rho_i) * d_base          # flotation thickness
    N      = rho_i * g * (H - H_flot)          # effective pressure [Pa]  (phi-free)
    n_hat  = N / (rho_i g H) = 1 - H_flot/H    # normalized N (dimensionless)

``n_hat`` IS the normalized effective pressure (normalized height above flotation)
under the standard ocean-connected-bed assumption — **no tunable phi**, and from a
higher-resolution dataset independent of Bedmap2. (g cancels in ``n_hat`` and in the
gating regression, so the *interaction* verdict does not depend on g or the absolute
Pa scale; g only sets the reported N magnitude.)

We then re-run the §H.1.6 gating regression
``ln u_* = a + b*TF + c*ln(n_hat) + d*(TF*ln n_hat)`` (predict ``d < 0``: the
TF->speed slope steepens toward flotation) and compare the BedMachine-N gating
**head-to-head** with the Bedmap2-rel gating on the identical GL points / TF / speed.

Verdict logic: if the more-direct, phi-free, independent-dataset N reproduces the
``d < 0`` gating (CI excludes 0) in the marine West and stays flat in the cold
interior, the §H.1.6 effective-pressure reading is robust to the proxy — a genuine
upgrade from "proxy" toward "measured N".

Honest scope: N still assumes an ocean-connected hydraulic head at the bed (the
relevant limit near grounding lines); a *truly* measured N needs boreholes/seismic.
This is the most-direct N obtainable from open gridded data.

Data: BedMachine v3 (Earthdata NSIDC-0756), Konrad 2018 GL points (CPOM), ITS_LIVE
240 m mosaic (anon S3), Schmidtko 2014 shelf TF (GEOMAR), Bedmap2 (BAS) for the
head-to-head rel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import rtn_ocean_efp_gate as RG                            # noqa: E402
from rtn_ocean_efp_gate import (_interaction_fit, _terciles)  # noqa: E402

ATL15_GLOB = os.path.expanduser("~/data_atl15/ATL15_A?_*01km*.nc")

RHO_I = 917.0
RHO_W = 1028.0
G = 9.81
BEDMACHINE = os.path.expanduser(
    "~/data_bedmachine/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")


def sample_bedmachine_N(glon, glat, path=BEDMACHINE, stride=4, search_m=10000.0):
    """Normalized effective pressure n_hat = 1 - H_flot/H (phi-free) and N [Pa] at
    each GL point, from the nearest grounded BedMachine cell (N>0) within search_m."""
    import netCDF4
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    ds = netCDF4.Dataset(path)
    x = np.asarray(ds.variables["x"][::stride], float)
    y = np.asarray(ds.variables["y"][::stride], float)
    H = np.asarray(ds.variables["thickness"][::stride, ::stride], float)
    bed = np.asarray(ds.variables["bed"][::stride, ::stride], float)
    mask = np.asarray(ds.variables["mask"][::stride, ::stride])
    ds.close()
    grounded = (mask == 2) & np.isfinite(H) & (H > 0) & np.isfinite(bed)
    d_base = np.where(bed < 0, -bed, 0.0)
    H_flot = (RHO_W / RHO_I) * d_base
    with np.errstate(invalid="ignore", divide="ignore"):
        n_hat = np.where(grounded & (H > 0), 1.0 - H_flot / H, np.nan)
    N_pa = RHO_I * G * (H - H_flot)
    valid = grounded & np.isfinite(n_hat) & (n_hat > 0)
    XX, YY = np.meshgrid(x, y)
    tree = cKDTree(np.c_[XX[valid], YY[valid]])
    nh_v, N_v, H_v = n_hat[valid], N_pa[valid], H[valid]
    px, py = Transformer.from_crs("EPSG:4326", "EPSG:3031",
                                  always_xy=True).transform(glon, glat)
    dist, idx = tree.query(np.c_[px, py], distance_upper_bound=search_m)
    hit = np.isfinite(dist)
    nh = np.full(glon.shape, np.nan); nh[hit] = nh_v[idx[hit]]
    Npa = np.full(glon.shape, np.nan); Npa[hit] = N_v[idx[hit]]
    Hb = np.full(glon.shape, np.nan); Hb[hit] = H_v[idx[hit]]
    return nh, Npa, Hb


def _fit_domains(spd, tf, proxy, marine):
    """Interaction + tercile gating fits per domain for a given N-proxy field, on
    the EXACT §H.1.6 good points (CONTINENTAL = all; MARINE_WEST = `marine` mask)."""
    domains = {"CONTINENTAL": np.ones(spd.size, bool), "MARINE_WEST": marine}
    res = {}
    for name, dm in domains.items():
        res[name] = {
            "n": int(dm.sum()),
            "n_usable": int((dm & np.isfinite(tf) & np.isfinite(proxy)
                             & (proxy > 0) & np.isfinite(spd) & (spd > 0)).sum()),
            "terciles_P1": _terciles(spd[dm], tf[dm], proxy[dm]),
            "interaction_P2": _interaction_fit(spd[dm], tf[dm], proxy[dm]),
        }
    return res


def _gate_ok(res):
    mw = res["MARINE_WEST"]
    it = mw.get("interaction_P2", {})
    d, ci = it.get("d_interaction"), it.get("d_ci95")
    t = mw.get("terciles_P1") or {}
    lo = t.get("low_rel_near_flotation", {}).get("TF_slope")
    hi = t.get("high_rel_well_grounded", {}).get("TF_slope")
    mono = lo is not None and hi is not None and lo > hi
    neg = d is not None and ci is not None and ci[1] < 0
    return bool(mono and neg), d, ci, lo, hi


def analyse(konrad_txt, schmidtko_txt, bin_dir, bedmachine=BEDMACHINE,
            dhdt_npz=None, itslive_npz=None, atl15_glob=ATL15_GLOB):
    """Reproduce the committed §H.1.6 good-point set + rel gating (RG.analyse), then
    swap ONLY the N proxy to the direct, phi-free BedMachine N at the SAME points."""
    dhdt_npz = dhdt_npz or os.path.expanduser("~/data_glmig/dhdt_at_konrad.npz")
    itslive_npz = itslive_npz or os.path.expanduser("~/data_glmig/itslive_speed_at_konrad_good.npz")
    # committed baseline: rel=m/H proxy on the canonical good points
    res_b2, diag = RG.analyse(bin_dir, konrad_txt, dhdt_npz, schmidtko_txt,
                              itslive_npz, atl15_glob)
    lon, lat, spd, tf = diag["lon"], diag["lat"], diag["spd"], diag["tf"]
    marine = diag["marine"]
    print(f"[good points: {lon.size}; with TF {np.isfinite(tf).sum()}; "
          f"marine {marine.sum()}]", flush=True)
    # upgrade: direct phi-free BedMachine N at the SAME good points
    n_hat, N_pa, H_bm = sample_bedmachine_N(lon, lat, bedmachine)
    print(f"[BedMachine n_hat valid at good points: {np.isfinite(n_hat).sum()}]", flush=True)
    res_bm = _fit_domains(spd, tf, n_hat, marine)
    both = np.isfinite(n_hat) & (n_hat > 0) & np.isfinite(diag["rel"]) & (diag["rel"] > 0)
    diag2 = dict(lon=lon, lat=lat, spd=spd, tf=tf, n_hat=n_hat, N_pa=N_pa,
                 rel=diag["rel"], marine=marine)
    return res_bm, res_b2, diag2, both


def make_figure(diag, res_bm, res_b2, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    order = ["low_rel_near_flotation", "mid_rel", "high_rel_well_grounded"]
    labels = ["low N\n(near flotation)", "mid N", "high N\n(well grounded)"]
    for res, col, off, lab in [(res_bm, "#c2185b", -0.15, "BedMachine N (phi-free)"),
                               (res_b2, "#1565c0", +0.15, "Bedmap2 rel (phi=0.9)")]:
        t = res["MARINE_WEST"].get("terciles_P1")
        if not t:
            continue
        xs, ys, es = [], [], []
        for i, k in enumerate(order):
            b = t.get(k, {})
            if b.get("TF_slope") is None:
                continue
            xs.append(i + off); ys.append(b["TF_slope"])
            ci = b["ci95"]; es.append([b["TF_slope"] - ci[0], ci[1] - b["TF_slope"]])
        if xs:
            ax[0].errorbar(xs, ys, yerr=np.array(es).T, fmt="o-", color=col,
                           capsize=3, ms=7, label=lab)
    ax[0].axhline(0, color="k", lw=1, ls="--")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set_ylabel(r"$d\,\ln u_*/dTF$  [per $\degree$C]")
    ax[0].set_title("(a) marine West: TF->speed steepens toward flotation\n"
                    "(direct BedMachine N vs Bedmap2 rel proxy)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    m = (diag["marine"] & np.isfinite(diag["n_hat"]) & (diag["n_hat"] > 0)
         & np.isfinite(diag["tf"]) & np.isfinite(diag["spd"]) & (diag["spd"] > 0))
    sc = ax[1].scatter(diag["tf"][m], diag["spd"][m], c=np.log10(diag["n_hat"][m]),
                       s=10, alpha=0.5, cmap="viridis_r")
    ax[1].set_yscale("log")
    cb = fig.colorbar(sc, ax=ax[1]); cb.set_label(r"$\log_{10}\,\hat N$ (low = near flotation)")
    ax[1].set_xlabel(r"ocean thermal forcing TF [$\degree$C]")
    ax[1].set_ylabel(r"GL surface speed $u_*$ [m/yr]")
    ax[1].set_title("Marine West: near-flotation (dark) responds most to TF")
    fig.suptitle("§H.1.6 upgrade — effective-pressure gating with a direct, phi-free "
                 "BedMachine effective pressure", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=135)
    plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--konrad", default=os.path.expanduser("~/data_glmig/konrad2018_glmig.txt"))
    ap.add_argument("--schmidtko", default=os.path.expanduser("~/data_glmig/schmidtko2014_shelf.txt"))
    ap.add_argument("--bin-dir", default=os.path.expanduser("~/data_bedmap/bedmap2_bin"))
    ap.add_argument("--bedmachine", default=BEDMACHINE)
    ap.add_argument("--out", default=os.path.join(_REPORTS, "efp_gate_direct_n.json"))
    a = ap.parse_args()
    res_bm, res_b2, diag, both = analyse(a.konrad, a.schmidtko, a.bin_dir, a.bedmachine)
    ok_bm, d_bm, ci_bm, lo_bm, hi_bm = _gate_ok(res_bm)
    ok_b2, d_b2, ci_b2, lo_b2, hi_b2 = _gate_ok(res_b2)
    Npa = diag["N_pa"][np.isfinite(diag["N_pa"])]
    verdict = (
        f"BedMachine direct phi-free N {'REPRODUCES' if ok_bm else 'does NOT reproduce'} "
        f"the marine-West gating (interaction d={d_bm:+.3f} CI[{ci_bm[0]:+.3f},{ci_bm[1]:+.3f}], "
        f"d ln u_*/dTF {lo_bm:+.2f}/degC near flotation vs {hi_bm:+.2f}/degC well grounded); "
        f"the Bedmap2-rel proxy gives d={d_b2:+.3f} CI[{ci_b2[0]:+.3f},{ci_b2[1]:+.3f}]. "
        f"{'The effective-pressure reading is robust to a more-direct, independent N.' if ok_bm and ok_b2 else ''}")
    out = {
        "what": ("§H.1.6 effective-pressure gating recomputed with a direct, phi-free "
                 "BedMachine effective pressure N = rho_i g (H - H_flot), n_hat=1-H_flot/H, "
                 "head-to-head vs the Bedmap2 rel=m/H proxy"),
        "N_definition": "N = rho_i g (H - H_flot); n_hat = 1 - H_flot/H (ocean-connected, phi-free)",
        "rho_i": RHO_I, "rho_w": RHO_W, "g": G,
        "median_N_MPa_at_GL": float(np.round(np.median(Npa) / 1e6, 4)) if Npa.size else None,
        "n_points_with_both_proxies": int(both.sum()),
        "bedmachine_direct_N": {"domains": res_bm, "gating_supported": ok_bm,
                                "interaction_d": d_bm, "interaction_ci": ci_bm},
        "bedmap2_rel_proxy": {"domains": res_b2, "gating_supported": ok_b2,
                              "interaction_d": d_b2, "interaction_ci": ci_b2},
        "verdict": verdict,
        "caveats": ("N assumes an ocean-connected hydraulic head at the bed (the relevant "
                    "limit near grounding lines); truly-measured N needs boreholes/seismic. "
                    "n_hat is the most-direct N obtainable from open gridded geometry. "
                    "Speed/N are correlated (fast ice rides near flotation); the interaction "
                    "term controls main effects, not every confound; bootstrap CIs ignore "
                    "spatial autocorrelation."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n=== §H.1.6 upgrade: direct phi-free BedMachine N ===")
    print(f"median N at GL: {out['median_N_MPa_at_GL']} MPa ; points with both proxies: {int(both.sum())}")
    print("VERDICT:", verdict)
    print(f"json -> {a.out}")
    make_figure(diag, res_bm, res_b2, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
