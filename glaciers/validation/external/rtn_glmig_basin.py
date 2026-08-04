r"""Follow-up to §H.1.3 — the u_* discriminant is regime-dependent.

§H.1.3 reported that, at continental scale, the residence number Ro = v_kin/v_obs
shows no organisation against the driving-stress proxy tau_d (log-log slope -0.11,
r -0.06) and concluded the §G.2 u_* law does not pace the grounding-line front.

This script shows that verdict is a Simpson's-paradox artefact compounded by a poor
proxy. Two robustness checks:

  (1) Per-basin split. Splitting the Konrad GL points into Antarctic sectors shows
      the marine, fast-streaming West-Antarctic basins (Amundsen, Bellingshausen/
      WAIS-Pacific) carry a *positive* Ro-vs-proxy slope, while the cold East-
      Antarctic interior carries a *negative* one. Averaging opposite-signed
      regimes gives a spurious "flat" continental result.

  (2) Sharper u_* proxy. tau_d = rho_i g H |grad s| is a driving-stress proxy that
      *anti*-correlates with speed in streaming ice (ice streams slide on low basal
      drag: low tau_d, high speed). Substituting measured ITS_LIVE surface speed —
      a direct u_* proxy — flips the continental slope from -0.06 to +0.15 and
      reveals the positive marine organisation that tau_d hid.

Reuses the committed sampling from rtn_glmig_test (same grids, masks, radii) but
retains per-point lat/lon so points can be binned by basin. Pass --itslive-npz
(produced by itslive_glmig_sample.py) to add the direct-speed proxy and figure.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from external.bedmap2_loader import load_fields  # noqa: E402
from rtn_glmig_test import _grid_coords, build_dhdt_at_konrad  # noqa: E402

RHO_I = 917.0
RHO_W = 1028.0
G = 9.81

# Antarctic sectors by lon/lat box. These boxes are deliberately NOT mutually
# exclusive (the per-sector slopes are robustness probes, not a partition):
#  - the Amundsen box is nested inside the Bellingshausen/WAIS-Pacific box; and
#  - Ross/Siple (150E -> -150W) and East Antarctica (-20E -> 160E) overlap in
#    the [150E, 160E] band, so the few grounding-line points there (south of
#    70S, the stricter Ross lat bound) are counted in BOTH sectors.
# Impact is minimal (a narrow band near the sector edge) and the CONTINENTAL row
# in _report is the single non-overlapping aggregate, so read the per-sector
# table in §H.1.4 knowing the sectors are not disjoint. The Ross/Siple sector
# crosses the +/-180 dateline, so its lon range has lo0 > lo1 and is matched by
# the wrap-aware branch in _in_sector.
SECTORS = {
    "Amundsen (PIG/Thwaites/Smith)": dict(lon=(-115.0, -95.0), lat=(-76.5, -71.0)),
    "Bellingshausen+WAIS-Pacific":   dict(lon=(-140.0, -60.0), lat=(-90.0, -65.0)),
    "Ross/Siple Coast":              dict(lon=(150.0, -150.0), lat=(-90.0, -70.0)),
    "East Antarctica":               dict(lon=(-20.0, 160.0),  lat=(-90.0, -64.0)),
}


def _in_sector(lon, lat, spec):
    (lo0, lo1), (la0, la1) = spec["lon"], spec["lat"]
    in_lat = (lat >= la0) & (lat <= la1)
    if lo0 <= lo1:
        in_lon = (lon >= lo0) & (lon <= lo1)
    else:                                  # dateline-crossing box (e.g. Ross: 150E..-150W)
        in_lon = (lon >= lo0) | (lon <= lo1)
    return in_lon & in_lat


def _fit(ro, x, nboot=1000, seed=0):
    """log-log slope/r of Ro vs proxy x (both > 0), with bootstrap 95% CI on slope.

    The bootstrap RNG is re-seeded (``seed=0``) on every call by design, for
    reproducibility. This means the resampling *index pattern* is identical from
    one sector to the next, but each call resamples a *different* subsetted
    (Ro, x) array, so the per-sector CIs are statistically independent; the fixed
    seed only makes each one deterministic, not correlated across sectors.
    """
    m = np.isfinite(ro) & np.isfinite(x) & (ro > 0) & (x > 0)
    out = {"n": int(m.sum()),
           "Ro_median": float(np.median(ro[m])) if m.any() else None,
           "slope": None, "r": None, "ci95": None}
    if m.sum() >= 30:
        lr, lx = np.log(ro[m]), np.log(x[m])
        slope, _ = np.polyfit(lx, lr, 1)
        out["slope"] = float(slope)
        out["r"] = float(np.corrcoef(lx, lr)[0, 1])
        rng = np.random.default_rng(seed)
        idx = np.arange(lx.size)
        bs = np.empty(nboot)
        for b in range(nboot):
            j = rng.choice(idx, idx.size, replace=True)   # one paired resample
            bs[b] = np.polyfit(lx[j], lr[j], 1)[0]
        out["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def sample(bin_dir, konrad_txt, dhdt_npz, stride=2, phi=0.9):
    """Return per-good-point arrays: lon, lat, Ro, tau_d (driving-stress proxy).

    Unlike ``rtn_glmig_test.analyse`` (which treats dH/dt as optional and falls
    back to the constant-free Test 1), ``sample`` *requires* a usable ``dhdt_npz``
    and raises ``FileNotFoundError`` if it is neither present nor buildable from a
    neighbouring Schroeder-2019 SEC netCDF. This is intentional: the residence
    number ``Ro = A * (dH/dt) / v_obs`` that the basin analysis regresses cannot
    be formed without per-point thinning rates, so there is no meaningful
    dH/dt-free fallback here.
    """
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    d = load_fields(bin_dir, stride=stride)
    H, bed, surf, mask = (d["thickness"], d["bed"], d["surface"],
                          d["icemask_grounded_and_shelves"])
    meta = d["_meta"]
    dx_km = meta["cellsize"] / 1000.0
    ny, nx = H.shape

    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)
    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    H_star = (RHO_W / (phi * RHO_I)) * d_base
    margin = np.where(grounded, H - H_star, np.nan)

    gy, gx = np.gradient(margin, dx_km, dx_km)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = 1.0 / np.hypot(gx, gy)

    s = np.where(grounded, surf, np.nan)
    sy, sx = np.gradient(s, dx_km * 1000.0, dx_km * 1000.0)
    tau_d = RHO_I * G * H * np.hypot(sx, sy)

    ys = _grid_coords(meta, stride, ny, ny, is_row=True)
    xs = _grid_coords(meta, stride, nx, nx, is_row=False)
    XX, YY = np.meshgrid(xs, ys)

    lat, lon, vgl, _ = np.loadtxt(konrad_txt, skiprows=5, unpack=True)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    px, py = tr.transform(lon, lat)

    valid = grounded & np.isfinite(amp) & (amp > 0)
    tree = cKDTree(np.c_[XX[valid], YY[valid]])
    A_src, tau_src = amp[valid], tau_d[valid]
    dist, idx = tree.query(np.c_[px, py], distance_upper_bound=10000.0)
    hit = np.isfinite(dist)
    A_pt = np.full(lat.shape, np.nan); A_pt[hit] = A_src[idx[hit]]
    tau_pt = np.full(lat.shape, np.nan); tau_pt[hit] = tau_src[idx[hit]]
    vgl_km = vgl / 1000.0

    if dhdt_npz and not os.path.exists(dhdt_npz):
        sec_nc = os.path.join(os.path.dirname(dhdt_npz), "schroder_sec_mrg.nc")
        if os.path.exists(sec_nc):
            build_dhdt_at_konrad(sec_nc, konrad_txt, dhdt_npz)
    if not (dhdt_npz and os.path.exists(dhdt_npz)):
        raise FileNotFoundError(
            f"per-point dH/dt cache not found at {dhdt_npz!r} and could not be built "
            f"(needs the Schroeder 2019 SEC netCDF as 'schroder_sec_mrg.nc' beside it; "
            f"see the 'glmig-data' env knowledge entry / module docstring). The "
            f"residence-number analysis requires per-point thinning rates.")
    thin = -np.load(dhdt_npz)["dhdt_m_per_yr"]

    good = (np.isfinite(A_pt) & (A_pt > 0) & np.isfinite(vgl_km) & (vgl_km > 0)
            & np.isfinite(thin) & (thin > 0.1) & np.isfinite(tau_pt) & (tau_pt > 0))
    Ro = (A_pt[good] * thin[good]) / vgl_km[good]
    return {"lon": lon[good], "lat": lat[good], "Ro": Ro, "tau_d": tau_pt[good]}


def _report(lon, lat, Ro, proxies):
    """proxies: dict name -> array. Returns nested summary and prints a table."""
    names = list(SECTORS) + ["CONTINENTAL"]
    masks = [_in_sector(lon, lat, SECTORS[n]) for n in SECTORS] + [np.ones(lon.size, bool)]
    summ = {}
    print(f"\n{'region':30s} {'proxy':12s} {'slope':>8s} {'ci95':>20s} {'r':>7s} {'Ro_med':>7s} {'n':>6s}")
    for name, m in zip(names, masks):
        summ[name] = {}
        for pname, arr in proxies.items():
            f = _fit(Ro[m], arr[m])
            summ[name][pname] = f
            ci = f"[{f['ci95'][0]:+.3f},{f['ci95'][1]:+.3f}]" if f["ci95"] else "  --  "
            sl = f"{f['slope']:+.3f}" if f["slope"] is not None else "  -- "
            rr = f"{f['r']:+.3f}" if f["r"] is not None else "  -- "
            rm = f"{f['Ro_median']:.2f}" if f["Ro_median"] is not None else " -- "
            print(f"{name:30s} {pname:12s} {sl:>8s} {ci:>20s} {rr:>7s} {rm:>7s} {f['n']:>6d}")
    return summ


def make_figure(lon, lat, Ro, tau, spd, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#d1495b", "#edae49", "#66a182", "#2e4057"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    names = list(SECTORS) + ["CONTINENTAL"]
    masks = [_in_sector(lon, lat, SECTORS[n]) for n in SECTORS] + [np.ones(lon.size, bool)]
    yc = np.arange(len(names))[::-1]
    for arr, off, col, lab in [(tau, +0.16, "#9aa0a6", r"$\tau_d=\rho gH|\nabla s|$ (driving stress)"),
                               (spd, -0.16, "#1f77b4", "ITS_LIVE surface speed (direct $u_*$ proxy)")]:
        labeled = False
        for k, m in enumerate(masks):
            f = _fit(Ro[m], arr[m])
            if f["slope"] is None:
                continue
            lo, hi = f["ci95"]
            # Attach the proxy's legend label to the first sector that actually
            # has data, not unconditionally to k == 0: if the first sector were
            # filtered out (slope is None) the legend entry would otherwise be lost.
            ax0.errorbar(f["slope"], yc[k] + off, xerr=[[f["slope"] - lo], [hi - f["slope"]]],
                         fmt="o", color=col, capsize=3, ms=6, label=None if labeled else lab)
            labeled = True
            ax0.text(hi + 0.03, yc[k] + off, f"r={f['r']:+.2f}", va="center", fontsize=8, color=col)
    ax0.axvline(0, color="k", lw=1, ls="--")
    ax0.set_yticks(yc); ax0.set_yticklabels([n.replace("+", "\n+") for n in names], fontsize=8.5)
    ax0.set_xlabel(r"log-log slope of  Ro  vs  $u_*$ proxy   (>0 $\Rightarrow$ $u_*$-paced rate-limiting)")
    ax0.set_title("Per-basin $u_*$ discriminant (95% bootstrap CI)")
    ax0.legend(loc="lower right", fontsize=8.5); ax0.grid(axis="x", alpha=0.3)

    for (name, spec), col in zip(SECTORS.items(), colors):
        m = _in_sector(lon, lat, spec) & np.isfinite(spd) & (spd > 0) & (Ro > 0)
        ax1.scatter(spd[m], Ro[m], s=6, alpha=0.25, color=col, label=name)
        f = _fit(Ro[m], spd[m])
        if f["slope"] is None:
            continue
        b = np.polyfit(np.log(spd[m]), np.log(Ro[m]), 1)[1]
        xx = np.array([np.nanpercentile(spd[m], 2), np.nanpercentile(spd[m], 98)])
        ax1.plot(xx, np.exp(b) * xx ** f["slope"], color=col, lw=2.4)
    ax1.axhline(1.0, color="k", ls=":", lw=1)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(r"ITS_LIVE surface speed  [m/yr]   ($u_*$ proxy)")
    ax1.set_ylabel("residence number  Ro = v_kin / v_obs")
    ax1.set_title("Marine streaming sectors: Ro rises with $u_*$ (interior falls)")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle("§H.1.4 — the $u_*$ signal is regime-dependent: a direct speed proxy reveals "
                 "positive $u_*$ organisation in the marine\nWest-Antarctic sectors that the "
                 "driving-stress proxy and continental average hid (Simpson's paradox)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=135)
    print(f"figure -> {os.path.abspath(path)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default=os.path.expanduser("~/data_bedmap/bedmap2_bin"))
    ap.add_argument("--konrad", default=os.path.expanduser("~/data_glmig/konrad2018_glmig.txt"))
    ap.add_argument("--dhdt-npz", default=os.path.expanduser("~/data_glmig/dhdt_at_konrad.npz"))
    ap.add_argument("--itslive-npz", default=os.path.expanduser(
        "~/data_glmig/itslive_speed_at_konrad_good.npz"),
        help="cached ITS_LIVE speed at good points (from itslive_glmig_sample.py)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports", "rtn_glmig_basin.json"))
    a = ap.parse_args()

    r = sample(a.bin_dir, a.konrad, a.dhdt_npz)
    lon, lat, Ro, tau = r["lon"], r["lat"], r["Ro"], r["tau_d"]
    proxies = {"tau_d": tau}
    spd = None
    if a.itslive_npz and os.path.exists(a.itslive_npz):
        z = np.load(a.itslive_npz)
        # align by good-point order (sampler stores the same lon/lat/Ro/tau)
        if z["Ro"].shape == Ro.shape and np.allclose(z["Ro"], Ro, equal_nan=True):
            spd = z["speed"]
            proxies["ITS_LIVE_speed"] = spd
        else:
            print("WARNING: itslive npz does not align with current good-point set; "
                  "re-run itslive_glmig_sample.py. Skipping speed proxy.")

    summ = _report(lon, lat, Ro, proxies)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(summ, fh, indent=2)
    print(f"\njson -> {os.path.abspath(a.out)}")

    if spd is not None:
        make_figure(lon, lat, Ro, tau, spd,
                    os.path.join(os.path.dirname(os.path.abspath(a.out)), "rtn_glmig_basin.png"))


if __name__ == "__main__":
    main()
