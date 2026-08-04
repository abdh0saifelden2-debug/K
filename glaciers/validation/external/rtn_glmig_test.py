r"""§H.1.3 — Does the §G.2 u_* law pace the grounding-line intrusion front?
 
Empirical test of the #5 "intrusion-clock" conjecture against OPEN observations
(no Earthdata auth):
 
  * v_obs  — observed grounding-line migration rate vgl [m/yr], 159k points,
             Konrad et al. (2018, Nat. Geosci.), CPOM data portal (open).
  * |grad m| — margin gradient of m = H - H* from Bedmap2 (this repo's loader),
             giving the level-set amplification A = 1/|grad m| [km advance / m thin].
  * dH/dt  — (optional) Schroeder et al. (2019, PANGAEA, CC-BY) SEC trend; if not
             present we run only the constant-free kinematic test (Test 1).
  * u_*    — proxy via driving stress tau_d = rho_i g H |grad s| from Bedmap2.
 
Provenance (open, no auth) — fetch once, then this script is reproducible::
 
    # v_obs: Konrad et al. 2018 grounding-line migration rates (~5 MB ASCII)
    curl -L -o ~/data_glmig/konrad2018_glmig.txt \
      https://www.cpom.ucl.ac.uk/csopr/icesheets3/data/gl_migration_konrad_et_al_2018.txt
    # dH/dt: Schroeder et al. 2019 multi-mission SEC (CC-BY, ~265 MB netCDF)
    curl -L -o ~/data_glmig/schroder_sec_mrg.nc \
      https://hs.pangaea.de/model/SchroderL-etal_2019/sec_mrg.nc
 
``build_dhdt_at_konrad`` then derives the per-point ``dH/dt`` (2010-2016 SEC
trend, nearest finite cell) and caches it to ``dhdt_at_konrad.npz``; ``analyse``
builds it on demand if the cache is absent but the SEC netCDF is present.
 
Tests
-----
Test 1 (constant-free, needs no dH/dt): Konrad's headline observation is that
  Antarctic ice streams retreat ~110 m per metre of thinning, i.e. the *observed*
  sensitivity v_obs/(dH/dt) ~ 0.11 km/m.  The level-set theory predicts this
  sensitivity *equals* A = 1/|grad m|.  We sample A at the 159k GL points and
  compare its distribution to 0.11 km/m.
 
Test 2 (needs dH/dt): per-point residence number Ro = v_kin/v_obs with
  v_kin = A * (dH/dt).  Ro ~ 1 => thinning-paced; Ro >> 1 => a rate-limiter
  (hydraulic clock) holds the front back.  Then regress log Ro on log tau_d
  (u_* proxy): a systematic slope is the signature that the §G.2 u_* law paces
  the front; a flat/scattered Ro falsifies the transfer (G.2 itself untouched).
"""
from __future__ import annotations
 
import json
import os
import sys
 
import numpy as np
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from external.bedmap2_loader import load_fields  # noqa: E402
from external.rtn_intrusion_clock import margin_field, amplification  # noqa: E402
 
RHO_I = 917.0
RHO_W = 1028.0
G = 9.81
KONRAD_OBS_SENSITIVITY_KM_PER_M = 0.110   # 110 m migration per m thinning (Konrad 2018)
 
 
def build_dhdt_at_konrad(sec_nc, konrad_txt, out_npz, t0=2010.0, t1=2017.0,
                         search_m=30000.0):
    """Derive per-Konrad-point dH/dt from the Schroeder SEC netCDF and cache it.
 
    Fits a per-cell linear SEC trend over [t0, t1] (Konrad's CryoSat-2 window),
    then samples the nearest finite-trend cell within ``search_m`` of each GL
    point (coastal SEC is NaN exactly at the GL, so the driver thinning is the
    adjacent finite cell). Writes ``dhdt_m_per_yr`` aligned to the Konrad rows.
    """
    import netCDF4 as nc
    from pyproj import Transformer
    from scipy.spatial import cKDTree
 
    ds = nc.Dataset(sec_nc)
    t = ds.variables["time"][:].astype("f8")
    x = ds.variables["x"][:].astype("f8")
    y = ds.variables["y"][:].astype("f8")
    w = (t >= t0) & (t <= t1)
    tt = t[w]
    S = np.array(ds.variables["sec"][w, :, :], dtype="f8")
    S = np.where(np.isfinite(S) & (np.abs(S) < 1e4), S, np.nan)
    tc = tt - tt.mean()
    num = np.nansum(tc[:, None, None] * (S - np.nanmean(S, axis=0, keepdims=True)),
                    axis=0)
    den = np.nansum(np.isfinite(S) * (tc[:, None, None] ** 2), axis=0)
    cnt = np.sum(np.isfinite(S), axis=0)
    dhdt = np.full(S.shape[1:], np.nan)
    gd = (cnt >= 12) & (den > 0)
    dhdt[gd] = num[gd] / den[gd]
 
    XX, YY = np.meshgrid(x, y)
    fin = np.isfinite(dhdt)
    tree = cKDTree(np.c_[XX[fin], YY[fin]])
    dvals = dhdt[fin]
 
    lat, lon = np.loadtxt(konrad_txt, skiprows=5, usecols=(0, 1), unpack=True)
    px, py = Transformer.from_crs("EPSG:4326", "EPSG:3031",
                                  always_xy=True).transform(lon, lat)
    dist, ii = tree.query(np.c_[px, py], distance_upper_bound=search_m)
    dh = np.where(np.isfinite(dist), dvals[np.clip(ii, 0, len(dvals) - 1)], np.nan)
    np.savez(out_npz, dhdt_m_per_yr=dh)
    return out_npz
 
 
def _grid_coords(meta, stride, n, axis_len, is_row):
    """Cell-center coordinate (m, EPSG:3031) for decimated index along one axis.

    The north-anchored y coordinate counts *down* from the top row, so it needs
    the true original row count. Reconstructing it as ``axis_len * stride`` is
    off by one whole row whenever the original grid dimension is odd (Bedmap2 is
    6667x6667; with stride=2, ``3334 * 2 = 6668 != 6667``), which shifts every
    coordinate ~1 km north. We therefore use the original grid size/cellsize
    recorded by the loader (``nrows_full``/``cellsize_full``) when present, and
    fall back to the old reconstruction only if those keys are absent. (The x
    branch counts *up* from the corner, so it was already exact.)

    ``axis_len`` is effectively *deprecated*: it is only read in the legacy
    fallback (``axis_len * stride``) for old caches that predate the loader's
    ``*_full`` metadata keys. All current callers pass ``axis_len == n`` and
    real ``load_fields`` metadata always carries ``nrows_full``, so the
    parameter is dead on the primary path; it is retained solely for
    backward-compatibility with those old caches.
    """
    cs = meta.get("cellsize_full", meta["cellsize"] / stride)   # original cellsize
    if is_row:                               # row 0 = northmost (max y)
        nrows_orig = meta.get("nrows_full", axis_len * stride)
        return meta["yll"] + (nrows_orig - np.arange(n) * stride - 0.5) * cs
    return meta["xll"] + (np.arange(n) * stride + 0.5) * cs
 
 
def analyse(bin_dir, konrad_txt, stride=2, phi=0.9, dhdt_npz=None):
    from pyproj import Transformer
 
    d = load_fields(bin_dir, stride=stride)
    H, bed, surf, mask = (d["thickness"], d["bed"], d["surface"],
                          d["icemask_grounded_and_shelves"])
    meta = d["_meta"]
    dx_km = meta["cellsize"] / 1000.0
    ny, nx = H.shape
 
    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)
    # margin m = H - H* and its level-set amplification A = 1/|grad m| via the
    # shared §H.1.2 driver functions (single source of truth; see
    # external.rtn_intrusion_clock) instead of an inline re-implementation.
    margin, H_star, d_base = margin_field(H, bed, phi)
    _grad_m, amp = amplification(margin, grounded, dx_km)  # A [km advance / m thin]
 
    # driving stress tau_d = rho_i g H |grad s|  (u_* proxy), s in m, grad over km->m
    s = np.where(grounded, surf, np.nan)
    sy, sx = np.gradient(s, dx_km * 1000.0, dx_km * 1000.0)
    tau_d = RHO_I * G * H * np.hypot(sx, sy)           # [Pa]
 
    # grid coordinates (EPSG:3031) of every cell, for nearest-grounded sampling
    ys = _grid_coords(meta, stride, ny, ny, is_row=True)
    xs = _grid_coords(meta, stride, nx, nx, is_row=False)
    XX, YY = np.meshgrid(xs, ys)
 
    lat, lon, vgl, dvgl = np.loadtxt(konrad_txt, skiprows=5, unpack=True)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    px, py = tr.transform(lon, lat)
 
    # sample A and tau_d at the nearest GROUNDED cell with a finite amplification
    # within a search radius (Konrad GL points sit on the grounded/shelf boundary;
    # the margin gradient that paces them is the adjacent grounded ice, <~10 km).
    from scipy.spatial import cKDTree
    valid = grounded & np.isfinite(amp) & (amp > 0)
    tree = cKDTree(np.c_[XX[valid], YY[valid]])
    A_src, tau_src = amp[valid], tau_d[valid]
    search_m = 10000.0
    dist, idx = tree.query(np.c_[px, py], distance_upper_bound=search_m)
    hit = np.isfinite(dist)
    A_pt = np.full(lat.shape, np.nan)
    tau_pt = np.full(lat.shape, np.nan)
    A_pt[hit] = A_src[idx[hit]]
    tau_pt[hit] = tau_src[idx[hit]]
    vgl_km = vgl / 1000.0
 
    ok = np.isfinite(A_pt) & np.isfinite(vgl_km) & (A_pt > 0) & (vgl_km > 0)
    A_ok = A_pt[ok]
 
    def pcts(a):
        return {k: float(np.nanpercentile(a, p))
                for k, p in (("p10", 10), ("p25", 25), ("p50", 50),
                             ("p75", 75), ("p90", 90))}
 
    summary = {
        "phi": phi, "stride": stride,
        "n_konrad_points": int(lat.size),
        "n_sampled_on_grounded": int(ok.sum()),
        "konrad_observed_sensitivity_km_per_m": KONRAD_OBS_SENSITIVITY_KM_PER_M,
        "kinematic_A_km_per_m": pcts(A_ok),
        "vgl_obs_km_per_yr": pcts(vgl_km[ok]),
    }
 
    # Test 2: needs an independent dH/dt per point. Build the cache on demand from
    # the Schroeder SEC netCDF if it sits next to the npz path (see module docstring).
    if dhdt_npz and not os.path.exists(dhdt_npz):
        sec_nc = os.path.join(os.path.dirname(dhdt_npz), "schroder_sec_mrg.nc")
        if os.path.exists(sec_nc):
            build_dhdt_at_konrad(sec_nc, konrad_txt, dhdt_npz)
    if dhdt_npz and os.path.exists(dhdt_npz):
        z = np.load(dhdt_npz)
        dhdt_pt = z["dhdt_m_per_yr"]                  # same order as konrad rows
        thin = -dhdt_pt                                # thinning positive
        good = ok & np.isfinite(thin) & (thin > 0.1) & np.isfinite(tau_pt) & (tau_pt > 0)
        v_kin = A_pt[good] * thin[good]               # km/yr
        Ro = v_kin / vgl_km[good]
        lro, ltau = np.log(Ro), np.log(tau_pt[good])
        slope, intercept = np.polyfit(ltau, lro, 1)
        r = float(np.corrcoef(ltau, lro)[0, 1])
        summary["test2_residence_number"] = {
            "n": int(good.sum()),
            "Ro": pcts(Ro),
            "Ro_median": float(np.median(Ro)),
            "loglog_slope_Ro_vs_tau_d": float(slope),
            "loglog_corr_r": r,
            "interpretation": (
                "Ro~1 => thinning-paced; Ro>>1 => hydraulic-limited. "
                "slope~0/|r| small => no u_* organisation (transfer falsified); "
                "systematic slope => u_* law paces the front."),
        }
        pts = {"v_obs": vgl_km[good], "v_kin": v_kin, "Ro": Ro,
               "tau_d": tau_pt[good], "slope": float(slope),
               "intercept": float(intercept), "r": r}
    else:
        pts = None
    return summary, pts
 
 
def make_figure(summary, pts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.3))
 
    # Panel A: observed vs kinematic front speed (Ro = v_kin/v_obs)
    vo, vk = pts["v_obs"] * 1000, pts["v_kin"] * 1000   # m/yr
    axa.hexbin(vo, vk, xscale="log", yscale="log", gridsize=45,
               mincnt=1, cmap="viridis")
    lim = [min(vo.min(), vk.min()), max(vo.max(), vk.max())]
    axa.plot(lim, lim, "w--", lw=1.5, label="Ro=1 (thinning-paced)")
    axa.set_xlabel("observed GL migration v_obs  [m/yr]  (Konrad 2018)")
    axa.set_ylabel("kinematic v_kin = A·(dH/dt)  [m/yr]")
    axa.set_title(f"Ro = v_kin/v_obs  (median {summary['test2_residence_number']['Ro_median']:.2f}, "
                  f"n={summary['test2_residence_number']['n']})")
    axa.legend(loc="upper left", fontsize=9)
 
    # Panel B: the u_* (sqrt-law) discriminant — Ro vs driving stress
    td, ro = pts["tau_d"] / 1000.0, pts["Ro"]            # kPa
    axb.hexbin(td, ro, xscale="log", yscale="log", gridsize=45,
               mincnt=1, cmap="magma")
    xx = np.array([td.min(), td.max()])
    axb.plot(xx, np.exp(pts["intercept"]) * xx ** pts["slope"], "c-", lw=2,
             label=f"slope={pts['slope']:.2f}, r={pts['r']:.2f}")
    axb.axhline(1.0, color="w", ls=":", lw=1)
    axb.set_xlabel(r"driving stress $\tau_d=\rho_i g H|\nabla s|$  [kPa]  (u_* proxy)")
    axb.set_ylabel("residence number Ro")
    axb.set_title("u_* discriminant: Ro vs τ_d (flat ⇒ no u_* pacing)")
    axb.legend(loc="upper right", fontsize=9)
 
    fig.suptitle("§H.1.3 — does the §G.2 u_* law pace the grounding-line front? "
                 "(Bedmap2 + Konrad 2018 v_obs + Schröder 2019 dH/dt)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    print(f"figure -> {os.path.abspath(path)}")
 
 
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default=os.path.expanduser(
        "~/data_bedmap/bedmap2_bin"))
    ap.add_argument("--konrad", default=os.path.expanduser(
        "~/data_glmig/konrad2018_glmig.txt"))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--phi", type=float, default=0.9)
    ap.add_argument("--dhdt-npz", default=os.path.expanduser(
        "~/data_glmig/dhdt_at_konrad.npz"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "reports", "rtn_glmig_test.json"))
    a = ap.parse_args()
 
    summary, pts = analyse(a.bin_dir, a.konrad, stride=a.stride, phi=a.phi,
                           dhdt_npz=a.dhdt_npz)
    print(json.dumps(summary, indent=2))
    obs = summary["konrad_observed_sensitivity_km_per_m"]
    med = summary["kinematic_A_km_per_m"]["p50"]
    print(f"\nTest 1: kinematic A median = {med:.3f} km/m  vs  "
          f"Konrad observed {obs:.3f} km/m  -> ratio {med/obs:.2f}")
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"json -> {os.path.abspath(a.out)}")
    if pts is not None:
        make_figure(summary, pts,
                    os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                 "rtn_glmig_test.png"))
 
 
if __name__ == "__main__":
    main()
