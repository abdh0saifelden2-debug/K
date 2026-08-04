r"""NR45 (ledger E11) -- ocean microrheology: the mesoscale ocean has a
measurable complex modulus, and its elasticity is carried by the eddy-trapped
(looper) floats.

Ledger E11 conjectured that Argo float dispersion, read through the
generalized Stokes-Einstein relation (GSER; Mason & Weitz 1995), turns the
committed non-Markov ocean-memory result (REPORT_NONMARKOV_ARGO.md) into a
*complex modulus* ``G*(omega)`` of the mesoscale ocean -- a new measurement
framing, not a new microphysical law.

Data: ANDRO deep-displacement atlas (Ollitrault & Rannou; SEANOE
doi:10.17882/47077, CC-BY): per-cycle park-depth displacements/velocities for
the global Argo array.  Filtered to park pressure 700-1300 dbar, |u| < 1.5
m/s, valid deep fixes: 1.38M cycles, 9379 floats, median cadence 10.0 days.
The companion 0.5-degree gridded atlas (mean_u, mean_v, EKE; the .nc in the
same deposit) supplies an *independent* Eulerian mean for residual-velocity
construction.  Raw files are NOT committed (0.5 GB); the derived per-float
accumulators sufficient to re-verify every verdict are committed to
``data/nr45_andro_gser_cache.json`` and regenerate bit-for-bit from the raw
deposit (``$ANDRO_DAT``, ``$ANDRO_NC``).

Method (all standard, composed):

1. per float: residual velocity ``v' = v_park - v_atlas(x)`` (atlas mean, NOT
   the float's own record mean -- see the bias control below); segments of
   near-regular cadence (7-13 d); per-float VACF and displacement-sum MSD
   accumulators to lag 300 d; per-float spin ``Omega = <u'v'_+1 - v'u'_+1> /
   <u'^2+v'^2>`` (discrete rotation measure -- "loopers" = top |spin|
   quartile).
2. ensemble MSD(t) -> local exponent ``alpha(t) = dln MSD/dln t`` ->
   Mason-form GSER: ``|G*(1/t)| ∝ 1/[MSD(t)·Gamma(1+alpha(t))]``,
   ``delta = pi*alpha/2``, so the **loss tangent tan(delta) and the elastic
   fraction G'/G'' = 1/tan(pi*alpha/2) are normalization-free** -- no
   ocean-"kT", no tracer radius, no calibration.
3. the finite-record bias control: removing the float's OWN record-mean
   velocity forces MSD to bend down at record-length lags (an apparent
   elastic window).  Removing the independent atlas mean instead is unbiased.
   Both are computed; the difference is reported as the bias, not physics.

Findings (figures/80_andro_gser.json):

* **The mesoscale ocean at ~1000 dbar is a viscoelastic LIQUID with a
  measured terminal-relaxation crossover.**  North-Atlantic box (25-45N,
  20-65W; 301 floats): ``alpha`` relaxes from 1.43 at 10 d through the
  velocity-decorrelation knee (VACF e-fold ~ 15 d, zero-crossing ~ 50 d) to
  ``alpha = 1.00-1.01`` across 90-140 d -- a clean terminal (pure-viscous)
  window with eddy diffusivity ``K = MSD/4t ~ 2.2e3 m^2/s`` at 100 d.
* **The elasticity is subpopulation-carried.**  Loopers (top |spin| quartile,
  n=76) dip to ``alpha = 0.86-0.95`` (median 0.91) over 100-190 d -- an
  elastic fraction ``G'/G'' = tan((1-alpha)pi/2) ~ 0.09-0.22`` at 3-6-month
  periods -- while non-loopers hold ``alpha ~ 1.00`` there (median 1.00;
  1.03 at 90-140 d -- elastic fraction consistent with 0).
  In GSER language: the eddy-trapped subpopulation feels the mesoscale field
  as a weak elastic solid at the eddy-coherence timescale; the untrapped
  majority feels a simple liquid.  This is the two-clocks split as
  *rheology*: fast rotational (trapped) clock => storage G'; slow dispersive
  clock => loss G''.
* **The float-mean bias is real and quantified**: with record-mean removal
  the 150-190 d ensemble median drops to ``alpha = 0.86`` vs ``0.93`` for the
  atlas-mean control (bias 0.07, i.e. a fake elastic window at record-length
  lags).  Prior Lagrangian "subdiffusion" claims at these lags should be read
  with this control in hand.
* Global (|lat|<60, 6936 floats): pooled ensemble ``alpha[90-140 d] = 1.18``
  (mixture superdiffusion from cross-float K heterogeneity -- a
  Richardson-like artefact of pooling, flagged, not claimed as physics).

CPU-only.  Tests: tests/test_andro_gser.py.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr45_andro_gser_cache.json")
FIG = os.path.join(HERE, "figures", "80_andro_gser.json")

DT_DAYS = 10.0
MAXLAG = 30
NA_BOX = dict(lat=(25.0, 45.0), lon=(-65.0, -20.0))
LOOPER_Q = 0.75


# --------------------------------------------------------------------------- #
# raw ingest (only when the ANDRO deposit is present)
# --------------------------------------------------------------------------- #
def _atlas(nc_path):
    import netCDF4 as nc
    a = nc.Dataset(nc_path)
    alon = np.array(a["longitude"][:])
    alat = np.array(a["latitude"][:])
    mu = np.array(a["mean_u"][:])
    mv = np.array(a["mean_v"][:])
    mu = np.where(np.abs(mu) > 1e3, np.nan, mu)
    mv = np.where(np.abs(mv) > 1e3, np.nan, mv)

    def at(lon, lat):
        i = np.clip(np.searchsorted(alon, lon), 0, len(alon) - 1)
        j = np.clip(np.searchsorted(alat, lat), 0, len(alat) - 1)
        return mu[i, j], mv[i, j]
    return at


def load_andro(dat_path=None, nc_path=None):
    import pandas as pd
    dat_path = dat_path or os.environ.get("ANDRO_DAT")
    df = pd.read_csv(dat_path, sep=r"\s+", header=None,
                     usecols=[0, 1, 2, 5, 6, 7, 8, 9, 34, 35],
                     names=["lon", "lat", "pres", "juld", "u", "v",
                            "eu", "ev", "wmo", "cyc"],
                     dtype=float, engine="c")
    m = (df.pres.between(700, 1300) & df.u.gt(-990) & df.v.gt(-990)
         & df.juld.gt(-9990) & df.u.abs().lt(150) & df.v.abs().lt(150)
         & df.lon.gt(-990) & df.lat.gt(-99))
    return df[m].copy()


def per_float_accumulators(d, atlas_at, *, mode="atlas", maxlag=MAXLAG,
                           min_seg=12):
    """Per-float (spin, MSD, VACF) accumulators from residual velocities."""
    out = []
    dts = 86400.0 * DT_DAYS
    for wmo, sub in d.groupby("wmo"):
        if len(sub) < 30:
            continue
        sub = sub.sort_values("juld")
        t = sub.juld.values
        u = sub.u.values / 100.0
        v = sub.v.values / 100.0
        if mode == "atlas":
            au, av = atlas_at(sub.lon.values, sub.lat.values)
            ur_all, vr_all = u - au / 100.0, v - av / 100.0
        else:
            ur_all, vr_all = u - u.mean(), v - v.mean()
        segs, cur = [], [0]
        for i in range(1, len(t)):
            if (7.0 <= t[i] - t[i - 1] <= 13.0 and np.isfinite(ur_all[i])
                    and np.isfinite(ur_all[i - 1])):
                cur.append(i)
            else:
                if len(cur) >= min_seg:
                    segs.append(np.array(cur))
                cur = [i]
        if len(cur) >= min_seg:
            segs.append(np.array(cur))
        if not segs:
            continue
        num = den = 0.0
        msd_s = np.zeros(maxlag + 1)
        msd_c = np.zeros(maxlag + 1)
        vac_n = np.zeros(maxlag + 1)
        vac_c = np.zeros(maxlag + 1)
        for idx in segs:
            ur, vr = ur_all[idx], vr_all[idx]
            if not (np.isfinite(ur).all() and np.isfinite(vr).all()):
                continue
            n = len(ur)
            num += np.sum(ur[:-1] * vr[1:] - vr[:-1] * ur[1:])
            den += np.sum(ur ** 2 + vr ** 2)
            c0 = (ur ** 2 + vr ** 2).mean()
            if c0 <= 0:
                continue
            for m in range(0, min(maxlag, n - 1) + 1):
                c = (ur[:n - m] * ur[m:] + vr[:n - m] * vr[m:]).mean()
                vac_n[m] += c / c0 * (n - m)
                vac_c[m] += n - m
            dx = np.insert(np.cumsum(ur * dts), 0, 0.0)
            dy = np.insert(np.cumsum(vr * dts), 0, 0.0)
            for m in range(1, min(maxlag, n - 1) + 1):
                ddx = dx[m:] - dx[:-m]
                ddy = dy[m:] - dy[:-m]
                msd_s[m] += np.sum(ddx ** 2 + ddy ** 2)
                msd_c[m] += len(ddx)
        if den <= 0 or msd_c[1] == 0:
            continue
        out.append(dict(wmo=float(wmo), spin=float(num / den),
                        msd_s=msd_s.tolist(), msd_c=msd_c.tolist(),
                        vac_n=vac_n.tolist(), vac_c=vac_c.tolist()))
    return out


def build_cache(dat_path=None, nc_path=None, write=True):
    d = load_andro(dat_path)
    at = _atlas(nc_path or os.environ.get("ANDRO_NC"))
    box = d[d.lat.between(*NA_BOX["lat"]) & d.lon.between(*NA_BOX["lon"])]
    cache = dict(
        meta=dict(n_cycles_valid=int(len(d)), n_floats=int(d.wmo.nunique()),
                  na_box=NA_BOX, dt_days=DT_DAYS, maxlag=MAXLAG,
                  provenance="ANDRO deep displacements, SEANOE "
                             "doi:10.17882/47077 (CC-BY); atlas mean from the "
                             "same deposit"),
        na_atlas=per_float_accumulators(box, at, mode="atlas"),
        na_floatmean=per_float_accumulators(box, at, mode="floatmean"),
        global_atlas_ens=_ens_only(per_float_accumulators(
            d[d.lat.abs() < 60], at, mode="atlas")),
    )
    if write:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w") as fh:
            json.dump(cache, fh)
    return cache


def _ens_only(rows):
    S = np.sum([r["msd_s"] for r in rows], axis=0)
    C = np.sum([r["msd_c"] for r in rows], axis=0)
    V = np.sum([r["vac_n"] for r in rows], axis=0)
    W = np.sum([r["vac_c"] for r in rows], axis=0)
    return dict(msd_s=S.tolist(), msd_c=C.tolist(), vac_n=V.tolist(),
                vac_c=W.tolist(), n_floats=len(rows))


def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# ensemble curves + GSER
# --------------------------------------------------------------------------- #
def ensemble(rows):
    S = np.sum([r["msd_s"] for r in rows], axis=0)
    C = np.sum([r["msd_c"] for r in rows], axis=0)
    V = np.sum([r["vac_n"] for r in rows], axis=0)
    W = np.sum([r["vac_c"] for r in rows], axis=0)
    msd = S / np.maximum(C, 1)
    vacf = V / np.maximum(W, 1)
    return msd, vacf


def alpha_of_t(msd, lags_days=None):
    lags = (np.arange(len(msd)) * DT_DAYS) if lags_days is None else lags_days
    return np.gradient(np.log(msd[1:]), np.log(lags[1:]))


def gser(msd):
    """Mason-form GSER, normalization-free parts: alpha, delta, G'/G''.

    For MSD ~ t^alpha the modulus phase is delta = pi*alpha/2 (0<alpha<=1:
    viscoelastic; alpha=1: pure viscous).  We report the elastic fraction
    G'/G'' = 1/tan(pi*alpha/2) clipped at alpha<=1, and |G*| up to a constant
    via 1/[MSD(t)*Gamma(1+alpha)]."""
    lags = np.arange(len(msd)) * DT_DAYS
    a = alpha_of_t(msd)
    Gmag = np.array([1.0 / (msd[i] * math.gamma(1.0 + min(max(a[i - 1], 0.01),
                                                          1.99)))
                     for i in range(1, len(msd))])
    delta = np.pi * np.clip(a, 0.01, 1.99) / 2.0
    with np.errstate(divide="ignore"):
        elastic_frac = np.where(a < 1.0,
                                1.0 / np.tan(np.pi * np.clip(a, 0.01, 1.0)
                                             / 2.0), 0.0)
    return dict(lags_days=lags[1:].tolist(), alpha=a.tolist(),
                Gmag_arb=Gmag.tolist(), delta_rad=delta.tolist(),
                elastic_frac=elastic_frac.tolist())


def window_median(arr, lags, lo, hi):
    lags = np.asarray(lags)
    arr = np.asarray(arr)
    sel = (lags >= lo) & (lags <= hi)
    return float(np.median(arr[sel]))


def analyze(cache):
    rows = cache["na_atlas"]
    rows_fm = cache["na_floatmean"]
    spins = np.array([abs(r["spin"]) for r in rows])
    thr = float(np.quantile(spins, LOOPER_Q))
    loop = [r for r in rows if abs(r["spin"]) >= thr]
    rest = [r for r in rows if abs(r["spin"]) < thr]

    lags = np.arange(1, MAXLAG + 1) * DT_DAYS
    out = dict(looper_threshold=thr, n_loopers=len(loop), n_rest=len(rest),
               n_floats=len(rows))
    for tag, rr in (("ens", rows), ("loop", loop), ("rest", rest),
                    ("ens_floatmean", rows_fm)):
        msd, vacf = ensemble(rr)
        a = alpha_of_t(msd)
        out[tag] = dict(msd=msd.tolist(), vacf=vacf.tolist(),
                        alpha=a.tolist(),
                        alpha_90_140=window_median(a, lags, 90, 140),
                        alpha_100_190=window_median(a, lags, 100, 190),
                        alpha_150_190=window_median(a, lags, 150, 190))
    g = _ens_only(cache["na_atlas"])  # container consistency
    ge = cache["global_atlas_ens"]
    msd_g = np.array(ge["msd_s"]) / np.maximum(np.array(ge["msd_c"]), 1)
    a_g = alpha_of_t(msd_g)
    out["global"] = dict(alpha_90_140=window_median(a_g, lags, 90, 140),
                         n_floats=ge["n_floats"])
    # eddy diffusivity at the terminal window (100 d)
    msd_ens = np.array(out["ens"]["msd"])
    out["K_m2_s_100d"] = float(msd_ens[10] / (4.0 * 10 * DT_DAYS * 86400.0))
    out["gser_ens"] = gser(np.array(out["ens"]["msd"]))
    out["gser_loop"] = gser(np.array(out["loop"]["msd"]))
    ef = np.array(out["gser_loop"]["elastic_frac"])
    gl = np.array(out["gser_loop"]["lags_days"])
    sel = (gl >= 100) & (gl <= 190)
    out["looper_elastic_frac_100_190"] = [float(np.min(ef[sel])),
                                          float(np.max(ef[sel]))]
    # VACF knees
    v_ens = np.array(out["ens"]["vacf"])
    zero_cross = np.argmax(v_ens < 0.0)
    out["vacf_zero_cross_days"] = float(zero_cross * DT_DAYS)
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    lags = np.arange(1, MAXLAG + 1) * DT_DAYS
    a_ens = np.array(out["ens"]["alpha"])
    a_loop = np.array(out["loop"]["alpha"])
    a_rest = np.array(out["rest"]["alpha"])
    a_fm = np.array(out["ens_floatmean"]["alpha"])
    w = lambda a, lo, hi: window_median(a, lags, lo, hi)
    terminal = abs(w(a_ens, 90, 140) - 1.0) <= 0.05
    bias = w(a_fm, 150, 190) < w(a_ens, 150, 190) - 0.04
    split = (w(a_loop, 100, 190) <= 0.95) and (w(a_rest, 90, 140) >= 0.97)
    ef_lo, ef_hi = out["looper_elastic_frac_100_190"]
    return dict(
        terminal_viscous_window=bool(terminal),
        floatmean_bias_quantified=bool(bias),
        elasticity_subpopulation_carried=bool(split),
        looper_elastic_frac_range=[ef_lo, ef_hi],
        looper_elastic_detected=bool(ef_hi >= 0.10),
        K_plausible=bool(500.0 <= out["K_m2_s_100d"] <= 5000.0),
        reading=(
            "GSER on ANDRO deep displacements: the mesoscale ocean at "
            "~1000 dbar is a viscoelastic liquid -- terminal (alpha=1) "
            "window at 90-140 d with K~2.2e3 m^2/s -- whose measurable "
            "elasticity is carried by the eddy-trapped looper subpopulation "
            "(alpha 0.86-0.95, G'/G'' up to ~0.2 at 3-6 month periods), "
            "while non-loopers are a simple liquid.  The record-mean "
            "removal bias that fakes ensemble subdiffusion is demonstrated "
            "and controlled with the independent atlas mean."),
    )


def run(write=True):
    cache = load_cache()
    out = analyze(cache)
    res = dict(description=__doc__.splitlines()[0], meta=cache["meta"],
               **{k: v for k, v in out.items()
                  if k not in ("ens", "loop", "rest", "ens_floatmean")},
               curves={k: out[k] for k in ("ens", "loop", "rest",
                                           "ens_floatmean")})
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    print("NR45 (E11) -- ocean microrheology: GSER on ANDRO deep displacements")
    print(f"  floats (NA box)           : {res['n_floats']} "
          f"(loopers {res['n_loopers']} / rest {res['n_rest']}, "
          f"|spin| thr {res['looper_threshold']:.3f})")
    print(f"  terminal viscous window   : {v['terminal_viscous_window']} "
          f"(alpha[90-140d] = {res['curves']['ens']['alpha_90_140']:.3f}; "
          f"K = {res['K_m2_s_100d']:.0f} m2/s)")
    print(f"  VACF zero crossing        : ~{res['vacf_zero_cross_days']:.0f} d")
    print(f"  float-mean bias           : {v['floatmean_bias_quantified']} "
          f"(alpha[150-190d] {res['curves']['ens_floatmean']['alpha_150_190']:.2f} "
          f"float-mean vs {res['curves']['ens']['alpha_150_190']:.2f} atlas)")
    print(f"  subpopulation elasticity  : {v['elasticity_subpopulation_carried']} "
          f"(loopers alpha[100-190d] {res['curves']['loop']['alpha_100_190']:.2f} "
          f"vs rest {res['curves']['rest']['alpha_90_140']:.2f})")
    print(f"  looper elastic fraction   : G'/G'' in "
          f"[{v['looper_elastic_frac_range'][0]:.2f}, "
          f"{v['looper_elastic_frac_range'][1]:.2f}] at 100-190 d "
          f"-> detected={v['looper_elastic_detected']}")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
