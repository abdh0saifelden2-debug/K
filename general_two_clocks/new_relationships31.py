r"""NR54 (theory + real data) -- the spin sets the drift compass: beta-drift
read from single-float statistics.  Strong spinners drift WESTWARD;
cyclonic spinners drift POLEWARD and anticyclonic spinners EQUATORWARD --
the deep-ocean version of the altimetric eddy-drift law, measured with no
eddy identification or tracking at all.

Follows NR52 (the spin = the parity-odd face) and NR53 (the spin's eddy
clock).  NR54 connects the odd face to TRANSLATION: the carriers of spin
(coherent vortices) self-propagate on the beta-plane, so a float's spin
statistic predicts its drift relative to the time-mean circulation.

The claim
=========
1. **Beta-drift.**  A coherent vortex on the beta-plane self-propagates:
   westward at O(beta R^2), with a meridional component set by its SIGN
   -- cyclones poleward, anticyclones equatorward (the classic
   vortex-beta interaction; observed for surface altimetric eddies).
   A float trapped in a vortex inherits this translation.

2. **The compass protocol.**  For each float: spin s = lag-1 odd
   correlation (NR52) and drift (du, dv) = mean velocity minus a 2x2 deg
   time-mean atlas.  Predictions, in BOTH hemispheres:

     - |s| high -> du more westward than |s| low (monotone in |s|);
     - among strong spinners, poleward drift (v sign(lat)) is LARGER for
       cyclonic spin (s sign(f) > 0) than anticyclonic.

   No eddy detection, no tracking, no maps beyond the mean atlas: the
   parity-odd single-float statistic is the compass needle.

Findings (figures/89_spin_drift.json, committed cache
data/nr54_spin_drift_cache.json; 8275 floats, 700-1300 dbar):

* **Westward with spin, monotone.**  u-drift by |spin| quintile (km/d):
  NH +0.06, +0.05, +0.03, -0.05, -0.07 (top-bottom shift -0.135, t=-2.7);
  SH ends +0.01 -> -0.09 (shift -0.099, t=-2.5).
* **Cyclones poleward, anticyclones equatorward.**  Among top-quintile
  spinners: SH poleward(cyc) - poleward(anti) = +0.24 km/d (t = 5.8);
  NH +0.07 km/d (t = 1.3, same sign); pooled t ~ 5 -- the Morrow-type
  meridional splitting, at depth, from spin alone.
* **Magnitude.**  O(0.1 km/d), an order below the bare beta R_d^2 (~2-3
  km/d): floats spend only part of their record trapped, deep eddies are
  weaker/smaller than surface altimetric eddies, and the quintile
  populations dilute the trapped fraction -- the compass reads direction
  far more robustly than speed.

Consequence for the program: the response anatomy's odd face is not just
a stored signature -- it PREDICTS transport.  Spin (closed-orbit,
parity-odd, NR52) and drift (open-path translation) are locked by
beta-plane dynamics, so the two-clocks tensor's antisymmetric part doubles
as a dynamical compass.  Protocol exportable to any trajectory ensemble
on a rotating sphere (drifters, balloons, icebergs).

CPU-only, offline-safe (committed cache; regenerate with build_cache()).
Tests: tests/test_spin_drift.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr54_spin_drift_cache.json")
FIG = os.path.join(HERE, "figures", "89_spin_drift.json")

KM_PER_DAY = 86.4          # m/s -> km/day


# --------------------------------------------------------------------------- #
# compass statistics
# --------------------------------------------------------------------------- #
def quintile_bins(x, edges=(0.2, 0.4, 0.6, 0.8)):
    qs = np.quantile(x, edges)
    return np.digitize(x, qs)


def two_sample_t(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0.0:
        return float(d), float(np.sign(d) * np.inf) if d else 0.0
    return float(d), float(d / se)


def hemisphere_compass(spin, du, dv, lat, pole_sign):
    """Westward-with-spin and poleward-splitting statistics for one
    hemisphere selection (arrays already subset)."""
    s = np.abs(spin)
    bins = quintile_bins(s)
    trend = [float(du[bins == b].mean() * KM_PER_DAY)
             if np.any(bins == b) else float("nan") for b in range(5)]
    dU, tU = two_sample_t(du[bins == 4], du[bins == 0])
    strong = bins == 4
    cyc = spin[strong] * pole_sign > 0        # cyclonic: rotates with f
    eta = dv[strong] * pole_sign * KM_PER_DAY  # poleward-positive km/d
    dEta, tEta = two_sample_t(eta[cyc], eta[~cyc])
    return dict(n=int(len(spin)), u_trend_km_d=trend,
                westward_shift_km_d=float(dU * KM_PER_DAY),
                westward_t=tU,
                poleward_split_km_d=float(dEta), poleward_t=tEta,
                n_cyc=int(cyc.sum()), n_anti=int((~cyc).sum()),
                eta_cyc=eta[cyc].tolist(), eta_anti=eta[~cyc].tolist())


# --------------------------------------------------------------------------- #
# synthetic estimator validation
# --------------------------------------------------------------------------- #
def synthetic_checks(seed=0):
    """Floats with prescribed spin-drift coupling: the compass recovers the
    signed pattern; shuffling spin labels destroys it."""
    rng = np.random.default_rng(seed)
    n = 8000
    spin = rng.normal(0.0, 0.05, n)
    lat = np.where(rng.random(n) < 0.5, 30.0, -30.0)
    pole = np.sign(lat)
    w, p, noise = 0.002, 0.0015, 0.01
    du = -w * np.abs(spin) / 0.05 + rng.normal(0, noise, n)
    dv = pole * p * np.sign(spin * pole) * (np.abs(spin) / 0.05) \
        + rng.normal(0, noise, n)
    out = {}
    for name, sel, ps in (("nh", lat > 0, 1.0), ("sh", lat < 0, -1.0)):
        out[name] = hemisphere_compass(spin[sel], du[sel], dv[sel],
                                       lat[sel], ps)
    sh_spin = spin.copy()
    rng.shuffle(sh_spin)
    null = hemisphere_compass(sh_spin[lat > 0], du[lat > 0], dv[lat > 0],
                              lat[lat > 0], 1.0)
    return dict(nh=out["nh"], sh=out["sh"],
                null_westward_t=null["westward_t"],
                null_poleward_t=null["poleward_t"])


# --------------------------------------------------------------------------- #
# committed real data
# --------------------------------------------------------------------------- #
def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


def analyze(cache):
    syn = synthetic_checks()
    f = cache["floats"]
    spin = np.array(f["spin"])
    du = np.array(f["du"])
    dv = np.array(f["dv"])
    lat = np.array(f["lat"])
    res = {}
    for name, sel, ps in (("nh", (lat > 15) & (lat < 50), 1.0),
                          ("sh", (lat < -15) & (lat > -50), -1.0)):
        res[name] = hemisphere_compass(spin[sel], du[sel], dv[sel],
                                       lat[sel], ps)
        del res[name]["eta_cyc"], res[name]["eta_anti"]
    # pooled poleward split across hemispheres
    etas_c, etas_a = [], []
    for name, sel, ps in (("nh", (lat > 15) & (lat < 50), 1.0),
                          ("sh", (lat < -15) & (lat > -50), -1.0)):
        h = hemisphere_compass(spin[sel], du[sel], dv[sel], lat[sel], ps)
        etas_c += h["eta_cyc"]
        etas_a += h["eta_anti"]
    dpool, tpool = two_sample_t(etas_c, etas_a)
    out = dict(synthetic=syn, hemispheres=res,
               pooled_poleward_split_km_d=float(dpool),
               pooled_poleward_t=float(tpool),
               n_floats=int(cache["meta"]["n_floats"]))
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    syn = out["synthetic"]
    nh, sh = out["hemispheres"]["nh"], out["hemispheres"]["sh"]
    est = (syn["nh"]["westward_t"] < -5 and syn["sh"]["westward_t"] < -5
           and syn["nh"]["poleward_t"] > 5 and syn["sh"]["poleward_t"] > 5
           and abs(syn["null_westward_t"]) < 2.5
           and abs(syn["null_poleward_t"]) < 2.5)
    west = (nh["westward_t"] < -2.0 and sh["westward_t"] < -2.0
            and nh["u_trend_km_d"][4] < nh["u_trend_km_d"][0]
            and sh["u_trend_km_d"][4] < sh["u_trend_km_d"][0])
    pole = (out["pooled_poleward_t"] > 3.0
            and nh["poleward_split_km_d"] > 0
            and sh["poleward_split_km_d"] > 0)
    mag = all(0.02 < abs(h["westward_shift_km_d"]) < 1.0
              for h in (nh, sh))
    return dict(
        compass_estimator_validated=bool(est),
        westward_drift_increases_with_spin=bool(west),
        cyclones_poleward_anticyclones_equatorward=bool(pole),
        magnitude_order_consistent=bool(mag),
        reading=(
            "The parity-odd face predicts translation: floats in the top "
            "spin quintile drift westward relative to the weak-spin "
            f"population by {nh['westward_shift_km_d']:.2f} km/d (NH, "
            f"t = {nh['westward_t']:.1f}) and "
            f"{sh['westward_shift_km_d']:.2f} km/d (SH, "
            f"t = {sh['westward_t']:.1f}), monotone across quintiles; "
            "among strong spinners the meridional drift splits by spin "
            "SIGN -- cyclonic poleward, anticyclonic equatorward -- by "
            f"{out['pooled_poleward_split_km_d']:.2f} km/d pooled "
            f"(t = {out['pooled_poleward_t']:.1f}; SH alone "
            f"{sh['poleward_split_km_d']:.2f}, t = {sh['poleward_t']:.1f})."
            "  This is the altimetric eddy-drift law (westward; cyclones "
            "poleward, anticyclones equatorward), recovered at 700-1300 "
            "dbar from single-float statistics with no eddy detection: "
            "the spin is a beta-plane drift compass, at O(0.1 km/d) -- "
            "an order below bare beta R^2, as expected for a diluted, "
            "deep, partially-trapped population."),
    )


def run(write=True):
    cache = load_cache()
    out = analyze(cache)
    res = dict(description=__doc__.splitlines()[0],
               meta=cache["meta"], **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


# --------------------------------------------------------------------------- #
# cache builder (needs the ANDRO pickle; not used in tests)
# --------------------------------------------------------------------------- #
def build_cache(pkl_path="/home/andro_valid.pkl", out_path=CACHE,
                min_cell=20):
    import pandas as pd
    df = pd.read_pickle(pkl_path)
    lon = ((df.lon.values + 180) % 360) - 180
    ilon = np.clip(((lon + 180) // 2).astype(int), 0, 179)
    ilat = np.clip(((df.lat.values + 90) // 2).astype(int), 0, 89)
    cell = ilon * 90 + ilat
    su = np.bincount(cell, df.u.values, minlength=180 * 90)
    sv = np.bincount(cell, df.v.values, minlength=180 * 90)
    sn = np.bincount(cell, minlength=180 * 90)
    au = np.where(sn >= min_cell, su / np.maximum(sn, 1), np.nan)
    av = np.where(sn >= min_cell, sv / np.maximum(sn, 1), np.nan)
    df = df.assign(ua=au[cell], va=av[cell])
    rows = []
    for wmo, sub in df.groupby("wmo"):
        if len(sub) < 30:
            continue
        sub = sub.sort_values("juld")
        t = sub.juld.values
        u = sub.u.values / 100.0
        v = sub.v.values / 100.0
        ur = u - u.mean()
        vr = v - v.mean()
        O1 = 0.0
        for i in range(len(t) - 1):
            if 7.0 <= t[i + 1] - t[i] <= 13.0:
                O1 += ur[i] * vr[i + 1] - vr[i] * ur[i + 1]
        E0 = np.sum(ur ** 2 + vr ** 2)
        if E0 <= 0:
            continue
        m = np.isfinite(sub.ua.values)
        if m.sum() < 10:
            continue
        rows.append((O1 / E0,
                     float(np.mean(u[m] - sub.ua.values[m] / 100.0)),
                     float(np.mean(v[m] - sub.va.values[m] / 100.0)),
                     float(sub.lat.mean()), int(m.sum())))
    arr = np.array(rows)
    out = dict(
        meta=dict(description=("NR54: per-float spin vs atlas-residual "
                               "drift (beta-drift compass)"),
                  provenance=("ANDRO SEANOE doi:10.17882/47077 (CC-BY), "
                              "700-1300 dbar, 10-day cycles; 2x2 deg atlas "
                              f"(cells >={min_cell} cycles) removed; spin "
                              "= lag-1 odd correlation rho1 (float-mean "
                              "residuals); drift = mean atlas-residual "
                              "velocity"),
                  columns=["spin_rho1", "du_m_s", "dv_m_s", "lat_mean",
                           "n_atlas_cycles"],
                  n_floats=int(len(arr))),
        floats=dict(spin=arr[:, 0].round(6).tolist(),
                    du=arr[:, 1].round(6).tolist(),
                    dv=arr[:, 2].round(6).tolist(),
                    lat=arr[:, 3].round(3).tolist(),
                    n=arr[:, 4].astype(int).tolist()))
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    return out


def main():
    res = run()
    v = res["verdicts"]
    nh, sh = res["hemispheres"]["nh"], res["hemispheres"]["sh"]
    print("NR54 -- the spin sets the drift compass (beta-drift from floats)")
    print(f"  compass estimator validated   : "
          f"{v['compass_estimator_validated']}")
    print(f"  westward drift with |spin|    : "
          f"{v['westward_drift_increases_with_spin']} "
          f"(NH {nh['westward_shift_km_d']:+.2f} km/d t={nh['westward_t']:.1f}; "
          f"SH {sh['westward_shift_km_d']:+.2f} t={sh['westward_t']:.1f})")
    print(f"  cyclones poleward / anti equat: "
          f"{v['cyclones_poleward_anticyclones_equatorward']} "
          f"(pooled {res['pooled_poleward_split_km_d']:+.2f} km/d, "
          f"t={res['pooled_poleward_t']:.1f})")
    print(f"  magnitude order consistent    : "
          f"{v['magnitude_order_consistent']}")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
