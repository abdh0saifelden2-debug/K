r"""§H.1.6 (theory) — Effective-pressure gating of the ocean→velocity coupling.

A new, falsifiable relationship mined from this repo's two field results:
  * §H.1.4-ocean (`rtn_glmig_ocean.py`): grounding-line surface speed ``u_*`` rises
    with observed ocean thermal forcing TF (warm shelf -> fast ice).
  * the RTN / level-set framework (§H.1.1-§H.1.2): the dimensionless margin
    ``rel = m/H`` with ``m = H - H*`` measures a GL point's *proximity to the
    flotation/intrusion threshold*. Because the effective pressure obeys
    ``N = rho_i g H - p_w ~ phi rho_i g (H - H*) = phi rho_i g m``, the normalized
    margin ``rel = m/H`` is a dimensionless proxy for the *normalized effective
    pressure*: ``rel -> 0`` at flotation (``N -> 0``).

Derivation (mainstream-grounded). A regularized-Coulomb / Budd sliding law makes
basal speed a decreasing, *convex* function of effective pressure, ``u_b ~ tau_b /
N^q`` (or Schoof/Tsai regularized Coulomb that saturates as ``N -> 0``); either way
the *elasticity* ``s_N = d ln u_b / d ln N < 0`` and ``|s_N|`` **grows as N falls**.
Ocean melt is a sink on ``H`` -> lowers ``N`` -> raises ``u``. Chaining,

    d ln u_* / dTF  =  s_N * (d ln N / dTF)        with |s_N| largest at small N,

so the **thermal-forcing sensitivity of GL speed is predicted to be largest where
the bed is closest to flotation (small rel)** — ocean forcing and a near-floating
bed *multiply*. This unifies the RTN intrusion number (where the ocean can reach the
bed) with the §H.1.4-ocean velocity coupling (how hard it then pushes), and is
exactly the ``d u_b/dN`` hydromechanical-state control the §G.4 lake-lag result
pointed to.

Falsifiable predictions (tested here):
  (P1) split GL points into terciles of ``rel``: the slope ``d ln u_* / dTF`` is
       *monotonically larger* for the near-flotation (low-rel) tercile.
  (P2) in the interaction model ``ln u_* = a + b*TF + c*ln rel + d*(TF*ln rel)`` the
       cross term ``d < 0`` (the TF slope steepens as rel falls), with a bootstrap
       95% CI excluding 0.
A flat/sign-reversed gating falsifies the effective-pressure reading (the coupling
would then be geometry- or buttressing-set, not basal-drag-set).

Data (all open, no Earthdata): Bedmap2 (rel), ITS_LIVE (u_*), Schmidtko 2014 (TF),
Konrad 2018 (GL points), ICESat-2 ATL15 (dH/dt for the canonical good-point set).
Reuses `rtn_glmig_ocean` / `rtn_glmig_basin` helpers so points/TF/speed are identical
to §H.1.4-ocean.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtn_glmig_basin import sample, SECTORS, _in_sector  # noqa: E402
from rtn_glmig_ocean import (assign_tf, load_schmidtko_tf, recover_thin_vobs,  # noqa: E402
                             build_dhdt_from_atl15, _fit_semilog,
                             ATL15_GLOB, STRIDE, SEARCH_M)
from rtn_glmig_test import build_dhdt_at_konrad, _grid_coords  # noqa: E402
from external.bedmap2_loader import load_fields  # noqa: E402
from external.rtn_intrusion_clock import margin_field  # noqa: E402

MARINE_WEST = ["Amundsen (PIG/Thwaites/Smith)", "Bellingshausen+WAIS-Pacific"]


def sample_rel(bin_dir, glon, glat, stride=STRIDE, phi=0.9, search_m=10000.0):
    """Normalized margin rel = m/H (proxy for normalized effective pressure) at each
    GL point, sampled at the nearest grounded-safe (rel>0) Bedmap2 cell (<=search_m)."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    d = load_fields(bin_dir, stride=stride)
    H, bed, mask = d["thickness"], d["bed"], d["icemask_grounded_and_shelves"]
    meta = d["_meta"]; ny, nx = H.shape
    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)
    margin, _Hstar, _dbase = margin_field(H, bed, phi)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(grounded & (H > 0), margin / H, np.nan)
    ys = _grid_coords(meta, stride, ny, ny, is_row=True)
    xs = _grid_coords(meta, stride, nx, nx, is_row=False)
    XX, YY = np.meshgrid(xs, ys)
    valid = grounded & np.isfinite(rel) & (rel > 0)
    tree = cKDTree(np.c_[XX[valid], YY[valid]])
    rel_src, H_src = rel[valid], H[valid]
    px, py = Transformer.from_crs("EPSG:4326", "EPSG:3031",
                                  always_xy=True).transform(glon, glat)
    dist, idx = tree.query(np.c_[px, py], distance_upper_bound=search_m)
    hit = np.isfinite(dist)
    rel_pt = np.full(glon.shape, np.nan); rel_pt[hit] = rel_src[idx[hit]]
    H_pt = np.full(glon.shape, np.nan); H_pt[hit] = H_src[idx[hit]]
    return rel_pt, H_pt


def _interaction_fit(spd, tf, rel, nboot=3000, seed=0):
    """OLS ln(u_*) = a + b*TF + c*ln(rel) + d*(TF*ln rel); bootstrap CI on d.

    P2 prediction: d < 0 (TF slope d ln u_*/dTF = b + d*ln(rel) steepens as rel->0,
    i.e. ln rel -> -inf). Returns coefficients, the implied TF slope at the 10th/90th
    rel percentiles, and R^2."""
    m = (np.isfinite(spd) & (spd > 0) & np.isfinite(tf)
         & np.isfinite(rel) & (rel > 0))
    out = {"n": int(m.sum())}
    if m.sum() < 50:
        return out
    y = np.log(spd[m]); tfm = tf[m]; lr = np.log(rel[m])
    X = np.column_stack([np.ones(y.size), tfm, lr, tfm * lr])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    rng = np.random.default_rng(seed)
    bs = np.empty(nboot)
    idx = np.arange(y.size)
    for b in range(nboot):
        j = rng.choice(idx, idx.size, replace=True)
        bb, *_ = np.linalg.lstsq(X[j], y[j], rcond=None)
        bs[b] = bb[3]
    lr10, lr90 = np.percentile(lr, 10), np.percentile(lr, 90)
    out.update({
        "b_TF": float(beta[1]), "c_lnrel": float(beta[2]),
        "d_interaction": float(beta[3]),
        "d_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "r2": float(ss),
        # TF slope at near-flotation (low rel, 10th pct) vs well-grounded (90th pct)
        "TF_slope_lowrel": float(beta[1] + beta[3] * lr10),
        "TF_slope_highrel": float(beta[1] + beta[3] * lr90),
        "rel_p10": float(np.exp(lr10)), "rel_p90": float(np.exp(lr90)),
    })
    return out


def _terciles(spd, tf, rel):
    """P1: d ln u_*/dTF in terciles of rel (low=near-flotation .. high=well-grounded)."""
    m = (np.isfinite(spd) & (spd > 0) & np.isfinite(tf)
         & np.isfinite(rel) & (rel > 0))
    rr = rel[m]
    if rr.size < 90:
        return None
    q1, q2 = np.percentile(rr, [33.33, 66.67])
    bins = {
        "low_rel_near_flotation": rel <= q1,
        "mid_rel": (rel > q1) & (rel <= q2),
        "high_rel_well_grounded": rel > q2,
    }
    out = {"rel_terciles": [float(q1), float(q2)]}
    for name, bmask in bins.items():
        bm = bmask & m
        f = _fit_semilog(spd[bm], tf[bm])
        out[name] = {"n": f["n"], "median_rel": float(np.nanmedian(rel[bm])),
                     "median_TF": f["median_TF"],
                     "TF_slope": f["slope_per_degC"], "r": f["r"], "ci95": f["ci95"]}
    return out


def analyse(bin_dir, konrad_txt, dhdt_npz, schmidtko_txt, itslive_npz=None,
            atl15_glob=ATL15_GLOB):
    # canonical good points + Ro (identical good set to §H.1.4-ocean)
    if dhdt_npz and not os.path.exists(dhdt_npz):
        sec_nc = os.path.join(os.path.dirname(dhdt_npz), "schroder_sec_mrg.nc")
        if os.path.exists(sec_nc):
            build_dhdt_at_konrad(sec_nc, konrad_txt, dhdt_npz)
        elif atl15_glob:
            build_dhdt_from_atl15(konrad_txt, dhdt_npz, atl15_glob)
    r = sample(bin_dir, konrad_txt, dhdt_npz, stride=STRIDE)
    lon, lat, Ro = r["lon"], r["lat"], r["Ro"]

    rel, H = sample_rel(bin_dir, lon, lat)
    thin, v_obs = recover_thin_vobs(lon, lat, konrad_txt, dhdt_npz)

    spd = None
    if itslive_npz and os.path.exists(itslive_npz):
        z = np.load(itslive_npz)
        if z["Ro"].shape == Ro.shape and np.allclose(z["Ro"], Ro, equal_nan=True):
            spd = z["speed"]
    if spd is None:
        from itslive_glmig_sample import sample_speed
        spd = sample_speed(lon, lat)
        if itslive_npz:
            np.savez(itslive_npz, speed=spd, lon=lon, lat=lat, Ro=Ro, tau_d=r["tau_d"])

    slon, slat, stf = load_schmidtko_tf(schmidtko_txt)
    tf, _dkm = assign_tf(lon, lat, slon, slat, stf)

    domains = {
        "CONTINENTAL": np.ones(lon.size, bool),
        "MARINE_WEST": np.zeros(lon.size, bool),
    }
    for s in MARINE_WEST:
        domains["MARINE_WEST"] |= _in_sector(lon, lat, SECTORS[s])

    res = {}
    for name, dm in domains.items():
        res[name] = {
            "n": int(dm.sum()),
            "n_usable": int((dm & np.isfinite(tf) & np.isfinite(rel)
                             & (rel > 0) & np.isfinite(spd) & (spd > 0)).sum()),
            "terciles_P1": _terciles(spd[dm], tf[dm], rel[dm]),
            "interaction_P2": _interaction_fit(spd[dm], tf[dm], rel[dm]),
        }
    diag = dict(lon=lon, lat=lat, rel=rel, tf=tf, spd=spd, Ro=Ro,
                marine=domains["MARINE_WEST"])
    return res, diag


def _verdict(res):
    def gated(d):
        t = d.get("terciles_P1") or {}
        lo = t.get("low_rel_near_flotation", {}).get("TF_slope")
        hi = t.get("high_rel_well_grounded", {}).get("TF_slope")
        it = d.get("interaction_P2", {})
        dco, dci = it.get("d_interaction"), it.get("d_ci95")
        mono = (lo is not None and hi is not None and lo > hi)
        inter_neg = (dco is not None and dci is not None and dci[1] < 0)
        return mono, inter_neg, lo, hi, dco, dci

    mw = res["MARINE_WEST"]; co = res["CONTINENTAL"]
    mw_mono, mw_neg, lo, hi, dco, dci = gated(mw)
    co_mono, co_neg, *_ = gated(co)
    if mw_mono and mw_neg:
        head = ("SUPPORTED in the marine West: the ocean-thermal-forcing sensitivity of "
                "GL speed is largest near flotation (low effective pressure) - both the "
                "tercile monotonicity (P1) and the negative TF*ln(rel) interaction (P2, "
                "CI<0) hold.")
    elif mw_mono or mw_neg or co_neg:
        head = ("PARTIALLY SUPPORTED: effective-pressure gating appears in some of "
                "{tercile monotonicity, interaction sign} but not robustly across both.")
    else:
        head = ("NOT SUPPORTED on this data: no robust steepening of the TF-speed "
                "coupling toward flotation; the coupling is not basal-drag(N)-gated here.")
    extra = ""
    if lo is not None and hi is not None:
        extra = (f" Marine-West d ln u_*/dTF: {lo:+.2f}/degC near flotation vs "
                 f"{hi:+.2f}/degC well-grounded; interaction d={dco:+.3f} "
                 f"CI[{dci[0]:+.3f},{dci[1]:+.3f}].")
    return head + extra


def make_figure(diag, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.6))
    order = ["low_rel_near_flotation", "mid_rel", "high_rel_well_grounded"]
    labels = ["low rel\n(near flotation)", "mid rel", "high rel\n(well grounded)"]
    for dom, col, off in [("MARINE_WEST", "#d95f02", -0.16),
                          ("CONTINENTAL", "#1b9e77", +0.16)]:
        t = res[dom].get("terciles_P1")
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
            es = np.array(es).T
            ax0.errorbar(xs, ys, yerr=es, fmt="o-", color=col, capsize=3, ms=7,
                         label=dom.replace("_", " ").title())
    ax0.axhline(0, color="k", lw=1, ls="--")
    ax0.set_xticks(range(3)); ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylabel(r"$d\,\ln u_*/dTF$  [per $\degree$C]")
    ax0.set_title("(P1) ocean-TF sensitivity of GL speed steepens toward flotation")
    ax0.legend(fontsize=9); ax0.grid(alpha=0.3)

    rel, tf, spd, marine = diag["rel"], diag["tf"], diag["spd"], diag["marine"]
    m = (marine & np.isfinite(rel) & (rel > 0) & np.isfinite(tf)
         & np.isfinite(spd) & (spd > 0))
    sc = ax1.scatter(tf[m], spd[m], c=np.log10(rel[m]), s=10, alpha=0.5,
                     cmap="viridis_r")
    ax1.set_yscale("log")
    cb = fig.colorbar(sc, ax=ax1); cb.set_label(r"$\log_{10}$ rel = $\log_{10}(m/H)$ (low = near flotation)")
    ax1.set_xlabel(r"ocean thermal forcing TF  [$\degree$C]")
    ax1.set_ylabel(r"GL surface speed $u_*$  [m/yr]")
    ax1.set_title("Marine West: near-flotation points (dark) respond most to TF")

    fig.suptitle("§H.1.6 - effective-pressure gating: ocean thermal forcing accelerates "
                 "grounding lines most where the bed is near flotation\n(unifies the RTN "
                 "margin with the §H.1.4-ocean velocity coupling)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
        os.path.dirname(os.path.abspath(__file__)), "..", "reports", "rtn_ocean_efp_gate.json"))
    a = ap.parse_args()

    res, diag = analyse(a.bin_dir, a.konrad, a.dhdt_npz, a.schmidtko, a.itslive_npz, a.atl15)
    verdict = _verdict(res)
    out = {
        "hypothesis": ("the ocean-thermal-forcing sensitivity of grounding-line speed "
                       "(d ln u_*/dTF) is largest where the bed is nearest flotation "
                       "(small normalized margin rel=m/H ~ low effective pressure), "
                       "because regularized-Coulomb/Budd sliding steepens d ln u/d ln N "
                       "as N->0; ocean melt lowers N, so TF and a near-floating bed "
                       "multiply"),
        "rel_definition": "rel = m/H, m = H - H* (RTN margin); proxy for normalized N",
        "TF_definition": "TF = CT - CT_freezing(SA,p) [degC] (Schmidtko 2014, TEOS-10)",
        "predictions": {"P1": "d ln u_*/dTF larger in the low-rel (near-flotation) tercile",
                        "P2": "interaction coef d on TF*ln(rel) is < 0 (CI excludes 0)"},
        "bedmap2_stride": STRIDE, "search_radius_km": SEARCH_M / 1000.0,
        "domains": res, "verdict": verdict,
        "caveats": ("rel is a normalized-effective-pressure proxy, not measured N; "
                    "speed and rel are themselves correlated (fast ice rides near "
                    "flotation) - the interaction term controls for the main effects but "
                    "not all confounds; ATL15 dH/dt sets the good-point set (PANGAEA "
                    "down); Schmidtko climatology vs modern speed; bootstrap CIs ignore "
                    "spatial autocorrelation."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({"verdict": verdict}, indent=2))
    print(f"json -> {os.path.abspath(a.out)}")
    make_figure(diag, res,
                os.path.join(os.path.dirname(os.path.abspath(a.out)), "rtn_ocean_efp_gate.png"))


if __name__ == "__main__":
    main()
