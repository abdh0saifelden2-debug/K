r"""New derived cross-relationship NR43, continuing the program (NR1-NR42) with the same
discipline: a *derived* two-clocks prediction, *confirmed on independent real data*
(CPU, deterministic; the raster fetch is offline-cached).  See
``REPORT_NEW_RELATIONSHIPS20.md`` for the write-up and
``tests/test_new_relationships20.py`` for the unit proof.

NR43 - THE ACTIVE-LAKE TWO CLOCKS IN ICESat-2: active subglacial lakes are the
       *episodic (fast) clock* riding on the *secular (slow) clock* of ice dynamics,
       and NASA ICESat-2 ATL15 surface-height-change (2019-2026) INDEPENDENTLY confirms
       it -- the non-secular dh variance is significantly enriched inside the 131
       Siegfried & Fricker (2018) active-lake outlines vs matched off-lake controls,
       the strongest-episodic lakes are the canonical active systems, the events are a
       fast-drain / slow-fill SAWTOOTH (the two-clock asymmetry), and hydraulically
       connected lakes are LAG-COUPLED (the NR39 lagged-coupling signature in real mass
       transfer).
       [two-clocks fast/slow separation x NR35 lake population x NR39 lagged coupling
        x ICESat-2 ATL15 (NSIDC, earthaccess) x Siegfried & Fricker 2018 outlines]

  The prediction (derived).  The program's two-clocks thesis separates every response
  into a slow (secular, quasi-steady) clock and a fast (episodic, memory-carrying) clock.
  For subglacial hydrology this predicts that an active lake's SURFACE expression is a
  fast, episodic dh residual localised to the lake, superposed on the slow secular
  ice-dynamic trend of the surrounding grounded ice.  Three consequences follow:
  (i)  LOCALISATION: the non-secular (detrended) dh variance is larger inside active-lake
       outlines than at matched off-lake controls in the same dynamic setting.
  (ii) ASYMMETRY: lake fill (upstream water supply, slow) and drainage (channelised
       evacuation, fast) run on different clocks, so the dh(t) event is an asymmetric
       sawtooth -- the drop steeper than the rise (the same fast-drain / slow-fill split
       as the G.4 drainage window and NR32/NR42).
  (iii)CONNECTIVITY: hydraulically connected lakes exchange water, so their dh(t) are
       lag-coupled (drain-one / fill-another) -- a real-data instance of NR39's lagged
       (imaginary-coherence) coupling that instantaneous common-mode forcing cannot make.

  The independent test.  ATL15 (ICESat-2 ATLAS, v005) is a DIFFERENT instrument and a
  DIFFERENT epoch (2019-2026) from the CryoSat-2 (2010-2020) series NR35 used, so it is a
  genuine out-of-sample confirmation.  Per lake we take the ATL15 10 km delta_h averaged
  over the pixels inside the SF2018 polygon, remove a low-order polynomial in time (the
  secular clock), and measure the residual (episodic) standard deviation.  Matched
  "pseudo-lake" controls -- random boxes with pixel counts drawn from the real-lake
  distribution, centred >=40 km from every lake -- give the null.  All numbers are cached
  offline in ``data/atl15_lake_dh_cache.json`` (168 KB) so the analysis and tests need no
  network; the fetch/extract path (earthaccess + the SF2018 outline .h5) is documented in
  the report.

  What is NEW here: (i) an independent-instrument, out-of-sample (ICESat-2-era)
  confirmation of the SF2018 active-lake population as an episodic fast-clock field
  (localisation significant at binomial p~1e-6); (ii) the fast-drain / slow-fill sawtooth
  asymmetry measured across the active subset (the two-clock signature in the height
  record itself, not just the velocity response NR35 bounded); (iii) the connected-lake
  lag coupling as a real-data NR39 instance, contrasted against unconnected control pairs.
  Honest scope: with 29 quarterly epochs the per-lake lag VALUES are not sharply resolved
  (best-of-lag selection), so connectivity is reported as a connected-vs-unconnected
  contrast, not a precise transport time.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "atl15_lake_dh_cache.json")
FIGDIR = os.path.join(HERE, "figures")

# Documented hydraulic connectivity (Fricker & Scambos 2009; Siegfried & Fricker 2018):
# the lower Whillans/Mercer ice-plain system and the MacAyeal Ice Stream system.
CONNECTED_PAIRS = [
    ("MercerSubglacialLake", "ConwaySubglacialLake"),
    ("MercerSubglacialLake", "WhillansSubglacialLake"),
    ("EngelhardtSubglacialLake", "WhillansSubglacialLake"),
    ("Mac1", "Mac2"),
    ("Mac1", "Mac3"),
    ("Mac2", "Mac3"),
]
# Cross-basin / cross-ice-stream pairs that are NOT hydraulically connected (control).
UNCONNECTED_PAIRS = [
    ("MercerSubglacialLake", "Byrd_1"),
    ("Mac1", "Thw_142"),
    ("EngelhardtSubglacialLake", "Byrd_2"),
    ("Mac2", "Thw_170"),
    ("ConwaySubglacialLake", "Byrd_1"),
    ("Mac3", "Rutford_1"),
]
KNOWN_ACTIVE = ["MercerSubglacialLake", "EngelhardtSubglacialLake", "ConwaySubglacialLake",
                "Thw_170", "Thw_142", "Mac1", "Byrd_1", "Byrd_2", "Whillans_6"]


def load_cache(path=CACHE):
    d = json.load(open(path))
    for r in d["per_lake"]:
        r["dh_arr"] = np.array([np.nan if v is None else v for v in r["dh"]], float)
        r["t_arr"] = np.array(r["t"], float)
    return d


def _detrended(r, deg=1):
    s, t = r["dh_arr"], r["t_arr"]
    m = np.isfinite(s)
    c = np.polyfit(t[m], s[m], deg)
    out = np.full_like(s, np.nan)
    out[m] = s[m] - np.polyval(c, t[m])
    return out


def discriminator(d):
    """In-lake vs matched-control non-secular dh variance: the localisation test."""
    lake = np.array([r["ns_std"] for r in d["per_lake"] if r["ns_std"] is not None])
    ctrl = np.array(d["controls"]["ns_std"], float)
    p95 = float(np.percentile(ctrl, 95))
    k = int((lake >= p95).sum()); n = len(lake)
    # permutation test on the median difference (deterministic seed)
    rng = np.random.default_rng(20260707)
    obs = float(np.median(lake) - np.median(ctrl))
    pool = np.concatenate([lake, ctrl]); ge = 0; NP = 5000
    for _ in range(NP):
        rng.shuffle(pool)
        if np.median(pool[:n]) - np.median(pool[n:]) >= obs:
            ge += 1
    out = {
        "n_lakes": n, "n_controls": len(ctrl),
        "lake_median": float(np.median(lake)), "lake_p90": float(np.percentile(lake, 90)),
        "ctrl_median": float(np.median(ctrl)), "ctrl_p90": float(np.percentile(ctrl, 90)),
        "ctrl_p95": p95, "detect_k": k, "detect_frac": k / n,
        "detect_enrichment_over_chance": (k / n) / 0.05,
        "perm_p_median": (ge + 1) / (NP + 1), "median_diff_m": obs,
    }
    try:
        from scipy.stats import binomtest, mannwhitneyu
        out["mannwhitney_p"] = float(mannwhitneyu(lake, ctrl, alternative="greater").pvalue)
        out["detect_binomial_p"] = float(
            binomtest(k, n, 0.05, alternative="greater").pvalue)
    except Exception:  # pragma: no cover
        out["mannwhitney_p"] = None
        out["detect_binomial_p"] = None
    return out


def size_dependence(d):
    p95 = float(np.percentile(np.array(d["controls"]["ns_std"], float), 95))
    res = [r for r in d["per_lake"] if r["ns_std"] and r["n_pix"] >= 2]
    sub = [r for r in d["per_lake"] if r["ns_std"] and r["n_pix"] < 2]
    def frac(g):
        return float(np.mean([r["ns_std"] >= p95 for r in g])) if g else None
    return {"resolved_n": len(res), "resolved_median": float(np.median([r["ns_std"] for r in res])),
            "resolved_detect_frac": frac(res),
            "subpixel_n": len(sub), "subpixel_median": float(np.median([r["ns_std"] for r in sub])),
            "subpixel_detect_frac": frac(sub)}


def fill_drain_asymmetry(d):
    """Across the active subset, the fastest fall (drainage) exceeds the fastest rise
    (fill): the two-clock sawtooth in the height record."""
    p95 = float(np.percentile(np.array(d["controls"]["ns_std"], float), 95))
    by = {r["name"]: r for r in d["per_lake"]}
    active = [r for r in d["per_lake"] if r["ns_std"] and r["ns_std"] >= p95]
    ratios = []; recs = []
    for r in active:
        dr = np.diff(_detrended(r))
        dr = dr[np.isfinite(dr)]
        if dr.size < 4:
            continue
        rise = float(np.max(dr)); fall = float(-np.min(dr))
        if rise > 1e-6:
            ratios.append(fall / rise)
            recs.append({"name": r["name"], "max_rise_m_q": rise, "max_fall_m_q": fall,
                         "fall_over_rise": fall / rise})
    ratios = np.array(ratios)
    n_fall_faster = int((ratios > 1).sum())
    # sign test that fall>rise is the majority (binomial vs 0.5)
    p_sign = None
    try:
        from scipy.stats import binomtest
        p_sign = float(binomtest(n_fall_faster, len(ratios), 0.5,
                                 alternative="greater").pvalue)
    except Exception:  # pragma: no cover
        pass
    recs.sort(key=lambda z: -z["fall_over_rise"])
    return {"n_active": len(ratios), "median_fall_over_rise": float(np.median(ratios)),
            "n_fall_faster": n_fall_faster, "sign_test_p": p_sign,
            "examples": recs[:6]}


def _best_lag_abscorr(a, b, by, maxlag=6):
    if a not in by or b not in by:
        return None
    ra = _detrended(by[a]); rb = _detrended(by[b])
    ra = ra - np.nanmean(ra); rb = rb - np.nanmean(rb)
    best = (0, 0.0)
    for L in range(-maxlag, maxlag + 1):
        if L >= 0:
            x, y = ra[L:], rb[:len(rb) - L] if L else rb
        else:
            x, y = ra[:len(ra) + L], rb[-L:]
        mm = np.isfinite(x) & np.isfinite(y)
        if mm.sum() < 8:
            continue
        c = float(np.corrcoef(x[mm], y[mm])[0, 1])
        if abs(c) > abs(best[1]):
            best = (L, c)
    return best


def connectivity_lag(d):
    """Connected lakes are more strongly lag-coupled than unconnected control pairs."""
    by = {r["name"]: r for r in d["per_lake"]}
    def collect(pairs):
        out = []
        for a, b in pairs:
            r = _best_lag_abscorr(a, b, by)
            if r is not None:
                out.append({"a": a, "b": b, "lag_q": r[0], "r": r[1]})
        return out
    conn = collect(CONNECTED_PAIRS); unc = collect(UNCONNECTED_PAIRS)
    conn_abs = np.array([abs(x["r"]) for x in conn]); unc_abs = np.array([abs(x["r"]) for x in unc])
    return {"connected": conn, "unconnected": unc,
            "connected_median_absr": float(np.median(conn_abs)) if conn_abs.size else None,
            "unconnected_median_absr": float(np.median(unc_abs)) if unc_abs.size else None,
            "contrast": float(np.median(conn_abs) - np.median(unc_abs))
            if conn_abs.size and unc_abs.size else None,
            "note": "29 quarterly epochs: lag VALUES are illustrative (best-of-lag "
                    "selection applied identically to both groups); the connected-vs-"
                    "unconnected |r| contrast is the falsifiable statement"}


def top_lakes(d, k=12):
    order = sorted([r for r in d["per_lake"] if r["ns_std"]], key=lambda r: -r["ns_std"])
    return [{"name": r["name"], "region": r["region"], "ns_std": round(r["ns_std"], 3),
             "n_pix": r["n_pix"]} for r in order[:k]]


def nr43(path=CACHE):
    d = load_cache(path)
    tops = top_lakes(d)
    topnames = {t["name"] for t in tops}
    return {
        "meta": d["_meta"],
        "discriminator": discriminator(d),
        "size_dependence": size_dependence(d),
        "fill_drain_asymmetry": fill_drain_asymmetry(d),
        "connectivity_lag": connectivity_lag(d),
        "top_lakes": tops,
        "top_includes_known_active": sorted(topnames & set(KNOWN_ACTIVE)),
    }


def make_figure(out, path_png, path_cache=CACHE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = load_cache(path_cache)
    lake = np.array([r["ns_std"] for r in d["per_lake"] if r["ns_std"] is not None])
    ctrl = np.array(d["controls"]["ns_std"], float)
    by = {r["name"]: r for r in d["per_lake"]}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 0.8, 33)
    ax[0].hist(ctrl, bins=bins, density=True, alpha=0.5, color="C7", label="off-lake control")
    ax[0].hist(lake, bins=bins, density=True, alpha=0.6, color="C0", label="in-lake (SF2018)")
    ax[0].axvline(np.percentile(ctrl, 95), color="C3", ls="--", lw=1, label="control p95")
    ax[0].set_xlabel("non-secular dh std [m]"); ax[0].set_ylabel("density")
    ax[0].set_title("ATL15 episodic dh: lakes vs control"); ax[0].legend(fontsize=8)
    for nm, c in [("MercerSubglacialLake", "C0"), ("Mac1", "C1"),
                  ("EngelhardtSubglacialLake", "C2")]:
        if nm in by:
            r = by[nm]; ax[1].plot(r["t_arr"], _detrended(r), c, lw=1.6,
                                   marker=".", label=nm.replace("SubglacialLake", ""))
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("year"); ax[1].set_ylabel("detrended dh [m]")
    ax[1].set_title("fast-drain / slow-fill sawtooth"); ax[1].legend(fontsize=8)
    fig.suptitle("NR43 - active-lake two clocks in ICESat-2 ATL15 (2019-2026)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    fig.savefig(path_png, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=os.path.join(FIGDIR, "nr43_active_lake_two_clocks.json"))
    ap.add_argument("--png-out", default=os.path.join(FIGDIR, "nr43_active_lake_two_clocks.png"))
    a = ap.parse_args()
    out = nr43()
    os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
    json.dump(out, open(a.json_out, "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in
                      ("discriminator", "size_dependence", "fill_drain_asymmetry",
                       "connectivity_lag", "top_includes_known_active")},
                     indent=1, default=str))
    make_figure(out, a.png_out)
    print("wrote", a.json_out, "and", a.png_out)


if __name__ == "__main__":
    main()
