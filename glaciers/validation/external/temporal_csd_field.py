r"""§I.3 FIELD TEST — temporal critical-slowing-down early-warning on REAL Antarctic
trunks: rolling variance + lag-1 autocorrelation of quarterly ITS_LIVE speed, 2014-2025.

What §I.3 registered (FUTURE_WORK.md; sn_master_curve.py OU demonstration)
--------------------------------------------------------------------------
Near the regularized-Coulomb flotation fold the restoring rate vanishes
(``lambda ∝ (1-R)²/R → 0``), so a velocity perturbation under slowly declining
effective pressure N shows **rising variance and rising lag-1 autocorrelation**
(critical slowing down; Scheffer 2009, Dakos 2008). Registered forecast (verbatim):
"an ice stream approaching ungrounding should show rising variance/AC1 in its surface
speed before it goes afloat", with the falsifier "a stream observed to approach
flotation with adequate sampling and NO variance/AC1 rise". Distinct from Boers &
Rypdal 2021 (Greenland melt CSD): different observable (velocity), mechanism (basal
drag stiffness), and threshold (flotation fold).

Who is "approaching flotation" 2014-2025 (setting the expectation, not the result):
the ASE trunks (Thwaites, Pine Island, Smith) thin at m/yr rates near their GLs —
N declines through the record. Rutford is the steady grounded control. The §I.6 unit
measured the SPATIAL face on the same corridors; this unit is the TEMPORAL face.

Method (real data, offline-replayable committed cache)
------------------------------------------------------
* Points: the SAME committed §I.6 flowline points, near-GL segments (d_gl <= 40 km)
  plus a matched upstream control band (80-140 km) on each stream.
* Series per point: ITS_LIVE v2 datacube image pairs (baseline <= 60 d) -> CALENDAR-
  QUARTER medians + MAD/sqrt(n) SEs (>= 5 pairs/quarter) -> deseasonalize (remove the
  mean quarter-of-year cycle) -> linear detrend -> residual series r(t).
* EWS estimators: centered rolling windows of 12 quarters: window variance MINUS the
  window noise floor (mean SE²) and window AC1 (on residuals). Trend statistic:
  Kendall tau of each rolling series vs time (Dakos 2008 standard).
* Honest nulls, per point: (i) 200 FOURIER PHASE surrogates of r(t) (same power
  spectrum / autocorrelation, stationary by construction) -> per-point one-sided p for
  tau_var and tau_ac1; (ii) the NOISE trend control: Kendall tau of the per-quarter
  SE² and of the per-quarter pair count vs time — ITS_LIVE sensor mix changes through
  2014-2025 (L8 -> +S2 2017 -> +L9 2022, S1 epochs), and a sampling-driven variance
  trend must not be read as CSD.
* Population read-out: near-GL vs upstream vs control-stream fractions of points with
  (a) tau_var > 0 at p < 0.1, (b) joint tau_var > 0 AND tau_ac1 > 0 (the §I.3 joint
  signature), compared by two-proportion tests; per-stream median taus with a
  points-are-correlated caveat (reported, not hidden).

Run modes
---------
  python temporal_csd_field.py --fetch      # network: quarterly medians -> cache
  python temporal_csd_field.py              # offline: analyze committed cache

No GPU. The committed cache holds per-point quarterly medians/SEs/counts only.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_EWSCACHE = os.path.join(_DATA, "spatial_ews_field_cache.json")
_CACHE = os.path.join(_DATA, "temporal_csd_field_cache.json")
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))

DT_MAX_DAYS = 60.0
MIN_PAIRS_PER_Q = 5
Q0, Q1 = (2014, 1), (2025, 4)
WIN = 12                       # rolling window, quarters (3 yr)
MIN_QUARTERS = 32              # require a reasonably filled series
NEAR_KM = 40.0
UP_KM = (80.0, 140.0)
N_SURR = 200

TRUNKS = ("Thwaites", "Pine_Island", "Smith")
CONTROL = "Rutford"


# ----------------------------------------------------------------- fetch
def _quarters():
    out = []
    for y in range(Q0[0], Q1[0] + 1):
        for q in range(1, 5):
            if (y, q) < Q0 or (y, q) > Q1:
                continue
            out.append((y, q))
    return out


def fetch():
    import itslive
    from pyproj import Transformer
    ews = json.load(open(_EWSCACHE))
    tr = Transformer.from_crs(3031, 4326, always_xy=True)
    qs = _quarters()
    qindex = {q: i for i, q in enumerate(qs)}
    cache = {"_description": "committed cache for the §I.3 temporal CSD field test: "
                             "per-point quarterly median speeds, SEs, pair counts",
             "_source": "ITS_LIVE v2 datacubes at the committed §I.6 flowline points; "
                        "pairs with baseline<=%.0f d; fetched %s" % (DT_MAX_DAYS, _today()),
             "_quarters": [f"{y}Q{q}" for (y, q) in qs],
             "streams": {}}
    for name, S in ews["streams"].items():
        pts = [p for p in S["points"]
               if p["d_gl_km"] <= NEAR_KM or UP_KM[0] <= p["d_gl_km"] <= UP_KM[1]]
        lonlat = [tr.transform(p["x"], p["y"]) for p in pts]
        rows = []
        B = 40
        for b0 in range(0, len(lonlat), B):
            ts = itslive.velocity_cubes.get_time_series(points=lonlat[b0:b0 + B],
                                                        variables=["v", "date_dt"])
            for k, t in enumerate(ts):
                d = t["time_series"]
                v = np.asarray(d["v"].values, float)
                dt = np.asarray(d["date_dt"].values.astype("timedelta64[D]").astype(float))
                md = d["mid_date"].values
                yr = md.astype("datetime64[Y]").astype(int) + 1970
                mo = md.astype("datetime64[M]").astype(int) % 12 + 1
                qq = (mo - 1) // 3 + 1
                ok = np.isfinite(v) & (dt <= DT_MAX_DAYS)
                med = [None] * len(qs); se = [None] * len(qs); cnt = [0] * len(qs)
                for (Y, Q) in qs:
                    m_ = ok & (yr == Y) & (qq == Q)
                    n = int(m_.sum())
                    if n >= MIN_PAIRS_PER_Q:
                        vv = v[m_]
                        mm = float(np.median(vv))
                        mad = float(np.median(np.abs(vv - mm)))
                        i = qindex[(Y, Q)]
                        med[i] = mm; se[i] = 1.4826 * mad / math.sqrt(n); cnt[i] = n
                p = dict(pts[b0 + k])
                p.update(q_median=med, q_se=se, q_n=cnt)
                rows.append(p)
            print(f"  [{name}] {min(b0 + B, len(lonlat))}/{len(lonlat)}")
        cache["streams"][name] = dict(expectation=S["expectation"], points=rows)
        _dump(cache, _CACHE)
    print(f"cache -> {_CACHE}")
    return cache


def _today():
    import datetime
    return datetime.date.today().isoformat()


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


# ----------------------------------------------------------------- estimators
def prep_series(p):
    """Quarterly series -> deseasonalized, detrended residuals (+ aligned SE²)."""
    med = np.asarray([np.nan if x is None else x for x in p["q_median"]], float)
    se = np.asarray([np.nan if x is None else x for x in p["q_se"]], float)
    n_ok = int(np.isfinite(med).sum())
    if n_ok < MIN_QUARTERS:
        return None
    t = np.arange(med.size, dtype=float)
    qoy = (np.arange(med.size) % 4)
    ok = np.isfinite(med)
    # JOINT season + trend removal (sequential subtraction leaves a sawtooth when a
    # trend is present: per-quarter means absorb phase-dependent trend offsets)
    cols = [np.ones(med.size), t - t.mean()]
    for q in (1, 2, 3):
        cols.append((qoy == q).astype(float))
    A = np.vstack(cols).T
    if ok.sum() < MIN_QUARTERS:
        return None
    coef, *_ = np.linalg.lstsq(A[ok], med[ok], rcond=None)
    r = np.full(med.size, np.nan)
    r[ok] = med[ok] - A[ok] @ coef
    vbar = float(np.nanmean(med))
    return dict(resid=r, se=se, t=t, vbar=vbar, n_ok=int(ok.sum()),
                frac_filled=float(ok.mean()))


def rolling_ews(r, se, win=WIN, min_fill=8):
    """Centered rolling noise-corrected variance and AC1 (NaN-tolerant)."""
    n = r.size
    var_w = np.full(n, np.nan)
    ac1_w = np.full(n, np.nan)
    for c in range(n):
        i0, i1 = max(0, c - win // 2), min(n, c + win - win // 2)
        seg = r[i0:i1]; sseg = se[i0:i1]
        ok = np.isfinite(seg)
        if ok.sum() < min_fill:
            continue
        v_raw = float(np.var(seg[ok], ddof=1))
        noise = float(np.nanmean(sseg[ok] ** 2)) if np.isfinite(sseg[ok]).any() else 0.0
        var_w[c] = max(v_raw - noise, 0.0)
        s = seg.copy()
        a, b = s[:-1], s[1:]
        m2 = np.isfinite(a) & np.isfinite(b)
        if m2.sum() >= min_fill - 1:
            aa, bb = a[m2] - a[m2].mean(), b[m2] - b[m2].mean()
            den = math.sqrt(float(np.sum(aa ** 2) * np.sum(bb ** 2)))
            ac1_w[c] = float(np.sum(aa * bb) / den) if den > 0 else np.nan
    return var_w, ac1_w


def _kendall_t(y):
    from scipy.stats import kendalltau
    ok = np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    t = np.arange(y.size, dtype=float)
    return float(kendalltau(t[ok], y[ok])[0])


def phase_surrogates(r, n_surr=N_SURR, seed=0):
    """Fourier-phase surrogates of a gappy series (gaps re-imposed after transform)."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(r)
    x = r.copy()
    x[~ok] = 0.0                      # zero-fill for FFT; gaps re-masked below
    X = np.fft.rfft(x)
    out = []
    for _ in range(n_surr):
        ph = np.exp(2j * np.pi * rng.random(X.size))
        ph[0] = 1.0
        if x.size % 2 == 0:
            ph[-1] = 1.0
        s = np.fft.irfft(X * ph, n=x.size)
        s = np.where(ok, s, np.nan)
        out.append(s)
    return out


def point_ews(p, seed=0):
    pr = prep_series(p)
    if pr is None:
        return None
    r, se = pr["resid"], pr["se"]
    var_w, ac1_w = rolling_ews(r, se)
    tau_v, tau_a = _kendall_t(var_w), _kendall_t(ac1_w)
    if not (np.isfinite(tau_v) and np.isfinite(tau_a)):
        return None
    # surrogate p-values (one-sided: rising)
    tv_s, ta_s = [], []
    for s in phase_surrogates(r, seed=seed):
        vw, aw = rolling_ews(s, se)
        tv, ta = _kendall_t(vw), _kendall_t(aw)
        if np.isfinite(tv):
            tv_s.append(tv)
        if np.isfinite(ta):
            ta_s.append(ta)
    p_v = float((np.sum(np.asarray(tv_s) >= tau_v) + 1) / (len(tv_s) + 1)) if tv_s else np.nan
    p_a = float((np.sum(np.asarray(ta_s) >= tau_a) + 1) / (len(ta_s) + 1)) if ta_s else np.nan
    # sampling-drift controls
    n_q = np.asarray(p["q_n"], float)
    se_q = np.asarray([np.nan if x is None else x for x in p["q_se"]], float)
    tau_n = _kendall_t(np.where(n_q > 0, n_q, np.nan))
    tau_se = _kendall_t(se_q ** 2)
    return dict(d_gl_km=p["d_gl_km"], vbar=pr["vbar"], n_quarters=pr["n_ok"],
                tau_var=tau_v, tau_ac1=tau_a, p_var=p_v, p_ac1=p_a,
                tau_npairs=tau_n, tau_se2=tau_se,
                sigma_rel_late_early=_late_early_ratio(var_w, pr["vbar"]))


def _late_early_ratio(var_w, vbar):
    ok = np.isfinite(var_w)
    if ok.sum() < 10:
        return np.nan
    idx = np.where(ok)[0]
    third = max(1, idx.size // 3)
    early = np.nanmedian(var_w[idx[:third]])
    late = np.nanmedian(var_w[idx[-third:]])
    if early <= 0:
        return np.nan
    return float(math.sqrt(late / early))


# ----------------------------------------------------------------- analysis
def analyze(cache=None, seed=0):
    cache = cache or json.load(open(_CACHE))
    res = {"what": "§I.3 temporal CSD FIELD TEST: rolling variance + AC1 trends of "
                   "quarterly ITS_LIVE speed residuals, 2014-2025",
           "registered_prediction": ("streams approaching flotation (ASE trunks, thinning "
                                     "-> N declining): rising variance AND AC1 near the GL; "
                                     "steady grounded control: no rise"),
           "method": ("quarterly medians (pairs<=60 d, >=5/quarter) -> deseasonalize -> "
                      "detrend -> 12-quarter rolling noise-corrected variance + AC1 -> "
                      "Kendall tau vs time; per-point one-sided p from 200 Fourier-phase "
                      "surrogates; sampling-drift controls (pair count, SE² trends)"),
           "groups": {}, "streams": {}}
    all_groups = {"trunk_nearGL": [], "trunk_upstream": [], "control_nearGL": [],
                  "control_upstream": []}
    for name, S in cache["streams"].items():
        rows = []
        for k, p in enumerate(S["points"]):
            st = point_ews(p, seed=seed + k)
            if st is not None:
                rows.append(st)
        res["streams"][name] = dict(expectation=S["expectation"], n_points=len(rows),
                                    points=rows)
        for st in rows:
            near = st["d_gl_km"] <= NEAR_KM
            if name == CONTROL:
                all_groups["control_nearGL" if near else "control_upstream"].append(st)
            elif name in TRUNKS:
                all_groups["trunk_nearGL" if near else "trunk_upstream"].append(st)

    def summarize(rows):
        if not rows:
            return dict(n=0)
        tv = np.asarray([r["tau_var"] for r in rows])
        ta = np.asarray([r["tau_ac1"] for r in rows])
        pv = np.asarray([r["p_var"] for r in rows])
        pa = np.asarray([r["p_ac1"] for r in rows])
        joint = (tv > 0) & (ta > 0)
        sig_v = (tv > 0) & (pv < 0.1)
        sig_joint = joint & (pv < 0.1) & (pa < 0.1)
        tn = np.asarray([r["tau_npairs"] for r in rows])
        tse = np.asarray([r["tau_se2"] for r in rows])
        return dict(n=len(rows),
                    median_tau_var=float(np.median(tv)),
                    median_tau_ac1=float(np.median(ta)),
                    frac_var_rising_sig=float(np.mean(sig_v)),
                    frac_joint=float(np.mean(joint)),
                    frac_joint_sig=float(np.mean(sig_joint)),
                    median_tau_npairs=float(np.nanmedian(tn)),
                    median_tau_se2=float(np.nanmedian(tse)),
                    median_amp_ratio=float(np.nanmedian(
                        [r["sigma_rel_late_early"] for r in rows])))
    for g, rows in all_groups.items():
        res["groups"][g] = summarize(rows)
    g = res["groups"]
    tn, cn = g.get("trunk_nearGL", {}), g.get("control_nearGL", {})
    res["headline"] = dict(
        trunk_nearGL_frac_joint_sig=tn.get("frac_joint_sig"),
        control_nearGL_frac_joint_sig=cn.get("frac_joint_sig"),
        trunk_nearGL_median_tau_var=tn.get("median_tau_var"),
        control_nearGL_median_tau_var=cn.get("median_tau_var"),
        sampling_drift_caveat=dict(
            trunk_nearGL_tau_npairs=tn.get("median_tau_npairs"),
            trunk_nearGL_tau_se2=tn.get("median_tau_se2")))
    res["verdict"] = _verdict(res)
    return res


def _verdict(res):
    g = res["groups"]
    tn, tu = g.get("trunk_nearGL", {}), g.get("trunk_upstream", {})
    cn = g.get("control_nearGL", {})
    if not tn.get("n"):
        return "insufficient trunk near-GL series"
    fj, fj_c = tn.get("frac_joint_sig", 0), cn.get("frac_joint_sig", 0) or 0
    mv, mv_c = tn.get("median_tau_var", 0), cn.get("median_tau_var", 0) or 0
    drift = abs(tn.get("median_tau_npairs") or 0)
    lines = []
    if fj >= 0.3 and fj >= 3 * max(fj_c, 0.03) and mv > 0.2 > abs(mv_c):
        lines.append("JOINT CSD SIGNATURE on the near-GL trunks (var+AC1 rising, "
                     "surrogate-significant) absent on the control")
    elif mv > 0.2 and fj >= 0.2:
        lines.append("PARTIAL: variance trends rise on the near-GL trunks with a "
                     "sub-threshold joint fraction")
    else:
        lines.append("REGISTERED NULL on this record: no population-level CSD rise on "
                     "the near-GL trunks 2014-2025")
    lines.append(f"(trunk near-GL: n={tn.get('n')}, median tau_var="
                 f"{tn.get('median_tau_var'):+.2f}, tau_ac1={tn.get('median_tau_ac1'):+.2f}, "
                 f"joint-sig frac={tn.get('frac_joint_sig'):.2f}; control near-GL "
                 f"n={cn.get('n')}, tau_var={cn.get('median_tau_var', float('nan')):+.2f}, "
                 f"joint-sig {cn.get('frac_joint_sig', float('nan')):.2f}; upstream trunks "
                 f"tau_var={tu.get('median_tau_var', float('nan')):+.2f})")
    if drift > 0.3:
        lines.append(f"CAVEAT: ITS_LIVE sampling density trends through the record "
                     f"(median tau_npairs={tn.get('median_tau_npairs'):+.2f}); the "
                     "noise-corrected rolling variance subtracts the per-window SE² "
                     "floor, but residual sampling drift cannot be fully excluded")
    return " | ".join(lines)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    groups = ["trunk_nearGL", "trunk_upstream", "control_nearGL", "control_upstream"]
    colors = dict(trunk_nearGL="#d95f02", trunk_upstream="#fdbf6f",
                  control_nearGL="#1b9e77", control_upstream="#a6dba0")
    # panel a: tau_var distributions
    for ax, key, lab in ((axes[0], "tau_var", r"Kendall $\tau$ of rolling variance"),
                         (axes[1], "tau_ac1", r"Kendall $\tau$ of rolling AC1")):
        data, labels = [], []
        for gname in groups:
            rows = _group_rows(res, gname)
            if rows:
                data.append([r[key] for r in rows]); labels.append(gname.replace("_", "\n"))
        if data:
            ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    ax = axes[2]
    for gname in groups:
        rows = _group_rows(res, gname)
        if rows:
            ax.scatter([r["tau_var"] for r in rows], [r["tau_ac1"] for r in rows],
                       s=14, alpha=0.75, label=gname, color=colors[gname])
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\tau$(variance)"); ax.set_ylabel(r"$\tau$(AC1)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("joint CSD quadrant = upper right")
    fig.suptitle("§I.3 FIELD TEST — temporal CSD early-warning, quarterly ITS_LIVE "
                 "2014-2025", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure -> {path}")


def _group_rows(res, gname):
    rows = []
    for name, S in res["streams"].items():
        for st in S["points"]:
            near = st["d_gl_km"] <= NEAR_KM
            g = (("control_" if name == CONTROL else "trunk_") +
                 ("nearGL" if near else "upstream"))
            if g == gname:
                rows.append(st)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default=os.path.join(_REPORTS, "temporal_csd_field.json"))
    a = ap.parse_args()
    if a.fetch:
        fetch()
    res = analyze()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res["groups"], indent=2))
    print("VERDICT:", res["verdict"])
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
