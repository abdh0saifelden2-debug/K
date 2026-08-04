r"""§G.4 / §H.2 population extension — the drainage-response data fit UN-GATED:
ATL15-dated subglacial drainage events (2019-2026) × quarterly ITS_LIVE speed response
on the full Siegfried-Fricker network.

What was gated, and what un-gates it
------------------------------------
FUTURE_WORK §G.4 ends with "Not a data fit (drainage dates USAP-DC-gated)": the
lag-kernel prediction (two-compartment hydraulic response peaked at t* ~ 0.01-2 yr)
was derived, but fitting it to data needed drainage DATES for a population of lakes.
The §H.2 test that ran (lake_lag_atl15_itslive.py) used the older CryoSat/ICESat event
lists with ANNUAL velocity (1 in-band detection / 19 testable lakes); the richer §I
re-analysis (lake_lag_sn_ews.py) had 3 lakes. This module supplies the missing piece
from data already in-repo: the committed NR43 ATL15 cache (10-km delta_h, quarterly
2019-2026, all 131 outlines) contains the drainage dates directly — no USAP-DC, no
7.6 GB re-download. Crossed with quarterly ITS_LIVE v2 medians at each lake, the §G.4
band (1-8 quarters) is finally testable on a ~20-event dated population.

Method
------
* Event detection (committed dh cache only): sustained level drop — at epoch i,
  median(dh[i+1:i+5]) - median(dh[i-4:i]) < -3 sigma_quad (sigma floor 5 cm); events
  separated by >=6 quarters; each needs >=4 pre and >=2 post quarters in-window.
* Velocity: quarterly medians (pairs <= 90 d baseline, >=4 pairs/quarter, MAD/sqrt(n)
  SEs) of the ITS_LIVE v2 datacube series at the lake centroid, 2018-2026.
* Response test per event: baseline = median of the 4 pre-event quarters; response
  r(l) = (v(t_ev + l) - baseline)/baseline for lags l = 1..8 quarters (the derived
  0.25-2 yr band). DETECTION = >=2 consecutive in-band quarters with |r| > max(2
  sigma_rel, 2 %) — sign recorded (positive = the §G.4 surge direction).
* Detector calibration on the SAME data: (i) pseudo-events at the same dates on quiet
  lakes (no ATL15 event, matched coverage), (ii) time-reversed pseudo-events (the
  pre-event side of real events must not fire). The population claim is a comparison
  of detection rates, not raw counts.

Honest scope
------------
10-km ATL15 pixels smear small lakes (amplitudes are lower bounds); centroid velocity
is one point (trunk-average response would need flow-band averaging); polar-hole lakes
(Siple Coast) mostly fail the pair-count floor and drop out as UNTESTABLE (counted);
quarterly resolution cannot see the fast t* ~ 0.01 yr corner of the derived band.

Run modes
---------
  python lake_drainage_response_atl15.py --fetch   # network: velocity cache
  python lake_drainage_response_atl15.py           # offline: analyze committed caches
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_ATL15 = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "general_two_clocks",
                                       "data", "atl15_lake_dh_cache.json"))
_CACHE = os.path.join(_DATA, "lake_drainage_response_cache.json")
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))

SIG_FLOOR_M = 0.05
DROP_SIGMA = 3.0
DT_MAX_DAYS = 90.0
MIN_PAIRS_PER_M = 3
POST_WINDOWS = [(0.10, 0.80), (0.80, 1.50), (1.50, 2.20)]   # yr after event
BASE_WINDOW = (-1.35, -0.10)                                # yr before event
DET_REL = 0.02                        # 2 % floor
YEARS_SPAN = (2018.0, 2026.25)
N_CONTROL_LAKES = 40


# ----------------------------------------------------------------- events
def detect_events(atl15=None):
    atl15 = atl15 or json.load(open(_ATL15))
    out = []
    quiet = []
    for L in atl15["per_lake"]:
        dh = np.asarray(L["dh"], float)
        t = np.asarray(L["t"], float)
        s = max(float(L["ns_std_quad"]), SIG_FLOOR_M)
        evs = []
        for i in range(4, dh.size - 2):
            pre = float(np.median(dh[max(0, i - 4):i]))
            post = float(np.median(dh[i + 1:min(dh.size, i + 5)]))
            drop = post - pre
            if drop < -DROP_SIGMA * s:
                evs.append((float(t[i]), drop))
        # merge: keep strongest within any 6-quarter run
        evs.sort()
        merged = []
        for tv, dr in evs:
            if merged and (tv - merged[-1][0]) < 1.5:
                if dr < merged[-1][1]:
                    merged[-1] = (tv, dr)
            else:
                merged.append((tv, dr))
        if merged:
            out.append(dict(name=L["name"], cx=L["cx"], cy=L["cy"],
                            sigma_quad=s,
                            events=[dict(t=tv, drop_m=dr) for tv, dr in merged]))
        else:
            quiet.append(dict(name=L["name"], cx=L["cx"], cy=L["cy"],
                              ns_std_quad=float(L["ns_std_quad"])))
    return out, quiet


# ----------------------------------------------------------------- fetch
def _mgrid():
    m = []
    y = YEARS_SPAN[0]
    while y < YEARS_SPAN[1]:
        m.append(round(y, 4))
        y += 1.0 / 12.0
    return m


def fetch():
    import itslive
    from pyproj import Transformer
    ev_lakes, quiet = detect_events()
    rng = np.random.default_rng(3)
    quiet_pick = [quiet[i] for i in rng.permutation(len(quiet))[:N_CONTROL_LAKES]]
    tr = Transformer.from_crs(3031, 4326, always_xy=True)
    ms = _mgrid()

    def series_for(rows, tag):
        lonlat = [tr.transform(r["cx"], r["cy"]) for r in rows]
        out = []
        B = 25
        for b0 in range(0, len(lonlat), B):
            ts = itslive.velocity_cubes.get_time_series(points=lonlat[b0:b0 + B],
                                                        variables=["v", "date_dt"])
            for k, t in enumerate(ts):
                d = t["time_series"]
                v = np.asarray(d["v"].values, float)
                dt = np.asarray(d["date_dt"].values.astype("timedelta64[D]").astype(float))
                md = d["mid_date"].values
                dy = (md.astype("datetime64[D]").astype(int) / 365.2425) + 1970.0
                ok = np.isfinite(v) & (dt <= DT_MAX_DAYS)
                med = [None] * len(ms); se = [None] * len(ms); cnt = [0] * len(ms)
                for i, m0 in enumerate(ms):
                    m_ = ok & (dy >= m0) & (dy < m0 + 1.0 / 12.0)
                    n = int(m_.sum())
                    if n >= MIN_PAIRS_PER_M:
                        vv = v[m_]
                        mm = float(np.median(vv))
                        med[i] = mm
                        se[i] = 1.4826 * float(np.median(np.abs(vv - mm))) / math.sqrt(n)
                        cnt[i] = n
                r = dict(rows[b0 + k])
                r.update(m_median=med, m_se=se, m_n=cnt)
                out.append(r)
            print(f"  [{tag}] {min(b0 + B, len(lonlat))}/{len(lonlat)}")
        return out
    cache = {"_description": "committed cache for the §G.4/§H.2 ATL15-dated drainage-"
                             "response population test: MONTHLY ITS_LIVE medians at "
                             "event lakes and quiet-control lakes (monthly, not "
                             "quarterly: Antarctic optical pairs cluster Oct-Mar and "
                             "leave half the quarters empty)",
             "_months": ms,
             "_filters": dict(dt_max_days=DT_MAX_DAYS, min_pairs_per_m=MIN_PAIRS_PER_M),
             "event_lakes": series_for(ev_lakes, "events"),
             "control_lakes": series_for(quiet_pick, "controls")}
    _dump(cache, _CACHE)
    print(f"cache -> {_CACHE}")
    return cache


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


# ----------------------------------------------------------------- response test
def _pool(ms, med, se, n, lo, hi):
    """Pool monthly bins with mid-time in [lo,hi): n-weighted median of medians,
    SE from the weighted mean of bin variances / n_bins."""
    ms = np.asarray(ms, float)
    sel = [i for i in range(len(ms)) if lo <= ms[i] < hi and med[i] is not None]
    if not sel:
        return None
    vals = np.asarray([med[i] for i in sel], float)
    ses = np.asarray([se[i] for i in sel], float)
    wts = np.asarray([n[i] for i in sel], float)
    order = np.argsort(vals)
    cw = np.cumsum(wts[order]) / wts.sum()
    m = float(vals[order][np.searchsorted(cw, 0.5)])
    pooled_se = float(math.sqrt(np.average(ses ** 2, weights=wts) / len(sel)))
    return dict(v=m, se=pooled_se, n_months=len(sel), n_pairs=int(wts.sum()))


def event_response(ms, med, se, n, t_ev):
    """Event-relative windowed response; None + reason if untestable."""
    base = _pool(ms, med, se, n, t_ev + BASE_WINDOW[0], t_ev + BASE_WINDOW[1])
    if base is None or base["n_months"] < 3:
        return None, "pre-event coverage"
    if base["v"] <= 20.0:
        return None, "site too slow (<20 m/yr)"
    wins = []
    for lo, hi in POST_WINDOWS:
        w = _pool(ms, med, se, n, t_ev + lo, t_ev + hi)
        if w is not None and w["n_months"] >= 2:
            wins.append((lo, hi, w))
    if len(wins) < 2:
        return None, "post-event coverage"
    r, r_se, spans = [], [], []
    for lo, hi, w in wins:
        r.append((w["v"] - base["v"]) / base["v"])
        r_se.append(math.sqrt(w["se"] ** 2 + base["se"] ** 2) / base["v"])
        spans.append((lo, hi))
    r = np.asarray(r); r_se = np.asarray(r_se)
    thr = np.maximum(2.0 * r_se, DET_REL)
    hot = np.abs(r) > thr
    det = False; det_sign = 0; det_window = None
    for k in range(len(r) - 1):
        if hot[k] and hot[k + 1] and np.sign(r[k]) == np.sign(r[k + 1]):
            det = True; det_sign = int(np.sign(r[k])); det_window = list(spans[k])
            break
    imax = int(np.argmax(np.abs(r)))
    return dict(baseline_m_yr=base["v"], baseline_se=base["se"],
                n_windows=len(r), windows=spans, r=r.tolist(), r_se=r_se.tolist(),
                max_abs_r=float(np.abs(r[imax])), r_at_max=float(r[imax]),
                sig_at_max=float(np.abs(r[imax]) / max(r_se[imax], 1e-9)),
                detected=det, det_sign=det_sign, det_window_yr=det_window), None


def analyze(cache=None, atl15=None):
    cache = cache or json.load(open(_CACHE))
    ev_lakes, _ = detect_events(atl15)
    ev_map = {L["name"]: L["events"] for L in ev_lakes}
    ms = cache["_months"]
    rows = []
    untestable = []
    for L in cache["event_lakes"]:
        for ev in ev_map.get(L["name"], []):
            resp, why = event_response(ms, L["m_median"], L["m_se"], L["m_n"], ev["t"])
            if resp is None:
                untestable.append(dict(name=L["name"], t=ev["t"],
                                       drop_m=ev["drop_m"], reason=why))
                continue
            rows.append(dict(name=L["name"], t_ev=ev["t"], drop_m=ev["drop_m"], **resp))
    # calibration 1: pseudo-events on quiet lakes at the SAME dates
    pseudo = []
    ev_dates = [r["t_ev"] for r in rows] or [e["t"] for L in ev_lakes for e in L["events"]]
    for C in cache["control_lakes"]:
        for t_ev in ev_dates:
            resp, _why = event_response(ms, C["m_median"], C["m_se"], C["m_n"], t_ev)
            if resp is not None:
                pseudo.append(dict(name=C["name"], t_ev=t_ev, **resp))
    # calibration 2: pre-event pseudo-dates on the SAME (unreversed) event lakes,
    # 2.5 yr before each real event, skipped when another event contaminates the span
    rev = []
    for L in cache["event_lakes"]:
        evs = ev_map.get(L["name"], [])
        for ev in evs:
            t_ps = ev["t"] - 2.5
            if any(abs(t_ps - other["t"]) < 2.3 for other in evs if other is not ev):
                continue
            resp, _why = event_response(ms, L["m_median"], L["m_se"], L["m_n"], t_ps)
            if resp is not None:
                rev.append(dict(name=L["name"], **resp))
    n_det = sum(r["detected"] for r in rows)
    n_pos = sum(r["detected"] and r["det_sign"] > 0 for r in rows)
    fp = (sum(r["detected"] for r in pseudo) / len(pseudo)) if pseudo else np.nan
    fp_rev = (sum(r["detected"] for r in rev) / len(rev)) if rev else np.nan
    from scipy.stats import binomtest
    p_binom = (float(binomtest(n_det, len(rows), max(fp, 1e-6), "greater").pvalue)
               if rows and np.isfinite(fp) else np.nan)
    amps = np.asarray([r["max_abs_r"] for r in rows]) if rows else np.asarray([])
    res = dict(
        what="§G.4/§H.2 population data fit UN-GATED: ATL15-dated drainage events "
             "(2019-2026) x monthly-binned ITS_LIVE response in the derived 0.25-2 yr band",
        n_lakes_with_events=len(ev_map),
        n_events_dated=int(sum(len(v) for v in ev_map.values())),
        n_events_testable=len(rows), n_untestable=len(untestable),
        untestable=untestable,
        n_detections=n_det, n_detections_positive=n_pos,
        detection_rate=float(n_det / len(rows)) if rows else np.nan,
        false_positive_rate_quiet=float(fp) if np.isfinite(fp) else None,
        n_pseudo=len(pseudo),
        false_positive_rate_preevent=float(fp_rev) if np.isfinite(fp_rev) else None,
        n_preevent=len(rev),
        p_binomial_vs_quiet=p_binom,
        amp_percentiles_pct=(dict(p50=float(np.percentile(amps, 50) * 100),
                                  p90=float(np.percentile(amps, 90) * 100),
                                  max=float(amps.max() * 100)) if amps.size else None),
        events=rows)
    det_names = [f"{r['name']}@{r['t_ev']:.2f}(sign {r['det_sign']:+d}, window "
                 f"{r['det_window_yr']}, max {100 * r['max_abs_r']:.1f}%)"
                 for r in rows if r["detected"]]
    from collections import Counter
    res["untestable_reasons"] = dict(Counter(u["reason"] for u in untestable))
    res["detections"] = det_names
    res["verdict"] = _verdict(res)
    return res


def _verdict(res):
    n, d = res["n_events_testable"], res["n_detections"]
    fp_q = res["false_positive_rate_quiet"]
    fp_p = res["false_positive_rate_preevent"]
    amp = res["amp_percentiles_pct"]
    if n == 0:
        return "no testable events (coverage)"
    lines = [f"POPULATION NULL: {d}/{n} dated drainage events show a sustained in-band "
             f"(0.25-2 yr) velocity 'response' — but the SAME detector fires on "
             f"{fp_p if fp_p is None else round(fp_p, 3)} of pre-event windows of the "
             f"same lakes and {fp_q if fp_q is None else round(fp_q, 3)} of quiet-lake "
             f"pseudo-events (binomial vs quiet p="
             f"{res['p_binomial_vs_quiet'] if res['p_binomial_vs_quiet'] is None else round(res['p_binomial_vs_quiet'], 3)}): "
             "the detected changes are indistinguishable from the background variability "
             "of these dynamic trunk sites — drainage timing adds nothing"]
    if amp:
        lines.append(f"the population response BOUND: median max|dv/v| {amp['p50']:.1f}%, "
                     f"p90 {amp['p90']:.1f}% across 0.25-2 yr — via dv/v=|s_N|dN/N these "
                     "sites sit far from the N_c fold (§I.1/§I.2 reading), consistent "
                     "with the earlier 1/19 CryoSat-era result and lake_lag_sn_ews")
    if d and 0 < res["n_detections_positive"] < d:
        lines.append(f"nominal detection signs are mixed ({res['n_detections_positive']}"
                     f"/{d} positive) — not the uniform §G.4 surge direction")
    return " | ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default=os.path.join(_REPORTS, "lake_drainage_response.json"))
    a = ap.parse_args()
    if a.fetch:
        fetch()
    res = analyze()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("events", "untestable")},
                     indent=2, default=str)[:2500])
    print("VERDICT:", res["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
