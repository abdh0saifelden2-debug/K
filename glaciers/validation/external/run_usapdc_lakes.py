r"""§V.2c — USAP-DC subglacial-lake real data: drainage forcing + thermal kernel.

This closes the one half of the §G.4 / §V.2 sliding-law test that every earlier
runner flagged as *unobtainable*.  ``run_sliding_real.py`` (lines 11-14) and
``REAL_DATA_RESULTS.md`` (§V.2 / §V.2b) state that the vetted lake
**volume-change time series** -- the drainage-event *dates and volumes* that are
the §G.4 forcing ``q_water`` -- live behind a USAP-DC login and so the forcing
side could only be approximated (CryoSat-2 *elevation* proxies for 5 lakes).

The USAP-DC datasets are in fact reachable (a bot-check, not a login), so this
module uses the **real vetted catalogues**:

  * **USAP-DC 601439** (Smith et al. 2009, ICESat active-lake inventory) --
    per-lake ``Volume_history.csv`` water-volume anomaly series (2003-2009) for
    124 lakes.  ``detect_drainages`` extracts discrete fill->drain events with
    real dates, drained volumes [km^3] and drain rates -> the genuine §G.4
    ``q_water`` forcing magnitudes (R1, R3).
  * **USAP-DC 601470** (Stubblefield et al. 2021; lake outlines after Siegfried
    & Fricker 2018) -- ``active_lake_statistics.dat`` gives each of 131 lakes a
    **BedMachine** ice thickness, an *independent* thickness source to the
    Bedmap2 sampling used in §V.2.  ``tau_ice = H^2/kappa_ice`` is recomputed and
    compared with the observed 0.02-2 yr surge-lag band (R2), with a
    BedMachine-vs-Bedmap2 cross-check at the same centroids (robustness).

Honest scope
------------
This promotes the **forcing** half from "USAP-DC-gated / approximated" to
"obtained and characterised on the vetted catalogue".  It does **not** promote
the lag *value* to [VERIFIED]: that still needs a co-temporal *velocity
response* (sub-annual GPS/GNSS, EarthScope-gated) for the 2003-2009 ICESat era,
which 601439 does not contain.  The §G.4 thermal kernel stays **[FALSIFIED]**.

Provenance (open bot-check, no login) -- fetch once, then this is reproducible::

    # USAP-DC 601470 (lake stats + H/beta maps)
    #   https://www.usap-dc.org/view/dataset/601470  -> data.zip
    #   unzip into ~/data_usapdc/601470/data/active_lake_statistics.dat
    # USAP-DC 601439 (per-lake volume-change time series)
    #   https://www.usap-dc.org/view/dataset/601439  -> subglacial_lakes_data.zip
    #   unzip into ~/data_usapdc/601439/<Lake>/Volume_history.csv
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from external import DataUnavailableError  # noqa: E402

KAPPA_ICE = 1.09e-6                      # ice thermal diffusivity [m^2 s^-1]
SEC_PER_YR = 365.25 * 86400.0
KM3YR_TO_M3S = 1.0e9 / SEC_PER_YR       # 1 km^3/yr -> m^3/s  (~31.69)

# Observed post-drainage surge-lag band (order of magnitude), same as §V.2:
# Stearns et al. 2008 (Byrd); Siegfried et al. 2016; literature -> days..~2 yr.
OBS_LAG_YR = (0.02, 2.0)

STATS_601470 = os.path.expanduser("~/data_usapdc/601470/data/active_lake_statistics.dat")
DIR_601439 = os.path.expanduser("~/data_usapdc/601439")
BEDMAP_DIR = os.path.expanduser("~/data_bedmap/bedmap2_bin")


# --------------------------------------------------------------------------- #
# 601470 -- lake geometry + Arthern et al. (2015) thickness (Stubblefield et al. 2021)
# --------------------------------------------------------------------------- #
def load_active_lake_stats(path=STATS_601470):
    """Load USAP-DC 601470 ``active_lake_statistics.dat`` (131 lakes).

    Columns (per the dataset readme): PS71 ``x``, ``y`` centroid [m]; Feret
    ``width``, ``length`` [km]; area-equivalent circle ``diam`` [km]; BedMachine
    ice ``thickness`` [m].
    """
    if not os.path.exists(path):
        raise DataUnavailableError(
            "USAP-DC 601470 active_lake_statistics.dat not found.\n"
            "Fetch https://www.usap-dc.org/view/dataset/601470 (bot-check, no "
            "login), unzip, and place at ~/data_usapdc/601470/data/."
        )
    a = np.loadtxt(path, delimiter=",", ndmin=2)  # ndmin=2 keeps a single-row file 2D so a[:, c] works
    # The empty-file shape is NumPy-version dependent (e.g. (0, 1) on >=2.x,
    # (1, 0) on some older builds); both are caught below -- the former by the
    # row check, the latter by the column check -- so this stays robust across
    # the numpy>=1.24 range in requirements.txt.
    if a.shape[0] == 0 or a.shape[1] < 6:
        raise DataUnavailableError(
            "USAP-DC 601470 active_lake_statistics.dat is empty or malformed "
            f"(parsed shape {a.shape}; expected >=1 row x >=6 columns).\n"
            "Re-fetch https://www.usap-dc.org/view/dataset/601470 (bot-check, no "
            "login), unzip, and place at ~/data_usapdc/601470/data/."
        )
    return {
        "x": a[:, 0], "y": a[:, 1],
        "feret_w_km": a[:, 2], "feret_l_km": a[:, 3], "equiv_diam_km": a[:, 4],
        "thickness_m": a[:, 5],
    }


def tau_ice_years(H_m):
    """Literal §G.4 memory timescale ``H^2 / kappa_ice`` in years."""
    H_m = np.asarray(H_m, float)
    return (H_m ** 2 / KAPPA_ICE) / SEC_PER_YR


def sample_bedmap_thickness(xs, ys, bin_dir=BEDMAP_DIR):
    """Nearest-cell Bedmap2 ice thickness [m] at EPSG:3031 points (NaN off-grid).

    Independent thickness source for the 601470 BedMachine cross-check.
    """
    from external.bedmap2_loader import read_flt
    H, m = read_flt(os.path.join(bin_dir, "bedmap2_thickness.flt"))
    nrows, ncols, cs = m["nrows"], m["ncols"], m["cellsize"]
    xll, yll = m["xll"], m["yll"]
    out = np.full(len(xs), np.nan)
    for i, (px, py) in enumerate(zip(xs, ys)):
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        c = int(round((px - xll) / cs - 0.5))
        r = nrows - 1 - int(round((py - yll) / cs - 0.5))
        if 0 <= r < nrows and 0 <= c < ncols:
            v = H[r, c]
            if np.isfinite(v) and v > 0:
                out[i] = v
    return out


# --------------------------------------------------------------------------- #
# 601439 -- per-lake volume-change series -> drainage events
# --------------------------------------------------------------------------- #
def _to_decimal_year(dt):
    start = datetime(dt.year, 1, 1)
    next_start = datetime(dt.year + 1, 1, 1)
    return dt.year + (dt - start).total_seconds() / (next_start - start).total_seconds()


def parse_volume_history(csv_path):
    """Parse a 601439 ``Volume_history.csv`` -> (t_decimal_yr, V_km3), date-sorted.

    Rows are ``index, dd-Mon-yyyy, cumulative_volume_anomaly_km3``.
    """
    rows = []
    with open(csv_path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                dt = datetime.strptime(parts[1], "%d-%b-%Y")
                v = float(parts[2])
            except ValueError:
                continue
            rows.append((_to_decimal_year(dt), v))
    rows.sort()
    if not rows:
        return np.array([]), np.array([])
    t, v = zip(*rows)
    return np.asarray(t, float), np.asarray(v, float)


def detect_drainages(t, v, drop_thresh=0.05, recover_frac=0.5):
    """Discrete fill->drain events in a cumulative volume-anomaly series.

    Distinct from ``lag_fit_real.detect_drainages`` (despite the shared name):
    this one is the ICESat volume-anomaly (601439) detector; that one operates
    on CryoSat-2 elevation series with a different signature/algorithm.

    A peak-to-trough state machine: track the running peak; once the series has
    fallen ``drop_thresh`` [km^3] below it and then recovered by ``recover_frac``
    of that drop (or the record ends), close one drainage event.

    Returns a list of dicts: ``t_peak``, ``t_trough`` [yr], ``drop_km3``,
    ``dur_yr`` (trough - peak), ``rate_km3_per_yr`` (= drop/dur).
    """
    n = len(v)
    events = []
    if n < 3:
        return events
    peak_i = 0
    trough_i = None

    def _close(pi, ti):
        drop = float(v[pi] - v[ti])
        if drop < drop_thresh:
            return
        dur = float(t[ti] - t[pi])
        rate = drop / dur if dur > 0 else float("nan")
        events.append({
            "t_peak": float(t[pi]), "t_trough": float(t[ti]),
            "drop_km3": drop, "dur_yr": dur, "rate_km3_per_yr": rate,
        })

    for i in range(1, n):
        if trough_i is None:
            if v[i] >= v[peak_i]:
                peak_i = i                       # still climbing / flat
            else:
                trough_i = i                     # drawdown begins
        else:
            if v[i] < v[trough_i]:
                trough_i = i                     # deeper trough
            else:
                drop = v[peak_i] - v[trough_i]
                if v[i] >= v[trough_i] + recover_frac * max(drop, drop_thresh):
                    _close(peak_i, trough_i)     # recovered -> event closed
                    peak_i = i
                    trough_i = None
    if trough_i is not None:
        _close(peak_i, trough_i)                 # open event at record end
    return events


def build_drainage_catalogue(dir_439=DIR_601439, drop_thresh=0.05):
    """Walk all 601439 lakes -> (events, per_lake) with real drainage forcing."""
    if not os.path.isdir(dir_439):
        raise DataUnavailableError(
            "USAP-DC 601439 directory not found.\n"
            "Fetch https://www.usap-dc.org/view/dataset/601439 (bot-check, no "
            "login), unzip, and place lakes at ~/data_usapdc/601439/<Lake>/."
        )
    events, per_lake = [], []
    for name in sorted(os.listdir(dir_439)):
        csv_path = os.path.join(dir_439, name, "Volume_history.csv")
        if not os.path.exists(csv_path):
            continue
        t, v = parse_volume_history(csv_path)
        if t.size < 3:
            continue
        evs = detect_drainages(t, v, drop_thresh=drop_thresh)
        for e in evs:
            e2 = dict(e); e2["lake"] = name
            events.append(e2)
        per_lake.append({
            "lake": name, "n_pts": int(t.size),
            "t_start": float(t[0]), "t_end": float(t[-1]),
            "v_min": float(v.min()), "v_max": float(v.max()),
            "n_events": len(evs),
        })
    return events, per_lake


def _pct(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else float("nan")


def _arr_max(a):
    """max of finite entries, or nan for an empty/all-nan array.

    Mirrors ``_pct`` so the summary's ``max`` field is nan (not ``None``) when
    no drainage events are found -- otherwise ``main()``'s ``:.3f`` format of
    ``None`` raises ``TypeError``."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else float("nan")


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def analyse(stats_path=STATS_601470, dir_439=DIR_601439, bin_dir=BEDMAP_DIR,
            drop_thresh=0.05):
    stats = load_active_lake_stats(stats_path)
    H_bm = stats["thickness_m"]
    ok = np.isfinite(H_bm) & (H_bm > 0)
    tau = tau_ice_years(H_bm[ok])

    # Independent Bedmap2 thickness at the same centroids (cross-check).
    try:
        H_b2 = sample_bedmap_thickness(stats["x"], stats["y"], bin_dir=bin_dir)
    except Exception:
        H_b2 = np.full(H_bm.shape, np.nan)
    both = ok & np.isfinite(H_b2)
    if both.sum() >= 3:
        dh = H_bm[both] - H_b2[both]
        cross = {
            "n": int(both.sum()),
            "median_bedmachine_m": float(np.median(H_bm[both])),
            "median_bedmap2_m": float(np.median(H_b2[both])),
            "median_abs_diff_m": float(np.median(np.abs(dh))),
            "corr": float(np.corrcoef(H_bm[both], H_b2[both])[0, 1]),
        }
    else:
        cross = {"n": int(both.sum()), "note": "too few co-located cells"}

    events, per_lake = build_drainage_catalogue(dir_439, drop_thresh=drop_thresh)
    drops = np.array([e["drop_km3"] for e in events], float)
    rates = np.array([e["rate_km3_per_yr"] for e in events], float)
    q_m3s = rates * KM3YR_TO_M3S

    lag_lo, lag_hi = OBS_LAG_YR
    summary = {
        "datasets": {
            "601470": "Stubblefield 2021 / Siegfried&Fricker 2018 lake stats "
                      "(131 lakes, BedMachine thickness)",
            "601439": "Smith et al. 2009 ICESat volume-change series (124 lakes)",
        },
        "R2_thermal_kernel": {
            "n_lakes": int(ok.sum()),
            "thickness_bedmachine_m": {"p5": _pct(H_bm[ok], 5),
                                       "p50": _pct(H_bm[ok], 50),
                                       "p95": _pct(H_bm[ok], 95)},
            "tau_ice_yr": {"p5": _pct(tau, 5), "p50": _pct(tau, 50),
                           "p95": _pct(tau, 95)},
            "obs_surge_lag_yr": [lag_lo, lag_hi],
            "median_overshoot_factor": _pct(tau, 50) / lag_hi,
            "verdict": "FALSIFIED as written (tau_ice >> observed surge lag)",
            "bedmachine_vs_bedmap2": cross,
        },
        "R1_drainage_forcing_601439": {
            "drop_thresh_km3": drop_thresh,
            "n_events": int(len(events)),
            "n_lakes_total": int(len(per_lake)),
            "n_lakes_with_event": int(sum(1 for p in per_lake if p["n_events"])),
            "drained_volume_km3": {"p5": _pct(drops, 5), "p50": _pct(drops, 50),
                                   "p95": _pct(drops, 95),
                                   "max": _arr_max(drops)},
            "largest_events": sorted(
                ({"lake": e["lake"], "t_peak": round(e["t_peak"], 2),
                  "drop_km3": round(e["drop_km3"], 3),
                  "rate_km3_per_yr": round(e["rate_km3_per_yr"], 3)}
                 for e in events),
                key=lambda d: -d["drop_km3"])[:8],
        },
        "R3_q_water_forcing": {
            "drain_rate_km3_per_yr": {"p5": _pct(rates, 5), "p50": _pct(rates, 50),
                                      "p95": _pct(rates, 95)},
            "q_water_m3_per_s": {"p5": _pct(q_m3s, 5), "p50": _pct(q_m3s, 50),
                                 "p95": _pct(q_m3s, 95)},
            "note": "q_water = drained_volume / drain_duration; literature "
                    "subglacial-flood discharges are O(10-100) m^3/s.",
        },
    }
    arrays = {"tau": tau, "H_bm": H_bm[ok], "H_bm_both": H_bm[both],
              "H_b2_both": H_b2[both], "drops": drops, "rates": rates,
              "events": events}
    return summary, arrays


def make_figure(summary, arrays, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) tau_ice vs observed surge-lag band
    tau = arrays["tau"]
    if tau.size:
        ax[0].hist(np.log10(tau), bins=20, color="#4477aa", alpha=0.85)
    lo, hi = summary["R2_thermal_kernel"]["obs_surge_lag_yr"]
    ax[0].axvspan(np.log10(lo), np.log10(hi), color="#cc6677", alpha=0.35,
                  label=f"obs surge lag {lo}-{hi} yr")
    ax[0].set_xlabel(r"$\log_{10}\,\tau_{ice}=H^2/\kappa$  [yr]")
    ax[0].set_ylabel("lakes")
    ax[0].set_title("(a) R2: thermal kernel FALSIFIED\n(601470 BedMachine H)")
    ax[0].legend(fontsize=8, loc="upper left")

    # (b) BedMachine vs Bedmap2 thickness at the same centroids
    hb, h2 = arrays["H_bm_both"], arrays["H_b2_both"]
    if hb.size:
        ax[1].scatter(h2, hb, s=14, color="#228833", alpha=0.7)
        lim = [0, float(max(hb.max(), h2.max())) * 1.05]
        ax[1].plot(lim, lim, "k--", lw=1)
        ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("Bedmap2 thickness [m]")
    ax[1].set_ylabel("BedMachine thickness [m] (601470)")
    cr = summary["R2_thermal_kernel"]["bedmachine_vs_bedmap2"].get("corr")
    have_cr = cr is not None and np.isfinite(cr)
    ax[1].set_title(f"(b) thickness cross-check\nr={cr:.3f}" if have_cr else "(b) cross-check")

    # (c) drained-volume distribution (601439)
    drops = arrays["drops"]
    if drops.size:
        ax[2].hist(drops, bins=20, color="#aa3377", alpha=0.85)
    n_ev = summary["R1_drainage_forcing_601439"]["n_events"]
    ax[2].set_xlabel("drained volume per event [km$^3$]")
    ax[2].set_ylabel("events")
    ax[2].set_title(f"(c) R1: real §G.4 forcing\n{n_ev} drainage events (601439)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"figure -> {os.path.abspath(out_png)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=STATS_601470)
    ap.add_argument("--dir439", default=DIR_601439)
    ap.add_argument("--bin-dir", default=BEDMAP_DIR)
    ap.add_argument("--drop-thresh", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports",
        "usapdc_lakes.json"))
    a = ap.parse_args()

    summary, arrays = analyse(a.stats, a.dir439, a.bin_dir, a.drop_thresh)
    print(json.dumps(summary, indent=2))

    r2 = summary["R2_thermal_kernel"]
    r1 = summary["R1_drainage_forcing_601439"]
    print(f"\nR2: tau_ice median {r2['tau_ice_yr']['p50']:.3e} yr vs obs "
          f"{r2['obs_surge_lag_yr']} yr -> {r2['median_overshoot_factor']:.1e}x too slow "
          f"[{r2['verdict'].split(' ')[0]}]")
    print(f"    BedMachine vs Bedmap2 thickness: r="
          f"{r2['bedmachine_vs_bedmap2'].get('corr', float('nan')):.3f}")
    print(f"R1: {r1['n_events']} drainage events over {r1['n_lakes_with_event']}/"
          f"{r1['n_lakes_total']} lakes; drained vol median "
          f"{r1['drained_volume_km3']['p50']:.3f} km^3, max "
          f"{r1['drained_volume_km3']['max']:.3f} km^3")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"json -> {os.path.abspath(a.out)}")

    # Also emit the drainage-event catalogue as a CSV with a 't' column,
    # directly consumable by lake_catalog_loader.load_drainage_events().
    events = arrays["events"]
    csv_out = os.path.splitext(a.out)[0] + "_events.csv"
    with open(csv_out, "w") as fh:
        fh.write("lake,t,drop_km3,dur_yr,rate_km3_per_yr\n")
        for e in sorted(events, key=lambda d: d["t_peak"]):
            fh.write(f"{e['lake']},{e['t_peak']:.4f},{e['drop_km3']:.4f},"
                     f"{e['dur_yr']:.4f},{e['rate_km3_per_yr']:.4f}\n")
    print(f"events csv -> {os.path.abspath(csv_out)}")

    make_figure(summary, arrays,
                os.path.join(os.path.dirname(os.path.abspath(a.out)),
                             "usapdc_lakes.png"))


if __name__ == "__main__":
    main()
