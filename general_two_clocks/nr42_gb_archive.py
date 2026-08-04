r"""NR42 real-data gate EXECUTED (ledger E9) -- the GB 1-s frequency archive,
binned by measured synchronous inertia: the registered band-limited
observability test of the grid-frequency drainage window.

NR42 (`new_relationships19.py`) transferred NR32's band-limited two-clocks
transmission verbatim to the power-grid System Frequency Response (SFR;
dictionary C<->M inertia, R<->1/beta, tau_sys = M/beta, tau_in <-> T_g) and
REGISTERED a real-data gate (`figures/nr42_grid_frequency_window.json`,
`real_data_gate`): band-limited disturbance variance in [2,30] s, binned by
inertia -- two-clocks window prediction = single-peaked with the maximum
INSIDE the inertia range (interior peak at c* ~ 30% of the all-synchronous
inertia scale); Markovian (quasi-static control) null = maximum at the
HIGHEST-inertia bin.

Data (all NESO Data Portal, open licence; raw CSVs NOT committed -- set
``$NR42_GB_DIR`` and run ``build_cache()`` to regenerate the committed
aggregate cache ``data/nr42_gb_archive_cache.json``):

* "Historic frequency data" -- 1-s GB system frequency, 7 months spanning the
  inertia transition and the Dynamic Containment (DC) introduction (Oct 2020):
  preDC = Jan 2018, Jul 2018, May 2020; DC = Jan 2022, Jul 2023, Jan 2025,
  Jul 2025 (10,416 joined half-hours, 18.7M 1-s samples kept).
* "System Inertia" -- settlement-period Outturn Inertia (GVA.s), the measured
  inertia proxy the gate asked for.
* "Historic demand data" -- settlement-period national demand ND (MW), for
  the demand-normalized control (available through Jul 2023; the DC-era
  demand-normalized stratum therefore covers Jan 2022 + Jul 2023).

Method: per half-hour Hann periodogram of the 1-s trace -> band variance in
the registered [2,30] s band (and the out-of-band 60-300 s secondary-control
band as specificity control) -> join Outturn Inertia H and demand by
settlement key -> within each era x stratum (all / day 10-17h / night 0-5h /
demand-normalized var/ND^2), octile-bin by H: median variance per bin,
argmax bin, bootstrap P(argmax interior), and per-half-hour Spearman rho(H,
var).  Per-era inertia-tercile mean spectra are pooled for the SFR-shape
exhibit and fitted with the identifiable SFR transmission shape
``|i w tau + (1-k) + k/(1+i w T_g)|^{-2}`` (tau = M/beta exactly -- NR42's
tau_sys in beta units).

Findings (figures/nr42_gb_archive.json):

1. **The Markovian null is REJECTED in the registered band, in every stratum
   of both eras** (10/10): the maximum is never at the highest-inertia bin,
   and Spearman rho <= 0.03 everywhere (pooled preDC rho = -0.24,
   p ~ 1e-59).  The quasi-static-control ordering ("more inertia = more
   observable") is the wrong ordering for real GB frequency noise.
2. **The window-side ordering is CONFIRMED on the range-clipped window**:
   variance RISES as inertia falls, toward the predicted sub-range peak --
   strongest in the controlled strata (preDC day rho = -0.64, preDC
   demand-normalized rho = -0.76, both with the maximum at the low-inertia
   edge).
3. **The interior peak itself is UNRESOLVED in-range, and the registered
   criterion says it must be**: the observed within-era inertia span is
   0.37-0.40 decades -- only ~1/3 of the [2,30] s band width (1.18 decades),
   below the criterion's own crossing requirement -- and the c* ~ 0.30
   readout puts the peak at 105-135 GVA.s for any all-synchronous scale in
   350-450 GVA.s, i.e. at/below the observed low edge (low-octile medians
   151/126 GVA.s).  The resolvable registered prediction on this window is
   the SIGN of the slope (finding 2), not the turning point.
4. **Band specificity**: out-of-band (60-300 s, secondary control/AGC) the
   ordering INVERTS -- rho = +0.18 (preDC) / +0.33 (DC) with the DC maximum
   at the highest-inertia edge (the demand/scheduling direction).  The
   in-band signal is not a generic variance artifact.
5. **DC-era flattening** (flagged reading, not registered): after Dynamic
   Containment procurement the pooled in-band ordering flattens (rho -0.24
   -> +0.03; demand-normalized -0.76 -> -0.53) -- converter-fast response
   raises in-band beta without adding storage, filling the very window the
   criterion monitors.
6. **Limit -- the ambient knee is NOT a calibration-free inertia readout**:
   fitted tau_sys does not rise with measured H across terciles (preDC
   8.9/7.6/6.0 s), consistent with beta co-varying with demand (response
   holding scales with demand, which co-varies with inertia).  The
   calibration-free critical-inertia readout of NR42 needs event-conditioned
   (infeed-loss) responses -- that piece of the gate stays open.

CPU-only.  Tests: tests/test_nr42_gb_archive.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr42_gb_archive_cache.json")
FIG = os.path.join(HERE, "figures", "nr42_gb_archive.json")

MONTHS = {
    "Jan2018": ("freq_January_2018.csv", "preDC"),
    "Jul2018": ("freq_July_2018.csv", "preDC"),
    "May2020": ("fNew 2020 5.csv", "preDC"),
    "Jan2022": ("fNew 2022 1.csv", "DC"),
    "Jul2023": ("freq_July_2023.csv", "DC"),
    "Jan2025": ("freq_January_2025.csv", "DC"),
    "Jul2025": ("freq_July_2025.csv", "DC"),
}
BANDS = {"b2_30": (1.0 / 30.0, 1.0 / 2.0),
         "b60_300": (1.0 / 300.0, 1.0 / 60.0)}
DAY_HOURS = (10, 17)
NIGHT_HOURS = (0, 5)
N_OCT = 8
N_BOOT = 2000
N_DECIM = 240
SEG = 1800


# --------------------------------------------------------------------------- #
# raw ingest (only when the NESO CSVs are present; $NR42_GB_DIR)
# --------------------------------------------------------------------------- #
def _settlement_maps(gb_dir):
    import glob

    import pandas as pd
    frames = []
    for p in sorted(glob.glob(os.path.join(gb_dir, "inertia_20*.csv"))):
        if p.endswith("inertia_2020.csv"):
            continue
        d = pd.read_csv(p)
        d.columns = [c.strip('" ').strip() for c in d.columns]
        frames.append(d)
    inr = pd.concat(frames)
    imap = dict(zip(inr["Settlement Date"].astype(str) + "|"
                    + inr["Settlement Period"].astype(str),
                    inr["Outturn Inertia"].astype(float)))
    dmap = {}
    for p in sorted(glob.glob(os.path.join(gb_dir, "demand_20*.csv"))):
        d = pd.read_csv(p)
        dt = pd.to_datetime(d["SETTLEMENT_DATE"], format="mixed",
                            dayfirst=True)
        key = (dt.dt.date.astype(str) + "|"
               + d["SETTLEMENT_PERIOD"].astype(int).astype(str))
        dmap.update(dict(zip(key, d["ND"].astype(float))))
    return imap, dmap


def _month_pass(path, imap, dmap):
    """One pass over a month of 1-s data: per-half-hour band variances,
    inertia H, demand, hour; plus the periodogram for spectra pooling."""
    import pandas as pd
    from zoneinfo import ZoneInfo
    lon = ZoneInfo("Europe/London")
    df = pd.read_csv(path)
    t = pd.to_datetime(df[df.columns[0]], utc=True, format="mixed")
    f = df[df.columns[1]].astype(float).values
    tl = t.dt.tz_convert(lon)
    key = (tl.dt.date.astype(str) + "|"
           + (tl.dt.hour * 2 + tl.dt.minute // 30 + 1).astype(int).astype(str))
    freqs = np.fft.rfftfreq(SEG, 1.0)
    w = np.hanning(SEG)
    wp = (w ** 2).mean()
    bsel = {name: (freqs >= lo) & (freqs <= hi)
            for name, (lo, hi) in BANDS.items()}
    rows, specs = [], []
    for k, idx in pd.Series(np.arange(len(f))).groupby(key.values):
        H = imap.get(k)
        if H is None:
            continue
        seg = f[idx.values]
        if len(seg) < 1710 or pd.isna(seg).mean() > 0.05:
            continue
        seg = pd.Series(seg[:SEG]).interpolate(limit=10).values
        if len(seg) < SEG or np.isnan(seg).any():
            continue
        seg = seg - seg.mean()
        P = np.abs(np.fft.rfft(seg * w)) ** 2 * 2.0 / (SEG ** 2 * wp)
        hh = int(k.split("|")[1])
        hh = (hh - 1) // 2
        rows.append(dict(H=float(H), demand=float(dmap.get(k, np.nan)),
                         hour=hh,
                         **{n: float(P[s].sum()) for n, s in bsel.items()}))
        specs.append((float(H), P))
    return rows, specs, freqs


def _octile_block(H, v, rng=None):
    """Octile-bin medians + argmax + Spearman + bootstrap interior prob."""
    from scipy.stats import spearmanr
    H = np.asarray(H)
    v = np.asarray(v)
    edges = np.quantile(H, np.linspace(0, 1, N_OCT + 1))
    edges[0] -= 1e-9
    which = np.clip(np.searchsorted(edges, H, side="right") - 1, 0, N_OCT - 1)
    hmed = [float(np.median(H[which == j])) for j in range(N_OCT)]
    vmed = [float(np.median(v[which == j])) for j in range(N_OCT)]
    n = [int(np.sum(which == j)) for j in range(N_OCT)]
    rho, p = spearmanr(H, v)
    rng = rng or np.random.default_rng(42)
    inter = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(v), len(v))
        wb, vb = which[idx], v[idx]
        med = np.array([np.median(vb[wb == j]) if np.any(wb == j) else -np.inf
                        for j in range(N_OCT)])
        if 1 <= int(np.argmax(med)) <= N_OCT - 2:
            inter += 1
    return dict(H_med=hmed, var_med=vmed, n=n, n_halfhours=int(len(v)),
                rho=float(rho), p=float(p),
                argmax_bin=int(np.argmax(vmed)),
                p_interior=float(inter / N_BOOT))


def build_cache(gb_dir=None, write=True):
    gb_dir = gb_dir or os.environ.get("NR42_GB_DIR", "/home/gridfreq")
    imap, dmap = _settlement_maps(gb_dir)
    era_rows = {"preDC": [], "DC": []}
    era_specs = {"preDC": [], "DC": []}
    freqs = None
    for tag, (fn, era) in MONTHS.items():
        rows, specs, freqs = _month_pass(os.path.join(gb_dir, fn), imap, dmap)
        era_rows[era].extend(rows)
        era_specs[era].extend(specs)
    rng = np.random.default_rng(7)
    ordering = {}
    for era, rows in era_rows.items():
        H = np.array([r["H"] for r in rows])
        dem = np.array([r["demand"] for r in rows])
        hr = np.array([r["hour"] for r in rows])
        for band in BANDS:
            v = np.array([r[band] for r in rows])
            strata = {
                "all": np.ones(len(v), bool),
                "day": (hr >= DAY_HOURS[0]) & (hr <= DAY_HOURS[1]),
                "night": (hr >= NIGHT_HOURS[0]) & (hr <= NIGHT_HOURS[1]),
                "demandnorm": np.isfinite(dem),
            }
            for st, m in strata.items():
                vv = v[m] / dem[m] ** 2 if st == "demandnorm" else v[m]
                ordering[f"{era}|{band}|{st}"] = _octile_block(H[m], vv,
                                                               rng=rng)
    # pooled per-era inertia-tercile mean spectra (log-decimated)
    spectra = {}
    for era, specs in era_specs.items():
        Hs = np.array([h for h, _ in specs])
        edges = np.quantile(Hs, [0.0, 1 / 3.0, 2 / 3.0, 1.0])
        fdec_edges = np.geomspace(freqs[1], 0.5, N_DECIM + 1)
        for j in range(3):
            sel = [P for h, P in specs if edges[j] <= h <= edges[j + 1]]
            mP = np.stack(sel).mean(axis=0)
            fd, pd_ = [], []
            for a, b in zip(fdec_edges[:-1], fdec_edges[1:]):
                m = (freqs >= a) & (freqs < b)
                if m.sum():
                    fd.append(float(np.exp(np.mean(np.log(freqs[m])))))
                    pd_.append(float(mP[m].mean()))
            spectra[f"{era}_T{j}"] = dict(
                H_med=float(np.median(Hs[(Hs >= edges[j])
                                         & (Hs <= edges[j + 1])])),
                n=len(sel), freqs=fd, P=pd_)
    cache = dict(
        meta=dict(
            months={k: v[0] for k, v in MONTHS.items()},
            eras={k: v[1] for k, v in MONTHS.items()},
            bands=BANDS, day_hours=DAY_HOURS, night_hours=NIGHT_HOURS,
            n_octiles=N_OCT, n_boot=N_BOOT,
            n_halfhours={e: len(r) for e, r in era_rows.items()},
            demand_coverage="ND available Jan2018-Jul2023; DC demandnorm "
                            "stratum = Jan2022+Jul2023 only",
            provenance="NESO Data Portal: 'Historic frequency data' (1-s "
                       "system frequency), 'System Inertia' (Outturn "
                       "Inertia, GVA.s), 'Historic demand data' (ND)"),
        ordering=ordering, spectra=spectra)
    if write:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w") as fh:
            json.dump(cache, fh)
    return cache


def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# SFR shape fit (identifiable form; tau = M/beta exactly)
# --------------------------------------------------------------------------- #
def sfr_psd_shape(w, tau, k, tg):
    """|H|^2 with H = 1/(i w tau + (1-k) + k/(1+i w tg)) -- the SFR
    transmission in beta-normalized units (tau = M/beta, k = governor share
    of beta, tg = governor lag).  Low-f -> 1; high-f -> 1/(w tau)^2."""
    den = 1j * w * tau + (1.0 - k) + k / (1.0 + 1j * w * tg)
    return 1.0 / np.abs(den) ** 2


def fit_sfr(freqs, P, fmin=1.0 / 120.0, fmax=0.4):
    from scipy.optimize import least_squares
    f = np.asarray(freqs)
    P = np.asarray(P)
    m = (f >= fmin) & (f <= fmax) & (P > 0)
    w = 2.0 * np.pi * f[m]
    y = np.log(P[m])

    def resid(q):
        la, lt, zk, ltg = q
        k = 1.0 / (1.0 + np.exp(-zk))
        return np.log(np.exp(la)
                      * sfr_psd_shape(w, np.exp(lt), k, np.exp(ltg))) - y

    best = None
    for t0 in (2.0, 6.0, 20.0):
        for tg0 in (2.0, 10.0):
            q0 = np.array([y[0], np.log(t0), 1.0, np.log(tg0)])
            res = least_squares(resid, q0, max_nfev=4000)
            if best is None or res.cost < best.cost:
                best = res
    la, lt, zk, ltg = best.x
    return dict(tau_sys=float(np.exp(lt)),
                gov_share=float(1.0 / (1.0 + np.exp(-zk))),
                t_g=float(np.exp(ltg)), amp=float(np.exp(la)),
                cost=float(best.cost), n=int(m.sum()))


# --------------------------------------------------------------------------- #
# analysis + verdicts
# --------------------------------------------------------------------------- #
def analyze(cache):
    reg = cache["ordering"]
    out = dict(ordering=reg)
    # observed inertia span vs the registered band, in decades
    band_dec = float(np.log10((1.0 / BANDS["b2_30"][0])
                              / (1.0 / BANDS["b2_30"][1])))
    span = {}
    for era in ("preDC", "DC"):
        h = reg[f"{era}|b2_30|all"]["H_med"]
        span[era] = dict(H_lo=h[0], H_hi=h[-1],
                         decades=float(np.log10(h[-1] / h[0])),
                         frac_of_band=float(np.log10(h[-1] / h[0])
                                            / band_dec))
    out["range_vs_band"] = dict(band_decades=band_dec, **span)
    # NR42 c* ~ 0.30 readout -> predicted peak location for plausible
    # all-synchronous scales (top of the observed preDC distribution)
    out["predicted_peak_H"] = {str(h0): 0.30 * h0 for h0 in (350, 400, 450)}
    # SFR shape fits on pooled tercile spectra
    fits = {}
    for era in ("preDC", "DC"):
        fits[era] = dict(
            H_med=[cache["spectra"][f"{era}_T{j}"]["H_med"]
                   for j in range(3)],
            tau_sys=[], gov_share=[], t_g=[], cost=[])
        for j in range(3):
            s = cache["spectra"][f"{era}_T{j}"]
            fr = fit_sfr(s["freqs"], s["P"])
            for key in ("tau_sys", "gov_share", "t_g", "cost"):
                fits[era][key].append(fr[key])
    out["sfr_fit"] = fits
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    reg = out["ordering"]
    b = "b2_30"
    combos = [(e, s) for e in ("preDC", "DC")
              for s in ("all", "day", "night", "demandnorm")]
    markov_rejected = (
        all(reg[f"{e}|{b}|{s}"]["argmax_bin"] != N_OCT - 1
            for e, s in combos)
        and all(reg[f"{e}|{b}|{s}"]["rho"] <= 0.05 for e, s in combos))
    pooled = reg[f"preDC|{b}|all"]
    day = reg[f"preDC|{b}|day"]
    dnorm = reg[f"preDC|{b}|demandnorm"]
    window_side = (pooled["rho"] < 0.0 and pooled["p"] < 1e-6
                   and day["rho"] <= -0.5 and dnorm["rho"] <= -0.5)
    interior_resolved = (
        1 <= day["argmax_bin"] <= N_OCT - 2
        and 1 <= dnorm["argmax_bin"] <= N_OCT - 2
        and day["p_interior"] >= 0.9 and dnorm["p_interior"] >= 0.9)
    under_spanned = (out["range_vs_band"]["preDC"]["frac_of_band"] <= 0.5
                     and out["range_vs_band"]["DC"]["frac_of_band"] <= 0.5)
    band_specific = (reg["preDC|b60_300|all"]["rho"] > 0.0
                     and reg["DC|b60_300|all"]["rho"] > 0.0
                     and reg["DC|b60_300|all"]["argmax_bin"] == N_OCT - 1)
    dc = reg[f"DC|{b}|all"]
    dc_flattened = (abs(dc["rho"]) < 0.10 and dc["rho"] > pooled["rho"]
                    and (reg[f"DC|{b}|demandnorm"]["rho"]
                         > reg[f"preDC|{b}|demandnorm"]["rho"]))
    taus = out["sfr_fit"]["preDC"]["tau_sys"]
    knee_not_inertia_readout = not (taus[0] < taus[1] < taus[2])
    return dict(
        markovian_null_rejected=bool(markov_rejected),
        window_side_ordering_confirmed=bool(window_side),
        interior_peak_resolved_in_range=bool(interior_resolved),
        inertia_range_under_spans_band=bool(under_spanned),
        band_specificity=bool(band_specific),
        dc_era_flattening=bool(dc_flattened),
        ambient_knee_not_inertia_readout=bool(knee_not_inertia_readout),
        reading=(
            "Registered NR42 gate on real GB archives: the Markovian "
            "(quasi-static control) ordering -- maximum observability at "
            "the highest-inertia bin -- is rejected in the [2,30] s band "
            "in every stratum of both eras; variance rises as inertia "
            "falls (preDC day/demand-normalized rho ~ -0.6), the "
            "window-side ordering, toward a predicted peak at/below the "
            "observed low edge.  The interior peak itself is unresolved "
            "exactly as the criterion requires: the observed inertia span "
            "(0.37-0.40 decades) under-spans the 1.18-decade band.  "
            "Out-of-band (60-300 s) the ordering inverts, so the in-band "
            "signal is not a variance artifact.  Post-DC the window "
            "flattens (converter-fast response raises in-band beta "
            "without storage).  The ambient PSD knee is tau_sys = M/beta "
            "with beta co-varying with demand -- NOT a calibration-free "
            "inertia readout; the event-conditioned readout stays gated."),
    )


def run(write=True):
    cache = load_cache()
    out = analyze(cache)
    res = dict(description=__doc__.splitlines()[0], meta=cache["meta"], **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    reg = res["ordering"]
    print("NR42 gate EXECUTED -- GB 1-s frequency archive vs Outturn Inertia")
    for era in ("preDC", "DC"):
        for st in ("all", "day", "night", "demandnorm"):
            r = reg[f"{era}|b2_30|{st}"]
            print(f"  [2,30]s {era:5s} {st:10s}: rho={r['rho']:+.2f} "
                  f"argmax bin {r['argmax_bin']}/7 "
                  f"(H={r['H_med'][r['argmax_bin']]:.0f}) "
                  f"p_int={r['p_interior']:.2f} n={r['n_halfhours']}")
    for era in ("preDC", "DC"):
        r = reg[f"{era}|b60_300|all"]
        print(f"  60-300s {era:5s} all       : rho={r['rho']:+.2f} "
              f"argmax bin {r['argmax_bin']}/7  (out-of-band control)")
    rv = res["range_vs_band"]
    print(f"  inertia span vs band      : preDC "
          f"{rv['preDC']['decades']:.2f} dec, DC {rv['DC']['decades']:.2f} "
          f"dec vs band {rv['band_decades']:.2f} dec "
          f"-> under-spanned={v['inertia_range_under_spans_band']}")
    ft = res["sfr_fit"]["preDC"]
    print(f"  SFR knee fit (preDC)      : tau_sys = "
          f"{['%.1f' % t for t in ft['tau_sys']]} s across H_med "
          f"{['%.0f' % h for h in ft['H_med']]} GVA.s "
          f"-> readout confounded={v['ambient_knee_not_inertia_readout']}")
    print(f"  verdicts: markov_rejected={v['markovian_null_rejected']} "
          f"window_side={v['window_side_ordering_confirmed']} "
          f"interior_resolved={v['interior_peak_resolved_in_range']} "
          f"band_specific={v['band_specificity']} "
          f"dc_flattening={v['dc_era_flattening']}")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
