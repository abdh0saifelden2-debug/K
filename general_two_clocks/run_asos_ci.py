"""P0-R2/R3/R4 -- extended ASOS ladder: bootstrap CIs, Haurwitz/Dai-Wang
comparison, and band coherence.

Extends the committed 3-station analysis (run_asos.py) to an 8-station
latitude ladder (24.6-47.4 N, coastal + interior) and adds what a referee
will ask for:

  * daily moving-block bootstrap CIs for S1_T, S2_P, and the bulk P-T
    correlation r (P0-R2);
  * a quantitative comparison of the measured S2(p) amplitudes against the
    classical migrating-tide climatology A_H = 1.16 cos^3(phi) hPa
    (Haurwitz 1956; Dai & Wang 1999) -- agreement in trend validates the
    protocol on a known planetary signal (P0-R3);
  * magnitude-squared coherence between P and T at the S1 and S2 bands with
    a phase-randomized surrogate 95% significance level -- the band-resolved
    replacement for the blunt bulk r (P0-R4).

Artifacts: figures/09b_asos_ci.json, figures/11b_tide_vs_latitude_ci.png
Data:      data_asos/<ID>_2020Q1.csv via asos/fetch_iem.py (IEM 1-min archive)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)

from asos.loader import load_station                       # noqa: E402
from run_asos import harmonic_amplitudes                   # noqa: E402

N_BOOT = 500
SEED = 0


def haurwitz_s2(lat_deg):
    """Migrating-S2 climatology A = 1.16 cos^3(phi) hPa (Haurwitz 1956)."""
    return 1.16 * np.cos(np.deg2rad(lat_deg)) ** 3


def block_bootstrap_station(df, n_boot=N_BOOT, seed=SEED):
    """Daily moving-block bootstrap of (S1_T, S2_P, r).

    Whole UTC days are resampled with replacement; each day keeps its own
    timestamps, so the 24h/12h design matrix (a pure function of time of day)
    is consistent under resampling.
    """
    rng = np.random.default_rng(seed)
    days = df.index.normalize()
    uniq = days.unique()
    by_day = {d: df.loc[days == d] for d in uniq}
    s1s, s2s, rs = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        boot = pd.concat([by_day[uniq[i]] for i in pick])
        s1t, _ = harmonic_amplitudes(boot["temp_c"])
        _, s2p = harmonic_amplitudes(boot["pres_hpa"])
        sub = boot.dropna(subset=["temp_c", "pres_hpa"])
        rs.append(float(sub["temp_c"].corr(sub["pres_hpa"])))
        s1s.append(s1t)
        s2s.append(s2p)
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return dict(S1_T_ci95=ci(s1s), S2_P_ci95=ci(s2s), r_ci95=ci(rs))


def band_coherence(df, seed=SEED, n_surr=200):
    """Welch magnitude-squared coherence between hourly P and T at the S1/S2
    bands, with a phase-randomized surrogate 95% significance level."""
    from scipy import signal

    h = df[["temp_c", "pres_hpa"]].resample("1h").mean().interpolate(limit=3)
    h = h.dropna()
    x = h["temp_c"].to_numpy()
    y = h["pres_hpa"].to_numpy()
    fs = 1.0                                   # per hour
    nperseg = 512
    f, C = signal.coherence(x, y, fs=fs, nperseg=nperseg)

    def band(fc, width=0.15):
        m = (f > fc * (1 - width)) & (f < fc * (1 + width))
        return float(C[m].max()) if m.any() else float("nan")

    c_s1 = band(1.0 / 24.0)
    c_s2 = band(1.0 / 12.0)
    # off-line sub-diurnal band: 9-11 h + 13-20 h, EXCLUDING the solar lines
    # (at the lines both fields are phase-locked to the sun, so coherence is
    # trivially high; off the lines is where the two-clocks decoupling shows)
    m_off = (((f > 1 / 20.0) & (f < 1 / 13.0)) |
             ((f > 1 / 11.0) & (f < 1 / 9.0)))
    c_off = float(np.median(C[m_off]))

    # surrogate significance: phase-randomize y, keep x
    rng = np.random.default_rng(seed)
    Y = np.fft.rfft(y - y.mean())
    surr_s1, surr_s2, surr_off = [], [], []
    for _ in range(n_surr):
        ph = np.exp(2j * np.pi * rng.random(Y.size))
        ph[0] = 1.0
        ys = np.fft.irfft(np.abs(Y) * ph, n=len(y))
        _, Cs = signal.coherence(x, ys, fs=fs, nperseg=nperseg)
        m1 = (f > (1 / 24) * 0.85) & (f < (1 / 24) * 1.15)
        m2 = (f > (1 / 12) * 0.85) & (f < (1 / 12) * 1.15)
        surr_s1.append(float(Cs[m1].max()))
        surr_s2.append(float(Cs[m2].max()))
        surr_off.append(float(np.median(Cs[m_off])))
    return dict(coh_S1=c_s1, coh_S2=c_s2, coh_offline=c_off,
                coh_S1_surr95=float(np.percentile(surr_s1, 95)),
                coh_S2_surr95=float(np.percentile(surr_s2, 95)),
                coh_offline_surr95=float(np.percentile(surr_off, 95)),
                nperseg=nperseg, n_hours=len(h))


def run(data_dir=None):
    data_dir = Path(data_dir or os.path.join(HERE, "data_asos"))
    csvs = sorted(data_dir.glob("*_2020Q1.csv"))
    if len(csvs) < 6:
        raise FileNotFoundError(
            f"expected >=6 station CSVs in {data_dir}; run asos/fetch_iem.py")
    stations = [load_station(p) for p in csvs]
    stations.sort(key=lambda s: s.lat)

    rows = []
    for st in stations:
        df = st.df
        s1t, _ = harmonic_amplitudes(df["temp_c"])
        _, s2p = harmonic_amplitudes(df["pres_hpa"])
        sub = df.dropna(subset=["temp_c", "pres_hpa"])
        r = float(sub["temp_c"].corr(sub["pres_hpa"]))
        print(f"  {st.station_id} ({st.lat:.1f}N): S1_T={s1t:.2f}C "
              f"S2_P={s2p:.3f}hPa r={r:+.3f} ... bootstrapping")
        ci = block_bootstrap_station(df)
        coh = band_coherence(df)
        ah = float(haurwitz_s2(st.lat))
        rows.append(dict(
            station=st.station_id, name=st.station_name,
            lat=float(st.lat), lon=float(st.lon),
            n_rows=int(len(df)),
            S1_T=float(s1t), S2_P=float(s2p), r=r,
            **ci, **coh,
            haurwitz_S2=ah, S2_over_haurwitz=float(s2p / ah)))

    lats = np.array([r["lat"] for r in rows])
    s2s = np.array([r["S2_P"] for r in rows])
    ahs = np.array([r["haurwitz_S2"] for r in rows])
    # cos^3 trend check: correlation of measured S2 with the climatology
    trend_r = float(np.corrcoef(s2s, ahs)[0, 1])
    ratio = s2s / ahs
    out = dict(
        what=("8-station ASOS 1-min ladder (2020Q1): S1_T/S2_P/r with daily "
              "block-bootstrap 95% CIs; Haurwitz 1956 / Dai & Wang 1999 "
              "migrating-S2 comparison; band coherence with surrogate "
              "significance"),
        source="IEM ASOS 1-min archive via asos/fetch_iem.py",
        n_boot=N_BOOT,
        stations=rows,
        s2_vs_haurwitz=dict(
            trend_corr=trend_r,
            ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
            ratio_median=float(np.median(ratio)),
            note=("station totals sit above the migrating-only formula, as "
                  "expected once non-migrating components are included "
                  "(Dai & Wang 1999); the cos^3 latitude decline is the "
                  "protocol validation")),
    )
    return out


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["stations"]
    lats = [r["lat"] for r in rows]
    s2 = [r["S2_P"] for r in rows]
    lo = [r["S2_P"] - r["S2_P_ci95"][0] for r in rows]
    hi = [r["S2_P_ci95"][1] - r["S2_P"] for r in rows]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].errorbar(lats, s2, yerr=[lo, hi], fmt="ko", ms=5, capsize=3,
                    label="measured S2(p), 95% CI")
    ll = np.linspace(20, 50, 100)
    axs[0].plot(ll, 1.16 * np.cos(np.deg2rad(ll)) ** 3, "b--", lw=1,
                label=r"Haurwitz: $1.16\cos^3\varphi$ (migrating)")
    for r in rows:
        axs[0].annotate(r["station"], (r["lat"], r["S2_P"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    axs[0].set_xlabel("latitude (deg N)")
    axs[0].set_ylabel("S2 pressure amplitude (hPa)")
    axs[0].set_title("semidiurnal tide vs latitude (8 stations, 2020Q1)")
    axs[0].legend(fontsize=8)

    x = np.arange(len(rows))
    axs[1].bar(x - 0.2, [r["coh_S1"] for r in rows], 0.4, label="coherence @ S1")
    axs[1].bar(x + 0.2, [r["coh_S2"] for r in rows], 0.4, label="coherence @ S2")
    axs[1].plot(x, [r["coh_S1_surr95"] for r in rows], "k_", ms=14,
                label="surrogate 95%")
    axs[1].set_xticks(x, [r["station"] for r in rows], fontsize=7)
    axs[1].set_ylabel(r"$\gamma^2$(P, T)")
    axs[1].set_title("band coherence (hourly, Welch)")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    res = run()
    out_dir = Path(HERE) / "figures"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "09b_asos_ci.json", "w") as fh:
        json.dump(res, fh, indent=2)
    make_figure(res, out_dir / "11b_tide_vs_latitude_ci.png")
    print("\nstation | lat | S2_P [95% CI] | /Haurwitz | r [95% CI] | coh S2 (surr95) | coh off-line (surr95)")
    for r in res["stations"]:
        print(f"  {r['station']} | {r['lat']:.1f} | {r['S2_P']:.3f} "
              f"[{r['S2_P_ci95'][0]:.3f},{r['S2_P_ci95'][1]:.3f}] | "
              f"{r['S2_over_haurwitz']:.2f} | {r['r']:+.3f} "
              f"[{r['r_ci95'][0]:+.3f},{r['r_ci95'][1]:+.3f}] | "
              f"{r['coh_S2']:.2f} ({r['coh_S2_surr95']:.2f}) | "
              f"{r['coh_offline']:.2f} ({r['coh_offline_surr95']:.2f})")
    t = res["s2_vs_haurwitz"]
    print(f"\nS2 vs Haurwitz cos^3: trend corr={t['trend_corr']:.3f}, "
          f"ratio median={t['ratio_median']:.2f} "
          f"[{t['ratio_min']:.2f}, {t['ratio_max']:.2f}]")
    return res


if __name__ == "__main__":
    main()
