"""Run every pressure/temperature test and emit figures + a Markdown report.

Usage:
    python run_analysis.py --data-dir <dir with NEON .h5> --out-dir figures

The script is deterministic: all numbers written into ``REPORT.md`` come straight
from the computations in :mod:`neon_pt.analysis`.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neon_pt import analysis as A
from neon_pt.loader import load_dataframe


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_diurnal(df, out_dir):
    cols = ["sw_in_wm2", "temp_air_c", "pres_kpa", "H_sensible_wm2", "ustar_ms", "wind_w_vari"]
    comp = A.diurnal_composite(df, cols)
    titles = {
        "sw_in_wm2": "Incoming shortwave (W m$^{-2}$)",
        "temp_air_c": "Air temperature (deg C)",
        "pres_kpa": "Barometric pressure (kPa)",
        "H_sensible_wm2": "Sensible heat flux H (W m$^{-2}$)",
        "ustar_ms": "Friction velocity u* (m s$^{-1}$)",
        "wind_w_vari": "Vertical velocity variance (m$^2$ s$^{-2}$)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, c in zip(axes.ravel(), cols):
        ax.plot(comp.index, comp[c], color="#1f77b4", lw=2)
        ax.set_title(titles[c])
        # Only mark zero where the sign of the quantity is meaningful (fluxes).
        if comp[c].min() < 0 < comp[c].max():
            ax.axhline(0, color="0.7", lw=0.8)
        else:
            ax.margins(y=0.15)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("local hour (PST)")
    fig.suptitle("Diurnal composites - WREF, Jan 2020 (the solar-driven cycle)", fontsize=13)
    return comp, _save(fig, out_dir, "01_diurnal_composites.png")


def fig_leadlag(df, out_dir):
    lh1, c1, b1 = A.lagged_xcorr(df["sw_in_wm2"], df["temp_air_c"])
    lh2, c2, b2 = A.lagged_xcorr(df["sw_in_wm2"], df["H_sensible_wm2"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lh1, c1, "-o", ms=3, label=f"SW->Temp (peak {b1:+.1f} h)")
    ax.plot(lh2, c2, "-s", ms=3, label=f"SW->Sensible heat (peak {b2:+.1f} h)")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("lag (hours); positive = response lags solar forcing")
    ax.set_ylabel("cross-correlation")
    ax.set_title("Solar radiation leads temperature and heat flux")
    ax.legend()
    ax.grid(alpha=0.3)
    return (b1, b2), _save(fig, out_dir, "02_lead_lag.png")


def fig_spectra(df, out_dir):
    periods = np.linspace(3.0, 480.0, 4000)
    sp_t = A.lomb_scargle_spectrum(df["temp_air_c"], periods)
    sp_p = A.lomb_scargle_spectrum(df["pres_kpa"], periods)

    t_diurnal = A.find_band_peak(sp_t, 20, 28)
    t_semidi = A.find_band_peak(sp_t, 10.5, 13.5)
    p_diurnal = A.find_band_peak(sp_p, 20, 28)
    p_semidi = A.find_band_peak(sp_p, 10.5, 13.5)
    # Single strongest peak across the synoptic-to-diurnal range.
    t_dominant = A.dominant_period(sp_t, 6, 240)
    p_dominant = A.dominant_period(sp_p, 6, 240)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, sp, name, color in [
        (axes[0], sp_t, "Air temperature", "#d62728"),
        (axes[1], sp_p, "Barometric pressure", "#1f77b4"),
    ]:
        ax.plot(sp.periods_h, sp.power, color=color, lw=1.3)
        ax.axvline(24, color="0.5", ls="--", lw=1, label="24 h (diurnal)")
        ax.axvline(12, color="0.5", ls=":", lw=1, label="12 h (semidiurnal)")
        ax.set_xlim(4, 60)
        ax.set_ylabel("normalised power")
        ax.set_title(f"{name} spectrum")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("period (hours)")
    fig.suptitle("Two clocks: temperature is diurnal (24 h); pressure carries a semidiurnal (12 h) tide", fontsize=12)
    info = {
        "t_diurnal": t_diurnal,
        "t_semidi": t_semidi,
        "p_diurnal": p_diurnal,
        "p_semidi": p_semidi,
        "t_dominant": t_dominant,
        "p_dominant": p_dominant,
    }
    return info, _save(fig, out_dir, "03_spectra_two_clocks.png")


def fig_coupling(df, out_dir, coup):
    sub = df[["temp_air_k", "pres_kpa"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(sub["temp_air_k"], sub["pres_kpa"], s=8, alpha=0.4, color="#2ca02c")
    xs = np.array([sub["temp_air_k"].min(), sub["temp_air_k"].max()])
    axes[0].plot(xs, coup.slope_kpa_per_k * xs + (sub["pres_kpa"].mean() - coup.slope_kpa_per_k * sub["temp_air_k"].mean()),
                 "k-", lw=1.5, label=f"fit: {coup.slope_kpa_per_k:+.3f} kPa/K, R^2={coup.r_squared:.2f}")
    # Ideal-gas (constant-density) expectation anchored at the data mean.
    p0, t0 = sub["pres_kpa"].mean(), sub["temp_air_k"].mean()
    axes[0].plot(xs, p0 + coup.ideal_gas_slope * (xs - t0), "r--", lw=1.5,
                 label=f"kinetic-theory lock: {coup.ideal_gas_slope:+.3f} kPa/K")
    axes[0].set_xlabel("temperature (K)")
    axes[0].set_ylabel("pressure (kPa)")
    axes[0].set_title("Bulk P vs T are nearly decoupled\n(not the steep kinetic-theory line)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Time series, normalised, to show they drift on different schedules.
    z = lambda s: (s - s.mean()) / s.std()
    axes[1].plot(df.index, z(df["temp_air_c"]), color="#d62728", lw=0.9, label="temperature (z)")
    axes[1].plot(df.index, z(df["pres_kpa"]), color="#1f77b4", lw=0.9, label="pressure (z)")
    axes[1].set_title(f"Normalised series (Pearson r = {coup.pearson_r:+.2f})")
    axes[1].set_xlabel("date (UTC)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.autofmt_xdate()
    return _save(fig, out_dir, "04_pressure_temperature_coupling.png")


def fig_shear(df, out_dir, shear):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    d = df[["wind_speed_ms", "ustar_ms"]].dropna()
    axes[0].scatter(d["wind_speed_ms"], d["ustar_ms"], s=8, alpha=0.4, color="#9467bd")
    axes[0].set_xlabel("horizontal wind speed (m s$^{-1}$)")
    axes[0].set_ylabel("friction velocity u* (m s$^{-1}$)")
    axes[0].set_title(f"Shear -> momentum flux (r = {shear.r_ustar_wind:.2f})")
    axes[0].grid(alpha=0.3)

    d2 = df[["H_sensible_wm2", "wind_w_vari"]].dropna()
    axes[1].scatter(d2["H_sensible_wm2"], d2["wind_w_vari"], s=8, alpha=0.4, color="#ff7f0e")
    axes[1].set_xlabel("sensible heat flux H (W m$^{-2}$)")
    axes[1].set_ylabel("vertical velocity variance (m$^2$ s$^{-2}$)")
    axes[1].set_title(f"Buoyancy -> turbulence (r = {shear.r_turb_heat:.2f})")
    axes[1].grid(alpha=0.3)
    return _save(fig, out_dir, "05_shear_turbulence.png")


def write_report(path, meta, df, comp, lags, spec, coup, shear, fig_paths):
    n = len(df)
    valid_p = int(df["pres_kpa"].notna().sum())
    valid_t = int(df["temp_air_c"].notna().sum())
    t_peak = A.peak_hour(comp["temp_air_c"])
    sw_peak = A.peak_hour(comp["sw_in_wm2"])
    p_peak = A.peak_hour(comp["pres_kpa"])
    p_trough = A.trough_hour(comp["pres_kpa"])
    h_peak = A.peak_hour(comp["H_sensible_wm2"])

    def fmt(pp):
        return f"{pp[0]:.1f} h (power {pp[1]:.3f})"

    lines = []
    w = lines.append
    w("# Pressure and Temperature: two clocks, not one")
    w("")
    w(f"Dataset: **NEON bundled eddy covariance (DP4.00200)**, site **{meta.site}** "
      f"({meta.ecosystem}), {meta.lat:.3f}, {meta.lon:.3f}, elev {meta.elevation_m:.0f} m, "
      f"canopy {meta.canopy_height_m:.0f} m. Month: January 2020, 30-min resolution "
      f"({n} intervals; {valid_p} valid pressure, {valid_t} valid temperature). "
      f"Local time = UTC{meta.utc_to_local_hours:+d} ({meta.time_zone}).")
    w("")
    w("This report tests the physical claims in the hypothesis against real "
      "surface-atmosphere measurements. Each number below is computed by "
      "`run_analysis.py`; figures are in `figures/`.")
    w("")

    w("## 1. The solar source drives the heating cycle")
    w("")
    w(f"- Incoming shortwave radiation peaks at **{sw_peak:.1f} h** local; air "
      f"temperature peaks later at **{t_peak:.1f} h**.")
    w(f"- Cross-correlation: solar radiation leads temperature by **{lags[0]:+.1f} h** "
      f"and leads the sensible heat flux by **{lags[1]:+.1f} h**.")
    w(f"- Sensible heat flux H peaks at **{h_peak:.1f} h** and is positive (upward, "
      "buoyant) by day, negative (downward) at night - exactly the "
      "\"hot air floats up\" mechanism.")
    w("")
    w(f"![diurnal]({os.path.basename(fig_paths['diurnal'])})")
    w(f"![leadlag]({os.path.basename(fig_paths['leadlag'])})")
    w("")
    w("**Verdict:** supported. The sun is the clock that sets temperature, and "
      "daytime heating does drive buoyant upward heat transport.")
    w("")

    w("## 2. The central claim: two different clocks for P and T")
    w("")
    w("Spectral (Lomb-Scargle) analysis of the full month:")
    w("")
    w("| signal | strongest diurnal (20-28 h) | strongest semidiurnal (10.5-13.5 h) |")
    w("|---|---|---|")
    w(f"| temperature | {fmt(spec['t_diurnal'])} | {fmt(spec['t_semidi'])} |")
    w(f"| pressure | {fmt(spec['p_diurnal'])} | {fmt(spec['p_semidi'])} |")
    w("")
    ratio_t = spec["t_semidi"][1] / spec["t_diurnal"][1] if spec["t_diurnal"][1] else float("nan")
    ratio_p = spec["p_semidi"][1] / spec["p_diurnal"][1] if spec["p_diurnal"][1] else float("nan")
    w(f"- Temperature is overwhelmingly **diurnal**: its 12 h power is only "
      f"**{ratio_t*100:.0f}%** of its 24 h power.")
    w(f"- Pressure carries a much stronger **semidiurnal (12 h) tide**: its 12 h power "
      f"is **{ratio_p*100:.0f}%** of its 24 h power - a qualitatively different "
      "spectral fingerprint.")
    w(f"- The single strongest period is **{spec['t_dominant']:.0f} h** for temperature "
      f"(the daily cycle) versus **{spec['p_dominant']:.0f} h** for pressure: "
      "pressure's dominant variability lives on the multi-day **synoptic** scale of "
      "passing weather systems, not the daily solar cycle.")
    w(f"- In the diurnal composite, pressure's daily extrema (min near "
      f"**{p_trough:.1f} h**, max near **{p_peak:.1f} h**) are offset from the "
      f"temperature peak at **{t_peak:.1f} h**.")
    w("")
    w(f"![spectra]({os.path.basename(fig_paths['spectra'])})")
    w("")
    w("**Verdict:** supported. Temperature and pressure are driven on different "
      "periodicities. The 12 h atmospheric (thermal) tide in pressure is a real, "
      "well-documented phenomenon and is the clearest evidence for the \"two "
      "clocks\" intuition.")
    w("")

    w("## 3. Is kinetic theory wrong to make P and T \"equal\"?")
    w("")
    w("Kinetic theory / the ideal gas law for a fixed parcel gives "
      "P = rho*R*T, i.e. at constant density P should rise steeply and almost "
      "perfectly with T.")
    w("")
    w(f"- Bulk fit of pressure on temperature: slope **{coup.slope_kpa_per_k:+.3f} "
      f"kPa/K**, **R^2 = {coup.r_squared:.2f}**, Pearson r = "
      f"**{coup.pearson_r:+.2f}** (p = {coup.pearson_p:.1e}).")
    w(f"- The constant-density (kinetic-theory) lock would require a slope of "
      f"**{coup.ideal_gas_slope:+.3f} kPa/K** - far steeper and of the opposite "
      "tightness to what the atmosphere shows.")
    w(f"- After removing the slow synoptic trend, fast P and T fluctuations "
      f"correlate at only r = **{coup.detrended_r:+.2f}**.")
    w(f"- What actually varies is air density: its coefficient of variation over "
      f"the month is **{coup.density_cv_pct:.1f}%**, absorbing the P-T mismatch.")
    w("")
    w(f"![coupling]({os.path.basename(fig_paths['coupling'])})")
    w("")
    w("**Verdict (nuanced):** Kinetic theory is *not* wrong - P = rho*R*T holds "
      "locally for every parcel. But it is wrong to read it as \"bulk P and T move "
      "together\": in the open atmosphere density is free to change, so P and T "
      "are effectively **decoupled** on the daily scale. The data back the "
      "hypothesis's spirit - P and T are not one locked quantity - while pointing "
      "to changing density (not a failure of the gas law) as the reason.")
    w("")

    w("## 4. Uneven streams shear against each other")
    w("")
    w(f"- Momentum flux tracks mechanical shear: u* vs wind speed r = "
      f"**{shear.r_ustar_wind:.2f}**.")
    w(f"- Turbulence (vertical velocity variance) rises mainly with **mechanical "
      f"shear** (r = **{shear.r_turb_wind:.2f}** vs wind speed); its link to daytime "
      f"buoyancy is weak in this winter month (r = **{shear.r_turb_heat:.2f}** vs "
      "upward heat flux), as expected when the sun is low and heat fluxes are small.")
    w(f"- Wind arrives from a wide spread of directions (circular std "
      f"**{shear.wind_dir_std_deg:.0f} deg**) - the \"uneven streams\" picture.")
    w("")
    w(f"![shear]({os.path.basename(fig_paths['shear'])})")
    w("")
    w("**Verdict:** supported in analogy, with a winter caveat. The momentum flux "
      "and turbulence that mix and distort the flow are real and measurable; in "
      "January they are dominated by mechanical wind shear rather than buoyancy. "
      "(The literal Marangoni effect is a liquid-surface-tension phenomenon; the "
      "atmospheric analogue here is shear- and buoyancy-driven turbulent stress.)")
    w("")

    w("## Summary")
    w("")
    w("| claim | result |")
    w("|---|---|")
    w("| Sun drives uneven heating and buoyant uplift | supported |")
    w("| Temperature and pressure run on two different clocks | supported (12 h pressure tide) |")
    w("| Bulk P and T are not locked \"equal\" | supported (R^2 low; density varies) |")
    w("| Kinetic theory is literally \"wrong\" | not quite - it holds locally; bulk decoupling is via density |")
    w("| Uneven sheared streams create turbulent stress | supported (analogue of Marangoni shear) |")
    w("")
    w("_Generated by `run_analysis.py` from the NEON DP4.00200 HDF5 bundle._")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="dir containing the NEON .h5 (searched recursively)")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--report", default="REPORT.md")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, meta = load_dataframe(args.data_dir)
    df = A.apply_qc(df)
    print(f"Loaded {len(df)} intervals for site {meta.site}")

    # fig_diurnal computes a 6-column diurnal composite; reuse it for the
    # report (it contains all 4 columns write_report needs) instead of
    # recomputing a separate 4-column composite.
    comp, p_diurnal = fig_diurnal(df, args.out_dir)
    lags, p_leadlag = fig_leadlag(df, args.out_dir)
    spec, p_spectra = fig_spectra(df, args.out_dir)
    coup = A.coupling_tests(df)
    p_coupling = fig_coupling(df, args.out_dir, coup)
    shear = A.shear_tests(df)
    p_shear = fig_shear(df, args.out_dir, shear)

    fig_paths = {
        "diurnal": p_diurnal,
        "leadlag": p_leadlag,
        "spectra": p_spectra,
        "coupling": p_coupling,
        "shear": p_shear,
    }
    write_report(args.report, meta, df, comp, lags, spec, coup, shear, fig_paths)
    print(f"Wrote {args.report} and figures to {args.out_dir}/")
    print(f"  P-T Pearson r = {coup.pearson_r:+.3f}, R^2 = {coup.r_squared:.3f}")
    print(f"  temp 24h/12h peaks: {spec['t_diurnal'][0]:.1f}h / {spec['t_semidi'][0]:.1f}h")
    print(f"  pres 24h/12h peaks: {spec['p_diurnal'][0]:.1f}h / {spec['p_semidi'][0]:.1f}h")


if __name__ == "__main__":
    main()
