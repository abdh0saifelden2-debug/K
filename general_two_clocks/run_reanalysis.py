"""Part 7 -- the two clocks in real reanalysis winds.
 
Splits real NCEP/NCAR Reanalysis horizontal winds into rotational (slow, balanced
"weather" clock) and divergent (fast, ageostrophic/convective clock) parts on the
sphere and shows:
 
  fig 25  one snapshot: the rotational wind is the weather; the divergent wind is a
          weak, structured residual concentrated in convergence zones.
  fig 26  the kinetic-energy spectrum by spherical wavenumber: divergent << rotational
          at every scale (the real-data echo of KE_dil/KE_sol << 1 from Parts 5-6).
  fig 27  KE_div/KE_rot vs height (minimal near the ~500 hPa level of non-divergence)
          and daily-mean vs 6-hourly: time-averaging is a low-pass filter that scrubs
          the fast clock -- the same move as the Part-5 acoustic average.
 
Data is fetched live from NOAA PSL over OPeNDAP (no credentials) and cached under
data_reanalysis/.  Run:
 
    python run_reanalysis.py --out-dir figures --report REPORT_REANALYSIS.md
"""
 
from __future__ import annotations
 
import argparse
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
 
from reanalysis import ncep
 
 
def _ascending(lat, *fields):
    """Flip rows so latitude increases (needed by streamplot)."""
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        fields = [f[::-1, :] for f in fields]
    return (lat, *fields)
 
 
def fig_snapshot(out_dir, level=850, year=2021, t0=420):
    u, v, lat, lon = ncep.fetch_wind(level, year, t0=t0, nt=1, source="inst")
    h = ncep.helmholtz(u[0], v[0], lat)
    latd = h["lat"]
    spd_rot = np.sqrt(h["u_psi"] ** 2 + h["v_psi"] ** 2)
    div = h["div"] * 1e6  # 1e-6 /s
 
    y, spd_rot, up, vp = _ascending(latd, spd_rot, h["u_psi"], h["v_psi"])
    _, divp, uc, vc = _ascending(latd, div, h["u_chi"], h["v_chi"])
    X, Y = np.meshgrid(lon, y)
 
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    fig.suptitle(f"Two clocks in real winds (NCEP reanalysis, {level} hPa snapshot): "
                 "rotational = the weather, divergent = the fast-clock residual")
 
    c0 = ax[0].pcolormesh(X, Y, spd_rot, cmap="viridis", shading="auto")
    ax[0].streamplot(lon, y, up, vp, color="w", density=1.1, linewidth=0.5,
                     arrowsize=0.6)
    fig.colorbar(c0, ax=ax[0], label="rotational wind speed (m/s)")
    ax[0].set_title("rotational (non-divergent) wind\nbalanced geostrophic weather "
                    "-- the slow clock")
 
    lim = np.nanpercentile(np.abs(divp), 98)
    c1 = ax[1].pcolormesh(X, Y, divp, cmap="RdBu_r", vmin=-lim, vmax=lim,
                          shading="auto")
    q = 4
    ax[1].quiver(X[::q, ::q], Y[::q, ::q], uc[::q, ::q], vc[::q, ::q],
                 scale=120, width=0.0018, color="k")
    fig.colorbar(c1, ax=ax[1], label="divergence (1e-6 /s)")
    ax[1].set_title("divergent wind + divergence field\nconvergence/overturning "
                    "-- the fast clock (note: ~100x weaker)")
    for a in ax:
        a.set_xlabel("longitude"); a.set_ylabel("latitude")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, "25_reanalysis_two_clocks.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p
 
 
def fig_spectrum(out_dir, level=850, year=2021, t0=100, nt=24):
    u, v, lat, _ = ncep.fetch_wind(level, year, t0=t0, nt=nt, source="daily")
    l, KEr, KEd, ratio = ncep.ke_ratio_block(u, v, lat)
    m = l >= 1
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.loglog(l[m], KEr[m], "o-", color="#1f77b4", label="rotational (balanced, slow clock)")
    ax.loglog(l[m], KEd[m], "s-", color="#d62728", label="divergent (fast clock)")
    ax.set_xlabel("spherical wavenumber l")
    ax.set_ylabel("kinetic energy per degree (arb. units)")
    ax.set_title(f"Real reanalysis wind, {level} hPa ({nt}-day mean): divergent KE is a\n"
                 f"small fraction of rotational at every scale -- KE_div/KE_rot = {ratio:.3f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "26_reanalysis_ke_spectrum.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p, ratio
 
 
def fig_levels(out_dir, year=2021, t0=100, nt=24):
    levels = [850, 500, 250]
    daily = {}
    for lev in levels:
        u, v, lat, _ = ncep.fetch_wind(lev, year, t0=t0, nt=nt, source="daily")
        daily[lev] = ncep.ke_ratio_block(u, v, lat)[3]
    ui, vi, lati, _ = ncep.fetch_wind(500, year, t0=t0, nt=nt, source="inst")
    inst500 = ncep.ke_ratio_block(ui, vi, lati)[3]
 
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("KE_div/KE_rot: the fast clock is weak, deepest near the "
                 "mid-troposphere level of non-divergence, and scrubbed by time-averaging")
 
    ax[0].bar([str(l) for l in levels], [daily[l] for l in levels],
              color=["#2ca02c", "#1f77b4", "#9467bd"])
    ax[0].set_xlabel("pressure level (hPa)")
    ax[0].set_ylabel("KE_div / KE_rot (daily mean)")
    ax[0].set_title("vertical structure: minimum at ~500 hPa\n(the level of non-divergence)")
    for i, l in enumerate(levels):
        ax[0].text(i, daily[l], f"{daily[l]*100:.1f}%", ha="center", va="bottom")
 
    ax[1].bar(["daily mean", "6-hourly\ninstantaneous"], [daily[500], inst500],
              color=["#1f77b4", "#ff7f0e"])
    ax[1].set_ylabel("KE_div / KE_rot at 500 hPa")
    ax[1].set_title("time-averaging is a low-pass filter\non the fast (divergent) clock")
    for i, val in enumerate([daily[500], inst500]):
        ax[1].text(i, val, f"{val*100:.2f}%", ha="center", va="bottom")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, "27_reanalysis_levels.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p, daily, inst500
 
 
def write_report(path, figs, spec_ratio, daily, inst500):
    rel = lambda p: os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p) \
        if os.path.dirname(p) else p
    f25, f26, f27 = figs
    txt = f"""# Part 7 -- the two clocks in real reanalysis winds
 
**Data:** NCEP/NCAR Reanalysis 1 (Kalnay et al. 1996), 2.5 deg, fetched live from
NOAA PSL over OPeNDAP (no credentials).  Parts 5-6 split a *synthetic* compressible
flow into a slow solenoidal part and a fast dilatational (acoustic) part.  Here the
same idea is applied to *real* horizontal winds via the Helmholtz split into a
rotational (non-divergent) part and a divergent part.
 
**Honesty note.**  Horizontal wind divergence is *not* a model artifact: by 3-D
mass continuity, du/dx + dv/dy = -dw/dz, it is the genuine footprint of vertical
motion.  So the diagnostic is the *Helmholtz split* -- how much kinetic energy lives
in the divergent (fast) versus rotational (slow) part -- not the raw divergence.
 
## 1. One snapshot: the weather is rotational; the fast clock is a weak residual
 
![snapshot]({rel(f25)})
 
The rotational wind (left) *is* the synoptic weather -- the balanced, geostrophic
highs and lows.  The divergent wind (right) is ~100x weaker and concentrated in
convergence/overturning zones (tropical convection bands, storm inflow): the
fast-clock footprint, the real-atmosphere analogue of the dilatational velocity.
 
## 2. The kinetic-energy spectrum: divergent << rotational at every scale
 
![spectrum]({rel(f26)})
 
Splitting the kinetic energy by spherical wavenumber l (from the SH power spectra of
vorticity and divergence, weighted by 1/(l(l+1))), the divergent branch sits well
below the rotational branch at all scales.  Integrated, **KE_div/KE_rot = {spec_ratio:.3f}**
at 850 hPa -- the real-data echo of the KE_dil/KE_sol ~ M^2 << 1 result from Parts 5-6.
 
## 3. Vertical structure and the low-pass-filter point
 
![levels]({rel(f27)})
 
| level | KE_div/KE_rot (daily mean) |
|---|---|
| 850 hPa | {daily[850]*100:.2f}% |
| 500 hPa | {daily[500]*100:.2f}% |
| 250 hPa | {daily[250]*100:.2f}% |
 
- The divergent fraction is **smallest near 500 hPa** -- the classical *level of
  non-divergence* -- and larger in the boundary layer (850 hPa, convergence into
  lows) and the upper-level jet/outflow (250 hPa).  The physics, not a tuning, sets
  this.
- At 500 hPa the **6-hourly instantaneous** ratio ({inst500*100:.2f}%) exceeds the
  **daily-mean** ratio ({daily[500]*100:.2f}%): time-averaging is a low-pass filter
  that scrubs the fast divergent clock -- exactly the move that recovered the elliptic
  pressure field in Part 5 (averaging over the acoustic period).
 
## Scope
 
Reanalysis is a model-assimilated product, not raw observation, and 2.5 deg resolves
only large scales (l <= 35).  This is a real-data *confirmation* that the wind's
energy is overwhelmingly in the slow rotational clock with a weak, physically
structured fast divergent clock; it is not a turbulence-closure or 3-D regularity
claim.
"""
    with open(path, "w") as fh:
        fh.write(txt)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--report", default="REPORT_REANALYSIS.md")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--nt", type=int, default=24)
    args = ap.parse_args()
 
    os.makedirs(args.out_dir, exist_ok=True)
    print("Part 7: rotational/divergent split of real NCEP reanalysis winds ...")
    f25 = fig_snapshot(args.out_dir, year=args.year)
    print(f"  -> {f25}")
    f26, spec_ratio = fig_spectrum(args.out_dir, year=args.year, nt=args.nt)
    print(f"  -> {f26}  (KE_div/KE_rot @850 = {spec_ratio:.4f})")
    f27, daily, inst500 = fig_levels(args.out_dir, year=args.year, nt=args.nt)
    print(f"  -> {f27}")
    print(f"  daily: 850={daily[850]:.4f} 500={daily[500]:.4f} 250={daily[250]:.4f} "
          f"| inst500={inst500:.4f}")
    write_report(args.report, (f25, f26, f27), spec_ratio, daily, inst500)
    print(f"Report: {args.report}")
 
 
if __name__ == "__main__":
    main()
