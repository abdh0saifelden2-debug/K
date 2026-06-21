"""Part 9 -- the two-clocks closure test in a subglacial cavity.
 
Turbulent meltwater flows through a cavity between a bumpy rock bed (warm) and a
flat ice base (cold), penalized into a doubly-periodic spectral box.  Flow
separates in the lee of the bedrock bumps, leaving recirculating wake eddies that
trap heat against the bed -- the glacier-relevant "fast clock" structures.
 
We run a high-resolution DNS to a developed turbulent state, then repeat the
Part-8b a-priori test *in this geometry*: sharp-spectral-filter the truth at k_c,
form the exact subgrid momentum force m_true and the exact subgrid heat flux, and
score Smagorinsky (K-theory) vs the projected-FDT closure on diagnostics a
glaciologist cares about:
 
  1. energy transfer T(k) and its *spatial* density Pi = ubar.m -- does the
     closure return energy to the lee wake (backscatter) or only drain it?
  2. structural solenoidality RMS(div m) -- a non-solenoidal force injects a
     spurious pressure, i.e. a wrong effective pressure on the bed.
  3. subgrid heat flux -- can the closure represent counter-gradient (up-gradient)
     heat trapping in the lee, or does down-gradient K-theory forbid it?
 
Generates figures 31-34 and REPORT_SUBGLACIAL.md.  Needs no downloaded data.
 
    python run_subglacial.py --out-dir figures --report REPORT_SUBGLACIAL.md
"""
 
from __future__ import annotations
 
import argparse
import os
import sys
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
 
# repo reorg: make sibling domain folders importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("general_two_clocks", "atmosphere", "glaciers", "ocean"):
    _p = os.path.join(_REPO_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _d, _p

from subglacial.flow import SubglacialFlow, SubglacialConfig  # noqa: E402
from subglacial.diag import (  # noqa: E402
    spatial_transfer, exact_sgs_heat_flux, eddy_diffusivity_heat_flux,
    heat_flux_divergence, countergradient_fraction, masked_corr,
)
from closure.sgs import (  # noqa: E402
    exact_sgs_force, transfer_spectrum, divergence_rms,
    smagorinsky_force, surrogate_force, projected_fdt_force,
)
 
N_DNS = 128
KC = 20
STEPS = 9000
RAMP = 2500
SNAPS = 6          # snapshots (post-spinup) to average a-priori scores over
SNAP_GAP = 250     # steps between snapshots
 
TEAL = "#1f9e9e"
CLAY = "#c4622d"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
 
 
# ---------------------------------------------------------------------------
# Figure 31 -- DNS field overview (wakes, heat trapping, wake pressure cores)
# ---------------------------------------------------------------------------
 
def fig_fields(f: SubglacialFlow, out_dir, num=31, scenario=""):
    X, Y, chi = f.X, f.Y, f.chi
    solid = chi > 0.5
    om = np.ma.masked_where(solid, f.vorticity())
    sp_ = np.ma.masked_where(solid, np.sqrt(f.u ** 2 + f.v ** 2))
    th = np.ma.masked_where(solid, f.theta)
    pr = np.ma.masked_where(solid, f.pressure())
    ytop = f.cfg.ice_base + 0.2
 
    fig, ax = plt.subplots(4, 1, figsize=(8.5, 11))
    sd = float(np.std(om.compressed()))
    panels = [
        (ax[0], sp_, "viridis", None, None, "speed |u| — jet over bumps, slow recirculating lee cavities"),
        (ax[1], om, "RdBu_r", -3 * sd, 3 * sd, "vorticity ω — separated shear layer + lee roll-up"),
        (ax[2], th, "inferno", 0.0, 1.0, "heat θ — warm meltwater trapped in lee cavities"),
        (ax[3], pr, "RdBu_r", None, None, "pressure p — low-pressure wake cores (the elliptic fast clock)"),
    ]
    for a, fld, cmap, vmin, vmax, title in panels:
        pcm = a.pcolormesh(X, Y, fld, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        a.contour(X, Y, chi, [0.5], colors="k", linewidths=0.9)
        a.set_ylim(0, ytop)
        a.set_aspect("equal")
        a.set_title(title, fontsize=10)
        fig.colorbar(pcm, ax=a, shrink=0.85, pad=0.01)
    ax[-1].set_xlabel("x")
    fig.suptitle(f"Part 9: subglacial cavity DNS — turbulent meltwater over a bumpy bed{scenario}",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{num}_subglacial_fields.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
 
 
# ---------------------------------------------------------------------------
# Figure 32 -- transfer spectrum + divergence scorecard
# ---------------------------------------------------------------------------
 
def fig_transfer_div(sp, ub, vb, m_true, m_smag, m_surr, m_fdt, div_scores, out_dir, kc, num=32, scenario=""):
    k, T_true = transfer_spectrum(sp, ub, vb, m_true[0], m_true[1], kc)
    _, T_smag = transfer_spectrum(sp, ub, vb, m_smag[0], m_smag[1], kc)
    _, T_surr = transfer_spectrum(sp, ub, vb, m_surr[0], m_surr[1], kc)
    _, T_fdt = transfer_spectrum(sp, ub, vb, m_fdt[0], m_fdt[1], kc)
    _corr = lambda a, b: (0.0 if np.std(a) == 0 or np.std(b) == 0
                          else float(np.corrcoef(a, b)[0, 1]))
    m = k >= 1
    c_smag = _corr(T_true[m], T_smag[m])
    c_fdt = _corr(T_true[m], T_fdt[m])
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    ax.axhline(0, color="gray", lw=0.8)
    ax.plot(k[m], T_true[m], "o-", color="black", ms=4, lw=2, label="exact SGS force (truth)")
    ax.plot(k[m], T_smag[m], "s--", color=CLAY, ms=3, lw=1.5,
            label=f"Smagorinsky (K-theory)  [corr {c_smag:+.2f}]")
    ax.plot(k[m], T_fdt[m], "D-", color=GREEN, ms=3, lw=1.5,
            label=f"projected-FDT  [corr {c_fdt:+.2f}]")
    lt = max(1.0, float(np.percentile(np.abs(np.r_[T_true[m], T_fdt[m]]), 65)))
    ax.set_yscale("symlog", linthresh=max(lt, 1e-12))
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("energy transfer  T(k) = Re⟨û*·m̂⟩   (symlog)")
    ax.set_title("T(k) > 0 = backscatter into the wake;  T(k) < 0 = dissipation\n"
                 "Smagorinsky stays negative at every scale (no backscatter); truth & FDT cross zero")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
 
    ax = axes[1]
    names = list(div_scores.keys())
    vals = [div_scores[n] for n in names]
    colors = ["black", CLAY, ORANGE, GREEN]
    bars = ax.bar(names, vals, color=colors, edgecolor="k", linewidth=0.6)
    ax.set_ylabel("relative RMS divergence  |∇·m| / |m|")
    ax.set_title("solenoidality — a non-zero ∇·m is a spurious effective pressure")
    ax.set_yscale("log")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.3, f"{v:.1e}",
                ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.suptitle(f"Part 9: subgrid-force structure in the cavity{scenario}", fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{num}_subglacial_transfer.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
 
 
# ---------------------------------------------------------------------------
# Figure 33 -- spatial backscatter map Pi = ubar.m in the lee
# ---------------------------------------------------------------------------
 
def fig_backscatter_map(f, ub, vb, m_true, m_smag, m_fdt, out_dir, num=33, scenario=""):
    X, Y, chi = f.X, f.Y, f.chi
    solid = chi > 0.5
    band = f.wake_band()
    ytop = f.cfg.ice_base + 0.2
    pi_true = spatial_transfer(ub, vb, m_true[0], m_true[1])
    pi_smag = spatial_transfer(ub, vb, m_smag[0], m_smag[1])
    pi_fdt = spatial_transfer(ub, vb, m_fdt[0], m_fdt[1])
    pis = [
        ("exact (truth)", pi_true, None),
        ("Smagorinsky (K-theory)", pi_smag, masked_corr(pi_smag, pi_true, band)),
        ("projected-FDT", pi_fdt, masked_corr(pi_fdt, pi_true, band)),
    ]
    scale = float(np.percentile(np.abs(pi_true[f.fluid]), 99)) + 1e-30
 
    fig, ax = plt.subplots(3, 1, figsize=(8.5, 8.6))
    for a, (name, pi, corr) in zip(ax, pis):
        pim = np.ma.masked_where(solid, pi)
        pcm = a.pcolormesh(X, Y, pim, cmap="RdBu_r", shading="auto",
                           vmin=-scale, vmax=scale)
        a.contour(X, Y, chi, [0.5], colors="k", linewidths=0.9)
        a.set_ylim(0, ytop)
        a.set_aspect("equal")
        tag = "" if corr is None else f"   (spatial corr. with truth in wake = {corr:+.2f})"
        a.set_title(f"{name}:  Π = ūbar·m{tag}", fontsize=10)
        fig.colorbar(pcm, ax=a, shrink=0.85, pad=0.01)
    ax[-1].set_xlabel("x")
    fig.suptitle(f"Part 9: where the subgrid force feeds (red) or drains (blue) the resolved wake{scenario}",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{num}_subglacial_backscatter_map.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
 
 
# ---------------------------------------------------------------------------
# Figure 34 -- subgrid heat flux: exact vs down-gradient K-theory
# ---------------------------------------------------------------------------
 
def fig_heatflux(f, sp, kc, cs, out_dir, num=34, scenario=""):
    u, v, th = f.u, f.v, f.theta
    qx, qy, tb, gx, gy = exact_sgs_heat_flux(sp, u, v, th, kc)
    from closure.sgs import sharp_filter
    ubf = sharp_filter(sp, u, kc)
    vbf = sharp_filter(sp, v, kc)
    qmx, qmy = eddy_diffusivity_heat_flux(sp, ubf, vbf, tb, kc, cs=cs)
    Q_true = heat_flux_divergence(sp, qx, qy)
    Q_mod = heat_flux_divergence(sp, qmx, qmy)
 
    X, Y, chi = f.X, f.Y, f.chi
    solid = chi > 0.5
    ytop = f.cfg.ice_base + 0.2
    sc = float(np.percentile(np.abs(Q_true[f.fluid]), 99)) + 1e-30
 
    # up-gradient (counter-gradient) regions of the EXACT flux
    align = qx * gx + qy * gy
    cg = np.ma.masked_where(solid | (align <= 0), np.ones_like(align))
 
    fig, ax = plt.subplots(3, 1, figsize=(8.5, 8.6))
    for a, fld, title in [
        (ax[0], np.ma.masked_where(solid, Q_true), "exact subgrid heating  Q = −∇·τ_θ (truth)"),
        (ax[1], np.ma.masked_where(solid, Q_mod), "K-theory model  Q = ∇·(κ_t ∇θ̄)  (down-gradient only)"),
    ]:
        pcm = a.pcolormesh(X, Y, fld, cmap="RdBu_r", shading="auto", vmin=-sc, vmax=sc)
        a.contour(X, Y, chi, [0.5], colors="k", linewidths=0.9)
        a.set_ylim(0, ytop); a.set_aspect("equal"); a.set_title(title, fontsize=10)
        fig.colorbar(pcm, ax=a, shrink=0.85, pad=0.01)
    a = ax[2]
    a.pcolormesh(X, Y, np.ma.masked_where(solid, tb), cmap="inferno", shading="auto", vmin=0, vmax=1)
    a.pcolormesh(X, Y, cg, cmap="spring", shading="auto", alpha=0.9)
    a.contour(X, Y, chi, [0.5], colors="k", linewidths=0.9)
    a.set_ylim(0, ytop); a.set_aspect("equal")
    a.set_title("filtered θ̄ with up-gradient (counter-gradient) cells highlighted — K-theory forbids these",
                fontsize=10)
    a.set_xlabel("x")
    fig.suptitle(f"Part 9: lee heat trapping is counter-gradient — invisible to eddy diffusivity{scenario}",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{num}_subglacial_heatflux.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    cg_frac = countergradient_fraction(qx, qy, gx, gy, f.wake_band())
    corr = masked_corr(Q_true, Q_mod, f.wake_band())
    return path, cg_frac, corr
 
 
_HAND_WRITTEN_MARKER = "<!-- BEGIN HAND-WRITTEN CONTENT (preserved across regeneration) -->"
 
 
def write_report(path, figs, scores, n, kc, steps, *, fignums=(31, 32, 33, 34),
                 bed_setup="a bumpy (synthetic) rock bed (warm, θ=1) and a flat ice base (cold, θ=0)",
                 bed_para="",
                 dem_caveat=("there is no ice\nrheology, no real bed DEM, and no coupled subglacial "
                             "hydrology or sliding law"),
                 title_note=""):
    f31, f32, f33, f34 = figs
    n1, n2, n3, n4 = fignums
    hand_written = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        idx = content.find(_HAND_WRITTEN_MARKER)
        if idx >= 0:
            hand_written = content[idx:]
 
    s = scores
    txt = f"""<!-- Benchmark section below is GENERATED by run_subglacial.py (n={n}, kc={kc}, steps={steps}).
     Re-run to refresh numbers. Hand-written content below the marker is preserved. -->
 
# Part 9 — the two-clocks closure test in a subglacial cavity{title_note}
 
**Setup:** a {n}² penalized pseudo-spectral DNS of turbulent meltwater in a cavity
between {bed_setup}, spun up for
{steps} steps with stochastic ring forcing (Re-sustaining ambient turbulence).
The developed field is sharp-spectral-filtered at k_c = {kc} and the exact subgrid
momentum force m_true and subgrid heat flux are compared against Smagorinsky
(K-theory) and the projected-FDT closure.  All scores below are means over
{SNAPS} snapshots; ± is one standard deviation.
 
## 1. The flow (figure {n1})
 
![fields]({os.path.basename(f31)})
 
Meltwater accelerates over the bumps and separates in their lee, leaving slow,
recirculating wake cavities that **trap warm water against the bed**. The pressure
panel shows the low-pressure wake cores — the global, elliptic "fast clock".
{bed_para}
 
## 2. Energy transfer and solenoidality (figure {n2})
 
![transfer]({os.path.basename(f32)})
 
| model | rel. RMS(∇·m) | transfer corr. with truth |
|---|---|---|
| truth | {s['div_truth'][0]:.2e} ± {s['div_truth'][1]:.0e} | 1.000 |
| Smagorinsky | {s['div_smag'][0]:.2e} ± {s['div_smag'][1]:.0e} | {s['corr_smag'][0]:.3f} ± {s['corr_smag'][1]:.3f} |
| surrogate | {s['div_surr'][0]:.2e} ± {s['div_surr'][1]:.0e} | {s['corr_surr'][0]:.3f} ± {s['corr_surr'][1]:.3f} |
| projected-FDT | {s['div_fdt'][0]:.2e} ± {s['div_fdt'][1]:.0e} | {s['corr_fdt'][0]:.3f} ± {s['corr_fdt'][1]:.3f} |
 
- The exact transfer has **backscatter** (T(k) > 0): the subgrid scales return
  energy to the resolved wake. Smagorinsky has T(k) ≤ 0 everywhere — purely
  dissipative, so it can only *drain* the wake.
- A non-solenoidal subgrid force injects a spurious pressure. The surrogate breaks
  ∇·m = 0; projected-FDT stays solenoidal by construction (correct bed pressure).
 
## 3. Where the wake is fed — spatial backscatter (figure {n3})
 
![backscatter map]({os.path.basename(f33)})
 
The exact transfer density Π = ūbar·m has coherent **backscatter patches (red)** in
the lee of the bumps — energy returned to the resolved wake exactly where the
separation eddy lives. Smagorinsky's Π tracks −|S̄|² and so drains the high-shear
separated layer; it has no mechanism to feed those lee patches.
 
Pointwise correlation of the modeled Π with the exact Π in the wake band is low for
*both* closures (Smagorinsky **{s['pi_corr_smag'][0]:+.2f} ± {s['pi_corr_smag'][1]:.2f}**,
projected-FDT **{s['pi_corr_fdt'][0]:+.2f} ± {s['pi_corr_fdt'][1]:.2f}**) — this is the
well-known limitation that *no* eddy-viscosity-type closure reproduces the
instantaneous subgrid field pointwise. The discriminating statement is therefore
the **scale-resolved** transfer of §2: Smagorinsky is anti-correlated with the truth
at the shell level ({s['corr_smag'][0]:+.2f}) and drains every scale, whereas
projected-FDT is positively correlated ({s['corr_fdt'][0]:+.2f}) because it returns
energy in the shells where the truth backscatters. A purely dissipative K-theory
LES therefore drifts toward a "dead wake".
 
## 4. Lee heat trapping is counter-gradient (figure {n4})
 
![heat flux]({os.path.basename(f34)})
 
- Up-gradient (counter-gradient) fraction of the **exact** subgrid heat flux in the
  wake band: **{s['cg_frac'][0]*100:.0f}% ± {s['cg_frac'][1]*100:.0f}%**.
- A positive eddy diffusivity (K-theory) gives a strictly **down-gradient** flux —
  its counter-gradient fraction is **0% by construction**, so it structurally
  cannot represent heat held against the mean gradient inside a recirculation
  cavity. Spatial correlation of the modeled subgrid heating with truth in the
  wake band is only {s['hf_corr'][0]:.2f} ± {s['hf_corr'][1]:.2f}.
 
## Conclusion
 
In a glacier-relevant geometry the same structural failure isolated in Part 8b
reappears, now with a physical reading:
 
1. **Wake (sliding/friction):** K-theory has no backscatter in the lee → it cannot
   sustain the separation eddy → it under-predicts the turbulent wake that controls
   basal drag. Projected-FDT restores the backscatter.
2. **Effective pressure:** a non-solenoidal closure injects a spurious pressure on
   the bed; the Leray-projected closure does not.
3. **Basal melt:** lee heat trapping is counter-gradient and therefore invisible to
   eddy diffusivity; K-theory under-predicts the heat retained against the bed.
 
## Scope and honesty
 
This is a **2D, a-priori (frozen-field) mechanism demonstration**, not an
operational glacier model. It shares Part 8b's caveats (single-instant, no
time-integration memory) and adds geometry-specific ones: the bed and ice are
embedded by volume **penalization** (a true solid wall breaks the clean
Leray-projection commutation that the periodic spectral method assumes); "melt" is
a **heat-flux/heat-content proxy**, not a Stefan phase change; {dem_caveat}. The
result upgrades the subglacial application from a qualitative analogy to a computed
2D structural claim — it does **not** constitute a validated glaciological
prediction.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)
        fh.write(hand_written if hand_written else _HAND_WRITTEN_MARKER + "\n")
 
 
def _mean_std(d):
    a = np.array(d)
    return float(a.mean()), float(a.std())
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--report", default=None)
    ap.add_argument("--n", type=int, default=N_DNS)
    ap.add_argument("--kc", type=int, default=KC)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--f-amp", type=float, default=1.5)
    ap.add_argument("--bed", choices=["idealized", "real"], default="idealized",
                    help="synthetic sinusoidal bed, or a real BEDMAP1 transect")
    ap.add_argument("--bed-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "subglacial", "data", "bedmap1_transect.csv"))
    args = ap.parse_args()
 
    os.makedirs(args.out_dir, exist_ok=True)
    kc, cs = args.kc, 0.16
 
    # scenario-specific geometry, output names, and report wording
    base = SubglacialConfig(n=args.n, sgs="none", f_amp=args.f_amp, k_f=10.0,
                            f_band=2.0, seed=1)
    fignums = (31, 32, 33, 34)
    scenario_fig, title_note = "", ""
    bed_setup = "a bumpy (synthetic) rock bed (warm, θ=1) and a flat ice base (cold, θ=0)"
    bed_para = ""
    dem_caveat = ("there is no ice\nrheology, no real bed DEM, and no coupled subglacial "
                  "hydrology or sliding law")
    if args.bed == "real":
        from subglacial.bedmap import bed_profile_from_transect
        prof, meta = bed_profile_from_transect(args.bed_file, args.n,
                                               base.bed_mean, base.bed_amp)
        cfg = SubglacialConfig(n=args.n, sgs="none", f_amp=args.f_amp, k_f=10.0,
                               f_band=2.0, seed=1, bed_profile=prof)
        fignums = (35, 36, 37, 38)
        scenario_fig = "  —  real BEDMAP1 Antarctic bed"
        title_note = " (real BEDMAP1 bed)"
        bed_setup = (
            f"a **real Antarctic bedrock transect** (BEDMAP1 airborne radar, "
            f"{meta['length_km']:.0f} km, {meta['relief_m']:.0f} m relief; warm, θ=1) "
            f"and a flat ice base (cold, θ=0)")
        bed_para = (
            "\nThe bed here is **not** synthetic: it is a measured BEDMAP1 airborne-radar "
            "bedrock transect (British Antarctic Survey / SCAR Bedmap, CC-BY-4.0; "
            "DOI:10.5285/f64815ec-4077-4432-9f55-0ce230f46029), mirrored for periodicity "
            "and non-dimensionalised to the cavity. Its real relief — asymmetric bumps and "
            "overdeepenings — sets where the flow separates.")
        dem_caveat = (
            "the bed geometry is now a **real measured transect** (BEDMAP1), but it is "
            "non-dimensionalised and made periodic to fit the spectral box; there is still "
            "no ice rheology and no coupled subglacial hydrology or sliding law")
    else:
        cfg = base
    if args.report is None:
        args.report = ("REPORT_SUBGLACIAL_REAL.md" if args.bed == "real"
                       else "REPORT_SUBGLACIAL.md")

    print(f"Part 9: subglacial closure test (n={args.n}, kc={kc}, spinup={args.steps}, bed={args.bed})")
    f = SubglacialFlow(cfg)
    print("  spinning up DNS truth ...")
    f.run(args.steps, ramp=RAMP, report_every=max(1, args.steps // 6))
    print(f"  developed: KE={f.kinetic_energy():.4e}  heat_in_wake={f.heat_in_wake():.4e}")
 
    sp = f.sp
    acc = {k: [] for k in (
        "div_truth", "div_smag", "div_surr", "div_fdt",
        "corr_smag", "corr_surr", "corr_fdt",
        "pi_corr_smag", "pi_corr_fdt", "cg_frac", "hf_corr")}
 
    snap_fields = None
    print(f"  scoring closures over {SNAPS} snapshots ...")
    for i in range(SNAPS):
        u, v = f.u, f.v
        ub, vb, mtx, mty = exact_sgs_force(sp, u, v, kc)
        msx, msy = smagorinsky_force(sp, ub, vb, kc, cs=cs)
        mqx, mqy = surrogate_force(sp, mtx, mty, kc, seed=7)
        mfx, mfy, _ = projected_fdt_force(sp, ub, vb, mtx, mty, kc, seed=7)
 
        acc["div_truth"].append(divergence_rms(sp, mtx, mty))
        acc["div_smag"].append(divergence_rms(sp, msx, msy))
        acc["div_surr"].append(divergence_rms(sp, mqx, mqy))
        acc["div_fdt"].append(divergence_rms(sp, mfx, mfy))
 
        k, Tt = transfer_spectrum(sp, ub, vb, mtx, mty, kc)
        _, Ts = transfer_spectrum(sp, ub, vb, msx, msy, kc)
        _, Tq = transfer_spectrum(sp, ub, vb, mqx, mqy, kc)
        _, Tf = transfer_spectrum(sp, ub, vb, mfx, mfy, kc)
        m = k >= 1
        cc = lambda a, b: (0.0 if np.std(a) == 0 or np.std(b) == 0
                           else float(np.corrcoef(a, b)[0, 1]))
        acc["corr_smag"].append(cc(Tt[m], Ts[m]))
        acc["corr_surr"].append(cc(Tt[m], Tq[m]))
        acc["corr_fdt"].append(cc(Tt[m], Tf[m]))
 
        band = f.wake_band()
        pi_t = spatial_transfer(ub, vb, mtx, mty)
        acc["pi_corr_smag"].append(masked_corr(spatial_transfer(ub, vb, msx, msy), pi_t, band))
        acc["pi_corr_fdt"].append(masked_corr(spatial_transfer(ub, vb, mfx, mfy), pi_t, band))
 
        qx, qy, tb, gx, gy = exact_sgs_heat_flux(sp, u, v, f.theta, kc)
        qmx, qmy = eddy_diffusivity_heat_flux(sp, ub, vb, tb, kc, cs=cs)
        acc["cg_frac"].append(countergradient_fraction(qx, qy, gx, gy, band))
        acc["hf_corr"].append(masked_corr(heat_flux_divergence(sp, qx, qy),
                                          heat_flux_divergence(sp, qmx, qmy), band))
 
        if i == SNAPS - 1:
            snap_fields = (ub, vb, (mtx, mty), (msx, msy), (mqx, mqy), (mfx, mfy))
        if i < SNAPS - 1:
            for _ in range(SNAP_GAP):
                f.step(cfg.U0)
 
    scores = {k: _mean_std(v) for k, v in acc.items()}
    print("  scores:")
    for k in ("div_smag", "div_fdt", "corr_smag", "corr_fdt",
              "pi_corr_smag", "pi_corr_fdt", "cg_frac", "hf_corr"):
        print(f"    {k:14s} {scores[k][0]:+.4f} ± {scores[k][1]:.4f}")
 
    print("  generating figures ...")
    ub, vb, m_true, m_smag, m_surr, m_fdt = snap_fields
    div_scores = {"truth": scores["div_truth"][0], "Smagorinsky": scores["div_smag"][0],
                  "surrogate": scores["div_surr"][0], "projected-FDT": scores["div_fdt"][0]}
    f31 = fig_fields(f, args.out_dir, num=fignums[0], scenario=scenario_fig)
    f32 = fig_transfer_div(sp, ub, vb, m_true, m_smag, m_surr, m_fdt, div_scores,
                           args.out_dir, kc, num=fignums[1], scenario=scenario_fig)
    f33 = fig_backscatter_map(f, ub, vb, m_true, m_smag, m_fdt, args.out_dir,
                              num=fignums[2], scenario=scenario_fig)
    f34, cg_frac, hf_corr = fig_heatflux(f, sp, kc, cs, args.out_dir,
                                         num=fignums[3], scenario=scenario_fig)
    for p in (f31, f32, f33, f34):
        print(f"  -> {p}")
 
    write_report(args.report, (f31, f32, f33, f34), scores, args.n, kc, args.steps,
                 fignums=fignums, bed_setup=bed_setup, bed_para=bed_para,
                 dem_caveat=dem_caveat, title_note=title_note)
    print(f"Report: {args.report}")
 
 
if __name__ == "__main__":
    main()
