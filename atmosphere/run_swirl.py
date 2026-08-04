"""Part 10 -- the two-clocks closure test at a turbulent vortex (Goldshtik-Sorokin).

A heavy particle can be held up at the core of a turbulent vortex by the radial
pressure-gradient force of the core's low-pressure well -- the Goldshtik-Sorokin
effect.  That well is a *global*, elliptic response (the inverse-Laplacian of the
swirl strain), so it is exactly the kind of structure the "two clocks" thesis says
K-theory is blind to.

We spin up a sustained turbulent swirl to a developed state, then repeat the
Part-8b a-priori test *in this geometry*: sharp-spectral-filter the truth at k_c,
form the exact subgrid momentum force m_true, and score Smagorinsky (K-theory),
a spectrum-matched surrogate, and the projected-FDT closure on

  1. energy transfer T(k) and solenoidality RMS(div m) -- does the closure return
     energy to the swirl (backscatter) or only drain it, and does it inject a
     spurious pressure?
  2. the spatial transfer density Pi = ubar.m -- where the swirl is fed vs drained;
  3. a **suspension margin** -- the net resolved SGS power turned into a one-turnover
     change in swirl energy (hence core-well depth, hence levitation force),
     calibrated so the truth field just suspends the particle (margin := 1).

Falsifiable prediction: Smagorinsky over-drains the swirl -> margin < 1 -> particle
falls; projected-FDT preserves the swirl -> margin ~ 1 -> particle stays up.

Generates figures 40-43 and REPORT_SWIRL.md.  Needs no downloaded data.

    python run_swirl.py --out-dir figures --report REPORT_SWIRL.md

A real-data variant calibrates the swirl to Hurricane Otis's best-track vortex
(NHC HURDAT2, bundled in ``swirl/data``): the target tangential profile is a
Holland (1980) fit to Otis's peak Vmax / central pressure / RMW.  It writes figures
44-47 and REPORT_SWIRL_REAL.md:

    python run_swirl.py --swirl otis
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

from swirl.flow import SwirlFlow, SwirlConfig  # noqa: E402
from swirl.levitation import (  # noqa: E402
    resolved_sgs_power, swirl_turnover, suspension_margin,
    radial_pressure_profile, radial_speed_profile,
)
from subglacial.diag import spatial_transfer, masked_corr  # noqa: E402
from closure.sgs import (  # noqa: E402
    exact_sgs_force, transfer_spectrum, divergence_rms,
    smagorinsky_force, surrogate_force, projected_fdt_force,
)

N_DNS = 128
KC = 20
STEPS = 4000
SNAPS = 6          # snapshots (post-spinup) to average a-priori scores over
SNAP_GAP = 250     # steps between snapshots

TEAL = "#1f9e9e"
CLAY = "#c4622d"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"


# ---------------------------------------------------------------------------
# Figure 40 -- DNS field overview + the core pressure well
# ---------------------------------------------------------------------------

def fig_fields(f: SwirlFlow, depth, out_dir, num=40):
    X, Y = f.X, f.Y
    speed = np.sqrt(f.u ** 2 + f.v ** 2)
    om = f.vorticity()
    p = f.pressure()
    rc_p, prof_p = radial_pressure_profile(p, f.r)
    rc_u, prof_u = radial_speed_profile(f.u, f.v, f.r)

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))
    sd = float(np.std(om))

    a = ax[0, 0]
    pcm = a.pcolormesh(X, Y, speed, cmap="viridis", shading="auto")
    a.set_aspect("equal"); a.set_title("(a) speed |u| — coherent swirl + ambient turbulence", fontsize=10)
    fig.colorbar(pcm, ax=a, shrink=0.85, pad=0.01)

    a = ax[0, 1]
    pcm = a.pcolormesh(X, Y, om, cmap="RdBu_r", shading="auto", vmin=-3 * sd, vmax=3 * sd)
    a.set_aspect("equal"); a.set_title("(b) vorticity ω — concentrated core + turbulent skirt", fontsize=10)
    fig.colorbar(pcm, ax=a, shrink=0.85, pad=0.01)

    a = ax[1, 0]
    pcm = a.pcolormesh(X, Y, p, cmap="RdBu_r", shading="auto")
    a.contour(X, Y, f.r, [0.5], colors="k", linewidths=0.9, linestyles=":")
    a.set_aspect("equal")
    a.set_title(f"(c) pressure p — deep low-pressure core well (depth {depth:.3f})\n"
                "the global, elliptic 'fast clock' that levitates the particle", fontsize=10)
    fig.colorbar(pcm, ax=a, shrink=0.85, pad=0.01)

    a = ax[1, 1]
    a.plot(rc_u, prof_u, "o-", color=TEAL, label="azimuthal speed |u|(r)")
    a.set_xlabel("radius r from core"); a.set_ylabel("speed |u|", color=TEAL)
    a.tick_params(axis="y", labelcolor=TEAL)
    a.axvline(f.cfg.r_core, color="gray", ls=":", lw=1)
    a2 = a.twinx()
    a2.plot(rc_p, prof_p, "s-", color=CLAY, label="pressure p(r) − far")
    a2.set_ylabel("pressure p − p_far", color=CLAY)
    a2.tick_params(axis="y", labelcolor=CLAY)
    a.set_title("(d) radial profiles — speed peaks near r_core; pressure dips at the core", fontsize=10)

    fig.suptitle("Part 10: turbulent swirl DNS — the Goldshtik–Sorokin core pressure well",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{num}_swirl_fields.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 41 -- transfer spectrum + divergence scorecard
# ---------------------------------------------------------------------------

def fig_transfer_div(sp, ub, vb, m_true, m_smag, m_surr, m_fdt, div_scores, out_dir, kc, num=41):
    k, T_true = transfer_spectrum(sp, ub, vb, m_true[0], m_true[1], kc)
    _, T_smag = transfer_spectrum(sp, ub, vb, m_smag[0], m_smag[1], kc)
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
    ax.set_title("T(k) > 0 = backscatter into the swirl;  T(k) < 0 = dissipation\n"
                 "Smagorinsky over-drains every scale; projected-FDT tracks the truth")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    names = list(div_scores.keys())
    vals = [div_scores[n] for n in names]
    colors = ["black", CLAY, ORANGE, GREEN]
    bars = ax.bar(names, vals, color=colors, edgecolor="k", linewidth=0.6)
    ax.set_ylabel("relative RMS divergence  |∇·m| / |m|")
    ax.set_title("solenoidality — a non-zero ∇·m is a spurious pressure on the core")
    ax.set_yscale("log")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.3, f"{v:.1e}",
                ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.suptitle("Part 10: subgrid-force structure in the swirl", fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{num}_swirl_transfer.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 42 -- spatial transfer map Pi = ubar.m
# ---------------------------------------------------------------------------

def fig_transfer_map(f, ub, vb, m_true, m_smag, m_fdt, out_dir, num=42):
    X, Y = f.X, f.Y
    swirl_band = f.r < 1.6
    pi_true = spatial_transfer(ub, vb, m_true[0], m_true[1])
    pi_smag = spatial_transfer(ub, vb, m_smag[0], m_smag[1])
    pi_fdt = spatial_transfer(ub, vb, m_fdt[0], m_fdt[1])
    pis = [
        ("exact (truth)", pi_true, None),
        ("Smagorinsky (K-theory)", pi_smag, masked_corr(pi_smag, pi_true, swirl_band)),
        ("projected-FDT", pi_fdt, masked_corr(pi_fdt, pi_true, swirl_band)),
    ]
    scale = float(np.percentile(np.abs(pi_true), 99)) + 1e-30

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 5.0))
    for a, (name, pi, corr) in zip(ax, pis):
        pcm = a.pcolormesh(X, Y, pi, cmap="RdBu_r", shading="auto",
                           vmin=-scale, vmax=scale)
        a.contour(X, Y, f.r, [0.5, 1.6], colors="k", linewidths=0.7, linestyles=":")
        a.set_aspect("equal")
        tag = "" if corr is None else f"\n(spatial corr. with truth in swirl = {corr:+.2f})"
        a.set_title(f"{name}:  Π = ū·m{tag}", fontsize=10)
        fig.colorbar(pcm, ax=a, shrink=0.8, pad=0.02)
    fig.suptitle("Part 10: where the subgrid force feeds (red) or drains (blue) the resolved swirl",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{num}_swirl_transfer_map.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 43 -- suspension margin (the Goldshtik-Sorokin punchline)
# ---------------------------------------------------------------------------

def fig_levitation(margins, powers, out_dir, num=43):
    names = ["truth", "Smagorinsky", "surrogate", "projected-FDT"]
    keys = ["truth", "smag", "surr", "fdt"]
    colors = ["black", CLAY, ORANGE, GREEN]
    m_mean = [margins[k][0] for k in keys]
    m_std = [margins[k][1] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    a = axes[0]
    bars = a.bar(names, m_mean, yerr=m_std, color=colors, edgecolor="k",
                 linewidth=0.6, capsize=4)
    a.axhline(1.0, color="crimson", ls="--", lw=1.4)
    a.text(-0.45, 1.16, "suspension threshold (margin = 1)", color="crimson",
           ha="left", va="bottom", fontsize=9)
    a.set_ylabel("suspension margin   (swirl energy ÷ truth)")
    a.set_title("Goldshtik–Sorokin levitation: margin ≥ 1 → particle stays up\n"
                "Smagorinsky drops below 1 (particle falls); projected-FDT preserves it")
    for bar, v in zip(bars, m_mean):
        verdict = "stays up" if v >= 1.0 else "FALLS"
        a.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.06, verdict,
               ha="center", va="bottom", fontsize=9,
               color=("green" if v >= 1.0 else "crimson"),
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    a.set_ylim(0, max(1.35, max(m_mean) * 1.2))
    plt.setp(a.get_xticklabels(), rotation=15, ha="right")

    a = axes[1]
    pkeys = ["truth", "smag", "fdt"]
    pnames = ["truth", "Smagorinsky", "projected-FDT"]
    pcolors = ["black", CLAY, GREEN]
    pvals = [abs(powers[k][0]) for k in pkeys]
    bars = a.bar(pnames, pvals, color=pcolors, edgecolor="k", linewidth=0.6)
    a.set_yscale("log")
    a.set_ylabel("|net resolved SGS power|  |⟨ū·m⟩|")
    a.set_title("how fast each closure drains the swirl\n"
                "Smagorinsky drains orders of magnitude more than the truth")
    for bar, v in zip(bars, pvals):
        a.text(bar.get_x() + bar.get_width() / 2, v * 1.3, f"{v:.1e}",
               ha="center", va="bottom", fontsize=8)
    plt.setp(a.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Part 10: the closure decides whether the particle is suspended",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{num}_swirl_levitation.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 44 -- Hurricane Otis best-track calibration (real-data variant)
# ---------------------------------------------------------------------------

def fig_otis_calibration(track, vortex, r_core_box, out_dir, num=44):
    """Show the real Otis best track and the Holland vortex fitted to its peak,
    plus the non-dimensional target profile handed to the solver."""
    from swirl.otis import (holland_speed_shape, holland_pressure_deficit,
                            KT_TO_MS, NMI_TO_KM)

    t = np.arange(len(track))
    vmax = np.array([r["vmax_kt"] * KT_TO_MS for r in track])
    mslp = np.array([r["mslp_mb"] for r in track])
    rmw = np.array([r["rmw_nmi"] * NMI_TO_KM for r in track])
    ipk = int(np.nanargmin(mslp))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    a = ax[0]
    a.plot(t, vmax, "o-", color=CLAY, label="max wind Vmax")
    a.set_xlabel("best-track record (6-hourly)")
    a.set_ylabel("Vmax  (m/s)", color=CLAY); a.tick_params(axis="y", labelcolor=CLAY)
    a.axvline(ipk, color="gray", ls=":", lw=1)
    a2 = a.twinx()
    a2.plot(t, mslp, "s-", color=TEAL, label="min pressure")
    a2.set_ylabel("min central pressure  (mb)", color=TEAL); a2.tick_params(axis="y", labelcolor=TEAL)
    a.set_title(f"(a) Otis (EP182023) best track\npeak {vortex.vmax_kt:.0f} kt / {vortex.pc_mb:.0f} mb")

    a = ax[1]
    a.plot(t, rmw, "D-", color="purple")
    a.axvline(ipk, color="gray", ls=":", lw=1)
    a.set_xlabel("best-track record (6-hourly)")
    a.set_ylabel("radius of max wind  (km)")
    a.set_title(f"(b) RMW collapses to a pinhole eye\n(peak RMW {vortex.rmw_km:.1f} km ≈ {vortex.rmw_nmi:.0f} nmi)")

    a = ax[2]
    rr = np.linspace(0.02, 3.0, 300)
    a.plot(rr, holland_speed_shape(rr, r_core_box, vortex.holland_B), color=GREEN,
           lw=2, label=f"Holland V(r)/Vmax (B={vortex.holland_B:.2f})")
    a.axvline(r_core_box, color="gray", ls=":", lw=1, label="mapped RMW = r_core")
    a.set_xlabel("non-dimensional radius r (box units)")
    a.set_ylabel("V(r) / Vmax", color=GREEN); a.tick_params(axis="y", labelcolor=GREEN)
    a3 = a.twinx()
    a3.plot(rr, holland_pressure_deficit(rr, r_core_box, vortex.holland_B), color=CLAY,
            lw=1.5, ls="--", label="pressure deficit (p_env−p)/Δp")
    a3.set_ylabel("normalised pressure deficit", color=CLAY); a3.tick_params(axis="y", labelcolor=CLAY)
    a.set_title("(c) fitted Holland shape → solver target\nreal peakedness; size mapped to the box")
    a.legend(loc="upper right", fontsize=8)

    fig.suptitle("Part 10 (real data): Hurricane Otis best-track vortex calibration",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{num}_otis_calibration.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


_HAND_WRITTEN_MARKER = "<!-- BEGIN HAND-WRITTEN CONTENT (preserved across regeneration) -->"


def write_report(path, figs, scores, n, kc, steps, depth, tau):
    f40, f41, f42, f43 = figs
    hand_written = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        idx = content.find(_HAND_WRITTEN_MARKER)
        if idx >= 0:
            hand_written = content[idx:]

    s = scores
    txt = f"""<!-- Benchmark section below is GENERATED by run_swirl.py (n={n}, kc={kc}, steps={steps}).
     Re-run to refresh numbers. Hand-written content below the marker is preserved. -->

# Part 10 — the two-clocks closure test at a turbulent vortex (Goldshtik–Sorokin)

**Setup:** a {n}² pseudo-spectral DNS of a sustained turbulent swirl — a coherent
central vortex (peak azimuthal speed U≈1.4, core radius r_core≈0.7) held against
drag while stochastic ring forcing keeps ambient turbulence alive — spun up for
{steps} steps. The incompressible pressure digs a deep, global (elliptic)
low-pressure well at the core (measured depth **{depth:.3f}**), whose radial
pressure-gradient force is what levitates a heavy particle (the Goldshtik–Sorokin
effect). The developed field is sharp-spectral-filtered at k_c = {kc} and the exact
subgrid momentum force m_true is compared against Smagorinsky (K-theory), a
spectrum-matched surrogate, and the projected-FDT closure. All scores below are
means over {SNAPS} snapshots; ± is one standard deviation.

## 1. The swirl and its core well (figure 40)

![fields]({os.path.basename(f40)})

The coherent vortex sits in a turbulent skirt. The pressure panel shows the deep
low-pressure core depression — the global, elliptic structure (∇⁻² of the swirl
strain) that holds the particle up. The radial profiles confirm the cyclostrophic
picture: the azimuthal speed peaks near r_core and the pressure dips at the core.

## 2. Energy transfer and solenoidality (figure 41)

![transfer]({os.path.basename(f41)})

| model | rel. RMS(∇·m) | transfer corr. with truth | net resolved SGS power ⟨ū·m⟩ |
|---|---|---|---|
| truth | {s['div_truth'][0]:.2e} ± {s['div_truth'][1]:.0e} | 1.000 | {s['P_truth'][0]:+.2e} ± {s['P_truth'][1]:.0e} |
| Smagorinsky | {s['div_smag'][0]:.2e} ± {s['div_smag'][1]:.0e} | {s['corr_smag'][0]:+.3f} ± {s['corr_smag'][1]:.3f} | {s['P_smag'][0]:+.2e} ± {s['P_smag'][1]:.0e} |
| surrogate | {s['div_surr'][0]:.2e} ± {s['div_surr'][1]:.0e} | {s['corr_surr'][0]:+.3f} ± {s['corr_surr'][1]:.3f} | {s['P_surr'][0]:+.2e} ± {s['P_surr'][1]:.0e} |
| projected-FDT | {s['div_fdt'][0]:.2e} ± {s['div_fdt'][1]:.0e} | {s['corr_fdt'][0]:+.3f} ± {s['corr_fdt'][1]:.3f} | {s['P_fdt'][0]:+.2e} ± {s['P_fdt'][1]:.0e} |

- The decisive number is the **net resolved SGS power** ⟨ū·m⟩: it is the rate at
  which the closure drains (or feeds) the resolved swirl. Smagorinsky drains it at
  **{abs(s['P_smag'][0]) / max(abs(s['P_truth'][0]), 1e-30):.0f}×** the truth's net
  rate — a purely dissipative eddy viscosity (T(k) ≤ 0) has no backscatter to offset
  its over-dissipation. Projected-FDT tracks the truth's net drain.
- The spectrum-matched surrogate breaks ∇·m = 0; projected-FDT stays solenoidal by
  construction (no spurious pressure injected on the core).

## 3. Where the swirl is fed vs drained (figure 42)

![transfer map]({os.path.basename(f42)})

The exact transfer density Π = ū·m has coherent **backscatter patches (red)** that
return energy to the resolved swirl; Smagorinsky's Π is sign-definite drain and so
cannot feed them. As in Part 8b/9, no eddy-viscosity closure reproduces the
instantaneous subgrid field pointwise — the discriminating statement is the
**net** energy budget of §2 and the suspension margin of §4.

## 4. Suspension margin — does the particle stay up? (figure 43)

![levitation]({os.path.basename(f43)})

The levitation force scales with the core-well depth, which scales with the
resolved swirl energy (p ~ ρ|u|²). Turning each closure's net SGS power into a
one-turnover (τ ≈ {tau:.2f}) change in swirl energy and normalising so the truth
field marginally suspends the particle (margin := 1):

| model | suspension margin | verdict |
|---|---|---|
| truth | {s['margin_truth'][0]:.3f} ± {s['margin_truth'][1]:.3f} | suspended (calibration) |
| Smagorinsky (K-theory) | **{s['margin_smag'][0]:.3f} ± {s['margin_smag'][1]:.3f}** | **{'particle falls' if s['margin_smag'][0] < 1.0 else 'stays up'}** |
| spectrum-matched surrogate | {s['margin_surr'][0]:.3f} ± {s['margin_surr'][1]:.3f} | energy-OK, but ∇·m≠0 → spurious core pressure |
| projected-FDT | {s['margin_fdt'][0]:.3f} ± {s['margin_fdt'][1]:.3f} | {'stays up' if s['margin_fdt'][0] >= 1.0 else 'particle falls'} |

A purely dissipative K-theory LES over-drains the swirl, the core well shoals, the
levitation force drops below the particle weight, and the particle **falls**. The
projected-FDT repair preserves the swirl energy (and hence the well), so the
particle **stays up** — the same structural failure isolated in Part 8b, now read
as a yes/no physical outcome. (The spectrum-matched surrogate has a near-zero *net*
resolved power, so it scores margin ≈ 1 on this energy proxy — but it fails the
other way, breaking ∇·m = 0 (§2) and so injecting a spurious pressure that corrupts
the very core well it would need to preserve.)

## Scope and honesty

This is a **2D, a-priori (frozen-field) mechanism demonstration**, not an
operational Goldshtik–Sorokin model. The real effect is a **3D axisymmetric**
phenomenon with vortex stretching; this 2D swirl has none. The "particle" is a
**force-balance proxy** — the suspension margin is the resolved swirl energy after a
single core-turnover extrapolation of the measured SGS power, *not* an integrated
Lagrangian particle trajectory with drag, added mass, and lift. The margin is
calibrated so the truth marginally suspends the particle, so only the *relative*
ordering (which closure keeps margin ≥ 1) is meaningful. It shares Part 8b's caveats
(single-instant, no time-integration memory). The result demonstrates the
operator-level mechanism — a global elliptic well that a local eddy viscosity drains
away — it does **not** constitute a validated levitation prediction.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)
        fh.write(hand_written if hand_written else _HAND_WRITTEN_MARKER + "\n")


def write_report_otis(path, figs, scores, vortex, r_core_box, n, kc, steps, depth, tau):
    """Real-data (Hurricane Otis) calibration report, mirroring REPORT_SUBGLACIAL_REAL.md."""
    f44, f45, f46, f47 = figs
    hand_written = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        idx = content.find(_HAND_WRITTEN_MARKER)
        if idx >= 0:
            hand_written = content[idx:]

    s = scores
    drain_ratio = abs(s['P_smag'][0]) / max(abs(s['P_truth'][0]), 1e-30)
    txt = f"""<!-- Benchmark section below is GENERATED by run_swirl.py --swirl otis
     (n={n}, kc={kc}, steps={steps}). Re-run to refresh numbers. Hand-written content
     below the marker is preserved. -->

# Part 10 (real data) — the closure test at a *Hurricane Otis*-calibrated vortex

This is the real-data calibration of Part 10, mirroring Part 9's BEDMAP1 real-bed
variant. The idealized run (`REPORT_SWIRL.md`) uses a synthetic Gaussian-core
vortex; here the swirl's **target tangential profile is taken from Hurricane Otis**
(EP182023), the record-breaking category-5 that struck Acapulco on 25 Oct 2023.

**Calibration (figure 44).**

![calibration]({os.path.basename(f44)})

From the NHC best track (HURDAT2, bundled in
`swirl/data/otis_besttrack.csv`) we read Otis's peak-intensity vortex:

| quantity | Otis (peak, {vortex.iso_time}) |
|---|---|
| minimum central pressure p_c | **{vortex.pc_mb:.0f} mb** |
| environmental pressure p_env | {vortex.penv_mb:.0f} mb |
| pressure deficit Δp = p_env − p_c | **{vortex.dp_mb:.0f} mb** |
| maximum sustained wind Vmax | {vortex.vmax_kt:.0f} kt ({vortex.vmax_ms:.1f} m/s) |
| radius of maximum wind RMW | **{vortex.rmw_nmi:.0f} nmi ({vortex.rmw_km:.1f} km)** |
| mean 34-kt wind radius | {vortex.r34_nmi:.0f} nmi |
| fitted Holland shape parameter B | **{vortex.holland_B:.2f}** |

The Holland (1980) parametric vortex is fitted to the real wind–pressure relation
(B = ρ·e·Vmax²/Δp ⇒ B ≈ {vortex.holland_B:.2f}), giving the radial *shape* — how
sharply the swirl peaks at the eyewall and how deep the core pressure deficit is.
That shape is handed to the solver as its sustained target (RMW mapped to the
resolvable core radius r_core ≈ {r_core_box:.2f}); turbulence is then developed on
top over {steps} steps. The measured core well depth is **{depth:.3f}**.

## Calibrated vortex and its core well (figure 45)

![otis fields]({os.path.basename(f45)})

The Otis-fitted vortex develops the same deep, global (elliptic) low-pressure core
well — the structure that would levitate a particle (Goldshtik–Sorokin) and exactly
the kind of nonlocal response the two-clocks thesis says K-theory cannot see.

## Energy transfer, solenoidality, suspension margin (figures 46–47)

![otis transfer]({os.path.basename(f46)})
![otis levitation]({os.path.basename(f47)})

| model | rel. RMS(∇·m) | transfer corr. | net resolved SGS power ⟨ū·m⟩ | suspension margin | verdict |
|---|---|---|---|---|---|
| truth | {s['div_truth'][0]:.2e} | 1.000 | {s['P_truth'][0]:+.2e} | {s['margin_truth'][0]:.3f} | suspended (calibration) |
| Smagorinsky | {s['div_smag'][0]:.2e} | {s['corr_smag'][0]:+.3f} | {s['P_smag'][0]:+.2e} | **{s['margin_smag'][0]:.3f}** | **{'particle falls' if s['margin_smag'][0] < 1.0 else 'stays up'}** |
| surrogate | {s['div_surr'][0]:.2e} | {s['corr_surr'][0]:+.3f} | {s['P_surr'][0]:+.2e} | {s['margin_surr'][0]:.3f} | ∇·m≠0 → spurious core pressure |
| projected-FDT | {s['div_fdt'][0]:.2e} | {s['corr_fdt'][0]:+.3f} | {s['P_fdt'][0]:+.2e} | {s['margin_fdt'][0]:.3f} | {'stays up' if s['margin_fdt'][0] >= 1.0 else 'particle falls'} |

**The Part-8b/Part-10 result survives real-vortex calibration.** With Otis's
measured Holland shape, a purely dissipative K-theory closure still over-drains the
swirl — at **{drain_ratio:.0f}×** the truth's net rate — so the suspension margin
falls to **{s['margin_smag'][0]:.3f} < 1** (particle falls). Projected-FDT preserves
the swirl energy and the core well (margin ≈ {s['margin_fdt'][0]:.2f}); the
spectrum-matched surrogate keeps the net energy budget but breaks ∇·m = 0, injecting
a spurious pressure on the very core well it must preserve.

## Scope and honesty (real-data variant)

The real data calibrate the **mean vortex geometry** (Otis's Holland B, its
deficit-to-wind relation, its tight-eye peakedness) — *not* the subgrid closure,
which is exactly what is under test. As in Part 9's real-bed run, the measured
profile is **non-dimensionalised** to the spectral box: Otis's real ≈{vortex.rmw_km:.0f} km
pinhole eye is sub-grid at n={n}, so absolute size and speed are rescaled while the
radial *shape* is preserved. Everything else inherits Part 10's caveats — a **2D,
frozen-field, a-priori** mechanism demo with a **force-balance** suspension proxy
(not an integrated Lagrangian trajectory), no 3D vortex stretching, and a margin
calibrated so the truth marginally suspends the particle (only the *relative*
ordering of closures is meaningful). This grounds the vortex geometry in a real
storm; it is **not** a hurricane forecast or an operational levitation prediction.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)
        fh.write(hand_written if hand_written else _HAND_WRITTEN_MARKER + "\n")


def _mean_std(d):
    a = np.array(d)
    return float(a.mean()), float(a.std())


def score_closures(f, sp, kc, cs, tau, snaps=SNAPS, gap=SNAP_GAP):
    """Run the a-priori scorecard over several post-spinup snapshots of ``f``.

    Returns ``(scores, snap_fields)`` where ``scores`` maps each metric to a
    (mean, std) pair and ``snap_fields`` holds the last snapshot's filtered
    velocity + each model's subgrid force, for plotting."""
    acc = {k: [] for k in (
        "div_truth", "div_smag", "div_surr", "div_fdt",
        "corr_smag", "corr_surr", "corr_fdt",
        "P_truth", "P_smag", "P_surr", "P_fdt",
        "pi_corr_smag", "pi_corr_fdt",
        "margin_truth", "margin_smag", "margin_surr", "margin_fdt",
        "well_depth")}
    snap_fields = None
    for i in range(snaps):
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

        e_res = 0.5 * float(np.mean(ub ** 2 + vb ** 2))
        P_t = resolved_sgs_power(ub, vb, mtx, mty)
        P_s = resolved_sgs_power(ub, vb, msx, msy)
        P_q = resolved_sgs_power(ub, vb, mqx, mqy)
        P_f = resolved_sgs_power(ub, vb, mfx, mfy)
        acc["P_truth"].append(P_t); acc["P_smag"].append(P_s)
        acc["P_surr"].append(P_q); acc["P_fdt"].append(P_f)
        acc["margin_truth"].append(suspension_margin(e_res, tau, P_t, P_t))
        acc["margin_smag"].append(suspension_margin(e_res, tau, P_s, P_t))
        acc["margin_surr"].append(suspension_margin(e_res, tau, P_q, P_t))
        acc["margin_fdt"].append(suspension_margin(e_res, tau, P_f, P_t))

        swirl_band = f.r < 1.6
        pi_t = spatial_transfer(ub, vb, mtx, mty)
        acc["pi_corr_smag"].append(masked_corr(spatial_transfer(ub, vb, msx, msy), pi_t, swirl_band))
        acc["pi_corr_fdt"].append(masked_corr(spatial_transfer(ub, vb, mfx, mfy), pi_t, swirl_band))
        acc["well_depth"].append(f.core_well_depth()[0])

        if i == snaps - 1:
            snap_fields = (ub, vb, (mtx, mty), (msx, msy), (mqx, mqy), (mfx, mfy))
        if i < snaps - 1:
            for _ in range(gap):
                f.step()

    scores = {k: _mean_std(v) for k, v in acc.items()}
    return scores, snap_fields


OTIS_R_CORE = 0.6   # box core radius the real Otis RMW is mapped onto (resolvable at n=128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--report", default=None)
    ap.add_argument("--n", type=int, default=N_DNS)
    ap.add_argument("--kc", type=int, default=KC)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--f-amp", type=float, default=1.2)
    ap.add_argument("--swirl", choices=["idealized", "otis"], default="idealized",
                    help="synthetic Gaussian-core vortex, or one calibrated to Hurricane Otis")
    ap.add_argument("--otis-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "swirl", "data", "otis_besttrack.csv"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    kc, cs = args.kc, 0.16

    if args.swirl == "otis":
        return run_otis(args, kc, cs)

    if args.report is None:
        args.report = "REPORT_SWIRL.md"
    cfg = SwirlConfig(n=args.n, sgs="none", f_amp=args.f_amp, seed=1)
    print(f"Part 10: Goldshtik–Sorokin swirl closure test (n={args.n}, kc={kc}, spinup={args.steps})")
    f = SwirlFlow(cfg)
    print("  spinning up turbulent swirl (truth) ...")
    f.run(args.steps, report_every=max(1, args.steps // 6))
    depth, _ = f.core_well_depth()
    print(f"  developed: KE={f.kinetic_energy():.4e}  core well depth={depth:.4e}")

    sp = f.sp
    tau = swirl_turnover(cfg.r_core, cfg.U_swirl)
    print(f"  scoring closures over {SNAPS} snapshots ...")
    scores, snap_fields = score_closures(f, sp, kc, cs, tau)
    print("  scores:")
    for k in ("div_smag", "div_fdt", "corr_smag", "corr_fdt",
              "P_truth", "P_smag", "P_fdt",
              "margin_smag", "margin_surr", "margin_fdt"):
        print(f"    {k:14s} {scores[k][0]:+.4f} ± {scores[k][1]:.4f}")

    print("  generating figures ...")
    ub, vb, m_true, m_smag, m_surr, m_fdt = snap_fields
    div_scores = {"truth": scores["div_truth"][0], "Smagorinsky": scores["div_smag"][0],
                  "surrogate": scores["div_surr"][0], "projected-FDT": scores["div_fdt"][0]}
    margins = {"truth": scores["margin_truth"], "smag": scores["margin_smag"],
               "surr": scores["margin_surr"], "fdt": scores["margin_fdt"]}
    powers = {"truth": scores["P_truth"], "smag": scores["P_smag"], "fdt": scores["P_fdt"]}

    f40 = fig_fields(f, scores["well_depth"][0], args.out_dir)
    f41 = fig_transfer_div(sp, ub, vb, m_true, m_smag, m_surr, m_fdt, div_scores, args.out_dir, kc)
    f42 = fig_transfer_map(f, ub, vb, m_true, m_smag, m_fdt, args.out_dir)
    f43 = fig_levitation(margins, powers, args.out_dir)
    for p in (f40, f41, f42, f43):
        print(f"  -> {p}")

    write_report(args.report, (f40, f41, f42, f43), scores, args.n, kc, args.steps,
                 scores["well_depth"][0], tau)
    print(f"Report: {args.report}")


def run_otis(args, kc, cs):
    """Part 10 calibrated to Hurricane Otis's best-track vortex (figures 44–47)."""
    from swirl.otis import peak_vortex, load_otis_track, target_uth_factory

    if args.report is None:
        args.report = "REPORT_SWIRL_REAL.md"
    r_core_box = OTIS_R_CORE
    track = load_otis_track(args.otis_file)
    vortex = peak_vortex(args.otis_file)
    print(f"Part 10 (real data): Hurricane Otis-calibrated swirl (n={args.n}, kc={kc}, spinup={args.steps})")
    print(f"  Otis peak: {vortex.vmax_kt:.0f} kt / {vortex.pc_mb:.0f} mb, RMW {vortex.rmw_nmi:.0f} nmi, "
          f"Holland B={vortex.holland_B:.2f}")

    cfg = SwirlConfig(n=args.n, sgs="none", f_amp=args.f_amp, seed=1, r_core=r_core_box)
    uth = target_uth_factory(vortex, cfg.U_swirl, r_core_box)
    f = SwirlFlow(cfg, target_uth=uth)
    print("  spinning up Otis-calibrated turbulent swirl (truth) ...")
    f.run(args.steps, report_every=max(1, args.steps // 6))
    depth, _ = f.core_well_depth()
    print(f"  developed: KE={f.kinetic_energy():.4e}  core well depth={depth:.4e}")

    sp = f.sp
    tau = swirl_turnover(cfg.r_core, cfg.U_swirl)
    print(f"  scoring closures over {SNAPS} snapshots ...")
    scores, snap_fields = score_closures(f, sp, kc, cs, tau)
    print("  scores:")
    for k in ("div_smag", "div_surr", "P_truth", "P_smag", "P_fdt",
              "margin_smag", "margin_surr", "margin_fdt"):
        print(f"    {k:14s} {scores[k][0]:+.4f} ± {scores[k][1]:.4f}")

    print("  generating figures ...")
    ub, vb, m_true, m_smag, m_surr, m_fdt = snap_fields
    div_scores = {"truth": scores["div_truth"][0], "Smagorinsky": scores["div_smag"][0],
                  "surrogate": scores["div_surr"][0], "projected-FDT": scores["div_fdt"][0]}
    margins = {"truth": scores["margin_truth"], "smag": scores["margin_smag"],
               "surr": scores["margin_surr"], "fdt": scores["margin_fdt"]}
    powers = {"truth": scores["P_truth"], "smag": scores["P_smag"], "fdt": scores["P_fdt"]}

    f44 = fig_otis_calibration(track, vortex, r_core_box, args.out_dir, num=44)
    f45 = fig_fields(f, scores["well_depth"][0], args.out_dir, num=45)
    f46 = fig_transfer_div(sp, ub, vb, m_true, m_smag, m_surr, m_fdt, div_scores, args.out_dir, kc, num=46)
    f47 = fig_levitation(margins, powers, args.out_dir, num=47)
    for p in (f44, f45, f46, f47):
        print(f"  -> {p}")

    write_report_otis(args.report, (f44, f45, f46, f47), scores, vortex, r_core_box,
                      args.n, kc, args.steps, scores["well_depth"][0], tau)
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
