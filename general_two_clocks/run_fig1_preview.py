"""Figure-1 preview composite for the manuscript (paper0, Fig.~1).

The projection/surrogate demonstration is the paper's central result, so the
manuscript previews it on page 1 as a four-panel composite:

    (a) the true correction the elliptic projection removes (u-component of the
        divergent part of the slow drift u*);
    (b) the power spectra of the true correction and of a phase-randomized
        surrogate --- identical by construction (Parseval);
    (c) the surrogate correction field (identical spectrum, scrambled phases);
    (d) RMS divergence left by no correction, by the surrogate (ensemble
        median, n = 100), and by the true projection.

Every panel is computed by the same deterministic Part-6 Boussinesq run that
generates figures 22-24 and REPORT_BOUSSINESQ.md (n = 128, t_end = 6.0, fixed
seeds; run_boussinesq.py), so the numbers printed on the panels reproduce that
report exactly.  This script only re-lays the panels.

Usage:
    python general_two_clocks/run_fig1_preview.py \
        [out_dir, default papers/tex/figures]

Writes fig01_preview_surrogate.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from run_boussinesq import simulate, DT_CAP, _rms  # noqa: E402
from boussinesq.solver import (  # noqa: E402
    divergence, project, radial_spectrum, phase_randomized,
)
from compressible.ns import helmholtz  # noqa: E402

N = 128
T_END = 6.0
N_ENS = 100
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE.parent / "papers/tex/figures"


def _fmt(v: float) -> str:
    """Two-significant-figure scientific notation, e.g. 2.75e-02."""
    return f"{v:.2e}"


def main() -> None:
    print(f"Boussinesq run for the Fig.~1 preview (n={N}, t_end={T_END}) ...")
    solver, st = simulate(N, T_END)
    sp = solver.sp

    # --- reproduce the figure-23/24 state exactly ---------------------------
    us, vs = solver.drift_velocity(st, DT_CAP)      # slow drift (divergent)
    up, vp = project(sp, us, vs)                    # fast elliptic clock
    rms_star = _rms(divergence(sp, us, vs))
    rms_proj = _rms(divergence(sp, up, vp))

    # the correction the projection removes = the gradient part of u*
    _, _, ud, vd = helmholtz(sp, us, vs)
    xu = phase_randomized(sp, ud, seed=11)          # spectrum-matched surrogate
    xv = phase_randomized(sp, vd, seed=12)

    ku, E_true = radial_spectrum(sp, ud)
    _, E_surr = radial_spectrum(sp, xu)

    # 100-surrogate ensemble (same scheme as REPORT_BOUSSINESQ.md)
    corrs, rmss = [], []
    for s in range(N_ENS):
        xu_s = phase_randomized(sp, ud, seed=1000 + 2 * s)
        xv_s = phase_randomized(sp, vd, seed=1001 + 2 * s)
        corrs.append(float(np.corrcoef(ud.ravel(), xu_s.ravel())[0, 1]))
        rmss.append(_rms(divergence(sp, us - xu_s, vs - xv_s)))
    corrs, rmss = np.asarray(corrs), np.asarray(rmss)
    rms_surr = float(np.median(rmss))
    corr_abs_p95 = float(np.percentile(np.abs(corrs), 95))
    print(f"  RMS div: no correction {_fmt(rms_star)}, projection {_fmt(rms_proj)}, "
          f"surrogate {_fmt(rms_surr)}; |corr| median "
          f"{np.median(np.abs(corrs)):.3f}, p95 {corr_abs_p95:.3f} (n={N_ENS})")

    # --- four-panel composite ----------------------------------------------
    fig = plt.figure(figsize=(12.0, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.22,
                          left=0.07, right=0.95, top=0.93, bottom=0.09)

    # (a) true correction field
    ax_a = fig.add_subplot(gs[0, 0])
    vmax = float(np.percentile(np.abs(ud), 99.5))
    im = ax_a.imshow(ud.T, origin="lower", extent=[0, sp.L, 0, sp.L],
                     cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax_a.set_title("(a) true projection correction "
                   "(divergent part of $\\mathbf{u}^*$,\n"
                   "removed by one elliptic solve)", fontsize=10)
    ax_a.set_xlabel("x"); ax_a.set_ylabel("y")
    fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)

    # (b) spectra: identical by construction
    ax_b = fig.add_subplot(gs[0, 1])
    m = ku > 0
    ax_b.loglog(ku[m], E_true[m], "o-", color="teal", ms=4,
                label="true correction")
    ax_b.loglog(ku[m], E_surr[m], "x--", color="crimson", ms=5,
                label="surrogate (random phases)")
    ax_b.set_title("(b) power spectra: identical by construction (Parseval)",
                   fontsize=10)
    ax_b.set_xlabel("radial wavenumber $k$")
    ax_b.set_ylabel("shell-summed power $E(k)$")
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3, which="both")

    # (c) surrogate field, same spectrum
    ax_c = fig.add_subplot(gs[1, 0])
    vmax = float(np.percentile(np.abs(xu), 99.5))
    im = ax_c.imshow(xu.T, origin="lower", extent=[0, sp.L, 0, sp.L],
                     cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax_c.set_title("(c) phase-randomized surrogate\n"
                   "(identical spectrum, scrambled phases)", fontsize=10)
    ax_c.set_xlabel("x"); ax_c.set_ylabel("y")
    fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)

    # (d) divergence after each treatment
    ax_d = fig.add_subplot(gs[1, 1])
    labels = ["no correction\n(div of $\\mathbf{u}^*$)",
              "surrogate\n(ensemble median, $n=100$)",
              "true\nprojection"]
    vals = [rms_star, rms_surr, rms_proj]
    colors = ["0.55", "crimson", "teal"]
    xpos = np.arange(3)
    ax_d.bar(xpos, vals, width=0.6, color=colors, log=True)
    for x, v in zip(xpos, vals):
        ax_d.text(x, v * 2.0, f"{_fmt(v)}", ha="center", va="bottom",
                  fontsize=9)
    ax_d.set_xticks(xpos)
    ax_d.set_xticklabels(labels, fontsize=8.5)
    ax_d.set_ylim(top=max(vals) * 60.0)
    ax_d.set_title("(d) RMS divergence $\\|\\nabla\\cdot\\mathbf{u}\\|$ left "
                   "by each correction", fontsize=10)
    ax_d.set_ylabel("RMS $\\nabla\\cdot\\mathbf{u}$ (log)")
    ax_d.grid(axis="y", alpha=0.3, which="both")

    fig.suptitle("Spectral equivalence is not dynamical equivalence: an elliptic "
                 "constraint is carried by phases, not by energy",
                 fontsize=12, y=0.985)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "fig01_preview_surrogate.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
