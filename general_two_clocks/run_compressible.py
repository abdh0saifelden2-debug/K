"""Demonstrate the hyperbolic -> elliptic crossover for the pressure field.

Two experiments on a 2D periodic grid:

1. Radiating pulse: an initial localized pressure bump propagates outward at the
   finite sound speed c (a hyperbolic wave) and fades by geometric spreading.

2. Crossover: under a fixed irrotational forcing f = grad(psi), the compressible
   pressure relaxes to *exactly* the elliptic Poisson field laplacian(p)=rho0 div f,
   but the relaxation time scales as L/c.  As c -> infinity the adjustment becomes
   instantaneous: the elliptic (incompressible / low-Mach) limit.

This is a pedagogical demonstration of the change of PDE type with compressibility.
It does NOT test 3D regularity / Beale-Kato-Majda; that needs rigorous analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compressible.solver import (
    LinearAcoustics2D,
    gaussian_bump,
    gradient_forcing,
    poisson_fft,
)


# ---------------------------------------------------------------------------
# Experiment 1 — radiating pulse
# ---------------------------------------------------------------------------

def fig_radiating_pulse(out_dir: Path, n: int = 256, L: float = 1.0,
                        c: float = 1.0, gamma: float = 0.3) -> Path:
    sim = LinearAcoustics2D(n, L, c, gamma=gamma, cfl=0.4)
    p0 = gaussian_bump(n, L, x0=L / 2, y0=L / 2, sigma=0.03, amp=1.0)
    sim.set_pressure(p0)

    # Snapshot times chosen so the wavefront (radius c*t) is well inside domain.
    snap_times = [0.0, 0.12, 0.24, 0.36]
    snaps = []
    peak_t, peak_amp = [], []
    next_idx = 0
    t_record = np.linspace(0, snap_times[-1], 200)
    rec_i = 0
    while sim.state.t <= snap_times[-1] + sim.dt:
        if next_idx < len(snap_times) and sim.state.t >= snap_times[next_idx] - 1e-9:
            snaps.append((sim.state.t, sim.state.p.copy()))
            next_idx += 1
        if rec_i < len(t_record) and sim.state.t >= t_record[rec_i] - 1e-9:
            peak_t.append(sim.state.t)
            peak_amp.append(np.max(np.abs(sim.state.p)))
            rec_i += 1
        sim.step()

    fig, axes = plt.subplots(1, 5, figsize=(19, 3.8))
    vmax = 0.25
    for ax, (t, p) in zip(axes[:4], snaps):
        im = ax.imshow(p.T, origin="lower", extent=[0, L, 0, L],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"t = {t:.2f}   (wavefront r ≈ {c*t:.2f})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[:4], fraction=0.012, pad=0.01, label="pressure")

    ax = axes[4]
    ax.plot(peak_t, peak_amp, "b-", lw=2)
    ax.set_xlabel("time")
    ax.set_ylabel("peak |pressure|")
    ax.set_title("amplitude fades\n(geometric spreading + damping)")
    ax.grid(alpha=0.3)

    fig.suptitle("Hyperbolic regime: a pressure pulse radiates at finite c and fades",
                 fontsize=13, y=1.02)
    path = out_dir / "16_acoustic_pulse.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Experiment 2 — hyperbolic -> elliptic crossover
# ---------------------------------------------------------------------------

def run_crossover(n: int, L: float, c_values: list[float],
                  damp_per_crossing: float = 1.5,
                  n_crossings: float = 8.0):
    """For each c, relax the acoustic field under a fixed forcing and compare
    against the elliptic Poisson solution.  Returns per-c error histories and
    the shared elliptic field."""
    dx = L / n
    psi = gaussian_bump(n, L, x0=L / 2, y0=L / 2, sigma=0.10, amp=1.0)
    rho0 = 1.0
    fx, fy = gradient_forcing(psi, dx)
    div_f = (np.roll(fx, -1, 0) - np.roll(fx, 1, 0)) / (2 * dx) \
        + (np.roll(fy, -1, 1) - np.roll(fy, 1, 1)) / (2 * dx)
    # Elliptic / instantaneous limit: laplacian(p) = rho0 * div f.
    p_elliptic = poisson_fft(rho0 * div_f, dx)
    p_elliptic -= p_elliptic.mean()
    norm_ell = np.linalg.norm(p_elliptic)

    results = {}
    for c in c_values:
        gamma = damp_per_crossing * c / L
        sim = LinearAcoustics2D(n, L, c, rho0=rho0, gamma=gamma, cfl=0.4)
        t_end = n_crossings * L / c
        t_hist, e_hist = [], []
        while sim.state.t < t_end - 1e-12:
            p = sim.state.p - sim.state.p.mean()
            t_hist.append(sim.state.t)
            e_hist.append(np.linalg.norm(p - p_elliptic) / norm_ell)
            sim.step(fx=fx, fy=fy)
        p_final = sim.state.p - sim.state.p.mean()
        results[c] = {
            "t": np.array(t_hist),
            "err": np.array(e_hist),
            "p_final": p_final,
        }
    return results, p_elliptic, norm_ell


def t_to_threshold(t: np.ndarray, err: np.ndarray, thresh: float) -> float:
    below = np.where(err < thresh)[0]
    return float(t[below[0]]) if below.size else float("nan")


def fig_crossover(results: dict, p_elliptic: np.ndarray, L: float,
                  out_dir: Path) -> tuple[Path, dict]:
    c_values = sorted(results)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(c_values)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    # Panel A: error vs physical time — larger c relaxes faster.
    axA = axes[0]
    t90 = {}
    for c, col in zip(c_values, colors):
        r = results[c]
        axA.plot(r["t"], r["err"], color=col, lw=2, label=f"c = {c:g}")
        t90[c] = t_to_threshold(r["t"], r["err"], 0.10)
    axA.axhline(0.10, color="k", ls=":", lw=1, label="10% error")
    axA.set_xlabel("physical time t")
    axA.set_ylabel("‖p(t) − p_elliptic‖ / ‖p_elliptic‖")
    axA.set_title("Compressible pressure relaxes to the elliptic field\n(faster at higher c)")
    axA.set_xlim(0, max(results[min(c_values)]["t"]))
    axA.legend(fontsize=8)
    axA.grid(alpha=0.3)

    # Panel B: collapse in rescaled time c*t/L — the only role of c is to set the clock.
    # Draw highest c first (thick) and lower c on top (thinner) so the overlap of
    # the collapsed curves is visible rather than hidden.
    axB = axes[1]
    lws = np.linspace(5.0, 1.6, len(c_values))
    for c, col, lw in zip(reversed(c_values), reversed(list(colors)), lws):
        r = results[c]
        axB.plot(r["t"] * c / L, r["err"], color=col, lw=lw, alpha=0.8,
                 label=f"c = {c:g}")
    axB.set_xlabel("rescaled time  c·t / L  (acoustic crossings)")
    axB.set_ylabel("relative error")
    axB.set_title("Curves collapse vs c·t/L\n→ relaxation time ∝ L/c")
    axB.legend(fontsize=8)
    axB.grid(alpha=0.3)

    # Panel C: t90 vs 1/c, expect linear through origin (t90 ∝ L/c).
    axC = axes[2]
    inv_c = np.array([1.0 / c for c in c_values])
    t90_arr = np.array([t90[c] for c in c_values])
    axC.plot(inv_c, t90_arr, "o", color="crimson", ms=9)
    # least-squares line through origin
    slope = float(np.sum(inv_c * t90_arr) / np.sum(inv_c**2))
    xs = np.linspace(0, inv_c.max() * 1.05, 50)
    axC.plot(xs, slope * xs, "k--", lw=1.5,
             label=f"t₉₀ ≈ {slope:.2f}·(L/c)")
    axC.plot(0, 0, "ks", ms=6)
    axC.set_xlabel("1 / c")
    axC.set_ylabel("time to reach 10% error,  t₉₀")
    axC.set_title("Adjustment time → 0 as c → ∞\n(the elliptic limit is instantaneous)")
    axC.legend(fontsize=9)
    axC.grid(alpha=0.3)

    fig.suptitle("Hyperbolic → elliptic crossover: the incompressible Poisson field "
                 "is the c → ∞ limit of finite-speed acoustics", fontsize=12, y=1.02)
    path = out_dir / "17_crossover.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path, {"t90": t90, "slope": slope}


def fig_fields_match(results: dict, p_elliptic: np.ndarray, L: float,
                     out_dir: Path) -> Path:
    c_max = max(results)
    p_final = results[c_max]["p_final"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    vmax = np.max(np.abs(p_elliptic)) * 1.05
    titles = [f"compressible steady state\n(c = {c_max:g})",
              "elliptic Poisson solution\n(c → ∞)",
              "difference"]
    fields = [p_final, p_elliptic, p_final - p_elliptic]
    for ax, fld, ttl in zip(axes, fields, titles):
        im = ax.imshow(fld.T, origin="lower", extent=[0, L, 0, L],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(ttl)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Same destination: high-c acoustic steady state matches the elliptic field",
                 fontsize=12, y=1.02)
    path = out_dir / "18_field_match.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(path: Path, c_values: list[float], stats: dict,
                 final_err: dict, fig_paths: list[Path]) -> None:
    c_sorted = sorted(c_values)
    lines = [
        "# Hyperbolic → elliptic crossover (compressible acoustics)\n",
        "\nA minimal 2D linear-acoustics solver on a periodic grid, used to show "
        "how the pressure field changes PDE type with compressibility.\n",
        "\n## Setup\n",
        "\nLinear acoustics about a fluid at rest:\n",
        "\n```\n"
        "dp/dt = -rho0 c^2 (div u)\n"
        "du/dt = -(1/rho0) grad p + f - gamma u\n"
        "```\n",
        "\n- Finite sound speed `c` ⇒ pressure obeys a **wave equation** "
        "(hyperbolic): signals travel at speed c.\n"
        "- As `c → ∞` the system becomes stiff and the steady-state pressure "
        "satisfies the **elliptic** Poisson equation `∇²p = rho0 ∇·f` — the "
        "incompressible / low-Mach limit.\n",
        "\n## Experiment 1 — radiating pulse\n",
        "\nAn initial localized pressure bump propagates outward as a ring at "
        "speed c and its peak amplitude fades (geometric spreading in 2D + light "
        "damping). This is the finite-speed, wave-like behaviour your acoustic "
        "intuition describes: a local energy anomaly the fluid tries to relax, "
        "radiating away rather than resolving instantly.\n",
        "\n## Experiment 2 — the crossover\n",
        "\nUnder a fixed irrotational forcing `f = ∇ψ`, the compressible pressure "
        "relaxes to the **same** field the elliptic Poisson solve gives, but the "
        "time to get there scales as `L/c`:\n",
        "\n| sound speed c | time to 10% error t₉₀ | final relative error |\n|---|---|---|\n",
    ]
    for c in c_sorted:
        t90 = stats["t90"][c]
        fe = final_err[c]
        lines.append(f"| {c:g} | {t90:.3f} | {fe:.2e} |\n")
    lines += [
        f"\nA line through the origin fits `t₉₀ ≈ {stats['slope']:.2f}·(L/c)`: the "
        "adjustment time is inversely proportional to the sound speed and → 0 as "
        "`c → ∞`. Plotting the error against the rescaled time `c·t/L` collapses "
        "all the curves, confirming that c's only role is to set the clock speed "
        "of the global pressure adjustment.\n",
        "\n## Interpretation\n",
        "\n- **Hyperbolic (finite c):** real, compressible fluids carry pressure "
        "information as sound waves at finite speed. A local divergence anomaly "
        "radiates outward and is resolved over an acoustic crossing time `~L/c`. "
        "In water `c ≈ 1500 m/s` (vs ~340 m/s in air), so this limit is "
        "approached even harder — hydraulic pressure is felt across large "
        "distances almost instantly.\n",
        "- **Elliptic (c → ∞):** in the incompressible limit the same pressure "
        "field is established *instantaneously and globally* by the Poisson "
        "solve. This is the regime of the Rayleigh-Bénard DNS in `REPORT_RB.md`, "
        "where pressure is smooth/global (a low-pass filter) and buoyancy is "
        "sharp/local.\n",
        "\nThe two are the same physics at different compressibility: the elliptic "
        "Poisson constraint is the low-Mach limit of finite-speed acoustics.\n",
        "\n## Scope\n",
        "\nThis **demonstrates** the crossover. It does **not** test 3D "
        "Navier-Stokes regularity / the Beale-Kato-Majda criterion, nor the "
        "wave-radiation-damping argument — those require rigorous PDE analysis "
        "(and, for a real flow, a full nonlinear compressible simulation), not a "
        "linear-acoustics demo.\n",
        "\n## Figures\n",
    ]
    for fp in fig_paths:
        lines.append(f"\n![{fp.stem}]({fp.name})")
    path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--report", default="REPORT_COMPRESSIBLE.md")
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L = 1.0

    print("Experiment 1: radiating pulse ...")
    p_pulse = fig_radiating_pulse(out_dir)
    print(f"  → {p_pulse}")

    print("Experiment 2: hyperbolic → elliptic crossover ...")
    c_values = [4.0, 8.0, 16.0, 32.0]
    results, p_elliptic, _ = run_crossover(args.n, L, c_values)
    final_err = {c: float(results[c]["err"][-1]) for c in c_values}
    for c in c_values:
        print(f"  c = {c:5g}  final rel. error = {final_err[c]:.2e}")
    p_cross, stats = fig_crossover(results, p_elliptic, L, out_dir)
    p_match = fig_fields_match(results, p_elliptic, L, out_dir)
    print(f"  t90 ≈ {stats['slope']:.3f}·(L/c)")
    print(f"  → {p_cross}\n  → {p_match}")

    report = Path(args.report)
    write_report(report, c_values, stats, final_err,
                 [p_pulse, p_cross, p_match])
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
