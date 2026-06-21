"""Part 6 -- Synthesis & the SPDE limit.
 
Reunites the two clocks in a single nonlinear flow using the projection
(fractional-step) method, the canonical incompressible algorithm:
 
  1. advance the slow, local advective-diffusive *drift* (the deterministic
     generator of the local Markov / heat semigroup) to an intermediate
     velocity u* -- which is generally *divergent* (the blind spot of local,
     memoryless forecasting);
  2. apply one elliptic Leray projection (the fast, boundary-aware pressure
     clock) to make it instantly divergence-free and structure-respecting;
  3. contrast that deterministic projection with an SPDE-style stochastic
     surrogate that matches the *spectrum* of the removed correction but not
     its structure -- "energy yes, structure no".
 
Generates figures 22-24 and REPORT_BOUSSINESQ.md.  Needs no downloaded data.
"""
 
from __future__ import annotations
 
import argparse
from pathlib import Path
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
 
from boussinesq.solver import (  # noqa: E402
    BoussinesqProjection, BoussinesqState, warm_blobs, divergence, project,
    projection_potential, radial_spectrum, phase_randomized)
from compressible.ns import helmholtz  # noqa: E402
 
NU = 2e-3
KAPPA = 2e-3
DT_CAP = 0.02
 
 
def simulate(n, t_end):
    solver = BoussinesqProjection(n, NU, KAPPA, cfl=0.4)
    u, v, b = warm_blobs(solver.sp)
    u, v = project(solver.sp, u, v)
    st = BoussinesqState(u, v, b, 0.0)
    while st.t < t_end:
        dt = min(solver.dt_cfl(st), DT_CAP)
        solver.step(st, dt)
        if not np.isfinite(st.u).all():
            raise RuntimeError(f"diverged at t={st.t}")
    return solver, st
 
 
def _rms(a):
    return float(np.sqrt(np.mean(a ** 2)))
 
 
# ---------------------------------------------------------------------------
# Figure 22 -- the developed mesoscale convection cycle
# ---------------------------------------------------------------------------
 
def fig_convection(solver, st, out_dir):
    sp = solver.sp
    xs = np.arange(sp.n) * sp.dx
    omega = sp.ddx(st.v) - sp.ddy(st.u)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
 
    im0 = axes[0].imshow(st.b.T, origin="lower", extent=[0, sp.L, 0, sp.L],
                         cmap="inferno", aspect="equal")
    axes[0].streamplot(xs, xs, st.u.T, st.v.T, color="cyan", density=1.1,
                       linewidth=0.6, arrowsize=0.7)
    axes[0].set_title("buoyancy (warm rises) + velocity streamlines\n"
                      "plumes lift; cooler fluid is entrained inward")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="buoyancy b")
 
    vmax = float(np.percentile(np.abs(omega), 99))
    im1 = axes[1].imshow(omega.T, origin="lower", extent=[0, sp.L, 0, sp.L],
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[1].set_title("vorticity omega = d_x v - d_y u\n"
                      "shear between rising/inflowing streams rolls up")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="omega")
 
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f"Mesoscale convection cell (t = {st.t:.1f}): diffuse -> buoyancy "
                 "-> plume -> pressure-driven entrainment -> shear",
                 fontsize=12, y=1.02)
    path = out_dir / "22_boussinesq_convection.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
 
 
# ---------------------------------------------------------------------------
# Figure 23 -- the projection step (the two clocks in one update)
# ---------------------------------------------------------------------------
 
def fig_projection_step(solver, st, out_dir):
    sp = solver.sp
    us, vs = solver.drift_velocity(st, DT_CAP)          # slow drift, no pressure
    div_star = divergence(sp, us, vs)
    up, vp = project(sp, us, vs)                          # fast elliptic clock
    div_proj = divergence(sp, up, vp)
    phi = projection_potential(sp, us, vs)
 
    rms_star, rms_proj = _rms(div_star), _rms(div_proj)
    vmax = float(np.percentile(np.abs(div_star), 99.5))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
 
    im0 = axes[0].imshow(div_star.T, origin="lower", extent=[0, sp.L, 0, sp.L],
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[0].set_title(f"slow drift u*: div(u*)\nRMS = {rms_star:.2e}  (NOT zero)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
 
    im1 = axes[1].imshow(div_proj.T, origin="lower", extent=[0, sp.L, 0, sp.L],
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[1].set_title(f"after Leray projection: div(P u*)\n"
                      f"RMS = {rms_proj:.1e}  (machine zero)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
 
    im2 = axes[2].imshow(phi.T, origin="lower", extent=[0, sp.L, 0, sp.L],
                         cmap="viridis", aspect="equal")
    axes[2].set_title("elliptic projection potential phi\n"
                      "(lap phi = div u*) -- the global pressure clock")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
 
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle("One Poisson solve (fast clock) makes the divergent slow drift "
                 f"globally consistent: RMS div {rms_star:.2e} -> {rms_proj:.1e}",
                 fontsize=12, y=1.03)
    path = out_dir / "23_projection_step.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path, rms_star, rms_proj
 
 
# ---------------------------------------------------------------------------
# Figure 24 -- the SPDE limit: energy yes, structure no
# ---------------------------------------------------------------------------
 
def fig_spde_limit(solver, st, out_dir):
    sp = solver.sp
    us, vs = solver.drift_velocity(st, DT_CAP)
    up, vp = project(sp, us, vs)
    div_proj = divergence(sp, up, vp)
 
    # the correction the projection removes (the curl-free / gradient part)
    _, _, ud, vd = helmholtz(sp, us, vs)
    xu = phase_randomized(sp, ud, seed=11)
    xv = phase_randomized(sp, vd, seed=12)
    # stochastic surrogate "balanced" field: replace deterministic correction
    # (subtract ud) with a spectrum-matched random field
    us_surr, vs_surr = us - xu, vs - xv
    div_surr = divergence(sp, us_surr, vs_surr)
 
    ku, E_true = radial_spectrum(sp, ud)
    _, E_surr = radial_spectrum(sp, xu)
    corr = float(np.corrcoef(ud.ravel(), xu.ravel())[0, 1])
    rms_proj, rms_surr = _rms(div_proj), _rms(div_surr)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
 
    m = ku > 0
    axes[0].loglog(ku[m], E_true[m], "o-", color="teal", ms=4,
                   label="true correction (removed by P)")
    axes[0].loglog(ku[m], E_surr[m], "x--", color="crimson", ms=5,
                   label="SPDE surrogate (random phases)")
    axes[0].set_xlabel("radial wavenumber k")
    axes[0].set_ylabel("shell-summed power E(k)")
    axes[0].set_title(f"spectra identical (Parseval)\ncorr(fields) = {corr:.3f} ~ 0")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
 
    vmax = float(np.percentile(np.abs(div_surr), 99.5))
    im1 = axes[1].imshow(div_proj.T, origin="lower", extent=[0, sp.L, 0, sp.L],
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[1].set_title(f"deterministic projection\ndiv RMS = {rms_proj:.1e} (= 0)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
 
    im2 = axes[2].imshow(div_surr.T, origin="lower", extent=[0, sp.L, 0, sp.L],
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[2].set_title(f"spectrum-matched surrogate\ndiv RMS = {rms_surr:.2e} (NOT 0)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
 
    fig.suptitle("Energy yes, structure no: matching the spectrum of the fast-clock "
                 "correction fails the pointwise div(u)=0 / elliptic test",
                 fontsize=12, y=1.03)
    path = out_dir / "24_spde_limit.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path, corr, rms_proj, rms_surr
 
 
# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
 
def write_report(report_path, fig_paths, stats):
    rms_star, rms_proj, corr, rms_surr = stats
    lines = [
        "# Part 6 -- Synthesis: the two clocks reunified by projection\n",
        "\nThe two clocks of the manuscript -- a slow, local, parabolic "
        "*temperature/advection* clock and a fast, global, elliptic *pressure* "
        "clock -- are reunited in one nonlinear flow using the **projection "
        "(fractional-step) method**, the canonical incompressible algorithm. The "
        "key point: this algorithm is not an arbitrary numerical trick, it is the "
        "two-clocks structure made literal.\n",
        "\n## The unified update\n",
        "\n```\n"
        "u*       = u^n + dt * [ -(u.grad)u + nu*lap(u) + (b-<b>) e_y ]   (slow drift)\n"
        "u^{n+1}  = P u*                                                  (fast projection)\n"
        "b^{n+1}  = b^n + dt * [ -(u.grad)b + kappa*lap(b) ]              (temperature clock)\n"
        "```\n",
        "\nThe drift is the **deterministic generator of the local Markov / heat "
        "semigroup** (its heat kernel is the Gaussian transition density) -- the "
        "*perfect* localized, memoryless forecaster, with no machine-learning or "
        "statistical noise to muddy the argument. The Leray projector "
        "`P = I - grad (lap)^-1 div` is exactly the divergence-free part of the "
        "Helmholtz split (`compressible/ns.py`).\n",
        "\n## 1. The mesoscale convection cycle (figure 22)\n",
        "\nFrom warm buoyancy blobs the flow reproduces the full mesoscale cell: "
        "heat diffuses, buoyant fluid lifts as plumes, the pressure field drives "
        "cooler fluid inward to fill the space (entrainment), and the shear "
        "between rising and inflowing streams rolls up into vortices.\n",
        "\n## 2. The slow drift is divergent; one Poisson solve fixes it (figure 23)\n",
        "\nThe intermediate velocity `u*` from the slow drift alone does **not** "
        f"conserve mass: RMS div(u*) = {rms_star:.2e}. This is the blind spot of "
        "any purely local, memoryless model -- it moves fluid based on local heat "
        "and history without knowing the global incompressibility constraint. A "
        "single elliptic Leray projection (the fast clock, which instantly 'feels' "
        "the whole domain) drops the divergence to machine zero: RMS div(P u*) = "
        f"{rms_proj:.1e}. The projection potential `phi` (lap phi = div u*) is the "
        "global elliptic pressure that does the work.\n",
        "\n## 3. The SPDE limit -- energy yes, structure no (figure 24)\n",
        "\nModern stochastic-climate / SPDE closures replace the unresolved fast "
        "clock with random forcing tuned to the right *spectrum*. We mimic this by "
        "replacing the deterministic correction (the gradient field `P` removes) "
        "with a phase-randomized surrogate of **identical power spectrum**. The "
        "result is decisive:\n",
        "\n- **Spectrum:** identical by construction (Parseval) -- the surrogate "
        "carries exactly the right energy at every wavenumber.\n",
        f"- **Structure:** pointwise correlation with the true correction is "
        f"{corr:.3f} ~ 0 -- the field is geometrically unrelated.\n",
        f"- **Constraint:** the deterministic projection gives RMS div = "
        f"{rms_proj:.1e} (= 0); the spectrum-matched surrogate leaves RMS div = "
        f"{rms_surr:.2e} -- it does **not** enforce incompressibility.\n",
        "\nMatching a spectrum is not the same as capturing the physics: the fast "
        "elliptic clock is a *boundary-aware, globally-coupled* operator, and no "
        "amount of correctly-coloured local noise reconstructs it. This is, "
        "concretely, why statistical anomaly predictors (Markov chains, HMMs) and "
        "noise-injection SPDE closures remain blind to events where the pressure "
        "clock instantly rewrites the system's geometry.\n",
        "\n## Scope\n",
        "\nA 2D pedagogical demonstration of how the projection method *is* the "
        "two-clocks synthesis, and of the structural ceiling of spectrum-matching "
        "surrogates. It does **not** prove 3D regularity / Beale-Kato-Majda, and "
        "the 'Markov chain' is the advection-diffusion operator, not a data-trained "
        "transition matrix.\n",
        "\n## Figures\n",
    ]
    for fp in fig_paths:
        name = Path(fp).stem
        lines.append(f"\n![{name}]({Path(fp).name})\n")
    report_path.write_text("".join(lines))
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--report", default="REPORT_BOUSSINESQ.md")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--t-end", type=float, default=6.0)
    args = ap.parse_args()
 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"Boussinesq projection run: n={args.n}, t_end={args.t_end} ...")
    solver, st = simulate(args.n, args.t_end)
    umax = float(np.max(np.sqrt(st.u ** 2 + st.v ** 2)))
    print(f"  developed: t={st.t:.2f}, umax={umax:.3f}, "
          f"div RMS={_rms(divergence(solver.sp, st.u, st.v)):.1e}")
 
    p22 = fig_convection(solver, st, out_dir)
    p23, rms_star, rms_proj = fig_projection_step(solver, st, out_dir)
    p24, corr, rms_proj2, rms_surr = fig_spde_limit(solver, st, out_dir)
    for p in (p22, p23, p24):
        print(f"  -> {p}")
    print(f"  div(u*)={rms_star:.2e}  div(P u*)={rms_proj:.1e}  "
          f"corr(surrogate)={corr:.3f}  div(surrogate)={rms_surr:.2e}")
 
    write_report(Path(args.report), [p22, p23, p24],
                 (rms_star, rms_proj, corr, rms_surr))
    print(f"Report: {args.report}")
 
 
if __name__ == "__main__":
    main()
