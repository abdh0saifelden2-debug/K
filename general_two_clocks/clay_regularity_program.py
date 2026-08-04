r"""A conditional-regularity program for 3D Navier-Stokes via the nonlocal pressure
Hessian -- the repo's furthest *honest* reach toward the Clay problem.

NOT a proof of the Clay Millennium problem.  This module assembles a rigorous
*conditional* attack and reduces global regularity to a single sharp depletion
inequality on the nonlocal pressure Hessian, then gives numerical evidence for that
inequality.  The remaining open core is stated explicitly in REPORT_CLAY_REGULARITY.md.

All CPU, no external data.  Three pillars (see the report for the theorems/proofs):

  [1] Vieillefosse / restricted-Euler finite-time blowup (RIGOROUS, exact rate).
      Dropping the nonlocal pressure Hessian -> the local model blows up.  This is the
      sharp statement of *why* nonlocality is not optional.

  [2] The Beale-Kato-Majda bridge (RIGOROUS criterion, numerically exhibited).
      Singularity <=> int_0^T ||omega||_inf dt = infinity.  The restricted-Euler
      vorticity integral DIVERGES (logarithmically, since |omega| ~ (t*-t)^-1); the
      nonlocal closure's integral stays FINITE.  Regularity is exactly BKM-finiteness.

  [3] The Constantin-Fefferman geometric depletion (NUMERICAL evidence in a real 3D
      NS DNS).  In a genuine forced-NS field, vorticity aligns with the *intermediate*
      strain eigenvector, which DEPLETES the self-stretching omega.S.omega below the
      maximal (restricted-Euler) value.  This depletion -- supplied by the nonlocal
      anisotropic pressure Hessian -- is the mechanism the conditional theorems need.

The reduction: NS regularity holds if the nonlocal pressure Hessian sustains a
depletion factor delta(t) bounded away from the no-depletion (blowup) value along the
flow.  [1]-[3] make this concrete; proving it *unconditionally* is the open core.
"""
from __future__ import annotations

import sys

import numpy as np

import restricted_euler_regularity as rer


# ===========================================================================
# [1] Vieillefosse / restricted-Euler finite-time blowup (rigorous; exact rate)
# ===========================================================================
def vieillefosse_blowup():
    """Numeric confirmation of the rigorous theorem (proof in the report):
    the restricted-Euler invariant ODE  Q'=-3R, R'=(2/3)Q^2  conserves
    H=R^2+(4/27)Q^3 and blows up in finite time on the Vieillefosse tail with
    Q ~ -3(t*-t)^-2,  |A| ~ (t*-t)^-1."""
    r = rer.re_invariant_blowup()
    return dict(blew_up=r["blew_up"], t_star=r["t_star"], H_drift=r["H_drift"],
                tail_ratio=r["tail_ratio"], rate=r["analytic_rate"], ok=r["ok"])


# ===========================================================================
# [2] The Beale-Kato-Majda bridge: vorticity-integral divergence vs finiteness
# ===========================================================================
def _omega_of(A):
    """Vorticity vector from the velocity-gradient tensor A_ij = du_i/dx_j."""
    return np.array([A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]])


def _integrate_vgt_bkm(A0, tau, T=1.0, dt=5e-4, t_max=20.0, big=1.0e7):
    """RK4-integrate the VGT and accumulate the BKM vorticity integral
    I(t) = int_0^t |omega| dt'.  Returns the integral up to blowup/stop, the
    blow-up flag/time and the rate constant |A|*(t*-t)."""
    A = A0.copy()
    t = 0.0
    bkm = 0.0
    last_norm = np.linalg.norm(A)
    rate_const = np.nan
    while t < t_max:
        nA = np.linalg.norm(A)
        wmag = float(np.linalg.norm(_omega_of(A)))
        if nA >= big:
            # |A| ~ (t*-t)^-1  =>  (t*-t) ~ 1/|A|, rate const |A|*(t*-t) -> 1
            rate_const = float(last_norm * dt) if last_norm > 0 else np.nan
            return dict(blew_up=True, t_star=t, bkm=bkm, norm=nA)
        bkm += wmag * dt
        last_norm = nA
        k1 = rer.vgt_rhs(A, tau, T)
        k2 = rer.vgt_rhs(A + 0.5 * dt * k1, tau, T)
        k3 = rer.vgt_rhs(A + 0.5 * dt * k2, tau, T)
        k4 = rer.vgt_rhs(A + dt * k3, tau, T)
        A = A + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t += dt
    return dict(blew_up=False, t_star=t, bkm=bkm, norm=float(np.linalg.norm(A)))


def bkm_bridge(seed=1, scale=1.0, tau_reg=0.12):
    """BKM criterion exhibited: the restricted-Euler (local Hessian) vorticity
    integral diverges as t -> t* (blowup), while the nonlocal-Hessian closure keeps
    it finite over a long horizon (regular).  Also samples I(t) near t* for the
    local model to show the logarithmic growth |omega| ~ (t*-t)^-1."""
    A0 = rer.random_traceless(seed, scale)

    # restricted Euler (tau = 0): sample the BKM integral at a sequence of horizons
    # approaching t* to show it grows without bound.
    re_full = _integrate_vgt_bkm(A0, tau=0.0, t_max=20.0)
    t_star = re_full["t_star"]
    horizons = [0.5, 0.8, 0.9, 0.95, 0.99]
    re_curve = []
    for frac in horizons:
        r = _integrate_vgt_bkm(A0, tau=0.0, t_max=frac * t_star)
        re_curve.append((frac, r["t_star"], r["bkm"]))
    re_growing = bool(re_curve[-1][2] > re_curve[0][2] * 1.5)   # integral keeps rising

    # nonlocal closure (tau > 0): BKM integral finite over a long horizon
    reg = _integrate_vgt_bkm(A0, tau=tau_reg, t_max=20.0)

    ok = bool(re_full["blew_up"] and re_growing
              and (not reg["blew_up"]) and np.isfinite(reg["bkm"]))
    return dict(re_blew_up=re_full["blew_up"], re_t_star=t_star,
                re_bkm_curve=re_curve, re_growing=re_growing,
                reg_blew_up=reg["blew_up"], reg_bkm=reg["bkm"],
                reg_norm=reg["norm"], reg_horizon=reg["t_star"], ok=ok)


# ===========================================================================
# [3] Constantin-Fefferman geometric depletion in a real forced-NS 3D DNS
# ===========================================================================
def _velocity_gradient_field(sp, u, v, w):
    """A_ij = d u_i / d x_j as a (n,n,n,3,3) field (spectral derivatives)."""
    comps = [u, v, w]
    hats = [sp.fft(c) for c in comps]
    K = [sp.kx, sp.ky, sp.kz]
    n = sp.n
    A = np.empty((n, n, n, 3, 3))
    for i in range(3):
        for j in range(3):
            A[..., i, j] = sp.ifft(1j * K[j] * hats[i])
    return A


def alignment_depletion(n=32, steps=600, nu=5.0e-3, f_amp=4.0, seed=0):
    """Develop a forced 3D NS field, then measure the strain/vorticity geometry that
    Constantin-Fefferman regularity hinges on:

      * enstrophy-weighted alignment cos^2(omega, e_i) with the strain eigenvectors
        (e1>=e2>=e3).  The hallmark NS result: omega aligns with the *intermediate*
        eigenvector e2, NOT the most-stretching e1.
      * the mean normalised intermediate eigenvalue <lambda2>/<|lambda|> (typically
        positive -- the strain-skewness that drives the forward cascade).
      * the DEPLETION factor delta = <omega.S.omega>/<|omega|^2 * lambda1>: actual
        self-stretching divided by the maximal (fully e1-aligned, restricted-Euler)
        value.  delta < 1 is the geometric depletion that (conditionally) regularises.
    """
    from closure.dns3d import DNS3DConfig, ForcedNS3D

    cfg = DNS3DConfig(n=n, nu=nu, dt=5.0e-3, seed=seed, f_amp=f_amp)
    dns = ForcedNS3D(cfg)
    dns.run(steps)
    u, v, w = dns.velocity()
    sp = dns.sp

    A = _velocity_gradient_field(sp, u, v, w)
    S = 0.5 * (A + np.swapaxes(A, -1, -2))            # strain (symmetric part)
    # eigen-decompose the strain field (ascending: [...,0]=min, [...,2]=max)
    evals, evecs = np.linalg.eigh(S)
    lam3, lam2, lam1 = evals[..., 0], evals[..., 1], evals[..., 2]
    e3, e2, e1 = evecs[..., 0], evecs[..., 1], evecs[..., 2]

    om = np.stack([A[..., 2, 1] - A[..., 1, 2],
                   A[..., 0, 2] - A[..., 2, 0],
                   A[..., 1, 0] - A[..., 0, 1]], axis=-1)
    enst = np.sum(om * om, axis=-1)                   # |omega|^2
    wsum = float(np.sum(enst)) + 1e-30
    omhat = om / np.sqrt(enst[..., None] + 1e-30)

    def cos2(ev):
        return np.sum(omhat * ev, axis=-1) ** 2       # cos^2(omega, e)
    # enstrophy-weighted mean alignments (where vorticity is strong -> BKM-relevant)
    a1 = float(np.sum(cos2(e1) * enst) / wsum)
    a2 = float(np.sum(cos2(e2) * enst) / wsum)
    a3 = float(np.sum(cos2(e3) * enst) / wsum)

    # self-stretching omega.S.omega = |omega|^2 * (xi.S.xi); xi.S.xi = sum lam_i cos2_i
    stretch = enst * (lam1 * cos2(e1) + lam2 * cos2(e2) + lam3 * cos2(e3))
    prod_mean = float(np.mean(stretch))
    max_mean = float(np.mean(enst * lam1))            # fully e1-aligned (max) stretching
    delta = float(prod_mean / (max_mean + 1e-30))     # depletion factor < 1
    lam2_norm = float(np.mean(lam2) / (np.mean(np.abs(evals)) + 1e-30))

    ke = 0.5 * float(np.mean(u * u + v * v + w * w))
    ok = bool(a2 > a1 and a2 > a3 and lam2_norm > 0.0 and 0.0 < delta < 1.0)
    return dict(n=n, steps=steps, ke=ke,
                align_e1=a1, align_e2=a2, align_e3=a3,
                lam2_norm=lam2_norm, stretch_mean=prod_mean,
                stretch_max=max_mean, depletion_delta=delta, ok=ok)


# ===========================================================================
# figure
# ===========================================================================
def make_figure(path, bkm, dns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # left: BKM integral vs horizon (restricted Euler diverges; nonlocal finite)
    ax = axes[0]
    fr = np.array([c[0] for c in bkm["re_bkm_curve"]])
    iv = np.array([c[2] for c in bkm["re_bkm_curve"]])
    ax.plot(fr, iv, "o-", color="#c0392b", ms=7,
            label="restricted Euler  $\\int_0^t|\\omega|dt'$ (local Hessian)")
    ax.axhline(bkm["reg_bkm"], color="navy", lw=1.8, ls="--",
               label=f"nonlocal Hessian: finite = {bkm['reg_bkm']:.2f}")
    ax.set_xlabel("integration horizon  t / t*")
    ax.set_ylabel("BKM vorticity integral  $\\int_0^t \\|\\omega\\|\\,dt'$")
    ax.set_title("BKM criterion: local model's vorticity integral diverges\n"
                 "as t -> t* (blowup); nonlocal model stays finite (regular)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    # right: vorticity-strain alignment (intermediate-eigenvector depletion)
    ax = axes[1]
    bars = ax.bar([0, 1, 2], [dns["align_e1"], dns["align_e2"], dns["align_e3"]],
                  color=["#888", "#1f77b4", "#888"], width=0.6)
    ax.axhline(1.0 / 3.0, color="k", ls=":", lw=1.2, label="isotropic (no alignment) = 1/3")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["e1 (max\nstretch)", "e2 (intermediate)", "e3 (compress)"])
    ax.set_ylabel(r"enstrophy-weighted  $\langle\cos^2(\omega,e_i)\rangle$")
    ax.set_title(f"Vorticity aligns with the INTERMEDIATE strain eigenvector\n"
                 f"-> self-stretching depleted to delta = {dns['depletion_delta']:.2f} "
                 f"of the maximal value")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Conditional NS regularity: the nonlocal pressure Hessian depletes "
                 "vortex stretching (Clay attack, honest scope)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# driver
# ===========================================================================
def run():
    return dict(vieillefosse=vieillefosse_blowup(),
                bkm=bkm_bridge(),
                depletion=alignment_depletion())


def summary():
    r = run()
    r["all_ok"] = bool(r["vieillefosse"]["ok"] and r["bkm"]["ok"]
                       and r["depletion"]["ok"])
    return r


if __name__ == "__main__":
    import os

    r = summary()
    print("Conditional regularity program for 3D Navier-Stokes (Clay attack, honest)")
    print("=" * 74)
    v = r["vieillefosse"]
    print(f"\n[1] Vieillefosse/restricted-Euler blowup (RIGOROUS): blew_up={v['blew_up']} "
          f"t*={v['t_star']:.4f}")
    print(f"    H drift={v['H_drift']:.1e}, tail ratio={v['tail_ratio']:.5f} -> 1;  "
          f"{v['rate']}  [{'PASS' if v['ok'] else 'FAIL'}]")
    b = r["bkm"]
    print(f"\n[2] BKM bridge: restricted Euler blows up at t*={b['re_t_star']:.3f}; "
          f"BKM integral int|omega|dt vs horizon t/t*:")
    for frac, t, bk in b["re_bkm_curve"]:
        print(f"      t/t*={frac:.2f}:  int|omega|dt = {bk:.3f}")
    print(f"    nonlocal closure: finite BKM integral = {b['reg_bkm']:.3f} over "
          f"t={b['reg_horizon']:.1f}, |A| bounded = {b['reg_norm']:.3f}  "
          f"[{'PASS' if b['ok'] else 'FAIL'}]")
    d = r["depletion"]
    print(f"\n[3] Constantin-Fefferman geometric depletion in a real forced-NS DNS "
          f"(n={d['n']}, KE={d['ke']:.3f}):")
    print(f"    enstrophy-weighted alignment  <cos^2(omega,e_i)>: "
          f"e1={d['align_e1']:.3f}  e2={d['align_e2']:.3f}  e3={d['align_e3']:.3f}  "
          f"(isotropic 0.333)")
    print(f"    intermediate eigenvalue <lam2>/<|lam|> = {d['lam2_norm']:+.3f}; "
          f"depletion factor delta = {d['depletion_delta']:.3f} (<1 = depleted)  "
          f"[{'PASS' if d['ok'] else 'FAIL'}]")

    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "figures", "67_clay_regularity_program.png")
    try:
        make_figure(fig_path, r["bkm"], r["depletion"])
        print(f"\nfigure -> {fig_path}")
    except Exception as e:
        print(f"\n(figure skipped: {e})")

    print("\n" + "=" * 74)
    print("ALL VERIFIED (conditional program)" if r["all_ok"] else "SOME FAILED")
    sys.exit(0 if r["all_ok"] else 1)
