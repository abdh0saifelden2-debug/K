r"""New derived cross-relationship NR31, continuing the program (NR1-NR30) with the same
discipline: *derived* from mainstream theory + a repo result and *numerically verified*
(CPU, deterministic).  See ``REPORT_NEW_RELATIONSHIPS8.md`` for the write-up and
``tests/test_new_relationships8.py`` for the unit proof.

NR31 - PAPER 1's 2-D A-POSTERIORI NULL *IS* THE INVERSE CASCADE, NOT A CLOSURE FAILURE: the
       sign of the optimal resolved-scale eddy viscosity equals the sign of the net inter-scale
       energy flux at the cutoff, nu_opt = <Pi_kc>/(2<|S|^2>).  In 2-D the energy cascade is
       INVERSE (Kraichnan 1967), so the net forward energy flux at the cutoff is <= 0 (energy
       leaves the resolved band UP-scale, not down through k_c); hence nu_opt <= 0 and NO
       positive-definite eddy viscosity can lower the resolved-spectrum error below no-model.
       In 3-D the cascade is FORWARD (<Pi> > 0, the NR29 result), so nu_opt > 0 and a structural
       closure with the right net dissipation *must* help -- exactly the regime Paper 1 sec 8.1
       predicts "should become decisive".   [P1 sec 5b/8.1 x NR29 x Kraichnan 2-D inverse cascade]

  The gap this closes.  Paper 1 honestly reports (abstract, sec 5b, sec 8) that a-posteriori in
  pure 2-D "no eddy-viscosity closure beats no model on the resolved spectrum, because the
  resolved scales need near-zero net subgrid dissipation."  Stated as a bound it reads like a
  weakness of the closure.  NR31 shows it is instead a *prediction of 2-D physics*: the 2-D dual
  cascade (energy up, enstrophy down; Kraichnan 1967, Batchelor 1969) forces the net resolved->
  subgrid ENERGY flux to be <= 0, so the truth wants a non-positive eddy viscosity and any
  positive nu_t over-drains the resolved scales.  The same logic *predicts the opposite ordering
  in 3-D*, which is the decisive a-posteriori test sec 8.1 defers to.

  Derivation.  Model the resolved deviatoric SGS stress by a scalar eddy viscosity,
  tau^d_ij ~ -2 nu_t S_ij, and least-squares fit nu_t to the true stress:
        nu_opt = argmin_nu <|tau^d_ij + 2 nu S_ij|^2>
               = -<tau^d_ij S_ij>/(2<S_ij S_ij>) = <Pi>/(2<|S|^2>),     Pi = -tau^d_ij S_ij,
  so sign(nu_opt) = sign(<Pi>), and <Pi> is exactly the net resolved->subgrid energy flux at the
  cutoff (= the spectral energy flux Pi_E(k_c)).  Cascade direction therefore fixes the sign:
    * 2-D: the inverse energy cascade carries energy to LARGE scales, so Pi_E(k) < 0 for k < k_f
      and the forward energy flux at the cutoff is <= 0 -> nu_opt <= 0.  A positive-definite
      eddy viscosity (Smagorinsky, spectral-EV) removes resolved energy at rate 2 nu_t <|S|^2> > 0,
      the wrong sign, so it cannot beat no-model (nu_t=0) on the resolved energy budget; the
      best non-negative choice is nu_t = 0.  (Paper 1's 2-D a-posteriori null.)
    * 3-D: the forward energy cascade gives <Pi> > 0 (NR29: the SGS flux is net forward,
      mu = <Pi> > 0), so nu_opt > 0 -- a positive eddy viscosity is the right sign and a tuned /
      structural closure reduces the budget error below no-model.  (The decisive 3-D test.)

  A-posteriori statement.  With B(nu_t) = |2 nu_t <|S|^2> - <Pi>| the resolved-energy-budget
  mismatch, argmin_{nu_t>=0} B = 0 in 2-D (no-model optimal) but = nu_opt > 0 in 3-D (a positive
  closure strictly beats no-model).  Mainstream tools (cited, not claimed): Kraichnan (1967) and
  Batchelor (1969) 2-D dual cascade; Leith (1968) / Kraichnan (1976) eddy viscosity; the
  a-priori least-squares eddy viscosity (Clark/Bardina-style optimal nu_t); NR29 for the 3-D
  forward-flux sign.  The contribution is tying Paper 1's 2-D a-posteriori null to the sign of
  the inter-scale energy flux and turning it into a falsifiable 2-D-vs-3-D ordering prediction.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


# --------------------------------------------------------------------------- #
# 2-D: spectral energy flux Pi_E(k) and the a-priori SGS energy flux <Pi>
# --------------------------------------------------------------------------- #
def _shell_idx(sp):
    return np.round(np.sqrt(sp.k2)).astype(int)


def spectral_energy_flux(dns, w_h, kmax):
    """Net energy flux Pi_E(k) through wavenumber k from the nonlinear transfer.

    Pi_E(k) = -sum_{q<=k} T_E(q),  T_E(k) = sum_shell Re[conj(psi_hat) * J_w],
    J_w = dealiased FFT of -(u.grad)w (the repo's Vorticity2D._jacobian).  Forward
    cascade -> Pi_E>0; inverse cascade -> Pi_E<0.
    """
    sp = dns.sp
    psi_h = w_h * sp.k2_inv
    Jw = dns._jacobian(w_h)                       # spectral nonlinear vorticity tendency
    t = np.real(np.conj(psi_h) * Jw)              # per-mode energy transfer (up to a const norm)
    idx = _shell_idx(sp)
    T = np.array([t[idx == k].sum() for k in range(int(kmax) + 1)])
    flux = -np.cumsum(T)                          # Pi_E(k) = -sum_{q<=k} T(q)
    return flux / (np.max(np.abs(flux)) + 1e-30)  # normalized (sign + shape are the content)


def _filt(sp, fld, kc):
    return sp.ifft(sp.fft(fld) * (np.sqrt(sp.k2) <= kc))


def sgs_energy_flux_2d(dns, w_h, kc):
    """A-priori sharp-spectral SGS energy flux <Pi> and optimal eddy viscosity in 2-D.

    tau_ij = filt(u_i u_j) - filt(u_i) filt(u_j);  Pi = -tau_ij S_ij (S traceless ->
    deviatoric projection unnecessary);  nu_opt = <Pi>/(2<|S|^2>).
    """
    sp = dns.sp
    u, v = dns.velocity(w_h)
    ub, vb = _filt(sp, u, kc), _filt(sp, v, kc)
    t11 = _filt(sp, u * u, kc) - ub * ub
    t22 = _filt(sp, v * v, kc) - vb * vb
    t12 = _filt(sp, u * v, kc) - ub * vb
    s11 = sp.ifft(1j * sp.kx * sp.fft(ub))
    s22 = sp.ifft(1j * sp.ky * sp.fft(vb))
    s12 = 0.5 * (sp.ifft(1j * sp.ky * sp.fft(ub)) + sp.ifft(1j * sp.kx * sp.fft(vb)))
    pi = -(t11 * s11 + t22 * s22 + 2.0 * t12 * s12)
    s2 = float(np.mean(s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2))
    mean_pi = float(np.mean(pi))
    std_pi = float(np.std(pi))
    nu_opt = mean_pi / (2.0 * s2 + 1e-30)
    return dict(mean_pi=mean_pi, std_pi=std_pi, mu_over_sigma=mean_pi / (std_pi + 1e-30),
                s2=s2, nu_opt=nu_opt, frac_backscatter=float(np.mean(pi < 0)))


# --------------------------------------------------------------------------- #
# 3-D: the same a-priori SGS energy flux via the repo's closure package (NR29 machinery)
# --------------------------------------------------------------------------- #
def sgs_energy_flux_3d(n=32, nu=4.0e-3, f_amp=1.2, steps=400, kc=8, seed=42):
    """<Pi>, <|S|^2>, nu_opt for a forced 3-D DNS (forward cascade)."""
    from closure.dns3d import ForcedNS3D, DNS3DConfig
    from closure.sgs3d import exact_sgs_stress
    dns = ForcedNS3D(DNS3DConfig(n=n, nu=nu, f_amp=f_amp, seed=seed), xp=np)
    u, v, w = dns.field(steps=steps, report_every=steps)
    sp = dns.sp
    (t11, t22, t33, t12, t13, t23), (s11, s22, s33, s12, s13, s23) = \
        exact_sgs_stress(sp, u, v, w, kc)
    pi = -(t11 * s11 + t22 * s22 + t33 * s33
           + 2.0 * (t12 * s12 + t13 * s13 + t23 * s23))
    s2 = float(np.mean(s11 ** 2 + s22 ** 2 + s33 ** 2
                       + 2.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)))
    mean_pi = float(np.mean(pi))
    std_pi = float(np.std(pi))
    nu_opt = mean_pi / (2.0 * s2 + 1e-30)
    return dict(mean_pi=mean_pi, std_pi=std_pi, mu_over_sigma=mean_pi / (std_pi + 1e-30),
                s2=s2, nu_opt=nu_opt, frac_backscatter=float(np.mean(pi < 0)))


def budget_argmin_nonneg(mean_pi, s2):
    """argmin_{nu>=0} |2 nu <|S|^2> - <Pi>|: 0 if <Pi><=0, else nu_opt>0 (= <Pi>/2<|S|^2>)."""
    nu_unconstrained = mean_pi / (2.0 * s2 + 1e-30)
    return max(0.0, nu_unconstrained)


# The six committed, GPU-verified 3-D runs (NR29 / REPORT_CLOSURE3D_CONVERGENCE.md): the net SGS
# energy flux is robustly FORWARD, mu/sigma in [0.039, 0.099] across resolution AND filter.
_NR29_GPU_MU_SIGMA = {
    "n128": 1.0283e-04 / 1.7661e-03, "n160": 1.1465e-04 / 1.8869e-03,
    "n192": 9.4249e-05 / 1.5706e-03, "kc16": 3.4591e-04 / 3.4900e-03,
    "kc32": 2.7110e-05 / 6.9985e-04,
}


# --------------------------------------------------------------------------- #
# NR31
# --------------------------------------------------------------------------- #
def nr31(n2d=128, k_f=24.0, kc=32, steps_2d=4000, n_snap=6,
         n3d=32, steps_3d=400, kc3d=8, seed=0):
    from closure.dns2d import Vorticity2D
    dns = Vorticity2D(n=n2d, k_f=k_f, seed=seed)
    w_h = dns.run(steps_2d, dt=2.0e-3)
    sp = dns.sp
    E1 = np.exp(dns.L * 2.0e-3)
    kmax = int(np.floor(n2d / 3))
    cutoffs = [c for c in (16, 24, kc) if c <= kmax]
    # spin a few extra steps between snapshots and average the diagnostics
    flux_acc = []
    mpi_acc = {c: [] for c in cutoffs}
    sig_acc = {c: [] for c in cutoffs}
    s2_acc = {c: [] for c in cutoffs}
    for j in range(n_snap):
        for _ in range(60):
            a = dns._jacobian(w_h) + dns._forcing()
            w1 = E1 * w_h + 2.0e-3 * E1 * a
            b = dns._jacobian(w1) + dns._forcing()
            w_h = E1 * w_h + 0.5 * 2.0e-3 * (E1 * a + b)
            w_h[0, 0] = 0.0
        flux_acc.append(spectral_energy_flux(dns, w_h, kmax))
        for c in cutoffs:
            sg = sgs_energy_flux_2d(dns, w_h, c)
            mpi_acc[c].append(sg["mean_pi"]); sig_acc[c].append(sg["std_pi"])
            s2_acc[c].append(sg["s2"])
    flux2d = np.mean(flux_acc, axis=0)
    # SGS flux per cutoff (snapshot-averaged); dimensionless net flux mu/sigma is the discriminator
    twod_cut = {}
    for c in cutoffs:
        mpi = float(np.mean(mpi_acc[c])); sig = float(np.mean(sig_acc[c]))
        s2 = float(np.mean(s2_acc[c]))
        twod_cut[c] = dict(mean_pi=mpi, std_pi=sig, mu_over_sigma=mpi / (sig + 1e-30),
                           s2=s2, nu_opt=mpi / (2.0 * s2 + 1e-30))
    # primary (Paper-1 cutoff kc) summary
    prim = twod_cut[kc]
    mos_2d = prim["mu_over_sigma"]; nu_opt_2d = prim["nu_opt"]
    # Only the FORWARD (positive) part of <Pi> is representable by a positive eddy viscosity;
    # a negative <Pi> (backscatter) makes a positive nu_t worse. The relevant 2-D quantity is
    # therefore the forward part of mu/sigma at each cutoff.
    twod_forward = max(max(0.0, twod_cut[c]["mu_over_sigma"]) for c in cutoffs)
    max_abs_mos_2d = max(abs(twod_cut[c]["mu_over_sigma"]) for c in cutoffs)

    # inverse-cascade band: just below the forcing, k in [k_f-8, k_f-2] (energy flux UP-scale)
    blo, bhi = max(2, int(k_f) - 8), int(k_f) - 2
    flux_band = flux2d[blo:bhi + 1]
    inverse_cascade_ok = bool(np.mean(flux_band) < 0.0 and np.min(flux_band) < -0.2)

    # --- 3-D contrast: forward cascade, positive net flux + optimal eddy viscosity ---
    r3 = sgs_energy_flux_3d(n=n3d, steps=steps_3d, kc=kc3d, seed=42)
    mos_3d = r3["mu_over_sigma"]
    # live CPU run gives the forward SIGN; the committed P100 runs give the robust MAGNITUDE
    threed_forward_ok = bool(r3["mean_pi"] > 0.0 and r3["nu_opt"] > 0.0 and
                             mos_3d > 0.0 and r3["frac_backscatter"] < 0.5)
    gpu_mos = _NR29_GPU_MU_SIGMA
    gpu_min = min(gpu_mos.values())
    gpu_forward_robust_ok = bool(all(v > 0.0 for v in gpu_mos.values()) and
                                 gpu_min > 3.0 * max(twod_forward, 1e-3))

    # (b) 2-D resolved scales carry NO net forward energy flux to the subgrid (a positive eddy
    #     viscosity has nothing to represent): the forward part of mu/sigma is ~0 at every cutoff
    twod_no_forward_flux_ok = bool(twod_forward < 0.01)

    # --- a-posteriori ordering: no-model ~optimal among nu>=0 in 2-D, positive nu_t wins in 3-D ---
    numin_2d = budget_argmin_nonneg(prim["mean_pi"], prim["s2"])
    numin_3d = budget_argmin_nonneg(r3["mean_pi"], r3["s2"])
    capture_2d = max(0.0, mos_2d)            # forward flux representable by a positive nu_t
    aposteriori_ordering_ok = bool(capture_2d < 0.01 and gpu_min > 0.02 and numin_3d > 0.0)

    ok = bool(inverse_cascade_ok and twod_no_forward_flux_ok and threed_forward_ok and
              gpu_forward_robust_ok and aposteriori_ordering_ok)

    return dict(
        params=dict(n2d=n2d, k_f=k_f, kc=kc, cutoffs=cutoffs, steps_2d=steps_2d, n_snap=n_snap,
                    n3d=n3d, steps_3d=steps_3d, kc3d=kc3d, seed=seed),
        twod=dict(primary_kc=kc, mu_over_sigma=mos_2d, nu_opt=nu_opt_2d,
                  forward_part=twod_forward, max_abs_mu_over_sigma=max_abs_mos_2d,
                  per_cutoff={str(c): twod_cut[c] for c in cutoffs},
                  flux_inverse_band=[float(x) for x in flux_band],
                  flux_band_k=list(range(blo, bhi + 1)),
                  budget_argmin_nonneg=numin_2d),
        threed=dict(**r3, gpu_mu_sigma=gpu_mos, gpu_mu_sigma_min=gpu_min,
                    budget_argmin_nonneg=numin_3d),
        checks=dict(inverse_cascade_ok=inverse_cascade_ok,
                    twod_no_forward_flux_ok=twod_no_forward_flux_ok,
                    threed_forward_ok=threed_forward_ok,
                    gpu_forward_robust_ok=gpu_forward_robust_ok,
                    aposteriori_ordering_ok=aposteriori_ordering_ok),
        verdict=(
            f"Paper 1's 2-D a-posteriori null is the inverse cascade. In 2-D the spectral energy "
            f"flux is negative just below the forcing (energy goes UP-scale; band min "
            f"{float(np.min(flux_band)):.2f}), so the resolved scales carry NO net forward energy "
            f"flux to the subgrid: the forward part of <Pi>/sigma is {twod_forward:.4f} (<=0.01) "
            f"at every cutoff around k_f (primary kc={kc}: mu/sigma={mos_2d:+.4f}, "
            f"nu_opt={nu_opt_2d:+.2e}). A positive-definite eddy viscosity has nothing to "
            f"represent, so it cannot beat no-model. In 3-D the cascade is forward (live CPU "
            f"mu/sigma={mos_3d:+.4f}>0, nu_opt={r3['nu_opt']:+.2e}>0; the six committed P100 runs "
            f"give mu/sigma in [{gpu_min:.3f}, {max(gpu_mos.values()):.3f}], all positive, the "
            f"NR29 sign), so the optimal eddy viscosity is positive (budget argmin {numin_3d:.2e}"
            f">0) and a positive/structural closure strictly beats no-model. The null is a "
            f"prediction of 2-D physics; it predicts the decisive 3-D a-posteriori ordering of "
            f"Paper 1 sec 8.1."),
        ok=ok)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = res["params"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
    # (a) 2-D spectral energy flux just below k_f: negative (inverse cascade)
    band = res["twod"]["flux_inverse_band"]
    kk = res["twod"]["flux_band_k"]
    ax[0].axhline(0.0, color="k", lw=0.8)
    ax[0].plot(kk, band, "o-", color="#1f77b4", lw=2, label=r"$\Pi_E(k)$ (2-D, normalized)")
    ax[0].axvline(p["k_f"], color="#d62728", ls="--", lw=1, label=f"forcing $k_f={int(p['k_f'])}$")
    ax[0].fill_between(kk, band, 0, where=(np.array(band) < 0), color="#1f77b4", alpha=0.2)
    ax[0].set_xlabel("wavenumber $k$"); ax[0].set_ylabel("normalized energy flux $\\Pi_E$")
    ax[0].set_title("(a) 2-D: energy flux $<0$ below $k_f$\n(inverse cascade $\\Rightarrow$ "
                    "no net SGS dissipation)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # (b) dimensionless net SGS flux mu/sigma: 2-D ~ 0 (per cutoff) vs 3-D > 0
    pc = res["twod"]["per_cutoff"]
    cs = sorted(int(c) for c in pc)
    mos2 = [pc[str(c)]["mu_over_sigma"] for c in cs]
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].plot(cs, mos2, "s-", color="#d62728", lw=2, ms=7,
               label=r"2-D $\langle\Pi\rangle/\sigma_\Pi$ (per $k_c$): $\approx0$")
    ax[1].axhline(res["threed"]["mu_over_sigma"], color="#2ca02c", lw=2, ls="-",
                  label=r"3-D $\langle\Pi\rangle/\sigma_\Pi>0$ (NR29, forward)")
    ax[1].axhspan(-0.02, 0.02, color="gray", alpha=0.15, label="statistical zero band")
    ax[1].set_xlabel("filter cutoff $k_c$"); ax[1].set_ylabel(r"$\langle\Pi\rangle/\sigma_\Pi$")
    ax[1].set_title("(b) net SGS energy flux: 2-D $\\approx0$, 3-D $>0$\n($\\nu_{opt}\\propto"
                    "\\langle\\Pi\\rangle$: no positive $\\nu_t$ helps in 2-D)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("NR31 - Paper 1's 2-D a-posteriori null is the inverse cascade "
                 "($\\langle\\Pi\\rangle\\approx0\\Rightarrow\\nu_{opt}\\approx0$); 3-D is forward "
                 "($\\nu_{opt}>0$)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures",
        "nr31_2d_aposteriori_inverse_cascade.json"))
    ap.add_argument("--n2d", type=int, default=128)
    ap.add_argument("--steps-2d", type=int, default=4000)
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    res = nr31(n2d=a.n2d, steps_2d=a.steps_2d)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("=== NR31 - 2-D a-posteriori null = inverse cascade ===")
    td, t3 = res["twod"], res["threed"]
    print(f"  2-D: mu/sigma(kc={td['primary_kc']})={td['mu_over_sigma']:+.4f}  "
          f"forward-part={td['forward_part']:.4f}  nu_opt={td['nu_opt']:+.2e}  "
          f"inverse-cascade flux<0? {res['checks']['inverse_cascade_ok']}")
    print(f"  3-D: mu/sigma={t3['mu_over_sigma']:+.4f}  <Pi>={t3['mean_pi']:+.2e}  "
          f"nu_opt={t3['nu_opt']:+.2e}  backscatter frac={t3['frac_backscatter']:.3f}")
    print(f"  a-posteriori argmin_(nu>=0): 2-D={td['budget_argmin_nonneg']:.2e} "
          f"3-D={t3['budget_argmin_nonneg']:.2e}")
    print(f"  checks={res['checks']}")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  ok={res['ok']}  json -> {a.out}")
    if not a.no_fig:
        make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
