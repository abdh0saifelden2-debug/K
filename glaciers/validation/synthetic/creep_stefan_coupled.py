r"""§D.6 (completion) — the coupled Stefan + Glen-creep amplitude sweep.

FUTURE_WORK.md §D.6 establishes that Glen's-law creep is a **[NULL]** for the
subglacial scallop amplitude: over the *solver clock* (seconds-to-hours) the creep
wall displacement is <1% of the roughness amplitude, so the rigid-wall Brinkman
penalization is justified (``creep_scaling_synthetic.py``).  §D.6 explicitly left
**one residual un-run**:

    "The full coupled Stefan+creep simulation across the amplitude sweep is not run
     here; it is unnecessary for the rigid-wall justification, which rests on the
     displacement-fraction bound above, but would quantify the (small) long-time
     smoothing correction."

This module runs that residual.  It does **not** re-run any DNS: the melt (Stefan)
amplitude-smoothing rate is taken from the *already-measured*, committed RESULT 14
harmonic decomposition (``glaciers/figures/56_scallop_amplitude_harmonics.json``,
``beta/a`` per mode -> the dimensional bridge ``tau_melt ~ lam^2/dT``).  It then
adds the Glen-creep amplitude sink and integrates the **coupled** amplitude ODE
across the amplitude sweep, over the *physical* multi-year melt timescale (not the
solver clock), to quantify how much creep shifts the melt-set scallop amplitude.

Physics
-------
The corrected RESULT 14 amplitude dynamics is smoothing-limited: ``a_dot|melt =
-beta_melt a`` (a damped, downstream-migrating mode ``s = -beta + i omega_mig``).
Glen creep adds a same-sign smoothing sink ``a_dot|creep = -r_creep a`` (§D.6 sign
argument: a corrugation under load can only relax, never amplify).  The coupled
linear amplitude equation is therefore

    a_dot = -(beta_melt + r_creep) a ,

so creep does two things, both quantified here: (i) it speeds the amplitude
e-folding by the factor ``1 + rho`` and (ii) if the melt physics sets a finite
attractor ``a*`` (the saturated scallop), the extra sink lowers it to
``a*_coupled / a*_melt = beta_melt / (beta_melt + r_creep) = 1/(1+rho)``, where

    rho := r_creep / beta_melt

is the dimensionless creep-to-melt smoothing ratio -- the single number that
decides whether creep matters.

The creep rate is ``r_creep = A sigma_dev^n`` (Glen, ``n=3``; Cuffey & Paterson
2010).  The *only* modelling choice is the deviatoric stress ``sigma_dev`` that
drives relaxation of a small corrugation.  We report two:

  * **Worst case (the §D.6 displacement bound's stress): ``sigma_dev = N``** -- the
    full effective pressure.  This is a deliberate upper bound; §D.6 only ever
    evaluates it over the *solver clock* (1 hr), where it gives <1%.  We show that
    *extrapolating it to the multi-year melt timescale is not physical*: at
    temperate ice and high ``N`` it would make ``rho >> 1`` (creep "dominating"),
    which is why §D.6 restricts the bound to the solver clock.

  * **Physical (topographic relief stress): ``sigma_dev = rho_i g a``** -- the
    gravitational stress of the corrugation's own relief, the standard driver for
    viscous relaxation of surface topography.  This is the stress that actually
    relaxes a scallop, and it is *much* smaller than ``N``.

The robust, modelling-choice-free statement is the **critical deviatoric stress**
at which creep would match melt smoothing,

    sigma_crit = (beta_melt / A)^(1/n) ,

compared with the actual topographic stress ``rho_i g a``.

Result (anchored at the Curl operating point, ``lam ~ 7.9 cm``, ``dT = 0.1 K``):
``beta_melt`` ~ (3 yr)^-1; ``sigma_crit`` ~ 0.16 MPa (temperate) / 0.35 MPa (cold);
the topographic stress is ~35-280 Pa across ``a/lam in [0.05, 0.40]``, so
``rho ~ 1e-11 .. 1e-8`` and the coupled amplitude correction ``1/(1+rho)`` differs
from 1 by < 1e-8.  Creep smoothing is 8-11 orders of magnitude slower than melt
smoothing -- the §D.6 long-time correction is **quantified and negligible**,
consistent with the [LIT] fact that morphologically identical scallops form on
**non-creeping limestone** (Curl 1966).  CPU-only, deterministic, no DNS re-run.
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
R14_JSON = os.path.join(REPO, "glaciers", "figures", "56_scallop_amplitude_harmonics.json")

# --- [LIT] / repo constants (match scallop_amplitude_harmonics.py & creep_scaling_synthetic.py)
KAPPA_ND = 8.0e-4          # solver thermal diffusivity (Candidate3Config.kappa)
K_TH_WATER = 0.56          # W m^-1 K^-1, water-side thermal conductivity
RHO_L_ICE = 3.0e8          # J m^-3, ice latent heat of fusion per unit volume
SECONDS_PER_YEAR = 3.15576e7
A_COLD = 2.4e-25           # Pa^-3 s^-1, Glen prefactor, ice ~ -10 C (Cuffey & Paterson 2010)
A_TEMP = 2.4e-24           # Pa^-3 s^-1, Glen prefactor, ice ~ 0 C
N_GLEN = 3
RHO_ICE = 917.0            # kg m^-3
G_GRAV = 9.81              # m s^-2
NU_WATER = 1.79e-6         # m^2 s^-1
RE_SCALLOP_CURL = 2200.0   # Blumberg & Curl (1974): lam = Re_* nu / u_*


def load_beta_per_a(nw=12, json_path=R14_JSON):
    """Measured dimensionless conduction smoothing gain beta/a and nondim wavelength
    for mode ``nw`` from the committed RESULT 14 artifact (no DNS re-run)."""
    with open(json_path) as fh:
        d = json.load(fh)
    row = next(r for r in d["conduction_beta"] if r["n_waves"] == nw)
    return float(row["beta_per_a"]), float(row["lam"])


def beta_melt(lam_phys, dT, beta_per_a, lam_nd,
              k_th=K_TH_WATER, rhoL=RHO_L_ICE, kappa_nd=KAPPA_ND):
    """Melt (Stefan) amplitude-smoothing rate [1/s] from the RESULT 14 dimensional
    bridge: r = (k_th dT / (rho L L0^2)) (beta/a)/kappa_nd,  L0 = lam_phys/lam_nd.
    (tau_melt = 1/r ~ lam^2/dT.)"""
    L0 = lam_phys / lam_nd
    return k_th * dT / (rhoL * L0 ** 2) * (beta_per_a / kappa_nd)


def r_creep(A, sigma_dev, n=N_GLEN):
    """Glen-creep amplitude-relaxation rate r = A sigma_dev^n  [1/s]."""
    return A * sigma_dev ** n


def sigma_crit(beta_m, A, n=N_GLEN):
    """Deviatoric stress at which creep smoothing matches melt smoothing
    (r_creep = beta_melt):  sigma_crit = (beta_melt / A)^(1/n)  [Pa]."""
    return (beta_m / A) ** (1.0 / n)


def sigma_topographic(a_phys, rho_i=RHO_ICE, g=G_GRAV):
    """Topographic relief deviatoric stress driving creep relaxation of a
    corrugation of amplitude a:  sigma ~ rho_i g a  [Pa] (gravitational relief)."""
    return rho_i * g * a_phys


def coupled_correction(beta_m, r_c):
    """Steady-amplitude correction factor a*_coupled/a*_melt = 1/(1+rho),
    rho = r_creep/beta_melt.  (Also the inverse e-folding-time speed-up.)"""
    rho = r_c / beta_m
    return 1.0 / (1.0 + rho), rho


def run(nw=12, u_star=0.05, dT=0.1, afracs=(0.05, 0.10, 0.20, 0.40),
        Ns=(0.1e6, 1.0e6), json_path=R14_JSON):
    beta_per_a, lam_nd = load_beta_per_a(nw, json_path)
    lam_phys = RE_SCALLOP_CURL * NU_WATER / u_star          # Curl selection
    bmelt = beta_melt(lam_phys, dT, beta_per_a, lam_nd)
    tau_melt_yr = 1.0 / bmelt / SECONDS_PER_YEAR

    s_crit = {"cold": sigma_crit(bmelt, A_COLD), "temperate": sigma_crit(bmelt, A_TEMP)}

    # (1) PHYSICAL stress sigma_dev = rho_i g a across the amplitude sweep
    physical = []
    for af in afracs:
        a_phys = af * lam_phys
        s_topo = sigma_topographic(a_phys)
        row = {"a_over_lam": af, "a_phys_m": a_phys, "sigma_topo_Pa": s_topo}
        for label, A in (("cold", A_COLD), ("temperate", A_TEMP)):
            rc = r_creep(A, s_topo)
            fac, rho = coupled_correction(bmelt, rc)
            row[f"rho_{label}"] = rho
            row[f"ampl_correction_{label}"] = fac          # a*_coupled/a*_melt
        physical.append(row)

    # (2) WORST-CASE stress sigma_dev = N (the §D.6 displacement-bound stress),
    #     extrapolated to the melt timescale -- shown to be non-physical at high N
    worst = []
    for label, A in (("cold", A_COLD), ("temperate", A_TEMP)):
        for N in Ns:
            rc = r_creep(A, N)
            fac, rho = coupled_correction(bmelt, rc)
            worst.append({"ice": label, "N_Pa": N, "N_MPa": N / 1e6,
                          "rho": rho, "ampl_correction": fac,
                          "creep_dominates_over_years": bool(rho > 1.0)})

    # ---- checks / verdict ----
    max_rho_phys = max(max(r["rho_cold"], r["rho_temperate"]) for r in physical)
    # creep negligible under the physical stress, across the whole sweep:
    physical_null = bool(max_rho_phys < 1e-6)
    # the worst-case sigma=N over years is NOT a valid long-time model (it would
    # make rho>1 somewhere) -> confirms why §D.6 restricts sigma=N to the solver clock:
    worstcase_overpredicts = bool(any(w["creep_dominates_over_years"] for w in worst))
    # topographic stress is far below the creep<->melt crossover stress:
    a_anchor = 0.10 * lam_phys
    margin_temperate = s_crit["temperate"] / sigma_topographic(a_anchor)
    stress_margin_ok = bool(margin_temperate > 100.0)
    # sign: creep is always a smoothing sink (correction factor <= 1):
    sign_ok = bool(all(r["ampl_correction_cold"] <= 1.0
                       and r["ampl_correction_temperate"] <= 1.0 for r in physical))

    ok = bool(physical_null and stress_margin_ok and sign_ok and worstcase_overpredicts)

    return {
        "params": {"nw": nw, "u_star_m_per_s": u_star, "dT_K": dT,
                   "lam_phys_m": lam_phys, "beta_per_a": beta_per_a, "lam_nd": lam_nd,
                   "afracs": list(afracs), "Ns_MPa": [N / 1e6 for N in Ns]},
        "beta_melt_per_s": bmelt, "tau_melt_yr": tau_melt_yr,
        "sigma_crit_MPa": {k: v / 1e6 for k, v in s_crit.items()},
        "physical_stress_sweep": physical,
        "worstcase_N_stress": worst,
        "max_rho_physical": max_rho_phys,
        "stress_margin_temperate": margin_temperate,
        "checks": {"physical_null": physical_null,
                   "stress_margin_ok": stress_margin_ok,
                   "sign_smoothing_only": sign_ok,
                   "worstcase_sigmaN_overpredicts_over_years": worstcase_overpredicts},
        "verdict": (
            "Coupled Stefan+creep amplitude sweep (§D.6 residual, quantified). The "
            "melt amplitude e-folds in ~%.1f yr; creep matches it only at a deviatoric "
            "stress sigma_crit ~ %.2f MPa (temperate) / %.2f MPa (cold). The physical "
            "topographic relief stress rho_i g a is ~%.0f-%.0f Pa across a/lam in "
            "[%.2f,%.2f] -- ~%.0fx below sigma_crit -- so rho = r_creep/beta_melt <= %.1e "
            "and the coupled amplitude correction 1/(1+rho) departs from 1 by < %.0e: the "
            "long-time creep smoothing correction is QUANTIFIED and NEGLIGIBLE, and is "
            "always same-sign smoothing (never enhancement). The worst-case sigma=N bound "
            "of §D.6 is solver-clock-only: extrapolated to the melt timescale it would give "
            "rho>1 at high N, which is why §D.6 (correctly) restricts it to the run window. "
            "Consistent with identical scallops on non-creeping limestone (Curl 1966)."
            % (tau_melt_yr, s_crit["temperate"] / 1e6, s_crit["cold"] / 1e6,
               physical[0]["sigma_topo_Pa"], physical[-1]["sigma_topo_Pa"],
               afracs[0], afracs[-1], margin_temperate, max_rho_phys, max_rho_phys)),
        "ok": ok,
    }


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # (a) rho vs a/lam under the physical stress (log) + worst-case sigma=N band
    af = [r["a_over_lam"] for r in res["physical_stress_sweep"]]
    ax[0].semilogy(af, [r["rho_temperate"] for r in res["physical_stress_sweep"]],
                   "o-", color="#1f77b4", label=r"physical $\sigma=\rho_i g a$ (temperate)")
    ax[0].semilogy(af, [r["rho_cold"] for r in res["physical_stress_sweep"]],
                   "s-", color="#2ca02c", label=r"physical $\sigma=\rho_i g a$ (cold)")
    for w in res["worstcase_N_stress"]:
        ax[0].axhline(w["rho"], ls=":", lw=0.8, color="#d62728" if w["ice"] == "temperate" else "#ff7f0e")
    ax[0].axhline(1.0, color="k", lw=1.0)
    ax[0].text(af[0], 1.3, r"creep = melt ($\rho=1$)", fontsize=8)
    ax[0].set_xlabel(r"amplitude $a/\lambda$"); ax[0].set_ylabel(r"$\rho = r_{creep}/\beta_{melt}$")
    ax[0].set_title("(a) creep/melt smoothing ratio\n(dotted red/orange = worst-case $\\sigma=N$)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    # (b) the coupled amplitude correction factor 1/(1+rho) under the physical stress
    ax[1].plot(af, [r["ampl_correction_temperate"] for r in res["physical_stress_sweep"]],
               "o-", color="#1f77b4", label="temperate")
    ax[1].plot(af, [r["ampl_correction_cold"] for r in res["physical_stress_sweep"]],
               "s-", color="#2ca02c", label="cold")
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set_ylim(1 - 1e-7, 1 + 1e-8)
    ax[1].set_xlabel(r"amplitude $a/\lambda$")
    ax[1].set_ylabel(r"$a^*_{coupled}/a^*_{melt}=1/(1+\rho)$")
    ax[1].set_title("(b) coupled steady-amplitude correction\n(departs from 1 by $<10^{-8}$)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(r"§D.6 coupled Stefan+creep: long-time creep smoothing is negligible "
                 r"($\beta_{melt}\!\sim\!(3\,\mathrm{yr})^{-1}$, $\sigma_{crit}\!\sim\!0.2\,$MPa $\gg\rho_i g a$)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO, "glaciers", "validation", "reports", "creep_stefan_coupled.json"))
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("=== §D.6 coupled Stefan + Glen-creep amplitude sweep ===")
    print(f"  beta_melt = {res['beta_melt_per_s']:.3e} /s  (tau_melt = {res['tau_melt_yr']:.2f} yr)")
    print(f"  sigma_crit (creep=melt): temperate {res['sigma_crit_MPa']['temperate']:.3f} MPa, "
          f"cold {res['sigma_crit_MPa']['cold']:.3f} MPa")
    print("  PHYSICAL stress (sigma=rho_i g a):")
    for r in res["physical_stress_sweep"]:
        print(f"    a/lam={r['a_over_lam']:.2f}: sigma_topo={r['sigma_topo_Pa']:.0f} Pa  "
              f"rho(temp)={r['rho_temperate']:.2e}  a*_corr={r['ampl_correction_temperate']:.10f}")
    print("  WORST-CASE sigma=N extrapolated to years (NOT physical at high N):")
    for w in res["worstcase_N_stress"]:
        print(f"    {w['ice']:>9} N={w['N_MPa']:.1f} MPa: rho={w['rho']:.2e}  "
              f"creep_dominates={w['creep_dominates_over_years']}")
    print(f"  max rho (physical) = {res['max_rho_physical']:.2e}; "
          f"stress margin (temperate) = {res['stress_margin_temperate']:.0f}x")
    print(f"  checks = {res['checks']}")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  ok = {res['ok']}  json -> {a.out}")
    if not a.no_fig:
        make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
