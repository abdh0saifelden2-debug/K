r"""§H.1.2 — the intrusion residence number Ro, DERIVED: is grounding-line intrusion
thinning-paced or hydraulic-limited?

What this sharpens (the §H.1.2 [HYP])
-------------------------------------
§H.1.2 verifies the *kinematics* of the RTN=1 intrusion front (level-set advance
`v_kin = (dH/dt)/|grad m|`, geometric amplification `A=1/|grad m|`) and conjectures
that the actual advance is **rate-limited by the subglacial hydraulic residence time**
(how fast the newly-exposed bed re-pressurises), with the residence number

    Ro = v_kin / v_obs   (>1 ⇒ hydraulic-limited; ≈1 ⇒ thinning-paced),

left as an *unmeasured* ratio pending DInSAR `v_obs`. This module **derives the
predicted Ro from physics** (no new data), so a future `v_obs` only has to locate the
system on the derived curve.

The derivation
--------------
For the front to advance a length `ℓ` (taken as the front smoothing scale ~ one ice
thickness `H`), the newly-near-flotation bed must re-pressurise to ocean head. That
re-pressurisation is the §G.4 subglacial-hydraulic residence process with timescale
`tau_hyd` (the §G.4 cavity<->channel residence band, 0.01-2 yr; `hydraulic_lag_derivation`).
The maximum hydraulic propagation speed over `ℓ` is `v_hyd = ℓ/tau_hyd`, so the front
moves at `v_obs = min(v_kin, v_hyd)` and

    Ro_pred = v_kin / v_hyd = v_kin * tau_hyd / ℓ,   Ro_obs = max(1, Ro_pred).

Equivalently the **regime boundary** is a critical residence time
`tau_crit = ℓ / v_kin`: `tau_hyd < tau_crit` ⇒ thinning-paced (`Ro≈1`);
`tau_hyd > tau_crit` ⇒ hydraulic-limited (`Ro≫1`). The implied hydraulic diffusivity
`D_hyd = ℓ^2/tau_hyd` (~0.06-12.7 m^2/s for `ℓ=2 km`, `tau_hyd=0.01-2 yr`) spans the
distributed (slow, low-D) to channelized (fast, high-D) subglacial-drainage range
(Werder et al. 2013; Hewitt 2013).

Result (runaway-tail cells, A=0.70 km/m)
----------------------------------------
`tau_crit = ℓ/v_kin ≈ 1.9 yr` at `dH/dt=1.5 m/yr` (≈1.4 yr at 2 m/yr). Since most of
the §G.4 residence band (median ~0.01-0.1 yr) is **below** `tau_crit`, the prediction is **thinning-paced
(Ro≈1)** for the runaway cells over most of the band; the hydraulic-limited case
(`Ro≫1`) needs residence at/above the slow ~2 yr end (or faster thinning). Falsifiable:
a DInSAR `v_obs` giving `Ro≫1` over runaway cells would place `tau_hyd` above `tau_crit`
and confirm hydraulic limitation; `Ro≈1` confirms thinning-pacing. No GPU, no download.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SEC_PER_YR = 365.25 * 86400.0
# §H.1.2 Bedmap2 amplification A=1/|grad m| [km per m of thinning] (documented values)
A_MEDIAN_KM_PER_M = 0.09
A_RUNAWAY_KM_PER_M = 0.70           # runaway-tail median (754 cells, A>=0.42 within 50 km of GL)
# §G.4 hydraulic residence band [yr] (hydraulic_lag_derivation / observed 0.02-2 yr)
TAU_HYD_BAND_YR = (0.01, 2.0)
ELL_M = 2000.0                       # front smoothing scale ~ ice thickness [m]


def v_kin_km_yr(A_km_per_m, dHdt_m_yr):
    """Kinematic front speed v_kin = A * dH/dt [km/yr]."""
    return A_km_per_m * dHdt_m_yr


def tau_crit_yr(ell_m, v_kin_km_yr_, ):
    """Critical residence time tau_crit = ell/v_kin [yr] (regime boundary)."""
    v_kin_m_yr = v_kin_km_yr_ * 1000.0
    return ell_m / v_kin_m_yr


def Ro_pred(tau_hyd_yr, v_kin_km_yr_, ell_m=ELL_M):
    """Predicted residence number Ro = v_kin*tau_hyd/ell ; Ro_obs = max(1, Ro_pred)."""
    v_kin_m_yr = v_kin_km_yr_ * 1000.0
    return v_kin_m_yr * tau_hyd_yr / ell_m


def D_hyd_m2_s(tau_hyd_yr, ell_m=ELL_M):
    """Implied subglacial hydraulic diffusivity D = ell^2/tau_hyd [m^2/s]."""
    return ell_m ** 2 / (tau_hyd_yr * SEC_PER_YR)


def run():
    dHdt = np.array([0.5, 1.0, 2.0, 5.0])           # thinning rate [m/yr]
    out_regimes = {}
    for label, A in (("margin_median", A_MEDIAN_KM_PER_M), ("runaway_tail", A_RUNAWAY_KM_PER_M)):
        vkin = v_kin_km_yr(A, dHdt)                  # km/yr per dH/dt
        tcrit = np.array([tau_crit_yr(ELL_M, v) for v in vkin])
        # Ro across the residence band, at dH/dt = 1 and 2 m/yr
        rows = []
        for dh, vk in zip(dHdt, vkin):
            ro_lo = Ro_pred(TAU_HYD_BAND_YR[0], vk)
            ro_hi = Ro_pred(TAU_HYD_BAND_YR[1], vk)
            rows.append(dict(dHdt_m_yr=float(dh), v_kin_km_yr=float(vk),
                             tau_crit_yr=float(tau_crit_yr(ELL_M, vk)),
                             Ro_at_fast_hyd=float(ro_lo), Ro_at_slow_hyd=float(ro_hi),
                             regime_over_band=("thinning-paced for tau_hyd<tau_crit, "
                                               "hydraulic-limited above")))
        out_regimes[label] = rows
    # regime map: Ro(tau_hyd, v_kin) grid for the figure
    tau = np.geomspace(TAU_HYD_BAND_YR[0], TAU_HYD_BAND_YR[1], 60)
    vk_grid = np.geomspace(0.05, 2.0, 60)            # km/yr
    Ro = np.array([[Ro_pred(t, v) for t in tau] for v in vk_grid])
    # headline: runaway cells at dH/dt=1.5
    vk_run = v_kin_km_yr(A_RUNAWAY_KM_PER_M, 1.5)
    tcrit_run = tau_crit_yr(ELL_M, vk_run)
    frac_band_thinning_paced = float(np.mean(tau < tcrit_run))
    return dict(
        what="derived residence number Ro for grounding-line intrusion: thinning-paced vs hydraulic-limited",
        definitions=dict(Ro="v_kin/v_obs (>1 hydraulic-limited)",
                         Ro_pred="v_kin*tau_hyd/ell", tau_crit="ell/v_kin (regime boundary)",
                         D_hyd="ell^2/tau_hyd (implied hydraulic diffusivity)"),
        ell_m=ELL_M, tau_hyd_band_yr=list(TAU_HYD_BAND_YR),
        D_hyd_implied_m2_s=[D_hyd_m2_s(TAU_HYD_BAND_YR[1]), D_hyd_m2_s(TAU_HYD_BAND_YR[0])],
        regimes=out_regimes,
        runaway_headline=dict(
            A_km_per_m=A_RUNAWAY_KM_PER_M, dHdt_m_yr=1.5, v_kin_km_yr=float(vk_run),
            tau_crit_yr=float(tcrit_run),
            frac_residence_band_thinning_paced=frac_band_thinning_paced,
            verdict=("thinning-paced (Ro~1) over %.0f%% of the G.4 residence band; "
                     "hydraulic-limited only at the slow (~2 yr) end" % (100 * frac_band_thinning_paced))),
        falsification=("measure v_obs (DInSAR/repeat radar) over the 754 runaway cells -> "
                       "Ro=v_kin/v_obs; Ro>>1 places tau_hyd>tau_crit (hydraulic-limited), "
                       "Ro~1 confirms thinning-pacing. Regress Ro vs a u_* proxy to test the "
                       "G.2 sqrt-law transfer (H.1.2 protocol)."),
        references="this repo §H.1.2, §G.4 (hydraulic_lag_derivation); Werder et al. 2013; Hewitt 2013; Schoof 2010",
        regime_map=dict(tau_hyd_yr=tau.tolist(), v_kin_km_yr=vk_grid.tolist(),
                        Ro=Ro.tolist(), tau_crit_curve_yr=[float(tau_crit_yr(ELL_M, v)) for v in vk_grid]),
        verdict=(
            f"Ro_pred=v_kin*tau_hyd/ell with regime boundary tau_crit=ell/v_kin. For the runaway "
            f"tail (A=0.70 km/m, dH/dt=1.5 m/yr) v_kin={vk_run:.2f} km/yr and tau_crit={tcrit_run:.2f} yr; "
            f"since {100*frac_band_thinning_paced:.0f}% of the G.4 residence band lies below tau_crit, "
            "intrusion is predicted THINNING-PACED (Ro~1) for these cells, turning hydraulic-limited "
            "only for residence at the slow ~2 yr end. The conjecture is now a derived, falsifiable "
            "regime prediction rather than an unmeasured ratio."),
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rm = res["regime_map"]
    tau = np.array(rm["tau_hyd_yr"]); vk = np.array(rm["v_kin_km_yr"]); Ro = np.array(rm["Ro"])
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    pcm = ax[0].pcolormesh(tau, vk, np.log10(Ro), shading="auto", cmap="RdBu_r",
                           vmin=-2, vmax=2)
    ax[0].plot(tau, [res["ell_m"] / (t * 1000) for t in tau], "k-", lw=2, label=r"$Ro=1$ (boundary)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"hydraulic residence $\tau_{hyd}$ [yr]  (§G.4 band)")
    ax[0].set_ylabel(r"kinematic speed $v_{kin}$ [km/yr]")
    ax[0].set_title("(a) intrusion regime map: $\\log_{10}Ro$\nblue=thinning-paced, red=hydraulic-limited")
    fig.colorbar(pcm, ax=ax[0], label=r"$\log_{10} Ro$")
    ax[0].legend(fontsize=8)
    # (b) Ro vs residence for margin-median and runaway, dH/dt=1.5
    tt = np.geomspace(*res["tau_hyd_band_yr"], 100)
    for label, A, c in (("margin median (A=0.09)", A_MEDIAN_KM_PER_M, "#1f77b4"),
                        ("runaway tail (A=0.70)", A_RUNAWAY_KM_PER_M, "#d62728")):
        vk1 = v_kin_km_yr(A, 1.5)
        ax[1].plot(tt, [Ro_pred(t, vk1) for t in tt], color=c, lw=2, label=label)
    ax[1].axhline(1.0, color="k", ls="--", lw=1, label="Ro=1")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\tau_{hyd}$ [yr]"); ax[1].set_ylabel("predicted Ro")
    ax[1].set_title("(b) Ro vs residence (dH/dt=1.5 m/yr)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("§H.1.2 intrusion residence number: thinning-paced vs hydraulic-limited", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "reports", "intrusion_residence_number.json")))
    a = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    h = res["runaway_headline"]
    print("=== §H.1.2 intrusion residence number Ro (derived) ===")
    print(f"  ell={res['ell_m']:.0f} m; D_hyd implied {res['D_hyd_implied_m2_s'][0]:.2f}-{res['D_hyd_implied_m2_s'][1]:.2f} m^2/s")
    print(f"  runaway tail: v_kin={h['v_kin_km_yr']:.2f} km/yr, tau_crit={h['tau_crit_yr']:.2f} yr")
    print(f"  -> {h['verdict']}")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
