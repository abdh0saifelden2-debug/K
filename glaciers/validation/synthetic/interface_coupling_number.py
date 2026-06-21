r"""§A.1 — the interface as a *coupling surface*: a frequency-dependent ice thermal
admittance and the criterion for when ice is a participating medium vs a passive BC.

What this closes (the §A.1 [HYP])
---------------------------------
FUTURE_WORK §A.1 proposes that the ice-water interface is "not a boundary condition
but a coupling surface where [the elliptic-pressure and parabolic-temperature]
operators interact", with ice "a participating medium with its own (slow) thermal
clock" — but left the *coupling-surface formalism* as `[HYP]` scaffolding. §B.2
already derived the ice-side interface-flux transfer function in closed form
(`ice_kernel_synthetic.py`):

    H(s) = q_ice'(s)/v'(s) = A[1 - sqrt(1 + 4 tau_d s)]/s,
    A = k_th theta_far Vbar^2/(2 kappa^2),   tau_d = kappa/Vbar^2.

This module turns the scaffolding into a **quantitative, frequency-resolved coupling
criterion**.

The deliverable — the interface coupling number Lambda(omega) [DERIVED]
----------------------------------------------------------------------
The interface velocity update is `rho_i L v = q_water - q_ice`. Writing the ice flux
as the linear response `q_ice'(s) = H(s) v'(s)` gives, in the frequency domain,

    v'(s) = q_water'(s) / (rho_i L + H(s)),

so the *passive-BC* limit `v = q_water/(rho_i L)` is corrected by the dimensionless
**interface coupling number**

    Lambda(omega) = |H(i omega)| / (rho_i L).                                   (★)

Two exact limits make it a clean criterion:

  * DC (slow forcing, omega -> 0):  H(0) = -rho c theta_far, so
        Lambda(0) = c_i|theta_far|/L = St  (the Stefan number, <= 0.06).
    The ice fully participates, but only at its latent-heat-limited weight St.
  * High frequency (fast forcing, omega tau_d >> 1):  H ~ -2A sqrt(tau_d/s) -> 0, so
        Lambda(omega) -> 0  (the ice is *frozen*; the passive-adiabatic BC is exact).

The crossover is the **ice thermal clock** 1/tau_d. Because the observed surge band
(0.02-2 yr) is far faster than tau_d = kappa/Vbar^2 (~10^3-10^4 yr for Vbar~0.01-1
m/yr), at surge timescales `Lambda << St <= 0.06`: ice is a passive BC to well under
1%. Ice only becomes a >few-% participant for forcing slower than tau_d (millennial
thinning). So "coupling surface vs passive BC" is decided by `Lambda(omega) = St *
|hat H(omega tau_d)|`: a high-pass-filtered Stefan weight.

Honest scope
------------
Lambda bounds the *quantitative* correction to the interface velocity; a small
Lambda does NOT preclude a *qualitative* (stability) effect — the §B.1 two-phase
Stefan run stabilises the melt amplitude (0.41 vs 0.21 water-only) even though
St<=0.06, because a small dissipative term can still flip a marginal mode. So:
passive BC is accurate to O(Lambda) for steady/lag *magnitudes*, but the
participating-ice (two-phase) treatment is still required where it changes stability.
theta_far and Vbar are problem inputs (per §B.2), so Lambda is bounded, not pinned.
No GPU, no data download.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SEC_PER_YR = 365.25 * 86400.0
# ice constants (Cuffey & Paterson 2010), identical to thermal_tail_amplitude.py
RHO_I, C_ICE, L_FUSION, K_TH = 917.0, 2009.0, 3.34e5, 2.1
RHO_C = RHO_I * C_ICE
KAPPA = K_TH / RHO_C            # ~1.14e-6 m^2/s
OBS_BAND_YR = (0.02, 2.0)


def stefan_number(theta_far, c_i=C_ICE, L=L_FUSION):
    return c_i * abs(theta_far) / L


def H_transfer(s, theta_far, Vbar, kappa=KAPPA):
    """§B.2 interface-flux transfer function H(s)=A[1-sqrt(1+4 tau_d s)]/s (complex s).
    Uses the principal sqrt branch (decaying root)."""
    A = K_TH * theta_far * Vbar ** 2 / (2.0 * kappa ** 2)
    tau_d = kappa / Vbar ** 2
    s = np.asarray(s, dtype=complex)
    out = np.empty_like(s)
    small = np.abs(s) * tau_d < 1e-8
    # series limit H(0) = -2 A tau_d = -rho c theta_far
    out[small] = -2.0 * A * tau_d
    ss = s[~small]
    out[~small] = A * (1.0 - np.sqrt(1.0 + 4.0 * tau_d * ss)) / ss
    return out


def coupling_number(omega, theta_far, Vbar, kappa=KAPPA, rho_i=RHO_I, L=L_FUSION):
    """Lambda(omega) = |H(i omega)|/(rho_i L)  (★)."""
    H = H_transfer(1j * np.asarray(omega, float), theta_far, Vbar, kappa)
    return np.abs(H) / (rho_i * L)


def tau_d_yr(Vbar, kappa=KAPPA):
    return kappa / Vbar ** 2 / SEC_PER_YR


def verify(theta_far=-1.0, Vbar=0.1 / SEC_PER_YR):
    """DC limit of Lambda equals the Stefan number; high-omega limit -> 0."""
    st = stefan_number(theta_far)
    lam_dc = float(coupling_number(1e-30, theta_far, Vbar))
    tau_d = kappa = KAPPA / Vbar ** 2
    lam_hi = float(coupling_number(1e6 / tau_d, theta_far, Vbar))
    return dict(stefan_number=st, lambda_dc=lam_dc,
                dc_rel_err=abs(lam_dc - st) / st, lambda_highfreq=lam_hi)


def run():
    theta_far, Vbar = -1.0, 0.1 / SEC_PER_YR        # representative ice-stream bed
    ver = verify(theta_far, Vbar)
    td = tau_d_yr(Vbar)
    # Lambda across forcing period from 1e-3 yr (fast surge) to 1e6 yr (deep slow)
    periods_yr = np.geomspace(1e-3, 1e6, 200)
    omega = 2 * np.pi / (periods_yr * SEC_PER_YR)
    lam = coupling_number(omega, theta_far, Vbar)
    st = stefan_number(theta_far)
    # in observed surge band: Lambda is tiny (ice frozen)
    in_band = (periods_yr >= OBS_BAND_YR[0]) & (periods_yr <= OBS_BAND_YR[1])
    lam_band_max = float(lam[in_band].max()) if in_band.any() else None
    # crossover period: where Lambda = St/2
    target = st / 2
    cross_idx = int(np.argmin(np.abs(lam - target)))
    cross_period_yr = float(periods_yr[cross_idx])
    # sweep over Vbar and theta_far: is the surge band always frozen (Lambda<<St)?
    rng = np.random.default_rng(0)
    band_ratios = []
    for _ in range(2000):
        th = np.exp(rng.uniform(np.log(0.1), np.log(10.0)))
        vb = np.exp(rng.uniform(np.log(1e-3), np.log(1.0))) / SEC_PER_YR
        Pmid = np.sqrt(OBS_BAND_YR[0] * OBS_BAND_YR[1]) * SEC_PER_YR
        lam_mid = float(coupling_number(2 * np.pi / Pmid, -th, vb))
        band_ratios.append(lam_mid / stefan_number(th))
    band_ratios = np.array(band_ratios)
    dec = max(1, periods_yr.size // 60)
    idx = list(range(0, periods_yr.size, dec))
    return dict(
        what="interface coupling number Lambda(omega)=|H(i omega)|/(rho_i L): the "
             "frequency-resolved weight of ice participation; passive BC valid when Lambda<<1",
        transfer_function="H(s)=A[1-sqrt(1+4 tau_d s)]/s (B.2); A=k_th theta_far Vbar^2/(2 kappa^2), tau_d=kappa/Vbar^2",
        dc_limit="Lambda(0)=c_i|theta_far|/L = St (Stefan number, <=0.06)",
        hf_limit="Lambda(omega tau_d>>1)->0 (ice frozen, passive-adiabatic BC exact)",
        verification=ver,
        tau_d_yr_baseline=td,
        stefan_number_baseline=st,
        lambda_in_surge_band_max=lam_band_max,
        crossover_period_yr=cross_period_yr,
        surge_band_frozen=bool(lam_band_max is not None and lam_band_max < 0.1 * st),
        sweep_surge_band_lambda_over_St=dict(
            p50=float(np.percentile(band_ratios, 50)),
            p95=float(np.percentile(band_ratios, 95)),
            max=float(band_ratios.max()),
            frac_below_0p1=float(np.mean(band_ratios < 0.1))),
        criterion=("passive BC error ~ Lambda(omega) = St*|hat H(omega tau_d)|; at the "
                   "observed surge band Lambda<<St<=0.06 (ice frozen), so ice is a passive "
                   "BC to <1% there; ice participates (up to St) only for forcing slower "
                   "than the ice clock tau_d~10^3-10^4 yr."),
        honest_scope=("Lambda bounds the QUANTITATIVE velocity correction; a small Lambda can "
                      "still flip STABILITY (the B.1 two-phase melt-amplitude stabilisation, "
                      "0.41 vs 0.21), which the participating-ice treatment is still needed for."),
        references="this repo §B.2 (ice_kernel_synthetic), §B.1; Cuffey & Paterson 2010; Stefan/Nye moving-boundary",
        series=dict(period_yr=[periods_yr[i] for i in idx],
                    lambda_=[float(lam[i]) for i in idx],
                    lambda_over_St=[float(lam[i] / st) for i in idx]),
        verdict=(
            f"Lambda(0)=St={st:.3f} (DC rel-err {ver['dc_rel_err']:.1e}) and Lambda->0 at high "
            f"frequency, crossover at the ice clock ~{td:.0f} yr (tau_d for Vbar=0.1 m/yr). In the "
            f"observed 0.02-2 yr surge band Lambda<{lam_band_max:.1e} (<<St), so the interface is "
            "a passive BC to <1% for the sliding-lag physics; ice becomes a participating medium "
            "(up to the St<=6% weight) only for millennial forcing. The coupling-surface scaffolding "
            "is now a frequency-resolved criterion Lambda(omega)=St*|hat H(omega tau_d)|."),
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = res["series"]
    P = np.array(s["period_yr"]); lam = np.array(s["lambda_"]); st = res["stefan_number_baseline"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.2))
    ax.plot(P, lam, color="#6a3d9a", lw=2, label=r"$\Lambda(\omega)=|H|/(\rho_i L)$")
    ax.axhline(st, color="k", ls="--", lw=1, label=f"St = {st:.3f} (DC limit)")
    ax.axvspan(OBS_BAND_YR[0], OBS_BAND_YR[1], color="#d95f0e", alpha=0.18,
               label="observed surge band 0.02-2 yr")
    ax.axvline(res["tau_d_yr_baseline"], color="gray", ls=":", lw=1,
               label=f"ice clock $\\tau_d$~{res['tau_d_yr_baseline']:.0f} yr")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("forcing period [yr]  (fast <- -> slow)")
    ax.set_ylabel(r"interface coupling number $\Lambda$")
    ax.set_title("§A.1 interface coupling number: ice is a passive BC at surge\n"
                 "timescales (frozen), a participating medium only for slow forcing")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "reports", "interface_coupling_number.json")))
    a = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    v = res["verification"]
    print("=== §A.1 interface coupling number ===")
    print(f"  Lambda(0)=St={v['stefan_number']:.4f}  (DC rel-err {v['dc_rel_err']:.1e}); "
          f"Lambda(high-omega)={v['lambda_highfreq']:.2e}")
    print(f"  ice clock tau_d ~ {res['tau_d_yr_baseline']:.0f} yr; surge-band Lambda_max < "
          f"{res['lambda_in_surge_band_max']:.1e}; frozen={res['surge_band_frozen']}")
    sw = res["sweep_surge_band_lambda_over_St"]
    print(f"  sweep Lambda/St in surge band: p50={sw['p50']:.1e} p95={sw['p95']:.1e} frac<0.1={sw['frac_below_0p1']:.2f}")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
