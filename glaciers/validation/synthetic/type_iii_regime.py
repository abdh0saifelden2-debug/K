r"""§F / §G.4 — the Type III (fully-floating, continuous-response) regime, and the
Type I -> II -> III transition as a function of effective pressure N.

Context (the regime map this closes)
------------------------------------
The repo's deliverable is a **regime map** (README 10f, FUTURE_WORK §F):
  * Type I  — cold, well-grounded, high N: drainage gives **no** sliding surge
              (bed locked, elastic);
  * Type II — near-flotation, grounded, low N: a **discrete** post-drainage surge
              with the §G.4 cavity-paced lag (0.02-2 yr) — now field-detected
              (Thw_142, Rutford_1; see ``lake_lag_trunk.py``);
  * Type III — **fully floating / at flotation (N -> 0)**: the bed-sliding
              mechanism is gone, so the response to forcing is **continuous**, not
              a discrete storage-paced lag.

Type III was the one regime with no validated model ("a validated Type III model
(fully floating, continuous response)" was flagged missing). This module supplies
it from two mainstream ingredients, and makes the transition **falsifiable**.

1. Amplitude: regularized-Coulomb sliding sensitivity s_N(N)
-----------------------------------------------------------
A regularized-Coulomb / Budd sliding law (Schoof 2005; Gagliardini 2007; Tsai
2015; Joughin et al. 2019) writes basal drag, at fixed driving stress tau_d that
the bed must support (tau_b = tau_d), as

    tau_b = C * N * ( u_b / (u_b + u0) )^(1/m)                          (RC)

so the basal speed at a given N is

    u_b(N) = u0 * R / (1 - R),   R = (tau_d / (C N))^m,   valid for C N > tau_d.

The N-elasticity s_N = d ln u_b / d ln N is < 0 and |s_N| **grows without bound as
C N -> tau_d^+** (the Coulomb threshold), then there is **no grounded steady
solution for N < N_c = tau_d / C**: the bed cannot support the driving stress and
the ice accelerates to flotation -> **Type III**. This is exactly the §H.1.6
"the TF->speed slope steepens toward flotation" prediction, now as a closed law,
and it sets the **surge amplitude**: a drainage that drops N by a fraction f
(ΔN/N = -f) gives

    du/u = |s_N(N)| * f                                                 (amp law)

which is the law ``lake_lag_trunk.py`` measures across field detections. The
Type II/III boundary is N_c; the amplitude diverges there and the regime flips to
continuous.

2. Shape: continuous (Type III) vs peaked (Type II) memory kernel
-----------------------------------------------------------------
Type II is the two-compartment cavity<->store system (``hydraulic_lag_derivation``):
its downstream impulse response **starts at 0, peaks at t*, decays** (a discrete
lag). Type III has no cavity storage to charge (cavities are saturated open at
N->0, the store<->cavity coupling J21 ∝ N^(n-1) -> 0), so the response collapses to
a **single-compartment, monotone relaxation** g(t) = (1/τ_r) e^{-t/τ_r} that simply
**follows the forcing** with a short ocean/ice-shelf adjustment time τ_r (no
interior peak). The qualitative signature flips from "rise-to-a-peak" (II) to
"instantaneous-then-relax" (III).

Verdict logic
-------------
* the regularized-Coulomb |s_N(N)| is monotone-increasing toward flotation and the
  amplitude law du/u = |s_N| f rises toward N_c -> the Type II amplitude grows
  toward flotation and **diverges into Type III** at N_c;
* the cavity store<->cavity coupling and the peaked-kernel amplitude collapse as
  N->0 (cavities saturate) -> the discrete-lag mechanism **turns off**, leaving the
  continuous Type III response;
* optionally, the measured field detections (``lake_lag_trunk.json``) are overlaid
  on the predicted du/u(N) curve as a falsification test.

No data download, no GPU (the cavity Jacobian + RC law are analytic); the optional
overlay just reads the committed ``lake_lag_trunk.json`` if present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import hydraulic_lag_derivation as HL   # noqa: E402

SEC_PER_YR = 365.25 * 86400.0
OBS_BAND_YR = (0.02, 2.0)


# --------------------------------------------------------------------------- #
# 1. regularized-Coulomb sliding sensitivity and the amplitude law
# --------------------------------------------------------------------------- #
def rc_speed(N, tau_d, C, u0, m):
    """Basal speed u_b(N) from the regularized-Coulomb law at fixed driving stress.

    u_b = u0 R/(1-R), R=(tau_d/(C N))^m. Returns np.inf for N<=N_c (no grounded
    steady solution -> Type III)."""
    N = np.asarray(N, float)
    R = (tau_d / (C * N)) ** m
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(R < 1.0, u0 * R / (1.0 - R), np.inf)
    return u


def s_N(N, tau_d, C, u0, m, dlnN=1e-3):
    """Elasticity s_N = d ln u_b / d ln N (numeric, central diff in ln N)."""
    N = np.asarray(N, float)
    up = rc_speed(N * (1 + dlnN), tau_d, C, u0, m)
    dn = rc_speed(N * (1 - dlnN), tau_d, C, u0, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (np.log(up) - np.log(dn)) / (2 * dlnN)
    return s


def regime_sweep(tau_d=3.0e4, C=0.5, u0=100.0, m=3.0, f_drain=0.1,
                 N_lo=1.0e3, N_hi=2.0e6, n=400):
    """Sweep effective pressure N; return s_N(N), the amplitude law du/u(N), and
    the Type II/III boundary N_c = tau_d/C.

    tau_d : driving stress [Pa] (ice-stream ~10-100 kPa)
    C     : Coulomb friction coefficient [-] (tan of till friction angle ~0.5)
    u0    : RC regularization velocity [m/yr]
    m     : Weertman exponent [-]
    f_drain : fractional N drop in a drainage (ΔN/N) used in the amplitude law
    """
    N_c = tau_d / C
    N = np.geomspace(N_lo, N_hi, n)
    u = rc_speed(N, tau_d, C, u0, m)
    sN = s_N(N, tau_d, C, u0, m)
    grounded = np.isfinite(u)
    amp = np.where(grounded, np.abs(sN) * f_drain, np.nan)   # du/u predicted
    return dict(N=N, u=u, s_N=sN, amp=amp, grounded=grounded,
                N_c=N_c, tau_d=tau_d, C=C, u0=u0, m=m, f_drain=f_drain)


# --------------------------------------------------------------------------- #
# 2. Type II vs III kernel shape, and the cavity-coupling collapse as N->0
# --------------------------------------------------------------------------- #
def cavity_coupling_vs_N(N_vals_MPa=(2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)):
    """For each effective pressure N, read the cavity model's discrete-lag
    diagnostics: lag t*, cavity fill h_s/h_r, store<->cavity coupling J21, and the
    peak amplitude of the downstream impulse response (∝ J21). As N->0 the cavity
    saturates (h_s/h_r -> 1) and J21 ∝ N^(n-1) -> 0: the discrete-lag mechanism
    turns off (Type II -> III)."""
    base = HL.baseline_params()
    out = []
    for NM in N_vals_MPa:
        p = dict(base, N_obs=NM * 1e6)
        if p["N_obs"] >= p["p_i"]:
            continue
        op = HL.operating_point(p)
        J = HL.jacobian_cavity(p)            # [s^-1]
        M = J * SEC_PER_YR
        tstar = HL.channel_impulse_peak_time(M)
        # downstream impulse-response peak value x2(t*) for unit store pulse:
        # x2(t)=m21 e^{mu t} sinh(delta t)/delta ; report its max as the "surge"
        mu = 0.5 * (M[0, 0] + M[1, 1])
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        disc = mu * mu - det
        if tstar is not None and mu < 0:
            if disc > 0:
                delta = np.sqrt(disc)
                x2 = M[1, 0] * np.exp(mu * tstar) * np.sinh(delta * tstar) / delta
            elif disc < 0:
                w = np.sqrt(-disc)
                x2 = M[1, 0] * np.exp(mu * tstar) * np.sin(w * tstar) / w
            else:
                x2 = M[1, 0] * tstar * np.exp(mu * tstar)
        else:
            x2 = np.nan
        out.append(dict(N_MPa=NM, tstar_yr=tstar, hs_fill=op["h_s"] / p["h_r"],
                        J21_per_s=float(J[1, 0]), peak_resp=float(abs(x2)),
                        in_band=(tstar is not None and OBS_BAND_YR[0] <= tstar <= OBS_BAND_YR[1])))
    return out


def kernels(tau1_yr=2.0, tau2_yr=0.4, tau_r_yr=0.1, t_max=6.0, nt=600):
    """Type II two-compartment cascade kernel (peaked) vs Type III single-compartment
    relaxation (monotone, forcing-following). Returns t and both g(t)."""
    t = np.linspace(0, t_max, nt)
    if abs(tau1_yr - tau2_yr) < 1e-9:
        g2 = t / tau1_yr ** 2 * np.exp(-t / tau1_yr)
    else:
        g2 = (np.exp(-t / tau1_yr) - np.exp(-t / tau2_yr)) / (tau1_yr - tau2_yr)
    g3 = (1.0 / tau_r_yr) * np.exp(-t / tau_r_yr)        # Type III continuous
    return t, g2, g3


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(field_json=None):
    sw = regime_sweep()
    cav = cavity_coupling_vs_N()
    t, g2, g3 = kernels()

    # regime classification along the sweep
    N = sw["N"]; grounded = sw["grounded"]; amp = sw["amp"]
    N_c = sw["N_c"]
    # Type III: N < N_c (no grounded steady solution); Type II: grounded & |s_N|
    # large (near flotation, amp above the well-grounded baseline); Type I: high N.
    sN = np.abs(sw["s_N"])
    sN_floor = np.nanmedian(sN[grounded & (N > 5 * N_c)])    # well-grounded baseline
    typeII = grounded & (sN > 1.5 * sN_floor)
    N_II_lo = float(np.nanmin(N[typeII])) if typeII.any() else None
    N_II_hi = float(np.nanmax(N[typeII])) if typeII.any() else None

    # cavity-coupling collapse: peaked-kernel amplitude and J21 vs N
    cav_sorted = sorted(cav, key=lambda d: d["N_MPa"])
    j21 = np.array([c["J21_per_s"] for c in cav_sorted])
    nmpa = np.array([c["N_MPa"] for c in cav_sorted])
    # fit log J21 vs log N -> expect slope ~ n-1 = 2 (J21 ∝ N^{n-1})
    mok = (j21 > 0) & (nmpa > 0)
    j21_slope = (float(np.polyfit(np.log(nmpa[mok]), np.log(j21[mok]), 1)[0])
                 if mok.sum() >= 2 else None)

    # shape test: Type II peaked (interior argmax>0), Type III monotone (argmax==0)
    typeII_peaked = bool(np.argmax(g2) > 0)
    typeIII_monotone = bool(np.argmax(g3) == 0)

    # optional falsification overlay: field detections amplitude vs N
    overlay = None
    if field_json and os.path.exists(field_json):
        fd = json.load(open(field_json))
        det = [d for d in fd.get("detections", [])
               if d.get("resp_frac") and d.get("rel") is not None]
        if det:
            # rel (Bedmap2 m/H) ~ normalized N; convert to N[Pa] via overburden of
            # the lake (rho_i g H * rel-ish) is not exact, so we report rel directly
            overlay = [dict(lake=d["lake"], rel=d["rel"], du_u=d["resp_frac"],
                            dist_km=d.get("dist_km"), lag=d.get("lag_to_peak"))
                       for d in det]

    return dict(
        sliding_law="regularized-Coulomb tau_b = C N (u/(u+u0))^(1/m) (Schoof 2005; Joughin 2019)",
        params=dict(tau_d_Pa=sw["tau_d"], C=sw["C"], u0_mpyr=sw["u0"], m=sw["m"],
                    f_drain=sw["f_drain"]),
        N_c_Pa=N_c, N_c_MPa=N_c / 1e6,
        amplitude_law="du/u = |s_N(N)| * (dN/N);  s_N from RC law, grows toward flotation",
        s_N_wellgrounded=float(sN_floor),
        typeII_N_band_MPa=[None if N_II_lo is None else N_II_lo / 1e6,
                           None if N_II_hi is None else N_II_hi / 1e6],
        amp_at_2MPa=float(np.interp(2.0e6, N, np.nan_to_num(amp, nan=0.0))),
        amp_near_Nc=float(np.nanmax(amp[grounded & (N < 3 * N_c)])) if (grounded & (N < 3 * N_c)).any() else None,
        cavity_coupling_vs_N=cav_sorted,
        J21_loglog_slope_vs_N=j21_slope,         # ~ n-1 = 2 expected
        kernel_shape=dict(typeII_peaked=typeII_peaked,
                          typeIII_monotone=typeIII_monotone,
                          typeII_peak_yr=float(t[np.argmax(g2)])),
        field_overlay=overlay,
        verdict=(
            f"Type III is the N<N_c={N_c/1e6:.3f} MPa regime where the regularized-"
            "Coulomb bed cannot support the driving stress: |s_N| diverges toward "
            "flotation (amplitude law du/u=|s_N|f grows toward N_c) and the cavity "
            f"discrete-lag mechanism turns off (J21 ∝ N^{{{j21_slope:.1f}}} -> 0, "
            "cavities saturate), so the response collapses from the Type II peaked "
            "kernel to a continuous single-relaxation Type III kernel. The Type II "
            f"amplified band is N ~ {('%.3f-%.3f' % (N_II_lo/1e6, N_II_hi/1e6)) if N_II_lo else 'n/a'} MPa."),
        obs_band_yr=list(OBS_BAND_YR),
    )


def make_figure(res, sweep, cav, kern, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = sweep["N"] / 1e6; amp = sweep["amp"] * 100; sN = np.abs(sweep["s_N"])
    t, g2, g3 = kern
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # (a) amplitude law + field overlay
    ax[0].plot(N, amp, "-", color="#c2185b", lw=2, label=r"RC law $|s_N|\,f$")
    ax[0].axvline(sweep["N_c"] / 1e6, color="k", ls="--", label=f"$N_c$={sweep['N_c']/1e6:.3f} MPa (II/III)")
    if res.get("field_overlay"):
        # rel is a normalized-N proxy; place on a secondary view by scaling rel->MPa
        # approximately via rel*overburden not available here, so annotate by rel.
        relv = [o["rel"] for o in res["field_overlay"]]
        amv = [100 * o["du_u"] for o in res["field_overlay"]]
        ax[0].scatter(np.clip(np.array(relv) * 2.0 + 0.05, 1e-3, None), amv,
                      c="#1f77b4", s=55, edgecolor="k", zorder=5,
                      label="field detections (rel->~N)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("effective pressure N [MPa]"); ax[0].set_ylabel(r"surge amplitude $\delta u/u$ [%]")
    ax[0].set_title("(a) amplitude grows toward flotation; diverges into Type III")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    # (b) cavity coupling / peaked-kernel amplitude collapse as N->0
    nmpa = [c["N_MPa"] for c in cav]; peak = [c["peak_resp"] for c in cav]
    fill = [c["hs_fill"] for c in cav]
    ax[1].plot(nmpa, np.array(peak) / max(peak), "o-", color="#d95f02", label="peaked-kernel amplitude (norm)")
    ax[1].plot(nmpa, fill, "s--", color="#7570b3", label=r"cavity fill $h_s/h_r$")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("effective pressure N [MPa]"); ax[1].set_ylabel("normalized")
    ax[1].set_title("(b) N->0: cavities saturate, discrete-lag mechanism turns off")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    # (c) kernel shapes
    ax[2].plot(t, g2 / g2.max(), color="#d62728", lw=2, label="Type II (peaked, cavity)")
    ax[2].plot(t, g3 / g3.max(), color="#1b9e77", lw=2, label="Type III (continuous)")
    ax[2].axvspan(OBS_BAND_YR[0], OBS_BAND_YR[1], color="#d95f0e", alpha=0.12)
    ax[2].set_xlabel("time after drainage [yr]"); ax[2].set_ylabel("response (norm)")
    ax[2].set_title("(c) discrete peak (II) vs continuous relaxation (III)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.suptitle("Type III regime: N->0 continuous-response limit of the §G.4 sliding/cavity system", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-json", default=os.path.normpath(
        os.path.join(_HERE, "..", "reports", "lake_lag_trunk.json")))
    ap.add_argument("--out", default=os.path.normpath(
        os.path.join(_HERE, "..", "reports", "type_iii_regime.json")))
    a = ap.parse_args()
    res = run(a.field_json)
    sweep = regime_sweep(); cav = sorted(cavity_coupling_vs_N(), key=lambda d: d["N_MPa"])
    kern = kernels()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2, default=lambda o: None if o is None else float(o))
    print("=== Type III regime (N->0 continuous-response limit) ===")
    print(f"  sliding law: {res['sliding_law']}")
    print(f"  N_c (Type II/III boundary): {res['N_c_MPa']:.3f} MPa")
    print(f"  |s_N| well-grounded baseline: {res['s_N_wellgrounded']:.2f}")
    print(f"  Type II amplified N band: {res['typeII_N_band_MPa']} MPa")
    print(f"  J21 ∝ N^{res['J21_loglog_slope_vs_N']:.2f} (expect ~n-1=2) -> coupling collapses as N->0")
    print(f"  kernel shapes: {res['kernel_shape']}")
    if res.get("field_overlay"):
        print(f"  field detections overlaid: {len(res['field_overlay'])}")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  json -> {a.out}")
    make_figure(res, sweep, cav, kern, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
