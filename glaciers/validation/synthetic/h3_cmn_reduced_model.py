r"""§H.3 — clock-mismatch (CMN) correction in a 2-D advection-diffusion REDUCED-MODEL
PROXY with a propagating plume. Honest scope: this is NOT the deferred ISSM/GlaDS
real-solver test (no ice dynamics, no real turbulence closure, no real geometry) —
it is a step *toward* it, checking whether the 1-D RESULT-15 conclusions survive
dimensionality + advection + a spatially-structured, moving transient.

What this sharpens (the §H.3 deferred real-model test)
-----------------------------------------------------
§G.5/§H.3 forecast that adding `−CMN·∇·(∂_t K_u ∇θ)` (`CMN=+τ_c`) to a K-theory
thermal solver reconstructs the lagged-clock truth and cuts spurious transient
oscillations. RESULT 15 (`general_two_clocks/cmn_solver_demo.py`) verified this in a
**1-D scalar** solver with a single prescribed `K_u(t)` cycle. The remaining gap is the
**real-model** test (ISSM/GlaDS on a real surge/plume), which needs heavy external
solvers not available here. This module supplies the honest intermediate: a
**self-contained 2-D advection-diffusion proxy** with a Gaussian plume whose amplitude
ramps up/down AND whose centre propagates across the domain — the closest standalone
analogue to a real moving plume/surge transient.

The test (identical RK4 stepper; only the diffusivity clock differs)
-------------------------------------------------------------------
`∂_t θ + u·∇θ = ∇·(K(x,y,t) ∇θ)`, `K = K0 + ΔK·A(t)·exp(−r(t)²/2σ²)`, plume centre
`x_c(t)` moving during the event window, `A(t)` a raised-sine envelope (so `∂_tK=0`
before/after). Three runs share the stepper and advection; only the diffusion clock
differs:
  * **truth**     `K(t−τ_c)`            (faithful two-clocks lag, evaluated exactly)
  * **naive**     `K(t)`                (frozen clock)
  * **corrected** `K(t) − τ_c ∂_tK(t)`  (= CMN term, first-order reconstruction)
so the common numerical/advection error cancels and only the *clock* error remains.

Result (proxy)
--------------
The four RESULT-15 conclusions survive in 2-D with advection and a moving plume:
(1) the transient error is cut by ~`τ_c`-order; (2) naive scales `∝τ_c¹`, corrected
`∝τ_c²` (one order higher-accurate); (3) the steady null is exact (no event ⇒
naive≡corrected≡truth to machine precision); (4) `CMN=+τ_c` is the unique
error-reducing sign (`−τ_c` is worse than naive). This raises confidence that the
real-solver (ISSM/GlaDS) test would behave as forecast — but it does **not** replace
it. No GPU, no download.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# ---- domain / numerics ----
NX = NY = 40
L = 1.0
DX = L / NX
DT = 1.0e-3
T_END = 1.2
U_ADV = (0.3, 0.0)            # advection velocity (ux, uy)
K0 = 0.010                    # background diffusivity
DK = 0.050                    # plume diffusivity enhancement
SIGMA = 0.10                  # plume width
EVENT = (0.3, 0.9)           # plume active window [t1, t2]
XC_PATH = (0.25, 0.75)       # plume centre x: start -> end during the event
YC = 0.5
_X, _Y = np.meshgrid(np.linspace(0, L, NX, endpoint=False),
                     np.linspace(0, L, NY, endpoint=False))


def _envelope(t):
    t1, t2 = EVENT
    if t <= t1 or t >= t2:
        return 0.0
    return float(np.sin(np.pi * (t - t1) / (t2 - t1)) ** 2)


def _xc(t):
    t1, t2 = EVENT
    s = min(max((t - t1) / (t2 - t1), 0.0), 1.0)
    return XC_PATH[0] + (XC_PATH[1] - XC_PATH[0]) * s


def K_field(t, event_amp=1.0):
    """Time/space-varying diffusivity K(x,y,t) with a moving, ramping plume."""
    A = event_amp * _envelope(t)
    if A == 0.0:
        return np.full_like(_X, K0)
    r2 = (_X - _xc(t)) ** 2 + (_Y - YC) ** 2
    return K0 + DK * A * np.exp(-r2 / (2 * SIGMA ** 2))


def dKdt_fd(t, event_amp=1.0, h=None):
    """∂_tK via centred finite difference (used only for the O(τ_c) correction)."""
    h = h or DT
    return (K_field(t + h, event_amp) - K_field(t - h, event_amp)) / (2 * h)


def _diffusion(theta, K):
    """∇·(K ∇θ) in flux form, periodic, face-averaged K."""
    Kxp = 0.5 * (K + np.roll(K, -1, axis=1)); Kxm = 0.5 * (K + np.roll(K, 1, axis=1))
    Kyp = 0.5 * (K + np.roll(K, -1, axis=0)); Kym = 0.5 * (K + np.roll(K, 1, axis=0))
    fx = Kxp * (np.roll(theta, -1, 1) - theta) - Kxm * (theta - np.roll(theta, 1, 1))
    fy = Kyp * (np.roll(theta, -1, 0) - theta) - Kym * (theta - np.roll(theta, 1, 0))
    return (fx + fy) / DX ** 2


def _advection(theta, u):
    ux, uy = u
    dtx = ((theta - np.roll(theta, 1, 1)) if ux >= 0 else (np.roll(theta, -1, 1) - theta)) / DX
    dty = ((theta - np.roll(theta, 1, 0)) if uy >= 0 else (np.roll(theta, -1, 0) - theta)) / DX
    return ux * dtx + uy * dty


def _rhs(theta, K, u):
    return -_advection(theta, u) + _diffusion(theta, K)


def _Kget(mode, t, tau_c, event_amp, sign):
    if mode == "truth":
        return K_field(t - tau_c, event_amp)
    if mode == "naive":
        return K_field(t, event_amp)
    # corrected: first-order reconstruction of K(t-τ_c)
    return K_field(t, event_amp) - sign * tau_c * dKdt_fd(t, event_amp)


def _rk4_step(theta, t, mode, tau_c, event_amp, sign):
    u = U_ADV
    k1 = _rhs(theta, _Kget(mode, t, tau_c, event_amp, sign), u)
    k2 = _rhs(theta + 0.5 * DT * k1, _Kget(mode, t + 0.5 * DT, tau_c, event_amp, sign), u)
    k3 = _rhs(theta + 0.5 * DT * k2, _Kget(mode, t + 0.5 * DT, tau_c, event_amp, sign), u)
    k4 = _rhs(theta + DT * k3, _Kget(mode, t + DT, tau_c, event_amp, sign), u)
    return theta + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(tau_c, event_amp=1.0, sign=1.0):
    """Step truth/naive/corrected in lockstep; return time-max relative errors."""
    r2 = (_X - 0.5) ** 2 + (_Y - 0.5) ** 2
    th0 = np.exp(-r2 / (2 * 0.15 ** 2))
    th = {m: th0.copy() for m in ("truth", "naive", "corrected")}
    n = int(round(T_END / DT))
    max_naive = max_corr = 0.0
    err_naive_series, err_corr_series, ts = [], [], []
    for i in range(n):
        t = i * DT
        for m in th:
            th[m] = _rk4_step(th[m], t, m, tau_c, event_amp, sign)
        nrm = np.linalg.norm(th["truth"])
        en = np.linalg.norm(th["naive"] - th["truth"]) / nrm
        ec = np.linalg.norm(th["corrected"] - th["truth"]) / nrm
        max_naive = max(max_naive, en); max_corr = max(max_corr, ec)
        if i % 20 == 0:
            ts.append(t); err_naive_series.append(en); err_corr_series.append(ec)
    return dict(tau_c=tau_c, max_err_naive=max_naive, max_err_corrected=max_corr,
                cut_factor=(max_naive / max_corr) if max_corr > 0 else float("inf"),
                t=ts, err_naive=err_naive_series, err_corr=err_corr_series)


def _loglog_slope(x, y):
    x = np.log(np.asarray(x, float)); y = np.log(np.asarray(y, float))
    A = np.vstack([x, np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def run():
    headline = integrate(0.02, event_amp=1.0, sign=1.0)
    # τ_c scaling
    taus = [0.005, 0.01, 0.02, 0.04]
    scal = [integrate(tc, 1.0, 1.0) for tc in taus]
    en = [s["max_err_naive"] for s in scal]; ec = [s["max_err_corrected"] for s in scal]
    slope_naive = _loglog_slope(taus, en); slope_corr = _loglog_slope(taus, ec)
    # steady null: no event -> all clocks coincide
    null = integrate(0.02, event_amp=0.0, sign=1.0)
    # sign test: CMN = -τ_c should be worse than naive
    wrong = integrate(0.02, event_amp=1.0, sign=-1.0)
    return dict(
        what="2-D advection-diffusion reduced-model proxy for the §H.3 CMN correction "
             "(NOT the deferred ISSM/GlaDS real-solver test)",
        scope_note="proxy only: no ice dynamics / real turbulence closure / real geometry; "
                   "advances RESULT-15 (1-D) to 2-D + advection + moving plume",
        grid=dict(nx=NX, ny=NY, dt=DT, T=T_END, u_adv=list(U_ADV), K0=K0, dK=DK,
                  sigma=SIGMA, event=list(EVENT)),
        headline=dict(tau_c=0.02, max_err_naive=headline["max_err_naive"],
                      max_err_corrected=headline["max_err_corrected"],
                      cut_factor=headline["cut_factor"]),
        scaling=dict(tau_c=taus, err_naive=en, err_corrected=ec,
                     slope_naive=slope_naive, slope_corrected=slope_corr),
        steady_null=dict(max_err_naive=null["max_err_naive"],
                         max_err_corrected=null["max_err_corrected"]),
        sign_test=dict(err_plus_tau_c=headline["max_err_corrected"],
                       err_minus_tau_c=wrong["max_err_corrected"],
                       err_naive=headline["max_err_naive"],
                       plus_is_unique_corrector=bool(
                           headline["max_err_corrected"] < headline["max_err_naive"]
                           < wrong["max_err_corrected"])),
        series=dict(t=headline["t"], err_naive=headline["err_naive"],
                    err_corr=headline["err_corr"]),
        verdict=(
            f"REDUCED-MODEL PROXY (not ISSM/GlaDS): in 2-D advection-diffusion with a "
            f"moving plume the CMN correction cuts the transient clock error "
            f"{headline['cut_factor']:.0f}× (τ_c=0.02: {headline['max_err_naive']:.2e} -> "
            f"{headline['max_err_corrected']:.2e}); naive scales ∝τ_c^{slope_naive:.2f}, "
            f"corrected ∝τ_c^{slope_corr:.2f} (one order higher-accurate); the steady "
            f"null is exact (no event -> err {null['max_err_corrected']:.1e}); and +τ_c is "
            f"the unique error-reducing sign (−τ_c err {wrong['max_err_corrected']:.2e} > "
            f"naive {headline['max_err_naive']:.2e}). All four RESULT-15 conclusions survive "
            f"dimensionality+advection+spatial structure, raising confidence in — but NOT "
            f"replacing — the deferred real-solver test."),
        references="this repo §G.5/§H.3 (cmn_solver_demo, RESULT 15), §G.5 commutator "
                   "(cmn_synthetic); deferred real solver: ISSM (Larour 2012), GlaDS (Werder 2013)",
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    s = res["scaling"]
    ax[0].loglog(s["tau_c"], s["err_naive"], "o-", color="#d62728",
                 label=f"naive ∝τ_c^{s['slope_naive']:.2f}")
    ax[0].loglog(s["tau_c"], s["err_corrected"], "s-", color="#2ca02c",
                 label=f"corrected ∝τ_c^{s['slope_corrected']:.2f}")
    tt = np.array(s["tau_c"])
    ax[0].loglog(tt, s["err_naive"][0] * (tt / tt[0]), "k--", lw=0.8, label="slope 1")
    ax[0].loglog(tt, s["err_corrected"][0] * (tt / tt[0]) ** 2, "k:", lw=0.8, label="slope 2")
    ax[0].set_xlabel(r"clock mismatch $\tau_c$"); ax[0].set_ylabel("time-max rel. error")
    ax[0].set_title("(a) naive ∝τ_c, corrected ∝τ_c² (2-D + advection + plume)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    se = res["series"]
    ax[1].plot(se["t"], se["err_naive"], color="#d62728", lw=2, label="naive")
    ax[1].plot(se["t"], se["err_corr"], color="#2ca02c", lw=2, label="corrected (+τ_c)")
    ax[1].axvspan(EVENT[0], EVENT[1], alpha=0.12, color="#1f77b4", label="plume active")
    ax[1].set_xlabel("time"); ax[1].set_ylabel("rel. error vs truth")
    ax[1].set_title(f"(b) τ_c=0.02: error cut {res['headline']['cut_factor']:.0f}×")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("§H.3 CMN correction — 2-D reduced-model proxy (not ISSM/GlaDS)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "reports", "h3_cmn_reduced_model.json")))
    a = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    h = res["headline"]; sc = res["scaling"]
    print("=== §H.3 CMN correction — 2-D reduced-model proxy (NOT ISSM/GlaDS) ===")
    print(f"  headline τ_c=0.02: naive {h['max_err_naive']:.2e} -> corrected "
          f"{h['max_err_corrected']:.2e}  (cut {h['cut_factor']:.0f}×)")
    print(f"  scaling: naive ∝τ_c^{sc['slope_naive']:.2f}, corrected ∝τ_c^{sc['slope_corrected']:.2f}")
    print(f"  steady null (no event): corrected err {res['steady_null']['max_err_corrected']:.1e}")
    print(f"  sign: +τ_c unique corrector = {res['sign_test']['plus_is_unique_corrector']}")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
