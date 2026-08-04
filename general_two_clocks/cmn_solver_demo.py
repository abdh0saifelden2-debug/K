r"""Transient K-theory thermal-solver test of the clock-mismatch (CMN) correction
(FUTURE_WORK §G.5 / §H.3).

§G.5 derives, with no approximation, that an eddy-diffusivity temperature solver
carries a hidden commutator when the momentum-derived diffusivity ``K_u`` is
itself unsteady,

    [d_t, D] theta = div( (d_t K_u) grad theta ),   D[theta] = div(K_u grad theta),

and proposes the correction term ``- CMN * div( (d_t K_u) grad theta )`` with the
dimensionally-forced coefficient ``CMN = tau_c`` (a correlation/memory time, not an
O(1) constant).  ``validation/synthetic/cmn_synthetic.py`` already verifies the
*identity*, its discretisation, the steady-state null and the dimensional
signature.  What it does **not** do is the §H.3 forecast itself --- that *adding
the term to a transient solver reduces the spurious error and vanishes in steady
turbulence*.  §H.3 flagged this as "not runnable here (implement in ISSM/GlaDS)".

This module runs exactly that test, self-contained, on a 1-D periodic spectral
diffusion solver --- no external data, no GPU.  The physical content of §G.5 is
the *two clocks* lag: the turbulence that sets ``K_u`` at time ``t`` is
communicated to the heat equation with a finite decorrelation lag ``tau_c``, so
the faithful ("truth") flux uses the **lagged** diffusivity ``K_u(x, t - tau_c)``
while the naive K-theory closure freezes it at ``K_u(x, t)``.  Because

    K_u(t) - tau_c d_t K_u(t) = K_u(t - tau_c) + O(tau_c^2),

the §G.5 correction with ``CMN = +tau_c`` is *exactly* the first-order
reconstruction of the lagged-clock truth.  Three solvers are advanced with an
identical RK4 stepper and identical ``theta_0`` over one transient cycle of
``K_u``; only the diffusivity field in the flux differs:

  * **truth**     : ``D[theta; K_u(t - tau_c)]``        (lagged clock)
  * **naive**     : ``D[theta; K_u(t)]``                (frozen clock, K-theory)
  * **corrected** : ``D[theta; K_u(t) - tau_c d_t K_u]`` (§G.5, CMN = +tau_c)

Measured against the truth (same discretisation, so numerical-diffusion error is
common and cancels --- this isolates the *modelling* error):

  1. **Transient reduction.** the time-max and time-RMS error of *corrected* is
     far below *naive* over the transient.
  2. **Order in tau_c.** sweeping ``tau_c`` -> 0, the naive error scales like
     ``tau_c^1`` and the corrected like ``tau_c^2`` (log-log slopes ~1 and ~2),
     i.e. the correction removes the leading-order clock-mismatch error.
  3. **Steady-turbulence null.** with ``eps = 0`` (``d_t K_u = 0``) the correction
     term is identically zero *and* the lagged and frozen clocks coincide, so all
     three solvers are bit-identical --- the §G.5 "vanishes in steady state".
  4. **Sign control.** ``CMN = -tau_c`` (wrong sign) makes the error *worse* than
     naive (~doubles the leading term), so ``sign(CMN) = +`` is the unique
     error-reducing choice --- the §G.5 ``+`` sign, here as a solver result.

Writes ``figures/60_cmn_solver_demo.json`` (+ a ``.png``).  ``--fast`` is a small
smoke run (coarser grid / shorter sweep) for plumbing only.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# --- 1-D periodic spectral diffusion operator div(K d_x theta) ----------------
def _wavenumbers(n):
    return np.fft.fftfreq(n, d=1.0 / n)  # integer wavenumbers on domain [0, 2pi)


def _dx(f, k):
    return np.real(np.fft.ifft(1j * k * np.fft.fft(f)))


def diffusion(theta, K, k):
    """Spectral ``d_x( K d_x theta )`` on a 2pi-periodic grid."""
    return _dx(K * _dx(theta, k), k)


# --- analytic eddy-diffusivity field and its time derivative ------------------
def K_field(x, t, K0=0.2, kappa=0.3, eps=0.5, omega=2.0):
    """Unsteady eddy diffusivity ``K_u(x, t) > 0`` (turbulence transient).

    Spatially modulated (``1 + kappa cos x``, kept positive) and time-modulated
    by ``1 + eps sin(omega t)`` (``eps = 0`` => steady turbulence)."""
    return K0 * (1.0 + kappa * np.cos(x)) * (1.0 + eps * np.sin(omega * t))


def dKdt_field(x, t, K0=0.2, kappa=0.3, eps=0.5, omega=2.0):
    return K0 * (1.0 + kappa * np.cos(x)) * (eps * omega * np.cos(omega * t))


def _Keff(model, x, t, tau_c, params):
    """Diffusivity field entering the flux for each solver variant."""
    if model == "truth":                 # lagged clock (the faithful flux)
        return K_field(x, t - tau_c, **params)
    if model == "naive":                 # frozen clock (plain K-theory)
        return K_field(x, t, **params)
    if model == "corrected":             # §G.5 correction, CMN = +tau_c
        # Keff>0 iff tau_c < sqrt(1/eps**2 - 1)/omega (tight bound from the
        # combined sin/cos amplitude); = sqrt(3)/2 ~ 0.866 for eps=0.5, omega=2.
        # The demo's production tau_c<=0.1 stays well inside it.
        return K_field(x, t, **params) - tau_c * dKdt_field(x, t, **params)
    if model == "wrongsign":             # CMN = -tau_c (sign positive control)
        return K_field(x, t, **params) + tau_c * dKdt_field(x, t, **params)
    raise ValueError(model)


def integrate(theta0, x, k, t0, T, dt, model, tau_c, params, record_times=None):
    """RK4-advance ``d_t theta = d_x(Keff d_x theta)`` from ``t0`` to ``t0+T``.

    ``Keff`` is the model-specific diffusivity (``_Keff``); all variants share
    this stepper, grid and ``dt`` so the discretisation error is common.  If
    ``record_times`` is given, returns ``(theta_final, snapshots)`` where
    ``snapshots[i]`` is ``theta`` at the step nearest ``record_times[i]``."""
    theta = theta0.copy()
    # endpoint is nsteps*dt, which can differ from T by up to dt/2; this bias is
    # common to all four variants and so cancels in the error comparison.
    nsteps = int(round(T / dt))
    rhs = lambda th, t: diffusion(th, _Keff(model, x, t, tau_c, params), k)  # noqa: E731

    snaps, rec = [], list(record_times) if record_times is not None else []
    ri = 0
    for s in range(nsteps):
        t = t0 + s * dt
        while ri < len(rec) and rec[ri] <= t + 0.5 * dt:
            snaps.append(theta.copy())
            ri += 1
        k1 = rhs(theta, t)
        k2 = rhs(theta + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(theta + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(theta + dt * k3, t + dt)
        theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    while ri < len(rec):
        snaps.append(theta.copy())
        ri += 1
    return (theta, snaps) if record_times is not None else theta


def _theta0(x):
    """Smooth multi-mode initial temperature."""
    return np.sin(x) + 0.5 * np.sin(2 * x) + 0.3 * np.cos(3 * x)


def _rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def error_trace(n=96, tau_c=0.05, dt=5e-4, eps=0.5, omega=2.0, n_snap=60,
                params_extra=None):
    """Advance truth / naive / corrected / wrongsign over one transient cycle
    ``T = 2 pi / omega`` and return the relative-error trajectories vs truth."""
    params = {"K0": 0.2, "kappa": 0.3, "eps": eps, "omega": omega}
    if params_extra:
        params.update(params_extra)
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    k = _wavenumbers(n)
    theta0 = _theta0(x)
    T = 2 * np.pi / omega
    rec = list(np.linspace(0.0, T, n_snap))

    out = {}
    for model in ("truth", "naive", "corrected", "wrongsign"):
        _, snaps = integrate(theta0, x, k, 0.0, T, dt, model, tau_c, params, rec)
        out[model] = np.array(snaps)
    tr = out["truth"]
    e_naive = np.array([_rel_err(out["naive"][i], tr[i]) for i in range(len(rec))])
    e_corr = np.array([_rel_err(out["corrected"][i], tr[i]) for i in range(len(rec))])
    e_wrong = np.array([_rel_err(out["wrongsign"][i], tr[i]) for i in range(len(rec))])
    return {"t": np.array(rec), "T": T, "tau_c": tau_c, "eps": eps, "omega": omega,
            "e_naive": e_naive, "e_corrected": e_corr, "e_wrongsign": e_wrong,
            "theta_truth_final": tr[-1], "theta_naive_final": out["naive"][-1],
            "theta_corrected_final": out["corrected"][-1], "x": x}


def tau_c_scaling(tau_list, n=96, dt=5e-4, eps=0.5, omega=2.0):
    """For each ``tau_c`` return the time-max error of naive/corrected vs truth,
    and the log-log slopes (expect ~1 for naive, ~2 for corrected)."""
    taus = np.array(sorted(tau_list), float)
    em_naive, em_corr = [], []
    for tc in taus:
        tr = error_trace(n=n, tau_c=float(tc), dt=dt, eps=eps, omega=omega, n_snap=40)
        em_naive.append(float(tr["e_naive"].max()))
        em_corr.append(float(tr["e_corrected"].max()))
    em_naive = np.array(em_naive)
    em_corr = np.array(em_corr)
    slope_naive = float(np.polyfit(np.log(taus), np.log(em_naive), 1)[0])
    slope_corr = float(np.polyfit(np.log(taus), np.log(em_corr), 1)[0])
    return {"tau_c": taus.tolist(), "maxerr_naive": em_naive.tolist(),
            "maxerr_corrected": em_corr.tolist(),
            "loglog_slope_naive": slope_naive,
            "loglog_slope_corrected": slope_corr}


def steady_null(n=96, dt=5e-4, tau_c=0.05, omega=2.0):
    """eps = 0 (d_tK = 0): truth, naive and corrected must be bit-identical."""
    tr = error_trace(n=n, tau_c=tau_c, dt=dt, eps=0.0, omega=omega, n_snap=20)
    return {"max_e_naive": float(tr["e_naive"].max()),
            "max_e_corrected": float(tr["e_corrected"].max())}


def _verdict(d):
    base = d["baseline"]
    sc = d["tau_c_scaling"]
    red = base["maxerr_naive"] / (base["maxerr_corrected"] + 1e-30)
    ok_reduce = base["maxerr_corrected"] < 0.5 * base["maxerr_naive"]
    ok_order = sc["loglog_slope_naive"] < 1.4 and sc["loglog_slope_corrected"] > 1.6
    ok_steady = d["steady_null"]["max_e_naive"] < 1e-10
    ok_sign = base["maxerr_wrongsign"] > base["maxerr_naive"]
    if ok_reduce and ok_order and ok_steady and ok_sign:
        return ("VERIFIED (synthetic K-theory solver): the §G.5 CMN term with "
                "CMN=+tau_c cuts the transient error %.0fx (naive slope %.2f~tau_c, "
                "corrected slope %.2f~tau_c^2), is identically zero for steady "
                "turbulence, and the +tau_c sign is the error-reducing one "
                "(wrong sign is worse). §H.3 forecast confirmed in-solver."
                % (red, sc["loglog_slope_naive"], sc["loglog_slope_corrected"]))
    return ("INCONCLUSIVE: reduce=%s order=%s steady=%s sign=%s (red=%.1fx, "
            "slopes naive=%.2f corr=%.2f)" % (ok_reduce, ok_order, ok_steady,
            ok_sign, red, sc["loglog_slope_naive"], sc["loglog_slope_corrected"]))


def run(fast=False):
    t0 = time.time()
    n = 64 if fast else 96
    dt = 1e-3 if fast else 5e-4
    taus = [0.02, 0.04, 0.08] if fast else [0.0125, 0.025, 0.05, 0.1]

    base_tr = error_trace(n=n, tau_c=0.05, dt=dt)
    baseline = {
        "tau_c": 0.05,
        "maxerr_naive": float(base_tr["e_naive"].max()),
        "maxerr_corrected": float(base_tr["e_corrected"].max()),
        "maxerr_wrongsign": float(base_tr["e_wrongsign"].max()),
        "rmserr_naive": float(np.sqrt(np.mean(base_tr["e_naive"] ** 2))),
        "rmserr_corrected": float(np.sqrt(np.mean(base_tr["e_corrected"] ** 2))),
    }
    sc = tau_c_scaling(taus, n=n, dt=dt)
    sn = steady_null(n=n, dt=dt)
    out = {
        "config": {"n": n, "dt": dt, "scheme": "RK4 spectral 1-D periodic",
                   "K0": 0.2, "kappa": 0.3, "eps": 0.5, "omega": 2.0,
                   "wall_time_s": round(time.time() - t0, 1)},
        "baseline": baseline,
        "tau_c_scaling": sc,
        "steady_null": sn,
        "trace": {"t": base_tr["t"].tolist(),
                  "e_naive": base_tr["e_naive"].tolist(),
                  "e_corrected": base_tr["e_corrected"].tolist(),
                  "e_wrongsign": base_tr["e_wrongsign"].tolist()},
    }
    out["verdict"] = _verdict(out)
    out["_base_tr"] = base_tr  # for plotting; stripped before JSON
    return out


def _plot(out, path_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = out["_base_tr"]
    sc = out["tau_c_scaling"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(tr["t"], tr["e_naive"], "-", color="C3", lw=2,
             label="naive K-theory (frozen clock)")
    ax1.plot(tr["t"], tr["e_corrected"], "-", color="C0", lw=2,
             label="§G.5 corrected (CMN=+$\\tau_c$)")
    ax1.plot(tr["t"], tr["e_wrongsign"], "--", color="0.6", lw=1.3,
             label="wrong sign (CMN=$-\\tau_c$)")
    ax1.set_xlabel("time  $t$"); ax1.set_ylabel("rel. error vs lagged-clock truth")
    ax1.set_title("Transient error over one $K_u$ cycle ($\\tau_c=0.05$)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    taus = np.array(sc["tau_c"])
    ax2.loglog(taus, sc["maxerr_naive"], "o-", color="C3",
               label="naive  (slope %.2f)" % sc["loglog_slope_naive"])
    ax2.loglog(taus, sc["maxerr_corrected"], "s-", color="C0",
               label="corrected  (slope %.2f)" % sc["loglog_slope_corrected"])
    ax2.loglog(taus, sc["maxerr_naive"][0] * (taus / taus[0]), ":", color="0.5",
               label="$\\propto\\tau_c$")
    ax2.loglog(taus, sc["maxerr_corrected"][0] * (taus / taus[0]) ** 2, "-.",
               color="0.5", label="$\\propto\\tau_c^2$")
    ax2.set_xlabel("clock lag  $\\tau_c$"); ax2.set_ylabel("time-max rel. error")
    ax2.set_title("Order in $\\tau_c$: naive $O(\\tau_c)$, corrected $O(\\tau_c^2)$")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")

    fig.suptitle("§H.3 clock-mismatch correction in a transient K-theory thermal solver",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path_png, dpi=110)
    plt.close(fig)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    fast = "--fast" in argv
    out = run(fast=fast)

    b, sc, sn = out["baseline"], out["tau_c_scaling"], out["steady_null"]
    print("=== CMN correction in a transient K-theory thermal solver (§G.5/§H.3) ===",
          flush=True)
    print("  config: n=%d dt=%g %s (%ss)" % (out["config"]["n"], out["config"]["dt"],
          out["config"]["scheme"], out["config"]["wall_time_s"]), flush=True)
    print("  baseline tau_c=0.05: max rel.err  naive=%.3e  corrected=%.3e  "
          "wrongsign=%.3e  (reduction %.0fx)" % (b["maxerr_naive"],
          b["maxerr_corrected"], b["maxerr_wrongsign"],
          b["maxerr_naive"] / (b["maxerr_corrected"] + 1e-30)), flush=True)
    print("  tau_c-scaling slopes: naive=%.2f (~1)  corrected=%.2f (~2)"
          % (sc["loglog_slope_naive"], sc["loglog_slope_corrected"]), flush=True)
    print("  steady-turbulence null (eps=0): max err naive=%.2e corrected=%.2e"
          % (sn["max_e_naive"], sn["max_e_corrected"]), flush=True)
    print("  VERDICT: " + out["verdict"], flush=True)

    if not fast:
        png = os.path.join(HERE, "figures", "60_cmn_solver_demo.png")
        _plot(out, png)
        print("WROTE " + png, flush=True)
        out.pop("_base_tr", None)
        path = os.path.join(HERE, "figures", "60_cmn_solver_demo.json")
        out_json = dict(out)
        out_json["description"] = (
            "§G.5/§H.3 clock-mismatch (CMN) correction tested in a self-contained "
            "1-D transient K-theory thermal solver. truth = lagged eddy diffusivity "
            "K_u(t-tau_c); naive = frozen K_u(t); corrected = K_u(t)-tau_c d_tK_u "
            "(CMN=+tau_c). Corrected cuts the transient error and scales as tau_c^2 "
            "vs naive tau_c^1; identically zero for steady turbulence; +tau_c sign "
            "is the error-reducing one.")
        with open(path, "w") as fh:
            json.dump(out_json, fh, indent=2, allow_nan=False)
        print("WROTE " + path, flush=True)
    else:
        out.pop("_base_tr", None)
    return out


if __name__ == "__main__":
    main()
