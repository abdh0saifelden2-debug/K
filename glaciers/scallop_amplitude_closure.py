r"""Amplitude (a/λ) closure for the scallop wall-flux enhancement — tests, and
**falsifies**, the §G.6 ``(a/λ)²`` closure ansatz (FUTURE_WORK §A.2/§G.6,
Caveat D).

Part C of ``REPORT_CANDIDATE3.md`` swept *wavelength* (``n_waves`` 4–24) and
turbulence *seeds* at a single **frozen** amplitude ``a/λ = 0.1``. The §G.1
mechanism module (``scallop_g1_populations.py``) later swept *amplitude* to earn
the area-partition decomposition, but recorded only the *partition* (why
``Nu<1``), never the **functional form** of ``Nu/Nu_flat(a/λ)`` posited by §G.6.
This module closes that gap: it reproduces the §G.1 amplitude sweep through the
``scallop_sweep`` flux core and fits the §G.6 effective-boundary-layer closure

    δ_T,eff = δ_T,flat · (1 + ζ (a/λ)²)   ⇒   Nu/Nu_flat = 1 / (1 + ζ (a/λ)²)

(conduction-limited flux ∝ 1/δ_T), i.e. a *quadratic* roll-off whose coefficient
``ζ`` cannot be fit — let alone its **exponent** confirmed — from one (a, λ) point.

This harness runs the (already-built but never-executed) ``amp_scan`` from
``scallop_sweep.py`` at the **production Part-C config** (``nx=ny=128``,
``spinup=3000``, ``measure=800``, ``U_drive=1.5``, ``n_waves=12``, local-normal Nu)
over a grid of ``a/λ`` and then *measures the closure*:

  * the mean-Nu **deficit** ``D(a/λ) = 1/(Nu/Nu_flat) − 1`` is fit as a power law
    ``D = ζ_p · (a/λ)^p`` (log–log slope ``p``, free exponent), and separately
    with the exponent **pinned to the §G.6 value** ``p = 2`` (giving ``ζ``); the
    free-``p`` vs pinned-``p`` residuals decide whether the quadratic ansatz holds;
  * the conduction-relative gain ``R_mean(a/λ)`` and local peak ``R_max(a/λ)`` are
    reported so the local lee enhancement (escaping the conduction limit) is
    separated from the sub-flat mean.

Writes ``figures/59_scallop_amplitude_closure.json``. Pure measurement on the
committed solver; no external data. ``--gpu`` uses CuPy; ``--fast`` is a tiny CPU
smoke (under-resolved — for plumbing only, not for the closure verdict).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_sweep import amp_scan, _run, _run_norm  # noqa: E402
from scallop_probe import _enh_stats, _json_safe, _nanmean  # noqa: E402
from subglacial.candidate3_roughness_feedback import Candidate3Config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def fit_closure(a_over_lam, nu_ratio):
    """Fit the mean-Nu deficit ``D = 1/(Nu/Nu_flat) − 1`` against ``a/λ``.

    Returns the free-exponent power-law fit ``D = ζ_free·(a/λ)^p`` (``p`` =
    log–log slope), the §G.6-pinned quadratic fit ``D = ζ_quad·(a/λ)²`` and the
    linear fit ``D = ζ_lin·(a/λ)``, each with an R² on ``log D`` / ``D`` so the
    quadratic ansatz can be accepted or corrected. All inputs are positive
    (Nu/Nu_flat < 1 ⇒ D > 0) over the resolved range."""
    x = np.asarray(a_over_lam, float)
    nu = np.asarray(nu_ratio, float)
    D = 1.0 / nu - 1.0                                   # >0 since Nu/Nu_flat<1
    lx, lD = np.log(x), np.log(D)

    # free-exponent power law: log D = log ζ + p log x
    p, log_zeta = np.polyfit(lx, lD, 1)
    zeta_free = float(np.exp(log_zeta))
    ss_res = float(np.sum((lD - (log_zeta + p * lx)) ** 2))
    ss_tot = float(np.sum((lD - lD.mean()) ** 2))
    r2_free = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dvar = float(np.sum((D - D.mean()) ** 2))

    def _pinned(power):
        # D = ζ x^power : ζ = least-squares slope through the origin in x^power
        xp = x ** power
        zeta = float(np.sum(xp * D) / np.sum(xp * xp))
        res = D - zeta * xp
        # R² relative to a constant; undefined (nan) if D has no variance.
        r2 = 1.0 - float(np.sum(res ** 2)) / dvar if dvar > 0 else float("nan")
        return zeta, r2

    zeta_quad, r2_quad = _pinned(2.0)                    # §G.6 ansatz p=2
    zeta_lin, r2_lin = _pinned(1.0)                      # linear alternative
    D_mean, D_std = float(D.mean()), float(D.std())

    if p < 0.5:
        # flat: D ~ const, neither power law explains the (lack of) variation.
        verdict = ("amplitude-INDEPENDENT: deficit D=1/(Nu/Nu_flat)-1=%.3f±%.3f "
                   "shows no power-law in a/λ (p_free≈%.2f, R²_logD=%.3f); the §G.6 "
                   "quadratic (R²=%.2f) and linear (R²=%.2f) laws are both rejected "
                   "(worse than a constant). Nu<1 is near-geometric, not an "
                   "amplitude roll-off." % (D_mean, D_std, p, r2_free, r2_quad, r2_lin))
    elif p < 1.5:
        verdict = ("exponent p≈%.2f in [0.5,1.5) — the §G.6 quadratic (a/λ)^2 "
                   "ansatz is NOT supported; roll-off is closer to linear "
                   "(ζ_lin≈%.2f)" % (p, zeta_lin))
    elif p > 2.5:
        verdict = ("exponent p≈%.2f > 2.5 — steeper than the §G.6 quadratic "
                   "ansatz" % p)
    else:
        verdict = ("exponent p≈%.2f consistent with the §G.6 quadratic (a/λ)^2 "
                   "ansatz; ζ≈%.2f" % (p, zeta_quad))

    return {
        "deficit_D": D.tolist(), "D_mean": D_mean, "D_std": D_std,
        "p_free": float(p), "zeta_free": zeta_free, "r2_logD_free": float(r2_free),
        "zeta_quadratic_p2": zeta_quad, "r2_quadratic_p2": float(r2_quad),
        "zeta_linear_p1": zeta_lin, "r2_linear_p1": float(r2_lin),
        "verdict": verdict,
    }


def _nu_ratio_one(nx, ny, n_waves, U_drive, spinup, measure, Ri, seed, a, Nu_flat,
                  xp):
    """Single (a, seed) frozen-boundary Nu/Nu_flat + R stats (one solver pair)."""
    dx = (4.0 * 2.0 * np.pi) / nx
    cfg0 = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.0, Ri=Ri,
                            seed=seed)
    yb, m_cond, _, _ = _run(cfg0, a, n_waves, U_drive=0.0, spinup=spinup, xp=xp)
    cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.4, Ri=Ri,
                           seed=seed)
    yb, m_v, m_n, _, _ = _run_norm(cfg, a, n_waves, U_drive, spinup, measure, xp)
    st = _enh_stats(yb, m_v, m_cond, dx)
    return _nanmean(m_n) / Nu_flat, st["R_mean"], st["R_max"]


def seed_robustness(nx, ny, n_waves, U_drive, spinup, measure, Ri, Nu_flat, lam,
                    a_over_lam, seeds, xp):
    """Per-seed Nu/Nu_flat at the extreme (and reference) a/λ so the across-
    amplitude variation can be compared against turbulent-seed scatter: if the
    amplitude spread exceeds the seed spread the (non-)trend is physical."""
    out = []
    for r in a_over_lam:
        a = r * lam
        per = []
        for sd in seeds:
            nu, rm, rx = _nu_ratio_one(nx, ny, n_waves, U_drive, spinup, measure,
                                       Ri, sd, a, Nu_flat, xp)
            per.append({"seed": sd, "Nu_ratio": nu, "R_mean": rm, "R_max": rx})
            print(f"    [seed-check a/λ={r:.3f} seed={sd}] Nu/Nu_flat={nu:.4f} "
                  f"R_mean={rm:.4f}", flush=True)
        nus = np.array([p["Nu_ratio"] for p in per])
        out.append({"a_over_lam": r, "per_seed": per,
                    "Nu_ratio_mean": float(nus.mean()),
                    "Nu_ratio_std": float(nus.std())})
    return out


def run(xp=np, nx=128, ny=128, n_waves=12, U_drive=1.5, spinup=3000,
        measure=800, Ri=0.0, seed=0,
        a_over_lam=(0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30),
        seed_check_a=(0.05, 0.30), seed_check_seeds=(0, 1, 2), fast=False):
    if fast:
        nx = ny = 64
        spinup = 300
        measure = 120
        a_over_lam = (0.05, 0.10, 0.15, 0.20, 0.30)
        seed_check_a = (0.05, 0.30)
        seed_check_seeds = (0, 1)
    t0 = time.time()
    scan = amp_scan(nx, ny, n_waves, U_drive, spinup, measure, Ri, seed,
                    list(a_over_lam), xp)
    rows = scan["rows"]
    a_grid = [r["a_over_lam"] for r in rows]
    nu_ratio = [r["Nu_ratio"] for r in rows]
    fit = fit_closure(a_grid, nu_ratio)
    print("=== seed robustness at extreme a/λ (amplitude spread vs seed spread) ===",
          flush=True)
    seed_rob = seed_robustness(nx, ny, n_waves, U_drive, spinup, measure, Ri,
                               scan["Nu_flat"], scan["lambda"],
                               list(seed_check_a), list(seed_check_seeds), xp)
    # amplitude spread of the mean-Nu across the full grid vs the worst per-a/λ
    # seed spread: the trend is physical only if amplitude_spread > seed_spread.
    nu_all = np.array(nu_ratio)
    amp_spread = float(nu_all.max() - nu_all.min())
    seed_spread = max((s["Nu_ratio_std"] for s in seed_rob), default=0.0)
    fit["amplitude_spread_Nu"] = amp_spread
    fit["worst_seed_std_Nu"] = float(seed_spread)
    fit["Nu_ratio_mean_over_grid"] = float(nu_all.mean())
    fit["Nu_ratio_std_over_grid"] = float(nu_all.std())
    out = {
        "config": {"nx": nx, "ny": ny, "n_waves": n_waves, "U_drive": U_drive,
                   "spinup": spinup, "measure": measure, "Ri": Ri, "seed": seed,
                   "lambda": scan["lambda"], "a_over_lam": list(a_over_lam),
                   "wall_time_s": round(time.time() - t0, 1)},
        "Nu_flat": scan["Nu_flat"],
        "rows": rows,
        "closure_fit": fit,
        "seed_robustness": seed_rob,
    }
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--gpu" in argv:
        import cupy as cp
        xp, tag = cp, "gpu"
    else:
        xp, tag = np, "cpu"
    fast = "--fast" in argv
    out = run(xp=xp, fast=fast)

    f = out["closure_fit"]
    print("=== scallop amplitude closure (Caveat D) ===", flush=True)
    print(f"  config: nx={out['config']['nx']} n_waves={out['config']['n_waves']} "
          f"U_drive={out['config']['U_drive']} spinup={out['config']['spinup']} "
          f"measure={out['config']['measure']} ({out['config']['wall_time_s']}s, {tag})",
          flush=True)
    print(f"  Nu_flat={out['Nu_flat']:.6e}", flush=True)
    for r in out["rows"]:
        print(f"    a/λ={r['a_over_lam']:.3f}  Nu/Nu_flat={r['Nu_ratio']:.4f}  "
              f"R_mean={r['R_mean']:.4f}  R_max={r['R_max']:.3f}  "
              f"corr={r['corr_excess_slope']:+.3f}", flush=True)
    print(f"  fit: D=1/(Nu/Nu_flat)-1 ~ ζ(a/λ)^p  ->  p_free={f['p_free']:.2f} "
          f"(R²_logD={f['r2_logD_free']:.3f}); pinned p=2: ζ={f['zeta_quadratic_p2']:.2f} "
          f"(R²={f['r2_quadratic_p2']:.3f}); pinned p=1: ζ={f['zeta_linear_p1']:.2f} "
          f"(R²={f['r2_linear_p1']:.3f})", flush=True)
    print(f"  VERDICT: {f['verdict']}", flush=True)

    if not fast:
        path = os.path.join(HERE, "figures", "59_scallop_amplitude_closure.json")
        with open(path, "w") as fh:
            json.dump(_json_safe(out), fh, indent=2, allow_nan=False)
        print("WROTE " + path, flush=True)
    return out


if __name__ == "__main__":
    main()
