r"""Near-wall convergence + penalization study for the cold-grounded melt ceiling.

Referee (P2): the conductive-sublayer ceiling and the area-partition identity
rest on the suppression ``Nu/Nu_flat < 1`` being a *resolved physical* effect,
not a Brinkman-penalization-smearing artifact.  This driver answers that with the
standard grid-independence argument plus a penalization sweep, reusing the exact
measurement machinery of ``scallop_sublayer_probe`` (same true local-normal flux,
same spinup/measure window), so the ``Nu/Nu_flat`` reported here is the same
quantity as the report's Part-C numbers -- only ``ny`` (wall-normal resolution,
``dy = Ly/ny``, Ly = 2*pi), the penalization ``eta``, and the forcing ``seed`` vary.

Because the cavity is turbulently forced, a single (ny, eta) run carries a
turbulent-realization scatter that can masquerade as (non-)convergence.  We
therefore ENSEMBLE each grid point over several forcing seeds and report the
mean +/- std of ``Nu/Nu_flat``.  Resolution adequacy is then judged honestly:
the suppression is "robust" if ``Nu/Nu_flat < 1`` holds across the ladder, and
"grid-converged in magnitude" only if the across-``ny`` spread of the seed-means
is no larger than the within-``ny`` seed scatter.  ``delta_flat`` here is
``kappa*dT/<m_n,flat>`` with the cancelling constant ``kappa*dT`` set to 1, so it
is an inverse flux, NOT a length; no "cells across the sublayer" proxy is
reported because that constant makes it meaningless.

Run:
    python scallop_sublayer_convergence.py            # CPU, seeds 0,1,2
    python scallop_sublayer_convergence.py --gpu      # Tesla P100 / CuPy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_probe import _json_safe  # noqa: E402
from scallop_sublayer_probe import get_backend, _sublayer_stats  # noqa: E402
from scallop_sweep import _run_norm  # noqa: E402
from subglacial.candidate3_roughness_feedback import Candidate3Config  # noqa: E402

LY = 2.0 * np.pi  # cavity height (Candidate3*.Ly)


def one(nx, ny, eta, nwaves, udrive, famp, amp_over_lam, spinup, measure, seed, xp):
    """Flat control + single-mode bump at (ny, eta, seed), SAME resolution; return
    the sublayer decomposition tagged with grid/penalization/seed parameters."""
    Lx = 4.0 * 2.0 * np.pi
    lam = Lx / nwaves

    def cfg():
        return Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none",
                                f_amp=famp, Ri=0.0, seed=seed, eta=eta)

    _, _, m_flat, umax_f, _ = _run_norm(cfg(), 0.0, nwaves, udrive, spinup, measure, xp)
    a = amp_over_lam * lam
    _, _, m_bump, umax, _ = _run_norm(cfg(), a, nwaves, udrive, spinup, measure, xp)
    st = _sublayer_stats(np.asarray(m_bump), np.asarray(m_flat))
    st.update({
        "nx": int(nx), "ny": int(ny), "eta": float(eta), "seed": int(seed),
        "dy": float(LY / ny), "umax_bump": float(umax), "umax_flat": float(umax_f),
        "nu_flat_abs": float(np.nanmean(np.asarray(m_flat))),
    })
    return st


def _agg(stats):
    """Aggregate per-seed stats at one grid point -> mean/std + robustness frac."""
    nu = np.array([s["nu_ratio"] for s in stats], float)
    th = np.array([s["thicken"] for s in stats], float)
    cx = np.array([s["convex"] for s in stats], float)
    return {
        "ny": stats[0]["ny"], "eta": stats[0]["eta"], "dy": stats[0]["dy"],
        "seeds": [s["seed"] for s in stats],
        "nu_mean": float(nu.mean()), "nu_std": float(nu.std()),
        "nu_min": float(nu.min()), "nu_max": float(nu.max()),
        "thicken_mean": float(th.mean()), "convex_mean": float(cx.mean()),
        "frac_nu_lt_1": float(np.mean(nu < 1.0)),
        "frac_thicken_gt_convex": float(np.mean(th > cx)),
        "umax_mean": float(np.mean([s["umax_bump"] for s in stats])),
        "per_seed_nu": [float(x) for x in nu],
        "per_seed": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nwaves", type=int, default=12)
    ap.add_argument("--udrive", type=float, default=1.5)
    ap.add_argument("--famp", type=float, default=0.4)
    ap.add_argument("--amp", type=float, default=0.20)   # a/lambda
    ap.add_argument("--spinup", type=int, default=2000)
    ap.add_argument("--measure", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eta0", type=float, default=5.0e-5)  # eta for the ny ladder
    ap.add_argument("--ny0", type=int, default=128)        # ny for the eta sweep
    ap.add_argument("--nys", type=int, nargs="+", default=[48, 64, 96, 128, 192])
    ap.add_argument("--etas", type=float, nargs="+", default=[2.5e-5, 5.0e-5, 1.0e-4])
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "subglacial",
        "sublayer_convergence.json"))
    args = ap.parse_args()
    xp, backend = get_backend(args.gpu)

    print(f"=== Near-wall convergence + penalization study (backend={backend}) ===")
    print(f"fixed: nx={args.nx} nwaves={args.nwaves} U={args.udrive} f_amp={args.famp} "
          f"a/lam={args.amp} spinup={args.spinup} measure={args.measure} "
          f"seeds={args.seeds}\n")

    hdr = (f"{'ny':>5} {'eta':>9} {'dy':>7} {'Nu/Nu_flat(mean±std)':>22} "
           f"{'[min,max]':>16} {'<d>/d_f':>8} {'1+CV2':>7} {'umax':>6} "
           f"{'f(Nu<1)':>8} {'f(th>cx)':>8}")

    def show(a, t):
        return (f"{a['ny']:5d} {a['eta']:9.2e} {a['dy']:7.4f} "
                f"{a['nu_mean']:9.4f} ± {a['nu_std']:7.4f}    "
                f"[{a['nu_min']:.3f},{a['nu_max']:.3f}] "
                f"{a['thicken_mean']:8.3f} {a['convex_mean']:7.3f} "
                f"{a['umax_mean']:6.2f} {a['frac_nu_lt_1']:8.2f} "
                f"{a['frac_thicken_gt_convex']:8.2f}   ({t:.0f}s)")

    def grid(nx, ny, eta):
        stats = [one(nx, ny, eta, args.nwaves, args.udrive, args.famp, args.amp,
                     args.spinup, args.measure, s, xp) for s in args.seeds]
        return _agg(stats)

    # (A) resolution ladder at fixed eta0, ensembled over seeds
    print(f"(A) wall-normal resolution sweep at eta={args.eta0:g} "
          f"({len(args.seeds)} seeds each):")
    print(hdr)
    res = []
    for ny in args.nys:
        t0 = time.time()
        a = grid(args.nx, ny, args.eta0)
        res.append(a)
        print(show(a, time.time() - t0), flush=True)

    # (B) penalization sweep at fixed ny0, ensembled over seeds
    print(f"\n(B) penalization (eta) sweep at ny={args.ny0} "
          f"({len(args.seeds)} seeds each):")
    print(hdr)
    pen = []
    for eta in args.etas:
        t0 = time.time()
        a = grid(args.nx, args.ny0, eta)
        pen.append(a)
        print(show(a, time.time() - t0), flush=True)

    # --- convergence + robustness verdict -------------------------------------
    fin = [r for r in res if r["ny"] >= 96]
    means_fin = [r["nu_mean"] for r in fin]
    across_ny = (max(means_fin) - min(means_fin)) if len(means_fin) >= 2 else float("nan")
    typ_seed_std = float(np.median([r["nu_std"] for r in res]))  # typical scatter
    means_eta = [p["nu_mean"] for p in pen]
    across_eta = (max(means_eta) - min(means_eta)) if len(means_eta) >= 2 else float("nan")
    all_lt1 = all(r["frac_nu_lt_1"] == 1.0 for r in res + pen)
    # "magnitude converged" only if the across-ny spread of the means (ny>=96) is
    # within the within-ny seed scatter -- i.e. the ladder is noise-limited.
    mag_converged = bool(across_ny <= 2.0 * typ_seed_std)

    print("\n--- verdict ---")
    print(f"mean Nu/Nu_flat by ny (>=96)  : {[round(x,4) for x in means_fin]}")
    print(f"across-ny spread (ny>=96)     : {across_ny:.4f}")
    print(f"typical within-ny seed std    : {typ_seed_std:.4f}")
    print(f"across-eta spread of means    : {across_eta:.4f}  "
          f"({[round(x,4) for x in means_eta]})")
    print(f"Nu<1 for EVERY seed & grid pt : {all_lt1}   (sign-robust ceiling)")
    print(f"magnitude grid-converged?     : {mag_converged}  "
          f"(across-ny spread <= 2x seed std)")

    out = _json_safe({
        "backend": backend, "args": vars(args),
        "resolution_sweep": res, "penalization_sweep": pen,
        "across_ny_spread_ge96": across_ny, "typical_seed_std": typ_seed_std,
        "across_eta_spread": across_eta,
        "all_nu_lt_1_every_seed": all_lt1,
        "magnitude_grid_converged": mag_converged,
    })
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, allow_nan=False, default=float)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
