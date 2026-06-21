r"""Candidate 2 -- double-diffusive (salt + temperature) sweep.

Sweeps the density ratio ``R_rho = alpha_S dS / (alpha_T dT)`` for each closure
and records the thermal/haline Nusselt numbers and the Turner flux ratio in the
penalised cavity (heat diffusivity ``kappa_T``, salt diffusivity
``kappa_S = kappa_T / Le``).

Pre-registered prediction: ``Nu_T`` is humped, peaking near ``R_rho ~ 2`` (the
salt-finger sweet spot) and falling to ~1 for ``R_rho < 1`` and ``R_rho >~ Le``,
with Smagorinsky suppressing the peak.

The honest finding this sweep is designed to expose: in the 2-D forced
penalised regime no finger hump appears.  Instead ``Nu_T`` **decreases
monotonically** with ``R_rho`` (stabilising salt suppresses convective heat
transport), the diffusivity contrast shows up robustly as ``Nu_S >> Nu_T`` (salt
is advection-dominated), and the flux can go counter-gradient (``Nu_T < 0``,
K-theory-invisible) at the most stabilising ratios.  A finger-resolved hump is a
low-forcing, high-resolution, long-integration GPU experiment.

Backend-agnostic: auto-detects CuPy (Kaggle/Colab GPU) else NumPy.

Usage (CPU):
    python run_candidate2.py --nx 128 --ny 64 --rrho 0.5,1,2,5,10 \
        --closures none,smagorinsky --spinup 1500 --measure 800 --out-dir figures

Usage (GPU, finger-resolved):
    python run_candidate2.py --nx 384 --ny 96 --rrho 0.5,1,1.5,2,2.5,3,5,10 \
        --f-amp 0.05 --ri-t 3.0 --closures none,smagorinsky,backscatter \
        --spinup 6000 --measure 4000 --out-dir figures --report REPORT_CANDIDATE2.md
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import sys
# repo reorg: make sibling domain folders importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("general_two_clocks", "atmosphere", "glaciers", "ocean"):
    _p = os.path.join(_REPO_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _d, _p

from subglacial.candidate2_doublediff import DoubleDiffConfig, run_case


def get_backend():
    try:
        import cupy as cp
        cp.zeros(1).sum()
        return cp, "cupy(GPU)"
    except (ImportError, RuntimeError, MemoryError):
        return np, "numpy(CPU)"


def _floats(s):
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


def _strs(s):
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def run_sweep(args, xp):
    results = []
    for closure in _strs(args.closures):
        for Rr in _floats(args.rrho):
            cfg = DoubleDiffConfig(
                nx=args.nx, ny=args.ny, A=args.aspect, R_rho=Rr, sgs=closure,
                Ri_T=args.ri_t, Le=args.le, f_amp=args.f_amp, seed=args.seed,
            )
            t0 = time.time()
            r = run_case(cfg, spinup=args.spinup, measure=args.measure,
                         sample_every=args.sample_every, xp=xp)
            rec = {"closure": closure, "R_rho": Rr, **{k: r[k] for k in (
                "Nu_T", "Nu_S", "gamma", "KE_mean", "umax")},
                "secs": time.time() - t0}
            results.append(rec)
            print(f"  [{closure:11s}] R_rho={Rr:<5g} Nu_T={rec['Nu_T']:6.3f} "
                  f"Nu_S={rec['Nu_S']:8.2f} gamma={rec['gamma']:+.3f} "
                  f"umax={rec['umax']:.3f} ({rec['secs']:.1f}s)")
    return results


def maybe_figure(results, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError) as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return None
    closures = sorted({r["closure"] for r in results})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for closure in closures:
        sub = sorted((r for r in results if r["closure"] == closure),
                     key=lambda r: r["R_rho"])
        rr = [r["R_rho"] for r in sub]
        axes[0].plot(rr, [r["Nu_T"] for r in sub], "o-", label=closure)
        axes[1].plot(rr, [r["Nu_S"] for r in sub], "o-", label=closure)
        axes[2].plot(rr, [r["gamma"] for r in sub], "o-", label=closure)
    axes[0].set(xlabel="R_rho", ylabel="Nu_T",
                title="thermal Nusselt vs R_rho\n(predicted hump @ R_rho~2)")
    axes[1].set(xlabel="R_rho", ylabel="Nu_S",
                title="haline Nusselt vs R_rho\n(Nu_S >> Nu_T: salt advection)")
    axes[2].set(xlabel="R_rho", ylabel="gamma = F_T/(R_rho F_S)",
                title="Turner flux ratio vs R_rho")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "48_candidate2_doublediff.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--aspect", type=float, default=4.0)
    p.add_argument("--rrho", default="0.5,1.0,1.5,2.0,2.5,3.0,5.0,10.0")
    p.add_argument("--closures", default="none,smagorinsky,backscatter")
    p.add_argument("--ri-t", dest="ri_t", type=float, default=2.0)
    p.add_argument("--le", type=float, default=100.0)
    p.add_argument("--f-amp", dest="f_amp", type=float, default=0.2)
    p.add_argument("--spinup", type=int, default=1500)
    p.add_argument("--measure", type=int, default=800)
    p.add_argument("--sample-every", dest="sample_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", dest="out_dir", default="figures")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    xp, backend = get_backend()
    print(f"Backend: {backend}")
    print(f"grid nx={args.nx} ny={args.ny} Le={args.le} Ri_T={args.ri_t} "
          f"f_amp={args.f_amp} spinup={args.spinup} measure={args.measure}")
    t0 = time.time()
    results = run_sweep(args, xp)
    path = maybe_figure(results, args)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "48_candidate2_doublediff.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"backend": backend, "args": vars(args), "results": results},
                  fh, indent=2)
    print(f"wrote {json_path}")
    if path:
        print(f"wrote {path}")

    # honest summary: is Nu_T humped at R_rho~2, or monotonic?
    lines = []
    for closure in sorted({r["closure"] for r in results}):
        sub = sorted((r for r in results if r["closure"] == closure),
                     key=lambda r: r["R_rho"])
        rr = [r["R_rho"] for r in sub]
        nt = [r["Nu_T"] for r in sub]
        i_peak = int(np.argmax(nt))
        interior = 0 < i_peak < len(nt) - 1
        verdict = (f"hump @ R_rho={rr[i_peak]:g}" if interior
                   else "monotonic (no interior hump)")
        lines.append(f"{closure:11s}: Nu_T={[round(x, 3) for x in nt]} "
                     f"-> {verdict}; Nu_S={[round(x, 1) for x in [r['Nu_S'] for r in sub]]}")
    print("\n".join(lines))
    print(f"total {time.time() - t0:.1f}s")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("# Candidate 2 -- double-diffusive layering (salt + temperature)\n\n")
            fh.write(f"Backend: {backend}\n\n```\n" + "\n".join(lines) + "\n```\n")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
