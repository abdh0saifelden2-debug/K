r"""Candidate 1 -- intermittent-plume sweep over a rough ice base.

Sweeps the buoyancy coupling ``Ri`` for each closure and records, over the
rough ice base, both

* the **intermittency statistics** of the interfacial melt field -- skewness
  ``S``, excess kurtosis ``K``, peak-to-mean ratio -- pooled over space and the
  measurement window, and
* the **turbulent heat flux** ``Fturb = <v' theta'>`` in the cavity interior.

Pre-registered prediction: ``S > 0``, ``K > 3``, ``peak/mean > 2`` humped at
intermediate ``Ri ~ 0.3-0.6``, with Smagorinsky suppressing the peaks.

The honest finding this sweep is designed to expose: the interfacial melt
intermittency over a rough base is *geometric* (thinner ice columns conduct
faster) and is therefore essentially independent of ``Ri`` and of the closure,
because the no-slip Brinkman wall pins the interfacial flux to molecular
conduction (the same scope boundary as the Stefan prototype / Candidate 4).
The flow-dependent, closure-dependent signal lives in ``Fturb``.

Backend-agnostic: auto-detects CuPy (Kaggle/Colab GPU) else NumPy.

Usage (CPU):
    python run_candidate1.py --nx 128 --ny 64 --ri 0.0,0.5,1.0 \
        --closures none,smagorinsky --spinup 400 --measure 600 --out-dir figures

Usage (GPU sweep):
    python run_candidate1.py --nx 384 --ny 96 \
        --ri 0.0,0.25,0.5,0.75,1.0,1.5 \
        --closures none,smagorinsky,backscatter \
        --spinup 2000 --measure 4000 --out-dir figures --report REPORT_CANDIDATE1.md
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from subglacial.candidate1_plumes import PlumeConfig, run_case


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
        for Ri in _floats(args.ri):
            cfg = PlumeConfig(
                nx=args.nx, ny=args.ny, A=args.aspect, Ri=Ri, sgs=closure,
                f_amp=args.f_amp, sigma_h=args.sigma_h, seed=args.seed,
                rough_seed=args.rough_seed,
            )
            t0 = time.time()
            r = run_case(cfg, spinup=args.spinup, measure=args.measure,
                         sample_every=args.sample_every, xp=xp)
            rec = {"closure": closure, "Ri": Ri, **{k: r[k] for k in (
                "skew", "kurt", "peak_mean", "fturb_mean", "KE_mean",
                "melt_mean", "umax", "sigma_h")}, "secs": time.time() - t0}
            results.append(rec)
            print(f"  [{closure:11s}] Ri={Ri:<4g} skew={rec['skew']:+.3f} "
                  f"kurt={rec['kurt']:+.3f} peak/mean={rec['peak_mean']:5.2f} "
                  f"Fturb={rec['fturb_mean']:+.3e} melt={rec['melt_mean']:.3e} "
                  f"({rec['secs']:.1f}s)")
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
                     key=lambda r: r["Ri"])
        ri = [r["Ri"] for r in sub]
        axes[0].plot(ri, [r["peak_mean"] for r in sub], "o-", label=closure)
        axes[1].plot(ri, [r["skew"] for r in sub], "o-", label=closure)
        axes[2].plot(ri, [r["fturb_mean"] for r in sub], "o-", label=closure)
    axes[0].set(xlabel="Ri", ylabel="peak/mean melt",
                title="melt peak/mean vs Ri\n(geometric -> flat)")
    axes[1].set(xlabel="Ri", ylabel="skewness", title="melt skewness vs Ri")
    axes[2].set(xlabel="Ri", ylabel="Fturb = <v' theta'>",
                title="turbulent heat flux vs Ri")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "candidate1_plumes.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--aspect", type=float, default=4.0)
    p.add_argument("--ri", default="0.0,0.25,0.5,0.75,1.0,1.5")
    p.add_argument("--closures", default="none,smagorinsky,backscatter")
    p.add_argument("--f-amp", dest="f_amp", type=float, default=0.6)
    p.add_argument("--sigma-h", dest="sigma_h", type=float, default=0.30)
    p.add_argument("--spinup", type=int, default=2000)
    p.add_argument("--measure", type=int, default=4000)
    p.add_argument("--sample-every", dest="sample_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rough-seed", dest="rough_seed", type=int, default=0)
    p.add_argument("--out-dir", dest="out_dir", default="figures")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    xp, backend = get_backend()
    print(f"Backend: {backend}")
    print(f"grid nx={args.nx} ny={args.ny} sigma_h={args.sigma_h} "
          f"spinup={args.spinup} measure={args.measure}")
    t0 = time.time()
    results = run_sweep(args, xp)
    path = maybe_figure(results, args)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "candidate1_plumes.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"backend": backend, "args": vars(args), "results": results},
                  fh, indent=2)
    print(f"wrote {json_path}")
    if path:
        print(f"wrote {path}")

    # honest summary: is the intermittency geometric (flat in Ri/closure)?
    lines = []
    for closure in sorted({r["closure"] for r in results}):
        sub = sorted((r for r in results if r["closure"] == closure),
                     key=lambda r: r["Ri"])
        pm = [r["peak_mean"] for r in sub]
        ft = [r["fturb_mean"] for r in sub]
        pm_spread = (max(pm) - min(pm)) / (np.mean(pm) + 1e-30)
        lines.append(f"{closure:11s}: peak/mean={[round(x, 2) for x in pm]} "
                     f"(rel spread {pm_spread:.1%}; ~0 => geometric) "
                     f"Fturb={[round(x, 5) for x in ft]}")
    print("\n".join(lines))
    print(f"total {time.time() - t0:.1f}s")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("# Candidate 1 -- intermittent plumes from ice-base roughness\n\n")
            fh.write(f"Backend: {backend}\n\n```\n" + "\n".join(lines) + "\n```\n")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
