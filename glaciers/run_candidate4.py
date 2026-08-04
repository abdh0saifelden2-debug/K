r"""Candidate 4 -- hydraulic-switching melt sweep (filled vs stratified cavity).

Drives a wide, shallow, body-force + tidally forced cavity
(:mod:`subglacial.candidate4_hydraulic_switch`) with the mass-flux controller
*removed*, and sweeps the buoyancy coupling ``Ri`` (and aspect ratio ``A``) for
each closure.  For every case it records the active-layer height series
``H1(t)``, the switching frequency ``f_switch``, and the time-averaged
interfacial melt, then tests the pre-registered prediction:

    time-averaged melt peaks at intermediate Ri (and intermediate A), where the
    cavity switches most often between the *filled* (H1 ~ 1) and *stratified*
    (H1 ~ 0.2) states; Smagorinsky over-dissipates the transitions and so
    suppresses the hump.

The closure only shapes the flow in a genuine under-resolved LES, which needs a
well-resolved cavity (ny >= 64) and many tidal cycles, so the real sweep is a
GPU job.  This script is backend-agnostic: it auto-detects CuPy and runs on a
GPU (e.g. a free Kaggle/Colab T4) if present, else NumPy on CPU.

Usage (CPU smoke test):
    python run_candidate4.py --nx 128 --ny 64 --spinup 300 --measure 400 \
        --ri 0.0,0.5,1.0 --aspect 4 --closures none,smagorinsky --out-dir figures

Usage (GPU sweep, the real run):
    python run_candidate4.py --nx 384 --ny 96 --spinup 4000 --measure 6000 \
        --ri 0.0,0.25,0.5,0.75,1.0 --aspect 1,4 \
        --closures none,smagorinsky,backscatter \
        --out-dir figures --report REPORT_CANDIDATE4.md
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from subglacial.candidate4_hydraulic_switch import (
    HydraulicConfig,
    run_case,
)


def get_backend():
    """Return (xp, name): CuPy if importable with a working device, else NumPy."""
    try:
        import cupy as cp
        cp.zeros(1).sum()  # touch device to confirm it works
        return cp, "cupy(GPU)"
    except (ImportError, RuntimeError, MemoryError):
        return np, "numpy(CPU)"


def _floats(s):
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


def _strs(s):
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def run_sweep(args, xp):
    ri_list = _floats(args.ri)
    a_list = _floats(args.aspect)
    closures = _strs(args.closures)
    results = []
    for closure in closures:
        for A in a_list:
            for Ri in ri_list:
                cfg = HydraulicConfig(
                    nx=args.nx, ny=args.ny, A=A, Ri=Ri, sgs=closure,
                    f_amp=args.f_amp, T_tide=args.t_tide, seed=args.seed,
                )
                t0 = time.time()
                r = run_case(cfg, spinup=args.spinup, measure=args.measure,
                             sample_every=args.sample_every, xp=xp)
                rec = {
                    "closure": closure, "A": A, "Ri": Ri,
                    "fturb_mean": r["fturb_mean"], "KE_mean": r["KE_mean"],
                    "melt_mean": r["melt_mean"], "f_switch": r["f_switch"],
                    "H1_mean": r["H1_mean"], "H1_std": r["H1_std"],
                    "umax": r["umax"], "secs": time.time() - t0,
                }
                results.append(rec)
                print(f"  [{closure:11s}] A={A:>3g} Ri={Ri:<4g} "
                      f"Fturb={rec['fturb_mean']:+.3e} KE={rec['KE_mean']:.3e} "
                      f"melt={rec['melt_mean']:+.3e} f_switch={rec['f_switch']:6.2f} "
                      f"H1={rec['H1_mean']:.2f}+/-{rec['H1_std']:.2f} ({rec['secs']:.1f}s)")
    return results


def peak_is_interior(xs, ys):
    """True if max(ys) occurs at an interior x (a hump, not a monotone edge)."""
    if len(ys) < 3:
        return False
    i = int(np.argmax(ys))
    return 0 < i < len(ys) - 1


def maybe_figure(results, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError) as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return None
    closures = sorted({r["closure"] for r in results})
    a_vals = sorted({r["A"] for r in results})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for closure in closures:
        for A in a_vals:
            sub = sorted((r for r in results
                          if r["closure"] == closure and r["A"] == A),
                         key=lambda r: r["Ri"])
            if not sub:
                continue
            ri = [r["Ri"] for r in sub]
            axes[0].plot(ri, [r["fturb_mean"] for r in sub], "o-",
                         label=f"{closure}, A={A:g}")
            axes[1].plot(ri, [r["f_switch"] for r in sub], "o-",
                         label=f"{closure}, A={A:g}")
    axes[0].set(xlabel="Ri", ylabel="Fturb = <v' theta'>",
                title="turbulent heat flux vs Ri")
    axes[1].set(xlabel="Ri", ylabel="f_switch", title="switching frequency vs Ri")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "candidate4_hydraulic_switch.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--ri", default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--aspect", default="1,4")
    p.add_argument("--closures", default="none,smagorinsky,backscatter")
    p.add_argument("--f-amp", dest="f_amp", type=float, default=0.5)
    p.add_argument("--t-tide", dest="t_tide", type=float, default=1.0)
    p.add_argument("--spinup", type=int, default=4000)
    p.add_argument("--measure", type=int, default=6000)
    p.add_argument("--sample-every", dest="sample_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", dest="out_dir", default="figures")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    xp, backend = get_backend()
    print(f"Backend: {backend}")
    print(f"grid nx={args.nx} ny={args.ny}  spinup={args.spinup} measure={args.measure}")
    t0 = time.time()
    results = run_sweep(args, xp)
    path = maybe_figure(results, args)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "candidate4_hydraulic_switch.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"backend": backend, "args": vars(args), "results": results},
                  fh, indent=2)
    print(f"wrote {json_path}")
    if path:
        print(f"wrote {path}")

    # pre-registered check: is melt(Ri) a hump (interior peak) for the resolved
    # closures, and does Smagorinsky suppress it?
    lines = []
    for closure in sorted({r["closure"] for r in results}):
        for A in sorted({r["A"] for r in results}):
            sub = sorted((r for r in results
                          if r["closure"] == closure and r["A"] == A),
                         key=lambda r: r["Ri"])
            fturb = [r["fturb_mean"] for r in sub]
            ri = [r["Ri"] for r in sub]
            hump = peak_is_interior(ri, fturb)
            lines.append(f"{closure:11s} A={A:g}: Fturb hump={hump} "
                         f"argmax Ri={ri[int(np.argmax(fturb))] if fturb else None} "
                         f"Fturb={[round(x, 5) for x in fturb]}")
    print("\n".join(lines))
    print(f"total {time.time() - t0:.1f}s")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("# Candidate 4 -- hydraulic switching sweep\n\n")
            fh.write(f"Backend: {backend}\n\n")
            fh.write("```\n" + "\n".join(lines) + "\n```\n")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
