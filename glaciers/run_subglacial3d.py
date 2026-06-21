r"""Part 9b -- 3D a-posteriori multi-site subglacial cavity melt-rate study.

Time-integrates the 3D penalized cavity (``subglacial/flow3d.py``) under each
closure (Smagorinsky K-theory vs the two-clocks projected-FDT backscatter), over
several *real* BEDMAP1 bed segments, and reports the **basal melt-rate factor**
projected-FDT / Smagorinsky at each site, time-averaged over the statistically
steady wake.

The closure only controls the melt rate in a genuine **under-resolved LES**: the
grid must be coarse relative to a developed turbulent cascade so the SGS model is
the dominant dissipation.  That needs n >= 128 with low molecular viscosity, which
is only tractable on a GPU.  This script is **backend-agnostic**: it auto-detects
CuPy and runs on a GPU (e.g. a free Colab/Kaggle T4) if present, else NumPy on CPU.

It prints, and writes to JSON, the melt factor *and* the LES-regime diagnostics
(turbulence intensity, SGS/molecular dissipation ratio) so the result is only
trusted when the run is actually closure-dominated.  Honest scope: an LES
demonstration, not a grid-converged DNS or a validated production glacier model.

Usage (CPU smoke test):
    python run_subglacial3d.py --n 32 --spinup 60 --snaps 3 --snap-every 20 \
        --sites 0.0,0.34 --closures smagorinsky,backscatter --out-dir figures

Usage (GPU LES, the real run):
    python run_subglacial3d.py --n 128 --spinup 3000 --snaps 8 --snap-every 200 \
        --out-dir figures --report REPORT_SUBGLACIAL3D.md
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from subglacial.bedmap import bed_window_profile
from subglacial.flow3d import Subglacial3DConfig, Subglacial3DFlow, divergence_rms3d


def get_backend():
    """Return (xp, name, to_np): CuPy if importable with a working device, else NumPy."""
    try:
        import cupy as cp
        cp.zeros(1).sum()  # touch device to confirm it works
        return cp, "cupy(GPU)", (lambda a: cp.asnumpy(a))
    except (ImportError, RuntimeError, MemoryError):
        return np, "numpy(CPU)", (lambda a: np.asarray(a))


DEFAULT_SITES = "0.0,0.34;0.34,0.67;0.67,1.0"   # 3 distinct real segments
BED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subglacial", "data", "bedmap1_transect.csv")


def parse_sites(s: str):
    out = []
    for i, chunk in enumerate(s.split(";")):
        chunk = chunk.strip()
        if not chunk:
            continue
        lo, hi = (float(x) for x in chunk.split(","))
        out.append((f"seg{i+1}", lo, hi))
    return out


def run_one(xp, to_np, n, sgs, bed_profile, args, seed):
    kappa = args.kappa if args.kappa is not None else args.nu
    cfg = Subglacial3DConfig(
        n=n, sgs=sgs, bed_profile=bed_profile,
        nu=args.nu, kappa=kappa, eta=5.0e-5, U0=1.0, dt=args.dt,
        cs=args.cs, backscatter=args.backscatter,
        f_amp=args.f_amp, k_f=args.k_f, f_band=args.f_band, seed=seed,
    )
    flow = Subglacial3DFlow(cfg, xp=xp)
    t0 = time.time()
    flow.run(args.spinup, ramp=max(1, args.spinup // 3))

    melt, tke, heat, drag = [], [], [], []
    ti, eps_mol, eps_sgs = [], [], []
    for _ in range(args.snaps):
        flow.run(args.snap_every)
        melt.append(flow.melt_flux()[0])
        tke.append(flow.wake_tke())
        heat.append(flow.heat_in_wake())
        drag.append(flow.basal_drag())
        ti.append(flow.turbulence_intensity())
        em, es = flow.dissipation_breakdown()
        eps_mol.append(em)
        eps_sgs.append(es)
    dt_wall = time.time() - t0

    def m(a):
        return float(np.mean(a))

    return {
        "sgs": sgs,
        "melt": m(melt), "melt_std": float(np.std(melt)),
        "wake_tke": m(tke), "heat_in_wake": m(heat), "basal_drag": m(drag),
        "turb_intensity": m(ti),
        "eps_molecular": m(eps_mol), "eps_sgs": m(eps_sgs),
        "sgs_dominance": m(eps_sgs) / (m(eps_mol) + 1e-30),
        "div_rms": divergence_rms3d(flow.sp, flow.u, flow.v, flow.w),
        "wall_s": dt_wall,
        "field_slice": to_np(flow.theta[:, :, n // 2]).tolist() if args.save_slices else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--spinup", type=int, default=3000)
    ap.add_argument("--snaps", type=int, default=8)
    ap.add_argument("--snap-every", type=int, default=200)
    ap.add_argument("--nu", type=float, default=5.0e-5,
                    help="molecular viscosity; low => high Re => SGS-dominated LES")
    ap.add_argument("--kappa", type=float, default=None,
                    help="thermal diffusivity; defaults to --nu (Prandtl number = 1). "
                         "Set != nu to explore Pr != 1 (e.g. stratification effects)")
    ap.add_argument("--dt", type=float, default=3.0e-4)
    ap.add_argument("--cs", type=float, default=0.17)
    ap.add_argument("--backscatter", type=float, default=0.7)
    ap.add_argument("--f-amp", type=float, default=2.0)
    ap.add_argument("--k-f", type=float, default=10.0)
    ap.add_argument("--f-band", type=float, default=3.0)
    ap.add_argument("--sites", default=DEFAULT_SITES)
    ap.add_argument("--closures", default="smagorinsky,backscatter")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--report", default=None)
    ap.add_argument("--json", default="subglacial3d_results.json")
    ap.add_argument("--save-slices", action="store_true")
    args = ap.parse_args()

    xp, backend, to_np = get_backend()
    os.makedirs(args.out_dir, exist_ok=True)
    sites = parse_sites(args.sites)
    closures = [c.strip() for c in args.closures.split(",") if c.strip()]

    print(f"backend = {backend}   n = {args.n}   sites = {[s[0] for s in sites]}   "
          f"closures = {closures}")
    print(f"LES regime: nu={args.nu:g} dt={args.dt:g} f_amp={args.f_amp:g} "
          f"k_f={args.k_f:g} cs={args.cs:g} backscatter={args.backscatter:g}")

    results = {"backend": backend, "n": args.n, "args": vars(args), "sites": {}}
    for name, lo, hi in sites:
        prof, meta = bed_window_profile(BED_FILE, args.n, 0.9, 0.55, lo, hi)
        print(f"\n=== site {name}  (real BEDMAP1 km {lo:.2f}-{hi:.2f} of line, "
              f"relief {meta['relief_m']:.0f} m) ===")
        site = {"meta": meta, "closures": {}}
        for c in closures:
            r = run_one(xp, to_np, args.n, c, prof, args, seed=hash((name, c)) % 2**31)
            site["closures"][c] = r
            print(f"  {c:12s} melt={r['melt']:.4e}+-{r['melt_std']:.1e}  "
                  f"TI={r['turb_intensity']:.2f}  SGS/mol={r['sgs_dominance']:.1f}  "
                  f"div={r['div_rms']:.1e}  ({r['wall_s']:.0f}s)")
        if "smagorinsky" in site["closures"] and "backscatter" in site["closures"]:
            ms = site["closures"]["smagorinsky"]["melt"]
            mb = site["closures"]["backscatter"]["melt"]
            site["melt_factor"] = mb / ms if ms != 0 else float("nan")
            print(f"  --> melt factor (backscatter/smagorinsky) = {site['melt_factor']:.2f}")
        results["sites"][name] = site

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.json}")

    factors = [results["sites"][s].get("melt_factor") for s in results["sites"]
               if "melt_factor" in results["sites"][s]]
    if factors:
        print(f"melt factors across sites: "
              f"{', '.join(f'{f:.2f}' for f in factors)}  "
              f"(mean {np.mean(factors):.2f})")

    _maybe_figures(results, args, to_np)
    if args.report:
        _write_report(results, args)


def _maybe_figures(results, args, to_np):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError) as e:  # pragma: no cover
        # ImportError: matplotlib not installed; ValueError: backend can't be
        # switched (e.g. pyplot already imported with another backend).
        print(f"(skipping figures: {e})")
        return
    sites = list(results["sites"].keys())
    closures = [c.strip() for c in args.closures.split(",") if c.strip()]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sites))
    width = 0.8 / max(1, len(closures))
    for j, c in enumerate(closures):
        vals = [results["sites"][s]["closures"][c]["melt"] for s in sites]
        ax.bar(x + j * width, vals, width, label=c)
    ax.set_xticks(x + width * (len(closures) - 1) / 2)
    ax.set_xticklabels(sites)
    ax.set_ylabel("basal melt flux (heat into ice base)")
    ax.set_title(f"3D LES basal melt rate by closure (n={args.n})")
    ax.legend()
    p = os.path.join(args.out_dir, "60_subglacial3d_melt.png")
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    print(f"wrote {p}")


def _write_report(results, args):
    lines = ["# Part 9b -- 3D a-posteriori multi-site cavity melt rate", ""]
    lines.append(f"Backend: `{results['backend']}`  |  resolution n={args.n}  |  "
                 f"closures: {args.closures}")
    lines.append("")
    lines.append("LES regime (closure-dominated only if turbulence intensity is "
                 "high **and** SGS/molecular dissipation >> 1):")
    lines.append("")
    lines.append("| site | relief (m) | closure | melt | turb. intensity | SGS/mol | div |")
    lines.append("|------|-----------|---------|------|-----------------|---------|-----|")
    for s, site in results["sites"].items():
        for c, r in site["closures"].items():
            lines.append(f"| {s} | {site['meta']['relief_m']:.0f} | {c} | "
                         f"{r['melt']:.3e} | {r['turb_intensity']:.2f} | "
                         f"{r['sgs_dominance']:.1f} | {r['div_rms']:.1e} |")
    lines.append("")
    for s, site in results["sites"].items():
        if "melt_factor" in site:
            lines.append(f"- **{s}**: melt factor (projected-FDT / Smagorinsky) = "
                         f"**{site['melt_factor']:.2f}**")
    lines.append("")
    lines.append("![melt](60_subglacial3d_melt.png)")
    lines.append("")
    lines.append("> Scope: a-posteriori **LES demonstration** over real BEDMAP1 bed "
                 "segments, not a grid-converged DNS and not a validated production "
                 "glacier model. The melt factor is only physically meaningful in "
                 "rows where the run is genuinely closure-dominated (SGS/mol >> 1).")
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
