r"""3D scallop geometry gate -- the *decisive* dimensionality test.

The 2D transverse-wave amplitude scan (``scallop_sweep.py --amp``) showed
``Nu/Nu_flat < 1`` at *every* amplitude: a spanwise-homogeneous wave forces the
flow *over* every bump, so more roughness only adds drag and the mean basal heat
flux never beats a flat wall.

This script tests whether a **3D streamwise-oriented ridge** changes the answer.
A ridge whose crest line runs in the streamwise (+x) direction varies in the
spanwise (z) direction only, so the flow can **channel between ridges** -- a
low-drag hydraulic path that the 2D topology cannot express.  If that
flow-concentration mechanism is real, the ridged bed delivers *more* heat to the
ice base than a flat bed: ``Nu/Nu_flat > 1``.

Geometry (3D penalized solver ``subglacial/flow3d.py``, triply-periodic
[0, 2*pi)^3; y vertical, x streamwise, z spanwise):

  * ``ridge``     -- h(x, z) = bed_mean + a*cos(kz*z)         (pure streamwise
                     ridges: continuous low-drag troughs along x)
  * ``eggcarton`` -- h(x, z) = bed_mean + a*sin(kx*x)*cos(kz*z) (ridges modulated
                     along x too; flow still meets bumps in x)
  * ``flat``      -- h = bed_mean (the control; identical mean cavity gap)

Observable: ``Nu`` = mean upward basal heat flux delivered to the (flat) ice base
(``Subglacial3DFlow.melt_flux``), time-averaged over snapshots after spinup.  The
measurement surface (ice base) is flat and identical in every run, so the only
difference is the bed geometry -> a fair geometric ``Nu/Nu_flat``.

Regime: this test is run in the **turbulent LES regime** of ``run_subglacial3d.py``
(nu=5e-5, dt=3e-4, f_amp=2, k_f=10, Smagorinsky SGS), NOT the solver's laminar
defaults.  Vertical heat transport across the cavity to the ice base requires a
developed (or at least secondary-flow) field; at the laminar default (nu=8e-4,
sgs='none') the flow is a steady horizontal shear, no warm bed fluid ever reaches
the ice-base band, and ``Nu`` collapses to diffusive noise (~0, even slightly
negative).  A closure is therefore required for the cascade to be resolved/closed;
the geometry gate asks whether 3D topology can beat flat *at all*, which is a
separate question from the 2D closure-independence check.

GATE:  Nu/Nu_flat > 1  -> channelization works; the scale-coupling framework has
       an empirical foundation and multi-mode is justified.
       Nu/Nu_flat <= 1 -> the no-slip (penalized) wall is the hard limit even in
       3D; document that boundary honestly.

Usage (CPU smoke):
    python scallop3d_probe.py --n 32 --spinup 400 --snaps 4 --snap-every 50 \
        --amps 0.10,0.20 --geom ridge

Usage (GPU, the real run):
    python scallop3d_probe.py --gpu --n 128 --spinup 3000 --snaps 8 \
        --snap-every 200 --amps 0.10,0.20 --geom ridge
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from subglacial.flow3d import (
    Subglacial3DConfig,
    Subglacial3DFlow,
    divergence_rms3d,
)


def get_backend(force_gpu: bool):
    """Return (xp, name). Prefer CuPy if --gpu or if a working device exists."""
    if force_gpu:
        import cupy as cp
        cp.zeros(1).sum()
        return cp, "cupy(GPU)"
    try:
        import cupy as cp
        cp.zeros(1).sum()
        return cp, "cupy(GPU)"
    except Exception:  # noqa: BLE001 - any import/runtime/device failure -> CPU
        return np, "numpy(CPU)"


def bed_field(geom, n, bed_mean, a, kx, kz):
    """2-D bed-top height h(x, z), shape (n, n) over (x, z) on [0, 2*pi)^2."""
    x = np.arange(n) * (2.0 * np.pi / n)
    z = np.arange(n) * (2.0 * np.pi / n)
    X, Z = np.meshgrid(x, z, indexing="ij")
    if geom == "flat":
        return np.full((n, n), bed_mean)
    if geom == "ridge":
        return bed_mean + a * np.cos(kz * Z)
    if geom == "eggcarton":
        return bed_mean + a * np.sin(kx * X) * np.cos(kz * Z)
    raise ValueError(f"unknown geom {geom!r}")


def run_one(xp, name, geom, n, a, kx, kz, args, seed=0):
    """Spin up a 3D cavity over the given bed and time-average the basal Nu."""
    hf = bed_field(geom, n, args.bed_mean, a, kx, kz)
    cfg = Subglacial3DConfig(
        n=n, sgs=args.sgs, bed_field=hf,
        nu=args.nu, kappa=(args.kappa if args.kappa is not None else args.nu),
        eta=5.0e-5, U0=args.U0, dt=args.dt,
        bed_mean=args.bed_mean, ice_base=args.ice_base,
        cs=args.cs, f_amp=args.f_amp, k_f=args.k_f, f_band=args.f_band,
        seed=seed,
    )
    flow = Subglacial3DFlow(cfg, xp=xp)
    t0 = time.time()
    flow.run(args.spinup, ramp=max(1, args.spinup // 3))
    nu_t, tke, ti = [], [], []
    for _ in range(args.snaps):
        flow.run(args.snap_every)
        nu_t.append(flow.melt_flux()[0])
        tke.append(flow.wake_tke())
        ti.append(flow.turbulence_intensity())
    dt_wall = time.time() - t0
    return {
        "geom": geom, "a": a, "a_over_lam": a / (2.0 * np.pi / kz),
        "kx": kx, "kz": kz,
        "Nu": float(np.mean(nu_t)), "Nu_std": float(np.std(nu_t)),
        "wake_tke": float(np.mean(tke)),
        "turb_intensity": float(np.mean(ti)),
        "div_rms": divergence_rms3d(flow.sp, flow.u, flow.v, flow.w),
        "umax": float(xp.abs(flow.u).max()),
        "fvol_frac": flow.fvol / (n ** 3),
        "wall_s": dt_wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--spinup", type=int, default=2500)
    ap.add_argument("--snaps", type=int, default=8)
    ap.add_argument("--snap-every", type=int, default=150)
    ap.add_argument("--amps", type=str, default="0.10,0.20",
                    help="comma-separated a/lambda values")
    ap.add_argument("--geom", type=str, default="ridge",
                    help="ridge | eggcarton (flat control always added)")
    ap.add_argument("--nwaves-span", type=int, default=3,
                    help="ridges across the spanwise box (kz); 3 ~ lambda_opt 2.094")
    ap.add_argument("--nu", type=float, default=5.0e-5,
                    help="molecular viscosity; low => high Re => turbulent LES")
    ap.add_argument("--kappa", type=float, default=None,
                    help="thermal diffusivity; defaults to --nu (Pr=1)")
    ap.add_argument("--dt", type=float, default=3.0e-4)
    ap.add_argument("--U0", type=float, default=1.0)
    ap.add_argument("--bed-mean", type=float, default=0.9)
    ap.add_argument("--ice-base", type=float, default=2.4)
    ap.add_argument("--sgs", type=str, default="smagorinsky",
                    help="'none' (laminar) | 'smagorinsky' | 'backscatter'")
    ap.add_argument("--cs", type=float, default=0.17)
    ap.add_argument("--f-amp", type=float, default=2.0)
    ap.add_argument("--k-f", type=float, default=10.0)
    ap.add_argument("--f-band", type=float, default=3.0)
    ap.add_argument("--out", type=str, default="scallop3d_gpu.json")
    args = ap.parse_args()

    xp, name = get_backend(args.gpu)
    n = args.n
    kz = args.nwaves_span
    kx = args.nwaves_span
    lam = 2.0 * np.pi / kz
    amps = [float(s) for s in args.amps.split(",") if s.strip()]
    print(f"BACKEND {name}  n={n}  geom={args.geom}  kz={kz} (lambda={lam:.3f}, "
          f"lambda/dx={lam/(2*np.pi/n):.1f})  amps(a/lam)={amps}", flush=True)

    # flat control first -> Nu_flat
    flat = run_one(xp, name, "flat", n, 0.0, kx, kz, args)
    Nu_flat = flat["Nu"]
    print(f"[flat ] Nu_flat={Nu_flat:.6e} ti={flat['turb_intensity']:.3f} "
          f"div={flat['div_rms']:.2e} umax={flat['umax']:.3f} "
          f"({flat['wall_s']:.0f}s)", flush=True)

    rows = [flat]
    for r in amps:
        a = r * lam
        res = run_one(xp, name, args.geom, n, a, kx, kz, args)
        res["Nu_flat"] = Nu_flat
        res["Nu_ratio"] = res["Nu"] / Nu_flat if Nu_flat != 0 else float("nan")
        rows.append(res)
        print(f"[{args.geom[:5]:5s}] a/lam={r:.2f} a={a:.3f}  Nu={res['Nu']:.6e}  "
              f"Nu/Nu_flat={res['Nu_ratio']:.4f}  ti={res['turb_intensity']:.3f} "
              f"div={res['div_rms']:.2e} umax={res['umax']:.3f} "
              f"({res['wall_s']:.0f}s)", flush=True)

    out = {"backend": name, "n": n, "geom": args.geom, "kx": kx, "kz": kz,
           "lambda": lam, "Nu_flat": Nu_flat, "rows": rows,
           "params": {k: getattr(args, k) for k in
                      ("nu", "kappa", "dt", "U0", "bed_mean", "ice_base", "sgs",
                       "f_amp", "k_f", "f_band", "spinup", "snaps", "snap_every")}}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("WROTE " + args.out, flush=True)
    # compact final verdict line
    best = max((rw.get("Nu_ratio", float("nan")) for rw in rows[1:]),
               default=float("nan"))
    verdict = "GATE PASS (channelization)" if best > 1.0 else "GATE FAIL (wall-limited)"
    print(f"VERDICT max Nu/Nu_flat={best:.4f} -> {verdict}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
