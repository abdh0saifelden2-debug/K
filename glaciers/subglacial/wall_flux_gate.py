r"""Decisive flow-ON vs flow-OFF gate for the finite-conductance ice wall.

The 2D and 3D Dirichlet-wall studies showed flow adds ~0% to basal heat because
the infinite-conductance ice pin makes the near-wall conductive sublayer the
controlling resistance. ``wall_flux.ThermalWall`` replaces that pin with a
finite-conductance (Robin-like) sink. This script runs the controlled 2x2:

    {flat, ridge} x {flow-ON (turbulent forcing), flow-OFF (pure conduction)}

all with the SAME finite ``cond_ratio``, and reports:

  * channelization delta = ridge(flow-ON) - ridge(flow-OFF):
        does the flow deliver heat beyond static geometry?
  * geometry gate       = ridge(flow-ON) vs flat(flow-ON):
        does the ridge beat a flat wall under flow?

Observables: ``wall_flux`` (heat absorbed by the ice sink, from ThermalWall) and
the solver's advective+diffusive ``melt_flux`` band proxy. The backend is
auto-detected: CuPy (GPU) is used whenever it is importable, otherwise NumPy
(CPU); ``--gpu`` *requires* the GPU and errors out if CuPy is unavailable.

Usage:
    python -m subglacial.wall_flux_gate --n 32 --spinup 400 --snaps 4 \
        --snap-every 50 --cond-ratio 0.05
    python -m subglacial.wall_flux_gate --gpu --n 128 --spinup 2000 --snaps 6 \
        --snap-every 150 --cond-ratio 0.05
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from subglacial.flow3d import Subglacial3DConfig, Subglacial3DFlow, divergence_rms3d
from subglacial.wall_flux import ThermalWall


def get_backend(force_gpu: bool):
    if force_gpu:
        import cupy as cp
        cp.zeros(1).sum()
        return cp, "cupy(GPU)"
    try:
        import cupy as cp
        cp.zeros(1).sum()
        return cp, "cupy(GPU)"
    except Exception:  # noqa: BLE001 - any import/device failure -> CPU
        return np, "numpy(CPU)"


def bed_field(geom, n, bed_mean, a, kz):
    """2-D bed-top height h(x, z) over [0, 2*pi)^2: flat or streamwise ridges."""
    z = np.arange(n) * (2.0 * np.pi / n)
    _, Z = np.meshgrid(np.arange(n) * (2.0 * np.pi / n), z, indexing="ij")
    if geom == "flat":
        return np.full((n, n), bed_mean)
    if geom == "ridge":
        return bed_mean + a * np.cos(kz * Z)
    raise ValueError(f"unknown geom {geom!r}")


def run_one(xp, geom, flow_on, args, seed=0):
    """One cavity run; returns time-averaged wall_flux and melt_flux."""
    hf = bed_field(geom, args.n, args.bed_mean, args.a, args.kz)
    tw = ThermalWall(cond_ratio=args.cond_ratio, mode=args.mode)
    cfg = Subglacial3DConfig(
        n=args.n, sgs=args.sgs, bed_field=hf,
        nu=args.nu, kappa=(args.kappa if args.kappa is not None else args.nu),
        eta=5.0e-5,
        U0=(args.U0 if flow_on else 0.0),
        f_amp=(args.f_amp if flow_on else 0.0),
        dt=args.dt, bed_mean=args.bed_mean, ice_base=args.ice_base,
        k_f=args.k_f, f_band=args.f_band, seed=seed,
        thermal_wall=tw,
    )
    flow = Subglacial3DFlow(cfg, xp=xp)
    t0 = time.time()
    flow.run(args.spinup, ramp=max(1, args.spinup // 3))
    wflux, mflux, umax = [], [], []
    for _ in range(args.snaps):
        flow.run(args.snap_every)
        wflux.append(tw._last_flux)
        mflux.append(flow.melt_flux()[0])
        umax.append(float(xp.abs(flow.u).max()))
    return {
        "geom": geom, "flow": "ON" if flow_on else "OFF",
        "wall_flux": float(np.mean(wflux)),
        "melt_flux": float(np.mean(mflux)),
        "umax": float(np.mean(umax)),
        "div_rms": divergence_rms3d(flow.sp, flow.u, flow.v, flow.w),
        "wall_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--spinup", type=int, default=400)
    ap.add_argument("--snaps", type=int, default=4)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--cond-ratio", type=float, default=0.05)
    ap.add_argument("--mode", type=str, default="robin_const")
    ap.add_argument("--a", type=float, default=0.30)       # ridge amplitude
    ap.add_argument("--kz", type=float, default=4.0)       # spanwise ridge count
    ap.add_argument("--nu", type=float, default=5.0e-5)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--U0", type=float, default=1.0)
    ap.add_argument("--f-amp", type=float, default=2.0)
    ap.add_argument("--k-f", type=float, default=6.0)
    ap.add_argument("--f-band", type=float, default=2.0)
    ap.add_argument("--sgs", type=str, default="smagorinsky")
    ap.add_argument("--dt", type=float, default=4.0e-4)
    ap.add_argument("--bed-mean", type=float, default=0.9)
    ap.add_argument("--ice-base", type=float, default=2.4)
    args = ap.parse_args()

    xp, backend = get_backend(args.gpu)
    print(f"backend={backend}  n={args.n}  cond_ratio={args.cond_ratio}  "
          f"mode={args.mode}  a={args.a} kz={args.kz}  spinup={args.spinup}", flush=True)

    rows = []
    for geom in ("flat", "ridge"):
        for flow_on in (True, False):
            r = run_one(xp, geom, flow_on, args)
            rows.append(r)
            print(f"  {geom:5s} flow-{r['flow']:3s}  "
                  f"wall_flux={r['wall_flux']:.6e}  melt_flux={r['melt_flux']:.6e}  "
                  f"umax={r['umax']:.3f}  div={r['div_rms']:.1e}  ({r['wall_s']:.0f}s)",
                  flush=True)

    def pick(geom, flow):
        return next(r for r in rows if r["geom"] == geom and r["flow"] == flow)

    rf_on, rf_off = pick("ridge", "ON"), pick("ridge", "OFF")
    ff_on = pick("flat", "ON")
    for obs in ("wall_flux", "melt_flux"):
        chan = rf_on[obs] - rf_off[obs]
        chan_ratio = rf_on[obs] / rf_off[obs] if rf_off[obs] != 0 else float("nan")
        gate = rf_on[obs] / ff_on[obs] if ff_on[obs] != 0 else float("nan")
        print(f"\n[{obs}] channelization delta (ridge ON-OFF) = {chan:.6e} "
              f"(ratio {chan_ratio:.4f})")
        print(f"[{obs}] geometry gate (ridge ON / flat ON)   = {gate:.4f}")


if __name__ == "__main__":
    main()
