r"""Navier bed-slip gate: is the *stagnant near-bed layer* the bottleneck?

Every prior study (2D/3D, laminar/turbulent, Dirichlet/Robin ice wall) found the
cavity is **wall-limited** -- flow adds ~0% to basal heat -- under a **no-slip**
bed.  The one remaining honest test is to relax the no-slip condition itself: if
the bed *slides* (Weertman/regelation), the stagnant penalty layer thins and
turbulent eddies can reach the warm bed, so heat advection could finally
penetrate the boundary layer.

This script sweeps the bed-slip knob ``bed_slip`` (s=1 no-slip ... s->0 free
slip) at fixed turbulent forcing and reports, for both a flat and a ridged bed:

  * ``melt_flux``  -- the solver's advective+diffusive heat-to-ice proxy
  * ``Nu/Nu_flat`` -- ridge melt_flux normalized by the flat-bed melt_flux at the
                      SAME slip (does the ridge geometry beat a flat wall?)
  * ``u_tang``     -- mean tangential speed in the bed transition band (the direct
                      check that slip actually energized the near-bed layer)

The honest gate:
  * if some s<1 lifts melt_flux / Nu well above the no-slip (s=1) value, the
    stagnant near-bed layer WAS the limit and the mechanism is real;
  * if melt_flux stays flat across all s even as ``u_tang`` grows large, the
    limit is the thermal conductive sublayer, not the momentum one -- genuinely
    wall-limited regardless of sliding.

Usage:
    python -m subglacial.slip_gate --n 32 --spinup 400 --snaps 4 --snap-every 50
    python -m subglacial.slip_gate --gpu --n 128 --spinup 2000 --snaps 6 \
        --snap-every 150 --slips 1.0 0.3 0.1 0.0
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from subglacial.flow3d import Subglacial3DConfig, Subglacial3DFlow, divergence_rms3d


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


def _u_tang(flow, xp):
    """Mean flow-tangential speed in the bed transition band (slip diagnostic)."""
    if flow.bed_slip is None:
        return 0.0
    band = (flow.chi_rock > 0.3) & (flow.chi_rock < 0.7)
    nx, ny, nz = flow._bn_x, flow._bn_y, flow._bn_z
    un = flow.u * nx + flow.v * ny + flow.w * nz
    ut = xp.sqrt((flow.u - un * nx) ** 2
                 + (flow.v - un * ny) ** 2
                 + (flow.w - un * nz) ** 2)
    denom = float(band.sum()) + 1e-30
    return float((ut * band).sum() / denom)


def run_one(xp, geom, slip, args, seed=0):
    """One turbulent cavity run at fixed bed-slip; returns time-averaged stats."""
    hf = bed_field(geom, args.n, args.bed_mean, args.a, args.kz)
    cfg = Subglacial3DConfig(
        n=args.n, sgs=args.sgs, bed_field=hf,
        nu=args.nu, kappa=(args.kappa if args.kappa is not None else args.nu),
        eta=5.0e-5, U0=args.U0, f_amp=args.f_amp,
        dt=args.dt, bed_mean=args.bed_mean, ice_base=args.ice_base,
        k_f=args.k_f, f_band=args.f_band, seed=seed,
        bed_slip=slip,                       # None = no-slip baseline (s=1); see caller
    )
    flow = Subglacial3DFlow(cfg, xp=xp)
    t0 = time.time()
    flow.run(args.spinup, ramp=max(1, args.spinup // 3))
    mflux, umax, utang = [], [], []
    for _ in range(args.snaps):
        flow.run(args.snap_every)
        mflux.append(flow.melt_flux()[0])
        umax.append(float(xp.abs(flow.u).max()))
        utang.append(_u_tang(flow, xp))
    return {
        "geom": geom, "slip": slip,
        "melt_flux": float(np.mean(mflux)),
        "umax": float(np.mean(umax)),
        "u_tang": float(np.mean(utang)),
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
    ap.add_argument("--slips", type=float, nargs="+", default=[1.0, 0.3, 0.1, 0.0])
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
    print(f"backend={backend}  n={args.n}  slips={args.slips}  a={args.a} "
          f"kz={args.kz}  nu={args.nu}  kappa={args.kappa}  spinup={args.spinup}",
          flush=True)

    rows = []
    for geom in ("flat", "ridge"):
        for slip in args.slips:
            # s=1.0 exactly is the no-slip baseline; pass None so it takes the
            # untouched no-slip code path (and is identical to bed_slip=1.0).
            bs = None if slip >= 1.0 else slip
            r = run_one(xp, geom, bs, args)
            r["slip"] = slip
            rows.append(r)
            print(f"  {geom:5s} slip={slip:4.2f}  melt_flux={r['melt_flux']:+.6e}  "
                  f"umax={r['umax']:7.3f}  u_tang={r['u_tang']:.4e}  "
                  f"div={r['div_rms']:.2e}  ({r['wall_s']:.1f}s)", flush=True)

    flat = {r["slip"]: r for r in rows if r["geom"] == "flat"}
    ridge = {r["slip"]: r for r in rows if r["geom"] == "ridge"}
    base_flat = flat[max(args.slips)]["melt_flux"]
    base_ridge = ridge[max(args.slips)]["melt_flux"]
    print("\n=== Navier bed-slip gate ===")
    print(f"{'slip':>6} {'flat_melt':>13} {'ridge_melt':>13} {'Nu/Nu_flat':>11} "
          f"{'ridge/ridge(noslip)':>20} {'u_tang(ridge)':>14}")
    for slip in sorted(args.slips, reverse=True):
        fm = flat[slip]["melt_flux"]
        rm = ridge[slip]["melt_flux"]
        nu_ratio = rm / fm if abs(fm) > 1e-30 else float("nan")
        rescue = rm / base_ridge if abs(base_ridge) > 1e-30 else float("nan")
        print(f"{slip:6.2f} {fm:+13.6e} {rm:+13.6e} {nu_ratio:11.4f} "
              f"{rescue:20.4f} {ridge[slip]['u_tang']:14.4e}")
    print(f"\nbaseline(no-slip) flat_melt={base_flat:+.6e}  "
          f"ridge_melt={base_ridge:+.6e}")
    print("Gate: if ridge/ridge(noslip) stays ~1 while u_tang grows, the bed "
          "stagnant layer is NOT the bottleneck (thermal sublayer is).")


if __name__ == "__main__":
    main()
