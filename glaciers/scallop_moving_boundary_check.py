r"""Moving-boundary consistency check for the frozen-interface migration (Paper 3).

Referee (P3): the harmonic mode ``s = -beta + i*omega_mig`` is measured on a
FROZEN interface (``Re(s)<0`` by construction).  Does the migration survive when
the boundary actually MOVES?  This driver seeds a single-mode ice base, spins up
the flow, then co-evolves the interface under the Stefan melt feedback
(``update_boundary``), tracking the fundamental complex Fourier mode ``Z(t)`` of
the interface SHAPE.  A coherent damped-migrating eigenmode makes ``log|Z|`` and
``arg(Z)`` both linear in ``t``, so

    s_mb = d/dt log Z = Re(s_mb) + i Im(s_mb),     I_mb = |Im| / (2 pi |Re|)

is well defined.  We check, at fixed ``(nw, U, a/lam)``:

  * coherence -- ``R^2`` of the ``log|Z|`` and ``arg(Z)`` linear fits,
  * ``Re(s_mb) < 0`` (decay) -- the decay-only regime is PHYSICAL, not merely
    imposed by freezing (this solver only smooths; no Roethlisberger opening),
  * the migration sign vs. the frozen-probe quadrature harmonic ``E_cos``,
  * ``I_mb`` vs. the frozen-probe ratio, and its INVARIANCE across the Stefan
    number ``St`` (St cancels in the ratio, so a clean ``I_mb`` must not move
    with St) and across resolution.

Scope: this validates that freezing does not distort the migration and that
``I`` is interface-motion-invariant WITHIN the decaying branch.  The GROWING
branch (``Re(s)>0``) that real ice ripples have is physically absent from this
smoothing-only solver, so matching it is out of scope and remains a caveat.

CPU only; no external data, no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_probe import ProbeFlow, _run, _json_safe  # noqa: E402
from scallop_amplitude_harmonics import project  # noqa: E402
from subglacial.candidate3_roughness_feedback import Candidate3Config  # noqa: E402

LX = 4.0 * 2.0 * np.pi


def frozen_harmonics(nw, U, afrac, *, nx, ny, f_amp, seed, spinup, measure):
    """Frozen-interface flux harmonics at one (nw, U, a/lam).

    Returns the quadrature (migration) ``E_cos`` and the in-phase parts split
    into conduction (flow-off) and flow-excess, plus the two ratio conventions:
    ``I_total`` (total in-phase denominator = what a moving boundary feels) and
    ``I_cond`` (conduction-only denominator, the Paper-3 ``constant_free_ratio``).
    """
    K = 2.0 * np.pi * nw / LX
    lam = LX / nw
    a = afrac * lam
    cfg0 = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.0, Ri=0.0, seed=seed)
    _, m_cond, _, _ = _run(cfg0, a, nw, U_drive=0.0, spinup=spinup, xp=np)
    cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=f_amp, Ri=0.0, seed=seed)
    _, m_flow, _, _ = _run(cfg, a, nw, U_drive=U, spinup=spinup, measure=measure, xp=np)
    Ec_cond, Es_cond = project(m_cond, K, nx)
    Ec_flow, Es_flow = project(m_flow - m_cond, K, nx)
    Es_total = Es_cond + Es_flow
    I_total = abs(Ec_flow) / (2.0 * np.pi * abs(Es_total)) if Es_total else float("nan")
    I_cond = abs(Ec_flow) / (2.0 * np.pi * abs(Es_cond)) if Es_cond else float("nan")
    return {
        "K": float(K), "lam": float(lam), "a": float(a),
        "E_cos": float(Ec_flow), "E_sin_flow": float(Es_flow),
        "E_sin_cond": float(Es_cond), "E_sin_total": float(Es_total),
        "E_cos_cond": float(Ec_cond),
        "I_total": float(I_total), "I_cond": float(I_cond),
    }


def moving_mode(nw, U, afrac, St, *, nx, ny, f_amp, seed, spinup,
                n_updates, steps_per_update):
    """Co-evolve the seeded single mode; return time series of the fundamental
    complex interface-shape mode Z(t) and the interface ceiling drift."""
    K = 2.0 * np.pi * nw / LX
    lam = LX / nw
    a0 = afrac * lam
    cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=f_amp,
                           Ri=0.0, seed=seed, St=St, N_update=steps_per_update)
    s = ProbeFlow(cfg, U_drive=U, xp=np)
    s.set_single_mode(a0, nw)
    for _ in range(spinup):
        s.step()
    xg = np.arange(nx) * s.sp.dx
    ph = np.exp(-1j * K * xg)
    clip = float(s.Ly - 4.0 * s.sp.dy)
    ts, Zs, ymax = [], [], []
    for _ in range(n_updates):
        acc = np.zeros(nx); cnt = np.zeros(nx)
        for _ in range(steps_per_update):
            s.step()
            m = s.melt_field()
            ok = np.isfinite(m); acc[ok] += m[ok]; cnt[ok] += 1.0
        m_avg = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
        s.update_boundary(m_avg)
        y = s._to_host(s.y_ice_x)
        if y.max() >= 0.98 * clip:    # stop before the ceiling clip distorts the shape
            break
        ts.append(float(s.t)); Zs.append(complex(np.sum((y - y.mean()) * ph) / nx))
        ymax.append(float(y.max()))
    return np.array(ts), np.array(Zs), np.array(ymax), clip


def fit_rate(ts, Zs, skip_frac=0.34):
    """Linear fit of log|Z| and unwrapped arg(Z) vs t over the post-transient
    window; return rates, R^2, decay factor and phase span."""
    n = len(ts)
    i0 = int(skip_frac * n)
    t = ts[i0:]
    amp = np.abs(Zs[i0:])
    pha = np.unwrap(np.angle(Zs))[i0:]
    if len(t) < 4 or np.any(amp <= 0):
        return None

    def lin(x, y):
        A = np.polyfit(x, y, 1)
        yh = np.polyval(A, x)
        ss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - np.sum((y - yh) ** 2) / ss if ss > 0 else float("nan")
        return float(A[0]), float(r2)

    re, r2a = lin(t, np.log(amp))
    im, r2p = lin(t, pha)
    return {
        "Re_s": re, "Im_s": im,
        "R2_amp": r2a, "R2_phase": r2p,
        "I_mb": abs(im) / (2.0 * np.pi * abs(re)) if re else float("nan"),
        "decay_factor": float(np.abs(Zs)[-1] / np.abs(Zs)[i0]),
        "phase_span_rad": float(np.unwrap(np.angle(Zs))[-1] - np.unwrap(np.angle(Zs))[i0]),
        "n_fit": int(len(t)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nw", type=int, default=12)
    ap.add_argument("--U", type=float, default=3.0)
    ap.add_argument("--afrac", type=float, default=0.10)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--famp", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spinup", type=int, default=1500)
    ap.add_argument("--steps_per_update", type=int, default=15)
    ap.add_argument("--n_updates", type=int, default=340)
    ap.add_argument("--sts", type=float, nargs="+", default=[1.0e-3, 5.0e-4])
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "subglacial",
        "moving_boundary_check.json"))
    args = ap.parse_args()

    fr = frozen_harmonics(args.nw, args.U, args.afrac, nx=args.nx, ny=args.ny,
                          f_amp=args.famp, seed=args.seed, spinup=args.spinup,
                          measure=400)
    print("=== Moving-boundary consistency check (Paper 3) ===")
    print(f"nw={args.nw} U={args.U} a/lam={args.afrac} nx={args.nx} ny={args.ny}\n")
    print("FROZEN harmonics:")
    print(f"  E_cos (migration)   = {fr['E_cos']:+.4e}")
    print(f"  E_sin flow/cond/tot = {fr['E_sin_flow']:+.4e} / {fr['E_sin_cond']:+.4e}"
          f" / {fr['E_sin_total']:+.4e}")
    print(f"  E_cos_cond (~0?)    = {fr['E_cos_cond']:+.4e}")
    print(f"  I_total={fr['I_total']:.3f}  I_cond(paper)={fr['I_cond']:.3f}\n")

    print("MOVING boundary (seed single mode -> co-evolve -> track Z(t)):")
    print(f"{'St':>9} {'n':>4} {'Re(s)':>11} {'Im(s)':>11} {'R2_amp':>7} {'R2_pha':>7} "
          f"{'decay':>6} {'dphi':>7} {'Re*St':>10} {'Im*St':>10} {'I_mb':>6}")
    runs = []
    for St in args.sts:
        ts, Zs, ymax, clip = moving_mode(
            args.nw, args.U, args.afrac, St, nx=args.nx, ny=args.ny,
            f_amp=args.famp, seed=args.seed, spinup=args.spinup,
            n_updates=args.n_updates, steps_per_update=args.steps_per_update)
        f = fit_rate(ts, Zs)
        if f is None:
            print(f"{St:9.1e}  fit failed (too few points)")
            continue
        f.update({"St": St, "ymax_end": float(ymax[-1]), "clip": clip,
                  "n_pts": int(len(ts))})
        runs.append(f)
        print(f"{St:9.1e} {len(ts):4d} {f['Re_s']:+.4e} {f['Im_s']:+.4e} "
              f"{f['R2_amp']:7.3f} {f['R2_phase']:7.3f} {f['decay_factor']:6.3f} "
              f"{f['phase_span_rad']:+7.3f} {f['Re_s']*St:+.3e} {f['Im_s']*St:+.3e} "
              f"{f['I_mb']:6.3f}", flush=True)

    # verdict
    good = [r for r in runs if r["R2_amp"] > 0.9 and r["R2_phase"] > 0.9]
    I_mbs = [r["I_mb"] for r in good]
    I_spread = (max(I_mbs) - min(I_mbs)) if len(I_mbs) >= 2 else float("nan")
    all_decay = all(r["Re_s"] < 0 for r in runs)
    mig_sign_ok = all(np.sign(r["Im_s"]) == np.sign(runs[0]["Im_s"]) for r in runs)
    print("\n--- verdict ---")
    print(f"clean runs (R2>0.9 amp & phase): {len(good)}/{len(runs)}")
    print(f"Re(s)<0 (decay) every run     : {all_decay}")
    print(f"migration sign consistent     : {mig_sign_ok}")
    print(f"I_mb across St (clean runs)    : {[round(x,3) for x in I_mbs]}  "
          f"spread={I_spread:.3f}")
    print(f"frozen I_total={fr['I_total']:.3f}  I_cond={fr['I_cond']:.3f}")

    out = {"args": vars(args), "frozen": fr, "moving": runs,
           "I_mb_spread_clean": I_spread, "all_decay": all_decay,
           "migration_sign_consistent": bool(mig_sign_ok), "n_clean": len(good)}
    with open(args.out, "w") as fh:
        json.dump(_json_safe(out), fh, indent=2, allow_nan=False, default=float)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
