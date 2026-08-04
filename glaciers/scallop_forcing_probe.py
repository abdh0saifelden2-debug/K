r"""§G.2 test -- is the scalloped-amplitude response to *time-dependent* drive
monostable?

The §G.2 reduced model is a 1st-order autonomous scalar ODE
``a_dot = (alpha a^{1/2} - beta a) / rho L`` with a single stable fixed point
``a* = (alpha/beta)^2``.  Such an ODE makes four falsifiable qualitative
predictions for the *full* moving-boundary solver's seeded-mode amplitude
``a(t)`` when the drive ``U_drive`` is made time-dependent:

  step    : a(t) relaxes *monotonically* to the new a*(U) -- no overshoot.
  reverse : up-then-down returns to the original a* -- *no hysteresis*.
  pulse   : a short spike gives a rise then relaxation back, *no overshoot*.
  sine    : a(t) tracks U(t) with a phase lag, *no resonance / sub-harmonic*.

This is a *qualitative* falsification harness, not a quantitative ODE fit:
``alpha``/``beta`` are never measured here, so we test the monostable
*signatures*, not the value of a*.

Pure driver -- **no solver changes**.  It reuses ``ProbeFlow`` +
``update_boundary`` exactly as ``scallop_sweep.a_sat_run`` does; the only new
ingredient is that ``U_drive(t)`` is mutated on the running solver each step.
``ProbeFlow._forcing`` reads ``self.U_drive`` fresh every step, so this needs
no edit to the spectral code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_probe import ProbeFlow  # noqa: E402
from scallop_sweep import mode_amp  # noqa: E402
from subglacial.candidate3_roughness_feedback import Candidate3Config  # noqa: E402

LX = 4.0 * 2.0 * np.pi


def run_forcing(U_of_t, *, nx, ny, n_waves, a0, spinup, n_update_steps,
                Ri=0.0, seed=0, St=None, xp=np):
    """Spin up at U_of_t(0), then evolve the Stefan boundary while driving the
    fluid with the time-dependent U_of_t(step).  Returns the recorded
    (step, t, U, amplitude) trajectory, sampled once per boundary update."""
    cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.4,
                           Ri=Ri, seed=seed)
    if St is not None:
        cfg.St = St
    s = ProbeFlow(cfg, U_drive=float(U_of_t(0)), xp=xp)
    s.set_single_mode(a0, n_waves)
    for _ in range(spinup):
        s.step()
    traj = []
    acc = np.zeros(nx); cnt = np.zeros(nx)
    for k in range(n_update_steps):
        s.U_drive = float(U_of_t(k))           # <-- time-dependent forcing
        s.step()
        mi = s.melt_field()
        ok = np.isfinite(mi); acc[ok] += mi[ok]; cnt[ok] += 1.0
        if (k + 1) % cfg.N_update == 0:
            m_mean = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
            s.update_boundary(m=m_mean)
            acc[:] = 0.0; cnt[:] = 0.0
            yb = s._to_host(s.y_ice_x).astype(float)
            traj.append((int(k + 1), float(s.t), float(s.U_drive),
                         float(mode_amp(yb, n_waves, LX, s.sp.dx))))
    return {"a0": float(a0), "n_waves": int(n_waves), "St": float(cfg.St),
            "N_update": int(cfg.N_update), "amp_t": traj,
            "amp_final": traj[-1][3] if traj else float(a0)}


# --------------------------------------------------------------------------- #
# forcing protocols U_of_t(step)
# --------------------------------------------------------------------------- #
def step_protocol(u_lo, u_hi, n_update_steps, frac=0.5):
    k_sw = int(frac * n_update_steps)
    return lambda k: (u_lo if k < k_sw else u_hi)


def pulse_protocol(u_base, u_peak, n_update_steps, start=0.25, width=0.15):
    k0 = int(start * n_update_steps); k1 = int((start + width) * n_update_steps)
    return lambda k: (u_peak if k0 <= k < k1 else u_base)


def sine_protocol(u0, du, n_update_steps, n_cycles):
    omega = 2.0 * math.pi * n_cycles / max(n_update_steps, 1)
    return lambda k: (u0 + du * math.sin(omega * k))


def _monotone_after(amp, k_idx, steps):
    """Is amp monotone (within float noise) over the samples *after* index
    where step >= k_idx?  Returns (is_monotone, direction)."""
    seg = [a for (st, a) in zip(steps, amp) if st >= k_idx]
    if len(seg) < 3:
        return True, 0
    d = np.diff(seg)
    tol = 1e-9 + 1e-3 * (max(seg) - min(seg) + 1e-30)
    up = bool(np.all(d > -tol)); dn = bool(np.all(d < tol))
    return (up or dn), (1 if up else (-1 if dn else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nwaves", type=int, default=12)
    ap.add_argument("--a0frac", type=float, default=0.20,
                    help="seed amplitude as a fraction of lambda")
    ap.add_argument("--spinup", type=int, default=3000)
    ap.add_argument("--updates", type=int, default=600,
                    help="number of boundary updates (x N_update steps each)")
    ap.add_argument("--ulo", type=float, default=1.0)
    ap.add_argument("--uhi", type=float, default=2.0)
    ap.add_argument("--ncycles", type=float, default=3.0)
    ap.add_argument("--ncycles_steps", type=int, default=0,
                    help="if >0, total boundary updates for the sine run")
    ap.add_argument("--Ri", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--St", type=float, default=-1.0)
    ap.add_argument("--protocols", nargs="+",
                    default=["constant", "step_up", "step_down", "pulse", "sine"])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", type=str, default="forcing_results.json")
    args = ap.parse_args()

    xp = np
    if args.gpu:
        import cupy as cp  # noqa: F401
        xp = cp

    lam = LX / args.nwaves
    a0 = args.a0frac * lam
    n_up = args.updates
    St = None if args.St < 0 else args.St

    # run_forcing's n_update_steps is the raw step count; we want `updates`
    # boundary updates, so multiply by N_update (default 10).
    cfg_probe = Candidate3Config(nx=args.nx, ny=args.ny)
    n_update_steps = n_up * cfg_probe.N_update

    # The sine run may use a longer (or shorter) horizon than the other
    # protocols so that several drive cycles fit in: --ncycles_steps gives the
    # number of *boundary updates* for the sine run (default: same as the rest).
    n_up_sine = args.ncycles_steps if args.ncycles_steps > 0 else n_up
    n_update_steps_sine = n_up_sine * cfg_probe.N_update

    protos = {
        "constant": step_protocol(args.ulo, args.ulo, n_update_steps),
        "step_up": step_protocol(args.ulo, args.uhi, n_update_steps),
        "step_down": step_protocol(args.uhi, args.ulo, n_update_steps),
        "pulse": pulse_protocol(args.ulo, args.uhi, n_update_steps),
        "sine": sine_protocol(0.5 * (args.ulo + args.uhi),
                              0.5 * (args.uhi - args.ulo),
                              n_update_steps_sine, args.ncycles),
    }
    # per-protocol horizon (raw solver steps); only the sine run differs
    proto_steps = {name: n_update_steps for name in protos}
    proto_steps["sine"] = n_update_steps_sine

    out = {"meta": {"nx": args.nx, "ny": args.ny, "nwaves": args.nwaves,
                    "lambda": lam, "a0": a0, "a0frac": args.a0frac,
                    "spinup": args.spinup, "updates": n_up,
                    "n_update_steps": n_update_steps,
                    "updates_sine": n_up_sine,
                    "n_update_steps_sine": n_update_steps_sine, "ulo": args.ulo,
                    "uhi": args.uhi, "ncycles": args.ncycles,
                    "ncycles_steps": args.ncycles_steps,
                    "Ri": args.Ri, "seed": args.seed,
                    "St": St if St is not None else cfg_probe.St,
                    "gpu": bool(args.gpu)}, "runs": {}}
    print("META " + json.dumps(out["meta"]), flush=True)

    for name in args.protocols:
        if name not in protos:
            print(f"!! unknown protocol {name}", flush=True); continue
        t0 = time.time()
        r = run_forcing(protos[name], nx=args.nx, ny=args.ny,
                        n_waves=args.nwaves, a0=a0, spinup=args.spinup,
                        n_update_steps=proto_steps[name], Ri=args.Ri,
                        seed=args.seed, St=St, xp=xp)
        steps = [t[0] for t in r["amp_t"]]
        amps = [t[3] for t in r["amp_t"]]
        out["runs"][name] = r
        a_first = amps[0] if amps else a0
        a_last = amps[-1] if amps else a0
        print(f"[{name:9s}] ({time.time()-t0:.0f}s) a0={a0:.4f} "
              f"a_first={a_first:.4f} a_final={a_last:.4f} "
              f"n_samples={len(amps)}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
