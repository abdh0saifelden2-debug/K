r"""Adapter: run the Paper-3 *branch test* on the **production solver**.

The Paper-3 branch test is ``scallop_field_test.harmonic_mode_rate``: given a raw
interface record ``H[t, x]`` it fits the dominant corrugation mode
``h(x, t) = Re{ a_k(t) e^{i(k x + phi_k(t))} }``, reads the complex modal rate
``s = d/dt ln|a_k| + i d/dt phi_k = Re(s) + i Im(s)`` and forms the constant-free
index ``I = |Im(s)| / (2*pi*|Re(s)|) = tau * c_mig / lambda``.  Until now it was
exercised only on a *synthetic* damped-migrating train (``synth_train``) or awaited
Bushuk et al. (2019) external ``h(x, t)`` arrays.

**This module is the missing bridge.**  It takes the SAME input as the normal
Paper-3 solver (a :class:`Candidate3Config` + drive ``U`` + a seeded single mode),
runs that solver (:class:`scallop_probe.ProbeFlow`) with a MOVING Stefan-melt
interface -- the identical co-evolution loop ``scallop_moving_boundary_check`` uses
-- and records the FULL interface profile ``y_ice_x`` at every boundary update,
assembling exactly the ``H[t, x]`` record the branch test consumes.  It then feeds
that record straight into ``harmonic_mode_rate``.  So the branch test is now driven
by solver-generated data ("same input -> normal solver -> branch test fed from
it"), not by a synthetic stand-in.

It also cross-checks the full-field branch test against the established
single-fundamental-mode ``Z(t)`` fit (``scallop_moving_boundary_check.fit_rate``):
the two must agree on ``Re(s)`` and ``Im(s)`` to a few percent, which validates that
feeding the solver record into ``harmonic_mode_rate`` reproduces the committed
moving-boundary measurement.

Scope / honest caveat (unchanged from Paper 3): this is a *smoothing-only* solver,
so the physical branch it produces is the **damped, downstream-migrating** one
(``Re(s) < 0``).  The **growing** branch (``Re(s) > 0``) that real ice ripples have
needs a Roethlisberger interface-opening mechanism this solver deliberately lacks;
the adapter measures whatever branch the solver actually realises, the *same way*
the pin recipe would measure Bushuk's raw arrays.

CPU only; no external data, no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_probe import ProbeFlow, _json_safe  # noqa: E402
from scallop_field_test import harmonic_mode_rate, synth_train  # noqa: E402
from scallop_moving_boundary_check import LX, fit_rate, frozen_harmonics  # noqa: E402
from subglacial.candidate3_roughness_feedback import Candidate3Config  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. drive the normal solver -> full interface record H[t, x]
# --------------------------------------------------------------------------- #
def solver_interface_record(nw, U, afrac, St, *, nx, ny, f_amp, seed,
                            spinup, n_updates, steps_per_update, xp=np):
    r"""Run the production :class:`ProbeFlow` solver with a moving Stefan-melt
    interface and return the raw record the branch test consumes.

    Same input as ``scallop_moving_boundary_check.moving_mode`` -- but instead of
    collapsing each frame to the single fundamental mode ``Z(t)``, it stores the
    **whole** interface profile ``y_ice_x`` so the branch test can pick and track
    the dominant corrugation itself, exactly as it would on a field record.

    Returns
    -------
    dict with ``x`` (Nx,), ``t`` (Nt,), ``H`` (Nt, Nx), ``clip``, ``ymax`` (Nt,),
    ``n_frames``, ``stopped_at_ceiling`` (bool).
    """
    lam = LX / nw
    a0 = afrac * lam
    cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=f_amp,
                           Ri=0.0, seed=seed, St=St, N_update=steps_per_update)
    s = ProbeFlow(cfg, U_drive=U, xp=xp)
    s.set_single_mode(a0, nw)
    for _ in range(spinup):
        s.step()
    x = np.arange(nx) * s.sp.dx
    clip = float(s.Ly - 4.0 * s.sp.dy)
    ts, Hs, ymax = [], [], []
    stopped = False
    for _ in range(n_updates):
        acc = np.zeros(nx)
        cnt = np.zeros(nx)
        for _ in range(steps_per_update):
            s.step()
            m = s.melt_field()
            m = s._to_host(m) if not isinstance(m, np.ndarray) else m
            ok = np.isfinite(m)
            acc[ok] += m[ok]
            cnt[ok] += 1.0
        m_avg = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
        s.update_boundary(m_avg)
        y = np.asarray(s._to_host(s.y_ice_x), float)
        if y.max() >= 0.98 * clip:    # stop before the ceiling clip distorts the shape
            stopped = True
            break
        ts.append(float(s.t))
        Hs.append(y.copy())
        ymax.append(float(y.max()))
    return dict(x=x, t=np.array(ts), H=np.array(Hs), clip=clip,
                ymax=np.array(ymax), n_frames=len(ts),
                stopped_at_ceiling=stopped, K=2.0 * np.pi * nw / LX, lam=lam)


# --------------------------------------------------------------------------- #
# 2. run the branch test on the solver record (+ cross-check vs single mode)
# --------------------------------------------------------------------------- #
def branch_test_from_solver(nw, U, afrac, St, *, nx, ny, f_amp, seed,
                            spinup, n_updates, steps_per_update, track_seeded=True,
                            skip_frac=0.34, xp=np):
    r"""End-to-end: normal solver -> H[t, x] -> branch test (harmonic_mode_rate).

    ``track_seeded`` forces the branch test to track the seeded rFFT bin ``nw``
    (the mode the solver was driven at); set ``False`` to auto-detect the dominant
    mode the way the field/pin recipe would on an unknown record.  ``skip_frac`` of
    the leading frames is dropped as the post-seed adjustment transient (the same
    convention ``scallop_moving_boundary_check.fit_rate`` uses), so the eigen-rate
    is read off the developed window -- and so the branch test and the
    single-fundamental-mode ``Z(t)`` cross-check are fit over the *same* frames.

    Returns a dict combining the branch-test output, a single-fundamental-mode
    ``Z(t)`` cross-check (``fit_rate``), provenance, and an agreement summary.
    """
    rec = solver_interface_record(nw, U, afrac, St, nx=nx, ny=ny, f_amp=f_amp,
                                  seed=seed, spinup=spinup, n_updates=n_updates,
                                  steps_per_update=steps_per_update, xp=xp)
    if rec["n_frames"] < 8:
        return {"ok": False, "reason": "too few interface frames for a fit",
                "n_frames": rec["n_frames"], "stopped_at_ceiling": rec["stopped_at_ceiling"]}

    x = rec["x"]
    i0 = int(skip_frac * rec["n_frames"])
    t, H = rec["t"][i0:], rec["H"][i0:]                  # developed (post-transient) window
    k_index = int(nw) if track_seeded else None
    branch = harmonic_mode_rate(x, t, H, k_index=k_index)

    # cross-check: collapse the SAME (post-transient) frames to the single
    # fundamental mode Z(t) = C[nw]/nx and fit it the committed way (fit_rate with
    # no further skip, since the transient is already dropped).  Z is the *same*
    # Fourier coefficient harmonic_mode_rate tracks, so the two must now agree.
    K = rec["K"]
    ph = np.exp(-1j * K * x)
    Z = np.array([np.sum((h - h.mean()) * ph) / x.size for h in H])
    zfit = fit_rate(t, Z, skip_frac=0.0)

    agree = None
    if zfit is not None:
        re_b, re_z = branch["Re_s"], zfit["Re_s"]
        im_b, im_z = branch["dphi_dt"], zfit["Im_s"]
        denom_re = max(abs(re_z), 1e-30)
        denom_im = max(abs(im_z), 1e-30)
        agree = {
            "Re_s_branch": re_b, "Re_s_Zfit": re_z,
            "Im_s_branch": im_b, "Im_s_Zfit": im_z,
            "rel_err_Re": abs(re_b - re_z) / denom_re,
            "rel_err_Im": abs(im_b - im_z) / denom_im,
            "R2_amp_Z": zfit["R2_amp"], "R2_phase_Z": zfit["R2_phase"],
            "consistent": bool(abs(re_b - re_z) <= 0.05 * denom_re
                               and abs(im_b - im_z) <= 0.05 * denom_im),
        }

    return {
        "ok": True,
        "provenance": dict(nw=nw, U=U, afrac=afrac, St=St, nx=nx, ny=ny,
                           f_amp=f_amp, seed=seed, spinup=spinup,
                           n_updates=n_updates, steps_per_update=steps_per_update,
                           tracked_k_index=k_index, n_frames=rec["n_frames"],
                           stopped_at_ceiling=rec["stopped_at_ceiling"]),
        "branch": branch,
        "z_crosscheck": zfit,
        "agreement": agree,
    }


# --------------------------------------------------------------------------- #
# 3. plumbing self-check (no solver): branch test recovers a known mode
# --------------------------------------------------------------------------- #
def selfcheck_synthetic(Re_s=-1.0 / 7200.0, c_mig=1.833e-6, lam=0.13, seed=0):
    """Confirm the branch-test plumbing the adapter feeds is intact: a synthetic
    damped-migrating train with known (Re(s), c_mig, lam) is recovered to <3%."""
    x, t, H = synth_train(Re_s, c_mig, lam, noise=0.01, seed=seed)
    rec = harmonic_mode_rate(x, t, H)
    I_true = (1.0 / abs(Re_s)) * c_mig / lam
    return dict(
        Re_s_true=Re_s, Re_s_rec=rec["Re_s"],
        c_mig_true=c_mig, c_mig_rec=rec["c_mig"], lam_true=lam, lam_rec=rec["lam"],
        I_true=I_true, I_rec=rec["I"], downstream=rec["downstream"],
        recovered=bool(rec["downstream"]
                       and abs(rec["Re_s"] - Re_s) <= 0.02 * abs(Re_s)
                       and abs(rec["c_mig"] - c_mig) <= 0.02 * c_mig
                       and abs(rec["I"] - I_true) <= 0.03 * I_true),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nw", type=int, default=12)
    ap.add_argument("--U", type=float, default=3.0)
    ap.add_argument("--afrac", type=float, default=0.10)
    ap.add_argument("--St", type=float, default=1.0e-3)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--famp", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spinup", type=int, default=1500)
    ap.add_argument("--steps_per_update", type=int, default=15)
    ap.add_argument("--n_updates", type=int, default=340)
    ap.add_argument("--auto-mode", action="store_true",
                    help="auto-detect the dominant mode (field-recipe style) instead "
                         "of tracking the seeded bin nw")
    ap.add_argument("--also-frozen", action="store_true",
                    help="also report the frozen-probe migration harmonics for context")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "subglacial",
        "branch_solver_adapter.json"))
    args = ap.parse_args()

    print("=== P3 branch test driven by the production solver ===")
    sc = selfcheck_synthetic()
    print(f"[plumbing] synthetic branch-test self-check recovered={sc['recovered']} "
          f"(I_rec={sc['I_rec']:.3f} vs I_true={sc['I_true']:.3f})")

    print(f"[solver] nw={args.nw} U={args.U} a/lam={args.afrac} St={args.St} "
          f"nx={args.nx} ny={args.ny} -> moving interface -> branch test ...", flush=True)
    res = branch_test_from_solver(
        args.nw, args.U, args.afrac, args.St, nx=args.nx, ny=args.ny,
        f_amp=args.famp, seed=args.seed, spinup=args.spinup,
        n_updates=args.n_updates, steps_per_update=args.steps_per_update,
        track_seeded=not args.auto_mode)

    if not res["ok"]:
        print(f"  FAILED: {res['reason']} (n_frames={res.get('n_frames')})")
    else:
        b = res["branch"]
        ag = res["agreement"]
        print(f"  frames={res['provenance']['n_frames']}  tracked bin="
              f"{res['provenance']['tracked_k_index']}")
        print(f"  Re(s)={b['Re_s']:+.4e}  Im(s)={b['dphi_dt']:+.4e}  "
              f"lam={b['lam']:.3f}  c_mig={b['c_mig']:.3e}  downstream={b['downstream']}")
        print(f"  I (=|Im|/2pi|Re|=tau*c_mig/lam) = {b['I']:.3f}")
        if ag is not None:
            print(f"  cross-check vs single-mode Z(t): Re rel-err={ag['rel_err_Re']:.2%} "
                  f"Im rel-err={ag['rel_err_Im']:.2%} (R2_amp={ag['R2_amp_Z']:.3f}, "
                  f"R2_pha={ag['R2_phase_Z']:.3f}) consistent={ag['consistent']}")

    out = {"plumbing_selfcheck": sc, "result": res}
    if args.also_frozen and res["ok"]:
        fr = frozen_harmonics(args.nw, args.U, args.afrac, nx=args.nx, ny=args.ny,
                              f_amp=args.famp, seed=args.seed, spinup=args.spinup,
                              measure=400)
        out["frozen_probe"] = fr
        print(f"  [frozen probe] E_cos(migration)={fr['E_cos']:+.3e}  "
              f"I_cond(paper)={fr['I_cond']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(_json_safe(out), fh, indent=2, allow_nan=False, default=float)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
