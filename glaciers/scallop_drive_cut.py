r"""RESULT 25 companion -- the REGIME-MATCHED solver counterpart of the Bushuk
exp-1b field pin: a mid-run drive cut.

The raw-array field test (``scallop_bushuk_raw.py``) measured
``I_obs = O(0.5-6.5)`` (pre-registered pin 3.6, flux-dynamic 1.2) on a train
whose wavelength was formed at ``U = 1.00 m/s`` and then observed under
``U = 0.16 m/s`` -- a far-from-equilibrium, anchoring-mismatched state.  The
near-equilibrium solver band ``[0.33, 0.88]`` is NOT the like-for-like
comparison; this driver builds it:

1. seed a single corrugation mode and co-evolve interface + flow at ``U_hi``
   (the analogue of exp 1a's formed train, migrating under its own drive);
2. CUT the drive to ``r_cut * U_hi`` (Bushuk's ratio ``r_cut = 0.16``) at a
   known time, keeping the interface evolving under the Stefan feedback;
3. track the fundamental complex shape mode ``Z(t)`` POST-CUT and fit
   ``s_cut = d/dt log Z`` exactly as the field pin fits the data
   (``Re(s)`` from ``log|Z|``, ``Im(s)`` from ``arg(Z)``), giving
   ``I_cut = |Im|/(2 pi |Re|)``;
4. controls: the SAME protocol with no cut (``I_nocut``, the near-equilibrium
   number), and a window-matched fit (first ``~0.33`` phase cycles post-cut,
   the span the 55-min Bushuk record resolves).

Question answered: does the anchoring mismatch push the solver's ``I`` from the
near-equilibrium band toward the observed ``O(1)`` values, i.e. is the
raw-array overshoot the *expected* signature of a drive-cut train?

Backend-agnostic (``--gpu`` -> CuPy, Tesla P100-verified family); the committed
headline numbers come from the GPU batch (seeds x n_w x a/lam grid).  CPU
``--quick`` smoke: ~1 min.  Writes ``subglacial/drive_cut.json`` (or --out).

Usage:
    python glaciers/scallop_drive_cut.py --quick          # CPU smoke
    python glaciers/scallop_drive_cut.py --gpu            # full grid (P100)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_probe import ProbeFlow  # noqa: E402
from scallop_moving_boundary_check import fit_rate  # noqa: E402
from subglacial.candidate3_roughness_feedback import Candidate3Config  # noqa: E402

LX = 4.0 * 2.0 * np.pi
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(HERE, "subglacial", "drive_cut.json")

BUSHUK_RATIO = 0.16          # U: 1.00 -> 0.16 m/s at 495 min (their table 1)
BUSHUK_I_OBS = (1.2, 3.6)    # flux-dynamic .. pre-registered pin (RESULT 25)
BUSHUK_PHASE_SPAN = 2.09     # rad of modal phase the 55-min record resolves


def evolve_cut(nw, U_hi, afrac, St, *, r_cut, nx, ny, f_amp, seed, spinup,
               n_updates_pre, n_updates_post, steps_per_update, xp=np):
    """Seed -> co-evolve at U_hi -> cut to r_cut*U_hi -> keep co-evolving.

    Returns (t, Z, i_cut, clip_hit): the fundamental-mode complex series with
    the index of the first post-cut sample.
    """
    K = 2.0 * np.pi * nw / LX
    lam = LX / nw
    a0 = afrac * lam
    cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=f_amp,
                           Ri=0.0, seed=seed, St=St,
                           N_update=steps_per_update)
    s = ProbeFlow(cfg, U_drive=U_hi, xp=xp)
    s.set_single_mode(a0, nw)
    for _ in range(spinup):
        s.step()
    xg = np.arange(nx) * s.sp.dx
    ph = np.exp(-1j * K * xg)
    clip = float(s.Ly - 4.0 * s.sp.dy)

    ts, Zs = [], []
    clip_hit = False

    def one_update():
        nonlocal clip_hit
        acc = np.zeros(nx)
        cnt = np.zeros(nx)
        for _ in range(steps_per_update):
            s.step()
            m = s.melt_field()
            ok = np.isfinite(m)
            acc[ok] += m[ok]
            cnt[ok] += 1.0
        m_avg = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
        s.update_boundary(m_avg)
        y = s._to_host(s.y_ice_x)
        if y.max() >= 0.98 * clip:
            clip_hit = True
            return False
        ts.append(float(s.t))
        Zs.append(complex(np.sum((y - y.mean()) * ph) / nx))
        return True

    for _ in range(n_updates_pre):
        if not one_update():
            break
    i_cut = len(ts)
    if not clip_hit:
        s.U_drive = r_cut * U_hi                     # THE CUT
        for _ in range(n_updates_post):
            if not one_update():
                break
    return np.array(ts), np.array(Zs), i_cut, clip_hit


def fit_window(ts, Zs, *, skip_frac=0.15, phase_span=None):
    """fit_rate on a sub-window; optionally stop once |unwrapped phase| spans
    ``phase_span`` rad (the Bushuk-record-matched fit)."""
    if len(ts) < 6:
        return None
    if phase_span is not None:
        pha = np.unwrap(np.angle(Zs))
        i0 = int(skip_frac * len(ts))
        rel = np.abs(pha - pha[i0])
        idx = np.where(rel[i0:] >= phase_span)[0]
        i1 = (i0 + idx[0] + 1) if idx.size else len(ts)
        if i1 - i0 < 6:
            i1 = min(len(ts), i0 + 6)
        return fit_rate(ts[i0:i1], Zs[i0:i1], skip_frac=0.0)
    return fit_rate(ts, Zs, skip_frac=skip_frac)


def run_case(nw, U_hi, afrac, St, *, r_cut, nx, ny, f_amp, seed, spinup,
             n_pre, n_post, spu, xp):
    t0 = time.time()
    ts, Zs, i_cut, clip_hit = evolve_cut(
        nw, U_hi, afrac, St, r_cut=r_cut, nx=nx, ny=ny, f_amp=f_amp,
        seed=seed, spinup=spinup, n_updates_pre=n_pre, n_updates_post=n_post,
        steps_per_update=spu, xp=xp)
    out = dict(nw=nw, U_hi=U_hi, afrac=afrac, St=St, seed=seed,
               r_cut=r_cut, n_samples=len(ts), i_cut=i_cut,
               clip_hit=bool(clip_hit), wall_s=round(time.time() - t0, 1))
    pre = fit_window(ts[:i_cut], Zs[:i_cut], skip_frac=0.3)
    post = fit_window(ts[i_cut:], Zs[i_cut:], skip_frac=0.15)
    postw = fit_window(ts[i_cut:], Zs[i_cut:], skip_frac=0.15,
                       phase_span=BUSHUK_PHASE_SPAN)
    out["pre_cut"] = pre
    out["post_cut"] = post
    out["post_cut_windowed"] = postw
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--quick", action="store_true", help="CPU smoke test")
    ap.add_argument("--nws", type=int, nargs="+", default=[8, 12])
    ap.add_argument("--afracs", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--U", type=float, default=3.0)
    ap.add_argument("--rcut", type=float, default=BUSHUK_RATIO)
    ap.add_argument("--St", type=float, default=1.0e-3)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--famp", type=float, default=0.4)
    ap.add_argument("--spinup", type=int, default=1500)
    ap.add_argument("--npre", type=int, default=120)
    ap.add_argument("--npost", type=int, default=300)
    ap.add_argument("--spu", type=int, default=15)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    xp = np
    if args.gpu:
        import cupy
        xp = cupy
    if args.quick:
        args.nws, args.afracs, args.seeds = [12], [0.10], [0]
        args.nx = args.ny = 96
        args.spinup, args.npre, args.npost = 600, 40, 90

    grid = [(nw, af, sd) for nw in args.nws for af in args.afracs
            for sd in args.seeds]
    print(f"=== drive-cut regime match (r_cut={args.rcut}, U_hi={args.U}) ===")
    print(f"grid: nw={args.nws} afrac={args.afracs} seeds={args.seeds} "
          f"nx={args.nx} backend={'cupy' if xp is not np else 'numpy'}\n")
    cases = []
    for nw, af, sd in grid:
        c = run_case(nw, args.U, af, args.St, r_cut=args.rcut, nx=args.nx,
                     ny=args.ny, f_amp=args.famp, seed=sd, spinup=args.spinup,
                     n_pre=args.npre, n_post=args.npost, spu=args.spu, xp=xp)
        cases.append(c)
        pre, post, pw = c["pre_cut"], c["post_cut"], c["post_cut_windowed"]

        def fmt(f):
            if f is None:
                return "  (fit failed)"
            return (f"Re={f['Re_s']:+.3e} Im={f['Im_s']:+.3e} "
                    f"I={f['I_mb']:.3f} R2=({f['R2_amp']:.2f},{f['R2_phase']:.2f})")
        print(f"nw={nw:2d} a/lam={af:.2f} seed={sd}  [{c['wall_s']}s]"
              f"{' CLIP' if c['clip_hit'] else ''}")
        print(f"   pre-cut : {fmt(pre)}")
        print(f"   post-cut: {fmt(post)}")
        print(f"   windowed: {fmt(pw)}", flush=True)

    # verdict: does the cut push I toward the observed O(1) band?
    def collect(key):
        vals = [c[key]["I_mb"] for c in cases
                if c[key] and c[key]["R2_phase"] > 0.8
                and np.isfinite(c[key]["I_mb"])]
        return vals
    I_pre, I_post, I_w = collect("pre_cut"), collect("post_cut"), collect("post_cut_windowed")
    verdict = dict(
        I_pre=[float(min(I_pre)), float(max(I_pre))] if I_pre else None,
        I_post=[float(min(I_post)), float(max(I_post))] if I_post else None,
        I_windowed=[float(min(I_w)), float(max(I_w))] if I_w else None,
        I_obs_bushuk=list(BUSHUK_I_OBS),
        cut_raises_I=bool(I_pre and I_post
                          and np.median(I_post) > np.median(I_pre)),
        post_cut_reaches_observed=bool(
            I_post and max(I_post) >= BUSHUK_I_OBS[0]),
    )
    out = dict(description=__doc__.splitlines()[0],
               backend=("cupy" if xp is not np else "numpy"),
               params=dict(U_hi=args.U, r_cut=args.rcut, St=args.St,
                           nx=args.nx, ny=args.ny, f_amp=args.famp,
                           spinup=args.spinup, n_pre=args.npre,
                           n_post=args.npost, steps_per_update=args.spu),
               cases=cases, verdict=verdict)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n--- verdict ---")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
