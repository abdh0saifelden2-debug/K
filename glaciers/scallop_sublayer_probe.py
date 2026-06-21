r"""Task A: test the corrected §1 thermal-sublayer decomposition against the fields.

Background (corrected §1, see the THEORY audit): there is NO geometric bound
``Nu_rough <= Nu_flat``.  Writing the local interfacial flux as
``q(x) = kappa*dT/dn = kappa*dT / delta_T(x)`` (``delta_T`` = local thermal
sublayer thickness, ``dT`` the bulk->melt drop, equal for bump and flat), the
cavity-mean Nusselt ratio decomposes exactly into a *mean-thickening* factor and
a *convexity (variance) boost*::

    Nu/Nu_flat = <q_bump>/<q_flat> = delta_flat * <1/delta_T>            (exact)
               = (delta_flat / <delta_T>) * (1 + CV_delta^2 + ...)       (2nd order)
                 \_____ thickening (<=1) _____/   \__ convexity (>=1) __/

So ``Nu/Nu_flat < 1`` is NOT a theorem -- it is the empirical statement that the
separated/stagnant lee cores thicken the *mean* sublayer faster than the thin
reattachment patches sharpen it, i.e. the (exact) condition

    harmonic_mean(delta_T) > delta_flat            <=>   Nu/Nu_flat < 1

or, to second order, ``<delta_T>/delta_flat > 1 + CV_delta^2``.

Because ``delta_T(x) = kappa*dT / m_n(x)`` and the flat sublayer is
``delta_flat = kappa*dT / <m_n,flat>``, the unknown constant ``kappa*dT`` cancels
in every dimensionless quantity below -- the whole test needs only the per-column
time-mean *normal* interfacial flux ``m_n(x)`` of the bump and the flat control.

This reuses the exact measurement machinery of ``scallop_sweep`` (true
local-normal gradient ``melt_normal``, same spinup/measure window), so the
``Nu/Nu_flat`` it reports here reproduces the report's Part-C numbers.

Usage (matches REPORT_CANDIDATE3.md Part C):
    python scallop_sublayer_probe.py --nx 128 --ny 128 --nwaves 12 \
        --udrive 1.5 --famp 0.4 --spinup 3000 --measure 800 \
        --amps 0.10 0.20 0.50
    python scallop_sublayer_probe.py --gpu ...        # Tesla P100 / CuPy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_probe import _json_safe  # noqa: E402
from scallop_sweep import _run_norm  # noqa: E402
from subglacial.candidate3_roughness_feedback import Candidate3Config  # noqa: E402


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


def _sublayer_stats(m_bump, m_flat):
    """Decompose Nu/Nu_flat into mean-thickening x convexity from the per-column
    normal-flux fields.  delta_T(x) ~ 1 / m_n(x); the kappa*dT constant cancels.

    Only columns with a *positive* normal flux define a physical sublayer
    (delta_T = kappa*dT/m_n > 0); columns with m_n <= 0 (flux reversal in deep
    separation) are reported but excluded from the delta_T statistics.
    """
    okb = np.isfinite(m_bump)
    okf = np.isfinite(m_flat)
    nu_bump = float(np.mean(m_bump[okb]))          # <m_n,bump>  (== report Nu_bump)
    nu_flat = float(np.mean(m_flat[okf]))          # <m_n,flat>  (== report Nu_flat)
    nu_ratio = nu_bump / nu_flat                   # exact Nu/Nu_flat

    posb = okb & (m_bump > 0.0)
    g = m_bump[posb]                               # 1/delta_T, up to kappa*dT
    n_pos, n_tot = int(posb.sum()), int(okb.sum())
    if n_pos == 0:
        # No positive-flux column defines a physical sublayer, so every
        # delta_T-derived quantity is undefined.  Return NaN explicitly instead
        # of letting an empty-array mean emit a RuntimeWarning and silently
        # propagate NaN through the stats.
        nan = float("nan")
        return {
            "nu_bump": nu_bump, "nu_flat": nu_flat, "nu_ratio": nu_ratio,
            "nu_ratio_pos": nan,
            "delta_mean": nan, "delta_flat": 1.0 / nu_flat, "cv2": nan,
            "thicken": nan, "convex": nan,
            "pred_ratio": nan, "harm_ratio": nan,
            "thicken_beats_convex": False,
            "nu_lt_1": bool(nu_ratio < 1.0),
            "n_pos": n_pos, "n_tot": n_tot,
        }
    # Nu ratio restricted to the positive-flux columns the delta_T statistics
    # use; equals nu_ratio only when every finite column is positive
    # (n_pos == n_tot).  This is the quantity the harmonic-mean identity and the
    # (1+CV^2) truncation below actually approximate -- NOT the full nu_ratio.
    nu_ratio_pos = float(np.mean(g)) / nu_flat

    inv = 1.0 / g                                  # delta_T, up to kappa*dT
    delta_mean = float(np.mean(inv))               # <delta_T>
    delta_std = float(np.std(inv))
    cv2 = (delta_std / delta_mean) ** 2            # CV_delta^2 (dimensionless)
    harm = 1.0 / float(np.mean(g))                 # harmonic_mean(delta_T) = kdT/<m_n>
    delta_flat = 1.0 / nu_flat                     # kappa*dT / <m_n,flat>

    thicken = delta_mean / delta_flat              # <delta_T>/delta_flat  (>1 suppresses)
    convex = 1.0 + cv2                             # variance boost (>1 enhances)
    pred_ratio = (delta_flat / delta_mean) * convex  # 2nd-order nu_ratio_pos
    harm_ratio = delta_flat / harm                 # exact via harmonic mean (==nu_ratio_pos)

    return {
        "nu_bump": nu_bump, "nu_flat": nu_flat, "nu_ratio": nu_ratio,
        "nu_ratio_pos": nu_ratio_pos,
        "delta_mean": delta_mean, "delta_flat": delta_flat, "cv2": cv2,
        "thicken": thicken, "convex": convex,
        "pred_ratio": pred_ratio, "harm_ratio": harm_ratio,
        "thicken_beats_convex": bool(thicken > convex),
        "nu_lt_1": bool(nu_ratio < 1.0),
        "n_pos": n_pos, "n_tot": n_tot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nwaves", type=int, default=12)
    ap.add_argument("--udrive", type=float, default=1.5)
    ap.add_argument("--famp", type=float, default=0.4)
    ap.add_argument("--spinup", type=int, default=3000)
    ap.add_argument("--measure", type=int, default=800)
    ap.add_argument("--ri", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amps", type=float, nargs="+", default=[0.10, 0.20, 0.50])
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    xp, backend = get_backend(args.gpu)
    Lx = 4.0 * 2.0 * np.pi
    lam = Lx / args.nwaves
    print(f"backend={backend}  nx={args.nx} ny={args.ny}  n_waves={args.nwaves} "
          f"lambda={lam:.3f}  U_drive={args.udrive}  f_amp={args.famp}  "
          f"spinup={args.spinup} measure={args.measure}  amps={args.amps}",
          flush=True)

    def cfg():
        return Candidate3Config(nx=args.nx, ny=args.ny, A=4.0, sgs="none",
                                f_amp=args.famp, Ri=args.ri, seed=args.seed)

    # flat-wall control, flow ON -> m_n,flat(x)
    t0 = time.time()
    _, _, m_flat, umax_f, _ = _run_norm(cfg(), 0.0, args.nwaves, args.udrive,
                                        args.spinup, args.measure, xp)
    print(f"FLAT  Nu_flat={float(np.nanmean(m_flat)):.6e}  umax={umax_f:.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for r in args.amps:
        a = r * lam
        t0 = time.time()
        _, _, m_bump, umax, _ = _run_norm(cfg(), a, args.nwaves, args.udrive,
                                          args.spinup, args.measure, xp)
        st = _sublayer_stats(np.asarray(m_bump), np.asarray(m_flat))
        st.update({"a_over_lam": r, "a": a, "umax": umax})
        rows.append(st)
        print(f"[a/λ={r:4.2f}] ({time.time()-t0:.0f}s) "
              f"Nu/Nu_flat={st['nu_ratio']:.4f}  "
              f"<δ>/δ_flat={st['thicken']:.4f}  1+CV²={st['convex']:.4f}  "
              f"pred Nu/Nu_flat={st['pred_ratio']:.4f}  "
              f"thicken>convex={st['thicken_beats_convex']}  "
              f"pos_cols={st['n_pos']}/{st['n_tot']}", flush=True)

    print("\n=== §1 thermal-sublayer decomposition  Nu/Nu_flat = (δ_flat/<δ>)(1+CV²) ===")
    print(f"{'a/λ':>5} {'Nu/Nu_flat':>11} {'Nu_pos':>8} {'<δ>/δ_flat':>11} {'1+CV²':>8} "
          f"{'pred':>8} {'|pred-Nu_pos|':>13} {'thick>conv':>10} {'Nu<1':>6}")
    for st in rows:
        print(f"{st['a_over_lam']:5.2f} {st['nu_ratio']:11.4f} {st['nu_ratio_pos']:8.4f} "
              f"{st['thicken']:11.4f} {st['convex']:8.4f} {st['pred_ratio']:8.4f} "
              f"{abs(st['pred_ratio']-st['nu_ratio_pos']):13.4f} "
              f"{str(st['thicken_beats_convex']):>10} {str(st['nu_lt_1']):>6}")
    print("\nReading: Nu<1 iff <δ>/δ_flat > 1+CV² (mean-thickening beats the "
          "convexity/variance boost). 'pred' is the 2nd-order truncation of the "
          "positive-column ratio Nu_pos; its gap to Nu_pos is the higher-moment "
          "remainder. Nu_pos differs from the full Nu/Nu_flat only via reversed-flux "
          "columns (n_pos<n_tot).")

    if args.out:
        # Sanitise NaN/Inf to null and serialise with allow_nan=False so the
        # n_pos==0 early return (and any degenerate run) writes strict RFC 8259
        # JSON rather than the non-standard NaN/Infinity tokens, matching
        # scallop_sweep.py.
        out = _json_safe({"backend": backend, "lambda": lam, "args": vars(args),
                          "rows": rows})
        # Only OSError (permissions, disk full) is recoverable here by echoing
        # the JSON to stdout: _json_safe already guarantees finite-only data, so
        # allow_nan=False never raises ValueError. Catching ValueError too would
        # be a trap -- the fallback json.dumps uses the same data and
        # allow_nan=False, so it would re-raise that very ValueError uncaught.
        try:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2, allow_nan=False)
            print(f"wrote {args.out}", flush=True)
        except OSError:
            print("SUBLAYER_JSON " + json.dumps(out, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
