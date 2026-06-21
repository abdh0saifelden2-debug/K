r"""§G.1 mechanism: why ``Nu/Nu_flat < 1`` is *distribution*-dominated.

Background (see FUTURE_WORK §G.1).  The cavity-mean Nusselt ratio is, by
definition, the area mean of the local normal interfacial flux normalised by the
flat-wall control,

    Nu/Nu_flat = <m_n,bump> / <m_n,flat>                         (exact)

The previously-proposed ``(1 + CV_delta^2)`` two-moment truncation is FALSIFIED
at scallop amplitudes (``CV_delta ~ O(1)``).  This module earns the mechanism
*without any truncation* by partitioning the interface into three physical
populations relative to the flat-wall conductance ``m_flat = <m_n,flat>``:

    reattachment (enhanced) :  m_n >= m_flat           (thin sublayer, delta_T < delta_flat)
    thickened   (suppressed):  0 < m_n <  m_flat       (thick sublayer, delta_T > delta_flat)
    reversed    (separated) :  m_n <= 0                (flux reversal, no physical sublayer)

Because the three populations tile the interface, their area-weighted flux
contributions sum *exactly* to the ratio:

    Nu/Nu_flat = C_reatt + C_thick + C_rev ,   C_p = (1/N) * sum_{p} m_n / m_flat

Writing each population's signed excess over its own area share, ``e_p = C_p - f_p``
(with ``f_reatt + f_thick + f_rev = 1``), gives the mechanism in one line:

    Nu/Nu_flat - 1 = e_reatt + e_thick + e_rev
                     \__ surplus >0 __/  \____ deficit <0 ____/

so ``Nu < 1`` iff the reattachment surplus is outweighed by the
thickened+reversed deficit.  This is exact, area-resolved, and reproduces the
ratio that the moment truncation misses.  The "fat tail in 1/delta_T" of the
reattachment patches is quantified by the skewness of the conductance and the
share of the harmonic mean carried by the top conductance decile.

Usage (Part-C config; matches REPORT_CANDIDATE3.md / §G.1):
    python scallop_g1_populations.py --nx 128 --ny 128 --nwaves 12 \
        --udrive 1.5 --famp 0.4 --spinup 3000 --measure 800 \
        --amps 0.05 0.10 0.15 0.20 0.30 0.50
    python scallop_g1_populations.py --fast            # quick CPU smoke test
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
        return cp, "cupy"
    try:
        import cupy as cp
        cp.zeros(1).sum()
        return cp, "cupy"
    except Exception:  # noqa: BLE001 - any import/device failure -> CPU
        return np, "numpy"


def _undefined_decomposition(n_valid, m_flat_mean):
    """All-NaN :func:`population_decomposition` result for a degenerate input --
    either no finite bump column (``n == 0``) or a non-finite flat reference
    (``m_flat_mean`` NaN/inf, e.g. every flat column non-finite).

    Without a finite flat-wall reference the area partition and every flux ratio
    are undefined, so they are returned as NaN (mirroring the ``n_pos == 0``
    guard in ``scallop_sublayer_probe._sublayer_stats``).  Returning early here
    is what keeps a NaN reference from silently misclassifying every positive
    column as "thickened" (an all-False ``v >= NaN`` would give ``f_reatt = 0``,
    ``f_thick ~ 1``) and from tripping a NaN-comparison ``RuntimeWarning`` on
    older NumPy.
    """
    nan = float("nan")
    return {
        "nu_ratio": nan, "nu_lt_1": False,
        "n_valid": int(n_valid), "m_flat_mean": m_flat_mean,
        "f_reatt": nan, "f_thick": nan, "f_rev": nan,
        "C_reatt": nan, "C_thick": nan, "C_rev": nan,
        "C_sum": nan,
        "e_reatt": nan, "e_thick": nan, "e_rev": nan,
        "surplus": nan, "deficit": nan,
        "deficit_beats_surplus": False,
        "cv2": nan, "cond_skew": nan,
        "top_decile_cond_share": nan,
        "pred_1pcv2": nan, "nu_ratio_pos": nan,
    }


def population_decomposition(m_bump, m_flat):
    """Partition the interface into reattachment / thickened / reversed zones and
    reconstruct ``Nu/Nu_flat`` exactly from their area-weighted flux shares.

    Parameters are the per-column time-mean *normal* interfacial flux of the bump
    and the flat control (``m_n(x)``); the unknown ``kappa*dT`` cancels in every
    ratio below.  Returns a dict of area fractions ``f_*``, flux contributions
    ``C_*`` (which sum to ``nu_ratio``), signed excesses ``e_*`` (which sum to
    ``nu_ratio - 1``), the harmonic-mean / tail diagnostics, and the (falsified)
    ``(1+CV^2)`` truncation for contrast.
    """
    mb = np.asarray(m_bump, dtype=float)
    mf = np.asarray(m_flat, dtype=float)
    okb = np.isfinite(mb)
    okf = np.isfinite(mf)
    n = int(okb.sum())
    nf = int(okf.sum())
    # ``<m_n,flat>``; guard the empty slice so an all-non-finite flat control
    # yields NaN cleanly (no empty-mean ``RuntimeWarning``) rather than calling
    # ``np.mean`` on an empty array; the degenerate guard just below then turns
    # that NaN into the all-NaN result.
    m_flat_mean = float(np.mean(mf[okf])) if nf else float("nan")
    # Degenerate inputs: no finite bump column (``n == 0``) or a non-positive /
    # non-finite flat reference (``m_flat_mean`` NaN/inf when every flat column
    # is non-finite, or ``<= 0`` -- which the partition invariant below forbids).
    # Either way the partition and all flux ratios are undefined, so return the
    # all-NaN result *before* the ``v >= m_flat_mean`` comparison below -- a NaN
    # reference would otherwise misclassify every positive column as thickened
    # (``f_reatt = 0``, ``f_thick ~ 1``) and could emit a NaN-comparison
    # ``RuntimeWarning`` on older NumPy; a zero reference would divide by 0 in
    # ``nu_ratio``.  (``frac()`` would also hit ``0.0/0.0`` when ``n == 0``.)
    # ``n_valid = n`` is 0 in the no-bump case by definition.
    if n == 0 or not np.isfinite(m_flat_mean) or m_flat_mean <= 0.0:
        return _undefined_decomposition(n, m_flat_mean)
    nu_ratio = float(np.mean(mb[okb])) / m_flat_mean  # exact Nu/Nu_flat

    v = mb[okb]                                      # valid bump columns
    # Partition assumes ``m_flat_mean > 0`` (always true physically: the
    # flat-wall mean flux is positive).  Under that invariant reatt/rev are
    # disjoint, so the three populations tile the interface and ``C_sum`` is
    # exactly ``nu_ratio``; if ``m_flat_mean <= 0`` the ``v == 0`` columns would
    # fall in both ``reatt`` and ``rev`` and double-count.
    reatt = v >= m_flat_mean                         # thin sublayer (enhanced)
    rev = v <= 0.0                                   # flux reversal (separated)
    thick = (~reatt) & (~rev)                        # 0 < m_n < m_flat (suppressed)

    def frac(mask):
        return float(np.count_nonzero(mask)) / float(n)

    def contrib(mask):
        # area-weighted flux share of this population in Nu/Nu_flat
        return float(np.sum(v[mask])) / (float(n) * m_flat_mean)

    f_reatt, f_thick, f_rev = frac(reatt), frac(thick), frac(rev)
    C_reatt, C_thick, C_rev = contrib(reatt), contrib(thick), contrib(rev)
    # signed excess over equal-flux (flat) baseline; sums to nu_ratio - 1
    e_reatt, e_thick, e_rev = C_reatt - f_reatt, C_thick - f_thick, C_rev - f_rev
    surplus = e_reatt                                # reattachment enhancement (>0)
    deficit = -(e_thick + e_rev)                     # thickened+reversed shortfall (>0)

    # --- delta_T distribution / fat-tail diagnostics (positive columns only) ---
    g = v[v > 0.0]                                   # 1/delta_T up to kappa*dT
    if g.size:
        inv = 1.0 / g                                # delta_T up to kappa*dT
        delta_mean = float(np.mean(inv))
        cv2 = float((np.std(inv) / delta_mean) ** 2)
        delta_flat = 1.0 / m_flat_mean
        # conductance skewness: fat positive tail = thin reattachment patches
        gm = float(np.mean(g)); gs = float(np.std(g))
        cond_skew = float(np.mean(((g - gm) / gs) ** 3)) if gs > 0 else float("nan")
        # share of the harmonic-mean numerator <1/delta_T> = <g> carried by the
        # top conductance decile (the thin-sublayer tail)
        thr = float(np.quantile(g, 0.9))
        top_decile_share = float(np.sum(g[g >= thr]) / np.sum(g))
        # the (1+CV^2) truncation that §G.1 reports as FALSIFIED, for contrast
        pred_1pcv2 = (delta_flat / delta_mean) * (1.0 + cv2)
        nu_ratio_pos = float(np.mean(g)) / m_flat_mean
    else:
        delta_mean = cv2 = cond_skew = top_decile_share = float("nan")
        pred_1pcv2 = nu_ratio_pos = float("nan")

    return {
        "nu_ratio": nu_ratio, "nu_lt_1": bool(nu_ratio < 1.0),
        "n_valid": n, "m_flat_mean": m_flat_mean,
        "f_reatt": f_reatt, "f_thick": f_thick, "f_rev": f_rev,
        "C_reatt": C_reatt, "C_thick": C_thick, "C_rev": C_rev,
        "C_sum": C_reatt + C_thick + C_rev,          # == nu_ratio (exactness check)
        "e_reatt": e_reatt, "e_thick": e_thick, "e_rev": e_rev,
        "surplus": surplus, "deficit": deficit,
        "deficit_beats_surplus": bool(deficit > surplus),
        "cv2": cv2, "cond_skew": cond_skew,
        "top_decile_cond_share": top_decile_share,
        "pred_1pcv2": pred_1pcv2, "nu_ratio_pos": nu_ratio_pos,
    }


def run(xp, nx=128, ny=128, n_waves=12, U_drive=1.5, f_amp=0.4,
        spinup=3000, measure=800, Ri=0.0, seed=0,
        amps=(0.05, 0.10, 0.15, 0.20, 0.30, 0.50)):
    Lx = 4.0 * 2.0 * np.pi
    lam = Lx / n_waves

    def cfg():
        return Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none",
                                f_amp=f_amp, Ri=Ri, seed=seed)

    t0 = time.time()
    _, _, m_flat, umax_f, _ = _run_norm(cfg(), 0.0, n_waves, U_drive,
                                        spinup, measure, xp)
    m_flat = np.asarray(m_flat)
    print(f"FLAT  Nu_flat={float(np.nanmean(m_flat)):.6e}  umax={umax_f:.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for r in amps:
        a = r * lam
        t0 = time.time()
        _, _, m_bump, umax, _ = _run_norm(cfg(), a, n_waves, U_drive,
                                          spinup, measure, xp)
        st = population_decomposition(np.asarray(m_bump), m_flat)
        st.update({"a_over_lam": float(r), "a": float(a), "umax": umax})
        rows.append(st)
        print(f"[a/λ={r:4.2f}] ({time.time()-t0:.0f}s) "
              f"Nu/Nu_flat={st['nu_ratio']:.4f}  "
              f"f(reatt/thick/rev)={st['f_reatt']:.3f}/{st['f_thick']:.3f}/{st['f_rev']:.3f}  "
              f"surplus={st['surplus']:.4f} deficit={st['deficit']:.4f}  "
              f"def>surp={st['deficit_beats_surplus']}", flush=True)

    return {"lambda": lam, "config": {
                "nx": nx, "ny": ny, "n_waves": n_waves, "U_drive": U_drive,
                "f_amp": f_amp, "spinup": spinup, "measure": measure,
                "Ri": Ri, "seed": seed, "amps": list(amps)},
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="quick CPU smoke test (small grid, short windows)")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nwaves", type=int, default=12)
    ap.add_argument("--udrive", type=float, default=1.5)
    ap.add_argument("--famp", type=float, default=0.4)
    ap.add_argument("--spinup", type=int, default=3000)
    ap.add_argument("--measure", type=int, default=800)
    ap.add_argument("--ri", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amps", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if args.fast:
        args.nx = args.ny = 64
        args.spinup, args.measure = 200, 120
        args.amps = [0.10, 0.30]

    xp, backend = get_backend(args.gpu)
    Lx = 4.0 * 2.0 * np.pi
    lam = Lx / args.nwaves
    print(f"backend={backend}  nx={args.nx} ny={args.ny}  n_waves={args.nwaves} "
          f"lambda={lam:.3f}  U_drive={args.udrive}  f_amp={args.famp}  "
          f"spinup={args.spinup} measure={args.measure}  amps={args.amps}",
          flush=True)

    res = run(xp, nx=args.nx, ny=args.ny, n_waves=args.nwaves,
              U_drive=args.udrive, f_amp=args.famp, spinup=args.spinup,
              measure=args.measure, Ri=args.ri, seed=args.seed, amps=args.amps)
    res["backend"] = backend

    print("\n=== §G.1 area-partition decomposition  Nu/Nu_flat = C_reatt + C_thick + C_rev ===")
    print(f"{'a/λ':>5} {'Nu/Nuf':>8} {'f_reatt':>8} {'f_thick':>8} {'f_rev':>7} "
          f"{'surplus':>8} {'deficit':>8} {'CV²':>6} {'skew':>6} {'top10%':>7} "
          f"{'pred1+CV²':>9} {'def>sur':>7}")
    for st in res["rows"]:
        print(f"{st['a_over_lam']:5.2f} {st['nu_ratio']:8.4f} {st['f_reatt']:8.3f} "
              f"{st['f_thick']:8.3f} {st['f_rev']:7.3f} {st['surplus']:8.4f} "
              f"{st['deficit']:8.4f} {st['cv2']:6.2f} {st['cond_skew']:6.2f} "
              f"{st['top_decile_cond_share']:7.3f} {st['pred_1pcv2']:9.4f} "
              f"{str(st['deficit_beats_surplus']):>7}")
    print("\nReading: the three populations tile the interface so C_reatt+C_thick+C_rev "
          "= Nu/Nu_flat exactly. Nu<1 iff the reattachment surplus (e_reatt) is "
          "outweighed by the thickened+reversed deficit. 'top10%' is the share of "
          "<1/δ_T> carried by the top conductance decile (the thin-sublayer fat tail); "
          "'pred1+CV²' is the falsified two-moment truncation, shown for contrast.")

    if args.out:
        out = _json_safe(res)
        try:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2, allow_nan=False)
            print(f"wrote {args.out}", flush=True)
        except OSError:
            print("G1_POP_JSON " + json.dumps(out, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
