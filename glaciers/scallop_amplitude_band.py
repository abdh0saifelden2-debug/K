r"""§D.5 critical amplitude band: separation onset and shedding onset.

Amplitude scan at fixed (optimal) wavelength ``n_waves=12``.  ``scallop_sweep``'s
``amp_scan`` already reports ``Nu/Nu_flat(a/λ)``; this probe adds the two
diagnostics the §D.5 hypothesis is actually about -- the *transitions* in the
flow state as the bump amplitude grows:

  * **separation (lower edge ``a_crit``).**  The mean drive is a steady ``+x``
    body force, so in attached flow the time-mean streamwise velocity is positive
    everywhere.  A coherent lee recirculation is the *only* source of time-mean
    ``u < 0``.  We therefore measure the **reverse-flow area fraction** of the
    time-mean field, ``rev_frac = area(fluid ∧ ⟨u⟩_t < 0) / area(fluid)``.  Below
    ``a_crit`` the bump is buried in the penalty/interface layer, no separation,
    ``rev_frac ≈ 0``; above it a lee bubble appears and ``rev_frac`` turns on.
    The turbulent forcing (``f_amp=0.4``) is left on; time-averaging removes the
    incoherent fluctuations and leaves the bump-locked recirculation.

  * **shedding (upper edge ``a_shed``).**  Whether steady recirculation gives way
    to unsteady vortex shedding cannot be read off the standard ``f_amp=0.4`` run
    (the stochastic force makes everything fluctuate).  *This solver has no
    deterministic laminar driven mode*: the steady ``U_drive`` body force is
    applied inside ``_forcing()`` and the whole forcing branch is gated behind
    ``cfg.f_amp > 0``, so ``f_amp=0`` leaves the fluid at rest.  We therefore run
    a **quasi-laminar** companion scan at a small ``f_amp`` (default ``0.02``):
    enough to sustain the mean current, little enough that the residual stochastic
    kinetic-energy jitter is a low, roughly amplitude-independent floor.  Over the
    window we record the total kinetic energy each step and report its coefficient
    of variation ``ke_cv = std(KE)/mean(KE)``; a rise of ``ke_cv`` well above the
    small-``a`` floor flags intrinsic unsteadiness.  Because the floor cannot be
    driven to zero in this solver, the shedding edge is reported honestly as
    *suggestive only* -- a clean ``a_shed`` needs a deterministic NS solver.

Both edges are *empirical* solver outputs.  This script earns the numbers; the
two diagnostic helpers (:func:`reverse_flow_fraction`, :func:`coeff_of_variation`)
are unit-tested on synthetic fields in ``tests/test_validation_synthetic.py``.

Usage:
    python scallop_amplitude_band.py --out band.json
    python scallop_amplitude_band.py --gpu --out band.json
    python scallop_amplitude_band.py --fast            # tiny CPU smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_probe import ProbeFlow, _json_safe, _nanmean  # noqa: E402
from scallop_sweep import melt_normal  # noqa: E402
from subglacial.candidate3_roughness_feedback import Candidate3Config  # noqa: E402


def reverse_flow_fraction(u_mean, fluid):
    """Area fraction of fluid cells whose *time-mean* streamwise velocity is
    negative.  ``u_mean`` and ``fluid`` are host arrays of identical shape;
    ``fluid`` is a boolean mask.  Returns NaN when there is no fluid."""
    f = np.asarray(fluid, dtype=bool)
    u = np.asarray(u_mean, dtype=float)
    n = int(f.sum())
    if n == 0:
        return float("nan")
    return float(np.count_nonzero(f & (u < 0.0))) / float(n)


def coeff_of_variation(series):
    """std/|mean| of a 1-D series (the relative unsteadiness).  Returns NaN for
    an empty series or a zero mean."""
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    m = float(np.mean(s))
    if m == 0.0:
        return float("nan")
    return float(np.std(s)) / abs(m)


def detrended_cv(series):
    """Relative fluctuation about a *linear* trend: std(series - linfit)/|mean|.
    This removes the slow spin-up/saturation drift of the driven flow so the
    metric reflects genuine oscillation (e.g. vortex shedding) rather than the
    flow still ramping up.  Returns NaN for a degenerate series."""
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 3:
        return float("nan")
    m = float(np.mean(s))
    if m == 0.0:
        return float("nan")
    t = np.arange(s.size, dtype=float)
    coeffs = np.polyfit(t, s, 1)
    resid = s - np.polyval(coeffs, t)
    return float(np.std(resid)) / abs(m)


def _scan_point(nx, ny, n_waves, U_drive, f_amp, spinup, measure, Ri, seed, a,
                xp):
    """Spin up a frozen single-mode cavity at amplitude ``a`` and measure, over
    the window: the time-mean reverse-flow fraction, the per-column normal melt
    (for Nu), and the kinetic-energy time series (for unsteadiness)."""
    cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=f_amp, Ri=Ri,
                           seed=seed)
    s = ProbeFlow(cfg, U_drive=U_drive, xp=xp)
    s.set_single_mode(a, n_waves)
    for _ in range(spinup):
        s.step()
    u_sum = xp.zeros_like(s.u)
    accn = np.zeros(nx)
    cntn = np.zeros(nx)
    ke = []
    m = max(measure, 1)
    for _ in range(m):
        s.step()
        u_sum = u_sum + s.u
        mn = melt_normal(s)
        ok = np.isfinite(mn)
        accn[ok] += mn[ok]
        cntn[ok] += 1.0
        ke.append(float(((s.u ** 2 + s.v ** 2) * s.fluid).sum() / s.fvol))
    u_mean = s._to_host(u_sum) / float(m)
    fluid_h = s._to_host(s.fluid).astype(bool)
    m_n = np.where(cntn > 0, accn / np.maximum(cntn, 1.0), np.nan)
    return {
        "rev_frac": reverse_flow_fraction(u_mean, fluid_h),
        "Nu_bump": _nanmean(m_n),
        "ke_cv": coeff_of_variation(ke),
        "ke_cv_detr": detrended_cv(ke),
        "umax": float(s.xp.abs(s.u).max()),
    }


def _flat_nu(nx, ny, n_waves, U_drive, f_amp, spinup, measure, Ri, seed, xp):
    cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=f_amp, Ri=Ri,
                           seed=seed)
    s = ProbeFlow(cfg, U_drive=U_drive, xp=xp)
    s.set_single_mode(0.0, n_waves)
    for _ in range(spinup):
        s.step()
    accn = np.zeros(nx)
    cntn = np.zeros(nx)
    for _ in range(max(measure, 1)):
        s.step()
        mn = melt_normal(s)
        ok = np.isfinite(mn)
        accn[ok] += mn[ok]
        cntn[ok] += 1.0
    m_n = np.where(cntn > 0, accn / np.maximum(cntn, 1.0), np.nan)
    return _nanmean(m_n)


def _first_onset(rows, key, thresh):
    """Smallest ``a_over_lam`` whose ``key`` first exceeds ``thresh`` and stays
    above it for the rest of the scan (so a single noisy point does not trip it).
    Returns None if never tripped."""
    n = len(rows)
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is None or not np.isfinite(v) or v <= thresh:
            continue
        # row i already passed the guard above; only the *rest* of the scan
        # needs to stay above thresh for this to be a sustained onset.
        if all((rows[j].get(key) is not None and np.isfinite(rows[j][key])
                and rows[j][key] > thresh) for j in range(i + 1, n)):
            return r["a_over_lam"]
    return None


def run(xp, nx=128, ny=128, n_waves=12, U_drive=1.5, spinup=3000, measure=800,
        Ri=0.0, seed=0, f_amp_lam=0.02,
        a_over_lam=(0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40,
                    0.50, 0.60),
        sep_thresh=0.005, shed_thresh=0.01):
    Lx = 4.0 * 2.0 * np.pi
    lam = Lx / n_waves
    out = {"config": {"nx": nx, "ny": ny, "n_waves": n_waves, "lambda": lam,
                      "U_drive": U_drive, "spinup": spinup, "measure": measure,
                      "Ri": Ri, "seed": seed, "a_over_lam": list(a_over_lam),
                      "f_amp_lam": f_amp_lam,
                      "sep_thresh": sep_thresh, "shed_thresh": shed_thresh}}

    # ---- forced scan (f_amp=0.4): separation a_crit + Nu/Nu_flat -------------
    print("=== forced scan (f_amp=0.4): separation + Nu ===", flush=True)
    nu_flat = _flat_nu(nx, ny, n_waves, U_drive, 0.4, spinup, measure, Ri, seed, xp)
    print(f"FLAT Nu_flat(normal)={nu_flat:.6e}", flush=True)
    forced = []
    for r in a_over_lam:
        t0 = time.time()
        d = _scan_point(nx, ny, n_waves, U_drive, 0.4, spinup, measure, Ri, seed,
                        r * lam, xp)
        d["a_over_lam"] = r
        d["a"] = r * lam
        # nu_flat is a strictly-positive flat-wall flux in practice; guard the
        # degenerate (all-invalid -> 0/NaN) control so it yields NaN (-> null via
        # _json_safe) rather than raising ZeroDivisionError.
        d["Nu_ratio"] = (d["Nu_bump"] / nu_flat
                         if np.isfinite(nu_flat) and nu_flat != 0.0
                         else float("nan"))
        forced.append(d)
        print(f"[a/λ={r:.2f}] ({time.time()-t0:.0f}s) rev_frac={d['rev_frac']:.4f} "
              f"Nu/Nu_flat={d['Nu_ratio']:.4f} umax={d['umax']:.3f}", flush=True)

    # ---- quasi-laminar scan (small f_amp): shedding a_shed -------------------
    print(f"=== quasi-laminar scan (f_amp={f_amp_lam}): shedding (KE coeff. of "
          "variation) ===", flush=True)
    laminar = []
    for r in a_over_lam:
        t0 = time.time()
        d = _scan_point(nx, ny, n_waves, U_drive, f_amp_lam, spinup, measure, Ri,
                        seed, r * lam, xp)
        d["a_over_lam"] = r
        d["a"] = r * lam
        laminar.append(d)
        print(f"[a/λ={r:.2f}] ({time.time()-t0:.0f}s) ke_cv_detr={d['ke_cv_detr']:.3e} "
              f"(raw {d['ke_cv']:.3e}) umax={d['umax']:.3f}", flush=True)

    out["nu_flat"] = nu_flat
    out["forced"] = forced
    out["laminar"] = laminar
    out["a_crit"] = _first_onset(forced, "rev_frac", sep_thresh)
    # detrended-CV floor from the smallest-amplitude quasi-laminar point (no
    # separation -> steady): shedding is flagged only well above this floor.
    ke_floor = laminar[0]["ke_cv_detr"] if laminar else float("nan")
    out["ke_cv_detr_floor"] = ke_floor
    shed_abs = shed_thresh
    if np.isfinite(ke_floor):
        shed_abs = max(shed_thresh, 3.0 * ke_floor)
    out["shed_thresh_effective"] = shed_abs
    out["a_shed"] = _first_onset(laminar, "ke_cv_detr", shed_abs)
    print(f"\n--> a_crit (separation onset, rev_frac>{sep_thresh}) = {out['a_crit']}",
          flush=True)
    print(f"--> ke_cv_detr floor (small a) = {ke_floor:.3e}; shedding flagged if "
          f"ke_cv_detr>{shed_abs:.3e}", flush=True)
    print(f"--> a_shed (suggestive; solver has no laminar mode) = {out['a_shed']}",
          flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--hires", action="store_true")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.gpu:
        import cupy as cp
        cp.zeros(1).sum()
        xp, backend = cp, "cupy(GPU)"
    else:
        xp, backend = np, "numpy(CPU)"

    kw = {}
    if args.fast:
        kw = dict(nx=64, ny=64, spinup=200, measure=120,
                  a_over_lam=(0.02, 0.10, 0.30))
    elif args.hires:
        kw = dict(nx=192, ny=192, spinup=4000, measure=1000)

    out = run(xp, **kw)
    out["backend"] = backend

    if args.out:
        safe = _json_safe(out)
        # Only OSError (permissions, disk full) is recoverable by echoing to
        # stdout: _json_safe guarantees finite-only data so allow_nan=False never
        # raises ValueError (catching it would re-raise uncaught in the fallback).
        try:
            with open(args.out, "w") as f:
                json.dump(safe, f, indent=2, allow_nan=False)
            print(f"wrote {args.out}", flush=True)
        except OSError:
            print("BAND_JSON " + json.dumps(safe, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
