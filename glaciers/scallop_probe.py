r"""Staged go/no-go probe for the *corrected* Candidate 3 (scallop / melting
instability), per the co-thinker's reasoning.

Question: with a single sinusoidal perturbation of the (frozen) ice base, does a
resolved mean current + turbulence create a *lee-side interfacial heat-flux
enhancement* that escapes the conduction limit of the penalised wall?

Decision rule:
  GO     -> some closure shows a coherent local enhancement  max R(x) >~ 1.10
            with the excess flux organised by the bump (not white noise).
  NO-GO  -> R(x) ~ 1 everywhere (flux conduction-pinned) -> the penalised wall is
            the hard boundary for *all* wall-driven emergent phenomena.

R(x) = m_flow(x) / m_cond(x), where m_cond is the SAME geometry run with the flow
OFF (empirical conduction baseline, same FD stencil as the flow case).

The stock Candidate-3 forcing is zero-mean isotropic, so there is no lee. We add
a steady x body force (U_drive) on the fluid to set up a mean current over the
bumps; a little ring turbulence (f_amp) trips separation.
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings

import numpy as np


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN / +-Inf) with ``None`` so the
    result serialises to valid (strict) JSON instead of the non-standard
    ``NaN`` / ``Infinity`` tokens, and never raises under ``allow_nan=False``.

    A degenerate run (e.g. ``Nu_flat`` ~ 0 making ``Nu_ratio`` infinite) would
    otherwise emit unparseable JSON or crash the serialiser.

    NumPy scalar/array types are coerced to their native Python equivalents:
    on NumPy 2.x ``np.float32`` is not a subclass of ``float`` and ``np.int64``
    is not a subclass of ``int``, so without this they would pass through
    unchanged and make ``json.dump`` raise an (uncaught) ``TypeError``. Dict
    keys are coerced via :func:`_json_safe_key` (which keeps a numpy-typed key
    such as ``np.int64`` from reaching ``json.dump`` while never collapsing a
    non-finite float key to ``None``)."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {_json_safe_key(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    # Unrecognised types pass through unchanged *by design*: the pipeline only
    # produces dicts/lists/numpy types/primitives (all handled above), and any
    # genuinely unsupported object should raise a loud ``TypeError`` in
    # ``json.dump`` rather than be silently coerced here.
    return obj


def _json_safe_key(k):
    """Coerce a dict *key* for strict-JSON serialisation.

    numpy-typed keys (e.g. ``np.int64``, which on NumPy 2.x is not an ``int``
    subclass) are mapped to their native Python equivalents so ``json.dump``
    does not raise ``TypeError``. A non-finite float key (``nan`` / ``+-inf``)
    has no valid strict-JSON form: routing it through :func:`_json_safe` like a
    value would turn every such key into ``None``, silently collapsing distinct
    non-finite keys into a single ``"null"`` key and losing data. Stringify it
    instead so each stays a distinct, lossless key.

    Limitation: distinct non-finite keys are separated only as far as ``repr``
    distinguishes them, so ``nan`` / ``+inf`` / ``-inf`` (the real-world case)
    stay distinct, but multiple *different* NaN objects all stringify to
    ``'nan'`` and would still collapse onto one key. Such dicts are vanishingly
    rare in practice."""
    safe = _json_safe(k)
    if safe is None and k is not None:
        return repr(k)
    return safe


def _nanmean(a):
    """``float(np.nanmean(a))`` that returns ``nan`` for an all-NaN/empty input
    without emitting the noisy ``RuntimeWarning: Mean of empty slice``. The NaN
    result is what callers want for a degenerate run; only the warning is
    suppressed (and only that specific one)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice",
                                category=RuntimeWarning)
        return float(np.nanmean(a))


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subglacial.candidate3_roughness_feedback import (  # noqa: E402
    Candidate3Config,
    RoughFeedbackFlow,
)


class ProbeFlow(RoughFeedbackFlow):
    """Candidate-3 cavity with (a) a single-mode frozen ice base and (b) an
    optional steady mean x-drive on the fluid."""

    def __init__(self, cfg, U_drive=0.0, xp=np):
        super().__init__(cfg, xp=xp)
        self.U_drive = float(U_drive)

    def set_single_mode(self, a, n_waves):
        """y_ice(x) = y_ice_mean + a sin(2 pi n_waves x / Lx)."""
        xp = self.xp
        x = np.arange(self.cfg.nx) * self.sp.dx
        y = self.cfg.y_ice_mean + a * np.sin(2.0 * np.pi * n_waves * x / self.Lx)
        self.y_ice_x = xp.asarray(y)
        self._rebuild_geometry(seed_theta=True)
        return y

    def _forcing(self):
        fx, fy = super()._forcing()
        if self.U_drive != 0.0:
            fx = fx + self.U_drive * self.fluid
        return fx, fy


def _run(cfg, a, n_waves, U_drive, spinup, measure=0, xp=np):
    """Spin up, then time-average the per-column melt over `measure` steps to
    separate the bump-locked (coherent) flux from turbulent fluctuations."""
    s = ProbeFlow(cfg, U_drive=U_drive, xp=xp)
    yb = s.set_single_mode(a, n_waves)
    for _ in range(spinup):
        s.step()
    if measure <= 0:
        m = s.melt_field()
    else:
        acc = np.zeros(cfg.nx)
        cnt = np.zeros(cfg.nx)
        for _ in range(measure):
            s.step()
            mi = s.melt_field()
            ok = np.isfinite(mi)
            acc[ok] += mi[ok]
            cnt[ok] += 1.0
        m = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
    umax = float(s.xp.abs(s.u).max())
    umean = float((s.u * s.fluid).sum() / s.fluid.sum())
    return yb, m, umax, umean


def _enh_stats(yb, m_flow, m_cond, dx):
    """Return R(x) stats + the bump-phase-locked coherent enhancement.

    corr_excess_slope: correlation of the time-mean flux excess with the bump
    slope (advective lee signature). coh_amp: amplitude of the excess projected
    onto the conduction shape variation -- the part of the enhancement that is
    organised by the geometry rather than residual turbulent scatter."""
    ok = np.isfinite(m_flow) & np.isfinite(m_cond) & (m_cond > 0)
    R = np.full(yb.size, np.nan)
    R[ok] = m_flow[ok] / m_cond[ok]
    Rok = R[np.isfinite(R)]
    if Rok.size:
        R_mean, R_max = float(np.mean(Rok)), float(np.max(Rok))
        R_min, R_std = float(np.min(Rok)), float(np.std(Rok))
    else:
        # zero valid columns (e.g. fluid domain entirely penalised): report NaN
        # stats rather than letting np.max/np.min crash on an empty array.
        R_mean = R_max = R_min = R_std = float("nan")
    slope = np.gradient(yb, dx)
    excess = np.where(ok, m_flow - m_cond, np.nan)
    good = np.isfinite(excess) & np.isfinite(slope)
    if good.sum() > 4 and np.std(excess[good]) > 1e-30 and np.std(slope[good]) > 1e-30:
        corr_slope = float(np.corrcoef(excess[good], slope[good])[0, 1])
    else:
        corr_slope = 0.0
    return {
        "R_mean": R_mean, "R_max": R_max,
        "R_min": R_min, "R_std": R_std,
        "corr_excess_slope": corr_slope,
        "m_cond_mean": _nanmean(m_cond),
        "m_flow_mean": _nanmean(m_flow),
    }


def probe(nx=128, ny=128, a=None, n_waves=12, U_drive=0.8, f_amp=0.4,
          spinup=3000, measure=600, Ri=0.0, seed=0, xp=np):
    Lx = 4.0 * 2.0 * np.pi
    dx, dy = Lx / nx, (2.0 * np.pi) / ny
    lam = Lx / n_waves
    if a is None:
        a = 0.1 * lam                         # enforce a/lambda = 0.1
    interface_w = 1.5 * dy                     # default cfg.interface = 1.5
    meta = {"nx": nx, "ny": ny, "dx": dx, "dy": dy, "lambda": lam,
            "lambda_over_dx": lam / dx, "a": a, "a_over_lambda": a / lam,
            "a_over_dy": a / dy, "interface_w": interface_w,
            "a_over_interface": a / interface_w, "U_drive": U_drive,
            "f_amp": f_amp, "spinup": spinup, "measure": measure, "Ri": Ri}

    # conduction baseline on the SAME bumpy geometry, flow OFF
    cfg0 = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.0, Ri=Ri, seed=seed)
    yb, m_cond, _, _ = _run(cfg0, a, n_waves, U_drive=0.0, spinup=spinup, xp=xp)

    # FLAT-WALL CONTROL: a=0, with drive+turbulence and its own conduction base.
    cfgf0 = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.0, Ri=Ri, seed=seed)
    ybf, mf_cond, _, _ = _run(cfgf0, 0.0, n_waves, U_drive=0.0, spinup=spinup, xp=xp)
    cfgf = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=f_amp, Ri=Ri, seed=seed)
    _, mf_flow, umax_f, umean_f = _run(cfgf, 0.0, n_waves, U_drive=U_drive,
                                       spinup=spinup, measure=measure, xp=xp)
    flat = _enh_stats(ybf, mf_flow, mf_cond, dx)
    flat["umax"] = umax_f
    flat["umean"] = umean_f

    results = {"_flat_control": flat}
    for closure in ("none", "smagorinsky", "backscatter"):
        cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs=closure, f_amp=f_amp,
                               Ri=Ri, seed=seed)
        _, m_flow, umax, umean = _run(cfg, a, n_waves, U_drive=U_drive,
                                      spinup=spinup, measure=measure, xp=xp)
        st = _enh_stats(yb, m_flow, m_cond, dx)
        st["umax"] = umax
        st["umean"] = umean
        results[closure] = st
    return meta, results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--a", type=float, default=-1.0)  # <0 => a = 0.1*lambda
    p.add_argument("--nwaves", type=int, default=12)
    p.add_argument("--udrive", type=float, default=0.15)
    p.add_argument("--famp", type=float, default=0.15)
    p.add_argument("--spinup", type=int, default=3000)
    p.add_argument("--measure", type=int, default=600)
    p.add_argument("--Ri", type=float, default=0.0)
    args = p.parse_args()

    if args.gpu:
        import cupy as cp
        xp = cp
    else:
        xp = np

    a = None if args.a < 0 else args.a
    meta, results = probe(nx=args.nx, ny=args.ny, a=a, n_waves=args.nwaves,
                          U_drive=args.udrive, f_amp=args.famp, spinup=args.spinup,
                          measure=args.measure, Ri=args.Ri, xp=xp)
    print("META " + json.dumps(_json_safe(meta), allow_nan=False))
    for k, v in results.items():
        print(f"RESULT {k} " + json.dumps(_json_safe(v), allow_nan=False))
