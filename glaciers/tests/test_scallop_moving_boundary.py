"""Tests for scallop_moving_boundary_check.py (Paper 3 moving-boundary check).

The deterministic test pins the core measurement -- ``fit_rate`` must recover a
known complex rate ``s = Re + i Im`` (and the observable ``I=|Im|/(2pi|Re|)``)
from a synthetic damped-migrating mode ``Z(t)=A0 exp(s t)`` to high precision;
this is the estimator that produces the manuscript's ``Re(s)``, ``Im(s)`` and
``I_mb``.  Two short CPU smokes confirm the frozen-probe harmonics and the
co-evolving run return finite, structurally sane output.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scallop_moving_boundary_check as mb


def test_fit_rate_recovers_synthetic_complex_rate():
    # Z(t) = A0 exp((Re + i Im) t): log|Z| and arg(Z) are exactly linear in t,
    # so fit_rate must recover the rate, R^2=1, and I=|Im|/(2pi|Re|) exactly.
    Re_true, Im_true = -0.4, -0.5
    ts = np.linspace(0.0, 5.0, 80)
    Z = 0.1 * np.exp(1j * 0.3) * np.exp((Re_true + 1j * Im_true) * ts)
    f = mb.fit_rate(ts, Z, skip_frac=0.0)
    assert f is not None
    assert abs(f["Re_s"] - Re_true) < 1e-6
    assert abs(f["Im_s"] - Im_true) < 1e-6
    assert f["R2_amp"] > 0.999999 and f["R2_phase"] > 0.999999
    I_expect = abs(Im_true) / (2.0 * np.pi * abs(Re_true))
    assert abs(f["I_mb"] - I_expect) < 1e-6
    # a pure-decay (no migration) mode has I=0 and Im=0
    Z0 = 0.1 * np.exp(-0.3 * ts)
    f0 = mb.fit_rate(ts, Z0, skip_frac=0.0)
    assert abs(f0["Im_s"]) < 1e-6 and f0["I_mb"] < 1e-6


def test_frozen_harmonics_smoke():
    fr = mb.frozen_harmonics(12, 2.0, 0.10, nx=64, ny=48, f_amp=0.4, seed=0,
                             spinup=150, measure=80)
    for k in ("E_cos", "E_sin_total", "I_total", "I_cond"):
        assert np.isfinite(fr[k])
    # conduction carries no quadrature (migration) component -> E_cos_cond ~ 0
    assert abs(fr["E_cos_cond"]) < 1e-10


def test_moving_mode_smoke():
    ts, Zs, ymax, clip = mb.moving_mode(
        12, 2.0, 0.10, 2.0e-3, nx=64, ny=48, f_amp=0.4, seed=0,
        spinup=150, n_updates=20, steps_per_update=8)
    assert len(ts) >= 4 and len(Zs) == len(ts)
    assert np.all(np.asarray(ymax) < clip)        # never recorded past the clip
    f = mb.fit_rate(ts, Zs)
    assert f is not None and np.isfinite(f["Re_s"]) and np.isfinite(f["Im_s"])
    assert 0.0 < f["decay_factor"] < 2.0
