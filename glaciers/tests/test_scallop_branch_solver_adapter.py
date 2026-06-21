"""Tests for the P3 branch-test -> production-solver adapter
(``scallop_branch_solver_adapter.py``).

The adapter drives the normal Paper-3 solver (``ProbeFlow``) to a moving
Stefan-melt interface, assembles the raw record ``H[t, x]`` and feeds it to the
branch test ``scallop_field_test.harmonic_mode_rate``.  Checks:

  1. *plumbing* -- the synthetic self-check recovers a known damped, downstream
     migrating mode (the branch-test path the adapter reuses is intact);
  2. *record shape* -- ``solver_interface_record`` returns ``x`` (Nx,), ``t``
     (Nt,) and ``H`` (Nt, Nx) with matching shapes from a small but real solver
     run, exactly the contract ``harmonic_mode_rate`` expects;
  3. *bridge fidelity* -- the full-field branch test (``harmonic_mode_rate`` on
     ``H[t, x]``) and the single-fundamental-mode ``Z(t)`` fit (the committed
     ``scallop_moving_boundary_check`` method) agree to <1% on ``Re(s)`` and
     ``Im(s)`` over the SAME developed window: they are the same Fourier
     coefficient ``C[nw]/nx``, so the adapter faithfully reproduces the committed
     moving-boundary measurement rather than inventing a new one.

CPU only; small grids so the solver runs in a few seconds.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scallop_branch_solver_adapter as ad


def _small_kw():
    return dict(nw=6, U=3.0, afrac=0.10, St=1.0e-3, nx=48, ny=48, f_amp=0.4,
                seed=0, spinup=120, n_updates=40, steps_per_update=8)


def test_plumbing_selfcheck_recovers_known_mode():
    """The synthetic branch-test self-check the adapter ships must recover a
    known damped, downstream-migrating mode (no solver needed)."""
    sc = ad.selfcheck_synthetic()
    assert sc["recovered"] is True
    assert sc["downstream"] is True
    assert sc["Re_s_rec"] < 0.0


def test_solver_record_shape_matches_branch_test_contract():
    """solver_interface_record must return the (x, t, H[Nt,Nx]) contract that
    harmonic_mode_rate consumes."""
    rec = ad.solver_interface_record(**_small_kw())
    nx = _small_kw()["nx"]
    assert rec["x"].shape == (nx,)
    assert rec["H"].ndim == 2
    assert rec["n_frames"] >= 8
    assert rec["H"].shape == (rec["n_frames"], nx)
    assert rec["t"].shape == (rec["n_frames"],)
    assert np.all(np.isfinite(rec["H"]))


def test_branch_test_matches_committed_single_mode_fit():
    """End-to-end bridge: feeding the solver record into harmonic_mode_rate must
    reproduce the committed single-fundamental-mode Z(t) fit (same coefficient,
    same window) to <1% on Re(s) and Im(s)."""
    res = ad.branch_test_from_solver(**_small_kw())
    assert res["ok"] is True
    assert res["provenance"]["tracked_k_index"] == _small_kw()["nw"]

    b = res["branch"]
    assert np.isfinite(b["I"])
    assert isinstance(b["downstream"], bool)

    ag = res["agreement"]
    assert ag is not None
    assert ag["consistent"] is True
    assert ag["rel_err_Re"] < 0.01
    assert ag["rel_err_Im"] < 0.01


def test_auto_mode_finds_the_seeded_bin():
    """With auto-detect (field-recipe style), the dominant corrugation the branch
    test locks onto is the seeded mode nw."""
    kw = _small_kw()
    res = ad.branch_test_from_solver(track_seeded=False, **kw)
    assert res["ok"] is True
    assert res["branch"]["k_index"] == kw["nw"]
