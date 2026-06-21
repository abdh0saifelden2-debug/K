"""Tests for the clock-mismatch (CMN) correction solver demo (``cmn_solver_demo``)
— the §H.3 model test of the §G.5 correction term.

Two layers, both CPU-only, fast and DNS-free:

  1. *live* small-grid solves verify the four structural claims directly: the
     §G.5 correction (a) cuts the transient error well below the naive frozen-
     clock K-theory solve, (b) is identically zero for steady turbulence
     (``d_tK=0``), (c) has the error-reducing ``+tau_c`` sign (wrong sign is
     worse), and (d) the naive error scales as ``tau_c`` while the corrected one
     scales as ``tau_c^2`` (the correction removes the leading-order term).

  2. the committed production artifact ``figures/60_cmn_solver_demo.json`` is
     checked for the same result at production resolution, and the verdict string
     records the §H.3 forecast as VERIFIED in-solver.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cmn_solver_demo as cmn  # noqa: E402

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures", "60_cmn_solver_demo.json")


def test_correction_reduces_transient_error_and_sign_control():
    """The §G.5 CMN term (CMN=+tau_c) cuts the transient error far below the
    naive frozen-clock solve, and the +tau_c sign is the error-reducing one —
    CMN=-tau_c is worse than doing nothing."""
    tr = cmn.error_trace(n=48, tau_c=0.05, dt=1e-3, n_snap=30)
    e_naive = tr["e_naive"].max()
    e_corr = tr["e_corrected"].max()
    e_wrong = tr["e_wrongsign"].max()
    assert e_corr < 0.25 * e_naive          # >=4x reduction (measured ~15x)
    assert e_wrong > e_naive                 # wrong sign strictly worse


def test_steady_turbulence_null():
    """eps=0 => d_tK=0: the correction term is identically zero AND the lagged
    and frozen clocks coincide, so naive and corrected are bit-identical to the
    truth (the §G.5 'vanishes in steady state')."""
    sn = cmn.steady_null(n=48, dt=1e-3)
    assert sn["max_e_naive"] == 0.0
    assert sn["max_e_corrected"] == 0.0


def test_tau_c_order_of_accuracy():
    """Sweeping tau_c->0, the naive error is O(tau_c^1) and the corrected error
    is O(tau_c^2): the correction removes the leading clock-mismatch term."""
    sc = cmn.tau_c_scaling([0.025, 0.05, 0.1], n=48, dt=1e-3)
    assert abs(sc["loglog_slope_naive"] - 1.0) < 0.25
    assert abs(sc["loglog_slope_corrected"] - 2.0) < 0.25
    assert sc["loglog_slope_corrected"] > sc["loglog_slope_naive"] + 0.5


def test_cmn_solver_artifact():
    """The committed production artifact records the §H.3 closure: a large
    transient-error reduction, the tau_c^1 (naive) vs tau_c^2 (corrected) order
    split, an exact steady-state null, and a VERIFIED verdict."""
    with open(ART) as f:
        d = json.load(f)

    cfg = d["config"]
    assert cfg["n"] >= 96 and cfg["scheme"].startswith("RK4")

    b = d["baseline"]
    assert b["maxerr_corrected"] < 0.2 * b["maxerr_naive"]   # measured ~15x
    assert b["maxerr_wrongsign"] > b["maxerr_naive"]         # sign control

    sc = d["tau_c_scaling"]
    assert abs(sc["loglog_slope_naive"] - 1.0) < 0.2
    assert abs(sc["loglog_slope_corrected"] - 2.0) < 0.2

    sn = d["steady_null"]
    assert sn["max_e_naive"] == 0.0 and sn["max_e_corrected"] == 0.0

    assert d["verdict"].startswith("VERIFIED")
