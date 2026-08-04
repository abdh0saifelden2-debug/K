"""Unit proofs for NR38 (`general_two_clocks/new_relationships15.py`):
the subglacial bed as a Randles impedance.  Ledger item E3.

Covered: the Warburg -45 deg limit; the nested reduction (Randles with
sigma_W=0 == the single-RC null); parameter recovery from the 6 tidal lines;
the Kramers-Kronig DC sum rule (causality, the NR23 face); the falsifiable
discrimination (the Warburg is needed on diffusive-bed data, and NOT spuriously
preferred on non-diffusive data); and the over-determination prediction.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships15 import (  # noqa: E402
    OMEGA, TAU_D, kk_dc_sumrule, randles_Z, run, single_rc_Z, warburg,
)


@pytest.fixture(scope="module")
def out():
    o, _ = run(seed=0)
    return o


def test_warburg_minus_45_degree_limit():
    """Semi-infinite Warburg (tau_d -> inf) has phase -45 deg at all w."""
    w = np.array([0.5, 5.0, 50.0])
    ph = np.degrees(np.angle(warburg(w, 1.0, tau_d=1e9)))
    assert np.allclose(ph, -45.0, atol=0.5)


def test_randles_reduces_to_single_rc_when_no_warburg():
    """sigma_W -> 0 collapses the Randles model onto the single-RC null."""
    w = OMEGA
    z_full = randles_Z(w, 0.2, 1.2, 0.5, 0.0, tau_d=TAU_D)
    z_null = single_rc_Z(w, 0.2, 1.2, 0.5)
    assert np.max(np.abs(z_full - z_null)) < 1e-12


def test_parameter_recovery(out):
    assert out["recovery_max_rel_err"] < 0.2      # 4 params from 6 lines @ 2% noise


def test_kk_dc_sum_rule_holds(out):
    """The fitted causal impedance satisfies the KK DC sum rule (NR23 face)."""
    assert out["kk_dc_rel_residual"] < 1e-3


def test_kk_sum_rule_direct():
    """Z'(0)-Z'(inf) = R_ct + sigma_W*sqrt(tau_d) matches the KK integral."""
    Zf = lambda w: randles_Z(w, 0.2, 1.0, 0.5, 0.8)
    lhs, rhs, rel = kk_dc_sumrule(Zf)
    assert abs(lhs - (1.0 + 0.8 * np.sqrt(TAU_D))) / lhs < 1e-3   # analytic DC
    assert rel < 1e-3


def test_warburg_is_needed_on_diffusive_data(out):
    """On a diffusive (Randles) bed the no-Warburg null is rejected decisively."""
    assert out["discrim_delta_aicc_warburg_needed"] > 10.0


def test_no_spurious_warburg_on_nondiffusive_data(out):
    """On a non-diffusive (single-RC) bed the Warburg must NOT be preferred."""
    assert out["control_delta_aicc_no_spurious_warburg"] < 2.0


def test_overdetermination_prediction(out):
    """Fit 4 constituents, predict the held-out 2 to <10%."""
    assert out["overdetermination_predict_rel_err"] < 0.1


def test_band_spans_multiple_decades(out):
    assert out["band_decades"] > 2.0
