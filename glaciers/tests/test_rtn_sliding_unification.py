"""Unit tests for the RTN <-> s_N unification (validation/synthetic/rtn_sliding_unification.py).

Guards (no external data, no GPU): RTN and the sliding-law divergence are ordered
crossings of one normalized effective pressure n_hat; the RTN=1 intrusion line sits
inland of the ungrounding fold, and the committed ocean-gating terciles are consistent.
"""
import os
import sys

import numpy as np

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rtn_sliding_unification as RU  # noqa: E402


def test_rtn_identity_and_threshold():
    # RTN = (1-n_hat)/phi; RTN>1 exactly at n_hat < 1-phi
    assert abs(RU.RTN_of_nhat(0.0, 0.9) - 1.0 / 0.9) < 1e-12
    assert RU.RTN_of_nhat(0.05, 0.9) > 1.0
    assert RU.RTN_of_nhat(0.20, 0.9) < 1.0


def test_intrusion_inland_of_fold():
    for H in (500.0, 1000.0, 2000.0, 3000.0):
        t = RU.thresholds(H)
        assert t["ordering_intrusion_inland_of_fold"]
        assert t["n_fold_ungrounding"] < t["n_RTN1_intrusion"]


def test_sN_diverges_below_intrusion():
    # |s_N| is still ~m at the intrusion line and only diverges much closer to the fold
    H = 2000.0
    t = RU.thresholds(H)
    assert t["sN_at_intrusion"] < 1.5 * RU.M_EXP
    assert RU.s_N_of_nhat(np.array(1.001 * t["n_fold_ungrounding"]), H) > 5 * RU.M_EXP


def test_run_and_field_consistency():
    res = RU.run(os.path.join(RU._REPORTS, "efp_gate_direct_n.json"))
    assert res["intrusion_inland_of_fold"]
    # committed gating terciles, if present, should be consistent
    if res.get("field_overlay"):
        assert res["field_consistent"]
