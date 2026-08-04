"""Unit tests for §A.1 interface coupling number (validation/synthetic/interface_coupling_number.py)."""
import os
import sys


_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import interface_coupling_number as IC  # noqa: E402

SEC = IC.SEC_PER_YR


def test_dc_limit_is_stefan_number():
    v = IC.verify(theta_far=-1.0, Vbar=0.1 / SEC)
    assert v["dc_rel_err"] < 1e-3
    assert abs(v["lambda_dc"] - IC.stefan_number(-1.0)) < 1e-6


def test_highfreq_limit_vanishes():
    v = IC.verify(theta_far=-1.0, Vbar=0.1 / SEC)
    assert v["lambda_highfreq"] < 1e-2 * v["stefan_number"]


def test_surge_band_is_frozen():
    res = IC.run()
    assert res["surge_band_frozen"]
    assert res["lambda_in_surge_band_max"] < 0.1 * res["stefan_number_baseline"]
    # always frozen across the literature sweep
    assert res["sweep_surge_band_lambda_over_St"]["frac_below_0p1"] > 0.99
