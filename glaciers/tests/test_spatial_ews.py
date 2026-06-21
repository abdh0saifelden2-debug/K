"""Unit tests for the spatial early-warning module (validation/synthetic/spatial_ews.py).

Guards (no external data, no GPU): the exact stationary covariance shows spatial
variance and along-flow correlation length rising toward the grounding line as
N -> N_c, matching the 1/sqrt(D*lambda) / sqrt(D/lambda) scalings.
"""
import os
import sys

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spatial_ews as SE  # noqa: E402


def test_lambda_vanishes_toward_flotation():
    lam_far = SE.lambda_profile(6.0e5)
    lam_near = SE.lambda_profile(1.1 * SE.N_C)
    assert lam_near < 1e-2 * lam_far


def test_spatial_variance_and_corrlength_rise_toward_GL():
    res = SE.run()
    assert res["kendall_tau_variance_toward_GL"] > 0.7
    assert res["kendall_tau_corrlength_toward_GL"] > 0.5
    assert res["var_ratio_GL_over_interior"] > 3.0
    assert res["corrlen_ratio_GL_over_interior"] > 1.5
    assert res["rising_spatial_ews"]


def test_interior_obeys_one_over_sqrt_lambda():
    res = SE.run()
    # Var * sqrt(lambda) is ~constant in the interior (the 1/sqrt(D*lambda) law)
    assert res["interior_var_sqrtlambda_cv"] < 0.2
