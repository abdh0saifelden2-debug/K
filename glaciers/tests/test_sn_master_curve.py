"""Unit tests for the s_N(N) master-curve / inversion / critical-slowing-down module.

Plant-and-recover guards for the new analytic results in
``validation/synthetic/sn_master_curve.py`` (no external data, no GPU):
  * the closed form |s_N|=m/(1-(N_c/N)^m) matches the repo's numeric s_N;
  * the near-flotation pole |s_N| ~ N_c/(N-N_c) is leading-order correct;
  * the amplitude law grows monotonically toward flotation;
  * the inversion recovers the flotation threshold N_c (the well-conditioned param);
  * the restoring rate vanishes at N_c (critical slowing down) and the equilibrium
    variance diverges toward flotation.
"""
import os
import sys

import numpy as np

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sn_master_curve as SM  # noqa: E402

N_C = SM.TAU_D / SM.C_FRIC


def test_closed_form_matches_numeric():
    v = SM.verify_closed_form()
    assert v["max_reldiff_vs_numeric"] < 1e-3
    assert abs(v["s_N_wellgrounded"] - SM.M_EXP) < 0.05


def test_near_flotation_pole_leading_order():
    v = SM.verify_closed_form()
    # leading-order pole within ~12% inside 1.1 N_c
    assert v["near_pole_max_reldiff"] is not None and v["near_pole_max_reldiff"] < 0.12


def test_amplitude_grows_toward_flotation():
    N = np.geomspace(1.05 * N_C, 50 * N_C, 30)
    amp = SM.amp_law(N)            # N ascending -> amp should be descending
    assert np.all(np.diff(amp) < 0)
    assert SM.s_N_closed(1.02 * N_C) > SM.s_N_closed(20 * N_C)


def test_inversion_recovers_threshold():
    fit = SM.inversion_synthetic_test(noise=0.10, n_pts=14, seed=3)
    assert fit["ok"]
    # N_c is the well-conditioned parameter -> recovered to <10%
    assert fit["recovery"]["N_c_relerr"] < 0.10


def test_inversion_robustness_Nc_better_than_m():
    rob = SM.inversion_robustness(noises=(0.10,), n_seed=10)["per_noise"][0]
    # the threshold is recovered far better than the exponent (degeneracy of m with f)
    assert rob["N_c_median_relerr"] < rob["m_median_relerr"]
    assert rob["N_c_median_relerr"] < 0.05


def test_critical_slowing_down():
    # restoring rate vanishes toward the N_c fold
    lam_near = SM.restoring_rate(np.array(1.02 * N_C))
    lam_far = SM.restoring_rate(np.array(20 * N_C))
    assert lam_near < 1e-2 * lam_far
    # equilibrium variance diverges toward flotation
    et = SM.ews_theory()
    assert et["var_ratio_Nc_over_ref"] > 50.0
