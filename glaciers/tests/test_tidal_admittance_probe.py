"""Unit tests for the tidal-admittance third probe (validation/synthetic/tidal_admittance_probe.py).

Guards (no external data, no GPU):
  * the fundamental tidal admittance -> |s_N| in the small-amplitude limit;
  * the 2f/1f harmonic ratio matches the analytic curvature law and grows toward
    flotation;
  * the tides-only inversion recovers the flotation-proximity R=(N_c/N)^m and m
    from velocity alone.
"""
import os
import sys

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tidal_admittance_probe as TA  # noqa: E402


def test_admittance_is_sN_small_eps():
    v = TA.verify_probe()
    assert v["max_admittance_vs_sN_relerr_smalleps"] < 0.03


def test_harmonic_matches_curvature_and_grows():
    v = TA.verify_probe()
    assert v["max_harmonic_vs_analytic_relerr"] < 0.08
    assert v["grows_toward_flotation"]
    assert v["ratio_2f_1f_near_flotation"] > v["ratio_2f_1f_wellgrounded"]


def test_tides_only_inversion_recovers_R_and_m():
    inv = TA.inversion_test()
    assert inv["max_R_relerr"] < 0.08
    assert inv["max_m_relerr"] < 0.05


def test_operational_ews_rises():
    ews = TA.tidal_ews()
    assert ews["admittance_rise"] > 2.0
    assert ews["harmonic_rise"] > 2.0
    # recovered proximity R approaches 1 toward flotation
    assert ews["R_recovered"][-1] > ews["R_recovered"][0]
