"""Unit tests for the §D.6 completion — coupled Stefan + Glen-creep amplitude sweep
(validation/synthetic/creep_stefan_coupled.py). Deterministic, CPU-only, no DNS."""
import os
import sys

import numpy as np

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import creep_stefan_coupled as CS  # noqa: E402

_RES = CS.run()


def test_sigma_crit_is_where_creep_matches_melt():
    # by definition r_creep(A, sigma_crit) == beta_melt
    bmelt = _RES["beta_melt_per_s"]
    for A in (CS.A_COLD, CS.A_TEMP):
        sc = CS.sigma_crit(bmelt, A)
        assert np.isclose(CS.r_creep(A, sc), bmelt, rtol=1e-9)


def test_creep_rate_glen_scaling():
    # r_creep = A sigma^3: cubic in stress, linear in A
    assert np.isclose(CS.r_creep(CS.A_TEMP, 2.0e5) / CS.r_creep(CS.A_TEMP, 1.0e5), 8.0, rtol=1e-9)
    assert np.isclose(CS.r_creep(2 * CS.A_TEMP, 1.0e5) / CS.r_creep(CS.A_TEMP, 1.0e5), 2.0, rtol=1e-9)


def test_coupled_correction_is_smoothing_only():
    # a*_coupled/a*_melt = 1/(1+rho) in (0,1]; >0 rho => factor < 1 (creep only smooths)
    fac0, rho0 = CS.coupled_correction(1e-8, 0.0)
    assert fac0 == 1.0 and rho0 == 0.0
    fac1, rho1 = CS.coupled_correction(1e-8, 1e-8)
    assert np.isclose(fac1, 0.5) and np.isclose(rho1, 1.0)
    assert 0.0 < fac1 <= 1.0


def test_beta_melt_matches_committed_anchor():
    # the melt e-folding time reproduces the committed RESULT 14 anchor (~3.03 yr)
    assert 2.8 < _RES["tau_melt_yr"] < 3.3


def test_physical_stress_creep_is_negligible():
    # under the physical topographic stress rho_i g a, creep/melt ratio is tiny
    # across the whole amplitude sweep, and the amplitude correction ~ 1
    assert _RES["max_rho_physical"] < 1e-6
    for r in _RES["physical_stress_sweep"]:
        assert r["rho_temperate"] < 1e-6 and r["rho_cold"] < 1e-6
        assert r["ampl_correction_temperate"] > 1.0 - 1e-6
        assert r["ampl_correction_temperate"] <= 1.0          # smoothing only
    # large safety margin between the crossover stress and the relief stress
    assert _RES["stress_margin_temperate"] > 100.0


def test_worstcase_sigmaN_is_solverclock_only():
    # the honest caveat: sigma=N extrapolated to the melt timescale would give
    # rho>1 at high N (so it is NOT a valid long-time model -> §D.6 keeps it on
    # the solver clock). At low N it is already < 1.
    worst = {(w["ice"], w["N_MPa"]): w for w in _RES["worstcase_N_stress"]}
    assert worst[("temperate", 1.0)]["rho"] > 1.0
    assert worst[("temperate", 1.0)]["creep_dominates_over_years"] is True
    assert worst[("cold", 0.1)]["rho"] < 1.0


def test_overall_ok():
    assert _RES["ok"]
