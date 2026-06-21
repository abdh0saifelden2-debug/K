"""Unit tests for §H.1.2 intrusion residence number Ro
(validation/synthetic/intrusion_residence_number.py)."""
import os
import sys

import numpy as np

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import intrusion_residence_number as IR  # noqa: E402


def test_v_kin_identity():
    # v_kin = A * dH/dt, linear in both
    assert IR.v_kin_km_yr(0.70, 1.5) == 0.70 * 1.5
    assert IR.v_kin_km_yr(0.09, 2.0) == 0.09 * 2.0


def test_tau_crit_identity_and_units():
    # tau_crit = ell/v_kin with ell in m, v_kin in km/yr -> yr
    vk = IR.v_kin_km_yr(IR.A_RUNAWAY_KM_PER_M, 1.5)   # km/yr
    tcrit = IR.tau_crit_yr(IR.ELL_M, vk)
    assert abs(tcrit - IR.ELL_M / (vk * 1000.0)) < 1e-12
    # runaway headline value
    assert abs(tcrit - 1.905) < 0.01


def test_Ro_equals_one_at_tau_crit():
    # Ro_pred = v_kin*tau_hyd/ell, so Ro==1 exactly when tau_hyd==tau_crit
    vk = IR.v_kin_km_yr(IR.A_RUNAWAY_KM_PER_M, 1.5)
    tcrit = IR.tau_crit_yr(IR.ELL_M, vk)
    assert abs(IR.Ro_pred(tcrit, vk) - 1.0) < 1e-9


def test_Ro_monotone_and_brackets_regime():
    vk = IR.v_kin_km_yr(IR.A_RUNAWAY_KM_PER_M, 1.5)
    tcrit = IR.tau_crit_yr(IR.ELL_M, vk)
    taus = np.geomspace(*IR.TAU_HYD_BAND_YR, 50)
    Ro = np.array([IR.Ro_pred(t, vk) for t in taus])
    # strictly increasing in residence time
    assert np.all(np.diff(Ro) > 0)
    # thinning-paced below tau_crit, hydraulic-limited above
    assert np.all(Ro[taus < tcrit] < 1.0)
    assert np.all(Ro[taus > tcrit] > 1.0)


def test_D_hyd_range_matches_reported():
    res = IR.run()
    lo, hi = res["D_hyd_implied_m2_s"]
    # ell^2/tau over band 0.01..2 yr -> ~0.06 .. ~12.7 m^2/s
    assert 0.05 < lo < 0.08
    assert 11.0 < hi < 14.0
    # explicit identity check
    assert abs(hi - IR.D_hyd_m2_s(IR.TAU_HYD_BAND_YR[0])) < 1e-6
    assert abs(lo - IR.D_hyd_m2_s(IR.TAU_HYD_BAND_YR[1])) < 1e-6


def test_runaway_headline_is_thinning_paced():
    res = IR.run()
    h = res["runaway_headline"]
    # runaway tail (A=0.70, dH/dt=1.5) sits thinning-paced over >90% of the band
    assert h["frac_residence_band_thinning_paced"] > 0.9
    assert "thinning-paced" in h["verdict"].lower()


def test_runaway_more_kinematically_amplified_than_margin():
    res = IR.run()
    # runaway tail has larger v_kin -> smaller tau_crit -> more easily hydraulic-limited
    run_rows = res["regimes"]["runaway_tail"]
    med_rows = res["regimes"]["margin_median"]
    for rr, mr in zip(run_rows, med_rows):
        assert rr["v_kin_km_yr"] > mr["v_kin_km_yr"]
        assert rr["tau_crit_yr"] < mr["tau_crit_yr"]
