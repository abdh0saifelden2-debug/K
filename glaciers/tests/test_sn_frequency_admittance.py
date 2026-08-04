"""Unit tests for the frequency-resolved s_N admittance module.

Plant-and-recover guards for the new analytic results in
``validation/synthetic/sn_frequency_admittance.py`` (no external data, no GPU):
  * the quasi-static limit A(omega->0) recovers the s_N(N) master curve;
  * the high-frequency prefactor s_N0*lambda vanishes at the N_c fold (periodic
    gains vanish where the quasi-static s_N0 diverges);
  * the corner N_corner(omega) increases with frequency (rolloff ordering);
  * corner-straddling multi-frequency admittance recovers s_N0 and lambda separately;
  * the phase lag rises toward flotation and -> 90 deg at N_c.
"""
import os
import sys

import numpy as np

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sn_frequency_admittance as FA  # noqa: E402

N_C = FA.N_C


def test_quasi_static_limit_recovers_master_curve():
    qs = FA.quasi_static_limit()
    assert qs["ok"] and qs["max_reldiff"] < 1e-5


def test_admittance_rolls_off_at_high_frequency():
    N = np.array(5.0 * N_C)
    s0 = float(FA.s_N0(N))
    g_slow = float(FA.admittance(1e-6, N)["gain"])
    g_fast = float(FA.admittance(FA.OMEGA["semidiurnal_M2"], N)["gain"])
    assert abs(g_slow - s0) / s0 < 1e-3          # slow probe == quasi-static s_N0
    assert g_fast < 0.5 * s0                     # fast probe rolled off


def test_highfreq_prefactor_vanishes_at_Nc():
    hp = FA.highfreq_prefactor()
    prod = np.array(hp["s0_times_lam"])
    # s_N0*lambda -> 0 at flotation (first point is closest to N_c)
    assert hp["vanishes_at_Nc"]
    assert prod[0] < prod[len(prod) // 2]


def test_corner_increases_with_frequency():
    cf = FA.corner_vs_frequency()
    assert cf["N_corner_increases_with_frequency"]
    rows = {r["probe"]: r["N_corner_MPa"] for r in cf["rows"]}
    assert rows["semidiurnal_M2"] > rows["fortnightly_MSf"] > rows["decadal_ocean"]


def test_spectroscopy_separates_sensitivity_from_memory():
    sp = FA.spectroscopy_test(noise=0.03, seed=1)
    rec = sp["recovery"]
    # both the sensitivity (-> N) and the relaxation rate lambda are recovered
    assert rec["N_relerr"] is not None and rec["N_relerr"] < 0.10
    assert rec["lam_relerr"] is not None and rec["lam_relerr"] < 0.30
    # and tau_h is a finite, positive memory time
    assert sp["recovered"]["tau_h_hat_yr"] and sp["recovered"]["tau_h_hat_yr"] > 0


def test_phase_lag_rises_toward_flotation():
    ph = FA.phase_vs_N()
    assert ph["monotone_rising_toward_flotation"]
    assert ph["lag_near_Nc"] > 89.0          # -> 90 deg at N_c
    assert ph["lag_near_Nc"] > ph["lag_well_grounded"]
