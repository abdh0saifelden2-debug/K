"""Tests for the Sec.G.2 / RESULT 14 *field test* (scallop_field_test.py).

Two things are checked:

  1. the §8.2 "pin" recipe -- harmonic_mode_rate() -- recovers a KNOWN complex
     growth rate ``s = Re(s) + i*Im(s)`` (hence ``I = |Im(s)|/(2pi|Re(s)|)`` and
     the downstream sign) from a synthetic damped, migrating interface, so it is
     ready to ingest Bushuk's raw ``h(x, t)`` arrays; and

  2. the committed Bushuk-2019 adjustment-regime figure bound reproduces the
     honest ``REPORT_SCALLOP_MIGRATION.md`` §8.1 result: downstream sign, point
     ``I_obs ~ 0.1`` a factor ~2-3 below the solver band -- a mild tension, NOT a
     falsification.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scallop_field_test as sf


def test_harmonic_mode_rate_recovers_known_mode():
    """A synthetic damped, downstream-migrating train with known (Re_s, c_mig,
    lam) must be recovered to <2% (and I = tau*c_mig/lam to <3%)."""
    Re_true = -1.0 / 7200.0                 # tau = 2 h
    c_true, lam_true = 1.833e-6, 0.13
    x, t, H = sf.synth_train(Re_true, c_true, lam_true, noise=0.01, seed=1)
    rec = sf.harmonic_mode_rate(x, t, H)

    assert rec["downstream"] is True
    assert abs(rec["Re_s"] - Re_true) <= 0.02 * abs(Re_true)
    assert rec["Re_s"] < 0.0                # damping
    assert abs(rec["c_mig"] - c_true) <= 0.02 * c_true
    assert abs(rec["lam"] - lam_true) <= 0.02 * lam_true

    # I is identical via the two definitions: |Im(s)|/(2pi|Re(s)|) == tau*c_mig/lam
    I_true = (1.0 / abs(Re_true)) * c_true / lam_true
    assert abs(rec["I"] - I_true) <= 0.03 * I_true
    assert abs(rec["I"] - sf.i_from_kinematics(rec["c_mig"], rec["lam"], rec["tau"])) < 1e-9


def test_upstream_migration_sign_is_detected():
    """An upstream-migrating train must flip the sign (downstream=False)."""
    x, t, H = sf.synth_train(-1.0 / 7200.0, 1.5e-6, 0.13, downstream=False, seed=2)
    rec = sf.harmonic_mode_rate(x, t, H)
    assert rec["downstream"] is False
    assert rec["c_phase"] < 0.0


def test_i_from_kinematics_definition():
    assert sf.i_from_kinematics(2.0, 4.0, 3.0) == 1.5   # tau*c/lam = 3*2/4


def test_bushuk_bound_matches_report():
    """Committed regime-matched bound reproduces REPORT §8.1 numbers + verdict."""
    b = sf.bushuk_adjustment_bound()

    # c_mig = 0.11 mm/min -> 1.833e-6 m/s
    assert abs(b["c_mig_m_per_s"] - 1.833e-6) < 1e-8
    # observed-lambda range ~0.05-0.15, point ~0.10
    lo, hi = b["I_observed_lambda"]
    assert abs(lo - 0.051) < 5e-3 and abs(hi - 0.152) < 5e-3
    assert abs(b["I_point_estimate"] - 0.10) < 0.02
    # full range incl Curl lambda ~ O(0.05-0.4)
    flo, fhi = b["I_obs_range"]
    assert 0.04 < flo < 0.07 and 0.35 < fhi < 0.45

    # honest verdict: downstream, same order, point below band, NOT falsified
    assert b["sign_downstream"] is True
    assert b["same_order_of_magnitude"] is True
    assert b["point_below_band"] is True
    assert 2.0 < b["factor_below_band"] < 4.0
    assert b["falsified"] is False
    # the point sits below the solver band's lower edge (the documented tension)
    band_lo, band_hi = b["solver_band"]
    assert b["I_point_estimate"] < band_lo
    assert band_lo >= 0.3 and band_hi <= 0.9


def test_run_assembles_and_self_checks():
    out = sf.run()
    v = out["verdict"]
    assert v["decomposition_recovers_known_mode"] is True
    assert v["field_sign_matches_downstream"] is True
    assert v["field_same_order_as_solver"] is True
    assert v["field_not_falsified"] is True
    assert v["pin_requires_raw_arrays"] is True
    # artifact written
    p = os.path.join(sf.HERE, "figures", "58_scallop_field_test.json")
    assert os.path.exists(p)
