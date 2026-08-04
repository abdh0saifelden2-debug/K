"""Unit tests for the new derived relationship NR30
(general_two_clocks/new_relationships7.py). Deterministic, CPU-only.

NR30: the subglacial hydraulic potential phi (parabolic, screened, lagged Darcy head with
storage) is NOT the Leray pressure (elliptic, instantaneous, memoryless constraint multiplier);
they coincide only in the singular no-storage limit tau_hyd -> 0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import new_relationships7 as NR7  # noqa: E402

_RES = None


def _res():
    global _RES
    if _RES is None:
        _RES = NR7.nr30(n=32, D_h=1.0)   # fast, deterministic; ok=True
    return _RES


# --- analytic closed forms ------------------------------------------------- #
def test_leray_transfer_is_real_and_instantaneous():
    # H_p(k) = 1/k^2 is REAL (zero phase) and omega-independent -> instantaneous, memoryless
    k2 = np.array([1.0, 4.0, 9.0, 25.0])
    Hp = NR7.leray_transfer(k2)
    assert np.allclose(Hp.imag, 0.0, atol=1e-15)
    assert np.allclose(Hp.real, 1.0 / k2, atol=1e-15)
    assert np.max(np.abs(np.angle(Hp))) < 1e-15


def test_hydraulic_transfer_is_screened_and_lagged():
    # H_phi(k,omega) = 1/(k^2 + i omega/D_h): |H_phi| < 1/k^2 (screening) and a nonzero lag
    k2, omega, D_h = 4.0, 2.0, 1.0
    Hphi = NR7.hydraulic_transfer(k2, omega, D_h)
    assert abs(Hphi) < 1.0 / k2                                  # screened below the Poisson kernel
    assert abs(abs(Hphi) - 1.0 / np.sqrt(k2 ** 2 + (omega / D_h) ** 2)) < 1e-12
    # phase lag closed form arctan(omega/(D_h k^2))
    lag = NR7.hydraulic_phase_lag(k2, omega, D_h)
    assert abs(lag - np.arctan((omega / D_h) / k2)) < 1e-12
    assert abs(-np.angle(Hphi) - lag) < 1e-12


def test_45_degree_crossover_is_the_hydraulic_clock():
    # the head lags by exactly 45 deg when omega = omega_c = D_h k^2  (NR28 p/phi crossover)
    k2, D_h = 9.0, 0.5
    omega_c = D_h * k2
    assert abs(NR7.hydraulic_phase_lag(k2, omega_c, D_h) - np.pi / 4.0) < 1e-12
    # monotone: lag rises 0 -> pi/2 with omega
    assert NR7.hydraulic_phase_lag(k2, 0.01 * omega_c, D_h) < np.pi / 8
    assert NR7.hydraulic_phase_lag(k2, 100.0 * omega_c, D_h) > 3 * np.pi / 8


def test_coincidence_only_in_no_storage_limit():
    # H_phi -> H_p iff omega/D_h -> 0 (steady state OR D_h -> infinity / tau_hyd -> 0)
    k2, omega = 4.0, 4.0
    Hp = NR7.leray_transfer(k2)
    gaps = [abs(NR7.hydraulic_transfer(k2, omega, D_h) - Hp) for D_h in (1.0, 10.0, 100.0, 1e4)]
    assert all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1))   # monotone -> 0
    assert gaps[-1] < 1e-3 and gaps[0] > 0.1                          # apart at finite storage
    # steady state (omega=0) also coincides exactly
    assert abs(NR7.hydraulic_transfer(k2, 0.0, 1.0) - Hp) < 1e-15


def test_hydraulic_time_closed_form():
    # tau_k = 1/(D_h k^2)
    assert abs(NR7.hydraulic_time(4.0, 2.0) - 1.0 / 8.0) < 1e-15


# --- NR30 simulated verification ------------------------------------------- #
def test_transfer_numeric_matches_closed_form():
    assert _res()["transfer"]["numeric_vs_closed_relerr"] < 1e-10


def test_driven_separator_head_lags_leray_does_not():
    sep = _res()["separator"]
    # the driven hydraulic head lags by the closed-form amount; the Leray pressure does not
    assert abs(sep["lag_phi"] - sep["lag_closed"]) < 0.02
    assert sep["lag_phi"] > 0.1
    assert abs(sep["lag_p"]) < 1e-3


def test_memory_time_recovered():
    m = _res()["memory"]
    assert abs(m["tau_measured"] - m["tau_closed"]) / m["tau_closed"] < 1e-3


def test_nr30_overall_ok():
    assert _res()["ok"] is True
