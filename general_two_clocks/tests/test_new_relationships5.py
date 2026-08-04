"""Unit tests for the new derived relationship NR28
(general_two_clocks/new_relationships5.py). Deterministic, CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import new_relationships5 as NR5  # noqa: E402

_RES = None


def _res():
    global _RES
    if _RES is None:
        _RES = NR5.nr28(n=3_000_000)   # seed=0, deterministic; ok=True
    return _RES


# --- analytic closed forms ------------------------------------------------- #
def test_phase_is_arctan_with_crossover_at_omega_c():
    wc = 0.5
    # 45deg crossover EXACTLY at omega = omega_c (no calibration needed)
    assert abs(NR5.phase_analytic(wc, wc) - np.pi / 4) < 1e-12
    # zero at DC, asymptotes to (but never reaches) 90deg
    assert abs(NR5.phase_analytic(0.0, wc)) < 1e-12
    assert NR5.phase_analytic(1e6 * wc, wc) < np.pi / 2
    assert NR5.phase_analytic(1e6 * wc, wc) > np.pi / 2 - 1e-3


def test_phase_one_sided_and_bounded_by_90deg():
    wc = 0.7
    w = np.linspace(1e-3, 50.0, 2000)
    phi = NR5.phase_analytic(w, wc)
    # pressure leads for every positive omega (never lags) ...
    assert np.all(phi > 0.0)
    # ... and the lead is strictly bounded by 90deg (a first-order lag, not a transport delay)
    assert np.all(phi < np.pi / 2)
    # monotonically increasing in omega
    assert np.all(np.diff(phi) > 0.0)


def test_coherence_closed_form_value_and_monotone_drop():
    wc, s_d, n_p, n_th = 0.5, 1.0, 0.05, 0.25
    # closed form at DC:  gamma^2 = s_d^2*Ht2 / [(s_d+n_p)(s_d*Ht2+n_th)], Ht2 = 1/wc^2
    g0 = NR5.coherence_analytic(0.0, wc, s_d, n_p, n_th)
    Ht2 = 1.0 / wc ** 2
    expect = (s_d ** 2 * Ht2) / ((s_d + n_p) * (s_d * Ht2 + n_th))
    assert abs(g0 - expect) < 1e-12
    assert abs(g0 - 0.896359) < 1e-4
    # coherence is high at low omega and drops monotonically toward 0 (clocks decouple)
    w = np.linspace(1e-3, 40.0, 4000)
    g = NR5.coherence_analytic(w, wc, s_d, n_p, n_th)
    assert np.all(np.diff(g) < 0.0)
    assert g[0] > 0.8 and g[-1] < 0.1


def test_coherence_decoupling_scales():
    wc = 0.5
    g_low = NR5.coherence_analytic(0.15 * wc, wc, 1.0, 0.05, 0.25)
    g_high = NR5.coherence_analytic(15.0 * wc, wc, 1.0, 0.05, 0.25)
    assert g_low > 0.8           # large scales: clocks coupled
    assert g_high < 0.1          # small scales: clocks decoupled
    assert g_high < 0.6 * g_low  # a genuine drop


# --- NR28 simulated verification ------------------------------------------- #
def test_nr28_phase_measures_the_parabolic_clock():
    ph = _res()["phase"]
    # the cross-spectral phase reproduces arctan(omega/omega_c) over the coherent band ...
    assert ph["rms_err_rad"] < 0.06
    # ... and omega_c = kappa k^2 is recovered FROM THE PHASE ALONE (no amplitude calibration)
    assert ph["fit_rel_err"] < 0.05         # tan(phi)=omega/omega_c linear fit
    assert ph["crossover_rel_err"] < 0.06   # interpolated 45deg crossover


def test_nr28_coherence_drop_is_the_decoupling():
    co = _res()["coherence"]
    assert co["drops"] is True
    assert co["gamma_low"] > 0.7            # coupled at large scales
    assert co["gamma_high"] < 0.2           # decoupled at small scales
    assert abs(co["gamma_low"] - co["gamma_an_low"]) < 0.08
    assert abs(co["gamma_high"] - co["gamma_an_high"]) < 0.08


def test_nr28_one_sided_and_bounded():
    ca = _res()["causality"]
    assert ca["one_sided_pressure_leads"] is True   # pressure leads at every coherent omega
    assert ca["bounded_by_90deg"] is True           # bounded lead -> diffusive, not transport


def test_nr28_overall_ok():
    assert _res()["ok"] is True
