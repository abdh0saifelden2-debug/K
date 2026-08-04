"""Unit tests for the new derived relationship NR27
(general_two_clocks/new_relationships4.py). Deterministic, CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import new_relationships4 as NR4  # noqa: E402

_RES = None


def _res():
    global _RES
    if _RES is None:
        _RES = NR4.nr27(n=3_000_000)
    return _RES


# --- analytic closed forms ------------------------------------------------- #
def test_response_is_noise_free_closed_form():
    # chi(w) = 1/(lam + i w): no D, no tau_c.  Static admittance |chi(0)| = 1/lam.
    lam = 0.1
    assert abs(NR4.chi(0.0, lam) - 1.0 / lam) < 1e-12
    w = 0.37
    assert abs(NR4.chi(w, lam) - 1.0 / (lam + 1j * w)) < 1e-12


def test_teff_white_limit_is_flat_and_equals_D():
    # tau_c -> 0: T_eff(w) -> D for all w (equilibrium FDT holds, flat effective temperature)
    w = np.array([0.01, 0.1, 1.0, 10.0])
    assert np.max(np.abs(NR4.teff_analytic(w, 1e-9, D=1.0) - 1.0)) < 1e-6
    # finite memory: half-value exactly at w = 1/tau_c
    tau_c = 2.0
    assert abs(NR4.teff_analytic(1.0 / tau_c, tau_c, D=1.0) - 0.5) < 1e-12


def test_spectrum_double_lorentzian_highf_slope():
    # white bath ~ w^-2; memory bath ~ w^-4 at high frequency (the model-free signature)
    w = np.array([20.0, 40.0])           # well above both corners for lam=0.1, tau_c=1
    s_white = NR4.spectrum(w, 0.1, 1e-3)
    s_mem = NR4.spectrum(w, 0.1, 1.0)
    slope_white = np.log(s_white[1] / s_white[0]) / np.log(w[1] / w[0])
    slope_mem = np.log(s_mem[1] / s_mem[0]) / np.log(w[1] / w[0])
    assert abs(slope_white + 2.0) < 0.05
    assert abs(slope_mem + 4.0) < 0.05


# --- NR27 simulated verification ------------------------------------------- #
def test_nr27_response_is_noise_free_and_measures_proximity():
    r = _res()["response"]
    # (a) noiseless lock-in recovers chi=1/(lam+iw) exactly; a noisy record recovers the
    # same chi (so the proximity read-out needs no noise calibration); lambda recovered.
    assert r["max_err_vs_analytic"] < 0.015
    assert r["max_D_independence_err"] < 0.06
    assert r["lam_resp_rel_err"] < 0.05


def test_nr27_spectrum_slope_steepens_with_memory():
    s = _res()["spectrum"]
    # (b) high-frequency PSD slope: ~-2 (white bath) -> ~-4 (finite-memory bath)
    assert abs(s["highf_slope_white"] + 2.0) < 0.5
    assert abs(s["highf_slope_memory"] + 4.0) < 0.5


def test_nr27_fdt_violation_measures_bath_memory():
    f = _res()["fdt"]
    # (c) T_eff flat (~1) for white bath (FDT holds); rolls off (<<1) for memory bath;
    # fitting the roll-off recovers tau_c and the plateau recovers D.
    assert abs(f["white_flat_ratio"] - 1.0) < 0.15
    assert f["mem_drop_ratio"] < 0.25
    assert abs(f["tau_c_fit"] - f["tau_c_true"]) / f["tau_c_true"] < 0.15
    assert abs(f["D_fit"] - f["D_true"]) / f["D_true"] < 0.15


def test_nr27_debias_recovers_De_from_response_and_fluctuation():
    d = _res()["debias"]
    # De = lambda*tau_c recovered from RESPONSE (lambda) x FDT roll-off (tau_c)
    assert d["De_rel_err"] < 0.15


def test_nr27_overall_ok():
    assert _res()["ok"]
