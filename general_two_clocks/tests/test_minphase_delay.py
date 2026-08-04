"""Unit proofs for NR48 (`general_two_clocks/new_relationships25.py`): the
two-clocks response is minimum-phase (gain and phase are a Bode/Hilbert
pair), and a genuine transport delay is the unique all-pass factor read
model-free off the excess phase.  Offline-safe -- the real-data reads come
from the committed cache ``data/nr48_minphase_cache.json`` (ocean GSER from
the NR45 ANDRO cache; delay clock from the BiSON x GOLF cross-spectrum).

Covered: the exact Bode gain-phase integral reconstructs the phase of
Debye / fractional / two-pole minimum-phase models from ``ln|chi|`` alone;
a pure delay ``exp(-i w tau)`` has zero Bode phase (all-pass) and its
excess-phase slope recovers ``tau`` to machine precision; the banded
transform (terminal-slope extension) beats raw truncation on a short band;
the committed cache schema; the ocean GSER minimum-phase verdict (~3 deg);
and the real BiSON x GOLF delay-clock gate (flat gain, |tau_d| < 30 s,
GOLF leads BiSON under the stated convention).
"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships25 import (  # noqa: E402
    CACHE, FIG, analyze, bode_phase_banded, bode_phase_from_logamp,
    delay_from_phase, excess_phase, load_cache, run, synthetic_checks,
)


@pytest.fixture(scope="module")
def cache():
    return load_cache()


@pytest.fixture(scope="module")
def result(cache):
    return analyze(cache)


# --------------------------------------------------------------------------- #
# the Bode gain-phase integral on exact minimum-phase models
# --------------------------------------------------------------------------- #
def test_bode_integral_reconstructs_debye_phase():
    w = np.geomspace(1e-3, 1e3, 3000)
    H = 1.0 / (1 + 1j * w)
    phi_b = bode_phase_from_logamp(np.log(w), np.log(np.abs(H)))
    mb = (w > 1e-2) & (w < 1e2)
    err = np.abs(np.unwrap(np.angle(H)) - phi_b)[mb]
    err -= np.median(err)
    assert np.median(np.degrees(np.abs(err))) < 0.1


def test_bode_integral_reconstructs_fractional_and_two_pole():
    syn = synthetic_checks()
    assert set(syn["recon_median_deg"]) == {"debye", "fractional", "two_pole"}
    assert max(syn["recon_median_deg"].values()) < 0.1


def test_power_law_gives_exact_half_pi_alpha():
    # |chi| ~ w^alpha -> Bode phase = pi*alpha/2 exactly (interior band)
    alpha = 0.6
    w = np.geomspace(1e-7, 1e7, 6000)
    phi_b = bode_phase_from_logamp(np.log(w), alpha * np.log(w))
    mb = (w > 1e-2) & (w < 1e2)
    assert np.allclose(phi_b[mb], np.pi * alpha / 2, atol=1e-3)


def test_uniform_grid_required():
    with pytest.raises(ValueError):
        bode_phase_from_logamp(np.array([0.0, 1.0, 3.0]), np.zeros(3))


# --------------------------------------------------------------------------- #
# the delay is all-pass: zero Bode phase, linear excess phase
# --------------------------------------------------------------------------- #
def test_pure_delay_is_allpass_and_recovered():
    syn = synthetic_checks()
    assert syn["delay_bode_phase_max"] < 1e-6          # |chi|=1 -> no Bode phase
    assert abs(syn["delay_tau_recovered"] - syn["delay_tau_true"]) < 1e-6


def test_delay_from_phase_sign_convention():
    # chi = exp(-i w tau), tau > 0  ->  delay_from_phase returns +tau
    f = np.linspace(0.1, 2.0, 200)
    tau = 1.7
    ph = -2 * np.pi * f * tau + 0.3
    tau_rec, off = delay_from_phase(f, ph)
    assert abs(tau_rec - tau) < 1e-9 and abs(off - 0.3) < 1e-9


def test_excess_phase_isolates_delay_from_relaxation():
    # Debye * delay: excess phase over the Bode part is linear with slope tau
    w = np.geomspace(1e-3, 1e3, 3000)
    tau = 0.25
    H = np.exp(-1j * w * tau) / (1 + 1j * w)
    exc, phi_b = excess_phase(np.log(w), np.log(np.abs(H)),
                              np.unwrap(np.angle(H)))
    mb = (w > 1e-2) & (w < 1e1)
    tau_rec, _ = delay_from_phase(w[mb] / (2 * np.pi), exc[mb])
    assert abs(tau_rec - tau) < 0.01


def test_banded_transform_beats_truncation_on_short_band():
    # fractional model on a 1.5-decade band: terminal-slope extension must
    # reduce the interior reconstruction error vs the raw truncated integral
    wfull = np.geomspace(1e-4, 1e4, 6000)
    Hfull = (1j * wfull) ** 0.6 / (1 + (1j * wfull) ** 0.6)
    band = (wfull >= 3e-2) & (wfull <= 1.0)
    lw = np.log(wfull[band])
    grid = np.linspace(lw.min(), lw.max(), 200)
    lnH = np.interp(grid, lw, np.log(np.abs(Hfull[band])))
    phi_true = np.interp(grid, lw, np.unwrap(np.angle(Hfull))[band])
    span = grid[-1] - grid[0]
    mb = (grid >= grid[0] + 0.15 * span) & (grid <= grid[-1] - 0.15 * span)

    def med_err(phi):
        d = (phi_true - phi)[mb]
        return np.median(np.degrees(np.abs(d - np.median(d))))

    err_raw = med_err(bode_phase_from_logamp(grid, lnH))
    err_band = med_err(bode_phase_banded(grid, lnH))
    assert err_band < err_raw
    assert err_band < 1.0


# --------------------------------------------------------------------------- #
# committed cache schema
# --------------------------------------------------------------------------- #
def test_cache_schema(cache):
    assert set(cache) >= {"meta", "ocean_gser", "delay_clock"}
    o = cache["ocean_gser"]
    assert set(o) >= {"log_w", "log_Gmag", "delta_meas_rad"}
    assert len(o["log_w"]) == len(o["log_Gmag"]) == len(o["delta_meas_rad"])
    assert all(np.isfinite(o[k]).all() for k in
               ("log_w", "log_Gmag", "delta_meas_rad"))
    d = cache["delay_clock"]
    assert set(d) >= {"freq_hz", "phase_rad", "g2", "gain"}
    n = len(d["freq_hz"])
    assert n > 100 and all(len(d[k]) == n for k in
                           ("phase_rad", "g2", "gain"))


# --------------------------------------------------------------------------- #
# real-data verdicts
# --------------------------------------------------------------------------- #
def test_ocean_gser_is_minimum_phase(result):
    assert result["ocean_minphase_median_deg"] < 6.0
    assert result["verdicts"]["gser_is_minimum_phase_on_real_ocean"]


def test_real_delay_clock_gate(result):
    # flat gain (no amplitude signature) + |tau_d| < 30 s, well-resolved
    assert result["delay_n_coherent"] >= 100
    assert result["delay_gain_log_std"] < 0.6
    assert 0.0 < abs(result["delay_tau_d_s"]) < 30.0
    assert result["verdicts"]["real_delay_clock_measured"]


def test_delay_sign_means_golf_leads_bison(result):
    # under S_xy = GOLF x conj(BiSON) and chi = exp(-i w tau_d), the fitted
    # tau_d is negative: GOLF leads BiSON by ~10 s (pure time-base offset)
    assert result["delay_tau_d_s"] < 0.0
    assert 5.0 < abs(result["delay_tau_d_s"]) < 15.0


def test_all_verdicts_pass_and_figure_written(result):
    v = result["verdicts"]
    for key in ("two_clocks_response_is_minimum_phase",
                "gser_is_minimum_phase_on_real_ocean",
                "delay_is_allpass_excess", "real_delay_clock_measured"):
        assert v[key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        on_disk = json.load(fh)
    assert on_disk["verdicts"]["real_delay_clock_measured"] is True
    assert on_disk["delay_tau_d_s"] == pytest.approx(
        res["delay_tau_d_s"], rel=1e-12)
