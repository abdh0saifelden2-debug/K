"""Unit proofs for NR49 (`general_two_clocks/new_relationships26.py`): the
two-snapshot transfer of an evolving interface factorizes into gain (growth
clock, Galilean-invariant) x all-pass (transport clock), and a Galilean
boost is exactly the spatial all-pass factor -- NR48's delay in the
conjugate variable.  Offline-safe: the real reads are the committed Bushuk
.mat (glaciers/subglacial/data/bushuk/) and the committed tracker cache
(glaciers/subglacial/data/bushuk_raw_derived.json).

Covered: the exact boost identity lambda -> lambda + ikV on synthetics
(sigma invariant, celerity recovered, machine precision); pure damping
cannot fake a celerity and a pure boost cannot fake a gain; the
phase-Nyquist aliasing bound; estimator internals (weighted median, SNR
gate, band selection); and the real-data verdicts -- lab-frame pattern
celerity vs the committed trackers, co-moving residual suppression,
frame-invariant growth spectrum, marginal pattern + damped roughness.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships26 import (  # noqa: E402
    DERIVED, FIG, MAT, analyze, band_indices, committed_trackers,
    frame_spectra, load_bushuk, pair_dispersion, run, snr_gate,
    synthetic_checks, weighted_median,
)


@pytest.fixture(scope="module")
def syn():
    return synthetic_checks()


@pytest.fixture(scope="module")
def result():
    return analyze()


# --------------------------------------------------------------------------- #
# the boost identity, exactly
# --------------------------------------------------------------------------- #
def test_boost_identity_machine_precision(syn):
    assert syn["sigma_max_err"] < 1e-9          # sigma(k) recovered
    assert syn["celerity_max_err"] < 1e-9       # c = V recovered (|kVdt|<pi)
    assert syn["boost_sigma_invariance"] < 1e-9  # boost never moves the gain
    assert syn["boost_residual_celerity"] < 1e-9  # boost removal -> c = 0


def test_damping_cannot_fake_celerity(syn):
    assert syn["damping_fitted_celerity"] < 1e-9


def test_pure_boost_is_allpass(syn):
    assert syn["pure_boost_fitted_gain"] < 1e-9


def test_phase_nyquist_wraps_beyond_half_wavelength():
    # beyond |kVdt| = pi the principal-value phase aliases: the fitted
    # celerity must NOT equal V there (this is why the bound exists)
    n, dx, dt, V = 256, 1.0, 1.0, 3.0
    k = 2.0 * np.pi * np.fft.rfftfreq(n, dx)
    H0 = np.fft.rfft(np.random.default_rng(1).standard_normal(n))
    H = np.array([H0 * np.exp(-1j * k * V * dt * j) for j in range(4)])
    _, cel, _ = pair_dispersion(H, k, dt_s=dt)
    beyond = (k * V * dt > 1.1 * np.pi) & (k * V * dt < 1.9 * np.pi)
    assert beyond.any()
    assert np.all(np.abs(np.median(cel, 0)[beyond] / 3600.0 - V) > 0.1)


# --------------------------------------------------------------------------- #
# estimator internals
# --------------------------------------------------------------------------- #
def test_weighted_median_basic():
    assert weighted_median([1.0, 2.0, 100.0], [1.0, 1.0, 0.01]) == 2.0
    assert weighted_median([5.0], [3.0]) == 5.0


def test_snr_gate_and_bands():
    rng = np.random.default_rng(2)
    x = np.arange(200) * 1.0
    frames = [np.sin(2 * np.pi * x / 50.0) + 0.01 * rng.standard_normal(200)
              for _ in range(3)]
    H, k = frame_spectra(frames, 1.0)
    g = snr_gate(H, k)
    idx = band_indices(k, g, 40.0, 60.0)
    assert len(idx) >= 1
    lam = 2 * np.pi / k[idx]
    assert np.all((lam >= 40.0) & (lam <= 60.0))


# --------------------------------------------------------------------------- #
# committed real data
# --------------------------------------------------------------------------- #
def test_committed_sources_exist():
    assert os.path.exists(MAT) and os.path.exists(DERIVED)


def test_bushuk_loads_expected_shapes():
    raw = load_bushuk()
    assert raw["Z_lab"].shape == (12, 166)
    assert raw["Z_adv"].shape[0] == 12
    assert np.allclose(np.diff(raw["t_s"]), 300.0)


def test_trackers_agree_with_bushuk_boost():
    trk = committed_trackers()
    raw = load_bushuk()
    boost = np.polyfit(raw["t_s"] / 3600.0, raw["xadv_mm"], 1)[0]
    assert abs(abs(boost) - trk["v_track_mm_hr"]) < 1.0
    assert abs(abs(boost) - trk["v_xcorr_mm_hr"]) < 5.0


# --------------------------------------------------------------------------- #
# real-data verdicts
# --------------------------------------------------------------------------- #
def test_migration_is_one_allpass_factor(result):
    assert 0.7 < result["celerity_ratio_lab_vs_xcorr"] < 1.3
    assert result["adv_residual_fraction"] < 0.2
    assert result["verdicts"]["migration_is_one_allpass_factor"]


def test_gain_is_boost_invariant(result):
    fr = result["frames"]
    assert abs(fr["lab"]["pattern"]["sigma_hr"]
               - fr["adv"]["pattern"]["sigma_hr"]) < 1.5
    assert fr["lab"]["rough"]["sigma_hr"] < -3.0
    assert fr["adv"]["rough"]["sigma_hr"] < -3.0
    assert result["verdicts"]["gain_is_boost_invariant"]


def test_marginal_pattern_damped_roughness(result):
    fr = result["frames"]
    assert abs(fr["lab"]["pattern"]["sigma_hr"]) < 1.0
    assert abs(fr["adv"]["pattern"]["sigma_hr"]) < 1.0
    assert result["verdicts"]["marginal_pattern_damped_roughness"]


def test_all_verdicts_and_figure(result):
    for key in ("boost_identity_exact", "migration_is_one_allpass_factor",
                "gain_is_boost_invariant",
                "marginal_pattern_damped_roughness"):
        assert result["verdicts"][key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["verdicts"]["migration_is_one_allpass_factor"] is True
    assert disk["adv_residual_fraction"] == pytest.approx(
        res["adv_residual_fraction"], rel=1e-12)
