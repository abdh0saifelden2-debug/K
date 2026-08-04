"""Unit proofs for the NR39 real-data gate execution
(`general_two_clocks/nr39_eeg_gate.py`): the imaginary-coherence
volume-conduction lemma on real resting EEG (OpenNeuro ds003775).
Offline-safe -- all numbers come from the committed derived cache
``data/nr39_eeg_gate_cache.json`` (regeneration via ``build_cache()`` with
``$NR39_EDF`` set + mne + the biosemi64 montage).

Covered: the phase-randomization surrogate (PSD preservation + cross-phase
destruction) and alpha coherency on synthetic mixed data (the lemma
``Im S_x = 0`` for instantaneous mixing, and a lagged path lighting up Im);
cache schema; and every committed verdict on the real EEG (volume conduction
dominates the REAL part; mixing anti-correlates with imaginary coherency;
ImCoh is inter-regional; the surrogate ceiling inverts with mixing).
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from nr39_eeg_gate import (  # noqa: E402
    BAND, FIG, _alpha_coherency, _phase_randomize, analyze, load_cache,
)


@pytest.fixture(scope="module")
def cache():
    return load_cache()


@pytest.fixture(scope="module")
def out(cache):
    return analyze(cache)


# --------------------------------------------------------------------------- #
# the lemma, on synthetic data (no EEG needed)
# --------------------------------------------------------------------------- #
def _coh(data, fs=256.0):
    L = int(4 * fs)
    H = np.hanning(L)
    ws = list(range(0, data.shape[1] - L, L // 2))
    fr = np.fft.rfftfreq(L, 1.0 / fs)
    sel = (fr >= BAND[0]) & (fr <= BAND[1])
    return _alpha_coherency(data, fs, ws, sel, H)


def _alpha_sources(k, T, fs, rng):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="band")
    return filtfilt(b, a, rng.standard_normal((k, T)), axis=1)


def test_instantaneous_mixing_has_zero_imaginary_coherency():
    # lemma: x = A s with real A and real-spectrum s -> Im coherency ~ 0
    rng = np.random.default_rng(0)
    fs = 256.0
    s = _alpha_sources(3, int(240 * fs), fs, rng)
    A = rng.standard_normal((6, 3))              # instantaneous linear mix
    C = _coh(A @ s, fs)
    iu = np.triu_indices(6, 1)
    im = np.abs(C.imag)[iu]
    re = np.abs(C.real)[iu]
    assert np.median(re) > 0.3                    # mixing -> large real part
    assert np.median(re) / np.median(im) > 5.0    # Im at the floor, Re large


def test_lagged_path_lights_up_imaginary_part():
    # a frequency-independent (Hilbert) phase shift = a genuine lagged copy
    from scipy.signal import hilbert
    rng = np.random.default_rng(1)
    fs = 256.0
    s = _alpha_sources(2, int(240 * fs), fs, rng)
    x = np.zeros_like(s)
    x[0] = s[0]
    x[1] = np.imag(hilbert(s[0])) + 0.5 * s[1]   # 90 deg lagged copy
    C = _coh(x, fs)
    assert abs(C.imag[0, 1]) > 0.3               # imaginary coherency lights up
    assert abs(C.imag[0, 1]) > abs(C.real[0, 1])  # lagged -> Im dominates


def test_phase_randomize_preserves_psd_destroys_crossphase():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 4096))
    xs = _phase_randomize(x, rng)
    px = np.abs(np.fft.rfft(x, axis=1))
    pxs = np.abs(np.fft.rfft(xs, axis=1))
    assert np.allclose(px, pxs, atol=1e-8)       # per-channel PSD preserved


# --------------------------------------------------------------------------- #
# committed cache schema
# --------------------------------------------------------------------------- #
def test_cache_schema(cache):
    m = cache["meta"]
    assert m["n_channels"] == 64
    assert m["n_pairs"] == 64 * 63 // 2 == 2016
    assert m["band_hz"] == list(BAND)
    assert "ds003775" in m["dataset"]
    for key in ("abs_imcoh", "abs_recoh", "dist", "surrogate_p95"):
        assert len(cache[key]) == m["n_pairs"]
    assert len(cache["pairs"]["i"]) == m["n_pairs"]
    assert 0.0 < cache["surrogate_median_imcoh"] < 0.05


# --------------------------------------------------------------------------- #
# the committed verdicts on real EEG
# --------------------------------------------------------------------------- #
def test_volume_conduction_dominates_real_part(out):
    assert out["verdicts"]["volume_conduction_dominates_real"]
    assert out["re_over_im"] >= 4.0              # measured 8.2
    assert out["med_re"] > out["med_im"]


def test_mixing_feeds_real_not_imaginary(out):
    # the decisive lemma signature: stronger mixing -> LESS imaginary coherency
    assert out["verdicts"]["mixing_feeds_real_not_imaginary"]
    assert out["spearman_re_im"] <= 0.0          # measured -0.25
    assert out["topmix_re_over_im"] > out["re_over_im"]     # 28 > 8
    assert out["topmix_med_im"] < out["med_im"]  # top-mixed pairs: smallest Im


def test_imcoh_is_interregional(out):
    assert out["verdicts"]["imcoh_is_interregional"]
    assert out["spearman_dist_im"] > 0.0         # Im rises with distance
    assert out["spearman_dist_re"] < 0.0         # mixing (Re) falls with dist


def test_surrogate_ceiling_inverts_with_mixing(out):
    assert out["verdicts"]["surrogate_ceiling_inverts_with_mixing"]
    assert out["surrogate_median_imcoh"] < 0.03  # null floor
    # the most-mixed pairs are the LEAST likely to clear the ceiling
    assert out["frac_topmix_clearing"] < out["frac_clearing_ceiling"]


def test_committed_figure_artifact(cache, out):
    with open(FIG) as fh:
        fig = json.load(fh)
    assert fig["verdicts"] == out["verdicts"]
    assert fig["re_over_im"] == pytest.approx(out["re_over_im"])
