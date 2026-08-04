"""Unit proofs for NR51 (`general_two_clocks/new_relationships28.py`):
closed spectral polygons are shape, open ones are motion.  Triad
correlators are exactly boost-blind, Im(bispectrum) is the parity-odd
shape coordinate (Elgar-Guza asymmetry), bulk skewness/asymmetry are exact
closed-triad sums, and on the real Bushuk scallops the parity breaking is
decisive in the flux quadrature but weak/unstable in the stored shape.
Offline-safe: committed .mat + committed flux cache.

Covered: boost-blindness of the biphase under raw circular shifts and
spectral boosts (machine precision) and the documented NON-covariance of
linear detrending; exact mirror flip (parity-odd); the mod-N moment sum
rules on synthetics and on every data frame; the leaning/skewed-symmetric
positive controls; and the real-data flux-vs-shape parity verdict.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships28 import (  # noqa: E402
    DERIVED, FIG, MAT, analyze, biphase_per_frame_deg, bispectrum,
    circular_stats_deg, clean_bins, detrend, dft_hilbert, load_frames,
    rfft_frames, run, synthetic_checks, triad_moment_sums,
)


@pytest.fixture(scope="module")
def syn():
    return synthetic_checks()


@pytest.fixture(scope="module")
def result():
    return analyze()


def _lean(n=512, amp=0.3, phi=np.pi / 2):
    th = 2.0 * np.pi * np.arange(n) / n
    return np.cos(th) + amp * np.cos(2 * th - phi)


# --------------------------------------------------------------------------- #
# boost-blindness (closed polygon) and its detrend caveat
# --------------------------------------------------------------------------- #
def test_biphase_boost_blind_raw(syn):
    assert syn["boost_biphase_spread_deg"] < 1e-9


def test_biphase_blind_to_spectral_boost():
    h = _lean()
    H = rfft_frames([h, h, h], window=False, detrend_frames=False)
    k = 2 * np.pi * np.fft.rfftfreq(len(h), 1.0)
    Hs = H * np.exp(-1j * k[None, :] * np.array([[3.7], [-11.2], [0.4]]))
    b0 = biphase_per_frame_deg(H, 1, 1)
    b1 = biphase_per_frame_deg(Hs, 1, 1)
    assert np.max(np.abs(b0 - b1)) < 1e-9


def test_pair_phase_carries_boost_but_triad_does_not():
    # open polygon (single mode phase) moves under a shift; closed does not
    h = _lean()
    F0 = np.fft.rfft(h)
    F1 = np.fft.rfft(np.roll(h, 41))
    assert abs(np.angle(F1[1]) - np.angle(F0[1])) > 0.1
    b0 = np.angle(F0[1] * F0[1] * np.conj(F0[2]))
    b1 = np.angle(F1[1] * F1[1] * np.conj(F1[2]))
    assert abs(b1 - b0) < 1e-9


def test_linear_detrend_is_not_boost_covariant():
    # documented caveat: the discrete ramp projects on every Fourier mode,
    # so linear detrending breaks the exact shift invariance
    h = _lean()
    b = [np.degrees(np.angle(np.fft.rfft(detrend(np.roll(h, s)))[1] ** 2
                             * np.conj(np.fft.rfft(
                                 detrend(np.roll(h, s)))[2])))
         for s in (0, 37)]
    assert abs(b[1] - b[0]) > 1.0


# --------------------------------------------------------------------------- #
# parity: mirror flips the biphase exactly
# --------------------------------------------------------------------------- #
def test_mirror_flips_biphase(syn):
    assert syn["mirror_flip_err_deg"] < 1e-9


def test_mirror_flips_asymmetry_sign():
    h = clean_bins(_lean())
    a = np.mean(dft_hilbert(h) ** 3)
    am = np.mean(dft_hilbert(h[::-1]) ** 3)
    assert a == pytest.approx(-am, rel=1e-9)


# --------------------------------------------------------------------------- #
# exact moment sum rules
# --------------------------------------------------------------------------- #
def test_sum_rules_synthetic(syn):
    for name in ("leaning", "skewed_symmetric"):
        assert syn[name]["sum_rule_re_err"] < 1e-12
        assert syn[name]["sum_rule_im_err"] < 1e-12


def test_sum_rules_random_field():
    rng = np.random.default_rng(3)
    h = clean_bins(rng.standard_normal(128))
    sr, si = triad_moment_sums(h)
    assert np.mean(h ** 3) == pytest.approx(sr, abs=1e-12)
    assert np.mean(dft_hilbert(h) ** 3) == pytest.approx(si, abs=1e-12)


def test_sum_rules_on_every_data_frame(result):
    assert result["data_sum_rule_re_max_err"] < 1e-9
    assert result["data_sum_rule_im_max_err"] < 1e-9


# --------------------------------------------------------------------------- #
# the detector separates the two parities
# --------------------------------------------------------------------------- #
def test_leaning_has_asymmetry_symmetric_does_not(syn):
    assert abs(syn["leaning"]["asymmetry"]) > 0.5
    assert abs(syn["skewed_symmetric"]["asymmetry"]) < 1e-12
    assert abs(syn["skewed_symmetric"]["skewness"]) > 0.5


def test_asymmetry_sign_tracks_lean_direction():
    hp = clean_bins(_lean(phi=+np.pi / 2))
    hm = clean_bins(_lean(phi=-np.pi / 2))
    ap = np.mean(dft_hilbert(hp) ** 3)
    am = np.mean(dft_hilbert(hm) ** 3)
    assert ap == pytest.approx(-am, rel=1e-9) and ap != 0.0


# --------------------------------------------------------------------------- #
# real data: flux breaks parity, shape barely does
# --------------------------------------------------------------------------- #
def test_committed_sources(result):
    assert os.path.exists(MAT) and os.path.exists(DERIVED)
    assert result["n_frames"] == 12


def test_flux_decisive_shape_weak(result):
    assert abs(result["t_flux_quadrature"]) > 4.0
    assert (abs(result["t_shape_asymmetry"])
            < 0.6 * abs(result["t_flux_quadrature"]))
    assert result["biphase_frame_circ_std_deg"] > 30.0
    assert result["verdicts"]["scallops_break_parity_in_flux_not_shape"]


def test_data_boost_and_mirror_exact(result):
    assert result["boost_blind_err_deg"] < 1e-6
    assert result["mirror_flip_err_deg"] < 1e-6


def test_all_verdicts_and_figure(result):
    for key in ("closed_polygons_boost_blind_and_parity_odd",
                "asymmetry_detector_separates_parities",
                "scallops_break_parity_in_flux_not_shape"):
        assert result["verdicts"][key] is True
    res = run(write=True)
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        disk = json.load(fh)
    assert disk["verdicts"][
        "scallops_break_parity_in_flux_not_shape"] is True
    assert disk["t_flux_quadrature"] == pytest.approx(
        res["t_flux_quadrature"], rel=1e-12)
