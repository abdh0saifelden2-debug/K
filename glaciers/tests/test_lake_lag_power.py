"""Tests for the matched-lag power analysis (P4a-R2)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "validation", "synthetic"))

import lake_lag_power as lp

ART = os.path.join(HERE, "validation", "reports", "lake_lag_power.json")


def test_kernel_peak_normalised():
    t = np.linspace(0, 3, 4000)
    for tau in (0.05, 0.25, 1.0):
        k = lp.kernel(t, tau)
        assert k.max() == pytest.approx(1.0, abs=1e-3)
        assert k[0] == 0.0


def test_aliasing_attenuates_short_transients():
    # a tau=0.1 yr transient loses most of its peak in an annual bin,
    # less in a quarterly bin; a sustained step is aliasing-immune
    q = lp.binned_signal(1.0, 0.1, 0.25).max()
    a = lp.binned_signal(1.0, 0.1, 1.0).max()
    assert a < q < 1.0
    assert a < 0.35
    s = lp.sustained_signal(1.0, 0.25)
    assert s[-1] == pytest.approx(1.0)


def test_calibrated_threshold_exceeds_nominal():
    k5, null_peaks = lp.calibrated_threshold(0.023, 0.25, n_mc=1200, seed=3)
    # max over 8 post bins: nominal 2-sigma is multiple-comparison inflated
    assert k5 > 2.3
    assert (null_peaks >= 2.0).mean() > 0.10
    # observed +0.56 sigma sits below the median noise maximum
    assert np.median(null_peaks) > 0.56


def test_power_monotone_in_amplitude():
    kt, _ = lp.calibrated_threshold(0.023, 0.25, n_mc=800, seed=5)
    p_small = (lp.mc_peaks(lp.sustained_signal(0.01, 0.25), 0.023, 0.25,
                           n_mc=600, seed=7) >= kt).mean()
    p_large = (lp.mc_peaks(lp.sustained_signal(0.15, 0.25), 0.023, 0.25,
                           n_mc=600, seed=7) >= kt).mean()
    assert p_large > 0.95 > p_small


@pytest.fixture(scope="module")
def art():
    with open(ART) as fh:
        return json.load(fh)


def test_artifact_consistency(art):
    fp = art["false_positive"]["quarterly_per_lake"]
    assert fp["nominal_2sigma_fp_rate"] > 0.10
    assert fp["median_noise_peak_sigma"] > art["observed_peak_sigma"]
    tbl = art["a95_fraction_of_trunk_speed"]
    q, s5, g = (tbl[k] for k in ("quarterly_per_lake", "quarterly_stacked5",
                                 "gps_daily"))
    # stacking beats per-lake; GPS beats both for the sub-annual transient
    assert s5["sustained"] < q["sustained"]
    assert g["transient_tau_0.1"] < s5["transient_tau_0.1"] < q["transient_tau_0.1"]
    # ITS_LIVE quarterly cannot see a 0.1-yr transient below ~30% of trunk speed
    assert q["transient_tau_0.1"] > 0.2
    # GPS route detects a few-percent transient
    assert g["transient_tau_0.1"] < 0.05
