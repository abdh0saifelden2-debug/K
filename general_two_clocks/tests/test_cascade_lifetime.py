"""Unit tests for Paper 3 — cascade lifetime distribution & power-law energy transport
(general_two_clocks/cascade_lifetime.py).

Validate the heavy-tail fitter and event machinery on deterministic synthetic data
(no downloads): power-law in -> power-law preferred + correct exponent; exponential
in -> power-law rejected; known excursions -> correct durations; E=τ^2 -> β=2.
Seeded, CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cascade_lifetime as L  # noqa: E402


def test_fitter_recovers_powerlaw():
    d = L.powerlaw_samples(n=5000, alpha=2.5, xmin=2.0, seed=0)
    ht = L.heavy_tail(d)
    assert ht is not None
    assert abs(ht["alpha"] - 2.5) < 0.2          # exponent recovered
    assert ht["pl_preferred"] is True            # PL beats exponential
    assert ht["d_aic"] > 0


def test_fitter_rejects_powerlaw_for_exponential():
    d = L.exponential_samples(n=5000, scale=5.0, xmin=2.0, seed=0)
    ht = L.heavy_tail(d)
    assert ht is not None
    assert ht["pl_preferred"] is False           # exponential data -> PL not preferred


def test_event_detection_counts_and_durations():
    # activity with three explicit rectangular excursions above thresh=0.5
    a = np.zeros(100)
    a[10:15] = 1.0   # duration 5
    a[30:33] = 1.0   # duration 3
    a[50:60] = 1.0   # duration 10
    dur, en = L.detect_events(a, 0.5)
    assert sorted(dur.tolist()) == [3.0, 5.0, 10.0]
    # excess energy = (1-0.5)*duration
    assert np.isclose(sorted(en.tolist()), [1.5, 2.5, 5.0]).all()


def test_energy_duration_scaling_recovers_beta():
    dur = np.arange(2, 200, 1.0)
    en = dur ** 2.0                              # E = τ^2 exactly
    ed = L.energy_duration(dur, en, xmin=2.0)
    assert abs(ed["beta"] - 2.0) < 1e-6
    assert ed["R2"] > 0.999


def test_xmin_selection_reasonable():
    d = L.powerlaw_samples(n=5000, alpha=2.5, xmin=5.0, seed=1)
    ks, xmin, alpha = L.select_xmin(d)
    assert xmin >= 2.0 and np.isfinite(alpha)
    assert ks < 0.1                              # good KS fit for true power law


def test_build_activity_nonnegative():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(2 ** 14))   # red-ish noise
    act = L.build_activity(x, k_fine=4, n_bands=10)
    assert np.all(act >= 0) and len(act) == len(x)
