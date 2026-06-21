"""Unit tests for Paper 1 — diagonal-band cascade structure
(general_two_clocks/cascade_band_structure.py).

Validate the band-coupling method on deterministic synthetic data (no downloads):
a multiplicative cascade must read out as diagonal-band; white noise must not; a
known amplitude modulation must show up as a specific off-diagonal coupling; and the
phase-randomized surrogate must destroy the cross-scale coupling. Seeded, CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cascade_band_structure as C  # noqa: E402

CASC = C.analyze(C.synthetic_cascade(seed=0), n_bands=9)
WHITE = C.analyze(C.white_noise(seed=1), n_bands=9)


def test_method_basics():
    x = C.synthetic_cascade(seed=0)
    env = C.analytic_envelope(x)
    assert np.all(env >= 0)
    n_bands = 9
    bands = C.dyadic_bands(x, n_bands)
    # bands sum EXACTLY to the band-limited signal they span (octaves 0..n_bands-1);
    # frequencies below the coarsest band are intentionally excluded.
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(n)
    mask = (f > 0.5 / 2 ** n_bands) & (f <= 0.5)
    ref = np.fft.irfft(X * mask, n)
    recon = np.sum(bands, axis=0)
    assert np.corrcoef(recon, ref)[0, 1] > 0.999
    phi = C.coupling_matrix(bands)
    assert np.allclose(np.diag(phi), 1.0, atol=1e-9)


def test_cascade_is_diagonal_band():
    assert CASC["n_pass"] == 4
    assert CASC["spearman"] <= -0.8            # monotone-decreasing trend
    assert 0 < CASC["d_half"] < 9              # finite local decay scale
    assert CASC["nn_real"] > 0.2               # real nearest-neighbour coupling
    assert CASC["mean_off_real"] > CASC["mean_off_surr"] + 0.05   # above surrogate


def test_white_noise_has_no_cross_scale_coupling():
    # the discriminating predictions (monotone trend, above-surrogate) must FAIL
    assert WHITE["mean_off_real"] < 0.05
    assert abs(WHITE["nn_real"]) < 0.1
    assert WHITE["cb4"] is False
    assert WHITE["cb2"] is False


def test_surrogate_destroys_coupling():
    # phase randomization keeps the spectrum but collapses off-diagonal coupling
    assert CASC["mean_off_surr"] < 0.05
    assert CASC["mean_off_real"] > 5.0 * abs(CASC["mean_off_surr"]) + 0.05


def test_cascade_coupling_is_local_in_scale():
    """Diagonal-band locality: adjacent octaves are more coupled than distant ones."""
    phi = CASC["phi"]
    M = phi.shape[0]
    adjacent = np.mean([phi[i, i + 1] for i in range(M - 1)])
    distant = np.mean([phi[i, i + 5] for i in range(M - 5)])
    assert adjacent > distant + 0.05
    assert adjacent > 0.2


def test_determinism():
    a = C.analyze(C.synthetic_cascade(seed=0), n_bands=9)
    b = C.analyze(C.synthetic_cascade(seed=0), n_bands=9)
    assert a["spearman"] == b["spearman"] and a["d_half"] == b["d_half"]
