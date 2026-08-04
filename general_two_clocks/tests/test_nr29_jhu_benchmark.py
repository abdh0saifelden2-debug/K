"""Offline unit proofs for the NR29 JHU benchmark machinery (no network)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nr29_jhu_benchmark import nr29_check, pi_field, sharp_filter  # noqa: E402


def _solenoidal_gaussian(n=32, seed=0, slope=-5.0 / 3.0):
    """Random solenoidal periodic velocity with a Kolmogorov-ish spectrum."""
    rng = np.random.default_rng(seed)
    k1 = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    kmag = np.sqrt(np.maximum(k2, 1e-12))
    amp = np.where(k2 > 0, kmag ** ((slope - 2.0) / 2.0), 0.0)
    u_hat = []
    for _ in range(3):
        ph = rng.normal(size=(n, n, n)) + 1j * rng.normal(size=(n, n, n))
        u_hat.append(amp * ph)
    # Leray-project to solenoidal
    kdotu = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, kk in enumerate((kx, ky, kz)):
            u_hat[i] = u_hat[i] - np.where(k2 > 0, kk * kdotu / k2, 0.0)
    u = np.stack([np.real(np.fft.ifftn(h)) for h in u_hat])
    return u / np.std(u)


def test_sharp_filter_kills_high_k_exactly():
    n = 16
    k1 = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    f = np.zeros((n, n, n), complex)
    f[1, 0, 0] = 1.0        # k=1 mode
    f[6, 0, 0] = 1.0        # k=6 mode
    g = sharp_filter(f, kc=3, k2=k2)
    assert g[1, 0, 0] == 1.0 and g[6, 0, 0] == 0.0


def test_pi_field_is_zero_when_filter_is_identity():
    """kc above Nyquist: filtered field == field, tau == 0, Pi == 0."""
    u = _solenoidal_gaussian(n=16, seed=1)
    pi = pi_field(u, kc=100)
    assert np.max(np.abs(pi)) < 1e-10


def test_pi_field_galilean_invariance():
    """Adding a uniform velocity must not change Pi (tau and grad-u invariant)."""
    u = _solenoidal_gaussian(n=16, seed=2)
    pi0 = pi_field(u, kc=4)
    u2 = u + np.array([1.7, -0.6, 0.9])[:, None, None, None]
    pi1 = pi_field(u2, kc=4)
    assert np.allclose(pi0, pi1, atol=1e-9)


def test_nr29_check_on_gaussian_synthetic_field():
    """For a genuinely Gaussian Pi sample the law is exact by construction."""
    rng = np.random.default_rng(3)
    pi = rng.normal(0.3, 1.0, size=200_000)
    r = nr29_check(pi)
    assert r["abs_err"] < 5e-3
    assert r["forward_cascade"] is True


def test_nr29_check_ktheory_limit():
    """A strictly positive Pi field: measured fraction 0, law -> ~0."""
    pi = np.abs(np.random.default_rng(4).normal(5.0, 0.5, 10_000))
    r = nr29_check(pi)
    assert r["backscatter_fraction"] == 0.0
    assert r["nr29_closed_form"] < 1e-8


def test_pi_field_tracks_law_on_synthetic_turbulence():
    """The random-phase solenoidal field is near-Gaussian: NR29 should track
    within a few percent at a mid-band kc (this is the offline stand-in for
    the JHU run; the committed artifact carries the real-data numbers)."""
    u = _solenoidal_gaussian(n=32, seed=5)
    pi = pi_field(u, kc=6)
    r = nr29_check(pi)
    assert 0.0 < r["backscatter_fraction"] < 1.0
    assert r["abs_err"] < 0.05


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
