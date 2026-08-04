"""Unit tests for the closure module (Part 8b).

Tests the key structural properties of the SGS closure benchmark without
requiring a full DNS spinup (uses small grids and short runs).
"""
from __future__ import annotations

import numpy as np
import pytest

from closure.dns2d import Vorticity2D
from closure.sgs import (
    exact_sgs_force, smagorinsky_force, surrogate_force, projected_fdt_force,
    divergence_rms, force_spectrum, transfer_spectrum, sharp_filter,
)


@pytest.fixture(scope="module")
def dns_field():
    """Small DNS field (64x64, short spinup) for fast testing."""
    dns = Vorticity2D(n=64, seed=42, k_f=12.0)
    u, v = dns.field(steps=500, dt=2.5e-3)
    return dns.sp, u, v


def test_dns_produces_nonzero_field(dns_field):
    """DNS should produce a field with nonzero kinetic energy."""
    sp, u, v = dns_field
    ke = 0.5 * np.mean(u**2 + v**2)
    assert ke > 0, "DNS field has zero KE"


def test_dns_field_is_divergence_free(dns_field):
    """Vorticity-streamfunction DNS produces a divergence-free velocity field."""
    sp, u, v = dns_field
    div_rms = divergence_rms(sp, u, v)
    assert div_rms < 1e-10, f"DNS field divergence too large: {div_rms}"


def test_exact_sgs_force_is_solenoidal(dns_field):
    """The exact SGS force m_true should be divergence-free (Leray projected)."""
    sp, u, v = dns_field
    kc = 8
    ub, vb, mx, my = exact_sgs_force(sp, u, v, kc)
    div_rms = divergence_rms(sp, mx, my)
    assert div_rms < 1e-10, f"Exact SGS force divergence: {div_rms}"


def test_smagorinsky_is_solenoidal(dns_field):
    """Smagorinsky force should be solenoidal (Leray projected in implementation)."""
    sp, u, v = dns_field
    kc = 8
    ub, vb, _, _ = exact_sgs_force(sp, u, v, kc)
    mx, my = smagorinsky_force(sp, ub, vb, kc)
    div_rms = divergence_rms(sp, mx, my)
    assert div_rms < 1e-10, f"Smagorinsky divergence: {div_rms}"


def test_surrogate_breaks_solenoidality(dns_field):
    """Phase-randomized surrogate should NOT be solenoidal."""
    sp, u, v = dns_field
    kc = 8
    _, _, mx, my = exact_sgs_force(sp, u, v, kc)
    mxs, mys = surrogate_force(sp, mx, my, kc, seed=7)
    div_rms = divergence_rms(sp, mxs, mys)
    assert div_rms > 0.1, f"Surrogate should have large divergence, got: {div_rms}"


def test_projected_fdt_is_solenoidal(dns_field):
    """Projected-FDT force should be divergence-free by construction."""
    sp, u, v = dns_field
    kc = 8
    ub, vb, mx, my = exact_sgs_force(sp, u, v, kc)
    mxf, myf, _ = projected_fdt_force(sp, ub, vb, mx, my, kc, seed=7)
    div_rms = divergence_rms(sp, mxf, myf)
    assert div_rms < 1e-10, f"Projected-FDT divergence: {div_rms}"


def test_smagorinsky_is_purely_dissipative(dns_field):
    """Smagorinsky should have T(k) <= 0 for all k (no backscatter)."""
    sp, u, v = dns_field
    kc = 8
    ub, vb, _, _ = exact_sgs_force(sp, u, v, kc)
    mx, my = smagorinsky_force(sp, ub, vb, kc)
    k, T = transfer_spectrum(sp, ub, vb, mx, my, kc)
    assert np.all(T[k >= 1] <= 0), "Smagorinsky should be purely dissipative (T<=0)"


def test_sharp_filter_removes_high_modes(dns_field):
    """Sharp filter should zero out all modes above kc."""
    sp, u, v = dns_field
    kc = 8
    uf = sharp_filter(sp, u, kc)
    uf_h = sp.fft(uf)
    kr = np.sqrt(sp.k2)
    high = np.abs(uf_h[kr > kc + 0.5])
    assert np.allclose(high, 0), "Sharp filter did not remove high-k modes"


def test_force_spectrum_parseval(dns_field):
    """Force spectrum summed over all shells should approximate total |F|^2."""
    sp, u, v = dns_field
    kc = 8
    _, _, mx, my = exact_sgs_force(sp, u, v, kc)
    k, E = force_spectrum(sp, mx, my, kc)
    # force_spectrum sums |fft(m)|^2 over shells; total should equal sum of all |F|^2
    # For a kc-filtered field, all power is within shells 0..kc
    total_fft = float((np.abs(sp.fft(mx))**2 + np.abs(sp.fft(my))**2).sum())
    total_spectral = E.sum()
    ratio = total_spectral / (total_fft + 1e-30)
    assert 0.95 < ratio < 1.05, f"Parseval check failed: ratio={ratio}"
