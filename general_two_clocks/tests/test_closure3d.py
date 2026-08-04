"""Unit tests for the 3D closure module (Part 9c).

Tests the structural properties of the 3D SGS closure benchmark on a small grid
plus the 3D-specific physics (vortex stretching, backscatter volume fraction) that
the 2D Part-9b test structurally cannot exercise.  CPU only (NumPy backend).
"""
from __future__ import annotations

import numpy as np
import pytest

from closure.dns3d import ForcedNS3D, DNS3DConfig
from closure.spectral3d import Spectral3D, divergence_rms3d, sharp_filter3d
from closure.sgs3d import (
    exact_sgs_force3d, smagorinsky_force3d, surrogate_force3d, projected_fdt_force3d,
    force_spectrum3d, transfer_spectrum3d, enstrophy_production,
    strain_vorticity_alignment, sgs_flux_stats,
)

KC = 6


@pytest.fixture(scope="module")
def dns_field():
    """Small developed 3D DNS field (32^3, short spinup) for fast testing."""
    dns = ForcedNS3D(DNS3DConfig(n=32, seed=3), xp=np)
    dns.run(300)
    u, v, w = dns.velocity()
    return dns.sp, u, v, w


def test_dns_produces_nonzero_field(dns_field):
    sp, u, v, w = dns_field
    ke = 0.5 * np.mean(u ** 2 + v ** 2 + w ** 2)
    assert ke > 0


def test_dns_field_is_nearly_divergence_free(dns_field):
    sp, u, v, w = dns_field
    # 3D velocity-form DNS: solenoidal up to Nyquist-mode FFT round-off
    assert divergence_rms3d(sp, u, v, w) < 1e-3


def test_exact_sgs_force_is_solenoidal(dns_field):
    sp, u, v, w = dns_field
    _, _, _, mx, my, mz = exact_sgs_force3d(sp, u, v, w, KC)
    assert divergence_rms3d(sp, mx, my, mz) < 1e-9


def test_smagorinsky_is_solenoidal(dns_field):
    sp, u, v, w = dns_field
    ub, vb, wb, *_ = exact_sgs_force3d(sp, u, v, w, KC)
    mx, my, mz = smagorinsky_force3d(sp, ub, vb, wb, KC)
    assert divergence_rms3d(sp, mx, my, mz) < 1e-9


def test_surrogate_breaks_solenoidality(dns_field):
    sp, u, v, w = dns_field
    _, _, _, mx, my, mz = exact_sgs_force3d(sp, u, v, w, KC)
    mrx, mry, mrz = surrogate_force3d(sp, mx, my, mz, KC, seed=7)
    assert divergence_rms3d(sp, mrx, mry, mrz) > 0.1


def test_projected_fdt_is_solenoidal(dns_field):
    sp, u, v, w = dns_field
    ub, vb, wb, mx, my, mz = exact_sgs_force3d(sp, u, v, w, KC)
    mfx, mfy, mfz, _ = projected_fdt_force3d(sp, ub, vb, wb, mx, my, mz, KC, seed=7)
    assert divergence_rms3d(sp, mfx, mfy, mfz) < 1e-9


def test_smagorinsky_is_purely_dissipative(dns_field):
    sp, u, v, w = dns_field
    ub, vb, wb, *_ = exact_sgs_force3d(sp, u, v, w, KC)
    mx, my, mz = smagorinsky_force3d(sp, ub, vb, wb, KC)
    k, T = transfer_spectrum3d(sp, ub, vb, wb, mx, my, mz, KC)
    assert np.all(T[k >= 1] <= 1e-9), "Smagorinsky must be purely dissipative (T<=0)"


def test_projected_fdt_tracks_true_transfer(dns_field):
    sp, u, v, w = dns_field
    ub, vb, wb, mx, my, mz = exact_sgs_force3d(sp, u, v, w, KC)
    mfx, mfy, mfz, _ = projected_fdt_force3d(sp, ub, vb, wb, mx, my, mz, KC, seed=7)
    k, Tt = transfer_spectrum3d(sp, ub, vb, wb, mx, my, mz, KC)
    _, Tf = transfer_spectrum3d(sp, ub, vb, wb, mfx, mfy, mfz, KC)
    m = k >= 1
    corr = np.corrcoef(Tt[m], Tf[m])[0, 1]
    assert corr > 0.6, f"projected-FDT should reproduce the shell transfer, corr={corr}"


def test_sharp_filter_removes_high_modes(dns_field):
    sp, u, v, w = dns_field
    uf = sharp_filter3d(sp, u, KC)
    high = np.abs(sp.fft(uf)[sp.kmag > KC + 0.5])
    assert np.allclose(high, 0)


def test_force_spectrum_parseval(dns_field):
    sp, u, v, w = dns_field
    _, _, _, mx, my, mz = exact_sgs_force3d(sp, u, v, w, KC)
    k, E = force_spectrum3d(sp, mx, my, mz, KC)
    total_fft = float((np.abs(sp.fft(mx)) ** 2 + np.abs(sp.fft(my)) ** 2
                       + np.abs(sp.fft(mz)) ** 2).sum())
    assert 0.95 < E.sum() / (total_fft + 1e-30) < 1.05


# --------------------------------------------------------------------------- #
# 3D-only physics: the whole reason Part 9c exists
# --------------------------------------------------------------------------- #

def test_vortex_stretching_zero_for_2d_flow():
    """A z-invariant, w=0 flow has omega_i S_ij omega_j == 0 identically -- the
    structural fact that makes the 2D benchmark blind to the stretching dynamics."""
    sp = Spectral3D(24, xp=np)
    X, Y, Z = sp.grid()
    u = np.cos(Y)            # u = u(y), v = v(x), w = 0  -> divergence-free, 2D
    v = np.sin(X)
    w = np.zeros_like(u)
    mean_prod, norm_prod = enstrophy_production(sp, u, v, w)
    assert abs(mean_prod) < 1e-10, f"2D flow must have zero stretching, got {mean_prod}"


def test_vortex_stretching_nonzero_and_positive_in_3d(dns_field):
    """The 3D DNS field has net positive vortex-stretching production (the forward
    cascade engine) -- nonzero, unlike any 2D flow."""
    sp, u, v, w = dns_field
    mean_prod, norm_prod = enstrophy_production(sp, u, v, w)
    assert abs(norm_prod) > 1e-3, "3D field must have nonzero stretching production"
    assert mean_prod > 0, "net vortex stretching should be positive (forward cascade)"


def test_smagorinsky_has_zero_backscatter_fraction(dns_field):
    """K-theory's positive-definite eddy viscosity has Pi = -tau:S >= 0 everywhere
    (zero backscatter volume), while real 3D turbulence has a large fraction."""
    sp, u, v, w = dns_field
    flux = sgs_flux_stats(sp, u, v, w, KC)
    assert flux["backscatter_fraction_smag"] == 0.0
    assert flux["backscatter_fraction_true"] > 0.05


def test_strain_vorticity_alignment_valid(dns_field):
    sp, u, v, w = dns_field
    al = strain_vorticity_alignment(sp, u, v, w, max_points=20000)
    for key in ("cos_extensional", "cos_intermediate", "cos_compressive"):
        assert 0.0 <= al[key] <= 1.0
    assert al["n_points"] > 0
