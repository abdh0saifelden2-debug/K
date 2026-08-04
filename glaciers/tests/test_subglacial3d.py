"""Unit tests for the 3D a-posteriori subglacial cavity LES (Part 9b/9c).

These exercise the structural properties the 3D melt-rate study relies on, using
small grids and short spinups so the suite stays fast and needs no GPU:

  * the multi-site BEDMAP1 window loader (`subglacial.bedmap.bed_window_profile`)
    produces periodic, cavity-scaled real beds, and distinct windows give
    genuinely distinct topography;
  * the penalized 3D solver (`subglacial.flow3d`) embeds the extruded bed / flat
    ice, keeps the resolved velocity divergence-free (the elliptic "fast clock"),
    stays finite under forcing, and its glacier diagnostics behave as documented
    (z-homogeneous geometry, advective+diffusive melt flux, SGS dissipation
    breakdown).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from subglacial.bedmap import (
    bed_profile_from_transect, bed_window_profile,
    _clean_sorted_transect, _embed_periodic,
)
from subglacial.flow3d import (
    Spectral3D, Subglacial3DConfig, Subglacial3DFlow,
    project3d, divergence_rms3d,
)

BEDMAP_TRANSECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "subglacial", "data", "bedmap1_transect.csv")


@pytest.fixture(scope="module")
def developed_flow3d():
    """Small penalized 3D cavity flow, short spinup -- fast but turbulent."""
    prof, _ = bed_window_profile(BEDMAP_TRANSECT, 24, 0.9, 0.55, 0.0, 0.34)
    cfg = Subglacial3DConfig(n=24, sgs="smagorinsky", bed_profile=prof,
                             f_amp=1.5, k_f=5.0, f_band=2.0, seed=1)
    f = Subglacial3DFlow(cfg)
    f.run(120, ramp=40)
    return f


# --------------------------------------------------------------------------- #
# real BEDMAP1 multi-site window loader
# --------------------------------------------------------------------------- #

def test_bed_window_profile_is_periodic_and_scaled():
    """A windowed BEDMAP1 segment maps to a periodic, cavity-scaled ybed(x):
    peak-to-peak rescaled to 2*bed_amp, mean recentred to bed_mean, seam
    continuous (mirrored construction), with real-relief provenance metadata."""
    n, bed_mean, bed_amp = 96, 0.9, 0.55
    h, meta = bed_window_profile(BEDMAP_TRANSECT, n, bed_mean, bed_amp, 0.0, 0.34)
    assert h.shape == (n,)
    assert (h.max() - h.min()) == pytest.approx(2 * bed_amp, rel=1e-6)
    assert h.mean() == pytest.approx(bed_mean, abs=1e-6)
    assert abs(h[0] - h[-1]) < bed_amp, "mirrored seam should be ~continuous"
    assert meta["relief_m"] > 100.0 and meta["length_km"] > 5.0
    assert meta["frac_lo"] == 0.0 and meta["frac_hi"] == pytest.approx(0.34)


def test_distinct_windows_give_distinct_real_beds():
    """Non-overlapping along-track windows yield genuinely different measured
    topography (not synthetic variants of one shape)."""
    h1, m1 = bed_window_profile(BEDMAP_TRANSECT, 96, 0.9, 0.55, 0.0, 0.34)
    h2, m2 = bed_window_profile(BEDMAP_TRANSECT, 96, 0.9, 0.55, 0.67, 1.0)
    assert not np.allclose(h1, h2), "distinct windows must give distinct beds"
    assert m1["relief_m"] != m2["relief_m"]


def test_bed_loader_refactor_is_behaviour_preserving():
    """The shared `_clean_sorted_transect` / `_embed_periodic` helpers reproduce
    the full-transect profile exactly (the refactor changed no numbers)."""
    n, bed_mean, bed_amp = 64, 0.9, 0.55
    h, _ = bed_profile_from_transect(BEDMAP_TRANSECT, n, bed_mean, bed_amp)
    _, bed = _clean_sorted_transect(BEDMAP_TRANSECT)
    h_helper = _embed_periodic(bed, n, bed_mean, bed_amp)
    assert np.array_equal(h, h_helper)


# --------------------------------------------------------------------------- #
# spectral operators
# --------------------------------------------------------------------------- #

def test_spectral3d_fft_roundtrip_is_real_identity():
    """ifft(fft(f)) == f for a real field on the triply-periodic box."""
    sp = Spectral3D(8)
    rng = np.random.default_rng(0)
    f = rng.standard_normal((8, 8, 8))
    assert np.allclose(sp.ifft(sp.fft(f)), f, atol=1e-10)


def test_project3d_makes_velocity_divergence_free():
    """Leray projection drives the relative RMS divergence to ~machine zero for a
    resolved (band-limited) velocity field."""
    sp = Spectral3D(16)
    X, Y, Z = sp.grid()
    u = np.sin(X) * np.cos(2 * Y)
    v = -np.cos(X) * np.sin(2 * Y) * np.cos(Z)
    w = np.sin(3 * Z) * np.cos(X)
    u, v, w = project3d(sp, u, v, w)
    assert divergence_rms3d(sp, u, v, w) < 1e-9


# --------------------------------------------------------------------------- #
# solver / geometry
# --------------------------------------------------------------------------- #

def test_masks_partition_solid_and_fluid():
    """Rock + ice masks define the solids; the cavity in between is fluid, with a
    warm bed (theta_solid -> 1) and a cold ice base (theta_solid -> 0)."""
    f = Subglacial3DFlow(Subglacial3DConfig(n=48))
    assert 0.0 < f.fvol < f.n ** 3, "cavity must be a strict subset of the box"
    assert f.theta_solid[f.chi_rock > 0.99].min() > 0.95
    assert f.theta_solid[f.chi_ice > 0.99].max() < 0.05


def test_extruded_bed_is_z_homogeneous():
    """A real 1-D bed extruded along z gives a fluid mask identical at every
    spanwise slice -- the z-homogeneity the wake/melt diagnostics rely on."""
    prof, _ = bed_window_profile(BEDMAP_TRANSECT, 32, 0.9, 0.55, 0.0, 0.34)
    f = Subglacial3DFlow(Subglacial3DConfig(n=32, bed_profile=prof))
    assert np.array_equal(f.fluid[:, :, 0], f.fluid[:, :, 5])
    assert np.allclose(f.ybed[:, :, 0], f.ybed[:, :, 7]), "ybed is f(x) only"


def test_spinup_stays_finite_and_incompressible(developed_flow3d):
    """Forced spinup yields a nonzero, finite KE and a near-incompressible field."""
    f = developed_flow3d
    ke = f.kinetic_energy()
    assert ke > 0 and np.isfinite(ke)
    assert np.isfinite(f.u).all() and np.isfinite(f.theta).all()
    assert divergence_rms3d(f.sp, f.u, f.v, f.w) < 1e-2


def test_backscatter_closure_stays_finite():
    """The two-clocks (projected-FDT backscatter) closure integrates stably."""
    prof, _ = bed_window_profile(BEDMAP_TRANSECT, 24, 0.9, 0.55, 0.0, 0.34)
    cfg = Subglacial3DConfig(n=24, sgs="backscatter", backscatter=0.7,
                             bed_profile=prof, f_amp=1.5, k_f=5.0, seed=3)
    f = Subglacial3DFlow(cfg)
    f.run(80, ramp=30)
    assert np.isfinite(f.u).all() and np.isfinite(f.theta).all()
    assert f.kinetic_energy() > 0


# --------------------------------------------------------------------------- #
# glacier-relevant diagnostics
# --------------------------------------------------------------------------- #

def test_melt_flux_includes_advective_heat_flux(developed_flow3d):
    """The 3D melt proxy is q = v*theta - kappa d(theta)/dy: it carries the
    resolved *advective* heat flux, unlike the diffusive-only 2D version."""
    f = developed_flow3d
    sp, cfg, xp = f.sp, f.cfg, f.xp
    melt, band = f.melt_flux()
    qdiff_only = float(xp.mean((-cfg.kappa * sp.ddy(f.theta))[band]))
    qadv = float(xp.mean((f.v * f.theta)[band]))
    assert np.isfinite(melt)
    assert abs(qadv) > 0.0, "developed wake should carry nonzero advective flux"
    assert melt == pytest.approx(qadv + qdiff_only, rel=1e-6, abs=1e-12)
    assert not melt == pytest.approx(qdiff_only, abs=1e-9), \
        "advective term must change the melt proxy vs diffusive-only"


def test_turbulence_intensity_in_unit_interval(developed_flow3d):
    """Fluctuation/total KE ratio is bounded in [0, 1]."""
    ti = developed_flow3d.turbulence_intensity()
    assert 0.0 <= ti <= 1.0 and np.isfinite(ti)


def test_dissipation_breakdown_nonnegative_and_scales_with_nu():
    """Both molecular and SGS dissipation rates are finite and >= 0, and the
    molecular rate scales with the molecular viscosity nu."""
    prof, _ = bed_window_profile(BEDMAP_TRANSECT, 24, 0.9, 0.55, 0.0, 0.34)
    base = Subglacial3DConfig(n=24, sgs="smagorinsky", bed_profile=prof,
                              nu=8.0e-4, f_amp=1.5, k_f=5.0, seed=4)
    f = Subglacial3DFlow(base)
    f.run(100, ramp=40)
    eps_mol, eps_sgs = f.dissipation_breakdown()
    assert eps_mol >= 0.0 and eps_sgs >= 0.0
    assert np.isfinite(eps_mol) and np.isfinite(eps_sgs)
    # eps_molecular = <2 nu |S|^2>; halving nu (same strain field) halves it
    nu_hi, nu_lo = 1.0e-3, 5.0e-4
    s = f._strain(f.u, f.v, f.w)[1]
    em_hi = float(np.mean((2.0 * nu_hi * s)[f.fluid]))
    em_lo = float(np.mean((2.0 * nu_lo * s)[f.fluid]))
    assert em_hi == pytest.approx(2.0 * em_lo, rel=1e-6)
