"""Unit tests for the subglacial cavity closure test (Part 9).

These exercise the structural properties the Part-9 benchmark relies on, using
small grids and short spinups so the suite stays fast and needs no data:

  * the penalized solver embeds the bumpy bed / flat ice and drives a cavity flow
    that stays divergence-free (the elliptic "fast clock" is respected);
  * the glacier-relevant diagnostics in ``subglacial.diag`` behave as advertised
    -- in particular the K-theory (eddy-diffusivity) heat flux is *structurally*
    down-gradient, so it can never represent the counter-gradient lee heat
    trapping that the exact subgrid flux exhibits.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from compressible.ns import Spectral2D
from closure.sgs import divergence_rms, transfer_spectrum, exact_sgs_force
from subglacial.flow import SubglacialFlow, SubglacialConfig
from subglacial.diag import (
    spatial_transfer, exact_sgs_heat_flux, eddy_diffusivity_heat_flux,
    heat_flux_divergence, countergradient_fraction, backscatter_fraction,
    masked_corr,
)


@pytest.fixture(scope="module")
def developed_flow():
    """Small penalized cavity flow, short spinup -- fast but turbulent."""
    cfg = SubglacialConfig(n=48, sgs="none", f_amp=1.5, k_f=8.0, f_band=2.0, seed=1)
    f = SubglacialFlow(cfg)
    f.run(800, ramp=400)
    return f


# --------------------------------------------------------------------------- #
# solver / geometry
# --------------------------------------------------------------------------- #

def test_masks_partition_solid_and_fluid():
    """Rock + ice masks define the solids; the cavity in between is fluid."""
    f = SubglacialFlow(SubglacialConfig(n=48))
    assert 0.0 < f.fvol < f.n * f.n, "cavity must be a strict subset of the box"
    # the heat penalty target is warm in the rock (theta_solid -> 1) and cold in
    # the ice (theta_solid -> 0)
    assert f.theta_solid[f.chi_rock > 0.99].min() > 0.95
    assert f.theta_solid[f.chi_ice > 0.99].max() < 0.05


def test_penalty_field_proportional_to_solid_fraction():
    """The implicit Brinkman penalty field is proportional to the solid fraction:
    strong in the bed/ice, weak in the open cavity."""
    f = SubglacialFlow(SubglacialConfig(n=48))
    # the penalty is exactly dt*chi/eta -- proportional to the local solid fraction
    assert np.allclose(f.pen, f.cfg.dt * f.chi / f.cfg.eta)
    assert f.pen[f.chi > 0.99].min() > 1.0, "penalty must be strong in the solid"
    assert f.pen[f.fluid].max() < f.pen[f.chi > 0.99].min(), \
        "penalty must be weaker in the cavity than in the solid"


def test_penalty_suppresses_velocity_in_solid(developed_flow):
    """The Brinkman penalty suppresses the flow inside the rock/ice relative to the
    open cavity (soft penalization -- it damps rather than perfectly zeroes)."""
    f = developed_flow
    speed = np.sqrt(f.u**2 + f.v**2)
    assert speed[f.chi > 0.95].mean() < 0.85 * speed[f.fluid].mean(), \
        "velocity should be suppressed in the solid relative to the cavity"


def test_exact_sgs_force_is_solenoidal_in_cavity(developed_flow):
    """The exact subgrid force (band-limited at k_c and Leray projected) is
    divergence-free to machine precision -- the elliptic 'fast clock' is exact."""
    f = developed_flow
    _, _, mx, my = exact_sgs_force(f.sp, f.u, f.v, kc=10)
    div_rms = divergence_rms(f.sp, mx, my)
    assert div_rms < 1e-9, f"exact SGS force divergence too large: {div_rms}"


def test_cavity_velocity_has_low_divergence(developed_flow):
    """The penalized velocity stays nearly incompressible (the near-Nyquist /
    sharp-interface residual is small relative to the field)."""
    f = developed_flow
    div_rms = divergence_rms(f.sp, f.u, f.v)
    assert div_rms < 1e-2, f"cavity velocity divergence too large: {div_rms}"


def test_spinup_produces_turbulent_field(developed_flow):
    """Forced spinup yields a nonzero, finite kinetic energy."""
    f = developed_flow
    ke = f.kinetic_energy()
    assert ke > 0 and np.isfinite(ke)
    assert np.isfinite(f.u).all() and np.isfinite(f.theta).all()


# --------------------------------------------------------------------------- #
# glacier-relevant diagnostics
# --------------------------------------------------------------------------- #

def test_spatial_transfer_is_dot_product():
    """Pi = ubar . m, pointwise."""
    rng = np.random.default_rng(0)
    ub, vb, mx, my = (rng.standard_normal((8, 8)) for _ in range(4))
    pi = spatial_transfer(ub, vb, mx, my)
    assert np.allclose(pi, ub * mx + vb * my)


def test_eddy_diffusivity_flux_is_structurally_downgradient():
    """K-theory flux q = -kappa_t grad(theta), kappa_t >= 0, so q is anti-parallel
    to the temperature gradient everywhere: its counter-gradient fraction is 0 by
    construction.  This is exactly the structural limitation Part 9 isolates."""
    sp = Spectral2D(32)
    X, Y = sp.grid()
    # an arbitrary smooth resolved velocity + temperature field
    ub = np.sin(X) * np.cos(2 * Y)
    vb = -np.cos(X) * np.sin(2 * Y)
    tb = np.sin(X + 0.5 * Y) + 0.3 * np.cos(2 * X)
    qx, qy = eddy_diffusivity_heat_flux(sp, ub, vb, tb, kc=12, cs=0.16)
    gx, gy = sp.ddx(tb), sp.ddy(tb)
    mask = np.ones_like(tb, dtype=bool)
    cg = countergradient_fraction(qx, qy, gx, gy, mask)
    assert cg == 0.0, f"eddy diffusivity must be down-gradient, got cg fraction {cg}"


def test_countergradient_fraction_detects_alignment():
    """Flux aligned with the gradient is 100% counter-gradient; anti-aligned 0%."""
    g = np.array([[1.0, -2.0], [0.5, 3.0]])
    mask = np.ones_like(g, dtype=bool)
    assert countergradient_fraction(g, g, g, g, mask) == 1.0      # parallel
    assert countergradient_fraction(-g, -g, g, g, mask) == 0.0    # anti-parallel


def test_exact_subgrid_heatflux_can_be_countergradient(developed_flow):
    """In a developed cavity the exact subgrid heat flux is counter-gradient over a
    nonzero fraction of the wake band -- the regime K-theory forbids."""
    f = developed_flow
    qx, qy, tb, gx, gy = exact_sgs_heat_flux(f.sp, f.u, f.v, f.theta, kc=10)
    cg = countergradient_fraction(qx, qy, gx, gy, f.wake_band())
    assert cg > 0.05, f"expected appreciable counter-gradient flux, got {cg}"


def test_heat_flux_divergence_is_minus_div():
    """Q = -div(tau_theta)."""
    sp = Spectral2D(32)
    X, Y = sp.grid()
    qx = np.sin(X) * np.cos(Y)
    qy = np.cos(X) * np.sin(2 * Y)
    Q = heat_flux_divergence(sp, qx, qy)
    assert np.allclose(Q, -(sp.ddx(qx) + sp.ddy(qy)))


def test_masked_corr_bounds():
    """Correlation of a field with itself is +1, with its negation -1."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 16))
    mask = np.ones_like(a, dtype=bool)
    assert masked_corr(a, a, mask) == pytest.approx(1.0, abs=1e-9)
    assert masked_corr(a, -a, mask) == pytest.approx(-1.0, abs=1e-9)


def test_backscatter_fraction_counts_positive_cells():
    """Backscatter fraction is the share of unmasked cells with Pi > 0."""
    pi = np.array([[1.0, -1.0], [2.0, -3.0]])
    mask = np.ones_like(pi, dtype=bool)
    assert backscatter_fraction(pi, mask) == 0.5


def test_smagorinsky_purely_dissipative_on_cavity_field(developed_flow):
    """K-theory has no backscatter even in this geometry: T(k) <= 0 for all k."""
    from closure.sgs import smagorinsky_force
    f = developed_flow
    kc = 10
    ub, vb, _, _ = exact_sgs_force(f.sp, f.u, f.v, kc)
    mx, my = smagorinsky_force(f.sp, ub, vb, kc, cs=0.16)
    k, T = transfer_spectrum(f.sp, ub, vb, mx, my, kc)
    assert np.all(T[k >= 1] <= 0), "Smagorinsky should be purely dissipative (T<=0)"


# --------------------------------------------------------------------------- #
# real BEDMAP1 bed profile (data-grounded geometry)
# --------------------------------------------------------------------------- #

BEDMAP_TRANSECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "subglacial", "data", "bedmap1_transect.csv")


def test_real_bed_profile_is_periodic_and_scaled():
    """The real BEDMAP1 transect maps to a periodic, cavity-scaled ybed(x):
    peak-to-peak rescaled to 2*bed_amp, mean recentred to bed_mean, seam
    continuous (mirrored construction), and the provenance is real relief."""
    from subglacial.bedmap import bed_profile_from_transect
    n, bed_mean, bed_amp = 128, 0.9, 0.55
    h, meta = bed_profile_from_transect(BEDMAP_TRANSECT, n, bed_mean, bed_amp)
    assert h.shape == (n,)
    assert (h.max() - h.min()) == pytest.approx(2 * bed_amp, rel=1e-6)
    assert h.mean() == pytest.approx(bed_mean, abs=1e-6)
    assert abs(h[0] - h[-1]) < bed_amp, "mirrored seam should be ~continuous"
    assert meta["relief_m"] > 500.0 and meta["length_km"] > 50.0


def test_real_bed_changes_geometry_but_keeps_a_cavity():
    """Injecting the real bed gives a different solid mask than the synthetic bed
    while still leaving an open cavity, and ybed stays a function of x only."""
    from subglacial.bedmap import bed_profile_from_transect
    h, _ = bed_profile_from_transect(BEDMAP_TRANSECT, 64, 0.9, 0.55)
    f_syn = SubglacialFlow(SubglacialConfig(n=64))
    f_real = SubglacialFlow(SubglacialConfig(n=64, bed_profile=h))
    assert 0.0 < f_real.fvol < f_real.n ** 2
    assert not np.allclose(f_syn.chi, f_real.chi), "real bed must change geometry"
    assert np.allclose(f_real.ybed[:, 0], f_real.ybed[:, 3]), "ybed is f(x) only"


# --------------------------------------------------------------------------- #
# elliptic (pressure) vs parabolic (temperature) locality -- the two clocks
# --------------------------------------------------------------------------- #

def test_elliptic_response_solves_poisson():
    """elliptic_response is the inverse Laplacian: lap(dp) = q (zero-mean)."""
    from subglacial.elliptic_demo import elliptic_response
    sp = Spectral2D(32)
    X, Y = sp.grid()
    q = np.sin(2 * X) * np.cos(3 * Y) + 0.5 * np.sin(X + Y)
    q = q - q.mean()
    dp = elliptic_response(sp, q)
    assert np.allclose(sp.laplacian(dp), q, atol=1e-10)


def test_geometry_bump_drives_global_pressure_local_temperature(developed_flow):
    """The SAME localized bed-bump source produces a *global* pressure response
    (elliptic inverse-Laplacian) but only a *local* temperature response (parabolic
    heat kernel): the far-field fraction is large for Δp, small for Δθ."""
    from subglacial.elliptic_demo import (
        geometry_bump, obstacle_source, elliptic_response, parabolic_response,
        diffusion_time_for_length, radial_from, far_fraction,
    )
    f = developed_flow
    bump, center = geometry_bump(f, x0=np.pi)
    q = obstacle_source(f, bump)
    tstar = diffusion_time_for_length(0.18, f.cfg.kappa)
    dp = elliptic_response(f.sp, q)
    dth = parabolic_response(f.sp, q, f.cfg.kappa, tstar)
    r = radial_from(f, center)
    ff_p = far_fraction(dp, r, f.fluid)
    ff_t = far_fraction(dth, r, f.fluid)
    assert ff_p > 0.4, f"pressure response should be global, far-frac {ff_p}"
    assert ff_t < 0.25, f"temperature response should be local, far-frac {ff_t}"
    assert ff_p > 2.0 * ff_t, "elliptic response is far more non-local than parabolic"
