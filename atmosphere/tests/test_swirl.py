"""Unit tests for the Goldshtik-Sorokin swirl closure test (Part 10).

These exercise the structural properties the Part-10 benchmark relies on, using a
small grid and short spinup so the suite stays fast and needs no data:

  * the swirl solver develops a coherent vortex with a real core pressure *well*
    (a depression), and keeps the velocity divergence-free (the elliptic clock);
  * the exact subgrid force is solenoidal in this geometry;
  * the levitation diagnostics behave as advertised -- in particular the
    suspension margin is calibrated so the truth field gives exactly 1, and a
    purely dissipative Smagorinsky closure over-drains the swirl so its margin
    drops below the truth's (the particle falls), while projected-FDT preserves it.
"""
from __future__ import annotations

import numpy as np
import pytest

from closure.sgs import (
    exact_sgs_force, smagorinsky_force, surrogate_force, projected_fdt_force,
    divergence_rms,
)
from swirl.flow import SwirlFlow, SwirlConfig
from swirl.levitation import (
    resolved_sgs_power, swirl_turnover, suspension_margin,
    radial_pressure_profile, radial_speed_profile,
)

KC = 10


@pytest.fixture(scope="module")
def developed_swirl():
    """Small sustained swirl, short spinup -- fast but turbulent."""
    cfg = SwirlConfig(n=64, f_amp=1.2, k_f=8.0, f_band=2.0, seed=1)
    f = SwirlFlow(cfg)
    f.run(700)
    return f


# --------------------------------------------------------------------------- #
# solver / geometry
# --------------------------------------------------------------------------- #

def test_spinup_produces_turbulent_field(developed_swirl):
    """Forced spinup yields a nonzero, finite kinetic energy."""
    f = developed_swirl
    ke = f.kinetic_energy()
    assert ke > 0 and np.isfinite(ke)
    assert np.isfinite(f.u).all() and np.isfinite(f.v).all()


def test_velocity_is_divergence_free(developed_swirl):
    """The Leray-projected velocity stays divergence-free to high accuracy --
    the elliptic 'fast clock' is respected each step."""
    f = developed_swirl
    div_rms = divergence_rms(f.sp, f.u, f.v)
    assert div_rms < 1e-9, f"swirl velocity divergence too large: {div_rms}"


def test_core_pressure_well_is_a_depression(developed_swirl):
    """The vortex core carries a low-pressure well: the core minimum lies well
    below the far-field reference, so the well depth is strictly positive."""
    f = developed_swirl
    depth, p = f.core_well_depth()
    assert depth > 0.0, f"core should be a pressure depression, got depth={depth}"
    assert float(p[f.core_mask()].mean()) < float(p[f.r > 2.2].mean())


def test_exact_sgs_force_is_solenoidal(developed_swirl):
    """The exact subgrid force (band-limited at k_c and Leray projected) is
    divergence-free."""
    f = developed_swirl
    _, _, mx, my = exact_sgs_force(f.sp, f.u, f.v, KC)
    div_rms = divergence_rms(f.sp, mx, my)
    assert div_rms < 1e-9, f"exact SGS force divergence too large: {div_rms}"


# --------------------------------------------------------------------------- #
# levitation diagnostics (pure functions)
# --------------------------------------------------------------------------- #

def test_swirl_turnover_formula():
    assert swirl_turnover(0.7, 1.4) == pytest.approx(2.0 * np.pi * 0.7 / 1.4)


def test_resolved_sgs_power_is_dot_product_mean():
    rng = np.random.default_rng(0)
    ub, vb, mx, my = (rng.standard_normal((8, 8)) for _ in range(4))
    assert resolved_sgs_power(ub, vb, mx, my) == pytest.approx(
        float(np.mean(ub * mx + vb * my)))


def test_truth_margin_is_unity():
    """The margin is calibrated so the truth field gives exactly 1."""
    assert suspension_margin(0.03, 3.14, -2e-5, -2e-5) == pytest.approx(1.0)


def test_stronger_drain_lowers_margin():
    """A more strongly draining closure (more negative power) gives a lower
    margin; a feeding (positive) power gives a margin above 1."""
    e, tau, p_t = 0.03, 3.14, -2e-5
    m_drain = suspension_margin(e, tau, -6e-3, p_t)
    m_feed = suspension_margin(e, tau, +1e-3, p_t)
    assert m_drain < 1.0 < m_feed


def test_margin_is_clamped_nonnegative():
    """A catastrophic drain cannot make the margin negative."""
    assert suspension_margin(0.01, 100.0, -10.0, -1e-5) == 0.0


def test_radial_profiles_shapes_and_core_dip(developed_swirl):
    """Radial profiles have matching lengths; the pressure profile dips negative
    near the core (referenced to the far field) and the speed peaks off-centre."""
    f = developed_swirl
    p = f.pressure()
    rc_p, prof_p = radial_pressure_profile(p, f.r)
    rc_u, prof_u = radial_speed_profile(f.u, f.v, f.r)
    assert rc_p.shape == prof_p.shape and rc_u.shape == prof_u.shape
    # innermost pressure bin is below the far-field reference (a dip)
    assert np.nanmin(prof_p[:3]) < 0.0
    # the azimuthal speed peaks away from r=0 (a vortex, not a plug)
    assert np.nanargmax(prof_u) > 0


# --------------------------------------------------------------------------- #
# the headline structural claim
# --------------------------------------------------------------------------- #

def test_smagorinsky_overdrains_and_drops_margin(developed_swirl):
    """Smagorinsky (K-theory) drains the resolved swirl far harder than the truth,
    so its suspension margin falls below 1 (particle falls), while projected-FDT
    tracks the truth's net drain and keeps the margin near 1 (particle stays up)."""
    f = developed_swirl
    sp = f.sp
    ub, vb, mtx, mty = exact_sgs_force(sp, f.u, f.v, KC)
    msx, msy = smagorinsky_force(sp, ub, vb, KC, cs=0.16)
    mfx, mfy, _ = projected_fdt_force(sp, ub, vb, mtx, mty, KC, seed=7)

    e_res = 0.5 * float(np.mean(ub ** 2 + vb ** 2))
    tau = swirl_turnover(f.cfg.r_core, f.cfg.U_swirl)
    p_t = resolved_sgs_power(ub, vb, mtx, mty)
    p_s = resolved_sgs_power(ub, vb, msx, msy)
    p_f = resolved_sgs_power(ub, vb, mfx, mfy)

    # Smagorinsky drains much harder than the truth
    assert abs(p_s) > 10.0 * abs(p_t)
    m_smag = suspension_margin(e_res, tau, p_s, p_t)
    m_fdt = suspension_margin(e_res, tau, p_f, p_t)
    assert m_smag < 0.9, f"Smagorinsky margin should fall below 1, got {m_smag}"
    assert m_smag < m_fdt, "FDT must preserve the swirl better than Smagorinsky"
    assert abs(m_fdt - 1.0) < abs(m_smag - 1.0)


def test_surrogate_breaks_solenoidality(developed_swirl):
    """The phase-randomized surrogate is NOT solenoidal -- it would inject a
    spurious pressure on the core well."""
    f = developed_swirl
    sp = f.sp
    _, _, mtx, mty = exact_sgs_force(sp, f.u, f.v, KC)
    mqx, mqy = surrogate_force(sp, mtx, mty, KC, seed=7)
    assert divergence_rms(sp, mqx, mqy) > 0.1
