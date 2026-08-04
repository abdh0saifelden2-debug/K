"""Unit tests for the real-data (Hurricane Otis) calibration of Part 10.

These check that the best-track loader and the Holland (1980) vortex fit reproduce
the documented Otis peak, and that a swirl driven by the real Otis profile still
shows the headline structural result: a real core pressure well, a Smagorinsky
closure that over-drains the swirl (suspension margin < 1, particle falls), and a
projected-FDT closure that preserves it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from closure.sgs import (
    exact_sgs_force, smagorinsky_force, projected_fdt_force,
)
from swirl.flow import SwirlFlow, SwirlConfig
from swirl.levitation import resolved_sgs_power, swirl_turnover, suspension_margin
from swirl.otis import (
    load_otis_track, peak_vortex, holland_B, holland_speed_shape,
    holland_pressure_deficit, target_uth_factory, KT_TO_MS,
)

KC = 10
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "swirl", "data", "otis_besttrack.csv")


# --------------------------------------------------------------------------- #
# best-track loader + Holland fit
# --------------------------------------------------------------------------- #

def test_track_loads_with_expected_columns():
    track = load_otis_track(DATA)
    assert len(track) > 10
    for key in ("iso_time", "status", "vmax_kt", "mslp_mb", "rmw_nmi"):
        assert key in track[0]


def test_peak_vortex_matches_documented_otis():
    """Peak record is the deepest pressure: Otis bottomed at 922 mb / 145 kt with a
    ~5 nmi pinhole eye (NHC best track)."""
    v = peak_vortex(DATA)
    assert v.pc_mb == pytest.approx(922.0)
    assert v.vmax_kt == pytest.approx(145.0)
    assert v.rmw_nmi == pytest.approx(5.0)
    assert v.dp_mb == pytest.approx(v.penv_mb - v.pc_mb)
    assert v.vmax_ms == pytest.approx(145.0 * KT_TO_MS)


def test_holland_B_in_physical_range():
    """The fitted Holland shape parameter sits in the observed envelope [1, 2.5]."""
    v = peak_vortex(DATA)
    assert 1.0 <= v.holland_B <= 2.5
    # monotone wind-pressure relation: a stronger Vmax at fixed dp raises B
    assert holland_B(80.0, 8500.0) > holland_B(60.0, 8500.0)
    assert holland_B(60.0, 1e9) == pytest.approx(1.0)  # clamped low


def test_holland_speed_shape_peaks_at_rmw():
    """V(r)/Vmax equals 1 at the radius of maximum wind and decays either side."""
    rmw, B = 0.6, 2.0
    assert holland_speed_shape(rmw, rmw, B) == pytest.approx(1.0)
    assert holland_speed_shape(0.5 * rmw, rmw, B) < 1.0
    assert holland_speed_shape(2.0 * rmw, rmw, B) < 1.0


def test_holland_pressure_deficit_is_monotone():
    """The pressure deficit is deepest at the core (→1) and vanishes far away (→0)."""
    rmw, B = 0.6, 2.0
    r = np.array([0.05, 0.3, 0.6, 1.2, 3.0])
    d = holland_pressure_deficit(r, rmw, B)
    assert d[0] > d[-1]
    assert np.all(np.diff(d) <= 1e-12)
    assert 0.0 <= d[-1] < d[0] <= 1.0


# --------------------------------------------------------------------------- #
# calibrated solver
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def otis_swirl():
    """Small, short Otis-calibrated swirl -- fast but develops a real core well."""
    v = peak_vortex(DATA)
    r_core = 0.6
    cfg = SwirlConfig(n=64, f_amp=1.2, k_f=8.0, f_band=2.0, seed=1, r_core=r_core)
    uth = target_uth_factory(v, cfg.U_swirl, r_core)
    f = SwirlFlow(cfg, target_uth=uth)
    f.run(700)
    return f


def test_otis_target_overrides_default_profile():
    """Supplying the Otis target profile changes the initial vortex (the Holland
    eyewall is sharper than the default Gaussian core)."""
    v = peak_vortex(DATA)
    cfg = SwirlConfig(n=64, r_core=0.6)
    uth = target_uth_factory(v, cfg.U_swirl, 0.6)
    f_otis = SwirlFlow(cfg, target_uth=uth)
    f_def = SwirlFlow(cfg)
    speed_otis = np.sqrt(f_otis.ut_x ** 2 + f_otis.ut_y ** 2)
    speed_def = np.sqrt(f_def.ut_x ** 2 + f_def.ut_y ** 2)
    assert not np.allclose(speed_otis, speed_def)
    assert np.isfinite(speed_otis).all()


def test_otis_core_is_a_pressure_well(otis_swirl):
    """The Otis-calibrated vortex digs a genuine low-pressure core depression."""
    depth, _ = otis_swirl.core_well_depth()
    assert depth > 0.0, f"Otis core should be a depression, got {depth}"


def test_otis_smagorinsky_overdrains_and_drops_margin(otis_swirl):
    """The headline result survives real-vortex calibration: Smagorinsky drains the
    Otis swirl far harder than the truth, dropping the suspension margin below 1,
    while projected-FDT preserves it."""
    f = otis_swirl
    sp = f.sp
    ub, vb, mtx, mty = exact_sgs_force(sp, f.u, f.v, KC)
    msx, msy = smagorinsky_force(sp, ub, vb, KC, cs=0.16)
    mfx, mfy, _ = projected_fdt_force(sp, ub, vb, mtx, mty, KC, seed=7)

    e_res = 0.5 * float(np.mean(ub ** 2 + vb ** 2))
    tau = swirl_turnover(f.cfg.r_core, f.cfg.U_swirl)
    p_t = resolved_sgs_power(ub, vb, mtx, mty)
    p_s = resolved_sgs_power(ub, vb, msx, msy)
    p_f = resolved_sgs_power(ub, vb, mfx, mfy)

    assert abs(p_s) > 10.0 * abs(p_t)
    m_smag = suspension_margin(e_res, tau, p_s, p_t)
    m_fdt = suspension_margin(e_res, tau, p_f, p_t)
    assert m_smag < 1.0, f"Otis Smagorinsky margin should fall below 1, got {m_smag}"
    assert abs(m_fdt - 1.0) < abs(m_smag - 1.0)
