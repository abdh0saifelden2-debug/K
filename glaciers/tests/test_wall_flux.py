"""Unit tests for the finite-conductance ice-side thermal wall model
(``subglacial.wall_flux.ThermalWall``) and its ``flow3d`` hook.

Small grids / short spinups keep the suite fast and GPU-free. The key invariant
is that ``cond_ratio == 1`` reduces the wall model *bit-for-bit* to the bare
solver's Dirichlet ice pin, so switching the hook on cannot silently perturb the
validated baseline; smaller ``cond_ratio`` then only weakens the ice pull.
"""
from __future__ import annotations

import numpy as np
import pytest

from subglacial.flow3d import Subglacial3DConfig, Subglacial3DFlow
from subglacial.wall_flux import ThermalWall


def _run(thermal_wall, steps=30, n=24, seed=1):
    cfg = Subglacial3DConfig(n=n, seed=seed, f_amp=1.5, sgs="none",
                             thermal_wall=thermal_wall)
    return Subglacial3DFlow(cfg).run(steps)


def _ice_band(f):
    return f.fluid & (f.Y > f.cfg.ice_base - 0.45) & (f.Y < f.cfg.ice_base)


def test_cond_ratio_one_reproduces_dirichlet_bit_for_bit():
    """``cond_ratio == 1`` (robin_const) must reproduce the bare Dirichlet ice
    pin exactly: identical theta field and identical melt-flux diagnostic. This
    guards against the earlier rock/ice-split formulation, which changed the
    rock/fluid transition target and was NOT equal to the solver's update."""
    f0 = _run(None)
    f1 = _run(ThermalWall(cond_ratio=1.0, mode="robin_const"))
    assert np.array_equal(f0.theta, f1.theta)
    assert f0.melt_flux()[0] == f1.melt_flux()[0]


def test_cond_ratio_one_ustar_mode_close_to_dirichlet():
    """robin_ustar scales the ice pull by u*(x,z)/mean(u*) (clipped to <=1), so
    cond_ratio=1 does NOT reduce bit-for-bit to Dirichlet (the local ratio
    varies); it is only the Reynolds-analogy variant. We require the run stays
    finite and the ice-band mean stays close to the Dirichlet baseline (the
    mean conductance is ~1), i.e. no gross over-/under-cooling."""
    f0 = _run(None)
    fu = _run(ThermalWall(cond_ratio=1.0, mode="robin_ustar"))
    band = _ice_band(f0)
    assert np.isfinite(fu.theta).all()
    base = f0.theta[band].mean()
    assert abs(fu.theta[band].mean() - base) <= 0.02 * abs(base)


def test_finite_conductance_does_not_overcool_ice_band():
    """A finite-conductance wall pulls the ice band toward T_melt less hard than
    the infinite-conductance Dirichlet pin, so the near-ice fluid must be at
    least as warm as the Dirichlet baseline (never colder)."""
    f0 = _run(None)
    flo = _run(ThermalWall(cond_ratio=0.05, mode="robin_const"))
    band = _ice_band(f0)
    assert flo.theta[band].mean() >= f0.theta[band].mean() - 1e-9


def test_wall_flux_diagnostic_finite_and_nonnegative():
    """The wall-flux diagnostic (heat absorbed by the ice sink) is recorded after
    a step, is finite, and is non-negative for a warm cavity draining into a cold
    ice wall."""
    tw = ThermalWall(cond_ratio=0.1, mode="robin_const")
    _run(tw)
    flux = tw._last_flux
    assert flux is not None and np.isfinite(flux) and flux >= 0.0


def test_cond_ratio_validation():
    with pytest.raises(ValueError):
        ThermalWall(cond_ratio=0.0)
    with pytest.raises(ValueError):
        ThermalWall(cond_ratio=1.5)
