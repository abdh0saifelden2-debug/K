"""Tests for Candidate 4 -- the hydraulic-switching cavity probe in
:mod:`subglacial.candidate4_hydraulic_switch`.

These are CPU-only smoke/behaviour tests (no GPU needed).  They check that

  * the well-resolved cavity actually develops flow (the previous dead-flow bug
    was an under-resolved cavity letting the Brinkman penalty bleed into the
    fluid; here ``umax`` must grow well above the penalty-bled O(1e-3) floor),
  * the diagnostics are sane and on the documented scales (active-layer height
    in [0, 1], finite kinetic energy and turbulent heat flux),
  * seeding the conductive profile makes the *turbulent heat flux* respond to
    buoyancy and to the closure -- Smagorinsky suppresses it relative to the
    unclosed run -- which is the flow-dependent observable that distinguishes
    the closures (the interfacial conductive melt is pinned to the wall and is
    deliberately *not* asserted to vary), and
  * the state-based switching detector counts genuine regime flips with
    hysteresis rather than sample-to-sample jitter.
"""

import numpy as np
import pytest

from subglacial.candidate4_hydraulic_switch import (
    HydraulicConfig,
    HydraulicSwitchFlow,
    detect_switching,
    run_case,
)


def _tiny_cfg(**kw):
    """A small but *resolved* cavity (ny>=64) cheap enough for CI."""
    base = dict(nx=96, ny=64, A=4.0, f_amp=0.8, Ri=0.0, seed=1)
    base.update(kw)
    return HydraulicConfig(**base)


def test_flow_develops_above_penalty_floor():
    """The resolved cavity develops real flow: umax >> the O(1e-3) penalty-bled
    floor that plagued the under-resolved (ny=48) configuration."""
    s = HydraulicSwitchFlow(_tiny_cfg(), xp=np)
    s.run(800)
    umax = float(np.abs(np.asarray(s.u)).max())
    assert umax > 0.05, f"flow too weak (penalty bleed?): umax={umax}"
    assert np.isfinite(s.kinetic_energy())
    assert s.kinetic_energy() > 1e-4


def test_diagnostics_in_range():
    s = HydraulicSwitchFlow(_tiny_cfg(), xp=np)
    s.run(500)
    H1 = s.active_layer_height()
    assert 0.0 <= H1 <= 1.0
    assert np.isfinite(s.turb_heat_flux())
    assert np.isfinite(s.melt_flux())
    # fluid mask occupies most of the (tall) cavity, not a thin sliver
    assert s.fluid.mean() > 0.5


def test_conduction_seed_sets_warm_fluid():
    """With init_conduction the fluid starts warm (mean theta well above 0);
    without it the fluid starts cold."""
    warm = HydraulicSwitchFlow(_tiny_cfg(init_conduction=True), xp=np)
    cold = HydraulicSwitchFlow(_tiny_cfg(init_conduction=False), xp=np)
    fw = np.asarray(warm.fluid)
    tw = np.asarray(warm.theta)[fw].mean()
    tc = np.asarray(cold.theta)[fw].mean()
    assert tw > 0.2, f"conductive seed should warm the fluid, got {tw}"
    assert tc < 0.05, f"cold start should leave fluid near 0, got {tc}"


def test_buoyancy_increases_turbulent_heat_flux():
    """Stronger buoyancy coupling (Ri) raises the vertical turbulent heat flux
    Fturb = <v' theta'> in the seeded conductive cavity."""
    def fturb(Ri):
        s = HydraulicSwitchFlow(_tiny_cfg(Ri=Ri), xp=np)
        s.run(2000)
        return s.turb_heat_flux()
    f0, f5 = fturb(0.0), fturb(5.0)
    assert f5 > f0 > 0.0, f"Fturb should grow with Ri: {f0} -> {f5}"


def test_smagorinsky_suppresses_turbulent_heat_flux():
    """K-theory (Smagorinsky) over-dissipates and so transports less heat than
    the unclosed run at the same buoyancy -- the two-clocks prediction."""
    def fturb(sgs):
        s = HydraulicSwitchFlow(_tiny_cfg(Ri=5.0, sgs=sgs), xp=np)
        s.run(2500)
        return s.turb_heat_flux()
    assert fturb("smagorinsky") < fturb("none")


def test_detect_switching_counts_regime_flips():
    """A clean square wave between filled (~0.9) and stratified (~0.2) yields
    exactly the number of flips, while small jitter in the filled state yields
    none (hysteresis suppresses noise)."""
    dt = 0.1
    square = np.array([0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.2, 0.2] * 2, dtype=float)
    f = detect_switching(square, dt)
    # 16 samples -> total_time = 15*dt; 7 flips in this series
    expected = 7 / (15 * dt)
    assert f == pytest.approx(expected, rel=1e-6)

    jitter = 0.9 + 0.02 * np.array([1, -1] * 16, dtype=float)
    assert detect_switching(jitter, dt) == 0.0


def test_detect_switching_short_series():
    assert detect_switching([0.5], 0.1) == 0.0
    assert detect_switching([], 0.1) == 0.0


def test_run_case_smoke():
    """run_case returns the documented diagnostics on sane scales."""
    cfg = _tiny_cfg()
    r = run_case(cfg, spinup=200, measure=300, sample_every=10, xp=np)
    for key in ("H1_mean", "H1_std", "KE_mean", "melt_mean",
                "fturb_mean", "f_switch", "umax"):
        assert key in r and np.isfinite(r[key])
    assert r["umax"] > 0.05
    assert 0.0 <= r["H1_mean"] <= 1.0
    assert r["t"].shape == r["fturb"].shape == r["H1"].shape
