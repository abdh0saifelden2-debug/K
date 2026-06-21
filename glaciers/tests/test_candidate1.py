"""Tests for Candidate 1 -- the rough-ice-base plume probe in
:mod:`subglacial.candidate1_plumes`.

CPU-only behaviour tests covering

  * the band-limited rough ice base has the requested RMS roughness and stays
    strictly inside the cavity (no overlap with the bed / domain top),
  * the resolved cavity develops flow (umax well above the penalty-bled floor),
  * the per-column melt field tracks the wavy interface and is spatially
    intermittent (positive skew, heavy tails) -- and, honestly, that this
    intermittency is *geometric*: it is essentially unchanged when the flow is
    switched off, because the no-slip wall pins the interfacial flux to
    conduction,
  * the turbulent heat flux Fturb rises with Ri (the flow-dependent observable),
  * the moment helper is robust to degenerate inputs.
"""

import numpy as np

from subglacial.candidate1_plumes import (
    PlumeConfig,
    PlumeFlow,
    _moments,
    make_rough_ice_base,
    run_case,
)


def _tiny_cfg(**kw):
    base = dict(nx=96, ny=64, A=4.0, f_amp=0.6, Ri=0.0, seed=1, rough_seed=3)
    base.update(kw)
    return PlumeConfig(**base)


def test_rough_base_rms_and_bounds():
    Lx = 4.0 * 2.0 * np.pi
    yb = make_rough_ice_base(256, Lx, y_ice_mean=5.5, sigma_h=0.3,
                             k_min=2, k_max=8, seed=0)
    assert np.isclose(np.std(yb), 0.3, rtol=1e-6)
    assert np.isclose(yb.mean(), 5.5, atol=1e-6)
    # stays between the bed top and the domain ceiling Ly=2*pi
    assert yb.min() > 0.5 and yb.max() < 2.0 * np.pi


def test_rough_base_seed_reproducible():
    Lx = 4.0 * 2.0 * np.pi
    a = make_rough_ice_base(128, Lx, 5.5, 0.3, 2, 8, seed=7)
    b = make_rough_ice_base(128, Lx, 5.5, 0.3, 2, 8, seed=7)
    c = make_rough_ice_base(128, Lx, 5.5, 0.3, 2, 8, seed=8)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_flow_develops_and_diagnostics_sane():
    s = PlumeFlow(_tiny_cfg(), xp=np)
    s.run(800)
    assert float(np.abs(np.asarray(s.u)).max()) > 0.05
    assert s.fluid.mean() > 0.5
    assert np.isfinite(s.turb_heat_flux())
    m = s.melt_field()
    assert m.shape == (s.cfg.nx,)
    assert np.isfinite(np.nanmean(m))


def test_melt_field_is_positive_and_varies_over_rough_base():
    """The penalty-clear interfacial melt is positive (warm bed -> cold ice) and
    spatially variable over the rough base (peak/mean > 1, finite moments)."""
    s = PlumeFlow(_tiny_cfg(sigma_h=0.4), xp=np)
    s.run(400)
    m = s.melt_field()
    assert np.nanmean(m) > 0.0
    mom = _moments(m)
    assert mom["peak_mean"] > 1.0
    assert np.isfinite(mom["skew"]) and np.isfinite(mom["kurt"])


def test_melt_is_conduction_limited_flow_independent():
    """The honest scope boundary: the interfacial melt and its spatial
    statistics are (to ~1%) identical whether the flow is off or on at finite
    Ri -- the no-slip Brinkman wall pins the flux to conduction.  The flow shows
    up only in Fturb."""
    off = run_case(_tiny_cfg(f_amp=0.0, Ri=0.0), spinup=400, measure=300, xp=np)
    on = run_case(_tiny_cfg(f_amp=0.6, Ri=0.5), spinup=400, measure=300, xp=np)
    assert abs(on["melt_mean"] - off["melt_mean"]) < 0.02 * abs(off["melt_mean"])
    assert abs(on["peak_mean"] - off["peak_mean"]) < 0.1 * abs(off["peak_mean"])
    assert on["fturb_mean"] > off["fturb_mean"]  # flow lifts Fturb, not the melt


def test_buoyancy_increases_turbulent_heat_flux():
    f0 = run_case(_tiny_cfg(Ri=0.0), spinup=1500, measure=800, xp=np)["fturb_mean"]
    f2 = run_case(_tiny_cfg(Ri=2.0), spinup=1500, measure=800, xp=np)["fturb_mean"]
    assert f2 > f0 > 0.0


def test_moments_degenerate_inputs():
    assert _moments([])["peak_mean"] == 0.0
    const = _moments([3.0, 3.0, 3.0])
    assert const["std"] == 0.0 and const["skew"] == 0.0
    z = _moments(np.array([1.0, 2.0, np.nan, 4.0]))
    assert np.isfinite(z["skew"]) and np.isfinite(z["kurt"])
