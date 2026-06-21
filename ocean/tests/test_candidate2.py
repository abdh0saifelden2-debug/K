"""Tests for Candidate 2 -- the double-diffusive (salt + temperature) probe in
:mod:`subglacial.candidate2_doublediff`.

CPU-only behaviour tests covering

  * the Lewis-number diffusivity contrast (``kappa_S = kappa_T / Le``),
  * the resolved cavity develops flow and carries two pinned scalars,
  * the *diffusivity-contrast* signature ``Nu_S >> Nu_T`` (salt is advection-
    dominated because its diffusivity is ~100x smaller),
  * the Turner flux ratio scaling ``gamma ~ 1/R_rho`` (since the same velocity
    advects two near-identical scalar ramps so ``F_T ~ F_S``),
  * and, honestly, that stabilising salinity *suppresses* thermal transport --
    ``Nu_T`` falls as ``R_rho`` grows -- i.e. there is no salt-finger hump in
    this forced penalised regime.
"""

import numpy as np

from subglacial.candidate2_doublediff import (
    DoubleDiffConfig,
    DoubleDiffFlow,
    run_case,
)


def _tiny_cfg(**kw):
    base = dict(nx=96, ny=64, A=4.0, f_amp=0.4, Ri_T=2.0, R_rho=2.0, seed=1)
    base.update(kw)
    return DoubleDiffConfig(**base)


def test_lewis_number_sets_salt_diffusivity():
    cfg = DoubleDiffConfig(kappa_T=8.0e-4, Le=100.0)
    assert np.isclose(cfg.kappa_S, 8.0e-6)
    assert cfg.kappa_S < cfg.kappa_T


def test_flow_develops_and_two_scalars_pinned():
    s = DoubleDiffFlow(_tiny_cfg(), xp=np)
    s.run(600)
    assert float(np.abs(np.asarray(s.u)).max()) > 0.05
    assert s.fluid.mean() > 0.5
    # both scalars stay bounded and pinned warm+salty at the bed, cold+fresh at ice
    for c in (np.asarray(s.theta), np.asarray(s.S)):
        assert np.isfinite(c).all()
        assert c.min() > -0.2 and c.max() < 1.2


def test_haline_nusselt_far_exceeds_thermal():
    """The diffusivity contrast means salt is transported almost entirely by
    advection: Nu_S is order-Le larger than Nu_T."""
    r = run_case(_tiny_cfg(), spinup=1200, measure=600, xp=np)
    assert r["Nu_S"] > 5.0 * r["Nu_T"]
    assert r["Nu_T"] > 1.0  # some convective enhancement above pure conduction


def test_turner_flux_ratio_scales_inverse_Rrho():
    """gamma = F_T/(R_rho F_S); with two near-identical advected ramps F_T~F_S,
    so gamma roughly halves when R_rho doubles."""
    lo = run_case(_tiny_cfg(R_rho=1.0), spinup=1000, measure=500, xp=np)["gamma"]
    hi = run_case(_tiny_cfg(R_rho=2.0), spinup=1000, measure=500, xp=np)["gamma"]
    assert lo > hi > 0.0
    assert abs(hi - 0.5 * lo) < 0.25 * lo


def test_no_finger_hump_salt_suppresses_heat_transport():
    """Honest scope: stabilising salinity monotonically suppresses thermal
    transport -- Nu_T at strongly stabilising R_rho is below Nu_T near R_rho~1.
    The predicted finger sweet-spot at R_rho~2 does not appear."""
    lowr = run_case(_tiny_cfg(R_rho=1.0), spinup=1500, measure=600, xp=np)["Nu_T"]
    midr = run_case(_tiny_cfg(R_rho=2.0), spinup=1500, measure=600, xp=np)["Nu_T"]
    highr = run_case(_tiny_cfg(R_rho=5.0), spinup=1500, measure=600, xp=np)["Nu_T"]
    assert highr < lowr           # stabilising salt suppresses heat transport
    assert midr < lowr + 1e-9     # no hump: R_rho=2 is not a maximum
