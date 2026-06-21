"""Tests for Candidate 3 -- the moving-boundary roughness-feedback probe in
:mod:`subglacial.candidate3_roughness_feedback`.
 
CPU-only behaviour tests covering
 
  * the moving boundary integrates without blowing up and the flow develops,
  * with melt switched off the roughness amplitude is *exactly* conserved (no
    spurious numerical drift from rebuilding the masks),
  * the honest scientific finding: the melt feedback is **negative** -- the
    roughness amplitude decays (``Lambda < 0``) because the thin (low) columns
    melt faster (``corr(m, y_ice) < 0``) so the boundary self-smooths,
  * and that this self-smoothing is **closure-independent** (the boundary is
    driven by conduction-limited melt, so Smagorinsky does not suppress it).
"""
 
import numpy as np
 
from subglacial.candidate3_roughness_feedback import (
    Candidate3Config,
    RoughFeedbackFlow,
    make_rough_ice_base,
    run_case,
)
 
 
def _tiny_cfg(**kw):
    base = dict(nx=96, ny=64, A=4.0, sigma_h_init=0.30, f_amp=0.6,
                St=2.0e-4, n_smooth=0, seed=1)
    base.update(kw)
    return Candidate3Config(**base)
 
 
def test_rough_ice_base_has_requested_rms():
    yb = make_rough_ice_base(128, 4 * 2 * np.pi, 5.5, 0.30, 2.0, 8.0, seed=3)
    assert np.isclose(yb.mean(), 5.5, atol=1e-6)
    assert np.isclose(yb.std(), 0.30, rtol=1e-3)
 
 
def test_moving_boundary_flow_develops_and_is_finite():
    s = RoughFeedbackFlow(_tiny_cfg(), xp=np)
    for _ in range(400):
        s.step()
    for _ in range(20):
        s.update_boundary()
    assert float(np.abs(np.asarray(s.u)).max()) > 0.05
    assert np.isfinite(np.asarray(s.y_ice_x)).all()
    assert np.isfinite(np.asarray(s.theta)).all()
    # the base stays inside the box
    yi = np.asarray(s.y_ice_x)
    assert yi.min() > s.cfg.y_bed and yi.max() < s.Ly
 
 
def test_no_melt_conserves_roughness():
    """With melting switched off (huge Stefan number) and no smoothing, rebuilding
    the penalty masks must not drift the roughness amplitude."""
    s = RoughFeedbackFlow(_tiny_cfg(St=1.0e12, n_smooth=0), xp=np)
    for _ in range(300):
        s.step()
    sig0 = s.sigma_h()
    for k in range(300):
        s.step()
        if k % 10 == 0:
            s.update_boundary()
    assert abs(s.sigma_h() - sig0) < 1e-6
 
 
def test_feedback_is_negative_roughness_self_smooths():
    """Honest scope: the melt feedback is negative -- the roughness decays and
    the thin columns melt faster, so the boundary flattens rather than running
    away."""
    r = run_case(_tiny_cfg(), spinup=400, n_steps=1200, record_every=20, xp=np)
    assert r["Lambda"] < 0.0          # roughness decays, not grows
    assert r["sigma_ratio"] < 1.0     # ends smaller than it started
    assert r["corr_m_h0"] < 0.0       # thin (low) columns melt faster
    assert r["corr_dh_h0"] < 0.0      # low columns rise more -> flattening
 
 
def test_self_smoothing_is_closure_independent():
    """Because the melt is conduction-limited, Smagorinsky does not suppress the
    (negative) feedback: the growth rate is essentially the same for the
    unclosed and Smagorinsky runs."""
    none = run_case(_tiny_cfg(sgs="none"), spinup=400, n_steps=1200,
                    record_every=20, xp=np)["Lambda"]
    smag = run_case(_tiny_cfg(sgs="smagorinsky"), spinup=400, n_steps=1200,
                    record_every=20, xp=np)["Lambda"]
    assert none < 0.0 and smag < 0.0
    assert abs(smag - none) < 0.1 * abs(none) + 1e-3
