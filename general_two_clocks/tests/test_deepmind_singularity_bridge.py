"""Tests for the DeepMind unstable-singularities bridge (deepmind_singularity_bridge.py).

These pin the verifiable math that connects DeepMind's "Discovery of Unstable Singularities"
(Wang et al., arXiv:2509.14185) to this repo's restricted-Euler / nonlocal-pressure-Hessian
regularity take.  Pure math, fast (no DNS, no data, no network).

What is asserted is deliberately conservative and matches the paper honestly:
  * the renormalized restricted-Euler self-similar profile is the Vieillefosse point (-3, 2);
  * its Jacobian eigenvalues are exactly {-6, +1} (trace -5, det -6);
  * the single +1 mode is the universal blow-up-time (time-translation) freedom, so AFTER
    quotienting it the caricature has instability order 0 -- DeepMind's *stable* type, NOT a
    member of their new order>=1 unstable ladder (which a fixed-exponent quadratic model cannot
    contain);
  * inside the full VGT closure, restricted-Euler blow-up is generic but the nonlocal Hessian
    makes it non-generic -- the "blow-up is fine-tuned" picture DeepMind's instability supports.
"""
from __future__ import annotations

import numpy as np

import deepmind_singularity_bridge as dmb


def test_renormalized_fixed_point_is_vieillefosse():
    """(q*, r*) = (-3, 2) is a genuine fixed point of the renormalized invariant system."""
    fp = np.array([-3.0, 2.0])
    assert np.linalg.norm(dmb.renorm_rhs(fp)) < 1e-12


def test_renorm_rhs_matches_analytic_formula():
    """renorm_rhs encodes q'=-2q-3r, r'=(2/3)q^2-3r at an arbitrary point (no fixed-point luck)."""
    q, r = -1.7, 0.9
    expected = np.array([-2.0 * q - 3.0 * r, (2.0 / 3.0) * q * q - 3.0 * r])
    assert np.allclose(dmb.renorm_rhs([q, r]), expected, atol=1e-12)


def test_jacobian_at_fixed_point_is_analytic():
    """Jacobian at (-3,2) is [[-2,-3],[-4,-3]]: trace -5, det -6 (so eigenvalues {-6, +1})."""
    J = dmb.jacobian([-3.0, 2.0])
    assert np.allclose(J, np.array([[-2.0, -3.0], [-4.0, -3.0]]), atol=1e-5)
    assert abs(np.trace(J) - (-5.0)) < 1e-5
    assert abs(np.linalg.det(J) - (-6.0)) < 1e-4


def test_self_similar_instability_order_zero_mod_timeshift():
    """The profile carries exactly one +1 mode (the universal blow-up-time freedom); removing
    it leaves instability order 0 -- DeepMind's classically-found *stable* singularity type."""
    ss = dmb.self_similar_instability()
    assert ss["residual"] < 1e-9
    assert np.allclose(sorted(ss["eigenvalues"]), [-6.0, 1.0], atol=1e-4)
    assert np.allclose(sorted(ss["eigenvalues_analytic"]), [-6.0, 1.0], atol=1e-6)
    assert ss["instability_order_raw"] == 1
    assert ss["has_trivial_timeshift_mode"] is True
    assert ss["instability_order_mod_trivial"] == 0
    assert "stable singularity" in ss["deepmind_class"]
    assert ss["ok"]


def test_fixed_point_is_a_saddle():
    """Perturb along each eigenvector: the -6 direction relaxes back to the profile shape, the
    +1 (time-shift) direction departs -- a saddle, exactly as the eigenvalues require."""
    sd = dmb.saddle_character(delta=1e-2)
    assert sd["eig_stable"] < -1.0          # the -6 contracting mode
    assert sd["eig_unstable"] > 0.5         # the +1 time-translation mode
    assert sd["dist_after_stable_perturb"] < 1e-2
    assert sd["dist_after_unstable_perturb"] > 1e-2
    assert sd["ok"]


def test_full_model_context_blowup_is_generic_then_depleted():
    """Restricted Euler (local/isotropic Hessian) blows up for ~all generic ICs; restoring the
    nonlocal anisotropic pressure Hessian keeps every seed bounded -- blow-up becomes non-generic.
    This is the repo-side statement that DeepMind's 'unstable / infinitely fine-tuned' result
    supports from the blow-up side."""
    fm = dmb.full_model_context()
    assert fm["restricted_euler_blowup_fraction"] > 0.8
    n_bounded, n_seeds = (int(x) for x in fm["nonlocal_hessian_bounded"].split("/"))
    assert n_bounded == n_seeds and n_seeds >= 1
    assert fm["ok"]


def test_summary_all_ok():
    """End-to-end: every sub-check passes, so the module's headline verdict is VERIFIED."""
    assert dmb.summary()["all_ok"] is True
