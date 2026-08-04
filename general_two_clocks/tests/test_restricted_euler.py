"""Unit tests for the regularity probe (restricted-Euler blowup vs the nonlocal
pressure Hessian as a regularity condition).  Pure math, fast (no DNS, no data)."""
from __future__ import annotations

import numpy as np
import pytest

import restricted_euler_regularity as rer


def test_restricted_euler_invariants_blow_up_on_vieillefosse_tail():
    """The restricted-Euler invariant system (isotropic/local pressure Hessian) blows
    up in finite time, conserving H=R^2+(4/27)Q^3 and riding onto the Vieillefosse
    tail (R^2/(-(4/27)Q^3) -> 1, with Q<0, R>0)."""
    r = rer.re_invariant_blowup()
    assert r["blew_up"] and r["t_star"] < 50.0
    assert r["H_drift"] < 1e-3                       # exact RE invariant conserved
    assert abs(r["tail_ratio"] - 1.0) < 1e-2         # on the Vieillefosse tail
    assert r["escapes_correct_corner"]
    assert r["ok"]


def test_tau0_rhs_is_exactly_restricted_euler():
    """vgt_rhs at tau=0 is exactly the restricted-Euler RHS -(A^2 - 1/3 tr(A^2) I)."""
    A = rer.random_traceless(7)
    A2 = A @ A
    re_rhs = -(A2 - (1.0 / 3.0) * np.trace(A2) * np.eye(3))
    assert np.allclose(rer.vgt_rhs(A, tau=0.0, T=1.0), re_rhs, atol=1e-12)


@pytest.mark.parametrize("tau", [0.0, 0.05, 0.15])
def test_pressure_hessian_satisfies_poisson_trace_constraint(tau):
    """Both the isotropic (tau=0) and anisotropic/nonlocal (tau>0) modelled pressure
    Hessians satisfy the incompressible Poisson constraint tr(P) = -tr(A^2)."""
    A = rer.random_traceless(3)
    P = rer.pressure_hessian(A, tau)
    assert abs(np.trace(P) + np.trace(A @ A)) < 1e-9
    if tau > 0:                                       # genuinely anisotropic
        assert not np.allclose(P, np.diag(np.diag(P)) * 0 + np.eye(3) * P[0, 0])


def test_restricted_euler_tensor_blows_up_for_generic_ic():
    """The full 3x3 restricted-Euler tensor (tau=0) blows up for every generic IC."""
    r = rer.re_tensor_blowup()
    assert r["n_blowup"] == r["n_seeds"]
    assert r["ok"]


def test_nonlocal_anisotropic_hessian_regularizes():
    """Restoring the anisotropic/nonlocal pressure Hessian (tau>0) keeps the tensor
    bounded for every generic IC -- no finite-time blowup."""
    r = rer.rfd_regularizes()
    assert r["n_bounded"] == r["n_seeds"]
    assert r["max_norm_over_seeds"] < 1e3
    assert r["ok"]


def test_memory_time_regularity_transition():
    """Sweeping the recent-deformation memory time: tau=0 (restricted Euler) blows up;
    sufficiently large tau is regular; once regular it stays regular (no re-entry)."""
    r = rer.tau_transition()
    assert r["re_blows"] and r["big_tau_regular"]
    assert r["monotone_delay"] and r["no_reentry"]
    assert r["ok"]


def test_generic_initial_conditions_blow_up_under_restricted_euler():
    """Restricted Euler blows up for essentially all generic initial conditions."""
    r = rer.generic_ic_blowup_fraction(n_ic=20)
    assert r["blowup_fraction"] > 0.8


def test_real_nonlocal_pressure_hessian_is_dominantly_anisotropic():
    """The real nonlocal pressure Hessian (FFT Poisson on a solenoidal field) satisfies
    the Poisson trace constraint tr(P)=-tr(A^2) (machine precision at the resolved size)
    and is dominantly anisotropic, so the restricted-Euler isotropic truncation discards
    an O(1) part."""
    r = rer.nonlocal_hessian_anisotropy(n=32)
    assert r["poisson_trace_relerr"] < 1e-9
    assert r["anisotropic_fraction_of_P"] > 0.3
