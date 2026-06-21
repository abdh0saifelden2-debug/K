"""Unit tests for the new derived relationship NR31
(general_two_clocks/new_relationships8.py). Deterministic, CPU-only.

NR31: Paper 1's 2-D a-posteriori null (no eddy-viscosity closure beats no-model) is the inverse
energy cascade -- the net resolved->subgrid forward energy flux is ~0 in 2-D (nu_opt~0) but >0 in
3-D (nu_opt>0, the NR29 forward sign), so a positive-definite eddy viscosity can only help in 3-D.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import new_relationships8 as NR8  # noqa: E402

_RES = None


def _res():
    global _RES
    if _RES is None:
        _RES = NR8.nr31(n2d=96, steps_2d=1500, n_snap=4, n3d=32, steps_3d=400)  # fast; ok=True
    return _RES


# --- analytic closed forms ------------------------------------------------- #
def test_optimal_viscosity_is_flux_over_strain():
    # nu_opt = <Pi>/(2<|S|^2>): sign(nu_opt) = sign(<Pi>) = cascade direction
    r3 = _res()["threed"]
    assert abs(r3["nu_opt"] - r3["mean_pi"] / (2.0 * r3["s2"])) < 1e-12 * abs(r3["nu_opt"]) + 1e-18
    assert np.sign(r3["nu_opt"]) == np.sign(r3["mean_pi"])


def test_budget_argmin_nonneg_sign():
    # argmin_{nu>=0}|2 nu s2 - <Pi>| = 0 for non-positive flux, = nu_opt>0 for forward flux
    assert NR8.budget_argmin_nonneg(-1.0, 2.0) == 0.0
    assert NR8.budget_argmin_nonneg(0.0, 2.0) == 0.0
    assert NR8.budget_argmin_nonneg(4.0, 2.0) == 1.0     # 4/(2*2)


def test_nr29_gpu_runs_are_all_forward():
    # the committed P100 3-D runs: net SGS energy flux mu/sigma is positive at every resolution
    # and filter (the robust 3-D forward cascade)
    assert all(v > 0.0 for v in NR8._NR29_GPU_MU_SIGMA.values())
    assert min(NR8._NR29_GPU_MU_SIGMA.values()) > 0.03


# --- NR31 simulated verification ------------------------------------------- #
def test_2d_inverse_cascade():
    r = _res()
    assert r["checks"]["inverse_cascade_ok"] is True
    band = r["twod"]["flux_inverse_band"]
    assert np.mean(band) < 0.0 and np.min(band) < -0.2     # energy flux UP-scale below k_f


def test_2d_has_no_net_forward_flux():
    r = _res()
    # the forward (representable-by-positive-nu_t) part of the 2-D net SGS flux is ~0
    assert r["twod"]["forward_part"] < 0.01
    assert r["checks"]["twod_no_forward_flux_ok"] is True


def test_3d_is_forward_cascade():
    r3 = _res()["threed"]
    assert r3["mean_pi"] > 0.0 and r3["nu_opt"] > 0.0     # forward, positive optimal eddy viscosity
    assert r3["frac_backscatter"] < 0.5                   # net forward (NR29)
    assert r3["mu_over_sigma"] > 0.0


def test_aposteriori_ordering():
    r = _res()
    # no-model is ~optimal among nu>=0 in 2-D (forward capture ~0); positive nu_t wins in 3-D
    assert max(0.0, r["twod"]["mu_over_sigma"]) < 0.01
    assert r["threed"]["budget_argmin_nonneg"] > 0.0
    assert r["checks"]["aposteriori_ordering_ok"] is True


def test_nr31_overall_ok():
    assert _res()["ok"] is True
