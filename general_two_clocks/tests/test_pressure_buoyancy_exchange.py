"""Unit tests for Paper 2 — in-place pressure-mediated buoyancy exchange
(general_two_clocks/pressure_buoyancy_exchange.py).

Pin the four pre-registered predictions plus the underlying Helmholtz/baroclinic
identities. Deterministic (seeded), CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pressure_buoyancy_exchange as P  # noqa: E402
from boussinesq.solver import project, divergence  # noqa: E402
from compressible.ns import Spectral2D  # noqa: E402

RES = P.compare()  # computed once (~2s; includes the developed-flow sim)


def test_all_four_preregistered_pass():
    assert RES["n_pass"] == 4
    assert RES["ok"] is True


# PR1 ----------------------------------------------------------------------- #
def test_pr1_pressure_response_is_nonlocal():
    assert RES["pr1"]["ratio"] > 1.5          # pressure correction broader than blob
    assert RES["pr1"]["far_fraction"] > 0.1   # substantial energy far from the source


# PR2 ----------------------------------------------------------------------- #
def test_pr2_pure_stratification_is_held_in_place():
    assert RES["pr2"]["frac_vertical"] < 1e-6
    assert RES["pr2"]["frac_horizontal"] > 1e-2
    assert RES["pr2"]["monotone"] is True


def test_pure_vertical_buoyancy_force_is_pure_gradient():
    """A b'(y)-only force has zero baroclinic torque, so projection removes all of it."""
    sp = Spectral2D(64)
    _, y = sp.grid()
    b = np.cos(3.0 * y) + 0.4 * np.sin(y)
    assert P.solenoidal_fraction(sp, b) < 1e-8


# baroclinic / Helmholtz identities ---------------------------------------- #
def test_projection_removes_only_curl_free_part():
    """F − P(F) is a pure gradient (curl-free); P(F) is divergence-free."""
    sp = Spectral2D(64)
    x, y = sp.grid()
    b = np.sin(2 * x) * np.cos(3 * y) + 0.3 * np.cos(x + y)
    bp = b - b.mean()
    z = np.zeros_like(bp)
    us, vs = project(sp, z, bp)               # F_sol
    # surviving force is divergence-free
    assert np.sqrt(np.mean(divergence(sp, us, vs) ** 2)) < 1e-9
    # the curl of the surviving force equals the baroclinic torque ∂_x b'
    curl_sol = sp.ddx(vs) - sp.ddy(us)
    assert np.allclose(curl_sol, sp.ddx(bp), atol=1e-9)


# PR3 ----------------------------------------------------------------------- #
def test_pr3_in_place_exchange():
    r = RES["pr3"]
    assert r["netmass_ratio"] < 1e-3          # no net vertical mass flux
    assert r["buoy_flux"] > 0                  # positive buoyancy flux (warm up)
    assert r["b_up"] > 0 > r["b_dn"]           # upward fluid warmer than downward
    assert 0.4 < r["vol_up"] < 0.6 and 0.4 < r["vol_dn"] < 0.6
    assert r["div_rms"] < 1e-6                 # the flow stays incompressible


# PR4 ----------------------------------------------------------------------- #
def test_pr4_linear_instantaneous_mediation():
    assert RES["pr4"]["superposition_resid"] < 1e-10
    assert abs(RES["pr4"]["scaling_slope"] - 1.0) < 0.02
