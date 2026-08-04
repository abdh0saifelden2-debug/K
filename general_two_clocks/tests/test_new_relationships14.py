"""Unit proofs for NR37 (`general_two_clocks/new_relationships14.py`):
pseudo-sound made explicit and the M^2 -> M^1 crossover of the NI density
fluctuation.  Ledger item E4.

Covered: the divergence-free driver field; the pseudo-sound identity (acoustic
channel = incompressible pressure / c^2, quadratic in amplitude); the entropy
channel linear in amplitude AND in the background gradient; the crossover Mach
linear in the gradient; the running exponent 1 -> 2; and a small nonlinear
confirmation that both exponents survive time evolution (and that beta = 0 is a
pure-acoustic control with no entropy channel).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships14 import (  # noqa: E402
    Spectral2D, crossover_sweep, leading_order, nonlinear_sweep,
    pseudo_sound_density, running_exponent, sim_point, solenoidal_unit_field,
)


# --------------------------------------------------------------- driver field
def test_driver_field_is_divergence_free():
    sp = Spectral2D(64)
    u, v = solenoidal_unit_field(sp, seed=0)
    div = sp.ddx(u) + sp.ddy(v)
    speed = float(np.sqrt(np.mean(u * u + v * v)))
    assert abs(speed - 1.0) < 1e-9                       # unit rms by construction
    assert float(np.sqrt(np.mean(div * div))) < 1e-10   # solenoidal to machine eps


# --------------------------------------------------------- acoustic (M^2) channel
def test_pseudo_sound_is_quadratic_in_amplitude():
    """rms(drho)/rho0 = rms(p_inc)/(rho0 c^2); p_inc ~ u^2 so doubling the
    velocity amplitude must quadruple the acoustic density response."""
    sp = Spectral2D(64)
    u, v = solenoidal_unit_field(sp, seed=1)
    c = 8.0
    d1 = pseudo_sound_density(sp, 0.1 * u, 0.1 * v, 1.0, c)
    d2 = pseudo_sound_density(sp, 0.2 * u, 0.2 * v, 1.0, c)
    assert abs(d2 / d1 - 4.0) < 1e-6


def test_acoustic_slope_is_two():
    lo = leading_order(n=64, seed=0)
    assert abs(lo["acoustic_slope"] - 2.0) < 0.03


# --------------------------------------------------------- entropy (M^1) channel
def test_entropy_slope_is_one():
    lo = leading_order(n=64, seed=0)
    assert abs(lo["entropy_slope"] - 1.0) < 0.03


def test_entropy_channel_is_linear_in_gradient():
    """The mean-gradient scalar is unforced feedback-free -> its rms is exactly
    linear in beta (double the gradient, double the response)."""
    th1 = sim_point(40, 8.0, 5e-3, 0, M=0.1, beta=1.0, t_end=0.4, kappa=5e-3)[1]
    th2 = sim_point(40, 8.0, 5e-3, 0, M=0.1, beta=2.0, t_end=0.4, kappa=5e-3)[1]
    assert th1 > 0
    assert abs(th2 / th1 - 2.0) < 0.05


# ----------------------------------------------------------------- crossover
def test_crossover_mach_is_linear_in_gradient():
    lo = leading_order(n=64, seed=0)
    a, b = lo["acoustic_prefactor_a"], lo["entropy_prefactor_b"]
    cr = crossover_sweep(a, b, betas=(0.01, 0.02, 0.04, 0.08))
    assert abs(cr["mstar_vs_beta_slope"] - 1.0) < 1e-6      # exact power law
    # and M^*(beta) = (b/a) beta explicitly
    for be, ms in zip(cr["betas"], cr["mstar"]):
        assert abs(ms - (b / a) * be) < 1e-12


def test_running_exponent_runs_from_one_to_two():
    lo = leading_order(n=64, seed=0)
    a, b = lo["acoustic_prefactor_a"], lo["entropy_prefactor_b"]
    m = np.logspace(-2.3, 0.2, 40)
    beta = 0.1 / (b / a)                                    # crossover near M~0.1
    e = running_exponent(m, a, b, beta)
    assert e[0] < 1.2         # gradient-dominated low-M limit -> 1
    assert e[-1] > 1.8        # pseudo-sound-dominated high-M limit -> 2
    assert np.all(np.diff(e) > -1e-9)                       # monotone increasing


# ------------------------------------------------------- nonlinear confirmation
def test_nonlinear_confirms_both_exponents():
    sw = nonlinear_sweep(n=44, machs=(0.05, 0.1, 0.2), beta=1.0, t_end=0.6)
    assert 1.75 <= sw["dyn_density_slope"] <= 2.15          # acoustic ~ M^2
    assert 0.7 <= sw["scalar_slope"] <= 1.2                 # entropy ~ M^1


def test_beta_zero_is_pure_acoustic_control():
    sw = nonlinear_sweep(n=44, machs=(0.05, 0.1, 0.2), beta=0.0, t_end=0.6)
    assert sw["scalar_slope"] is None                       # no entropy channel
    assert max(sw["scalar_rms"]) < 1e-8
    assert 1.75 <= sw["dyn_density_slope"] <= 2.15
