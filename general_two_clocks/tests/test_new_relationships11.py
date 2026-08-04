"""Unit proofs for NR34 (`general_two_clocks/new_relationships11.py`):
the B.2 ice kernel as an exact tempered half-derivative, its Warburg/Caputo
limit, the Mittag-Leffler closed loop, and the Ste^2 visibility window.
"""
import os
import sys

import numpy as np
import pytest
from scipy.special import erfcx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships11 import (  # noqa: E402
    H_exact, H_tempered, H_warburg, coeff_A, coeff_W, coeff_W_classic,
    half_integral, ml_step_response, stefan_number, tau_diff, tau_frac,
    tempered_identity_error, volterra_step, warburg_phase_deg)


# ------------------------------------------------------ exact operator form --
def test_tempered_identity_is_exact():
    """H(s) = A/s + W sqrt(s + 1/(4 tau_d))/s to machine precision on a
    complex grid: the kernel IS a tempered half-derivative, not 'like' one."""
    assert tempered_identity_error() < 1e-12


def test_tempered_identity_dimensional_params():
    """Same identity with dimensional (non-unit) parameters."""
    kw = dict(kappa=1.09e-6, Vbar=3.2e-8, theta_far=-0.7, rho_c=917 * 2100.0)
    s = np.array([1e-9 + 1e-10j, 1e-6 + 1e-6j, 1e-3 + 1e-5j])
    e = np.abs(H_exact(s, **kw) - H_tempered(s, **kw)) / np.abs(H_exact(s, **kw))
    assert np.max(e) < 1e-12


def test_warburg_coefficient_equals_classic_half_space_law():
    """W = -2 A sqrt(tau_d) == (k_th/sqrt(kappa)) |thetabar'(0)| identically:
    the fractional coefficient is the textbook Caputo flux coefficient
    evaluated on the advected base gradient."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        k, V, rc = rng.uniform(0.01, 10, 3)
        th = -rng.uniform(0.01, 300)
        assert np.isclose(coeff_W(k, V, th, rc), coeff_W_classic(k, V, th, rc),
                          rtol=1e-12)
        assert coeff_W(k, V, th, rc) > 0          # negative feedback sign


def test_warburg_limit_and_error_rate():
    """|H - W/sqrt(s)|/|H| falls like (s tau_d)^{-1/2} (the A/s next order)."""
    errs = []
    for x in (1e2, 1e4, 1e6):
        e = abs((H_exact(complex(x)) - H_warburg(complex(x)))
                / H_exact(complex(x)))
        errs.append(float(e))
        # prefactor 1/2: err ~ 0.5 (s tau_d)^{-1/2}
        assert np.isclose(e, 0.5 * x ** -0.5, rtol=0.15)
    assert errs[0] > errs[1] > errs[2]


def test_phase_tends_to_minus_45deg():
    """arg H(i omega) -> -45 deg (Warburg constant-phase element)."""
    p3, p5 = warburg_phase_deg(1e3), warburg_phase_deg(1e5)
    assert abs(p5 + 45.0) < 0.2
    assert abs(p5 + 45.0) < abs(p3 + 45.0)


# --------------------------------------------------- Mittag-Leffler solution --
def test_ml_solves_the_fractional_integral_equation():
    """v(t) = v0 erfcx(sqrt(t/tau_f)) satisfies rhoL v + W I^{1/2} v = q0
    (quadrature on a fine grid) -- the fractional Langevin step problem."""
    rhoL, q0 = 0.2, 1.0                       # Ste = 5 in nondim units
    W = coeff_W()
    tf = tau_frac(rhoL=rhoL)
    tg = np.linspace(0.0, 5.0 * tf, 4001)
    v = ml_step_response(tg, rhoL=rhoL, q0=q0)
    lhs = rhoL * v + W * half_integral(v, tg)
    assert np.max(np.abs(lhs[1:] - q0)) < 2e-3


def test_ml_asymptotics():
    """Short time 1 - 2 sqrt(t/(pi tau_f)); long time sqrt(tau_f/(pi t)) --
    the heavy algebraic (non-exponential) fractional-relaxation tail."""
    rhoL = 1.0
    tf = tau_frac(rhoL=rhoL)
    t_s = 1e-6 * tf
    v = ml_step_response(np.array([t_s]), rhoL=rhoL)[0]
    assert np.isclose(v, 1 - 2 * np.sqrt(t_s / (np.pi * tf)), rtol=1e-4)
    t_l = 1e6 * tf
    v = ml_step_response(np.array([t_l]), rhoL=rhoL)[0]
    assert np.isclose(v, np.sqrt(tf / (np.pi * t_l)), rtol=1e-2)
    # and it is NOT exponential: v(10 tau_f) >> exp(-10)
    v10 = ml_step_response(np.array([10 * tf]), rhoL=rhoL)[0]
    assert v10 > 100 * np.exp(-10)


def test_erfcx_is_mittag_leffler_half():
    """E_{1/2}(-x) = e^{x^2} erfc(x): check against the series
    sum_k (-x)^k / Gamma(k/2 + 1) at small-moderate x."""
    from math import gamma
    for x in (0.1, 0.5, 1.5):
        series = sum((-x) ** k / gamma(k / 2.0 + 1.0) for k in range(80))
        assert np.isclose(erfcx(x), series, rtol=1e-10)


# ------------------------------------------------------------- closed loop --
def test_volterra_solver_exact_on_warburg_kernel():
    """Product-integration solver vs the exact E_{1/2} answer on the pure
    Warburg kernel: discretisation error only."""
    rhoL = 0.2
    tf = tau_frac(rhoL=rhoL)
    tg, v = volterra_step(5.0 * tf, n=1500, rhoL=rhoL, kernel="warburg")
    vml = ml_step_response(tg, rhoL=rhoL)
    assert np.max(np.abs(v - vml) / vml) < 5e-3


def test_full_kernel_converges_to_fractional_limit_at_rate_one_over_ste():
    """The closed loop with the FULL tempered kernel deviates from the ML
    stage by O(1/Ste) inside the window and halves when Ste doubles."""
    errs = {}
    for ste in (5.0, 10.0, 20.0):
        rl = 1.0 / ste
        tfk = tau_frac(rhoL=rl)
        tg, v = volterra_step(5.0 * tfk, n=1200, rhoL=rl)
        vml = ml_step_response(tg, rhoL=rl)
        errs[ste] = float(np.max(np.abs(v - vml) / vml))
    assert errs[5.0] > errs[10.0] > errs[20.0]
    assert np.isclose(errs[5.0] / errs[10.0], 2.0, rtol=0.2)
    assert np.isclose(errs[10.0] / errs[20.0], 2.0, rtol=0.2)


def test_glacial_ste_quasisteady_no_fractional_stage():
    """Ste = 6.3e-3: tau_f/tau_d ~ 2.5e4, so the response never leaves the
    O(Ste) neighbourhood of v0 and lands on v_inf = v0/(1+Ste)."""
    ste = 6.3e-3
    rl = 1.0 / ste
    tg, v = volterra_step(30.0, n=1200, rhoL=rl)
    v0 = 1.0 / rl
    vinf = v0 / (1.0 + ste)
    assert abs(v[-1] / vinf - 1.0) < 1e-3
    assert np.max(np.abs(v / v0 - 1.0)) < 1.5 * ste


def test_window_law_is_ste_squared():
    """tau_f / tau_d = Ste^{-2} algebraically, for random parameters."""
    rng = np.random.default_rng(4)
    for _ in range(30):
        k, V, rc, rhoL = rng.uniform(0.05, 20, 4)
        th = -rng.uniform(0.01, 50)
        ste = stefan_number(th, rc, rhoL)
        ratio = tau_frac(k, V, th, rc, rhoL) / tau_diff(k, V)
        assert np.isclose(ratio, ste ** -2, rtol=1e-12)


def test_dc_gain_gives_quasisteady_limit():
    """H(0) = -rho_c theta_far => v_inf/v0 = 1/(1 + Ste): the closed loop's
    long-time limit is controlled by the kernel's DC gain alone."""
    th, rc, rhoL = -2.0, 1.5, 4.0
    H0 = -rc * th
    ste = stefan_number(th, rc, rhoL)
    assert np.isclose(1.0 / (1.0 + H0 / rhoL), 1.0 / (1.0 + ste), rtol=1e-12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
