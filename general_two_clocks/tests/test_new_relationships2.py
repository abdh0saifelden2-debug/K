"""Unit tests for the new derived relationships NR22-NR24
(general_two_clocks/new_relationships2.py). Deterministic, CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import new_relationships2 as NR2  # noqa: E402


# helpers -------------------------------------------------------------------- #
def test_ar1_stationary_moments():
    # AR(1) sample var -> sigma^2/(1-a^2), AC1 -> a
    a, sigma = 0.9, 1.0
    x = NR2.ar1(2 ** 20, a, sigma=sigma, seed=3)
    assert abs(np.var(x) - sigma ** 2 / (1 - a ** 2)) / (sigma ** 2 / (1 - a ** 2)) < 0.03
    assert abs(NR2._ac1(x) - a) < 0.01


# NR22 — EWS indicator coupling --------------------------------------------- #
def test_nr22_invariant_constant_along_fold():
    r = NR2.nr22()
    # Var*(1-AC1^2) stays = sigma_eps^2 across the approach-the-fold sweep ...
    assert r["fold_inv_err"] < 0.05
    # ... while BOTH precursors genuinely rise (critical slowing down)
    assert r["ac1_rises"] and r["var_rises"]
    assert r["ok"]


def test_nr22_discriminator_separates_csd_from_noise():
    r = NR2.nr22()
    # louder forcing (a fixed): invariant tracks sigma^2 (rises), AC1 stays flat
    assert r["loud_tracks"] < 0.05
    assert r["ac1_flat"] < 0.02
    # the two sweeps are genuinely different: fold holds invariant flat, loud does not
    loud_inv = [row[3] for row in r["loud"]]
    assert max(loud_inv) / min(loud_inv) > 3.0     # invariant spans >3x under noise inflation


# NR23 — Kramers-Kronig DC sum rule ----------------------------------------- #
def test_nr23_dc_sum_rule_matches():
    r = NR2.nr23()
    assert r["max_rel_err"] < 1e-3
    assert r["ok"]


def test_nr23_sign_tracks_backscatter():
    # the net eddy viscosity sign (incl. nu_eff<0 backscatter) == migration-integral sign
    z0, rhs = NR2._kk_dc([-1.5, 0.4, 0.3], [10.0, 2.0, 1.0])
    assert z0 < 0 and rhs < 0
    assert abs(z0 - rhs) / abs(z0) < 1e-3
    # a dissipative kernel is positive on both sides
    z0p, rhsp = NR2._kk_dc([1.0, 0.5], [2.0, 10.0])
    assert z0p > 0 and rhsp > 0


# NR24 — eddy diffusivity = zero-frequency PSD ------------------------------ #
def test_nr24_D_equals_half_S0():
    r = NR2.nr24()
    # independent (Welch periodogram) S(0) reproduces the dispersion diffusivity
    assert r["max_rel_err"] < 0.12
    assert r["ok"]


def test_nr24_fold_divergence_exponent():
    r = NR2.nr24()
    # D ~ (1-a)^{-2} approaching the fold (transport face of the variance EWS)
    assert abs(r["fold_slope"] + 2.0) < 1e-6
    assert r["div_rel"] < 0.12


def test_nr24_long_memory_kills_finite_diffusivity():
    r = NR2.nr24()
    # white noise: window-diffusivity flat (finite D); 1/f memory: grows (S(0)->inf)
    assert r["white_flat"] and r["mem_grows"]
    assert r["growth"][1.0] > r["growth"][0.0] + 0.3


# NR25 — second fluctuation-dissipation theorem ---------------------------- #
def test_nr25_second_fdt():
    r = NR2.nr25()
    # friction kernel (dissipation) == random-force ACF/kT (fluctuation)
    assert r["max_rel_err"] < 0.02
    assert r["ok"]


def test_nr25_two_sides_computed_independently():
    # the dissipation side is a deterministic bath sum (no sampling); the
    # fluctuation side is a Gibbs Monte-Carlo ACF — they must still agree
    w, c = NR2._cl_bath(40)
    t = np.array([0.0, 1.0, 2.5])
    K = NR2._K_friction(t, w, c)                  # deterministic
    FF = NR2._FF_sampled(t, w, c, kT=1.0, M=200000, seed=2)  # stochastic
    assert K[0] > 0                               # K(0)=sum c^2/w^2 > 0
    assert np.max(np.abs(FF - K)) / np.max(np.abs(K)) < 0.02


def test_run_all_pass():
    r = NR2.run()
    assert all(v["ok"] for v in r.values())
