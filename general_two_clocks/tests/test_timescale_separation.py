"""Tests for timescale_separation.py — checklist items #1 (elliptic↔parabolic
timescale separation) and #4 (singular-perturbation theory of the decoupling).
Deterministic, CPU-only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timescale_separation as TS  # noqa: E402


# #1 — fast clock: elliptic pressure constraint is instantaneous ------------- #
def test_fast_clock_instantaneous_projection():
    r = TS.fast_clock_projection()
    assert r["ok"]
    assert r["div_before"] > 0.1            # there really was divergence to remove
    assert r["div_after"] < 1e-10           # enforced to ~machine zero in one step
    assert r["drop"] > 1e8                  # many orders of magnitude, single step


# #1 — slow clock: parabolic transport relaxes at finite rate κk² ------------ #
def test_slow_clock_parabolic_rate():
    r = TS.slow_clock_diffusion()
    assert r["ok"]
    assert r["max_rel_err"] < 0.05          # measured rate matches 1/(κk²)
    taus = [tt for _, _, tt in r["rows"]]
    assert taus[0] > taus[1] > taus[2]      # τ_slow ∝ k^-2


# #4 — singular-perturbation limit: ε = κk/c_s ∝ c_s^-1 → 0 (Mach→0) --------- #
def test_singular_perturbation_limit():
    r = TS.acoustic_clock()
    assert r["ok"]
    assert r["omega_err"] < 0.02            # acoustic ω = c_s·k recovered
    assert abs(r["eps_exponent"] + 1.0) < 0.05    # ε ∝ c_s^-1 (singular limit)
    eps = [row[3] for row in r["rows"]]
    assert all(eps[i] > eps[i + 1] for i in range(len(eps) - 1))   # ε ↓ as c_s ↑
