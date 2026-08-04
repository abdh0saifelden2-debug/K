"""Unit tests for the new derived relationships NR16-NR21
(general_two_clocks/new_relationships.py). Deterministic, CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import new_relationships as NR  # noqa: E402
import pressure_buoyancy_exchange as PB  # noqa: E402
from compressible.ns import Spectral2D  # noqa: E402


# NR18 — exact identity ------------------------------------------------------ #
def test_nr18_exact_identity():
    r = NR.nr18(n=96, seed=0)
    assert r["max_abs_err"] < 1e-9
    assert r["ok"]


def test_nr18_limits():
    sp = Spectral2D(64)
    x, y = sp.grid()
    b_vert = np.cos(3 * y) + 0.3 * np.sin(y)          # b'(y): held in place
    b_horiz = np.cos(2 * x) + 0.4 * np.sin(x)         # b'(x): full exchange
    assert PB.solenoidal_fraction(sp, b_vert) < 1e-9
    assert abs(PB.solenoidal_fraction(sp, b_horiz) - 1.0) < 1e-6
    # spectral predictor matches both
    assert abs(NR.exchange_fraction_spectral(b_vert)) < 1e-9
    assert abs(NR.exchange_fraction_spectral(b_horiz) - 1.0) < 1e-6


# NR16 — Lowen–Teich --------------------------------------------------------- #
def test_nr16_lowen_teich():
    r = NR.nr16(alphas=(1.5, 2.0, 2.5), n=2 ** 16, n_real=4)
    # spectral slope tracks 3 - alpha (decreasing with alpha), within tolerance
    for a, g, pred in r["rows"]:
        assert abs(g - pred) < 0.4
    gammas = [g for _, g, _ in r["rows"]]
    assert gammas[0] > gammas[1] > gammas[2]          # monotone decreasing in alpha


def test_telegraph_is_dichotomous():
    x = NR.telegraph_powerlaw(2 ** 14, alpha=2.0, seed=0)
    assert set(np.unique(x)).issubset({-1.0, 1.0})
    assert np.isfinite(x).all()


# NR17 — power-law spectrum -> growing memory -------------------------------- #
def test_nr17_memory_grows_with_spectral_slope():
    r = NR.nr17(gammas=(0.0, 0.5, 1.0, 1.5), n=2 ** 15)
    assert r["spearman"] > 0.9
    assert r["tau_int"][0] < 2.0                       # white noise ~ memoryless
    assert r["tau_int"][-1] > r["tau_int"][0] + 3      # memory grows
    # recovered spectral slopes track gamma
    assert r["slopes"][-1] > r["slopes"][0] + 1.0


def test_power_law_noise_slope():
    x = NR.power_law_noise(2 ** 16, gamma=1.0, seed=0)
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x))
    m = (f > 2e-4) & (f < 5e-2)
    slope = -np.polyfit(np.log(f[m]), np.log(np.abs(X[m]) ** 2 + 1e-300), 1)[0]
    assert abs(slope - 1.0) < 0.25


# NR19 — Green–Kubo / Taylor diffusivity ------------------------------------- #
def test_nr19_green_kubo_diffusivity():
    r = NR.nr19()                                       # defaults n=4000, n_traj=5000
    assert r["ok"]
    assert r["max_rel_err"] < 0.08
    # the two independent diffusivity estimators (dispersion vs Green–Kubo) agree
    for _, d_disp, d_gk in r["rows"]:
        assert abs(d_disp - d_gk) / d_gk < 0.08
    # the dispersion estimator matches the exact AR(1) random-walk value 0.5/(1-phi)^2
    assert abs(r["rows"][0][1] - 0.5 / (1 - 0.8) ** 2) / (0.5 / (1 - 0.8) ** 2) < 0.05
    assert abs(r["rows"][1][1] - 0.5 / (1 - 0.95) ** 2) / (0.5 / (1 - 0.95) ** 2) < 0.05


def test_nr19_global_mean_removal_beats_per_row():
    # the fix: removing the known zero mean globally must not inject the per-row
    # sample-mean tail bias (which pulled Green–Kubo ~19% low vs theory).
    V = NR._ar1_ensemble(4000, 4000, 0.8, seed=3)
    d_disp, d_gk = NR._D_from_ensemble(V)
    assert abs(d_gk - 12.5) / 12.5 < 0.06                # GK now near AR(1) theory
    assert abs(d_disp - d_gk) / d_gk < 0.06


# NR20 — additive diffusivity for multiscale memory ------------------------- #
def test_nr20_additive_diffusivity():
    r = NR.nr20(n=3000, n_traj=3000)
    assert r["ok"]
    assert r["rel_err"] < 0.05
    assert abs(r["d_total"] - r["d_sum"]) / r["d_sum"] < 0.05
    # the long-memory (slow) clock dominates transport even at equal energy
    assert r["d_slow"] > 5 * r["d_fast"]


# NR21 — long-range dependence: DFA/Hurst exponent = (gamma+1)/2 ------------- #
def test_nr21_dfa_hurst_tracks_spectral_slope():
    r = NR.nr21()
    assert r["ok"]
    assert r["max_abs_err"] < 0.06
    # white noise -> H = 1/2 (no long-range dependence)
    assert abs(r["rows"][0][1] - 0.5) < 0.05
    # alpha_DFA increases monotonically with the spectral slope gamma
    a = [row[1] for row in r["rows"]]
    assert all(a[i] < a[i + 1] for i in range(len(a) - 1))


def test_dfa_white_noise_is_half():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2 ** 15)
    scales = np.unique(np.floor(np.logspace(1.0, 3.0, 16)).astype(int))
    F = NR.dfa(x, scales)
    a = np.polyfit(np.log(scales), np.log(F), 1)[0]
    assert abs(a - 0.5) < 0.06
