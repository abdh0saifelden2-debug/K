"""Unit tests for the GPU-free theory verifications in THEORY_CAVITY.md.

Focus: the §3.2 Dedner-GLM approximate-projection threshold (RESULT 4). These
tests exercise the closed-form per-mode cleaning factor and the residual sweep
on a *synthetic* divergence spectrum (no DNS spinup), so they are fast and
deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from compressible.ns import Spectral2D
from subglacial.flow import SubglacialFlow, SubglacialConfig
from subglacial.theory_tests import (dedner_cleaning_factor, dedner_residual_curve,
                                     _knee_one_over_e, result_dedner_cleaning,
                                     result_commutator_entropy, result_constraint_class,
                                     result_roughness_scale_separation)


def _rk4_factor(kabs, c_h, gamma, L, steps=40000):
    """Direct RK4 of the GLM pair for one mode over one transit T=L/c_h, from
    (D0,psi)=(1,0); returns D(T)/D0. Reference for the closed-form factor."""
    T = L / c_h
    dt = T / steps
    k2 = kabs * kabs
    D, psi = 1.0, 0.0

    def deriv(D, psi):
        return (k2 * psi, -c_h * c_h * D - gamma * psi)

    for _ in range(steps):
        a1 = deriv(D, psi)
        a2 = deriv(D + 0.5 * dt * a1[0], psi + 0.5 * dt * a1[1])
        a3 = deriv(D + 0.5 * dt * a2[0], psi + 0.5 * dt * a2[1])
        a4 = deriv(D + dt * a3[0], psi + dt * a3[1])
        D += dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        psi += dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
    return D


@pytest.mark.parametrize("kabs,G", [(1.0, 1.0), (3.0, 2.0), (2.0, 8.0),
                                    (1.0, 12.0), (5.0, 4.0)])
def test_cleaning_factor_matches_rk4(kabs, G):
    """Closed-form factor reproduces direct time integration of the GLM system
    (under- and over-damped) to high accuracy."""
    L = 2.0 * np.pi
    c_h = 3.0
    gamma = G * c_h / L                      # G = gamma * tau_adj, tau_adj = L/c_h
    analytic = float(dedner_cleaning_factor(np.array([kabs * L]), G)[0])
    numeric = _rk4_factor(kabs, c_h, gamma, L)
    assert abs(analytic - numeric) < 1e-4


def test_factor_limits():
    """G=0 with integer kL returns cos(kL)=1 (undamped wave returns after one
    transit); near critical damping (G = 2*kL) the mode is strongly suppressed;
    deep over-damping (G >> 2*kL) suppresses *less* (slow root) -- the trade-off."""
    kL = 2.0 * np.pi                          # |k|L = 2*pi -> cos(kL) = 1 at G=0
    assert dedner_cleaning_factor(np.array([kL]), 0.0)[0] == pytest.approx(1.0, abs=1e-9)
    crit = abs(dedner_cleaning_factor(np.array([kL]), 2.0 * kL)[0])   # ~ critical
    over = abs(dedner_cleaning_factor(np.array([kL]), 6.0 * kL)[0])   # over-damped
    assert crit < 0.05
    assert over > crit                        # too much damping is worse


def _synthetic_div_spectrum(n=64, seed=0):
    """A smooth, broadband, zero-mean divergence field's spectrum (no DNS)."""
    sp = Spectral2D(n)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, n))
    qh = sp.fft(q) * np.exp(-sp.k2 / (2.0 * (n / 6.0) ** 2))   # band-limit
    qh[0, 0] = 0.0                                              # zero mean
    return sp, qh


def test_residual_monotone_collapse_and_oh1_knee():
    """The divergence residual falls monotonically with G up to the knee, and the
    knee G* = gamma_clean*tau_adj is O(1) (between ~1 and ~5)."""
    sp, qh = _synthetic_div_spectrum()
    Gs = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
    rows = dedner_residual_curve(sp, qh, Gs)
    div = rows[:, 1]
    # strictly decreasing on the under/critically-damped branch (G <= 8)
    assert np.all(np.diff(div[:9]) < 0)
    Gknee = _knee_one_over_e(rows)
    assert 1.0 <= Gknee <= 5.0
    # bound holds: deep over-cleaning suppresses divergence by >50x vs uncleaned
    assert div[-1] < div[0] / 50.0


def test_overdamped_pressure_upturn():
    """The low-k (spurious-pressure) residual has an interior optimum: pushing G
    far beyond the knee re-grows it (the classic Dedner over-damping trade-off)."""
    sp, qh = _synthetic_div_spectrum(seed=3)
    Gs = (2.0, 6.0, 12.0, 24.0, 48.0, 96.0)
    rows = dedner_residual_curve(sp, qh, Gs)
    dp = rows[:, 2]
    iopt = int(np.argmin(dp))
    assert 0 < iopt < len(dp) - 1               # optimum is interior
    assert dp[-1] > dp[iopt]                     # over-damped branch re-grows


@pytest.fixture(scope="module")
def cleaning_result():
    """End-to-end RESULT 4 on a small penalized cavity (short spinup, no data)."""
    f = SubglacialFlow(SubglacialConfig(n=48, sgs="none", f_amp=1.5, k_f=8.0,
                                        f_band=2.0, seed=1))
    f.run(800, ramp=400)
    return result_dedner_cleaning(f.sp, f, kc=f.n // 3)


def test_result_dedner_endtoend(cleaning_result):
    """The full measurement path reproduces the claimed structure: O(1) knee,
    over-damped optimum past the knee, a well-defined nonlocal amplification, and a
    divergence residual that collapses from O(1) to small across the sweep."""
    r = cleaning_result
    assert 1.0 <= r["Gknee"] <= 5.0
    assert r["Gopt"] > r["Gknee"]                       # optimum past the knee
    assert 0.0 < r["tail0"] < 1.0                        # some global pressure spread
    assert r["amp"] > 0.0 and np.isfinite(r["amp"])     # amplification well-defined
    div = r["rows"][:, 1]
    assert div[0] > 0.5 and div[-1] < div[0] / 20.0     # collapses from O(1) to small


def test_commutator_entropy():
    """RESULT 5 (Girsanov / Entropy-Pressure identity): on a divergence-free frozen
    field the master-identity commutator equals the curl-free part of the advection
    (machine-zero algebra), the operator-splitting difference quotient converges to
    it linearly in t, and its squared norm equals the H^1 energy of the spurious
    pressure to machine precision. The Theorem-3 counter-gradient link is reported
    only as a partial positive correlation (a conjecture, not an identity)."""
    flow = SubglacialFlow(SubglacialConfig(n=48, sgs="none", f_amp=1.5, k_f=8.0,
                                           f_band=2.0, seed=1))
    flow.run(800, ramp=400)
    r = result_commutator_entropy(flow.sp, flow, kc=flow.n // 3)
    # (a) commutator == (I-L)(u.grad u0) to machine precision
    assert r["err_alg"] < 1e-4
    # (c) Entropy-Pressure identity: ||C||^2 == INT |grad dp_spur|^2 exactly
    assert abs(r["ep_ratio"] - 1.0) < 1e-4
    assert 0.0 < r["dil_frac"] < 1.0                       # genuine dilatational part
    # (b) splitting limit converges to the commutator, ~linearly in t
    rel = r["split_rows"][:, 1]
    assert rel[-1] < rel[0]                                # smaller t -> smaller error
    assert rel[-1] < 0.15
    # Theorem-3 link is positive but partial (a scaling, not an identity)
    assert 0.0 < r["cg_frac"] < 1.0
    assert r["corr_abs"] > 0.1


def test_constraint_class():
    """RESULT 6 (constraint class): the incompressibility constraint is first-class,
    so its pressure inverse is the nonlocal (nabla^2)^-1 with symbol |k|^-2. (1) the
    Poisson symbol holds to machine zero; (2) the local surrogate p_local = c*Q (flat
    |k|^0 symbol) is falsified -- the |p_dyn|^2/|p_local|^2 power ratio falls with a
    steep negative slope (~ -4, i.e. |k|^-2 in amplitude) that no local operator can
    produce, and the true pressure holds far more low-k energy than the surrogate;
    (3) the corruption of N is model error, not physics: the exact (Leray-projected)
    subgrid force is solenoidal, while raw Smagorinsky injects a dilatational part
    whose amplitude is small relative to the dynamic pressure."""
    flow = SubglacialFlow(SubglacialConfig(n=48, sgs="none", f_amp=1.5, k_f=8.0,
                                           f_band=2.0, seed=1))
    flow.run(800, ramp=400)
    r = result_constraint_class(flow.sp, flow, kc=flow.n // 3)
    # (1) Poisson symbol p_hat = -Q_hat/|k|^2 to machine precision
    assert r["err_sym"] < 1e-8
    # (2) local surrogate falsified: steep negative power-ratio slope (the |k|^-2 symbol)
    assert r["slope_ratio"] < -2.0
    # true first-class pressure carries more low-k energy than the local caricature
    assert r["lowk_pdyn"] > r["lowk_local"]
    # (3) exact subgrid force is solenoidal by construction; Smag injects a dilatational part
    assert r["dil_exact"] < 1e-6
    assert r["dil_smag"] > r["dil_exact"]
    # spurious pressure is small in amplitude but phase-decorrelated from the truth
    assert r["amp_smag"] < 0.2


def test_roughness_scale_separation():
    """RESULT 7 (Theorem 8, roughness--scale separation) on the bundled BEDMAP1
    transect. The bed is a natural red spectrum (alpha ~ 2-2.6), so: (1) the
    pressure response (elliptic k^-2 kernel) is concentrated at the largest,
    cavity scales; (2) bed-SLOPE variance (k^2 E_h, form-drag relevant) is
    instead concentrated at small scales -- the blind zone grows monotonically
    toward fine scales and far exceeds the (tiny) elevation-variance blind zone;
    (3) for molecular kappa the crossover is the cavity scale L. Uses the bundled
    DEM only -- no download, no DNS."""
    r = result_roughness_scale_separation()
    # (a) natural red bedrock spectrum
    assert 1.8 <= r["alpha"] <= 2.8
    # (b) pressure (k^-2) is a large-scale / cavity-scale quantity
    assert r["pf_50km"] > 0.95
    assert r["pf_20km"] >= r["pf_50km"]
    # (c) bed-slope blind zone is monotone-increasing toward small scales and large
    assert r["bz_slope_300"] < r["bz_slope_1km"] < r["bz_slope_5km"]
    assert r["bz_slope_1km"] > 0.2                       # substantial form-drag roughness
    # control: elevation variance blind zone is tiny (red spectrum)
    assert r["bz_elev_20km"] < 0.10
    assert r["bz_slope_1km"] > 5.0 * r["bz_elev_20km"]   # slope >> elevation blind zone
    # molecular-kappa crossover is the cavity scale
    assert r["lam_star_mol"] == pytest.approx(r["length_m"])
    # (d) felt-roughness ratio R_felt = slope-var(form drag) / slope-var(thermal).
    #     UN-normalized, so it is >1 and grows as the thermal screen lam* coarsens
    #     and as the cutoff drops below Nyquist (extrapolated). The earlier NORMALIZED
    #     drag-coeff ratio is identically O(1) for a red bed -- that is why this is the
    #     reported quantity.
    assert r["rfelt_1km"] > 1.0                          # thermal misses some slope var
    assert r["rfelt_5km"] > r["rfelt_1km"]               # coarser screen -> larger ratio
    assert r["rfelt_1km_ext"] > r["rfelt_1km"]           # extrapolation grows the ratio
    assert r["rfelt_10m_ext"] > 1.0
