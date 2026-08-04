"""Unit proofs for NR60 (`general_two_clocks/new_relationships37.py`):
Gonella's rotary coefficient is an entropy-production spectral density.

Covered: the exact 2x2 closure tr[S^{-T}S - I] = 4(Im S_uv)^2/det S and
its Stokes / rotary-coefficient forms; the OU-rotator integral closing on
NR57's 2 f^2/nu; spectral KL = phase-space-current EPR for arbitrary OU
systems (2-D and 3-D); the reversible null; sampling as a data-processing
lower bound; the rectilinear-waves-are-reversible theorem; the
Gaussian-lower-bounds-true-EPR demonstration (biased ring); mirror/
reversal invariance; and the real-ocean readouts (equator reversible,
tropical eddy-clock localization, polar NH-DC/SH-eddy split, hemispheric
mirror, no determinant clipping).  Offline-safe: committed NR53 cache.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships37 import (  # noqa: E402
    BANDS, EPS, analyze, band_correlations, band_epr_spectrum,
    epr_integrand_2x2, epr_integrand_general, epr_integrand_rotary,
    epr_integrand_stokes, load_bands, ou_epr_current, ou_epr_spectral,
    ring_gaussian_epr, ring_true_epr, rotator_epr_exact,
    wave_field_spectral_epr,
)


@pytest.fixture(scope="module")
def out():
    return analyze(write=False)


def _rand_spd(rng):
    X = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    return X @ X.conj().T + 0.1 * np.eye(2)


def test_identity_2x2_and_stokes_forms():
    rng = np.random.default_rng(0)
    for _ in range(200):
        S = _rand_spd(rng)
        g = epr_integrand_general(S)
        assert abs(g - epr_integrand_2x2(S)) < 1e-11 * (1 + abs(g))
        assert abs(g - epr_integrand_stokes(S)) < 1e-11 * (1 + abs(g))


def test_rotary_reduction_gonella_form():
    # isotropic sector: S+- = Ss -+ Sa, density = 4 C_R^2/(1-C_R^2)
    rng = np.random.default_rng(1)
    for _ in range(100):
        Ss = rng.uniform(0.5, 3.0)
        Sa = rng.uniform(-0.9, 0.9) * Ss
        S = np.array([[Ss, -1j * Sa], [1j * Sa, Ss]])
        Sp, Sm = Ss + Sa, Ss - Sa
        CR = (Sp - Sm) / (Sp + Sm)
        want = 4.0 * CR ** 2 / (1.0 - CR ** 2)
        assert abs(epr_integrand_general(S) - want) < 1e-10 * (1 + want)
        assert abs(epr_integrand_rotary(Sp, Sm) - want) < 1e-10 * (1 + want)


def test_rotator_integral_closes_on_nr57(out):
    for x in out["synthetic"]["rotator"]:
        assert abs(x["spectral"] - x["exact"]) < 1e-5 * x["exact"]
        assert abs(x["current"] - x["exact"]) < 1e-9 * x["exact"]
        assert abs(rotator_epr_exact(x["nu"], x["f"]) -
                   2.0 * x["f"] ** 2 / x["nu"]) == 0.0


def test_spectral_equals_current_for_general_ou(out):
    for g in out["synthetic"]["general_ou"]:
        assert abs(g["spectral"] - g["current"]) < 1e-4 * abs(g["current"])


def test_reversible_process_has_zero_epr(out):
    rev = out["synthetic"]["reversible"]
    assert abs(rev["spectral"]) < 1e-6
    assert abs(rev["current"]) < 1e-10


def test_mirror_and_reversal_invariance():
    rng = np.random.default_rng(2)
    for _ in range(50):
        S = _rand_spd(rng)
        M = np.diag([1.0, -1.0])
        assert abs(epr_integrand_general(S) -
                   epr_integrand_general(M @ S @ M)) < 1e-11
        assert abs(epr_integrand_general(S) -
                   epr_integrand_general(S.T)) < 1e-11


def test_frame_rotation_and_scale_invariance():
    rng = np.random.default_rng(3)
    for _ in range(50):
        S = _rand_spd(rng)
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        g = epr_integrand_general(S)
        assert abs(g - epr_integrand_general(R @ S @ R.T)) < 1e-10 * (1 + g)
        assert abs(g - epr_integrand_general(7.3 * S)) < 1e-10 * (1 + g)


def test_sampling_is_data_processing(out):
    rates = [x["rate"] for x in out["synthetic"]["sampling"]]
    cont = out["synthetic"]["cont_exact"]
    assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))
    assert all(r < cont for r in rates)
    assert abs(rates[-1] - cont) < 0.06 * cont


def test_rectilinear_waves_are_reversible():
    vmax, dmax = wave_field_spectral_epr(n_modes=9, seed=5)
    assert vmax < 1e-12 and dmax < 1e-10
    # circular counter-demo: one rotating mode has nonzero density
    Ss, Sa = 1.0, 0.7
    S = np.array([[Ss, -1j * Sa], [1j * Sa, Ss]])
    assert epr_integrand_general(S) > 1.0


def test_gaussian_lower_bounds_true_epr():
    true = ring_true_epr(0.5, 0.1)
    gauss = ring_gaussian_epr(0.5, 0.1)
    assert 0.0 < gauss < true
    assert abs(ring_gaussian_epr(0.3, 0.3)) < 1e-10
    assert ring_true_epr(0.4, 0.4) == 0.0


def test_ocean_equator_is_reversible(out):
    r = out["real"]
    others = [r["bands"][n]["sigma_nats_per_step"]
              for n in BANDS if n != "eq"]
    assert r["eq_sigma"] <= min(others)
    assert r["eq_sigma"] < r["eq_null_q95"]
    assert r["nhtrop_sigma"] > 10.0 * r["nhtrop_null_q95"]


def test_ocean_tropical_epr_peaks_at_eddy_clock(out):
    for n in ("nh_trop", "sh_trop"):
        b = out["real"]["bands"][n]
        assert 20.0 < b["peak_period_days"] < 90.0
        for p in b["peak_periods_tapers"].values():
            assert 20.0 < p < 90.0
        assert b["frac_below_120d"] < 0.05


def test_ocean_polar_split_and_mirror(out):
    b = out["real"]["bands"]
    assert b["nh_polar"]["peak_period_days"] > 120.0
    assert b["nh_polar"]["frac_below_120d"] > 0.4
    assert b["sh_polar"]["peak_period_days"] < 90.0
    ratio = (b["nh_trop"]["sigma_nats_per_step"] /
             b["sh_trop"]["sigma_nats_per_step"])
    assert abs(np.log(ratio)) < np.log(3.0)


def test_ocean_no_determinant_clipping(out):
    for n in BANDS:
        assert out["real"]["bands"][n]["min_det_margin"] > 0.0
        assert out["real"]["bands"][n]["max_abs_CR"] < 1.0


def test_band_estimator_recovers_rotator_density():
    # sampled damped rotator: analytic C_s, C_a -> density peak near f
    dt, tau, f = 1.0, 6.0, 0.9
    m = np.arange(0, 13)
    Cs = np.exp(-m * dt / tau) * np.cos(f * m * dt)
    Ca = np.exp(-m * dt / tau) * np.sin(f * m * dt)
    thetas, Ss, Sa, dens, sigma = band_epr_spectrum(Cs, Ca)
    pk = thetas[int(np.nanargmax(dens))]
    assert abs(pk - f * dt) < 0.25
    assert sigma > 0.0


def test_real_bands_load_and_normalize():
    cache = load_bands()
    for n in BANDS:
        Cs, Ca = band_correlations(cache["bands"][n])
        assert Cs[0] == 1.0 and Ca[0] == 0.0
        assert len(Cs) == 13 and len(Ca) == 13


def test_figure_json_written():
    from new_relationships37 import FIG
    assert os.path.exists(FIG)
    import json
    with open(FIG) as fh:
        d = json.load(fh)
    assert all(d["verdicts"].values())
