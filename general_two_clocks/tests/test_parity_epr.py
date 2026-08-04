"""Unit proofs for NR62 (`general_two_clocks/new_relationships39.py`): the
reversible reference phase is set by time-parity (0 deg equal / +-90 deg
opposite), and entropy production is the departure from it.

Covered: the parity-aware KL integrand and its 2x2 closed form for both
parity products; the even-odd oscillator (v=dx/dt => S_xv purely
imaginary) is reversible with parity (Re carrier 0) while the naive
equal-parity reading flags it irreversible; two-temperature coupling is
irreversible (equal parity) and reversible at equal T; the non-reciprocal
coupling raises EPR; parity EPR non-negativity; and the real solar
readouts (p-modes near the -90-deg reference, naive over-counts 8.5x,
epoch stability, convective background irreversible, V-V control flags the
instrument delay).  Offline-safe: committed NR46 cache.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships39 import (  # noqa: E402
    analyze, coherency, damped_oscillator_xspectrum, epr_integrand_2x2_parity,
    epr_integrand_parity, nonreciprocal_osc, parity_reversal, sde_epr,
    sde_psd, two_temperature,
)


@pytest.fixture(scope="module")
def out():
    return analyze(write=False)


def _spd(rng):
    X = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    return X @ X.conj().T + 0.05 * np.eye(2)


def test_closed_form_both_parities():
    rng = np.random.default_rng(0)
    for _ in range(300):
        S = _spd(rng)
        for eps in (np.array([1, 1]), np.array([1, -1]), np.array([-1, 1])):
            g = epr_integrand_parity(S, eps)
            assert abs(g - epr_integrand_2x2_parity(S, eps)) < 1e-9 * (1 + g)
            assert g >= -1e-9                          # non-negative


def test_parity_reversal_involutive_and_equalparity_matches_naive():
    rng = np.random.default_rng(1)
    for _ in range(50):
        S = _spd(rng)
        eps = np.array([1, -1])
        assert np.allclose(parity_reversal(parity_reversal(S, eps), eps), S)
        # equal parity reduces to the plain transpose reversal (NR60/61)
        g_eq = epr_integrand_parity(S, np.array([1, 1]))
        gen = float(np.real(np.trace(np.linalg.solve(S.T, S))) - 2)
        assert abs(g_eq - gen) < 1e-9 * (1 + abs(gen))


def test_oscillator_quadrature_reversible_with_parity(out):
    re_max, im_min = damped_oscillator_xspectrum()
    assert re_max < 0.05          # Re coherency ~ 0 -> opposite-parity: rev.
    assert im_min > 0.3           # Im coherency O(1) -> naive: "irreversible"
    assert out["synthetic"]["osc_re_carrier_max"] < 0.05
    assert out["synthetic"]["osc_im_carrier_min"] > 0.3


def test_two_temperature_irreversible_and_equalT_reversible(out):
    s = out["synthetic"]
    assert s["two_temp_epr"] > 1e-3
    assert abs(s["two_temp_equal_epr"]) < 1e-6
    # explicit heat-current monotonicity in Delta T
    e1 = sde_epr(*two_temperature(T1=1.0, T2=2.0)[:2], np.array([1, 1]))
    e2 = sde_epr(*two_temperature(T1=1.0, T2=4.0)[:2], np.array([1, 1]))
    assert e2 > e1 > 0


def test_nonreciprocal_increases_epr(out):
    s = out["synthetic"]
    assert s["nonrecip_epr"] > s["nonrecip_epr_a0"] > 1e-3


def test_parity_epr_nonnegative(out):
    assert out["synthetic"]["positivity_min"] > -1e-9


def test_sde_psd_hermitian_pd():
    A, D, _ = two_temperature()
    for w in (-2.0, 0.3, 1.7):
        S = sde_psd(A, D, w)
        assert np.allclose(S, S.conj().T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(S) > 0)


def test_solar_pmodes_near_adiabatic_reference(out):
    so = out["solar"]
    assert so["nonadiab_departure_med_deg"] < 30.0
    assert -140.0 < so["mode_phase_med_deg"] < -80.0


def test_solar_naive_overcounts(out):
    so = out["solar"]
    assert so["naive_overcount"] > 3.0
    assert so["epr_naive_med"] > so["epr_parity_med"] > 0


def test_solar_irreversibility_epoch_stable(out):
    so = out["solar"]
    assert so["n_epochs"] >= 10
    assert so["epoch_phase_std_deg"] < 5.0
    assert so["epoch_cos2_std"] < 0.05


def test_solar_background_irreversible(out):
    assert out["solar"]["back_median_departure_deg"] > 40.0


def test_vv_control_flags_instrument_delay(out):
    so = out["solar"]
    assert so["vv_top_g2"] > 0.8
    assert 0.0 < so["vv_quad_carrier"] < 0.5


def test_figure_written_all_true():
    from new_relationships39 import FIG
    import json
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        d = json.load(fh)
    assert all(d["verdicts"].values())
