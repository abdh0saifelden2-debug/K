"""Unit proofs for NR61 (`general_two_clocks/new_relationships38.py`):
imaginary coherency is the pairwise entropy-production spectral density.

Covered: the exact 2x2 closure tr[S^{-T}S-I] = 4(Im S12)^2/det S = 4
ImCoh^2/(1-|Coh|^2) and cross-consistency with NR60; the Qian d=1
always-reversible theorem; volume conduction (real instantaneous mixing)
is reversible for any mixing; a pure inter-channel delay is irreversible
with ImCoh sign = direction and EPR magnitude direction-invariant; VAR(1)
unidirectional coupling is irreversible (no-coupling reversible), the
reciprocal-symmetric OU reversible; the all-pass (phi=pi/2) is the maximal
irreversibility at fixed coherence and the instantaneous (phi=0) the
reversible null; and the real-EEG readouts (volume-conduction pairs most
reversible, irreversibility rises with distance while VC falls, network
majority-irreversible).  Offline-safe: committed NR39 cache.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships38 import (  # noqa: E402
    analyze, coherency, epr_density_from_coherency, epr_integrand_2x2,
    epr_integrand_general, instantaneous_mix_psd, integrate_density,
    var1_epr, var1_psd,
)
import new_relationships37 as nr60  # noqa: E402


@pytest.fixture(scope="module")
def out():
    return analyze(write=False)


def _spd(rng):
    X = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    return X @ X.conj().T + 0.05 * np.eye(2)


def test_closed_form_and_nr60_consistency():
    rng = np.random.default_rng(0)
    for _ in range(300):
        S = _spd(rng)
        g = epr_integrand_general(S)
        assert abs(g - epr_integrand_2x2(S)) < 1e-11 * (1 + abs(g))
        # identical object to NR60's rotary/Stokes integrand
        assert abs(g - nr60.epr_integrand_general(S)) < 1e-12 * (1 + abs(g))
        assert abs(g - nr60.epr_integrand_stokes(S)) < 1e-11 * (1 + abs(g))


def test_one_dimensional_always_reversible(out):
    assert out["synthetic"]["oneD_max_epr"] < 1e-12
    rng = np.random.default_rng(1)
    for _ in range(100):
        assert abs(epr_integrand_general(
            np.array([[rng.uniform(0.1, 5.0) + 0j]]))) < 1e-13


def test_volume_conduction_is_reversible(out):
    assert out["synthetic"]["vc_reversible_max_epr"] < 1e-10
    # explicit: any real mixing of independent real auto-spectra -> 0
    thetas = np.linspace(-np.pi, np.pi, 2001)
    M = np.array([[1.3, 0.7], [0.4, 0.9]])
    S = instantaneous_mix_psd(lambda t: 1.0 / (1.3 - np.cos(t)),
                              lambda t: 2.0 / (1.6 - 0.5 * np.cos(t)),
                              M, thetas)
    integ = np.array([epr_integrand_general(s) for s in S])
    assert abs(integrate_density(thetas, integ)) < 1e-10
    for s in S[::200]:
        assert abs(coherency(s)[1]) < 1e-10          # ImCoh == 0


def test_pure_delay_irreversible_and_signed(out):
    s = out["synthetic"]
    assert s["delay_epr_pos"] > 1e-3
    assert s["delay_imcoh_pos"] * s["delay_imcoh_neg"] < 0
    assert abs(s["delay_epr_pos"] - s["delay_epr_neg"]) < 1e-6


def test_var_coupling_direction_and_reversibility(out):
    s = out["synthetic"]
    assert s["var_uni_epr"] > 1e-3
    assert abs(s["var_nocpl_epr"]) < 1e-12
    assert abs(s["var_uni_epr"] - s["var_rev_epr"]) < 1e-9
    assert s["var_imcoh_uni"] * s["var_imcoh_rev"] < 0
    # a diagonal (uncoupled) VAR has a diagonal PSD -> reversible
    epr, thetas, S = var1_epr(np.diag([0.6, 0.3]), np.eye(2))
    assert abs(epr) < 1e-12


def test_reciprocal_symmetric_reversible(out):
    s = out["synthetic"]
    assert abs(s["ou_reciprocal_reversible"]) < 1e-9
    assert s["ou_antisym_irreversible"] > 1e-6


def test_allpass_is_maximal_irreversibility(out):
    dens = out["synthetic"]["phase_sweep_density"]
    assert abs(dens[0]) < 1e-12                       # phi=0 instantaneous
    assert abs(dens[-1]) < 1e-12                      # phi=pi
    assert np.argmax(dens) == len(dens) // 2          # phi=pi/2 all-pass
    # closed form: density = r^2 sin^2 phi/(1-r^2)
    r = 0.6
    for p in np.linspace(0, np.pi, 13):
        want = r ** 2 * np.sin(p) ** 2 / (1 - r ** 2)
        got = epr_density_from_coherency(r * np.sin(p), r)
        assert abs(got - want) < 1e-12


def test_mirror_reversal_invariance():
    rng = np.random.default_rng(2)
    for _ in range(50):
        S = _spd(rng)
        M = np.diag([1.0, -1.0])
        assert abs(epr_integrand_general(S) -
                   epr_integrand_general(M @ S @ M)) < 1e-11
        assert abs(epr_integrand_general(S) -
                   epr_integrand_general(S.T)) < 1e-11   # T-reversal even


def test_eeg_volume_conduction_most_reversible(out):
    e = out["eeg"]
    assert e["vc_median_coh"] > e["median_coh"]
    assert e["vc_median_epr"] < e["median_epr"]
    assert e["vc_frac_above_surrogate"] < 0.6 * e["frac_above_surrogate"]


def test_eeg_irreversibility_rises_with_distance(out):
    e = out["eeg"]
    assert e["sp_epr_dist"] > 0
    assert e["sp_im_dist"] > 0
    assert e["sp_re_dist"] < 0
    assert e["sp_re_im"] < 0                           # NR39 mixing lemma


def test_eeg_network_majority_irreversible(out):
    e = out["eeg"]
    assert e["frac_above_surrogate"] > 0.5
    assert e["reversible_null_share"] < 0.3
    assert e["total_epr"] > 0


def test_var_psd_hermitian_pd():
    thetas = np.linspace(-np.pi, np.pi, 101)
    S = var1_psd(np.array([[0.5, 0.1], [0.3, 0.4]]), np.eye(2), thetas)
    for s in S:
        assert np.allclose(s, s.conj().T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(s) > 0)


def test_figure_written_all_true():
    from new_relationships38 import FIG
    import json
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        d = json.load(fh)
    assert all(d["verdicts"].values())
