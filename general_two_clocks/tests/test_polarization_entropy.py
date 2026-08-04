"""Unit-proofs for NR67 (new_relationships44.py) — Lagrangian polarization entropy
and the parity split of purity. Exact optics identities + the ANDRO read-out.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

import new_relationships44 as R  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(R.STOKES_CACHE)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed NR56 Stokes cache missing")


# ---------------------------------------------------------- exact optics identities
@pytest.mark.parametrize("Q,U,V", [(0.3, -0.2, 0.4), (0.1, 0.5, -0.6),
                                   (0.7, 0.0, 0.2), (0.0, 0.0, 0.0)])
def test_coherency_hermitian_trace_one_and_eigenvalues(Q, U, V):
    J = R.coherency_matrix(Q, U, V)
    assert np.allclose(J, J.conj().T)
    assert abs(np.trace(J).real - 1.0) < 1e-12
    w = np.linalg.eigvalsh(J)
    p = math.sqrt(Q * Q + U * U + V * V)
    assert np.allclose(sorted(w), [(1 - p) / 2, (1 + p) / 2])
    assert w.min() >= -1e-12                       # positive semidefinite


@pytest.mark.parametrize("Q,U,V", [(0.3, -0.2, 0.4), (0.1, 0.5, -0.6),
                                   (0.7, 0.0, 0.2)])
def test_entropy_matches_von_neumann(Q, U, V):
    J = R.coherency_matrix(Q, U, V)
    w = np.linalg.eigvalsh(J)
    S_vn = -sum(l * math.log2(l) for l in w if l > 0)
    S, p = R.polarization_entropy(Q, U, V)
    assert abs(S - S_vn) < 1e-12
    assert 0.0 <= S <= 1.0


def test_pure_states_zero_entropy_opposite_parity():
    Sw, pw = R.polarization_entropy(1.0, 0.0, 0.0)   # wave
    Sv, pv = R.polarization_entropy(0.0, 0.0, 1.0)   # vortex
    assert Sw < 1e-12 and Sv < 1e-12                 # both pure
    assert abs(pw - 1) < 1e-12 and abs(pv - 1) < 1e-12
    assert R.parity_split(1, 0, 0)[2] == 0.0         # wave: fully reversible
    assert R.parity_split(0, 0, 1)[2] == 1.0         # vortex: fully irreversible


def test_isotropic_is_max_entropy():
    S, p = R.polarization_entropy(0.0, 0.0, 0.0)
    assert abs(S - 1.0) < 1e-12 and p == 0.0


def test_entropy_monotone_decreasing_in_purity():
    ps = np.linspace(0, 1, 50)
    S = [R.polarization_entropy(p, 0.0, 0.0)[0] for p in ps]
    assert all(S[i] >= S[i + 1] - 1e-12 for i in range(len(S) - 1))


# ---------------------------------------------------------- parity split theorem
def test_parity_split_pythagoras_and_rotation_invariance():
    Q, U, V = 0.36, -0.15, 0.28
    pe, po, f = R.parity_split(Q, U, V)
    assert abs(pe ** 2 + po ** 2 - (Q ** 2 + U ** 2 + V ** 2)) < 1e-12
    # rotate (Q,U) by 2θ: f_odd invariant, entropy invariant
    th = 0.7
    Qr = Q * math.cos(2 * th) + U * math.sin(2 * th)
    Ur = -Q * math.sin(2 * th) + U * math.cos(2 * th)
    _, _, fr = R.parity_split(Qr, Ur, V)
    assert abs(f - fr) < 1e-12
    assert abs(R.polarization_entropy(Q, U, V)[0]
               - R.polarization_entropy(Qr, Ur, V)[0]) < 1e-12


def test_f_odd_bounds_and_mixture():
    assert R.parity_split(0.5, 0.0, 0.0)[2] == 0.0
    assert R.parity_split(0.0, 0.0, 0.5)[2] == 1.0
    _, _, f = R.parity_split(math.sqrt(0.5), 0.0, math.sqrt(0.5))
    assert abs(f - 0.5) < 1e-12


def test_synthetic_verdicts():
    syn = R.synthetic_checks()
    assert all(c["eig_matches"] and c["entropy_matches"]
               for c in syn["eig_entropy_checks"])
    assert syn["parity_split"]["pythagoras"]
    assert syn["parity_split"]["rotation_invariant_f_odd"]


# ---------------------------------------------------------- real ANDRO data
@needs_cache
def test_ocean_equator_is_coherent_minimum():
    oc = R.ocean_entropy()
    assert oc["n_floats"] > 5000
    # equator least depolarized (higher p, lower entropy) than poleward
    assert oc["equator"]["p"] > oc["poleward"]["p"]
    assert oc["equator"]["S"] < oc["poleward"]["S"]
    # and most reversible (lowest circular fraction)
    assert oc["equator"]["f_odd"] < oc["poleward"]["f_odd"]


@needs_cache
def test_ocean_per_float_parity_and_entropy_trends():
    oc = R.ocean_entropy()
    pf = oc["per_float"]
    # irreversible fraction rises with |f| (reactive parity face of NR66)
    assert pf["f_odd_vs_absf"]["rho"] > 0 and pf["f_odd_vs_absf"]["p"] < 1e-6
    # entropy rises poleward (equatorial coherent jets are the pure limit)
    assert pf["entropy_vs_abslat"]["rho"] > 0 and pf["entropy_vs_abslat"]["p"] < 1e-6
    assert pf["poleward_mean_f_odd"] > pf["equator_mean_f_odd"]


@needs_cache
def test_full_verdict():
    res = R.analyze()
    v = res["verdict"]
    assert v["eig_and_entropy_exact"]
    assert v["parity_pythagoras_and_invariant"]
    assert v["ocean_equator_least_depolarized"]
    assert v["ocean_equator_most_reversible"]
    assert v["ocean_per_float_f_odd_rises"]
    assert v["ocean_per_float_entropy_rises_poleward"]
