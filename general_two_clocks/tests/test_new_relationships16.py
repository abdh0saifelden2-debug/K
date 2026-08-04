"""Unit proofs for NR39 (`general_two_clocks/new_relationships16.py`):
imaginary coherence = instantaneous-mixing projection invariance.  Ledger E7.

Covered: the algebraic lemma (real mixing of real-spectrum sources has zero
imaginary cross-spectrum) and its failure mode; that pure instantaneous mixing
sits at the estimator null while a single injected lagged path is detected on
exactly that pair; the volume-conduction Re>>Im signature; and the P0 phase-
surrogate ceiling.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships16 import (  # noqa: E402
    max_imag_lagged_source, max_imag_real_mixing, run,
)


@pytest.fixture(scope="module")
def out():
    o, _ = run(seed=0)
    return o


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_real_instantaneous_mixing_has_zero_imaginary_cross_spectrum(seed):
    """The lemma: real A, real diagonal S_s => Im(A S_s A^T) = 0 exactly."""
    assert max_imag_real_mixing(seed) < 1e-12


def test_lagged_source_breaks_the_lemma():
    """A single complex-Hermitian (lagged) source coupling makes Im nonzero."""
    assert max_imag_lagged_source(0) > 0.1


def test_pure_mixing_sits_at_the_null(out):
    sa = out["synthetic_array"]
    assert sa["max_abs_imcoh_pure_mixing"] < 3 * sa["analytic_null_1sigma"]


def test_injected_lagged_pair_is_detected(out):
    sa = out["synthetic_array"]
    assert sa["injected_over_null"] > 4.0
    # and it stands out from every other (pure-mixing) pair
    assert sa["imcoh_injected_pair"] > 5.0 * sa["max_imcoh_other_pairs"]


def test_mixing_lives_on_the_real_axis(out):
    """Volume-conduction signature: Re coherency >> Im coherency."""
    assert out["synthetic_array"]["mixing_dominated_re_over_im"] > 5.0


def test_clears_phase_surrogate_ceiling(out):
    """Corollary 2 (P0 ceiling): the real coupling exceeds its surrogate p95."""
    sc = out["surrogate_ceiling"]
    assert sc["clears_ceiling"]
    assert sc["injected_imcoh"] > 2.0 * sc["surrogate_p95"]
