"""Unit tests for the molecular foundation of the regularity program
(general_two_clocks/molecular_regularity.py).

These pin the three claims of the bottom rung of the
Newton -> Boltzmann -> compressible NS -> incompressible NS hierarchy:
(1) the hard-disk gas is regular by construction (exact conservation, no blowup);
(2) it relaxes to the Boltzmann level (Maxwell-Boltzmann + H-theorem);
(3) its pressure is collisional (the molecular origin of the EOS).
Deterministic (seeded), CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import molecular_regularity as M  # noqa: E402

RES = M.compare(seed=0)  # computed once (~2s)


# --------------------------------------------------------------------------- #
# (1) regular by construction
# --------------------------------------------------------------------------- #
def test_initial_state_is_valid():
    r, v = M.initial_state(N=100, L=30.0, R=0.5, v0=1.0, seed=0)
    # disks do not overlap (minimum-image pairwise distance > one diameter)
    dr = M._min_image(r[:, None, :] - r[None, :, :], 30.0)
    d = np.sqrt(np.einsum("ijk,ijk->ij", dr, dr))
    d[np.diag_indices(len(r))] = np.inf
    assert d.min() > 2.0 * 0.5
    # the beam start carries exactly zero total momentum
    assert np.allclose(v.sum(0), 0.0, atol=1e-12)


def test_regular_by_construction():
    # Newtonian elastic dynamics conserves momentum and energy to round-off ...
    assert RES["mom_residual"] < 1e-9
    assert RES["energy_relerr"] < 1e-9
    # ... and never hits a finite-time singularity (positive inter-event times)
    assert RES["min_dt"] > 0.0
    assert RES["n_events"] >= 1000


# --------------------------------------------------------------------------- #
# (2) the Boltzmann level emerges
# --------------------------------------------------------------------------- #
def test_h_theorem():
    # binned Boltzmann H decreases overall, with a negative trend
    assert RES["H1"] < RES["H0"]
    assert RES["H_slope"] < 0.0
    assert RES["H_q1"] - RES["H_q4"] > 0.3


def test_maxwell_boltzmann_relaxation():
    # v_x Gaussianises: kurtosis 1 (bimodal beam) -> ~3 (Gaussian)
    assert RES["kurt_vx_init"] < 1.5
    assert RES["kurt_vx_fin"] > 2.5
    # energy equipartitions from the x-beam into y (starts at ~0)
    assert RES["equip_init"] < 0.05
    assert RES["equip_fin"] > 0.7


# --------------------------------------------------------------------------- #
# (3) pressure is collisional (the EOS)
# --------------------------------------------------------------------------- #
def test_collisional_pressure():
    # the collisional virial is positive (repulsive) and lifts P above ideal-gas
    assert RES["P_coll"] > 0.0
    assert RES["P_ratio"] > 1.0
    assert RES["P_ratio"] < 1.6           # dilute: a modest excluded-volume correction


def test_compare_ok():
    assert RES["ok"] is True
