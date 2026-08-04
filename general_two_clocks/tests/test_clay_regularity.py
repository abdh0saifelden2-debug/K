r"""Tests for the conditional NS-regularity program (clay_regularity_program.py).

These verify the three pillars of the *conditional* Clay attack (NOT a proof of the
Millennium problem): the rigorous restricted-Euler blowup, the BKM vorticity-integral
bridge, and the Constantin-Fefferman geometric depletion measured in a real 3D NS DNS.
"""
from __future__ import annotations

import numpy as np

import clay_regularity_program as clay


def test_omega_from_velocity_gradient():
    """_omega_of extracts the correct vorticity vector from A_ij = du_i/dx_j."""
    # solid-body rotation about z: u=(-y, x, 0) -> A antisymmetric, omega=(0,0,2)
    A = np.array([[0.0, -1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    om = clay._omega_of(A)
    assert np.allclose(om, [0.0, 0.0, 2.0])


def test_vieillefosse_blowup_rigorous():
    """The restricted-Euler (local pressure Hessian) model blows up in finite time on
    the Vieillefosse tail with the exact analytic rate -- the rigorous statement that
    discarding the nonlocal Hessian destroys regularity."""
    r = clay.vieillefosse_blowup()
    assert r["blew_up"]
    assert r["t_star"] < 50.0
    assert abs(r["tail_ratio"] - 1.0) < 1e-2          # on the Vieillefosse tail
    assert r["H_drift"] < 1e-3                         # exact RE invariant conserved
    assert r["ok"]


def test_bkm_bridge_divergence_vs_finiteness():
    """BKM criterion: the restricted-Euler vorticity integral int|omega|dt grows
    without bound as t -> t* (blowup), while the nonlocal-Hessian closure keeps it
    finite over a long horizon (regular)."""
    b = clay.bkm_bridge()
    assert b["re_blew_up"]
    # BKM integral strictly increasing along the horizons (accumulating)
    vals = [bk for _, _, bk in b["re_bkm_curve"]]
    assert all(v2 > v1 for v1, v2 in zip(vals, vals[1:]))
    # and accelerating toward t* (last gap > first gap): signature of divergence
    gaps = [v2 - v1 for v1, v2 in zip(vals, vals[1:])]
    assert gaps[-1] > gaps[0]
    # the nonlocal closure stays finite and bounded
    assert not b["reg_blew_up"]
    assert np.isfinite(b["reg_bkm"])
    assert b["reg_bkm"] < vals[-1]                     # finite vs diverging
    assert b["ok"]


def test_constantin_fefferman_depletion_in_dns():
    """In a real forced-3D-NS field, vorticity aligns with the INTERMEDIATE strain
    eigenvector (not the most-stretching one), the intermediate eigenvalue is positive,
    and the self-stretching is depleted below its maximal value -- the geometric
    depletion the conditional regularity theorems rely on."""
    d = clay.alignment_depletion()                     # default n=32, 600 steps (~15s)
    # intermediate-eigenvector alignment (the hallmark NS result)
    assert d["align_e2"] > d["align_e1"]
    assert d["align_e2"] > d["align_e3"]
    assert d["align_e3"] < 1.0 / 3.0                   # vorticity avoids compression
    # positive strain skewness (forward-cascade geometry)
    assert d["lam2_norm"] > 0.0
    # geometric depletion: stretching well below the fully-aligned (maximal) value
    assert 0.0 < d["depletion_delta"] < 1.0
    assert d["ok"]
