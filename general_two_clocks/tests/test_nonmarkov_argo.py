"""Unit tests for Paper 4 — non-Markovian multiscale memory
(general_two_clocks/nonmarkov_argo.py).

Validate the autocorrelation, the Chapman-Kolmogorov excess, and the single/double
exponential model selection on deterministic synthetic processes (no Argo download):
AR(1) must read as Markov (excess ~0, single-exp); a two-timescale process must read
as non-Markovian (excess>0, double-exp, separated timescales). Seeded, CPU-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nonmarkov_argo as M  # noqa: E402


def test_excess_zero_for_geometric_acf():
    rho = 0.7
    C = rho ** np.arange(20)            # exact AR(1) autocorrelation
    assert abs(M.nonmarkov_excess(C)) < 1e-12
    assert abs(M.nonmarkov_excess_k(C, 3)) < 1e-12
    assert abs(M.nonmarkov_excess_k(C, 5)) < 1e-12


def test_acf_white_noise_is_flat():
    w = np.random.default_rng(0).standard_normal(5000)
    C = M.acf_nan(w, 20)
    assert C[0] == 1.0
    assert np.all(np.abs(C[1:]) < 0.1)


def test_acf_nan_aware():
    x = np.sin(np.arange(500) * 0.3)
    x[::7] = np.nan                      # scattered gaps
    C = M.acf_nan(x, 10)
    assert C[0] == 1.0 and np.isfinite(C[1])


def test_ar1_reads_as_markov():
    ex2 = [M.nonmarkov_excess(M.acf_nan(M.ar1(phi=0.8, seed=s), 30)) for s in range(20)]
    assert abs(np.median(ex2)) < 0.02                       # excess ~ 0
    dA = [M.fit_models(M.acf_nan(M.ar1(phi=0.8, seed=s), 30), step=1.0)["d_aic"]
          for s in range(20)]
    assert np.median(dA) < 50                               # double not preferred


def test_two_timescale_reads_as_nonmarkov():
    ex2 = [M.nonmarkov_excess(M.acf_nan(M.two_timescale(seed=s), 30)) for s in range(20)]
    assert np.median(ex2) > 0.02                            # positive CK excess
    f = M.fit_models(M.acf_nan(M.two_timescale(seed=0), 30), step=1.0)
    assert f["d_aic"] > 50 and f["sep"] > 3 and f["double_preferred"]


def test_fit_recovers_single_timescale():
    phi = 0.85
    C = M.acf_nan(M.ar1(phi=phi, seed=3), 40)
    f = M.fit_models(C, step=1.0)
    expected = -1.0 / np.log(phi)
    assert abs(f["tau_single"] - expected) / expected < 0.4
