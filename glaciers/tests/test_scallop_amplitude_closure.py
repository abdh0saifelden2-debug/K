"""Tests for the scallop amplitude (a/λ) closure (``scallop_amplitude_closure``)
— the measurement that closes **Caveat D** (FUTURE_WORK §A.2/§G.6).

Two layers, both CPU-only and DNS-free:

  1. :func:`fit_closure` is exercised on *synthetic* Nu(a/λ) curves with a known
     exponent — it must recover ``p≈2`` for a quadratic ``δ_T,eff`` closure,
     ``p≈1`` for a linear one, and flag an amplitude-INDEPENDENT deficit (``p≈0``,
     both power laws rejected) — so the fit logic that delivers the verdict is
     trustworthy independent of the (expensive) solver run.

  2. the committed production artifact ``figures/59_scallop_amplitude_closure.json``
     is checked for the structural result it records: the spatial-mean normal flux
     is sub-flat (``Nu/Nu_flat < 1``) at *every* amplitude, the local lee
     enhancement ``R_max`` grows with ``a/λ`` while the mean deficit does **not**,
     and the §G.6 quadratic ``(a/λ)²`` ansatz is rejected (worse than a constant).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scallop_amplitude_closure as ac  # noqa: E402

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures", "59_scallop_amplitude_closure.json")


def test_fit_closure_recovers_known_exponents():
    """fit_closure recovers the generating exponent: p≈2 for the §G.6 quadratic
    δ_T,eff closure, p≈1 for a linear one, p≈0 (amplitude-independent, both power
    laws rejected) for a constant deficit."""
    x = np.array([0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])

    # quadratic D = ζ x² : Nu = 1/(1+ζ x²)
    nu_q = 1.0 / (1.0 + 8.0 * x ** 2)
    fq = ac.fit_closure(x, nu_q)
    assert abs(fq["p_free"] - 2.0) < 0.1
    assert fq["r2_quadratic_p2"] > 0.999
    assert abs(fq["zeta_quadratic_p2"] - 8.0) < 0.1
    assert "quadratic" in fq["verdict"]

    # linear D = ζ x : Nu = 1/(1+ζ x)
    nu_l = 1.0 / (1.0 + 2.0 * x)
    fl = ac.fit_closure(x, nu_l)
    assert abs(fl["p_free"] - 1.0) < 0.1
    assert fl["r2_linear_p1"] > 0.999
    assert "linear" in fl["verdict"]

    # flat: a scattered but trendless deficit (like the measured data) -> p≈0,
    # both power laws rejected, verdict flags amplitude-independence.
    D_flat = np.array([0.10, 0.15, 0.07, 0.05, 0.13, 0.11, 0.12])
    nu_f = 1.0 / (1.0 + D_flat)
    ff = ac.fit_closure(x, nu_f)
    assert abs(ff["p_free"]) < 0.5
    assert ff["r2_quadratic_p2"] <= 0.0          # quadratic worse than a constant
    assert "amplitude-INDEPENDENT" in ff["verdict"]
    assert abs(ff["D_mean"] - D_flat.mean()) < 1e-6


def test_amplitude_closure_artifact():
    """The committed production sweep records the Caveat-D closure: sub-flat mean
    Nu at every amplitude, a growing *local* lee enhancement, and a falsified
    §G.6 quadratic amplitude law (deficit is amplitude-independent)."""
    with open(ART) as f:
        d = json.load(f)

    cfg = d["config"]
    assert cfg["nx"] == 128 and cfg["ny"] == 128      # production resolution
    assert cfg["n_waves"] == 12                       # fluid-selected wavelength

    rows = d["rows"]
    assert len(rows) >= 6
    nu = np.array([r["Nu_ratio"] for r in rows])
    rmax = np.array([r["R_max"] for r in rows])
    a_over = np.array([r["a_over_lam"] for r in rows])

    # (1) spatial-mean normal flux is sub-flat at EVERY amplitude (Nu<1)
    assert np.all(nu < 1.0)
    # (2) local lee enhancement R_max grows strongly with amplitude (escapes the
    #     conduction limit locally) while the mean stays suppressed
    assert rmax[-1] > rmax[0]
    assert rmax[a_over.argmax()] > 1.5 * rmax[a_over.argmin()]

    fit = d["closure_fit"]
    # (3) the §G.6 quadratic (a/λ)^2 closure is falsified: no power-law in a/λ
    #     (p≈0), and the pinned quadratic fit is worse than a constant (R²<0).
    assert abs(fit["p_free"]) < 0.5
    assert fit["r2_quadratic_p2"] < 0.0
    assert fit["r2_quadratic_p2"] < fit["r2_logD_free"]
    assert "amplitude-INDEPENDENT" in fit["verdict"]
    # (4) the (non-)trend is physical: the amplitude deficit is small & flat, with
    #     a mean Nu/Nu_flat well below 1 (the §G.1 area-partition suppression).
    assert 0.0 < fit["D_mean"] < 0.25
    assert fit["Nu_ratio_mean_over_grid"] < 0.97
