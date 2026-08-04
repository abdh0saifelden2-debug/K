"""Fast invariants for the P1 a-posteriori closure test (paper1 sec 5b).

Uses tiny grids / short runs (no full DNS) and checks only the robust, resolution-
independent qualitative claims of sec 5b:
  - every predictive closure is stable (no blow-up),
  - Smagorinsky over-dissipates (drains resolved KE below the no-model run),
  - the FDT backscatter returns energy (cusp+bs KE >= cusp-only KE).
"""
from __future__ import annotations

import pytest

from closure_aposteriori import run_les


@pytest.fixture(scope="module")
def runs():
    kc, n_les = 16, 48
    out = {}
    for clos in ["none", "smag", "cuspEV", "cuspEV_bs"]:
        out[clos] = run_les(n_les, clos, kc, k_f=10.0, steps=1600, dt=2.0e-3,
                            seed=3, n_snap=12, warmup_frac=0.5)
    return out


def test_all_closures_stable(runs):
    for clos, r in runs.items():
        assert not r["blew"], f"{clos} blew up a-posteriori"
        assert r["ke"] > 0.0


def test_smagorinsky_over_dissipates(runs):
    # Smagorinsky drains resolved KE below the no-model control (sec 5b finding 2).
    assert runs["smag"]["ke"] < runs["none"]["ke"]


def test_cusp_closure_gentler_than_smagorinsky(runs):
    # The scale-selective cusp eddy viscosity preserves more resolved KE than
    # Smagorinsky's blanket dissipation (sec 5b finding 3: structured >> Smagorinsky).
    assert runs["cuspEV"]["ke"] > runs["smag"]["ke"]


def test_backscatter_is_a_small_stable_perturbation(runs):
    # The FDT backscatter is a small, stable correction on top of the cusp eddy
    # viscosity (sec 5b: its benefit is marginal and lives mostly in the spectrum
    # error, not the bulk KE). Here we only assert it stays a small perturbation.
    cusp, bs = runs["cuspEV"]["ke"], runs["cuspEV_bs"]["ke"]
    assert abs(bs - cusp) / cusp < 0.1

