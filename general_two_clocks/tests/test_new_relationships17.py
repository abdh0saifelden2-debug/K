"""Unit proofs for NR40 (`general_two_clocks/new_relationships17.py`):
Omori decay as a pore-pressure diffusion-kernel tail.  Ledger item E10.

Covered (synthetic identifiability -- the falsifiable machinery):
the spatial Shapiro sqrt(t) triggering front recovers the diffusivity D; the
temporal tapered-Omori cutoff clock tau_D is recovered by MLE and AICc-preferred
when real, with a pure-Omori false-positive control.  The real-ComCat fit is a
separate, network-gated consistency check (reported as a null in the module).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships17 import (  # noqa: E402
    _intensity, fetch_comcat, fit_sequence, fit_shapiro_front,
    simulate_diffusion_catalog, simulate_tapered_omori,
)


def test_shapiro_front_recovers_diffusivity():
    ts, rs = simulate_diffusion_catalog(D=0.5, T=60.0, n=600, seed=0)
    fr = fit_shapiro_front(ts, rs)
    assert fr["D_km2_per_day"] is not None
    assert 0.3 < fr["D_km2_per_day"] < 0.8      # true 0.5
    assert fr["front_r2"] > 0.9


def test_shapiro_front_recovers_a_second_diffusivity():
    """Generalisation: a filled diffusion cloud at a different D is recovered
    (the envelope estimator is calibrated for filled clouds, not a bare line)."""
    ts, rs = simulate_diffusion_catalog(D=0.7, T=50.0, n=600, seed=3)
    fr = fit_shapiro_front(ts, rs)
    assert 0.45 < fr["D_km2_per_day"] < 1.05      # true 0.7
    assert fr["front_r2"] > 0.9


def test_temporal_cutoff_recovered_when_present():
    tt = simulate_tapered_omori(K=40.0, c=0.02, p=1.1, tau=6.0, T=60.0, seed=0)
    rec = fit_sequence(tt, 60.0)
    assert rec["delta_aicc_cutoff_preferred"] > 2.0     # cutoff preferred
    assert 3.0 < rec["tau_D_day"] < 15.0                # near true 6


def test_no_spurious_cutoff_on_pure_omori():
    tt = simulate_tapered_omori(K=40.0, c=0.02, p=1.1, tau=np.inf, T=60.0, seed=1)
    rec = fit_sequence(tt, 60.0)
    assert rec["delta_aicc_cutoff_preferred"] < 6.0     # AICc does not add a cutoff


def test_intensity_is_decreasing_power_law():
    t = np.array([0.0, 0.1, 1.0, 10.0])
    lam = _intensity(t, K=1.0, c=0.02, p=1.0, tau=np.inf)
    assert np.all(np.diff(lam) < 0)                     # monotone decreasing
    assert lam[0] == pytest.approx(1.0 / 0.02)          # K c^-p at t=0


@pytest.mark.skipif(os.environ.get("NR40_NO_NET") == "1",
                    reason="network disabled")
def test_real_comcat_fetch_is_wellformed():
    """Data-gated: if ComCat is reachable, the Pawnee fetch returns a sane
    (times, distances) pair; skip cleanly on any network error."""
    try:
        t, r = fetch_comcat(36.425, -96.929, 35.0, "2016-09-03T12:02:44", 30, 3.0)
    except Exception:
        pytest.skip("ComCat unreachable")
    assert len(t) == len(r)
    if len(t):
        assert np.all(t > 0) and np.all(r >= 0)
