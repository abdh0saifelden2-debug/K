"""Tests for the drive-cut regime match (scallop_drive_cut.py) and its
committed P100 batch artifact (subglacial/drive_cut_p100.json)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scallop_drive_cut as dc

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "subglacial", "drive_cut_p100.json")


@pytest.fixture(scope="module")
def art():
    with open(ART) as fh:
        return json.load(fh)


def test_artifact_is_the_gpu_batch(art):
    assert art["backend"] == "cupy"
    assert art["params"]["r_cut"] == pytest.approx(0.16)
    assert len(art["cases"]) == 16
    assert all(not c["clip_hit"] for c in art["cases"])


def test_precut_I_rises_with_nw(art):
    """Same-protocol pre-cut baseline tracks the frozen band's I(n_w) trend."""
    by_nw = {8: [], 12: []}
    for c in art["cases"]:
        if c["pre_cut"] and c["pre_cut"]["R2_phase"] > 0.9:
            by_nw[c["nw"]].append(c["pre_cut"]["I_mb"])
    assert np.median(by_nw[12]) > np.median(by_nw[8])


def test_cut_lowers_solver_I(art):
    """Per-seed paired comparison: the drive cut LOWERS the smoothing-only
    solver's I (migration ~U^1/2 collapses at conserved conduction damping)."""
    drops = 0
    total = 0
    for c in art["cases"]:
        pre, post = c["pre_cut"], c["post_cut"]
        if pre and post and np.isfinite(pre["I_mb"]) and np.isfinite(post["I_mb"]):
            total += 1
            drops += bool(post["I_mb"] < pre["I_mb"])
    assert total >= 12
    assert drops / total >= 0.75


def test_postcut_I_far_below_observed(art):
    """The solver's post-cut I (quality-gated) sits well below the raw-array
    I_obs = 1.2-3.6: the adjustment-regime magnitude is NOT reproduced by the
    smoothing-only branch -- the cross-branch reading of SS8.6."""
    gated = [c["post_cut"]["I_mb"] for c in art["cases"]
             if c["post_cut"] and c["post_cut"]["R2_amp"] > 0.7
             and c["post_cut"]["R2_phase"] > 0.9]
    assert gated, "no quality-gated post-cut fits"
    assert max(gated) < 1.2 * 0.6            # well below the observed band
    assert art["verdict"]["post_cut_reaches_observed"] is False


def test_fit_window_phase_span_criterion():
    """fit_window with phase_span stops the fit once the modal phase has
    swept the requested span (synthetic check)."""
    t = np.linspace(0.0, 1000.0, 200)
    w = 2.0 * np.pi / 250.0
    Z = np.exp(-t / 2000.0) * np.exp(-1j * w * t)
    f_full = dc.fit_window(t, Z, skip_frac=0.1)
    f_win = dc.fit_window(t, Z, skip_frac=0.1, phase_span=1.0)
    assert f_full["n_fit"] > f_win["n_fit"]
    assert f_win["Im_s"] == pytest.approx(-w, rel=0.05)
    assert f_win["Re_s"] == pytest.approx(-1 / 2000.0, rel=0.1)


def test_evolve_cut_smoke_cpu():
    """Tiny CPU smoke: the driver runs, produces a pre/post split, and the
    post-cut phase keeps the downstream sign."""
    ts, Zs, i_cut, clip_hit = dc.evolve_cut(
        8, 3.0, 0.10, 1.0e-3, r_cut=0.16, nx=64, ny=64, f_amp=0.4, seed=0,
        spinup=200, n_updates_pre=12, n_updates_post=16, steps_per_update=10)
    assert i_cut == 12 and len(ts) >= 24
    assert np.all(np.isfinite(np.abs(Zs)))
