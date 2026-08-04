"""Unit proofs for NR63 (`general_two_clocks/new_relationships40.py`):
active subglacial lake systems are gain-clock synchronized, not
transport-clock cascades (the NR49 gain/all-pass split on the real
ICESat-2 lake network).

Covered: the gain (zero-lag) / all-pass (lead-lag asymmetry)
decomposition and the signed-lead ordering; the synthetic method
validation (synchronous system -> gain, ~0 net lead; delayed-attenuated
chain -> recovered order and direction, all-pass above surrogate); the
phase-randomization surrogate calibration; and the real findings
(connected lakes synchronized above both the independent-noise null and
the between-system baseline; directed transport below the ICESat-2 floor
for the large systems).  Offline-safe: committed ATL15 cache.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships40 import (  # noqa: E402
    allpass, analyze, build_matrix, detrend_std, gain, lagged_corr,
    load_lakes, make_cascade, make_common_mode, net_lead_order,
    phase_randomize, signed_lead,
)


@pytest.fixture(scope="module")
def out():
    return analyze(write=False)


def test_gain_allpass_basic_properties():
    t = np.arange(40)
    rng = np.random.default_rng(0)
    a = detrend_std(np.cumsum(rng.standard_normal(40)), t)
    # identical series: gain 1, zero all-pass
    assert abs(gain(a, a) - 1.0) < 1e-9
    assert allpass(a, a) < 1e-9
    # a pure delayed copy: nonzero all-pass, signed lead recovers direction
    b = np.zeros(40)
    b[3:] = a[:-3]
    b = detrend_std(b, t)
    L, c = signed_lead(a, b)
    assert L > 0 and c > 0.5           # a leads b
    assert allpass(a, b) > 0.1


def test_lagged_corr_symmetry():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(30)
    b = rng.standard_normal(30)
    for L in (1, 3, 5):
        assert abs(lagged_corr(a, b, L) - lagged_corr(b, a, -L)) < 1e-12


def test_phase_randomize_preserves_spectrum():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(29))
    y = phase_randomize(x, rng)
    px = np.abs(np.fft.rfft((x - x.mean()) / (x.std() + 1e-9)))
    py = np.abs(np.fft.rfft(y))
    assert np.allclose(np.sort(px)[1:], np.sort(py)[1:], rtol=0.3, atol=0.5)
    assert abs(y.mean()) < 1e-9


def test_method_detects_synchronous_gain(out):
    s = out["synthetic"]
    assert s["sync_gain"] > 0.5
    g, q95 = s["sync_gain_vs_surrogate"]
    assert g > q95                       # surrogate calibrated


def test_method_detects_cascade_direction(out):
    s = out["synthetic"]
    assert s["cascade_order_rho"] > 0.85
    assert s["cascade_allpass"] > 1.5 * s["sync_allpass"]


def test_cascade_order_recovered_directly():
    tt = np.arange(60)
    Xk = np.array([detrend_std(x, tt) for x in make_cascade(seed=3)])
    r = net_lead_order(Xk)
    from scipy.stats import spearmanr
    # net lead should decrease down the chain (source leads)
    assert abs(spearmanr(r, np.arange(len(Xk))).statistic) > 0.85


def test_real_connected_lakes_synchronized(out):
    r = out["real"]
    assert r["within_median"] > r["surrogate_within_q95"]
    assert r["within_p_vs_surrogate"] < 0.05
    assert r["within_median"] > 0.4


def test_real_within_exceeds_between(out):
    r = out["real"]
    assert r["within_median"] > r["between_median"]
    assert r["mwu_within_gt_between_p"] < 1e-6


def test_real_transport_below_floor(out):
    r = out["real"]
    # among the LARGE systems (n>=10) the lead-lag ordering does not align
    # with along-flow geometry -> directed transport below the ATL15 floor
    big = [v["along_axis_rho"] for v in r["systems"].values() if v["n"] >= 10]
    assert len(big) >= 3
    assert max(big) < 0.4


def test_real_matrix_shapes():
    cache = load_lakes()
    X, sysname, pos, t = build_matrix(cache)
    assert X.shape[0] == 131 and X.shape[1] == len(t)
    assert len(sysname) == 131 and pos.shape == (131, 2)
    # detrended + standardized
    assert abs(X[0].mean()) < 1e-6 and abs(X[0].std() - 1.0) < 1e-3


def test_figure_written_all_true():
    from new_relationships40 import FIG
    import json
    assert os.path.exists(FIG)
    with open(FIG) as fh:
        d = json.load(fh)
    assert all(d["verdicts"].values())
