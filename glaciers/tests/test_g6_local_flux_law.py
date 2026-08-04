"""Unit tests for §G.6 local lee-flux law
(validation/synthetic/g6_local_flux_law.py)."""
import os
import sys

import numpy as np

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g6_local_flux_law as G6  # noqa: E402


def test_sweep_loads_ordered():
    s = G6.load_sweep()
    x = s["a_over_lam"]
    assert len(x) == 8
    assert np.all(np.diff(x) > 0)                      # ordered by amplitude
    assert s["R_max"][0] < s["R_max"][-1]              # grows with amplitude


def test_local_flux_is_linear_not_quadratic():
    s = G6.load_sweep()
    f = G6.fit_local_flux(s["a_over_lam"], s["R_max"])
    # linear explains R_max well; the §G.6 quadratic is decisively worse
    assert f["linear"]["r2"] > 0.9
    assert f["quadratic_G6"]["r2"] < f["linear"]["r2"] - 0.3
    # free exponent is sub-quadratic (closer to 1 than 2)
    assert 0.4 < f["free_power"]["exponent"] < 1.3


def test_slope_proportional_respects_flat_limit():
    s = G6.load_sweep()
    f = G6.fit_local_flux(s["a_over_lam"], s["R_max"])
    # origin-respecting (R_max->1 at a/λ->0) law still fits well
    assert f["slope_proportional"]["r2"] > 0.85
    assert f["slope_proportional"]["coef_b"] > 0


def test_separation_onset_brackets():
    s = G6.load_sweep()
    onset = G6.separation_onset(s["a_over_lam"], s["R_min"])
    # R_min crosses zero between a/λ = 0.10 and 0.125
    assert 0.10 <= onset <= 0.125


def test_mean_nu_stays_flat():
    res = G6.run()
    # mean conductance suppression is amplitude-flat (the §A.2/§G.6 falsification)
    assert 0.85 < res["Nu_ratio_mean"] < 0.95
    assert res["Nu_ratio_std"] < 0.05


def test_run_headline():
    res = G6.run()
    assert res["quadratic_rejected"] is True
    lo, hi = res["R_max_range"]
    assert 1.8 < lo < 2.0
    assert 4.0 < hi < 4.5
    # R_mean saturates: late rise much smaller than early rise
    rmean = np.array(res["R_mean"])
    early = rmean[3] - rmean[0]
    late = rmean[-1] - rmean[3]
    assert late < early
