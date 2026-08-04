"""Unit tests for the real-data lake-lag §I extension
(validation/external/lake_lag_sn_ews.py)."""
import os
import sys

import numpy as np

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
if _VALIDATION not in sys.path:
    sys.path.insert(0, _VALIDATION)

from external import lake_lag_sn_ews as LL  # noqa: E402

RES = LL.run()


def test_matched_data_loads():
    d = LL.load_matched()
    assert "lakes" in d and len(d["lakes"]) == 5
    assert d["_obs_lag_band_yr"] == [0.02, 2.0]


def test_event_response_detects_planted_surge():
    # planted clear +10% sustained speed-up after the trough -> surge_detection True
    t = np.array([2013., 2014., 2015., 2016., 2017., 2018.])
    v = np.array([400., 401., 399., 440., 441., 439.])   # jump after 2015.5
    ev = dict(drop_m=5.0, t_peak=2015.0, t_trough=2015.5)
    r = LL.event_response(t, v, 0.01, ev, [0.02, 2.0])
    assert r["resolved"] and r["surge_detection"] is True
    # a flat series -> no detection
    vflat = np.full_like(t, 400.0)
    assert LL.event_response(t, vflat, 0.01, ev, [0.02, 2.0])["surge_detection"] is False


def test_ews_detector_fires_on_planted_csd():
    # rising variance + high lag-1 AC -> joint CSD True (detector is not dead)
    t = np.arange(12.0)
    rng = np.random.default_rng(0)
    early = 0.1 * rng.standard_normal(6)
    late = np.cumsum(2.0 * rng.standard_normal(6))     # growing, autocorrelated
    v = 400.0 + np.concatenate([early, late])
    m = LL.ews_metrics(t, v)
    assert m["joint_csd_signature"] is True
    # a stable series -> no CSD
    stable = 400.0 + 0.2 * rng.standard_normal(12)
    assert LL.ews_metrics(t, stable)["joint_csd_signature"] is False


def test_real_data_no_universal_surge_no_precursor():
    s = RES["summary"]
    assert s["n_velocity_lakes"] == 3
    assert s["n_surge_detections"] == 0          # no clean positive surge
    assert s["max_abs_dv_over_v"] <= 0.05        # all responses <= ~5%
    assert s["n_lakes_with_joint_csd"] == 0      # EWS true-negative on stable lakes
    assert s["any_universal_surge"] is False
    assert s["any_csd_precursor"] is False


def test_scope_is_honest():
    # must disclose the gated full population and not fake an ATL15 download
    assert "gated" in RES["honest_limits"].lower()
    assert "ATL15" in RES["honest_limits"]
    assert RES["summary"]["n_velocity_lakes"] < 10   # marquee subset, not the 131 catalog
