"""Unit-proofs for NR65 (new_relationships42.py) — the phase face of the tidal EIS.

Planted-signal proofs run through the ACTUAL §I.5 harmonic machinery
(glaciers/validation/external/tidal_admittance_field.py): absolute-epoch phase
recovery across disjoint observing seasons, the diffusive-wave loss tangent
k_i/k_r = 1, unwrap and sign/wrap flagging. Real-cache tests replay the committed
harmonic cache.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.normpath(os.path.join(
    _HERE, "..", "..", "glaciers", "validation", "external")))

import new_relationships42 as R  # noqa: E402
import new_relationships41 as R41  # noqa: E402
import tidal_admittance_field as TAF  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_HAVE = os.path.exists(R._CACHE)
needs_cache = pytest.mark.skipif(not _HAVE, reason="committed §I.5 cache missing")

W_MSF_H = 2.0 * math.pi / TAF.CONSTITUENTS["MSf"]          # rad/hour
OMEGA_MSF = R.OMEGA["MSf"]                                 # rad/s


def _track(day0, n_days, phi0, amp_m, rng, gap_at=None, trend=0.0):
    """5-min cadence MSf track in days-since-1970 with optional 3-day gap."""
    t = day0 + np.arange(int(n_days * 288)) / 288.0
    if gap_at is not None:
        keep = (t < day0 + gap_at) | (t > day0 + gap_at + 3.0)
        t = t[keep]
    y = (amp_m * np.cos(W_MSF_H * (t * 24.0) + phi0)
         + trend * (t - t[0]) + 2e-3 * rng.standard_normal(t.size))
    return t, y


# --------------------------------------------------- absolute-epoch phase recovery
def test_harmonic_fit_recovers_absolute_phase_across_seasons():
    """Two 'stations' observing the SAME wave in different years must return the
    same absolute phase at the common 1970 epoch (t_ref=0)."""
    rng = np.random.default_rng(7)
    phi0 = 0.83
    for day0, ndays in ((16000.0, 60.0), (16811.0, 45.0)):
        t, y = _track(day0, ndays, phi0, 0.12, rng, gap_at=20.0, trend=0.004)
        segs = TAF._segments(t)
        assert len(segs) == 2                      # the 3-day gap splits the track
        fit, _ = TAF.harmonic_fit(t, y, segs, ["MSf"])
        A, ph, sig = fit["MSf"]
        assert abs(A - 0.12) < 0.005
        dphi = (ph - phi0 + math.pi) % (2.0 * math.pi) - math.pi
        assert abs(dphi) < 0.02, dphi


def test_planted_diffusive_wave_gives_unit_loss_tangent():
    """Stations on exp(i w t - (1+i) d/delta): k_r = k_i = 1/delta, so the
    pipeline must return loss tangent 1 and the same K on both faces."""
    rng = np.random.default_rng(3)
    delta_km = 30.0
    d_st = [4.0, 12.0, 25.0, 40.0, 55.0]
    amps, sigs, phs, snrs = [], [], [], []
    for i, d in enumerate(d_st):
        A0 = 0.15 * math.exp(-d / delta_km)
        phi = 0.4 - d / delta_km
        t, y = _track(16200.0 + 37.0 * i, 70.0, phi, A0, rng)   # staggered epochs
        fit, _ = TAF.harmonic_fit(t, y, TAF._segments(t), ["MSf"])
        A, ph, sig = fit["MSf"]
        amps.append(A); sigs.append(sig); phs.append(ph)
        snrs.append(A / sig)
    f = R.fit_phase_slope(d_st, phs, snrs)
    assert f and not f["two_point"] and not f["sign_violation"]
    k_r = -f["slope_rad_km"]
    assert abs(k_r - 1.0 / delta_km) / (1.0 / delta_km) < 0.10
    fa = R41.fit_delta(d_st, amps, sigs)
    k_i = 1.0 / fa["delta_km"]
    assert abs(k_i * delta_km - 1.0) < 0.10
    assert abs(k_i / k_r - 1.0) < 0.15
    dv = R._derived(k_r, f["err_rad_km"], OMEGA_MSF)
    K_true = OMEGA_MSF / (2.0 * (1.0 / (delta_km * 1e3)) ** 2)
    assert abs(dv["K_phase_m2_s"] - K_true) / K_true < 0.25
    assert abs(dv["c_m_s"] - OMEGA_MSF * delta_km * 1e3) / (
        OMEGA_MSF * delta_km * 1e3) < 0.15


# ------------------------------------------------------------------ fit machinery
def test_unwrap_along_recovers_wrapped_profile():
    d = np.array([5.0, 20.0, 40.0, 60.0, 80.0])
    k = 0.06                                       # total lag 4.8 rad -> wraps
    true = 2.0 - k * d
    wrapped = (true + math.pi) % (2.0 * math.pi) - math.pi
    un = R.unwrap_along(d, wrapped)
    assert np.allclose(np.diff(un[np.argsort(d)]),
                       np.diff(true[np.argsort(d)]), atol=1e-9)
    f = R.fit_phase_slope(d, wrapped, [50.0] * 5)
    assert abs(-f["slope_rad_km"] - k) < 1e-6


def test_sign_violation_flagged():
    d = [5.0, 20.0, 40.0]
    ph = [0.1, 0.5, 0.9]                           # phase ADVANCES upstream
    f = R.fit_phase_slope(d, ph, [30.0] * 3)
    assert f["sign_violation"]


def test_two_point_wrap_risk_flagged():
    f = R.fit_phase_slope([10.0, 60.0], [1.4, -0.6], [30.0, 30.0])
    assert f["two_point"] and f["wrap_risk"]
    f2 = R.fit_phase_slope([10.0, 60.0], [1.4, 0.9], [30.0, 30.0])
    assert f2["two_point"] and not f2["wrap_risk"]


def test_low_snr_stations_excluded():
    f = R.fit_phase_slope([5.0, 20.0, 40.0, 60.0],
                          [1.0, 0.7, 0.3, 2.9], [40.0, 40.0, 40.0, 1.0])
    assert f["n"] == 3 and not f["sign_violation"]


# ------------------------------------------------------------------ real cache
@needs_cache
def test_foundation_phase_profile_and_loss_tangent():
    res = R.analyze()
    e = res["branches"]["Foundation"]
    f = e["phase_fit"]
    assert f["n"] == 5 and not f["two_point"] and not f["sign_violation"]
    dv = e["derived"]
    assert 25.0 < dv["delta_phase_km"] < 90.0
    assert 10.0 < dv["c_km_day"] < 40.0
    lt = e["loss_tangent"]
    assert 0.4 < lt["k_i_over_k_r"] < 2.0          # parabolic prediction: 1
    assert 0.2 < dv["K_phase_m2_s"] / e["K_amp_m2_s"] < 5.0
    o = e["ordering"]
    assert o["tau"] < 0 and o["p_one_sided"] <= 0.01   # 5/5 strictly monotone


@needs_cache
def test_evans_phase_profile_present():
    res = R.analyze()
    e = res["branches"]["Evans"]
    assert e["phase_fit"]["n"] >= 4
    assert not e["phase_fit"]["sign_violation"]
    assert e["derived"]["c_m_s"] > 0


@needs_cache
def test_rutford_two_point_wrap_flagged_with_alt_branch():
    res = R.analyze()
    e = res["branches"]["Rutford"]
    f = e["phase_fit"]
    assert f["two_point"] and f["wrap_risk"]
    assert "alt_branch" in e


@needs_cache
def test_ocean_epoch_control_tight():
    res = R.analyze()
    oc = res["ocean_epoch_control"]
    assert len(oc["pairs"]) >= 2
    assert oc["max_dphi_rad"] < 0.15


@needs_cache
def test_verdict_mentions_foundation_and_control():
    res = R.analyze()
    assert "Foundation" in res["verdict"]
    assert "ocean epoch control" in res["verdict"]
