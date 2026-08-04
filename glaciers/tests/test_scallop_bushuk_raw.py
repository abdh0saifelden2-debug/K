"""Tests for RESULT 25 -- the paper-3 field pin on the raw Bushuk h(x,t)
arrays (scallop_bushuk_raw.py).

Three layers:

1. machinery ground truth (no data): the flux-transfer decomposition
   ``s_flux = -m_k/h_k`` recovers a KNOWN complex rate from a synthetic
   damped, downstream-migrating train, and is exactly parity-symmetric
   (``Im = 0``) for a symmetric (non-migrating) profile;
2. committed derived statistics: every headline verdict re-verifies offline
   from ``subglacial/data/bushuk_raw_derived.json`` (downstream sign, damping,
   parity break, c_mig agreement, I values, controls);
3. raw reproduction (skipped unless the .mat is present): derive() from the
   raw file reproduces the committed derived statistics.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scallop_bushuk_raw as sbr


# --------------------------------------------------------------------------- #
# 1. machinery on synthetic ground truth
# --------------------------------------------------------------------------- #
def _synth_flux_series(g, w, nx=166, L=0.1486, dt=300.0, nfr=12):
    """h(x,t) = a e^{gt} cos(kx - wt): melt m = -dh/dt sampled like the data."""
    x = np.arange(nx) * L / nx
    k = 2.0 * np.pi / L

    def h(tt):
        return 6e-3 * np.exp(g * tt) * np.cos(k * x - w * tt)

    s_vals = []
    for i in range(1, nfr):
        t0, t1 = (i - 1) * dt, i * dt
        m = -(h(t1) - h(t0)) / dt
        hm = 0.5 * (h(t0) + h(t1))
        mF = np.fft.rfft(m - m.mean())
        hF = np.fft.rfft(hm - hm.mean())
        s_vals.append(-mF[1] / hF[1])
    return np.array(s_vals), k


def test_flux_transfer_recovers_known_mode():
    g, w = -1.0 / 36000.0, 2.0 * np.pi / 6000.0     # tau=10h, T=100 min
    s, k = _synth_flux_series(g, w)
    assert abs(s.real.mean() - g) <= 0.05 * abs(g)
    # downstream (+x) migration => Im(s) = -w
    assert abs(s.imag.mean() + w) <= 0.05 * w
    c = -s.imag.mean() / k
    assert c > 0.0                                   # downstream
    I = abs(s.imag.mean()) / (2.0 * np.pi * abs(s.real.mean()))
    I_true = w / (2.0 * np.pi * abs(g))
    assert abs(I - I_true) <= 0.06 * I_true


def test_flux_transfer_parity_null_for_symmetric_profile():
    """No migration (w=0) => the quadrature is machine zero: the K-theory
    parity statement the real record is tested against."""
    g, w = -1.0 / 36000.0, 0.0
    s, _ = _synth_flux_series(g, w)
    assert np.max(np.abs(s.imag)) < 1e-12 * np.max(np.abs(s.real))


def test_xcorr_shift_subgrid():
    x = np.linspace(0.0, 1.0, 200, endpoint=False)

    def prof(xx):                                   # multi-mode: no shift alias
        return (np.cos(2 * np.pi * xx) + 0.4 * np.sin(4 * np.pi * xx)
                + 0.2 * np.cos(6 * np.pi * xx + 0.7))

    a, b = prof(x - 0.013), prof(x)
    dx = float(x[1] - x[0])
    s = sbr._xcorr_shift(a, b, dx)
    assert abs(s - 0.013) < 5e-4


# --------------------------------------------------------------------------- #
# 2. committed derived statistics -> the headline verdicts, offline
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ds():
    return sbr.load_derived()


def test_meta_is_the_adjustment_sequence(ds):
    m = ds["meta"]
    assert m["n_frames"] == 12
    assert abs(m["dt_s"] - 300.0) < 1e-9
    assert abs(m["Lx_mm"] - 148.57) < 0.1
    assert m["timestring"][0] == "191702" and m["timestring"][-1] == "201202"


def test_dominant_mode_is_the_window_scale_not_the_digitized_28mm(ds):
    """The raw arrays rule out the superseded digitized crest-comb reading:
    corrugation power concentrates at the ~149 mm window mode; the ~30 mm
    (bin-5) power is <1% of it."""
    top = {int(k): p for k, p in ds["modal_power_top"]}
    assert max(top, key=top.get) == 1
    assert top.get(5, 0.0) < 0.01


def test_downstream_sign_unanimous(ds):
    v = sbr.verdict(ds)
    assert v["downstream_sign_unanimous"] is True


def test_damping_on_the_mean(ds):
    v = sbr.verdict(ds)
    assert v["damped_on_the_mean"] is True


def test_parity_break_significant(ds):
    fp = sbr.flux_parity(ds, n_boot=5000, seed=1)
    assert fp["bin1"]["p_Im_ge_0"] < 1e-3
    assert fp["bin1"]["Im_mean"] < 0.0            # downstream quadrature


def test_migration_speed_agreement(ds):
    cs = sbr.migration_speeds(ds)
    assert cs["all_downstream"] is True
    assert cs["spread_frac"] < 0.10               # four methods within 10%
    for k in ("bushuk_tracking", "xcorr", "modal_phase", "flux_quadrature"):
        assert 0.85 < cs[k] < 1.15                # mm/min


def test_flux_quadrature_accounts_for_migration(ds):
    """The dynamical closure: c from the melt-flux quadrature equals the
    kinematic trackers to <10% -- the flux phase lag IS the migration."""
    cs = sbr.migration_speeds(ds)
    kin = np.mean([cs["bushuk_tracking"], cs["xcorr"], cs["modal_phase"]])
    assert abs(cs["flux_quadrature"] - kin) / kin < 0.10


def test_pin_value_and_battery(ds):
    v = sbr.verdict(ds)
    assert 3.0 < v["I_pin"] < 4.3                 # pre-registered harness, verbatim
    lo, hi = v["I_battery"]
    assert 0.3 < lo < 1.0 and 5.0 < hi < 8.0      # honest estimator spread
    assert 0.8 < v["I_flux"] < 1.8


def test_battery_all_downstream_all_finite(ds):
    for tag, r in ds["variants"].items():
        assert bool(r["downstream"]) is True, tag
        assert np.isfinite(r["I"]), tag


def test_advected_frame_control(ds):
    ct = sbr.controls(ds)
    assert ct["advected_resid_frac"] < 0.01       # <1% of the lab-frame speed


def test_crest_evolution_angle_matches_published(ds):
    ct = sbr.controls(ds)
    assert ct["phi_agrees"] is True
    assert abs(ct["phi_deg"] - 15.0) < 1.5


def test_mean_melt_rate_magnitude(ds):
    assert 0.2 < ds["mean_melt_rate_mm_min"] < 0.3


def test_supersedes_digitized_reading(ds):
    """The raw-array I sits well below the digitized 10.3 and the battery is
    disjoint from the digitized battery's upper half."""
    v = sbr.verdict(ds)
    assert v["I_pin"] < 0.5 * v["digitized_superseded"]["I"]
    assert v["I_battery"][1] < v["digitized_superseded"]["battery"][1]


# --------------------------------------------------------------------------- #
# 3. raw .mat reproduction (only when the file is present)
# --------------------------------------------------------------------------- #
_MAT = os.environ.get("BUSHUK_MAT", sbr.MAT_DEFAULT)


@pytest.mark.skipif(not os.path.exists(_MAT), reason="raw Bushuk .mat not present")
def test_raw_reproduces_committed_derived(ds):
    raw = sbr.load_mat(_MAT)
    fresh = sbr.derive(raw)
    a = np.array(fresh["modal"]["1"]); b = np.array(ds["modal"]["1"])
    assert np.allclose(a, b, rtol=1e-10, atol=1e-12)
    fa = np.array(fresh["flux"]["1"]); fb = np.array(ds["flux"]["1"])
    assert np.allclose(fa, fb, rtol=1e-10, atol=1e-18)
    assert fresh["variants"]["verbatim"]["I"] == pytest.approx(
        ds["variants"]["verbatim"]["I"], rel=1e-12)
