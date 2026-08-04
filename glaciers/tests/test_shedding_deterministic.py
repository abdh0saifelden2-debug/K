"""Tests for the §D.5 deterministic-limit shedding probe
(:mod:`scallop_shedding_deterministic`).

CPU-only.  No heavy DNS is run; the tests exercise:
- helper numerics (_detrend, _spectral_peak)
- the report-generation branch (write_report both writes a file and picks the
  correct verdict wording for the no-steady-base, shedding-found, and
  steady-no-shedding branches).
"""

import numpy as np

from scallop_shedding_deterministic import (
    _detrend,
    _spectral_peak,
    write_report,
)


def test_detrend_removes_linear_ramp():
    ramp = np.linspace(10.0, 20.0, 200)
    res = _detrend(ramp)
    assert abs(res.mean()) < 1e-10
    assert np.std(res) < 0.02


def test_spectral_peak_narrow_band():
    """A pure sine should give high spectral concentration."""
    t = np.arange(512)
    sig = np.sin(2 * np.pi * t / 32.0)
    conc, period = _spectral_peak(sig)
    assert conc > 0.5
    assert 28 < period < 36


def test_spectral_peak_broadband():
    """White noise should give low spectral concentration."""
    rng = np.random.default_rng(42)
    conc, _ = _spectral_peak(rng.standard_normal(512))
    assert conc < 0.1


def _make_results(base_steady, intrinsic):
    """Minimal results list with one amplitude and two f_amp points."""
    return [{"a_over_lam": 0.25, "base_steady": base_steady,
             "intrinsic": intrinsic, "ratio_rise": float("nan"),
             "points": [
                 {"ke_drift": 0.76, "std_detr_abs": 0.08, "cv_detr": 0.01,
                  "std_over_famp": 8.0, "spec_conc": 0.89,
                  "spec_period": 2000.0, "mean_ke": 7.5, "umax": 3.7,
                  "f_amp": 0.01},
                 {"ke_drift": 0.76, "std_detr_abs": 0.08, "cv_detr": 0.01,
                  "std_over_famp": 80.0, "spec_conc": 0.89,
                  "spec_period": 2000.0, "mean_ke": 7.5, "umax": 3.7,
                  "f_amp": 0.001}]}]


_CFG = {"nx": 64, "ny": 64, "n_waves": 12, "U_drive": 1.5,
        "spinup": 800, "measure": 600, "f_amps": [0.01, 0.001],
        "amps": [0.25], "backend": "numpy"}


def test_write_report_no_steady_base(tmp_path):
    """When no steady base flow exists the report says 'not testable'."""
    p = tmp_path / "rep.md"
    write_report(str(p), config=_CFG,
                 results=_make_results(False, False),
                 a_shed_found=False, a_shed=None, any_steady=False)
    txt = p.read_text()
    assert "not testable" in txt.lower()
    assert "AUTO-GENERATED" in txt


def test_write_report_shedding_found(tmp_path):
    """When shedding IS found the report says VERIFIED."""
    p = tmp_path / "rep.md"
    write_report(str(p), config=_CFG,
                 results=_make_results(True, True),
                 a_shed_found=True, a_shed=0.25, any_steady=True)
    txt = p.read_text()
    assert "[VERIFIED]" in txt
    assert "0.25" in txt


def test_write_report_steady_no_shedding(tmp_path):
    """Steady base, but no intrinsic shedding → NULL within reach."""
    p = tmp_path / "rep.md"
    write_report(str(p), config=_CFG,
                 results=_make_results(True, False),
                 a_shed_found=False, a_shed=None, any_steady=True)
    txt = p.read_text()
    assert "not earned" in txt.lower()
