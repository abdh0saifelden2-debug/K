"""Tests for the CATS2008 tide loader and the grounding-zone tidal-forcing probe.

The pure helpers + the missing-data error path run everywhere.  The real
prediction/probe tests are skipped unless ``pyTMD`` and the (large, gated) CATS2008
model + BedMachine NetCDF are present locally (env overrides
``K_CATS2008_DIR`` / ``K_BEDMACHINE``), mirroring the data-gated bedmachine tests.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from external import DataUnavailableError  # noqa: E402
from external.tide_loader import (  # noqa: E402
    ocean_pressure_from_draft, _times_to_delta_seconds, load_tide, RHO_W, G,
)

_HAS_PYTMD = importlib.util.find_spec("pyTMD") is not None
_CATS_DIR = os.environ.get("K_CATS2008_DIR", "/home/data_tides")
_HAS_CATS = os.path.isdir(os.path.join(_CATS_DIR, "CATS2008"))
_BEDMACHINE = os.environ.get(
    "K_BEDMACHINE",
    "/home/data_bedmachine/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")
_HAS_BM = os.path.exists(_BEDMACHINE)


def test_ocean_pressure_from_draft_matches_hydrostatic():
    """p_ocean = rho_w g (draft + eta), elementwise."""
    p = ocean_pressure_from_draft(500.0, eta_m=1.5)
    assert abs(p - RHO_W * G * 501.5) < 1e-6
    arr = ocean_pressure_from_draft(np.array([100.0, 200.0]), eta_m=np.array([0.0, 1.0]))
    np.testing.assert_allclose(arr, RHO_W * G * np.array([100.0, 201.0]))


def test_times_to_delta_seconds_datetime_and_numeric():
    import datetime
    ep = (1992, 1, 1, 0, 0, 0)
    t = [datetime.datetime(1992, 1, 1, 0, 0, 0), datetime.datetime(1992, 1, 1, 1, 0, 0)]
    d = _times_to_delta_seconds(t, ep)
    np.testing.assert_allclose(d, [0.0, 3600.0])
    # already-numeric seconds pass through unchanged
    np.testing.assert_allclose(_times_to_delta_seconds([0.0, 7200.0], ep), [0.0, 7200.0])


def test_load_tide_missing_dir_raises():
    """No model dir -> DataUnavailableError with a provisioning hint (no pyTMD import)."""
    with pytest.raises(DataUnavailableError):
        load_tide("/no/such/CATS2008/dir", x=[0.0], y=[-75.0], times=[0.0])


@pytest.mark.skipif(not (_HAS_PYTMD and _HAS_CATS),
                    reason="needs pyTMD + local CATS2008 model")
def test_load_tide_predicts_sane_antarctic_tide():
    """A real CATS2008 prediction at a Filchner-Ronne cavity point is meter-scale."""
    import datetime
    times = [datetime.datetime(2020, 1, 1) + datetime.timedelta(hours=i)
             for i in range(24 * 5)]
    eta = load_tide(_CATS_DIR, x=np.array([-50.0]), y=np.array([-78.0]), times=times)
    assert eta.shape == (1, len(times))
    rng = float(np.nanmax(eta) - np.nanmin(eta))
    assert 0.5 < rng < 4.0, f"FRIS tidal range {rng:.2f} m outside expected band"


@pytest.mark.skipif(not (_HAS_PYTMD and _HAS_CATS and _HAS_BM),
                    reason="needs pyTMD + CATS2008 + BedMachine NetCDF")
def test_grounding_zone_probe_finite():
    """The CATS2008 x BedMachine GZ probe returns finite, physical medians."""
    from external.tidal_forcing_gz import analyse
    summary, arrays = analyse(_BEDMACHINE, _CATS_DIR, stride=16, gl_dist_km=10.0,
                              n_sample=40, days=3)
    assert summary["n_valid"] > 0
    assert 0.1 < summary["eta_amp_m"]["p50"] < 4.0       # meter-scale tides
    assert summary["ratio_dp_over_pi"]["p50"] > 0.0      # positive forcing floor
