"""Unit proofs for NR35 (`general_two_clocks/new_relationships12.py`):
the two-sided persistence bracket -- vanish ceiling, melt-corrected floor,
threshold-gauge algebra, and the classifier/anchor machinery (synthetic; the
data application is exercised only if the local datasets are present).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships12 import (  # noqa: E402
    DT_CEIL_YR, DT_FLOOR_YR, GLEN_A, GLEN_N, YR, n_floor_melt_corrected,
    n_threshold, phi_bound, survival_efolds)


# ------------------------------------------------------------ threshold ----
def test_threshold_is_the_efold_root():
    """N*(Dt) makes the closure integral exactly one e-fold: c(N*) Dt = 1."""
    for dt_yr in (1.0, 3.0, 4.0, 10.0):
        dt = dt_yr * YR
        n_star = n_threshold(dt)
        assert np.isclose(2.0 * GLEN_A * (n_star / GLEN_N) ** GLEN_N * dt,
                          1.0, rtol=1e-12)


def test_two_sides_share_the_same_function():
    """Persist floor (at mbar=0) and vanish ceiling are the SAME constant at
    the same Dt -- one repeat pair is a binary classifier; the conservative
    spans (3 vs 4 yr) create the honest deadband [N*_ceil, N*_floor]."""
    dt = 4.0 * YR
    assert np.isclose(n_floor_melt_corrected(dt, 0.0), n_threshold(dt),
                      rtol=1e-12)
    assert n_threshold(DT_CEIL_YR * YR) < n_threshold(DT_FLOOR_YR * YR)
    # archive dividend, two-sided: N* ~ Dt^{-1/3}
    assert np.isclose(n_threshold(8 * YR) / n_threshold(1 * YR),
                      8.0 ** (-1.0 / 3.0), rtol=1e-12)


def test_vanish_ceiling_is_melt_robust():
    """Vanish means int (c - m) dt >= 1 with m >= 0, so c Dt >= 1 + int m dt
    >= 1: any opening only STRENGTHENS the inferred N >= N*.  Check by
    explicit survival integral: at N = N*(Dt), adding melt makes the cavity
    survive (net e-folds < 1) -- so a cell that nevertheless vanished must
    have N strictly above N* when m > 0."""
    dt = 4.0 * YR
    n_star = n_threshold(dt)
    assert np.isclose(survival_efolds(n_star, dt), 1.0, rtol=1e-12)
    for m_per_yr in (0.1, 0.5, 2.0):
        assert survival_efolds(n_star, dt, m_per_yr / YR) < 1.0
        # the N that reaches one net e-fold under melt is strictly larger
        n_req = GLEN_N * ((1.0 / dt + m_per_yr / YR) / (2 * GLEN_A)) ** (1 / 3)
        assert n_req > n_star
        assert np.isclose(survival_efolds(n_req, dt, m_per_yr / YR), 1.0,
                          rtol=1e-12)


def test_melt_corrected_floor_monotone_and_reduces():
    """The corrected floor rises monotonically with measured mbar and
    reduces exactly to NR33's floor at mbar = 0."""
    dt = 4.2 * YR
    prev = n_floor_melt_corrected(dt, 0.0)
    assert np.isclose(prev, n_threshold(dt), rtol=1e-12)
    for m in (0.05, 0.15, 0.8, 3.0):
        cur = n_floor_melt_corrected(dt, m / YR)
        assert cur > prev
        prev = cur
    # closed-form check of the exponent-1/3 structure
    assert np.isclose(
        n_floor_melt_corrected(dt, 7.0 / dt) / n_threshold(dt),
        8.0 ** (1.0 / 3.0), rtol=1e-12)


def test_phi_bounds_directions():
    """Floor N <= X -> phi >= 1 - X/p_i; ceiling N >= Y -> phi <= 1 - Y/p_i;
    both between 0 and 1 for glacial overburden."""
    H = 3886.0
    n_star = n_threshold(4.0 * YR)
    lo = phi_bound(H, n_threshold(3.0 * YR))
    hi = phi_bound(H, n_star)
    assert 0.97 < lo < hi < 1.0


# ------------------------------------------------------------- classifier --
def _classify(sA, sB, thr_on=0.2, thr_off=0.1):
    persist = (sA >= thr_on) & (sB >= thr_on)
    vanish = (sA >= thr_on) & (sB < thr_off)
    appear = (sA < thr_off) & (sB >= thr_on)
    dark = (sA < thr_off) & (sB < thr_off)
    return persist, vanish, appear, dark


def test_deadband_classifier_logic():
    """Deadband kills threshold flicker: a cell oscillating inside
    [thr_off, thr_on) is classified neither persist nor vanish."""
    sA = np.array([0.5, 0.5, 0.05, 0.05, 0.15])
    sB = np.array([0.5, 0.05, 0.5, 0.05, 0.12])
    p, v, a, d = _classify(sA, sB)
    assert list(p) == [True, False, False, False, False]
    assert list(v) == [False, True, False, False, False]
    assert list(a) == [False, False, True, False, False]
    assert list(d) == [False, False, False, True, False]
    # the flicker cell is in no class (the deadband's whole point)
    assert not (p | v | a | d)[4]


def test_classes_are_disjoint_random():
    rng = np.random.default_rng(8)
    sA, sB = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500)
    p, v, a, d = _classify(sA, sB)
    stack = np.stack([p, v, a, d]).astype(int).sum(axis=0)
    assert stack.max() <= 1


# ---------------------------------------------------------- data (optional) --
_HAVE_DATA = (os.path.isdir("/home/K/_data_specularity")
              and os.path.isdir("/home/K/_data_bedmap2/bedmap2_bin")
              and os.path.isdir(os.path.expanduser("~/data_usapdc/601439")))


@pytest.mark.skipif(not _HAVE_DATA, reason="local ICECAP/601439 data absent")
def test_geotiff_centroids_cross_validate_601470():
    """PIL-parsed 601439 z-grid centroids: CRS pinned to EPSG:3031 and the
    median nearest-neighbour distance to the independent 601470 outline
    catalogue is sub-cell (< 2 km)."""
    from new_relationships12 import (lake_centroids_from_geotiffs,
                                     LAKE_STATS_601470)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "glaciers", "validation"))
    from external.rtn_spec_sign_pin import load_lake_stats
    cents = lake_centroids_from_geotiffs()
    assert len(cents) > 100
    cat = load_lake_stats(os.path.expanduser(LAKE_STATS_601470))
    cx, cy = np.asarray(cat["x"]), np.asarray(cat["y"])
    dmin = [np.min(np.hypot(cx - x, cy - y)) / 1e3 for _, x, y in cents]
    assert np.median(dmin) < 2.0


@pytest.mark.skipif(not _HAVE_DATA, reason="local ICECAP/601439 data absent")
def test_totten2_dated_anchor_arithmetic():
    """The one covered dated lake: last trough 2007.8, Dt_dated ~ 4.2 yr,
    dated floor 3.49 bar, melt-corrected bracket above it and below 6 bar."""
    from new_relationships12 import DIR_601439, EPOCH_B_YR
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "glaciers", "validation"))
    from external.run_usapdc_lakes import parse_volume_history, detect_drainages
    t, v = parse_volume_history(os.path.join(
        DIR_601439, "Totten_2", "Volume_history.csv"))
    ev = detect_drainages(t, v)
    assert len(ev) == 2
    t_d = ev[-1]["t_trough"]
    assert 2007.5 < t_d < 2008.0
    dt = (EPOCH_B_YR - t_d) * YR
    n_fl = n_threshold(dt)
    assert 3.3 < n_fl / 1e5 < 3.7
    # melt correction bracket from the two normalisations
    after = t >= t_d
    dv_dt = max(float(np.polyfit(t[after], v[after], 1)[0]), 0.0)
    m_lo = dv_dt / (v.max() - v.min())
    m_hi = dv_dt / (v[-1] - v.min())
    assert 0 < m_lo < m_hi
    lo = n_floor_melt_corrected(dt, m_lo / YR) / 1e5
    hi = n_floor_melt_corrected(dt, m_hi / YR) / 1e5
    assert n_fl / 1e5 < lo < hi < 6.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
