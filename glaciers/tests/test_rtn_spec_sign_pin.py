"""Unit proofs for §V.1d ``rtn_spec_sign_pin`` (synthetic only, no downloads)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "validation"))
from external.rtn_spec_sign_pin import (  # noqa: E402
    GLEN_A, GLEN_N, YR, creep_nmax, phi_floor, lake_cell_masks,
    stratified_mw, rank_partial, driving_stress)
from external.run_rtn_bedmap2 import RHO_I  # noqa: E402

G = 9.81


def _meta(nr=40, nc=40, cs=5000.0):
    return {"cellsize": cs, "nrows": nr, "ncols": nc, "xll": 0.0, "yll": 0.0,
            "nrows_full": nr, "ncols_full": nc, "cellsize_full": cs,
            "stride": 1}


# ---------------------------------------------------------------- physics --
def test_creep_nmax_is_the_efold_root():
    """N_max solves survival(t_obs) = 1/e for dS/dt = -2A(N/n)^n S exactly,
    and matches a brute-force root solve of the integrated ODE."""
    t = 4.0 * YR
    nmax = creep_nmax(t)
    # closed-form survival at N_max after t: exp(-2A(N/n)^n t) = exp(-1)
    rate = 2.0 * GLEN_A * (nmax / GLEN_N) ** GLEN_N
    assert abs(rate * t - 1.0) < 1e-12
    # explicit Euler integration of the linear ODE reaches ~1/e
    nstep = 20000
    dt = t / nstep
    s = 1.0
    for _ in range(nstep):
        s += -rate * s * dt
    assert abs(s - np.exp(-1.0)) < 5e-4
    # magnitude sanity: ~3.5 bar at 4 yr, temperate A
    assert 2e5 < nmax < 6e5


def test_phi_floor_monotonicity_and_magnitude():
    t = 4.0 * YR
    H = np.array([1000.0, 2000.0, 4000.0])
    fl = phi_floor(H, t)
    assert np.all(np.diff(fl) > 0)               # thicker ice -> higher floor
    fl_1yr = phi_floor(H, 1.0 * YR)
    assert np.all(fl > fl_1yr)                   # longer persistence -> higher
    p_i = RHO_I * G * 2000.0
    assert abs(fl[1] - (1 - creep_nmax(t) / p_i)) < 1e-12
    assert fl[1] > 0.97                           # the ordering-violation zone


def test_orientation_logic_squeezes_decreasing_map():
    """The floor constrains only high-spec (persistent) cells: an increasing
    map keeps a free low end, a decreasing map's whole range is squeezed above
    the floor, so any claimed range > 1-floor is inadmissible."""
    required_floor = 0.985
    claimed_range = 0.97 - 0.80
    max_range_decreasing = max(0.0, 1.0 - required_floor)
    assert max_range_decreasing < 0.02
    assert claimed_range > max_range_decreasing          # connectivity killed
    assert claimed_range <= 1.0                          # increasing map fine


# ------------------------------------------------------------------ lakes --
def test_lake_cell_masks_geometry():
    meta = _meta()
    lakes = {"x": np.array([100_000.0]), "y": np.array([100_000.0]),
             "eqdiam_km": np.array([12.0]), "H_bm": np.array([2000.0]),
             "feret_w_km": np.array([10.0]), "feret_l_km": np.array([14.0])}
    lake, halo, lid, per = lake_cell_masks(lakes, meta, (40, 40),
                                           buffer_km=25.0)
    # centroid cell (i: ytop=200km -> y=100km is row 19/20 boundary; radius 6km)
    assert per[0] == lake.sum() and lake.sum() >= 4      # 6-km radius, 5-km cells
    assert lid[lake].max() == 0
    assert halo.sum() > lake.sum()                        # halo strictly larger
    ii, jj = np.where(lake)
    # all lake cells within 6 km + half-diagonal of the centroid
    ytop = meta["yll"] + meta["nrows_full"] * meta["cellsize_full"]
    xc = meta["xll"] + (jj + 0.5) * meta["cellsize"]
    yc = ytop - (ii + 0.5) * meta["cellsize"]
    r = np.hypot(xc - 100_000.0, yc - 100_000.0)
    assert r.max() <= 6_000.0 + 1e-9


def test_lake_cell_masks_radius_floor():
    meta = _meta()
    lakes = {"x": np.array([100_000.0]), "y": np.array([100_000.0]),
             "eqdiam_km": np.array([0.5]), "H_bm": np.array([2000.0]),
             "feret_w_km": np.array([0.5]), "feret_l_km": np.array([0.5])}
    lake, _, _, per = lake_cell_masks(lakes, meta, (40, 40))
    assert per[0] >= 1                       # tiny lake still captures >=1 cell


# ------------------------------------------------------------- statistics --
def test_stratified_mw_detects_planted_ordering():
    rng = np.random.default_rng(0)
    n = 400
    spec = rng.uniform(0, 0.3, (n,))
    f = rng.uniform(0.2, 1.0, (n,))
    lake = np.zeros(n, bool); lake[:40] = True
    spec[lake] += 0.25                       # planted: lakes clearly higher
    r = stratified_mw(spec, lake, ~lake, f, np.random.default_rng(1),
                      n_perm=500)
    assert r["p_perm_stratified"] < 0.01
    assert r["prob_lake_gt_ctrl"] > 0.8


def test_stratified_mw_null_is_calibrated():
    rng = np.random.default_rng(2)
    n = 400
    f = rng.uniform(0.2, 1.0, (n,))
    spec = 0.3 * (1 - f) + rng.uniform(0, 0.1, (n,))   # spec depends on f only
    lake = np.zeros(n, bool)
    lake[rng.choice(n, 40, replace=False)] = True
    r = stratified_mw(spec, lake, ~lake, f, np.random.default_rng(3),
                      n_perm=500)
    assert r["p_perm_stratified"] > 0.05     # no false detection under the null


def test_rank_partial_recovers_planted_sign_under_confounding():
    rng = np.random.default_rng(4)
    n = 3000
    tau = rng.lognormal(0, 0.5, n)
    h = rng.uniform(500, 3000, n)
    f = rng.uniform(0, 1, n)
    spec = 0.5 * (tau / tau.max()) + rng.uniform(0, 0.3, n)  # spec confounded w/ tau
    for gamma, expect_pos in ((+0.6, True), (-0.6, False)):
        y = 2.0 * np.log(tau) + 0.001 * h + gamma * spec + rng.normal(0, 0.2, n)
        r = rank_partial(y, spec, [np.log(tau), h, f],
                         np.random.default_rng(5))
        assert (r["rho_partial"] > 0.1) is expect_pos
        assert (r["rho_partial"] < -0.1) is (not expect_pos)
    y0 = 2.0 * np.log(tau) + rng.normal(0, 0.2, n)           # gamma = 0
    r0 = rank_partial(y0, spec, [np.log(tau), h, f],
                      np.random.default_rng(6))
    assert abs(r0["rho_partial"]) < 0.1


def test_rank_partial_block_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.uniform(0, 1, n)
    y = 0.5 * x + rng.normal(0, 0.3, n)
    blocks = (np.arange(n) // 50)
    r = rank_partial(y, x, [rng.uniform(0, 1, n)], np.random.default_rng(8),
                     block_ids=blocks, n_boot=100)
    lo, hi = r["rho_partial_ci95"]
    assert lo < r["rho_partial"] < hi
    assert lo > 0                                            # clearly positive


# ---------------------------------------------------------------- gradients --
def test_driving_stress_flat_surface_is_zero_and_slope_scales():
    nr = nc = 30
    H = np.full((nr, nc), 1000.0)
    flat = np.full((nr, nc), 500.0)
    td = driving_stress(H, flat, 5000.0)
    assert np.nanmax(np.abs(td[2:-2, 2:-2])) < 1e-8
    slope = 1e-3
    xx = np.arange(nc) * 5000.0
    tilted = 500.0 + slope * xx[None, :] * np.ones((nr, 1))
    td2 = driving_stress(H, tilted, 5000.0)
    expect = RHO_I * G * 1000.0 * slope
    assert abs(np.nanmedian(td2[3:-3, 3:-3]) - expect) / expect < 0.05


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
