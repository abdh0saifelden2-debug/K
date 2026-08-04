"""Unit proofs for NR33 (persistence-pressure bound) -- synthetic/closed-form only."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from new_relationships10 import (  # noqa: E402
    GLEN_A, GLEN_N, RHO_I, G, YR, nmax_cylinder, nmax_sheet, phi_floor,
    verify_efold, verify_sheet_crossover, verify_archive_dividend,
    magnitude_table)


def test_nmax_cylinder_closed_form_is_the_efold_root():
    t = 4.0 * YR
    n = nmax_cylinder(t)
    assert abs(2.0 * GLEN_A * (n / GLEN_N) ** GLEN_N * t - 1.0) < 1e-12


def test_efold_ode_integration_matches():
    r = verify_efold()
    assert r["abs_err"] < 5e-5


def test_magnitudes_land_in_the_documented_bands():
    # ~3.5 bar at 4 yr (cylinder); ~0.1 bar sheet at h=0.1m, w=500m
    assert 3.0 < nmax_cylinder(4 * YR) / 1e5 < 4.0
    assert 0.05 < nmax_sheet(4 * YR) / 1e5 < 0.2
    # phi floor > 0.97 for >= 2 km ice at 4 yr
    assert phi_floor(2000.0, 4 * YR) > 0.97
    assert phi_floor(2000.0, 4 * YR, geometry="sheet") > 0.999


def test_sheet_ratio_analytic_and_ordering():
    rows = verify_sheet_crossover()
    assert all(r["match"] for r in rows)
    # every wide flat sheet (w >= h): strictly tighter than the cylinder
    wide = [r for r in rows if r["w_over_h"] >= 1.0]
    assert wide and all(r["nmax_ratio_sheet_over_cyl"] < 1.0 for r in wide)
    # coincidence at the (unphysical, tall-slot) aspect w/h = 2/n^n
    cross = [r for r in rows
             if abs(r["w_over_h"] - 2.0 / GLEN_N ** GLEN_N) < 1e-12]
    assert cross and abs(cross[0]["nmax_ratio_sheet_over_cyl"] - 1.0) < 1e-12


def test_archive_dividend_slope():
    r = verify_archive_dividend()
    assert r["abs_err"] < 1e-10


def test_phi_floor_monotone_in_H_and_t():
    H = np.array([1500.0, 2500.0, 3500.0])
    f4 = phi_floor(H, 4 * YR)
    f1 = phi_floor(H, 1 * YR)
    assert np.all(np.diff(f4) > 0)
    assert np.all(f4 > f1)


def test_cold_ice_systematic_direction():
    """Lower A (colder ice) must RAISE N_max as A^{-1/n} (documented systematic)."""
    t = 4.0 * YR
    n_temperate = nmax_cylinder(t)
    n_cold = nmax_cylinder(t, A=GLEN_A / 10.0)
    assert n_cold > n_temperate
    assert abs(n_cold / n_temperate - 10.0 ** (1.0 / GLEN_N)) < 1e-12


def test_magnitude_table_shape_and_floors():
    rows = magnitude_table()
    assert len(rows) == 12
    for r in rows:
        assert r["phi_floor_sheet"] > r["phi_floor_cyl"]     # sheet is tighter
        p_i = RHO_I * G * r["H_m"]
        assert abs(r["phi_floor_cyl"] -
                   (1 - r["Nmax_cyl_bar"] * 1e5 / p_i)) < 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
