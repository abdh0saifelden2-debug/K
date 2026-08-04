r"""Deterministic tests for the Rossby-number two-clocks analysis (rossby_clocks.py).

No network: the Helmholtz pipeline and the scaling fits are tested on planted fields
with known rotational/divergent content. (The real-NCEP verdict lives in
run_rossby_clocks.py / REPORT_ROSSBY_CLOCKS.md.)
"""
from __future__ import annotations

import numpy as np

from reanalysis import rossby_clocks as rc


def _ncep_grid():
    lat = np.linspace(90.0, -90.0, 73)
    lon = np.linspace(0.0, 357.5, 144)
    phi, lam = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    return lat, lon, phi, lam


def test_coriolis_sign_and_zero():
    f = rc.coriolis(np.array([-90.0, 0.0, 90.0]))
    assert abs(f[1]) < 1e-12                       # zero at the equator
    assert f[2] > 0 and f[0] < 0                   # sign flips across the equator
    assert abs(f[2] - 2 * rc.OMEGA) < 1e-12        # 2*Omega at the pole


def test_helmholtz_separates_rotational_and_divergent():
    """A pure-streamfunction wind has ~zero divergent KE; a pure-velocity-potential
    wind has ~zero rotational KE — the Helmholtz split is exact to round-off."""
    lat, lon, phi, lam = _ncep_grid()
    pattern = np.sin(2 * lam) * np.cos(phi) ** 2
    z = np.zeros_like(pattern)

    u, v = rc.wind_from_potentials(pattern, z, lat)      # pure rotational
    _, ker, ked = rc.ke_by_latitude(u, v, lat)
    assert ked.sum() / ker.sum() < 1e-6

    u, v = rc.wind_from_potentials(z, pattern, lat)      # pure divergent
    _, ker, ked = rc.ke_by_latitude(u, v, lat)
    assert ker.sum() / ked.sum() < 1e-6


def test_block_profile_recovers_planted_latitude_structure():
    """A divergent field modulated to be strong in the tropics yields a divergent
    fraction that peaks in the tropics and is small in the extratropics."""
    lat, lon, phi, lam = _ncep_grid()
    rot = np.sin(2 * lam) * np.cos(phi) ** 2                       # broad rotational
    trop_mask = np.exp(-(np.rad2deg(phi) / 12.0) ** 2)            # tropics-localized
    div = 0.3 * trop_mask * np.sin(3 * lam)                       # divergent, tropical
    u, v = rc.wind_from_potentials(rot, div, lat)
    block_u = u[None, :, :]
    block_v = v[None, :, :]
    latd, ker, ked, ratio = rc.block_profile(block_u, block_v, lat)
    con = rc.tropics_extratropics_contrast(latd, ratio)
    assert con["tropics"] > con["extratropics"]                   # peaks in the tropics
    assert con["contrast"] > 3.0


def test_fit_f2_recovers_minus_two():
    """The extratropical fit recovers the predicted Ro^2 ~ f^-2 (slope -2) exactly on
    a synthetic ratio = |sin phi|^-2."""
    latd = np.linspace(-80.0, 80.0, 100)
    sphi = np.abs(np.sin(np.deg2rad(latd)))
    ratio = np.where(sphi > 0, sphi ** -2.0, 1.0)
    fit = rc.fit_f2_scaling(latd, ratio)
    assert abs(fit["slope"] + 2.0) < 1e-6
    assert fit["r"] < -0.99                                       # strong anti-correlation
