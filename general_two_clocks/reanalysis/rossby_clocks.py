r"""The Rossby-number two clocks in real reanalysis winds.

The rotating-flow analog of the Mach->0 elliptic-pressure limit
(`general_two_clocks/REPORT_MACH_REGULARITY.md`, `REPORT_NS.md`).

Two-clocks dictionary (compressible  <->  rotating geophysical):

    sound speed  c           <->   Coriolis frequency  f = 2 Omega sin(phi)
    Mach number  M = U/c     <->   Rossby number  Ro = U/(f L)
    acoustic adjustment      <->   geostrophic (inertia-gravity) adjustment
    dilatational/acoustic u  <->   divergent/ageostrophic wind  (the FAST clock)
    solenoidal/vortical u    <->   rotational/balanced wind      (the SLOW clock)
    elliptic Poisson p       <->   elliptic balanced/QG geopotential (nonlocal)

Prediction.  Geostrophic balance `f k x u_g = -grad(Phi)` makes the leading
(rotational) wind balanced; the ageostrophic correction is
`u_a = -(1/f) k x Du_g/Dt ~ (U^2/L)/f = Ro U`, which carries the divergence.  Hence

    KE_div / KE_rot  ~  Ro^2  =  [U/(fL)]^2   ~   f^{-2}  ~  sin(phi)^{-2}

at roughly fixed eddy speed `U` and scale `L`.  So the FAST (divergent) clock's
energy fraction should (i) be largest in the tropics (`f -> 0`, `Ro >~ 1`, balance
fails) and (ii) fall off as `f^{-2}` (slope -2 in `log(ratio)` vs `log|sin phi|`)
through the extratropics -- the rotating mirror of `KE_dil/KE_sol ~ M^2`.

Validation verdict (real NCEP/NCAR Reanalysis, no credentials) is computed by
`run_rossby_clocks.py`; see `REPORT_ROSSBY_CLOCKS.md`.  The Helmholtz machinery
(spherical-harmonic rotational/divergent split) is reused from `reanalysis/ncep.py`.
"""
from __future__ import annotations

import numpy as np

from reanalysis import ncep

OMEGA = 7.292e-5            # Earth's rotation rate (rad/s)


def coriolis(lat_deg: np.ndarray) -> np.ndarray:
    """f = 2 Omega sin(phi)."""
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat_deg))


# --------------------------------------------------------------------------- #
# building winds from potentials (for deterministic tests)
# --------------------------------------------------------------------------- #
def wind_from_potentials(psi: np.ndarray, chi: np.ndarray, lat: np.ndarray):
    """Reconstruct (u, v) from a streamfunction psi (rotational) and velocity
    potential chi (divergent) on the lat/lon sphere, using the SAME spherical
    operators as ncep.helmholtz:
        rotational: u_psi = -(1/a) d psi/d phi,  v_psi = (1/(a cos phi)) d psi/d lambda
        divergent : u_chi = (1/(a cos phi)) d chi/d lambda,  v_chi = (1/a) d chi/d phi
    """
    a = ncep.A_EARTH
    phi = np.deg2rad(lat)
    cosp = np.cos(phi)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        u_psi = -np.gradient(psi, phi, axis=0) / a
        v_psi = ncep._dlon(psi) / (a * cosp)
        u_chi = ncep._dlon(chi) / (a * cosp)
        v_chi = np.gradient(chi, phi, axis=0) / a
    u = u_psi + u_chi
    v = v_psi + v_chi
    # scrub the singular pole rows (cos phi = 0) with the adjacent ring
    for arr in (u, v):
        arr[0] = arr[1]
        arr[-1] = arr[-2]
    return u, v


# --------------------------------------------------------------------------- #
# latitude-resolved rotational / divergent kinetic energy
# --------------------------------------------------------------------------- #
def ke_by_latitude(u: np.ndarray, v: np.ndarray, lat: np.ndarray):
    """Zonal-mean rotational and divergent KE as functions of latitude, via the
    spherical Helmholtz split.  Returns (lat_dh, ke_rot(phi), ke_div(phi))."""
    h = ncep.helmholtz(u, v, lat)
    ke_rot = 0.5 * np.mean(h["u_psi"] ** 2 + h["v_psi"] ** 2, axis=1)
    ke_div = 0.5 * np.mean(h["u_chi"] ** 2 + h["v_chi"] ** 2, axis=1)
    return h["lat"], ke_rot, ke_div


def block_profile(u: np.ndarray, v: np.ndarray, lat: np.ndarray):
    """Time-average ke_by_latitude over a (nt, nlat, nlon) block.  Returns
    (lat_dh, ke_rot(phi), ke_div(phi), ratio(phi) = ke_div/ke_rot)."""
    KEr = KEd = None
    for t in range(u.shape[0]):
        latd, ker, ked = ke_by_latitude(u[t], v[t], lat)
        KEr = ker if KEr is None else KEr + ker
        KEd = ked if KEd is None else KEd + ked
    KEr /= u.shape[0]
    KEd /= u.shape[0]
    ratio = KEd / np.maximum(KEr, 1e-30)
    return latd, KEr, KEd, ratio


# --------------------------------------------------------------------------- #
# the two predictions, tested
# --------------------------------------------------------------------------- #
def fit_f2_scaling(latd: np.ndarray, ratio: np.ndarray,
                   lat_min: float = 20.0, lat_max: float = 75.0):
    """Extratropical power-law slope of log(ratio) vs log|sin phi| (proportional to
    log|f|).  The model predicts slope ~ -2  (KE_div/KE_rot ~ Ro^2 ~ f^{-2})."""
    sphi = np.abs(np.sin(np.deg2rad(latd)))
    m = (np.abs(latd) >= lat_min) & (np.abs(latd) <= lat_max) & (ratio > 0) & (sphi > 0)
    slope, intercept = np.polyfit(np.log(sphi[m]), np.log(ratio[m]), 1)
    # correlation coefficient of the fit
    x, y = np.log(sphi[m]), np.log(ratio[m])
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(slope=float(slope), intercept=float(intercept), r=r, n=int(m.sum()))


def tropics_extratropics_contrast(latd: np.ndarray, ratio: np.ndarray,
                                  trop=15.0, ext=(30.0, 60.0)):
    """Mean divergent fraction in the tropics vs the extratropics, and the ratio."""
    mt = np.abs(latd) < trop
    me = (np.abs(latd) >= ext[0]) & (np.abs(latd) <= ext[1])
    t = float(np.mean(ratio[mt]))
    e = float(np.mean(ratio[me]))
    return dict(tropics=t, extratropics=e, contrast=float(t / max(e, 1e-30)))
