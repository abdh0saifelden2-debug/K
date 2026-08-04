"""Rotational/divergent (Helmholtz) split of real reanalysis winds on the sphere.
 
The horizontal wind splits into a *rotational* (non-divergent, balanced) part and
a much weaker *divergent* part.  The rotational part is the slow, geostrophic
"weather" clock; the divergent part is the fast clock -- ageostrophic adjustment,
gravity-wave/tidal and convective overturning -- the same role the dilatational
(acoustic) velocity plays in the compressible runs of Parts 5-6.
 
Horizontal wind divergence is *not* a model artifact: by 3-D mass continuity
du/dx + dv/dy = -dw/dz it is the genuine footprint of vertical motion.  The honest
diagnostic is therefore the *Helmholtz split*, not the raw divergence.
 
Data: NCEP/NCAR Reanalysis 1 (Kalnay et al. 1996), served freely by NOAA PSL over
OPeNDAP -- no credentials.  The 2.5 deg grid (73 lat x 144 lon) minus its south
pole row is exactly a Driscoll-Healy grid (n=72, sampling=2), so the
spherical-harmonic transforms used for the Laplacian inversions need no regridding.
"""
 
from __future__ import annotations
 
import os
 
import numpy as np
import pyshtools as pysh
 
A_EARTH = 6.371e6  # m
 
_DODS = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis"
SOURCES = {
    "daily": f"{_DODS}/Dailies/pressure/{{var}}.{{year}}.nc",  # daily means
    "inst": f"{_DODS}/pressure/{{var}}.{{year}}.nc",            # 6-hourly snapshots
}
 
 
def fetch_wind(level: int, year: int = 2021, t0: int = 100, nt: int = 24,
               source: str = "daily", cache_dir: str = "data_reanalysis"):
    """Download (and cache) a (nt, nlat, nlon) block of u and v at one level.
 
    Returns (u, v, lat, lon) with lat running 90..-90 and lon 0..357.5.
    """
    if source not in SOURCES:
        raise ValueError(
            f"Unknown source {source!r}; expected one of {list(SOURCES.keys())}"
        )
    key = f"{source}_{year}_{level}_{t0}_{nt}"
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"wind_{key}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["u"], d["v"], d["lat"], d["lon"]
 
    import netCDF4
    url = SOURCES[source]
    out = {}
    lat = lon = None
    for var in ("uwnd", "vwnd"):
        remote = url.format(var=var, year=year)
        try:
            ds = netCDF4.Dataset(remote)
        except OSError as exc:
            raise OSError(
                f"Failed to open remote dataset {remote!r}: {exc}"
            ) from exc
        try:
            lev = list(ds.variables["level"][:])
            if level not in lev:
                raise ValueError(
                    f"Pressure level {level} hPa not in dataset "
                    f"(available: {lev})"
                )
            if lat is None:
                lat = np.ma.filled(np.ma.asarray(ds.variables["lat"][:]).astype(float), np.nan)
                lon = np.ma.filled(np.ma.asarray(ds.variables["lon"][:]).astype(float), np.nan)
            raw = ds.variables[var][t0:t0 + nt, lev.index(level), :, :]
            out[var] = np.ma.filled(np.ma.asarray(raw).astype(float), np.nan)
        finally:
            ds.close()
    u, v = out["uwnd"], out["vwnd"]
    np.savez(path, u=u, v=v, lat=lat, lon=lon)
    return u, v, lat, lon
 
 
def _dlon(f: np.ndarray) -> np.ndarray:
    """Exact longitude derivative (periodic) via FFT, d/d(lambda) in radians."""
    nlon = f.shape[-1]
    k = np.fft.fftfreq(nlon, d=1.0 / nlon)
    return np.real(np.fft.ifft(1j * k * np.fft.fft(f, axis=-1), axis=-1))
 
 
def vorticity_divergence(u: np.ndarray, v: np.ndarray, lat: np.ndarray):
    """Relative vorticity and horizontal divergence on the lat/lon sphere.
 
    Longitude derivatives are spectral (periodic); latitude derivatives are
    centred finite differences.  The two singular pole rows (cos(lat)=0) are
    replaced by the zonal mean of the adjacent ring; they carry ~zero area weight.
    """
    nlat = u.shape[0]
    phi = np.deg2rad(lat)
    cosp = np.cos(phi)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        div = (_dlon(u) + np.gradient(v * cosp, phi, axis=0)) / (A_EARTH * cosp)
        vor = (_dlon(v) - np.gradient(u * cosp, phi, axis=0)) / (A_EARTH * cosp)
    for arr in (div, vor):
        arr[0] = np.nanmean(arr[1])
        arr[nlat - 1] = np.nanmean(arr[nlat - 2])
    return vor, div
 
 
def _inverse_laplacian(field_dh: np.ndarray) -> np.ndarray:
    """Solve lap(s) = field on the sphere; return the scalar potential grid.
 
    field_dh must be a Driscoll-Healy grid (nlat even, nlon = 2*nlat).
    """
    cilm = pysh.expand.SHExpandDH(field_dh, sampling=2)
    lmax = cilm.shape[1] - 1
    l = np.arange(lmax + 1)
    fac = np.zeros(lmax + 1)
    fac[1:] = -A_EARTH ** 2 / (l[1:] * (l[1:] + 1))   # lap Y_lm = -l(l+1)/a^2 Y_lm
    return pysh.expand.MakeGridDH(cilm * fac[None, :, None], sampling=2)
 
 
def helmholtz(u: np.ndarray, v: np.ndarray, lat: np.ndarray):
    """Split (u, v) into rotational and divergent parts on the sphere.
 
    Returns a dict with vorticity, divergence, streamfunction psi, velocity
    potential chi, and the reconstructed rotational/divergent winds.  All grids
    are the Driscoll-Healy (south-pole-trimmed) 72x144 grid; lat_dh = lat[:-1].
    """
    vor, div = vorticity_divergence(u, v, lat)
    vor_dh, div_dh = vor[:-1], div[:-1]
    lat_dh = lat[:-1]
 
    psi = _inverse_laplacian(vor_dh)     # lap(psi) = vorticity
    chi = _inverse_laplacian(div_dh)     # lap(chi) = divergence
 
    cg = np.cos(np.deg2rad(lat_dh))[:, None]
    phg = np.deg2rad(lat_dh)
    # divergent wind = grad(chi); rotational wind = k x grad(psi)
    u_chi = _dlon(chi) / (A_EARTH * cg)
    v_chi = np.gradient(chi, phg, axis=0) / A_EARTH
    u_psi = -np.gradient(psi, phg, axis=0) / A_EARTH
    v_psi = _dlon(psi) / (A_EARTH * cg)
    return dict(vor=vor_dh, div=div_dh, psi=psi, chi=chi, lat=lat_dh,
                u_chi=u_chi, v_chi=v_chi, u_psi=u_psi, v_psi=v_psi)
 
 
def ke_spectra(u: np.ndarray, v: np.ndarray, lat: np.ndarray):
    """Rotational and divergent kinetic-energy spectra by spherical degree l.
 
    Uses the identity KE_rot ~ sum_l S_zeta(l)/(l(l+1)),
    KE_div ~ sum_l S_delta(l)/(l(l+1)), where S is the SH power per degree of
    vorticity / divergence.  Returns (l, KE_rot(l), KE_div(l)).
    """
    vor, div = vorticity_divergence(u, v, lat)
    s_z = pysh.spectralanalysis.spectrum(pysh.expand.SHExpandDH(vor[:-1], sampling=2))
    s_d = pysh.spectralanalysis.spectrum(pysh.expand.SHExpandDH(div[:-1], sampling=2))
    l = np.arange(len(s_z))
    w = np.zeros_like(l, dtype=float)
    w[1:] = 1.0 / (l[1:] * (l[1:] + 1))
    return l, s_z * w, s_d * w
 
 
def ke_ratio_block(u: np.ndarray, v: np.ndarray, lat: np.ndarray):
    """Time-averaged KE_div/KE_rot and the mean rotational/divergent spectra
    over a (nt, nlat, nlon) block."""
    KEr = KEd = None
    for t in range(u.shape[0]):
        l, kr, kd = ke_spectra(u[t], v[t], lat)
        KEr = kr if KEr is None else KEr + kr
        KEd = kd if KEd is None else KEd + kd
    KEr /= u.shape[0]
    KEd /= u.shape[0]
    return l, KEr, KEd, float(KEd.sum() / KEr.sum())
