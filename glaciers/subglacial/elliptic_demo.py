r"""Geometry -> global pressure: the two clocks made literal.

The bed geometry enters the dynamics through the *pressure* boundary condition.
This module isolates that fact with a controlled experiment on a frozen, developed
cavity field: add a single localized bump to the bed (a new little obstacle) and
ask how the flow responds.

A localized obstacle in an incompressible stream introduces a *localized*
continuity source ``q = -div(bump * u)`` (fluid piles up on the stoss side and is
starved in the lee).  We then apply the SAME source ``q`` to the two operators the
"two clocks" thesis distinguishes -- both are functions of the Laplacian, so this
is an apples-to-apples test:

  * elliptic / pressure clock     ->  dp  = (lap)^{-1} q          (steady Poisson)
  * parabolic / temperature clock ->  dth = exp(t* lap) q         (heat kernel)

The elliptic inverse-Laplacian is *global*: in 2D its Green's function is
~ln r, so a localized source is felt across the entire domain instantaneously.
The heat kernel is *local*: ``exp(t lap)`` is a Gaussian of width ``sqrt(4 kappa t)``,
so the same source stays a confined blob.  That contrast -- identical source,
global vs local response -- is the whole thesis in one picture, and it is driven
purely by the solid geometry.
"""
from __future__ import annotations

import numpy as np

from compressible.ns import Spectral2D


def geometry_bump(f, x0: float, dy: float = 0.12, sigma: float = 0.18):
    """A localized solid bump added to the bed top at along-cavity position ``x0``,
    restricted to the part poking into the fluid cavity."""
    X, Y = f.X, f.Y
    y0 = float(np.interp(x0, X[:, 0], f.ybed[:, 0])) + dy
    dxp = np.angle(np.exp(1j * (X - x0)))            # periodic distance in x
    bump = np.exp(-(dxp ** 2 + (Y - y0) ** 2) / (2.0 * sigma ** 2))
    return bump * f.fluid, (x0, y0)


def obstacle_source(f, bump):
    """Localized continuity source a bump introduces in the oncoming stream:
    ``q = -div(bump * u_vec)`` (zero-mean)."""
    sp = f.sp
    qx, qy = bump * f.u, bump * f.v
    q = -(sp.ddx(qx) + sp.ddy(qy))
    return q - q.mean()


def elliptic_response(sp: Spectral2D, q: np.ndarray):
    """Pressure (elliptic) response to source ``q``: solve lap(dp) = q."""
    dp_h = -sp.fft(q) * sp.k2_inv
    dp_h[0, 0] = 0.0
    return sp.ifft(dp_h).real


def parabolic_response(sp: Spectral2D, q: np.ndarray, kappa: float, tstar: float):
    """Temperature (parabolic) response to the same source after diffusion time
    ``tstar``: apply the heat kernel ``exp(t* kappa lap)``."""
    return sp.ifft(np.exp(-kappa * sp.k2 * tstar) * sp.fft(q)).real


def diffusion_time_for_length(length: float, kappa: float) -> float:
    """Diffusion time for the heat kernel to spread a given length: t = L^2/(4 kappa)."""
    return length ** 2 / (4.0 * kappa)


def radial_from(f, center):
    """Periodic-in-x radial distance from ``center`` = (x0, y0)."""
    X, Y = f.X, f.Y
    x0, y0 = center
    dxp = np.angle(np.exp(1j * (X - x0)))
    return np.sqrt(dxp ** 2 + (Y - y0) ** 2)


def radial_profile(field, r, mask, rmax=2.8, nbin=24):
    """Mean |field| in radial bins (within ``mask``), normalised to its peak."""
    m = np.abs(field) * mask
    bins = np.linspace(0.0, rmax, nbin)
    idx = np.digitize(r.ravel(), bins)
    fr = m.ravel()
    prof = np.array([fr[idx == i].mean() if np.any(idx == i) else np.nan
                     for i in range(1, len(bins))])
    rc = 0.5 * (bins[1:] + bins[:-1])
    peak = np.nanmax(prof)
    return rc, prof / (peak + 1e-30)


def far_fraction(field, r, mask, rc: float = 0.6) -> float:
    """Fraction of the response magnitude located farther than ``rc`` from the
    source -- a scalar measure of non-locality (global ~1, local ~0)."""
    m = np.abs(field) * mask
    tot = m.sum() + 1e-30
    return float(m[(r > rc) & mask].sum() / tot)
