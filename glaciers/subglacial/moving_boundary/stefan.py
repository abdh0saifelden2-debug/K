r"""One-phase Stefan problem: analytic Neumann solution + enthalpy-method solver.

This is the *validated core* of the moving-boundary (free-surface) machinery.
Before coupling a melting ice base to the turbulent cavity flow, we first prove
that the numerics reproduce the textbook moving-front behaviour: a melt front
that advances as ``X(t) = 2 lambda sqrt(alpha t)`` with the correct
Stefan-number prefactor ``lambda``.

Problem (one-phase melting, fusion temperature ``T_m`` taken as 0):

      x = 0                      x = X(t)                 x = Lx
        | <------ liquid ------> | <------ solid -------> |
      T = T_w (> 0)            T = T_m = 0            T = T_m = 0

A semi-infinite solid initially at the fusion temperature is heated at ``x = 0``
to ``T_w > T_m``.  Only the liquid carries sensible heat (one-phase); the solid
stays at ``T_m``.  The liquid/solid interface ``X(t)`` advances into the solid.

Neumann similarity solution
---------------------------
With thermal diffusivity ``alpha = k / (rho c)`` and Stefan number

    St = c (T_w - T_m) / L_f ,

the interface position is ``X(t) = 2 lambda sqrt(alpha t)`` where ``lambda``
solves the transcendental equation

    lambda * exp(lambda**2) * erf(lambda) = St / sqrt(pi) .

The liquid temperature profile is

    T(x, t) = T_w * (1 - erf(x / (2 sqrt(alpha t))) / erf(lambda)) .

Enthalpy method
---------------
We solve on a fixed grid for the volumetric enthalpy ``H = c T + L_f phi`` where
``phi in [0, 1]`` is the liquid fraction.  Inverting ``H``:

    H < 0          -> solid,  T = H / c,        phi = 0
    0 <= H <= L_f  -> interface, T = 0,         phi = H / L_f
    H > L_f        -> liquid, T = (H - L_f)/c,   phi = 1

The update is the conservative ``dH/dt = d/dx (k dT/dx)`` (rho = 1), integrated
explicitly (FTCS).  The latent heat is handled implicitly by the phase map, so
no front-tracking is needed -- the front is wherever ``phi`` crosses 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf


# --------------------------------------------------------------------------- #
# Analytic Neumann solution
# --------------------------------------------------------------------------- #
def neumann_lambda(stefan: float) -> float:
    r"""Solve ``lambda e^{lambda^2} erf(lambda) = St / sqrt(pi)`` for ``lambda``.

    Parameters
    ----------
    stefan : float
        Stefan number ``St = c (T_w - T_m) / L_f`` (must be > 0).

    Returns
    -------
    float
        The similarity constant ``lambda`` in ``X(t) = 2 lambda sqrt(alpha t)``.
    """
    if stefan <= 0.0:
        raise ValueError("Stefan number must be positive for melting.")
    rhs = stefan / np.sqrt(np.pi)

    def f(lam: float) -> float:
        return lam * np.exp(lam * lam) * erf(lam) - rhs

    # f is monotone increasing from 0 (at lam->0) to +inf; bracket the root.
    hi = 1.0
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Failed to bracket Neumann lambda.")
    return float(brentq(f, 1e-12, hi, xtol=1e-13, rtol=1e-12))


def neumann_front(t, stefan: float, alpha: float):
    """Analytic interface position ``X(t)`` of the one-phase Stefan problem."""
    lam = neumann_lambda(stefan)
    return 2.0 * lam * np.sqrt(alpha * np.asarray(t, dtype=float) * 1.0)


def neumann_temperature(x, t: float, T_w: float, stefan: float, alpha: float):
    """Analytic liquid temperature profile ``T(x, t)`` (``T_m = 0``).

    Returns ``T_w`` clamped to the melt temperature beyond the front.
    """
    x = np.asarray(x, dtype=float)
    lam = neumann_lambda(stefan)
    if t <= 0.0:
        return np.where(x <= 0.0, T_w, 0.0)
    eta = x / (2.0 * np.sqrt(alpha * t))
    T = T_w * (1.0 - erf(eta) / erf(lam))
    return np.where(x <= 2.0 * lam * np.sqrt(alpha * t), np.maximum(T, 0.0), 0.0)


# --------------------------------------------------------------------------- #
# Enthalpy-method numerical solver
# --------------------------------------------------------------------------- #
@dataclass
class Stefan1DConfig:
    """Parameters for the 1-D enthalpy Stefan solver (rho = 1, T_m = 0)."""

    nx: int = 400          # number of grid points
    Lx: float = 1.0        # domain length
    k: float = 1.0         # thermal conductivity
    c: float = 1.0         # specific heat (alpha = k / c since rho = 1)
    Lf: float = 1.0        # latent heat of fusion
    T_w: float = 1.0       # imposed warm temperature at x = 0 (T_m = 0)
    dt: float | None = None  # time step; default = 0.4 * dx^2 / alpha (stable)
    cfl: float = 0.4       # explicit-diffusion safety factor when dt is None


class Stefan1D:
    """Fixed-grid enthalpy-method solver for the one-phase Stefan problem."""

    def __init__(self, cfg: Stefan1DConfig):
        self.cfg = cfg
        self.x = np.linspace(0.0, cfg.Lx, cfg.nx)
        self.dx = self.x[1] - self.x[0]
        self.alpha = cfg.k / cfg.c
        if cfg.dt is None:
            self.dt = cfg.cfl * self.dx * self.dx / self.alpha
        else:
            self.dt = cfg.dt
            if self.dt > 0.5 * self.dx * self.dx / self.alpha:
                raise ValueError(
                    "dt exceeds explicit-diffusion stability limit "
                    f"({0.5 * self.dx * self.dx / self.alpha:.3e})."
                )
        self.t = 0.0
        # Initial enthalpy: all solid at the fusion temperature (H = 0),
        # except the warm Dirichlet node at x = 0 (liquid at T_w).
        self.H = np.zeros(cfg.nx)
        self.H[0] = cfg.Lf + cfg.c * cfg.T_w

    # -- enthalpy <-> (temperature, liquid fraction) maps ------------------- #
    def temperature(self, H=None):
        cfg = self.cfg
        H = self.H if H is None else H
        T = np.empty_like(H)
        solid = H < 0.0
        liquid = H > cfg.Lf
        mush = ~(solid | liquid)
        T[solid] = H[solid] / cfg.c
        T[mush] = 0.0
        T[liquid] = (H[liquid] - cfg.Lf) / cfg.c
        return T

    def liquid_fraction(self, H=None):
        cfg = self.cfg
        H = self.H if H is None else H
        return np.clip(H / cfg.Lf, 0.0, 1.0)

    # -- front position (phi = 0.5 contour) -------------------------------- #
    def front_position(self) -> float:
        phi = self.liquid_fraction()
        # first index from the warm wall where liquid fraction drops below 0.5
        below = np.where(phi < 0.5)[0]
        if below.size == 0:
            return self.cfg.Lx
        j = below[0]
        if j == 0:
            return 0.0
        # linear interpolation of the phi = 0.5 crossing between j-1 and j
        phi0, phi1 = phi[j - 1], phi[j]
        if phi0 == phi1:
            return float(self.x[j])
        frac = (phi0 - 0.5) / (phi0 - phi1)
        return float(self.x[j - 1] + frac * self.dx)

    # -- time stepping ------------------------------------------------------ #
    def step(self):
        cfg = self.cfg
        T = self.temperature()
        # Dirichlet warm wall at x = 0, insulated (Neumann) far wall at x = Lx.
        lap = np.zeros_like(T)
        lap[1:-1] = (T[2:] - 2.0 * T[1:-1] + T[:-2]) / (self.dx * self.dx)
        # Zero-flux (dT/dx = 0) at the far wall via a mirror ghost node
        # T[n] = T[n-2], giving lap[-1] = 2*(T[n-2] - T[n-1])/dx^2.
        lap[-1] = 2.0 * (T[-2] - T[-1]) / (self.dx * self.dx)
        self.H[1:] = self.H[1:] + self.dt * cfg.k * lap[1:]
        # re-pin the warm Dirichlet node (keep it liquid at T_w)
        self.H[0] = cfg.Lf + cfg.c * cfg.T_w
        self.t += self.dt

    def run(self, t_end: float):
        while self.t < t_end - 1e-12:
            self.step()
        return self.t
