"""2D linear acoustics on a periodic grid, plus an FFT Poisson solver.

Linear acoustic system (small perturbations about a fluid at rest, density rho0):

    dp/dt = -rho0 * c^2 * (div u)
    du/dt = -(1/rho0) * grad p + f - gamma * u

with an optional steady body force f and linear (Rayleigh) damping gamma.

* c is the sound speed.  Information propagates at finite speed c -> the
  pressure obeys a wave equation (hyperbolic).
* As c -> infinity the system is stiff: any divergence in u is corrected
  almost instantly, and the steady-state pressure satisfies the elliptic
  Poisson equation  laplacian(p) = rho0 * div(f).  That is the incompressible
  / low-Mach limit.

The time integrator is the standard staggered-in-time (symplectic) update used
in FDTD acoustics: advance p, then use the *new* p to advance u.  It is stable
under the CFL condition  c * dt / dx <= 1/sqrt(2)  in 2D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Periodic finite-difference operators (2nd-order central)
# ---------------------------------------------------------------------------

def ddx(field: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * dx)


def ddy(field: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * dx)


def divergence(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    return ddx(u, dx) + ddy(v, dx)


# ---------------------------------------------------------------------------
# Elliptic reference: solve laplacian(p) = rhs on a periodic grid via FFT
# ---------------------------------------------------------------------------

def poisson_fft(rhs: np.ndarray, dx: float) -> np.ndarray:
    """Solve laplacian(p) = rhs on a periodic square grid (zero-mean solution).

    This is the *elliptic* / instantaneous pressure field — the c -> infinity
    limit of the acoustic system.
    """
    n = rhs.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid divide-by-zero; the mean is set to zero below
    rhs_hat = np.fft.fft2(rhs)
    p_hat = -rhs_hat / k2
    p_hat[0, 0] = 0.0
    p = np.real(np.fft.ifft2(p_hat))
    return p


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class AcousticState:
    p: np.ndarray
    u: np.ndarray
    v: np.ndarray
    t: float


class LinearAcoustics2D:
    """Explicit staggered-in-time integrator for 2D linear acoustics."""

    def __init__(self, n: int, L: float, c: float, rho0: float = 1.0,
                 gamma: float = 0.0, cfl: float = 0.4):
        self.n = n
        self.L = L
        self.dx = L / n
        self.c = c
        self.rho0 = rho0
        self.gamma = gamma
        # CFL-limited time step (2D acoustic limit is dx/(c*sqrt(2))).
        self.dt = cfl * self.dx / (c * np.sqrt(2.0))
        self.x = (np.arange(n) + 0.5) * self.dx
        self.state = AcousticState(
            p=np.zeros((n, n)),
            u=np.zeros((n, n)),
            v=np.zeros((n, n)),
            t=0.0,
        )

    def set_pressure(self, p0: np.ndarray) -> None:
        self.state.p = p0.copy()

    def step(self, fx: np.ndarray | None = None,
             fy: np.ndarray | None = None) -> None:
        s = self.state
        dx, dt = self.dx, self.dt
        # Advance pressure from current velocity divergence.
        div = divergence(s.u, s.v, dx)
        s.p = s.p - dt * self.rho0 * self.c**2 * div
        # Advance velocity using the updated pressure (symplectic).
        gp_x = ddx(s.p, dx)
        gp_y = ddy(s.p, dx)
        damp = 1.0 - dt * self.gamma
        ax = -(1.0 / self.rho0) * gp_x
        ay = -(1.0 / self.rho0) * gp_y
        if fx is not None:
            ax = ax + fx
        if fy is not None:
            ay = ay + fy
        s.u = damp * s.u + dt * ax
        s.v = damp * s.v + dt * ay
        s.t += dt

    def run(self, t_end: float, fx: np.ndarray | None = None,
            fy: np.ndarray | None = None) -> None:
        while self.state.t < t_end - 1e-12:
            self.step(fx=fx, fy=fy)


# ---------------------------------------------------------------------------
# Forcing helpers
# ---------------------------------------------------------------------------

def gaussian_bump(n: int, L: float, x0: float, y0: float,
                  sigma: float, amp: float = 1.0) -> np.ndarray:
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    xx, yy = np.meshgrid(x, x, indexing="ij")
    # periodic-aware distance
    dxp = np.minimum(np.abs(xx - x0), L - np.abs(xx - x0))
    dyp = np.minimum(np.abs(yy - y0), L - np.abs(yy - y0))
    r2 = dxp**2 + dyp**2
    return amp * np.exp(-r2 / (2.0 * sigma**2))


def gradient_forcing(psi: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Irrotational forcing f = grad(psi).  Its divergence is laplacian(psi)."""
    return ddx(psi, dx), ddy(psi, dx)
