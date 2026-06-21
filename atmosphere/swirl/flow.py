r"""Pseudo-spectral solver for a sustained turbulent swirl (Goldshtik-Sorokin).

A coherent central vortex is held in a doubly-periodic [0, 2*pi)^2 spectral box
against drag, while ring-band stochastic forcing keeps an ambient turbulent field
alive on top of it.  The developed flow is a *turbulent swirl*: a smooth vortex
core surrounded by fine-scale turbulence.

The physically important object is the vortex core's **pressure well**.  The
incompressible pressure solves a Poisson equation lap(p) = -d_i u_j d_j u_i, so
the swirl (with its strong cyclostrophic balance rho u_theta^2 / r = dp/dr) digs a
deep, *global* (elliptic) low-pressure depression at the core.  That radial
pressure-gradient force is exactly what holds a heavy particle up against gravity
in the Goldshtik-Sorokin effect.

Subgrid models (applied a-priori in the closure scorecard, not inside the DNS):
the same Part-8b operators (Smagorinsky, spectrum-matched surrogate, projected-FDT)
are scored on how they transfer energy to/from the resolved swirl.  An optional
in-solver `sgs` mode mirrors `subglacial.flow` for sanity checks.

This is a pedagogical demonstration solver, not a validated production code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compressible.ns import Spectral2D, helmholtz, incompressible_pressure


@dataclass
class SwirlConfig:
    n: int = 128
    nu: float = 6.0e-4            # kinematic viscosity (sets the Reynolds number)
    dt: float = 3.0e-4
    U_swirl: float = 1.4          # peak azimuthal speed of the target core vortex
    r_core: float = 0.7           # core radius of the target vortex
    mu: float = 0.06              # linear drag (sink that the swirl is sustained against)
    gamma: float = 2.5            # relaxation rate toward the target coherent vortex
    k_f: float = 12.0             # turbulence forcing ring wavenumber
    f_band: float = 2.0           # forcing ring half-width
    f_amp: float = 1.2            # forcing amplitude (stationary RMS)
    f_tau: float = 0.05           # forcing correlation time (Ornstein-Uhlenbeck)
    sgs: str = "none"             # 'none' | 'smagorinsky' | 'backscatter'
    cs: float = 0.16              # Smagorinsky constant
    backscatter: float = 0.6      # fraction of SGS dissipation returned (0..1)
    seed: int = 1


class SwirlFlow:
    """Velocity-form penalty-free 2D NS: a central vortex sustained against drag
    plus ring-band turbulence forcing."""

    def __init__(self, cfg: SwirlConfig, target_uth=None):
        self.cfg = cfg
        self.sp = Spectral2D(cfg.n)
        sp = self.sp
        self.n = cfg.n
        self.dx = sp.dx
        self.rng = np.random.default_rng(cfg.seed)

        X, Y = sp.grid()
        self.X, self.Y = X, Y
        self.x0 = np.pi
        self.y0 = np.pi

        # periodic distance from the box centre
        dxp = np.angle(np.exp(1j * (X - self.x0)))
        dyp = np.angle(np.exp(1j * (Y - self.y0)))
        r = np.sqrt(dxp ** 2 + dyp ** 2) + 1e-12
        self.r = r

        # target steady vortex.  By default a smooth (Gaussian-core) azimuthal
        # profile peaking near r_core; if ``target_uth(r)`` is supplied (e.g. a
        # real Holland fit from swirl.otis) it overrides the shape.
        if target_uth is not None:
            uth = np.asarray(target_uth(r), dtype=float)
        else:
            uth = cfg.U_swirl * (1.0 - np.exp(-(r ** 2) / (2 * cfg.r_core ** 2))) / (r / cfg.r_core + 1e-9)
            uth *= np.exp(-(r ** 2) / (2 * (1.8 * cfg.r_core) ** 2))
        # unit tangential vectors (counter-clockwise): (-sin, cos) about the centre
        self.ut_x = -uth * (dyp / r)
        self.ut_y = uth * (dxp / r)

        # spectral helpers
        self.k2 = sp.k2
        self.kabs = np.sqrt(sp.k2)
        self.visc = np.exp(-cfg.nu * sp.k2 * cfg.dt)
        kcut = cfg.n / 2.0
        self.specfilt = np.exp(-36.0 * (self.kabs / kcut) ** 16)
        self.ring = ((self.kabs >= cfg.k_f - cfg.f_band)
                     & (self.kabs <= cfg.k_f + cfg.f_band)).astype(float)
        self.Fx = np.zeros((cfg.n, cfg.n))   # persistent OU forcing state
        self.Fy = np.zeros((cfg.n, cfg.n))

        # state -- start from the target vortex
        self.u = self.ut_x.copy()
        self.v = self.ut_y.copy()
        self.t = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------ #
    # operators
    # ------------------------------------------------------------------ #
    def _project(self, u, v):
        us, vs, _, _ = helmholtz(self.sp, u, v)
        return us, vs

    def _advect(self, u, v, f):
        return -(u * self.sp.ddx(f) + v * self.sp.ddy(f))

    def _forcing(self):
        """Time-correlated (Ornstein-Uhlenbeck) solenoidal ring-band body force
        that sustains ambient turbulence on top of the coherent vortex."""
        sp = self.sp
        cfg = self.cfg
        wx = sp.fft(self.rng.standard_normal((self.n, self.n))) * self.ring
        wy = sp.fft(self.rng.standard_normal((self.n, self.n))) * self.ring
        nx, ny = self._project(sp.ifft(wx), sp.ifft(wy))
        rms = float(np.sqrt(np.mean(nx ** 2 + ny ** 2))) + 1e-30
        nx *= cfg.f_amp / rms
        ny *= cfg.f_amp / rms
        a = cfg.dt / cfg.f_tau
        self.Fx = (1.0 - a) * self.Fx + np.sqrt(2.0 * a) * nx
        self.Fy = (1.0 - a) * self.Fy + np.sqrt(2.0 * a) * ny
        return self.Fx, self.Fy

    def _sgs_force(self, u, v):
        cfg = self.cfg
        if cfg.sgs == "none":
            z = np.zeros_like(u)
            return z, z
        sp = self.sp
        delta = sp.dx
        ux, uy = sp.ddx(u), sp.ddy(u)
        vx, vy = sp.ddx(v), sp.ddy(v)
        s11, s22, s12 = ux, vy, 0.5 * (uy + vx)
        smag = np.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2))
        nu_t = (cfg.cs * delta) ** 2 * smag
        mx = sp.ddx(2 * nu_t * s11) + sp.ddy(2 * nu_t * s12)
        my = sp.ddx(2 * nu_t * s12) + sp.ddy(2 * nu_t * s22)
        eps = 2.0 * nu_t * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2)
        if cfg.sgs == "backscatter" and cfg.backscatter > 0.0:
            amp = np.sqrt(np.maximum(eps, 0.0) / cfg.dt)
            fx, fy = self._project(amp * self.rng.standard_normal(u.shape),
                                   amp * self.rng.standard_normal(u.shape))
            inj = float(np.mean(fx * u + fy * v))
            drained = float(np.mean(eps))
            if abs(inj) > 1e-30 and drained > 0.0:
                scale = float(np.clip(cfg.backscatter * drained / inj, -5.0, 5.0))
                mx = mx + scale * fx
                my = my + scale * fy
        return mx, my

    # ------------------------------------------------------------------ #
    # time step
    # ------------------------------------------------------------------ #
    def step(self):
        cfg = self.cfg
        sp = self.sp
        u, v = self.u, self.v
        Nu = self._advect(u, v, u)
        Nv = self._advect(u, v, v)
        mx, my = self._sgs_force(u, v)
        Nu = Nu + mx
        Nv = Nv + my
        # sustain the coherent vortex (relax toward target) + drag
        Nu = Nu + cfg.gamma * (self.ut_x - u) - cfg.mu * u
        Nv = Nv + cfg.gamma * (self.ut_y - v) - cfg.mu * v
        if cfg.f_amp > 0.0:
            fx, fy = self._forcing()
            Nu = Nu + fx
            Nv = Nv + fy
        uh = self.specfilt * self.visc * (sp.fft(u) + cfg.dt * sp.fft(Nu) * sp.dealias)
        vh = self.specfilt * self.visc * (sp.fft(v) + cfg.dt * sp.fft(Nv) * sp.dealias)
        u1, v1 = sp.ifft(uh), sp.ifft(vh)
        u1, v1 = self._project(u1, v1)
        self.u, self.v = u1, v1
        self.t += cfg.dt
        self.step_count += 1

    def run(self, steps, ramp=0, report_every=0):
        for s in range(steps):
            self.step()
            if report_every and s % report_every == 0:
                d, _ = self.core_well_depth()
                print(f"    step {s:6d}  KE={self.kinetic_energy():.4e}  "
                      f"well_depth={d:.4e}")
        return self

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def velocity(self):
        return self.u, self.v

    def vorticity(self):
        return self.sp.ddx(self.v) - self.sp.ddy(self.u)

    def pressure(self):
        """Elliptic (incompressible) pressure: lap(p) = -d_i u_j d_j u_i."""
        return incompressible_pressure(self.sp, self.u, self.v, 1.0).real

    def kinetic_energy(self):
        return 0.5 * float(np.mean(self.u ** 2 + self.v ** 2))

    def core_mask(self, r_in: float = 0.5):
        """Boolean mask of the vortex core (r < r_in)."""
        return self.r < r_in

    def core_well_depth(self, r_in: float = 0.5, r_far: float = 2.2):
        """Depth of the core pressure well: (far-field reference) - (core minimum).
        Positive = a real depression at the core.  Returns (depth, pressure)."""
        p = self.pressure()
        far = float(np.mean(p[self.r > r_far]))
        pmin = float(p[self.r < r_in].min())
        return far - pmin, p
