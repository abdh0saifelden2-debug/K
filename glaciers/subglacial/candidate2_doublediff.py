r"""Candidate 2 -- double-diffusive layering (salt + temperature).

Heat diffuses ~100x faster than salt in water.  In an ice-shelf cavity the
meltwater is cold *and* fresh while the ocean is warm *and* salty, so the two
buoyancy-active scalars have opposing gradients with very different
diffusivities -- the classic *salt-finger* configuration.  The pre-registered
prediction is that the thermal transport is enhanced in a hump:

* ``Nu_T`` (thermal Nusselt) peaks near the density ratio ``R_rho ~ 2`` (the
  salt-finger sweet spot),
* ``Nu_T -> 1`` for ``R_rho < 1`` (diffusive convection, weak) and for
  ``R_rho >~ Le`` (stably stratified, fingers shut off),
* Smagorinsky over-dissipates the fingers, lowering ``Nu_T``.

Implementation (2-D x-y, backend-agnostic; pass ``xp=cupy`` for GPU).  Same
resolved penalised cavity as Candidates 1/4 (so the Brinkman penalty does not
bleed into the fluid), but with **two** advected/diffused scalars:

* temperature ``theta`` with diffusivity ``kappa_T`` and
* salinity ``S`` with diffusivity ``kappa_S = kappa_T / Le`` (Lewis number
  ``Le = Pr_S/Pr_T ~ 100``),

both seeded with the per-column conduction ramp (warm+salty bed -> cold+fresh
ice) and pinned to those wall values by the penalty.  The buoyancy is
``b = Ri_T (theta - <theta>) - Ri_T R_rho (S - <S>)`` so temperature is
destabilising and salt is stabilising; sweeping ``R_rho`` moves through the
finger regime.

Honest scope (same boundary as the Stefan prototype / Candidates 1, 4): the
*wall* flux is conduction-limited, so ``Nu_T`` is measured from the **interior
turbulent fluxes** ``<v'theta'>`` / ``<v'S'>`` (finger transport), not from the
no-slip wall gradient.  All of ``Nu_T``, ``Nu_S`` and the Turner flux ratio are
reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _make_rng(xp, seed):
    try:
        return xp.random.default_rng(seed)
    except AttributeError:  # pragma: no cover - older CuPy
        return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# anisotropic 2-D spectral operators on [0, Lx) x [0, Ly)
# --------------------------------------------------------------------------- #
class SpectralAniso2D:
    """Fourier operators on a rectangular doubly-periodic box (Lx may != Ly)."""

    def __init__(self, nx, ny, Lx, Ly, xp=np):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx / nx, Ly / ny
        self.xp = xp
        kx = xp.asarray(2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx))
        ky = xp.asarray(2.0 * np.pi * np.fft.fftfreq(ny, d=self.dy))
        self.kx, self.ky = xp.meshgrid(kx, ky, indexing="ij")
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2_inv = xp.where(self.k2 > 0, 1.0 / xp.where(self.k2 > 0, self.k2, 1.0), 0.0)
        kxmax, kymax = np.max(np.abs(kx)) * 2.0 / 3.0, np.max(np.abs(ky)) * 2.0 / 3.0
        self.dealias = (xp.abs(self.kx) <= kxmax) & (xp.abs(self.ky) <= kymax)
        self.kabs = xp.sqrt(self.k2)

    def grid(self):
        x = self.xp.arange(self.nx) * self.dx
        y = self.xp.arange(self.ny) * self.dy
        return self.xp.meshgrid(x, y, indexing="ij")

    def fft(self, f):
        return self.xp.fft.fft2(f)

    def ifft(self, F):
        return self.xp.real(self.xp.fft.ifft2(F))

    def ddx(self, f):
        return self.ifft(1j * self.kx * self.fft(f))

    def ddy(self, f):
        return self.ifft(1j * self.ky * self.fft(f))


def project2d(sp, u, v):
    """Leray projection: remove the divergent part of (u, v)."""
    uh, vh = sp.fft(u), sp.fft(v)
    div_h = 1j * sp.kx * uh + 1j * sp.ky * vh
    phi_h = -div_h * sp.k2_inv
    u = u - sp.ifft(1j * sp.kx * phi_h)
    v = v - sp.ifft(1j * sp.ky * phi_h)
    return u, v


@dataclass
class DoubleDiffConfig:
    # grid / domain
    nx: int = 256
    ny: int = 96
    A: float = 4.0                # cavity aspect ratio Lx/Ly (Lx = A * 2*pi)
    # physics
    nu: float = 8.0e-4
    kappa_T: float = 8.0e-4
    Le: float = 100.0             # Lewis number = kappa_T / kappa_S (Pr_S/Pr_T)
    eta: float = 5.0e-5
    dt: float = 4.0e-4
    # geometry: tall cavity so the penalty does not bleed into the fluid gap
    y_bed: float = 0.30
    y_ice: float = 5.50
    interface: float = 1.5
    # optional scalloped ice wall (§D.2): y_ice(x) = y_ice + a*sin(2*pi*n*x/Lx)
    # with a = wall_amp * (Lx / wall_nwaves).  Defaults (0) keep the flat wall
    # bit-for-bit, so the smooth-wall Candidate-2 results are unchanged.
    wall_amp: float = 0.0
    wall_nwaves: int = 0
    # ambient turbulence forcing (seeds the fingers)
    f_amp: float = 0.6
    k_f: float = 6.0
    f_band: float = 2.0
    f_tau: float = 0.05
    # double-diffusive stratification / closure
    Ri_T: float = 1.0             # thermal buoyancy strength (destabilising)
    R_rho: float = 2.0            # density ratio alpha_S dS / (alpha_T dT)
    sgs: str = "none"             # 'none' | 'smagorinsky' | 'backscatter'
    cs: float = 0.16
    backscatter: float = 0.6
    bs_tau: float = 0.0
    seed: int = 0

    @property
    def kappa_S(self) -> float:
        return self.kappa_T / self.Le


class DoubleDiffFlow:
    """Resolved penalised cavity carrying two buoyancy-active scalars (heat and
    salt) with a Lewis-number diffusivity contrast; measures the thermal/haline
    Nusselt numbers and the Turner flux ratio."""

    def __init__(self, cfg: DoubleDiffConfig, xp=np):
        self.cfg = cfg
        self.xp = xp
        Lx, Ly = cfg.A * 2.0 * np.pi, 2.0 * np.pi
        self.Ly = Ly
        self.sp = SpectralAniso2D(cfg.nx, cfg.ny, Lx, Ly, xp)
        sp = self.sp
        self.rng = _make_rng(xp, cfg.seed)

        X, Y = sp.grid()
        self.X, self.Y = X, Y

        # ice-wall height: flat (scalar ``cfg.y_ice``) or scalloped per-column
        # ``y_ice(x)`` when ``wall_amp>0`` and ``wall_nwaves>0`` (§D.2).
        if cfg.wall_amp != 0.0 and cfg.wall_nwaves > 0:
            lam = Lx / cfg.wall_nwaves
            a = cfg.wall_amp * lam
            y_ice_x = cfg.y_ice + a * xp.sin(2.0 * np.pi * cfg.wall_nwaves
                                             * X[:, 0:1] / Lx)   # (nx, 1)
        else:
            y_ice_x = xp.full((cfg.nx, 1), float(cfg.y_ice))
        self.y_ice_x = y_ice_x[:, 0]                              # (nx,)

        d = cfg.interface * sp.dy
        chi_rock = 0.5 * (1.0 + xp.tanh((cfg.y_bed - Y) / d))
        chi_ice = 0.5 * (1.0 + xp.tanh((Y - y_ice_x) / d))
        self.chi = xp.clip(chi_rock + chi_ice, 0.0, 1.0)
        self.fluid = (self.chi < 0.5)
        self.fvol = float(self.fluid.sum()) + 1e-30
        # warm + salty bed (value 1), cold + fresh ice (value 0)
        self.scal_solid = chi_rock * 1.0 + chi_ice * 0.0

        self.k2 = sp.k2
        self.visc_u = xp.exp(-cfg.nu * sp.k2 * cfg.dt)
        self.visc_T = xp.exp(-cfg.kappa_T * sp.k2 * cfg.dt)
        self.visc_S = xp.exp(-cfg.kappa_S * sp.k2 * cfg.dt)
        # Mild spectral safety filter -- essentially unfiltered except at the
        # highest (finer-direction) modes.  Do NOT "fix" this to
        # ``min(pi/dx, pi/dy)``: on the anisotropic grid (Lx != Ly) that
        # applies the *coarse* x-direction Nyquist isotropically to ``kabs``
        # and over-smooths the well-resolved y-direction, which corrupts the
        # diagnostics (e.g. flips candidate3's feedback sign).
        kcut = min(cfg.nx, cfg.ny) / 2.0
        self.specfilt = xp.exp(-36.0 * (sp.kabs / (kcut + 1e-30)) ** 16)
        self.pen = cfg.dt * self.chi / cfg.eta
        self.ring = ((sp.kabs >= cfg.k_f - cfg.f_band)
                     & (sp.kabs <= cfg.k_f + cfg.f_band)).astype(float)
        self.Fx = xp.zeros((cfg.nx, cfg.ny))
        self.Fy = xp.zeros((cfg.nx, cfg.ny))

        # per-column linear conduction ramp (1 at the bed -> 0 at the ice);
        # with a scalloped wall the local cavity height varies with ``x``.
        Hc_x = xp.maximum(y_ice_x - cfg.y_bed, 1e-30)            # (nx, 1)
        # bulk-gradient normalisation for the cavity-mean Nusselt: the mean ice
        # height equals ``cfg.y_ice`` (sin averages to 0), so this is unchanged
        # at ``wall_amp=0`` and is the correct mean for the scalloped wall too.
        self.Hcav = max(cfg.y_ice - cfg.y_bed, 1e-30)
        ramp = xp.clip((y_ice_x - Y) / Hc_x, 0.0, 1.0)
        self.ramp = ramp

        self.u = xp.zeros((cfg.nx, cfg.ny))
        self.v = xp.zeros((cfg.nx, cfg.ny))
        self.theta = ramp.copy()        # temperature
        self.S = ramp.copy()            # salinity
        self.t = 0.0
        self.step_count = 0
        self.bs_x = xp.zeros((cfg.nx, cfg.ny))
        self.bs_y = xp.zeros((cfg.nx, cfg.ny))

    # ------------------------------------------------------------------ #
    def _advect(self, u, v, f):
        return -(u * self.sp.ddx(f) + v * self.sp.ddy(f))

    def _forcing(self):
        sp, cfg, xp = self.sp, self.cfg, self.xp
        shape = (cfg.nx, cfg.ny)
        wx = sp.fft(self.rng.standard_normal(shape)) * self.ring
        wy = sp.fft(self.rng.standard_normal(shape)) * self.ring
        nx, ny = project2d(sp, sp.ifft(wx), sp.ifft(wy))
        nx, ny = nx * self.fluid, ny * self.fluid
        rms = float(xp.sqrt(xp.mean(nx ** 2 + ny ** 2))) + 1e-30
        nx, ny = nx * (cfg.f_amp / rms), ny * (cfg.f_amp / rms)
        a = cfg.dt / cfg.f_tau
        self.Fx = (1.0 - a) * self.Fx + np.sqrt(2.0 * a) * nx
        self.Fy = (1.0 - a) * self.Fy + np.sqrt(2.0 * a) * ny
        return self.Fx, self.Fy

    def _sgs_force(self, u, v):
        cfg, sp, xp = self.cfg, self.sp, self.xp
        if cfg.sgs == "none":
            z = xp.zeros_like(u)
            return z, xp.zeros_like(u)
        delta = np.sqrt(sp.dx * sp.dy)
        ux, uy, vx, vy = sp.ddx(u), sp.ddy(u), sp.ddx(v), sp.ddy(v)
        s11, s22, s12 = ux, vy, 0.5 * (uy + vx)
        smag = xp.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2))
        nu_t = (cfg.cs * delta) ** 2 * smag * self.fluid
        mx = sp.ddx(2.0 * nu_t * s11) + sp.ddy(2.0 * nu_t * s12)
        my = sp.ddx(2.0 * nu_t * s12) + sp.ddy(2.0 * nu_t * s22)
        eps_mean = float(xp.mean(2.0 * nu_t * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2)))
        if cfg.sgs == "backscatter" and cfg.backscatter > 0.0 and eps_mean > 0.0:
            amp = np.sqrt(max(eps_mean, 0.0) / max(cfg.dt, 1e-30))
            if cfg.bs_tau > 0.0:
                ex = np.exp(-cfg.dt / cfg.bs_tau)
                sg = np.sqrt(max(1.0 - ex * ex, 0.0))
                self.bs_x = ex * self.bs_x + sg * self.rng.standard_normal(u.shape)
                self.bs_y = ex * self.bs_y + sg * self.rng.standard_normal(u.shape)
                wx, wy = self.bs_x, self.bs_y
            else:
                wx = self.rng.standard_normal(u.shape)
                wy = self.rng.standard_normal(u.shape)
            fx, fy = project2d(sp, amp * wx, amp * wy)
            inj = float(xp.mean((fx * u + fy * v) * self.fluid))
            if abs(inj) > 1e-30:
                scale = float(np.clip(cfg.backscatter * eps_mean / inj, -5.0, 5.0))
                mx, my = mx + scale * fx, my + scale * fy
        return mx, my

    # ------------------------------------------------------------------ #
    def step(self):
        cfg, sp, xp = self.cfg, self.sp, self.xp
        u, v, theta, S = self.u, self.v, self.theta, self.S

        Nu = self._advect(u, v, u)
        Nv = self._advect(u, v, v)
        Nt = self._advect(u, v, theta)
        Ns = self._advect(u, v, S)
        mx, my = self._sgs_force(u, v)
        Nu = Nu + mx
        Nv = Nv + my
        if cfg.f_amp > 0.0:
            fx, fy = self._forcing()
            Nu, Nv = Nu + fx, Nv + fy
        # double-diffusive buoyancy: warm (theta) lifts, salt (S) sinks
        if cfg.Ri_T != 0.0:
            inv = 1.0 / xp.mean(self.fluid)
            tref = float(xp.mean(theta * self.fluid) * inv)
            sref = float(xp.mean(S * self.fluid) * inv)
            b = (cfg.Ri_T * (theta - tref)
                 - cfg.Ri_T * cfg.R_rho * (S - sref))
            Nv = Nv + b * self.fluid

        uh = self.specfilt * self.visc_u * (sp.fft(u) + cfg.dt * sp.fft(Nu) * sp.dealias)
        vh = self.specfilt * self.visc_u * (sp.fft(v) + cfg.dt * sp.fft(Nv) * sp.dealias)
        th = self.specfilt * self.visc_T * (sp.fft(theta) + cfg.dt * sp.fft(Nt) * sp.dealias)
        sh = self.specfilt * self.visc_S * (sp.fft(S) + cfg.dt * sp.fft(Ns) * sp.dealias)
        u1, v1, t1, s1 = sp.ifft(uh), sp.ifft(vh), sp.ifft(th), sp.ifft(sh)

        u1 = u1 / (1.0 + self.pen)
        v1 = v1 / (1.0 + self.pen)
        t1 = (t1 + self.pen * self.scal_solid) / (1.0 + self.pen)
        s1 = (s1 + self.pen * self.scal_solid) / (1.0 + self.pen)
        u1, v1 = project2d(sp, u1, v1)

        self.u, self.v, self.theta, self.S = u1, v1, t1, s1
        self.t += cfg.dt
        self.step_count += 1

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def kinetic_energy(self):
        return 0.5 * float(self.xp.mean((self.u ** 2 + self.v ** 2) * self.fluid))

    def _turb_flux(self, scal):
        """Interior turbulent vertical flux ``<v' c'>`` (means removed)."""
        ff = self.fluid
        fvol = self.fvol
        vbar = float((self.v * ff).sum() / fvol)
        cbar = float((scal * ff).sum() / fvol)
        return float(((self.v - vbar) * (scal - cbar) * ff).sum() / fvol)

    def turb_heat_flux(self):
        return self._turb_flux(self.theta)

    def turb_salt_flux(self):
        return self._turb_flux(self.S)

    def nusselt(self):
        """Thermal/haline Nusselt numbers from the interior turbulent flux,
        ``Nu = 1 + <v'c'> / (kappa_c * dC/dy)`` with the imposed bulk gradient
        ``dC/dy = 1 / H_cav``.  Also returns the Turner flux ratio
        ``gamma = F_T / (R_rho F_S)`` (the heat-to-salt buoyancy-flux ratio)."""
        cfg = self.cfg
        grad = 1.0 / self.Hcav
        F_T = self.turb_heat_flux()
        F_S = self.turb_salt_flux()
        Nu_T = 1.0 + F_T / (cfg.kappa_T * grad)
        Nu_S = 1.0 + F_S / (cfg.kappa_S * grad)
        gamma = F_T / (cfg.R_rho * F_S) if abs(F_S) > 1e-30 else 0.0
        return dict(Nu_T=Nu_T, Nu_S=Nu_S, gamma=gamma, F_T=F_T, F_S=F_S)


def run_case(cfg: DoubleDiffConfig, spinup, measure, sample_every=10, xp=np):
    """Run one case; average the Nusselt numbers / flux ratio over the
    measurement window (fingers need a long spinup to organise)."""
    s = DoubleDiffFlow(cfg, xp=xp)
    s.run(spinup)
    NuT, NuS, gam, ke = [], [], [], []
    nblocks = max(measure // sample_every, 1)
    for _ in range(nblocks):
        s.run(sample_every)
        d = s.nusselt()
        NuT.append(d["Nu_T"])
        NuS.append(d["Nu_S"])
        gam.append(d["gamma"])
        ke.append(s.kinetic_energy())
    return {
        "Nu_T": float(np.mean(NuT)), "Nu_S": float(np.mean(NuS)),
        "gamma": float(np.mean(gam)), "KE_mean": float(np.mean(ke)),
        "umax": float(s.xp.abs(s.u).max()),
        "R_rho": cfg.R_rho, "Le": cfg.Le,
    }


if __name__ == "__main__":
    for Rr in (1.0, 2.0, 5.0):
        cfg = DoubleDiffConfig(nx=128, ny=64, A=4.0, R_rho=Rr)
        r = run_case(cfg, spinup=600, measure=600, xp=np)
        print(f"R_rho={Rr:4.1f}  Nu_T={r['Nu_T']:6.3f}  Nu_S={r['Nu_S']:8.2f}  "
              f"gamma={r['gamma']:+.3f}  KE={r['KE_mean']:.3e}  umax={r['umax']:.3f}")
