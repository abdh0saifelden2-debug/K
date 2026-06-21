r"""Penalized pseudo-spectral solver for 2D subglacial cavity flow + heat.
 
Geometry (doubly-periodic [0, 2*pi)^2, volume penalization embeds the solids):
 
      y = 2*pi  +-----------------------------+
                |          ICE  (solid)        |   theta = 0 (cold ice base)
        y_ice   +-----------------------------+
                |                              |
                |     meltwater CAVITY         |   flow driven in +x
                |   (fluid; lee wakes here)    |
        y_bed(x)+~~~~/\~~~/\~~~~/\~~~~/\~~~~~~~+
                |        ROCK  (solid)         |   theta = 1 (warm bed)
      y = 0     +-----------------------------+
 
The bottom rock bed is bumpy; the top ice base is flat.  Both are stationary
solids imposed by a Brinkman penalty (u -> 0 in solid).  The heat tracer is
penalized to theta = 1 in the rock (geothermal/frictional warmth) and theta = 0
in the ice; in the cavity it is advected and diffused.  The mean cavity flow is
held at U0 by a constant-mass-flux body force.
 
Subgrid models (applied in the cavity only):
  * 'none'         -- DNS (no model); use at high resolution for truth.
  * 'smagorinsky'  -- K-theory: m = div(2 nu_t S), nu_t=(Cs*Delta)^2 |S| >= 0.
                      Positive-definite => purely dissipative, no backscatter.
  * 'backscatter'  -- the Part-8 repair: the same Smagorinsky dissipation PLUS a
                      Leray-projected stochastic force whose local variance is
                      tied to the Smagorinsky dissipation rate (FDT-style), so a
                      controlled fraction of the drained energy is returned to the
                      resolved scales without breaking incompressibility.
 
This is a pedagogical demonstration solver, not a validated production code.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
 
from compressible.ns import Spectral2D, helmholtz
 
 
@dataclass
class SubglacialConfig:
    n: int = 256
    nu: float = 6.0e-4            # kinematic viscosity (sets the Reynolds number)
    kappa: float = 6.0e-4         # heat diffusivity (Pr = nu/kappa = 1)
    eta: float = 5.0e-5           # Brinkman permeability (penalty = dt/eta in solid)
    U0: float = 1.0               # target bulk cavity velocity (+x)
    dt: float = 3.0e-4
    bed_mean: float = 0.9         # mean rock-bed top height
    bed_amp: float = 0.55         # bedrock bump amplitude (strong constriction)
    ice_base: float = 2.4         # flat ice-base height (solid above)
    interface: float = 5.0        # solid-interface smoothing width (in grid cells)
    sgs: str = "none"             # 'none' | 'smagorinsky' | 'backscatter'
    cs: float = 0.16              # Smagorinsky constant
    backscatter: float = 0.6      # fraction of SGS dissipation returned (0..1)
    bs_tau: float = 0.0           # backscatter memory time (0 = white noise, as before;
                                  # >0 = OU-correlated stochastic force with this tau_mem)
    f_amp: float = 0.0            # ambient turbulence forcing amplitude (0 = off)
    k_f: float = 14.0             # forcing ring wavenumber
    f_band: float = 1.5           # forcing ring half-width
    f_tau: float = 0.05           # forcing correlation time (Ornstein-Uhlenbeck)
    seed: int = 0
    # optional real bed: 1-D bed-top height per grid column (length n).  When set,
    # it replaces the synthetic sinusoidal bed (see subglacial/bedmap.py).
    bed_profile: "np.ndarray | None" = None
 
 
class SubglacialFlow:
    def __init__(self, cfg: SubglacialConfig):
        self.cfg = cfg
        self.sp = Spectral2D(cfg.n)
        sp = self.sp
        self.n = cfg.n
        self.dx = sp.dx
        self.rng = np.random.default_rng(cfg.seed)
 
        X, Y = sp.grid()
        self.X, self.Y = X, Y
 
        # --- rock bed (bumpy) and ice base (flat) masks via smoothed Heaviside ---
        if cfg.bed_profile is not None:
            # real measured bed: a 1-D height per x-column (X varies along axis 0)
            hp = np.asarray(cfg.bed_profile, dtype=float).ravel()
            if hp.shape[0] != cfg.n:
                hp = np.interp(np.linspace(0.0, 1.0, cfg.n, endpoint=False),
                               np.linspace(0.0, 1.0, hp.shape[0], endpoint=False), hp)
            ybed = np.broadcast_to(hp[:, None], (cfg.n, cfg.n)).copy()
        else:
            ybed = cfg.bed_mean + cfg.bed_amp * (
                0.6 * np.sin(3 * X) + 0.4 * np.sin(5 * X + 0.7)
                + 0.5 * np.exp(-((X - np.pi) ** 2) / 0.15)
            )
        self.ybed = ybed
        d = cfg.interface * sp.dx
        chi_rock = 0.5 * (1.0 + np.tanh((ybed - Y) / d))      # 1 below bed top
        chi_ice = 0.5 * (1.0 + np.tanh((Y - cfg.ice_base) / d))  # 1 above ice base
        self.chi_rock = chi_rock
        self.chi_ice = chi_ice
        self.chi = np.clip(chi_rock + chi_ice, 0.0, 1.0)       # all solid
        self.fluid = (self.chi < 0.5)
        self.fvol = float(self.fluid.sum())
 
        # heat penalty target: warm rock (1), cold ice (0)
        self.theta_solid = chi_rock * 1.0 + chi_ice * 0.0
 
        # spectral helpers
        self.k2 = sp.k2
        self.kabs = np.sqrt(sp.k2)
        self.visc_u = np.exp(-cfg.nu * sp.k2 * cfg.dt)
        self.visc_t = np.exp(-cfg.kappa * sp.k2 * cfg.dt)
        kcut = cfg.n / 2.0
        self.specfilt = np.exp(-36.0 * (self.kabs / kcut) ** 16)
        self.pen = cfg.dt * self.chi / cfg.eta                 # implicit penalty field
        self.ring = ((self.kabs >= cfg.k_f - cfg.f_band)
                     & (self.kabs <= cfg.k_f + cfg.f_band)).astype(float)
        self.Fx = np.zeros((cfg.n, cfg.n))   # persistent OU forcing state
        self.Fy = np.zeros((cfg.n, cfg.n))
        self.bs_x = np.zeros((cfg.n, cfg.n))  # persistent OU backscatter state (bs_tau>0)
        self.bs_y = np.zeros((cfg.n, cfg.n))
 
        # state
        self.u = np.zeros((cfg.n, cfg.n))
        self.v = np.zeros((cfg.n, cfg.n))
        self.theta = self.theta_solid.copy()
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
        confined to the cavity.  A persistent forcing field with correlation time
        f_tau and stationary RMS f_amp; correlated-in-time forcing injects energy
        efficiently (rate ~ dt, not dt^2) and sustains a turbulent cavity."""
        sp = self.sp
        cfg = self.cfg
        wx = sp.fft(self.rng.standard_normal((self.n, self.n))) * self.ring
        wy = sp.fft(self.rng.standard_normal((self.n, self.n))) * self.ring
        nx, ny = self._project(sp.ifft(wx), sp.ifft(wy))
        nx *= self.fluid
        ny *= self.fluid
        rms = float(np.sqrt(np.mean(nx ** 2 + ny ** 2))) + 1e-30
        nx *= cfg.f_amp / rms
        ny *= cfg.f_amp / rms
        a = cfg.dt / cfg.f_tau
        self.Fx = (1.0 - a) * self.Fx + np.sqrt(2.0 * a) * nx
        self.Fy = (1.0 - a) * self.Fy + np.sqrt(2.0 * a) * ny
        return self.Fx, self.Fy
 
    def _strain(self, u, v):
        sp = self.sp
        ux, uy = sp.ddx(u), sp.ddy(u)
        vx, vy = sp.ddx(v), sp.ddy(v)
        s11, s22, s12 = ux, vy, 0.5 * (uy + vx)
        smag = np.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2))
        return s11, s22, s12, smag
 
    # ------------------------------------------------------------------ #
    # subgrid model -> momentum force (mx, my) acting in the cavity
    # ------------------------------------------------------------------ #
    def _sgs_force(self, u, v):
        cfg = self.cfg
        if cfg.sgs == "none":
            z = np.zeros_like(u)
            return z, z, 0.0
        sp = self.sp
        delta = sp.dx
        s11, s22, s12, smag = self._strain(u, v)
        nu_t = (cfg.cs * delta) ** 2 * smag
        nu_t *= self.fluid                                   # model acts in cavity
        # Smagorinsky stress divergence: m = div(2 nu_t S)
        tau11 = 2.0 * nu_t * s11
        tau22 = 2.0 * nu_t * s22
        tau12 = 2.0 * nu_t * s12
        mx = sp.ddx(tau11) + sp.ddy(tau12)
        my = sp.ddx(tau12) + sp.ddy(tau22)
        # local SGS dissipation rate eps = 2 nu_t |S|^2 (energy drained by Smagorinsky)
        eps = 2.0 * nu_t * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2)
        eps_mean = float(np.mean(eps * self.fluid))
 
        if cfg.sgs == "backscatter" and cfg.backscatter > 0.0:
            # FDT-linked, Leray-projected stochastic backscatter.  Local forcing
            # variance is tied to the local Smagorinsky dissipation (return a
            # fraction `backscatter` of the drained energy), then projected
            # divergence-free so the fast (elliptic) clock is respected exactly.
            amp = np.sqrt(np.maximum(eps, 0.0) / cfg.dt) * self.fluid
            wx = self.rng.standard_normal(u.shape)
            wy = self.rng.standard_normal(u.shape)
            fx, fy = self._project(amp * wx, amp * wy)       # solenoidal white noise
            # scale so the injected power = backscatter * drained power
            inj = float(np.mean((fx * u + fy * v) * self.fluid))
            drained = eps_mean
            if abs(inj) > 1e-30 and drained > 0.0:
                scale = cfg.backscatter * drained / inj
                # keep the sign sane (net energy return), clip runaway amplitudes
                scale = float(np.clip(scale, -5.0, 5.0))
                gx, gy = scale * fx, scale * fy              # FDT-scaled backscatter force
                if cfg.bs_tau > 0.0:
                    # OU memory on the FDT-scaled force itself: tau_mem = bs_tau, via
                    # the EXACT exponential discretization F_n = F_{n-1} e^{-dt/tau}
                    # + sqrt(1-e^{-2 dt/tau}) (scale*noise).  It preserves the
                    # stationary variance (so the mean energy return is unchanged),
                    # stays solenoidal (linear combo of projected fields -> the fast
                    # elliptic clock is respected), and now carries BOTH the amplitude
                    # and the pattern of previous shear states (the Theorem-10 memory).
                    # bs_tau=0 leaves the white path byte-identical.
                    ex = float(np.exp(-cfg.dt / cfg.bs_tau))
                    sg = float(np.sqrt(1.0 - ex * ex))
                    self.bs_x = ex * self.bs_x + sg * gx
                    self.bs_y = ex * self.bs_y + sg * gy
                    gx, gy = self.bs_x, self.bs_y
                mx = mx + gx
                my = my + gy
        return mx, my, eps_mean
 
    # ------------------------------------------------------------------ #
    # time step
    # ------------------------------------------------------------------ #
    def step(self, Utarget=None, fext=None):
        cfg = self.cfg
        sp = self.sp
        u, v, theta = self.u, self.v, self.theta
        if Utarget is None:
            Utarget = cfg.U0
 
        Nu = self._advect(u, v, u)
        Nv = self._advect(u, v, v)
        Nt = self._advect(u, v, theta)
        mx, my, eps_mean = self._sgs_force(u, v)
        Nu = Nu + mx
        Nv = Nv + my
        if fext is not None:                                 # optional external body force
            Nu = Nu + fext[0]
            Nv = Nv + fext[1]
        if cfg.f_amp > 0.0:
            fx, fy = self._forcing()
            Nu = Nu + fx
            Nv = Nv + fy
 
        uh = self.specfilt * self.visc_u * (sp.fft(u) + cfg.dt * sp.fft(Nu) * sp.dealias)
        vh = self.specfilt * self.visc_u * (sp.fft(v) + cfg.dt * sp.fft(Nv) * sp.dealias)
        th = self.specfilt * self.visc_t * (sp.fft(theta) + cfg.dt * sp.fft(Nt) * sp.dealias)
        u1, v1, t1 = sp.ifft(uh), sp.ifft(vh), sp.ifft(th)
 
        # implicit Brinkman penalty: u -> 0 in solid, theta -> theta_solid in solid
        u1 = u1 / (1.0 + self.pen)
        v1 = v1 / (1.0 + self.pen)
        t1 = (t1 + self.pen * self.theta_solid) / (1.0 + self.pen)
 
        # Leray projection (elliptic pressure solve)
        u1, v1 = self._project(u1, v1)
 
        # constant-mass-flux forcing: relax mean cavity u toward Utarget
        umean = float((u1 * self.fluid).sum() / self.fvol)
        u1 = u1 + 0.5 * (Utarget - umean)
        u1, v1 = self._project(u1, v1)
 
        self.u, self.v, self.theta = u1, v1, t1
        self.t += cfg.dt
        self.step_count += 1
        return eps_mean
 
    def run(self, steps, ramp=0, report_every=0):
        cfg = self.cfg
        for s in range(steps):
            Ut = cfg.U0 * (min(1.0, (s + 1) / ramp) if ramp > 0 else 1.0)
            self.step(Ut)
            if report_every and s % report_every == 0:
                print(f"    step {s:6d}  umax={np.abs(self.u).max():.3f}  "
                      f"KE={self.kinetic_energy():.4e}")
        return self
 
    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def velocity(self):
        return self.u, self.v
 
    def vorticity(self):
        return self.sp.ddx(self.v) - self.sp.ddy(self.u)
 
    def pressure(self):
        """Elliptic (incompressible) pressure in the cavity: lap(p) = -d_i u_j d_j u_i."""
        sp = self.sp
        u, v = self.u, self.v
        ux, uy = sp.ddx(u), sp.ddy(u)
        vx, vy = sp.ddx(v), sp.ddy(v)
        src = -(ux * ux + 2.0 * uy * vx + vy * vy)
        p_h = -sp.fft(src) * sp.k2_inv
        p_h[0, 0] = 0.0
        return sp.ifft(p_h)
 
    def kinetic_energy(self):
        return 0.5 * float(np.mean((self.u ** 2 + self.v ** 2) * self.fluid))
 
    def wake_band(self):
        """Boolean mask of the cavity band just above the bedrock bumps (the lee
        wake region), where separation eddies live."""
        lo = self.ybed + 0.12
        hi = self.cfg.ice_base - 0.25
        return self.fluid & (self.Y > lo) & (self.Y < hi)
 
    def wake_tke(self, umean_x):
        """Turbulent KE in the wake band, using a supplied mean x-velocity field
        (or scalar) to define fluctuations."""
        band = self.wake_band()
        up = self.u - umean_x
        vp = self.v
        tke = 0.5 * (up ** 2 + vp ** 2)
        return float(np.mean(tke[band]))
 
    def melt_flux(self):
        """Heat flux delivered to the ice base = -kappa * d(theta)/dy, summed over
        the cavity cells just below the ice (a melt-rate proxy)."""
        sp = self.sp
        dthdy = sp.ddy(self.theta)
        # band of fluid within ~0.3 below the ice base
        band = self.fluid & (self.Y > self.cfg.ice_base - 0.45) & (self.Y < self.cfg.ice_base)
        flux = -self.cfg.kappa * dthdy
        return float(np.mean(flux[band])), band
 
    def heat_in_wake(self):
        """Mean heat content (theta) trapped in the wake band -- the lee heat trap."""
        band = self.wake_band()
        return float(np.mean(self.theta[band]))
