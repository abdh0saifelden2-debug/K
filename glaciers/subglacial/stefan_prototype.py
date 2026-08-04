r"""Minimal moving-boundary (Stefan) prototype for the subglacial cavity.

This is the *Option-3* prototype: it demonstrates the feedback loop

    melt  ->  ice-base motion  ->  flow change  ->  melt

inside the existing pseudo-spectral + Brinkman-penalty machinery, **without**
building a full phase-field or level-set code.  The single new ingredient is a
*moving* ice base: the flat ``ice_base`` of :mod:`subglacial.flow` is replaced
by a 1-D height ``H_c(x, t)`` that recedes as the ice melts.

Geometry (doubly-periodic ``[0, 2*pi)^2``)::

      y = Ly   +-----------------------------+
               |          ICE  (solid)        |   theta = theta_ice (cold)
      H_c(x,t) +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+   <- MOVING melt front
               |        meltwater CAVITY       |   body-force driven (+x)
      y_bed    +-----------------------------+
               |        warm bed  (solid)     |   theta = theta_bed (warm)
      y = 0    +-----------------------------+

Differences from :class:`subglacial.flow.SubglacialFlow`:

* **Moving mask.**  The ice penalty mask is rebuilt from a level-set
  ``phi(x, y) = y - H_c(x)`` every ``n_mask`` steps.  ``H_c`` is advanced
  explicitly by the Stefan condition ``dH_c/dt = m / St`` (a melting ice base
  *recedes upward*, growing the cavity; see :meth:`stefan_update`).
* **Body-force drive.**  The constant-mass-flux controller of ``flow.py`` is
  *disabled*.  The flow is driven by a tidal body force
  ``f(t) = f0 + df * sin(omega_tide t)`` acting in ``+x`` within the cavity, so
  the cavity flow is natural rather than pinned.
* **Melt diagnostic.**  ``m(x, t)`` is the conductive heat flux arriving at the
  interface, ``-kappa d(theta)/dy`` sampled in the fluid just below ``H_c``.

Honest scope (a *proof of concept*, not an operational cavity model):

* 1-D ice base ``H_c(x, t)`` -- no overhangs / detached ice.
* explicit mask update -- CFL-like limit ``dt < St * dy / max(m)``.
* diffuse (Brinkman) interface, ~3-5 cells thick -- not a sharp front.
* ice is a *stationary* penalty solid -- no viscous creep.
* constant melting point ``theta_ice`` -- no pressure-melting / regelation.
* 2-D only.

The validated 1-D analytic reference (Neumann similarity solution) lives in
:mod:`subglacial.moving_boundary.stefan` and is used by ``tests/test_stefan.py``
to check that the no-flow limit of this prototype recovers ``s(t) ~ sqrt(t)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compressible.ns import Spectral2D, helmholtz


@dataclass
class StefanConfig:
    n: int = 64                  # grid points per axis (domain is [0, 2*pi)^2)
    nu: float = 3.0e-3           # kinematic viscosity
    kappa: float = 3.0e-3        # thermal diffusivity
    eta: float = 1.0e-3          # Brinkman permeability (penalty = dt/eta in solid)
    dt: float = 1.0e-3
    # geometry
    y_bed: float = 0.25          # flat warm-bed top (solid below)
    H0: float = 1.5              # initial mean ice-base height
    eps: float = 0.05            # ice-base perturbation amplitude
    k0: int = 2                  # ice-base perturbation wavenumber
    interface: float = 2.0       # mask smoothing width (in grid cells); kept sharp so
                                 # the cavity core is penalty-free (chi ~ 0) once the
                                 # mass-flux controller is removed (body-force drive only)
    # thermodynamics
    St: float = 10.0             # Stefan number (rho_i L_f / (rho_w c_p dTheta)); >>1 => slow melt
    theta_bed: float = 1.0       # warm bed temperature
    theta_ice: float = 0.0       # cold ice (melting-point) temperature
    beta: float = 0.5            # Boussinesq buoyancy: warm fluid (theta>0) rises (+y),
                                 # driving the convection that carries heat to the ice base
    # drive (mass-flux controller disabled; body force only)
    f0: float = 0.1              # mean body force (+x)
    df: float = 0.05             # tidal body-force amplitude
    T_tide: float = 1.0          # tidal period (omega = 2*pi / T_tide)
    # moving boundary
    n_mask: int = 10             # rebuild ice mask + advance H_c every n_mask steps
    melt_band: int = 2           # # fluid cells below interface used for the flux sample
    freeze: bool = False         # if True, hold H_c fixed (diagnostic / control runs)
    # subgrid model (cavity only): 'none' | 'smagorinsky'
    sgs: str = "none"
    cs: float = 0.16
    seed: int = 0


class StefanPrototype:
    """2-D penalized cavity flow with a melting (moving) ice base."""

    def __init__(self, cfg: StefanConfig):
        self.cfg = cfg
        self.sp = Spectral2D(cfg.n)
        sp = self.sp
        self.n = cfg.n
        self.dx = sp.dx
        self.Ly = sp.L                       # domain height (= 2*pi)

        X, Y = sp.grid()                     # X varies along axis 0, Y along axis 1
        self.X, self.Y = X, Y
        self.xcol = X[:, 0].copy()           # x coordinate per column (length n)
        self.dmask = cfg.interface * sp.dx   # mask smoothing length

        # warm bed: flat solid band below y_bed (fixed; never melts)
        self.chi_bed = 0.5 * (1.0 + np.tanh((cfg.y_bed - Y) / self.dmask))

        # moving ice base H_c(x): mean H0 + small sinusoidal perturbation
        self.Hc = cfg.H0 + cfg.eps * np.sin(cfg.k0 * self.xcol)
        self.Hc0 = self.Hc.copy()

        # spectral helpers
        self.k2 = sp.k2
        self.visc_u = np.exp(-cfg.nu * sp.k2 * cfg.dt)
        self.visc_t = np.exp(-cfg.kappa * sp.k2 * cfg.dt)
        kabs = np.sqrt(sp.k2)
        kcut = cfg.n / 2.0
        self.specfilt = np.exp(-36.0 * (kabs / kcut) ** 16)

        # build the initial ice mask + derived penalty fields
        self._rebuild_masks()

        # state: start the cavity isothermal-ish (warm bed pinned by penalty)
        self.u = np.zeros((cfg.n, cfg.n))
        self.v = np.zeros((cfg.n, cfg.n))
        self.theta = self.theta_solid.copy()
        self.t = 0.0
        self.step_count = 0

    def init_warm_cavity(self):
        """Seed the cavity with a warm->cold linear temperature profile (warm fluid
        in contact with the ice base).  This is the physically relevant initial
        state for melting -- the ice base sees warm water immediately, rather than
        waiting for heat to diffuse up from a distant bed -- and is what makes the
        no-flow limit recover the Neumann similarity solution."""
        cfg = self.cfg
        span = np.maximum(self.Hc[:, None] - cfg.y_bed, 1e-9)
        frac = np.clip((self.Hc[:, None] - self.Y) / span, 0.0, 1.0)
        prof = cfg.theta_ice + frac * (cfg.theta_bed - cfg.theta_ice)
        self.theta = np.where(self.fluid, prof, self.theta_solid)
        return self

    # ------------------------------------------------------------------ #
    # masks / moving boundary
    # ------------------------------------------------------------------ #
    def _rebuild_masks(self):
        """Recompute the ice mask (and all penalty-derived fields) from H_c."""
        cfg = self.cfg
        Y = self.Y
        # level set phi = y - H_c(x);  ice where phi > 0 (above the base)
        chi_ice = 0.5 * (1.0 + np.tanh((Y - self.Hc[:, None]) / self.dmask))
        self.chi_ice = chi_ice
        self.chi = np.clip(self.chi_bed + chi_ice, 0.0, 1.0)     # all solid
        self.fluid = self.chi < 0.5
        self.fvol = float(self.fluid.sum()) + 1e-30
        # heat penalty target: warm bed (theta_bed) and cold ice (theta_ice)
        self.theta_solid = self.chi_bed * cfg.theta_bed + chi_ice * cfg.theta_ice
        self.pen = cfg.dt * self.chi / cfg.eta                   # implicit penalty field

    def _interface_index(self):
        """For each column, the y-index of the fluid cell just below the ice base."""
        # Y[:, j] = j * dx ; interface at Y == Hc.  Largest fluid index per column.
        jint = np.floor(self.Hc / self.dx).astype(int) - 1
        return np.clip(jint, 1, self.n - 2)

    # ------------------------------------------------------------------ #
    # operators
    # ------------------------------------------------------------------ #
    def _project(self, u, v):
        us, vs, _, _ = helmholtz(self.sp, u, v)
        return us, vs

    def _advect(self, u, v, f):
        return -(u * self.sp.ddx(f) + v * self.sp.ddy(f))

    def _body_force(self):
        """Tidal body force f(t) = f0 + df sin(omega t) in +x, acting in the cavity."""
        cfg = self.cfg
        omega = 2.0 * np.pi / cfg.T_tide
        fx = (cfg.f0 + cfg.df * np.sin(omega * self.t)) * self.fluid
        return fx

    def _sgs_force(self, u, v):
        cfg = self.cfg
        if cfg.sgs != "smagorinsky":
            return np.zeros_like(u), np.zeros_like(v)
        sp = self.sp
        delta = sp.dx
        ux, uy = sp.ddx(u), sp.ddy(u)
        vx, vy = sp.ddx(v), sp.ddy(v)
        s11, s22, s12 = ux, vy, 0.5 * (uy + vx)
        smag = np.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2))
        nu_t = (cfg.cs * delta) ** 2 * smag * self.fluid
        mx = sp.ddx(2.0 * nu_t * s11) + sp.ddy(2.0 * nu_t * s12)
        my = sp.ddx(2.0 * nu_t * s12) + sp.ddy(2.0 * nu_t * s22)
        return mx, my

    # ------------------------------------------------------------------ #
    # melt rate + Stefan boundary motion
    # ------------------------------------------------------------------ #
    def melt_rate(self):
        """Per-column melt rate m(x) = -kappa d(theta)/dy sampled in the fluid just
        below the ice base (the conductive heat flux delivered to the interface).

        Positive m => heat into the ice => melting.  Averaged over ``melt_band``
        fluid cells under the interface for a smoother estimate."""
        cfg = self.cfg
        dthdy = self.sp.ddy(self.theta)
        jint = self._interface_index()
        m = np.zeros(self.n)
        for b in range(cfg.melt_band):
            jj = np.clip(jint - b, 1, self.n - 2)
            m += -cfg.kappa * dthdy[np.arange(self.n), jj]
        m /= max(cfg.melt_band, 1)
        return m

    def stefan_update(self, n_sub):
        """Advance the ice base by the explicit Stefan condition over ``n_sub`` steps:

            dH_c/dt = + m / St      (melting => interface recedes upward => cavity grows)

        ``m`` is the interfacial flux from :meth:`melt_rate`.  The motion is
        accumulated over the ``n_sub`` substeps since the last mask rebuild."""
        cfg = self.cfg
        if cfg.freeze:
            return
        m = self.melt_rate()
        self.Hc = self.Hc + (n_sub * cfg.dt) * m / cfg.St
        # keep the base inside the domain, above the warm bed
        self.Hc = np.clip(self.Hc, cfg.y_bed + 2.0 * self.dx, self.Ly - 2.0 * self.dx)
        self._rebuild_masks()

    # ------------------------------------------------------------------ #
    # time step
    # ------------------------------------------------------------------ #
    def step(self):
        cfg = self.cfg
        sp = self.sp
        u, v, theta = self.u, self.v, self.theta

        Nu = self._advect(u, v, u)
        Nv = self._advect(u, v, v)
        Nt = self._advect(u, v, theta)
        mx, my = self._sgs_force(u, v)
        Nu = Nu + mx + self._body_force()
        # Boussinesq buoyancy: warm cavity fluid rises, driving convection that
        # transports heat from the warm bed up to the (melting) ice base.
        Nv = Nv + my + cfg.beta * theta * self.fluid

        uh = self.specfilt * self.visc_u * (sp.fft(u) + cfg.dt * sp.fft(Nu) * sp.dealias)
        vh = self.specfilt * self.visc_u * (sp.fft(v) + cfg.dt * sp.fft(Nv) * sp.dealias)
        th = self.specfilt * self.visc_t * (sp.fft(theta) + cfg.dt * sp.fft(Nt) * sp.dealias)
        u1, v1, t1 = sp.ifft(uh), sp.ifft(vh), sp.ifft(th)

        # implicit Brinkman penalty: u -> 0 in solid, theta -> theta_solid in solid
        u1 = u1 / (1.0 + self.pen)
        v1 = v1 / (1.0 + self.pen)
        t1 = (t1 + self.pen * self.theta_solid) / (1.0 + self.pen)

        # Leray projection (elliptic pressure solve) -- NO mass-flux controller
        u1, v1 = self._project(u1, v1)

        self.u, self.v, self.theta = u1, v1, t1
        self.t += cfg.dt
        self.step_count += 1

    def run(self, steps, report_every=0):
        """Advance ``steps`` flow steps, rebuilding the ice mask + moving H_c every
        ``n_mask`` steps (explicit Stefan update)."""
        cfg = self.cfg
        since = 0
        for s in range(steps):
            self.step()
            since += 1
            if since >= cfg.n_mask:
                self.stefan_update(since)
                since = 0
            if report_every and s % report_every == 0:
                print(f"    step {s:6d}  t={self.t:.3f}  Hbar={self.Hc.mean():.4f}  "
                      f"sig_h={self.roughness():.3e}  KE={self.kinetic_energy():.3e}")
        if since > 0:
            self.stefan_update(since)
        return self

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def mean_height(self):
        return float(self.Hc.mean())

    def melt_distance(self):
        """Mean melt-front displacement |H_c - H_c(0)| (the Stefan front position)."""
        return float(np.abs(self.Hc - self.Hc0).mean())

    def roughness(self):
        """RMS ice-base roughness sigma_h(t)."""
        return float(np.sqrt(np.mean((self.Hc - self.Hc.mean()) ** 2)))

    def height_spectrum(self):
        """Power spectrum E_h(k) = |hat H_c(k)|^2 of the ice-base height."""
        H = np.fft.rfft(self.Hc - self.Hc.mean())
        return np.abs(H) ** 2

    def kinetic_energy(self):
        return 0.5 * float(np.mean((self.u ** 2 + self.v ** 2) * self.fluid))

    def nusselt(self):
        """Normalized mean interfacial heat transfer Nu = <m>_x / (kappa dTheta / H0)."""
        cfg = self.cfg
        dT = cfg.theta_bed - cfg.theta_ice
        ref = cfg.kappa * dT / cfg.H0 + 1e-30
        return float(self.melt_rate().mean() / ref)

    def penalty_consistency(self):
        """Ratio <|u|^2>_ice / <|u|^2>_fluid (should be << 1: flow is killed in solid)."""
        sol = ~self.fluid
        usol = float(np.mean((self.u ** 2 + self.v ** 2)[sol])) if sol.any() else 0.0
        ufl = float(np.mean((self.u ** 2 + self.v ** 2)[self.fluid])) + 1e-30
        return usol / ufl
