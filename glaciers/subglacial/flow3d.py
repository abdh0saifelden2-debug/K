r"""Penalized pseudo-spectral solver for **3D** subglacial cavity flow + heat.

This is the 3D, *a-posteriori* (time-stepped) extension of ``subglacial/flow.py``.
Where Part 9 froze a single turbulent field and scored closures a-priori, here we
actually integrate the cavity flow forward in time **with a closure active**, so
the closure shapes the developing wake and the heat it delivers to the ice base.
That lets us read a closure-dependent **basal melt rate** -- the quantity a
glaciologist cares about -- instead of an instantaneous transfer score.

Geometry (triply-periodic [0, 2*pi)^3; volume penalization embeds the solids).
``y`` is vertical, ``x`` is streamwise (mean flow +x), ``z`` is spanwise.  The
rock bed ``y_bed(x)`` is a real measured transect *extruded* along the spanwise
direction (homogeneous in z), so the resolved turbulence is fully 3D while the
topography stays the measured 1-D relief:

      y = 2*pi  +-----------------------------+
                |          ICE  (solid)        |   theta = 0 (cold ice base)
        y_ice   +-----------------------------+
                |       meltwater CAVITY       |   flow driven in +x, 3D wakes
        y_bed(x)+~~~~/\~~~/\~~~~/\~~~~/\~~~~~~~+
                |        ROCK  (solid)         |   theta = 1 (warm bed)
      y = 0     +-----------------------------+

Subgrid models (applied in the cavity only), identical in spirit to the 2D code:
  * 'none'         -- no model (use only as a high-cost reference).
  * 'smagorinsky'  -- K-theory: m = div(2 nu_t S), nu_t=(Cs*Delta)^2 |S| >= 0.
                      Positive-definite => purely dissipative, no backscatter.
  * 'backscatter'  -- the two-clocks repair: the same Smagorinsky dissipation PLUS
                      a Leray-projected (divergence-free) stochastic force whose
                      local variance is tied to the Smagorinsky dissipation rate
                      (FDT-style), returning a controlled fraction of the drained
                      energy to the resolved wake without injecting spurious
                      pressure.

**Backend-agnostic:** every solver constructor accepts an array module ``xp``
that defaults to NumPy.  Pass ``xp=cupy`` (e.g. on a free Colab/Kaggle T4 GPU) to
run the same code on the GPU, which is what makes a genuine **under-resolved LES**
(n >= 128, low molecular viscosity so the SGS carries the dissipation) tractable.
At the low resolutions affordable on CPU the SGS term is a negligible correction
and the closures are nearly indistinguishable -- the closure only controls the
melt rate once the grid is coarse relative to a developed cascade.

This is an LES demonstration solver, not a grid-converged DNS and not a validated
production glacier model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _make_rng(xp, seed):
    """Return a Generator for the given backend (numpy or cupy)."""
    try:
        return xp.random.default_rng(seed)
    except AttributeError:  # pragma: no cover - very old cupy
        return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# minimal spectral operators on [0, 2*pi)^3
# --------------------------------------------------------------------------- #
class Spectral3D:
    def __init__(self, n: int, xp=np):
        self.n = n
        self.xp = xp
        self.L = 2.0 * np.pi
        self.dx = self.L / n
        k = xp.asarray(np.fft.fftfreq(n, d=1.0 / n))  # integer wavenumbers
        self.kx, self.ky, self.kz = xp.meshgrid(k, k, k, indexing="ij")
        self.k2 = self.kx ** 2 + self.ky ** 2 + self.kz ** 2
        self.k2_inv = xp.zeros_like(self.k2)
        self.k2_inv[self.k2 > 0] = 1.0 / self.k2[self.k2 > 0]
        kmax = n // 3
        ax = (xp.abs(k) <= kmax)
        self.dealias = ax[:, None, None] & ax[None, :, None] & ax[None, None, :]

    def grid(self):
        xp = self.xp
        x = xp.arange(self.n) * self.dx
        return xp.meshgrid(x, x, x, indexing="ij")

    def fft(self, f):
        return self.xp.fft.fftn(f)

    def ifft(self, F):
        return self.xp.real(self.xp.fft.ifftn(F))

    def ddx(self, f):
        return self.ifft(1j * self.kx * self.fft(f))

    def ddy(self, f):
        return self.ifft(1j * self.ky * self.fft(f))

    def ddz(self, f):
        return self.ifft(1j * self.kz * self.fft(f))


def project3d(sp: Spectral3D, u, v, w):
    """Leray projection: remove the dilatational (curl-free) part so div(u)=0."""
    uh, vh, wh = sp.fft(u), sp.fft(v), sp.fft(w)
    div_h = 1j * (sp.kx * uh + sp.ky * vh + sp.kz * wh)
    phi_h = -div_h * sp.k2_inv
    u = u - sp.ifft(1j * sp.kx * phi_h)
    v = v - sp.ifft(1j * sp.ky * phi_h)
    w = w - sp.ifft(1j * sp.kz * phi_h)
    return u, v, w


def divergence_rms3d(sp: Spectral3D, u, v, w):
    xp = sp.xp
    duh = 1j * (sp.kx * sp.fft(u) + sp.ky * sp.fft(v) + sp.kz * sp.fft(w))
    div = sp.ifft(duh)
    rms = float(xp.sqrt(xp.mean(div ** 2)))
    scale = float(xp.sqrt(xp.mean(u ** 2 + v ** 2 + w ** 2))) + 1e-30
    return rms / scale


@dataclass
class Subglacial3DConfig:
    n: int = 48
    nu: float = 8.0e-4
    kappa: float = 8.0e-4          # Pr = nu/kappa = 1
    eta: float = 5.0e-5            # Brinkman permeability (penalty = dt/eta in solid)
    U0: float = 1.0               # target bulk cavity velocity (+x)
    dt: float = 4.0e-4
    bed_mean: float = 0.9
    bed_amp: float = 0.55
    ice_base: float = 2.4
    interface: float = 4.0        # solid-interface smoothing (grid cells)
    sgs: str = "none"             # 'none' | 'smagorinsky' | 'backscatter'
    cs: float = 0.16
    backscatter: float = 0.6      # fraction of SGS dissipation returned (0..1)
    bs_tau: float = 0.0           # backscatter memory time (0 = white; >0 = OU colored)
    Ri: float = 0.0               # Richardson number (Boussinesq buoyancy coupling)
    f_amp: float = 1.5            # ambient turbulence forcing amplitude
    k_f: float = 6.0              # forcing ring wavenumber
    f_band: float = 2.0
    f_tau: float = 0.05
    seed: int = 0
    bed_profile: "np.ndarray | None" = None   # real 1-D bed-top height per x-column
    bed_field: "np.ndarray | None" = None     # 2-D bed-top height over (x, z);
    # enables spanwise-varying topography (e.g. streamwise-oriented ridges) so
    # the flow can channel *between* ridges -- the 3D hydraulic topology the
    # z-homogeneous ``bed_profile`` extrusion cannot express. Takes precedence
    # over ``bed_profile`` when both are set.
    thermal_wall: "object | None" = None      # optional ice-side thermal BC object
    # (e.g. ``subglacial.wall_flux.ThermalWall``).  When None (default) the ice
    # boundary is the hard Dirichlet pin theta->0 (infinite-conductance sink).
    # When set, the object's ``apply(flow, t1)`` method replaces the ice pin with
    # a finite-conductance (Robin) flux BC; the rock pin theta->1 is preserved.
    # All wall-model physics lives in that separate object -- the solver only
    # delegates the post-diffusion thermal update to it.
    bed_slip: "float | None" = None            # Navier tangential-slip knob on the
    # ROCK bed.  None (default) => the velocity wall is the usual no-slip Brinkman
    # penalization (u->0 in the solid), unchanged.  A float ``s`` in [0, 1] reduces
    # ONLY the bed-*tangential* penalty (along the local extruded bed-surface
    # normal) while the bed-*normal* component stays fully impermeable and the ice
    # wall stays fully no-slip:  s=1 is bit-for-bit the no-slip baseline, s->0 is
    # free slip (infinite slip length).  This is the volume-penalization analogue
    # of Navier slip ``u_t = lambda_s d u_t/d n`` -- Weertman bed sliding -- and is
    # the controlled test of whether the *stagnant near-bed layer* (rather than the
    # thermal conductive sublayer) is what limits basal heat delivery.

class Subglacial3DFlow:
    def __init__(self, cfg: Subglacial3DConfig, xp=np):
        self.cfg = cfg
        self.xp = xp
        self.sp = Spectral3D(cfg.n, xp)
        sp = self.sp
        self.n = cfg.n
        self.dx = sp.dx
        self.rng = _make_rng(xp, cfg.seed)

        X, Y, Z = sp.grid()
        self.X, self.Y, self.Z = X, Y, Z

        # rock bed and ice base (flat). A 2-D ``bed_field`` h(x, z) gives a
        # spanwise-varying bed (streamwise ridges); ``bed_profile`` gives a 1-D
        # h(x) extruded along z; otherwise a synthetic z-homogeneous relief.
        if cfg.bed_field is not None:
            hf = np.asarray(cfg.bed_field, dtype=float)
            if hf.shape != (cfg.n, cfg.n):
                raise ValueError(
                    f"bed_field must have shape (n, n)=({cfg.n}, {cfg.n}) over "
                    f"(x, z); got {hf.shape}")
            ybed = xp.asarray(np.broadcast_to(hf[:, None, :],
                                              (cfg.n, cfg.n, cfg.n)).copy())
        elif cfg.bed_profile is not None:
            hp = np.asarray(cfg.bed_profile, dtype=float).ravel()
            if hp.shape[0] != cfg.n:
                hp = np.interp(np.linspace(0.0, 1.0, cfg.n, endpoint=False),
                               np.linspace(0.0, 1.0, hp.shape[0], endpoint=False), hp)
            ybed = xp.asarray(np.broadcast_to(hp[:, None, None],
                                              (cfg.n, cfg.n, cfg.n)).copy())
        else:
            ybed = cfg.bed_mean + cfg.bed_amp * (
                0.6 * xp.sin(3 * X) + 0.4 * xp.sin(5 * X + 0.7)
                + 0.5 * xp.exp(-((X - np.pi) ** 2) / 0.15)
            )
        self.ybed = ybed
        d = cfg.interface * sp.dx
        chi_rock = 0.5 * (1.0 + xp.tanh((ybed - Y) / d))
        chi_ice = 0.5 * (1.0 + xp.tanh((Y - cfg.ice_base) / d))
        self.chi_rock = chi_rock
        self.chi_ice = chi_ice
        self.chi = xp.clip(chi_rock + chi_ice, 0.0, 1.0)
        self.fluid = (self.chi < 0.5)
        self.fvol = float(self.fluid.sum())
        self.theta_solid = chi_rock * 1.0 + chi_ice * 0.0

        self.k2 = sp.k2
        self.kabs = xp.sqrt(sp.k2)
        self.visc_u = xp.exp(-cfg.nu * sp.k2 * cfg.dt)
        self.visc_t = xp.exp(-cfg.kappa * sp.k2 * cfg.dt)
        kcut = cfg.n / 2.0
        self.specfilt = xp.exp(-36.0 * (self.kabs / kcut) ** 16)
        self.pen = cfg.dt * self.chi / cfg.eta
        # Optional bed Navier-slip: precompute the rock-only penalty and the
        # (extruded) bed-surface unit normal.  Built only when requested so the
        # default no-slip path keeps its exact trajectory and memory footprint.
        self.bed_slip = cfg.bed_slip
        if cfg.bed_slip is not None:
            if not (0.0 <= float(cfg.bed_slip) <= 1.0):
                raise ValueError("bed_slip must be in [0, 1]")
            self.pen_rock = cfg.dt * chi_rock / cfg.eta
            yb = self.ybed[:, 0, :]
            h2d = yb.get() if hasattr(yb, "get") else np.asarray(yb)
            kk = np.fft.fftfreq(cfg.n, d=1.0 / cfg.n)
            H = np.fft.fft2(h2d)
            hx = np.real(np.fft.ifft2(1j * kk[:, None] * H))
            hz = np.real(np.fft.ifft2(1j * kk[None, :] * H))
            inv = 1.0 / np.sqrt(1.0 + hx ** 2 + hz ** 2)
            n2 = (-hx * inv, inv, -hz * inv)            # (n_x, n_y, n_z) over (x, z)
            def _bc(a2):
                return xp.asarray(np.broadcast_to(
                    a2[:, None, :], (cfg.n, cfg.n, cfg.n)).copy())
            self._bn_x, self._bn_y, self._bn_z = (_bc(c) for c in n2)
        self.ring = ((self.kabs >= cfg.k_f - cfg.f_band)
                     & (self.kabs <= cfg.k_f + cfg.f_band)).astype(float)
        self.Fx = xp.zeros((cfg.n, cfg.n, cfg.n))
        self.Fy = xp.zeros((cfg.n, cfg.n, cfg.n))
        self.Fz = xp.zeros((cfg.n, cfg.n, cfg.n))

        self.u = xp.zeros((cfg.n, cfg.n, cfg.n))
        self.v = xp.zeros((cfg.n, cfg.n, cfg.n))
        self.w = xp.zeros((cfg.n, cfg.n, cfg.n))
        self.theta = self.theta_solid.copy()
        self.t = 0.0
        self.step_count = 0
        # OU backscatter state (colored-FDT, bs_tau > 0)
        self.bs_x = xp.zeros((cfg.n, cfg.n, cfg.n))
        self.bs_y = xp.zeros((cfg.n, cfg.n, cfg.n))
        self.bs_z = xp.zeros((cfg.n, cfg.n, cfg.n))
        # SGS force history for tau_mem diagnostics
        self._sgs_history: list = []
        self._sgs_times: list = []
        # cache of the SGS force applied on the most recent step() so the
        # tau_mem diagnostic can be recorded without re-invoking _sgs_force
        # (which would advance the OU state and consume RNG).
        self._last_sgs_force = None

    # ------------------------------------------------------------------ #
    def _advect(self, u, v, w, f):
        sp = self.sp
        return -(u * sp.ddx(f) + v * sp.ddy(f) + w * sp.ddz(f))

    def _forcing(self):
        sp, cfg, xp = self.sp, self.cfg, self.xp
        shape = (self.n, self.n, self.n)
        wx = sp.fft(self.rng.standard_normal(shape)) * self.ring
        wy = sp.fft(self.rng.standard_normal(shape)) * self.ring
        wz = sp.fft(self.rng.standard_normal(shape)) * self.ring
        nx, ny, nz = project3d(sp, sp.ifft(wx), sp.ifft(wy), sp.ifft(wz))
        nx = nx * self.fluid
        ny = ny * self.fluid
        nz = nz * self.fluid
        rms = float(xp.sqrt(xp.mean(nx ** 2 + ny ** 2 + nz ** 2))) + 1e-30
        nx = nx * (cfg.f_amp / rms)
        ny = ny * (cfg.f_amp / rms)
        nz = nz * (cfg.f_amp / rms)
        a = cfg.dt / cfg.f_tau
        self.Fx = (1.0 - a) * self.Fx + np.sqrt(2.0 * a) * nx
        self.Fy = (1.0 - a) * self.Fy + np.sqrt(2.0 * a) * ny
        self.Fz = (1.0 - a) * self.Fz + np.sqrt(2.0 * a) * nz
        return self.Fx, self.Fy, self.Fz

    def _strain(self, u, v, w):
        sp = self.sp
        ux, uy, uz = sp.ddx(u), sp.ddy(u), sp.ddz(u)
        vx, vy, vz = sp.ddx(v), sp.ddy(v), sp.ddz(v)
        wx, wy, wz = sp.ddx(w), sp.ddy(w), sp.ddz(w)
        s11, s22, s33 = ux, vy, wz
        s12 = 0.5 * (uy + vx)
        s13 = 0.5 * (uz + wx)
        s23 = 0.5 * (vz + wy)
        sumsq = s11 ** 2 + s22 ** 2 + s33 ** 2 + 2.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
        smag = self.xp.sqrt(2.0 * sumsq)
        return (s11, s22, s33, s12, s13, s23), sumsq, smag

    def _sgs_force(self, u, v, w):
        cfg, sp, xp = self.cfg, self.sp, self.xp
        if cfg.sgs == "none":
            z = xp.zeros_like(u)
            return z, z, z, 0.0
        delta = sp.dx
        (s11, s22, s33, s12, s13, s23), sumsq, smag = self._strain(u, v, w)
        nu_t = (cfg.cs * delta) ** 2 * smag * self.fluid
        t11, t22, t33 = 2.0 * nu_t * s11, 2.0 * nu_t * s22, 2.0 * nu_t * s33
        t12, t13, t23 = 2.0 * nu_t * s12, 2.0 * nu_t * s13, 2.0 * nu_t * s23
        mx = sp.ddx(t11) + sp.ddy(t12) + sp.ddz(t13)
        my = sp.ddx(t12) + sp.ddy(t22) + sp.ddz(t23)
        mz = sp.ddx(t13) + sp.ddy(t23) + sp.ddz(t33)
        eps = 2.0 * nu_t * sumsq                      # local SGS dissipation rate
        eps_mean = float(xp.mean(eps * self.fluid))

        if cfg.sgs == "backscatter" and cfg.backscatter > 0.0:
            amp = xp.sqrt(xp.maximum(eps, 0.0) / cfg.dt) * self.fluid
            if cfg.bs_tau > 0.0:
                # Colored-FDT: OU process with exact exponential discretization
                ex = float(np.exp(-cfg.dt / cfg.bs_tau))
                sg = float(np.sqrt(1.0 - ex**2))
                self.bs_x = ex * self.bs_x + sg * self.rng.standard_normal(u.shape)
                self.bs_y = ex * self.bs_y + sg * self.rng.standard_normal(u.shape)
                self.bs_z = ex * self.bs_z + sg * self.rng.standard_normal(u.shape)
                wx, wy, wz = self.bs_x, self.bs_y, self.bs_z
            else:
                wx = self.rng.standard_normal(u.shape)
                wy = self.rng.standard_normal(u.shape)
                wz = self.rng.standard_normal(u.shape)
            fx, fy, fz = project3d(sp, amp * wx, amp * wy, amp * wz)
            inj = float(xp.mean((fx * u + fy * v + fz * w) * self.fluid))
            if abs(inj) > 1e-30 and eps_mean > 0.0:
                scale = float(np.clip(cfg.backscatter * eps_mean / inj, -5.0, 5.0))
                mx = mx + scale * fx
                my = my + scale * fy
                mz = mz + scale * fz
        return mx, my, mz, eps_mean

    # ------------------------------------------------------------------ #
    def step(self, Utarget=None):
        cfg, sp, xp = self.cfg, self.sp, self.xp
        u, v, w, theta = self.u, self.v, self.w, self.theta
        if Utarget is None:
            Utarget = cfg.U0

        Nu = self._advect(u, v, w, u)
        Nv = self._advect(u, v, w, v)
        Nw = self._advect(u, v, w, w)
        Nt = self._advect(u, v, w, theta)
        mx, my, mz, eps_mean = self._sgs_force(u, v, w)
        self._last_sgs_force = (mx, my, mz)
        Nu = Nu + mx
        Nv = Nv + my
        Nw = Nw + mz
        if cfg.f_amp > 0.0:
            fx, fy, fz = self._forcing()
            Nu = Nu + fx
            Nv = Nv + fy
            Nw = Nw + fz
        # Boussinesq buoyancy: vertical body force proportional to temperature anomaly
        if cfg.Ri != 0.0:
            theta_ref = float(xp.mean(theta * self.fluid) / xp.mean(self.fluid))
            Nv = Nv + cfg.Ri * (theta - theta_ref) * self.fluid

        uh = self.specfilt * self.visc_u * (sp.fft(u) + cfg.dt * sp.fft(Nu) * sp.dealias)
        vh = self.specfilt * self.visc_u * (sp.fft(v) + cfg.dt * sp.fft(Nv) * sp.dealias)
        wh = self.specfilt * self.visc_u * (sp.fft(w) + cfg.dt * sp.fft(Nw) * sp.dealias)
        th = self.specfilt * self.visc_t * (sp.fft(theta) + cfg.dt * sp.fft(Nt) * sp.dealias)
        u1, v1, w1, t1 = sp.ifft(uh), sp.ifft(vh), sp.ifft(wh), sp.ifft(th)

        # implicit Brinkman penalty: velocity -> 0 in solid, theta -> theta_solid
        if self.bed_slip is None:
            u1 = u1 / (1.0 + self.pen)
            v1 = v1 / (1.0 + self.pen)
            w1 = w1 / (1.0 + self.pen)
        else:
            # No-slip baseline, then add a bed-tangential slip *correction* that
            # vanishes exactly where the rock penalty is zero (ice + fluid) and at
            # s=1 -- so those cases stay bit-for-bit the no-slip trajectory.  Only
            # the bed-tangential penalty is softened (pen_t = pen - (1-s)*pen_rock),
            # leaving the bed-normal component (along n) fully impermeable.
            s = float(self.cfg.bed_slip)
            ub = u1 / (1.0 + self.pen)
            vb = v1 / (1.0 + self.pen)
            wb = w1 / (1.0 + self.pen)
            pen_t = self.pen - (1.0 - s) * self.pen_rock
            fac = 1.0 / (1.0 + pen_t) - 1.0 / (1.0 + self.pen)
            nx_, ny_, nz_ = self._bn_x, self._bn_y, self._bn_z
            un = u1 * nx_ + v1 * ny_ + w1 * nz_
            u1 = ub + (u1 - un * nx_) * fac
            v1 = vb + (v1 - un * ny_) * fac
            w1 = wb + (w1 - un * nz_) * fac
        if cfg.thermal_wall is None:
            t1 = (t1 + self.pen * self.theta_solid) / (1.0 + self.pen)
        else:
            # delegate the ice-side thermal update to the wall-model object;
            # it preserves the rock pin (theta->1) and replaces the ice Dirichlet
            # pin with a finite-conductance flux BC.  velocity penalty unchanged.
            t1 = cfg.thermal_wall.apply(self, t1)

        u1, v1, w1 = project3d(sp, u1, v1, w1)

        # constant-mass-flux: relax mean cavity streamwise velocity toward Utarget
        umean = float((u1 * self.fluid).sum() / self.fvol)
        u1 = u1 + 0.5 * (Utarget - umean)
        u1, v1, w1 = project3d(sp, u1, v1, w1)

        self.u, self.v, self.w, self.theta = u1, v1, w1, t1
        self.t += cfg.dt
        self.step_count += 1
        return eps_mean

    def run(self, steps, ramp=0, report_every=0):
        cfg = self.cfg
        for s in range(steps):
            Ut = cfg.U0 * (min(1.0, (s + 1) / ramp) if ramp > 0 else 1.0)
            self.step(Ut)
            if report_every and s % report_every == 0:
                print(f"    step {s:6d}  umax={float(self.xp.abs(self.u).max()):.3f}  "
                      f"KE={self.kinetic_energy():.4e}  melt={self.melt_flux()[0]:.4e}")
        return self

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def speed(self):
        return self.xp.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2)

    def kinetic_energy(self):
        xp = self.xp
        return 0.5 * float(xp.mean((self.u ** 2 + self.v ** 2 + self.w ** 2) * self.fluid))

    def wake_band(self):
        """Cavity band just above the bed bumps where lee separation eddies live."""
        lo = self.ybed + 0.12
        hi = self.cfg.ice_base - 0.25
        return self.fluid & (self.Y > lo) & (self.Y < hi)

    def wake_tke(self):
        """Turbulent KE in the wake band (fluctuations about the spanwise mean
        streamwise velocity at each height)."""
        xp = self.xp
        band = self.wake_band()
        umean_xy = (self.u * self.fluid).mean(axis=2, keepdims=True)
        up = self.u - umean_xy
        tke = 0.5 * (up ** 2 + self.v ** 2 + self.w ** 2)
        return float(xp.mean(tke[band]))

    def melt_flux(self):
        """Basal-melt proxy: net upward heat flux delivered to the cold ice base,
        (v*theta - kappa d(theta)/dy), averaged over the cavity band just below the
        ice base.  More heat reaching the ice => faster basal melt."""
        sp, cfg, xp = self.sp, self.cfg, self.xp
        band = self.fluid & (self.Y > cfg.ice_base - 0.45) & (self.Y < cfg.ice_base)
        dthdy = sp.ddy(self.theta)
        qdiff = -cfg.kappa * dthdy
        qadv = self.v * self.theta
        q = qadv + qdiff
        return float(xp.mean(q[band])), band

    def heat_in_wake(self):
        """Mean heat content (theta) trapped in the wake band -- the lee heat trap."""
        band = self.wake_band()
        return float(self.xp.mean(self.theta[band]))

    def basal_drag(self):
        """Proxy for turbulent basal drag: mean wake TKE * resolved shear at the bed
        band -- the momentum the wake can extract.  Reported relative, not absolute."""
        xp = self.xp
        band = self.wake_band()
        shear = xp.abs(self.sp.ddy(self.u))
        return float(xp.mean((shear * self.speed())[band]))

    # ------------------------------------------------------------------ #
    # LES-regime diagnostics: only meaningful if the SGS actually dominates
    # ------------------------------------------------------------------ #
    def turbulence_intensity(self):
        """Ratio of fluctuation KE to total cavity KE.  Near 0 => essentially a
        laminar/mean flow (closure can't matter); near 1 => developed turbulence."""
        xp = self.xp
        f = self.fluid
        umean_xy = (self.u * f).mean(axis=2, keepdims=True)
        up = self.u - umean_xy
        fluc = 0.5 * (up ** 2 + self.v ** 2 + self.w ** 2)
        tot = 0.5 * (self.u ** 2 + self.v ** 2 + self.w ** 2)
        return float(xp.mean(fluc[f]) / (xp.mean(tot[f]) + 1e-30))

    def dissipation_breakdown(self):
        """Return (eps_molecular, eps_sgs): volume-averaged dissipation rates from
        molecular viscosity vs the Smagorinsky SGS in the cavity.  SGS >> molecular
        is the signature of a genuine under-resolved LES (closure-dominated)."""
        cfg, sp, xp = self.cfg, self.sp, self.xp
        (s11, s22, s33, s12, s13, s23), sumsq, smag = self._strain(self.u, self.v, self.w)
        eps_mol = float(xp.mean((2.0 * cfg.nu * sumsq)[self.fluid]))
        nu_t = (cfg.cs * sp.dx) ** 2 * smag
        eps_sgs = float(xp.mean((2.0 * nu_t * sumsq)[self.fluid]))
        return eps_mol, eps_sgs

    def spectrum(self):
        """Shell-averaged kinetic-energy spectrum E(k) of the cavity velocity.
        A power-law inertial range that does not decay before the grid scale is
        the marker of an under-resolved LES."""
        sp, xp = self.sp, self.xp
        uh = sp.fft(self.u * self.fluid)
        vh = sp.fft(self.v * self.fluid)
        wh = sp.fft(self.w * self.fluid)
        e = 0.5 * (xp.abs(uh) ** 2 + xp.abs(vh) ** 2 + xp.abs(wh) ** 2) / (self.n ** 6)
        kbin = xp.rint(self.kabs).astype(int)
        kmax = int(self.n // 2)
        E = xp.zeros(kmax + 1)
        for kk in range(kmax + 1):
            E[kk] = xp.sum(e[kbin == kk])
        k = xp.arange(kmax + 1)
        return k, E

    # ------------------------------------------------------------------ #
    # Direction C diagnostics: tau_mem from SGS force autocorrelation
    # ------------------------------------------------------------------ #
    def clear_sgs_history(self):
        """Reset the recorded SGS force history (call before a measurement run)."""
        self._sgs_history = []
        self._sgs_times = []

    def record_sgs_force(self):
        """Record the spatially-averaged magnitude of the SGS force applied on the
        most recent step(), for tau_mem estimation.

        Reads the cached force from the last step rather than re-invoking
        _sgs_force, so it neither advances the OU backscatter state nor consumes
        RNG -- the simulation trajectory is identical whether or not recording
        is enabled."""
        if self._last_sgs_force is None:
            return
        xp = self.xp
        mx, my, mz = self._last_sgs_force
        fmag = float(xp.mean(xp.sqrt(mx ** 2 + my ** 2 + mz ** 2) * self.fluid))
        self._sgs_history.append(fmag)
        self._sgs_times.append(self.t)

    def _sample_dt(self):
        """Physical time between consecutive recorded SGS-force samples, inferred
        from the recorded times.  record_sgs_force() is typically called only
        every RECORD_EVERY steps, so this is RECORD_EVERY * dt, not dt."""
        if len(self._sgs_times) >= 2:
            d = np.diff(np.asarray(self._sgs_times, dtype=float))
            d = d[d > 0]
            if d.size:
                return float(np.median(d))
        return float(self.cfg.dt)

    def tau_mem_from_history(self):
        """Compute decorrelation time from recorded SGS force history via FFT ACF."""
        if len(self._sgs_history) < 10:
            return 0.0
        x = np.array(self._sgs_history)
        x = x - x.mean()
        if x.std() < 1e-30:
            return 0.0
        n = len(x)
        fft = np.fft.fft(x, n=2 * n)
        acf = np.fft.ifft(fft * np.conj(fft)).real[:n]
        acf = acf / acf[0]
        sample_dt = self._sample_dt()
        # find e-folding time
        below = np.where(acf < 1.0 / np.e)[0]
        if len(below) == 0:
            return float(n * sample_dt)
        return float(below[0] * sample_dt)

    def buoyancy_frequency(self):
        """Estimate N_BV from mean vertical temperature gradient in the cavity."""
        xp = self.xp
        dthdy = self.sp.ddy(self.theta)
        grad_mean = float(xp.mean(xp.abs(dthdy) * self.fluid) / xp.mean(self.fluid))
        if self.cfg.Ri <= 0.0 or grad_mean <= 0.0:
            return 0.0
        return float(np.sqrt(self.cfg.Ri * grad_mean))
