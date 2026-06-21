r"""Candidate 3 -- ice-base roughness feedback (melting reshapes the geometry).
 
Melt is faster where the ice base is thin; thinning is supposed to grow
roughness, roughness to enhance turbulence, and turbulence to enhance melt -- a
positive feedback that runs away until the flow chokes.  The pre-registered
prediction is therefore that the roughness amplitude ``sigma_h(t)`` **grows**
exponentially at early times (growth rate ``Lambda > 0``), saturates/reverses
when the roughness chokes the flow, and that Smagorinsky suppresses the growth.
 
Implementation (2-D x-y, backend-agnostic; pass ``xp=cupy`` for GPU).  Same
resolved penalised cavity as Candidates 1/2/4, but the ice base is now a
**moving boundary**: every ``N_update`` steps the per-column interfacial melt
``m(x)`` is computed (local finite difference clear of the Brinkman penalty,
exactly as Candidate 1) and the ice base is advanced,
 
    y_ice(x) <- y_ice(x) + dt_eff * m(x) / St    (melt raises the base),
 
with a light 1-2-1 smoothing to suppress grid-scale noise, after which the
penalty masks are rebuilt.  ``St`` is the Stefan number (large => slow melting).
 
Honest scope (the boundary this probe is built to expose): because the no-slip
Brinkman wall pins the interfacial flux to *conduction*, the melt is
**geometric** -- it is larger where the fluid column is thinner
(``m ~ kappa dT/H``).  Melting then raises the base fastest exactly where it is
lowest, so the feedback is **negative**: roughness *self-smooths*
(``Lambda < 0``), and -- like the mean melt in Candidate 1 -- the rate is
closure-independent.  The positive, turbulence-driven feedback the hypothesis
imagines needs the flow->wall-flux leg that lives in ``Fturb``, not in the
conductive wall gradient.  The runaway does not occur here, and this module
measures the (negative) growth rate and the melt/ice-height anti-correlation
that explains it.

Corrected hypothesis (scallop / melting instability).  The roughness-*growth*
framing above is a false analogy to eroding rock beds; an ice base is a melting
surface, so ``Lambda < 0`` (large-scale self-smoothing) is the *correct* physics.
The right question is a stability one -- does a *resolved* finite-amplitude
perturbation organise the flow (lee separation -> recirculation -> reattachment)
into a heat-transfer-enhancing scallop?  The staged go/no-go probe built on this
class (``scallop_probe.py`` / ``scallop_battery.py``, see ``REPORT_CANDIDATE3.md``)
answers yes: with a single resolved sine mode and a steady mean current, the
interfacial flux is enhanced above both conduction and a flat-wall control,
closure-independently -- a resolved-scale effect that escapes the conduction
limit which pins the *subgrid* mechanisms of Candidates 1/4.
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
 
 
def make_rough_ice_base(nx, Lx, y_ice_mean, sigma_h, k_min, k_max, seed=0):
    """Band-limited random ice-base height ``y_ice(x)`` (power ``~k^-2.5``)."""
    rng = np.random.default_rng(seed)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    kfund = 2.0 * np.pi / Lx
    m = np.abs(kx) / kfund
    weight = np.where((m >= k_min) & (m <= k_max), np.maximum(m, 1.0) ** (-1.25), 0.0)
    W = np.fft.fft(rng.standard_normal(nx))
    h = np.real(np.fft.ifft(W * weight))
    h -= h.mean()
    h *= sigma_h / (h.std() + 1e-30)
    return y_ice_mean + h
 
 
def _smooth121(arr):
    """Periodic 1-2-1 smoother on a 1-D array (suppresses grid-scale noise on
    the moving boundary)."""
    return 0.25 * (np.roll(arr, 1) + 2.0 * arr + np.roll(arr, -1))


def _thomas_solve(a, b, c, d):
    """Vectorised Thomas algorithm for many independent tridiagonal systems.

    ``a`` (sub-diagonal), ``b`` (main), ``c`` (super-diagonal) and ``d`` (RHS)
    are each shape ``(nsys, m)``; ``a[:, 0]`` and ``c[:, -1]`` are unused. One
    system per row is solved (loop is over the ``m`` nodes, not the systems).
    Returns ``x`` of shape ``(nsys, m)``."""
    m = b.shape[1]
    cp = np.empty_like(b)
    dp = np.empty_like(b)
    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = d[:, 0] / b[:, 0]
    for k in range(1, m):
        denom = b[:, k] - a[:, k] * cp[:, k - 1]
        cp[:, k] = c[:, k] / denom
        dp[:, k] = (d[:, k] - a[:, k] * dp[:, k - 1]) / denom
    x = np.empty_like(b)
    x[:, -1] = dp[:, -1]
    for k in range(m - 2, -1, -1):
        x[:, k] = dp[:, k] - cp[:, k] * x[:, k + 1]
    return x
 
 
@dataclass
class Candidate3Config:
    # grid / domain
    nx: int = 256
    ny: int = 96
    A: float = 4.0                # cavity aspect ratio Lx/Ly (Lx = A * 2*pi)
    # physics
    nu: float = 8.0e-4
    kappa: float = 8.0e-4
    eta: float = 5.0e-5
    dt: float = 4.0e-4
    # geometry: tall cavity so the penalty does not bleed into the fluid gap
    y_bed: float = 0.30
    y_ice_mean: float = 4.80     # clear of the box top so peaks never hit the clip
    interface: float = 1.5
    # ice-base roughness (band-limited red noise, RMS sigma_h_init)
    sigma_h_init: float = 0.30
    k_min: float = 2.0
    k_max: float = 8.0
    rough_seed: int = 0
    # ambient turbulence forcing (seeds the instability)
    f_amp: float = 0.6
    k_f: float = 6.0
    f_band: float = 2.0
    f_tau: float = 0.05
    # moving-boundary melt feedback
    St: float = 2.0e-4            # Stefan number (large => slow melting)
    N_update: int = 10
    n_smooth: int = 0             # 1-2-1 passes/update; 0 => pure melt feedback
    # ice-side conduction in the Stefan balance (opt-in; default OFF reproduces
    # the water-only update exactly).  Resolves a thin 1-D conduction layer in
    # the ice above each interface column: rho*L*v = q_water - q_ice, with
    # q_ice = -kappa_ice dT_ice/dn at the interface.  T_ice is pinned to the
    # melting temperature (theta=0) at the interface and to ``T_ice_cold`` at a
    # fixed englacial reservoir ``n_ice`` cells above it; it diffuses (and is
    # advected by the moving interface) between boundary updates, giving the
    # interface thermal "memory".
    ice_side: bool = False
    ice_kappa_ratio: float = 8.0  # kappa_ice / kappa_water (ice ~8x water)
    T_ice_cold: float = -1.0      # cold englacial reservoir temp (theta units)
    n_ice: int = 16               # ice conduction-layer cells (depth = n_ice*dy)
    # stratification / closure
    Ri: float = 0.0
    sgs: str = "none"             # 'none' | 'smagorinsky' | 'backscatter'
    cs: float = 0.16
    backscatter: float = 0.6
    bs_tau: float = 0.0
    seed: int = 0
 
 
class RoughFeedbackFlow:
    """Resolved penalised cavity with a *moving* rough ice base that melts and
    reshapes; measures the evolution of the roughness amplitude and the
    melt/ice-height correlation that sets the feedback sign."""
 
    def __init__(self, cfg: Candidate3Config, xp=np):
        self.cfg = cfg
        self.xp = xp
        self.Lx = cfg.A * 2.0 * np.pi
        self.Ly = 2.0 * np.pi
        self.sp = SpectralAniso2D(cfg.nx, cfg.ny, self.Lx, self.Ly, xp)
        sp = self.sp
        self.rng = _make_rng(xp, cfg.seed)
 
        self.X, self.Y = sp.grid()
 
        yb = make_rough_ice_base(cfg.nx, self.Lx, cfg.y_ice_mean, cfg.sigma_h_init,
                                 cfg.k_min, cfg.k_max, cfg.rough_seed)
        self.y_ice_x = xp.asarray(yb)                 # (nx,)
 
        self.k2 = sp.k2
        self.visc_u = xp.exp(-cfg.nu * sp.k2 * cfg.dt)
        self.visc_t = xp.exp(-cfg.kappa * sp.k2 * cfg.dt)
        # Mild spectral safety filter -- essentially unfiltered except at the
        # highest (finer-direction) modes.  Do NOT "fix" this to
        # ``min(pi/dx, pi/dy)``: on the anisotropic grid (Lx != Ly) that
        # applies the *coarse* x-direction Nyquist isotropically to ``kabs``
        # and over-smooths the well-resolved y-direction, which corrupts the
        # diagnostics (e.g. flips candidate3's feedback sign).
        kcut = min(cfg.nx, cfg.ny) / 2.0
        self.specfilt = xp.exp(-36.0 * (sp.kabs / (kcut + 1e-30)) ** 16)
        self.ring = ((sp.kabs >= cfg.k_f - cfg.f_band)
                     & (sp.kabs <= cfg.k_f + cfg.f_band)).astype(float)
        self.Fx = xp.zeros((cfg.nx, cfg.ny))
        self.Fy = xp.zeros((cfg.nx, cfg.ny))
 
        self.u = xp.zeros((cfg.nx, cfg.ny))
        self.v = xp.zeros((cfg.nx, cfg.ny))
        self.bs_x = xp.zeros((cfg.nx, cfg.ny))
        self.bs_y = xp.zeros((cfg.nx, cfg.ny))
        self.t = 0.0
        self.step_count = 0
 
        self._rebuild_geometry(seed_theta=True)
        self._init_ice_layer()
 
    # ------------------------------------------------------------------ #
    def _init_ice_layer(self):
        """Set up the 1-D ice-side conduction layer (host NumPy; the boundary
        update is host-side already).  A no-op unless ``cfg.ice_side``.

        ``T_ice[x, k]`` is the ice temperature ``k`` cells above the interface
        of column ``x`` (``k=0`` is the interface, pinned to the melting
        temperature ``theta=0``; ``k=n_ice-1`` is the fixed cold reservoir).
        Initialised to the steady linear conduction profile."""
        cfg = self.cfg
        if not cfg.ice_side:
            self.T_ice = None
            return
        n_ice = max(int(cfg.n_ice), 2)
        self.ice_dxi = float(self.sp.dy)
        self.kappa_ice = float(cfg.ice_kappa_ratio) * float(cfg.kappa)
        ramp = np.linspace(0.0, 1.0, n_ice)
        self.T_ice = (cfg.T_ice_cold * ramp)[None, :] * np.ones((cfg.nx, 1))
        self.v_int = np.zeros(cfg.nx)        # interface velocity from last update

    def _evolve_ice_layer(self, dt_eff):
        """Advance the per-column ice conduction layer over ``dt_eff`` in the
        interface-attached frame: dT/dt = kappa_ice d2T/dxi2 + v_int dT/dxi,
        with T(interface)=0 and T(reservoir)=T_ice_cold.  ``v_int>0`` (melting)
        advects cold ice toward the interface, the mechanism that raises
        ``q_ice``.

        Integrated with an *implicit* backward-Euler tridiagonal (Thomas) solve
        and sign-aware upwind advection.  This is unconditionally stable, so the
        sub-step count is an accuracy/cost choice only -- it can never diverge,
        unlike the old explicit scheme whose ``nsub`` honoured the diffusion CFL
        but ignored the advection CFL ``|v_int| h / dxi`` and so overflowed to
        NaN once ``v_int = m / St`` grew large at finite scallop amplitude."""
        T = self.T_ice
        n = T.shape[1]
        dxi, kap = self.ice_dxi, self.kappa_ice
        Tc = self.cfg.T_ice_cold
        vi = self.v_int
        if n <= 2:                           # only the two fixed boundaries
            T[:, 0] = 0.0
            T[:, -1] = Tc
            self.T_ice = T
            return
        # sub-stepping purely for temporal accuracy (stability is guaranteed)
        diff_dt = 0.25 * dxi * dxi / max(kap, 1e-30)
        vmax = float(np.max(np.abs(vi))) if vi.size else 0.0
        adv_dt = dxi / (vmax + 1e-30)
        sub_dt = 0.5 * min(diff_dt, adv_dt)
        nsub = int(np.clip(np.ceil(dt_eff / max(sub_dt, 1e-30)), 1, 500))
        h = dt_eff / nsub
        r = h * kap / (dxi * dxi)            # diffusion number (scalar)
        c = h * vi / dxi                     # advection number per column (nx,)
        cp = np.maximum(c, 0.0)[:, None]     # v_int>0: upwind from reservoir side
        cm = np.minimum(c, 0.0)[:, None]     # v_int<0: upwind from interface side
        nx, mint = T.shape[0], n - 2         # interior unknowns k = 1 .. n-2
        ones = np.ones((nx, mint))
        aL = -r + cm * ones                  # sub-diagonal
        aD = (1.0 + 2.0 * r) + (cp - cm) * ones   # main diagonal (>0, dominant)
        aU = -r - cp * ones                  # super-diagonal
        for _ in range(nsub):
            d = T[:, 1:-1].copy()
            d[:, -1] -= aU[:, -1] * Tc       # fixed reservoir BC -> RHS
            # interface BC T[:,0]=0 contributes nothing to the k=1 row
            T[:, 1:-1] = _thomas_solve(aL, aD, aU, d)
            T[:, 0] = 0.0
            T[:, -1] = Tc
        self.T_ice = T

    def _ice_loss(self):
        """Per-column conductive heat loss into the ice ``q_ice`` (same units as
        ``melt_field``): ``kappa_ice * (T_interface - T_ice[1]) / dxi`` >= 0 when
        the ice is colder than the interface (reduces the net melt)."""
        T = self.T_ice
        return self.kappa_ice * (0.0 - T[:, 1]) / self.ice_dxi
 
    # ------------------------------------------------------------------ #
    def _rebuild_geometry(self, seed_theta=False):
        """(Re)build the penalty masks / conduction profiles from the current
        ``y_ice_x``.  Called at init and after each boundary update."""
        cfg, sp, xp = self.cfg, self.sp, self.xp
        Y = self.Y
        y_ice_2d = self.y_ice_x[:, None]
        d = cfg.interface * sp.dy
        chi_rock = 0.5 * (1.0 + xp.tanh((cfg.y_bed - Y) / d))
        chi_ice = 0.5 * (1.0 + xp.tanh((Y - y_ice_2d) / d))
        self.chi = xp.clip(chi_rock + chi_ice, 0.0, 1.0)
        self.fluid = (self.chi < 0.5)
        self.fvol = float(self.fluid.sum()) + 1e-30
        self.theta_solid = chi_rock * 1.0 + chi_ice * 0.0
        self.pen = cfg.dt * self.chi / cfg.eta
        Hc = xp.maximum(y_ice_2d - cfg.y_bed, 1e-30)
        self.theta_cond = xp.clip((y_ice_2d - Y) / Hc, 0.0, 1.0)
        if seed_theta:
            self.theta = self.theta_cond.copy()
 
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
            return xp.zeros_like(u), xp.zeros_like(u)
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
        u, v, theta = self.u, self.v, self.theta
 
        Nu = self._advect(u, v, u)
        Nv = self._advect(u, v, v)
        Nt = self._advect(u, v, theta)
        mx, my = self._sgs_force(u, v)
        Nu, Nv = Nu + mx, Nv + my
        if cfg.f_amp > 0.0:
            fx, fy = self._forcing()
            Nu, Nv = Nu + fx, Nv + fy
        if cfg.Ri != 0.0:
            theta_ref = float(xp.mean(theta * self.fluid) / xp.mean(self.fluid))
            Nv = Nv + cfg.Ri * (theta - theta_ref) * self.fluid
 
        uh = self.specfilt * self.visc_u * (sp.fft(u) + cfg.dt * sp.fft(Nu) * sp.dealias)
        vh = self.specfilt * self.visc_u * (sp.fft(v) + cfg.dt * sp.fft(Nv) * sp.dealias)
        th = self.specfilt * self.visc_t * (sp.fft(theta) + cfg.dt * sp.fft(Nt) * sp.dealias)
        u1, v1, t1 = sp.ifft(uh), sp.ifft(vh), sp.ifft(th)
 
        u1 = u1 / (1.0 + self.pen)
        v1 = v1 / (1.0 + self.pen)
        t1 = (t1 + self.pen * self.theta_solid) / (1.0 + self.pen)
        u1, v1 = project2d(sp, u1, v1)
 
        self.u, self.v, self.theta = u1, v1, t1
        self.t += cfg.dt
        self.step_count += 1
 
    # ------------------------------------------------------------------ #
    def _to_host(self, arr):
        """Return ``arr`` as a host NumPy array (works for NumPy and CuPy).
 
        ``numpy.asarray(cupy_array)`` raises in CuPy >= 10, so the device-side
        diagnostics and the moving-boundary update route through this helper
        before any host-only NumPy logic (``np.where``, boolean indexing, the
        finite-difference stencil, ...)."""
        if self.xp is np:
            return np.asarray(arr)
        return self.xp.asnumpy(arr)
     
    def melt_field(self, top_skip=3, span=5):
        """Per-column interfacial melt ``m(x) = -kappa dtheta/dy`` via a local
        finite difference in the penalty-clear fluid (identical to Candidate 1).
        NaN where a column lacks enough clean fluid."""
        cfg = self.cfg
        theta = self._to_host(self.theta)
        fl = self._to_host(self.fluid).astype(bool)
        idx = np.arange(cfg.ny)
        jtop = np.where(fl, idx[None, :], -1).max(axis=1)
        cols = np.arange(cfg.nx)
        out = np.full(cfg.nx, np.nan)
        jhi = jtop - top_skip
        jlo = jhi - span
        ok = (jtop >= 0) & (jlo >= 0)
        dthdy = (theta[cols[ok], jhi[ok]] - theta[cols[ok], jlo[ok]]) / (span * self.sp.dy)
        out[ok] = -cfg.kappa * dthdy
        return out
 
    def melt_height_correlation(self):
        """Pearson correlation between per-column melt ``m(x)`` and ice-base
        height ``y_ice(x)``.  Negative => thin (low) columns melt faster =>
        melting flattens the roughness (negative geometric feedback)."""
        m = self.melt_field()
        h = self._to_host(self.y_ice_x)
        ok = np.isfinite(m)
        if ok.sum() < 4:
            return 0.0
        mm, hh = m[ok], h[ok]
        ms, hs = mm.std(), hh.std()
        if ms < 1e-30 or hs < 1e-30:
            return 0.0
        return float(np.mean((mm - mm.mean()) * (hh - hh.mean())) / (ms * hs))
 
    def update_boundary(self, m=None):
        """Advance the moving ice base by one melt update: raise the base by
        ``dt_eff * m(x) / St`` (melt deepens the cavity), smooth, rebuild.
 
        ``m`` may be a pre-computed (e.g. time-averaged) per-column melt; if
        ``None`` the instantaneous field is used.  Time-averaging is physical:
        with ``St >> 1`` the boundary is slow relative to the turbulence, so it
        responds to the mean melt, not the instantaneous near-wall fluctuation."""
        cfg, xp = self.cfg, self.xp
        if m is None:
            m = self.melt_field()
        m = np.where(np.isfinite(m), m, np.nanmean(m[np.isfinite(m)]) if np.isfinite(m).any() else 0.0)
        dt_eff = cfg.dt * cfg.N_update
        # ice-side conduction: rho*L*v = q_water - q_ice.  ``m`` above is the
        # water-side q_water; subtract the conductive loss into the (cold) ice.
        if cfg.ice_side and self.T_ice is not None:
            self._evolve_ice_layer(dt_eff)
            self.m_ice = self._ice_loss()
            m = m - self.m_ice
            self.v_int = m / cfg.St          # interface velocity for next advect
        y = self._to_host(self.y_ice_x).astype(float) + dt_eff * m / cfg.St
        for _ in range(max(cfg.n_smooth, 0)):
            y = _smooth121(y)
        # keep the base inside the box with a little headroom for the penalty
        y = np.clip(y, cfg.y_bed + 4.0 * self.sp.dy, self.Ly - 4.0 * self.sp.dy)
        self.y_ice_x = xp.asarray(y)
        self._rebuild_geometry(seed_theta=False)
 
    def sigma_h(self):
        return float(self.xp.std(self.y_ice_x))
 
    def kinetic_energy(self):
        return 0.5 * float(self.xp.mean((self.u ** 2 + self.v ** 2) * self.fluid))
 
    # ------------------------------------------------------------------ #
    def run_feedback(self, spinup, n_steps, record_every=10):
        """Spin the flow up on the frozen base, then evolve the moving boundary
        for ``n_steps`` (updating it every ``cfg.N_update`` steps) while
        recording ``sigma_h``, mean/peak melt and the melt/height correlation."""
        cfg = self.cfg
        for _ in range(spinup):
            self.step()
        # feedback-sign diagnostics on the *initial* (post-spinup) rough base:
        # negative corr(m, y_ice) means the thin (low) columns melt faster.
        h0 = self._to_host(self.y_ice_x).astype(float).copy()
        corr_m_h0 = self.melt_height_correlation()
        hist = {"t": [], "sigma_h": [], "m_mean": [], "m_peak": [], "corr": [],
                "h0": h0, "corr_m_h0": corr_m_h0}
        for k in range(n_steps):
            self.step()
            if k % cfg.N_update == 0:
                self.update_boundary()
            if k % record_every == 0:
                m = self.melt_field()
                hist["t"].append(self.t)
                hist["sigma_h"].append(self.sigma_h())
                hist["m_mean"].append(float(np.nanmean(m)))
                hist["m_peak"].append(float(np.nanmax(m)))
                hist["corr"].append(self.melt_height_correlation())
        hist["h_final"] = self._to_host(self.y_ice_x).astype(float).copy()
        return hist
 
 
def _growth_rate(t, sigma):
    """Lambda = d(ln sigma_h)/dt from a least-squares line fit to log sigma."""
    t = np.asarray(t, dtype=float)
    s = np.asarray(sigma, dtype=float)
    ok = (s > 0) & np.isfinite(s)
    if ok.sum() < 3:
        return 0.0
    A = np.polyfit(t[ok], np.log(s[ok]), 1)
    return float(A[0])
 
 
def run_case(cfg: Candidate3Config, spinup, n_steps, record_every=10, xp=np):
    """Run one moving-boundary case; return the roughness growth rate, its
    start/end amplitude, mean melt/height correlation and the history."""
    s = RoughFeedbackFlow(cfg, xp=xp)
    hist = s.run_feedback(spinup, n_steps, record_every=record_every)
    lam = _growth_rate(hist["t"], hist["sigma_h"])
    sig = hist["sigma_h"]
    # integrated feedback sign: correlate the *net* boundary motion with the
    # initial roughness.  corr(dh, h0) < 0 means the low columns rose more, i.e.
    # the boundary flattened -> negative (smoothing) feedback.
    dh = hist["h_final"] - hist["h0"]
    h0 = hist["h0"]
    if dh.std() > 1e-30 and h0.std() > 1e-30:
        corr_dh_h0 = float(np.corrcoef(dh, h0)[0, 1])
    else:
        corr_dh_h0 = 0.0
    return {
        "Lambda": lam,
        "sigma_h0": float(sig[0]) if sig else 0.0,
        "sigma_hT": float(sig[-1]) if sig else 0.0,
        "sigma_ratio": float(sig[-1] / sig[0]) if sig and sig[0] > 0 else 0.0,
        "corr_m_h0": float(hist["corr_m_h0"]),
        "corr_dh_h0": corr_dh_h0,
        "m_mean": float(np.nanmean(hist["m_mean"])) if hist["m_mean"] else 0.0,
        "KE_mean": s.kinetic_energy(),
        "umax": float(s.xp.abs(s.u).max()),
        "hist": hist,
    }
 
 
if __name__ == "__main__":
    for closure in ("none", "smagorinsky", "backscatter"):
        cfg = Candidate3Config(nx=128, ny=64, A=4.0, sigma_h_init=0.30,
                               St=2.0e-4, sgs=closure)
        r = run_case(cfg, spinup=400, n_steps=1500, record_every=20, xp=np)
        print(f"[{closure:11s}] Lambda={r['Lambda']:+.3e}  "
              f"sigma_h {r['sigma_h0']:.4f}->{r['sigma_hT']:.4f} "
              f"(x{r['sigma_ratio']:.3f})  corr(m,h0)={r['corr_m_h0']:+.3f}  "
              f"corr(dh,h0)={r['corr_dh_h0']:+.3f}  umax={r['umax']:.3f}")
