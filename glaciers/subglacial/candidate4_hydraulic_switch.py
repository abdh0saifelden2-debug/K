r"""Candidate 4 -- hydraulic switching between *filled* and *stratified* cavity
states (anisotropic, body-force driven; no mass-flux controller).
 
A wide shallow cavity can sit in two regimes:
 
* **filled**     -- the flow fills the whole gap, strong shear at the ice base,
                    the active layer reaches the ice (``H1 ~ Ly``);
* **stratified** -- the flow concentrates in a thin basal layer, the upper cavity
                    is nearly stagnant (``H1 ~ 0.2 Ly``).
 
Tidal pressure fluctuations push the system back and forth; the time-averaged
melt is hypothesised to peak where *switching* is most frequent (intermediate
``Ri`` and intermediate aspect ratio ``A``).
 
This module is the minimal realisation of that experiment.  Differences from
:class:`subglacial.flow3d.Subglacial3DFlow` (the spec's "no solver change"):
 
* **Anisotropic domain** ``Lx = A * 2*pi``, ``Ly = 2*pi`` -- a genuinely wide,
  shallow cavity (the prior probe was ~cubic, which cannot support two states).
* **No mass-flux controller.**  The ``u1 += 0.5 (Utarget - umean)`` reset that
  pins the profile is *removed*; the flow is driven only by a tidal body force
  ``f_body(t) = f0 + df sin(omega t)`` in ``+x``, so stratification is free to
  reshape the profile.
* **Active-layer diagnostic** ``H1(t)`` from the horizontally-averaged streamwise
  profile, plus a switching-event counter ``f_switch``.
 
It is 2-D (x, y); the active-layer/hydraulic state is a vertical-profile property
that 2-D captures, at a fraction of the 3-D cost.  Backend-agnostic: pass
``xp=cupy`` for the GPU sweep.
 
Honest scope: like the Stefan prototype (THEORY_CAVITY.md S12), the melt read
here is a conductive interfacial flux; whether it responds to the hydraulic
*state* depends on the flow reaching the ice base (filled regime) versus being
excluded from it (stratified regime) -- which is exactly what ``H1`` measures.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
 
 
def _make_rng(xp, seed):
    try:
        return xp.random.default_rng(seed)
    except AttributeError:  # pragma: no cover
        return np.random.default_rng(seed)
 
 
# --------------------------------------------------------------------------- #
# anisotropic 2-D spectral operators on [0, Lx) x [0, Ly)
# --------------------------------------------------------------------------- #
class SpectralAniso2D:
    """Fourier operators on a rectangular doubly-periodic box.
 
    ``nx, ny`` grid points over physical lengths ``Lx, Ly``.  Wavenumbers are the
    *physical* angular wavenumbers ``2*pi * fftfreq(n, d=dx)`` so derivatives are
    correct for ``Lx != Ly``.
    """
 
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
class HydraulicConfig:
    # grid / domain
    nx: int = 256
    ny: int = 96
    A: float = 4.0                # cavity aspect ratio Lx/H_cav (Lx = A * 2*pi)
    # physics
    nu: float = 8.0e-4
    kappa: float = 8.0e-4
    eta: float = 5.0e-5           # Brinkman permeability
    dt: float = 4.0e-4
    # geometry: thin bed/ice penalty layers, cavity fills the rest of Ly=2*pi.
    # The cavity height H_cav = y_ice - y_bed must span most of Ly so the
    # tanh penalty interface does not bleed into the (thin) fluid gap -- an
    # under-resolved gap divides the velocity by ~(1+pen) each step and kills
    # the flow.  H_cav ~ 5.7 here => ~57 fluid rows at ny=64, ~87 at ny=96.
    y_bed: float = 0.30           # warm rock-bed top
    y_ice: float = 5.98           # cold ice base (~Ly - 0.30); cavity = [y_bed, y_ice]
    interface: float = 1.5        # mask smoothing (cells, in y)
    # drive (mass-flux controller DISABLED; body force only)
    f0: float = 0.10
    df: float = 0.05
    T_tide: float = 1.0
    # ambient turbulence forcing (seeds instability; set f_amp=0 for laminar)
    f_amp: float = 0.5
    k_f: float = 6.0
    f_band: float = 2.0
    f_tau: float = 0.05
    # stratification / closure
    Ri: float = 0.0               # Boussinesq buoyancy coupling (as in flow3d)
    # Start the cavity from the linear conductive temperature profile (warm bed
    # theta=1 -> cold ice theta=0).  Without this the fluid starts cold and the
    # warm-bed signal needs ~L^2/kappa ~ 1e5 steps to diffuse in before buoyancy
    # can do anything; seeding the (unstable) conductive profile lets ring-forced
    # perturbations drive convective plumes within a tractable run.
    init_conduction: bool = True
    sgs: str = "none"             # 'none' | 'smagorinsky' | 'backscatter'
    cs: float = 0.16
    backscatter: float = 0.6
    bs_tau: float = 0.0
    seed: int = 0
 
 
class HydraulicSwitchFlow:
    def __init__(self, cfg: HydraulicConfig, xp=np):
        self.cfg = cfg
        self.xp = xp
        Lx, Ly = cfg.A * 2.0 * np.pi, 2.0 * np.pi
        self.Ly = Ly
        self.sp = SpectralAniso2D(cfg.nx, cfg.ny, Lx, Ly, xp)
        sp = self.sp
        self.rng = _make_rng(xp, cfg.seed)
 
        X, Y = sp.grid()
        self.X, self.Y = X, Y
        self.ycol = (xp.arange(cfg.ny) * sp.dy)        # y per row
 
        d = cfg.interface * sp.dy
        chi_rock = 0.5 * (1.0 + xp.tanh((cfg.y_bed - Y) / d))
        chi_ice = 0.5 * (1.0 + xp.tanh((Y - cfg.y_ice) / d))
        self.chi = xp.clip(chi_rock + chi_ice, 0.0, 1.0)
        self.fluid = (self.chi < 0.5)
        self.fvol = float(self.fluid.sum()) + 1e-30
        self.theta_solid = chi_rock * 1.0 + chi_ice * 0.0
        self.chi_ice = chi_ice
 
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
        self.pen = cfg.dt * self.chi / cfg.eta
        self.ring = ((sp.kabs >= cfg.k_f - cfg.f_band)
                     & (sp.kabs <= cfg.k_f + cfg.f_band)).astype(float)
        self.Fx = xp.zeros((cfg.nx, cfg.ny))
        self.Fy = xp.zeros((cfg.nx, cfg.ny))
 
        # linear conduction ramp: 1 at/below the bed, 0 at/above the ice base.
        Hc = max(cfg.y_ice - cfg.y_bed, 1e-30)
        self.theta_cond = xp.clip((cfg.y_ice - Y) / Hc, 0.0, 1.0)

        self.u = xp.zeros((cfg.nx, cfg.ny))
        self.v = xp.zeros((cfg.nx, cfg.ny))
        self.theta = (self.theta_cond.copy() if cfg.init_conduction
                      else self.theta_solid.copy())
        self.t = 0.0
        self.step_count = 0
        self.bs_x = xp.zeros((cfg.nx, cfg.ny))
        self.bs_y = xp.zeros((cfg.nx, cfg.ny))
 
    # ------------------------------------------------------------------ #
    def _advect(self, u, v, f):
        return -(u * self.sp.ddx(f) + v * self.sp.ddy(f))
 
    def _body_force(self):
        cfg = self.cfg
        omega = 2.0 * np.pi / cfg.T_tide
        return (cfg.f0 + cfg.df * np.sin(omega * self.t)) * self.fluid
 
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
            return z, z
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
        Nu = Nu + mx + self._body_force()
        Nv = Nv + my
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
 
        # implicit Brinkman penalty (NO mass-flux controller afterwards)
        u1 = u1 / (1.0 + self.pen)
        v1 = v1 / (1.0 + self.pen)
        t1 = (t1 + self.pen * self.theta_solid) / (1.0 + self.pen)
        u1, v1 = project2d(sp, u1, v1)
 
        self.u, self.v, self.theta = u1, v1, t1
        self.t += cfg.dt
        self.step_count += 1
 
    def run(self, steps, report_every=0):
        for s in range(steps):
            self.step()
            if report_every and s % report_every == 0:
                print(f"    step {s:6d}  t={self.t:.3f}  H1={self.active_layer_height():.3f}"
                      f"  umax={float(self.xp.abs(self.u).max()):.3f}  "
                      f"melt={self.melt_flux():.3e}")
        return self
 
    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def _to_host(self, arr):
        """Return ``arr`` as a host NumPy array (works for NumPy and CuPy).

        ``numpy.asarray(cupy_array)`` raises in CuPy >= 10, so the device-side
        diagnostics route through this helper before any host-only NumPy logic
        (boolean fancy indexing, ``np.where``, ...)."""
        if self.xp is np:
            return np.asarray(arr)
        return self.xp.asnumpy(arr)

    def umean_profile(self):
        """Horizontally-averaged streamwise velocity <u>_x(y) over the cavity."""
        num = (self.u * self.fluid).sum(axis=0)
        den = self.fluid.sum(axis=0) + 1e-30
        return self._to_host(num / den)
 
    def active_layer_height(self, threshold=0.1):
        """Height H1 of the active layer: the highest y (measured from the bed)
        where <u>_x exceeds ``threshold`` of its column max, normalised by the
        cavity height.  H1 ~ 1 => filled; H1 ~ 0.2 => thin basal layer."""
        prof = np.abs(self.umean_profile())
        fluid_rows = self._to_host(self.fluid.any(axis=0)).astype(bool)
        if not fluid_rows.any():
            return 0.0
        umax = prof[fluid_rows].max() + 1e-30
        ys = self._to_host(self.ycol)
        active = fluid_rows & (prof > threshold * umax)
        if not active.any():
            return 0.0
        y_lo, y_hi = ys[fluid_rows].min(), ys[fluid_rows].max()
        H = max(y_hi - y_lo, 1e-30)
        return float((ys[active].max() - y_lo) / H)
 
    def kinetic_energy(self):
        return 0.5 * float(self.xp.mean((self.u ** 2 + self.v ** 2) * self.fluid))

    def turb_heat_flux(self):
        """Vertical turbulent heat flux Fturb = <v' theta'> over the cavity
        interior (fluid-averaged, means removed).  Unlike the interfacial
        conductive melt -- which is pinned to molecular conduction at the no-slip
        Brinkman wall and is insensitive to the bulk flow -- this is the
        flow-dependent transport that would set the melt at a real stress-driven
        interface.  It is the observable that distinguishes the closures."""
        xp = self.xp
        ff = self.fluid
        fvol = self.fvol
        vbar = float((self.v * ff).sum() / fvol)
        tbar = float((self.theta * ff).sum() / fvol)
        return float(((self.v - vbar) * (self.theta - tbar) * ff).sum() / fvol)
 
    def melt_flux(self, band=2):
        """Mean interfacial melt flux m = <-kappa dtheta/dy> sampled in the fluid
        just below the ice base.  For each column we take the *topmost fluid row*
        (the cell nearest the ice) and average over the ``band`` fluid cells
        below it, so the sample tracks the actual fluid/ice interface rather than
        a fixed row index (robust to resolution and a wavy ice base)."""
        cfg = self.cfg
        dthdy = self._to_host(self.sp.ddy(self.theta))
        fl = self._to_host(self.fluid).astype(bool)
        idx = np.arange(cfg.ny)
        # topmost fluid row per column (-1 if the column has no fluid)
        jtop = np.where(fl, idx[None, :], -1).max(axis=1)
        cols = np.arange(cfg.nx)
        acc, cnt = 0.0, 0
        for b in range(band):
            jj = jtop - b
            ok = jj >= 0
            if not ok.any():
                continue
            m = -cfg.kappa * dthdy[cols[ok], jj[ok]]
            acc += float(m.mean())
            cnt += 1
        return acc / max(cnt, 1)
 
 
def detect_switching(H1_series, dt, filled=0.7, stratified=0.45):
    """Count filled<->stratified transitions in an H1(t) series and return the
    switching frequency f_switch = (# state flips) / total_time.

    State-based with hysteresis: a *filled* state is registered when H1 rises
    above ``filled`` and a *stratified* state when H1 drops below ``stratified``;
    values in the dead band hold the current state.  This counts genuine regime
    flips rather than every noisy excursion (the old rate-threshold counted
    sample-to-sample jitter, inflating f_switch by ~100x)."""
    H1 = np.asarray(H1_series, dtype=float)
    if H1.size < 3:
        return 0.0
    state = 0          # -1 stratified, +1 filled, 0 undetermined
    flips = 0
    for h in H1:
        if h >= filled:
            s = 1
        elif h <= stratified:
            s = -1
        else:
            s = state
        if state != 0 and s != 0 and s != state:
            flips += 1
        if s != 0:
            state = s
    total_time = (H1.size - 1) * dt
    return flips / max(total_time, 1e-30)
 
 
def run_case(cfg: HydraulicConfig, spinup, measure, sample_every=5, xp=np):
    """Run one case; return diagnostics dict including the H1(t) series and the
    time-averaged melt over the measurement window."""
    s = HydraulicSwitchFlow(cfg, xp=xp)
    s.run(spinup)
    ts, H1, KE, melt, fturb = [], [], [], [], []
    nblocks = max(measure // sample_every, 1)
    for _ in range(nblocks):
        s.run(sample_every)
        ts.append(s.t)
        H1.append(s.active_layer_height())
        KE.append(s.kinetic_energy())
        melt.append(s.melt_flux())
        fturb.append(s.turb_heat_flux())
    dt_sample = sample_every * cfg.dt
    return {
        "t": np.array(ts), "H1": np.array(H1), "KE": np.array(KE),
        "melt": np.array(melt), "melt_mean": float(np.mean(melt)),
        "fturb": np.array(fturb), "fturb_mean": float(np.mean(fturb)),
        "KE_mean": float(np.mean(KE)),
        "H1_mean": float(np.mean(H1)), "H1_std": float(np.std(H1)),
        "f_switch": detect_switching(H1, dt_sample),
        "umax": float(s.xp.abs(s.u).max()),
    }
 
 
if __name__ == "__main__":
    # tiny CPU smoke test (well-resolved cavity; ny>=64 so the penalty does not
    # bleed into the fluid gap)
    cfg = HydraulicConfig(nx=128, ny=64, A=4.0, f_amp=0.5, Ri=0.5)
    r = run_case(cfg, spinup=400, measure=600, xp=np)
    print(f"H1_mean={r['H1_mean']:.3f} H1_std={r['H1_std']:.3f} "
          f"f_switch={r['f_switch']:.3f} melt={r['melt_mean']:.3e} umax={r['umax']:.3f}")
