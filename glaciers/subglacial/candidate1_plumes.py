r"""Candidate 1 -- intermittent convective plumes from ice-base roughness.

Bedrock/ice-base bumps protrude through the thermal boundary layer; where a
crest sticks down into warm water, a localised plume of enhanced melt forms.
The pre-registered prediction is that the *time-and-space statistics* of the
interfacial melt are intermittent at intermediate stratification ``Ri``:

* skewness ``S > 0`` and excess kurtosis ``K > 3`` (rare, intense bursts),
* peak-to-mean ratio ``> 2``, humped at intermediate ``Ri ~ 0.3-0.6``,
* Smagorinsky over-dissipates the bursts, so its peak/mean is lower than the
  unclosed / backscatter runs.

Implementation notes (2-D, x-y; backend-agnostic, pass ``xp=cupy`` for GPU):

* Same resolved anisotropic penalised cavity as Candidate 4 (so the Brinkman
  penalty does not bleed into the fluid), but the **ice base is rough**:
  ``y_ice(x) = y_ice_mean + h(x)`` with ``h`` a band-limited random surface whose
  amplitude spectrum follows the BEDMAP-like ``a_k ~ k^-1.25`` (power ``~k^-2.5``),
  normalised to an RMS roughness ``sigma_h``.
* The cavity is seeded with the (per-column) conductive temperature profile so
  the warm bed / cold ice contrast is present from ``t=0`` (otherwise the
  thermal field needs ~L^2/kappa ~ 1e5 steps to develop and buoyancy does
  nothing -- see Candidate 4 / THEORY_CAVITY.md S12).
* The melt observable is sampled just *above* the roughness crests, at the
  topmost fluid row of each column, so it tracks the wavy interface rather than
  a fixed row.

Honest scope: as documented for the Stefan prototype and Candidate 4, the
interfacial flux is conductive at the no-slip Brinkman wall; the plume signal
lives in the *spatial intermittency* of that flux over the rough base and in the
turbulent heat flux ``Fturb = <v' theta'>``, not in a change of the mean wall
gradient.  This module reports all three.
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
    """Band-limited random ice-base height ``y_ice(x)`` on a periodic line.

    The amplitude spectrum follows ``a_k ~ k^-1.25`` (power ``~k^-2.5``, the
    BEDMAP-like red-noise slope) over mode indices ``[k_min, k_max]`` (in units
    of the fundamental ``2*pi/Lx``), with random phases, normalised to RMS
    ``sigma_h``.  Returned as a length-``nx`` NumPy array (mean ``y_ice_mean``)."""
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


@dataclass
class PlumeConfig:
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
    y_ice_mean: float = 5.50      # mean ice-base height (leaves room for bumps)
    interface: float = 1.5
    # ice-base roughness (band-limited red noise, RMS sigma_h)
    sigma_h: float = 0.30
    k_min: float = 2.0
    k_max: float = 8.0
    rough_seed: int = 0
    # ambient turbulence forcing (seeds the instability)
    f_amp: float = 0.6
    k_f: float = 6.0
    f_band: float = 2.0
    f_tau: float = 0.05
    # stratification / closure
    Ri: float = 0.0
    init_conduction: bool = True
    sgs: str = "none"             # 'none' | 'smagorinsky' | 'backscatter'
    cs: float = 0.16
    backscatter: float = 0.6
    bs_tau: float = 0.0
    seed: int = 0


class PlumeFlow:
    """Resolved penalised cavity with a *rough* ice base; measures the spatial
    intermittency of the interfacial melt and the turbulent heat flux."""

    def __init__(self, cfg: PlumeConfig, xp=np):
        self.cfg = cfg
        self.xp = xp
        Lx, Ly = cfg.A * 2.0 * np.pi, 2.0 * np.pi
        self.Ly = Ly
        self.sp = SpectralAniso2D(cfg.nx, cfg.ny, Lx, Ly, xp)
        sp = self.sp
        self.rng = _make_rng(xp, cfg.seed)

        X, Y = sp.grid()
        self.X, self.Y = X, Y

        # rough ice base y_ice(x); penalty interfaces use the wavy boundary
        yb = make_rough_ice_base(cfg.nx, Lx, cfg.y_ice_mean, cfg.sigma_h,
                                 cfg.k_min, cfg.k_max, cfg.rough_seed)
        self.y_ice_x = xp.asarray(yb)                 # (nx,)
        y_ice_2d = self.y_ice_x[:, None]              # broadcast over y
        d = cfg.interface * sp.dy
        chi_rock = 0.5 * (1.0 + xp.tanh((cfg.y_bed - Y) / d))
        chi_ice = 0.5 * (1.0 + xp.tanh((Y - y_ice_2d) / d))
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

        # per-column linear conduction ramp: 1 at the bed -> 0 at the (wavy) ice
        Hc = xp.maximum(y_ice_2d - cfg.y_bed, 1e-30)
        self.theta_cond = xp.clip((y_ice_2d - Y) / Hc, 0.0, 1.0)

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
        Nu = Nu + mx
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

        u1 = u1 / (1.0 + self.pen)
        v1 = v1 / (1.0 + self.pen)
        t1 = (t1 + self.pen * self.theta_solid) / (1.0 + self.pen)
        u1, v1 = project2d(sp, u1, v1)

        self.u, self.v, self.theta = u1, v1, t1
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

    def turb_heat_flux(self):
        """Fturb = <v' theta'> over the cavity interior (means removed)."""
        ff = self.fluid
        fvol = self.fvol
        vbar = float((self.v * ff).sum() / fvol)
        tbar = float((self.theta * ff).sum() / fvol)
        return float(((self.v - vbar) * (self.theta - tbar) * ff).sum() / fvol)

    def _to_host(self, arr):
        """Return ``arr`` as a host NumPy array (works for NumPy and CuPy).
 
        ``numpy.asarray(cupy_array)`` raises in CuPy >= 10, so the device-side
        diagnostics route through this helper before any host-only NumPy logic
        (boolean fancy indexing, ``np.where``, ...)."""
        if self.xp is np:
            return np.asarray(arr)
        return self.xp.asnumpy(arr)
      
    def melt_field(self, top_skip=3, span=5):
        """Per-column interfacial melt ``m(x) = -kappa dtheta/dy`` evaluated just
        below the (wavy) ice base.  The gradient is a *local finite difference*
        over a ``span``-cell window that starts ``top_skip`` cells below the
        topmost fluid cell of each column -- i.e. at ``ice_base - top_skip*dy``,
        clear of the Brinkman penalty bleed (the spec's "ice_base + 2 dy, inside
        the fluid, not in the penalty zone").  A local FD is used deliberately:
        the global spectral ``ddy`` rings (Gibbs) against the sharp penalty
        interface and flips the gradient sign on coarse grids / large roughness.
        Returns a length-``nx`` NumPy array; columns without enough clean fluid
        are NaN and dropped by the statistics helpers."""
        cfg = self.cfg
        theta = self._to_host(self.theta)
        fl = self._to_host(self.fluid).astype(bool)
        idx = np.arange(cfg.ny)
        jtop = np.where(fl, idx[None, :], -1).max(axis=1)     # (nx,)
        cols = np.arange(cfg.nx)
        out = np.full(cfg.nx, np.nan)
        jhi = jtop - top_skip                 # upper sample (near, cold)
        jlo = jhi - span                      # lower sample (deep, warm)
        ok = (jtop >= 0) & (jlo >= 0)
        # theta warm at bed, cold at ice => theta[jhi] < theta[jlo] => melt > 0
        dthdy = (theta[cols[ok], jhi[ok]] - theta[cols[ok], jlo[ok]]) / (span * self.sp.dy)
        out[ok] = -cfg.kappa * dthdy
        return out

    def melt_mean(self, **kw):
        return float(np.nanmean(self.melt_field(**kw)))


def _moments(samples):
    """Return (mean, std, skewness, excess_kurtosis, peak/mean) of a 1-D sample
    set, robust to empty / zero-variance inputs."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return dict(mean=0.0, std=0.0, skew=0.0, kurt=0.0, peak_mean=0.0)
    mu = float(s.mean())
    sd = float(s.std())
    if sd < 1e-30:
        return dict(mean=mu, std=0.0, skew=0.0, kurt=0.0,
                    peak_mean=float(s.max() / mu) if abs(mu) > 1e-30 else 0.0)
    z = (s - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    peak_mean = float(s.max() / mu) if abs(mu) > 1e-30 else 0.0
    return dict(mean=mu, std=sd, skew=skew, kurt=kurt, peak_mean=peak_mean)


def run_case(cfg: PlumeConfig, spinup, measure, sample_every=5, xp=np):
    """Run one case; pool the per-column interfacial melt over the measurement
    window and return its intermittency statistics plus mean melt / Fturb."""
    s = PlumeFlow(cfg, xp=xp)
    s.run(spinup)
    melt_samples = []
    fturb, ke, mmean = [], [], []
    nblocks = max(measure // sample_every, 1)
    for _ in range(nblocks):
        s.run(sample_every)
        m = s.melt_field()
        melt_samples.append(m)
        mmean.append(float(np.nanmean(m)))
        fturb.append(s.turb_heat_flux())
        ke.append(s.kinetic_energy())
    pooled = np.concatenate(melt_samples) if melt_samples else np.array([])
    mom = _moments(pooled)
    return {
        "melt_mean": float(np.mean(mmean)), "melt_std": mom["std"],
        "skew": mom["skew"], "kurt": mom["kurt"], "peak_mean": mom["peak_mean"],
        "fturb_mean": float(np.mean(fturb)), "KE_mean": float(np.mean(ke)),
        "umax": float(s.xp.abs(s.u).max()),
        "sigma_h": float(s.xp.std(s.y_ice_x)),
    }


if __name__ == "__main__":
    cfg = PlumeConfig(nx=128, ny=64, A=4.0, f_amp=0.6, Ri=0.5)
    r = run_case(cfg, spinup=400, measure=600, xp=np)
    print(f"melt_mean={r['melt_mean']:.3e} skew={r['skew']:.3f} kurt={r['kurt']:.3f} "
          f"peak/mean={r['peak_mean']:.2f} Fturb={r['fturb_mean']:.3e} "
          f"umax={r['umax']:.3f} sigma_h={r['sigma_h']:.3f}")
