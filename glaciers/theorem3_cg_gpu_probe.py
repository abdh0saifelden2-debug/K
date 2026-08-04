r"""Theorem 3: the Counter-Gradient Parameter C_G — GPU self-contained probe.

THEORY_CAVITY.md §5 defines the melt-mechanism diagnostic

    C_G = <F . grad(theta_bar)> / <|F| |grad(theta_bar)|>,

the magnitude-weighted alignment of the turbulent heat flux F with the mean
temperature gradient. A scalar (down-gradient) eddy diffusivity forces F =
-kappa_t grad(theta_bar), hence C_G = -1 ("K-theory by construction"); a lee
recirculation can carry heat *counter*-gradient, C_G > -1. §9 item 5 asks the
LES to *report C_G alongside R(Ri)* and check whether the melt enhancement and
the departure C_G > -1 co-locate in Ri.

The temperature equation in this cavity LES has **no** modelled SGS heat flux
(theta is advected + molecularly diffused + spectrally filtered), so the
available — and physically meaningful — flux is the *resolved* turbulent heat
flux F(x) = <u'(x) theta'(x)>_t accumulated over the measurement window, with
primes the fluctuation about the time mean at each grid point. This is the
standard counter-gradient diagnostic and reduces to -1 for pure down-gradient
transport, exactly matching the theory's K-theory limit.

The solver (CavityFlow) is byte-identical to the validated RESULT-8 probe
``direction_c_gpu_probe.py`` so C_G(Ri) is directly comparable to the RESULT-8
melt sweep (same geometry, Ri grid, forcing, n). We sweep the two closures the
theory contrasts:

* **two-clocks** (colored-FDT backscatter, bs_tau>0)  -- can it go counter-gradient?
* **Smagorinsky** (backscatter=0)                     -- the K-theory baseline.

Self-contained: no repo imports beyond numpy/cupy, so the whole file can be
pasted into one Kaggle cell.

Usage (Kaggle notebook cell, P100 GPU):
    !python theorem3_cg_gpu_probe.py
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

# --- Backend: CuPy if available, else NumPy (detected at run, not import) ---
try:
    import cupy as cp
    cp.zeros(1).sum()
    xp = cp
    BACKEND = "cupy(GPU)"
    def to_np(a):
        return cp.asnumpy(a)
except (ImportError, RuntimeError, MemoryError):
    xp = np
    BACKEND = "numpy(CPU)"
    def to_np(a):
        return np.asarray(a)


# =========================================================================== #
# Strict-JSON helper (mirrors scallop_probe._json_safe): non-finite -> None
# =========================================================================== #
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# =========================================================================== #
# Counter-gradient alignment (pure NumPy; unit-tested without a GPU)
# =========================================================================== #
def counter_gradient_alignment(flux, grad, mask):
    """Magnitude-weighted alignment C_G = <F.g>/<|F||g|> over ``mask`` cells.

    Parameters
    ----------
    flux : sequence of 3 host arrays (Fx, Fy, Fz) -- turbulent heat flux.
    grad : sequence of 3 host arrays (gx, gy, gz) -- mean-temperature gradient.
    mask : boolean host array selecting the averaging region (the fluid).

    Returns ``-1.0`` for pure down-gradient flux (F = -c*grad, c>0, the
    K-theory limit), ``+1.0`` for pure counter-gradient, ``nan`` if the region
    is empty or both fields vanish there.
    """
    fx, fy, fz = (np.asarray(f, dtype=float) for f in flux)
    gx, gy, gz = (np.asarray(g, dtype=float) for g in grad)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    dot = (fx * gx + fy * gy + fz * gz)[m]
    fmag = np.sqrt((fx * fx + fy * fy + fz * fz)[m])
    gmag = np.sqrt((gx * gx + gy * gy + gz * gz)[m])
    denom = float(np.mean(fmag * gmag))
    if denom <= 1e-300:
        return float("nan")
    return float(np.mean(dot) / denom)


# =========================================================================== #
# Spectral operators on [0, 2*pi)^3
# =========================================================================== #
class Spectral3D:
    def __init__(self, n: int):
        self.n = n
        self.L = 2.0 * np.pi
        self.dx = self.L / n
        k = xp.asarray(np.fft.fftfreq(n, d=1.0 / n))
        self.kx, self.ky, self.kz = xp.meshgrid(k, k, k, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_inv = xp.zeros_like(self.k2)
        self.k2_inv[self.k2 > 0] = 1.0 / self.k2[self.k2 > 0]
        kmax = n // 3
        ax = (xp.abs(k) <= kmax)
        self.dealias = ax[:, None, None] & ax[None, :, None] & ax[None, None, :]

    def grid(self):
        x = xp.arange(self.n) * self.dx
        return xp.meshgrid(x, x, x, indexing="ij")

    def fft(self, f):
        return xp.fft.fftn(f)

    def ifft(self, F):
        return xp.real(xp.fft.ifftn(F))

    def ddx(self, f):
        return self.ifft(1j * self.kx * self.fft(f))

    def ddy(self, f):
        return self.ifft(1j * self.ky * self.fft(f))

    def ddz(self, f):
        return self.ifft(1j * self.kz * self.fft(f))


def project3d(sp: Spectral3D, u, v, w):
    """Leray projection: remove dilatational part so div(u)=0."""
    uh, vh, wh = sp.fft(u), sp.fft(v), sp.fft(w)
    div_h = 1j * (sp.kx * uh + sp.ky * vh + sp.kz * wh)
    phi_h = -div_h * sp.k2_inv
    u = u - sp.ifft(1j * sp.kx * phi_h)
    v = v - sp.ifft(1j * sp.ky * phi_h)
    w = w - sp.ifft(1j * sp.kz * phi_h)
    return u, v, w


# =========================================================================== #
# 3D subglacial cavity solver with buoyancy + colored-FDT
# (byte-identical physics to direction_c_gpu_probe.py -- the RESULT-8 solver)
# =========================================================================== #
@dataclass
class CavityConfig:
    n: int = 64
    nu: float = 5.0e-4
    kappa: float = 5.0e-4
    eta: float = 5.0e-5
    U0: float = 1.0
    dt: float = 3.0e-4
    bed_mean: float = 0.9
    bed_amp: float = 0.55
    ice_base: float = 2.4
    interface: float = 4.0
    sgs: str = "backscatter"
    cs: float = 0.17
    backscatter: float = 0.7
    bs_tau: float = 0.0        # 0 = white-FDT; >0 = colored-FDT (OU memory)
    Ri: float = 0.0            # Richardson number (Boussinesq buoyancy)
    f_amp: float = 2.0
    k_f: float = 8.0
    f_band: float = 2.0
    f_tau: float = 0.05
    seed: int = 0


class CavityFlow:
    def __init__(self, cfg: CavityConfig):
        self.cfg = cfg
        self.sp = Spectral3D(cfg.n)
        sp = self.sp
        n = cfg.n
        self.rng = xp.random.default_rng(cfg.seed) if hasattr(xp.random, 'default_rng') \
            else np.random.default_rng(cfg.seed)

        X, Y, Z = sp.grid()
        self.X, self.Y, self.Z = X, Y, Z

        # bed + ice geometry
        ybed = cfg.bed_mean + cfg.bed_amp * (
            0.6 * xp.sin(3 * X) + 0.4 * xp.sin(5 * X + 0.7)
            + 0.5 * xp.exp(-((X - np.pi)**2) / 0.15)
        )
        self.ybed = ybed
        d = cfg.interface * sp.dx
        chi_rock = 0.5 * (1.0 + xp.tanh((ybed - Y) / d))
        chi_ice = 0.5 * (1.0 + xp.tanh((Y - cfg.ice_base) / d))
        self.chi = xp.clip(chi_rock + chi_ice, 0.0, 1.0)
        self.fluid = (self.chi < 0.5)
        self.fvol = float(self.fluid.sum())
        self.theta_solid = chi_rock * 1.0 + chi_ice * 0.0

        self.kabs = xp.sqrt(sp.k2)
        self.visc_u = xp.exp(-cfg.nu * sp.k2 * cfg.dt)
        self.visc_t = xp.exp(-cfg.kappa * sp.k2 * cfg.dt)
        kcut = n / 2.0
        self.specfilt = xp.exp(-36.0 * (self.kabs / kcut)**16)
        self.pen = cfg.dt * self.chi / cfg.eta
        self.ring = ((self.kabs >= cfg.k_f - cfg.f_band)
                     & (self.kabs <= cfg.k_f + cfg.f_band)).astype(float)

        # state
        self.u = xp.zeros((n, n, n))
        self.v = xp.zeros((n, n, n))
        self.w = xp.zeros((n, n, n))
        self.theta = self.theta_solid.copy()
        self.t = 0.0

        # forcing state (OU)
        self.Fx = xp.zeros((n, n, n))
        self.Fy = xp.zeros((n, n, n))
        self.Fz = xp.zeros((n, n, n))

        # OU backscatter state
        self.bs_x = xp.zeros((n, n, n))
        self.bs_y = xp.zeros((n, n, n))
        self.bs_z = xp.zeros((n, n, n))

    def _advect(self, u, v, w, f):
        sp = self.sp
        return -(u * sp.ddx(f) + v * sp.ddy(f) + w * sp.ddz(f))

    def _forcing(self):
        sp, cfg = self.sp, self.cfg
        n = cfg.n
        shape = (n, n, n)
        wx = sp.fft(self._randn(shape)) * self.ring
        wy = sp.fft(self._randn(shape)) * self.ring
        wz = sp.fft(self._randn(shape)) * self.ring
        nx, ny, nz = project3d(sp, sp.ifft(wx), sp.ifft(wy), sp.ifft(wz))
        nx = nx * self.fluid
        ny = ny * self.fluid
        nz = nz * self.fluid
        rms = float(xp.sqrt(xp.mean(nx**2 + ny**2 + nz**2))) + 1e-30
        nx, ny, nz = nx * (cfg.f_amp / rms), ny * (cfg.f_amp / rms), nz * (cfg.f_amp / rms)
        a = cfg.dt / cfg.f_tau
        self.Fx = (1.0 - a) * self.Fx + np.sqrt(2.0 * a) * nx
        self.Fy = (1.0 - a) * self.Fy + np.sqrt(2.0 * a) * ny
        self.Fz = (1.0 - a) * self.Fz + np.sqrt(2.0 * a) * nz
        return self.Fx, self.Fy, self.Fz

    def _randn(self, shape):
        if hasattr(self.rng, 'standard_normal'):
            return xp.asarray(self.rng.standard_normal(shape))
        return xp.random.standard_normal(shape)

    def _strain(self, u, v, w):
        sp = self.sp
        ux, uy, uz = sp.ddx(u), sp.ddy(u), sp.ddz(u)
        vx, vy, vz = sp.ddx(v), sp.ddy(v), sp.ddz(v)
        wx, wy, wz = sp.ddx(w), sp.ddy(w), sp.ddz(w)
        s11, s22, s33 = ux, vy, wz
        s12 = 0.5 * (uy + vx)
        s13 = 0.5 * (uz + wx)
        s23 = 0.5 * (vz + wy)
        sumsq = s11**2 + s22**2 + s33**2 + 2.0*(s12**2 + s13**2 + s23**2)
        smag = xp.sqrt(2.0 * sumsq)
        return sumsq, smag

    def _sgs_force(self, u, v, w):
        cfg, sp = self.cfg, self.sp
        if cfg.sgs == "none":
            z = xp.zeros_like(u)
            return z, z, z, 0.0
        delta = sp.dx
        sumsq, smag = self._strain(u, v, w)
        nu_t = (cfg.cs * delta)**2 * smag * self.fluid

        ux, uy, uz = sp.ddx(u), sp.ddy(u), sp.ddz(u)
        vx, vy, vz = sp.ddx(v), sp.ddy(v), sp.ddz(v)
        wx, wy, wz = sp.ddx(w), sp.ddy(w), sp.ddz(w)
        s11, s22, s33 = ux, vy, wz
        s12 = 0.5 * (uy + vx)
        s13 = 0.5 * (uz + wx)
        s23 = 0.5 * (vz + wy)
        t11, t22, t33 = 2*nu_t*s11, 2*nu_t*s22, 2*nu_t*s33
        t12, t13, t23 = 2*nu_t*s12, 2*nu_t*s13, 2*nu_t*s23
        mx = sp.ddx(t11) + sp.ddy(t12) + sp.ddz(t13)
        my = sp.ddx(t12) + sp.ddy(t22) + sp.ddz(t23)
        mz = sp.ddx(t13) + sp.ddy(t23) + sp.ddz(t33)
        eps = 2.0 * nu_t * sumsq
        eps_mean = float(xp.mean(eps * self.fluid))

        if cfg.sgs == "backscatter" and cfg.backscatter > 0.0:
            amp = xp.sqrt(xp.maximum(eps, 0.0) / cfg.dt) * self.fluid
            wx_n = self._randn(u.shape)
            wy_n = self._randn(u.shape)
            wz_n = self._randn(u.shape)
            fx, fy, fz = project3d(sp, amp * wx_n, amp * wy_n, amp * wz_n)
            inj = float(xp.mean((fx * u + fy * v + fz * w) * self.fluid))
            if abs(inj) > 1e-30 and eps_mean > 0.0:
                scale = float(np.clip(cfg.backscatter * eps_mean / inj, -5.0, 5.0))
                gx, gy, gz = scale * fx, scale * fy, scale * fz
                if cfg.bs_tau > 0.0:
                    ex = float(np.exp(-cfg.dt / cfg.bs_tau))
                    sg = float(np.sqrt(1.0 - ex * ex))
                    self.bs_x = ex * self.bs_x + sg * gx
                    self.bs_y = ex * self.bs_y + sg * gy
                    self.bs_z = ex * self.bs_z + sg * gz
                    gx, gy, gz = self.bs_x, self.bs_y, self.bs_z
                mx = mx + gx
                my = my + gy
                mz = mz + gz
        return mx, my, mz, eps_mean

    def step(self):
        cfg, sp = self.cfg, self.sp
        u, v, w, theta = self.u, self.v, self.w, self.theta

        Nu = self._advect(u, v, w, u)
        Nv = self._advect(u, v, w, v)
        Nw = self._advect(u, v, w, w)
        Nt = self._advect(u, v, w, theta)
        mx, my, mz, eps_mean = self._sgs_force(u, v, w)
        Nu = Nu + mx
        Nv = Nv + my
        Nw = Nw + mz

        # Boussinesq buoyancy
        if cfg.Ri != 0.0:
            theta_ref = float(xp.mean(theta[self.fluid]))
            Nv = Nv + cfg.Ri * (theta - theta_ref) * self.fluid

        if cfg.f_amp > 0.0:
            fx, fy, fz = self._forcing()
            Nu = Nu + fx; Nv = Nv + fy; Nw = Nw + fz

        uh = self.specfilt * self.visc_u * (sp.fft(u) + cfg.dt * sp.fft(Nu) * sp.dealias)
        vh = self.specfilt * self.visc_u * (sp.fft(v) + cfg.dt * sp.fft(Nv) * sp.dealias)
        wh = self.specfilt * self.visc_u * (sp.fft(w) + cfg.dt * sp.fft(Nw) * sp.dealias)
        th = self.specfilt * self.visc_t * (sp.fft(theta) + cfg.dt * sp.fft(Nt) * sp.dealias)
        u1, v1, w1, t1 = sp.ifft(uh), sp.ifft(vh), sp.ifft(wh), sp.ifft(th)

        # Brinkman penalty
        u1 = u1 / (1.0 + self.pen)
        v1 = v1 / (1.0 + self.pen)
        w1 = w1 / (1.0 + self.pen)
        t1 = (t1 + self.pen * self.theta_solid) / (1.0 + self.pen)

        u1, v1, w1 = project3d(sp, u1, v1, w1)

        # constant mass flux
        umean = float((u1 * self.fluid).sum() / self.fvol)
        u1 = u1 + 0.5 * (cfg.U0 - umean)
        u1, v1, w1 = project3d(sp, u1, v1, w1)

        self.u, self.v, self.w, self.theta = u1, v1, w1, t1
        self.t += cfg.dt
        return eps_mean

    def run(self, steps, ramp=0):
        for s in range(steps):
            if ramp > 0:
                orig_U0 = self.cfg.U0
                self.cfg.U0 = orig_U0 * min(1.0, (s + 1) / ramp)
                self.step()
                self.cfg.U0 = orig_U0
            else:
                self.step()

    def melt_flux(self):
        sp, cfg = self.sp, self.cfg
        band = self.fluid & (self.Y > cfg.ice_base - 0.45) & (self.Y < cfg.ice_base)
        dthdy = sp.ddy(self.theta)
        qdiff = -cfg.kappa * dthdy
        qadv = self.v * self.theta
        q = qadv + qdiff
        return float(xp.mean(q[band]))

    def dissipation_breakdown(self):
        cfg, sp = self.cfg, self.sp
        sumsq, smag = self._strain(self.u, self.v, self.w)
        eps_mol = float(xp.mean((2.0 * cfg.nu * sumsq)[self.fluid]))
        nu_t = (cfg.cs * sp.dx)**2 * smag
        eps_sgs = float(xp.mean((2.0 * nu_t * sumsq)[self.fluid]))
        return eps_mol, eps_sgs


# =========================================================================== #
# Counter-gradient accumulator: running time-means of theta, u_i, u_i*theta
# so the resolved flux F_i = <u_i theta> - <u_i><theta> needs no field storage.
# =========================================================================== #
class CGAccumulator:
    def __init__(self, flow: CavityFlow):
        self.flow = flow
        shp = flow.u.shape
        self.n = 0
        self.s_th = xp.zeros(shp)
        self.s_u = xp.zeros(shp)
        self.s_v = xp.zeros(shp)
        self.s_w = xp.zeros(shp)
        self.s_uth = xp.zeros(shp)
        self.s_vth = xp.zeros(shp)
        self.s_wth = xp.zeros(shp)

    def sample(self):
        f = self.flow
        self.s_th += f.theta
        self.s_u += f.u
        self.s_v += f.v
        self.s_w += f.w
        self.s_uth += f.u * f.theta
        self.s_vth += f.v * f.theta
        self.s_wth += f.w * f.theta
        self.n += 1

    def finalize(self):
        """Return (C_G_fluid, C_G_band, |F|_fluid_mean) on the host."""
        f = self.flow
        sp, cfg = f.sp, f.cfg
        if self.n == 0:
            return float("nan"), float("nan"), float("nan")
        inv = 1.0 / self.n
        th_bar = self.s_th * inv
        u_bar = self.s_u * inv
        v_bar = self.s_v * inv
        w_bar = self.s_w * inv
        # resolved turbulent heat flux F_i = <u_i theta> - <u_i><theta>
        Fx = self.s_uth * inv - u_bar * th_bar
        Fy = self.s_vth * inv - v_bar * th_bar
        Fz = self.s_wth * inv - w_bar * th_bar
        gx = sp.ddx(th_bar)
        gy = sp.ddy(th_bar)
        gz = sp.ddz(th_bar)

        fluid = to_np(f.fluid)
        band = fluid & (to_np(f.Y) > cfg.ice_base - 0.45) & (to_np(f.Y) < cfg.ice_base)
        flux = (to_np(Fx), to_np(Fy), to_np(Fz))
        grad = (to_np(gx), to_np(gy), to_np(gz))

        cg_fluid = counter_gradient_alignment(flux, grad, fluid)
        cg_band = counter_gradient_alignment(flux, grad, band)
        fmag = np.sqrt(flux[0]**2 + flux[1]**2 + flux[2]**2)
        fmag_mean = float(np.mean(fmag[fluid])) if fluid.any() else float("nan")
        return cg_fluid, cg_band, fmag_mean


# =========================================================================== #
# Probe: Ri sweep measuring C_G(Ri) for two-clocks vs Smagorinsky
# =========================================================================== #
RI_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]   # matches RESULT 8
N = 64
SPINUP = 400
MEASURE = 600
SAMPLE_EVERY = 5          # accumulate C_G statistics every N steps
BS_TAU = 0.05             # colored-FDT memory time (two-clocks closure)
NSEEDS = 3


def run_one(Ri, closure, seed):
    """closure in {'two_clocks', 'smagorinsky'}."""
    backscatter = 0.7 if closure == "two_clocks" else 0.0
    bs_tau = BS_TAU if closure == "two_clocks" else 0.0
    cfg = CavityConfig(
        n=N, nu=5.0e-4, kappa=5.0e-4,
        sgs="backscatter", cs=0.17, backscatter=backscatter,
        bs_tau=bs_tau, Ri=Ri,
        f_amp=2.0, k_f=8.0, f_band=2.0, f_tau=0.05,
        seed=seed,
    )
    flow = CavityFlow(cfg)
    flow.run(SPINUP, ramp=max(1, SPINUP // 3))

    acc = CGAccumulator(flow)
    melt_samples = []
    for s in range(MEASURE):
        flow.step()
        if s % SAMPLE_EVERY == 0:
            acc.sample()
        if s % 20 == 0:
            melt_samples.append(flow.melt_flux())

    cg_fluid, cg_band, fmag = acc.finalize()
    eps_mol, eps_sgs = flow.dissipation_breakdown()
    melt_mean = float(np.mean(melt_samples)) if melt_samples else float("nan")
    return {
        "Ri": Ri, "closure": closure, "seed": seed,
        "C_G_fluid": cg_fluid, "C_G_band": cg_band, "flux_mag": fmag,
        "melt": melt_mean,
        "eps_mol": eps_mol, "eps_sgs": eps_sgs,
        "sgs_dominance": eps_sgs / (eps_mol + 1e-30),
    }


def _stat(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def ensemble(Ri, closure):
    rows = [run_one(Ri, closure, seed=s + 42) for s in range(NSEEDS)]
    cgf_m, cgf_s = _stat([r["C_G_fluid"] for r in rows])
    cgb_m, cgb_s = _stat([r["C_G_band"] for r in rows])
    melt_m, melt_s = _stat([r["melt"] for r in rows])
    return {
        "Ri": Ri, "closure": closure,
        "C_G_fluid_mean": cgf_m, "C_G_fluid_std": cgf_s,
        "C_G_band_mean": cgb_m, "C_G_band_std": cgb_s,
        "melt_mean": melt_m, "melt_std": melt_s,
        "sgs_dominance": float(np.mean([r["sgs_dominance"] for r in rows])),
    }


def main():
    print("=== Theorem 3: Counter-Gradient Parameter C_G probe ===")
    print(f"    Backend: {BACKEND}")
    print(f"    n={N}, bs_tau={BS_TAU}, nseeds={NSEEDS}, "
          f"spinup={SPINUP}, measure={MEASURE}, sample_every={SAMPLE_EVERY}")
    print(f"    Ri sweep: {RI_SWEEP}")
    print()

    two_clocks, smag = [], []
    t0 = time.time()
    for Ri in RI_SWEEP:
        print(f"--- Ri = {Ri:.2f} ---")
        tt = time.time()
        rc = ensemble(Ri, "two_clocks")
        two_clocks.append(rc)
        print(f"  two-clocks : C_G(fluid)={rc['C_G_fluid_mean']:+.4f} "
              f"C_G(band)={rc['C_G_band_mean']:+.4f} melt={rc['melt_mean']:.4e} "
              f"D={rc['sgs_dominance']:.2f} ({time.time()-tt:.1f}s)")
        ts = time.time()
        rs = ensemble(Ri, "smagorinsky")
        smag.append(rs)
        print(f"  smagorinsky: C_G(fluid)={rs['C_G_fluid_mean']:+.4f} "
              f"C_G(band)={rs['C_G_band_mean']:+.4f} melt={rs['melt_mean']:.4e} "
              f"D={rs['sgs_dominance']:.2f} ({time.time()-ts:.1f}s)")
        Rr = (rc["melt_mean"] / rs["melt_mean"]
              if abs(rs["melt_mean"]) > 1e-30 else float("nan"))
        print(f"  R(2c/smag) = {Rr:.4f}\n")

    total = time.time() - t0
    print("=" * 78)
    print(f"{'Ri':>5} | {'C_G 2c(fl)':>10} | {'C_G 2c(bd)':>10} | "
          f"{'C_G sm(fl)':>10} | {'C_G sm(bd)':>10} | {'R(2c/sm)':>8}")
    print("-" * 78)
    R_values = []
    for rc, rs in zip(two_clocks, smag):
        Rr = (rc["melt_mean"] / rs["melt_mean"]
              if abs(rs["melt_mean"]) > 1e-30 else float("nan"))
        R_values.append(Rr)
        print(f"{rc['Ri']:5.2f} | {rc['C_G_fluid_mean']:+10.4f} | "
              f"{rc['C_G_band_mean']:+10.4f} | {rs['C_G_fluid_mean']:+10.4f} | "
              f"{rs['C_G_band_mean']:+10.4f} | {Rr:8.4f}")
    print("=" * 78)

    # Verdict: does the two-clocks closure go counter-gradient (C_G > -1)
    # anywhere, and does any departure co-locate (in Ri) with a melt hump?
    cg_2c_band = [r["C_G_band_mean"] for r in two_clocks]
    finite = [c for c in cg_2c_band if c is not None and np.isfinite(c)]
    max_dep = (max(c + 1.0 for c in finite) if finite else float("nan"))
    peak_idx = int(np.nanargmax(R_values)) if any(np.isfinite(R_values)) else 0
    has_melt_hump = (np.isfinite(R_values[peak_idx]) and R_values[peak_idx] > 1.05
                     and peak_idx not in (0, len(RI_SWEEP) - 1))
    print(f"\nmax (C_G+1) over Ri (two-clocks, band) = {max_dep:+.4f}")
    print(f"melt hump present = {has_melt_hump} "
          f"(peak R={R_values[peak_idx]:.4f} at Ri={RI_SWEEP[peak_idx]:.2f})")
    if np.isfinite(max_dep) and max_dep > 0.1 and has_melt_hump:
        verdict = "SUPPORTED: counter-gradient flux co-locates with a melt hump"
    elif np.isfinite(max_dep) and max_dep > 0.1:
        verdict = ("PARTIAL: counter-gradient flux present but no co-located "
                   "melt hump")
    else:
        verdict = ("NULL: flux stays down-gradient (C_G ~ -1); no counter-"
                   "gradient mechanism at these parameters")
    print(f"VERDICT: {verdict}\n")

    output = _json_safe({
        "backend": BACKEND, "n": N, "bs_tau": BS_TAU, "nseeds": NSEEDS,
        "spinup": SPINUP, "measure": MEASURE, "sample_every": SAMPLE_EVERY,
        "Ri_sweep": RI_SWEEP, "R_values": R_values,
        "max_counter_gradient_departure": max_dep,
        "has_melt_hump": bool(has_melt_hump),
        "verdict": verdict, "wall_time_s": total,
        "two_clocks": two_clocks, "smagorinsky": smag,
    })
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "52_theorem3_cg.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=1, allow_nan=False)
    print(f"Results saved to {out_path}  (wall {total:.1f}s)")


if __name__ == "__main__":
    main()
