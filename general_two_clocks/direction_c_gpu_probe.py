r"""Direction C: Stratification Resonance Test — GPU self-contained probe.

Run on Kaggle with GPU T4 (CuPy backend). Tests whether the melt enhancement
hump in the regime equation arises from resonance between tau_mem and T_BV.

Self-contained: all solver code is embedded. No external imports needed beyond
numpy and cupy.

Usage (Kaggle notebook cell):
    !python direction_c_gpu_probe.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

# --- Backend: CuPy if available, else NumPy ---
try:
    import cupy as cp
    cp.zeros(1).sum()
    xp = cp
    BACKEND = "cupy(GPU)"
    def to_np(a): return cp.asnumpy(a)
except (ImportError, RuntimeError, MemoryError):
    xp = np
    BACKEND = "numpy(CPU)"
    def to_np(a): return np.asarray(a)

print(f"Backend: {BACKEND}")


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

        # SGS force history for tau_mem measurement
        self._sgs_history = []
        self._sgs_times = []
        # cache of the SGS force applied on the most recent step() so the
        # tau_mem diagnostic can be recorded without re-invoking _sgs_force
        # (which would advance the OU state and consume RNG).
        self._last_sgs_force = None

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

        # Smagorinsky stress divergence
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
        self._last_sgs_force = (mx, my, mz)
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
                # temporarily set U0 for ramp
                orig_U0 = self.cfg.U0
                self.cfg.U0 = orig_U0 * min(1.0, (s + 1) / ramp)
                self.step()
                self.cfg.U0 = orig_U0
            else:
                self.step()

    def record_sgs_force(self):
        """Record the magnitude of the SGS force applied on the most recent
        step().  Reads the cached force rather than re-invoking _sgs_force, so
        it neither advances the OU backscatter state nor consumes RNG -- the
        simulation trajectory is identical whether or not recording is on."""
        if self._last_sgs_force is None:
            return
        mx, my, mz = self._last_sgs_force
        fmag = float(xp.mean(xp.sqrt(mx**2 + my**2 + mz**2) * self.fluid))
        self._sgs_history.append(fmag)
        self._sgs_times.append(self.t)

    def clear_sgs_history(self):
        self._sgs_history = []
        self._sgs_times = []

    def _sample_dt(self):
        """Physical time between consecutive recorded samples, inferred from the
        recorded times.  record_sgs_force() is called only every RECORD_EVERY
        steps, so this is RECORD_EVERY * dt, not dt."""
        if len(self._sgs_times) >= 2:
            d = np.diff(np.asarray(self._sgs_times, dtype=float))
            d = d[d > 0]
            if d.size:
                return float(np.median(d))
        return float(self.cfg.dt)

    def tau_mem_from_history(self):
        if len(self._sgs_history) < 10:
            return 0.0
        h = np.array(self._sgs_history)
        h = h - h.mean()
        if np.std(h) < 1e-30:
            return 0.0
        n = len(h)
        fh = np.fft.fft(h, n=2 * n)
        acf = np.fft.ifft(fh * np.conj(fh)).real[:n]
        acf = acf / acf[0]
        dt = self._sample_dt()
        threshold = 1.0 / np.e
        crossed = np.where(acf < threshold)[0]
        if len(crossed) == 0:
            return n * dt
        idx = crossed[0]
        if idx > 0:
            return dt * (idx - 1 + (acf[idx-1] - threshold) / (acf[idx-1] - acf[idx] + 1e-30))
        return 0.0

    def buoyancy_frequency(self):
        dthdy_mean = float(xp.mean(xp.abs(self.sp.ddy(self.theta)) * self.fluid))
        N2 = self.cfg.Ri * dthdy_mean
        return float(np.sqrt(max(N2, 0.0)))

    def melt_flux(self):
        sp, cfg = self.sp, self.cfg
        band = self.fluid & (self.Y > cfg.ice_base - 0.45) & (self.Y < cfg.ice_base)
        dthdy = sp.ddy(self.theta)
        qdiff = -cfg.kappa * dthdy
        qadv = self.v * self.theta
        q = qadv + qdiff
        return float(xp.mean(q[band]))

    def kinetic_energy(self):
        return 0.5 * float(xp.mean((self.u**2 + self.v**2 + self.w**2) * self.fluid))

    def dissipation_breakdown(self):
        cfg, sp = self.cfg, self.sp
        sumsq, smag = self._strain(self.u, self.v, self.w)
        eps_mol = float(xp.mean((2.0 * cfg.nu * sumsq)[self.fluid]))
        nu_t = (cfg.cs * sp.dx)**2 * smag
        eps_sgs = float(xp.mean((2.0 * nu_t * sumsq)[self.fluid]))
        return eps_mol, eps_sgs


# =========================================================================== #
# Probe: Ri sweep
# =========================================================================== #
RI_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
N = 64                    # grid (64^3 on GPU is fast)
SPINUP = 400              # steps to develop turbulence
MEASURE = 600             # steps for measurement
RECORD_EVERY = 3          # record SGS force every N steps
BS_TAU = 0.05             # colored-FDT memory time
NSEEDS = 3                # ensemble size


def run_one(Ri, bs_tau, seed):
    cfg = CavityConfig(
        n=N, nu=5.0e-4, kappa=5.0e-4,
        sgs="backscatter", cs=0.17, backscatter=0.7,
        bs_tau=bs_tau, Ri=Ri,
        f_amp=2.0, k_f=8.0, f_band=2.0, f_tau=0.05,
        seed=seed,
    )
    flow = CavityFlow(cfg)
    flow.run(SPINUP, ramp=max(1, SPINUP // 3))

    flow.clear_sgs_history()
    melt_samples = []
    for s in range(MEASURE):
        flow.step()
        if s % RECORD_EVERY == 0:
            flow.record_sgs_force()
        if s % 20 == 0:
            melt_samples.append(flow.melt_flux())

    tau_mem = flow.tau_mem_from_history()
    N_BV = flow.buoyancy_frequency()
    melt_mean = float(np.mean(melt_samples)) if melt_samples else 0.0
    eps_mol, eps_sgs = flow.dissipation_breakdown()

    return {
        "Ri": Ri, "bs_tau_set": bs_tau, "seed": seed,
        "melt": melt_mean,
        "tau_mem": tau_mem, "N_BV": N_BV,
        "tau_x_N": tau_mem * N_BV,
        "eps_mol": eps_mol, "eps_sgs": eps_sgs,
        "sgs_dominance": eps_sgs / (eps_mol + 1e-30),
    }


def ensemble(Ri, bs_tau):
    results = [run_one(Ri, bs_tau, seed=s + 42) for s in range(NSEEDS)]
    melts = [r["melt"] for r in results]
    taus = [r["tau_mem"] for r in results]
    nbvs = [r["N_BV"] for r in results]
    prods = [r["tau_x_N"] for r in results]
    return {
        "Ri": Ri, "bs_tau_set": bs_tau,
        "melt_mean": float(np.mean(melts)),
        "melt_std": float(np.std(melts)),
        "tau_mem_mean": float(np.mean(taus)),
        "tau_mem_std": float(np.std(taus)),
        "N_BV_mean": float(np.mean(nbvs)),
        "product_mean": float(np.mean(prods)),
        "product_std": float(np.std(prods)),
        "sgs_dominance": float(np.mean([r["sgs_dominance"] for r in results])),
    }


def main():
    print("=== Direction C: Stratification Resonance Probe ===")
    print(f"    Backend: {BACKEND}")
    print(f"    n={N}, bs_tau={BS_TAU}, nseeds={NSEEDS}")
    print(f"    spinup={SPINUP}, measure={MEASURE}")
    print(f"    Ri sweep: {RI_SWEEP}")
    print()

    all_white = []
    all_colored = []
    t0 = time.time()

    for Ri in RI_SWEEP:
        print(f"--- Ri = {Ri:.2f} ---")
        tw = time.time()
        rw = ensemble(Ri, 0.0)
        print(f"  white-FDT: melt={rw['melt_mean']:.4e} "
              f"({time.time()-tw:.1f}s) sgs_dom={rw['sgs_dominance']:.2f}")
        all_white.append(rw)

        tc = time.time()
        rc = ensemble(Ri, BS_TAU)
        print(f"  colored-FDT: melt={rc['melt_mean']:.4e} "
              f"({time.time()-tc:.1f}s) tau_mem={rc['tau_mem_mean']:.4e} "
              f"N_BV={rc['N_BV_mean']:.4f} product={rc['product_mean']:.3f}")
        all_colored.append(rc)

        R = rc["melt_mean"] / rw["melt_mean"] if abs(rw["melt_mean"]) > 1e-30 else float("nan")
        print(f"  R(colored/white) = {R:.4f}")
        print()

    total = time.time() - t0
    print(f"\n{'='*90}")
    print(f"Total wall time: {total:.1f}s")
    print(f"{'='*90}\n")

    # Summary table
    print(f"{'Ri':>5} | {'melt_w':>10} | {'melt_c':>10} | {'R':>7} | "
          f"{'tau_mem':>8} | {'N_BV':>6} | {'tau*N':>7} | {'sgs_dom':>7} | {'~2pi?':>5}")
    print("-" * 90)

    R_values = []
    for rw, rc in zip(all_white, all_colored):
        Ri = rw["Ri"]
        R = rc["melt_mean"] / rw["melt_mean"] if abs(rw["melt_mean"]) > 1e-30 else 1.0
        R_values.append(R)
        near = "YES" if abs(rc["product_mean"] - 2*np.pi) < 2.0 else "no"
        print(f"{Ri:5.2f} | {rw['melt_mean']:10.4e} | {rc['melt_mean']:10.4e} | "
              f"{R:7.4f} | {rc['tau_mem_mean']:8.4e} | {rc['N_BV_mean']:6.4f} | "
              f"{rc['product_mean']:7.3f} | {rc['sgs_dominance']:7.2f} | {near:>5}")

    print(f"\n{'='*90}")

    # Interpretation
    peak_idx = int(np.argmax(R_values))
    peak_Ri = RI_SWEEP[peak_idx]
    peak_R = R_values[peak_idx]
    has_hump = peak_R > 1.05 and peak_idx not in [0, len(RI_SWEEP) - 1]

    print(f"\nPeak R = {peak_R:.4f} at Ri = {peak_Ri:.2f}")
    if has_hump:
        prod_at_peak = all_colored[peak_idx]["product_mean"]
        print(f"tau_mem * N_BV at peak = {prod_at_peak:.3f} (resonance: {2*np.pi:.3f})")
        if abs(prod_at_peak - 2*np.pi) < 2.0:
            print("VERDICT: HUMP + RESONANCE MATCH => mechanism verified")
        else:
            print("VERDICT: HUMP but resonance condition NOT matched")
    else:
        if peak_R < 1.02:
            print("VERDICT: NULL — no stratification resonance from memory")
        else:
            print("VERDICT: Weak effect — inconclusive at this resolution/parameter")

    # Check if SGS dominates
    mean_sgs_dom = float(np.mean([r["sgs_dominance"] for r in all_colored]))
    if mean_sgs_dom < 1.0:
        print(f"\nWARNING: SGS dominance = {mean_sgs_dom:.2f} < 1.0")
        print("The closure is NOT dominant at this resolution.")
        print("Results may not reflect the regime where the hump lives.")
        print("Consider: lower nu, higher n, or stronger forcing.")

    # Save results
    output = {
        "backend": BACKEND, "n": N, "bs_tau": BS_TAU, "nseeds": NSEEDS,
        "spinup": SPINUP, "measure": MEASURE,
        "Ri_sweep": RI_SWEEP, "R_values": R_values,
        "peak_Ri": peak_Ri, "peak_R": peak_R, "has_hump": has_hump,
        "mean_sgs_dominance": mean_sgs_dom,
        "wall_time_s": total,
        "white": all_white, "colored": all_colored,
    }
    import os
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = os.path.join(out_dir, "direction_c_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
