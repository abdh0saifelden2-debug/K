r"""Forced 3D incompressible Navier-Stokes DNS (velocity form, pseudo-spectral).

The 3D analogue of ``closure/dns2d.py``.  It manufactures a *truth* turbulent
velocity field (u, v, w) for the Part-9c closure benchmark -- the genuine 3D
extension of Part 9b.

Why velocity form (not vorticity-streamfunction)?  The 2D vorticity trick is
special to 2D (scalar vorticity, no stretching).  In 3D vorticity is a vector
and the equation carries the **vortex-stretching term** omega . grad(u) -- the
term that is *identically zero in 2D* and that drives the forward energy cascade
and the strain/vorticity geometry (Constantin-Fefferman) that regularity hinges
on.  So we solve the primitive velocity equation and fold the pressure Poisson
solve into the Leray projection P (the dimension-agnostic elliptic clock):

    d_t u = P[ -(u.grad)u + f ] + nu lap(u),     div(u) = 0.

The stiff linear (viscous) part is integrated *exactly* with an integrating
factor exp(-nu k^2 dt); the projected, dealiased nonlinear+forcing part is
advanced with an integrating-factor Heun (RK2) step -- the same scheme as the 2D
code.  Forcing is a random solenoidal band at low |k| renormalised to a fixed
physical-space RMS each step, which sustains a statistically steady forward
cascade with a developed inertial range.  The state is kept as the *real*
physical velocity (re-projected to solenoidal each step), so it stays exactly
real and divergence-free to machine precision.

**Backend-agnostic:** the constructor takes ``xp`` (NumPy default; pass
``xp=cupy`` for a GPU).  A real 128^3-256^3 DNS is only tractable on a GPU; on a
CPU use n<=64 for smoke tests.  Pedagogical demonstration code, not a validated
production solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from closure.spectral3d import Spectral3D, project3d_spectral


def _make_rng(xp, seed):
    try:
        return xp.random.default_rng(seed)
    except AttributeError:  # pragma: no cover - very old cupy
        return np.random.default_rng(seed)


@dataclass
class DNS3DConfig:
    n: int = 64
    nu: float = 3.5e-3          # molecular viscosity (sets the dissipation scale)
    k_f: float = 2.5            # forcing-band centre wavenumber (low k -> forward cascade)
    f_band: float = 1.5         # forcing-band half-width in |k|
    f_amp: float = 1.2          # physical-space RMS of the per-step forcing
    dt: float = 5.0e-3
    seed: int = 0


class ForcedNS3D:
    def __init__(self, cfg: DNS3DConfig, xp=np):
        self.cfg = cfg
        self.xp = xp
        self.sp = Spectral3D(cfg.n, xp)
        self.n = cfg.n
        sp = self.sp
        # exact viscous integrating factor exp(L dt), L = -nu k^2 (mode 0 -> 1)
        self.E1 = xp.exp(-cfg.nu * sp.k2 * cfg.dt)
        kmag = sp.kmag
        self.ring = ((kmag >= cfg.k_f - cfg.f_band)
                     & (kmag <= cfg.k_f + cfg.f_band))
        self.rng = _make_rng(xp, cfg.seed)
        # real physical velocity state (small broadband solenoidal seed)
        u = 0.05 * self.rng.standard_normal((cfg.n,) * 3)
        v = 0.05 * self.rng.standard_normal((cfg.n,) * 3)
        w = 0.05 * self.rng.standard_normal((cfg.n,) * 3)
        uh, vh, wh = project3d_spectral(sp, sp.fft(u), sp.fft(v), sp.fft(w))
        self.u, self.v, self.w = sp.ifft(uh), sp.ifft(vh), sp.ifft(wh)
        self.t = 0.0

    # --- projected, dealiased nonlinear term  N = -P[(u.grad)u] (spectral) ---
    def _nonlinear(self, uh, vh, wh):
        sp = self.sp
        u, v, w = sp.ifft(uh), sp.ifft(vh), sp.ifft(wh)
        ux, uy, uz = sp.ifft(1j * sp.kx * uh), sp.ifft(1j * sp.ky * uh), sp.ifft(1j * sp.kz * uh)
        vx, vy, vz = sp.ifft(1j * sp.kx * vh), sp.ifft(1j * sp.ky * vh), sp.ifft(1j * sp.kz * vh)
        wx, wy, wz = sp.ifft(1j * sp.kx * wh), sp.ifft(1j * sp.ky * wh), sp.ifft(1j * sp.kz * wh)
        ax = u * ux + v * uy + w * uz
        ay = u * vx + v * vy + w * vz
        az = u * wx + v * wy + w * wz
        axh = sp.fft(ax) * sp.dealias
        ayh = sp.fft(ay) * sp.dealias
        azh = sp.fft(az) * sp.dealias
        axh, ayh, azh = project3d_spectral(sp, axh, ayh, azh)
        return -axh, -ayh, -azh

    def _forcing(self):
        """Random solenoidal forcing in the low-k band, normalised to a fixed
        physical-space RMS of ``f_amp`` so the injection rate is controllable."""
        sp, xp, cfg = self.sp, self.xp, self.cfg
        shape = (self.n,) * 3
        fxh = sp.fft(self.rng.standard_normal(shape)) * self.ring
        fyh = sp.fft(self.rng.standard_normal(shape)) * self.ring
        fzh = sp.fft(self.rng.standard_normal(shape)) * self.ring
        fxh, fyh, fzh = project3d_spectral(sp, fxh, fyh, fzh)
        fx, fy, fz = sp.ifft(fxh), sp.ifft(fyh), sp.ifft(fzh)
        rms = float(xp.sqrt(xp.mean(fx ** 2 + fy ** 2 + fz ** 2))) + 1e-30
        s = cfg.f_amp / rms
        return fxh * s, fyh * s, fzh * s

    def step(self):
        sp, E1 = self.sp, self.E1
        dt = self.cfg.dt
        uh, vh, wh = sp.fft(self.u), sp.fft(self.v), sp.fft(self.w)
        fxh, fyh, fzh = self._forcing()
        ax, ay, az = self._nonlinear(uh, vh, wh)
        ax, ay, az = ax + fxh, ay + fyh, az + fzh
        u1h = E1 * uh + dt * E1 * ax
        v1h = E1 * vh + dt * E1 * ay
        w1h = E1 * wh + dt * E1 * az
        bx, by, bz = self._nonlinear(u1h, v1h, w1h)
        bx, by, bz = bx + fxh, by + fyh, bz + fzh
        uh = E1 * uh + 0.5 * dt * (E1 * ax + bx)
        vh = E1 * vh + 0.5 * dt * (E1 * ay + by)
        wh = E1 * wh + 0.5 * dt * (E1 * az + bz)
        # re-project and store the *real* physical field (keeps it solenoidal and
        # exactly real -> no Hermitian-symmetry drift across many steps)
        uh, vh, wh = project3d_spectral(sp, uh, vh, wh)
        self.u, self.v, self.w = sp.ifft(uh), sp.ifft(vh), sp.ifft(wh)
        self.t += dt

    def run(self, steps: int, report_every: int = 0):
        xp = self.xp
        for s in range(steps):
            self.step()
            if report_every and s % report_every == 0:
                ke = 0.5 * float(xp.mean(self.u ** 2 + self.v ** 2 + self.w ** 2))
                print(f"  step {s:5d}  KE={ke:.4e}  umax={float(xp.abs(self.u).max()):.3f}")
        return self

    def velocity(self):
        return self.u, self.v, self.w

    def field(self, steps: int = 3000, report_every: int = 0):
        """Return a developed divergence-free 3D velocity field (u, v, w)."""
        self.run(steps, report_every=report_every)
        return self.velocity()
