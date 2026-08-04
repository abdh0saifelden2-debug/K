"""Forced 2D incompressible Navier-Stokes DNS (vorticity-streamfunction, spectral).
 
Used only to manufacture a *truth* turbulent velocity field for the Part-8b
closure benchmark.  Vorticity form is automatically divergence-free:
 
    d(omega)/dt + J(psi, omega) = -nu k^2 omega - nu_h k^(2p) omega - mu omega + F
    laplacian(psi) = -omega,   u = d(psi)/dy,   v = -d(psi)/dx
 
Forcing F is a random, Hermitian-symmetric ring at wavenumber ``k_f`` (built as the
band-passed FFT of white noise, so it stays real); a linear drag ``mu`` saturates
the 2D inverse energy cascade; small hyperviscosity ``nu_h`` keeps the enstrophy
range clean.  Integrated with an integrating-factor RK2 (Heun) scheme so the stiff
linear part is handled exactly.
 
Pedagogical demonstration code, not a validated production solver.
"""
 
from __future__ import annotations
 
import numpy as np
 
from compressible.ns import Spectral2D
 
 
class Vorticity2D:
    def __init__(self, n: int = 256, nu: float = 2.0e-4, nu_h: float = 2.0e-17,
                 hyper_p: int = 4, mu: float = 0.02, k_f: float = 24.0,
                 f_amp: float = 2.5, seed: int = 0):
        self.sp = Spectral2D(n)
        self.n = n
        k2 = self.sp.k2
        # linear operator L(k): viscosity + hyperviscosity + drag (mode (0,0) = 0)
        self.L = -nu * k2 - nu_h * k2 ** hyper_p - mu
        self.L[0, 0] = 0.0
        # forcing ring mask in |k|
        kr = np.sqrt(k2)
        self.ring = ((kr >= k_f - 1.0) & (kr <= k_f + 1.0)).astype(float)
        self.f_amp = f_amp
        self.rng = np.random.default_rng(seed)
 
    # --- velocity from vorticity (spectral) ---
    def velocity(self, w_h: np.ndarray):
        sp = self.sp
        psi_h = w_h * sp.k2_inv          # laplacian(psi) = -omega -> psi_h = w_h/k^2
        u = sp.ifft(1j * sp.ky * psi_h)
        v = sp.ifft(-1j * sp.kx * psi_h)
        return u, v
 
    def _jacobian(self, w_h: np.ndarray) -> np.ndarray:
        """-(u.grad)omega, dealiased -> spectral tendency."""
        sp = self.sp
        u, v = self.velocity(w_h)
        wx = sp.ifft(1j * sp.kx * w_h)
        wy = sp.ifft(1j * sp.ky * w_h)
        adv = -(u * wx + v * wy)
        return sp.fft(adv) * sp.dealias
 
    def _forcing(self) -> np.ndarray:
        """Random ring-band vorticity forcing (spectral), normalized to a fixed
        *physical-space* RMS of ``f_amp`` so the injection rate is controllable."""
        white = self.rng.standard_normal((self.n, self.n))
        f_h = np.fft.fft2(white) * self.ring
        f_phys = np.real(np.fft.ifft2(f_h))
        rms = float(np.sqrt(np.mean(f_phys ** 2))) + 1e-30
        return f_h * (self.f_amp / rms)
 
    def run(self, steps: int, dt: float = 2.5e-3, spinup_report: bool = False):
        sp = self.sp
        w_h = self._forcing() * 0.0
        # seed with a little broadband vorticity
        w_h = np.fft.fft2(0.1 * self.rng.standard_normal((self.n, self.n)))
        w_h[0, 0] = 0.0
        E1 = np.exp(self.L * dt)
        for s in range(steps):
            a = self._jacobian(w_h) + self._forcing()
            w1 = E1 * w_h + dt * E1 * a
            b = self._jacobian(w1) + self._forcing()
            w_h = E1 * w_h + 0.5 * dt * (E1 * a + b)
            w_h[0, 0] = 0.0
            if spinup_report and s % max(1, steps // 10) == 0:
                u, v = self.velocity(w_h)
                ke = 0.5 * float(np.mean(u * u + v * v))
                print(f"  step {s:5d}  KE={ke:.4e}")
        return w_h
 
    def field(self, steps: int = 4000, dt: float = 2.5e-3, spinup_report: bool = False):
        """Return a developed divergence-free velocity field (u, v)."""
        w_h = self.run(steps, dt=dt, spinup_report=spinup_report)
        return self.velocity(w_h)
