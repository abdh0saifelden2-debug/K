r"""Backend-agnostic 3D pseudo-spectral toolkit for the Part-9c closure benchmark.

This is the 3D analogue of the ``Spectral2D`` / ``helmholtz`` helpers in
``compressible/ns.py`` that power the 2D Part-9b benchmark.  Everything here is
written against an array module ``xp`` (NumPy by default, CuPy on a GPU) so the
*same* code runs a small CPU smoke test and a real 128^3-256^3 DNS on a Kaggle
Tesla P100 -- exactly the pattern used by ``glaciers/subglacial/flow3d.py``.

Domain is the triply-periodic box [0, 2*pi)^3 so integer wavenumbers are used.
The Leray projector P_k = I - k k^T / |k|^2 is diagonal in Fourier space and is
the *same* nonlocal elliptic operator the two-clocks framework insists on (it is
dimension-agnostic -- REPORT_THEORY.md sec 6).  In 3D it is the only place the
pressure Poisson coupling enters the velocity equation.

Pedagogical/demonstration code -- not a validated production solver.
"""

from __future__ import annotations

import numpy as np


def to_host(a):
    """Return a NumPy view of ``a`` whether it is a NumPy or CuPy array.

    Avoids ``np.asarray(<cupy array>)`` which raises in CuPy >= 10 (the classic
    device->host trap documented in the GPU testing notes)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


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
        self.kmag = xp.sqrt(self.k2)
        # 2/3-rule dealiasing mask
        kmax = n // 3
        ax = (xp.abs(k) <= kmax)
        self.dealias = ax[:, None, None] & ax[None, :, None] & ax[None, None, :]

    # spatial grid
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

    def laplacian(self, f):
        return self.ifft(-self.k2 * self.fft(f))

    def dealias_field(self, f):
        return self.ifft(self.fft(f) * self.dealias)


# ---------------------------------------------------------------------------
# Helmholtz / Leray projection (the dimension-agnostic elliptic clock)
# ---------------------------------------------------------------------------

def helmholtz3d(sp: Spectral3D, fx, fy, fz):
    """Split (fx, fy, fz) into solenoidal (div-free) and dilatational (curl-free)
    parts.  Returns (sx, sy, sz, dx, dy, dz)."""
    fxh, fyh, fzh = sp.fft(fx), sp.fft(fy), sp.fft(fz)
    div_h = 1j * (sp.kx * fxh + sp.ky * fyh + sp.kz * fzh)
    phi_h = -div_h * sp.k2_inv                 # laplacian(phi) = div
    dx = sp.ifft(1j * sp.kx * phi_h)
    dy = sp.ifft(1j * sp.ky * phi_h)
    dz = sp.ifft(1j * sp.kz * phi_h)
    return fx - dx, fy - dy, fz - dz, dx, dy, dz


def project3d(sp: Spectral3D, fx, fy, fz):
    """Leray projection: solenoidal (divergence-free) part of (fx, fy, fz)."""
    sx, sy, sz, _, _, _ = helmholtz3d(sp, fx, fy, fz)
    return sx, sy, sz


def project3d_spectral(sp: Spectral3D, fxh, fyh, fzh):
    """Leray projection acting directly on spectral coefficients (P_k = I - kk^T/k^2)."""
    kdotf = sp.kx * fxh + sp.ky * fyh + sp.kz * fzh
    fac = kdotf * sp.k2_inv
    return (fxh - sp.kx * fac, fyh - sp.ky * fac, fzh - sp.kz * fac)


def divergence_rms3d(sp: Spectral3D, fx, fy, fz) -> float:
    """Relative RMS divergence  |div f| / |f|  (nondimensional)."""
    xp = sp.xp
    duh = 1j * (sp.kx * sp.fft(fx) + sp.ky * sp.fft(fy) + sp.kz * sp.fft(fz))
    div = sp.ifft(duh)
    rms = float(xp.sqrt(xp.mean(div ** 2)))
    scale = float(xp.sqrt(xp.mean(fx ** 2 + fy ** 2 + fz ** 2))) + 1e-30
    return rms / scale


# ---------------------------------------------------------------------------
# shell binning + sharp spectral filter
# ---------------------------------------------------------------------------

def shell_index(sp: Spectral3D):
    return sp.xp.rint(sp.kmag).astype(int)


def sharp_filter3d(sp: Spectral3D, f, kc: float):
    """Keep only modes with |k| <= kc (sharp spectral cutoff)."""
    F = sp.fft(f)
    F = sp.xp.where(sp.kmag > kc, 0.0, F)
    return sp.ifft(F)
