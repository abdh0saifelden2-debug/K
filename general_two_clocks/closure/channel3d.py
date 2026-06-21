r"""Backend-agnostic 3D **wall-bounded (channel)** toolkit for the closure benchmark.

This is the wall-bounded counterpart of ``closure/spectral3d.py`` (the triply-periodic
Part-9c toolkit).  It exists to test the *one* place the periodic-box proof of the
two-clocks closure is known to break: at a solid wall the sharp filter ``F`` and the
Leray projector ``P`` **no longer commute**, because ``P`` stops being a Fourier
multiplier (it becomes a pressure-Poisson solve with a wall boundary condition).
Part 9b/9c established the structural verdicts in periodic boxes where ``[F,P]=0``;
P1's stated open item is whether they survive walls.

Geometry: channel ``[0,Lx) x [0,Ly] x [0,Lz)`` -- Fourier (periodic) in x,z, solid
walls at ``y=0,Ly``.  Wall-normal y uses a single 2nd-order finite-difference matrix
``Dy`` (one-sided at the walls, centred in the interior).  The *same* code runs a
CPU smoke test and a high-resolution Tesla P100 (CuPy) run via the repo's NumPy/CuPy
``xp`` pattern.

Consistency is the whole game.  We build ONE first-derivative operator ``Dy`` and use
it everywhere -- in the divergence, in the gradient, and (squared) in the Laplacian.
The wall-aware Leray projector ``project_wall`` then solves the genuine discrete Hodge
problem

    L phi = div(w),   L = Dy@Dy - k_h^2 I,   (Dy phi)|_wall = w_y|_wall  (no-penetration)

per horizontal wavenumber ``(kx,kz)``.  Because ``L = DIV . GRAD`` is the exact
composition of the same operators that measure the divergence, the projected field
``P w = w - grad phi`` is divergence-free **to machine precision in the interior** and
satisfies no-penetration ``(Pw)_y = 0`` **to machine precision at the walls** -- the
real elliptic pressure constraint with the wall geometry, not the periodic multiplier
``I - kk^T/k^2`` that ignores the wall.

The contrast operator ``project_fourier`` is that wall-blind periodic multiplier; it
leaves an O(1) no-penetration violation at the walls.  ``fourier_filter`` is the
triply-periodic sharp cutoff: it shares the 3-D Fourier eigenbasis with
``project_fourier``, so ``[F_per, P_per] = 0`` to machine precision -- the clean
periodic baseline against which the channel commutator is measured.

Pedagogical/demonstration code -- not a validated production channel solver.
"""
from __future__ import annotations

import numpy as np


def to_host(a):
    """NumPy view of ``a`` whether NumPy or CuPy (avoids the device->host trap)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


class Channel3D:
    def __init__(self, nx: int, ny: int, nz: int, Lx=2.0 * np.pi, Ly=2.0,
                 Lz=2.0 * np.pi, xp=np):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.xp = xp
        # derivative wavenumbers: the Nyquist mode of an even-length real FFT
        # cannot be differentiated consistently (i*k_nyq*field is not Hermitian),
        # so it is zeroed -- the standard pseudo-spectral convention.  Used in the
        # divergence, the gradient and the wall projector so all three agree.
        kxd = np.fft.fftfreq(nx, d=Lx / nx) * 2 * np.pi
        kzd = np.fft.fftfreq(nz, d=Lz / nz) * 2 * np.pi
        if nx % 2 == 0:
            kxd[nx // 2] = 0.0
        if nz % 2 == 0:
            kzd[nz // 2] = 0.0
        self.kxd, self.kzd = xp.asarray(kxd), xp.asarray(kzd)
        self.y = xp.asarray(np.linspace(0.0, Ly, ny))
        self.dy = float(Ly / (ny - 1))
        KX, KZ = xp.meshgrid(self.kxd, self.kzd, indexing="ij")
        self.KX, self.KZ = KX, KZ                      # (nx, nz) derivative k's
        self.K2h = KX ** 2 + KZ ** 2                    # horizontal |k|^2
        # the single 2nd-order wall-normal first-derivative matrix (ny, ny)
        self.Dy = self._build_Dy()
        self.Dy2 = self.Dy @ self.Dy                    # consistent Laplacian core

    # ---- the one wall-normal first-derivative operator (used everywhere) ----
    def _build_Dy(self):
        ny, dy = self.ny, self.dy
        D = np.zeros((ny, ny))
        i = np.arange(1, ny - 1)
        D[i, i - 1] = -1.0 / (2 * dy)
        D[i, i + 1] = +1.0 / (2 * dy)
        # 2nd-order one-sided at the walls
        D[0, 0], D[0, 1], D[0, 2] = -3 / (2 * dy), 4 / (2 * dy), -1 / (2 * dy)
        D[-1, -1], D[-1, -2], D[-1, -3] = 3 / (2 * dy), -4 / (2 * dy), 1 / (2 * dy)
        return self.xp.asarray(D)

    def _apply_Dy(self, f):
        """Apply Dy along the y axis (axis 1) of an (nx, ny, nz) array."""
        return self.xp.einsum("ab,xbz->xaz", self.Dy, f)

    # ---- horizontal transforms (FFT in x,z; y is a physical index) ----
    def fft_h(self, f):
        return self.xp.fft.fft2(f, axes=(0, 2))

    def ifft_h(self, F):
        return self.xp.real(self.xp.fft.ifft2(F, axes=(0, 2)))

    def dy1(self, f):
        """d/dy via the single consistent FD operator (axis 1)."""
        return self._apply_Dy(f)

    def grid(self):
        xp = self.xp
        x = xp.arange(self.nx) * (self.Lx / self.nx)
        z = xp.arange(self.nz) * (self.Lz / self.nz)
        return xp.meshgrid(x, self.y, z, indexing="ij")

    # ---- divergence (spectral x,z; the SAME Dy in y) ----
    def divergence(self, u, v, w):
        uh, wh = self.fft_h(u), self.fft_h(w)
        dudx = self.ifft_h(1j * self.KX[:, None, :] * uh)
        dwdz = self.ifft_h(1j * self.KZ[:, None, :] * wh)
        return dudx + self._apply_Dy(v) + dwdz

    def divergence_rms(self, u, v, w) -> float:
        """Relative RMS divergence over the WHOLE field (wall rows included)."""
        xp = self.xp
        d = self.divergence(u, v, w)
        return float(xp.sqrt(xp.mean(d ** 2)) / (self._scale(u, v, w)))

    def divergence_rms_interior(self, u, v, w) -> float:
        """Relative RMS divergence over interior rows only (the rows where the
        discrete Hodge solve enforces div=0 exactly)."""
        xp = self.xp
        d = self.divergence(u, v, w)[:, 1:-1, :]
        return float(xp.sqrt(xp.mean(d ** 2)) / (self._scale(u, v, w)))

    def no_penetration_rms(self, u, v, w) -> float:
        """Relative RMS of the wall-normal velocity ON the two walls."""
        xp = self.xp
        vw = xp.concatenate([v[:, :1, :], v[:, -1:, :]], axis=1)
        return float(xp.sqrt(xp.mean(vw ** 2)) / (self._scale(u, v, w)))

    def _scale(self, u, v, w) -> float:
        xp = self.xp
        return float(xp.sqrt(xp.mean(u ** 2 + v ** 2 + w ** 2))) + 1e-30

    # ---- the wall-aware Leray projector (discrete Hodge with wall BC) ----
    def project_wall(self, u, v, w):
        """Leray projection with the channel no-penetration wall BC.

        Solves ``(Dy@Dy - k_h^2) phi = div(w)`` per horizontal wavenumber with
        ``(Dy phi)|_wall = v|_wall`` (so the projected wall-normal velocity
        vanishes on the walls), then ``P w = w - grad phi``.  Returns the
        divergence-free, no-penetration part of ``(u, v, w)``."""
        xp = self.xp
        nx, ny, nz = self.nx, self.ny, self.nz
        uh, vh, wh = self.fft_h(u), self.fft_h(v), self.fft_h(w)
        divh = (1j * self.KX[:, None, :] * uh
                + self._apply_Dy(vh)
                + 1j * self.KZ[:, None, :] * wh)
        # pack horizontal modes onto axis 0: (nmodes, ny)
        divm = xp.transpose(divh, (0, 2, 1)).reshape(nx * nz, ny)
        vm = xp.transpose(vh, (0, 2, 1)).reshape(nx * nz, ny)
        k2 = self.K2h.reshape(nx * nz)
        eye = xp.eye(ny)
        L = self.Dy2[None, :, :] - k2[:, None, None] * eye[None, :, :]  # (m, ny, ny) real
        # replace the two wall rows by the no-penetration Neumann BC (Dy phi = v)
        L[:, 0, :] = self.Dy[0, :]
        L[:, -1, :] = self.Dy[-1, :]
        rhs = divm.copy()
        rhs[:, 0] = vm[:, 0]
        rhs[:, -1] = vm[:, -1]
        # the k=0 horizontal mean is a pure-Neumann problem: pin phi[0]=0 (gauge)
        k0 = np.where(to_host(k2) <= 1e-14)[0]
        if k0.size:
            j0 = int(k0[0])
            L[j0, 0, :] = 0.0
            L[j0, 0, 0] = 1.0
            rhs[j0, 0] = 0.0
        # one real batched solve for the real and imaginary parts at once
        rr = xp.stack([rhs.real, rhs.imag], axis=-1)        # (m, ny, 2)
        sol = xp.linalg.solve(L, rr)                         # (m, ny, 2)
        phim = sol[..., 0] + 1j * sol[..., 1]
        phih = xp.transpose(phim.reshape(nx, nz, ny), (0, 2, 1))
        uph = uh - 1j * self.KX[:, None, :] * phih
        wph = wh - 1j * self.KZ[:, None, :] * phih
        vph = vh - self._apply_Dy(phih)
        return self.ifft_h(uph), self.ifft_h(vph), self.ifft_h(wph)

    def project_fourier(self, u, v, w):
        """The NAIVE triply-periodic Leray projector ``I - kk^T/k^2`` applied as
        if y were periodic (the wall is absent).  Wall-blind by construction: it
        leaves an O(1) no-penetration violation at the walls.  Shares the 3-D
        Fourier eigenbasis with ``fourier_filter`` (so they commute exactly)."""
        xp = self.xp
        KX, KY, KZ, k2inv = self._fourier3d()
        U = xp.fft.fftn(u); V = xp.fft.fftn(v); W = xp.fft.fftn(w)
        kdf = (KX * U + KY * V + KZ * W) * k2inv
        U = U - KX * kdf; V = V - KY * kdf; W = W - KZ * kdf
        re = lambda A: xp.real(xp.fft.ifftn(A))
        return re(U), re(V), re(W)

    def fourier_filter(self, f, kc_h, kc_y):
        """Triply-periodic sharp spectral cutoff (|kx|<=kc_h, |ky|<=kc_y,
        |kz|<=kc_h).  Same 3-D Fourier basis as ``project_fourier``."""
        xp = self.xp
        KX, KY, KZ, _ = self._fourier3d()
        keep = (xp.abs(KX) <= kc_h) & (xp.abs(KY) <= kc_y) & (xp.abs(KZ) <= kc_h)
        F = xp.fft.fftn(f) * keep
        return xp.real(xp.fft.ifftn(F))

    def _fourier3d(self):
        """kx,ky,kz and 1/k^2 for the naive periodic operators (cached).  Uses the
        same Nyquist-zeroed convention as the wall operators so the periodic
        projector/filter stay real and commute to machine precision."""
        xp = self.xp
        if not hasattr(self, "_f3d"):
            ky = np.fft.fftfreq(self.ny, d=self.Ly / self.ny) * 2 * np.pi
            if self.ny % 2 == 0:
                ky[self.ny // 2] = 0.0
            ky = xp.asarray(ky)
            KX, KY, KZ = xp.meshgrid(self.kxd, ky, self.kzd, indexing="ij")
            k2 = KX ** 2 + KY ** 2 + KZ ** 2
            k2inv = xp.where(k2 > 0, 1.0 / xp.where(k2 > 0, k2, 1.0), 0.0)
            self._f3d = (KX, KY, KZ, k2inv)
        return self._f3d

    # ---- the wall-respecting sharp filter F (horizontal cutoff + DCT-y) ----
    def sharp_filter(self, f, kc_h: float, ny_keep: int):
        """Low-pass: sharp horizontal Fourier cutoff |k_h|<=kc_h AND a wall-aware
        wall-normal coarsening that keeps the lowest ``ny_keep`` cosine (DCT-I)
        modes.  The DCT-I basis has zero wall-normal derivative at the walls, so
        the filter respects the no-flux wall structure (it does NOT assume
        periodicity in y -- that is exactly the property the naive projector lacks)."""
        xp = self.xp
        F = self.fft_h(f)
        F = xp.where(self.K2h[:, None, :] > kc_h ** 2, 0.0, F)
        out = self.ifft_h(F)
        return self._dct_lowpass_y(out, ny_keep)

    def _dct_lowpass_y(self, f, ny_keep):
        """Keep the lowest ny_keep DCT-I modes along y (wall-respecting filter)."""
        xp = self.xp
        ny = self.ny
        mirror = xp.concatenate([f, f[:, -2:0:-1, :]], axis=1)   # length 2(ny-1)
        C = xp.fft.rfft(mirror, axis=1)
        mask = xp.zeros(C.shape[1])
        mask[:ny_keep] = 1.0
        C = C * mask[None, :, None]
        back = xp.fft.irfft(C, n=2 * (ny - 1), axis=1)
        return back[:, :ny, :]


def random_channel_field(ch: "Channel3D", seed: int, k0: float = 2.5, nym: int = 5):
    """A deterministic, smooth, multi-scale, *divergence-laden* channel field.

    Built as random horizontal Fourier coefficients with a Gaussian spectral
    envelope (decay scale ``k0``) times the lowest ``nym`` wall-normal cosine
    profiles.  The cosines are nonzero on the walls on purpose, so the raw field
    violates both incompressibility and no-penetration -- exactly the input that
    forces the Leray projector to do real work.  NumPy ``default_rng(seed)`` makes
    every run bit-reproducible."""
    xp = ch.xp
    rng = np.random.default_rng(seed)
    nx, ny, nz = ch.nx, ch.ny, ch.nz
    env = to_host(ch.K2h)
    env = np.exp(-env / (2.0 * k0 ** 2))
    y = to_host(ch.y)
    comps = []
    for _ in range(3):
        fh = np.zeros((nx, ny, nz), dtype=complex)
        for m in range(1, nym + 1):
            prof = np.cos(m * np.pi * y / ch.Ly)
            coeff = (rng.standard_normal((nx, nz))
                     + 1j * rng.standard_normal((nx, nz))) * env / m
            fh += coeff[:, None, :] * prof[None, :, None]
        comps.append(ch.ifft_h(xp.asarray(fh)))
    return comps
