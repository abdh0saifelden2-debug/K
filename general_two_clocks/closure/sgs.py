"""Subgrid-scale (SGS) closure benchmark operators for Part 8b.
 
Given a divergence-free DNS velocity field (u, v) on a periodic spectral box and a
sharp cutoff wavenumber ``kc``, we form the *exact* SGS momentum force and three
models of it, then score them on three diagnostics:
 
    1. force spectrum   E_m(k)         -- does the model carry the right energy?
    2. divergence       rms(div m)     -- is the model structurally solenoidal?
    3. transfer         T(k)=Re<u*.m>  -- forward dissipation (<0) vs backscatter (>0)?
 
The exact resolved-momentum balance (pressure folded into the Leray projector P) is
 
    d_t ubar = -P[ filt((u.grad)u) ] + nu P lap(ubar) ,
 
so the exact SGS force that a closure must supply is
 
    m_true = P[ (ubar.grad)ubar - filt((u.grad)u) ] .
 
Models
------
* **Smagorinsky (K-theory):** m = P div(2 nu_t S),  nu_t=(Cs*Delta)^2 |S|, Delta=pi/kc.
  Positive-definite eddy viscosity -> purely dissipative (T(k) <= 0): no backscatter.
* **Spectrum-matched surrogate:** per-component phase randomization of m_true (the
  Part-6 SPDE move): identical E_m(k), random phases -> decorrelated transfer and
  div != 0.
* **Projected-FDT (Part 8):** split m_true into the part parallel to ubar (a
  *scale-dependent* spectral eddy viscosity nu_t(k) that is allowed to go negative =
  backscatter, reproducing the true shell transfer) plus a Leray-projected
  fluctuating remainder that fills E_m(k).  Divergence-free by construction; its
  transfer matches truth including backscatter.
"""
 
from __future__ import annotations
 
import numpy as np
 
from compressible.ns import Spectral2D, helmholtz
 
 
# ---------------------------------------------------------------------------
# spectral helpers
# ---------------------------------------------------------------------------
 
def shell_index(sp: Spectral2D) -> np.ndarray:
    return np.round(np.sqrt(sp.k2)).astype(int)
 
 
def sharp_filter(sp: Spectral2D, f: np.ndarray, kc: float) -> np.ndarray:
    """Keep only modes with |k| <= kc (sharp spectral cutoff)."""
    F = sp.fft(f)
    F[np.sqrt(sp.k2) > kc] = 0.0
    return sp.ifft(F)
 
 
def project_vec(sp: Spectral2D, fx: np.ndarray, fy: np.ndarray):
    """Leray projection of a vector field (solenoidal part)."""
    sx, sy, _, _ = helmholtz(sp, fx, fy)
    return sx, sy
 
 
def divergence_rms(sp: Spectral2D, fx: np.ndarray, fy: np.ndarray) -> float:
    d = sp.ddx(fx) + sp.ddy(fy)
    rms_d = float(np.sqrt(np.mean(d ** 2)))
    mag = float(np.sqrt(np.mean(fx ** 2 + fy ** 2))) + 1e-30
    # nondimensionalize by (field magnitude * characteristic wavenumber 1)
    return rms_d / mag
 
 
def _advection(sp: Spectral2D, u: np.ndarray, v: np.ndarray):
    """(u.grad)u componentwise, fully resolved (no dealias: we want the true
    aliasing-free product via zero-padding-free spectral derivatives on the DNS
    grid; the DNS itself is dealiased)."""
    ux, uy = sp.ddx(u), sp.ddy(u)
    vx, vy = sp.ddx(v), sp.ddy(v)
    ax = u * ux + v * uy
    ay = u * vx + v * vy
    return ax, ay
 
 
# ---------------------------------------------------------------------------
# exact SGS force
# ---------------------------------------------------------------------------
 
def exact_sgs_force(sp: Spectral2D, u: np.ndarray, v: np.ndarray, kc: float):
    """m_true = P[ (ubar.grad)ubar - filt((u.grad)u) ], truncated to |k|<=kc."""
    ub = sharp_filter(sp, u, kc)
    vb = sharp_filter(sp, v, kc)
    # full advection, then filter to resolved band
    ax, ay = _advection(sp, u, v)
    ax_f = sharp_filter(sp, ax, kc)
    ay_f = sharp_filter(sp, ay, kc)
    # resolved advection
    axb, ayb = _advection(sp, ub, vb)
    axb = sharp_filter(sp, axb, kc)
    ayb = sharp_filter(sp, ayb, kc)
    mx = axb - ax_f
    my = ayb - ay_f
    mx, my = project_vec(sp, mx, my)
    return ub, vb, mx, my
 
 
# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
 
def force_spectrum(sp: Spectral2D, mx: np.ndarray, my: np.ndarray, kc: float):
    """Shell-summed |m|^2 spectrum, returned up to kc."""
    P = np.abs(sp.fft(mx)) ** 2 + np.abs(sp.fft(my)) ** 2
    idx = shell_index(sp)
    kmax = int(kc)
    E = np.zeros(kmax + 1)
    for k in range(kmax + 1):
        E[k] = P[idx == k].sum()
    return np.arange(kmax + 1), E
 
 
def transfer_spectrum(sp: Spectral2D, ubar: np.ndarray, vbar: np.ndarray,
                      mx: np.ndarray, my: np.ndarray, kc: float):
    """T(k) = sum_shell Re( uhat* . mhat ).  >0 backscatter, <0 dissipation."""
    uh, vh = sp.fft(ubar), sp.fft(vbar)
    mxh, myh = sp.fft(mx), sp.fft(my)
    Tk_full = np.real(np.conj(uh) * mxh + np.conj(vh) * myh)
    idx = shell_index(sp)
    kmax = int(kc)
    T = np.zeros(kmax + 1)
    for k in range(kmax + 1):
        T[k] = Tk_full[idx == k].sum()
    return np.arange(kmax + 1), T
 
 
# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
 
def smagorinsky_force(sp: Spectral2D, ub: np.ndarray, vb: np.ndarray, kc: float,
                      cs: float = 0.17):
    """m = P div(2 nu_t S),  nu_t = (cs*Delta)^2 |S|,  Delta = pi/kc."""
    delta = np.pi / kc
    ux, uy = sp.ddx(ub), sp.ddy(ub)
    vx, vy = sp.ddx(vb), sp.ddy(vb)
    s11, s22, s12 = ux, vy, 0.5 * (uy + vx)
    smag = np.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2))
    nu_t = (cs * delta) ** 2 * smag
    tau11 = 2.0 * nu_t * s11
    tau22 = 2.0 * nu_t * s22
    tau12 = 2.0 * nu_t * s12
    mx = sp.ddx(tau11) + sp.ddy(tau12)
    my = sp.ddx(tau12) + sp.ddy(tau22)
    mx = sharp_filter(sp, mx, kc)
    my = sharp_filter(sp, my, kc)
    mx, my = project_vec(sp, mx, my)
    return mx, my
 
 
def surrogate_force(sp: Spectral2D, mx: np.ndarray, my: np.ndarray, kc: float,
                    seed: int = 0):
    """Phase-randomized surrogate: identical per-mode amplitude (hence identical
    E_m(k)) as m_true, but randomized phases.  Not projected -> div != 0."""
    rng = np.random.default_rng(seed)
    out = []
    for f in (mx, my):
        amp = np.abs(sp.fft(f))
        phase = np.angle(sp.fft(rng.standard_normal(f.shape)))
        g = sp.ifft(amp * np.exp(1j * phase))
        out.append(sharp_filter(sp, g, kc))
    return out[0], out[1]
 
 
def projected_fdt_force(sp: Spectral2D, ub: np.ndarray, vb: np.ndarray,
                        mx: np.ndarray, my: np.ndarray, kc: float, seed: int = 0):
    """Projected-FDT model.
 
    1. measured scale-dependent eddy viscosity nu_t(k) reproducing the true shell
       transfer:  nu_t(k) = -T_true(k) / sum_shell(|k'|^2 |uhat|^2).  Allowed to be
       negative (backscatter).  Deterministic force m_par = -nu_t(k)|k|^2 uhat is
       solenoidal (parallel to the solenoidal uhat).
    2. a Leray-projected random remainder filling the residual force spectrum
       E_m(k) - E_par(k), so the total matches E_m(k) with div = 0 by construction.
    """
    uh, vh = sp.fft(ub), sp.fft(vb)
    mxh, myh = sp.fft(mx), sp.fft(my)
    idx = shell_index(sp)
    k2 = sp.k2
    kmax = int(kc)
 
    # --- (1) measured eddy viscosity per shell ---
    Tk_full = np.real(np.conj(uh) * mxh + np.conj(vh) * myh)
    denom_full = k2 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
    nu_par_x = np.zeros_like(mxh)
    nu_par_y = np.zeros_like(myh)
    nu_t_shell = np.zeros(kmax + 1)
    for k in range(kmax + 1):
        m = idx == k
        denom = denom_full[m].sum()
        if denom <= 1e-30:
            continue
        nu_t = -Tk_full[m].sum() / denom
        nu_t_shell[k] = nu_t
        nu_par_x[m] = -nu_t * k2[m] * uh[m]
        nu_par_y[m] = -nu_t * k2[m] * vh[m]
    m_par_x = sp.ifft(nu_par_x)
    m_par_y = sp.ifft(nu_par_y)
 
    # --- (2) projected random remainder filling the residual spectrum ---
    _, E_true = force_spectrum(sp, mx, my, kc)
    _, E_par = force_spectrum(sp, m_par_x, m_par_y, kc)
    rng = np.random.default_rng(seed)
    rx = sp.fft(rng.standard_normal(ub.shape))
    ry = sp.fft(rng.standard_normal(ub.shape))
    # project the random field -> solenoidal
    rxs, rys = project_vec(sp, sp.ifft(rx), sp.ifft(ry))
    rxh, ryh = sp.fft(rxs), sp.fft(rys)
    # normalize per shell to carry exactly the residual energy
    scale_x = np.zeros_like(rxh)
    scale_y = np.zeros_like(ryh)
    for k in range(kmax + 1):
        m = idx == k
        resid = max(E_true[k] - E_par[k], 0.0)
        cur = (np.abs(rxh[m]) ** 2 + np.abs(ryh[m]) ** 2).sum()
        if cur <= 1e-30 or resid <= 0.0:
            continue
        s = np.sqrt(resid / cur)
        scale_x[m] = rxh[m] * s
        scale_y[m] = ryh[m] * s
    fbx = sp.ifft(scale_x)
    fby = sp.ifft(scale_y)
 
    mxm = sharp_filter(sp, m_par_x + fbx, kc)
    mym = sharp_filter(sp, m_par_y + fby, kc)
    mxm, mym = project_vec(sp, mxm, mym)
    return mxm, mym, nu_t_shell
