r"""3D subgrid-scale (SGS) closure operators + 3D-specific diagnostics (Part 9c).

The genuine 3D extension of ``closure/sgs.py`` (Part 9b).  Given a divergence-free
3D DNS velocity field (u, v, w) on a periodic spectral box and a sharp cutoff
wavenumber ``kc``, we form the *exact* SGS momentum force and three models, then
score them on the same three structural diagnostics as 2D **plus** the diagnostics
that only exist in 3D:

Shared with 2D (Part 9b)
    1. force spectrum    E_m(k)
    2. divergence        rms(div m)
    3. transfer          T(k) = Re<u*.m>     (<0 forward dissipation, >0 backscatter)

New in 3D -- the physics the 2D test structurally cannot see
    4. vortex stretching  <omega_i S_ij omega_j>  (identically 0 in 2D; the engine
       of the forward cascade and of singularity formation).
    5. strain/vorticity alignment  -- the Constantin-Fefferman geometric-depletion
       geometry: vorticity preferentially aligns with the *intermediate* strain
       eigenvector.  No 2D analogue (2D vorticity is orthogonal to the plane).
    6. SGS backscatter volume fraction  -- fraction of space where the local SGS
       energy flux Pi = -tau^d_ij S_ij < 0 (energy flowing up-scale).  Smagorinsky
       gives *exactly zero* (positive-definite eddy viscosity); real 3D turbulence
       has a large backscatter fraction; projected-FDT reproduces it.

The exact resolved-momentum balance (pressure folded into the Leray projector P) is
    d_t ubar = -P[ filt((u.grad)u) ] + nu P lap(ubar)
so the exact SGS force a closure must supply is
    m_true = P[ (ubar.grad)ubar - filt((u.grad)u) ] .

Backend-agnostic (``sp.xp`` is NumPy or CuPy).  Pedagogical demonstration code.
"""

from __future__ import annotations

import numpy as np

from closure.spectral3d import (
    Spectral3D, project3d, divergence_rms3d, shell_index, sharp_filter3d, to_host,
)

__all__ = [
    "exact_sgs_force3d", "smagorinsky_force3d", "surrogate_force3d",
    "projected_fdt_force3d", "force_spectrum3d", "transfer_spectrum3d",
    "divergence_rms3d", "energy_spectrum3d", "vorticity3d", "strain3d",
    "enstrophy_production", "strain_vorticity_alignment", "exact_sgs_stress",
    "sgs_flux_stats",
]


# ---------------------------------------------------------------------------
# advection helper
# ---------------------------------------------------------------------------

def _advection3d(sp: Spectral3D, u, v, w):
    """(u.grad)u componentwise (fully resolved on the DNS grid)."""
    ux, uy, uz = sp.ddx(u), sp.ddy(u), sp.ddz(u)
    vx, vy, vz = sp.ddx(v), sp.ddy(v), sp.ddz(v)
    wx, wy, wz = sp.ddx(w), sp.ddy(w), sp.ddz(w)
    ax = u * ux + v * uy + w * uz
    ay = u * vx + v * vy + w * vz
    az = u * wx + v * wy + w * wz
    return ax, ay, az


# ---------------------------------------------------------------------------
# exact SGS force
# ---------------------------------------------------------------------------

def exact_sgs_force3d(sp: Spectral3D, u, v, w, kc: float):
    """m_true = P[ (ubar.grad)ubar - filt((u.grad)u) ], truncated to |k|<=kc."""
    ub = sharp_filter3d(sp, u, kc)
    vb = sharp_filter3d(sp, v, kc)
    wb = sharp_filter3d(sp, w, kc)
    ax, ay, az = _advection3d(sp, u, v, w)
    ax_f = sharp_filter3d(sp, ax, kc)
    ay_f = sharp_filter3d(sp, ay, kc)
    az_f = sharp_filter3d(sp, az, kc)
    axb, ayb, azb = _advection3d(sp, ub, vb, wb)
    axb = sharp_filter3d(sp, axb, kc)
    ayb = sharp_filter3d(sp, ayb, kc)
    azb = sharp_filter3d(sp, azb, kc)
    mx, my, mz = project3d(sp, axb - ax_f, ayb - ay_f, azb - az_f)
    return ub, vb, wb, mx, my, mz


# ---------------------------------------------------------------------------
# diagnostics: spectra
# ---------------------------------------------------------------------------

def _shell_sum(sp: Spectral3D, P, kmax: int):
    idx = shell_index(sp)
    xp = sp.xp
    E = xp.zeros(kmax + 1)
    for k in range(kmax + 1):
        E[k] = P[idx == k].sum()
    return np.arange(kmax + 1), to_host(E)


def force_spectrum3d(sp: Spectral3D, mx, my, mz, kc: float):
    """Shell-summed |m|^2 spectrum up to kc."""
    P = sp.xp.abs(sp.fft(mx)) ** 2 + sp.xp.abs(sp.fft(my)) ** 2 + sp.xp.abs(sp.fft(mz)) ** 2
    return _shell_sum(sp, P, int(kc))


def energy_spectrum3d(sp: Spectral3D, u, v, w):
    """Shell-summed kinetic-energy spectrum E(k) of the full field (for the
    inertial-range figure)."""
    n = sp.n
    P = 0.5 * (sp.xp.abs(sp.fft(u)) ** 2 + sp.xp.abs(sp.fft(v)) ** 2
               + sp.xp.abs(sp.fft(w)) ** 2) / (n ** 6)
    return _shell_sum(sp, P, int(n // 2))


def transfer_spectrum3d(sp: Spectral3D, ub, vb, wb, mx, my, mz, kc: float):
    """T(k) = sum_shell Re( uhat* . mhat ).  >0 backscatter, <0 dissipation."""
    uh, vh, wh = sp.fft(ub), sp.fft(vb), sp.fft(wb)
    mxh, myh, mzh = sp.fft(mx), sp.fft(my), sp.fft(mz)
    Tk = sp.xp.real(sp.xp.conj(uh) * mxh + sp.xp.conj(vh) * myh + sp.xp.conj(wh) * mzh)
    return _shell_sum(sp, Tk, int(kc))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def smagorinsky_force3d(sp: Spectral3D, ub, vb, wb, kc: float, cs: float = 0.16):
    """K-theory: m = div(2 nu_t S),  nu_t = (cs*Delta)^2 |S|,  Delta = pi/kc.
    Positive-definite eddy viscosity -> purely dissipative (no backscatter)."""
    delta = np.pi / kc
    ux, uy, uz = sp.ddx(ub), sp.ddy(ub), sp.ddz(ub)
    vx, vy, vz = sp.ddx(vb), sp.ddy(vb), sp.ddz(vb)
    wx, wy, wz = sp.ddx(wb), sp.ddy(wb), sp.ddz(wb)
    s11, s22, s33 = ux, vy, wz
    s12, s13, s23 = 0.5 * (uy + vx), 0.5 * (uz + wx), 0.5 * (vz + wy)
    smag = sp.xp.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + s33 ** 2
                             + 2.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)))
    nu_t = (cs * delta) ** 2 * smag
    t11, t22, t33 = 2.0 * nu_t * s11, 2.0 * nu_t * s22, 2.0 * nu_t * s33
    t12, t13, t23 = 2.0 * nu_t * s12, 2.0 * nu_t * s13, 2.0 * nu_t * s23
    mx = sp.ddx(t11) + sp.ddy(t12) + sp.ddz(t13)
    my = sp.ddx(t12) + sp.ddy(t22) + sp.ddz(t23)
    mz = sp.ddx(t13) + sp.ddy(t23) + sp.ddz(t33)
    mx, my, mz = (sharp_filter3d(sp, mx, kc), sharp_filter3d(sp, my, kc),
                  sharp_filter3d(sp, mz, kc))
    return project3d(sp, mx, my, mz)


def surrogate_force3d(sp: Spectral3D, mx, my, mz, kc: float, seed: int = 0):
    """Phase-randomized surrogate: identical per-mode amplitude (hence identical
    E_m(k)) as m_true, randomized phases.  Not projected -> div != 0."""
    rng = (sp.xp.random.default_rng(seed) if hasattr(sp.xp.random, "default_rng")
           else np.random.default_rng(seed))
    out = []
    for f in (mx, my, mz):
        amp = sp.xp.abs(sp.fft(f))
        phase = sp.xp.angle(sp.fft(rng.standard_normal(f.shape)))
        g = sp.ifft(amp * sp.xp.exp(1j * phase))
        out.append(sharp_filter3d(sp, g, kc))
    return out[0], out[1], out[2]


def projected_fdt_force3d(sp: Spectral3D, ub, vb, wb, mx, my, mz, kc: float, seed: int = 0):
    """Projected-FDT model (3D).

    1. measured scale-dependent eddy viscosity nu_t(k) reproducing the true shell
       transfer:  nu_t(k) = -T_true(k) / sum_shell(|k'|^2 |uhat|^2).  Allowed to be
       negative (backscatter).  m_par = -nu_t(k)|k|^2 uhat is solenoidal.
    2. a Leray-projected random remainder filling the residual force spectrum
       E_m(k) - E_par(k), so the total matches E_m(k) with div = 0 by construction.
    """
    xp = sp.xp
    uh, vh, wh = sp.fft(ub), sp.fft(vb), sp.fft(wb)
    mxh, myh, mzh = sp.fft(mx), sp.fft(my), sp.fft(mz)
    idx = shell_index(sp)
    k2 = sp.k2
    kmax = int(kc)

    Tk = xp.real(xp.conj(uh) * mxh + xp.conj(vh) * myh + xp.conj(wh) * mzh)
    denom_full = k2 * (xp.abs(uh) ** 2 + xp.abs(vh) ** 2 + xp.abs(wh) ** 2)
    par_x = xp.zeros_like(mxh)
    par_y = xp.zeros_like(myh)
    par_z = xp.zeros_like(mzh)
    nu_t_shell = xp.zeros(kmax + 1)
    for k in range(kmax + 1):
        m = idx == k
        denom = denom_full[m].sum()
        if float(denom) <= 1e-30:
            continue
        nu_t = -Tk[m].sum() / denom
        nu_t_shell[k] = nu_t
        par_x[m] = -nu_t * k2[m] * uh[m]
        par_y[m] = -nu_t * k2[m] * vh[m]
        par_z[m] = -nu_t * k2[m] * wh[m]
    m_par_x, m_par_y, m_par_z = sp.ifft(par_x), sp.ifft(par_y), sp.ifft(par_z)

    _, E_true = force_spectrum3d(sp, mx, my, mz, kc)
    _, E_par = force_spectrum3d(sp, m_par_x, m_par_y, m_par_z, kc)
    rng = (xp.random.default_rng(seed) if hasattr(xp.random, "default_rng")
           else np.random.default_rng(seed))
    rxs, rys, rzs = project3d(sp, rng.standard_normal(ub.shape),
                              rng.standard_normal(ub.shape),
                              rng.standard_normal(ub.shape))
    rxh, ryh, rzh = sp.fft(rxs), sp.fft(rys), sp.fft(rzs)
    scale_x = xp.zeros_like(rxh)
    scale_y = xp.zeros_like(ryh)
    scale_z = xp.zeros_like(rzh)
    for k in range(kmax + 1):
        m = idx == k
        resid = max(float(E_true[k]) - float(E_par[k]), 0.0)
        cur = float((xp.abs(rxh[m]) ** 2 + xp.abs(ryh[m]) ** 2 + xp.abs(rzh[m]) ** 2).sum())
        if cur <= 1e-30 or resid <= 0.0:
            continue
        s = np.sqrt(resid / cur)
        scale_x[m] = rxh[m] * s
        scale_y[m] = ryh[m] * s
        scale_z[m] = rzh[m] * s
    fbx, fby, fbz = sp.ifft(scale_x), sp.ifft(scale_y), sp.ifft(scale_z)

    mxm = sharp_filter3d(sp, m_par_x + fbx, kc)
    mym = sharp_filter3d(sp, m_par_y + fby, kc)
    mzm = sharp_filter3d(sp, m_par_z + fbz, kc)
    mxm, mym, mzm = project3d(sp, mxm, mym, mzm)
    return mxm, mym, mzm, to_host(nu_t_shell)


# ---------------------------------------------------------------------------
# 3D-specific physics diagnostics
# ---------------------------------------------------------------------------

def vorticity3d(sp: Spectral3D, u, v, w):
    """omega = curl(u)."""
    ox = sp.ddy(w) - sp.ddz(v)
    oy = sp.ddz(u) - sp.ddx(w)
    oz = sp.ddx(v) - sp.ddy(u)
    return ox, oy, oz


def strain3d(sp: Spectral3D, u, v, w):
    """Symmetric strain-rate components and |S| = sqrt(2 S_ij S_ij)."""
    ux, uy, uz = sp.ddx(u), sp.ddy(u), sp.ddz(u)
    vx, vy, vz = sp.ddx(v), sp.ddy(v), sp.ddz(v)
    wx, wy, wz = sp.ddx(w), sp.ddy(w), sp.ddz(w)
    s11, s22, s33 = ux, vy, wz
    s12, s13, s23 = 0.5 * (uy + vx), 0.5 * (uz + wx), 0.5 * (vz + wy)
    sumsq = s11 ** 2 + s22 ** 2 + s33 ** 2 + 2.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    smag = sp.xp.sqrt(2.0 * sumsq)
    return (s11, s22, s33, s12, s13, s23), sumsq, smag


def enstrophy_production(sp: Spectral3D, u, v, w):
    """Return (mean_production, normalized_production).

    Vortex-stretching enstrophy production  P = omega_i S_ij omega_j, the term that
    is *identically zero* in 2D.  Normalized by <|omega|^2> <|S|> so it is a pure,
    dimensionless geometry number comparable across runs."""
    xp = sp.xp
    ox, oy, oz = vorticity3d(sp, u, v, w)
    (s11, s22, s33, s12, s13, s23), _, smag = strain3d(sp, u, v, w)
    prod = (ox * ox * s11 + oy * oy * s22 + oz * oz * s33
            + 2.0 * (ox * oy * s12 + ox * oz * s13 + oy * oz * s23))
    enst = ox * ox + oy * oy + oz * oz
    mean_prod = float(xp.mean(prod))
    norm = float(xp.mean(enst) * xp.mean(smag)) + 1e-30
    return mean_prod, mean_prod / norm


def strain_vorticity_alignment(sp: Spectral3D, u, v, w, max_points: int = 400_000,
                               seed: int = 0):
    """Mean |cos(theta)| between vorticity and each strain eigenvector.

    Eigenvectors are ordered by eigenvalue: e1 (largest/extensional),
    e2 (intermediate), e3 (smallest/compressive).  The Constantin-Fefferman /
    Ashurst result is that vorticity aligns preferentially with the *intermediate*
    eigenvector e2 -- the geometric depletion of vortex stretching.  Returns a dict
    with mean |cos| for each eigenvector plus the mean eigenvalue conditioned on the
    e2 alignment.  Eigendecomposition is done on the host (subsampled for speed)."""
    ox, oy, oz = vorticity3d(sp, u, v, w)
    (s11, s22, s33, s12, s13, s23), _, _ = strain3d(sp, u, v, w)
    xp = sp.xp
    fields = [c.ravel() for c in (s11, s22, s33, s12, s13, s23, ox, oy, oz)]
    npts = int(fields[0].shape[0])
    rng = np.random.default_rng(seed)
    if npts > max_points:
        sel_h = rng.choice(npts, size=max_points, replace=False)
        sel = xp.asarray(sel_h)
        fields = [c[sel] for c in fields]
    # move only the (subsampled) data to the host for the eigendecomposition
    s11, s22, s33, s12, s13, s23, ox, oy, oz = [to_host(c) for c in fields]
    S = np.empty((s11.shape[0], 3, 3))
    S[:, 0, 0] = s11; S[:, 1, 1] = s22; S[:, 2, 2] = s33
    S[:, 0, 1] = S[:, 1, 0] = s12
    S[:, 0, 2] = S[:, 2, 0] = s13
    S[:, 1, 2] = S[:, 2, 1] = s23
    w_vec = np.stack([ox, oy, oz], axis=1)
    wn = np.linalg.norm(w_vec, axis=1)
    good = wn > 1e-12
    S, w_vec, wn = S[good], w_vec[good], wn[good]
    evals, evecs = np.linalg.eigh(S)          # ascending: [:,0]=min, [:,2]=max
    what = w_vec / wn[:, None]
    # |cos| with min (e3), intermediate (e2), max (e1) eigenvectors
    cos_min = np.abs(np.einsum("pi,pi->p", what, evecs[:, :, 0]))
    cos_int = np.abs(np.einsum("pi,pi->p", what, evecs[:, :, 1]))
    cos_max = np.abs(np.einsum("pi,pi->p", what, evecs[:, :, 2]))
    return {
        "cos_extensional": float(cos_max.mean()),   # e1 (largest eigenvalue)
        "cos_intermediate": float(cos_int.mean()),  # e2
        "cos_compressive": float(cos_min.mean()),   # e3 (most negative)
        "hist_extensional": np.histogram(cos_max, bins=20, range=(0, 1), density=True)[0],
        "hist_intermediate": np.histogram(cos_int, bins=20, range=(0, 1), density=True)[0],
        "hist_compressive": np.histogram(cos_min, bins=20, range=(0, 1), density=True)[0],
        "n_points": int(S.shape[0]),
    }


def exact_sgs_stress(sp: Spectral3D, u, v, w, kc: float):
    """Exact (deviatoric) SGS stress tau^d_ij = filt(u_i u_j) - ubar_i ubar_j and
    the resolved strain S_ij of ubar.  Returns (tau_d_components, S_components)."""
    ub = sharp_filter3d(sp, u, kc)
    vb = sharp_filter3d(sp, v, kc)
    wb = sharp_filter3d(sp, w, kc)
    def fij(a, b):
        return sharp_filter3d(sp, a * b, kc)
    t11 = fij(u, u) - ub * ub
    t22 = fij(v, v) - vb * vb
    t33 = fij(w, w) - wb * wb
    t12 = fij(u, v) - ub * vb
    t13 = fij(u, w) - ub * wb
    t23 = fij(v, w) - vb * wb
    tr = (t11 + t22 + t33) / 3.0
    t11, t22, t33 = t11 - tr, t22 - tr, t33 - tr   # deviatoric
    (s11, s22, s33, s12, s13, s23), _, _ = strain3d(sp, ub, vb, wb)
    return (t11, t22, t33, t12, t13, t23), (s11, s22, s33, s12, s13, s23)


def sgs_flux_stats(sp: Spectral3D, u, v, w, kc: float, cs: float = 0.16):
    """SGS energy flux Pi = -tau^d_ij S_ij to subgrid scales.

    Returns a dict comparing the *exact* SGS flux with the Smagorinsky model:
      - mean_flux_true / mean_flux_smag   (net; >0 = forward cascade)
      - backscatter_fraction_true         (volume fraction with Pi < 0)
      - backscatter_fraction_smag         (= 0 exactly for positive eddy viscosity)
    This is the headline 3D structural diagnostic: K-theory's positive-definite
    eddy viscosity cannot represent the large backscatter fraction of real 3D
    turbulence."""
    xp = sp.xp
    (t11, t22, t33, t12, t13, t23), (s11, s22, s33, s12, s13, s23) = \
        exact_sgs_stress(sp, u, v, w, kc)
    pi_true = -(t11 * s11 + t22 * s22 + t33 * s33
                + 2.0 * (t12 * s12 + t13 * s13 + t23 * s23))
    # Smagorinsky model stress tau^model_ij = -2 nu_t S_ij -> Pi = 2 nu_t |S|^2 >= 0
    delta = np.pi / kc
    smag = xp.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + s33 ** 2
                          + 2.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)))
    nu_t = (cs * delta) ** 2 * smag
    pi_smag = 2.0 * nu_t * (s11 ** 2 + s22 ** 2 + s33 ** 2
                            + 2.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2))
    return {
        "mean_flux_true": float(xp.mean(pi_true)),
        "mean_flux_smag": float(xp.mean(pi_smag)),
        "backscatter_fraction_true": float(xp.mean((pi_true < 0).astype(xp.float64))),
        "backscatter_fraction_smag": float(xp.mean((pi_smag < 0).astype(xp.float64))),
        "pi_true_std": float(xp.std(pi_true)),
    }
