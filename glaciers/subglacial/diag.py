"""Glacier-relevant a-priori closure diagnostics for the subglacial cavity.
 
These reuse the Part-8 sharp-filter machinery (``closure.sgs``) but add the two
diagnostics a glaciologist cares about:
 
  * **spatial energy transfer** Pi(x) = ubar . m  -- *where* the subgrid force
    feeds or drains resolved kinetic energy.  Backscatter (Pi > 0) localized in
    the lee of bedrock bumps is exactly what keeps a wake eddy alive; a
    positive-definite (K-theory) closure cannot produce it there.
 
  * **subgrid heat flux** tau_theta = bar(u*theta) - ubar*thetabar, and the
    eddy-diffusivity (K-theory) model -kappa_t grad(thetabar).  Counter-gradient
    (up-gradient) subgrid heat flux in the lee -- heat held against the mean
    gradient inside a recirculation cavity -- is structurally impossible for a
    positive eddy diffusivity, so K-theory cannot represent lee heat trapping.
"""
 
from __future__ import annotations
 
import numpy as np
 
from compressible.ns import Spectral2D
from closure.sgs import sharp_filter
 
 
def spatial_transfer(ub, vb, mx, my):
    """Pointwise resolved-scale energy transfer density Pi(x) = ubar . m.
    Pi > 0: subgrid returns energy to the resolved wake (backscatter);
    Pi < 0: forward dissipation."""
    return ub * mx + vb * my
 
 
def exact_sgs_heat_flux(sp: Spectral2D, u, v, theta, kc: float):
    """Exact subgrid heat-flux vector tau_theta = bar(u*theta) - ubar*thetabar
    and the resolved temperature gradient, all at the sharp cutoff kc."""
    ub = sharp_filter(sp, u, kc)
    vb = sharp_filter(sp, v, kc)
    tb = sharp_filter(sp, theta, kc)
    qx = sharp_filter(sp, u * theta, kc) - ub * tb
    qy = sharp_filter(sp, v * theta, kc) - vb * tb
    gx, gy = sp.ddx(tb), sp.ddy(tb)
    return qx, qy, tb, gx, gy
 
 
def eddy_diffusivity_heat_flux(sp: Spectral2D, ub, vb, tb, kc: float,
                               cs: float = 0.16, pr_t: float = 1.0):
    """K-theory scalar flux q = -kappa_t grad(thetabar), kappa_t = nu_t / Pr_t,
    nu_t = (cs*Delta)^2 |S|, Delta = pi/kc.  Down-gradient by construction."""
    delta = np.pi / kc
    ux, uy = sp.ddx(ub), sp.ddy(ub)
    vx, vy = sp.ddx(vb), sp.ddy(vb)
    s11, s22, s12 = ux, vy, 0.5 * (uy + vx)
    smag = np.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2))
    nu_t = (cs * delta) ** 2 * smag
    kappa_t = nu_t / pr_t
    gx, gy = sp.ddx(tb), sp.ddy(tb)
    return -kappa_t * gx, -kappa_t * gy
 
 
def heat_flux_divergence(sp: Spectral2D, qx, qy):
    """Subgrid heating field Q = -div(tau_theta)."""
    return -(sp.ddx(qx) + sp.ddy(qy))
 
 
def countergradient_fraction(qx, qy, gx, gy, mask):
    """Fraction of (masked) cells where the subgrid heat flux is up-gradient,
    i.e. tau_theta . grad(thetabar) > 0 -- the regime K-theory cannot represent."""
    align = qx * gx + qy * gy
    m = mask & np.isfinite(align)
    if m.sum() == 0:
        return 0.0
    return float(np.mean(align[m] > 0.0))
 
 
def backscatter_fraction(pi, mask):
    """Fraction of (masked) cells with Pi > 0 (backscatter)."""
    m = mask & np.isfinite(pi)
    if m.sum() == 0:
        return 0.0
    return float(np.mean(pi[m] > 0.0))
 
 
def masked_corr(a, b, mask):
    """Pearson correlation of two fields over a boolean mask (0 if degenerate)."""
    m = mask & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return 0.0
    aa, bb = a[m], b[m]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return 0.0
    c = np.corrcoef(aa, bb)[0, 1]
    return 0.0 if np.isnan(c) else float(c)
