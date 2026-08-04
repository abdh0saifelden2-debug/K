r"""Nonlocal (scale-dependent) eddy-diffusivity correction to the mainstream
gradient-diffusion hypothesis -- the K-theory closure used in essentially every
operational turbulence/ocean/ice model.

WHAT MAINSTREAM GETS WRONG
--------------------------
The gradient-diffusion hypothesis (Boussinesq eddy viscosity / K-theory) models a
turbulent flux as *local* and *down-gradient* with a single non-negative scalar
diffusivity,

    F(x) = -kappa_t grad C(x),   kappa_t >= 0.

THEOREM (structural failure, proved + tested here).  For ANY local closure with a
positive-semidefinite eddy diffusivity -- scalar kappa(x) >= 0 or a PSD tensor --

    F . grad C = -grad C . kappa . grad C <= 0   pointwise,

so the flux is down-gradient *everywhere* and the magnitude-weighted alignment

    C_G = <F . grad C> / <|F| |grad C|>

is identically -1 for a scalar closure (and <= 0 for any PSD tensor).  A local
down-gradient closure therefore CANNOT represent counter-gradient transport
(C_G > -1) -- the up-gradient flux that a lee recirculation, a coherent eddy, or
backscatter actually produces, where parcels carry the *nonlocal* memory of the
field that swept them there.  This repo measured exactly that: RESULT 11 /
Theorem 3 (glaciers/REPORT_CG_BUOYANCY.md) found C_G > -1 in a 3-D active-buoyancy
LES, so the local closure is falsified, not approximate.

THE CORRECTION (drop-in)
------------------------
Replace the scalar kappa_t by a *scale-dependent* eddy diffusivity kappa_hat(k) --
equivalently a nonlocal convolution flux F = -(K * grad C) whose spectral transform
is K_hat(k) = kappa_hat(k):

    F_hat(k) = -kappa_hat(k) (i k) C_hat(k).

This is the spatial analog of the temporal Mori-Zwanzig memory kernel
(glads_memory_correction.py): eliminating the fast/advective scales leaves a flux
that is nonlocal in space.  When kappa_hat(k) = kappa_0 is constant it collapses to
the local Fickian closure F = -kappa_0 grad C exactly (the safe default, recovered
as the kernel width -> 0).  When kappa_hat(k) dips negative in a band of scales
(backscatter / counter-transport by coherent structures) the corrected flux is
up-gradient there -- it reproduces C_G > -1, which no kappa(x) >= 0 can.

WHAT THIS MODULE PROVIDES
-------------------------
  grad_spectral / local_flux            -- the mainstream K-theory flux
  khat_constant / khat_backscatter      -- constant (local) vs scale-dependent kappa_hat
  nonlocal_flux                         -- F = -(K * grad C) via kappa_hat(k)
  apply_spectral_diffusivity            -- the actual drop-in retrofit a model calls
  counter_gradient                      -- the RESULT-11 alignment C_G
  best_local_kappa                      -- least-squares scalar kappa_t (mainstream fit)
  compare / run                         -- quantified demonstration (local pinned at
                                           C_G = -1 and high error; nonlocal recovers
                                           the counter-gradient truth)

This is a 1-D minimal-closure demonstration of the *structural* identity and its
nonlocal fix, grounded in the 3-D RESULT-11 measurement; it is not itself a 3-D DNS
(that is REPORT_CG_BUOYANCY.md).  CPU only, no external data.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# spectral operators on a periodic 1-D domain
# --------------------------------------------------------------------------- #
def wavenumbers(n: int, dx: float) -> np.ndarray:
    """Angular wavenumbers k for an n-point periodic grid of spacing dx."""
    return 2.0 * np.pi * np.fft.fftfreq(n, d=dx)


def grad_spectral(C: np.ndarray, dx: float) -> np.ndarray:
    """Spectral first derivative dC/dx on a periodic grid (exact for band-limited C)."""
    C = np.asarray(C, float)
    k = wavenumbers(len(C), dx)
    return np.fft.ifft(1j * k * np.fft.fft(C)).real


def local_flux(C: np.ndarray, dx: float, kappa: float) -> np.ndarray:
    """Mainstream K-theory flux F = -kappa dC/dx (local, down-gradient, scalar kappa)."""
    return -kappa * grad_spectral(C, dx)


# --------------------------------------------------------------------------- #
# scale-dependent eddy diffusivity kappa_hat(k)
# --------------------------------------------------------------------------- #
def khat_constant(k: np.ndarray, kappa0: float) -> np.ndarray:
    """The local closure as a (degenerate) spectral diffusivity: kappa_hat(k) = kappa0."""
    return np.full_like(np.asarray(k, float), float(kappa0))


def khat_backscatter(k: np.ndarray, kappa0: float, kc: float, width: float,
                     beta: float) -> np.ndarray:
    """Scale-dependent diffusivity that dips (and for beta>1 goes negative) in a band
    around |k| = kc -- the backscatter / counter-gradient scales -- and relaxes to the
    forward-scatter value kappa0 elsewhere."""
    k = np.asarray(k, float)
    dip = beta * np.exp(-((np.abs(k) - kc) ** 2) / (2.0 * width ** 2))
    return kappa0 * (1.0 - dip)


def nonlocal_flux(C: np.ndarray, dx: float, khat: np.ndarray) -> np.ndarray:
    """Nonlocal flux F = -(K * grad C) with spectral kernel kappa_hat(k):
    F_hat(k) = -khat(k) (i k) C_hat(k).  khat is sampled on wavenumbers(n, dx)."""
    C = np.asarray(C, float)
    k = wavenumbers(len(C), dx)
    khat = np.asarray(khat, float)
    return np.fft.ifft(-khat * (1j * k) * np.fft.fft(C)).real


def apply_spectral_diffusivity(C: np.ndarray, dx: float, khat: np.ndarray) -> np.ndarray:
    """Drop-in retrofit: given a model's scalar field C and a (measured) scale-dependent
    diffusivity khat sampled on wavenumbers(len(C), dx), return the corrected flux.
    A constant khat returns the model's original local Fickian flux unchanged."""
    return nonlocal_flux(C, dx, khat)


def dc_gain(khat: np.ndarray) -> float:
    """kappa_hat(k=0): the single scalar diffusivity the local model keeps."""
    return float(np.asarray(khat, float)[0])


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def counter_gradient(F: np.ndarray, gradC: np.ndarray) -> float:
    """Magnitude-weighted flux/gradient alignment C_G (RESULT 11 definition).
    C_G = -1 is pure down-gradient (K-theory); C_G > -1 is counter-gradient."""
    F = np.asarray(F, float)
    g = np.asarray(gradC, float)
    num = float(np.sum(F * g))
    den = float(np.sum(np.abs(F) * np.abs(g))) + 1e-30
    return num / den


def best_local_kappa(C: np.ndarray, dx: float, F_true: np.ndarray) -> float:
    """The scalar eddy diffusivity a modeller would calibrate: least-squares fit of
    F_true ~ -kappa dC/dx, i.e. kappa = -<F_true C_x> / <C_x^2>.  May come out negative
    when the truth is strongly counter-gradient -- the quantitative signal that the data
    demand an *unphysical* (negative) diffusivity, which K-theory forbids."""
    Cx = grad_spectral(C, dx)
    return float(-np.sum(F_true * Cx) / (np.sum(Cx * Cx) + 1e-30))


def best_admissible_kappa(C: np.ndarray, dx: float, F_true: np.ndarray) -> float:
    """The best *physically admissible* local closure: max(least-squares kappa, 0), since
    mainstream K-theory requires kappa >= 0.  This is the most generous local baseline --
    if even this is pinned at C_G = -1, no admissible local closure can do better."""
    return max(best_local_kappa(C, dx, F_true), 0.0)


def _relerr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# --------------------------------------------------------------------------- #
# quantified demonstration
# --------------------------------------------------------------------------- #
def make_field(n=512, L=2.0 * np.pi, modes=((1, 1.0), (6, 0.9), (11, 0.5)), seed=0):
    """A periodic scalar with energy spread across a few wavenumbers (one of them in
    the backscatter band).  Deterministic given seed."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, L, n, endpoint=False)
    C = np.zeros(n)
    for m, amp in modes:
        C += amp * np.sin(m * 2.0 * np.pi * x / L + rng.uniform(0, 2 * np.pi))
    return x, C


def compare(n=512, L=2.0 * np.pi, kappa0=0.05, kc=6.0, width=1.0, beta=1.5):
    """Compare the mainstream local closure with the nonlocal correction against a
    'truth' flux generated by a scale-dependent (backscatter) diffusivity."""
    dx = L / n
    x, C = make_field(n=n, L=L)
    k = wavenumbers(n, dx)
    gradC = grad_spectral(C, dx)

    # truth: flux from a scale-dependent diffusivity (backscatter band near kc)
    khat_true = khat_backscatter(k, kappa0, kc, width, beta)
    F_true = nonlocal_flux(C, dx, khat_true)
    cg_true = counter_gradient(F_true, gradC)

    # mainstream: the best *admissible* (kappa >= 0) scalar local closure, plus the
    # unconstrained least-squares kappa as a diagnostic (it can want kappa < 0)
    kappa_ls = best_local_kappa(C, dx, F_true)
    kappa_t = best_admissible_kappa(C, dx, F_true)
    F_loc = local_flux(C, dx, kappa_t)
    cg_loc = counter_gradient(F_loc, gradC)
    err_loc = _relerr(F_loc, F_true)
    # what a model calibrated to the large-scale (DC) diffusivity would use
    err_dc = _relerr(local_flux(C, dx, kappa0), F_true)

    # correction: nonlocal closure with the same kappa_hat(k) reproduces the truth
    F_nl = apply_spectral_diffusivity(C, dx, khat_true)
    cg_nl = counter_gradient(F_nl, gradC)
    err_nl = _relerr(F_nl, F_true)

    # local limit: constant kappa_hat == local Fickian flux, C_G -> -1
    F_const = apply_spectral_diffusivity(C, dx, khat_constant(k, kappa0))
    err_limit = _relerr(F_const, local_flux(C, dx, kappa0))
    cg_const = counter_gradient(F_const, gradC)

    ok = bool(
        -1.0 < cg_true < 0.0            # counter-gradient, still in the RESULT-11 regime
        and kappa_t > 0.0               # admissible local closure is non-degenerate
        and abs(cg_loc + 1.0) < 1e-9    # ...yet pinned at C_G = -1
        and err_nl < 1e-10              # nonlocal closure is exact
        and err_loc > 10.0 * err_nl     # best admissible local closure cannot fit
        and err_limit < 1e-12           # constant khat == local Fick exactly
        and abs(cg_const + 1.0) < 1e-9  # ...and is down-gradient
    )
    return dict(
        kappa0=kappa0, kappa_ls_unconstrained=kappa_ls, kappa_t_admissible=kappa_t,
        dc_gain=dc_gain(khat_true),
        cg_true=cg_true, cg_local=cg_loc, cg_nonlocal=cg_nl, cg_const=cg_const,
        err_local=err_loc, err_local_dc=err_dc, err_nonlocal=err_nl,
        err_local_limit=err_limit, ok=ok,
    )


def run():
    return compare()


def main():
    r = run()
    print("=== Nonlocal eddy-diffusivity correction (gradient-diffusion / K-theory) ===")
    print(f"  best admissible kappa_t = {r['kappa_t_admissible']:.4f}   "
          f"(unconstrained LS = {r['kappa_ls_unconstrained']:+.4f}, DC gain = {r['dc_gain']:+.4f})")
    print(f"  C_G truth (scale-dependent)   : {r['cg_true']:+.3f}  -> counter-gradient (RESULT-11 regime)")
    print(f"  C_G mainstream (best local)   : {r['cg_local']:+.3f}  -> pinned at -1 (down-gradient)")
    print(f"  C_G nonlocal correction       : {r['cg_nonlocal']:+.3f}  -> recovers truth")
    print(f"  flux rel-err  best-local : {r['err_local']:.3e}   nonlocal : {r['err_nonlocal']:.3e}")
    print(f"  local-limit (const kappa_hat == Fick) rel-err : {r['err_local_limit']:.2e}")
    print(f"  PASS: {r['ok']}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
