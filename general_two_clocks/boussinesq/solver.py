"""2D incompressible Boussinesq convection (pseudo-spectral, projection method).
 
Demonstration companion to ``compressible/ns.py``.  On a doubly-periodic box we
advance the **slow** advective-diffusive *drift* (the deterministic generator of
the local Markov / heat semigroup -- the parabolic temperature clock) for the
velocity and buoyancy, then enforce incompressibility with a single elliptic
**Leray projection** (the fast, boundary-aware pressure clock):
 
    u*        = u^n + dt * [ -(u.grad)u + nu*lap(u) + (b - <b>) e_y ]   (slow drift)
    u^{n+1}   = P u*                                                    (fast projection)
    b^{n+1}   = b^n + dt * [ -(u.grad)b + kappa*lap(b) ]                (temperature clock)
 
with the Leray projector
 
    P = I - grad (lap)^{-1} div ,
 
i.e. exactly the divergence-free (solenoidal) part of the Helmholtz split in
``compressible/ns.py``.  This *is* Chorin's fractional-step method -- the
canonical incompressible algorithm -- here framed as the synthesis of the two
clocks: a local, memoryless drift made globally consistent by one instantaneous
elliptic solve.
 
Why the drift is "the perfect localized Markov chain": the advection-diffusion
operator is the deterministic generator of a continuous-time Markov process (its
heat kernel is the Gaussian transition density).  Using the operator -- rather
than a data-trained transition matrix -- isolates the *fundamental* limitation
of local, memoryless forecasting without ML/statistical noise.
 
Pedagogical demonstration solver, not a validated production code.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
 
from compressible.ns import Spectral2D, helmholtz
 
 
# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
 
def divergence(sp: Spectral2D, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return sp.ddx(u) + sp.ddy(v)
 
 
def project(sp: Spectral2D, u: np.ndarray, v: np.ndarray):
    """Leray projection P(u,v): the divergence-free (solenoidal) part."""
    u_sol, v_sol, _, _ = helmholtz(sp, u, v)
    return u_sol, v_sol
 
 
def projection_potential(sp: Spectral2D, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The scalar phi removed by the projection: lap(phi) = div(u,v), so that
    P(u,v) = (u,v) - grad(phi).  This is the elliptic 'pressure' field."""
    div_h = 1j * sp.kx * sp.fft(u) + 1j * sp.ky * sp.fft(v)
    phi_h = -div_h * sp.k2_inv          # lap -> -k^2  =>  phi_h = -div_h / k^2
    phi_h[0, 0] = 0.0
    return sp.ifft(phi_h)
 
 
def _dealias_advect(sp: Spectral2D, q: np.ndarray, u: np.ndarray,
                    v: np.ndarray) -> np.ndarray:
    """-(u.grad) q with 2/3-rule dealiasing of derivatives and the product."""
    qx = sp.ddx_dealias(q)
    qy = sp.ddy_dealias(q)
    adv = -(u * qx + v * qy)
    return sp.ifft(sp.fft(adv) * sp.dealias)
 
 
@dataclass
class BoussinesqState:
    u: np.ndarray
    v: np.ndarray
    b: np.ndarray
    t: float
 
 
class BoussinesqProjection:
    """Incompressible Boussinesq convection by explicit-drift + Leray projection."""
 
    def __init__(self, n: int, nu: float = 1e-3, kappa: float = 1e-3,
                 cfl: float = 0.4):
        self.sp = Spectral2D(n)
        self.nu = nu
        self.kappa = kappa
        self.cfl = cfl
 
    # --- the slow clock: advective-diffusive drift, NO pressure ---
    def drift(self, u, v, b):
        sp = self.sp
        du = _dealias_advect(sp, u, u, v) + self.nu * sp.laplacian(u)
        dv = (_dealias_advect(sp, v, u, v) + self.nu * sp.laplacian(v)
              + (b - float(b.mean())))          # buoyancy anomaly forces +y
        db = _dealias_advect(sp, b, u, v) + self.kappa * sp.laplacian(b)
        return du, dv, db
 
    def drift_velocity(self, st: BoussinesqState, dt: float):
        """One explicit drift step for velocity only -> the 'ignorant'
        intermediate u* (generally divergent, before any projection)."""
        du, dv, _ = self.drift(st.u, st.v, st.b)
        return st.u + dt * du, st.v + dt * dv
 
    # --- full projected step (RK2 midpoint, projecting the velocity tendency) ---
    def step(self, st: BoussinesqState, dt: float) -> None:
        sp = self.sp
        du1, dv1, db1 = self.drift(st.u, st.v, st.b)
        pu1, pv1 = project(sp, du1, dv1)
        u1 = st.u + 0.5 * dt * pu1
        v1 = st.v + 0.5 * dt * pv1
        b1 = st.b + 0.5 * dt * db1
 
        du2, dv2, db2 = self.drift(u1, v1, b1)
        pu2, pv2 = project(sp, du2, dv2)
        u = st.u + dt * pu2
        v = st.v + dt * pv2
        b = st.b + dt * db2
 
        u, v = project(sp, u, v)            # scrub round-off divergence
        st.u, st.v, st.b, st.t = u, v, b, st.t + dt
 
    def dt_cfl(self, st: BoussinesqState) -> float:
        umax = float(np.max(np.sqrt(st.u**2 + st.v**2)))
        return self.cfl * self.sp.dx / (umax + 1e-6)
 
 
# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------
 
def warm_blobs(sp: Spectral2D, amp: float = 1.0, seed: int = 0):
    """A few warm buoyancy blobs in the lower half + tiny noise; zero velocity.
    Warm (positive-anomaly) fluid is buoyant and rises into cooler surroundings,
    which must rush in to fill the space -- the mesoscale convection cycle."""
    x, y = sp.grid()
    b = np.zeros_like(x)
    centers = [(np.pi * 0.65, np.pi * 0.55), (np.pi * 1.35, np.pi * 0.5),
               (np.pi, np.pi * 0.7)]
    for cx, cy in centers:
        b += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 0.30 ** 2))
    rng = np.random.default_rng(seed)
    b += 0.01 * rng.standard_normal(b.shape)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    return u, v, b
 
 
# ---------------------------------------------------------------------------
# Spectral diagnostics (for the SPDE contrast)
# ---------------------------------------------------------------------------
 
def radial_spectrum(sp: Spectral2D, f: np.ndarray):
    """Isotropic (shell-averaged) power spectrum of a real field."""
    P = np.abs(sp.fft(f)) ** 2
    kr = np.sqrt(sp.k2)
    kmax = int(np.floor(kr.max()))
    kbin = np.arange(0, kmax + 1)
    E = np.zeros(kmax + 1)
    idx = np.round(kr).astype(int)
    for k in kbin:
        E[k] = P[idx == k].sum()
    return kbin, E
 
 
def phase_randomized(sp: Spectral2D, f: np.ndarray, seed: int) -> np.ndarray:
    """A surrogate field with the *identical* per-mode amplitude (hence identical
    power spectrum) as f, but randomized phases -- destroying spatial structure.
    This is the 'inject noise with the right spectrum' SPDE/stochastic move.
 
    The random phases are taken from the FFT of white noise so they inherit the
    Hermitian symmetry of a real signal; multiplying by the (real, symmetric)
    target amplitude keeps the result real with the spectrum preserved exactly
    (Parseval), unlike naive random phases which would halve the energy."""
    amp = np.abs(sp.fft(f))
    rng = np.random.default_rng(seed)
    rand_phase = np.angle(sp.fft(rng.standard_normal(f.shape)))
    return sp.ifft(amp * np.exp(1j * rand_phase))
