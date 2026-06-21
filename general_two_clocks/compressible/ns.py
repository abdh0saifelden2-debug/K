"""Nonlinear 2D isothermal compressible Navier-Stokes (pseudo-spectral).
 
We solve, on a doubly-periodic square box, the *fully nonlinear* compressible
equations with an isothermal closure p = c^2 * rho (constant sound speed c):
 
    d(rho)/dt   + div(m)                       = 0
    d(m_i)/dt   + d_j( m_i u_j )  + d_i p       = d_j tau_ij
 
with m = rho u the momentum, u = m / rho, and the (constant-mu) compressible
viscous stress
 
    tau_ij = mu ( d_i u_j + d_j u_i ) - (2/3) mu (div u) delta_ij .
 
Isothermal closure keeps the sound speed c an explicit, tunable constant, so the
Mach number M = U / c can be swept directly while retaining full nonlinear
advection.  At low Mach the flow is smooth (no shocks), so a pseudo-spectral
method with 2/3-rule dealiasing is accurate and stable.
 
The domain is [0, 2*pi)^2 so integer wavenumbers are used.
 
This is a pedagogical, *demonstration* solver — not a validated production code.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
 
 
# ---------------------------------------------------------------------------
# Spectral operators on [0, 2*pi)^2
# ---------------------------------------------------------------------------
 
class Spectral2D:
    def __init__(self, n: int):
        self.n = n
        self.L = 2.0 * np.pi
        self.dx = self.L / n
        k = np.fft.fftfreq(n, d=1.0 / n)  # integer wavenumbers 0,1,..,-1
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.k2_inv = np.zeros_like(self.k2)
        self.k2_inv[self.k2 > 0] = 1.0 / self.k2[self.k2 > 0]
        # 2/3-rule dealiasing mask
        kmax = n // 3
        self.dealias = (np.abs(k)[:, None] <= kmax) & (np.abs(k)[None, :] <= kmax)
 
    # spatial grid
    def grid(self):
        x = np.arange(self.n) * self.dx
        return np.meshgrid(x, x, indexing="ij")
 
    def fft(self, f):
        return np.fft.fft2(f)
 
    def ifft(self, F):
        return np.real(np.fft.ifft2(F))
 
    def ddx(self, f):
        return self.ifft(1j * self.kx * self.fft(f))
 
    def ddy(self, f):
        return self.ifft(1j * self.ky * self.fft(f))
 
    def laplacian(self, f):
        return self.ifft(-self.k2 * self.fft(f))
 
    def ddx_dealias(self, f):
        F = self.fft(f) * self.dealias
        return self.ifft(1j * self.kx * F)
 
    def ddy_dealias(self, f):
        F = self.fft(f) * self.dealias
        return self.ifft(1j * self.ky * F)
 
 
# ---------------------------------------------------------------------------
# Helmholtz decomposition + incompressible reference pressure
# ---------------------------------------------------------------------------
 
def helmholtz(sp: Spectral2D, u: np.ndarray, v: np.ndarray):
    """Split (u, v) into solenoidal (divergence-free) and dilatational
    (curl-free) parts.  Returns (u_sol, v_sol, u_dil, v_dil)."""
    uh, vh = sp.fft(u), sp.fft(v)
    div_h = 1j * sp.kx * uh + 1j * sp.ky * vh
    phi_h = -div_h * sp.k2_inv  # potential: laplacian(phi) = div
    u_dil = sp.ifft(1j * sp.kx * phi_h)
    v_dil = sp.ifft(1j * sp.ky * phi_h)
    u_sol = u - u_dil
    v_sol = v - v_dil
    return u_sol, v_sol, u_dil, v_dil
 
 
def incompressible_pressure(sp: Spectral2D, u: np.ndarray, v: np.ndarray,
                            rho0: float) -> np.ndarray:
    """Elliptic (incompressible) pressure: laplacian(p) = -rho0 d_i u_j d_j u_i."""
    ux, uy = sp.ddx(u), sp.ddy(u)
    vx, vy = sp.ddx(v), sp.ddy(v)
    src = -rho0 * (ux * ux + 2.0 * uy * vx + vy * vy)
    p_h = -sp.fft(src) * sp.k2_inv
    p_h[0, 0] = 0.0
    return sp.ifft(p_h)
 
 
# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
 
@dataclass
class NSState:
    rho: np.ndarray
    mx: np.ndarray
    my: np.ndarray
    t: float
    theta: np.ndarray | None = None      # optional passive scalar (mean-gradient)
 
class IsothermalCompressibleNS:
    def __init__(self, n: int, c: float, mu: float, rho0: float = 1.0,
                 cfl: float = 0.3, *, cs: float = 0.0, backscatter: float = 0.0,
                 bs_tau: float = 0.0, kappa: float = 0.0, scalar_grad: float = 0.0,
                 seed: int = 0):
        self.sp = Spectral2D(n)
        self.c = c
        self.mu = mu
        self.rho0 = rho0
        self.cfl = cfl
        # --- optional Smagorinsky + (white/colored) backscatter closure -------
        # All default to 0 -> the original rhs/step path is untouched; the demo
        # and the 82-test suite are unaffected.  cs>0 adds an eddy viscosity;
        # backscatter>0 returns a fraction of the drained energy as a solenoidal
        # stochastic force whose variance is tied to the instantaneous SGS
        # dissipation (FDT).  bs_tau>0 gives that force an OU memory time
        # tau_mem=bs_tau (exact-exponential), so the closure carries memory of
        # previous shear states.  Same construction as subglacial/flow.py, here
        # ported to the finite-c compressible solver where the pressure
        # adjustment time tau_adj = L/c is finite (Direction A).
        self.cs = cs
        self.backscatter = backscatter
        self.bs_tau = bs_tau
        self.kappa = kappa
        self.scalar_grad = scalar_grad
        self.rng = np.random.default_rng(seed)
        self.bs_x = np.zeros((n, n))         # persistent OU backscatter state
        self.bs_y = np.zeros((n, n))
        self._dt = 0.0                       # step dt frozen for the FDT variance
        self._last_eps = 0.0                 # last mean SGS dissipation (diagnostic)
 
    def rhs(self, rho, mx, my):
        sp = self.sp
        u = mx / rho
        v = my / rho
        p = self.c**2 * rho
 
        # Continuity (linear in conserved variables — full spectral accuracy).
        drho = -sp.ddx(mx) - sp.ddy(my)
 
        # Momentum: nonlinear convective fluxes (dealiased) + pressure gradient
        # (linear, full spectral accuracy).
        dmx = -(sp.ddx_dealias(mx * u) + sp.ddy_dealias(mx * v)) - sp.ddx(p)
        dmy = -(sp.ddx_dealias(my * u) + sp.ddy_dealias(my * v)) - sp.ddy(p)
 
        # Compressible viscous stress (constant mu).
        div = sp.ddx(u) + sp.ddy(v)
        visc_x = self.mu * (sp.laplacian(u) + (1.0 / 3.0) * sp.ddx(div))
        visc_y = self.mu * (sp.laplacian(v) + (1.0 / 3.0) * sp.ddy(div))
        dmx = dmx + visc_x
        dmy = dmy + visc_y
        return drho, dmx, dmy
 
    def step(self, state: NSState, dt: float) -> None:
        rho, mx, my = state.rho, state.mx, state.my
        k1 = self.rhs(rho, mx, my)
        k2 = self.rhs(rho + 0.5 * dt * k1[0], mx + 0.5 * dt * k1[1],
                      my + 0.5 * dt * k1[2])
        k3 = self.rhs(rho + 0.5 * dt * k2[0], mx + 0.5 * dt * k2[1],
                      my + 0.5 * dt * k2[2])
        k4 = self.rhs(rho + dt * k3[0], mx + dt * k3[1], my + dt * k3[2])
        state.rho = rho + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        state.mx = mx + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        state.my = my + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        state.t += dt
 
    def dt_cfl(self, state: NSState) -> float:
        u = state.mx / state.rho
        v = state.my / state.rho
        umax = float(np.max(np.sqrt(u**2 + v**2)))
        return self.cfl * self.sp.dx / (self.c + umax + 1e-12)
 
    # ------------------------------------------------------------------ #
    # optional closure (Smagorinsky + white/colored backscatter) and a
    # mean-gradient passive scalar, used by the finite-c tidal phase-lag
    # probe (Direction A).  Inactive unless cs/backscatter/scalar_grad > 0.
    # ------------------------------------------------------------------ #
    def _project(self, fx, fy):
        """Solenoidal (divergence-free) part of (fx, fy)."""
        us, vs, _, _ = helmholtz(self.sp, fx, fy)
        return us, vs
 
    def _sgs_force(self, u, v):
        """Smagorinsky eddy-viscosity acceleration + optional FDT backscatter.
        Returns (ax, ay, eps_mean); the stochastic part is frozen by the caller
        across the RK4 substages.  Mirrors subglacial/flow.py._sgs_force."""
        sp = self.sp
        d = sp.dx
        ux, uy = sp.ddx(u), sp.ddy(u)
        vx, vy = sp.ddx(v), sp.ddy(v)
        s11, s22, s12 = ux, vy, 0.5 * (uy + vx)
        smag = np.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2))
        nu_t = (self.cs * d) ** 2 * smag
        tau11 = 2.0 * nu_t * s11
        tau22 = 2.0 * nu_t * s22
        tau12 = 2.0 * nu_t * s12
        ax = sp.ddx(tau11) + sp.ddy(tau12)
        ay = sp.ddx(tau12) + sp.ddy(tau22)
        eps = 2.0 * nu_t * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2)
        eps_mean = float(np.mean(eps))
        self._last_eps = eps_mean
        if self.backscatter > 0.0 and self._dt > 0.0:
            # clip pointwise eps to avoid overflow in eps/dt when dt is tiny
            eps_clip = np.clip(eps, 0.0, max(100.0 * eps_mean, 1e-20))
            amp = np.sqrt(eps_clip / self._dt)
            fx, fy = self._project(amp * self.rng.standard_normal(u.shape),
                                   amp * self.rng.standard_normal(u.shape))
            inj = float(np.mean(fx * u + fy * v))
            if abs(inj) > 1e-30 and eps_mean > 0.0:
                scale = float(np.clip(self.backscatter * eps_mean / inj, -5.0, 5.0))
                gx, gy = scale * fx, scale * fy          # FDT-scaled backscatter
                if self.bs_tau > 0.0:
                    ex = float(np.exp(-self._dt / self.bs_tau))
                    sg = float(np.sqrt(1.0 - ex * ex))
                    self.bs_x = ex * self.bs_x + sg * gx
                    self.bs_y = ex * self.bs_y + sg * gy
                    # clip OU state to prevent runaway accumulation
                    ou_rms = float(np.sqrt(np.mean(self.bs_x**2 + self.bs_y**2)))
                    g_rms = float(np.sqrt(np.mean(gx**2 + gy**2))) + 1e-30
                    if ou_rms > 3.0 * g_rms:
                        r = 3.0 * g_rms / ou_rms
                        self.bs_x *= r; self.bs_y *= r
                    gx, gy = self.bs_x, self.bs_y
                ax = ax + gx
                ay = ay + gy
        return ax, ay, eps_mean
 
    def rhs_forced(self, rho, mx, my, theta, fx_acc, fy_acc):
        """rhs() plus a frozen body acceleration (fx_acc, fy_acc) and a
        mean-gradient passive scalar theta (advected + diffused)."""
        drho, dmx, dmy = self.rhs(rho, mx, my)
        dmx = dmx + rho * fx_acc                      # body force density = rho*accel
        dmy = dmy + rho * fy_acc
        dtheta = None
        if theta is not None:
            sp = self.sp
            u = mx / rho
            v = my / rho
            dtheta = (-(u * sp.ddx_dealias(theta) + v * sp.ddy_dealias(theta))
                      + self.kappa * sp.laplacian(theta)
                      - self.scalar_grad * v)           # mean-gradient production
        return drho, dmx, dmy, dtheta
 
    def step_closure(self, state: NSState, dt: float, fext=None) -> None:
        """RK4 step including the optional closure and scalar.  The SGS+
        backscatter+external acceleration is computed once from the start-of-step
        state and frozen across the substages (so the stochastic draw is
        consistent and the elliptic/acoustic clock is untouched)."""
        self._dt = dt
        rho, mx, my, th = state.rho, state.mx, state.my, state.theta
        ax = np.zeros_like(rho)
        ay = np.zeros_like(rho)
        if self.cs > 0.0 or self.backscatter > 0.0:
            sx, sy, _ = self._sgs_force(mx / rho, my / rho)
            ax = ax + sx
            ay = ay + sy
        if fext is not None:
            ax = ax + fext[0]
            ay = ay + fext[1]
 
        def f(r, mxx, myy, tt):
            return self.rhs_forced(r, mxx, myy, tt, ax, ay)
 
        k1 = f(rho, mx, my, th)
        def comb(a, k, h):
            return None if a is None else a + h * k
        k2 = f(rho + 0.5*dt*k1[0], mx + 0.5*dt*k1[1], my + 0.5*dt*k1[2],
               comb(th, k1[3], 0.5*dt))
        k3 = f(rho + 0.5*dt*k2[0], mx + 0.5*dt*k2[1], my + 0.5*dt*k2[2],
               comb(th, k2[3], 0.5*dt))
        k4 = f(rho + dt*k3[0], mx + dt*k3[1], my + dt*k3[2],
               comb(th, k3[3], dt))
        state.rho = np.maximum(rho + dt/6.0 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]), 0.01 * self.rho0)
        state.mx = mx + dt/6.0 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        state.my = my + dt/6.0 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        if th is not None:
            state.theta = th + dt/6.0 * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        state.t += dt
 
# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
 
def taylor_green(sp: Spectral2D, u0: float, rho0: float = 1.0):
    """Smooth solenoidal Taylor-Green velocity, uniform density."""
    x, y = sp.grid()
    u = u0 * np.sin(x) * np.cos(y)
    v = -u0 * np.cos(x) * np.sin(y)
    rho = np.full_like(u, rho0)
    return rho, rho * u, rho * v
