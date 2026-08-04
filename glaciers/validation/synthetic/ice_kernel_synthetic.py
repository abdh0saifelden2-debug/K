r"""§V.4 synthetic unit test for the ice-side boundary memory kernel (§B.2).

No external data.  §B.2 claims that solving the ice heat equation with a *moving*
(ablating) interface gives an ice-side conductive flux that depends on the **history**
of the interface velocity, i.e. a memory kernel.  Here we *derive* that kernel in
closed form and check it against a direct finite-difference solve of the governing
PDE.

Derivation (linearised, semi-infinite ice in the interface-attached frame).  Let the
ice occupy ``xi > 0`` with the melting interface at ``xi = 0``.  With a mean ablation
speed ``Vbar`` the steady temperature perturbation ``thetabar = T - T_melt`` obeys

    kappa thetabar'' + Vbar thetabar' = 0,   thetabar(0)=0, thetabar(inf)=theta_far,

so ``thetabar(xi) = theta_far (1 - exp(-Vbar xi / kappa))`` and the steady ice-side
gradient is ``thetabar'(0) = theta_far Vbar / kappa``.  Perturb the interface speed
``Vbar -> Vbar + v'(t)``.  The perturbation field obeys

    d_t theta' = kappa theta'_xixi + Vbar theta'_xi + v'(t) thetabar'(xi),
    theta'(0,t) = 0  (interface pinned at T_melt),   theta'(inf,t) = 0.

Laplace-transforming in time (zero initial perturbation) and solving the resulting
ODE with the decaying root gives the *interface-flux transfer function*

    H(s) = q_ice'(s) / v'(s)
         = A * (1 - sqrt(1 + 4 tau_d s)) / s,    A = k_th theta_far Vbar^2 / (2 kappa^2),

with the advective thermal time ``tau_d = kappa / Vbar^2``.  Its inverse Laplace
transform is the memory kernel

    G(t) = A [ erfc( sqrt(t / 4 tau_d) ) - 2 sqrt(tau_d / (pi t)) exp(-t / 4 tau_d) ].

Key consequences (the actual [HYP] -> [DERIVED] content):

  * **Short-time tail ~ t^{-1/2}.**  As ``t -> 0`` the singular term dominates,
    ``G(t) ~ |A| * 2 sqrt(tau_d / pi) * t^{-1/2}`` -- confirming the ``sqrt(t)``
    diffusive response §B.2 asserted.
  * **Exponential cutoff at the diffusion time, decaying at LONG lag.**  The cutoff
    is ``exp(-t / 4 tau_d)`` (with ``tau_d = kappa/Vbar^2``, replaced by ``H^2/kappa``
    if the ice is thin).  This *corrects* §B.2's schematic ``exp(-H^2 / 4 kappa t)``,
    whose argument is inverted -- that form would suppress *short* lags, the opposite
    of a causal memory tail.
  * **Dimensional closure.**  ``[A] = W/m^3`` and the bracket is dimensionless, so
    ``[G] = W/m^3``; the resolvent kernel ``K_ice = G / (rho_i L)`` that appears in
    ``v = (1/rho_i L)[q_water - (K_ice * q_water)]`` has units ``1/s`` -- a *rate*,
    resolving §B.2's note that ``(kappa/t)^{1/2}`` was a velocity, not a rate.
  * **DC gain = quasi-steady sensitivity.**  ``H(0) = -rho c theta_far = integral G
    dt`` equals ``d qbar_ice / d Vbar`` from the steady profile -- an independent
    cross-check that the normalisation is right.

This validates the *kernel and its normalisation* by matching the analytic step
response ``S(t) = integral_0^t G`` to a direct PDE solve.  The melt-rate *amplitude*
in any particular cavity still depends on ``theta_far, Vbar`` there, which remain
problem inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc


# --- analytic kernel ---------------------------------------------------------
def kernel_G(t, kappa=1.0, Vbar=1.0, theta_far=-1.0, rho_c=1.0):
    """Closed-form ice-side memory kernel G(t) [units W/m^3]."""
    t = np.asarray(t, dtype=float)
    k_th = rho_c * kappa
    tau_d = kappa / Vbar**2
    A = k_th * theta_far * Vbar**2 / (2.0 * kappa**2)
    out = np.zeros_like(t)
    pos = t > 0
    tp = t[pos]
    out[pos] = A * (erfc(np.sqrt(tp / (4.0 * tau_d)))
                    - 2.0 * np.sqrt(tau_d / (np.pi * tp)) * np.exp(-tp / (4.0 * tau_d)))
    return out


def dc_gain(theta_far=-1.0, rho_c=1.0):
    """H(0) = -rho c theta_far = d qbar_ice / d Vbar (quasi-steady sensitivity)."""
    return -rho_c * theta_far


# --- direct PDE solve of the linearised moving-frame heat equation -----------
def pde_step_response(kappa=1.0, Vbar=1.0, theta_far=-1.0, rho_c=1.0,
                      L=40.0, n=800, dt=1e-3, t_end=12.0):
    """Finite-difference step response: apply v'(t)=1 (Heaviside) and record the
    interface flux perturbation q_ice'(t) = -k_th d_xi theta'|_0."""
    k_th = rho_c * kappa
    dx = L / n
    # Explicit forward-Euler stability for the central-difference advection-diffusion
    # update: violate any of these and the integration silently blows up, so fail
    # loudly if a future parameter change crosses a limit.
    diff_limit = dx**2 / (2.0 * kappa)            # diffusion: dt < dx^2/(2 kappa)
    if not dt < diff_limit:
        raise ValueError(
            f"diffusion-unstable: dt={dt:g} >= dx^2/(2 kappa)={diff_limit:g}")
    pe_grid = abs(Vbar) * dx / kappa              # grid Peclet < 2 for central diff
    if not pe_grid < 2.0:
        raise ValueError(
            f"grid-Peclet too large: |Vbar| dx/kappa={pe_grid:g} >= 2")
    cfl = abs(Vbar) * dt / dx                     # advective CFL < 1
    if not cfl < 1.0:
        raise ValueError(f"CFL-unstable: |Vbar| dt/dx={cfl:g} >= 1")
    xi = np.linspace(0.0, L, n + 1)
    # background gradient forcing shape thetabar'(xi) = theta_far (Vbar/kappa) e^{-Vbar xi/kappa}
    forcing_shape = theta_far * (Vbar / kappa) * np.exp(-Vbar * xi / kappa)
    th = np.zeros(n + 1)  # theta' (zero initial perturbation)
    nsteps = int(round(t_end / dt))
    ts = np.empty(nsteps)
    qs = np.empty(nsteps)
    for i in range(nsteps):
        lap = np.empty_like(th)
        lap[1:-1] = (th[2:] - 2 * th[1:-1] + th[:-2]) / dx**2
        lap[0] = lap[-1] = 0.0
        adv = np.empty_like(th)
        adv[1:-1] = (th[2:] - th[:-2]) / (2 * dx)  # central; diffusion-dominated => stable
        adv[0] = adv[-1] = 0.0
        th = th + dt * (kappa * lap + Vbar * adv + 1.0 * forcing_shape)  # v'(t)=1
        th[0] = 0.0   # interface pinned
        th[-1] = 0.0  # far field
        # interface flux perturbation q' = -k_th d_xi theta'|_0 (one-sided)
        dtheta0 = (-3 * th[0] + 4 * th[1] - th[2]) / (2 * dx)
        ts[i] = (i + 1) * dt
        qs[i] = -k_th * dtheta0
    return ts, qs


def _integral_G(t_end, kappa, Vbar, theta_far, rho_c, M=400001):
    """Accurate integral_0^{t_end} G dt via the u=sqrt(t) substitution, which
    removes the integrable t^{-1/2} singularity at the origin (dt = 2u du, so the
    singular factor t^{-1/2} ~ 1/u is cancelled by 2u)."""
    u = np.linspace(0.0, np.sqrt(t_end), M)
    t = u**2
    G = kernel_G(t, kappa, Vbar, theta_far, rho_c)
    integrand = 2.0 * u * G          # G dt = G * 2u du; finite limit at u=0
    integrand[0] = 0.0               # single endpoint, negligible at this resolution
    S = np.concatenate([[0.0],
                        np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(u))])
    return t, S                      # S[k] = integral_0^{t[k]} G dt


def run(dt=1e-3, n=800, t_end=12.0):
    kappa = Vbar = rho_c = 1.0
    theta_far = -1.0
    ts, q_num = pde_step_response(kappa, Vbar, theta_far, rho_c, n=n, dt=dt, t_end=t_end)

    # analytic step response S(t) = integral_0^t G dt', evaluated accurately in the
    # u=sqrt(t) variable, then interpolated onto the PDE time grid.
    t_grid, S_grid = _integral_G(t_end, kappa, Vbar, theta_far, rho_c)
    S_an = np.interp(ts, t_grid, S_grid)

    # (1) step-response match (skip t < 0.5 where the t^{-1/2} cusp dominates the
    #     finite-difference flux estimate at the wall)
    i0 = int(0.5 / dt)
    denom = np.linalg.norm(q_num[i0:])
    rel_step = np.linalg.norm(q_num[i0:] - S_an[i0:]) / (denom + 1e-30)

    # (2) DC gain: late-time step response -> H(0) = -rho c theta_far
    g0 = dc_gain(theta_far, rho_c)
    rel_dc = abs(q_num[-1] - g0) / abs(g0)

    # (3) short-time power law: log-log slope of |G| vs t near 0 -> -1/2
    tt = np.array([1e-4, 4e-4, 1.6e-3, 6.4e-3])
    gg = np.abs(kernel_G(tt, kappa, Vbar, theta_far, rho_c))
    slope = np.polyfit(np.log(tt), np.log(gg), 1)[0]
    rel_slope = abs(slope + 0.5)

    # (4) analytic identity: integral_0^inf G dt == H(0) (DC gain)
    _, S_big = _integral_G(400.0, kappa, Vbar, theta_far, rho_c)
    rel_norm = abs(S_big[-1] - g0) / abs(g0)

    ok = bool(rel_step < 0.05 and rel_dc < 0.05 and rel_slope < 0.02 and rel_norm < 0.02)
    return {
        "step_response_rel_err": float(rel_step),
        "dc_gain_rel_err": float(rel_dc),
        "shorttime_slope": float(slope),
        "shorttime_slope_rel_err": float(rel_slope),
        "kernel_norm_rel_err": float(rel_norm),
        "pass": ok,
    }


def main():
    r = run()
    print("=== §V.4 ice-side memory kernel synthetic unit test ===")
    print(f"  PDE step-response rel-err   : {r['step_response_rel_err']:.2e}"
          f"  (analytic int G vs finite-diff)")
    print(f"  DC-gain rel-err             : {r['dc_gain_rel_err']:.2e}"
          f"  (late-time -> -rho c theta_far)")
    print(f"  short-time log-log slope    : {r['shorttime_slope']:.4f}"
          f"  (~ -0.5 => t^(-1/2) tail)")
    print(f"  kernel norm int G vs H(0)   : {r['kernel_norm_rel_err']:.2e}")
    print(f"  PASS                        : {r['pass']}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
