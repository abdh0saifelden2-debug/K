r"""§V.3 synthetic unit test for the clock-mismatch correction (§G.5).

No external data.  §G.5 proposes adding ``-CMN * div( (d_t K_u) grad theta )`` to
the temperature equation, on the grounds that the eddy-diffusivity operator
``D[theta] = div(K grad theta)`` does **not commute** with ``d_t`` when ``K`` is
time-dependent:

    [d_t, D] theta = d_t( div(K grad theta) ) - div( K grad(d_t theta) )
                   = div( (d_t K) grad theta ).                       (identity)

The correction term is exactly ``-CMN`` times this commutator.  This unit test
checks, on a periodic spectral grid with a *time-dependent* ``K_u(x,y,t)`` and
``theta(x,y,t)``:

  1. **Commutator identity** -- the finite-difference-in-time commutator
     ``[d_t, D] theta`` matches the closed form ``div((d_t K) grad theta)`` to
     discretisation order (error -> 0 as dt -> 0).
  2. **Vanishes for steady turbulence** -- with ``d_t K = 0`` the term is ~0
     (machine/round-off), confirming §G.5's "maximal in transients, zero in
     steady state" property.
  3. **Scales with the transient rate** -- doubling ``d_t K`` doubles the term.
  4. **Dimensional signature** -- evaluated where ``K`` is rate-independent but
     ``d_tK`` is not, the diffusion term ``div(K grad theta)`` is independent of
     the transient rate ``omega`` while the commutator term scales *linearly*
     with ``omega``. The commutator therefore carries one extra inverse-time
     factor (units ``Theta/T^2``) relative to the tendency ``d_t theta``
     (``Theta/T``), so the coefficient ``CMN`` multiplying it must have units of
     **time** -- it is a correlation/memory time ``tau_c``, *not* a dimensionless
     ``O(1)`` constant.

This validates the *identity, its discretisation, and the dimensional signature
of the coefficient* -- not the (heuristic) sign or magnitude of ``tau_c``, which
remain [HYP].
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# --- spectral periodic operators on [0, 2pi)^2 -------------------------------
def _k(n):
    return np.fft.fftfreq(n, d=1.0 / n)  # integer wavenumbers (domain 2pi)


def grad(f):
    n = f.shape[0]
    kx = _k(n)[:, None]
    ky = _k(n)[None, :]
    F = np.fft.fft2(f)
    fx = np.real(np.fft.ifft2(1j * kx * F))
    fy = np.real(np.fft.ifft2(1j * ky * F))
    return fx, fy


def div(fx, fy):
    n = fx.shape[0]
    kx = _k(n)[:, None]
    ky = _k(n)[None, :]
    return np.real(np.fft.ifft2(1j * kx * np.fft.fft2(fx)
                                + 1j * ky * np.fft.fft2(fy)))


def D(K, theta):
    """Diffusion operator div(K grad theta)."""
    tx, ty = grad(theta)
    return div(K * tx, K * ty)


# --- time-dependent analytic fields ------------------------------------------
def theta_field(n, t):
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.sin(X) * np.cos(Y) * (1.0 + 0.5 * np.sin(0.7 * t))


OMEGA = 0.9  # fixed temporal frequency of the eddy-diffusivity transient


def K_field(n, t, eps=0.4, steady=False, omega=OMEGA):
    """Time-dependent eddy diffusivity; ``eps`` is the transient amplitude
    (``eps=0`` => steady turbulence), ``omega`` the transient rate."""
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    base = 0.3 + 0.1 * np.cos(X) * np.cos(Y)
    if steady:
        return base
    return base * (1.0 + eps * np.sin(omega * t))


def dKdt_field(n, t, eps=0.4, omega=OMEGA):
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    base = 0.3 + 0.1 * np.cos(X) * np.cos(Y)
    return base * (eps * omega * np.cos(omega * t))


def D_term(n, t, eps=0.4, omega=OMEGA):
    """The §G.5 correction integrand div((d_tK) grad theta) (closed form)."""
    tx, ty = grad(theta_field(n, t))
    dK = dKdt_field(n, t, eps, omega)
    return div(dK * tx, dK * ty)


def commutator_fd(n, t, dt, eps=0.4):
    """Finite-difference-in-time commutator [d_t, D] theta."""
    F = lambda tt: D(K_field(n, tt, eps), theta_field(n, tt))  # noqa: E731
    dF = (F(t + dt) - F(t - dt)) / (2 * dt)
    dtheta = (theta_field(n, t + dt) - theta_field(n, t - dt)) / (2 * dt)
    return dF - D(K_field(n, t, eps), dtheta)


def run(n=64, t=1.3, dt=1e-3, eps=0.4):
    # (1) identity: FD commutator vs closed form div((d_tK) grad theta)
    lhs = commutator_fd(n, t, dt, eps)
    rhs = D_term(n, t, eps)
    rel = np.linalg.norm(lhs - rhs) / (np.linalg.norm(rhs) + 1e-30)

    # (2) steady K -> term ~ 0
    steady_term = D_term(n, t, eps=0.0)  # d_tK = 0
    steady_mag = np.linalg.norm(steady_term) / (np.linalg.norm(rhs) + 1e-30)

    # (3) linearity in transient amplitude: doubling d_tK doubles the term
    # (frequency fixed, so only the amplitude scales -- no phase confound)
    ratio = np.linalg.norm(D_term(n, t, eps=0.4)) / \
        (np.linalg.norm(D_term(n, t, eps=0.2)) + 1e-30)

    # (4) dimensional signature. Evaluate at t=0, where K = base (independent of
    #     omega) but d_tK = base*eps*omega (linear in omega). The diffusion term
    #     div(K grad theta) is then rate-independent, while the commutator term
    #     div((d_tK) grad theta) doubles when omega doubles -> it carries one
    #     extra factor of inverse time, so CMN must have units of TIME.
    diff_1x = np.linalg.norm(D(K_field(n, 0.0, eps, omega=OMEGA), theta_field(n, 0.0)))
    diff_2x = np.linalg.norm(D(K_field(n, 0.0, eps, omega=2 * OMEGA), theta_field(n, 0.0)))
    diff_rate_sens = abs(diff_2x - diff_1x) / (diff_1x + 1e-30)   # ~0 (rate-independent)
    term_1x = np.linalg.norm(D_term(n, 0.0, eps, omega=OMEGA))
    term_2x = np.linalg.norm(D_term(n, 0.0, eps, omega=2 * OMEGA))
    term_rate_ratio = term_2x / (term_1x + 1e-30)                # ~2 (linear in rate)

    ok = bool(rel < 1e-3 and steady_mag < 1e-12 and abs(ratio - 2.0) < 0.02
              and diff_rate_sens < 1e-12 and abs(term_rate_ratio - 2.0) < 0.02)
    return {
        "identity_rel_err": float(rel),
        "steady_term_rel_mag": float(steady_mag),
        "amp_doubling_ratio": float(ratio),
        "diff_rate_sensitivity": float(diff_rate_sens),
        "term_rate_ratio": float(term_rate_ratio),
        "pass": ok,
    }


def main():
    r = run()
    print("=== §V.3 clock-mismatch (CMN) synthetic unit test ===")
    print(f"  commutator identity rel-err : {r['identity_rel_err']:.2e}  (-> 0 as dt->0)")
    print(f"  steady-K term rel-magnitude : {r['steady_term_rel_mag']:.2e}"
          f"  (~0 => vanishes in steady turbulence)")
    print(f"  amp-doubling term ratio     : {r['amp_doubling_ratio']:.3f}"
          f"  (~2 => linear in d_tK)")
    print(f"  diffusion rate-sensitivity  : {r['diff_rate_sensitivity']:.2e}"
          f"  (~0 => div(K grad theta) is rate-independent)")
    print(f"  commutator rate-ratio       : {r['term_rate_ratio']:.3f}"
          f"  (~2 => extra 1/T factor => CMN has units of TIME)")
    print(f"  PASS                        : {r['pass']}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
