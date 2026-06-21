r"""Timescale separation of the two clocks, and its identity with the
singular-perturbation parameter of the incompressible limit.

Directly addresses two of the user's checklist titles:

  [#1] "Two clocks: timescale separation between elliptic pressure constraints
        and parabolic scalar transport"
  [#4] "Two clocks, one incompressible fluid: a singular perturbation theory of
        mesoscale transport decoupling"

The point this module *measures* (not just asserts) is that #1 and #4 are the same
statement seen twice:

  FAST clock  — the elliptic pressure constraint  ∇²φ = ∇·u*  is INSTANTANEOUS:
    one Leray projection ℙ = I − ∇Δ⁻¹∇· drives ‖∇·u‖ to machine zero in a SINGLE
    step, independent of the time step.  Its relaxation time is τ_fast → 0.

  SLOW clock  — parabolic scalar transport  ∂ₜb = κΔb  relaxes a mode k at the
    FINITE rate κk², i.e. τ_slow = 1/(κk²).

  SEPARATION — the dimensionless ratio  ε = τ_fast / τ_slow  → 0.  That ε is not a
    nuisance; it is the singular-perturbation small parameter.  Regularising the
    constraint with a finite sound speed c_s (compressible flow) gives the pressure
    clock a finite acoustic time τ_fast = 1/(c_s k), so

        ε(k) = τ_fast / τ_slow = κ k / c_s   →  0   as  c_s → ∞   (Mach → 0).

  The incompressible model is therefore the ε→0 SINGULAR limit of compressible
  flow: the fast clock is squeezed to zero relaxation time, becoming the
  instantaneous elliptic projection.  "Mesoscale transport decoupling" (#4) is the
  statement that the slow (κk²) transport decouples from the infinitely-fast
  constraint.

Mainstream, named not invented: low-Mach singular-perturbation theory (Majda 1984;
Klein 1995); rigorous incompressible limit of compressible NS (Lions–Masmoudi 1998;
Schochet); Chorin's projection method.  CPU, deterministic.
"""
from __future__ import annotations

import numpy as np

from boussinesq import solver as B
from compressible.ns import Spectral2D


# --------------------------------------------------------------------------- #
# FAST clock — elliptic pressure constraint is instantaneous (one projection)
# --------------------------------------------------------------------------- #
def fast_clock_projection(n=64, seed=0):
    """The Leray projection enforces ∇·u = 0 in ONE step (τ_fast → 0). Build a
    smooth field with a real divergent (gradient) part and project once."""
    sp = Spectral2D(n)
    x, y = sp.grid()
    phi = (np.cos(x) + 0.5 * np.sin(2 * y) + 0.3 * np.cos(x) * np.cos(y)
           + 0.2 * np.cos(2 * x) * np.sin(y))
    u = np.sin(y) + sp.ifft(1j * sp.kx * sp.fft(phi)).real      # solenoidal + ∇φ
    v = -np.sin(x) + sp.ifft(1j * sp.ky * sp.fft(phi)).real
    d0 = float(np.sqrt(np.mean(B.divergence(sp, u, v) ** 2)))
    us, vs = B.project(sp, u, v)
    d1 = float(np.sqrt(np.mean(B.divergence(sp, us, vs) ** 2)))
    return dict(div_before=d0, div_after=d1, drop=d0 / max(d1, 1e-300),
                ok=bool(d1 < 1e-10 and d0 > 0.1))


# --------------------------------------------------------------------------- #
# SLOW clock — parabolic transport relaxes a mode k at the finite rate κk²
# --------------------------------------------------------------------------- #
def slow_clock_diffusion(n=64, kappa=0.02, ks=(1, 2, 4)):
    """Forward-integrate ∂ₜb = κΔb in spectral space and RECOVER the decay rate of
    each mode k; compare τ_meas to the theoretical τ_slow = 1/(κk²)."""
    sp = Spectral2D(n)
    x, y = sp.grid()
    rows = []
    worst = 0.0
    for k in ks:
        b = np.cos(k * x)
        bh = sp.fft(b)
        k2 = sp.k2
        dt = 0.1 / (kappa * sp.k2.max())          # stable explicit step
        T = 1.0 / (kappa * k ** 2)                 # one slow-clock time
        nsteps = int(np.ceil(T / dt))
        amp0 = float(np.abs(bh).max())
        ts, amps = [], []
        for s in range(nsteps + 1):
            ts.append(s * dt)
            amps.append(float(np.abs(bh).max()))
            bh = bh * (1.0 - dt * kappa * k2)      # explicit Euler in spectral space
        ts = np.array(ts)
        amps = np.array(amps) / amp0
        rate = -np.polyfit(ts, np.log(amps), 1)[0]   # measured decay rate
        tau_meas = 1.0 / rate
        tau_th = 1.0 / (kappa * k ** 2)
        rows.append((k, tau_meas, tau_th))
        worst = max(worst, abs(tau_meas - tau_th) / tau_th)
    return dict(rows=rows, max_rel_err=float(worst), ok=bool(worst < 0.05))


# --------------------------------------------------------------------------- #
# Compressible bridge — acoustic clock τ_fast = 1/(c_s k); ε = κk/c_s → 0
# --------------------------------------------------------------------------- #
def _acoustic_omega(c, k, N=256, n_steps=4000):
    """Linearised 1-D acoustics u_t=-p_x, p_t=-c² u_x: a k-mode is a sound wave at
    ω = c·k. Measure ω from zero-crossings of p at a fixed point."""
    L = 2 * np.pi
    dx = L / N
    x = np.arange(N) * dx
    p = np.cos(k * x).astype(float)
    u = np.zeros(N)
    dt = 0.2 * dx / c

    def ddx(f):
        return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)

    rec = np.empty(n_steps)
    for s in range(n_steps):
        u = u - dt * ddx(p)
        p = p - dt * c ** 2 * ddx(u)
        rec[s] = p[0]
    t = np.arange(n_steps) * dt
    sign = np.sign(rec)
    zc = np.where(np.diff(sign) != 0)[0]
    period = 2.0 * np.mean(np.diff(t[zc]))         # crossings are half-periods apart
    return 2.0 * np.pi / period


def acoustic_clock(kappa=0.02, k=1, c_list=(5.0, 10.0, 20.0, 40.0)):
    """The pressure clock at finite sound speed: τ_fast = 1/ω = 1/(c_s k), so the
    separation ε = τ_fast/τ_slow = κk/c_s ∝ c_s⁻¹ → 0 (the Mach→0 singular limit)."""
    rows = []
    om_err = 0.0
    for c in c_list:
        w = _acoustic_omega(c, k)
        tau_fast = 1.0 / w
        tau_slow = 1.0 / (kappa * k ** 2)
        eps = tau_fast / tau_slow
        rows.append((c, w, c * k, eps, kappa * k / c))
        om_err = max(om_err, abs(w - c * k) / (c * k))
    cs = np.array([r[0] for r in rows])
    eps = np.array([r[3] for r in rows])
    slope = float(np.polyfit(np.log(cs), np.log(eps), 1)[0])   # expect −1
    return dict(rows=rows, eps_exponent=slope, omega_err=float(om_err),
                ok=bool(abs(slope + 1.0) < 0.05 and om_err < 0.02))


def run():
    return dict(fast=fast_clock_projection(), slow=slow_clock_diffusion(),
                acoustic=acoustic_clock())


def main():
    f = fast_clock_projection()
    s = slow_clock_diffusion()
    a = acoustic_clock()
    print("=== FAST clock — elliptic pressure constraint (instantaneous) ===")
    print(f"  ||div u|| {f['div_before']:.3e} -> {f['div_after']:.3e} in ONE projection "
          f"(drop x{f['drop']:.1e})  ok={f['ok']}")
    print("=== SLOW clock — parabolic transport, τ_slow = 1/(κk²) ===")
    for k, tm, tt in s["rows"]:
        print(f"  k={k}: τ_meas={tm:.2f}  τ_theory=1/(κk²)={tt:.2f}")
    print(f"  max rel.err={s['max_rel_err']:.3f}  ok={s['ok']}")
    print("=== SINGULAR limit — acoustic clock τ_fast=1/(c_s k); ε=κk/c_s ∝ c_s⁻¹ ===")
    for c, w, ck, eps, pred in a["rows"]:
        print(f"  c_s={c:5.0f}: ω_meas={w:.2f} (c_s·k={ck:.2f})  ε={eps:.4f}  (κk/c_s={pred:.4f})")
    print(f"  ε ∝ c_s^{a['eps_exponent']:.3f}  (singular limit: exponent → −1)  ok={a['ok']}")
    ok = all(d["ok"] for d in (f, s, a))
    print(f"PASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
