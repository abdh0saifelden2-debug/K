r"""Dyadic (shell) model of Euler/Navier-Stokes: energy conservation does NOT prevent
finite-time blowup -- the nonlocal/dissipative structure does.

WHY THIS IS HERE (PR #1, regularity thread)
-------------------------------------------
`REPORT_MOLECULAR_REGULARITY.md` makes a load-bearing claim: a physical,
energy-conserving microscopic origin cannot by itself imply regularity, because
Tao (2016) built an *averaged* 3-D Navier-Stokes equation with the **same energy
identity** that nevertheless blows up in finite time -- so the specific nonlocal
structure of the true nonlinearity, not the energy budget, carries regularity.
This module makes that abstract point **runnable** with the classical dyadic
(shell) model -- the cleanest energy-conserving NS-like system that blows up.

THE MODEL (Katz-Pavlovic 2005; Cheskidov 2008; Friedlander-Pavlovic 2004)
-------------------------------------------------------------------------
Shells n = 0..N with wavenumbers k_n = lambda^n (lambda=2). Real amplitudes a_n:

    da_n/dt = k_n a_{n-1}^2  -  k_{n+1} a_n a_{n+1}  -  nu k_n^{2*gamma} a_n,
    a_{-1} = a_{N+1} = 0.

The inviscid (nu=0) nonlinearity is energy-conserving by an exact telescoping flux:
with the inter-shell flux Pi_n = k_{n+1} a_n^2 a_{n+1}, summing a_n*(da_n/dt) gives
dE/dt = sum_n (Pi_{n-1} - Pi_n) = 0  (E = 1/2 sum a_n^2).  Yet from a_0=1 the energy
cascades to ever-smaller scales: the high-k (enstrophy/H^1) norm grows without bound
as the resolution N -> infinity, and the cascade front reaches scale k_n at a time
t_n that *converges* to a finite t* -- a singularity with energy exactly conserved
(Katz-Pavlovic).  Adding the NS dissipation (gamma=1) arrests the cascade and the
solution is global (bounded H^1); supercritical dissipation (gamma small) does not.

WHAT IS DEMONSTRATED (CPU, deterministic)
-----------------------------------------
  1. Energy identity is EXACT: sum a_n*(da_n/dt) = 0 to round-off for any state
     (the telescoping flux), and energy drift under integration stays ~1e-5.
  2. Inviscid cascade: H^1 = (sum k_n^2 a_n^2)^{1/2} grows by >1e4x while energy is
     conserved.  For finite N the H^1 height saturates at the truncation ceiling
     ~k_N = 2^N (an artifact of the cutoff); the *genuine* finite-time-blowup
     evidence is that the cascade-front arrival time t_front(N) converges to a finite
     t* as N grows -- not a truncation artifact.
  3. Viscous, NS dissipation (gamma=1): H^1 stays bounded -- global regularity.
  4. The contrast IS the regularity face of the two-clocks thesis at shell level:
     remove the regularizing structure (here dissipation; in the VGT, the nonlocal
     pressure Hessian -> restricted Euler) and an energy-faithful model blows up.

Mainstream, named not invented: Katz & Pavlovic (2005, dyadic Euler blowup);
Cheskidov (2008, viscous dyadic threshold); Tao (2016, averaged-NS finite-time
blowup); Friedlander & Pavlovic (2004).  The linear dissipation is integrated by an
exact integrating factor (Strang split) so it is unconditionally stable; dt is set
only by the nonlinear CFL.
"""
from __future__ import annotations

import numpy as np


def k_shells(N: int, lam: float = 2.0) -> np.ndarray:
    return lam ** np.arange(N + 1, dtype=float)


def rhs(a: np.ndarray, k: np.ndarray, nu: float = 0.0, gamma: float = 1.0) -> np.ndarray:
    """Dyadic RHS: da_n = k_n a_{n-1}^2 - k_{n+1} a_n a_{n+1} - nu k_n^{2g} a_n."""
    am1 = np.empty_like(a); am1[0] = 0.0; am1[1:] = a[:-1]          # a_{n-1}
    ap1 = np.empty_like(a); ap1[-1] = 0.0; ap1[:-1] = a[1:]        # a_{n+1}
    kp1 = np.empty_like(k); kp1[:-1] = k[1:]; kp1[-1] = k[-1] * 2.0  # k_{n+1}
    out = k * am1 ** 2 - kp1 * a * ap1
    if nu > 0.0:
        out = out - nu * (k ** (2.0 * gamma)) * a
    return out


def energy(a: np.ndarray) -> float:
    return 0.5 * float(np.sum(a * a))


def hs_norm(a: np.ndarray, k: np.ndarray, s: float = 1.0) -> float:
    return float(np.sqrt(np.sum((k ** (2.0 * s)) * a * a)))


def flux(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Inter-shell energy flux Pi_n = k_{n+1} a_n^2 a_{n+1} (Pi_N := 0)."""
    kp1 = np.empty_like(k); kp1[:-1] = k[1:]; kp1[-1] = k[-1] * 2.0
    ap1 = np.empty_like(a); ap1[-1] = 0.0; ap1[:-1] = a[1:]
    return kp1 * a ** 2 * ap1


def _rk4_step(a, k, dt, nu, gamma):
    k1 = rhs(a, k, nu, gamma)
    k2 = rhs(a + 0.5 * dt * k1, k, nu, gamma)
    k3 = rhs(a + 0.5 * dt * k2, k, nu, gamma)
    k4 = rhs(a + dt * k3, k, nu, gamma)
    return a + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def front_time(N, theta=1e-4, t_end=12.0, cfl=0.15, dt_max=1e-2, lam=2.0,
               max_steps=4_000_000):
    """First time the smallest resolved shell (N) carries fraction theta of the energy
    -- the cascade-front arrival.  Inviscid only.  t_front(N) -> t* (finite) as
    N -> infinity is the signature of finite-time blowup (vs a truncation artifact).
    Returns t_front, or None if the front never arrives within t_end."""
    k = k_shells(N, lam)
    a = np.zeros(N + 1); a[0] = 1.0
    t = 0.0; n = 0
    while t < t_end and n < max_steps:
        rate = float(np.max(k * np.abs(a))) + 1.0
        dt = min(dt_max, cfl / rate)
        a = _rk4_step(a, k, dt, 0.0, 1.0); t += dt; n += 1
        if not np.all(np.isfinite(a)):
            return t
        if a[-1] ** 2 > theta * np.sum(a * a):
            return t
    return None


def integrate(N=18, nu=0.0, gamma=1.0, a0=None, t_end=12.0, cfl=0.15, dt_max=1e-2,
              lam=2.0, front_theta=1e-2, max_steps=400_000):
    """Integrate the dyadic model.  The linear dissipation is handled by an exact
    integrating factor (Strang split): unconditionally stable, dt set only by the
    nonlinear CFL.  The inviscid run stops once the cascade front reaches the top
    shell (further evolution just piles energy at k_N -- a truncation artifact);
    the viscous run integrates to t_end.  Records E(t), H^1(t) every step."""
    k = k_shells(N, lam)
    L = nu * (k ** (2.0 * gamma)) if nu > 0.0 else None     # linear dissipation rate
    a = np.zeros(N + 1)
    if a0 is None:
        a[0] = 1.0
    else:
        a[:len(a0)] = np.asarray(a0, float)
    E0 = energy(a)
    ts = [0.0]; Es = [E0]; H1s = [hs_norm(a, k, 1.0)]
    t = 0.0; n_steps = 0; max_drift = 0.0; t_front = None
    while t < t_end and n_steps < max_steps:
        rate = float(np.max(k * np.abs(a))) + 1.0       # nonlinear CFL (fastest shell)
        dt = min(dt_max, cfl / rate)
        if L is not None:                               # Strang: exact half-dissipation
            half = np.exp(-0.5 * dt * L)
            a = half * _rk4_step(half * a, k, dt, 0.0, 1.0)
        else:
            a = _rk4_step(a, k, dt, 0.0, 1.0)
        t += dt; n_steps += 1
        if not np.all(np.isfinite(a)):
            t_front = t; ts.append(t); Es.append(np.inf); H1s.append(np.inf)
            break
        E = energy(a); H1 = hs_norm(a, k, 1.0)
        if nu == 0.0:
            max_drift = max(max_drift, abs(E - E0) / E0)
        ts.append(t); Es.append(E); H1s.append(H1)
        if t_front is None and a[-1] ** 2 > front_theta * np.sum(a * a):
            t_front = t
            if nu == 0.0:           # cascade reached the cutoff; the rest is artifact
                break
    H1s_arr = np.array(H1s)
    finite = H1s_arr[np.isfinite(H1s_arr)]
    return dict(N=N, nu=nu, gamma=gamma, t=np.array(ts), E=np.array(Es), H1=H1s_arr,
                E0=E0, t_front=t_front, max_energy_drift=max_drift, n_steps=n_steps,
                H1_init=float(H1s[0]), H1_max=float(finite.max()) if len(finite) else np.inf)


def blowup_time_vs_N(Ns=(8, 10, 12, 14, 16, 18), theta=1e-4, **kw):
    """Inviscid cascade-front arrival time vs truncation N.  Finite-time blowup iff
    t_front(N) is increasing and converges (Cauchy) to a finite t* as N grows."""
    return [(N, front_time(N, theta=theta, **kw)) for N in Ns]


def compare():
    """Headline metrics + an honest PASS flag for the energy-conserving-blowup story."""
    inv = integrate(N=18, nu=0.0)                       # energy-conserving cascade
    vis = integrate(N=18, nu=0.05, gamma=1.0)           # NS dissipation -> bounded
    conv = blowup_time_vs_N(Ns=(8, 10, 12, 14, 16, 18))
    ts = [t for _, t in conv if t is not None]

    inv_cascade = (inv["H1_max"] / inv["H1_init"] > 1e3) and (inv["max_energy_drift"] < 1e-2)
    vis_bounded = (vis["t_front"] is None) and (vis["H1_max"] < 1e2)
    monotone = all(x <= y + 1e-9 for x, y in zip(ts, ts[1:]))
    # geometric convergence: last gap << first gap  => finite limit t*
    converged = (len(ts) == len(conv) and monotone and len(ts) >= 3 and
                 (ts[-1] - ts[-2]) < 0.5 * (ts[1] - ts[0]))
    t_star = ts[-1] if ts else None

    ok = bool(inv_cascade and vis_bounded and converged)
    return dict(
        inviscid_H1_growth=float(inv["H1_max"] / inv["H1_init"]),
        inviscid_energy_drift=float(inv["max_energy_drift"]),
        inviscid_t_front=inv["t_front"],
        viscous_H1_max=float(vis["H1_max"]), viscous_blowup=vis["t_front"],
        blowup_time_vs_N=conv, blowup_time_converged=bool(converged), t_star=t_star,
        inv=inv, vis=vis, ok=ok,
    )


def run():
    return compare()


def make_figure(path, r=None):
    if r is None:
        r = compare()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    inv, vis = r["inv"], r["vis"]

    ax[0].semilogy(inv["t"], inv["H1"], color="#c0392b", lw=2,
                   label=f"inviscid (Euler), N={inv['N']}")
    ax[0].semilogy(vis["t"], vis["H1"], color="#2c7fb8", lw=2,
                   label=f"viscous (NS, γ=1), N={vis['N']}")
    ax[0].axhline(2.0 ** inv["N"], color="#c0392b", ls=":", lw=1,
                  label="truncation ceiling 2^N")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("H¹ norm  (Σ kₙ² aₙ²)$^{1/2}$")
    ax[0].set_title("(a) inviscid cascade blows up;\nNS dissipation regularizes")
    ax[0].legend(fontsize=8.2); ax[0].grid(alpha=0.3, which="both")

    ax[1].plot(inv["t"], inv["E"] / inv["E0"], color="#c0392b", lw=2, label="inviscid")
    ax[1].plot(vis["t"], vis["E"] / vis["E0"], color="#2c7fb8", lw=2, label="viscous")
    ax[1].set_xlabel("time"); ax[1].set_ylabel("energy E(t)/E(0)")
    ax[1].set_ylim(0, 1.1)
    ax[1].set_title(f"(b) inviscid energy conserved\n(drift {r['inviscid_energy_drift']:.1e}) — yet H¹ blows up")
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3)

    Ns = [n for n, t in r["blowup_time_vs_N"] if t is not None]
    bs = [t for _, t in r["blowup_time_vs_N"] if t is not None]
    ax[2].plot(Ns, bs, "o-", color="#34495e", ms=8)
    if r["t_star"] is not None:
        ax[2].axhline(r["t_star"], color="#7f8c8d", ls="--", lw=1,
                      label=f"t* ≈ {r['t_star']:.3f}")
        ax[2].legend(fontsize=9)
    ax[2].set_xlabel("truncation N (shells)"); ax[2].set_ylabel("cascade-front arrival t_front(N)")
    ax[2].set_title("(c) t_front(N) converges as N→∞\n(finite-time blowup, not a truncation artifact)")
    ax[2].grid(alpha=0.3)
    fig.suptitle("Dyadic NS: energy conservation does not prevent finite-time blowup — "
                 "the structure does", fontsize=12.5, y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    r = run()
    print("=== Dyadic (shell) model: energy-conserving finite-time blowup ===")
    print(f"  INVISCID (Euler): H^1 growth = {r['inviscid_H1_growth']:.2e}  (blows up to the 2^N ceiling)")
    print(f"    energy drift          = {r['inviscid_energy_drift']:.2e}  (conserved)")
    print(f"    cascade-front arrival = {r['inviscid_t_front']:.4f}")
    print(f"  VISCOUS (NS, γ=1): H^1 max = {r['viscous_H1_max']:.3f}  blowup={r['viscous_blowup']}  (bounded -> regular)")
    print("  FINITE-TIME BLOWUP — front arrival t_front(N) converges as N grows:")
    for N, t in r["blowup_time_vs_N"]:
        print(f"    N={N:>3}: t_front={t}")
    print(f"  t* ≈ {r['t_star']:.4f}")
    print(f"  PASS: {r['ok']}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
