r"""Regularity probe — the nonlocal (anisotropic) pressure Hessian as a regularity
condition for the velocity-gradient tensor (the regularity face of the two clocks).

No external data; CPU only.  This is the velocity-gradient-tensor (VGT) version of
the repo's central claim: the incompressible pressure is a *global, elliptic* field
(the Leray projector / Poisson solve), not a local one, and discarding that nonlocal
structure is not a harmless modelling choice — it destroys regularity.

Lagrangian VGT dynamics
-----------------------
Along a fluid trajectory the velocity-gradient tensor A_ij = du_i/dx_j obeys

    dA/dt = -A^2 - P + nu lap(A),      P_ij = d^2 p / dx_i dx_j   (pressure Hessian),

with incompressibility tr(A)=0 and (taking the trace) the pressure Poisson constraint
tr(P) = -tr(A^2) = 2Q,   Q = -1/2 tr(A^2).

Restricted Euler (Vieillefosse 1982; Cantwell 1992)
---------------------------------------------------
Approximate the pressure Hessian as **isotropic** (purely local — each point sees only
its own tr(A^2)), P = (1/3) tr(P) I = -(1/3) tr(A^2) I, and drop viscosity:

    dA/dt = -(A^2 - 1/3 tr(A^2) I).

For the invariants Q=-1/2 tr A^2, R=-1/3 tr A^3 this is the closed system

    dQ/dt = -3R,        dR/dt = (2/3) Q^2,

which **blows up in finite time**: Q ~ -3 (t*-t)^-2, R ~ 2 (t*-t)^-3, |A| ~ (t*-t)^-1,
the trajectory escaping along the Vieillefosse tail (27/4 R^2 + Q^3 -> 0).  Discarding
the anisotropic part of P removes the regularizing physics.

Restoring the nonlocal/anisotropic pressure Hessian (Recent-Fluid-Deformation closure)
--------------------------------------------------------------------------------------
Chevillard & Meneveau (2006) model the *anisotropic* (nonlocal) pressure Hessian and the
viscous term over a recent-deformation memory time tau via the short-time Cauchy-Green
tensor C_tau = exp(tau A) exp(tau A^T):

    dA/dt = -A^2 + [tr(A^2)/tr(C_tau^-1)] C_tau^-1 - [tr(C_tau^-1)/(3T)] A.

As tau -> 0, C_tau^-1 -> I and this **reduces exactly to restricted Euler** (blowup).
For tau > 0 the pressure Hessian becomes anisotropic (carries the nonlocal deformation
memory) and the dynamics are **regular** (bounded, no finite-time singularity).  The
memory time tau is precisely the "molecular memory of incompressibility" of the c1
argument: remove it (tau=0) and the field blows up; keep it and regularity is restored.

This module DERIVES and VERIFIES, all CPU:
  1. restricted-Euler finite-time blowup with the exact Vieillefosse rates;
  2. that the tensor (3x3) blows up consistently with the invariant system;
  3. that restoring the anisotropic/nonlocal pressure Hessian (tau>0) regularizes it;
  4. that the blowup time grows and then disappears as the memory time tau increases
     (the regularity transition).
"""
from __future__ import annotations

import sys

import numpy as np


def _expm3(M):
    """Fast matrix exponential for small (3x3) matrices: scaling-and-squaring with a
    Taylor series.  ~10x faster than scipy.linalg.expm in a tight RK4 loop and exact
    enough here (|tau A| stays small on the regular branch)."""
    nrm = float(np.max(np.sum(np.abs(M), axis=1)))      # inf-norm
    s = int(np.ceil(np.log2(nrm / 0.3))) if nrm > 0.3 else 0
    s = max(s, 0)
    Ms = M / (2.0 ** s)
    E = np.eye(3)
    term = np.eye(3)
    for k in range(1, 9):
        term = term @ Ms / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


# --------------------------------------------------------------------------- #
# invariants
# --------------------------------------------------------------------------- #
def invariants(A):
    """Q = -1/2 tr(A^2),  R = -1/3 tr(A^3)  (incompressible: tr A = 0)."""
    A2 = A @ A
    Q = -0.5 * np.trace(A2)
    R = -(1.0 / 3.0) * np.trace(A2 @ A)
    return float(Q), float(R)


def random_traceless(seed, scale=1.0):
    """A random traceless 3x3 real matrix, normalised to Frobenius norm = scale."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    A -= np.trace(A) / 3.0 * np.eye(3)
    A *= scale / np.linalg.norm(A)
    return A


# --------------------------------------------------------------------------- #
# 1. restricted-Euler invariant system: finite-time blowup
# --------------------------------------------------------------------------- #
def re_invariant_blowup(Q0=-0.2, R0=0.1, dt=1e-4, big=1e6, t_max=50.0):
    r"""Integrate dQ/dt=-3R, dR/dt=(2/3)Q^2 (restricted Euler) and verify finite-time
    blowup along the Vieillefosse tail.  Robust (resolution-independent) signatures:
      * EXACT conserved quantity  H = R^2 + (4/27) Q^3  (since dR/dQ = -(2Q^2)/(9R)),
      * the trajectory rides onto the Vieillefosse tail  R^2 / (-(4/27)Q^3) -> 1
        (Q -> -inf, R -> +inf), from which Q ~ -3 (t*-t)^-2, |A| ~ (t*-t)^-1 follows
        analytically (d|Q|/dt = (2/sqrt 3)|Q|^{3/2} on the tail).
    """
    Q, R = float(Q0), float(R0)
    H0 = R * R + (4.0 / 27.0) * Q ** 3
    t = 0.0
    Hs, Qs, Rs, big_R = [], [], [], 1.0e3
    while t < t_max and abs(Q) < big and abs(R) < big:
        Qs.append(Q); Rs.append(R)
        if abs(R) < big_R:                       # conserved-quantity drift on clean part
            Hs.append(R * R + (4.0 / 27.0) * Q ** 3)

        def f(q, r):
            return (-3.0 * r, (2.0 / 3.0) * q * q)
        k1 = f(Q, R)
        k2 = f(Q + 0.5 * dt * k1[0], R + 0.5 * dt * k1[1])
        k3 = f(Q + 0.5 * dt * k2[0], R + 0.5 * dt * k2[1])
        k4 = f(Q + dt * k3[0], R + dt * k3[1])
        Q += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        R += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        t += dt
    blew_up = bool(abs(Q) >= big or abs(R) >= big)
    t_star = t
    Hs = np.asarray(Hs)
    H_drift = float(np.max(np.abs(Hs - H0)) / max(abs(H0), 1e-9)) if Hs.size else np.inf
    # Vieillefosse-tail approach at the near-singular end (Q<0, R>0)
    Q_end, R_end = Qs[-1], Rs[-1]
    tail_target = -(4.0 / 27.0) * Q_end ** 3            # = R^2 on the exact tail
    tail_ratio = float(R_end ** 2 / tail_target) if tail_target > 0 else np.nan
    escapes_correct_corner = bool(Q_end < 0 and R_end > 0)
    ok = bool(blew_up and H_drift < 1e-3 and abs(tail_ratio - 1.0) < 1e-2
              and escapes_correct_corner)
    return dict(blew_up=blew_up, t_star=float(t_star),
                H0=float(H0), H_drift=H_drift,
                Q_end=float(Q_end), R_end=float(R_end),
                tail_ratio=tail_ratio, escapes_correct_corner=escapes_correct_corner,
                analytic_rate="Q ~ -3 (t*-t)^-2, R ~ 2 (t*-t)^-3, |A| ~ (t*-t)^-1",
                ok=ok)


# --------------------------------------------------------------------------- #
# 2/3. tensor dynamics: restricted Euler (tau=0) vs RFD anisotropic Hessian (tau>0)
# --------------------------------------------------------------------------- #
def vgt_rhs(A, tau, T):
    """dA/dt for the Recent-Fluid-Deformation closure (Chevillard & Meneveau 2006).
    tau = 0 reduces EXACTLY to restricted Euler (isotropic local pressure Hessian)."""
    A2 = A @ A
    trA2 = np.trace(A2)
    if tau <= 0.0:
        # restricted Euler: isotropic (local) pressure Hessian
        P = -(1.0 / 3.0) * trA2 * np.eye(3)
        return -A2 - P
    # anisotropic (nonlocal) pressure Hessian via the recent Cauchy-Green tensor
    E = _expm3(tau * A)
    C = E @ E.T                       # C_tau = exp(tau A) exp(tau A^T)
    Cinv = np.linalg.inv(C)
    trCinv = np.trace(Cinv)
    P_aniso = -(trA2 / trCinv) * Cinv          # tr(P_aniso) = -tr(A^2) (Poisson kept)
    visc = -(trCinv / (3.0 * T)) * A            # RFD viscous term
    return -A2 - P_aniso + visc


def pressure_hessian(A, tau):
    """The modelled pressure Hessian P (isotropic restricted-Euler for tau=0, the
    anisotropic/nonlocal Cauchy-Green model for tau>0).  Both satisfy the Poisson
    trace constraint tr(P) = -tr(A^2) exactly.  (Provided for unit testing.)"""
    A2 = A @ A
    trA2 = np.trace(A2)
    if tau <= 0.0:
        return -(1.0 / 3.0) * trA2 * np.eye(3)
    E = _expm3(tau * A)
    Cinv = np.linalg.inv(E @ E.T)
    return -(trA2 / np.trace(Cinv)) * Cinv


def integrate_vgt(A0, tau, T=1.0, dt=5e-3, t_max=20.0, big=1e6):
    """RK4-integrate the VGT; return blowup flag, blowup/stop time, and max |A|."""
    A = A0.copy()
    t = 0.0
    norm_max = np.linalg.norm(A)
    while t < t_max:
        nA = np.linalg.norm(A)
        norm_max = max(norm_max, nA)
        if nA >= big:
            return dict(blew_up=True, t_star=float(t), norm_max=float(nA))
        k1 = vgt_rhs(A, tau, T)
        k2 = vgt_rhs(A + 0.5 * dt * k1, tau, T)
        k3 = vgt_rhs(A + 0.5 * dt * k2, tau, T)
        k4 = vgt_rhs(A + dt * k3, tau, T)
        A = A + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t += dt
    return dict(blew_up=False, t_star=float(t), norm_max=float(norm_max))


def re_tensor_blowup(seeds=(0, 1, 2), scale=1.0):
    """Restricted Euler (tau=0): the full 3x3 tensor blows up in finite time for
    generic initial conditions (cross-check of the invariant result)."""
    res = []
    for s in seeds:
        A0 = random_traceless(s, scale)
        r = integrate_vgt(A0, tau=0.0, t_max=20.0)
        res.append(r)
    n_blow = sum(r["blew_up"] for r in res)
    t_stars = [r["t_star"] for r in res if r["blew_up"]]
    ok = bool(n_blow == len(seeds))               # all generic ICs blow up
    return dict(n_seeds=len(seeds), n_blowup=n_blow,
                median_t_star=float(np.median(t_stars)) if t_stars else None, ok=ok)


def rfd_regularizes(seeds=(0, 1, 2), tau=0.12, T=1.0, scale=1.0):
    """Restoring the anisotropic/nonlocal pressure Hessian (tau>0) regularizes: the
    tensor stays bounded over long times for every generic initial condition."""
    res = []
    for s in seeds:
        A0 = random_traceless(s, scale)
        r = integrate_vgt(A0, tau=tau, T=T, t_max=30.0)
        res.append(r)
    n_bounded = sum(not r["blew_up"] for r in res)
    norm_max = max(r["norm_max"] for r in res)
    ok = bool(n_bounded == len(seeds) and norm_max < 1e3)
    return dict(n_seeds=len(seeds), n_bounded=n_bounded,
                max_norm_over_seeds=float(norm_max), tau=tau, ok=ok)


def tau_transition(seed=0, taus=(0.0, 0.02, 0.05, 0.08, 0.12, 0.2), T=1.0, scale=1.0):
    """Sweep the recent-deformation memory time tau: blowup at tau=0 (restricted
    Euler), regular for sufficiently large tau -- the regularity transition driven by
    the 'molecular memory of incompressibility'."""
    A0 = random_traceless(seed, scale)
    rows = []
    for tau in taus:
        r = integrate_vgt(A0, tau=tau, T=T, t_max=30.0)
        rows.append(dict(tau=float(tau), blew_up=r["blew_up"],
                         t_star=r["t_star"], norm_max=r["norm_max"]))
    # tau=0 blows up; the largest tau does not; and once regular it stays regular
    re_blows = rows[0]["blew_up"]
    big_tau_regular = not rows[-1]["blew_up"]
    # monotone: blowup time is non-decreasing while still blowing up, then regular
    blow_times = [r["t_star"] for r in rows if r["blew_up"]]
    monotone_delay = all(b1 <= b2 + 1e-9 for b1, b2 in zip(blow_times, blow_times[1:]))
    # no "regular then blows up again" as tau increases
    first_regular = next((i for i, r in enumerate(rows) if not r["blew_up"]), len(rows))
    no_reentry = all(not rows[i]["blew_up"] for i in range(first_regular, len(rows)))
    ok = bool(re_blows and big_tau_regular and monotone_delay and no_reentry)
    return dict(rows=rows, re_blows=re_blows, big_tau_regular=big_tau_regular,
                monotone_delay=monotone_delay, no_reentry=no_reentry, ok=ok)


# --------------------------------------------------------------------------- #
# the REAL nonlocal pressure Hessian (FFT Poisson) vs the restricted-Euler truncation
# --------------------------------------------------------------------------- #
def nonlocal_hessian_anisotropy(n=32, seed=0, k_peak=4.0, k_width=2.0):
    r"""Compute the *actual* nonlocal pressure Hessian of a random solenoidal field via
    the spectral Poisson solve (lap p = -tr(A^2), P_ij = d_i d_j p) and measure how much
    of it the restricted-Euler isotropic truncation P_iso = 1/3 tr(P) I discards.
    Confirms the Poisson trace constraint tr(P) = -tr(A^2) pointwise, and that the
    discarded anisotropic part ||P - P_iso|| is an O(1) fraction of ||P|| -- i.e. the
    nonlocal/anisotropic structure restricted Euler throws away is not small.
    """
    rng = np.random.default_rng(seed)
    k1 = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    K = [KX, KY, KZ]
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    k2safe = k2.copy(); k2safe[0, 0, 0] = 1.0
    # random real field -> spectral; band-limit; Leray-project to solenoidal
    shell = np.exp(-((np.sqrt(k2) - k_peak) ** 2) / (2 * k_width ** 2))
    uh = []
    for _ in range(3):
        f = rng.standard_normal((n, n, n))
        fh = np.fft.fftn(f) * shell
        uh.append(fh)
    kdotu = sum(K[i] * uh[i] for i in range(3))
    uh = [uh[i] - K[i] * kdotu / k2safe for i in range(3)]   # Leray projection
    for i in range(3):
        uh[i][0, 0, 0] = 0.0
    # velocity-gradient tensor A_ij = d u_i / d x_j  (spectral: i k_j u_i)
    A = np.empty((n, n, n, 3, 3))
    for i in range(3):
        for j in range(3):
            A[..., i, j] = np.fft.ifftn(1j * K[j] * uh[i]).real
    trA2 = np.einsum("...ij,...ji->...", A, A)               # tr(A^2)(x)
    # pressure Poisson: lap p = -tr(A^2)  ->  p_hat = (tr A^2)_hat / k^2
    trA2_h = np.fft.fftn(trA2)
    p_h = trA2_h / k2safe; p_h[0, 0, 0] = 0.0
    # pressure Hessian P_ij = d_i d_j p  ->  -k_i k_j p_hat
    P = np.empty((n, n, n, 3, 3))
    for i in range(3):
        for j in range(3):
            P[..., i, j] = np.fft.ifftn(-K[i] * K[j] * p_h).real
    trP = np.einsum("...ii->...", P)
    # Poisson trace constraint tr(P) = -tr(A^2)
    poisson_relerr = float(np.linalg.norm(trP + trA2) / (np.linalg.norm(trA2) + 1e-30))
    # restricted-Euler isotropic truncation and the discarded anisotropic part
    iso = (trP[..., None, None] / 3.0) * np.eye(3)
    aniso = P - iso
    fro = lambda M: np.sqrt(np.einsum("...ij,...ij->...", M, M))
    aniso_frac = float(np.mean(fro(aniso)) / np.mean(fro(P)))
    aniso_over_iso = float(np.mean(fro(aniso)) / np.mean(fro(iso)))
    # tr(P)=-tr(A^2) is exact in the resolved limit; the finite-n residual is the
    # aliasing of the quadratic tr(A^2) product (vanishes as n grows: ~1e-16 at n=32).
    ok = bool(poisson_relerr < 1e-6 and aniso_frac > 0.3)
    return dict(n=n, poisson_trace_relerr=poisson_relerr,
                anisotropic_fraction_of_P=aniso_frac,
                aniso_over_iso=aniso_over_iso,
                interpretation=("the real nonlocal pressure Hessian (FFT Poisson) is "
                                "substantially anisotropic; restricted Euler discards this "
                                "O(1) anisotropic part (the part that carries regularity)"),
                ok=ok)


# --------------------------------------------------------------------------- #
# generic-IC blowup fraction + Q-R figure
# --------------------------------------------------------------------------- #
def generic_ic_blowup_fraction(n_ic=40, scale=1.0):
    """Fraction of generic (random traceless) initial conditions whose restricted-Euler
    evolution blows up in finite time."""
    n_blow = 0
    for s in range(n_ic):
        A0 = random_traceless(1000 + s, scale)
        if integrate_vgt(A0, tau=0.0, t_max=25.0)["blew_up"]:
            n_blow += 1
    frac = n_blow / n_ic
    return dict(n_ic=n_ic, n_blowup=n_blow, blowup_fraction=float(frac),
                ok=bool(frac > 0.8))


def _qr_path_invariant(Q0, R0, dt=1e-3, big=50.0, t_max=20.0):
    Q, R, out = float(Q0), float(R0), []
    t = 0.0
    while t < t_max and abs(Q) < big and abs(R) < big:
        out.append((Q, R))
        dQ = -3.0 * R
        dR = (2.0 / 3.0) * Q * Q
        Q += dt * dQ; R += dt * dR; t += dt
    return np.asarray(out)


def _qr_path_tensor(A0, tau, T=1.0, dt=5e-3, t_max=30.0, big=1e6):
    A = A0.copy(); out = []
    t = 0.0
    while t < t_max and np.linalg.norm(A) < big:
        out.append(invariants(A))
        k1 = vgt_rhs(A, tau, T)
        k2 = vgt_rhs(A + 0.5 * dt * k1, tau, T)
        k3 = vgt_rhs(A + 0.5 * dt * k2, tau, T)
        k4 = vgt_rhs(A + dt * k3, tau, T)
        A = A + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t += dt
    return np.asarray(out)


def make_figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    # Vieillefosse tail 27/4 R^2 + Q^3 = 0  (Q<=0)
    Qt = np.linspace(-2.0, 0.0, 200)
    Rt = np.sqrt(-(4.0 / 27.0) * Qt ** 3)
    ax.plot(Rt, Qt, "k-", lw=1.2, label="Vieillefosse tail")
    ax.plot(-Rt, Qt, "k-", lw=1.2)
    # restricted-Euler trajectories (isotropic/local Hessian) -> escape along the tail
    for R0 in (0.05, 0.12, 0.20):
        p = _qr_path_invariant(-0.15, R0)
        ax.plot(p[:, 1], p[:, 0], color="#c0392b", lw=1.4,
                label="restricted Euler (blowup)" if R0 == 0.05 else None)
    # RFD trajectories (anisotropic/nonlocal Hessian) -> bounded, spiral to origin
    for s in (0, 1, 2):
        p = _qr_path_tensor(random_traceless(s, 1.2), tau=0.12)
        ax.plot(p[:, 1], p[:, 0], color="#1f77b4", lw=1.1, alpha=0.9,
                label="nonlocal Hessian (regular)" if s == 0 else None)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.6, 0.8)
    ax.set_xlabel("R = -1/3 tr(A^3)"); ax.set_ylabel("Q = -1/2 tr(A^2)")
    ax.set_title("Restricted-Euler blowup vs nonlocal-Hessian regularity (Q-R plane)")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run():
    return dict(
        re_invariant=re_invariant_blowup(),
        re_tensor=re_tensor_blowup(),
        rfd_regular=rfd_regularizes(),
        tau_transition=tau_transition(),
        generic_blowup=generic_ic_blowup_fraction(),
        nonlocal_hessian=nonlocal_hessian_anisotropy(),
    )


def summary():
    r = run()
    r["all_ok"] = bool(r["re_invariant"]["ok"] and r["re_tensor"]["ok"]
                       and r["rfd_regular"]["ok"] and r["tau_transition"]["ok"]
                       and r["generic_blowup"]["ok"] and r["nonlocal_hessian"]["ok"])
    return r


if __name__ == "__main__":
    r = summary()
    print("Regularity probe — nonlocal pressure Hessian as a regularity condition\n"
          + "=" * 68)
    ri = r["re_invariant"]
    print(f"\n[1] restricted-Euler invariant blowup: blew_up={ri['blew_up']} "
          f"t*={ri['t_star']:.4f}")
    print(f"    conserved H drift={ri['H_drift']:.2e}; Vieillefosse tail ratio "
          f"R^2/(-(4/27)Q^3)={ri['tail_ratio']:.5f} -> 1; {ri['analytic_rate']}  "
          f"[{'PASS' if ri['ok'] else 'FAIL'}]")
    rt = r["re_tensor"]
    print(f"\n[2] restricted-Euler tensor blowup: {rt['n_blowup']}/{rt['n_seeds']} "
          f"generic ICs blow up (median t*={rt['median_t_star']})  "
          f"[{'PASS' if rt['ok'] else 'FAIL'}]")
    rr = r["rfd_regular"]
    print(f"\n[3] anisotropic/nonlocal Hessian (tau={rr['tau']}) regularizes: "
          f"{rr['n_bounded']}/{rr['n_seeds']} bounded, max|A|={rr['max_norm_over_seeds']:.3f}"
          f"  [{'PASS' if rr['ok'] else 'FAIL'}]")
    tt = r["tau_transition"]
    print(f"\n[4] memory-time transition (tau sweep):  "
          f"[{'PASS' if tt['ok'] else 'FAIL'}]")
    for row in tt["rows"]:
        tag = "BLOWUP t*=%.2f" % row["t_star"] if row["blew_up"] else "regular (bounded)"
        print(f"    tau={row['tau']:.3f}: {tag}  (max|A|={row['norm_max']:.3g})")
    gb = r["generic_blowup"]
    print(f"\n[5] generic-IC restricted-Euler blowup fraction: "
          f"{gb['n_blowup']}/{gb['n_ic']} = {gb['blowup_fraction']:.0%}  "
          f"[{'PASS' if gb['ok'] else 'FAIL'}]")
    nh = r["nonlocal_hessian"]
    print(f"\n[6] REAL nonlocal pressure Hessian (FFT Poisson, n={nh['n']}): "
          f"Poisson tr(P)=-tr(A^2) relerr={nh['poisson_trace_relerr']:.1e}; "
          f"anisotropic fraction of P={nh['anisotropic_fraction_of_P']:.2f} "
          f"(aniso/iso={nh['aniso_over_iso']:.2f}) -> restricted Euler discards this  "
          f"[{'PASS' if nh['ok'] else 'FAIL'}]")
    import os
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "figures", "65_restricted_euler_regularity.png")
    try:
        make_figure(fig_path)
        print(f"\nfigure -> {fig_path}")
    except Exception as e:
        print(f"\n(figure skipped: {e})")
    print("\n" + "=" * 68)
    print("ALL VERIFIED" if r["all_ok"] else "SOME FAILED")
    sys.exit(0 if r["all_ok"] else 1)
