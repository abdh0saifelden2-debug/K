r"""§V.5d — the cavity↔channel Mori–Zwanzig memory kernel is a *distributed*
operator projection, not a 2×2 linear-algebra identity (paper4a §4.5).

Motivation
----------
`hydraulic_mz_projection_synthetic.py` (§V.5b) proves the surge-lag memory of the
*lumped* cavity↔channel system is an exact Mori–Zwanzig kernel `K(τ)=−ab e^{−τ/τ₂}`.
A fair, recurring objection (paper4a §4.4; referee) is that eliminating one variable
of a 2×2 system is *just a linear-algebra identity*.  `hydraulic_nonlinear_kernel.py`
(§V.5c) answers half of it (the linear kernel is the small-amplitude limit of the
nonlinear Röthlisberger system).  This module answers the other half — **the spatial
half** — listed as the open gate in paper4a §8 ("a spatially-resolved coupled
GlaDS/Röthlisberger PDE derivation remains future work").

The distributed (1-D flowline) model
------------------------------------
Two coupled *fields* on a periodic flowline `x∈[0,L)` (anomalies about a steady
drainage state); `s(x,t)` the **resolved** cavity store (water-pressure anomaly),
`q(x,t)` the **eliminated** channel variable (cross-section / flux anomaly):

    ∂ₜs = D_s ∂ₓₓ s − s/τ₁ − a q                          (cavity)
    ∂ₜq = D_q ∂ₓₓ q − U ∂ₓ q + b s − q/τ₂                 (channel: along-flow transport)

Channels *route water downstream* (advection `U`) and spread it (`D_q`); the creep
closure sets the channel relaxation `τ₂` (Röthlisberger 1972; Werder et al. 2013;
Hewitt 2013).  Discretising `x` on `N` nodes gives `Ẋ = M X`, `X=(S,Q)∈ℝ²ᴺ`,
`M=[[A_ss,A_sq],[A_qs,A_qq]]` with `A_ss=D_s∇²−I/τ₁`, `A_sq=−aI`, `A_qs=bI`,
`A_qq=D_q∇²−U∂ₓ−I/τ₂`.

Exact Mori projection of the channel *field* (linear Nakajima–Zwanzig, no Markov):

    Ṡ(t) = A_ss S(t) + ∫₀ᵗ 𝒦(t−τ) S(τ) dτ + R(t),
    𝒦(τ) = A_sq e^{A_qq τ} A_qs            (a matrix-valued / non-local-in-space kernel)
    R(t) = A_sq e^{A_qq t} Q(0).

Checks (all machine-precision or a clean limit)
-----------------------------------------------
  A. **Projection exact (operator Schur complement).**  In Laplace space the GLE
     transfer operator `[(sI−A_ss) − A_sq(sI−A_qq)⁻¹A_qs]⁻¹` maps `S₀+A_sq(sI−A_qq)⁻¹Q₀`
     to exactly the (S-block of the) full resolvent `(sI−M)⁻¹X₀`, at random complex `s`
     for the spatially-coupled operator — max rel err ≤ 1e-10.  So the *distributed*
     projection is exact; the kernel is the exact projected operator.
  B. **The kernel IS the eliminated channel operator's Green's function**
     `A_sq e^{A_qq τ} A_qs` (vs direct `expm`, ≤ 1e-12), and it is **spatially
     non-local** with a resolved along-flow range `ℓ_mem ≫ Δx`.
  C. **The reduced field-GLE reproduces the full PDE trajectory** (slow-channel
     regime) to rel-err ≤ 1e-4 — the channel-eliminated distributed model *is* the
     resolved dynamics.
  D. **Memory is necessary, as a dose-response.**  The memoryless local closure
     `Ṡ=(A_ss+𝒦_DC)S` (𝒦_DC=∫₀^∞𝒦=−A_sq A_qq⁻¹A_qs) incurs a trajectory error that
     grows monotonically with the channel time `τ₂` and → 0 as `τ₂→0` (the Markovian
     limit recovered from the projection, now in the distributed system).
  E. **The lumped 2×2 kernel is the no-transport, single-node limit.**  With
     `D_q=U=0` the distributed kernel `𝒦(τ)` is **exactly diagonal** and its diagonal
     equals the committed lumped `−ab e^{−τ/τ₂}` to machine precision (≤ 1e-12); turning
     channel transport on moves the kernel mass off-diagonal (spatial non-locality
     grows with `U,D_q`), and the along-flow memory range scales as `ℓ_mem ~ Uτ₂+√(D_qτ₂)`.

So the §V.5b 2×2 identity is the *spatially-local, single-node* limit of an exact
projection of a genuine distributed (PDE) operator — and the distributed system makes
a new, falsifiable structural prediction: the surge-lag memory has a finite along-flow
footprint `ℓ_mem` set by channel transport, which a memoryless (local) closure cannot
produce.  This validates the projection *structure* of the linearised distributed
model; the fully nonlinear spatially-resolved coupled PDE remains future work.

No external data; CPU only.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


# --- the distributed operator ------------------------------------------------
def ring_ops(N, dx):
    """Periodic 1-D Laplacian and central first-difference on N nodes."""
    i = np.arange(N)
    I = np.eye(N)
    Lap = (-2.0 * I + I[(i + 1) % N] + I[(i - 1) % N]) / dx ** 2
    D1 = (I[(i + 1) % N] - I[(i - 1) % N]) / (2 * dx)
    return Lap, D1


def build(N=24, L=1.0, Ds=0.005, tau1=2.0, Dq=0.02, U=0.6, tau2=0.5, a=0.15, b=0.30):
    """Assemble the 2N×2N distributed cavity↔channel generator M and its blocks."""
    dx = L / N
    Lap, D1 = ring_ops(N, dx)
    I = np.eye(N)
    A_ss = Ds * Lap - I / tau1
    A_sq = -a * I
    A_qs = b * I
    A_qq = Dq * Lap - U * D1 - I / tau2
    M = np.block([[A_ss, A_sq], [A_qs, A_qq]])
    return {"N": N, "L": L, "dx": dx, "A_ss": A_ss, "A_sq": A_sq, "A_qs": A_qs,
            "A_qq": A_qq, "M": M, "Ds": Ds, "tau1": tau1, "Dq": Dq, "U": U,
            "tau2": tau2, "a": a, "b": b}


def kernel_mats(P, t):
    """𝒦(τ)=A_sq e^{A_qq τ} A_qs at each τ in `t` (eigendecomposition of A_qq)."""
    lam, W = np.linalg.eig(P["A_qq"])
    Winv = np.linalg.inv(W)
    A_sq, A_qs = P["A_sq"], P["A_qs"]
    out = np.empty((len(t), P["N"], P["N"]))
    for k, tk in enumerate(t):
        E = (W * np.exp(lam * tk)) @ Winv
        out[k] = (A_sq @ E @ A_qs).real
    return out


def memory_range(P, K_at_tau):
    """Along-flow 2nd-moment range ℓ_mem of the (real) kernel matrix at one τ:
    response at all x to a unit channel source at the mid-node, RMS spatial spread."""
    N, dx = P["N"], P["dx"]
    src = N // 2
    prof = np.abs(K_at_tau[:, src])
    prof = np.roll(prof, N // 2 - src)
    if prof.sum() == 0:
        return 0.0
    prof = prof / prof.sum()
    x = (np.arange(N) - N // 2) * dx
    return float(np.sqrt(np.sum(prof * x ** 2)))


# --- A. exact projection in Laplace space (operator Schur complement) --------
def check_projection_exact(P, seed=0, ntrials=40):
    rng = np.random.default_rng(seed)
    N, M = P["N"], P["M"]
    A_ss, A_sq, A_qs, A_qq = P["A_ss"], P["A_sq"], P["A_qs"], P["A_qq"]
    I = np.eye(N)
    max_err = 0.0
    for _ in range(ntrials):
        s = rng.uniform(0.3, 6.0) + 1j * rng.uniform(-3.0, 3.0)
        S0 = rng.normal(size=N)
        Q0 = rng.normal(size=N)
        X0 = np.concatenate([S0, Q0])
        full = np.linalg.solve(s * np.eye(2 * N) - M, X0)[:N]
        Rqq = np.linalg.inv(s * I - A_qq)
        GS = np.linalg.inv((s * I - A_ss) - A_sq @ Rqq @ A_qs)
        gle = GS @ (S0 + A_sq @ Rqq @ Q0)
        max_err = max(max_err, float(np.max(np.abs(full - gle)) /
                                   (np.max(np.abs(full)) + 1e-30)))
    return {"max_rel_err": max_err, "n_tested": ntrials, "pass": bool(max_err <= 1e-10)}


# --- B. kernel == channel Green's function + spatial non-locality ------------
def check_kernel_is_greens(P, T=12.0, n=2400):
    t = np.linspace(0.0, T, n)
    K = kernel_mats(P, t)
    err = 0.0
    for tk in (0.0, 0.3, 1.0, 3.0):
        k = int(tk / T * (n - 1))
        Kd = P["A_sq"] @ expm(P["A_qq"] * t[k]) @ P["A_qs"]
        err = max(err, float(np.max(np.abs(Kd - K[k]))))
    k2 = int(P["tau2"] / T * (n - 1))
    ell = memory_range(P, K[k2])
    resolved = ell > 5 * P["dx"]
    return {"kernel_vs_expm_max_err": err, "ell_mem_at_tau2": ell, "dx": P["dx"],
            "spatially_resolved_nonlocal": bool(resolved),
            "pass": bool(err <= 1e-12 and resolved)}


# --- C. reduced field-GLE reproduces the full PDE ----------------------------
def _full_traj(P, S0, t):
    N = P["N"]
    X0 = np.concatenate([S0, np.zeros(N)])
    lam, V = np.linalg.eig(P["M"])
    c = np.linalg.inv(V) @ X0
    S = np.empty((len(t), N))
    for k, tk in enumerate(t):
        S[k] = (V @ (np.exp(lam * tk) * c)).real[:N]
    return S


def _gle_traj(P, S0, t, K):
    N = P["N"]
    dt = t[1] - t[0]
    A_ss = P["A_ss"]
    S = np.zeros((len(t), N))
    S[0] = S0

    def mem(i, hist):
        if i == 0:
            return np.zeros(N)
        Ksub = K[i::-1][:i + 1]                  # 𝒦(t_i − t_j), j=0..i
        w = np.ones(i + 1)
        w[0] = 0.5
        w[-1] = 0.5
        return dt * np.einsum("k,kij,kj->i", w, Ksub, hist[:i + 1])

    for i in range(len(t) - 1):
        f0 = A_ss @ S[i] + mem(i, S[:i + 1])
        Spred = S[i] + dt * f0
        f1 = A_ss @ Spred + mem(i + 1, np.vstack([S[:i + 1], Spred]))
        S[i + 1] = S[i] + 0.5 * dt * (f0 + f1)
    return S


def _local_traj(P, S0, t):
    """Memoryless local closure Ṡ=(A_ss+𝒦_DC)S, 𝒦_DC=−A_sq A_qq⁻¹ A_qs=∫₀^∞𝒦."""
    G = P["A_ss"] - P["A_sq"] @ np.linalg.inv(P["A_qq"]) @ P["A_qs"]
    lam, V = np.linalg.eig(G)
    c = np.linalg.inv(V) @ S0
    S = np.empty((len(t), P["N"]))
    for k, tk in enumerate(t):
        S[k] = (V @ (np.exp(lam * tk) * c)).real
    return S


def _bump(P, width=1.5):
    x = np.arange(P["N"])
    src = P["N"] // 4
    return np.exp(-0.5 * ((x - src) / width) ** 2)


def check_gle_trajectory(T=10.0, n=1401):
    P = build(tau2=1.5)                          # slow channel: a non-trivial memory test
    t = np.linspace(0.0, T, n)
    S0 = _bump(P)
    K = kernel_mats(P, t)
    Sf = _full_traj(P, S0, t)
    Sg = _gle_traj(P, S0, t, K)
    rel = float(np.max(np.abs(Sg - Sf)) / np.max(np.abs(Sf)))
    return {"rel_err_trajectory": rel, "tau2": 1.5, "pass": bool(rel <= 1e-4)}


# --- D. memory necessity (dose-response in channel slowness) -----------------
def check_memory_necessity(T=10.0, n=1201, tau2_list=(0.05, 0.2, 0.5, 1.0, 2.0, 3.0)):
    t = np.linspace(0.0, T, n)
    errs = []
    for tau2 in tau2_list:
        P = build(tau2=tau2)
        S0 = _bump(P)
        Sf = _full_traj(P, S0, t)
        Sl = _local_traj(P, S0, t)
        errs.append(float(np.max(np.abs(Sl - Sf)) / np.max(np.abs(Sf))))
    errs = np.array(errs)
    monotone = bool(np.all(np.diff(errs) > -1e-12))
    vanishes_fast = bool(errs[0] < 1e-3)
    grows = bool(errs[-1] > 10 * errs[0])
    return {"tau2_list": list(tau2_list), "local_closure_err": errs.tolist(),
            "monotone_increasing": monotone, "vanishes_as_tau2_to_0": vanishes_fast,
            "grows_with_channel_slowness": grows,
            "pass": bool(monotone and vanishes_fast and grows)}


# --- E. lumped 2×2 = no-transport single-node limit; transport ⇒ non-locality -
def check_lumped_is_no_transport_limit(T=6.0, n=1200):
    a, b, tau2 = 0.15, 0.30, 0.5
    t = np.linspace(0.0, T, n)
    P0 = build(Dq=0.0, U=0.0, a=a, b=b, tau2=tau2)         # no channel transport
    K0 = kernel_mats(P0, t)
    k = 10
    offdiag = float(np.max(np.abs(K0[k] - np.diag(np.diag(K0[k])))))
    diagvals = np.array([K0[j][0, 0] for j in range(n)])
    lumped = -a * b * np.exp(-t / tau2)
    diag_err = float(np.max(np.abs(diagvals - lumped)))

    # transport ON ⇒ kernel mass moves off-diagonal; ℓ_mem grows with U and Dq
    offfrac, ells, configs = [], [], [(0.0, 0.0), (0.01, 0.3), (0.02, 0.6), (0.05, 1.0)]
    for Dq, U in configs:
        P = build(Dq=Dq, U=U)
        tt = np.linspace(0.0, 6.0, 1200)
        K = kernel_mats(P, tt)
        kk = int(P["tau2"] / 6.0 * 1199)
        Kt = np.abs(K[kk])
        offfrac.append(float((Kt.sum() - np.trace(Kt)) / Kt.sum()))
        ells.append(memory_range(P, K[kk]))
    nonlocality_grows = bool(offfrac[0] < 1e-9 and offfrac[1] > offfrac[0]
                             and offfrac[-1] > 0.5)
    ell_grows = bool(ells[-1] > ells[1] > ells[0])
    return {"no_transport_offdiag_max": offdiag,
            "no_transport_diag_vs_lumped_err": diag_err,
            "transport_configs_DqU": configs, "offdiag_fraction": offfrac,
            "ell_mem": ells, "nonlocality_grows_with_transport": nonlocality_grows,
            "ell_grows_with_transport": ell_grows,
            "pass": bool(offdiag <= 1e-12 and diag_err <= 1e-12
                         and nonlocality_grows and ell_grows)}


# --- F. the along-flow memory-footprint length law --------------------------
def memory_footprint_length_analytic(Dq, U, tau2):
    """Downstream e-folding length of the time-integrated (steady) memory
    influence `𝒦_DC(x)=ab·A_qq⁻¹δ`, i.e. the decaying root of the continuum
    operator `D_q∂ₓₓ−U∂ₓ−1/τ₂`:  ℓ = 2D_q/(√(U²+4D_q/τ₂)−U).
    Limits: U≫ → Uτ₂ (ballistic); U→0 → √(D_qτ₂) (diffusive)."""
    return 2.0 * Dq / (np.sqrt(U ** 2 + 4.0 * Dq / tau2) - U)


def _footprint_numeric(N, L, Dq, U, tau2, a=0.15, b=0.30):
    dx = L / N
    Lap, D1 = ring_ops(N, dx)
    I = np.eye(N)
    A_qq = Dq * Lap - U * D1 - I / tau2
    Kdc = a * b * np.linalg.inv(A_qq)               # 𝒦_DC = −A_sq A_qq⁻¹ A_qs = ab·A_qq⁻¹
    src = N // 2
    col = np.abs(Kdc[:, src])
    xs = (np.arange(src + 1, src + 1 + N // 3)) % N  # downstream side
    g = col[xs]
    g = g[g > 0]
    xfit = np.arange(len(g)) * dx
    slope = np.polyfit(xfit, np.log(g), 1)[0]
    return -1.0 / slope


def check_memory_footprint_law(N=400, L=4.0):
    """F. The numeric steady memory-influence decay length matches the analytic
    ℓ=2D_q/(√(U²+4D_q/τ₂)−U) across (D_q,U,τ₂), and the two limits hold."""
    cases = [(0.02, 0.0, 0.5), (0.02, 0.3, 0.5), (0.02, 0.6, 0.5),
             (0.02, 1.2, 0.5), (0.05, 0.6, 0.5), (0.02, 0.6, 1.5), (0.01, 0.0, 1.0)]
    rows, max_rel = [], 0.0
    for Dq, U, tau2 in cases:
        ea = memory_footprint_length_analytic(Dq, U, tau2)
        en = _footprint_numeric(N, L, Dq, U, tau2)
        rel = abs(ea - en) / ea
        max_rel = max(max_rel, rel)
        rows.append({"Dq": Dq, "U": U, "tau2": tau2, "ell_analytic": ea,
                     "ell_numeric": en, "rel_err": rel})
    # limits: U=0 diffusive √(Dq τ2); strongly advective → Uτ2
    diff = memory_footprint_length_analytic(0.02, 0.0, 0.5)
    diffusive_ok = abs(diff - np.sqrt(0.02 * 0.5)) / diff < 1e-9
    ea_big = memory_footprint_length_analytic(0.001, 3.0, 0.5)
    ballistic_ok = abs(ea_big - 3.0 * 0.5) / (3.0 * 0.5) < 0.02
    return {"cases": rows, "max_rel_err": max_rel,
            "diffusive_limit_ok": bool(diffusive_ok),
            "ballistic_limit_ok": bool(ballistic_ok),
            "pass": bool(max_rel <= 1e-2 and diffusive_ok and ballistic_ok)}


# --- orchestrator ------------------------------------------------------------
def run():
    P = build()
    ev = np.linalg.eigvals(P["M"])
    stable = bool(np.all(ev.real < 0))
    A = check_projection_exact(P)
    B = check_kernel_is_greens(P)
    C = check_gle_trajectory()
    D = check_memory_necessity()
    E = check_lumped_is_no_transport_limit()
    F = check_memory_footprint_law()
    out = {"what": "distributed (1-D flowline) cavity↔channel Mori–Zwanzig projection: "
                   "the lumped 2×2 kernel is its no-transport single-node limit; channel "
                   "transport makes the surge-lag memory kernel spatially non-local, with "
                   "an along-flow footprint ℓ=2D_q/(√(U²+4D_q/τ₂)−U)",
           "stable_generator": stable, "max_real_eig": float(ev.real.max()),
           "projection_exact": A, "kernel_is_greens": B, "gle_trajectory": C,
           "memory_necessity": D, "lumped_is_no_transport_limit": E,
           "memory_footprint_law": F}
    out["pass"] = bool(stable and A["pass"] and B["pass"] and C["pass"]
                       and D["pass"] and E["pass"] and F["pass"])
    return out


def make_figure(out, path="glaciers/validation/reports/78_hydraulic_mz_spatial.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = build()
    t = np.linspace(0.0, 6.0, 1200)
    K = kernel_mats(P, t)
    k2 = int(P["tau2"] / 6.0 * 1199)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (1) the spatial memory kernel matrix at τ=τ₂ (non-local off-diagonal spread)
    im = ax[0].imshow(np.abs(K[k2]), origin="lower", cmap="magma", aspect="auto")
    ax[0].set_title(r"$|\mathcal{K}(\tau{=}\tau_2)|$ — spatially non-local")
    ax[0].set_xlabel("channel source node $x'$")
    ax[0].set_ylabel("cavity response node $x$")
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    # (2) along-flow influence profile + ℓ_mem, vs the lumped (no-transport) δ-spike
    N, dx, src = P["N"], P["dx"], P["N"] // 2
    prof = np.roll(np.abs(K[k2][:, src]), N // 2 - src)
    prof = prof / prof.max()
    x = (np.arange(N) - N // 2) * dx
    ax[1].plot(x, prof, "o-", color="C3", label="distributed kernel")
    ax[1].axvline(0, color="k", ls=":", lw=1)
    spike = np.zeros(N); spike[N // 2] = 1.0
    ax[1].plot(x, spike, "s-", color="0.5", ms=4, label="lumped 2×2 (no transport)")
    ell = out["kernel_is_greens"]["ell_mem_at_tau2"]
    ax[1].set_title(rf"along-flow memory range $\ell_{{\rm mem}}\approx{ell:.2f}$")
    ax[1].set_xlabel("along-flow offset $x-x'$")
    ax[1].set_ylabel("memory weight (norm.)")
    ax[1].legend(fontsize=8)

    # (3) memory-necessity dose-response: local-closure error vs channel time τ₂
    D = out["memory_necessity"]
    ax[2].plot(D["tau2_list"], np.array(D["local_closure_err"]) * 100, "o-", color="C0")
    ax[2].set_xlabel(r"channel relaxation time $\tau_2$")
    ax[2].set_ylabel("memoryless-closure error (%)")
    ax[2].set_title("memory necessity vs channel time\n(→0 = Markovian limit)")
    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    import json
    import os
    out = run()
    A, B, C = out["projection_exact"], out["kernel_is_greens"], out["gle_trajectory"]
    D, E = out["memory_necessity"], out["lumped_is_no_transport_limit"]
    F = out["memory_footprint_law"]
    print("§V.5d distributed cavity↔channel Mori–Zwanzig projection (paper4a §4.5)")
    print(f"  stable generator: {out['stable_generator']} (max Re eig {out['max_real_eig']:.4f})")
    print(f"(A) projection exact (operator Laplace, random complex s): max rel err "
          f"{A['max_rel_err']:.1e}")
    print(f"(B) kernel == channel Green's fn: err {B['kernel_vs_expm_max_err']:.1e}; "
          f"along-flow memory range ℓ_mem={B['ell_mem_at_tau2']:.3f} (Δx={B['dx']:.3f}, "
          f"resolved/non-local={B['spatially_resolved_nonlocal']})")
    print(f"(C) reduced field-GLE reproduces full PDE (τ₂={C['tau2']}): rel-err "
          f"{C['rel_err_trajectory']:.1e}")
    print(f"(D) memory necessity dose-response: local-closure err "
          f"{[round(e, 4) for e in D['local_closure_err']]} over τ₂={D['tau2_list']} "
          f"(monotone={D['monotone_increasing']}, →0={D['vanishes_as_tau2_to_0']})")
    print(f"(E) lumped 2×2 = no-transport limit: off-diag {E['no_transport_offdiag_max']:.1e}, "
          f"diag vs −ab e^{{−τ/τ₂}} err {E['no_transport_diag_vs_lumped_err']:.1e}; "
          f"off-diag fraction {[round(f, 3) for f in E['offdiag_fraction']]} "
          f"(non-locality grows={E['nonlocality_grows_with_transport']})")
    print(f"(F) along-flow footprint law ℓ=2D_q/(√(U²+4D_q/τ₂)−U): max rel err "
          f"{F['max_rel_err']:.1e} over 7 cases (diffusive limit {F['diffusive_limit_ok']}, "
          f"ballistic limit {F['ballistic_limit_ok']})")
    print(f"PASS={out['pass']}")
    os.makedirs("glaciers/validation/reports", exist_ok=True)
    with open("glaciers/validation/reports/hydraulic_mz_spatial.json", "w") as fh:
        json.dump(out, fh, indent=2)
    p = make_figure(out)
    print("[saved] glaciers/validation/reports/hydraulic_mz_spatial.json")
    print(f"[saved] {p}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
