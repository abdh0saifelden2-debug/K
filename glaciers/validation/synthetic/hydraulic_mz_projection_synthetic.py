r"""§V.5b synthetic unit test for the §G.4 Mori–Zwanzig *projection* (RESULT 21).

No external data; CPU only.  §G.4 reassigns the surge-lag memory to the hydraulic
cavity↔channel subsystem and makes a precise structural claim (the ontology table,
"two potentials + a hydraulic impedance kernel"):

    "Its linearised lumped Green's function **is** a memory kernel in the exact
     Mori–Zwanzig sense"  — §G.4

`hydraulic_kernel_synthetic.py` (RESULT, §V.5) already shows the *observable* of the
coupled 2-compartment model rises to an interior peak `t*` (the shape).  What it
does **not** do — and what makes the quoted claim true rather than a metaphor — is
the actual **projection**: eliminate the channel variable from the coupled system
and show the *resolved* variable then obeys a **closed generalized Langevin
equation** (GLE) whose convolution kernel is exactly the eliminated subsystem's
Green's function.  That is the content of "in the exact Mori–Zwanzig sense", and it
was never demonstrated.  This module demonstrates it, for the *same* linear model
`hydraulic_kernel_synthetic.coupled_response` builds.

Exact projection (linear Nakajima–Zwanzig / Mori, no Markov approximation)
--------------------------------------------------------------------------
The lumped linearisation is ``ẋ = M x + F`` with ``x = (s, q)`` — ``s`` the
*resolved* store (cavity water pressure ``p_w``) and ``q`` the *eliminated* channel
variable — and

    M = [[M_ss, M_sq],          M_ss = −1/τ₁,  M_sq = −a,
         [M_qs, M_qq]],         M_qs =  b,     M_qq = −1/τ₂.

Solving the ``q`` row by variation of parameters and substituting into the ``s`` row
gives a term that is **non-local in time** — a memory integral:

    ṡ(t) = M_ss s(t) + ∫₀ᵗ K(t−τ) s(τ) dτ + R(t),                              (GLE)

    K(τ) = M_sq M_qs e^{M_qq τ}              (the memory kernel)               (K)
    R(t) = M_sq e^{M_qq t} q(0) + F_s(t) + M_sq ∫₀ᵗ e^{M_qq(t−τ)} F_q(τ) dτ.   (R)

`K(τ) = −ab·e^{−τ/τ₂}` is **exactly** the channel subsystem's own Green's function
`e^{M_qq τ}` weighted by the up/down couplings `M_sq M_qs` — i.e. the eliminated
compartment's *impedance* kernel.  `R(t)` is the Mori "random force": in the
homogeneous problem it is carried entirely by the eliminated initial state `q(0)`
and is orthogonal to the resolved variable's own history.  Checks:

  A. **The projection is exact (machine precision).**  In Laplace space the GLE
     transfer function ``ŝ(s) = [s₀(s−M_qq) + M_sq q₀] / [(s−M_ss)(s−M_qq) −
     M_sq M_qs]`` must equal the full 2×2 resolvent's (1,1) component term-for-term.
     Verified at random complex ``s`` for random stable overdamped ``M``
     (max abs err ≤ 1e-10) — so ``K`` is the *exact* kernel, not an approximation.

  B. **The kernel IS the eliminated subsystem's Green's function.**  Probing the
     channel alone (``q̇ = M_qq q``, ``q(0)=M_qs``) and reading ``M_sq·q(τ)``
     reproduces ``K(τ)`` to ≤1e-10 — the kernel is the channel impedance response,
     not a fit.

  C. **The reduced GLE reproduces the resolved trajectory.**  Time-stepping (GLE)
     with kernel (K) and force (R) matches the exact ``expm`` solution of the full
     2×2 system to rel-err ≤1e-5 over a transient — the reduced non-Markovian model
     *is* the resolved dynamics.

  D. **Memory is necessary for the lag/peak.**  The full (memory-carrying) downstream
     response peaks at the interior ``t* = τ₁τ₂ln(τ₁/τ₂)/(τ₁−τ₂)`` (tie-in to
     `hydraulic_kernel_synthetic`), whereas the **Markovian** adiabatic-elimination
     closure (drop the memory: slave ``q = −M_qq⁻¹(M_qs s)``) collapses to a
     first-order response that is monotone with argmax at ``t=0``.  So the surge lag
     is a property of the *memory kernel*, exactly §G.4's claim.

  E. **Markovian limit = adiabatic elimination (DC gain).**  ``∫₀^∞ K dτ =
     M_sq M_qs (−M_qq)⁻¹`` (the kernel's DC gain), and as the channel time
     ``τ₂→0`` at fixed DC gain ``K→(∫K)·δ(τ)`` so the GLE collapses to the
     instantaneous closure ``ṡ = (M_ss + ∫K) s`` — the local limit the model uses
     when the eliminated subsystem is fast (the §D.4 FDT-Markov limit, here *derived
     from the projection* rather than assumed).

This validates the **projection structure** — that the §G.4 hydraulic term is a
genuine Mori–Zwanzig memory kernel — not any physical lag value (that is
`hydraulic_lag_derivation.py`, RESULT) nor a real surge (USAP-DC-gated, §H.2).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

# the coupled model under test is the *same* one used for the shape result
import hydraulic_kernel_synthetic as hks  # noqa: E402


def _trapz(y, dx):
    """Uniform-grid trapezoidal integral (version-agnostic: np.trapz was renamed
    to np.trapezoid in NumPy 2.0)."""
    y = np.asarray(y, float)
    return float(dx * (y.sum() - 0.5 * (y[0] + y[-1])))


def system_matrix(tau1, tau2, a, b):
    """The §G.4 lumped cavity↔channel linearisation (identical to
    ``hydraulic_kernel_synthetic.coupled_response``)."""
    return np.array([[-1.0 / tau1, -a], [b, -1.0 / tau2]])


def memory_kernel(t, M):
    """Exact MZ memory kernel (K): ``K(τ) = M_sq M_qs e^{M_qq τ}`` (s resolved,
    q eliminated)."""
    t = np.asarray(t, float)
    M_sq, M_qs, M_qq = M[0, 1], M[1, 0], M[1, 1]
    return np.where(t >= 0.0, M_sq * M_qs * np.exp(M_qq * t), 0.0)


# --- A. exact projection in Laplace space ------------------------------------
def _s_full_laplace(sv, M, x0):
    """(1,1)-projected resolved transfer ŝ(s) from the *full* resolvent
    (sI−M)⁻¹ x0, evaluated by direct 2×2 inverse (complex s)."""
    s0, q0 = x0
    out = np.empty_like(np.asarray(sv, complex))
    for i, s in enumerate(np.atleast_1d(sv)):
        R = np.linalg.inv(s * np.eye(2) - M)
        out[i] = (R @ x0)[0]
    return out


def _s_gle_laplace(sv, M, x0):
    """ŝ(s) reconstructed from the GLE with kernel (K): closed form
    ``[s0 (s−M_qq) + M_sq q0] / [(s−M_ss)(s−M_qq) − M_sq M_qs]``."""
    s0, q0 = x0
    M_ss, M_sq, M_qs, M_qq = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    sv = np.atleast_1d(np.asarray(sv, complex))
    num = s0 * (sv - M_qq) + M_sq * q0
    den = (sv - M_ss) * (sv - M_qq) - M_sq * M_qs
    return num / den


def run_projection_exact(seed=0, n=400):
    """A. The GLE transfer function equals the full resolvent (1,1) at random
    complex s, for random stable overdamped systems — the projection is exact."""
    rng = np.random.default_rng(seed)
    max_err = 0.0
    n_tested = 0
    for _ in range(n):
        tau1 = rng.uniform(0.3, 4.0)
        tau2 = rng.uniform(0.1, 0.9) * tau1            # keep τ₂ < τ₁
        a = rng.uniform(0.01, 0.4)
        b = rng.uniform(0.01, 0.4)
        M = system_matrix(tau1, tau2, a, b)
        # overdamped (real, negative eigenvalues) — physical regime
        ev = np.linalg.eigvals(M)
        if np.any(np.abs(ev.imag) > 1e-12) or np.any(ev.real >= 0):
            continue
        n_tested += 1
        x0 = rng.normal(size=2)
        sv = rng.uniform(0.2, 8.0, size=12) + 1j * rng.uniform(-4.0, 4.0, size=12)
        err = np.max(np.abs(_s_full_laplace(sv, M, x0) - _s_gle_laplace(sv, M, x0)))
        max_err = max(max_err, float(err))
    # require systems to have actually been tested: if the overdamped filter ever
    # rejected *all* draws, max_err would stay 0.0 and the test would pass
    # vacuously, so a non-empty sample is part of the pass criterion.
    ok = n_tested > 0 and max_err <= 1e-10
    return {"max_abs_err": max_err, "n_drawn": n, "n_tested": n_tested,
            "pass": bool(ok)}


# --- B. kernel == eliminated subsystem Green's function ----------------------
def run_kernel_is_greens_function(tau1=2.0, tau2=0.5, a=0.15, b=0.30, T=20.0):
    """B. Probing the channel alone (q̇=M_qq q, q(0)=M_qs) and reading M_sq·q(τ)
    reproduces K(τ) — the memory kernel is the channel impedance Green's function."""
    M = system_matrix(tau1, tau2, a, b)
    M_sq, M_qs, M_qq = M[0, 1], M[1, 0], M[1, 1]
    t = np.linspace(0.0, T, 4001)
    K_analytic = memory_kernel(t, M)
    q_probe = M_qs * np.exp(M_qq * t)        # channel Green's function, q(0)=M_qs
    K_from_probe = M_sq * q_probe
    err = float(np.max(np.abs(K_from_probe - K_analytic)))
    # the kernel decays at the channel rate 1/τ₂ and has the back-coupling sign
    rate = -np.polyfit(t[K_analytic != 0][:50],
                       np.log(np.abs(K_analytic[K_analytic != 0][:50])), 1)[0]
    ok = err <= 1e-10 and abs(rate - 1.0 / tau2) <= 1e-6 and np.all(K_analytic <= 0)
    return {"kernel_vs_greens_max_err": err, "decay_rate": float(rate),
            "expected_rate_inv_tau2": 1.0 / tau2,
            "kernel_sign_nonpositive": bool(np.all(K_analytic <= 0)), "pass": bool(ok)}


# --- C. reduced GLE reproduces the full resolved trajectory ------------------
def _full_trajectory(M, x0, t):
    """Exact full 2×2 solution s(t)=[expm(Mt) x0]_0 (homogeneous, F=0)."""
    s = np.empty_like(t)
    for i, ti in enumerate(t):
        s[i] = (expm(M * ti) @ x0)[0]
    return s


def _gle_trajectory(M, x0, t):
    """Integrate the reduced GLE  ṡ = M_ss s + ∫₀ᵗ K(t−τ) s(τ)dτ + R(t)  on a
    uniform grid (trapezoidal memory, F=0 ⇒ R(t)=M_sq e^{M_qq t} q0).  Heun step."""
    M_ss, M_sq, M_qq = M[0, 0], M[0, 1], M[1, 1]
    s0, q0 = x0
    dt = t[1] - t[0]
    K = memory_kernel(t, M)
    s = np.empty_like(t)
    s[0] = s0

    def rhs(i, si):
        # memory integral ∫₀^{t_i} K(t_i-τ) s(τ) dτ via trapezoid over known history
        if i == 0:
            mem = 0.0
        else:
            ker = K[i::-1][:i + 1]              # K(t_i - t_j), j=0..i
            integ = ker * np.concatenate([s[:i], [si]])
            mem = _trapz(integ, dx=dt)
        R = M_sq * np.exp(M_qq * t[i]) * q0
        return M_ss * si + mem + R

    for i in range(len(t) - 1):
        f0 = rhs(i, s[i])
        s_pred = s[i] + dt * f0
        f1 = rhs(i + 1, s_pred)
        s[i + 1] = s[i] + 0.5 * dt * (f0 + f1)
    return s


def run_gle_reproduces_trajectory(tau1=2.0, tau2=0.5, a=0.15, b=0.30, T=20.0, n=8001):
    """C. The reduced GLE trajectory matches the exact full-system s(t)."""
    M = system_matrix(tau1, tau2, a, b)
    x0 = np.array([1.0, 0.0])                  # excite the resolved store only
    t = np.linspace(0.0, T, n)
    s_full = _full_trajectory(M, x0, t)
    s_gle = _gle_trajectory(M, x0, t)
    scale = np.max(np.abs(s_full))
    rel = float(np.max(np.abs(s_gle - s_full)) / scale)
    ok = rel <= 1e-5
    return {"rel_err_trajectory": rel, "pass": bool(ok)}


# --- D. memory is necessary for the lag/peak ---------------------------------
def run_memory_makes_the_peak(tau1=2.0, tau2=0.5, a=0.15, b=0.30, T=30.0, n=30001):
    """D. Full (memory) downstream response peaks at the interior time set by the
    *coupled* eigenvalues; the Markovian adiabatic closure (no memory) is monotone
    with argmax at t=0."""
    M = system_matrix(tau1, tau2, a, b)
    M_ss, M_sq, M_qs, M_qq = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    t = np.linspace(0.0, T, n)

    # full coupled downstream (channel) impulse response — reuse the shape module
    G_full, _M = hks.coupled_response(t, tau1, tau2, a, b)
    k_full = int(np.argmax(G_full))
    full_interior = 0 < k_full < len(t) - 1
    # the feedback (a,b) shifts the timescales: the correct peak uses the *coupled*
    # eigenvalues τ̃ = −1/Re(eig), not the bare (τ₁,τ₂).
    ev = np.linalg.eigvals(M)
    tt = np.sort(-1.0 / ev.real)                     # coupled relaxation times
    tstar = hks.cascade_peak_time(tt[1], tt[0])      # same g2 peak formula
    peak_err = abs(t[k_full] - tstar) / tstar

    # Markovian adiabatic elimination: q slaved instantly q≈ -M_qq^{-1} M_qs s,
    # so s obeys ṡ=(M_ss + M_sq M_qs/(-M_qq)) s = r_eff s, and the *channel*
    # observable q_slave = (M_qs/(-M_qq)) s inherits s's monotone decay.
    r_eff = M_ss + M_sq * M_qs / (-M_qq)
    s_mark = np.exp(r_eff * t)                       # s(0)=1, F=0
    q_slave = (M_qs / (-M_qq)) * s_mark
    k_mark = int(np.argmax(np.abs(q_slave)))
    mark_monotone = bool(np.all(np.diff(np.abs(q_slave)) <= 1e-12))

    ok = (full_interior and peak_err < 5e-3
          and k_mark == 0 and mark_monotone and r_eff < 0)
    return {"full_peak_time": float(t[k_full]),
            "tstar_coupled_eigs": float(tstar), "peak_rel_err": float(peak_err),
            "full_peaks_interior": bool(full_interior),
            "markovian_argmax_index": k_mark, "markovian_monotone": mark_monotone,
            "markovian_eff_rate": float(r_eff), "pass": bool(ok)}


# --- E. Markovian limit = adiabatic elimination (DC gain) --------------------
def run_markovian_limit(tau1=2.0, a=0.15, b=0.30, T=400.0, n=400001):
    """E. ∫₀^∞ K dτ = M_sq M_qs/(−M_qq); shrinking τ₂ at fixed DC gain collapses
    K toward (∫K)·δ(τ) — the local-closure limit."""
    # base system
    tau2 = 0.5
    M = system_matrix(tau1, tau2, a, b)
    M_sq, M_qs, M_qq = M[0, 1], M[1, 0], M[1, 1]
    t = np.linspace(0.0, T, n)
    K = memory_kernel(t, M)
    dc_numeric = _trapz(K, dx=t[1] - t[0])
    dc_analytic = M_sq * M_qs / (-M_qq)
    dc_err = abs(dc_numeric - dc_analytic)

    # shrink the channel time at fixed DC gain: rescale b so M_sq M_qs/(-M_qq) const.
    # DC = (-a)(b) * tau2  ⇒ keep a*b*tau2 fixed; as tau2→0 the kernel mass
    # concentrates near τ=0 (peak 1/τ2 → ∞, width τ2 → 0) i.e. → δ(τ).
    dc_fixed = dc_analytic
    fwhm = []
    taus = [0.5, 0.1, 0.02]
    for t2 in taus:
        # hold DC gain fixed: DC = -a*b2*t2 ⇒ b2 = -dc_fixed/(a*t2)  (dc_fixed<0 ⇒ b2>0)
        b2 = -dc_fixed / (a * t2)
        Mt = system_matrix(tau1, t2, a, b2)
        Kt = memory_kernel(t, Mt)
        # width: time for |K| to fall to half its peak (≈ t2 ln2) -> shrinks with t2
        peak = np.abs(Kt[0])
        half = np.where(np.abs(Kt) <= 0.5 * peak)[0]
        fwhm.append(float(t[half[0]]) if len(half) else float("inf"))
    widths_shrink = fwhm[0] > fwhm[1] > fwhm[2]
    ok = dc_err <= 1e-8 and widths_shrink and fwhm[2] < 0.05
    return {"dc_gain_numeric": dc_numeric, "dc_gain_analytic": dc_analytic,
            "dc_gain_err": dc_err, "kernel_widths_vs_tau2": dict(zip(taus, fwhm)),
            "widths_shrink_to_delta": bool(widths_shrink), "pass": bool(ok)}


# --- orchestrator ------------------------------------------------------------
def run():
    A = run_projection_exact()
    B = run_kernel_is_greens_function()
    C = run_gle_reproduces_trajectory()
    D = run_memory_makes_the_peak()
    E = run_markovian_limit()
    out = {"projection_exact": A, "kernel_is_greens": B, "gle_trajectory": C,
           "memory_makes_peak": D, "markovian_limit": E}
    out["pass"] = bool(A["pass"] and B["pass"] and C["pass"] and D["pass"] and E["pass"])
    return out


def main():
    r = run()
    A, B, C = r["projection_exact"], r["kernel_is_greens"], r["gle_trajectory"]
    D, E = r["memory_makes_peak"], r["markovian_limit"]
    print("§G.4 hydraulic Mori–Zwanzig projection — synthetic calibration")
    print(f"(A) projection exact (Laplace, random stable M): max abs err "
          f"{A['max_abs_err']:.1e}")
    print(f"(B) kernel == channel Green's fn: err {B['kernel_vs_greens_max_err']:.1e}, "
          f"decay rate {B['decay_rate']:.4f} vs 1/τ₂={B['expected_rate_inv_tau2']:.4f}")
    print(f"(C) reduced GLE reproduces full s(t): rel-err {C['rel_err_trajectory']:.1e}")
    print(f"(D) memory⇒peak: full peaks at t*={D['full_peak_time']:.3f} "
          f"(coupled-eig analytic {D['tstar_coupled_eigs']:.3f}, "
          f"rel-err {D['peak_rel_err']:.1e}); Markovian monotone="
          f"{D['markovian_monotone']} argmax_idx={D['markovian_argmax_index']}")
    print(f"(E) Markovian limit: ∫K err {E['dc_gain_err']:.1e}, widths "
          f"{[round(w, 4) for w in E['kernel_widths_vs_tau2'].values()]} → δ "
          f"({E['widths_shrink_to_delta']})")
    print(f"PASS={r['pass']}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
