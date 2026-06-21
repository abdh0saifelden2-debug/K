r"""§V.5c — the nonlinear cavity<->channel kernel, and why the linear MZ kernel is its
small-amplitude limit (paper4a §4, answering "the MZ result is just a 2x2 identity").

The committed projection (`hydraulic_mz_projection_synthetic.py`) is exact but *linear*:
eliminating the channel from `M=[[-1/tau1,-a],[b,-1/tau2]]` gives the memory kernel
`K(tau)=-ab e^{-tau/tau2}`.  A fair objection is that this is a 2x2 linear-algebra
identity.  This module shows the linear kernel is the **leading term of a controlled
expansion** of the *physically nonlinear* Roethlisberger cavity<->channel system, and
that the leading nonlinear correction is a real, falsifiable geophysical prediction.

Nonlinear lumped model (Roethlisberger 1972; Schoof 2010; Hewitt 2013; Kingslake)
---------------------------------------------------------------------------------
State: cavity water pressure ``p`` (resolved store), channel cross-section ``S``
(eliminated channel).  Effective pressure ``N = p_i - p``.

    C dp/dt = Q_in(t) - Q_out,              Q_out = kq * S * sqrt(max(p, eps))   (turbulent channel outflow)
    dS/dt   = ko * Q_out  -  kc * S * N**n  (melt opening ~ discharge; creep closure ~ S N^n, Glen n=3)

The **only** nonlinearities are physical: the turbulent ``sqrt(p)`` discharge and the
Glen-law ``N**3`` creep closure.  Linearising at the steady state ``(p*, S*)`` recovers
exactly the paper's ``M`` (overdamped, real eigenvalues), so the linear MZ kernel is the
small-amplitude limit.  The leading correction, driven by ``N**3``, makes the channel
relaxation time — hence the surge lag ``t*`` — **amplitude-dependent**: a larger drainage
drives ``p`` up, ``N`` down, weakens creep closure (``N**3`` collapses), keeps the channel
open longer, and *lengthens* the lag.  Prediction: **bigger floods have longer surge lags.**

No external data; CPU only.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# nonlinear lumped model
# ---------------------------------------------------------------------------
class CavityChannel:
    def __init__(self, p_i=1.0, p_star=0.6, S_star=1.0, Qin0=1.0, C=0.04,
                 ko=0.5, n=3.0, eps=1e-9):
        self.p_i, self.p_star, self.S_star, self.Qin0 = p_i, p_star, S_star, Qin0
        self.C, self.ko, self.n, self.eps = C, ko, n, eps
        # calibrate kq so Q_out(p*,S*) = Qin0 ; kc so opening = closure at steady state
        self.kq = Qin0 / (S_star * np.sqrt(p_star))
        Nstar = p_i - p_star
        self.kc = ko * Qin0 / (S_star * Nstar ** n)

    def Qout(self, p, S):
        return self.kq * S * np.sqrt(max(p, self.eps))

    def rhs(self, t, y, Qin):
        p, S = y
        N = max(self.p_i - p, self.eps)
        Qout = self.kq * S * np.sqrt(max(p, self.eps))
        dp = (Qin(t) - Qout) / self.C
        dS = self.ko * Qout - self.kc * S * N ** self.n
        return [dp, dS]

    def jacobian(self):
        """Numerical 2x2 Jacobian at the steady state (the paper's M)."""
        y0 = np.array([self.p_star, self.S_star])
        J = np.zeros((2, 2))
        h = 1e-6
        f0 = np.array(self.rhs(0.0, y0, lambda t: self.Qin0))
        for j in range(2):
            yp = y0.copy(); yp[j] += h
            fj = np.array(self.rhs(0.0, yp, lambda t: self.Qin0))
            J[:, j] = (fj - f0) / h
        return J


def pulse(t0, width, dQ):
    return lambda t: np.exp(-((t - t0) / width) ** 2) * dQ


def peak_time(t, g):
    """Interior-peak time of a response that starts near 0 (returns (t*, is_interior))."""
    k = int(np.argmax(g))
    interior = 0 < k < len(g) - 1
    return float(t[k]), bool(interior)


def linear_response(M, C, t, t0, width, dQ):
    """Linearised system d/dt[dp,dS] = M[dp,dS] + [pulse/C, 0]."""
    pf = pulse(t0, width, dQ)
    def f(tt, z):
        return M @ z + np.array([pf(tt) / C, 0.0])
    sol = solve_ivp(f, (t[0], t[-1]), [0.0, 0.0], t_eval=t, rtol=1e-9, atol=1e-12, max_step=width / 4)
    return sol.y[0], sol.y[1]   # dp, dS


def nonlinear_response(model, t, t0, width, dQ):
    """Full nonlinear response minus steady state."""
    Qin = lambda tt: model.Qin0 + pulse(t0, width, dQ)(tt)
    sol = solve_ivp(model.rhs, (t[0], t[-1]), [model.p_star, model.S_star], args=(Qin,),
                    t_eval=t, rtol=1e-9, atol=1e-12, max_step=width / 4)
    return sol.y[0] - model.p_star, sol.y[1] - model.S_star


def run():
    m = CavityChannel()
    M = m.jacobian()
    eig = np.linalg.eigvals(M)
    overdamped = bool(np.all(np.isreal(eig)) and np.all(np.real(eig) < 0))
    # relaxation time of the slowest coupled eigenmode (sets the lag)
    tau2_lin = float(-1.0 / np.max(np.real(eig)))

    t = np.linspace(0.0, 2.0, 4000)   # years
    t0, width = 0.05, 0.01
    base_dQ = 0.02 * m.Qin0           # reference small flood (2% of baseflux)

    # (1) small-amplitude limit: nonlinear -> linear as dQ -> 0
    amps = np.array([0.02, 0.05, 0.10, 0.20, 0.40, 0.80])
    tstar_nl, tstar_lin, interior_ok = [], [], True
    for a in amps:
        dQ = a * m.Qin0
        _, dS_nl = nonlinear_response(m, t, t0, width, dQ)
        _, dS_lin = linear_response(M, m.C, t, t0, width, dQ)
        # normalize sign (channel opens -> dS>0); measure peak lag relative to pulse center
        tnl, intnl = peak_time(t, dS_nl)
        tli, intli = peak_time(t, dS_lin)
        tstar_nl.append(tnl - t0); tstar_lin.append(tli - t0)
        interior_ok = interior_ok and intnl and intli
    tstar_nl = np.array(tstar_nl); tstar_lin = np.array(tstar_lin)

    # convergence: |t*_nl - t*_lin| -> 0 as amplitude -> 0 (small-amplitude limit)
    rel_gap = np.abs(tstar_nl - tstar_lin) / tstar_lin
    converges = bool(rel_gap[0] < 0.05 and rel_gap[0] < rel_gap[-1])

    # (2) the nonlinear prediction: bigger floods -> longer lag (monotone increase of t*_nl)
    lag_increases = bool(np.all(np.diff(tstar_nl) > -1e-6) and tstar_nl[-1] > tstar_nl[0] * 1.05)
    # slope of lag vs flood size (a falsifiable number)
    A = np.vstack([amps, np.ones_like(amps)]).T
    slope = float(np.linalg.lstsq(A, tstar_nl, rcond=None)[0][0])

    out = {
        "what": "linear MZ kernel = small-amplitude limit of the nonlinear Roethlisberger "
                "cavity<->channel system; Glen-N^3 creep makes the surge lag amplitude-dependent",
        "M": M.tolist(),
        "eigenvalues": [float(np.real(e)) for e in eig],
        "overdamped_real": overdamped,
        "tau2_linear_yr": tau2_lin,
        "flood_amplitudes_frac": amps.tolist(),
        "tstar_nonlinear_yr": tstar_nl.tolist(),
        "tstar_linear_yr": tstar_lin.tolist(),
        "rel_gap": rel_gap.tolist(),
        "linear_is_small_amplitude_limit": converges,
        "bigger_floods_longer_lag": lag_increases,
        "dlag_d(floodfrac)_yr": slope,
        "interior_peaks": interior_ok,
    }
    out["pass"] = bool(overdamped and converges and lag_increases and interior_ok)
    return out, t, M, m


def _print(out):
    print("=== nonlinear cavity<->channel kernel (paper4a §4.x) ===")
    print(f"  M = {np.array(out['M']).round(3).tolist()}  eig={np.round(out['eigenvalues'],3).tolist()}"
          f"  overdamped={out['overdamped_real']}")
    print(f"  linear kernel tau2 = {out['tau2_linear_yr']:.4f} yr")
    print(f"  flood frac : {np.round(out['flood_amplitudes_frac'],2).tolist()}")
    print(f"  t*_nonlin  : {np.round(out['tstar_nonlinear_yr'],4).tolist()} yr")
    print(f"  t*_linear  : {np.round(out['tstar_linear_yr'],4).tolist()} yr")
    print(f"  rel-gap    : {np.round(out['rel_gap'],4).tolist()}")
    print(f"  linear = small-amplitude limit : {out['linear_is_small_amplitude_limit']}")
    print(f"  bigger floods -> longer lag    : {out['bigger_floods_longer_lag']}"
          f"  (d t*/d floodfrac = {out['dlag_d(floodfrac)_yr']:.4f} yr)")
    print(f"  PASS: {out['pass']}")


def robustness_sweep():
    """Is 'bigger floods -> longer lag' a parameter artefact?  Sweep steady effective
    pressure N*, storage C, and melt-opening ko over physically reasonable ranges and
    confirm the sign holds for every overdamped configuration.  Returns a summary dict."""
    t = np.linspace(0.0, 2.0, 3000)
    t0, w = 0.05, 0.01
    amps = np.array([0.02, 0.1, 0.4, 0.8])
    n_over = 0; n_incr = 0; slopes_by_N = {}
    for p_star in (0.45, 0.6, 0.75):          # N* = 0.55, 0.40, 0.25
        for C in (0.02, 0.04, 0.08):
            for ko in (0.3, 0.5, 0.8):
                m = CavityChannel(p_star=p_star, C=C, ko=ko)
                eig = np.linalg.eigvals(m.jacobian())
                if not (np.all(np.isreal(eig)) and np.all(np.real(eig) < 0)):
                    continue
                ts = []
                for a in amps:
                    _, dS = nonlinear_response(m, t, t0, w, a * m.Qin0)
                    ts.append(t[int(np.argmax(dS))] - t0)
                ts = np.array(ts)
                n_over += 1
                if ts[-1] > ts[0] * 1.02 and np.all(np.diff(ts) > -1e-6):
                    n_incr += 1
                slopes_by_N.setdefault(round(1 - p_star, 2), []).append(float(np.polyfit(amps, ts, 1)[0]))
    mean_slope = {N: float(np.mean(s)) for N, s in sorted(slopes_by_N.items())}
    return {"n_overdamped": n_over, "n_lag_increases": n_incr,
            "robust_all": bool(n_incr == n_over and n_over > 0),
            "mean_slope_by_Nstar_yr": mean_slope,
            "strengthens_toward_flotation": bool(
                len(mean_slope) >= 2 and
                list(mean_slope.values())[0] > list(mean_slope.values())[-1])}


def make_figure(out, path="glaciers/validation/reports/hydraulic_nonlinear_kernel.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    amps = np.array(out["flood_amplitudes_frac"]) * 100.0
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(amps, np.array(out["tstar_linear_yr"]) * 365.25, "k--", lw=2,
            label="linear MZ kernel (amplitude-independent)")
    ax.plot(amps, np.array(out["tstar_nonlinear_yr"]) * 365.25, "o-", color="C0", lw=2,
            label="nonlinear Roethlisberger (Glen N$^3$ creep)")
    ax.set_xlabel("drainage flood size (% of base flux)")
    ax.set_ylabel("surge lag  t*  (days)")
    ax.set_title("Linear kernel = small-amplitude limit; bigger floods lag longer")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    out, t, M, m = run()
    _print(out)
    rob = robustness_sweep()
    out["robustness"] = rob
    print(f"  robustness: bigger-floods-longer-lag holds in {rob['n_lag_increases']}/{rob['n_overdamped']} "
          f"overdamped configs; mean slope by N* = {rob['mean_slope_by_Nstar_yr']} yr "
          f"(strengthens toward flotation: {rob['strengthens_toward_flotation']})")
    import json, os
    os.makedirs("glaciers/validation/reports", exist_ok=True)
    with open("glaciers/validation/reports/hydraulic_nonlinear_kernel.json", "w") as fh:
        json.dump(out, fh, indent=2)
    p = make_figure(out)
    print("[saved] glaciers/validation/reports/hydraulic_nonlinear_kernel.json")
    print(f"[saved] {p}")
