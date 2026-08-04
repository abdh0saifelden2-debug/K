r"""Cross-cutting relationships, batch 5 (NR15) — DERIVED here, VERIFIED one-by-one.
Continuation of cross_relationships{,2,3,4}.py (NR1-NR14).  No external data; CPU only.

  NR15 The backscatter budget: the SIGNED time-integral of the Mori-Zwanzig memory
       kernel is the *net* eddy viscosity, nu_eff = \int_0^inf K(tau) dtau.  A kernel
       with a dominant negative (backscatter) lobe gives nu_eff < 0 -- Kraichnan's
       negative eddy viscosity -- so the resolved mode GROWS (energy returned from
       sub-grid scales), which a strictly positive K-theory eddy viscosity is
       structurally unable to represent.  This is the dynamical, energy-budget face of
       NR2 (Re Z < 0 = backscatter): K-theory keeps only the gross forward transport
       and projects out the backscatter lobe, fixing the sign of net dissipation it can
       express.  Verifies on the exponential-memory GLE (exact Markovian embedding):
       (i) nu_eff = \int K = c_f - c_b; (ii) net-dissipative kernel -> decay; (iii)
       backscatter-dominated kernel -> growth (nu_eff < 0); (iv) the crossover is
       exactly at the backscatter fraction beta = c_b/c_f = 1.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.integrate import quad


def _kernel(tau, c_f, c_b, tau1, tau2):
    """K(tau) = c_f/tau1 e^{-tau/tau1} - c_b/tau2 e^{-tau/tau2}  (forward minus
    backscatter lobe); each exponential integrates to its weight, so int K = c_f-c_b."""
    return c_f / tau1 * np.exp(-tau / tau1) - c_b / tau2 * np.exp(-tau / tau2)


def _gle_evolve(c_f, c_b, tau1, tau2, T=40.0, dt=2e-3):
    r"""Exact Markovian embedding of the exponential-memory GLE
        dx/dt = -\int_0^t K(t-s) x(s) ds,
    with K = c_f/tau1 e^{-./tau1} - c_b/tau2 e^{-./tau2}.  Auxiliary
    y_i = \int_0^t (1/tau_i) e^{-(t-s)/tau_i} x(s) ds obey y_i' = (x - y_i)/tau_i and
    dx/dt = -(c_f y1 - c_b y2).  Returns (t, x)."""
    n = int(T / dt)
    t = np.arange(n) * dt
    x = np.empty(n)
    xi, y1, y2 = 1.0, 0.0, 0.0

    def deriv(x_, y1_, y2_):
        dy1 = (x_ - y1_) / tau1
        dy2 = (x_ - y2_) / tau2
        dx = -(c_f * y1_ - c_b * y2_)
        return dx, dy1, dy2

    for i in range(n):
        x[i] = xi
        k1 = deriv(xi, y1, y2)
        k2 = deriv(xi + 0.5 * dt * k1[0], y1 + 0.5 * dt * k1[1], y2 + 0.5 * dt * k1[2])
        k3 = deriv(xi + 0.5 * dt * k2[0], y1 + 0.5 * dt * k2[1], y2 + 0.5 * dt * k2[2])
        k4 = deriv(xi + dt * k3[0], y1 + dt * k3[1], y2 + dt * k3[2])
        xi += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        y1 += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        y2 += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
    return t, x


def _late_rate(t, x):
    """Asymptotic growth/decay rate from a log-linear fit on |x| over the last half."""
    i0 = len(t) // 2
    ax = np.abs(x[i0:])
    ax = np.clip(ax, 1e-300, None)
    return float(np.polyfit(t[i0:], np.log(ax), 1)[0])


def nr15_backscatter_budget(tau1=0.5, tau2=0.5):
    r"""net eddy viscosity nu_eff = int_0^inf K = c_f - c_b sets the sign of the
    resolved-mode evolution: nu_eff>0 -> decay; nu_eff<0 (backscatter-dominated) ->
    growth (Kraichnan negative eddy viscosity), unreachable by any positive K-theory.
    """
    # (i) signed integral identity
    cf0, cb0 = 1.0, 0.3
    I_num = quad(_kernel, 0, 200, args=(cf0, cb0, tau1, tau2))[0]
    int_ok = abs(I_num - (cf0 - cb0)) < 1e-6

    # (ii) net-dissipative kernel -> decay
    t, x_diss = _gle_evolve(cf0, cb0, tau1, tau2)
    rate_diss = _late_rate(t, x_diss)
    decays = bool(abs(x_diss[-1]) < 0.1 and rate_diss < 0)

    # (iii) backscatter-dominated kernel -> growth (nu_eff<0)
    cf1, cb1 = 0.4, 1.0
    t, x_back = _gle_evolve(cf1, cb1, tau1, tau2)
    rate_back = _late_rate(t, x_back)
    grows = bool(abs(x_back[-1]) > 5.0 and rate_back > 0)

    # (iv) crossover at beta = c_b/c_f = 1 (nu_eff = 0): sweep c_b at fixed c_f
    cf = 1.0
    betas = np.linspace(0.4, 1.6, 13)
    rates = np.array([_late_rate(*_gle_evolve(cf, b * cf, tau1, tau2)) for b in betas])
    # rate sign should flip from - to + as beta passes 1 (nu_eff = cf(1-beta))
    sign_flip = bool(np.all(rates[betas < 0.98] < 0) and np.all(rates[betas > 1.02] > 0))
    # net vs gross: gross forward = c_f, net = c_f - c_b; backscatter fraction
    beta_back = cb1 / cf1

    ok = bool(int_ok and decays and grows and sign_flip)
    return dict(name="NR15 backscatter budget = signed kernel integral",
                int_K=float(I_num), int_K_expected=float(cf0 - cb0), int_identity_ok=int_ok,
                rate_dissipative=rate_diss, x_final_dissipative=float(x_diss[-1]),
                rate_backscatter=rate_back, x_final_backscatter=float(x_back[-1]),
                beta_backscatter_case=float(beta_back),
                crossover_beta=1.0, rate_sign_flips_at_beta1=sign_flip,
                interpretation=("nu_eff = int K = c_f - c_b is the NET eddy viscosity; a "
                                "dominant negative (backscatter) lobe makes nu_eff<0 "
                                "(Kraichnan) -> resolved-mode growth, impossible for a "
                                "positive K-theory; dynamical face of NR2's Re Z<0"),
                mainstream="Kraichnan 1976 negative eddy viscosity; Mori-Zwanzig 2nd-FDT; "
                           "Leith/Frederiksen stochastic backscatter",
                ok=ok)


ALL = [nr15_backscatter_budget]


def summary():
    return [f() for f in ALL]


if __name__ == "__main__":
    print("Cross-cutting relationships batch 5 (NR15) — verification\n" + "=" * 56)
    allok = True
    for r in summary():
        allok &= r["ok"]
        print(f"\n[{'PASS' if r['ok'] else 'FAIL'}] {r['name']}")
        print(f"   link: {r['interpretation']}")
        print(f"   lit:  {r['mainstream']}")
        for k, v in r.items():
            if k in ("name", "interpretation", "mainstream", "ok"):
                continue
            if isinstance(v, float):
                print(f"   {k} = {v:.4g}")
            else:
                print(f"   {k} = {v}")
    print("\n" + "=" * 56)
    print("ALL VERIFIED" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
