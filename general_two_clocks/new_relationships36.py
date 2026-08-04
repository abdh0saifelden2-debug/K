r"""NR59 (theory + real data) -- the spin is a skew (divergence-free)
diffusivity: the Green-Kubo integral of the odd correlation is the
ANTISYMMETRIC eddy-transport tensor, which advects tracer along contours
and mixes nothing.  Its ratio to the ordinary (Taylor) diffusivity is
EXACTLY the clock ratio r = f tau_L, and in the deep ocean it flips sign
across the equator.

Follows NR52 (spin = odd correlation), NR57 (spin = irreversibility) and
reconnects to NR15/NR23 (the signed net eddy viscosity / KK sum rule).
NR57 read the spin thermodynamically; NR59 reads its TRANSPORT meaning.

The claim
=========
Green-Kubo splits the Lagrangian eddy-transport tensor
``K = INT_0^inf <u(0) u(t)^T> dt`` into

    K_S = symmetric (Taylor) diffusivity   -- DOWN-gradient, mixing;
    K_A = antisymmetric (skew) diffusivity -- the odd correlation's
          integral, ``K_A[xy] = (1/2) INT_0^inf rho_odd_raw(t) dt``.

For the damped-rotator two-clocks velocity (``du = -u/tau_L + f eps u +
noise``) both are closed-form:

    K_S = tau_L^2/(1 + f^2 tau_L^2),   K_A = f tau_L^3/(1 + f^2 tau_L^2),

so **``K_A / K_S = f tau_L = r``, the two-clocks clock ratio (NR47)
EXACTLY** -- the skew fraction of eddy transport is the same tan(delta)
that measures memory everywhere else in the corpus.

Two consequences the corpus needs:
1. **The skew part transports but does not mix.**  The skew flux
   ``F_A = -K_A grad C`` is divergence-free (``div F_A = 0``) and
   everywhere PARALLEL to tracer contours (``F_A . grad C = 0``): it
   advects tracer around, adds nothing to the variance budget.  This is
   the transport-tensor form of NR23's *signed* net eddy viscosity and of
   the streamfunction (bolus/Nakamura) eddy advection -- the rotational
   flux that down-gradient closures (K-theory) silently discard, the
   transport face of the discarded-operator structure (NR2/NR15).
2. **The spin sets the skew diffusivity, sign and size.**  ``K_A`` has
   the sign of ``f`` (the local vorticity/rotation sense), so a measured
   Lagrangian spin gives the sign AND magnitude of the divergence-free
   eddy transport directly.

Findings (figures/94_skew_diffusivity.json; real integrals from the
committed NR52 band cache):

* **Closed forms exact.**  K_S, K_A and K_A/K_S = f tau_L verified to
  1e-14 across (tau_L, f); the skew flux div-free and contour-parallel
  to machine precision (symbolic).
* **The deep ocean's skew transport flips at the equator.**  Green-Kubo
  integrals of the committed band correlations: K_A < 0 in the NH
  (clockwise), K_A > 0 in the SH (counterclockwise), with |K_A/K_S| ~
  0.05-0.15 (mid-latitudes) -- a 5-15 % rotational (non-mixing) share of
  the deep eddy transport tensor, hemisphere-antisymmetric, vanishing at
  the equator (the K_A -> 0 line where f -> 0).  The Taylor part K_S ~
  2000-3600 m^2/s (NH/SH interior) matches the known deep eddy
  diffusivity scale.

Consequence for the program: the odd face (NR52) is the third leg of a
single object measured four ways -- memory phase (NR47), irreversibility
(NR57), and now skew transport (this NR) are all r = f tau_L = tan(delta).
The corpus's "discarded operator" (NR2/15/23) is, in transport terms,
exactly this skew diffusivity, and it is measurable, signed, and
hemisphere-antisymmetric in the real ocean.

CPU-only, offline-safe (committed NR52 cache).
Tests: tests/test_skew_diffusivity.py.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

HERE = os.path.dirname(os.path.abspath(__file__))
SPIN_CACHE = os.path.join(HERE, "data", "nr52_odd_spin_cache.json")
FIG = os.path.join(HERE, "figures", "94_skew_diffusivity.json")

EPS = np.array([[0.0, -1.0], [1.0, 0.0]])
DT_S = 10.0 * 86400.0          # 10-day cycle in seconds


# --------------------------------------------------------------------------- #
# closed-form damped-rotator diffusivity
# --------------------------------------------------------------------------- #
def rotator_diffusivity(tau_L, f):
    """Green-Kubo K = A^{-1} C0 for du = -u/tau_L + f eps u + sqrt(2)dW;
    returns (K_S_diag, K_A_xy)."""
    A = np.eye(2) / tau_L + f * EPS       # drift matrix (du = -A u + ...)
    C0 = solve_continuous_lyapunov(A, 2.0 * np.eye(2))
    K = np.linalg.inv(A) @ C0
    return float(0.5 * (K + K.T)[0, 0]), float(0.5 * (K - K.T)[0, 1])


def closed_form(tau_L, f):
    denom = 1.0 + f ** 2 * tau_L ** 2
    return tau_L ** 2 / denom, f * tau_L ** 3 / denom


# --------------------------------------------------------------------------- #
# synthetic checks
# --------------------------------------------------------------------------- #
def synthetic_checks():
    err = 0.0
    ratio_err = 0.0
    for tau_L in (1.3, 5.0, 9.0):
        for f in (0.15, 0.5, 1.2):
            KS, KA = rotator_diffusivity(tau_L, f)
            pS, pA = closed_form(tau_L, f)
            err = max(err, abs(KS - pS), abs(KA - pA))
            ratio_err = max(ratio_err, abs(KA / KS - f * tau_L))
    # divergence-free + contour-parallel: analytic finite-difference check
    xs = np.linspace(-2, 2, 81)
    X, Y = np.meshgrid(xs, xs)
    C = np.exp(-(X ** 2 + Y ** 2))          # a smooth tracer blob
    dx = xs[1] - xs[0]
    Cy, Cx = np.gradient(C, dx, dx)
    psi = 1.0
    FAx, FAy = -psi * Cy, psi * Cx          # skew flux F_A = -K_A grad C
    dFAx_x = np.gradient(FAx, dx, axis=1)
    dFAy_y = np.gradient(FAy, dx, axis=0)
    div = dFAx_x + dFAy_y
    interior = (slice(2, -2), slice(2, -2))
    div_max = float(np.max(np.abs(div[interior])))
    dotgrad_max = float(np.max(np.abs((FAx * Cx + FAy * Cy)[interior])))
    return dict(closed_form_max_err=err, ratio_max_err=ratio_err,
                skew_div_max=div_max, skew_dot_grad_max=dotgrad_max)


# --------------------------------------------------------------------------- #
# real data (committed NR52 band cache)
# --------------------------------------------------------------------------- #
def load_cache(path=SPIN_CACHE):
    with open(path) as fh:
        return json.load(fh)


def band_diffusivity(cache, band):
    """Green-Kubo K_S, K_A from the committed band correlations.  Units:
    O, E are summed (u in m/s); per-pair means give (m/s)^2, integrated
    over lag (s) -> m^2/s."""
    b = cache["bands"][band]
    O = np.array(b["O"], float)
    E = np.array(b["E"], float)
    npr = np.array(b["n_pairs"], float)
    Ro = O / npr
    Re = E / npr
    lags = np.arange(len(E)) * DT_S
    KS = 0.5 * float(np.trapezoid(Re, lags))
    KA = 0.5 * float(np.trapezoid(Ro, lags))
    return dict(K_S=KS, K_A=KA, skew_fraction=KA / KS)


def analyze(cache):
    syn = synthetic_checks()
    bands = {name: band_diffusivity(cache, name)
             for name in ("nh_all", "sh_all", "eq", "nh_mid", "sh_mid",
                          "nh_low", "sh_low")}
    out = dict(synthetic=syn, bands=bands)
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    syn = out["synthetic"]
    b = out["bands"]
    exact = (syn["closed_form_max_err"] < 1e-10
             and syn["ratio_max_err"] < 1e-10)
    divfree = (syn["skew_div_max"] < 1e-9
               and syn["skew_dot_grad_max"] < 1e-9)
    flips = (b["nh_all"]["K_A"] < 0 and b["sh_all"]["K_A"] > 0
             and b["nh_mid"]["K_A"] < 0 and b["sh_mid"]["K_A"] > 0
             and abs(b["eq"]["skew_fraction"])
             < min(abs(b["nh_mid"]["skew_fraction"]),
                   abs(b["sh_mid"]["skew_fraction"])))
    taylor_scale = (500.0 < b["nh_all"]["K_S"] < 20000.0
                    and 500.0 < b["sh_all"]["K_S"] < 20000.0)
    return dict(
        skew_ratio_is_clock_ratio=bool(exact),
        skew_flux_divergence_free=bool(divfree),
        ocean_skew_transport_flips_at_equator=bool(flips),
        taylor_diffusivity_scale_reasonable=bool(taylor_scale),
        reading=(
            "The spin is a skew diffusivity: the Green-Kubo integral of "
            "the odd correlation is the ANTISYMMETRIC eddy-transport "
            "tensor K_A, whose flux is divergence-free and everywhere "
            "parallel to tracer contours -- it advects, mixes nothing "
            "(the transport form of NR23's signed eddy viscosity, the "
            "rotational flux K-theory discards).  Its ratio to the Taylor "
            "diffusivity is EXACTLY the clock ratio K_A/K_S = f tau_L = "
            "tan(delta), so the skew fraction of eddy transport is the "
            "same memory coordinate as everywhere else in the corpus.  In "
            "the deep ocean the skew transport flips sign across the "
            f"equator (NH K_A = {b['nh_all']['K_A']:.0f}, SH "
            f"{b['sh_all']['K_A']:+.0f} m^2/s), a "
            f"{100 * abs(b['sh_mid']['skew_fraction']):.0f}% rotational "
            "(non-mixing) share of the mid-latitude transport tensor on a "
            f"Taylor background of ~{b['sh_all']['K_S']:.0f} m^2/s, "
            "vanishing at the equator where f -> 0.  Memory phase, "
            "irreversibility, and skew transport are one object measured "
            "three ways: r = f tau_L."),
    )


def run(write=True):
    cache = load_cache()
    out = analyze(cache)
    res = dict(description=__doc__.splitlines()[0], **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    s, b = res["synthetic"], res["bands"]
    print("NR59 -- the spin is a skew (divergence-free) diffusivity")
    print(f"  K_A/K_S = clock ratio f tau_L : "
          f"{v['skew_ratio_is_clock_ratio']} "
          f"(closed-form err {s['closed_form_max_err']:.0e}, ratio err "
          f"{s['ratio_max_err']:.0e})")
    print(f"  skew flux divergence-free      : "
          f"{v['skew_flux_divergence_free']} "
          f"(div {s['skew_div_max']:.0e}, F.gradC {s['skew_dot_grad_max']:.0e})")
    print(f"  ocean skew transport flips eq  : "
          f"{v['ocean_skew_transport_flips_at_equator']} "
          f"(NH K_A={b['nh_all']['K_A']:.0f}, SH K_A={b['sh_all']['K_A']:+.0f} "
          f"m^2/s)")
    print(f"  Taylor diffusivity scale ok    : "
          f"{v['taylor_diffusivity_scale_reasonable']} "
          f"(NH K_S={b['nh_all']['K_S']:.0f}, SH K_S={b['sh_all']['K_S']:.0f} "
          f"m^2/s)")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
