r"""§V.5 synthetic unit test for the hydraulic memory-kernel *shape* (§G.4 / §K).

No external data.  §G.4 reassigns the surge-lag memory from ice-thermal diffusion
to the **hydraulic** cavity/channel subsystem, and notes (caveat i) that a single
first-order ``RC`` storage is *monotone* (``K = (1/RC) e^{-t/RC}``, maximal at
``t=0``) and therefore **cannot** reproduce the observed post-drainage speed-up that
*rises to a peak* at 0.02-2 yr.  The open question was whether reproducing the peak
needs the full nonlinear GlaDS/Roethlisberger model.

This module shows it does **not**: the peak is the generic, *linear* signature of a
**two-compartment** hydraulic system (cavity storage charging, then channel
cross-section opening).  Concretely we compare three linearised lumped models.

1. **One storage (first-order).**  ``C dx/dt = f(t) - x/R`` gives impulse response
   ``g1(t) = (1/C) e^{-t/RC}`` -- *monotone decreasing*, argmax at ``t=0``.  This is
   exactly §G.4's ``K_hydraulic`` and its "no peak" caveat.

2. **Two storages in cascade (cavity -> channel).**  Water first charges the cavity
   (time ``tau1``), which then drives channel opening (time ``tau2``).  The
   downstream impulse response is

       g2(t) = (e^{-t/tau1} - e^{-t/tau2}) / (tau1 - tau2),

   which satisfies ``g2(0) = 0`` and **rises to an interior peak** at

       t* = tau1 tau2 ln(tau1/tau2) / (tau1 - tau2)  > 0,

   then decays.  The peak is produced by purely monotone, overdamped components.

3. **Two-compartment with feedback (matrix).**  The coupled cavity<->channel
   linearisation ``d/dt [x1,x2] = M [x1,x2] + [f,0]`` with
   ``M = [[-1/tau1, -a],[ b, -1/tau2]]`` (channel opening ``b`` driven by cavity
   throughput; channel growth lowering cavity resistance ``a``).  For overdamped
   (real-eigenvalue) parameters the downstream channel response again **starts at 0,
   peaks at an interior time, and decays monotonically** (one peak, no oscillation).

The conclusion is structural and *derivable*: the observed rise-to-a-peak follows
from any two coupled linear storages with monotone parts -- it needs neither
nonlinearity nor a special pulse shape.  What stays **[HYP]** is the *value* of the
lag (``t*``), which depends on the regional ``R, C, tau1, tau2`` set by the
nonlinear GlaDS/Roethlisberger hydrology -- this test plants those timescales, it
does not derive them.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


# --- analytic impulse responses ----------------------------------------------
def g1(t, tau):
    """First-order (single storage) impulse response: monotone decaying."""
    t = np.asarray(t, float)
    return np.where(t >= 0, np.exp(-t / tau) / tau, 0.0)


def g2(t, tau1, tau2):
    """Two-stage cascade (cavity -> channel) impulse response: g2(0)=0, then peaks."""
    t = np.asarray(t, float)
    if abs(tau1 - tau2) < 1e-12:                       # critically-damped limit
        return np.where(t >= 0, t * np.exp(-t / tau1) / tau1**2, 0.0)
    return np.where(t >= 0,
                    (np.exp(-t / tau1) - np.exp(-t / tau2)) / (tau1 - tau2), 0.0)


def cascade_peak_time(tau1, tau2):
    """Analytic interior peak time of g2."""
    return tau1 * tau2 * np.log(tau1 / tau2) / (tau1 - tau2)


def coupled_response(t, tau1, tau2, a, b):
    """Downstream (channel, x2) impulse response of the coupled 2x2 system via the
    matrix exponential: x(t) = expm(M t) @ [1, 0] (unit impulse into the cavity)."""
    M = np.array([[-1.0 / tau1, -a], [b, -1.0 / tau2]])
    x0 = np.array([1.0, 0.0])
    out = np.empty_like(np.asarray(t, float))
    for i, ti in enumerate(np.atleast_1d(t)):
        out[i] = (expm(M * ti) @ x0)[1] if ti >= 0 else 0.0
    return out, M


def _interior_peak(t, g):
    """Return (index, is_interior, monotone_after_peak) for a sampled response."""
    k = int(np.argmax(g))
    is_interior = 0 < k < len(g) - 1
    monotone_after = bool(np.all(np.diff(g[k:]) <= 1e-12))
    return k, is_interior, monotone_after


def run():
    # planted (illustrative, NOT derived) hydraulic timescales
    tau1, tau2 = 2.0, 0.5            # cavity-charge vs channel-open times
    t = np.linspace(0.0, 30.0, 6001)

    # (1) one storage -> monotone, peak at t=0
    G1 = g1(t, tau1)
    k1 = int(np.argmax(G1))
    mono1 = bool(np.all(np.diff(G1) <= 1e-12))
    first_order_monotone = bool(k1 == 0 and mono1)

    # (2) cascade -> g2(0)=0, interior peak matching the analytic t*
    G2 = g2(t, tau1, tau2)
    k2, interior2, monoaft2 = _interior_peak(t, G2)
    tstar = cascade_peak_time(tau1, tau2)
    tstar_rel_err = abs(t[k2] - tstar) / tstar
    starts_at_zero = abs(G2[0]) < 1e-12
    cascade_peaks = bool(interior2 and monoaft2 and starts_at_zero and tstar_rel_err < 0.01)

    # (3) coupled 2-compartment (real eigenvalues) -> downstream response peaks once
    a, b = 0.15, 0.30
    G3, M = coupled_response(t, tau1, tau2, a, b)
    eig = np.linalg.eigvals(M)
    overdamped = bool(np.all(np.abs(eig.imag) < 1e-9) and np.all(eig.real < 0))
    k3, interior3, monoaft3 = _interior_peak(t, G3)
    starts_at_zero3 = abs(G3[0]) < 1e-12
    coupled_peaks = bool(overdamped and interior3 and monoaft3 and starts_at_zero3)

    ok = bool(first_order_monotone and cascade_peaks and coupled_peaks)
    return {
        "first_order_argmax_idx": k1,
        "first_order_monotone": first_order_monotone,
        "cascade_starts_at_zero": bool(starts_at_zero),
        "cascade_peak_time": float(t[k2]),
        "cascade_peak_time_analytic": float(tstar),
        "cascade_peak_time_rel_err": float(tstar_rel_err),
        "cascade_peaks": cascade_peaks,
        "coupled_overdamped": overdamped,
        "coupled_peak_time": float(t[k3]),
        "coupled_peaks": coupled_peaks,
        "pass": ok,
    }


def main():
    r = run()
    print("=== §V.5 hydraulic memory-kernel shape synthetic unit test ===")
    print(f"  1st-order (single storage)  : argmax idx={r['first_order_argmax_idx']}"
          f" monotone={r['first_order_monotone']}  (=> peak at t=0, no lag peak)")
    print(f"  cascade peak time           : {r['cascade_peak_time']:.4f}"
          f"  (analytic {r['cascade_peak_time_analytic']:.4f},"
          f" rel-err {r['cascade_peak_time_rel_err']:.2e})")
    print(f"  cascade starts at 0 & peaks : {r['cascade_peaks']}")
    print(f"  coupled overdamped & peaks  : {r['coupled_peaks']}"
          f"  (peak t={r['coupled_peak_time']:.3f})")
    print(f"  PASS                        : {r['pass']}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
