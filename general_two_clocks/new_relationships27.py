r"""NR50 (theory) -- the phase-area budget: Bode's area theorem makes the
integrated memory phase a conserved quantity fixed by the endpoint gains
(the waterbed), and the banded budget classifies every response into
transport (phase, no gain), causal memory (gain that pays phase), and
static weighting (gain that pays none).

Follows NR47 (memory phase = the universal coordinate), NR48 (minimum-phase
/ all-pass factorisation in omega), NR49 (gain x all-pass in k).  NR50 is
the INTEGRAL law of the same structure.

The claim
=========
1. **Phase-area theorem (exact).**  For a minimum-phase response with
   finite, nonzero endpoint gains (a genuine two-clocks response: an
   instantaneous clock chi(inf) plus a relaxed limit chi(0)),

       ``INT delta(omega) dln omega = (pi/2) [ln|chi(inf)| - ln|chi(0)|]``

   over the full line (Fubini on the Bode integral; the coth kernel has
   total weight pi^2/2).  The TOTAL memory phase is fixed by the two clock
   endpoints alone -- redistribution of relaxation times moves phase
   across scales but cannot create or destroy it (control theory's
   waterbed).  "Total memory is conserved; only its scale-distribution is
   free."

2. **The banded budget.**  Over a finite band, three distinct objects pay
   into the measured phase area ``A = INT phi_meas dln omega``:

     - transport (all-pass ``e^{-i omega tau_d}``): pays
       ``-tau_d (omega_2 - omega_1)`` of area with NO gain;
     - causal memory (minimum-phase): pays ``(pi/2) Delta ln|chi|`` --
       the endpoint-gain difference, waterbed-conserved;
     - static spectral weighting (a real positive per-frequency factor,
       e.g. instrument mode-sensitivity ratios): gain with NO phase -- a
       continuum of elliptic elements, NR39's invisibility.

   Measuring BOTH the phase area and the endpoint gains therefore
   CLASSIFIES the gain structure: if the phase area matches
   ``(pi/2) Delta ln|chi|`` the gain is causal dynamics; if the phase
   area falls far short, the gain tilt is static weighting.

Findings (figures/85_phase_area_budget.json):

* **Exact (synthetic).**  Lead-lag ``(1+i w tau_2)/(1+i w tau_1)``: the
  full-line area equals ``(pi/2) ln(tau_2/tau_1)`` to < 1e-6.  Waterbed: a
  two-pole/two-zero system with the SAME endpoint ratio but different
  interior structure has the SAME area to < 1e-6 while its delta(omega)
  differs pointwise -- redistribution without creation.
* **Ocean GSER (NR45/NR48 cache) -- the gain is causal dynamics.**  The
  measured banded phase area (6.04 rad-efold) matches the endpoint-gain
  prediction ``(pi/2) Delta ln|G*|`` (5.65) to ~6 %: the mesoscale
  modulus pays its full Bode phase, closing the budget with zero
  transport -- the integral (endpoint-only) face of NR48's pointwise
  minimum-phase verdict.
* **Solar delay pair (NR46/NR48 cache) -- the gain is static weighting.**
  BiSON x GOLF: the measured phase area (0.213) is explained by the
  fitted transport term alone (0.207, ~3 %), while a minimum-phase
  reading of the observed gain tilt would predict 0.76 -- 3.5x the
  measured area.  The phase VETOES the causal reading of the gain: the
  instrument-pair gain tilt is per-mode static sensitivity (elliptic
  continuum, no phase), and the only genuine dynamics between the two
  time series is the 10 s time-base transport of NR48.

Consequence for the program: the memory phase is not only pointwise
amplitude-determined (NR48) -- its INTEGRAL is an invariant of the two
clock endpoints, so "how much memory" is a two-number measurement
(chi(0), chi(inf)), robust to everything in between; and the budget
turns any measured (gain, phase) pair into a three-way classification of
the underlying physics: transport / causal memory / static weighting.

CPU-only, offline-safe (reads the committed NR48 cache).
Tests: tests/test_phase_area_budget.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr48_minphase_cache.json")
FIG = os.path.join(HERE, "figures", "85_phase_area_budget.json")


# --------------------------------------------------------------------------- #
# the area theorem
# --------------------------------------------------------------------------- #
def phase_area(log_w, phase_rad):
    """Banded phase area A = INT phase dln omega (rad * e-fold)."""
    return float(np.trapezoid(np.asarray(phase_rad, float),
                              np.asarray(log_w, float)))


def endpoint_area(log_absH_lo, log_absH_hi):
    """Minimum-phase (causal-memory) prediction from endpoint gains alone:
    (pi/2) * [ln|H|(hi) - ln|H|(lo)]."""
    return float(np.pi / 2.0 * (log_absH_hi - log_absH_lo))


def transport_area(freq_hz, tau_d_s, phase_offset=0.0):
    """Area paid by a pure all-pass delay phi = -omega tau_d + phi0 over the
    band spanned by freq_hz (analytic: -tau_d (w2 - w1) + phi0 ln(w2/w1))."""
    w = 2.0 * np.pi * np.asarray(freq_hz, float)
    return float(-tau_d_s * (w.max() - w.min())
                 + phase_offset * np.log(w.max() / w.min()))


def robust_endpoints(values, n=3):
    v = np.asarray(values, float)
    n = min(n, len(v) // 2)
    return float(np.median(v[:n])), float(np.median(v[-n:]))


# --------------------------------------------------------------------------- #
# synthetic exactness: area theorem + waterbed
# --------------------------------------------------------------------------- #
def synthetic_checks():
    w = np.geomspace(1e-8, 1e8, 60001)
    u = np.log(w)
    t1, t2 = 3.0, 0.2
    H1 = (1 + 1j * w * t2) / (1 + 1j * w * t1)
    area1 = phase_area(u, np.angle(H1))
    pred = endpoint_area(0.0, np.log(t2 / t1))        # ln|H(0)|=0, ln|H(inf)|
    # waterbed: same endpoint ratio, different interior structure
    a1, b1, a2 = 5.0, 0.7, 1.1
    b2 = (t2 / t1) * a1 * b1 / a2
    H2 = ((1 + 1j * w * a2) * (1 + 1j * w * b2)
          / ((1 + 1j * w * a1) * (1 + 1j * w * b1)))
    area2 = phase_area(u, np.angle(H2))
    mid = np.argmin(np.abs(u))
    return dict(
        leadlag_area=area1, leadlag_pred=pred,
        leadlag_err=abs(area1 - pred),
        waterbed_area=area2, waterbed_diff=abs(area2 - area1),
        shapes_differ=float(np.max(np.abs(np.angle(H2) - np.angle(H1)))),
        mid_phase_1=float(np.angle(H1)[mid]),
        mid_phase_2=float(np.angle(H2)[mid]))


# --------------------------------------------------------------------------- #
# real-data budgets (committed NR48 cache)
# --------------------------------------------------------------------------- #
def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


def ocean_budget(cache):
    o = cache["ocean_gser"]
    lw = np.array(o["log_w"])
    lnG = np.array(o["log_Gmag"])
    dm = np.array(o["delta_meas_rad"])
    area_meas = phase_area(lw, dm)
    # the banded identity pairs the integral with the TRUE band endpoints;
    # lnG comes from the ensemble-averaged MSD and is smooth at the edges
    area_pred = endpoint_area(float(lnG[0]), float(lnG[-1]))
    return dict(area_meas=area_meas, area_pred_minphase=area_pred,
                rel_closure=abs(area_meas - area_pred) / abs(area_meas))


def solar_budget(cache):
    d = cache["delay_clock"]
    fr = np.array(d["freq_hz"])
    ph = np.array(d["phase_rad"])
    g2 = np.array(d["g2"])
    gain = np.array(d["gain"])
    coh = g2 >= 0.5
    f = fr[coh]
    p = np.unwrap(ph[coh])
    lg = np.log(2.0 * np.pi * f)
    area_meas = phase_area(lg, p)
    # transport fit (NR48 convention): phi = -w tau_d + phi0, weights g2
    wgt = g2[coh]
    A = np.vstack([2.0 * np.pi * f, np.ones_like(f)]).T
    sol, *_ = np.linalg.lstsq(A * wgt[:, None], p * wgt, rcond=None)
    tau_d, phi0 = float(-sol[0]), float(sol[1])
    area_tr = phase_area(lg, sol[0] * 2.0 * np.pi * f + phi0)
    # naive minimum-phase reading of the gain tilt
    lo, hi = robust_endpoints(np.log(gain[coh]), n=10)
    area_minphase_naive = endpoint_area(lo, hi)
    return dict(area_meas=area_meas, area_transport=area_tr,
                tau_d_s=tau_d, phi0_rad=phi0,
                rel_closure_transport=abs(area_meas - area_tr)
                / abs(area_meas),
                area_minphase_naive=area_minphase_naive,
                naive_overshoot=abs(area_minphase_naive) / abs(area_meas),
                n_coherent=int(coh.sum()))


def analyze(cache):
    syn = synthetic_checks()
    oc = ocean_budget(cache)
    so = solar_budget(cache)
    out = dict(synthetic=syn, ocean=oc, solar=so)
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    syn = out["synthetic"]
    oc = out["ocean"]
    so = out["solar"]
    exact = (syn["leadlag_err"] < 1e-6 and syn["waterbed_diff"] < 1e-6
             and syn["shapes_differ"] > 0.1)
    ocean_ok = oc["rel_closure"] < 0.10
    solar_ok = (so["rel_closure_transport"] < 0.10
                and so["naive_overshoot"] > 2.0)
    return dict(
        area_theorem_and_waterbed_exact=bool(exact),
        ocean_gain_is_causal_memory=bool(ocean_ok),
        solar_gain_is_static_weighting=bool(solar_ok),
        reading=(
            "The integrated memory phase is a conserved quantity: over the "
            "full line it equals (pi/2) ln[chi(inf)/chi(0)] exactly "
            "(lead-lag to {:.0e}; waterbed redistribution changes the "
            "phase shape by {:.2f} rad pointwise yet moves the area by "
            "only {:.0e}).  On real data the banded budget classifies the "
            "gain: the ocean modulus pays its full Bode phase "
            "(area {:.2f} vs endpoint prediction {:.2f}, {:.0%} closure) "
            "-- causal memory; the solar pair's phase area ({:.3f}) is "
            "pure transport ({:.3f}, tau_d = {:.1f} s), while a causal "
            "reading of its gain tilt would demand {:.2f} -- {:.1f}x too "
            "much -- so that tilt is static per-mode weighting, gain "
            "without phase, the elliptic continuum."
        ).format(syn["leadlag_err"], syn["shapes_differ"],
                 syn["waterbed_diff"], oc["area_meas"],
                 oc["area_pred_minphase"], oc["rel_closure"],
                 so["area_meas"], so["area_transport"], so["tau_d_s"],
                 so["area_minphase_naive"], so["naive_overshoot"]),
    )


def run(write=True):
    cache = load_cache()
    out = analyze(cache)
    res = dict(description=__doc__.splitlines()[0],
               meta=cache["meta"], **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    syn, oc, so = res["synthetic"], res["ocean"], res["solar"]
    print("NR50 -- the phase-area budget (conserved memory + classification)")
    print(f"  area theorem + waterbed exact : "
          f"{v['area_theorem_and_waterbed_exact']} "
          f"(err {syn['leadlag_err']:.1e}, waterbed {syn['waterbed_diff']:.1e},"
          f" shapes differ {syn['shapes_differ']:.2f} rad)")
    print(f"  ocean gain = causal memory    : "
          f"{v['ocean_gain_is_causal_memory']} "
          f"(area {oc['area_meas']:.2f} vs pred {oc['area_pred_minphase']:.2f},"
          f" closure {oc['rel_closure']:.1%})")
    print(f"  solar gain = static weighting : "
          f"{v['solar_gain_is_static_weighting']} "
          f"(transport closes {so['rel_closure_transport']:.1%}; causal "
          f"reading overshoots x{so['naive_overshoot']:.1f})")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
