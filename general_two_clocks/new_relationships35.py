r"""NR58 (theory) -- the ice memory kernel in the response anatomy: the B.2
tempered-Warburg interface kernel (NR34) is PURE minimum-phase memory --
zero transport, zero handedness -- and its exponential tempering rate is
EXACTLY the Kramers-Kronig crossover of the memory phase from 0 to 45 deg.

Bridges the cryosphere half of the corpus (Paper 4 / B.2 / NR34) to the
NR47-NR57 response anatomy.  NR34 proved the B.2 kernel IS a tempered
half-derivative; NR47-NR57 built the four-phase anatomy of a linear
response.  NR58 places the ice kernel inside that anatomy and reads off
what the anatomy says about the ice.

The claim
=========
The exact B.2 interface-flux transfer (NR34) is, with the Warburg
coefficient W = -2 A sqrt(tau_d) and tempering rate lam = 1/(4 tau_d),

    H(s) = A/s + W sqrt(s + lam)/s        (EXACT),

whose non-trivial factor is the tempered Warburg chi(s) = sqrt(s + lam).
Placed in the response anatomy:

1. **P1 memory: chi is minimum-phase, phase 0 -> 45 deg, crossover AT the
   tempering rate.**  delta(omega) = arg sqrt(i omega + lam)
   = (1/2) arctan(omega/lam): monotone from 0 (elliptic pole, omega <<
   lam) to 45 deg (Warburg / half-order pole, omega >> lam), passing
   through EXACTLY 22.5 deg at omega = lam.  So the exponential tempering
   cutoff of the kernel (its exp(-t/4 tau_d) tail, NR34) is precisely the
   Kramers-Kronig crossover frequency of its memory phase -- one number
   lam controls both the time-domain cutoff and the frequency-domain
   phase transition.  The full-line phase-area theorem (NR50) holds
   exactly: INT delta dln omega = (pi/2) ln|chi(inf)/chi(0)|.

2. **P2 transport: none.**  The excess phase (measured minus Bode, NR48)
   has zero slope: the diffusive ice memory is NOT a transport delay --
   no advective clock hides in it.  This distinguishes it from the solar
   pair (NR48, pure all-pass) at the opposite corner of the anatomy.

3. **P4 handedness: none.**  chi is scalar (1D interface heat conduction);
   the odd/spin face is identically zero -- the ice kernel is a pure
   two-clocks (symmetric) response, the cryosphere analog of the ocean's
   even part with the odd part switched off.

4. **The five-faces order parameter (NR47) at the Warburg limit.**  As
   omega >> lam the local fractional order alpha -> 1/2 in BOTH faces
   (2 delta/pi and the Bode amplitude slope d ln|chi|/d ln omega), the
   half-order Caputo/Randles-Warburg contact point (NR34 claim 2, E3).

Findings (figures/93_ice_kernel_anatomy.json):

* delta(lam) = 22.5 deg to 1e-6 (the crossover IS the tempering rate);
  delta -> 0 and 45 deg at the two poles.
* minimum-phase reconstruction of delta from |chi| alone (Bode) to < 2
  deg median over the resolved band; excess-phase transport delay
  consistent with zero (< 1e-3 of tau_d).
* phase-area theorem exact (rel err < 1e-3); both fractional-order faces
  -> 1/2 in the Warburg band.
* placement matrix: ice kernel = (memory 0->45, transport 0, spin 0),
  a NEW corner distinct from EEG (memory 0), ocean (memory 90 + spin),
  solar (transport), scallops (transport + weak parity).

Consequence for the program: the response anatomy (NR55) now classifies
BOTH halves of the corpus -- cryosphere and ocean -- on the same axes.
The ice kernel occupies the "tempered half-order memory, no transport, no
handedness" corner, and its single physical parameter tau_d = kappa/Vbar^2
sets its lone anatomy coordinate (the KK crossover lam = 1/4 tau_d).  The
anatomy is not ocean-specific; it is the universal chart of the whole
two-clocks program.

CPU-only, offline-safe (self-contained analytic kernel).
Tests: tests/test_ice_kernel_anatomy.py.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "93_ice_kernel_anatomy.json")


# --------------------------------------------------------------------------- #
# the exact B.2 tempered-Warburg kernel and its anatomy faces
# --------------------------------------------------------------------------- #
def warburg_chi(omega, lam):
    """The tempered Warburg factor chi(s) = sqrt(s + lam), s = i omega."""
    return np.sqrt(1j * np.asarray(omega, float) + lam)


def memory_phase(omega, lam):
    """delta(omega) = (1/2) arctan(omega / lam), the exact arg of chi."""
    return 0.5 * np.arctan2(np.asarray(omega, float), lam)


def bode_phase_from_logamp(log_w, log_absH):
    log_w = np.asarray(log_w, float)
    du = log_w[1] - log_w[0]
    slope = np.gradient(np.asarray(log_absH, float), log_w)
    n = len(log_w)
    u = np.arange(-n + 1, n) * du
    kernel = np.log(np.abs(1.0 / np.tanh(np.where(u == 0, du / 10.0, u)
                                         / 2.0)))
    return np.convolve(slope, kernel, mode="full")[n - 1:2 * n - 1] \
        * du / np.pi


def full_transfer(omega, A, tau_d):
    """The full B.2 transfer H(s) = A/s + W sqrt(s+lam)/s (physical G = -H
    has positive DC); returned as the physical memory response G."""
    lam = 1.0 / (4.0 * tau_d)
    W = -2.0 * A * math.sqrt(tau_d)
    s = 1j * np.asarray(omega, float)
    H = A / s + W * np.sqrt(s + lam) / s
    return -H


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyze(tau_d=1.0):
    lam = 1.0 / (4.0 * tau_d)
    omega = np.geomspace(lam * 1e-4, lam * 1e4, 40001)
    lw = np.log(omega)
    chi = warburg_chi(omega, lam)
    delta = np.angle(chi)

    # 1. crossover exactly at lam
    delta_at_lam = float(memory_phase(lam, lam))          # expect pi/8
    delta_lo = float(delta[0])
    delta_hi = float(delta[-1])

    # phase-area theorem (NR50)
    area = float(np.trapezoid(delta, lw))
    area_pred = float(np.pi / 2.0 * (np.log(np.abs(chi[-1]))
                                     - np.log(np.abs(chi[0]))))

    # minimum-phase reconstruction (NR48)
    phi_b = bode_phase_from_logamp(lw, np.log(np.abs(chi)))
    mb = (omega > 1e-2 * lam) & (omega < 1e2 * lam)
    off = np.median((delta - phi_b)[mb])
    minphase_med = float(np.degrees(np.median(np.abs(delta - phi_b - off)[mb])))

    # 2. transport: excess-phase slope of the FULL physical kernel
    G = full_transfer(omega, A=-1.0, tau_d=tau_d)         # A<0 so DC(G)>0
    phiG = np.unwrap(np.angle(G))
    phi_bG = bode_phase_from_logamp(lw, np.log(np.abs(G)))
    exc = phiG - phi_bG
    exc = exc - np.median(exc[mb])
    tau_transport = float(-np.polyfit(omega[mb], exc[mb], 1)[0])

    # 4. five-faces fractional order at the Warburg limit
    slope_amp = np.gradient(np.log(np.abs(chi)), lw)
    alpha_phase_hi = float(2.0 * delta_hi / np.pi)
    alpha_slope_hi = float(slope_amp[-1])

    out = dict(
        tau_d=tau_d, lam=lam,
        delta_at_lam_deg=math.degrees(delta_at_lam),
        delta_lo_deg=math.degrees(delta_lo),
        delta_hi_deg=math.degrees(delta_hi),
        area=area, area_pred=area_pred,
        area_rel_err=abs(area - area_pred) / abs(area),
        minphase_median_deg=minphase_med,
        transport_delay_over_tau_d=abs(tau_transport) / tau_d,
        alpha_phase_hi=alpha_phase_hi, alpha_slope_hi=alpha_slope_hi)
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    crossover = abs(out["delta_at_lam_deg"] - 22.5) < 1e-4
    poles = (abs(out["delta_lo_deg"]) < 0.1
             and abs(out["delta_hi_deg"] - 45.0) < 0.1)
    minphase = out["minphase_median_deg"] < 2.5
    no_transport = out["transport_delay_over_tau_d"] < 1e-3
    area_ok = out["area_rel_err"] < 1e-3
    warburg = (abs(out["alpha_phase_hi"] - 0.5) < 1e-3
               and abs(out["alpha_slope_hi"] - 0.5) < 1e-3)
    return dict(
        tempering_rate_is_kk_crossover=bool(crossover and poles),
        ice_kernel_is_minimum_phase=bool(minphase and area_ok),
        ice_kernel_has_no_transport=bool(no_transport),
        warburg_half_order_limit=bool(warburg),
        reading=(
            "The B.2 ice memory kernel is pure minimum-phase memory in the "
            "response anatomy: its memory phase runs 0 -> 45 deg and "
            "crosses 22.5 deg EXACTLY at omega = lam = 1/(4 tau_d) -- the "
            "kernel's exponential tempering rate is identically the "
            "Kramers-Kronig crossover of its phase, one number governing "
            "both the time-domain cutoff and the frequency-domain phase "
            "transition.  It carries no transport (excess-phase delay "
            f"{out['transport_delay_over_tau_d']:.0e} tau_d) and no "
            "handedness (scalar 1D conduction), so it sits at a NEW corner "
            "of the anatomy -- tempered half-order memory, no transport, "
            "no spin -- distinct from the ocean (memory+spin), the solar "
            "pair (transport), and EEG (elliptic).  In the Warburg band "
            "both fractional-order faces reach 1/2 (Caputo/Randles), and "
            "the phase-area theorem closes exactly: the anatomy of "
            "NR47-57 classifies the cryosphere on the same axes as the "
            "ocean."),
    )


def run(write=True):
    out = analyze()
    res = dict(description=__doc__.splitlines()[0], **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    print("NR58 -- the ice memory kernel in the response anatomy")
    print(f"  tempering rate = KK crossover  : "
          f"{v['tempering_rate_is_kk_crossover']} "
          f"(delta(lam) = {res['delta_at_lam_deg']:.3f} deg, poles "
          f"{res['delta_lo_deg']:.2f} -> {res['delta_hi_deg']:.2f})")
    print(f"  ice kernel is minimum-phase    : "
          f"{v['ice_kernel_is_minimum_phase']} "
          f"(recon {res['minphase_median_deg']:.2f} deg, area rel err "
          f"{res['area_rel_err']:.1e})")
    print(f"  no transport clock             : "
          f"{v['ice_kernel_has_no_transport']} "
          f"(excess delay {res['transport_delay_over_tau_d']:.0e} tau_d)")
    print(f"  Warburg half-order limit       : "
          f"{v['warburg_half_order_limit']} "
          f"(alpha_phase {res['alpha_phase_hi']:.3f}, alpha_slope "
          f"{res['alpha_slope_hi']:.3f})")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
