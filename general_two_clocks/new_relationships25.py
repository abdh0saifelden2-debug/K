r"""NR48 (theory) -- the two-clocks response is minimum-phase, so the memory
phase is the Bode transform of the amplitude; a genuine transport delay is a
third, all-pass clock read off the excess phase.

Follows NR47 (the memory phase delta = arg chi as the universal two-clocks
coordinate).  NR47 used the *local* Bode relation ``delta ~ (pi/2) dln|chi|/
dln omega`` (exact only for a pure power law).  NR48 upgrades it to the exact
*global* Bode gain-phase integral and draws two consequences the corpus needs.

The claim
=========
1. **A passive relaxational (two-clocks) response is minimum-phase**, so gain
   and phase are a Hilbert pair (Bode 1945):

       ``delta(omega0) = (1/pi) INT (dln|chi|/du) ln|coth(|u|/2)| du``,
       ``u = ln(omega/omega0)``.

   The phase carries NO information independent of the amplitude spectrum.
   This is exactly *why GSER works*: recovering the full complex modulus
   ``G*(omega)`` from the amplitude/MSD alone (Mason-Weitz) is legitimate
   only because the modulus is minimum-phase -- NR48 states that assumption
   and checks it on the real ocean data.

2. **A pure transport delay is the unique all-pass factor**
   ``exp(-i omega tau_d)``: ``|chi| = 1`` (no amplitude signature, zero Bode
   phase) but a linear phase ``-omega tau_d``.  So the *excess phase*
   ``phi_excess = phi_measured - Bode[ln|chi|]`` isolates advective/transport
   delay, and its slope is ``tau_d`` -- a model-free delay measurement,
   cleanly separated from relaxational memory.

3. **Three clocks, not two.**  The two-clocks axis (NR47) gets a third,
   orthogonal element:
     - elliptic (instantaneous): ``delta = 0``, ``|chi|`` flat, zero excess;
     - parabolic (relaxational memory): ``delta`` = Bode transform of the
       amplitude, minimum-phase;
     - delay (advective): all-pass, ``|chi| = 1``, excess phase linear in
       ``omega`` with slope ``tau_d``.
   Any measured response decomposes uniquely into (amplitude -> minimum-phase
   part) + (all-pass delay).  This is the standard control-theory
   minimum-phase / all-pass factorisation, here made the *measurement
   protocol* for the two-clocks program.

Findings (figures/83_minphase_delay.json):

* **Synthetic (exact).**  For Debye, fractional and two-pole minimum-phase
  models the Bode integral reconstructs the phase from ``ln|chi|`` alone to
  ~0.01-0.04 deg median.  A pure delay ``exp(-i omega tau_d)`` gives zero
  Bode phase and an excess-phase slope that recovers ``tau_d`` to machine
  precision.
* **Ocean GSER is minimum-phase (real data, NR45).**  The GSER phase
  ``delta = pi*alpha/2`` measured from the MSD slope agrees with the Bode
  transform of the measured ``ln|G*|`` to ~3 deg median over the interior of
  the resolved band (the finite 1.5-decade band is extended with its terminal
  power-law slopes before the integral -- exactly the Mason-Weitz local
  power-law assumption).  This is the Kramers-Kronig/minimum-phase assumption
  implicit in every microrheology inversion, now verified on the mesoscale
  ocean.
* **A delay clock, measured model-free on real data (NR46 pair).**  BiSON x
  GOLF velocity-velocity (two independent Doppler instruments, 2003-2006):
  flat gain (minimum-phase Bode phase ~ 0) and a phase LINEAR in frequency ->
  excess-phase slope = a pure timing delay.  With the cross-spectrum
  ``S_xy = GOLF x conj(BiSON)`` and the delay convention
  ``chi = exp(-i omega tau_d)``, the fit gives ``tau_d = -10 s``: GOLF leads
  BiSON by ~10 s -- a pure time-base offset between the two instruments,
  extracted with no model.  This is the third clock in isolation: no
  amplitude signature, all phase.

Consequence for the whole program: the "memory phase" of NR47 is
amplitude-determined for every relaxational (parabolic) system, so an
amplitude spectrum suffices to read the memory clock; and a nonzero excess
phase is the unambiguous signature of a genuine propagation/advection time
that no relaxation can fake -- the delay counterpart of NR39's "instantaneous
mixing is invisible to the imaginary part."

CPU-only, offline-safe.  Tests: tests/test_minphase_delay.py.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr48_minphase_cache.json")
FIG = os.path.join(HERE, "figures", "83_minphase_delay.json")


# --------------------------------------------------------------------------- #
# the Bode gain-phase integral (minimum-phase) + delay factorisation
# --------------------------------------------------------------------------- #
def bode_phase_from_logamp(log_w, log_absH):
    """Exact Bode gain-phase integral on a uniform log-omega grid:
    phi(w0) = (1/pi) INT (dln|H|/du) ln|coth(|u|/2)| du, u = ln(w/w0).

    Returns the minimum-phase phase (radians), determined by |H| alone up to
    an additive constant fixed by the high-frequency limit."""
    log_w = np.asarray(log_w, float)
    du = log_w[1] - log_w[0]
    if not np.allclose(np.diff(log_w), du, rtol=1e-4):
        raise ValueError("log_w must be a uniform grid")
    slope = np.gradient(np.asarray(log_absH, float), log_w)
    n = len(log_w)
    u = np.arange(-n + 1, n) * du
    kernel = np.log(np.abs(1.0 / np.tanh(np.where(u == 0, du / 10.0, u) / 2.0)))
    return np.convolve(slope, kernel, mode="full")[n - 1:2 * n - 1] * du / np.pi


def bode_phase_banded(log_w, log_absH, pad_decades=2.0, slope_frac=0.2):
    """Bode phase on a finite band: extend ln|H| beyond the measured band
    with its terminal power-law slopes (robust fit over the outer
    ``slope_frac`` of the band) before the integral, then return the phase on
    the original grid.  The extension is the Mason-Weitz assumption that the
    terminal power laws continue; without it the finite-band integral is
    biased near the edges."""
    log_w = np.asarray(log_w, float)
    log_absH = np.asarray(log_absH, float)
    du = log_w[1] - log_w[0]
    npad = max(1, int(round(pad_decades * math.log(10.0) / du)))
    k = max(3, int(round(slope_frac * len(log_w))))
    sl_lo = np.polyfit(log_w[:k], log_absH[:k], 1)[0]
    sl_hi = np.polyfit(log_w[-k:], log_absH[-k:], 1)[0]
    ext = np.concatenate([
        log_absH[0] + sl_lo * (np.arange(-npad, 0) * du),
        log_absH,
        log_absH[-1] + sl_hi * (np.arange(1, npad + 1) * du)])
    grid = np.concatenate([
        log_w[0] + np.arange(-npad, 0) * du,
        log_w,
        log_w[-1] + np.arange(1, npad + 1) * du])
    return bode_phase_from_logamp(grid, ext)[npad:npad + len(log_w)]


def excess_phase(log_w, log_absH, phi_measured):
    """phi_excess = phi_measured - Bode[ln|H|], aligned to zero mean; its
    slope in omega is the transport delay tau_d."""
    phi_b = bode_phase_from_logamp(log_w, log_absH)
    exc = np.asarray(phi_measured, float) - phi_b
    return exc - np.median(exc), phi_b


def delay_from_phase(freq_hz, phase_rad, weights=None):
    """Fit a linear phase(omega) -> transport delay tau_d = -slope/(2 pi)."""
    w = 2.0 * np.pi * np.asarray(freq_hz, float)
    ph = np.unwrap(np.asarray(phase_rad, float))
    W = np.ones_like(w) if weights is None else np.asarray(weights, float)
    A = np.vstack([w, np.ones_like(w)]).T
    WA = A * W[:, None]
    sol, *_ = np.linalg.lstsq(WA, ph * W, rcond=None)
    return float(-sol[0]), float(sol[1])       # tau_d, phase offset


# --------------------------------------------------------------------------- #
# synthetic minimum-phase models (self-contained)
# --------------------------------------------------------------------------- #
def _minphase_models(w):
    return {
        "debye": 1.0 / (1 + 1j * w * 1.0),
        "fractional": (1j * w) ** 0.6 / (1 + (1j * w) ** 0.6),
        "two_pole": 1.0 / ((1 + 1j * w * 2.0) * (1 + 1j * w * 0.2)),
    }


def synthetic_checks():
    w = np.geomspace(1e-3, 1e3, 3000)
    lw = np.log(w)
    recon = {}
    for name, H in _minphase_models(w).items():
        phi_true = np.unwrap(np.angle(H))
        phi_b = bode_phase_from_logamp(lw, np.log(np.abs(H)))
        mb = (w > 1e-2) & (w < 1e2)
        off = np.median((phi_true - phi_b)[mb])
        recon[name] = float(np.median(np.degrees(
            np.abs(phi_true - phi_b - off)[mb])))
    # pure delay: all-pass, recover tau_d from excess phase
    tau = 0.3
    Hd = np.exp(-1j * w * tau)
    mb = (w > 0.1) & (w < 5)
    tau_rec, _ = delay_from_phase(w[mb] / (2 * np.pi), np.angle(Hd)[mb])
    bode_delay = float(np.max(np.abs(bode_phase_from_logamp(
        lw, np.log(np.abs(Hd))))))
    return dict(recon_median_deg=recon, delay_tau_true=tau,
                delay_tau_recovered=tau_rec, delay_bode_phase_max=bode_delay)


# --------------------------------------------------------------------------- #
# real-data anchors (committed cache)
# --------------------------------------------------------------------------- #
def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


def analyze(cache):
    syn = synthetic_checks()

    # ocean GSER minimum-phase self-consistency (terminal-slope extension,
    # interior 70% of the resolved band; edges of a finite Bode integral are
    # truncation-biased)
    o = cache["ocean_gser"]
    lw = np.array(o["log_w"])
    grid = np.linspace(lw.min() + 1e-9, lw.max() - 1e-9, 200)
    lnG = np.interp(grid, lw, np.array(o["log_Gmag"]))
    dm = np.interp(grid, lw, np.array(o["delta_meas_rad"]))
    phi_b = bode_phase_banded(grid, lnG)
    span = grid[-1] - grid[0]
    mb = (grid >= grid[0] + 0.15 * span) & (grid <= grid[-1] - 0.15 * span)
    off = np.median((dm - phi_b)[mb])
    ocean_err = np.degrees(np.abs(dm - phi_b - off)[mb])
    ocean_med = float(np.median(ocean_err))

    # real delay clock (BiSON x GOLF V-V)
    d = cache["delay_clock"]
    fr = np.array(d["freq_hz"])
    ph = np.array(d["phase_rad"])
    g2 = np.array(d["g2"])
    gain = np.array(d["gain"])
    coh = g2 >= 0.5
    tau_d, _ = delay_from_phase(fr[coh], ph[coh], weights=g2[coh])
    gain_flatness = float(np.std(np.log(gain[coh])))

    out = dict(
        synthetic=syn,
        ocean_minphase_median_deg=ocean_med,
        delay_tau_d_s=tau_d,
        delay_gain_log_std=gain_flatness,
        delay_n_coherent=int(coh.sum()))
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    syn = out["synthetic"]
    minphase = (max(syn["recon_median_deg"].values()) < 0.1
                and abs(syn["delay_tau_recovered"] - syn["delay_tau_true"])
                < 1e-6
                and syn["delay_bode_phase_max"] < 1e-6)
    gser_minphase = out["ocean_minphase_median_deg"] < 6.0
    real_delay = (0.0 < abs(out["delay_tau_d_s"]) < 30.0
                  and out["delay_gain_log_std"] < 0.6)
    return dict(
        two_clocks_response_is_minimum_phase=bool(minphase),
        gser_is_minimum_phase_on_real_ocean=bool(gser_minphase),
        delay_is_allpass_excess=bool(
            syn["delay_bode_phase_max"] < 1e-6
            and abs(syn["delay_tau_recovered"] - syn["delay_tau_true"])
            < 1e-6),
        real_delay_clock_measured=bool(real_delay),
        reading=(
            "The two-clocks response is minimum-phase: gain and phase are a "
            "Bode/Hilbert pair, so the memory phase is the amplitude "
            "spectrum's transform (reconstructed to ~0.01-0.04 deg on "
            "synthetics), and GSER's amplitude-only modulus inversion is "
            "legitimate exactly because the ocean modulus is minimum-phase "
            f"(delta=pi*alpha/2 matches Bode[ln|G*|] to "
            f"{out['ocean_minphase_median_deg']:.1f} deg on ANDRO).  A pure "
            "transport delay is the unique all-pass factor -- no amplitude "
            "signature, linear excess phase -- so the third clock is read "
            "model-free from the excess-phase slope: BiSON x GOLF gives "
            f"tau_d = {out['delay_tau_d_s']:.1f} s under the "
            "GOLFxconj(BiSON), chi=exp(-i w tau_d) convention (GOLF leads "
            f"BiSON by {abs(out['delay_tau_d_s']):.0f} s) -- a pure "
            "time-base offset between two independent velocity instruments.  "
            "Three clocks: "
            "instantaneous (delta=0), relaxational (amplitude-determined "
            "phase), advective (all-pass delay)."),
    )


def run(write=True):
    cache = load_cache()
    out = analyze(cache)
    res = dict(description=__doc__.splitlines()[0], meta=cache["meta"], **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    s = res["synthetic"]
    print("NR48 -- minimum-phase two-clocks + the delay clock")
    print(f"  minimum-phase reconstruction : "
          f"{v['two_clocks_response_is_minimum_phase']} "
          f"(Bode phase from |chi| alone, median err "
          f"{max(s['recon_median_deg'].values()):.3f} deg worst model)")
    print(f"  GSER = minimum-phase (ocean)  : "
          f"{v['gser_is_minimum_phase_on_real_ocean']} "
          f"(delta=pi*alpha/2 vs Bode[ln|G*|] median "
          f"{res['ocean_minphase_median_deg']:.1f} deg)")
    print(f"  delay = all-pass excess       : {v['delay_is_allpass_excess']} "
          f"(synthetic tau recovered "
          f"{s['delay_tau_recovered']:.4f} vs {s['delay_tau_true']:.4f})")
    print(f"  real delay clock (BiSONxGOLF) : "
          f"{v['real_delay_clock_measured']} (tau_d = "
          f"{res['delay_tau_d_s']:.1f} s, gain log-std "
          f"{res['delay_gain_log_std']:.2f}, "
          f"{res['delay_n_coherent']} coherent bins)")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
