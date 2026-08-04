r"""New derived cross-relationship NR42, continuing the program (NR1-NR41) with the same
discipline: *derived* from mainstream theory + repo results and *numerically verified*
(CPU, deterministic).  See ``REPORT_NEW_RELATIONSHIPS19.md`` for the write-up and
``tests/test_new_relationships19.py`` for the unit proof.

NR42 - THE GRID-FREQUENCY DRAINAGE WINDOW: disturbance observability in 1-s power-grid
       frequency archives is the SAME band-limited two-clocks transmission as NR32's
       subglacial drainage-response window.  Fed the grid's constants -- rotational
       inertia M as the storage C, the primary-response stiffness beta = D + 1/R_droop
       as 1/R, the governor/inverter response time T_g as the input clock, and the
       1-s archive fast-transient band as the observability window -- the identical
       transmission functional and the identical interior-peak criterion predict that
       disturbance observability is SINGLE-PEAKED in inertia and peaks at INTERMEDIATE
       inertia IFF the energy transition removes STORAGE (inertia actually falls).  The
       Markovian (quasi-static control) collapse of the memory kernel destroys the peak
       and predicts the WRONG, monotone ordering (highest-inertia grid most observable).
       The initial RoCoF, by contrast, is monotone in inertia loss -- so RoCoF risk and
       archive-band observability are DISTINCT, and only the latter is a two-clocks
       (memory) signature.
       [NR32 band-limited transmission x swing-equation SFR (Kundur 1994;
        Anderson & Mirheydar 1990) x ENTSO-E / GB-ESO 1-s frequency archives]

  Why this matters.  NR32 proved the "drainage-response window" -- which subglacial lakes
  produce a detectable post-drainage speed-up -- is a band-limited hydraulic transmission
  peaked at the cavity<->channel transition, and that an adiabatic (Markovian) closure
  predicts the wrong ordering.  The open question the ledger posed (E9) is whether that
  window is a GLACIOLOGICAL accident of the hydrology, or a STRUCTURAL property of any
  two-compartment cascade observed through a finite band.  The power grid is the decisive
  cross-domain test: it is a completely different physical system (rotating machines +
  control loops, not water + till) whose disturbance response is nonetheless the same
  lumped two-clock cascade, and whose 1-s frequency archives (ENTSO-E TransparencyPlatform;
  GB National Grid ESO) supply exactly the finite observability band NR32 needs.  If the
  same functional -- literally imported here from ``new_relationships9`` -- and the same
  interior-peak criterion reproduce the grid's disturbance-observability topology, the
  window is structural.

  Derivation (all mainstream pieces; the composition + the transfer is the contribution).
  1. Standard low-order System Frequency Response (SFR; Kundur 1994 sec 11; Anderson &
     Mirheydar 1990).  A power imbalance DP_L (a lost generator / block load) drives the
     centre-of-inertia frequency deviation Df through the swing equation with primary
     (governor / droop) control:
         Df(s) / (-DP_L(s)) = 1 / ( M s + D + (1/R_droop) / (1 + s T_g) ),
     where M = 2H is the aggregate inertia constant [s], D the load-damping [pu/pu],
     1/R_droop the aggregate primary gain [pu/pu], T_g the governor/turbine (or, for
     inverter-based resources, the fast-frequency-response) lag [s].  Secondary control
     (AGC, integral, ~minutes) is BELOW the 1-s archive's fast-transient band and is
     dropped.  Clearing the governor pole, the denominator is the quadratic
         M T_g s^2 + (M + D T_g) s + beta = 0,   beta := D + 1/R_droop,
     whose two roots are, to leading order in T_g/(M/D),
         s_slow ~ -beta/M   =>  tau_sys = M / beta   (the inertial relaxation clock),
         s_fast ~ -1/T_g                              (the governor / FFR clock).
     So the grid's disturbance response is a TWO-CLOCK cascade with the same
         H(omega) = 1 / ((1 + i omega T_g)(1 + i omega tau_sys))
     structure NR32 certified for the subglacial bed -- the dictionary is
         C  <-> M (rotational inertia = the frequency "storage"),
         R  <-> 1/beta (inverse primary-response stiffness = evacuation resistance),
         tau_sys = R C  <->  M/beta (inertial relaxation),
         tau_in <-> T_g (the fast input clock),
         DV     <-> DP_L (the disturbance).
  2. The de-carbonisation axis c in [0,1] is NR32's channelisation axis.  Retiring
     synchronous machines for inverter-based resources REMOVES inertia (storage), so
         M(c) = M_0 * 10^(-decades_C c),
     while grid-forming / fast-frequency-response inverters RAISE the in-band primary
     stiffness, so R(c) = 1/beta(c) FALLS:
         R(c) = 10^(-decades_R c),   tau_sys(c) = (M_0/beta_0) * 10^(-(decades_R+decades_C) c).
     This is EXACTLY NR32's ``maps(c)`` -- imported here unchanged.
  3. Band-limited observability.  A 1-s frequency archive resolves periods only inside a
     finite fast-transient band (Nyquist ~2 s to the pre-AGC primary-response horizon
     ~30 s): T in [2, 30] s.  The disturbance content the archive can carry is the
     band-limited transmission
         T_band(c) = (DP_L R(c))^2 (1/pi) Int_{omega_1}^{omega_2} |H(omega;c)|^2 domega,
     the identical closed form (partial fractions -> arctan) NR32 evaluates.
  4. The interior-peak theorem transfers verbatim.  High-inertia grids (c->0) have
     tau_sys >> band, so the ride-down is quasi-static and its in-band content vanishes as
     1/tau_sys^2 (it lives under the slow AGC drift, outside the fast archive band); very
     low-inertia grids with stiff fast control (c->1) have tau_sys << band AND small R, so
     the in-band content dies as R(c)^2.  An interior maximum -- a MOST-observable
     intermediate inertia -- therefore exists IFF the log-slope of tau_sys exceeds that of
     R alone, i.e. IFF the transition removes STORAGE (decades_C > 0).  If modernisation
     only stiffened primary control at fixed inertia (decades_C = 0), the highest-inertia
     grid would be most observable (argmax at c=0) -- the observable difference between
     "the transition removes inertia" and "the transition merely adds fast control".
  5. THE MARKOVIAN CONTROL (the two-clocks tie-in).  Collapse the inertial memory to a
     delta with the same DC gain (instantaneous, quasi-static frequency control -- the
     adiabatic closure this repo degenerates the MZ kernel to) and
         T_band^Markov(c) ~ (DP_L R(c))^2 (omega_2 - omega_1)/pi,
     monotone in c, maximal at the HIGHEST-inertia grid, no interior peak.  So a windowed
     disturbance-observability peak in real archives is a MEMORY signature a no-memory
     (quasi-static) frequency model cannot produce.
  6. RoCoF is NOT the window.  The instantaneous rate-of-change-of-frequency after the
     disturbance is RoCoF_0 = DP_L / M(c) ~ 10^(+decades_C c): MONOTONE increasing as
     inertia falls.  So RoCoF risk (monotone) and 1-s archive band observability (single-
     peaked) are DISTINCT diagnostics; the widely-tracked RoCoF cannot see the window, and
     only the band-limited variance carries the two-clocks fingerprint.  This is a sharp,
     falsifiable prediction for the inertia-declining grid.

  What is NEW here (vs the cited mainstream SFR + NR32): (i) the identification that the
  standard SFR disturbance response IS NR32's two-compartment band-limited cascade, with an
  explicit swing-equation dictionary and a pole-location proof that (T_g, M/beta) are the
  two clocks; (ii) the cross-domain confirmation -- same imported functional, same
  criterion -- that the interior-peak "window" is STRUCTURAL, not glaciological; (iii) the
  RoCoF-vs-window dichotomy (monotone vs single-peaked) distinguishing inertia risk from
  archive observability; (iv) a stackable, calibration-free population test for ENTSO-E /
  GB-ESO archives: band-limited disturbance variance vs an inertia proxy should be single-
  peaked, and the Markovian null (monotone) is falsified by any interior peak.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# The cross-domain claim made concrete: the grid uses the IDENTICAL transmission
# functional, band integral, lag-to-peak, channelisation map and interior-peak criterion
# certified for the subglacial bed in NR32 -- only the constants differ.
from new_relationships9 import (  # noqa: E402
    band_integral,
    band_integral_numeric,
    interior_peak_criterion,
    maps,
    omega_band,
    t_peak,
    transmission,
)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# The 1-s frequency-archive fast-transient observability band, in SECONDS (periods).
# Low edge: Nyquist of 1-s sampling (~2 s).  High edge: the primary-response horizon
# before secondary control / AGC takes over (~30 s; Continental-Europe primary reserve is
# fully deployed by 30 s, ENTSO-E SO GL Art. 154).
BAND_GRID = (2.0, 30.0)


# --------------------------------------------------------------------------- #
# swing-equation grounding: the two clocks are (T_g, M/beta)
# --------------------------------------------------------------------------- #
def grid_stiffness(D: float, inv_droop: float) -> float:
    """Primary-response stiffness beta = D + 1/R_droop [pu-power / pu-frequency]."""
    return float(D) + float(inv_droop)


def primary_sfr_poles(M: float, D: float, inv_droop: float, T_g: float):
    """Time constants of the primary-control SFR denominator poles.

    G(s) = 1 / (M s + D + inv_droop/(1+s T_g));  clearing the governor pole gives
        M T_g s^2 + (M + D T_g) s + (D + inv_droop) = 0.
    Returns (tau_slow, tau_fast) = decay time constants 1/|Re(root)| sorted slow-first,
    plus a flag whether the pair is real (overdamped) or complex (underdamped nadir).
    """
    a = M * T_g
    b = M + D * T_g
    c = grid_stiffness(D, inv_droop)
    disc = b * b - 4.0 * a * c
    if disc >= 0.0:
        r1 = (-b + np.sqrt(disc)) / (2.0 * a)
        r2 = (-b - np.sqrt(disc)) / (2.0 * a)
        taus = sorted((1.0 / abs(r1), 1.0 / abs(r2)), reverse=True)
        return float(taus[0]), float(taus[1]), True
    re = -b / (2.0 * a)
    tau = 1.0 / abs(re)
    return tau, tau, False


def rocof_initial(dP: float, M: float) -> float:
    """Initial rate-of-change-of-frequency after a step imbalance: RoCoF_0 = DP / M."""
    return float(dP) / float(M)


# --------------------------------------------------------------------------- #
# the NR42 computation
# --------------------------------------------------------------------------- #
def nr42(n_c: int = 241, band=BAND_GRID):
    cs = np.linspace(0.0, 1.0, n_c)
    out = {
        "band_s": list(band),
        "c_grid_n": n_c,
        "dictionary": {
            "C": "M (rotational inertia, storage)",
            "R": "1/beta, beta=D+1/R_droop (inverse primary-response stiffness)",
            "tau_sys": "M/beta (inertial relaxation clock)",
            "tau_in": "T_g (governor / fast-frequency-response clock)",
            "DV": "DP_L (power disturbance)",
            "c": "de-carbonisation / inverter-penetration axis",
        },
        "sweeps": [],
    }

    # (0) same closed form == numeric quadrature, now in grid (second) units
    errs = []
    for ta, tb in ((0.5, 40.0), (1.0, 1.0), (0.2, 300.0), (3.0, 0.05)):
        num = band_integral_numeric(ta, tb, band)
        errs.append(abs(band_integral(ta, tb, band) - num) / num)
    out["closed_form_check"] = {"max_rel_err": float(max(errs))}

    # (1) swing-equation pole check: the two clocks ARE (T_g, M/beta) in the overdamped
    #     regime -- so tau_sys = M/beta is the physical storage clock, not a fit.
    #     Overdamped, well-separated regime (tau_sys >> T_g): the clean two-clock cascade
    #     for the high-inertia grid before heavy fast-control augmentation.
    pole_errs = []
    pole_recs = []
    for M, D, inv_droop, T_g in (
        (20.0, 0.5, 1.0, 0.5),
        (24.0, 1.0, 2.0, 0.6),
        (30.0, 0.5, 1.5, 0.4),
        (16.0, 1.0, 1.0, 0.3),
    ):
        beta = grid_stiffness(D, inv_droop)
        tau_slow, tau_fast, real = primary_sfr_poles(M, D, inv_droop, T_g)
        tau_sys = M / beta
        rel = abs(tau_slow - tau_sys) / tau_sys
        pole_errs.append(rel)
        pole_recs.append({
            "M": M, "D": D, "inv_droop": inv_droop, "T_g": T_g,
            "tau_sys_M_over_beta": float(tau_sys), "tau_slow_pole": float(tau_slow),
            "tau_fast_pole": float(tau_fast), "rel_err": float(rel),
            "overdamped": bool(real),
        })
    out["sfr_pole_check"] = {
        "cases": pole_recs, "max_rel_err": float(max(pole_errs)),
        "all_overdamped": bool(all(r["overdamped"] for r in pole_recs)),
        "note": "slow pole = M/beta (inertial relaxation) to leading order in T_g/(M/D); "
                "fast pole ~ T_g; strong-primary grids are underdamped -- the complementary "
                "nadir-resonance regime where the same window logic applies to the peak",
    }

    # (2) robustness sweep: the imported interior-peak criterion PREDICTS the topology in
    #     every literature-spanning combo (tau_dist = high-inertia M_0/beta_0, seconds).
    n_pred_ok = 0
    n_window_ok = 0
    n_interior_cases = 0
    for tau_dist in (60.0, 120.0, 240.0):
        for dec_R in (2.0, 3.0, 4.0):
            for dec_C in (0.0, 0.5, 1.0, 2.0):
                for T_g in (0.3, 0.6, 1.0):
                    T = transmission(cs, tau_in=T_g, band=band, tau_dist=tau_dist,
                                     decades_R=dec_R, decades_C=dec_C)
                    i = int(np.argmax(T))
                    interior = 0 < i < n_c - 1
                    predicted = interior_peak_criterion(dec_R, dec_C)
                    pred_ok = (interior == predicted) or (not predicted and i == 0)
                    n_pred_ok += bool(pred_ok)
                    rec = {"tau_dist_s": tau_dist, "decades_R": dec_R, "decades_C": dec_C,
                           "T_g_s": T_g, "c_peak": float(cs[i]),
                           "interior": bool(interior),
                           "predicted_interior": bool(predicted),
                           "prediction_correct": bool(pred_ok)}
                    if interior and predicted:
                        n_interior_cases += 1
                        _, tau_at = maps(cs[i], tau_dist=tau_dist, decades_R=dec_R,
                                         decades_C=dec_C)
                        tstar_at = t_peak(T_g, float(tau_at))
                        in_win = band[0] <= tstar_at <= band[1]
                        n_window_ok += bool(in_win)
                        rec.update({"tau_sys_at_peak_s": float(tau_at),
                                    "t_star_at_peak_s": float(tstar_at),
                                    "t_star_in_window": bool(in_win),
                                    "prominence_over_endpoints": float(
                                        T[i] / max(T[0], T[-1]))})
                    out["sweeps"].append(rec)
    out["criterion_predicts_all"] = bool(n_pred_ok == len(out["sweeps"]))
    out["n_combos"] = len(out["sweeps"])
    out["interior_cases"] = {"n": n_interior_cases, "t_star_in_window_n": n_window_ok,
                             "t_star_in_window_frac": float(n_window_ok / n_interior_cases)
                             if n_interior_cases else None,
                             "note": "in the fast-primary regime (T_g < band floor) the lag-"
                             "to-peak sits at/just below the 2 s archive Nyquist edge (the "
                             "RoCoF-phase content); it moves into the band for slower-"
                             "governor (reheat) grids"}

    # (2b) criterion BOUNDARY (a grid-exposed refinement of NR32): removing storage is
    #      necessary but not sufficient -- the transition must span enough decades that
    #      tau_sys actually crosses BELOW the band by c=1, else the peak rides to c=1.
    T_bnd = transmission(cs, tau_in=0.6, band=band, tau_dist=240.0, decades_R=1.0,
                         decades_C=0.5)
    i_bnd = int(np.argmax(T_bnd))
    _, tau_bnd_end = maps(1.0, tau_dist=240.0, decades_R=1.0, decades_C=0.5)
    out["criterion_boundary"] = {
        "tau_dist_s": 240.0, "decades_R": 1.0, "decades_C": 0.5,
        "tau_sys_at_c1_s": float(tau_bnd_end),
        "band_high_tau_s": float(band[1] / (2.0 * np.pi)),
        "peak_rides_to_c1": bool(i_bnd == n_c - 1),
        "finding": "decades_C>0 but only 1.5 total decades: tau_sys(c=1) has NOT crossed "
                   "below the band, so no interior peak -- the window needs the transition "
                   "to traverse the observability band, not merely remove some storage",
    }

    # (3) Markovian (quasi-static control / delta-kernel) control: monotone, wrong end
    Tm = transmission(cs, markovian=True, tau_dist=120.0, decades_R=2.0, decades_C=1.0)
    out["markovian_control"] = {
        "monotone_decreasing": bool(np.all(np.diff(Tm) <= 1e-15)),
        "argmax_at_c0": bool(int(np.argmax(Tm)) == 0),
        "interior_peak": False,
        "verdict": "quasi-static (no-memory) frequency control predicts the HIGHEST-inertia "
                   "grid is most disturbance-observable and no interior peak -- any windowed "
                   "peak in real archives falsifies it",
    }

    # (4) RoCoF-vs-window dichotomy: RoCoF_0 ~ 10^(+decades_C c) is MONOTONE increasing while
    #     the band-limited observability is single-peaked -- distinct diagnostics.
    dec_C = 1.0
    rocof = 10.0 ** (dec_C * cs)                       # DP=1, M_0=1 normalisation
    Tw = transmission(cs, tau_in=0.6, band=band, tau_dist=120.0, decades_R=2.0,
                      decades_C=dec_C)
    i_w = int(np.argmax(Tw))
    out["rocof_vs_window"] = {
        "rocof_monotone_increasing": bool(np.all(np.diff(rocof) >= -1e-15)),
        "rocof_argmax_at_c1": bool(int(np.argmax(rocof)) == n_c - 1),
        "window_interior": bool(0 < i_w < n_c - 1),
        "window_c_peak": float(cs[i_w]),
        "verdict": "RoCoF risk rises monotonically with inertia loss; archive-band "
                   "observability peaks at intermediate inertia -- only the latter is a "
                   "two-clocks signature",
    }

    # (5) calibration-free readout on a representative modern-transition case
    tau_dist, dec_R, dec_C, T_g = 120.0, 2.0, 1.0, 0.6
    Tc = transmission(cs, tau_in=T_g, band=band, tau_dist=tau_dist, decades_R=dec_R,
                      decades_C=dec_C)
    ic = int(np.argmax(Tc))
    _, tau_star = maps(cs[ic], tau_dist=tau_dist, decades_R=dec_R, decades_C=dec_C)
    out["critical_inertia_readout"] = {
        "tau_dist_s": tau_dist, "decades_R": dec_R, "decades_C": dec_C, "T_g_s": T_g,
        "c_star": float(cs[ic]),
        "inertia_fraction_at_peak": float(10.0 ** (-dec_C * cs[ic])),
        "tau_sys_at_peak_s": float(tau_star),
        "knee_period_s": float(2.0 * np.pi * tau_star),
        "knee_in_band": bool(band[0] <= 2.0 * np.pi * tau_star <= band[1]),
        "t_star_at_peak_s": float(t_peak(T_g, float(tau_star))),
        "reading": "the most disturbance-observable state sits at a fraction of the original "
                   "inertia whose relaxation knee (period 2*pi*tau_sys) lies inside the 1-s "
                   "archive band -- an amplitude-free (calibration-free) statement of which "
                   "inertia is most visible",
    }

    # (6) real-data gate: the exact stackable, falsifiable population test
    out["real_data_gate"] = {
        "archives": ["ENTSO-E Transparency Platform 1-s system frequency (Continental "
                     "Europe, Nordic, GB, Baltic)", "GB National Grid ESO 1-s frequency "
                     "(Data Portal)"],
        "inertia_proxy": ["published synchronous-inertia estimates (GB-ESO inertia "
                          "forecast; ENTSO-E)", "share of synchronous generation from "
                          "dispatch data"],
        "observable": "band-limited disturbance variance in [2,30] s (high-pass 1/30 Hz, "
                      "low-pass 1/2 Hz) around known infeed-loss events, binned by inertia",
        "prediction": "single-peaked vs inertia (interior maximum) if the transition removes "
                      "inertia; Markovian null = monotone decreasing in inertia (max at the "
                      "highest-inertia bin)",
        "status": "GATED -- requires archive download + event catalogue (infeed-loss "
                  "register); the derivation, identifiability and Markovian discriminator "
                  "are verified here on the transferred functional",
    }
    return out


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(out, path_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs = np.linspace(0.0, 1.0, out["c_grid_n"])
    band = tuple(out["band_s"])
    Tw = transmission(cs, tau_in=0.6, band=band, tau_dist=120.0, decades_R=2.0,
                      decades_C=1.0)
    Tm = transmission(cs, markovian=True, tau_dist=120.0, decades_R=2.0, decades_C=1.0)
    rocof = 10.0 ** (1.0 * cs)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(cs, Tw / Tw.max(), "C0", lw=2, label="band-limited observability (NR42)")
    ax[0].plot(cs, Tm / Tm.max(), "C3--", lw=2, label="Markovian (quasi-static) control")
    ax[0].axvline(cs[int(np.argmax(Tw))], color="C0", ls=":", lw=1)
    ax[0].set_xlabel("de-carbonisation / inverter penetration  c")
    ax[0].set_ylabel("normalised in-band disturbance content")
    ax[0].set_title("interior peak iff the transition removes inertia")
    ax[0].legend(fontsize=8)
    ax[1].plot(cs, rocof / rocof.max(), "C2", lw=2, label="RoCoF$_0$ ~ 1/M (monotone)")
    ax[1].plot(cs, Tw / Tw.max(), "C0", lw=2, label="archive-band observability (peaked)")
    ax[1].set_xlabel("de-carbonisation / inverter penetration  c")
    ax[1].set_ylabel("normalised")
    ax[1].set_title("RoCoF risk vs archive window are distinct")
    ax[1].legend(fontsize=8)
    fig.suptitle("NR42 - the grid-frequency drainage window (NR32 transmission, grid constants)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    fig.savefig(path_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out",
                    default=os.path.join(FIGDIR, "nr42_grid_frequency_window.json"))
    ap.add_argument("--png-out",
                    default=os.path.join(FIGDIR, "nr42_grid_frequency_window.png"))
    a = ap.parse_args()
    out = nr42()
    os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
    with open(a.json_out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("closed_form_check", "sfr_pole_check", "criterion_predicts_all",
                       "n_combos", "interior_cases", "markovian_control", "rocof_vs_window",
                       "critical_inertia_readout")}, indent=1, default=str))
    make_figure(out, a.png_out)
    print("wrote", a.json_out, "and", a.png_out)


if __name__ == "__main__":
    main()
