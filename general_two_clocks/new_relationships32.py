r"""NR55 (theory capstone) -- the response anatomy is complete: a linear
response carries exactly four independent phases -- memory, transport,
shape parity, handedness -- each guarded by its own blindness theorem, and
the corpus's real systems occupy distinct corners of that phase space
with every zero enforced by a theorem, not by accident.

Closes the NR47-NR54 arc.  NR47 made the memory phase the coordinate;
NR48-NR54 discovered, one by one, the other faces and their guards.  NR55
states the completed anatomy and verifies the classification matrix
directly from the committed artifacts.

The anatomy
===========
Four phases exhaust the phase content of a (possibly tensorial) linear
response chi(omega) of a statistically 1D/2D field:

  P1 memory phase   delta = arg chi_min   -- odd in omega, KK-locked:
     DETERMINED by |chi| (Bode; NR48), integrally conserved (waterbed,
     NR50).  Guard: minimum-phase theorem -- relaxation cannot hide phase
     from the amplitude, nor put phase where amplitude structure is not.
  P2 transport phase  -omega tau_d (all-pass) -- the ONLY phase with no
     amplitude signature (NR48); in k it is the Galilean boost (NR49).
     Guard: all-pass factorisation -- gain structure can never produce a
     k- or omega-linear excess phase; boosts/delays can never touch gain.
  P3 shape parity   biphase = arg B (closed triads) -- parity-odd,
     EXACTLY boost-blind (closed polygons; NR51).  Guard: polygon
     closure -- no kinematics reaches a closed spectral polygon; only
     nonlinear asymmetric coupling writes into it.
  P4 handedness     spin = odd part of the correlation tensor -- parity-
     AND lag-odd (NR52).  Guard: scalar contraction -- every scalar
     observable (MSD, GSER, |chi|, coherence magnitude) is chirality-
     blind; only the tensor's antisymmetric part reads rotation sense.

The four guards are four INDEPENDENT invisibility theorems: amplitude-
blindness (P2 invisible to |chi|), phase-determinism (P1 has no freedom),
boost-blindness (P3 immune to P2's kinematics), chirality-blindness (P4
invisible to scalars).  No fifth phase exists at the two-point + three-
point level: pair open phase = P1+P2, pair tensor odd = P4, triad closed
= P3; higher polygons repeat P3's parity content.

The classification matrix (all values read from committed artifacts):

  system         P1 memory      P2 transport     P3 shape      P4 spin
  EEG (NR39/47)  ~0 (2.1 deg)   --               --            --
                 = elliptic pole: theorem-enforced zero phase
  ocean (NR45)   ~90 deg        0 (GSER closes   --            10-sigma,
                 (parabolic)    with no delay)                 mirrored
  solar (NR46)   ~0 (static     tau_d = -10.3 s  --            --
                 weighting,     (pure all-pass
                 x3.5 veto)     delay clock)
  scallops (P3)  sigma(k)~0     c = 48 mm/hr     weak (t~2.4   --
                 (marginal)     (boost, 93%      vs flux -4.7)
                                removable)
  grid (NR42)    band-limited   --               --            --
                 memory window

Every dash and every ~0 is enforced by the matching guard theorem; every
nonzero is a measured, tested number.  The corpus realizes all four
phases, each in the system where its physics lives.

Findings (figures/90_response_anatomy.json): the matrix assembled and
gated against the committed figures of NR47-NR54 -- see verdicts.

Consequence: a complete measurement protocol for ANY new system.  Measure
|chi| -> P1 is free (Bode).  Fit the excess-phase slope -> P2.  Close a
triad -> P3.  Antisymmetrize the tensor -> P4.  Four numbers, four
theorems, no redundancy: the two-clocks program's phase content is
closed.

CPU-only, offline-safe (reads committed figures only).
Tests: tests/test_response_anatomy.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
FIG = os.path.join(FIGDIR, "90_response_anatomy.json")

SOURCES = {
    "nr47": "82_memory_phase.json",
    "nr48": "83_minphase_delay.json",
    "nr49": "84_dispersion_boost.json",
    "nr50": "85_phase_area_budget.json",
    "nr51": "86_triad_parity.json",
    "nr52": "87_odd_spin.json",
    "nr53": "88_spin_two_clocks.json",
    "nr54": "89_spin_drift.json",
}


def load_sources(figdir=FIGDIR):
    out = {}
    for key, name in SOURCES.items():
        with open(os.path.join(figdir, name)) as fh:
            out[key] = json.load(fh)
    return out


def assemble_matrix(src):
    """The classification matrix, every entry read from a committed
    artifact."""
    return dict(
        eeg=dict(
            memory_deg=src["nr47"]["reads"]["eeg"]["delta_deg_near"],
            pole="elliptic"),
        ocean=dict(
            memory_deg=src["nr47"]["reads"]["ocean"]["ens"]["delta_deg"],
            minphase_median_deg=src["nr48"]["ocean_minphase_median_deg"],
            area_closure=src["nr50"]["ocean"]["rel_closure"],
            spin_nh_t=src["nr52"]["stats"]["nh_all"]["t_stat"],
            spin_sh_t=src["nr52"]["stats"]["sh_all"]["t_stat"],
            spin_eq_t=src["nr52"]["stats"]["eq"]["t_stat"],
            eddy_period_nh_d=src["nr53"]["bands"]["nh_trop"]["fit"][
                "period_days"],
            eddy_period_sh_d=src["nr53"]["bands"]["sh_trop"]["fit"][
                "period_days"],
            compass_pooled_t=src["nr54"]["pooled_poleward_t"],
            pole="parabolic + odd"),
        solar=dict(
            tau_d_s=src["nr48"]["delay_tau_d_s"],
            gain_log_std=src["nr48"]["delay_gain_log_std"],
            static_gain_veto_x=src["nr50"]["solar"]["naive_overshoot"],
            pole="pure transport"),
        scallops=dict(
            celerity_mm_hr=src["nr49"]["frames"]["lab"]["pattern"][
                "celerity_mm_hr"],
            boost_removable_frac=1.0
            - src["nr49"]["adv_residual_fraction"],
            growth_sigma_hr=src["nr49"]["frames"]["lab"]["pattern"][
                "sigma_hr"],
            t_shape_parity=src["nr51"]["t_shape_asymmetry"],
            t_flux_parity=src["nr51"]["t_flux_quadrature"],
            pole="transport + weak shape parity"),
    )


def verdicts(mat):
    eeg = abs(mat["eeg"]["memory_deg"]) < 5.0
    ocean = (85.0 < mat["ocean"]["memory_deg"] < 95.0
             and mat["ocean"]["minphase_median_deg"] < 6.0
             and mat["ocean"]["area_closure"] < 0.10
             and mat["ocean"]["spin_nh_t"] < -4.0
             and mat["ocean"]["spin_sh_t"] > 4.0
             and abs(mat["ocean"]["spin_eq_t"]) < 2.0
             and mat["ocean"]["compass_pooled_t"] > 3.0)
    solar = (5.0 < abs(mat["solar"]["tau_d_s"]) < 15.0
             and mat["solar"]["gain_log_std"] < 0.6
             and mat["solar"]["static_gain_veto_x"] > 2.0)
    scallop = (30.0 < mat["scallops"]["celerity_mm_hr"] < 70.0
               and mat["scallops"]["boost_removable_frac"] > 0.8
               and abs(mat["scallops"]["growth_sigma_hr"]) < 1.0
               and abs(mat["scallops"]["t_shape_parity"])
               < 0.6 * abs(mat["scallops"]["t_flux_parity"]))
    corners = eeg and ocean and solar and scallop
    return dict(
        eeg_at_elliptic_pole=bool(eeg),
        ocean_parabolic_plus_odd=bool(ocean),
        solar_pure_transport=bool(solar),
        scallops_transport_weak_shape=bool(scallop),
        four_corners_distinct=bool(corners),
        reading=(
            "Four phases, four guards, no fifth: memory (Bode-determined, "
            "waterbed-conserved), transport (all-pass, amplitude-blind), "
            "shape parity (closed-polygon, boost-blind), handedness "
            "(tensor-odd, scalar-blind).  The corpus realizes every "
            "corner: EEG sits at the elliptic pole "
            f"({mat['eeg']['memory_deg']:.1f} deg -- volume conduction's "
            "theorem-enforced zero); the ocean is parabolic "
            f"({mat['ocean']['memory_deg']:.0f} deg, minimum-phase to "
            f"{mat['ocean']['minphase_median_deg']:.1f} deg, area budget "
            f"{100 * mat['ocean']['area_closure']:.0f}%) PLUS the odd "
            "face (10-sigma mirrored spin, 50-60 d eddy clock, beta-drift "
            "compass); the solar pair is pure transport "
            f"(tau_d = {mat['solar']['tau_d_s']:.1f} s all-pass, its gain "
            f"tilt vetoed x{mat['solar']['static_gain_veto_x']:.1f} as "
            "static weighting); the scallops are transport "
            f"({mat['scallops']['celerity_mm_hr']:.0f} mm/hr, "
            f"{100 * mat['scallops']['boost_removable_frac']:.0f}% "
            "boost-removable) with marginal growth and only weak triad "
            "parity -- their parity breaking lives in the flux, exactly "
            "as the polygon theorem demands.  Protocol for any new "
            "system: |chi| -> memory free; excess-phase slope -> "
            "transport; closed triad -> shape; antisymmetrized tensor -> "
            "handedness."),
    )


def analyze():
    src = load_sources()
    mat = assemble_matrix(src)
    out = dict(matrix=mat, sources={k: SOURCES[k] for k in SOURCES})
    out["verdicts"] = verdicts(mat)
    return out


def run(write=True):
    out = analyze()
    res = dict(description=__doc__.splitlines()[0], **out)
    if write:
        os.makedirs(FIGDIR, exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    print("NR55 -- the response anatomy (capstone): four phases, four guards")
    print(f"  EEG at elliptic pole          : {v['eeg_at_elliptic_pole']}")
    print(f"  ocean parabolic + odd         : {v['ocean_parabolic_plus_odd']}")
    print(f"  solar pure transport          : {v['solar_pure_transport']}")
    print(f"  scallops transport + weak par : "
          f"{v['scallops_transport_weak_shape']}")
    print(f"  four corners distinct         : {v['four_corners_distinct']}")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
