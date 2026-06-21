"""Regression test for RESULT 14 (scallop_amplitude_harmonics.py): the scallop
amplitude law, corrected by a frozen-interface wall-flux harmonic decomposition.

A short CPU run must reproduce the three structural verdicts that overturn the
FUTURE_WORK Sec.G.2 ansatz ``rho L a_dot = alpha a^{1/2} - beta a``:

  1. the pure-conduction smoothing ``beta`` is ~K-independent -> the ``beta ~ K^2``
     (lambda^-2) curvature ansatz is FALSIFIED (also not Mullins-Sekerka |k|);
  2. the in-phase flow-excess flux is smoothing (negative) at every driven point
     -> there is NO autonomous ``+alpha a^{1/2}`` growth term; and
  3. the genuine flow channel is a QUADRATURE migration term that is ~0 at U=0
     and grows with drive (the damped, downstream-migrating mode).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scallop_amplitude_harmonics as h


def test_amplitude_law_harmonic_decomposition():
    out = h.run(nx=96, ny=96, spinup=800, measure=300,
                nwaves=(6, 8, 10, 12, 16, 20), flow_nwaves=(12,),
                U_list=(0.0, 2.0), seeds=(0,))

    # (1) conduction beta/a is far from the K^2 curvature ansatz (exponent ~0,
    # certainly well below the predicted +2): the ansatz is falsified.
    p = out["conduction_beta_K_exponent"]
    assert np.isfinite(p) and abs(p) < 1.0
    assert out["verdict"]["curvature_K2_ansatz_falsified"]

    fr = out["flow_excess"]["12"]
    by_U = {r["U"]: r for r in fr}

    # (2) no autonomous growth term: the in-phase (shape) flow excess is smoothing
    # (negative) under drive -- flow never reinforces the corrugation.
    assert by_U[2.0]["inphase_mean"] < 0.0
    assert out["verdict"]["no_autonomous_growth_term"]

    # (3) the flow channel is migration (quadrature): ~0 at U=0, and larger under
    # drive -- a damped, downstream-migrating mode rather than growth-saturation.
    assert abs(by_U[0.0]["migration_mean"]) < 1e-5
    assert abs(by_U[2.0]["migration_mean"]) > abs(by_U[0.0]["migration_mean"])
    assert out["verdict"]["migration_zero_at_U0"]
    assert out["verdict"]["migration_grows_with_drive"]


def test_amplitude_generalization_holds_beyond_swept_amplitude():
    """RESULT 22: the two structural verdicts that overturn the Sec.G.2 ansatz
    must not be artefacts of the single swept amplitude ``a0/lambda = 0.20``.
    Re-measuring at two amplitudes (two seeds) the pure-conduction ``beta``
    stays ~K-independent (nowhere near the ``K^2`` curvature ansatz) and the
    driven in-phase flow excess stays smoothing (negative) -- no autonomous
    ``+alpha a^{1/2}`` growth term at either amplitude.  (Low amplitudes
    ``<0.20`` are noise-limited at this short spinup and are covered by the full
    ``figures/56`` run; here we use the signal-rich ``0.20`` and ``0.30``.)"""
    g = h.amplitude_generalization_scan(
        afracs=(0.20, 0.30), nx=96, ny=96, spinup=800, measure=300,
        cond_nwaves=(6, 12, 20), flow_nw=12, U_list=(2.0,), seeds=(0, 1))

    assert len(g["rows"]) == 2
    for r in g["rows"]:
        # K-independent: nowhere near the +2 curvature ansatz (nor +1 M-S)
        assert np.isfinite(r["K_exponent"]) and abs(r["K_exponent"]) < 1.0
        # smoothing-only: the strongest (most positive) driven in-phase flow
        # excess is still negative -> flow never reinforces the corrugation
        assert r["inphase_max_driven"] < 0.0

    # headline verdicts (robust at every amplitude): no autonomous growth term,
    # and the K^2 curvature ansatz stays falsified.
    assert g["verdict"]["smoothing_only_all_amplitudes"]
    assert g["verdict"]["curvature_ansatz_falsified_all_amplitudes"]
    # finer descriptors hold in this signal-rich (>=0.20) regime: strict
    # K-independence and a flat beta/a (they degrade only at shallow amplitude).
    assert g["verdict"]["K_independent_all_amplitudes"]
    assert g["verdict"]["beta_per_a_amplitude_flat"]


def test_drive_window_generalization_holds_beyond_swept_drive():
    """RESULT 23: the two *flow* verdicts must not be artefacts of the swept
    drive window ``U in [1.5, 3.0]`` (the other half of Caveat D).  Pushing the
    mean drive to ``U = 6`` (two seeds), the driven in-phase flow excess stays
    smoothing (negative) at every drive -- strong-drive lee separation opens no
    autonomous ``+alpha a^{1/2}`` growth channel -- and the quadrature migration
    keeps its **sub-kinematic** friction-velocity ``U^{0.5-0.8}`` scaling (never
    ``U^1``) while vanishing (to noise) at ``U = 0`` (parity control).
    Monotonicity is *not* asserted: at high fidelity the migration saturates /
    rolls over near ``U ~ 4.5-6`` (finer descriptor), so the headline verdicts
    are the smoothing-only, sub-kinematic scaling, and parity facts."""
    d = h.drive_window_scan(
        U_list=(0.0, 3.0, 6.0), nx=96, ny=96, spinup=800, measure=300,
        flow_nw=12, afrac=0.20, seeds=(0, 1))

    # every driven point is smoothing (most positive driven in-phase < 0)
    assert d["inphase_max_driven"] < 0.0
    assert all(r["smoothing"] for r in d["rows"])
    # migration keeps the sub-kinematic friction-velocity (~U^0.5-0.8) scaling
    # out to U=6 -- well below the kinematic U^1 alternative
    assert np.isfinite(d["migration_U_exponent"])
    assert 0.4 <= d["migration_U_exponent"] <= 1.0

    # headline verdicts (robust): no growth channel, sub-kinematic migration,
    # vanishing parity control. (migration_monotone_in_U is a reported finer
    # descriptor only -- it degrades at the strongest drive, see figures/56.)
    assert d["verdict"]["smoothing_only_all_drives"]
    assert d["verdict"]["migration_sqrt_law_holds"]
    assert d["verdict"]["parity_control_at_U0"]


def test_physical_bridge_scaling_and_sign():
    """The Stefan-condition bridge (to_physical) must give positive, finite SI
    rates with the derived scalings: amplitude e-fold time ``tau ~ lam^2 / dT``
    and migration speed ``c_mig ~ lam^-1 * dT``.  (No solver run needed -- the
    bridge is a closed-form map from the measured dimensionless coefficients.)"""
    out = {
        "conduction_beta": [
            {"n_waves": 8, "lam": 3.1416, "K": 2.0, "a": 0.628,
             "beta_per_a": 8.0e-5},
        ],
        "flow_excess": {
            "12": [
                {"n_waves": 12, "lam": 2.0944, "K": 3.0, "a": 0.419, "U": 3.0,
                 "migration_mean": -1.1e-4},
            ],
        },
    }
    p1 = h.to_physical(out, dT=0.1, lam_phys=0.05)
    tau1 = p1["conduction_efold"][0]["tau_amp_yr"]
    cmig1 = p1["migration"]["12"]["c_mig_m_per_yr"]
    # positive, finite, physically sensible magnitudes (months-years; cm/yr)
    assert np.isfinite(tau1) and 0.01 < tau1 < 100.0
    assert np.isfinite(cmig1) and cmig1 > 0.0

    # tau ~ lam^2 : doubling lambda -> ~4x e-fold time
    tau2 = h.to_physical(out, dT=0.1, lam_phys=0.10)["conduction_efold"][0]["tau_amp_yr"]
    assert abs(tau2 / tau1 - 4.0) < 0.1
    # tau ~ 1/dT : tripling dT -> ~1/3 e-fold time
    tau3 = h.to_physical(out, dT=0.3, lam_phys=0.05)["conduction_efold"][0]["tau_amp_yr"]
    assert abs(tau3 / tau1 - 1.0 / 3.0) < 0.02
    # migration ~ 1/lam : doubling lambda -> ~1/2 the speed
    cmig2 = h.to_physical(out, dT=0.1, lam_phys=0.10)["migration"]["12"]["c_mig_m_per_yr"]
    assert abs(cmig2 / cmig1 - 0.5) < 0.02
 
 
def test_anchored_subglacial_curl_wavelength():
    """The literature-anchored single point: the wavelength is *selected* by the
    Curl law ``lam = Re_* nu / u_*`` (not free), and the same nondim mode is used
    for both beta and migration.  Check the wavelength selection, physically
    sensible magnitudes (~yr e-fold, ~cm/yr migration) and the dT scalings."""
    out = {
        "conduction_beta": [
            {"n_waves": 12, "lam": 2.0944, "K": 3.0, "a": 0.419,
             "beta_per_a": 6.3e-5},
        ],
        "flow_excess": {
            "12": [
                {"n_waves": 12, "lam": 2.0944, "K": 3.0, "a": 0.419, "U": 0.0,
                 "migration_mean": 0.0},
                {"n_waves": 12, "lam": 2.0944, "K": 3.0, "a": 0.419, "U": 3.0,
                 "migration_mean": -1.14e-4},
            ],
        },
    }
    a = h.anchored_subglacial(out, u_star=0.05, dT=0.1, nw=12)
    # Curl selection: lam = Re_* nu / u_* = 2200 * 1.79e-6 / 0.05 ~ 0.0788 m
    assert abs(a["lam_phys_m"] - 0.0788) < 1e-3
    assert a["U_drive"] == 3.0                       # strongest-drive migration used
    # physically sensible: e-fold of order years, migration of order cm/yr
    assert 0.5 < a["tau_amp_yr"] < 20.0
    assert 0.3 < a["c_mig_m_per_yr"] * 100 < 20.0
    # tau ~ 1/dT, c_mig ~ dT
    a3 = h.anchored_subglacial(out, u_star=0.05, dT=0.3, nw=12)
    assert abs(a3["tau_amp_yr"] / a["tau_amp_yr"] - 1.0 / 3.0) < 0.02
    assert abs(a3["c_mig_m_per_yr"] / a["c_mig_m_per_yr"] - 3.0) < 0.02


def test_constant_free_ratio_cancels_ice_constants():
    """``I = tau*c_mig/lam`` must be independent of dT, k_th and rho_L (they
    cancel exactly).  Check the closed form, and that rebuilding the same
    combination from ``to_physical`` at two very different ``(dT, lam_phys)``
    operating points returns the identical value."""
    out = {
        "conduction_beta": [
            {"n_waves": 12, "lam": 2.0944, "K": 3.0, "a": 0.419,
             "beta_per_a": 6.3e-5},
        ],
        "flow_excess": {
            "12": [
                {"n_waves": 12, "lam": 2.0944, "K": 3.0, "a": 0.419, "U": 0.0,
                 "migration_mean": 0.0},
                {"n_waves": 12, "lam": 2.0944, "K": 3.0, "a": 0.419, "U": 3.0,
                 "migration_mean": -1.14e-4},
            ],
        },
    }
    cf = h.constant_free_ratio(out)
    ratio = cf["ratios"][0]["tau_cmig_over_lam"]
    # closed form: (-E_cos) / (lam_nd * (beta/a) * a * K)
    expect = 1.14e-4 / (2.0944 * 6.3e-5 * 0.419 * 3.0)
    assert abs(ratio - expect) < 1e-9

    # ice-constant + dT independence: tau*c_mig/lam from to_physical must equal
    # `ratio` for any (dT, lam_phys) -- every physical constant cancels.
    for dT, lam in ((0.05, 0.03), (0.4, 0.20)):
        phys = h.to_physical(out, dT=dT, lam_phys=lam)
        ce = next(c for c in phys["conduction_efold"] if c["n_waves"] == 12)
        cmig = phys["migration"]["12"]["c_mig_m_per_s"]
        rebuilt = ce["tau_amp_s"] * cmig / lam
        assert abs(rebuilt - ratio) < 1e-6 * abs(ratio)
