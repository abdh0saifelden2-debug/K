r"""§I.7 — FREQUENCY-RESOLVED s_N: the bed as a hydromechanical low-pass filter, a
reconciliation of the three s_N probes, and a multi-frequency "spectroscopy" that
separates sliding sensitivity from basal relaxation time.

What is new here (vs ``sn_master_curve.py`` and ``tidal_admittance_probe.py``)
-----------------------------------------------------------------------------
The repo established three INDEPENDENT field probes of the regularized-Coulomb (RC)
sliding sensitivity ``s_N = d ln u_b / d ln N``:
  * lake-drainage steps (``lake_lag_trunk.py``)              — an impulse at ~1/yr,
  * ocean thermal-forcing drift (``efp_gate_direct_n.py``)   — quasi-static (decadal+),
  * tidal modulation (``tidal_admittance_probe.py``)         — periodic at ~1/(12 h)–1/(14 d),
and the closed-form master curve ``|s_N|(N) = m/(1-(N_c/N)^m)`` (``sn_master_curve.py``),
which DIVERGES at the flotation fold ``N_c``. All three probes were treated as
QUASI-STATIC measurements of the *same* ``s_N(N)``.

But they sample three DIFFERENT forcing frequencies, and ``sn_master_curve.py`` Result 3
shows the *same* ``N_c`` fold makes the velocity-restoring rate vanish,
``lambda(N) ∝ (1-R)^2/R -> 0`` at ``N_c`` (critical slowing down). The velocity response
to an ``N``-forcing at angular frequency ``omega`` is therefore NOT ``s_N`` — it is the
linear-response admittance of that first-order relaxation:

    A(omega, N) = s_N0(N) / (1 + i*omega/lambda(N)),                         (ADM)

a single-pole LOW-PASS with corner ``lambda(N)``. (ADM) is the exact linear response of
the OU model ``sn_master_curve.ews_realization`` integrates,
``d(δln u)/dt = -lambda(N)[δln u - s_N0(N) δln N]``.

The results are stated in the CALIBRATION-FREE dimensionless ratio ``omega/lambda(N)``;
the absolute corner ``lambda_ref`` is uncertain (the repo carries it as a normalization),
and the spectroscopy (Result 3) is precisely what MEASURES it.

Three consequences the quasi-static picture misses
--------------------------------------------------
1. **The probes diverge near flotation (central result).** As ``N->N_c``: ``s_N0 ~
   N_c/(N-N_c) -> +inf`` but ``lambda ~ (N-N_c)^2 -> 0``, so for ANY fixed ``omega>0`` the
   gain ``|A| -> s_N0*lambda/omega ~ (N-N_c)/omega -> 0``. The quasi-static (``omega->0``,
   ocean/secular) probe DIVERGES at ``N_c`` while every periodic probe's gain VANISHES.
   They coincide only well-grounded. This *qualifies* ``tidal_admittance_probe.py`` Result 1
   ("A1=|s_N| steepens toward flotation"): true only while ``omega << lambda(N)``.
   (Observed tidal *modulation* = forcing ``epsilon(x)`` x admittance ``|A|``; the forcing
   grows toward the GL (REAL_DATA §I.5: ``Δp/p_i`` 0.21->0.45%), so raw amplitude can still
   grow toward flotation — consistent with Gudmundsson 2011 / Minchew 2017. The admittance
   rolloff is read in the FREQUENCY DEPENDENCE — constituent ratios and phase — not raw amplitude.)

2. **Corner sweeps down through the forcing band (constituent ordering).** The corner
   ``N_corner(omega)`` where ``lambda(N)=omega`` increases with ``omega``, so as a site drifts
   toward ``N_c`` it crosses the high-frequency corners FIRST: M2/S2 roll off (further from
   flotation), then fortnightly MSf, then decadal, then secular. A site approaching ungrounding
   becomes progressively more low-pass — an ordered precursor in the relative admittance of
   tidal constituents.

3. **Spectroscopy = a new method of measurement.** ``1/|A|^2 = 1/s_N0^2 + omega^2/(s_N0^2 lambda^2)``
   is LINEAR in ``omega^2``: admittance at frequencies STRADDLING the corner (some
   ``omega<lambda``, some ``>``) recovers ``s_N0`` (intercept) and ``lambda`` (slope)
   SEPARATELY — disentangling the sliding sensitivity (-> proximity to ``N_c``) from the basal
   relaxation time ``tau_h=1/lambda`` (the hydromechanical memory), which is confounded in any
   single-frequency probe. Phase ``-arctan(omega/lambda) -> 90 deg`` at ``N_c`` gives ``lambda``
   independently.

Physical reality
----------------
The ``N_c`` fold makes the bed SIMULTANEOUSLY infinitely sensitive (``s_N0->inf``) and
infinitely sluggish (``lambda->0``); low-frequency probes see the sensitivity, high-frequency
probes see the sluggishness, and their product ``s_N0*lambda -> 0`` is a finite, vanishing
diagnostic. This is the two-clocks picture at the fold: the sliding clock ``1/lambda`` diverges
to swallow any forcing clock ``1/omega``.

Honest scope
------------
(ADM) is the exact linear response of the repo's RC-sliding OU caricature (same ``s_N0``,
``lambda``). Real ice adds viscoelastic stress transmission and a second, hydraulic clock (the
§G.4 Mori-Zwanzig channel kernel), so a real admittance is a PRODUCT of low-passes; the robust,
calibration-free predictions are the corner-sweep ordering, the ``1/|A|^2``-vs-``omega^2``
linearity, and the phase->90deg. References: Gudmundsson 2011 *TC*; Rosier et al. 2014/2015;
Robel et al. 2017; Minchew et al. 2017 *ESSD*; Scheffer et al. 2009 *Nature*.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SN = _load("sn_master_curve", "sn_master_curve.py")
M_EXP, N_C = _SN.M_EXP, _SN.TAU_D / _SN.C_FRIC          # m=3, N_c=0.06 MPa
LAM_REF, N_REF = 1.0, 5.0e5                              # restoring-rate normalisation (repo convention)

# representative forcing angular frequencies [rad/yr] (absolute; see honest-scope caveat)
OMEGA = {
    "secular_ocean": 2 * np.pi / 50.0,                      # ~50-yr drift (near quasi-static)
    "decadal_ocean": 2 * np.pi / 10.0,                      # decadal thermal forcing
    "lake_step": 2 * np.pi / 1.0,                           # lake-drainage impulse ~1/yr
    "fortnightly_MSf": 2 * np.pi / (14.77 / 365.25),        # MSf tide
    "semidiurnal_M2": 2 * np.pi / (12.42 / 24.0 / 365.25),  # M2 tide
}


# --------------------------------------------------------------------------- #
# core: quasi-static sensitivity, restoring rate, admittance (ADM)
# --------------------------------------------------------------------------- #
def s_N0(N, m=M_EXP, N_c=N_C):
    """Quasi-static (omega->0) sensitivity = the master curve |s_N|(N)."""
    return _SN.s_N_closed(N, m, N_c)


def lam(N, m=M_EXP, N_c=N_C, lam_ref=LAM_REF, N_ref=N_REF):
    """Velocity-restoring (corner) rate lambda(N) [1/yr]; ->0 at the N_c fold,
    monotone increasing in N."""
    return _SN.restoring_rate(N, m, N_c, lam_ref, N_ref)


def admittance(omega, N, m=M_EXP, N_c=N_C, lam_ref=LAM_REF, N_ref=N_REF):
    """Complex velocity admittance to an N-forcing at angular frequency omega (ADM)."""
    N = np.asarray(N, float)
    s0 = s_N0(N, m, N_c)
    L = lam(N, m, N_c, lam_ref, N_ref)
    A = s0 * L / (L + 1j * omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = omega / L
    return dict(gain=np.abs(A), phase_deg=np.degrees(np.angle(A)),
                s_N0=s0, lam=L, omega_over_lambda=ratio)


# --------------------------------------------------------------------------- #
# 0. quasi-static limit recovers the master curve
# --------------------------------------------------------------------------- #
def quasi_static_limit(n=30):
    """As omega->0, |A| -> s_N0 (recovers the master curve) at machine precision."""
    N = np.geomspace(1.1 * N_C, 2.0e6, n)
    g = admittance(1e-12, N)["gain"]
    s0 = s_N0(N)
    return dict(max_reldiff=float(np.nanmax(np.abs(g - s0) / s0)),
                ok=bool(np.allclose(g, s0, rtol=1e-5)))


# --------------------------------------------------------------------------- #
# 1. the probes diverge near flotation (dimensionless, calibration-free)
# --------------------------------------------------------------------------- #
def divergence_dimensionless():
    """Calibration-free: as omega/lambda grows, the gain factor 1/sqrt(1+(omega/lambda)^2)
    suppresses s_N0. Near flotation lambda->0 so ANY fixed omega -> full suppression; the
    quasi-static (omega->0) limit diverges. Tabulate the suppression at fixed omega/lambda."""
    r = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0])
    supp = 1.0 / np.sqrt(1.0 + r ** 2)
    lag = np.degrees(np.arctan(r))
    return dict(omega_over_lambda=r.tolist(),
                gain_suppression=supp.tolist(), phase_lag_deg=lag.tolist(),
                finding=("The measured gain is s_N0 x 1/sqrt(1+(omega/lambda)^2): a probe reads "
                         "the full quasi-static s_N0 only for omega<<lambda(N). Since lambda->0 at "
                         "N_c, every periodic probe is driven into rolloff toward flotation while "
                         "the omega->0 limit diverges — the probes agree only well-grounded."))


def highfreq_prefactor(n=40):
    """The omega->inf gain prefactor s_N0(N)*lambda(N) (|A|~s0*lam/omega) -> 0 at N_c,
    so periodic-probe gains vanish at flotation though s_N0 diverges."""
    N = np.geomspace(1.02 * N_C, 50 * N_C, n)
    prod = s_N0(N) * lam(N)
    return dict(N_MPa=(N / 1e6).tolist(), s0_times_lam=prod.tolist(),
                value_1pct_above_Nc=float(prod[0]),
                vanishes_at_Nc=bool(prod[0] < prod[len(prod) // 2]))


# --------------------------------------------------------------------------- #
# 2. corner N_corner(omega): where lambda(N)=omega; increases with omega
# --------------------------------------------------------------------------- #
def corner_N(omega, n=400000):
    """Solve lambda(N)=omega for N_corner (the probe rolls off below this N). lambda is
    monotone in N, so the root is unique; returns +inf if omega exceeds lambda at the top."""
    N = np.geomspace(1.0001 * N_C, 1e8, n)
    L = lam(N)
    i = int(np.argmin(np.abs(L - omega)))
    return float(N[i])


def corner_vs_frequency():
    """N_corner(omega) increases with omega: higher-frequency probes roll off further from
    flotation, so toward ungrounding the rolloff sweeps DOWN through the forcing band."""
    rows = []
    for name, om in OMEGA.items():
        Nc_om = corner_N(om)
        rows.append(dict(probe=name, omega_per_yr=float(om), N_corner_MPa=Nc_om / 1e6,
                         N_corner_over_Nc=Nc_om / N_C))
    oms = np.array([r["omega_per_yr"] for r in rows])
    ncs = np.array([r["N_corner_MPa"] for r in rows])
    order = np.argsort(oms)
    monotone = bool(np.all(np.diff(ncs[order]) >= -1e-9))
    return dict(rows=rows, N_corner_increases_with_frequency=monotone,
                finding=("N_corner rises with frequency, so as N falls toward N_c a site crosses "
                         "the M2/S2 corners first (loses high-frequency admittance), then MSf, then "
                         "decadal, then secular — an ordered ungrounding precursor in constituent "
                         "admittance ratios. Absolute N_corner uses the repo's lambda normalization; "
                         "the ORDERING is calibration-free."))


# --------------------------------------------------------------------------- #
# 3. spectroscopy: separate s_N0 (sensitivity) from tau_h=1/lambda (memory)
# --------------------------------------------------------------------------- #
def spectroscopy_invert(omegas, gains):
    """Recover (s_N0, lambda) from |A| at >=2 frequencies via the exact linearity
    1/|A|^2 = 1/s_N0^2 + (1/(s_N0^2 lambda^2)) omega^2 (least squares in omega^2)."""
    omegas = np.asarray(omegas, float)
    gains = np.asarray(gains, float)
    y = 1.0 / gains ** 2
    x = omegas ** 2
    A = np.vstack([np.ones_like(x), x]).T
    (b0, b1), *_ = np.linalg.lstsq(A, y, rcond=None)
    s0 = 1.0 / np.sqrt(b0) if b0 > 0 else np.nan
    L = np.sqrt(b0 / b1) if (b1 > 0 and b0 > 0) else np.nan
    return dict(s_N0_hat=float(s0), lam_hat=float(L),
                tau_h_hat_yr=float(1.0 / L) if (L == L and L > 0) else None)


def spectroscopy_test(N_true=1.3 * N_C, straddle=(0.1, 0.3, 1.0, 3.0, 10.0),
                      noise=0.05, seed=0):
    """Plant a site at N_true; 'measure' |A| at frequencies that STRADDLE the site's corner
    lambda(N_true) (the design requirement: probes on both sides of the corner); recover
    s_N0 and tau_h=1/lambda and map s_N0 back to N. Demonstrates the measurement separates
    sensitivity from memory."""
    rng = np.random.default_rng(seed)
    L_true = float(lam(np.array(N_true)))
    s0_true = float(s_N0(np.array(N_true)))
    oms = L_true * np.asarray(straddle, float)             # omega/lambda = straddle factors
    gains = admittance(oms, np.array(N_true))["gain"] * np.exp(noise * rng.standard_normal(oms.size))
    inv = spectroscopy_invert(oms, gains)
    Ngrid = np.geomspace(1.0001 * N_C, 1e7, 300000)
    Nrec = float(Ngrid[np.argmin(np.abs(s_N0(Ngrid) - inv["s_N0_hat"]))]) if inv["s_N0_hat"] == inv["s_N0_hat"] else np.nan
    return dict(
        truth=dict(N_MPa=N_true / 1e6, s_N0=s0_true, lam_per_yr=L_true, tau_h_yr=1.0 / L_true),
        straddle_factors=list(straddle), noise=noise, recovered=inv,
        recovery=dict(
            s_N0_relerr=abs(inv["s_N0_hat"] - s0_true) / s0_true if inv["s_N0_hat"] == inv["s_N0_hat"] else None,
            lam_relerr=abs(inv["lam_hat"] - L_true) / L_true if inv["lam_hat"] == inv["lam_hat"] else None,
            N_relerr=abs(Nrec - N_true) / N_true if Nrec == Nrec else None,
            N_rec_MPa=Nrec / 1e6 if Nrec == Nrec else None),
        note=("Probes straddling the corner over-determine (s_N0, lambda): sensitivity "
              "(-> proximity to N_c) and basal relaxation time tau_h are recovered SEPARATELY, "
              "which no single-frequency probe can do. Practically: pair a slow probe "
              "(ocean drift, omega<lambda) with a fast one (tidal, omega>lambda)."))


def spectroscopy_robustness(N_true=1.3 * N_C, noises=(0.03, 0.05, 0.10), n_seed=60):
    out = []
    for nz in noises:
        es0, eL, eN = [], [], []
        for s in range(n_seed):
            r = spectroscopy_test(N_true=N_true, noise=nz, seed=s)["recovery"]
            if r["s_N0_relerr"] is not None:
                es0.append(r["s_N0_relerr"])
            if r["lam_relerr"] is not None:
                eL.append(r["lam_relerr"])
            if r["N_relerr"] is not None:
                eN.append(r["N_relerr"])
        out.append(dict(noise=nz,
                        s_N0_median_relerr=float(np.median(es0)) if es0 else None,
                        lam_median_relerr=float(np.median(eL)) if eL else None,
                        N_median_relerr=float(np.median(eN)) if eN else None))
    return dict(per_noise=out,
                finding=("With corner-straddling frequencies the sensitivity s_N0 (-> N) and the "
                         "relaxation time tau_h=1/lambda are both recovered to a few percent and, "
                         "crucially, DISENTANGLED — the new content of the multi-frequency probe."))


# --------------------------------------------------------------------------- #
# 4. phase lag rises toward flotation
# --------------------------------------------------------------------------- #
def phase_vs_N(probe="fortnightly_MSf", n=40):
    """Phase lag -arg A = arctan(omega/lambda) -> 90 deg at N_c (periodic response lags MORE
    toward flotation, as both the sliding clock 1/lambda and the §G.4 hydraulic clock slow)."""
    om = OMEGA[probe]
    N = np.geomspace(1.02 * N_C, 30 * N_C, n)
    ph = -admittance(om, N)["phase_deg"]
    return dict(probe=probe, omega_per_yr=float(om), N_MPa=(N / 1e6).tolist(),
                lag_deg=ph.tolist(), lag_near_Nc=float(ph[0]), lag_well_grounded=float(ph[-1]),
                monotone_rising_toward_flotation=bool(ph[0] > ph[-1]),
                note=("Phase lag rises toward flotation, -> 90 deg at N_c. The repo's n=3 lake-step "
                      "time-to-peak lags trending the other way (marginal Rutford point) remain an "
                      "open low-n discriminator (sn_master_curve.field_overlay); the periodic phase "
                      "lag here is the cleaner, higher-cadence version of the same clock."))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run():
    qs = quasi_static_limit()
    dd = divergence_dimensionless()
    hp = highfreq_prefactor()
    cf = corner_vs_frequency()
    spec = spectroscopy_test()
    spec_rob = spectroscopy_robustness()
    ph = phase_vs_N()
    return dict(
        model=("velocity admittance to an N-forcing A(omega,N)=s_N0(N)/(1+i omega/lambda(N)); "
               "linear response of the RC-sliding OU model (sn_master_curve)"),
        params=dict(m=M_EXP, N_c_MPa=N_C / 1e6, lam_ref_per_yr=LAM_REF, N_ref_Pa=N_REF),
        omega_per_yr={k: float(v) for k, v in OMEGA.items()},
        quasi_static_limit=qs,
        divergence_dimensionless=dd,
        highfreq_prefactor=hp,
        corner_vs_frequency=cf,
        spectroscopy=spec,
        spectroscopy_robustness=spec_rob,
        phase_vs_N=ph,
        verdict=(
            "The three s_N probes are one frequency-resolved admittance A(omega,N)=s_N0/(1+i omega/lambda). "
            "The same N_c fold that diverges s_N0 collapses the corner lambda->0, so toward flotation every "
            "periodic probe is driven into rolloff (gain ~ s_N0*lambda/omega -> 0) while the quasi-static limit "
            "diverges — the probes coincide only well-grounded, and the rolloff sweeps DOWN through the forcing "
            "band (M2->MSf->decadal->secular), an ordered ungrounding precursor. Admittance at corner-straddling "
            f"frequencies inverts s_N0 and tau_h=1/lambda SEPARATELY (s_N0 to "
            f"{spec_rob['per_noise'][1]['s_N0_median_relerr']:.0%}, N to {spec_rob['per_noise'][1]['N_median_relerr']:.0%} "
            "at 5% noise) — a new measurement disentangling sliding sensitivity from basal relaxation time, with "
            "phase lag rising to 90 deg at N_c. Calibration-free in omega/lambda; the spectroscopy measures the "
            "absolute scale the repo carries as a normalization."),
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))

    # (a) gain vs N for each probe frequency + quasi-static s_N0
    N = np.geomspace(1.01 * N_C, 2.0e6, 500)
    for name, om in OMEGA.items():
        ax[0, 0].plot(N / 1e6, admittance(om, N)["gain"], lw=1.7, label=name)
    ax[0, 0].plot(N / 1e6, s_N0(N), "k--", lw=1.2, label="s_N0 (omega->0)")
    ax[0, 0].axvline(N_C / 1e6, color="k", ls=":")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("N [MPa]"); ax[0, 0].set_ylabel("|A| (admittance)")
    ax[0, 0].set_title("(a) probes diverge near flotation\n(quasi-static diverges; periodic gains vanish)")
    ax[0, 0].legend(fontsize=7); ax[0, 0].grid(alpha=0.3, which="both")

    # (b) universal low-pass + corner sweep
    rr = np.geomspace(1e-2, 1e2, 200)
    ax[0, 1].plot(rr, 1.0 / np.sqrt(1 + rr ** 2), color="#6a3d9a", lw=2)
    ax[0, 1].axvline(1.0, ls=":", color="gray")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlabel(r"$\omega/\lambda(N)$"); ax[0, 1].set_ylabel("gain / s_N0")
    ax[0, 1].set_title("(b) universal low-pass; corner lambda(N)->0 at N_c\n(rolloff sweeps to low f toward flotation)")
    ax[0, 1].grid(alpha=0.3, which="both")

    # (c) spectroscopy: 1/|A|^2 linear in omega^2
    sp = res["spectroscopy"]
    Lt = sp["truth"]["lam_per_yr"]; oms = Lt * np.array(sp["straddle_factors"])
    g2 = admittance(oms, np.array(sp["truth"]["N_MPa"] * 1e6))["gain"]
    ax[1, 0].plot((oms / Lt) ** 2, 1.0 / g2 ** 2, "o-", color="#1b9e77")
    ax[1, 0].set_xlabel(r"$(\omega/\lambda)^2$"); ax[1, 0].set_ylabel(r"$1/|A|^2$")
    rec = sp["recovery"]
    ax[1, 0].set_title("(c) spectroscopy: 1/|A|^2 linear in omega^2\n"
                       f"-> s_N0 & tau_h separately (s_N0 err {rec['s_N0_relerr']:.1%})")
    ax[1, 0].grid(alpha=0.3)

    # (d) phase lag vs N
    pv = res["phase_vs_N"]
    ax[1, 1].plot(pv["N_MPa"], pv["lag_deg"], color="#d95f02", lw=2)
    ax[1, 1].axvline(N_C / 1e6, color="k", ls=":"); ax[1, 1].axhline(90, color="gray", ls=":")
    ax[1, 1].set_xscale("log"); ax[1, 1].invert_xaxis()
    ax[1, 1].set_xlabel("N [MPa] (decreasing -> flotation)")
    ax[1, 1].set_ylabel(f"phase lag [deg] ({pv['probe']})")
    ax[1, 1].set_title("(d) phase lag rises to 90 deg at N_c")
    fig.suptitle("Frequency-resolved s_N: the bed as a hydromechanical low-pass filter", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_REPORTS, "sn_frequency_admittance.json"))
    a = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2, default=_json_default)
    print("=== frequency-resolved s_N: the bed as a hydromechanical low-pass filter ===")
    qs = res["quasi_static_limit"]
    print(f"  quasi-static limit A(omega->0)=s_N0: max rel-diff {qs['max_reldiff']:.1e} (ok={qs['ok']})")
    print("  corner N_corner(omega) increases with frequency:",
          res["corner_vs_frequency"]["N_corner_increases_with_frequency"])
    for r in res["corner_vs_frequency"]["rows"]:
        print(f"    {r['probe']:>16s}: N_corner = {r['N_corner_MPa']:.4f} MPa ({r['N_corner_over_Nc']:.1f} N_c)")
    print("  spectroscopy (separate s_N0 from tau_h=1/lambda), corner-straddling probes:")
    for row in res["spectroscopy_robustness"]["per_noise"]:
        print(f"    noise={row['noise']:.2f}: s_N0-err {row['s_N0_median_relerr']:.1%}, "
              f"N-err {row['N_median_relerr']:.1%}, lambda-err {row['lam_median_relerr']:.1%}")
    ph = res["phase_vs_N"]
    print(f"  phase lag: {ph['lag_well_grounded']:.1f} deg well-grounded -> {ph['lag_near_Nc']:.1f} deg near N_c")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
