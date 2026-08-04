r"""§G.4 / §H.1.6 / §F — the s_N(N) MASTER CURVE, an inversion method, and a
critical-slowing-down early-warning signal for ungrounding.

What is new here (vs ``type_iii_regime.py`` and ``efp_probe_theory.py``)
-----------------------------------------------------------------------
The repo already established (PR #1-#4) that two independent field probes — lake
drainage steps (``lake_lag_trunk.py``) and ocean thermal-forcing gating
(``efp_gate_direct_n.py``) — both measure the *same* regularized-Coulomb (RC)
sliding-law sensitivity ``s_N = d ln u_b / d ln N``, and that ``|s_N|`` grows
toward flotation. That synthesis was **qualitative** (the *sign* and the *shape*).
``type_iii_regime.py`` evaluates ``s_N`` only *numerically*.

This module pushes the same physics three concrete steps further, all analytic +
self-contained (no GPU, no downloads; it optionally reads the committed field
JSONs):

1. **Closed-form master curve.** Solving the RC law ``tau_b = C N (u/(u+u0))^(1/m)``
   at fixed driving stress ``tau_b = tau_d`` gives, exactly,

       |s_N|(N) = m / ( 1 - (N_c/N)^m ),     N_c = tau_d / C,            (MASTER)

   a single closed curve with TWO parameters (``m``, ``N_c``). Far from flotation
   ``|s_N| -> m``; near it ``|s_N| ~ N_c/(N - N_c)`` (a simple pole at N_c). This
   *derives* the near-flotation weakening that Joughin, Smith & Schoof (2019) put
   in by hand (a linear ``h_af < h_T`` ramp with a fixed ``u0 = 300 m/yr``,
   "no reliable knowledge of basal water pressure"). Verified to <1e-4 against the
   repo's numeric ``type_iii_regime.s_N``.

2. **An inversion method (a new "method of measurement").** Because (MASTER) has
   only (``m``, ``N_c``), a *population* of drainage steps with known ``N`` and
   measured amplitudes ``du/u = |s_N(N)| f`` over-determines the sliding law: one
   can **recover ``m`` and ``N_c`` (the flotation/Type-III threshold) directly from
   field data** instead of tuning them. We verify the inversion plant->recover on
   synthetic populations and report what the current (uncalibrated, n~3) field set
   can and cannot yet constrain.

3. **A critical-slowing-down early-warning signal (a new physical reality).** The
   same pole has a dynamical consequence the static picture misses. Near flotation
   the basal drag saturates onto its Coulomb plateau, so the velocity-restoring
   *stiffness* ``d tau_b/du ∝ (1-R)^2/R`` (``R=(N_c/N)^m``) -> 0 as ``N -> N_c``:
   the grounded steady branch ends in a fold and the restoring rate vanishes —
   textbook **critical slowing down** (Scheffer et al. 2009; Dakos et al. 2008).
   So an ice stream drifting toward ungrounding should show, *before* it goes
   afloat, a **rising variance and lag-1 autocorrelation of its speed** — an
   early-warning signal for marine grounding-line retreat / MISI onset. This is a
   *different* mechanism and observable from the Greenland surface-melt CSD of
   Boers & Rypdal (2021): the precursor here is in *ice velocity*, driven by the
   *basal-sliding* ``s_N`` divergence, and the threshold is *ungrounding*.

Honest scope
------------
(MASTER) and the inversion are exact for the fixed-``u0`` RC law (the repo /
Joughin-2019 practical form); with the cavitation form ``u0 = N^m Λ_o`` the
well-grounded asymptote changes (``|s_N| -> 0`` instead of ``m``) but the ``N_c``
pole is unchanged — flagged in ``WELL_GROUNDED_NOTE``. The CSD signal is a robust
qualitative prediction (variance/AC1 rise); the *rate* prefactor and how close to
``N_c`` it stays resolvable are sweep-rate dependent (shown). The field overlay
inherits the project's standing caveats: the drainage ``rel=m/H`` is a centroid
proxy for ``N`` (not the trunk ``N`` where the surge is measured) and the per-event
fractional drop ``f`` is uncalibrated, so the field set constrains the *sign/shape*,
not yet a calibrated ``(m, N_c)``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_T3 = _load("type_iii_regime", "type_iii_regime.py")

# RC law defaults (identical to type_iii_regime.regime_sweep, so N_c = 0.06 MPa)
TAU_D = 3.0e4      # driving stress [Pa]
C_FRIC = 0.5       # Coulomb coefficient (tan of till friction angle)
U0 = 100.0         # RC regularization velocity [m/yr]
M_EXP = 3.0        # Weertman exponent
RHO_I, G = 917.0, 9.81
SEC_PER_YR = 365.25 * 86400.0
WELL_GROUNDED_NOTE = (
    "Fixed-u0 RC law -> |s_N|->m far from flotation. With Joughin-2019 cavitation "
    "u0=N^m*Lambda_o, s_N=-m R/(1-R) -> 0 far from flotation but still diverges at "
    "N_c=tau_d/C; the master-curve N_c pole is convention-independent.")


# --------------------------------------------------------------------------- #
# 1. closed-form master curve
# --------------------------------------------------------------------------- #
def s_N_closed(N, m=M_EXP, N_c=TAU_D / C_FRIC):
    """|s_N|(N) = m / (1 - (N_c/N)^m); +inf for N <= N_c (no grounded steady soln)."""
    N = np.asarray(N, float)
    R = (N_c / N) ** m
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(N > N_c, m / (1.0 - R), np.inf)
    return out


def amp_law(N, m=M_EXP, N_c=TAU_D / C_FRIC, f=0.05):
    """Predicted surge amplitude du/u = |s_N(N)| * f for a fractional drop f=|dN/N|."""
    return s_N_closed(N, m, N_c) * f


def pole_approx(N, N_c=TAU_D / C_FRIC):
    """Leading near-flotation behaviour |s_N| ~ N_c/(N-N_c)."""
    N = np.asarray(N, float)
    return N_c / (N - N_c)


def verify_closed_form(m=M_EXP, N_c=TAU_D / C_FRIC, n=40):
    """Closed form vs the repo's numeric type_iii_regime.s_N (max rel-diff)."""
    N = np.geomspace(1.05 * N_c, 2.0e6, n)
    closed = s_N_closed(N, m, N_c)
    numeric = np.abs(_T3.s_N(N, TAU_D, C_FRIC, U0, m))
    rel = np.abs(closed - numeric) / closed
    # pole check in the asymptotic near-flotation regime (leading order)
    pole = pole_approx(N[N < 1.1 * N_c], N_c)
    pole_true = closed[N < 1.1 * N_c]
    pole_rel = float(np.nanmax(np.abs(pole - pole_true) / pole_true)) if pole.size else None
    return dict(max_reldiff_vs_numeric=float(np.nanmax(rel)),
                s_N_wellgrounded=float(closed[-1]),
                near_pole_max_reldiff=pole_rel)


# --------------------------------------------------------------------------- #
# 2. inversion method: recover (m, N_c) from a population of drainage steps
# --------------------------------------------------------------------------- #
def _fit_core(N_i, amp_i):
    """Bounded least-squares fit of (m, N_c, f) to amp = f*m/(1-(N_c/N)^m). No bootstrap."""
    from scipy.optimize import least_squares
    N_i = np.asarray(N_i, float)
    amp_i = np.asarray(amp_i, float)
    Nmin = float(N_i.min())

    def resid(p, Nv, av):
        m, N_c, f = p
        R = (N_c / Nv) ** m
        return np.log(np.clip(f * m / (1.0 - R), 1e-12, None)) - np.log(av)

    lb = [1.0, 1e2, 1e-4]
    ub = [12.0, 0.999 * Nmin, 10.0]
    x0 = [3.0, min(0.5 * Nmin, 0.5 * ub[1]), float(np.median(amp_i)) / 3.0]
    x0 = [min(max(x0[i], lb[i]), ub[i]) for i in range(3)]
    sol = least_squares(resid, x0, args=(N_i, amp_i), bounds=(lb, ub), max_nfev=4000)
    return sol.x, float(sol.cost)


def invert_population(N_i, amp_i):
    """Recover (m, N_c, f) from field points (N_i [Pa], amp_i = |s_N(N_i)| f).

    Fits log(amp)=log f+log m-log(1-(N_c/N)^m) by bounded least squares + a 1-sigma
    bootstrap. Needs >=3 points spanning N. N_c (the threshold) is well-conditioned;
    m is degenerate with f far from flotation (see inversion_robustness)."""
    N_i = np.asarray(N_i, float)
    amp_i = np.asarray(amp_i, float)
    ok = np.isfinite(N_i) & np.isfinite(amp_i) & (amp_i > 0) & (N_i > 0)
    N_i, amp_i = N_i[ok], amp_i[ok]
    if N_i.size < 3:
        return dict(ok=False, note="need >=3 finite, spanning points", n=int(N_i.size))
    (m_hat, Nc_hat, f_hat), cost = _fit_core(N_i, amp_i)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(120):
        idx = rng.integers(0, N_i.size, N_i.size)
        if np.unique(N_i[idx]).size < 3:
            continue
        try:
            boots.append(_fit_core(N_i[idx], amp_i[idx])[0])
        except Exception:
            pass
    boots = np.array(boots) if boots else np.empty((0, 3))
    std = (np.nanstd(boots, axis=0).tolist() if boots.size else [None, None, None])
    return dict(ok=True, n=int(N_i.size), m_hat=float(m_hat),
                N_c_hat_Pa=float(Nc_hat), N_c_hat_MPa=float(Nc_hat / 1e6),
                f_hat=float(f_hat), cost=cost,
                m_std=std[0], N_c_std_MPa=(None if std[1] is None else std[1] / 1e6))


def inversion_synthetic_test(m_true=3.0, N_c_true=6.0e4, f_true=0.05,
                             noise=0.12, n_pts=14, seed=0):
    """Plant (m, N_c, f); generate a noisy, near-N_c-spanning drainage population; recover."""
    rng = np.random.default_rng(seed)
    N_i = np.geomspace(1.05 * N_c_true, 30 * N_c_true, n_pts)
    amp = amp_law(N_i, m_true, N_c_true, f_true) * np.exp(noise * rng.standard_normal(n_pts))
    fit = invert_population(N_i, amp)
    fit["truth"] = dict(m=m_true, N_c_MPa=N_c_true / 1e6, f=f_true, noise=noise, n=n_pts)
    if fit.get("ok"):
        fit["recovery"] = dict(
            m_relerr=abs(fit["m_hat"] - m_true) / m_true,
            N_c_relerr=abs(fit["N_c_hat_Pa"] - N_c_true) / N_c_true,
            f_relerr=abs(fit["f_hat"] - f_true) / f_true)
    return fit


def inversion_robustness(m_true=3.0, N_c_true=6.0e4, f_true=0.05, n_pts=14,
                         noises=(0.05, 0.10, 0.20), n_seed=25):
    """Which parameters can a drainage-step population constrain? Median recovery
    error of N_c (robust threshold) vs m (degenerate far from flotation) over seeds."""
    out = []
    for noise in noises:
        em, enc = [], []
        for seed in range(n_seed):
            rng = np.random.default_rng(1000 + seed)
            N_i = np.geomspace(1.05 * N_c_true, 30 * N_c_true, n_pts)
            amp = amp_law(N_i, m_true, N_c_true, f_true) * np.exp(
                noise * rng.standard_normal(n_pts))
            (mh, nch, fh), _ = _fit_core(N_i, amp)
            em.append(abs(mh - m_true) / m_true)
            enc.append(abs(nch - N_c_true) / N_c_true)
        out.append(dict(noise=noise, N_c_median_relerr=float(np.median(enc)),
                        m_median_relerr=float(np.median(em))))
    return dict(per_noise=out,
                finding=("N_c (flotation/Type-III threshold, the MISI-relevant quantity) is "
                         "recovered to a few percent and is robust to noise; m is only weakly "
                         "constrained by amplitude alone (degenerate with f where amp~=f*m far "
                         "from flotation), so pinning m needs near-N_c sampling or a co-located "
                         "hydrology dN."))


# --------------------------------------------------------------------------- #
# 3. critical slowing down -> early-warning signal for ungrounding
# --------------------------------------------------------------------------- #
def drag_stiffness(N, m=M_EXP, N_c=TAU_D / C_FRIC):
    """Velocity-restoring basal-drag stiffness d tau_b/du (∝ (1-R)^2/R), the rate
    that -> 0 at the N_c fold. Returned in arbitrary (normalisable) units."""
    N = np.asarray(N, float)
    R = (N_c / N) ** m
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(N > N_c, (1.0 - R) ** 2 / R, 0.0)
    return out


def restoring_rate(N, m=M_EXP, N_c=TAU_D / C_FRIC, lam_ref=1.0, N_ref=5.0e5):
    """Restoring rate lambda(N) [1/yr], normalised so lambda(N_ref)=lam_ref. The
    velocity perturbation relaxes as exp(-lambda t); lambda ∝ drag stiffness."""
    return lam_ref * drag_stiffness(N, m, N_c) / drag_stiffness(np.array(N_ref), m, N_c)


def ews_theory(m=M_EXP, N_c=TAU_D / C_FRIC, lam_ref=1.0, N_ref=5.0e5, dt_yr=0.1,
               D=1.0, n=60):
    """Equilibrium early-warning curves: stationary variance D/lambda(N) and
    lag-1 autocorrelation exp(-lambda dt) as N -> N_c."""
    N = np.geomspace(1.2 * N_c, N_ref, n)
    lam = restoring_rate(N, m, N_c, lam_ref, N_ref)
    var_eq = D / lam
    ac1 = np.exp(-lam * dt_yr)
    return dict(N_MPa=(N / 1e6).tolist(), lambda_per_yr=lam.tolist(),
                var_eq=var_eq.tolist(), ac1=ac1.tolist(),
                var_ratio_Nc_over_ref=float(var_eq[0] / var_eq[-1]))


def ews_realization(m=M_EXP, N_c=TAU_D / C_FRIC, N_start=5.0e5, N_stop=2.0 * 6.0e4,
                    T_yr=3000.0, dt_yr=0.1, window_yr=30.0, lam_ref=1.0,
                    N_ref=5.0e5, D=1.0, seed=1):
    """Integrate an Ornstein-Uhlenbeck velocity perturbation while N slowly
    declines from N_start to N_stop, then measure rolling variance + AC1 and the
    Kendall-tau trend (the operational early-warning estimator)."""
    from scipy.stats import kendalltau
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T_yr, dt_yr)
    N = np.linspace(N_start, N_stop, t.size)
    x = np.zeros(t.size)
    for i in range(1, t.size):
        lam = float(restoring_rate(np.array(N[i]), m, N_c, lam_ref, N_ref))
        x[i] = x[i - 1] - lam * x[i - 1] * dt_yr + np.sqrt(2.0 * D * dt_yr) * rng.standard_normal()
    w = int(window_yr / dt_yr)
    var = np.full(t.size, np.nan)
    ac1 = np.full(t.size, np.nan)
    for i in range(w, t.size):
        seg = x[i - w:i]
        var[i] = float(np.var(seg))
        ac1[i] = float(np.corrcoef(seg[:-1], seg[1:])[0, 1])
    ok = np.isfinite(var) & np.isfinite(ac1)
    tau_var = float(kendalltau(t[ok], var[ok])[0])
    tau_ac1 = float(kendalltau(t[ok], ac1[ok])[0])
    # decimate for storage
    dec = max(1, ok.sum() // 60)
    idx = np.where(ok)[0][::dec]
    return dict(
        kendall_tau_variance=tau_var, kendall_tau_ac1=tau_ac1,
        rising_ews=bool(tau_var > 0.5 and tau_ac1 > 0.5),
        N_window_MPa=[float(N_start / 1e6), float(N_stop / 1e6)],
        series=dict(t_yr=t[idx].tolist(), N_MPa=(N[idx] / 1e6).tolist(),
                    variance=var[idx].tolist(), ac1=ac1[idx].tolist()))


# --------------------------------------------------------------------------- #
# 4. field connection (optional, reads committed JSONs)
# --------------------------------------------------------------------------- #
def field_points(trunk_json):
    """Drainage detections -> (N proxy, amp, lag). N ~ rel * rho_i g H (ocean-
    connected n_hat); flags the lag-vs-N tension honestly."""
    if not (trunk_json and os.path.exists(trunk_json)):
        return None
    fd = json.load(open(trunk_json))
    pts = []
    for d in fd.get("detections", []):
        if d.get("resp_frac") and d.get("rel") is not None and d.get("H"):
            N = d["rel"] * RHO_I * G * d["H"]    # Pa (can be <0 => at/over flotation)
            pts.append(dict(lake=d["lake"], rel=d["rel"], H=d["H"],
                            N_MPa=N / 1e6, amp=d["resp_frac"], lag_yr=d.get("lag_to_peak"),
                            detrend_sigma=d.get("detrend_sigma")))
    if not pts:
        return None
    amp = np.array([p["amp"] for p in pts])
    lag = np.array([p["lag_yr"] for p in pts if p["lag_yr"]])
    rel = np.array([p["rel"] for p in pts])
    out = dict(points=pts, n=len(pts),
               corr_rel_lnamp=float(np.corrcoef(rel, np.log(amp))[0, 1]) if len(pts) >= 3 else None)
    if len(pts) >= 3 and lag.size == len(pts):
        out["corr_lnamp_lnlag"] = float(np.corrcoef(np.log(amp), np.log(lag))[0, 1])
        out["corr_rel_lnlag"] = float(np.corrcoef(rel, np.log(lag))[0, 1])
        out["lag_vs_N_tension"] = (
            "Amplitude rises toward flotation (corr(rel,ln amp)<0), as the master "
            "curve predicts. But the baseline cavity model predicts the discrete "
            "lag also RISES toward flotation, whereas these n=3 lags trend the "
            "opposite way (driven by the marginal Rutford 2.86-sigma point). So the "
            "lag-vs-N sign is an OPEN discriminator: more in-band detections are "
            "needed to settle whether lag grows or shrinks toward flotation.")
    return out


def gating_Nc_consistency(efp_json):
    """Use the ocean-gating TF-slope ratio (near-flotation / well-grounded) as an
    independent |s_N| ratio and back out an order-of-magnitude N_c from the master
    curve. Qualified: rel->N for the terciles is approximate."""
    if not (efp_json and os.path.exists(efp_json)):
        return None
    d = json.load(open(efp_json))
    try:
        mw = d["bedmachine_direct_N"]["domains"]["MARINE_WEST"]["terciles_P1"]
        s_lo = mw["low_rel_near_flotation"]["TF_slope"]
        s_hi = mw["high_rel_well_grounded"]["TF_slope"]
        rel_lo = mw["low_rel_near_flotation"]["median_rel"]
        rel_mid = mw["mid_rel"]["median_rel"]
        med_N = d["median_N_MPa_at_GL"]          # top-level overall median N [MPa]
    except Exception:
        return None
    m = M_EXP
    # master curve: |s_N(N)|/m = 1/(1-(N_c/N)^m). The well-grounded high tercile has
    # |s_N| ~ m, so the measured TF-slope ratio is the |s_N| ratio:
    ratio = s_lo / s_hi                       # |s_N(low N)| / |s_N(high N)| ~ s_N_lo/m
    frac = max(1.0 - 1.0 / ratio, 1e-6)       # = (N_c/N_lo)^m
    # near-flotation tercile N from a transparent rel∝N scaling off the overall
    # median N (the mid tercile): N_lo ~ med_N * rel_lo/rel_mid.
    N_lo = max(med_N * rel_lo / max(rel_mid, 1e-6), 1e-3)   # MPa
    N_c_est = N_lo * frac ** (1.0 / m)        # MPa
    return dict(TF_slope_ratio_lowrel_over_highrel=float(ratio),
                implied_sN_ratio=float(ratio), N_lo_MPa_approx=float(N_lo),
                N_c_estimate_MPa=float(N_c_est), N_c_RC_default_MPa=TAU_D / C_FRIC / 1e6,
                note=("order-of-magnitude only: rel->N for terciles uses N~med_N*rel_lo/rel_mid "
                      "and assumes dlnN/dTF ~constant across terciles + well-grounded |s_N|~m; "
                      "an independent cross-check that the ocean-gating curvature implies an N_c "
                      "of the same order as the RC default 0.06 MPa."))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(trunk_json=None, efp_json=None):
    verify = verify_closed_form()
    inv = inversion_synthetic_test()
    inv_robust = inversion_robustness()
    ews_t = ews_theory()
    ews_r = ews_realization()
    field = field_points(trunk_json)
    gating = gating_Nc_consistency(efp_json)
    N_c = TAU_D / C_FRIC
    return dict(
        sliding_law="regularized-Coulomb tau_b = C N (u/(u+u0))^(1/m) (Schoof 2005; Joughin et al. 2019)",
        params=dict(tau_d_Pa=TAU_D, C=C_FRIC, u0_mpyr=U0, m=M_EXP, N_c_MPa=N_c / 1e6),
        master_curve="|s_N|(N) = m / (1 - (N_c/N)^m);  near flotation ~ N_c/(N-N_c)",
        well_grounded_note=WELL_GROUNDED_NOTE,
        closed_form_verification=verify,
        inversion_method=inv,
        inversion_robustness=inv_robust,
        critical_slowing_down=dict(
            mechanism=("near flotation the Coulomb plateau makes basal drag velocity-"
                       "insensitive: d tau_b/du ∝ (1-R)^2/R -> 0 at the N_c fold, so the "
                       "restoring rate vanishes (critical slowing down)."),
            theory=ews_t, realization=ews_r,
            new_observable=("rising variance + lag-1 autocorrelation of ICE SPEED as a "
                            "precursor to ungrounding (MISI), distinct from the Greenland "
                            "surface-melt CSD of Boers & Rypdal 2021 (different observable, "
                            "mechanism, and threshold)."),
            references="Scheffer et al. 2009 Nature; Dakos et al. 2008 PNAS; Boers & Rypdal 2021 PNAS"),
        field_overlay=field,
        gating_Nc_consistency=gating,
        verdict=(
            f"Closed-form s_N(N) master curve verified to {verify['max_reldiff_vs_numeric']:.1e} "
            f"vs the repo's numeric s_N; |s_N|->{verify['s_N_wellgrounded']:.2f} well-grounded "
            f"and diverges as N_c/(N-N_c) at N_c={N_c/1e6:.3f} MPa. The two-parameter law is "
            "INVERTIBLE: a natural drainage-step population recovers the flotation threshold "
            f"N_c to a few percent (robustly: {inv_robust['per_noise'][1]['N_c_median_relerr']:.1%} "
            f"at 10% amplitude noise), while m is degenerate with f far from flotation "
            f"({inv_robust['per_noise'][1]['m_median_relerr']:.0%}). The same N_c fold "
            "implies a critical-slowing-down early-warning signal (rising velocity variance + "
            f"AC1) toward ungrounding (Kendall tau var={ews_r['kendall_tau_variance']:.2f}, "
            f"AC1={ews_r['kendall_tau_ac1']:.2f})."),
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N_c = TAU_D / C_FRIC
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))

    # (a) master curve + field overlay
    N = np.geomspace(1.02 * N_c, 2.0e6, 400)
    ax[0, 0].plot(N / 1e6, s_N_closed(N), color="#c2185b", lw=2,
                  label=r"$|s_N|=m/(1-(N_c/N)^m)$")
    ax[0, 0].plot(N / 1e6, pole_approx(N), "k:", lw=1, label=r"$\sim N_c/(N-N_c)$")
    ax[0, 0].axvline(N_c / 1e6, color="k", ls="--", label=f"$N_c$={N_c/1e6:.3f} MPa")
    ax[0, 0].axhline(M_EXP, color="gray", ls=":", lw=1, label=f"$m$={M_EXP:.0f}")
    fo = res.get("field_overlay")
    if fo:
        Np = np.array([p["N_MPa"] for p in fo["points"]])
        ampp = np.array([p["amp"] for p in fo["points"]])
        # plot field as |s_N| ~ amp/f_hat using inversion f (illustrative)
        f_il = res["inversion_method"].get("f_hat", 0.05)
        ax[0, 0].scatter(np.clip(Np, 1.02 * N_c / 1e6, None), ampp / f_il,
                         c="#1f77b4", s=55, edgecolor="k", zorder=5,
                         label="field du/u  /  f (illustrative)")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("effective pressure N [MPa]"); ax[0, 0].set_ylabel(r"$|s_N|$")
    ax[0, 0].set_title("(a) closed-form s_N(N) master curve"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3, which="both")

    # (b) inversion plant->recover
    inv = res["inversion_method"]
    tru, rec = inv["truth"], inv.get("recovery", {})
    txt = (f"plant:   m={tru['m']:.2f}  N_c={tru['N_c_MPa']:.3f} MPa  f={tru['f']:.3f}\n"
           f"recover: m={inv.get('m_hat',float('nan')):.2f}  "
           f"N_c={inv.get('N_c_hat_MPa',float('nan')):.3f} MPa  "
           f"f={inv.get('f_hat',float('nan')):.3f}\n"
           f"rel-err: m {rec.get('m_relerr',float('nan')):.1%}, "
           f"N_c {rec.get('N_c_relerr',float('nan')):.1%}, "
           f"f {rec.get('f_relerr',float('nan')):.1%}")
    ax[0, 1].axis("off")
    ax[0, 1].text(0.02, 0.95, "inversion method (synthetic plant -> recover)",
                  fontsize=11, weight="bold", va="top")
    ax[0, 1].text(0.02, 0.78, txt, fontsize=10, va="top", family="monospace")
    ax[0, 1].text(0.02, 0.45,
                  "A population of drainage steps with known N and measured\n"
                  "amplitude du/u = |s_N(N)| f over-determines (m, N_c): the\n"
                  "flotation threshold and Weertman exponent are MEASURED,\n"
                  "not tuned (cf. Joughin 2019 fixed u0 + ad hoc h_T ramp).",
                  fontsize=9, va="top")

    # (c) EWS theory: variance & AC1 vs N
    et = res["critical_slowing_down"]["theory"]
    Nm = np.array(et["N_MPa"]); var = np.array(et["var_eq"]); ac1 = np.array(et["ac1"])
    axb = ax[1, 0]; axb2 = axb.twinx()
    axb.plot(Nm, var / var[-1], color="#d95f02", lw=2, label="variance / ref")
    axb2.plot(Nm, ac1, color="#1b9e77", lw=2, ls="--", label="lag-1 autocorr")
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.invert_xaxis()
    axb.axvline(N_c / 1e6, color="k", ls="--")
    axb.set_xlabel("N [MPa]  (decreasing -> flotation)")
    axb.set_ylabel("stationary variance (norm)", color="#d95f02")
    axb2.set_ylabel("lag-1 autocorrelation", color="#1b9e77")
    axb.set_title("(c) early-warning theory: variance & AC1 diverge at N_c")

    # (d) EWS realization
    er = res["critical_slowing_down"]["realization"]["series"]
    t = np.array(er["t_yr"]); v = np.array(er["variance"]); a = np.array(er["ac1"])
    axd = ax[1, 1]; axd2 = axd.twinx()
    axd.plot(t, v, color="#d95f02", lw=1.5, label="rolling variance")
    axd2.plot(t, a, color="#1b9e77", lw=1.5, ls="--", label="rolling AC1")
    axd.set_xlabel("time [yr]  (N declining toward flotation)")
    axd.set_ylabel("rolling variance", color="#d95f02")
    axd2.set_ylabel("rolling AC1", color="#1b9e77")
    kr = res["critical_slowing_down"]["realization"]
    axd.set_title(f"(d) realization: rising EWS "
                  f"(Kendall tau var={kr['kendall_tau_variance']:.2f}, "
                  f"AC1={kr['kendall_tau_ac1']:.2f})")
    fig.suptitle("s_N(N) master curve, inversion, and a critical-slowing-down "
                 "early-warning signal for ungrounding", fontsize=12)
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
    ap.add_argument("--trunk-json", default=os.path.join(_REPORTS, "lake_lag_trunk.json"))
    ap.add_argument("--efp-json", default=os.path.join(_REPORTS, "efp_gate_direct_n.json"))
    ap.add_argument("--out", default=os.path.join(_REPORTS, "sn_master_curve.json"))
    a = ap.parse_args()
    res = run(a.trunk_json, a.efp_json)
    rob = res["inversion_robustness"]["per_noise"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2, default=_json_default)
    print("=== s_N(N) master curve / inversion / critical slowing down ===")
    v = res["closed_form_verification"]
    print(f"  closed form vs numeric: max rel-diff {v['max_reldiff_vs_numeric']:.2e}; "
          f"|s_N|_wellgrounded={v['s_N_wellgrounded']:.3f}; pole rel-diff {v['near_pole_max_reldiff']:.2e}")
    inv = res["inversion_method"]
    if inv.get("ok"):
        r = inv["recovery"]
        print(f"  inversion plant->recover: m {r['m_relerr']:.1%}, N_c {r['N_c_relerr']:.1%}, f {r['f_relerr']:.1%}")
    print("  inversion robustness (N_c robust, m degenerate):")
    for row in rob:
        print(f"    noise={row['noise']:.2f}: N_c-err {row['N_c_median_relerr']:.1%}, m-err {row['m_median_relerr']:.1%}")
    cr = res["critical_slowing_down"]["realization"]
    print(f"  EWS realization: Kendall tau var={cr['kendall_tau_variance']:.2f}, "
          f"AC1={cr['kendall_tau_ac1']:.2f}, rising={cr['rising_ews']}")
    if res.get("field_overlay"):
        fo = res["field_overlay"]
        print(f"  field overlay: n={fo['n']} corr(rel,ln amp)={fo.get('corr_rel_lnamp')}")
        if fo.get("lag_vs_N_tension"):
            print("    lag-vs-N: OPEN discriminator (see JSON)")
    if res.get("gating_Nc_consistency"):
        print(f"  gating N_c estimate: {res['gating_Nc_consistency']['N_c_estimate_MPa']:.3f} MPa "
              f"(RC default 0.06)")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
