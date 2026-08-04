r"""NR38 -- the subglacial bed is a Randles circuit: P4a's MZ kernel, P4b's tidal
admittance, and NR11's phase lag are ONE object, the bed impedance Z(omega),
and the six-constituent tidal admittance is an *impedance-spectroscopy sweep*
with Kramers-Kronig validation.

Ledger item E3 (papers/RESEARCH_RECAP_AND_HORIZON.md).  The Warburg -45 deg
element derived in NR34 (E2) is, as the ledger notes, "the E3 entry point": the
bed's response to a periodic (tidal) pressure load is the electrochemical-
impedance-spectroscopy (EIS) canon (Randles 1947; Barsoukov & Macdonald 2005)
with renamed constants.

The circuit (every element already in the repo)
-----------------------------------------------
  Z(w) = R_ch  +  1 / ( i w C  +  1/( R_ct + Z_W(w) ) )                  (Randles)

  * R_ch  -- series channel resistance  = P4a/NR32 channel resistance R(c);
  * C     -- distributed storage        = NR32's two-compartment storage C(c)
             (the R_ct||C arc reproduces NR11's `arctan(w R_ct C)` phase lag);
  * R_ct  -- interfacial/transfer resistance;
  * Z_W   -- tempered Warburg  Z_W(w) = sigma_W / sqrt(i w + 1/tau_d) = the NR34
             tempered half-derivative element (finite tau_d -> finite DC, so Z is
             Kramers-Kronig-clean; tau_d is NR34's tempering time).  The semi-
             infinite limit tau_d -> inf gives the -45 deg Warburg line.

  Null (no diffusion):  Z = R_ch + R_ct/(1 + i w R_ct C)   -- Randles WITHOUT the
  Warburg (a single Debye/RC arc).  "Is the bed diffusive?" == "is the Warburg
  element needed?" == nested AICc test.

The measurement
---------------
Tidal forcing samples Z(w) at the constituent lines, chosen to span the widest
band (a real EIS sweep): Ssa (semiannual), Mf (fortnightly), O1/K1 (diurnal),
M2/S2 (semidiurnal).  The complex admittance Y = 1/Z at 6 lines is 12 real
numbers constraining 4 parameters -- an over-determined fit with a mature
identification + KK-validation theory.

What is proved here (in-repo, CPU, deterministic)
-------------------------------------------------
1. Recovery: a synthetic Randles bed sampled ONLY at the 6 tidal lines with
   realistic noise is inverted back to its 4 parameters.
2. KK causality (NR9/NR23 face): the fitted Z satisfies the Kramers-Kronig DC
   sum rule  Z'(0) - Z'(inf) = (2/pi) INT_0^inf (-Im Z)/w dw  to small residual.
3. Discrimination (falsifiable): the 6-line sweep RESOLVES the Warburg element --
   the no-Warburg null is rejected by AICc on diffusive-bed data, and (control)
   NOT spuriously rejected on non-diffusive (single-RC) data.  The tidal
   admittance can thus *detect whether the bed diffuses*.

External data (data-gated): open GNSS tidal admittance (Rutford/Whillans;
Gudmundsson 2006; Rosier et al.) -- not required for the proofs above.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import least_squares

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

CONSTITUENTS_CPD = {
    "Ssa": 0.005476, "Mf": 0.073202,
    "O1": 0.929536, "K1": 1.002738,
    "M2": 1.932274, "S2": 2.000000,
}
OMEGA = 2.0 * np.pi * np.array(list(CONSTITUENTS_CPD.values()))   # rad/day
TAU_D = 200.0          # NR34 tempering time [day]; finite -> KK-clean


# ------------------------------------------------------------- circuit models
def warburg(w, sigma_w, tau_d=TAU_D):
    return sigma_w / np.sqrt(1j * w + 1.0 / tau_d)


def randles_Z(w, R_ch, R_ct, C, sigma_w, tau_d=TAU_D):
    return R_ch + 1.0 / (1j * w * C + 1.0 / (R_ct + warburg(w, sigma_w, tau_d)))


def single_rc_Z(w, R_ch, R_ct, C):
    """Randles with the Warburg removed -- the non-diffusive null (one RC arc)."""
    return R_ch + R_ct / (1.0 + 1j * w * R_ct * C)


# ------------------------------------------------------------------- fitting
def _resid(Zmodel, Zobs, scale):
    r = (Zmodel - Zobs) / scale
    return np.concatenate([r.real, r.imag])


def _aicc(sumsq_norm, n, k):
    aic = n * np.log(sumsq_norm / n) + 2 * k
    denom = n - k - 1
    return aic + (2 * k * (k + 1) / denom if denom > 0 else np.inf)


def _fit(model, w, Zobs, p0):
    scale = float(np.mean(np.abs(Zobs)))
    lp0 = np.log(np.clip(p0, 1e-9, None))
    sol = least_squares(lambda lp: _resid(model(w, *np.exp(lp)), Zobs, scale),
                        lp0, method="lm", max_nfev=40000)
    ssq = float(np.sum(sol.fun ** 2))
    return np.exp(sol.x), _aicc(ssq, 2 * len(w), len(p0)), ssq * scale ** 2


def fit_randles(w, Zobs):
    a = np.abs(Zobs)
    p0 = [a.min() * 0.5, np.ptp(a) + 1e-3, 1.0 / (OMEGA.mean() * (np.ptp(a) + 1e-3)),
          a.mean()]
    p, aicc, rss = _fit(randles_Z, w, Zobs, p0)
    return {"params": dict(zip(("R_ch", "R_ct", "C", "sigma_w"), p.tolist())),
            "aicc": aicc, "rss": rss}


def fit_single_rc(w, Zobs):
    a = np.abs(Zobs)
    p0 = [a.min() * 0.5, np.ptp(a) + 1e-3, 1.0 / (OMEGA.mean() * (np.ptp(a) + 1e-3))]
    p, aicc, rss = _fit(single_rc_Z, w, Zobs, p0)
    return {"params": p.tolist(), "aicc": aicc, "rss": rss}


# --------------------------------------------------------- Kramers-Kronig (DC)
def kk_dc_sumrule(Zfun, wlo=1e-6, whi=1e6, npts=200001):
    """Causality check via the KK DC sum rule (the NR23 face):
        Z'(0) - Z'(inf) = (2/pi) INT_0^inf (-Im Z(w)) / w dw.
    Returns (lhs, rhs, rel_err)."""
    lhs = float(Zfun(np.array([wlo]))[0].real - Zfun(np.array([whi]))[0].real)
    w = np.logspace(np.log10(wlo), np.log10(whi), npts)
    integrand = (-Zfun(w).imag) / w
    rhs = (2.0 / np.pi) * float(np.sum(0.5 * (integrand[1:] + integrand[:-1])
                                       * np.diff(w)))
    return lhs, rhs, abs(lhs - rhs) / (abs(lhs) + 1e-30)


# ----------------------------------------------------------------------- run
def run(seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    true = dict(R_ch=0.20, R_ct=1.20, C=0.50, sigma_w=0.90)
    Zt = randles_Z(OMEGA, **true)
    Zobs = Zt * (1.0 + noise * (rng.standard_normal(Zt.shape)
                                + 1j * rng.standard_normal(Zt.shape)))

    fit = fit_randles(OMEGA, Zobs)
    rel_err = {k: abs(fit["params"][k] - true[k]) / true[k] for k in true}

    pf = fit["params"]
    Zfit = lambda w: randles_Z(w, pf["R_ch"], pf["R_ct"], pf["C"], pf["sigma_w"])
    kk_lhs, kk_rhs, kk_rel = kk_dc_sumrule(Zfit)

    # nested discrimination: does the diffusive-bed data NEED the Warburg?
    nullf = fit_single_rc(OMEGA, Zobs)
    d_aicc_diffusive = nullf["aicc"] - fit["aicc"]     # >0 => Warburg needed

    # control: non-diffusive (single-RC) data must NOT prefer the Warburg
    tc = (0.20, 1.60, 0.45)
    Zc = single_rc_Z(OMEGA, *tc)
    Zc = Zc * (1.0 + noise * (rng.standard_normal(Zc.shape)
                              + 1j * rng.standard_normal(Zc.shape)))
    d_aicc_control = fit_single_rc(OMEGA, Zc)["aicc"] - fit_randles(OMEGA, Zc)["aicc"]

    # over-determination: fit 4 constituents, predict the held-out 2
    idx_fit, idx_pred = [0, 1, 2, 4], [3, 5]
    f4 = fit_randles(OMEGA[idx_fit], Zobs[idx_fit])["params"]
    Zpred = randles_Z(OMEGA[idx_pred], f4["R_ch"], f4["R_ct"], f4["C"], f4["sigma_w"])
    pred_rel = float(np.mean(np.abs(Zpred - Zt[idx_pred]) / np.abs(Zt[idx_pred])))

    out = {
        "constituents_cpd": CONSTITUENTS_CPD,
        "band_decades": float(np.log10(OMEGA.max() / OMEGA.min())),
        "true_params": true, "fit_params": fit["params"],
        "recovery_rel_err": rel_err,
        "recovery_max_rel_err": float(max(rel_err.values())),
        "kk_dc_lhs": kk_lhs, "kk_dc_rhs": kk_rhs, "kk_dc_rel_residual": kk_rel,
        "discrim_delta_aicc_warburg_needed": d_aicc_diffusive,
        "control_delta_aicc_no_spurious_warburg": d_aicc_control,
        "overdetermination_predict_rel_err": pred_rel,
        "warburg_element": "sigma_W/sqrt(i w + 1/tau_d), tau_d=%.0f d (NR34 tempered)" % TAU_D,
    }
    return out, {"Zt": Zt, "Zobs": Zobs, "Zfit": Zfit}


# --------------------------------------------------------------------- figure
def make_figure(out, aux, path_png):                        # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w_dense = np.logspace(np.log10(OMEGA.min()), np.log10(OMEGA.max()), 400)
    Zc = aux["Zfit"](w_dense)
    Zobs = aux["Zobs"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(Zc.real, -Zc.imag, "-", color="teal", lw=1.5, label="fitted Randles")
    ax.plot(Zobs.real, -Zobs.imag, "o", color="crimson", ms=8, label="tidal lines")
    for name, w in zip(CONSTITUENTS_CPD, OMEGA):
        z = aux["Zfit"](np.array([w]))[0]
        ax.annotate(name, (z.real, -z.imag), fontsize=8)
    ax.set_xlabel("Re Z"); ax.set_ylabel("-Im Z")
    ax.set_title("Nyquist: bed impedance sampled by the tides")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_aspect("equal", "box")

    ax = axes[1]
    ax.loglog(w_dense, np.abs(Zc), "-", color="teal")
    ax.loglog(OMEGA, np.abs(Zobs), "o", color="crimson", ms=8)
    ax.set_xlabel("w [rad/day]"); ax.set_ylabel("|Z|")
    ax.set_title("Bode magnitude"); ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.semilogx(w_dense, np.degrees(np.angle(Zc)), "-", color="teal")
    ax.semilogx(OMEGA, np.degrees(np.angle(Zobs)), "o", color="crimson", ms=8)
    ax.axhline(-45, ls="--", color="0.5", lw=1)
    ax.set_xlabel("w [rad/day]"); ax.set_ylabel("phase [deg]")
    ax.set_title("Bode phase (-45 deg = Warburg)"); ax.grid(alpha=0.3, which="both")

    fig.suptitle("NR38 (E3): the subglacial bed as a Randles impedance", y=1.03)
    fig.tight_layout()
    fig.savefig(path_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():                                                 # pragma: no cover
    out, aux = run()
    os.makedirs(FIGDIR, exist_ok=True)
    jpath = os.path.join(FIGDIR, "nr38_bed_randles_impedance.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, aux, os.path.join(FIGDIR, "nr38_bed_randles_impedance.png"))
    print(json.dumps(out, indent=1))
    print(f"wrote {jpath} and the png")


if __name__ == "__main__":                                  # pragma: no cover
    main()
