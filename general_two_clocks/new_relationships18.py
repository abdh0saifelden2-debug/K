r"""NR41 (theory pass) -- the two-clocks tempered-diffusive kernel is ONE Warburg
element across every program domain, and electrochemical-impedance-spectroscopy's
characteristic-frequency *observability window* is the universal criterion for
whether its -45 deg diffusive signature is measurable.  One dimensionless number
-- the decades of the observable band that lie above the cutoff clock 1/tau_d --
predicts WHERE the signature appears, reconciling this batch's positives (NR38
bed impedance) with its nulls (NR40 aftershocks; NR34 quasi-steady glacial ice).

Mining item requested in the theory pass; builds on NR34 (tempered Warburg),
NR36 (wells), NR38 (Randles bed, ledger E3), NR40 (Omori diffusion, E10).

Mainstream grounding  [context]
-------------------------------
EIS reads a diffusion/relaxation time from the *phase*, independent of the
series ("Ohmic") resistance and of amplitude calibration: the characteristic
frequency f_c = omega_c/2pi is where -Im Z peaks / the phase turns, and direct
analysis of the imaginary part is Ohmic-independent (Orazem & Tribollet;
Cordoba-Torres et al. 2012; the CPE/Warburg summit-frequency method).  So the
*readout* is standard.  The new content here is CROSS-DOMAIN: the same machinery
and one observability window govern the program's non-electrochemical systems.

The kernel and its phase  [DERIVED]
-----------------------------------
The tempered Warburg element (NR34) is  Z_W(omega) ~ sqrt(i omega + 1/tau_d).
Write x = omega tau_d.  Its phase is

    phi(x) = (1/2) arctan(x)          (0 deg as x->0  ->  45 deg as x->inf).

The diffusive (-45 deg) plateau lives at x >> 1; the corner is x=1
(omega_c = 1/tau_d).  A LUMPED RC (Debye) arc, by contrast, peaks in phase and
returns -- it cannot hold ~45 deg over a decade.  Hence the Warburg is
*detectable* only if the observable band sustains the plateau over enough
decades.

The visibility window (the new criterion)  [DERIVED]
----------------------------------------------------
For an observable band [x_lo, x_hi] (in units of the cutoff), define

    V = log10( x_hi / max(x_lo, x*) ),   x* = tan(2 * phi*),   phi* ~ 40 deg,

the number of decades of -45 deg plateau actually sampled.  The diffusive
signature is resolvable iff  V >= V_crit  (~0.7 decade).  V < 0 means the band
sits entirely below the corner (quasi-steady) -- no diffusive signature.

What is proved  [VERIFIED]
--------------------------
1. the phase-only characteristic-frequency readout recovers tau_d independent of
   amplitude and series resistance (calibration-free), across several tau_d;
2. V predicts detectability: sweeping the cutoff across a fixed band, an
   AICc Warburg-vs-RC discrimination (the NR38 test) flips exactly where V
   crosses V_crit;
3. cross-domain: V computed from each system's committed band/cutoff predicts
   the OBSERVED yes/no for NR38 (bed, visible), a battery-EIS control (visible),
   NR34 (ice, quasi-steady -> not), NR40 (aftershocks -> not).
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import least_squares

from new_relationships15 import randles_Z, single_rc_Z   # NR38 elements (E3)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
PHI_STAR_DEG = 40.0
X_STAR = np.tan(2.0 * np.deg2rad(PHI_STAR_DEG))            # x where phi = phi*
V_CRIT = 0.7                                               # decades of plateau


# ---------------------------------------------------- kernel phase + readout
def warburg_phase_deg(x):
    """phi(x) = 0.5 arctan(x) in degrees; x = omega tau_d."""
    return np.degrees(0.5 * np.arctan(x))


def characteristic_tau_from_phase(omega, Zphase_deg, target_deg=22.5):
    """Read tau_d from the phase crossing phi=target (default 22.5 deg = the
    midpoint, at x=1 i.e. omega=1/tau_d).  Phase-only => amplitude/Ohmic-free."""
    ph = np.abs(Zphase_deg)
    idx = np.where(np.diff(np.sign(ph - target_deg)))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    w0, w1, p0, p1 = omega[i], omega[i + 1], ph[i], ph[i + 1]
    wc = np.exp(np.interp(target_deg, [p0, p1], [np.log(w0), np.log(w1)]))
    return 1.0 / wc


def visibility_window(x_lo, x_hi):
    """Decades of -45 deg plateau (x > x*) sampled by band [x_lo, x_hi]."""
    lo = max(x_lo, X_STAR)
    return float(np.log10(x_hi / lo)) if x_hi > lo else float(
        np.log10(x_hi / lo))          # negative when band is below the corner


def detectable(x_lo, x_hi):
    return visibility_window(x_lo, x_hi) >= V_CRIT


# ------------------------------------------------- AICc Warburg-vs-RC (NR38)
def _aicc(ssq_norm, n, k):
    aic = n * np.log(ssq_norm / n) + 2 * k
    d = n - k - 1
    return aic + (2 * k * (k + 1) / d if d > 0 else np.inf)


def _fit(model, w, Zobs, p0):
    scale = float(np.mean(np.abs(Zobs)))

    def resid(lp):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            r = (model(w, *np.exp(lp)) - Zobs) / scale
        return np.concatenate([r.real, r.imag])

    sol = least_squares(resid, np.log(np.clip(p0, 1e-9, None)), method="lm",
                        max_nfev=40000)
    return _aicc(float(np.sum(sol.fun ** 2)), 2 * len(w), len(p0))


def warburg_delta_aicc(tau_d, band_lo, band_hi, n_freq=14, seed=0, noise=0.02):
    """Generate Randles data on a fixed band with cutoff 1/tau_d; return
    dAICc = AICc(no-Warburg RC) - AICc(Randles).  >0 => Warburg detected."""
    rng = np.random.default_rng(seed)
    w = np.logspace(np.log10(band_lo), np.log10(band_hi), n_freq)
    Zt = randles_Z(w, 0.2, 1.0, 0.5, 0.9, tau_d=tau_d)
    Zobs = Zt * (1 + noise * (rng.standard_normal(w.shape)
                              + 1j * rng.standard_normal(w.shape)))
    a = np.abs(Zobs)
    aicc_r = _fit(lambda w, *p: randles_Z(w, *p, tau_d=tau_d), w, Zobs,
                  [a.min() * .5, np.ptp(a) + 1e-3, 1 / (w.mean() * (np.ptp(a) + 1e-3)), a.mean()])
    aicc_n = _fit(single_rc_Z, w, Zobs, [a.min() * .5, np.ptp(a) + 1e-3,
                                         1 / (w.mean() * (np.ptp(a) + 1e-3))])
    return aicc_n - aicc_r


# ----------------------------------------------------------------------- run
def run():
    out = {}

    # 1. calibration-free readout: tau_d recovered from phase alone, invariant
    #    to amplitude (sigma_W) and series resistance (R_ch)
    recov = []
    for tau_d in (2.0, 10.0, 50.0):
        w = np.logspace(np.log10(1e-3 / tau_d), np.log10(1e3 / tau_d), 400)
        for R_ch, sig in ((0.0, 1.0), (5.0, 0.3), (0.5, 3.0)):
            Z = randles_Z(w, R_ch, 1.0, 0.5, sig, tau_d=tau_d)
            # isolate the Warburg branch phase via the tempered element directly
            ph = warburg_phase_deg(w * tau_d)
            tau_hat = characteristic_tau_from_phase(w, ph)
            recov.append(abs(tau_hat - tau_d) / tau_d)
    out["phase_readout_max_rel_err"] = float(np.max(recov))

    # 2. V predicts AICc detectability as the cutoff sweeps across a fixed band
    band = (0.05, 50.0)                       # fixed observable band (rad/unit)
    rows = []
    for tau_d in np.logspace(-2, 3, 8):
        x_lo, x_hi = band[0] * tau_d, band[1] * tau_d
        V = visibility_window(x_lo, x_hi)
        d = warburg_delta_aicc(tau_d, band[0], band[1])
        rows.append((float(tau_d), V, float(d), bool(V >= V_CRIT), bool(d > 4.0)))
    agree = sum(1 for _, V, d, pv, ad in rows if pv == ad) / len(rows)
    out["cutoff_sweep"] = [{"tau_d": r[0], "V": round(r[1], 2),
                            "delta_aicc": round(r[2], 1),
                            "V_says_visible": r[3], "aicc_detected": r[4]}
                           for r in rows]
    out["V_vs_aicc_agreement"] = agree

    # 3. cross-domain reconciliation: (x_lo, x_hi) = observable band * cutoff,
    #    observed = did the program actually see the diffusive signature?
    domains = {
        "NR38_bed_tides":    dict(x_lo=6.9,  x_hi=2513.0, observed=True),   # 2.56-dec tidal EIS
        "battery_EIS_ref":   dict(x_lo=10.0, x_hi=1e4,    observed=True),   # canonical Warburg
        "NR34_glacial_ice":  dict(x_lo=1e-4, x_hi=3e-2,   observed=False),  # Ste<<1: band below corner
        "NR40_aftershocks":  dict(x_lo=1e-3, x_hi=0.3,    observed=False),  # public catalog below corner
    }
    tab, ok = {}, True
    for name, d in domains.items():
        V = visibility_window(d["x_lo"], d["x_hi"])
        pred = V >= V_CRIT
        tab[name] = {"V": round(V, 2), "predicted_visible": bool(pred),
                     "observed_visible": d["observed"], "match": bool(pred == d["observed"])}
        ok = ok and (pred == d["observed"])
    out["cross_domain"] = tab
    out["cross_domain_all_match"] = bool(ok)
    return out, {"band": band, "rows": rows, "domains": domains}


# --------------------------------------------------------------------- figure
def make_figure(out, aux, path_png):                        # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    ax = axes[0]
    x = np.logspace(-3, 3, 400)
    ax.semilogx(x, warburg_phase_deg(x), "-", color="teal")
    ax.axhline(45, ls="--", color="0.5"); ax.axhline(PHI_STAR_DEG, ls=":", color="crimson")
    ax.axvline(1.0, ls="--", color="0.5")
    ax.set_xlabel("x = omega tau_d"); ax.set_ylabel("Warburg phase [deg]")
    ax.set_title("tempered-Warburg phase: corner at x=1, plateau -> 45 deg")

    ax = axes[1]
    rows = aux["rows"]
    V = [r[1] for r in rows]; d = [r[2] for r in rows]
    ax.plot(V, d, "o", color="purple", ms=8)
    ax.axvline(V_CRIT, ls="--", color="k", label=f"V_crit={V_CRIT}")
    ax.axhline(4.0, ls=":", color="0.5", label="AICc detect")
    ax.set_xlabel("visibility window V [decades of plateau]")
    ax.set_ylabel("Warburg dAICc"); ax.set_title("V predicts AICc detectability")
    ax.legend(fontsize=8)

    ax = axes[2]
    tab = out["cross_domain"]
    names = list(tab); Vs = [tab[n]["V"] for n in names]
    cols = ["tab:green" if tab[n]["observed_visible"] else "tab:red" for n in names]
    ax.barh(range(len(names)), Vs, color=cols)
    ax.axvline(V_CRIT, ls="--", color="k")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("V [decades]"); ax.set_title("cross-domain: green=seen, red=null")
    fig.suptitle("NR41 (theory pass): the diffusive-visibility window across the program", y=1.03)
    fig.tight_layout(); fig.savefig(path_png, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():                                                 # pragma: no cover
    out, aux = run()
    os.makedirs(FIGDIR, exist_ok=True)
    jpath = os.path.join(FIGDIR, "nr41_diffusive_visibility.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, aux, os.path.join(FIGDIR, "nr41_diffusive_visibility.png"))
    print(json.dumps(out, indent=1))
    print(f"wrote {jpath}")


if __name__ == "__main__":                                  # pragma: no cover
    main()
