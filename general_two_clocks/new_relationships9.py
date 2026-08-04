r"""New derived cross-relationship NR32, continuing the program (NR1-NR31) with the same
discipline: *derived* from mainstream theory + repo results and *numerically verified*
(CPU, deterministic).  See ``REPORT_NEW_RELATIONSHIPS9.md`` for the write-up and
``tests/test_new_relationships9.py`` for the unit proof.

NR32 - THE DRAINAGE-RESPONSE WINDOW: which lakes can produce a detectable post-drainage
       speed-up is set by a BAND-LIMITED hydraulic transmission that is *peaked at the
       cavity<->channel transition* -- and the Markovian (adiabatic) collapse of the
       hydraulic memory kernel DESTROYS the peak, predicting instead that the most
       distributed beds respond most.  The 18/19-null + one transition-zone detection of
       the ATL15xITS_LIVE population test is the peaked prediction, not the monotone one.
       Radar bed-echo specularity content is the field map of the window.
       [P4a MZ kernel x sec V.2d population result x Schroeder 2013 specularity transition]

  The gap this closes.  The sec V.2d modern matched-lag test found the sec G.4 surge is NOT a
  universal consequence of lake drainage: 18/19 well-resolved drained lakes show no response
  (<= ~3%), while ONE (Thw_142, Thwaites) shows a +8.5%, 4.5-sigma step with lag-to-peak
  1.125 yr inside the derived 0.02-2 yr band.  The honest verdict said "the response needs a
  dynamically-primed bed" -- but left "primed" undefined.  NR32 derives what "primed" means:
  the bed's hydraulic relaxation time must fall INSIDE the sliding-observable window, which
  happens on the way from distributed to channelized drainage -- i.e. at the very
  distributed->channelized transition that radar specularity was invented to map (Schroeder
  et al. 2013 PNAS, on Thwaites itself).

  Derivation (all mainstream pieces; the composition is the contribution).
  1. Lumped linear hydrology (the same 2-compartment linearisation whose Green's function the
     repo certified as an exact Mori-Zwanzig kernel; Werder et al. 2013 / Hewitt 2013 physics):
     a drainage impulse DV first charges the local storage over an input time tau_in (flood
     arrival/spread), then relaxes through the drainage system with
         tau_sys(c) = R(c) * C(c),
     where R = evacuation resistance and C = storage capacitance both FALL with the
     channelization state c in [0,1] (linked-cavity/canal -> R-channel; Kamb 1987;
     Roethlisberger 1972; Schoof 2010).  The pressure (equivalently effective-pressure)
     response to the impulse is the two-stage cascade
         Dp_hat(omega) = DV * R(c) * H(omega),
         H(omega) = 1 / ((1 + i omega tau_in)(1 + i omega tau_sys)),
     with lag-to-peak  t* = tau_in tau_sys ln(tau_in/tau_sys) / (tau_in - tau_sys).
  2. The sliding response is observable only inside a finite band: the derived surge window
     T in [0.02, 2] yr (repo sec G.4; velocity records cannot resolve faster, secular trends
     swallow slower).  The detectable signal is therefore the BAND-LIMITED transmission
         T_band(c) = (DV R(c))^2 * (1/pi) * Int_{omega_1}^{omega_2} |H(omega; c)|^2 domega,
     omega_i = 2 pi / T_i -- computable in closed form (partial fractions -> arctan).
  3. The two ends die for different reasons: deep-distributed beds (c->0) have tau_sys >>
     T_max, so the response is quasi-static and its in-band spectral content vanishes as
     1/tau_sys^2 (it hides in the secular trend); mature channelized beds (c->1) evacuate with
     tau_sys << T_min AND with small R, so the in-band content dies as R(c)^2.  The interior
     maximum therefore exists IFF the log-slope of tau_sys = R C exceeds that of R alone --
     i.e. IFF channelization also removes STORAGE (decades_C > 0).  This is a sharp, derived
     dichotomy: if channels merely conducted (fixed storage), the most distributed bed would
     respond most; because channelization drains storage (sheet/cavity area -> channel
     volume), the response is peaked at the transition, where tau_sys(c) crosses the window
     and the lag-to-peak t*(c) sits inside [0.02, 2] yr.
  4. THE MARKOVIAN CONTROL (the two-clocks tie-in): collapse the memory to a delta with the
     same DC gain (the adiabatic-elimination closure the repo's MZ certification degenerates
     to) and the window selectivity is destroyed:
         T_band^Markov(c) ~ (DV R(c))^2 * (omega_2 - omega_1)/pi,
     monotone in c (maximal at the MOST distributed bed, no interior peak).  So the observed
     "peaked" detectability -- nulls at distributed Siple-coast/interior lakes AND at fast
     channelized outlets, detection at the Thwaites transition -- is a MEMORY signature that
     an adiabatic (K-theory-style) hydrology closure cannot reproduce.
  5. The field map: radar bed-echo specularity content measures exactly the distributed(high)
     -> channelized(low) axis (Schroeder et al. 2013, 2015; Young et al. 2016; Dow et al.
     2020).  NR32 therefore predicts drainage-response detectability is a SINGLE-PEAKED
     function of upstream specularity, and lag-to-peak within detections falls with
     channelization -- a stackable, falsifiable population test for the NISAR era.  The one
     in-band detection reads consistently: Thw_142's 1.125 yr lag EXCLUDES the mature-channel
     end (which would respond within the drainage quarter), while its detected amplitude
     excludes the deep-distributed end (below the population floor), jointly placing it in
     the transition neighbourhood -- where Schroeder et al. (2013) mapped Thwaites'
     distributed->channelized transition.  (Greenland's seasonal story -- early-summer
     speedups that self-limit after channelization, Bartholomew et al. 2010; Sundal et al.
     2011; Schoof 2010 -- is the same window statement evaluated where distributed tau_sys
     already sits in/above the observing band: the monotone limit of the peaked law.  The
     Stearns et al. 2008 Byrd trunk speed-up after an upstream lake flood is a second
     literature in-band case, gauged on the outlet trunk the flood transited rather than at
     the slow lake centroid -- the gauge-position corollary of the same transmission.)

  What is NEW here (vs the cited mainstream): (i) the band-limited transmission form and its
  interior-peak theorem tying "dynamically primed" to tau_sys inside the window; (ii) the
  Markovian control showing the peak (and hence WHICH lakes respond) is memory-carried --
  the same delta-collapse diagnosed across this repo; (iii) specularity as the remote-sensing
  coordinate of the window, making the population prediction stackable from existing radar +
  altimetry archives; (iv) the Thw_142 lag -> tau_sys -> transition-state inversion.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# ----------------------------------------------------------------------------- #
# closed forms
# ----------------------------------------------------------------------------- #
YR = 1.0  # all times in years

BAND = (0.02, 2.0)          # observable lag window [yr] (repo sec G.4 / sec V.2d)


def omega_band(band=BAND):
    T1, T2 = band
    return 2.0 * np.pi / T2, 2.0 * np.pi / T1     # omega_lo, omega_hi


def band_integral(tau_a, tau_b, band=BAND):
    r"""Closed form of  Int_{omega_1}^{omega_2} |H|^2 domega  for the cascade
    H = 1/((1+i w tau_a)(1+i w tau_b)):
      |H|^2 = 1/((1+w^2 a^2)(1+w^2 b^2))
            = [a^2/(a^2-b^2)] 1/(1+w^2 a^2) - [b^2/(a^2-b^2)] 1/(1+w^2 b^2),
      Int 1/(1+w^2 a^2) dw = arctan(a w)/a.
    Degenerate a==b handled by the analytic limit.
    """
    w1, w2 = omega_band(band)
    a, b = float(tau_a), float(tau_b)
    if abs(a - b) < 1e-12 * max(a, b):
        # Int dw / (1+a^2 w^2)^2 = [ w/(2(1+a^2w^2)) + arctan(a w)/(2a) ]
        def F(w):
            return w / (2.0 * (1.0 + a * a * w * w)) + np.arctan(a * w) / (2.0 * a)
        return F(w2) - F(w1)
    ca = a * a / (a * a - b * b)
    cb = b * b / (a * a - b * b)
    return (ca * (np.arctan(a * w2) - np.arctan(a * w1)) / a
            - cb * (np.arctan(b * w2) - np.arctan(b * w1)) / b)


def band_integral_numeric(tau_a, tau_b, band=BAND, n=200_001):
    w1, w2 = omega_band(band)
    w = np.linspace(w1, w2, n)
    h2 = 1.0 / ((1.0 + (w * tau_a) ** 2) * (1.0 + (w * tau_b) ** 2))
    return np.trapezoid(h2, w)


def t_peak(tau_a, tau_b):
    """Lag-to-peak of the cascade impulse response (interior maximum)."""
    a, b = float(tau_a), float(tau_b)
    if abs(a - b) < 1e-12 * max(a, b):
        return a
    return a * b * np.log(a / b) / (a - b)


# ----------------------------------------------------------------------------- #
# channelization maps (log-linear; wide literature-spanned sweeps prove robustness)
# ----------------------------------------------------------------------------- #
def maps(c, tau_dist=20.0, decades_R=3.0, decades_C=1.0):
    """(R(c), tau_sys(c)) with R and C log-linear in the channelization c.

    R falls ``decades_R`` decades (linked-cavity -> R-channel conductivity
    contrast) and the storage capacitance C falls ``decades_C`` decades
    (distributed sheet/cavity area -> channel volume), so
    tau_sys = R C falls ``decades_R + decades_C`` decades from ``tau_dist``.
    """
    c = np.asarray(c, float)
    R = 10.0 ** (-decades_R * c)                     # R(0)=1 (normalised)
    tau_sys = tau_dist * 10.0 ** (-(decades_R + decades_C) * c)
    return R, tau_sys


def interior_peak_criterion(decades_R, decades_C):
    r"""NR32's derived existence condition for the interior transmission peak.

    With r = -dlnR/dc = ln10*decades_R and s = -dln tau/dc = ln10*(decades_R+decades_C):
      dlnT/dc = -2r + 2s   while tau_sys is ABOVE the window (B ~ tau^-2),
      dlnT/dc = -2r        once tau_sys is BELOW the window (B flat),
    so T rises at c=0 iff s > r  <=>  decades_C > 0, and always falls at c=1:
    an interior maximum exists  IFF channelization removes storage (decades_C>0).
    If channelization only lowered resistance at fixed storage (decades_C=0),
    the distributed end would dominate (argmax at c=0) -- the observable
    difference between "channels drain the bed" and "channels merely conduct".
    """
    return decades_C > 0.0


def transmission(c, tau_in=0.1, band=BAND, markovian=False, **mkw):
    """Band-limited transmission T_band(c) (arbitrary units; shapes are the content)."""
    R, tau_sys = maps(c, **mkw)
    if markovian:
        w1, w2 = omega_band(band)
        return R ** 2 * (w2 - w1) / np.pi
    out = np.array([band_integral(tau_in, tb, band) for tb in np.atleast_1d(tau_sys)])
    return R ** 2 * out / np.pi


# ----------------------------------------------------------------------------- #
# the NR32 computation
# ----------------------------------------------------------------------------- #
def nr32(n_c=241, band=BAND):
    cs = np.linspace(0.0, 1.0, n_c)
    out = {"band_yr": list(band), "c_grid_n": n_c, "sweeps": [], "closed_form_check": None,
           "markovian_control": {}, "thw142": {}, "spec_anchor": None}

    # closed form == numeric quadrature
    errs = []
    for ta, tb in ((0.05, 3.0), (0.1, 0.1), (0.02, 40.0), (0.3, 0.005)):
        errs.append(abs(band_integral(ta, tb, band) - band_integral_numeric(ta, tb, band))
                    / band_integral_numeric(ta, tb, band))
    out["closed_form_check"] = {"max_rel_err": float(max(errs))}

    # robustness sweep: the derived criterion PREDICTS the peak topology in every combo
    n_pred_ok = 0
    n_window_ok = 0
    n_interior_cases = 0
    for tau_dist in (10.0, 20.0, 50.0):
        for dec_R in (2.0, 3.0, 4.0):
            for dec_C in (0.0, 0.5, 1.0, 2.0):
                for tau_in in (0.05, 0.1, 0.25):
                    T = transmission(cs, tau_in=tau_in, band=band, tau_dist=tau_dist,
                                     decades_R=dec_R, decades_C=dec_C)
                    i = int(np.argmax(T))
                    interior = 0 < i < n_c - 1
                    predicted = interior_peak_criterion(dec_R, dec_C)
                    pred_ok = (interior == predicted) or (not predicted and i == 0)
                    n_pred_ok += bool(pred_ok)
                    rec = {"tau_dist": tau_dist, "decades_R": dec_R, "decades_C": dec_C,
                           "tau_in": tau_in, "c_peak": float(cs[i]),
                           "interior": bool(interior), "predicted_interior": bool(predicted),
                           "prediction_correct": bool(pred_ok)}
                    if interior and predicted:
                        n_interior_cases += 1
                        _, tau_at = maps(cs[i], tau_dist=tau_dist, decades_R=dec_R,
                                         decades_C=dec_C)
                        tstar_at = t_peak(tau_in, float(tau_at))
                        in_win = band[0] <= tstar_at <= band[1]
                        n_window_ok += bool(in_win)
                        rec.update({"tau_sys_at_peak_yr": float(tau_at),
                                    "t_star_at_peak_yr": float(tstar_at),
                                    "t_star_in_window": bool(in_win),
                                    "prominence_over_endpoints": float(
                                        T[i] / max(T[0], T[-1]))})
                    out["sweeps"].append(rec)
    out["criterion_predicts_all"] = bool(n_pred_ok == len(out["sweeps"]))
    out["n_combos"] = len(out["sweeps"])
    out["interior_cases"] = {"n": n_interior_cases, "t_star_in_window_n": n_window_ok,
                             "all_in_window": bool(n_window_ok == n_interior_cases)}

    # Markovian (adiabatic / delta-kernel) control: monotone, peak at the WRONG end
    Tm = transmission(cs, markovian=True)
    out["markovian_control"] = {
        "monotone_decreasing": bool(np.all(np.diff(Tm) <= 1e-15)),
        "argmax_at_c0": bool(int(np.argmax(Tm)) == 0),
        "interior_peak": False,
        "verdict": "delta-collapsed kernel predicts distributed beds respond MOST and no "
                   "interior peak -- falsified by the 18/19 nulls incl. distributed lakes",
    }

    # Thw_142: what the one in-band detection pins down (honest two-sided read)
    #   forcing duration from ATL15: drawdown 2021.75 -> 2022.0  =>  tau_in ~ 0.25 yr
    #   observed lag-to-peak: 1.125 yr (quarterly resolution +-0.125)
    tau_in_obs = 0.25
    t_star_obs = 1.125
    # (i) the LAG bounds the channelized end: a mature-channel tau_sys would respond
    #     within the same quarter (t* < 0.25 yr). Find the excluded tau_sys range.
    taus = np.logspace(-3, 4, 4001)
    tstars = np.array([t_peak(tau_in_obs, t) for t in taus])
    tau_min_lag = float(taus[np.argmax(tstars >= 0.25)])   # smallest tau with t*>=1 quarter
    # (ii) the DETECTABILITY bounds both ends: with the population floor ~3% and the
    #     observed +8.5%, the detection requires T_band(c) >= (3/8.5) * T_peak.
    Tc = transmission(cs, tau_in=tau_in_obs)
    thr = (0.03 / 0.085) * float(Tc.max())
    det = cs[Tc >= thr]
    _, tau_c = maps(cs)
    lag_ok = cs[np.array([t_peak(tau_in_obs, float(t)) for t in tau_c]) >= 0.25]
    joint = np.intersect1d(np.round(det, 6), np.round(lag_ok, 6))
    out["thw142"] = {
        "tau_in_yr_from_ATL15_drawdown": tau_in_obs,
        "t_star_obs_yr": t_star_obs,
        "lag_floor_tau_sys_yr": tau_min_lag,
        "lag_excludes": "mature-channel states tau_sys < %.3f yr (same-quarter response)"
                        % tau_min_lag,
        "lag_insensitive_above": "t* grows only logarithmically in tau_sys (t* ~ "
                                 "tau_in ln(tau_sys/tau_in)), so the lag alone cannot "
                                 "discriminate transition from deep-distributed",
        "detectable_c_interval_central_map": [float(det.min()), float(det.max())]
        if det.size else None,
        "joint_c_interval": [float(joint.min()), float(joint.max())] if joint.size else None,
        "reading": "the detection + lag jointly place Thw_142 in the transition "
                   "neighbourhood (window-crossing tau_sys), excluding both the mature-"
                   "channel end (lag) and the deep-distributed end (amplitude floor) -- "
                   "consistent with the Schroeder et al. (2013) distributed->channelized "
                   "transition mapped at Thwaites",
    }

    # optional: ICECAP specularity at the East Antarctic drained-null lakes (data-gated)
    out["spec_anchor"] = _spec_anchor()
    return out


# ----------------------------------------------------------------------------- #
# optional real-data anchor: mean ICECAP specularity near EAIS drained lakes
# ----------------------------------------------------------------------------- #
_EAIS_LAKES = {  # ATL15-modern-drained lakes inside the ICECAP lon window (sec V.2d catalogue)
    "Byrd_1": (148.31, -81.03), "Byrd_2": (146.88, -80.69),
    "Byrd_s10": (139.02, -81.83), "David_s2": (152.93, -75.24),
}


def _spec_anchor(spec_dir="/home/K/_data_specularity", radius_km=30.0):
    files = glob.glob(os.path.join(spec_dir, "**", "*_spec.txt"), recursive=True)
    if not files:
        return {"available": False,
                "hint": "USAP-DC 601371 profiles not present; see "
                        "glaciers/validation/external/rtn_specularity_real.py"}
    try:
        from pyproj import Transformer
    except Exception:
        return {"available": False, "hint": "pyproj not installed"}
    lons, lats, specs = [], [], []
    for f in files:
        a = np.loadtxt(f, comments="#", usecols=(3, 4, 5), ndmin=2)
        if a.size:
            lons.append(a[:, 0]); lats.append(a[:, 1]); specs.append(a[:, 2])
    lon = np.concatenate(lons); lat = np.concatenate(lats); sp = np.concatenate(specs)
    m = np.isfinite(sp)
    lon, lat, sp = lon[m], lat[m], np.clip(sp[m], 0, 1)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x, y = tr.transform(lon, lat)
    x, y = np.asarray(x), np.asarray(y)
    res = {"available": True, "radius_km": radius_km,
           "coverage_median_spec": float(np.median(sp)), "lakes": {}}
    for name, (lo, la) in _EAIS_LAKES.items():
        xl, yl = tr.transform(lo, la)
        d = np.hypot(x - xl, y - yl)
        sel = d < radius_km * 1e3
        res["lakes"][name] = {
            "n_pts": int(sel.sum()),
            "mean_spec": float(np.mean(sp[sel])) if sel.any() else None,
            "p90_spec": float(np.percentile(sp[sel], 90)) if sel.any() else None,
        }
    return res


# ----------------------------------------------------------------------------- #
def make_figure(out, path_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = np.linspace(0, 1, out["c_grid_n"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    T = transmission(cs)
    Tm = transmission(cs, markovian=True)
    ax[0].semilogy(cs, T / T.max(), "C0", lw=2, label="memory kernel (NR32)")
    ax[0].semilogy(cs, Tm / Tm.max(), "C3--", lw=2, label="Markovian ($\\delta$) control")
    ax[0].set_xlabel("channelization state c  (0=distributed, 1=R-channel)")
    ax[0].set_ylabel("band-limited transmission (norm.)")
    ax[0].set_title("(a) the detectability window is peaked;\nthe $\\delta$-kernel closure is monotone")
    ax[0].legend(frameon=False, fontsize=9)

    _, tau_sys = maps(cs)
    ts = np.array([t_peak(0.25, tb) for tb in tau_sys])
    ax[1].semilogy(cs, ts, "C0", lw=2)
    ax[1].axhspan(*out["band_yr"], color="0.85", label="observable window 0.02–2 yr")
    ax[1].axhline(out["thw142"]["t_star_obs_yr"], color="C1", lw=1.5, ls=":",
                  label="Thw_142 lag 1.125 yr")
    ax[1].set_xlabel("channelization state c"); ax[1].set_ylabel("lag-to-peak t* [yr]")
    ax[1].set_title("(b) lag map: t*(c) crosses the window\nexactly where (a) peaks")
    ax[1].legend(frameon=False, fontsize=9)

    sa = out.get("spec_anchor") or {}
    if sa.get("available"):
        names = [n for n, v in sa["lakes"].items() if v["n_pts"] > 0]
        vals = [sa["lakes"][n]["mean_spec"] for n in names]
        ax[2].bar(range(len(names)), vals, color="C0")
        ax[2].axhline(sa["coverage_median_spec"], color="k", ls="--", lw=1,
                      label="ICECAP coverage median")
        ax[2].set_xticks(range(len(names)))
        ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax[2].set_ylabel(f"mean specularity (r<{sa['radius_km']:.0f} km)")
        ax[2].set_title("(c) EAIS drained-null lakes on the\nspecularity (channelization) axis")
        ax[2].legend(frameon=False, fontsize=8)
    else:
        ax[2].axis("off")
        ax[2].text(0.05, 0.5, "spec anchor: data not present\n(USAP-DC 601371)", fontsize=9)
    fig.suptitle("NR32 — the drainage-response window: peaked at the cavity↔channel transition",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=os.path.join(FIGDIR, "nr32_drainage_response_window.json"))
    ap.add_argument("--png-out", default=os.path.join(FIGDIR, "nr32_drainage_response_window.png"))
    a = ap.parse_args()
    out = nr32()
    os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
    with open(a.json_out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("closed_form_check", "criterion_predicts_all", "n_combos",
                       "interior_cases", "markovian_control", "thw142")}, indent=1))
    sa = out.get("spec_anchor") or {}
    if sa.get("available"):
        print("spec anchor:", json.dumps(sa, indent=1))
    make_figure(out, a.png_out)
    print("wrote", a.json_out, "and", a.png_out)


if __name__ == "__main__":
    main()
