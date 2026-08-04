r"""NR65 — the phase face of the tidal EIS: the MSf phase-lag profile measures the
bed's complex wavenumber, and its 45-degree loss angle is the parabolic clock's
fingerprint (k_r = k_i), closing the impedance reading of NR64 with a second,
independent in-situ diffusivity.

The claim (two clocks, field face)
----------------------------------
NR64 measured |Z|: the upstream ATTENUATION length delta_amp = 1/k_i of each tidal
constituent on the committed §I.5 harmonic cache. A diffusive (parabolic) bed
transmits exp(i*omega*t - (1+i)*d/delta): amplitude decay and phase lag advance at
the SAME rate,

    k_r = k_i = sqrt(omega / 2K)      (loss angle 45 deg, |k_i/k_r| = 1)
    c_phase = omega/k_r = sqrt(2*omega*K),   K_phase = omega / (2 k_r^2)

so the PHASE-lag profile phi(d) is a second, independent measurement of the same
hydraulic diffusivity K that the amplitude ladder gave — the two must agree with NO
free parameter. The competing readings decouple the two faces: in-situ generation by
the nonlinear sliding law alone (Rosier, Gudmundsson & Green 2014, their Fig. 6b)
predicts MSf phase "almost constant" upstream (k_r ~ 0 while k_i > 0); purely
elastic stress transmission likewise screens amplitude without accumulating travel
time. The measured loss tangent k_i/k_r is therefore a sharper discriminant than
NR64's underpowered Mm/MSf ratio: 1 = parabolic transport, >>1 = reactive screening
or in-situ generation.

Mainstream anchors
------------------
* Gudmundsson 2006 (Nature) measured 1-2 m/s propagation of the tidal modulation on
  Rutford (GL,+10,+20,+40 km); Adalgeirsdottir et al. 2008 found locally 10+/-4 m/s
  over a 3-km array — "propagation velocity is variable along the ice stream".
* Rosier, Gudmundsson & Green 2014 (The Cryosphere 8, 1763): viscoelastic coupling-
  length model gives period-dependent phase velocity — 1.45 m/s semidiurnal vs
  0.27 m/s (= 23 km/d) fortnightly, ratio ~ sqrt(T_MSf/T_M2) = 5.3, i.e. their own
  model output already carries the sqrt(omega) dispersion of a parabolic clock.
* Minchew et al. 2017 (JGR): InSAR on Rutford — the MSf signal propagates upstream
  at ~29 km/d (0.34 m/s) decaying quasi-linearly over ~85 km, and originates over
  the ice SHELF (buttressing nonlinearity; also Robel et al. 2017).
* Rosier et al. 2017 (ESSD 9, 849; THIS archive): "decay length scales are
  relatively uniform for all ice streams but the speed at which the Msf signal
  propagates upstream shows more variation" — stated qualitatively; the per-branch
  complex wavenumber (k_r AND k_i jointly) is not published. That joint read-out,
  its loss tangent, and the K_phase-vs-K_amp closure are the NR65 contribution.

Why cross-epoch phases are legitimate: MSf on grounded ice is the M2xS2 difference
frequency; its forcing phase is astronomically locked (phi_M2 - phi_S2), so
deployments from different years share a common clock once all records are reduced
to one absolute epoch (done in the rebuilt §I.5 cache: hours since 1970-01-01,
y = A cos(w*tt + ph)). The EMPIRICAL control is the floating-station M2 height
phase: truly-afloat neighbours (separation < 60 km, |d_GL| >= 5 km) deployed in
different seasons must agree — measured spread ~0.04 rad (checked below), which is
the cross-epoch phase floor.

Honest scope
------------
Formal per-station sigma_phi (sig_A/A ~ 1e-3 rad) is unrealistically small against
seasonal/epoch systematics — branch errors are jackknife over stations, as in NR64.
MSf on grounded ice is partly GENERATED along the path (Gudmundsson 2007/2011), so
k_r is an EFFECTIVE transmission phase gradient exactly as delta_amp is an effective
attenuation (the K values are linear-response equivalents; distributed generation
biases both faces together, which is why their RATIO is the robust discriminant).
Two-point branches carry a 2*pi branch ambiguity: |dphi| > pi/2 is flagged wrap_risk
and the alternative branch is reported. The semidiurnal phase profile is NOT fitted
on grounded ice (mm-level amplitudes at the OTL floor with uncorrected OTL phase).
Grounding-zone floating stations (|d_GL| < 5 km) are excluded from the ocean control
(flexure modifies their phase).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import new_relationships41 as R41                                  # noqa: E402

_CACHE = R41._CACHE
_FIGDIR = os.path.join(_HERE, "figures")

OMEGA = {c: 2.0 * math.pi / (T * 3600.0) for c, T in R41.PERIODS_H.items()}  # rad/s
BRANCHES = R41.BRANCHES
MIN_SIG = R41.MIN_SIG
WRAP_RISK_RAD = math.pi / 2.0
# literature reference points (see docstring)
REFS = {
    "minchew2017_rutford_insar_km_day": 29.0,
    "rosier2014_model_msf_m_s": 0.27,
    "rosier2014_model_semidiurnal_m_s": 1.45,
    "gudmundsson2006_modulation_m_s": [1.0, 2.0],
}


def _stations(cache):
    return {st["name"]: st for st in cache["stations"]}


def _phase_row(st, c):
    """(A, sig_A, phase, sig_phi) for constituent c, or None."""
    d = st["constituents"].get(c)
    if not d or d.get("ph_disp_rad") is None:
        return None
    return d["A_disp_m"], d["sig_disp_m"], d["ph_disp_rad"], d.get("sig_ph_rad")


def unwrap_along(d_km, ph):
    """Nearest-2pi-branch unwrap walking upstream (d ascending)."""
    o = np.argsort(d_km)
    out = np.array(ph, float)
    for i, j in zip(o[:-1], o[1:]):
        out[j] += 2.0 * math.pi * round((out[i] - out[j]) / (2.0 * math.pi))
    return out


def kendall_exact_one_sided(d, ph):
    """Exact one-sided permutation p for NEGATIVE ordering (lag accumulates
    upstream): p = P(tau_perm <= tau_obs). Enumerates all n! for n <= 8."""
    from itertools import permutations
    from scipy.stats import kendalltau
    d = np.asarray(d, float); ph = np.asarray(ph, float)
    n = d.size
    if n < 3 or n > 8:
        return None
    tau = float(kendalltau(d, ph)[0])
    taus = [kendalltau(d, np.asarray(p))[0] for p in permutations(ph)]
    p = float(np.mean([t <= tau + 1e-12 for t in taus]))
    return dict(tau=tau, p_one_sided=p, n=int(n))


def fit_phase_slope(d_km, ph_rad, snr, min_sig=MIN_SIG):
    """Weighted LSQ phi = a + b*d with NR64-style weights (SNR^2, capped) and
    jackknife slope error. Returns dict or None. k_r = -b (registered sign:
    phase DECREASES upstream for a wave entering at the grounding line)."""
    d, p, w = [], [], []
    for dk, phk, sk in zip(d_km, ph_rad, snr):
        if sk is None or sk < min_sig:
            continue
        d.append(dk); p.append(phk); w.append(sk ** 2)
    n = len(d)
    if n < 2:
        return None
    d = np.asarray(d); w = np.asarray(w)
    p = unwrap_along(d, np.asarray(p))
    w = np.minimum(w, np.median(w) * 10)          # one hyper-precise station cap

    def _slope(dd, pp, ww):
        W = ww.sum()
        dm = (ww * dd).sum() / W
        pm = (ww * pp).sum() / W
        var = (ww * (dd - dm) ** 2).sum()
        if var <= 0:
            return np.nan, np.nan
        b = (ww * (dd - dm) * (pp - pm)).sum() / var
        return b, pm - b * dm
    b, a = _slope(d, p, w)
    if not np.isfinite(b):
        return None
    raw_steps = np.abs(np.diff(p[np.argsort(d)]))
    out = dict(slope_rad_km=float(b), intercept_rad=float(a), n=int(n),
               two_point=(n == 2),
               wrap_risk=bool(n == 2 and raw_steps.size and raw_steps[0] > WRAP_RISK_RAD),
               sign_violation=bool(b >= 0.0), err_rad_km=np.nan)
    if n > 2:
        jk = []
        for k in range(n):
            m = np.ones(n, bool); m[k] = False
            bk, _ = _slope(d[m], p[m], w[m])
            if np.isfinite(bk):
                jk.append(bk)
        if len(jk) == n:
            out["err_rad_km"] = float(math.sqrt(
                (n - 1) / n * np.sum((np.asarray(jk) - np.mean(jk)) ** 2)))
    return out


def _derived(k_r_rad_km, err_rad_km, omega_rad_s):
    """delta_phase, c, K_phase (+ propagated errors) from k_r [rad/km]."""
    k_m = k_r_rad_km / 1e3                       # rad/m
    c = omega_rad_s / k_m                        # m/s
    rel = (err_rad_km / k_r_rad_km) if (np.isfinite(err_rad_km)
                                        and k_r_rad_km > 0) else np.nan
    return dict(
        delta_phase_km=float(1.0 / k_r_rad_km),
        delta_phase_err_km=float(rel / k_r_rad_km) if np.isfinite(rel) else np.nan,
        c_m_s=float(c), c_km_day=float(c * 86.4),
        c_err_m_s=float(c * rel) if np.isfinite(rel) else np.nan,
        K_phase_m2_s=float(omega_rad_s / (2.0 * k_m ** 2)),
        K_phase_err_m2_s=(float(omega_rad_s / (2.0 * k_m ** 2) * 2.0 * rel)
                          if np.isfinite(rel) else np.nan))


def ocean_epoch_control(cache, min_snr=20.0, min_dgl_km=5.0, max_sep_km=60.0):
    """Truly-afloat neighbour pairs must share the M2 height phase across
    deployment epochs. Returns per-pair |dphi| and the max spread."""
    rows = []
    for st in cache["stations"]:
        if st.get("grounded_measures") or st.get("d_gl_km") is None:
            continue
        if abs(st["d_gl_km"]) < min_dgl_km:
            continue
        cc = st["constituents"].get("M2")
        if not cc or cc.get("ph_h_rad") is None or cc["sig_h_m"] <= 0:
            continue
        if cc["A_h_m"] / cc["sig_h_m"] < min_snr:
            continue
        rows.append((st["name"], st["x"], st["y"], cc["ph_h_rad"]))
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            sep = math.hypot(rows[i][1] - rows[j][1],
                             rows[i][2] - rows[j][2]) / 1e3
            if sep > max_sep_km:
                continue
            dphi = abs((rows[i][3] - rows[j][3] + math.pi) %
                       (2.0 * math.pi) - math.pi)
            pairs.append(dict(pair=[rows[i][0], rows[j][0]],
                              sep_km=float(sep), dphi_rad=float(dphi)))
    return dict(stations=[r[0] for r in rows], pairs=pairs,
                max_dphi_rad=(float(max(p["dphi_rad"] for p in pairs))
                              if pairs else None))


def analyze(cache=None, constituent="MSf"):
    cache = cache or json.load(open(_CACHE))
    S = _stations(cache)
    omega = OMEGA[constituent]
    res = {"what": "NR65: MSf phase-lag profile -> complex wavenumber of the "
                   "subglacial bed; loss tangent k_i/k_r vs the parabolic-clock "
                   "prediction 1; phase velocity + independent K_phase closure "
                   "against NR64's amplitude-derived K",
           "constituent": constituent, "omega_rad_s": float(omega),
           "references": REFS, "branches": {}}
    for br, names in BRANCHES.items():
        d_km, amps, sigs, phs, snrs = [], [], [], [], []
        st_rows = []
        for nm in names:
            st = S.get(nm)
            if not st or st.get("d_gl_km") is None:
                continue
            row = _phase_row(st, constituent)
            if row is None:
                continue
            A, sA, ph, sph = row
            snr = (A / sA) if sA > 0 else 0.0
            d_km.append(st["d_gl_km"]); amps.append(A); sigs.append(sA)
            phs.append(ph); snrs.append(snr)
            st_rows.append(dict(name=nm, d_gl_km=float(st["d_gl_km"]),
                                A_disp_m=float(A), snr=float(snr),
                                ph_rad=float(ph),
                                sig_ph_rad=(float(sph) if sph else None),
                                used=bool(snr >= MIN_SIG)))
        entry = dict(stations=st_rows)
        used = [(dk, pk) for dk, pk, sk in zip(d_km, phs, snrs) if sk >= MIN_SIG]
        if len(used) >= 3:
            du = [u[0] for u in used]
            pu = unwrap_along(du, [u[1] for u in used])
            ordering = kendall_exact_one_sided(du, pu)
            if ordering:
                entry["ordering"] = ordering
        fit = fit_phase_slope(d_km, phs, snrs)
        if fit:
            entry["phase_fit"] = fit
            if not fit["sign_violation"]:
                k_r = -fit["slope_rad_km"]
                entry["derived"] = _derived(k_r, fit["err_rad_km"], omega)
                # amplitude face on the SAME stations (NR64 machinery)
                fa = R41.fit_delta(d_km, amps, sigs)
                if fa and np.isfinite(fa["delta_km"]):
                    k_i = 1.0 / fa["delta_km"]
                    entry["amp_fit"] = fa
                    entry["K_amp_m2_s"] = float(
                        (fa["delta_km"] * 1e3) ** 2 * omega / 2.0)
                    ratio = k_i / k_r
                    rel_a = (fa["err_km"] / fa["delta_km"]
                             if np.isfinite(fa.get("err_km", np.nan)) else np.nan)
                    rel_p = (fit["err_rad_km"] / k_r
                             if np.isfinite(fit["err_rad_km"]) else np.nan)
                    err = (abs(ratio) * math.hypot(rel_a, rel_p)
                           if np.isfinite(rel_a) and np.isfinite(rel_p) else np.nan)
                    entry["loss_tangent"] = dict(
                        k_i_over_k_r=float(ratio), err=float(err),
                        parabolic_prediction=1.0)
                    entry["K_geometric_m2_s"] = float(
                        omega / (2.0 * (k_r / 1e3) * (k_i / 1e3)))
            if fit["two_point"] and fit["wrap_risk"]:
                # alternative 2pi branch for the far station
                dd = np.asarray([r["d_gl_km"] for r in st_rows
                                 if r["used"]])
                pp = np.asarray([r["ph_rad"] for r in st_rows if r["used"]])
                o = np.argsort(dd)
                dphi = pp[o][1] - pp[o][0]
                alt = dphi + math.copysign(2.0 * math.pi, dphi)   # one more turn
                b_alt = alt / (dd[o][1] - dd[o][0])
                entry["alt_branch"] = dict(slope_rad_km=float(b_alt))
                if b_alt < 0:
                    entry["alt_branch"]["derived"] = _derived(-b_alt, np.nan, omega)
        res["branches"][br] = entry
    res["ocean_epoch_control"] = ocean_epoch_control(cache)
    res["verdict"] = _verdict(res)
    return res


def _verdict(res):
    lines = []
    for br, e in res["branches"].items():
        f = e.get("phase_fit")
        if not f:
            continue
        if f["sign_violation"]:
            lines.append(f"{br}: phase INCREASES upstream (sign violation"
                         f"{', 2pt' if f['two_point'] else ''}) — no transmission "
                         f"read-out")
            continue
        dv = e["derived"]
        tag = "2pt" + ("+wrap-risk" if f.get("wrap_risk") else "") \
            if f["two_point"] else f"n={f['n']}"
        s = (f"{br}: k_r={-f['slope_rad_km']:.4f} rad/km ({tag}) -> "
             f"delta_phi={dv['delta_phase_km']:.0f} km, "
             f"c={dv['c_km_day']:.0f} km/d ({dv['c_m_s']:.2f} m/s), "
             f"K_phi={dv['K_phase_m2_s']:.0f} m2/s")
        o = e.get("ordering")
        if o:
            s += (f"; ordering tau={o['tau']:+.2f} "
                  f"(exact one-sided p={o['p_one_sided']:.3f})")
        lt = e.get("loss_tangent")
        if lt:
            s += (f"; k_i/k_r={lt['k_i_over_k_r']:.2f}"
                  + (f"+/-{lt['err']:.2f}" if np.isfinite(lt["err"]) else "")
                  + f" vs 1 (parabolic), K_amp={e['K_amp_m2_s']:.0f}")
        lines.append(s)
    oc = res["ocean_epoch_control"]
    n_ok = sum(1 for e in res["branches"].values()
               if e.get("phase_fit") and not e["phase_fit"]["sign_violation"])
    n_fit = sum(1 for e in res["branches"].values() if e.get("phase_fit"))
    lines.append(f"sign concordance: {n_ok}/{n_fit} branches accumulate lag "
                 f"upstream (registered direction)")
    if oc.get("max_dphi_rad") is not None:
        lines.append(f"ocean epoch control: max |dphi(M2,h)| = "
                     f"{oc['max_dphi_rad']:.3f} rad across afloat neighbour pairs "
                     f"({len(oc['pairs'])} pairs) — cross-epoch phase floor")
    lines.append(f"references: Minchew-2017 Rutford InSAR "
                 f"{REFS['minchew2017_rutford_insar_km_day']:.0f} km/d; "
                 f"Rosier-2014 model MSf {REFS['rosier2014_model_msf_m_s']} m/s "
                 f"(= {REFS['rosier2014_model_msf_m_s'] * 86.4:.0f} km/d)")
    return " | ".join(lines)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    colors = dict(zip(res["branches"], plt.cm.tab10.colors))
    for br, e in res["branches"].items():
        rows = [r for r in e["stations"] if r["used"]]
        if not rows:
            continue
        d = np.array([r["d_gl_km"] for r in rows])
        p = unwrap_along(d, np.array([r["ph_rad"] for r in rows]))
        c = colors[br]
        ax[0].plot(d, p, "o", color=c, ms=6, label=br)
        for i, r in enumerate(rows):
            ax[0].annotate(r["name"], (d[i], p[i]),
                           fontsize=6, xytext=(3, 3), textcoords="offset points")
        f = e.get("phase_fit")
        if f and not f["sign_violation"]:
            dd = np.linspace(d.min(), d.max(), 10)
            ax[0].plot(dd, f["intercept_rad"] + f["slope_rad_km"] * dd, "-",
                       color=c, lw=1, alpha=0.7)
    ax[0].set_xlabel("distance upstream of grounding line [km]")
    ax[0].set_ylabel(r"MSf phase $\phi$ [rad] (common 1970 epoch)")
    ax[0].set_title("the MSf phase lag accumulates upstream:\n"
                    r"$\phi(d)=\phi_0-k_r d$ — travelling, not standing", fontsize=10)
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
    # panel b: k_i vs k_r
    ks = []
    for br, e in res["branches"].items():
        if "derived" not in e or "loss_tangent" not in e:
            continue
        k_r = 1.0 / e["derived"]["delta_phase_km"]
        k_i = 1.0 / e["amp_fit"]["delta_km"]
        ks.append((br, k_r, k_i, e["phase_fit"]["two_point"]))
    if ks:
        lo = 0.5 * min(min(k[1], k[2]) for k in ks)
        hi = 2.0 * max(max(k[1], k[2]) for k in ks)
        ax[1].plot([lo, hi], [lo, hi], "k--", lw=1,
                   label=r"$k_i=k_r$ (parabolic, 45$^\circ$ loss angle)")
        for br, k_r, k_i, twop in ks:
            ax[1].plot(k_r, k_i, "s" if twop else "o", color=colors[br], ms=8)
            ax[1].annotate(br + (" (2pt)" if twop else ""), (k_r, k_i), fontsize=7,
                           xytext=(4, 4), textcoords="offset points")
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel(r"$k_r$ = phase-lag rate [rad km$^{-1}$]")
        ax[1].set_ylabel(r"$k_i$ = attenuation rate [km$^{-1}$]")
        ax[1].set_title("the bed's complex wavenumber:\n"
                        "diffusion predicts the diagonal with NO free parameter",
                        fontsize=10)
        ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8)
    fig.suptitle("NR65 — phase face of the tidal EIS (BAS Filchner-Ronne GPS, "
                 "MSf constituent)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        _FIGDIR, "100_tidal_phase_dispersion.json"))
    a = ap.parse_args()
    res = analyze()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("VERDICT:", res["verdict"])
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
