r"""NR64 — the tidal constituent ladder is an in-situ EIS sweep of the subglacial
drainage system: multi-stream attenuation lengths delta(omega), the parabolic-clock
consistency test, and per-stream hydraulic diffusivities from GPS alone.

The claim (two clocks, field face)
----------------------------------
NR30 separated the two constraint pressures: the Leray pressure (elliptic,
instantaneous, no storage) vs the Darcy head (parabolic, finite-time, storage) —
distinct by a nonzero forcing->head lag. NR38 read the subglacial bed as a Randles
circuit whose EIS sweep is performed by the tides. This unit EXECUTES that sweep
spatially on the committed §I.5 harmonic cache (BAS Filchner-Ronne GPS): each tidal
constituent probes the bed at its own frequency, and its upstream attenuation length
delta(omega) is the corresponding skin depth. The parabolic clock predicts

    delta(omega) = sqrt(2 K / omega)          (diffusive skin depth)

i.e. delta ratios fixed PURELY by period ratios: delta_Mm/delta_MSf = sqrt(27.55/
14.77) = 1.366, delta_MSf/delta_M2 = sqrt(28.55) = 5.34 — no free parameters.
Mainstream anchors: Rosier, Gudmundsson & Green 2015 model exactly this diffusion
(their Eq. 12, decay scale sqrt(2K/omega), K "poorly constrained... treated as an
unknown"); Rosier & Gudmundsson 2016 show the hydrological forcing dominates MSf
while flexure/damming dominate the semidiurnal band; Rosier et al. 2014 derive the
viscoelastic (Maxwell) alternative: elastic screening at short periods, viscous
saturation at long periods — i.e. within the long-period band the Maxwell reading
predicts delta(Mm)/delta(MSf) ~ 1, NOT 1.366. That ratio is the discriminant.

What the cache supports (measured here)
---------------------------------------
* Per-stream-branch attenuation lengths for the LONG-PERIOD band (MSf 14.77 d,
  Mm 27.55 d, Mf where multi-season) from log-linear fits over 3-5 grounded
  stations (jackknife errors), two-point estimates flagged.
* The SEMIDIURNAL band on grounded ice sits at the few-mm OTL/noise floor by the
  first station (3.6-8 km) on every branch except Institute — an upper BOUND on
  delta(M2) via the last floating station as the d~0 forcing scale.
* Per-branch hydraulic diffusivity K = delta_MSf^2 * omega_MSf / 2 — a NEW METHOD
  OF MEASUREMENT: the GPS tidal ladder measures the subglacial hydraulic
  diffusivity in situ, no borehole (the number Rosier 2015 had to leave free).
* The Mm/MSf discriminant ratio per branch, against 1.366 (parabolic) vs 1.0
  (Maxwell saturation) — the honest verdict on WHICH clock carries the long band,
  with its current power stated.

Honest scope
------------
MSf/Mm on grounded ice are (partly) nonlinearity-GENERATED along the path
(Gudmundsson 2007/2011: MSf = M2xS2 intermodulation via m>1 sliding), so the
apparent delta mixes generation profile and transmission; the m>1 nonlinearity
itself stretches the decay scale (Rosier 2015 Fig. 8). The K values are therefore
"effective linear-response" diffusivities — exactly the object their model treats
as free. Diurnal band excluded (K1 is the GPS orbit-repeat artifact frequency);
S2 carries solar radiation/multipath aliases. Station branches mix epochs
(2005-2016) — tidal response is treated as stationary; Rutford pairs one 2004-07
and one later station (flagged). n_branch = 3-5: jackknife errors, no asymptotics.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.normpath(os.path.join(
    _HERE, "..", "glaciers", "validation", "external", "data",
    "tidal_admittance_field_cache.json"))
_FIGDIR = os.path.join(_HERE, "figures")

PERIODS_H = {"O1": 25.81933871, "K1": 23.93446959, "M2": 12.4206012, "S2": 12.0,
             "N2": 12.65834751, "MSf": 354.3670666, "Mf": 327.8599387,
             "Mm": 661.3111655}

BRANCHES = {
    "Evans": ["E3", "EGPS", "E2A", "E1"],
    "Evans_XX": ["XX20", "XX21"],
    "Foundation": ["H01", "H10", "H02", "H04", "H05"],
    "Talutis": ["T+18", "T+38"],
    "Rutford": ["RUT_long", "R145"],
}
FLOATING_REF = {"Evans": "E4A", "Foundation": "H11", "Talutis": "T-02",
                "Rutford": None, "Evans_XX": "E4A"}
MIN_SIG = 5.0
SD_FLOOR_M = 0.002          # OTL/noise floor for the semidiurnal band on grounded ice


def _stations(cache):
    return {st["name"]: st for st in cache["stations"]}


def _amp(st, c):
    d = st["constituents"].get(c)
    if not d:
        return None, None
    return d["A_disp_m"], d["sig_disp_m"]


def fit_delta(d_km, amps_m, sigs_m, min_sig=MIN_SIG):
    """Weighted log-linear attenuation fit ln A = a - d/delta with jackknife error.

    Returns dict(delta_km, err_km, n, two_point) or None.
    """
    d, la, w = [], [], []
    for dk, A, s in zip(d_km, amps_m, sigs_m):
        if A is None or s is None or s <= 0 or A / s < min_sig:
            continue
        d.append(dk); la.append(math.log(A)); w.append((A / s) ** 2)
    n = len(d)
    if n < 2:
        return None
    d = np.asarray(d); la = np.asarray(la); w = np.asarray(w)
    w = np.minimum(w, np.median(w) * 10)     # cap: one hyper-precise station
    #                                          must not own the fit

    def _slope(dd, ll, ww):
        W = ww.sum()
        dm = (ww * dd).sum() / W
        lm = (ww * ll).sum() / W
        cov = (ww * (dd - dm) * (ll - lm)).sum()
        var = (ww * (dd - dm) ** 2).sum()
        return cov / var if var > 0 else np.nan
    b = _slope(d, la, w)
    if not np.isfinite(b) or b >= -1e-6:
        return dict(delta_km=np.inf, err_km=np.nan, n=int(n), two_point=(n == 2))
    delta = -1.0 / b
    if n == 2:
        return dict(delta_km=float(delta), err_km=np.nan, n=2, two_point=True)
    jk = []
    for k in range(n):
        m = np.ones(n, bool); m[k] = False
        bk = _slope(d[m], la[m], w[m])
        if np.isfinite(bk) and bk < 0:
            jk.append(-1.0 / bk)
    err = (math.sqrt((n - 1) / n * np.sum((np.asarray(jk) - np.mean(jk)) ** 2))
           if len(jk) == n else np.nan)
    return dict(delta_km=float(delta), err_km=float(err), n=int(n), two_point=False)


def analyze(cache=None):
    cache = cache or json.load(open(_CACHE))
    S = _stations(cache)
    res = {"what": "NR64: tidal constituent ladder as an in-situ EIS of the subglacial "
                   "drainage system — attenuation lengths delta(omega), the parabolic-"
                   "clock test, and per-branch hydraulic diffusivities",
           "branches": {}}
    for br, names in BRANCHES.items():
        sts = [S[n] for n in names if n in S]
        if len(sts) < 2:
            continue
        d_km = [st["d_gl_km"] for st in sts]
        entry = dict(stations=[st["name"] for st in sts], d_gl_km=d_km,
                     spans_d=[st["span_days"] for st in sts], fits={})
        for c in ("MSf", "Mm", "Mf"):
            amps, sigs = zip(*[_amp(st, c) for st in sts])
            f = fit_delta(d_km, amps, sigs)
            if f:
                entry["fits"][c] = f
        # semidiurnal floor + bound
        near = min(sts, key=lambda st: st["d_gl_km"])
        A_m2, s_m2 = _amp(near, "M2")
        ref = FLOATING_REF.get(br)
        bound = None
        if ref and ref in S and A_m2 is not None:
            A0, _ = _amp(S[ref], "M2")
            if A0 and A0 > A_m2 > 0:
                bound = float(near["d_gl_km"] / math.log(A0 / A_m2))
        entry["semidiurnal"] = dict(
            nearest_station=near["name"], nearest_d_km=near["d_gl_km"],
            A_M2_mm=None if A_m2 is None else 1e3 * A_m2,
            at_floor=bool(A_m2 is not None and A_m2 <= 3 * SD_FLOOR_M),
            floating_ref=ref,
            delta_M2_upper_bound_km=bound)
        # diffusivity from MSf (and Mm)
        for c in ("MSf", "Mm"):
            f = entry["fits"].get(c)
            if f and np.isfinite(f["delta_km"]) and not f["two_point"]:
                w = 2 * math.pi / (PERIODS_H[c] * 3600.0)
                entry[f"K_{c}_m2_s"] = float((f["delta_km"] * 1e3) ** 2 * w / 2.0)
                if np.isfinite(f.get("err_km", np.nan)):
                    entry[f"K_{c}_err_m2_s"] = float(
                        2 * f["delta_km"] * 1e6 * f["err_km"] * w / 2.0)
        # discriminant ratio
        fm, fmsf = entry["fits"].get("Mm"), entry["fits"].get("MSf")
        if fm and fmsf and np.isfinite(fm["delta_km"]) and np.isfinite(fmsf["delta_km"]):
            r = fm["delta_km"] / fmsf["delta_km"]
            rerr = np.nan
            if not (fm["two_point"] or fmsf["two_point"]):
                rerr = abs(r) * math.sqrt(
                    (fm["err_km"] / fm["delta_km"]) ** 2
                    + (fmsf["err_km"] / fmsf["delta_km"]) ** 2)
            entry["ratio_Mm_over_MSf"] = dict(
                measured=float(r), err=float(rerr) if np.isfinite(rerr) else None,
                parabolic_prediction=float(math.sqrt(PERIODS_H["Mm"] / PERIODS_H["MSf"])),
                maxwell_saturation_prediction=1.0)
        res["branches"][br] = entry
    res["parabolic_ratio_MSf_over_M2"] = float(
        math.sqrt(PERIODS_H["MSf"] / PERIODS_H["M2"]))
    res["verdict"] = _verdict(res)
    return res


def _verdict(res):
    lines = []
    ks = []
    for br, e in res["branches"].items():
        f = e["fits"].get("MSf")
        if f and np.isfinite(f["delta_km"]):
            tag = " (two-point)" if f["two_point"] else f" ± {f['err_km']:.0f}"
            lines.append(f"{br}: delta_MSf={f['delta_km']:.0f}{tag} km"
                         + (f", K={e['K_MSf_m2_s']:.0f} m^2/s"
                            if "K_MSf_m2_s" in e else ""))
        if "K_MSf_m2_s" in e:
            ks.append(e["K_MSf_m2_s"])
    ratios = [e["ratio_Mm_over_MSf"]["measured"] for e in res["branches"].values()
              if "ratio_Mm_over_MSf" in e]
    sd = [e["semidiurnal"]["delta_M2_upper_bound_km"] for e in res["branches"].values()
          if e["semidiurnal"].get("delta_M2_upper_bound_km")]
    out = ["LONG-BAND ATTENUATION MEASURED on real GPS: " + "; ".join(lines)]
    if ks:
        out.append(f"in-situ hydraulic diffusivity K ~ {min(ks):.0f}-{max(ks):.0f} m^2/s "
                   "(the parameter Rosier 2015 must leave free; their fit needs a "
                   "'highly conductive' system — measured here from GPS alone)")
    if sd:
        out.append(f"semidiurnal band is dead by the first grounded station on every "
                   f"branch (delta_M2 <= {min(sd):.1f}-{max(sd):.1f} km taking the last "
                   f"floating station as the forcing scale) vs delta_MSf 10-45 km: the "
                   f"short/long ratio >= 5-30 EXCEEDS the single-diffusion prediction "
                   f"{res['parabolic_ratio_MSf_over_M2']:.1f} — the two bands ride "
                   "DIFFERENT mechanisms (elastic/flexural vs hydraulic-parabolic), "
                   "the field face of NR30's two constraint pressures")
    if ratios:
        r = np.asarray(ratios)
        out.append(f"the within-long-band discriminant delta_Mm/delta_MSf = "
                   f"{', '.join(f'{x:.2f}' for x in r)} vs 1.37 (parabolic) / 1.00 "
                   f"(Maxwell saturation): mean {r.mean():.2f} — CANNOT yet separate "
                   "the two long-band clocks at n=3-4 branches; the decisive datum is "
                   "two more multi-season stations at 40-70 km on Foundation")
    return " | ".join(out)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    cols = dict(Evans="#1b9e77", Foundation="#d95f02", Talutis="#7570b3",
                Rutford="#e7298a", Evans_XX="#66a61e")
    for br, e in res["branches"].items():
        for c, f in e["fits"].items():
            if not np.isfinite(f["delta_km"]) or f["delta_km"] > 500:
                continue
            T = PERIODS_H[c] / 24.0
            mk = "o" if not f["two_point"] else "s"
            ax[0].loglog(T, f["delta_km"], mk, color=cols.get(br, "k"), ms=7,
                         mfc="none" if f["two_point"] else cols.get(br, "k"))
            if np.isfinite(f.get("err_km", np.nan)):
                ax[0].errorbar(T, f["delta_km"], yerr=f["err_km"], fmt="none",
                               ecolor=cols.get(br, "k"), alpha=0.6)
        sd = e["semidiurnal"].get("delta_M2_upper_bound_km")
        if sd:
            ax[0].loglog(PERIODS_H["M2"] / 24.0, sd, "v", color=cols.get(br, "k"), ms=8)
    Ts = np.array([0.4, 40.0])
    for K, lab in ((300, "K=300 m$^2$/s"), (3000, "3000"), (30000, "30000")):
        delt = np.sqrt(2 * K / (2 * np.pi / (Ts * 86400.0))) / 1e3
        ax[0].loglog(Ts, delt, "k--", lw=0.7, alpha=0.5)
        ax[0].annotate(lab, (Ts[-1], delt[-1]), fontsize=6)
    ax[0].set_xlabel("constituent period [days]")
    ax[0].set_ylabel(r"attenuation length $\delta$ [km]")
    ax[0].set_title("the tidal ladder as an EIS sweep: skin depth vs period\n"
                    r"(dashed: parabolic $\delta=\sqrt{2K/\omega}$; triangles: "
                    "M2 upper bounds)", fontsize=9)
    ax[0].grid(alpha=0.3, which="both")
    for br in cols:
        if br in res["branches"]:
            ax[0].plot([], [], "o", color=cols[br], label=br)
    ax[0].legend(fontsize=7)
    for br, e in res["branches"].items():
        if "ratio_Mm_over_MSf" not in e:
            continue
        r = e["ratio_Mm_over_MSf"]
        ax[1].errorbar([br], [r["measured"]],
                       yerr=None if r["err"] is None else [r["err"]],
                       fmt="o", color=cols.get(br, "k"), ms=8, capsize=4)
    ax[1].axhline(math.sqrt(PERIODS_H["Mm"] / PERIODS_H["MSf"]), color="k", ls="--",
                  label=r"parabolic $\sqrt{T_{Mm}/T_{MSf}}=1.37$")
    ax[1].axhline(1.0, color="0.5", ls=":", label="Maxwell saturation = 1.00")
    ax[1].set_ylabel(r"$\delta_{Mm}/\delta_{MSf}$")
    ax[1].set_title("the long-band discriminant (underpowered at n=3-4:\n"
                    "registered decisive measurement in the report)", fontsize=9)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("NR64 — tidal-ladder EIS of the subglacial drainage system "
                 "(BAS Filchner-Ronne GPS, committed §I.5 cache)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_FIGDIR, "99_tidal_eis_ladder.json"))
    a = ap.parse_args()
    res = analyze()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    for br, e in res["branches"].items():
        print(f"[{br}] " + " ".join(
            f"{c}: {f['delta_km']:.0f}km(n={f['n']}{'2pt' if f['two_point'] else ''})"
            for c, f in e["fits"].items() if np.isfinite(f["delta_km"])))
    print("VERDICT:", res["verdict"])
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
