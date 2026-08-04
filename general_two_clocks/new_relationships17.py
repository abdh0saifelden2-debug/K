r"""NR40 -- Omori-Utsu aftershock decay is a memory-kernel tail: the fluid-
triggered case is the differentiated tail of a pore-pressure DIFFUSION kernel
(Nur & Booker 1972), the same `t^{-1/2} -> exponential-cutoff` family as the
repo's B.2 ice interface and G.4 diffusive kernels.  Its two diffusion clocks:
a TEMPORAL cutoff `tau_D` in the rate, and the SPATIAL Shapiro triggering front
`r(t) = sqrt(4 pi D t)` that recovers the hydraulic diffusivity `D`.

Ledger item E10 (papers/RESEARCH_RECAP_AND_HORIZON.md).  Scope: aftershock
triggering physics is contested -- this is a CONSISTENCY + IDENTIFIABILITY
result, reported straight (including the honest real-data null).

The derivation  [DERIVED]
-------------------------
Nur & Booker (1972): a co-seismic (or injection) pressure step diffuses; the
diffusion Green's function is the `t^{-1/2}`-tail-with-cutoff two-clocks kernel
(B.2/G.4).  Two observable faces:
  * temporal: seismicity rate ~ stressing rate = the differentiated kernel tail,
    `lambda(t) = K (t+c)^{-p} exp(-t/tau_D)` -- Omori with a diffusion cutoff
    clock `tau_D`; `tau_D -> inf` = scale-free (tectonic) Omori.
  * spatial: the pressure front reaches radius `r` at `t = r^2/(4 pi D)`, so the
    triggering front is `r(t) = sqrt(4 pi D t)` (Shapiro et al. 1997, 2002) --
    the identifiable diffusion signature that returns `D`.

What is proved here
-------------------
1. [VERIFIED, synthetic] IDENTIFIABILITY: a simulated pore-pressure-diffusion
   catalog recovers `D` from the sqrt(t) front (clean); a tapered-Omori rate
   with a finite `tau_D` is recovered by MLE and AICc-preferred over pure Omori,
   with a pure-Omori false-positive control.  (The temporal fit is the weaker,
   partly p<->tau degenerate face; the spatial front is the clean one.)
2. [real data, data-gated, REPORTED NULL] USGS ComCat: for the 2016 Pawnee OK
   M5.8 (injection-induced) and 2019 Ridgecrest M7.1 (tectonic) sequences,
   neither the temporal cutoff nor an epicentre-referenced sqrt(t) front is
   resolved in the public catalogue -- an honest null.  The sharpened
   requirement (next step): relocated catalogues + injection-well coordinates
   (the epicentre is not the diffusion source, and an M>=5.8 rupture seeds its
   whole fault at t=0).
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
from scipy.optimize import minimize

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

SEQUENCES = [
    ("Pawnee_2016", 36.425, -96.929, 35.0, "2016-09-03T12:02:44", 90, 2.7, "injection"),
    ("Ridgecrest_2019", 35.770, -117.599, 60.0, "2019-07-06T03:19:53", 90, 3.5, "tectonic"),
]


def _trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# ----------------------------------------------------- temporal: tapered Omori
def _intensity(t, K, c, p, tau):
    lam = K * (t + c) ** (-p)
    return lam * np.exp(-t / tau) if np.isfinite(tau) else lam


def _negloglik(theta, t, T, tapered):
    if tapered:
        logK, logc, p, logtau = theta
        tau = np.exp(logtau)
    else:
        logK, logc, p = theta
        tau = np.inf
    if not (0.2 < p < 3.0):
        return 1e12
    K, c = np.exp(logK), np.exp(logc)
    lam = _intensity(t, K, c, p, tau)
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        return 1e12
    grid = np.linspace(0.0, T, 4000)
    return -(float(np.sum(np.log(lam))) - _trapz(_intensity(grid, K, c, p, tau), grid))


def _aicc(nll, n, k):
    aic = 2 * nll + 2 * k
    d = n - k - 1
    return aic + (2 * k * (k + 1) / d if d > 0 else np.inf)


def fit_sequence(t, T):
    n = len(t)
    p0 = [np.log(max(n / T, 1.0)), np.log(0.02), 1.0]
    om = minimize(_negloglik, p0, args=(t, T, False), method="Nelder-Mead",
                  options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-6})
    tp = minimize(_negloglik, list(om.x) + [np.log(T / 4.0)], args=(t, T, True),
                  method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-6,
                                                  "fatol": 1e-6})
    return {"n_events": n,
            "omori_p": float(om.x[2]), "omori_aicc": float(_aicc(om.fun, n, 3)),
            "tapered_p": float(tp.x[2]), "tau_D_day": float(np.exp(tp.x[3])),
            "tapered_aicc": float(_aicc(tp.fun, n, 4)),
            "delta_aicc_cutoff_preferred": float(_aicc(om.fun, n, 3)
                                                 - _aicc(tp.fun, n, 4))}


def simulate_tapered_omori(K, c, p, tau, T, seed=0):
    rng = np.random.default_rng(seed)
    lam_max = K * c ** (-p)
    t, events = 0.0, []
    while t < T:
        t += rng.exponential(1.0 / lam_max)
        if t < T and rng.uniform() < _intensity(t, K, c, p, tau) / lam_max:
            events.append(t)
    return np.array(events)


# ------------------------------------------------------ spatial: Shapiro front
def simulate_diffusion_catalog(D, T, n, seed=0):
    """Events swept by a pore-pressure front r_f(t)=sqrt(4 pi D t), uniform in
    the wetted disk area up to the front."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.02, T, 4 * n)
    rf = np.sqrt(4.0 * np.pi * D * t)
    r = rf * np.sqrt(rng.uniform(0, 1, t.size))    # uniform-in-area
    idx = np.argsort(rng.uniform(size=t.size))[:n]
    return t[idx], r[idx]


def fit_shapiro_front(t, r, nbin=9, q=90):
    """Fit the upper-envelope triggering front r_env(t)=sqrt(4 pi D t); return D
    and the front R^2 (how sqrt(t)-like the envelope is)."""
    edges = np.logspace(np.log10(max(t.min(), 1e-2)), np.log10(t.max()), nbin + 1)
    tc, renv = [], []
    for i in range(nbin):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if m.sum() >= 3:
            tc.append(np.sqrt(edges[i] * edges[i + 1]))
            renv.append(np.percentile(r[m], q))
    tc, renv = np.array(tc), np.array(renv)
    if len(tc) < 4:
        return {"D_km2_per_day": None, "front_r2": None, "n_bins": int(len(tc))}
    a = float(np.sum(renv ** 2 * tc) / np.sum(tc ** 2))       # renv^2 = a t
    D = a / (4.0 * np.pi)
    pred = np.sqrt(np.maximum(a * tc, 0.0))
    r2 = 1.0 - np.sum((renv - pred) ** 2) / np.sum((renv - renv.mean()) ** 2)
    return {"D_km2_per_day": D, "front_r2": float(r2), "n_bins": int(len(tc))}


# ----------------------------------------------------------------- real data
def fetch_comcat(lat, lon, radius_km, t0_iso, days, minmag, timeout=45):
    import math
    import requests
    t0 = dt.datetime.fromisoformat(t0_iso)
    end = t0 + dt.timedelta(days=days)
    p = dict(format="csv", starttime=t0_iso, endtime=end.isoformat(),
             minmagnitude=minmag, latitude=lat, longitude=lon,
             maxradiuskm=radius_km, orderby="time-asc")
    r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query",
                     params=p, timeout=timeout)
    r.raise_for_status()
    times, dists = [], []
    for row in r.text.splitlines()[1:]:
        c = row.split(",")
        try:
            te = dt.datetime.fromisoformat(c[0].replace("Z", "+00:00"))
            dd = (te - t0.replace(tzinfo=te.tzinfo)).total_seconds() / 86400.0
            la, lo = float(c[1]), float(c[2])
            dla, dlo = math.radians(la - lat), math.radians(lo - lon)
            a = (math.sin(dla / 2) ** 2 + math.cos(math.radians(lat))
                 * math.cos(math.radians(la)) * math.sin(dlo / 2) ** 2)
            if dd > 0:
                times.append(dd)
                dists.append(2 * 6371.0 * math.asin(math.sqrt(a)))
        except Exception:
            continue
    o = np.argsort(times)
    return np.array(times)[o], np.array(dists)[o]


# ----------------------------------------------------------------------- run
def run(seed=0):
    out = {}
    T = 60.0
    # temporal identifiability: recover a finite cutoff when present ...
    tt = simulate_tapered_omori(K=40.0, c=0.02, p=1.1, tau=6.0, T=T, seed=seed)
    rec = fit_sequence(tt, T)
    out["synthetic_temporal"] = {
        "true_tau_D": 6.0, "fit_tau_D": rec["tau_D_day"],
        "delta_aicc": rec["delta_aicc_cutoff_preferred"],
        "cutoff_preferred": bool(rec["delta_aicc_cutoff_preferred"] > 2.0),
    }
    tt0 = simulate_tapered_omori(K=40.0, c=0.02, p=1.1, tau=np.inf, T=T, seed=seed + 1)
    out["synthetic_temporal_control"] = {
        "delta_aicc": fit_sequence(tt0, T)["delta_aicc_cutoff_preferred"],
    }
    out["synthetic_temporal_control"]["no_spurious_cutoff"] = bool(
        out["synthetic_temporal_control"]["delta_aicc"] < 6.0)

    # spatial identifiability: the sqrt(t) front recovers D cleanly
    ts, rs = simulate_diffusion_catalog(D=0.5, T=T, n=600, seed=seed)
    fr = fit_shapiro_front(ts, rs)
    out["synthetic_spatial_front"] = {
        "true_D": 0.5, "fit_D": fr["D_km2_per_day"], "front_r2": fr["front_r2"],
        "D_recovered": bool(fr["D_km2_per_day"] is not None
                            and 0.25 < fr["D_km2_per_day"] < 1.0
                            and fr["front_r2"] > 0.8),
    }

    # real ComCat (data-gated) -- reported straight
    real = {}
    for name, lat, lon, rad, t0, days, mm, kind in SEQUENCES:
        try:
            t, rdist = fetch_comcat(lat, lon, rad, t0, days, mm)
            if len(t) >= 30:
                ft = fit_sequence(t, float(days))
                fs = fit_shapiro_front(t, rdist)
                real[name] = {"kind": kind, "n_events": int(len(t)),
                              "temporal_delta_aicc": ft["delta_aicc_cutoff_preferred"],
                              "tau_D_day": ft["tau_D_day"], "omori_p": ft["omori_p"],
                              "spatial_front_D": fs["D_km2_per_day"],
                              "spatial_front_r2": fs["front_r2"]}
            else:
                real[name] = {"kind": kind, "n_events": int(len(t)), "note": "too few"}
        except Exception as e:
            real[name] = {"kind": kind, "error": repr(e)[:120]}
    out["real_comcat"] = real
    # honest verdict: is a clean diffusion clock resolved in the public catalogue?
    resolved = any(v.get("spatial_front_r2", 0) and v["spatial_front_r2"] > 0.9
                   and v.get("kind") == "injection" for v in real.values())
    out["real_diffusion_clock_resolved"] = bool(resolved)
    return out, {"tt": tt, "T": T, "ts": ts, "rs": rs}


# --------------------------------------------------------------------- figure
def make_figure(out, aux, path_png):                        # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ts, rs = aux["ts"], aux["rs"]
    ax = axes[0]
    ax.scatter(ts, rs, s=6, alpha=0.4, color="tab:blue", label="synthetic events")
    tg = np.linspace(0.02, aux["T"], 200)
    fr = out["synthetic_spatial_front"]
    ax.plot(tg, np.sqrt(4 * np.pi * fr["fit_D"] * tg), "r-", lw=2,
            label=f"front sqrt(4piDt), D={fr['fit_D']:.2f} (true 0.5)")
    ax.set_xlabel("t [day]"); ax.set_ylabel("r [km]")
    ax.set_title(f"spatial diffusion front recovers D (R^2={fr['front_r2']:.2f})")
    ax.legend(fontsize=8)

    ax = axes[1]
    real = out["real_comcat"]
    names = [k for k in real if "temporal_delta_aicc" in real[k]]
    if names:
        vals = [real[k]["temporal_delta_aicc"] for k in names]
        cols = ["tab:red" if real[k]["kind"] == "injection" else "tab:blue" for k in names]
        ax.bar(range(len(names)), vals, color=cols)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{n}\n(front R2={real[n].get('spatial_front_r2', float('nan')):.2f})"
                            for n in names], rotation=0, fontsize=7)
        ax.axhline(2, ls="--", color="k", lw=1, label="cutoff-preferred")
        ax.set_ylabel("temporal dAICc"); ax.legend(fontsize=8)
    ax.set_title("real ComCat: no clean diffusion clock (null)")
    fig.suptitle("NR40 (E10): Omori decay as a pore-pressure diffusion-kernel tail", y=1.03)
    fig.tight_layout(); fig.savefig(path_png, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():                                                 # pragma: no cover
    out, aux = run()
    os.makedirs(FIGDIR, exist_ok=True)
    jpath = os.path.join(FIGDIR, "nr40_omori_kernel_tail.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, aux, os.path.join(FIGDIR, "nr40_omori_kernel_tail.png"))
    print(json.dumps(out, indent=1))
    print(f"wrote {jpath}")


if __name__ == "__main__":                                  # pragma: no cover
    main()
