r"""NR69 (theory + reference-atmosphere data) — the atmosphere is a LADDER of clock
crossovers: one rising molecular diffusivity D(z) ∝ 1/n against two ceilings — the
eddy ceiling K_zz (turbopause) and the ballistic ceiling v̄H/3 (exobase) — with the
Jeans/blow-off discriminant λ_c ≈ 2γ as the top rung. The rung ORDER is forced by
the atmosphere being subsonically stirred (K_zz ≪ v̄H/3), and the exobase is the
altitude where ANY local (Markovian) closure dies — Paper 1's "what closure
discards" become "where closure dies".

Where this sits
===============
NR68 measured rung 1: the turbopause as the flow-clock/molecular-clock handoff
(H_He/H_N2: 1 → 28/4 on the committed NRLMSIS 2.1 cache). NR69 shows the SAME
structure repeats twice more overhead, and that all three mainstream boundaries —
turbopause, exobase, escape regime — are crossings of one clock pair read at three
different levels:

  rung 1  turbopause   K_zz = D_i          flow clock loses composition to the
                                           molecular clock (NR68)
  rung 2  exobase      l = H  (Kn = 1)     collision clock loses to the transit
                                           clock: the fluid/closure description dies
  rung 3  escape       λ = v_esc²/v_p²     gravity clock loses the light tail:
                       vs λ_c ≈ 2γ         Jeans leak (λ > λ_c) or blow-off (λ < λ_c)

Mainstream anchors: the exobase is DEFINED by mean free path = scale height, i.e.
Knudsen number Kn ≈ 1 (Chamberlain 1963; Bauer & Lammer 2004; Lammer et al. 2022
review); the hydrodynamic/Jeans transition is sharp at λ_c = 2γ ≈ 2.8 (diatomic) –
3.3 (monatomic) (Gruzinov 2011; Volkov et al. 2011 ApJL); K_zz at the turbopause is
100–1000 m²/s (Colegrove et al. 1965: 4×10⁶ cm²/s; Kelley et al. 2003: 250–1000;
Vlasov & Kelley 2014 infer K_zz from exactly the MSIS He/Ar/N2 profiles NR68 reads).

The one-curve/two-ceilings theorem
==================================
Hard-sphere kinetics: mean free path l = 1/(√2 n σ), thermal speed
v̄ = √(8kT/πm̄), molecular diffusivity D ≈ (1/3) v̄ l. Then EXACTLY (same v̄):

    Kn(z) = l/H = 3 D(z) / (v̄ H)   and   Kn = τ_coll / τ_transit,

with τ_coll = l/v̄ the collision (memory) time and τ_transit = H/v̄ the transit
time across one gradient length. So:

  * the turbopause is where the rising D(z) crosses the EDDY ceiling K_zz;
  * the exobase is where D(z) crosses the BALLISTIC ceiling v̄H/3 (i.e. Kn = 1);
  * z_turbo < z_exo is FORCED iff K_zz < v̄H/3 — the subsonic-stirring condition.
    Measured: v̄H/3 ~ 10⁵–10⁶ m²/s up high vs K_zz ~ 10²–10³ m²/s → the ceilings
    are 2.5–4 decades apart; the ladder order is not contingent, it is a theorem
    about subsonically-stirred atmospheres.

Markovianizability (the Paper-1 reading)
========================================
Kn = τ_coll/τ_transit is the memory-time/system-time ratio of the GLE: below the
exobase the collision kernel is short-lived (Kn ≪ 1) and a LOCAL closure exists —
scalar-blind eddy diffusion below rung 1, scalar-selective molecular diffusion
between rungs 1 and 2. At rung 2 the kernel support reaches the system's own
evolution time: no local closure of any kind survives, and transport goes exact-
ballistic (collisionless streaming — no dissipation left to model). Rung 3 then
asks what ballistic motion does against gravity: λ > λ_c → molecule-by-molecule
Jeans leak of the light tail; λ < λ_c → bulk hydrodynamic blow-off. Earth sits in
the Jeans regime for every species including H (λ_H ≈ 9 at the exobase), and the
critical BLOW-OFF temperature this implies for H, T_c = m_H g r/(λ_c k) ≈ 2400 K,
is ~3× the actual exospheric temperature — why Earth keeps its water on Gyr
timescales while losing ~3 kg/s of hydrogen (classical Jeans ordering, NR68).

What is measured here (committed cache, offline)
================================================
From the NR68 NRLMSIS 2.1 cache (4 real-driver conditions, 80–600 km):
  1. l(z) = 1/(√2 n_tot σ) with σ = 3×10⁻¹⁹ m² (generic hard-sphere; Chamberlain &
     Hunten 1987). Kn(z) = l/H with H = kT/(m̄(z) g(z)), m̄ = ρ/n_tot.
  2. z_exo: the Kn = 1 crossing — lands in the mainstream 450–600 km band.
  3. z_turbo(D = K_zz) for K_zz ∈ {100, 400, 1000} m²/s — an INDEPENDENT
     turbopause estimate, to compare with NR68's composition-discriminant band.
  4. the ceiling ratio (v̄H/3)/K_zz at z_turbo — the subsonic-forcing margin.
  5. λ_i(z_exo) for H, He, O, N2 and the blow-off critical temperature per species.

Honest scope
------------
σ is a single generic cross-section: a factor-2 change shifts z_exo by ~H ln2 ≈
one scale height (~40 km) — the exobase is a BAND, as in the literature (350–700 km
over the solar cycle). D uses the (1/3)v̄l hard-sphere estimate with the mean mass,
adequate for a rung locator but NOT a species-resolved Chapman–Enskog D_i — rung 1
precision belongs to NR68's composition discriminant, which needs no σ at all.
Kinetic corrections make true escape ≈ 2–2.5× the Jeans rate near Kn ~ 0.2 (Volkov
2017); nothing here uses the absolute rate, only orderings and crossings.
NRLMSIS 2.1 is the mainstream empirical reference atmosphere, not a raw instrument
record. λ_c is bracketed 2.8–3.3 (γ-dependent); Earth clears it by a factor ≥ 2.7
in λ so the bracket width is immaterial.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from new_relationships45 import (CACHE, COL, K_B, M_U, MASS, g_of_z, load_cache,
                                 _profile, jeans_parameter)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "104_clock_ladder.json")

SIGMA_HS = 3.0e-19          # generic hard-sphere collision cross-section [m^2]
KZZ_REF = (100.0, 400.0, 1000.0)   # eddy ceiling band [m^2/s]: Kelley/Colegrove
LAMBDA_C = (2.8, 3.3)       # blow-off/Jeans transition (Gruzinov 2011; Volkov 2011)
MAJORS = ("N2", "O2", "O", "He", "H", "Ar")


def number_density_total(arr):
    """Total number density from the major species columns [m^-3]."""
    cols = [COL[s] for s in MAJORS]
    return np.nansum(arr[:, cols], axis=1)


def mean_mass_kg(arr):
    """Mean molecular mass rho/n_tot [kg]."""
    return arr[:, COL["rho"]] / number_density_total(arr)


def mean_free_path(n_tot, sigma=SIGMA_HS):
    """Hard-sphere l = 1/(sqrt(2) n sigma) [m]."""
    return 1.0 / (math.sqrt(2.0) * n_tot * sigma)


def thermal_speed(T, m_kg):
    """Mean speed sqrt(8kT/(pi m)) [m/s]."""
    return np.sqrt(8.0 * K_B * np.asarray(T) / (math.pi * m_kg))


def ladder_profiles(alts_km, arr, sigma=SIGMA_HS):
    """All z-profiles the ladder needs."""
    z = np.asarray(alts_km) * 1e3
    T = arr[:, COL["T"]]
    n = number_density_total(arr)
    m = mean_mass_kg(arr)
    g = g_of_z(z)
    H = K_B * T / (m * g)                       # local mean scale height [m]
    l = mean_free_path(n, sigma)                # mean free path [m]
    v = thermal_speed(T, m)                     # mean thermal speed [m/s]
    D = (v * l) / 3.0                           # molecular diffusivity [m^2/s]
    ballistic = (v * H) / 3.0                   # ballistic ceiling [m^2/s]
    Kn = l / H                                  # Knudsen number
    tau_coll = l / v
    tau_transit = H / v
    return dict(alt_km=np.asarray(alts_km), T=T, n=n, m_kg=m, H_m=H, l_m=l,
                v_ms=v, D=D, ballistic=ballistic, Kn=Kn,
                tau_coll=tau_coll, tau_transit=tau_transit)


def _first_upcross(alts_km, y, threshold):
    """Lowest altitude where y crosses ABOVE threshold (log-interp)."""
    y = np.asarray(y)
    for i in range(1, len(alts_km)):
        if y[i - 1] < threshold <= y[i]:
            f = (math.log(threshold) - math.log(y[i - 1])) / \
                (math.log(y[i]) - math.log(y[i - 1]))
            return float(alts_km[i - 1] + f * (alts_km[i] - alts_km[i - 1]))
    return None


def analyze_condition(alts_km, arr, sigma=SIGMA_HS):
    p = ladder_profiles(alts_km, arr, sigma)
    alts = p["alt_km"]
    # rung 2: exobase Kn = 1
    z_exo = _first_upcross(alts, p["Kn"], 1.0)
    i_exo = int(np.argmin(np.abs(alts - z_exo)))
    # rung 1: D = K_zz for the reference eddy band
    z_turbo = {f"Kzz={int(k)}": _first_upcross(alts, p["D"], k) for k in KZZ_REF}
    # subsonic forcing margin at the middle turbopause
    z_t = z_turbo["Kzz=400"]
    i_t = int(np.argmin(np.abs(alts - z_t)))
    ceiling_ratio = float(p["ballistic"][i_t] / 400.0)
    # rung 3: Jeans parameters and blow-off critical temperatures at the exobase
    T_exo = float(p["T"][i_exo])
    lam = {s: float(jeans_parameter(alts[i_exo], T_exo, MASS[s]))
           for s in ("H", "He", "O", "N2")}
    r_exo = 6.371e6 + alts[i_exo] * 1e3
    g_exo = g_of_z(alts[i_exo] * 1e3)
    T_blowoff = {s: {f"lc={lc}": float(MASS[s] * M_U * g_exo * r_exo / (lc * K_B))
                     for lc in LAMBDA_C} for s in ("H", "He")}
    # exact identity check: Kn == 3 D/(v H) and Kn == tau_coll/tau_transit
    ident1 = float(np.max(np.abs(p["Kn"] - 3.0 * p["D"] / (p["v_ms"] * p["H_m"]))))
    ident2 = float(np.max(np.abs(p["Kn"] - p["tau_coll"] / p["tau_transit"])))
    return dict(
        z_exobase_km=z_exo, T_exobase_K=T_exo,
        z_turbopause_km=z_turbo, ceiling_ratio_at_turbo=ceiling_ratio,
        jeans_at_exobase=lam, T_blowoff_K=T_blowoff,
        identity_residuals=dict(Kn_vs_3D_over_vH=ident1,
                                Kn_vs_tau_ratio=ident2),
        profile=dict(alt_km=alts.tolist(),
                     Kn=p["Kn"].tolist(), D=p["D"].tolist(),
                     ballistic=p["ballistic"].tolist()))


def analyze(sigma=SIGMA_HS):
    cache = load_cache()
    alts = np.array(cache["_alts_km"], float)
    res = {"sigma_m2": sigma, "Kzz_ref_m2s": list(KZZ_REF),
           "lambda_c": list(LAMBDA_C), "conditions": {}}
    for name, cond in cache["conditions"].items():
        res["conditions"][name] = analyze_condition(alts, _profile(cond), sigma)
    res["verdict"] = _verdict(res)
    return res


def _verdict(res):
    lines = []
    for name, e in res["conditions"].items():
        zt = e["z_turbopause_km"]
        lines.append(
            f"{name}: rung1 D=Kzz at {zt['Kzz=100']:.0f}-{zt['Kzz=1000']:.0f} km"
            f" (400: {zt['Kzz=400']:.0f}) < rung2 exobase Kn=1 at "
            f"{e['z_exobase_km']:.0f} km (T={e['T_exobase_K']:.0f} K), "
            f"ceilings {e['ceiling_ratio_at_turbo']:.0f}x apart")
    first = next(iter(res["conditions"].values()))
    j = first["jeans_at_exobase"]
    tb = first["T_blowoff_K"]["H"]
    lines.append(
        f"rung3 at exobase: lam_H={j['H']:.1f} > lambda_c={res['lambda_c'][0]}"
        f" (Jeans regime, no blow-off); H blow-off would need T >"
        f" {min(tb.values()):.0f} K vs actual {first['T_exobase_K']:.0f} K")
    return " | ".join(lines)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    e = res["conditions"]["midlat_equinox"]
    prof = e["profile"]
    alt = np.array(prof["alt_km"])
    D = np.array(prof["D"]); bal = np.array(prof["ballistic"])
    ax[0].semilogx(D, alt, lw=1.8, label=r"molecular $D(z)=\bar v\,l/3$ (rising)")
    ax[0].semilogx(bal, alt, lw=1.4, ls="-.", color="tab:purple",
                   label=r"ballistic ceiling $\bar v H/3$")
    ax[0].axvspan(KZZ_REF[0], KZZ_REF[-1], color="tab:orange", alpha=0.25,
                  label=r"eddy ceiling $K_{zz}$ (100-1000 m$^2$/s)")
    zt = e["z_turbopause_km"]; ze = e["z_exobase_km"]
    ax[0].axhline(zt["Kzz=400"], color="tab:orange", ls=":", lw=1)
    ax[0].axhline(ze, color="tab:purple", ls=":", lw=1)
    ax[0].annotate(f"rung 1: turbopause {zt['Kzz=400']:.0f} km",
                   (2e2, zt["Kzz=400"] + 8), fontsize=9, color="tab:orange")
    ax[0].annotate(f"rung 2: exobase {ze:.0f} km (Kn=1)",
                   (2e-2, ze + 8), fontsize=9, color="tab:purple")
    ax[0].set_xlabel(r"diffusivity [m$^2$/s]")
    ax[0].set_ylabel("altitude [km]")
    ax[0].set_title("one rising diffusivity, two ceilings:\n"
                    "the ladder order is forced by $K_{zz}\\ll \\bar v H/3$",
                    fontsize=10)
    ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8, loc="lower left")
    # panel b: rung 3 — Jeans parameter per species at the exobase, vs lambda_c
    species = ("H", "He", "O", "N2")
    names = list(res["conditions"])
    x = np.arange(len(species))
    for k, nm in enumerate(names):
        lam = [res["conditions"][nm]["jeans_at_exobase"][s] for s in species]
        ax[1].bar(x + 0.2 * k - 0.3, lam, width=0.18, label=nm)
    ax[1].axhspan(LAMBDA_C[0], LAMBDA_C[1], color="r", alpha=0.3,
                  label=r"blow-off threshold $\lambda_c=2\gamma$")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x); ax[1].set_xticklabels(species)
    ax[1].set_ylabel(r"Jeans parameter $\lambda$ at the exobase")
    ax[1].set_title("rung 3: every species sits ABOVE the blow-off line —\n"
                    "Earth leaks molecule-by-molecule (Jeans), never in bulk",
                    fontsize=10)
    ax[1].grid(alpha=0.3, axis="y"); ax[1].legend(fontsize=7)
    fig.suptitle("NR69 — the atmosphere is a ladder of clock crossovers "
                 "(NRLMSIS 2.1)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=130)
    plt.close(fig)
    print(f"figure -> {os.path.splitext(path)[0]}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FIG)
    a = ap.parse_args()
    res = analyze()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    slim = {k: v for k, v in res.items() if k != "conditions"}
    slim["conditions"] = {n: {kk: vv for kk, vv in e.items() if kk != "profile"}
                          for n, e in res["conditions"].items()}
    with open(a.out, "w") as fh:
        json.dump(slim, fh, indent=2)
    print("VERDICT:", res["verdict"])
    make_figure(res, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
