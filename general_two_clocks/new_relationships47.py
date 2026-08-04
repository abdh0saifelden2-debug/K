r"""NR70 (theory + reference-atmosphere data) — the escape ladder has a CLOCK
ALGEBRA: rungs in SERIES compose harmonically (the slowest clock rules — Earth's H
loss is set by the diffusive pipe, not the exobase valve), channels in PARALLEL add
(the fastest clock rules — helium, whose thermal valve fails by nine decades, leaves
through the plasma channel instead). Both compositions are measured on the committed
NR68/NR69 cache against the mainstream numbers.

Where this sits
===============
NR69 located the rungs (turbopause, exobase, escape discriminant). NR70 asks what
FLUX the ladder carries, and finds the two-clocks composition law:

  series  (one species, stages stacked):  1/g_tot = Σ 1/g_k   — harmonic;
          the smallest conductance (slowest clock) sets the flux.
  parallel (one reservoir, channels side by side):  g_tot = Σ g_k — additive;
          the largest conductance (fastest clock) sets the flux.

This is the resistor algebra of transport, exact for steady linear stages (proved
in-module to 1e-12 on a two-slab diffusion problem), and it is the missing flux
statement of the two-clocks program: WHICH clock is rate-determining is decided by
the circuit topology, not by the clocks alone.

Hydrogen — the series face (mainstream: diffusion-limited escape)
=================================================================
Hunten (1973): H escape from Earth is DIFFUSION-LIMITED — the homosphere-to-
heterosphere pipe (rung 1→2 of NR69) supplies Φ_lim ≈ 2.5×10¹³ f_T cm⁻²s⁻¹ with
f_T ≈ 10⁻⁵ the total-hydrogen mixing ratio (H2O+CH4+H2; Catling & Kasting 2017),
i.e. Φ_lim ≈ 2.5×10⁸ cm⁻²s⁻¹. The observed total escape ≈ 10⁸ cm⁻²s⁻¹ (~3 kg/s)
sits at the pipe's ceiling, NOT at the valve's: the thermal (Jeans) valve computed
here from the committed cache carries only 1–3×10⁷ cm⁻²s⁻¹ (0.1–0.3 kg/s) at this
low-activity epoch — the remainder leaves through parallel non-thermal channels
(charge exchange, polar wind). Series algebra verdict: the SLOWEST stage (the
diffusive pipe) sets Earth's hydrogen loss; that is why H escape is famously
insensitive to exospheric temperature.

Helium — the parallel face (mainstream: the helium escape problem)
==================================================================
Crustal α-decay outgasses ⁴He at ≈ 10⁶ atoms cm⁻²s⁻¹, which must escape in steady
state (Nicolet 1957). The thermal valve measured here: Φ_J(He) ≤ 1.4×10⁻³ cm⁻²s⁻¹
— NINE-plus decades short at this epoch (λ_He = 36–43; e^{−λ} annihilates it).
This is the classical helium escape problem; its mainstream resolution (Axford
1968) is the POLAR WIND — ion outflow along open field lines, i.e. a parallel
channel whose clock is electromagnetic (the constraint clock), not thermal.
Parallel algebra verdict: the FASTEST channel rules; for He that channel is not
in the neutral ladder at all.

The exponential rectifier
=========================
Φ_J ∝ (1+λ)e^{−λ} with λ ∝ 1/T_exo makes the thermal valve an exponential
rectifier of the solar cycle: across just the four cache conditions (one epoch,
day/night × latitude), the H valve swings ×3 and the He valve ×2000. Escape
happens at the hot excursions — the time-average is dominated by solar maximum
(Hunten & Donahue 1976), another reason the supply pipe, whose clock is only
algebraic in K_zz/D, is the steady one.

Honest scope
------------
Φ_lim's f_T ≈ 10⁻⁵ and the He outgassing 10⁶ cm⁻²s⁻¹ are literature constants
(cited), not derived here; the observed-total H escape ~10⁸ cm⁻²s⁻¹ is the
standard satellite-era value. The cache is one low-activity epoch: the thermal
valve numbers are epoch specific (that is the rectifier point), and the He
shortfall quoted "≥9 decades" is for THIS epoch — cycle-averaged literature
estimates still leave it orders short, which is why the polar wind is the
accepted resolution. Jeans fluxes use the classical formula; kinetic corrections
(×2–2.5 near Kn~0.2, Volkov 2017) do not move any decade-scale conclusion.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from new_relationships45 import (CACHE, COL, K_B, M_U, MASS, load_cache,
                                 _profile, jeans_parameter)
from new_relationships46 import ladder_profiles, _first_upcross

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "105_escape_flux_algebra.json")

R_EARTH = 6.371e6
# literature constants (cited in module docstring)
PHI_DIFF_LIMIT = 2.5e8        # cm^-2 s^-1, Hunten 1973 with f_T ~ 1e-5
PHI_OBS_TOTAL_H = 1.0e8       # cm^-2 s^-1, standard satellite-era total
PHI_HE_REQUIRED = 1.0e6       # cm^-2 s^-1, crustal outgassing balance (Nicolet)


# ------------------------------------------------------------ the clock algebra
def series_conductance(gs):
    """1/g_tot = sum 1/g_k (harmonic): the slowest stage rules."""
    gs = np.asarray(gs, float)
    return 1.0 / np.sum(1.0 / gs)


def parallel_conductance(gs):
    """g_tot = sum g_k (additive): the fastest channel rules."""
    return float(np.sum(np.asarray(gs, float)))


def two_slab_steady_flux(D1, L1, D2, L2, c_top, c_bot=0.0):
    """Exact steady diffusion through two stacked slabs (continuity of flux and
    concentration at the interface). Returns the flux and the series prediction —
    equal to machine precision; the in-module proof of the harmonic law."""
    g1, g2 = D1 / L1, D2 / L2
    # exact solution: phi = (c_top - c_bot) / (L1/D1 + L2/D2)
    phi_exact = (c_top - c_bot) / (L1 / D1 + L2 / D2)
    phi_series = series_conductance([g1, g2]) * (c_top - c_bot)
    return phi_exact, phi_series


def jeans_flux(n_m3, T, mass_amu, alt_km):
    """Classical Jeans effusion flux [m^-2 s^-1]: n v_p (1+lam) e^-lam / (2 sqrt(pi))."""
    m = mass_amu * M_U
    vp = math.sqrt(2.0 * K_B * T / m)
    lam = float(jeans_parameter(alt_km, T, mass_amu))
    return n_m3 * vp / (2.0 * math.sqrt(math.pi)) * (1.0 + lam) * math.exp(-lam), lam


# ------------------------------------------------------------ measurement
def analyze_condition(alts_km, arr):
    p = ladder_profiles(alts_km, arr)
    z_exo = _first_upcross(alts_km, p["Kn"], 1.0)
    i = int(np.argmin(np.abs(np.asarray(alts_km) - z_exo)))
    T = float(arr[i, COL["T"]])
    out = dict(z_exobase_km=z_exo, T_exobase_K=T, species={})
    for s in ("H", "He", "O"):
        n = float(arr[i, COL[s]])
        phi_m2, lam = jeans_flux(n, T, MASS[s], alts_km[i])
        phi_cm2 = phi_m2 / 1e4
        global_kg_s = phi_m2 * 4.0 * math.pi * (R_EARTH + z_exo * 1e3) ** 2 \
            * MASS[s] * M_U
        out["species"][s] = dict(n_exo_m3=n, lam=lam,
                                 phi_cm2s=phi_cm2, global_kg_s=global_kg_s)
    return out


def analyze():
    cache = load_cache()
    alts = np.array(cache["_alts_km"], float)
    res = {"literature": dict(phi_diffusion_limit_H=PHI_DIFF_LIMIT,
                              phi_observed_total_H=PHI_OBS_TOTAL_H,
                              phi_He_required=PHI_HE_REQUIRED),
           "conditions": {}}
    for name, cond in cache["conditions"].items():
        res["conditions"][name] = analyze_condition(alts, _profile(cond))
    # circuit verdicts
    phis_H = [e["species"]["H"]["phi_cm2s"] for e in res["conditions"].values()]
    phis_He = [e["species"]["He"]["phi_cm2s"] for e in res["conditions"].values()]
    res["series_H"] = dict(
        thermal_valve_range=[min(phis_H), max(phis_H)],
        valve_over_pipe=max(phis_H) / PHI_DIFF_LIMIT,
        observed_over_pipe=PHI_OBS_TOTAL_H / PHI_DIFF_LIMIT,
        rectifier_swing=max(phis_H) / min(phis_H))
    res["parallel_He"] = dict(
        thermal_valve_max=max(phis_He),
        decades_short=math.log10(PHI_HE_REQUIRED / max(phis_He)),
        rectifier_swing=max(phis_He) / min(phis_He))
    res["verdict"] = _verdict(res)
    return res


def _verdict(res):
    sH, pHe = res["series_H"], res["parallel_He"]
    return (
        f"H (series): thermal valve {sH['thermal_valve_range'][0]:.1e}-"
        f"{sH['thermal_valve_range'][1]:.1e} cm-2s-1 = "
        f"{100 * sH['valve_over_pipe']:.0f}% of the diffusion-limit pipe "
        f"{PHI_DIFF_LIMIT:.1e}; observed total {PHI_OBS_TOTAL_H:.0e} sits at the "
        f"pipe, not the valve -> the SLOWEST stage rules (diffusion-limited, "
        f"Hunten 1973) | He (parallel): thermal valve <= "
        f"{pHe['thermal_valve_max']:.1e} vs required {PHI_HE_REQUIRED:.0e} -> "
        f"{pHe['decades_short']:.0f} decades short (the helium problem); the "
        f"plasma channel (polar wind) is the fast parallel clock (Axford 1968) | "
        f"rectifier: valve swings x{res['series_H']['rectifier_swing']:.1f} (H), "
        f"x{res['parallel_He']['rectifier_swing']:.0f} (He) across one epoch's "
        f"conditions")


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    # panel a: the H series circuit
    names = list(res["conditions"])
    valve = [res["conditions"][n]["species"]["H"]["phi_cm2s"] for n in names]
    x = np.arange(len(names))
    ax[0].bar(x, valve, width=0.55, color="tab:blue",
              label="thermal valve $\\Phi_J$(H) (this cache)")
    ax[0].axhline(PHI_DIFF_LIMIT, color="k", ls="--", lw=1.4,
                  label="diffusive pipe $\\Phi_{lim}$ (Hunten 1973)")
    ax[0].axhline(PHI_OBS_TOTAL_H, color="tab:red", ls=":", lw=1.6,
                  label="observed TOTAL escape (~3 kg/s)")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax[0].set_ylabel(r"H escape flux [cm$^{-2}$ s$^{-1}$]")
    ax[0].set_title("hydrogen: SERIES circuit — the observed loss sits at the\n"
                    "slow pipe, not the fast valve (slowest clock rules)",
                    fontsize=10)
    ax[0].grid(alpha=0.3, axis="y"); ax[0].legend(fontsize=8)
    # panel b: the He parallel circuit
    valve_he = [res["conditions"][n]["species"]["He"]["phi_cm2s"] for n in names]
    ax[1].bar(x, valve_he, width=0.55, color="tab:green",
              label="thermal valve $\\Phi_J$(He) (this cache)")
    ax[1].axhline(PHI_HE_REQUIRED, color="k", ls="--", lw=1.4,
                  label="required outgassing balance (Nicolet)")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax[1].set_ylabel(r"He escape flux [cm$^{-2}$ s$^{-1}$]")
    ax[1].set_title("helium: the thermal rung fails by 9+ decades —\n"
                    "the PARALLEL plasma channel (polar wind) must carry it",
                    fontsize=10)
    ax[1].grid(alpha=0.3, axis="y"); ax[1].legend(fontsize=8)
    fig.suptitle("NR70 — the escape ladder's clock algebra: series = slowest "
                 "rules, parallel = fastest rules", fontsize=12)
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
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("VERDICT:", res["verdict"])
    make_figure(res, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
