r"""NR68 (theory + real data) — the atmosphere's vertical structure IS the two-clocks
competition made spatial: the turbopause is the flow-clock/molecular-clock crossover,
the homosphere is Paper-1 scalar-independent mixing, the heterosphere is per-species
diffusive separation, and Jeans escape is the top boundary where the thermal clock
finally beats the gravity constraint for the light, fast tail.

Where this sits
===============
Paper 1 measured, in turbulence, that the eddy-mixing memory time is
SCALAR-INDEPENDENT: τ_c(salt)/τ_c(heat) ≈ 1 across a 100× molecular-Lewis contrast,
because the memory belongs to the FLOW, not the diffusing substance. NR68 finds the
same statement written vertically in the real atmosphere, where the mainstream
aeronomy community has measured it for 60 years without reading it as a clock
competition.

The two clocks, stacked in altitude
===================================
A parcel of air has two ways to transport a minor species upward:
  * the FLOW clock — turbulent eddy mixing, coefficient K_zz (gravity-wave breaking;
    Colegrove 1965; Hines 1997). It is SCALAR-BLIND: it moves every species at the
    same rate, so it drives all constituents toward ONE shared scale height set by the
    MEAN air mass, H_bar = kT/(m_bar g). (Constant mixing ratio — Paper-1's
    scalar-independence, exactly.)
  * the MOLECULAR clock — species diffusion, coefficient D_i ∝ 1/(m_i^{1/2} n). It is
    SCALAR-SELECTIVE: in diffusive equilibrium each species settles under its OWN
    scale height H_i = kT/(m_i g), light species reaching higher ("cream from milk").

The crossover is the **turbopause / homopause**, defined in the mainstream literature
as exactly the altitude where D_i = K_zz (Banks & Kockarts 1973; Vlasov & Kelley 2014).
Below it (homosphere) the flow clock wins → mean-mass mixing. Above it (heterosphere)
the molecular clock wins → per-species separation. This is Paper-1's a-priori benchmark
rendered as a vertical coordinate: the same "one clock is scalar-blind, the other is
scalar-selective" split, with altitude playing the role of the closure regime.

The measurement (mainstream reference atmosphere, no fit)
========================================================
On the committed NRLMSIS 2.1 cache (the community-standard empirical atmosphere;
Emmert et al. 2021), read the per-species density scale height H_i(z) = −1/(d ln n_i/dz)
for the chemically-inert tracers He, Ar, N2, O directly from the model profiles, and
compare to the two clocks' predictions:

  Homosphere test (scalar-INDEPENDENCE):  H_He/H_N2 → 1 at low altitude
                                          (all species share H_bar).
  Heterosphere test (scalar-SELECTIVITY): H_i → kT/(m_i g), so
                                          H_He/H_N2 → m_N2/m_He = 28/4 = 7 aloft.

The ratio H_He/H_N2 is the two-clocks DISCRIMINANT: 1 = flow clock (mixed),
m_N2/m_He = 7 = molecular clock (separated). The crossover altitude where it leaves 1
is the turbopause. Same for Ar/N2 (predicted 28/40 = 0.70 aloft) and O/N2 (28/16 = 1.75).

Jeans escape — the top boundary
===============================
At the exobase the two clocks stop being about *transport* and become about *escape*:
the gravity constraint (escape speed v_esc = √(2GM/r)) races the thermal clock (the
Maxwell tail, most-probable speed v_p = √(2kT/m) ∝ 1/√m). The Jeans parameter
λ = v_esc²/v_p² = GMm/(kT r) = (m g r)/(kT) is the ratio of the two clocks; the Jeans
escape flux ∝ (1+λ)e^{−λ} is astronomically larger for light m (small λ). That is why
H and He leak from Earth while N2/O2 are bound — the SAME m-vs-thermal competition that
sets the heterospheric scale heights, taken to the altitude where the confining clock
loses entirely to the fast tail. NR68 computes λ per species at the exobase from the
cache and shows the escape ordering H > He ≫ O > N2 follows from λ alone.

Two theorems
============
1. **The turbopause is a two-clocks crossover, not a substance boundary.** It is
   defined by D_i = K_zz — a race between the scalar-selective molecular clock and the
   scalar-blind flow clock — exactly the Paper-1 competition, with altitude as the
   control parameter. Homosphere = flow-clock regime (mixing-ratio constant,
   scalar-independent); heterosphere = molecular-clock regime (per-species H_i).
2. **Jeans escape is the same competition at the exobase.** The Jeans parameter
   λ = m g r/(kT) is the gravity-clock/thermal-clock ratio; escape selects the light,
   fast tail — the identical m-dependence that separates the heterosphere, taken to
   its limit. Gravity is the constraint clock; the Maxwell tail is the diffusive clock;
   the atmosphere's very retention of its heavy species is the constraint clock winning
   everywhere except the light tail.

Honest scope
------------
NRLMSIS 2.1 is a data-assimilative empirical MODEL, not a single instrument — but it is
THE mainstream reference atmosphere, constrained by decades of mass-spectrometer,
incoherent-scatter-radar and satellite-drag data; the He/Ar/N2 heterospheric profiles
it encodes are the observational barometric law (Vlasov & Kelley 2014 verify [He]
follows the diffusive barometric law to high precision). We read only the
chemically-inert species (He, Ar, N2) plus O where its photochemistry is slow; reactive
minor species (NO, O3) are excluded because sources/sinks, not the two clocks, set their
profiles. Thermal diffusion (the small α_T correction, Banks & Kockarts) is neglected;
it shifts H_i by a few % and does not touch the crossover. The turbopause is not a sharp
level (it is species- and season-dependent; Garcia 2014) — we report the crossover as
the altitude band where the discriminant departs from 1, consistent with the mainstream
100–110 km value. Jeans escape here is the classical thermal estimate (non-thermal
escape — charge exchange, polar wind — also matters for the real H budget and is out of
scope). This is a READING of established aeronomy in the two-clocks language + a
quantitative discriminant, not a new atmospheric measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr68_msis_cache.json")
FIG = os.path.join(HERE, "figures", "103_atmosphere_two_clocks.json")

# physical constants (SI)
K_B = 1.380649e-23
M_U = 1.66053907e-27
G_GRAV = 6.674e-11
M_EARTH = 5.972e24
R_EARTH = 6.371e6

# molar masses (amu) of the read-out species
MASS = {"N2": 28.0134, "O2": 31.9988, "O": 15.999, "He": 4.0026,
        "H": 1.008, "Ar": 39.948}
# column index in the cached MSIS profile
COL = {"rho": 0, "N2": 1, "O2": 2, "O": 3, "He": 4, "H": 5, "Ar": 6,
       "N": 7, "AnomO": 8, "NO": 9, "T": 10}
INERT = ("He", "Ar", "N2", "O")           # chemically-inert / slow read-out set


def g_of_z(z_m):
    """Altitude-dependent gravity."""
    return G_GRAV * M_EARTH / (R_EARTH + z_m) ** 2


def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


def _profile(cond):
    """(alts_km, arr[nalt, ncol]) for a named condition dict."""
    return np.array(cond["profile"], float)


def measured_scale_height(alts_km, n):
    """H_i(z) = -1/(d ln n / dz), km, from the density profile."""
    z = np.asarray(alts_km) * 1e3
    ln = np.log(np.maximum(n, 1e-300))
    dlnn_dz = np.gradient(ln, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -1.0 / dlnn_dz
    return H / 1e3


def theoretical_scale_height(alts_km, T, mass_amu):
    """Diffusive-equilibrium H_i = kT/(m_i g), km."""
    z = np.asarray(alts_km) * 1e3
    g = g_of_z(z)
    return K_B * np.asarray(T) / (mass_amu * M_U * g) / 1e3


def jeans_parameter(alts_km, T, mass_amu):
    """λ(z) = m g r / (kT) = v_esc²/v_p²; the gravity-clock/thermal-clock ratio."""
    z = np.asarray(alts_km) * 1e3
    r = R_EARTH + z
    g = g_of_z(z)
    return (mass_amu * M_U) * g * r / (K_B * np.asarray(T))


def jeans_flux_factor(lam):
    """Relative Jeans escape flux ∝ (1+λ) e^{−λ} (up to the shared n·v_p prefactor)."""
    return (1.0 + lam) * np.exp(-lam)


def analyze_condition(alts_km, arr):
    T = arr[:, COL["T"]]
    Hmeas = {s: measured_scale_height(alts_km, arr[:, COL[s]]) for s in INERT}
    Hth = {s: theoretical_scale_height(alts_km, T, MASS[s]) for s in INERT}
    alts = np.asarray(alts_km)
    # two-clocks discriminant: H_He / H_N2 (1 = mixed/flow clock, 7 = separated)
    ratio = Hmeas["He"] / Hmeas["N2"]
    mass_ratio = MASS["N2"] / MASS["He"]
    # turbopause: the transition is gradual, so report a BAND, not one number.
    #   onset   — discriminant first clearly departs from 1 (r > 1.25):
    #             the flow clock's first loss
    #   log-mid — geometric-mean crossover r > sqrt(mass_ratio):
    #             the clocks' log-scale handoff (compare to the conventional
    #             D_i = K_zz homopause, ~95-120 km in the literature)
    #   lin-mid — linear halfway point r > (1+mass_ratio)/2 (transition largely done)
    def _first_crossing(threshold):
        for i in range(1, len(alts)):
            if ratio[i - 1] < threshold <= ratio[i]:
                return float(np.interp(threshold, [ratio[i - 1], ratio[i]],
                                       [alts[i - 1], alts[i]]))
        return None

    turbo = dict(onset_km=_first_crossing(1.25),
                 log_mid_km=_first_crossing(float(np.sqrt(mass_ratio))),
                 linear_mid_km=_first_crossing(0.5 * (1.0 + mass_ratio)))
    # homosphere sample (low) and heterosphere sample (high)
    i_lo = int(np.argmin(np.abs(alts - 85.0)))
    i_hi = int(np.argmin(np.abs(alts - 450.0)))
    het = {}
    for s in ("He", "Ar", "O"):
        het[s] = dict(
            measured_ratio_vs_N2=float(Hmeas[s][i_hi] / Hmeas["N2"][i_hi]),
            mass_prediction=float(MASS["N2"] / MASS[s]))
    # Jeans at the top of the cache (exobase proxy)
    i_exo = len(alts) - 1
    lam = {s: float(jeans_parameter(alts[i_exo], T[i_exo], MASS[s]))
           for s in ("H", "He", "O", "N2")}
    flux = {s: float(jeans_flux_factor(lam[s])) for s in lam}
    return dict(
        turbopause=turbo,
        homosphere=dict(alt_km=float(alts[i_lo]),
                        He_over_N2=float(ratio[i_lo]),
                        Ar_over_N2=float(Hmeas["Ar"][i_lo] / Hmeas["N2"][i_lo])),
        heterosphere=dict(alt_km=float(alts[i_hi]),
                          He_over_N2=float(ratio[i_hi]),
                          mass_prediction_He=mass_ratio, species=het),
        jeans=dict(alt_km=float(alts[i_exo]), T_K=float(T[i_exo]),
                   lam=lam, flux_factor=flux),
        discriminant_profile=dict(alt_km=alts.tolist(),
                                  He_over_N2=ratio.tolist()),
        Hmeas_vs_Hth_He=dict(
            alt_km=alts.tolist(),
            measured=Hmeas["He"].tolist(), theoretical=Hth["He"].tolist()))


def analyze(cache=None):
    cache = cache or load_cache()
    alts = cache["_alts_km"]
    res = {"what": "NR68: the atmosphere's vertical structure is the two-clocks "
                   "competition — turbopause crossover, homosphere scalar-independent "
                   "mixing (Paper-1), heterosphere per-species separation, Jeans "
                   "escape as the exobase limit",
           "model": cache["_model"], "conditions": {}}
    for name, cond in cache["conditions"].items():
        res["conditions"][name] = analyze_condition(alts, _profile(cond))
    res["verdict"] = _verdict(res)
    return res


def _verdict(res):
    lines = []
    for name, e in res["conditions"].items():
        het = e["heterosphere"]
        lines.append(
            f"{name}: turbopause onset≈{e['turbopause']['onset_km']:.0f} km, "
            f"crossover≈{e['turbopause']['log_mid_km']:.0f} km; "
            f"H_He/H_N2 = {e['homosphere']['He_over_N2']:.2f} (85 km, mixed→1) "
            f"→ {het['He_over_N2']:.2f} (450 km) vs mass ratio "
            f"{het['mass_prediction_He']:.2f}")
    # Jeans ordering (use first condition)
    first = next(iter(res["conditions"].values()))
    j = first["jeans"]
    order = sorted(j["flux_factor"], key=lambda s: -j["flux_factor"][s])
    lines.append("Jeans escape ordering at the exobase (flux factor): "
                 + " > ".join(f"{s}(λ={j['lam'][s]:.1f})" for s in order))
    return " | ".join(lines)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for name, e in res["conditions"].items():
        d = e["discriminant_profile"]
        ax[0].plot(d["He_over_N2"], d["alt_km"], label=name, lw=1.6)
    mr = next(iter(res["conditions"].values()))["heterosphere"]["mass_prediction_He"]
    ax[0].axvline(1.0, color="grey", ls=":", lw=1, label="flow clock (mixed, =1)")
    ax[0].axvline(mr, color="k", ls="--", lw=1,
                  label=f"molecular clock (separated, =m$_{{N2}}$/m$_{{He}}$={mr:.1f})")
    ax[0].set_xlabel(r"two-clocks discriminant  $H_{He}/H_{N_2}$")
    ax[0].set_ylabel("altitude [km]")
    ax[0].set_title("the turbopause is where the flow clock hands off\n"
                    "to the molecular clock (scalar-blind → scalar-selective)",
                    fontsize=10)
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
    # panel b: measured vs diffusive-equilibrium He scale height
    e0 = next(iter(res["conditions"].values()))
    h = e0["Hmeas_vs_Hth_He"]
    ax[1].plot(h["measured"], h["alt_km"], label="measured $H_{He}$ (MSIS)", lw=1.8)
    ax[1].plot(h["theoretical"], h["alt_km"], "--",
               label=r"diffusive eq. $kT/m_{He}g$", lw=1.4)
    ax[1].set_xlabel("He scale height [km]")
    ax[1].set_ylabel("altitude [km]")
    ax[1].set_title("above the turbopause He follows its OWN\n"
                    "diffusive scale height (the molecular clock)", fontsize=10)
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle("NR68 — the atmosphere's vertical structure is the two clocks "
                 "(NRLMSIS 2.1)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FIG)
    a = ap.parse_args()
    res = analyze()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({k: v for k, v in res.items()}, fh, indent=2)
    print("VERDICT:", res["verdict"])
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
