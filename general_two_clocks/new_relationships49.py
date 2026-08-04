r"""NR72 (theory + matched reference models) — the F2 peak is a two-clocks
crossover WHEN it is chemistry that builds it, and the dimensionless clock ratio
R(h) = beta * H_p^2 / D_a evaluated AT the empirical peak is a parameter-free
classifier of what built the layer: daytime midlatitude peaks sit at R ~ 5-30
(balance-formed, the chemical clock handing off to the diffusion clock), while
midnight and equatorial peaks sit at R <= 1e-2 — two-plus decades away — because
wind lifting / plasmaspheric flux and the ExB fountain (a THIRD clock) hold those
layers far above the chemical handoff.

Where this sits
===============
Classical F2-layer theory (Rishbeth & Garriott 1969; Schunk & Nagy 2009 ch. 13)
says the daytime peak forms near the level where the O+ chemical loss rate
beta = k1[N2] + k2[O2] equals the ambipolar-diffusion rate D_a/H^2: below it
photochemistry wins (production ~ loss), above it diffusion wins (the NR71
diffusive-equilibrium topside). That is verbatim a two-clocks crossover:
tau_chem = 1/beta (the reactive/molecular clock) racing tau_diff = H_p^2/D_a
(the transport clock), with the peak at the handoff. NR68 read the turbopause
as K_zz = D_i; NR72 reads hmF2 as beta = D_a/H_p^2 — the same audit, one story
lower in the ionosphere, and now with a twist: sometimes the audit FAILS, and
the failure is the measurement.

The audit quantity
==================
    R(h) = beta(h) * H_p(h)^2 / D_a(h)        (dimensionless clock ratio)

  beta = k1(T)[N2] + k2(T)[O2]  — O+ loss (St-Maurice & Torr 1978 rates)
  D_a  = k(Te+Ti)/(m_O nu_in), nu_in = O+-O charge-exchange collision rate
         (Banks 1966; Schunk & Nagy) — ambipolar diffusion on NR71's welded column
  H_p  = k(Te+Ti)/(m_O g)       — NR71's doubled plasma scale height

R falls with altitude with a measured e-folding of ~16 km (beta collapses with
[N2] while D_a ~ 1/[O] grows), so the R = 1 level is robust: an O(1) convention
change (H_p vs the neutral scale height, factors in nu_in) moves it by <= 25 km.
Neutrals from the committed NRLMSIS 2.1 profiles, plasma from the committed
IRI2016 profiles, MATCHED epochs/locations (one cache, both models, no tuning).

Measured (5 matched conditions, 2019 epochs)
============================================
  R evaluated at IRI's own empirical peak height hmF2:

    midlat equinox noon    R(hmF2) ~ 17      hmF2 = 228 km   balance-formed
    midlat winter noon     R(hmF2) ~ 27      hmF2 = 217 km   balance-formed
    midlat summer noon     R(hmF2) ~  5      hmF2 = 245 km   balance-formed
    midlat equinox 00LT    R(hmF2) ~ 4e-3    hmF2 = 332 km   dynamics-formed
    equator equinox noon   R(hmF2) ~ 2e-3    hmF2 = 359 km   dynamics-formed

  The two families are separated by >2.5 DECADES with no case in between.
  Daytime midlatitude peaks sit 1-2 N2 scale heights below the naive R = 1
  level (R ~ 5-30 there), exactly where full F2-layer solutions put the peak
  relative to the balance level; the midnight and equatorial peaks sit 78-85 km
  ABOVE the R = 1 level — no chemical handoff builds them. Mainstream dynamics
  names both failures: the nighttime layer is lifted and maintained by
  equatorward thermospheric wind and downward plasmaspheric flux (Rishbeth's
  servo picture), and the equatorial layer rides the ExB fountain (Appleton
  anomaly) — a third, electrodynamic clock that the two-clock audit correctly
  refuses to absorb.

Honest scope
------------
NRLMSIS and IRI are the mainstream empirical reference models (not raw
instrument records) and are independently constructed — their consistency here
is a nontrivial cross-check but not a first-principles measurement. Rate
coefficients k1, k2 and nu_in carry ~20-30% uncertainties (moving R by the same
factor — negligible against a 2.5-decade gap and a 16-km e-fold). Effective
temperatures for k1, k2 use (Ti+Tn)/2 (drift contributions to Teff neglected —
fine at midlatitude quiet time). Vertical (not field-aligned) gradients; for
the daytime midlat columns dip > 60 deg so projection factors are O(1) and
absorbed in the convention robustness. The classifier statement needs only R
computed IDENTICALLY across conditions, so convention factors cancel where it
matters. No claim that R predicts hmF2 exactly — the claim is which physics
builds the peak, and its 2.5-decade separation.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr72_f2peak_cache.json")
FIG = os.path.join(HERE, "figures", "107_f2peak_two_clocks.json")

K_B = 1.380649e-23
M_U = 1.66053907e-27
M_O = 15.999 * M_U
R_EARTH_KM = 6371.0


def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


def g_of_alt_km(alt_km):
    return 9.80665 * (R_EARTH_KM / (R_EARTH_KM + np.asarray(alt_km))) ** 2


def o_plus_loss_rate(N2_m3, O2_m3, Teff):
    """beta = k1[N2] + k2[O2] [1/s]; St-Maurice & Torr 1978 polynomial rates
    (validity T < ~1700 K, satisfied in these windows)."""
    t3 = np.asarray(Teff) / 300.0
    k1 = (1.533e-12 - 5.92e-13 * t3 + 8.6e-14 * t3 ** 2) * 1e-6      # m^3/s
    k2 = (2.82e-11 - 7.74e-12 * t3 + 1.073e-12 * t3 ** 2
          - 5.17e-14 * t3 ** 3 + 9.65e-16 * t3 ** 4) * 1e-6
    return k1 * np.asarray(N2_m3) + k2 * np.asarray(O2_m3)


def o_plus_o_collision_rate(O_m3, Tr):
    """Resonant O+-O charge-exchange collision frequency [1/s] (Banks; Schunk &
    Nagy): nu = 3.67e-11 n(O)[cm^-3] sqrt(Tr) (1 - 0.064 log10 Tr)^2."""
    Tr = np.asarray(Tr)
    return 3.67e-11 * (np.asarray(O_m3) / 1e6) * np.sqrt(Tr) \
        * (1.0 - 0.064 * np.log10(Tr)) ** 2


def clock_ratio_profile(alt_km, msis, iri):
    """R(z) = beta * H_p^2 / D_a and its ingredients."""
    N2, O2, O = (np.asarray(msis[k], float) for k in ("N2", "O2", "O"))
    Tn = np.asarray(msis["Tn"], float)
    Te, Ti = np.asarray(iri["Te"], float), np.asarray(iri["Ti"], float)
    g = g_of_alt_km(alt_km)
    beta = o_plus_loss_rate(N2, O2, 0.5 * (Ti + Tn))
    nu = o_plus_o_collision_rate(O, 0.5 * (Ti + Tn))
    Da = K_B * (Te + Ti) / (M_O * nu)
    Hp = K_B * (Te + Ti) / (M_O * g)
    R = beta * Hp ** 2 / Da
    return dict(R=R, beta=beta, Da=Da, Hp_m=Hp,
                tau_chem=1.0 / beta, tau_diff=Hp ** 2 / Da)


def crossing_altitude(alt_km, R, level=1.0):
    """Altitude where R falls through `level` (log interpolation)."""
    alt = np.asarray(alt_km)
    lnR = np.log(np.maximum(np.asarray(R), 1e-300))
    ln0 = np.log(level)
    for i in range(1, len(alt)):
        if lnR[i - 1] > ln0 >= lnR[i]:
            f = (lnR[i - 1] - ln0) / (lnR[i - 1] - lnR[i])
            return float(alt[i - 1] + f * (alt[i] - alt[i - 1]))
    return None


def r_efold_km(alt_km, R, z_lo=220.0, z_hi=320.0):
    """e-folding length of R(z) in the crossover zone [km]."""
    alt = np.asarray(alt_km)
    sel = (alt >= z_lo) & (alt <= z_hi)
    slope = np.polyfit(alt[sel], np.log(np.asarray(R)[sel]), 1)[0]
    return float(-1.0 / slope)


def analyze():
    cache = load_cache()
    alt = np.array(cache["_alt_km"], float)
    res = {"conditions": {}}
    for name, cond in cache["conditions"].items():
        p = clock_ratio_profile(alt, cond["msis"], cond["iri"])
        R_at_peak = float(np.exp(np.interp(cond["hmF2_km"], alt,
                                           np.log(np.maximum(p["R"], 1e-300)))))
        z1 = crossing_altitude(alt, p["R"], 1.0)
        res["conditions"][name] = dict(
            hmF2_km=cond["hmF2_km"],
            R_at_peak=R_at_peak,
            z_R1_km=z1,
            peak_minus_z1_km=cond["hmF2_km"] - z1 if z1 else None,
            efold_km=r_efold_km(alt, p["R"]),
            balance_formed=bool(R_at_peak > 1.0),
            profile=dict(alt_km=alt.tolist(), R=p["R"].tolist()))
    res["classifier_gap_decades"] = _gap(res)
    res["verdict"] = _verdict(res)
    return res


def _gap(res):
    formed = [np.log10(e["R_at_peak"]) for e in res["conditions"].values()
              if e["balance_formed"]]
    dyn = [np.log10(e["R_at_peak"]) for e in res["conditions"].values()
           if not e["balance_formed"]]
    if not formed or not dyn:
        return None
    return float(min(formed) - max(dyn))


def _verdict(res):
    lines = []
    for name, e in res["conditions"].items():
        tag = "balance-formed" if e["balance_formed"] else "dynamics-formed"
        lines.append(f"{name}: R(hmF2)={e['R_at_peak']:.2g} at "
                     f"{e['hmF2_km']:.0f} km ({tag}; R=1 level at "
                     f"{e['z_R1_km']:.0f} km, e-fold {e['efold_km']:.0f} km)")
    lines.append(f"classifier gap = {res['classifier_gap_decades']:.1f} decades "
                 f"between balance-formed (daytime midlat) and dynamics-formed "
                 f"(midnight wind/flux, equatorial ExB fountain) peaks")
    return " | ".join(lines)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for name, e in res["conditions"].items():
        p = e["profile"]
        ls = "-" if e["balance_formed"] else "--"
        line, = ax[0].semilogx(p["R"], p["alt_km"], ls, lw=1.6, label=name)
        ax[0].plot(e["R_at_peak"], e["hmF2_km"], "o", ms=7,
                   color=line.get_color())
    ax[0].axvline(1.0, color="k", lw=0.9)
    ax[0].set_xlim(1e-6, 1e6)
    ax[0].set_xlabel(r"clock ratio $R=\beta H_p^2/D_a$  "
                     r"($\bullet$ = at IRI's empirical peak)")
    ax[0].set_ylabel("altitude [km]")
    ax[0].set_title("the chemical clock hands off to the diffusion clock at "
                    "R=1;\ndots: where each empirical F2 peak actually sits",
                    fontsize=10)
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
    names = list(res["conditions"])
    x = np.arange(len(names))
    vals = [res["conditions"][n]["R_at_peak"] for n in names]
    cols = ["tab:blue" if res["conditions"][n]["balance_formed"]
            else "tab:red" for n in names]
    ax[1].bar(x, vals, color=cols, width=0.55)
    ax[1].axhline(1.0, color="k", lw=0.9)
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax[1].set_ylabel("R at the empirical peak")
    ax[1].set_title("R(hmF2) classifies what BUILT the layer:\n"
                    "blue = chemical handoff; red = wind/flux & ExB fountain "
                    f"(gap {res['classifier_gap_decades']:.1f} decades)",
                    fontsize=10)
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("NR72 — the F2 peak audited by the two clocks "
                 "(NRLMSIS 2.1 + IRI2016, matched)", fontsize=12)
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
    slim = {"classifier_gap_decades": res["classifier_gap_decades"],
            "verdict": res["verdict"],
            "conditions": {n: {k: v for k, v in e.items() if k != "profile"}
                           for n, e in res["conditions"].items()}}
    with open(a.out, "w") as fh:
        json.dump(slim, fh, indent=2)
    print("VERDICT:", res["verdict"])
    make_figure(res, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
