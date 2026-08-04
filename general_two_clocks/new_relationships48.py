r"""NR71 (theory + reference-ionosphere data) — quasineutrality is the ionosphere's
incompressibility, and the AMBIPOLAR FIELD is its divergence-cleaning multiplier:
an elliptic constraint (Debye length ~cm vs 100-km gradients) whose Lagrange
multiplier is readable off the realized profile with NO ion dynamics — exactly as
pressure is readable from the incompressibility projection in Paper 1. Its static
value lifts Te/(Te+Ti) of the ion weight and raises the plasma scale height to
k(Te+Ti)/(m_i g) (the textbook doubling); its measured EXCESS over static at high
dip latitudes is the polar-wind driver — the parallel channel NR70 required.

Where this sits
===============
Paper 1's elliptic face: incompressibility is enforced instantaneously by a
constraint field (pressure) that carries no dynamics of its own — discard it and
the closure fails (the divergence-cleaning window). The ionosphere repeats this
structure with charge instead of volume:

  constraint      div u = 0            <->  n_e = n_i   (quasineutrality)
  ellipticity     Poisson, c_s -> inf  <->  Debye length ~cm << gradients ~100 km
  multiplier      pressure p           <->  ambipolar potential / field E
  read-out        p from projection    <->  eE = -(1/n_e) d(n_e k T_e)/dz  (exact)
  discard cost    closure collapse     <->  barometric clock fails by ~1.5x

Theorems (exact, verified in tests)
===================================
1. MULTIPLIER READ-OUT. Inertialess electrons (m_e g and electron inertia
   negligible): 0 = -d(n_e k T_e)/dz - n_e e E  =>  eE = -k T_e dln(n_e)/dz
   - k dT_e/dz. The multiplier is fixed by the constraint plus the OBSERVED
   profile; no ion equation is used. (Schunk & Nagy 2009, ch. 5.)
2. STATIC LIFT AND THE DOUBLED CLOCK. Adding the ion balance with the same E and
   quasineutrality: dln(n)/dz = -m_i g/k(T_e+T_i) - dln(T_e+T_i)/dz, so the
   isothermal plasma scale height is H_p = k(T_e+T_i)/(m_i g) — the barometric
   clock with the SUM of both temperatures (for T_e = T_i, exactly double), and
   the static lift fraction is eE/(m_i g) = T_e/(T_e+T_i). The light species'
   thermal clock is handed to the heavy species through the constraint — the
   inverse of NR68's separation: there the flow clock GAVE every species one
   scale height; here the constraint WELDS two species into one column again.
3. EXCESS = OUTFLOW DRIVER. With steady field-aligned flux the ion balance gains
   a friction/inertia term; the profile steepens and the read-out lift EXCEEDS
   Te/(Te+Ti). lift > 1 means net upward force on O+ — an outflow-driving column
   (Banks & Holzer 1968 polar wind; Axford 1968 — the He resolution of NR70).

Measured (committed IRI2016 cache; Bilitza et al.)
==================================================
Topside O+ window (above hmF2+75 km, O+ fraction > 85%), vertical gradients (for
straight field lines the dip angle cancels in both the scale height and the lift
fraction — both are field-aligned statements):

  * constraint lift on the electron-density scale height: H_meas is 1.45-1.64x
    the single-fluid barometric clock kTi/(m_eff g) — the naive clock is excluded;
  * full static-DE ratio H_meas/H_DE: equator 0.92 vs high-dip midlat 0.69-0.76
    — low latitude sits near static equilibrium, high latitude is steepened;
  * the multiplier itself: lift = eE/(m_eff g) = 0.63 at the equator vs its
    static prediction 0.55 (excess +0.09), but 0.75-1.03 at 60N vs static
    0.53-0.66 (excess +0.22-0.36) — up to and beyond FULL weight support.
    The high-dip excess is the outflow-driving configuration; the equatorial
    column is statically supported.

Honest scope
------------
IRI2016 is the mainstream empirical ionosphere (fitted to ionosonde/ISR/topside-
sounder data), not a raw instrument record, and its topside is an empirical shape
(NeQuick-type), so the high-dip excess is IRI's encoding of the observed steepened
topside; attributing it to field-aligned outflow is the mainstream reading (ion
upflow/polar wind at auroral-subauroral latitudes), not proven from IRI alone.
The 0N,0E control is not a field-aligned column (low dip; fountain geometry) —
its near-static vertical profile is an empirical topside fact, quoted as control
only for the excess contrast. m_eff uses IRI's own ion composition; windows are
O+-dominated so light-ion H+/He+ corrections are small. T_e, T_i are IRI's
empirical models. No absolute outflow flux is claimed — only the sign and size of
the multiplier excess.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr71_iri_cache.json")
FIG = os.path.join(HERE, "figures", "106_ambipolar_multiplier.json")

K_B = 1.380649e-23
M_U = 1.66053907e-27
R_EARTH_KM = 6371.0

ION_MASS = {"nO+": 15.999, "nH+": 1.008, "nHe+": 4.0026,
            "nO2+": 31.9988, "nNO+": 30.006}


def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


def g_of_alt_km(alt_km):
    return 9.80665 * (R_EARTH_KM / (R_EARTH_KM + np.asarray(alt_km))) ** 2


def effective_ion_mass_amu(prof):
    """Density-weighted ion mass from the IRI composition."""
    num = sum(ION_MASS[k] * np.asarray(prof[k], float) for k in ION_MASS)
    den = sum(np.asarray(prof[k], float) for k in ION_MASS)
    return num / np.maximum(den, 1.0)


def measured_scale_height_m(alt_km, n):
    z = np.asarray(alt_km) * 1e3
    dln = np.gradient(np.log(np.maximum(np.asarray(n, float), 1.0)), z)
    with np.errstate(divide="ignore"):
        return -1.0 / dln


def multiplier_readout(alt_km, ne, Te):
    """Theorem 1: eE = -(1/n_e) d(n_e k T_e)/dz  [N] — exact, electrons only."""
    z = np.asarray(alt_km) * 1e3
    ne = np.maximum(np.asarray(ne, float), 1.0)
    pe = ne * K_B * np.asarray(Te, float)
    return -np.gradient(pe, z) / ne


def static_lift_fraction(Te, Ti):
    """Theorem 2: eE/(m_i g) = Te/(Te+Ti) in static diffusive equilibrium."""
    Te, Ti = np.asarray(Te, float), np.asarray(Ti, float)
    return Te / (Te + Ti)


def plasma_scale_height_m(alt_km, Te, Ti, m_amu, with_T_gradient=True):
    """Theorem 2: 1/H = m g/(k(Te+Ti)) + dln(Te+Ti)/dz (static DE)."""
    z = np.asarray(alt_km) * 1e3
    Tp = np.asarray(Te, float) + np.asarray(Ti, float)
    inv = (np.asarray(m_amu) * M_U * g_of_alt_km(alt_km)) / (K_B * Tp)
    if with_T_gradient:
        inv = inv + np.gradient(np.log(Tp), z)
    return 1.0 / inv


def topside_window(alt_km, cond):
    prof = cond["profile"]
    ne = np.asarray(prof["ne"], float)
    frac_op = np.asarray(prof["nO+"], float) / np.maximum(ne, 1.0)
    sel = (np.asarray(alt_km) > cond["hmF2_km"] + 75.0) & \
          (np.asarray(alt_km) < 600.0) & (frac_op > 0.85)
    if sel.sum() < 3:
        sel = (np.asarray(alt_km) > cond["hmF2_km"] + 75.0) & \
              (np.asarray(alt_km) < 550.0)
    return sel


def analyze_condition(alt_km, cond):
    prof = cond["profile"]
    ne = np.asarray(prof["ne"], float)
    Te = np.asarray(prof["Te"], float)
    Ti = np.asarray(prof["Ti"], float)
    m_eff = effective_ion_mass_amu(prof)
    sel = topside_window(alt_km, cond)
    g = g_of_alt_km(alt_km)
    H_meas = measured_scale_height_m(alt_km, ne)
    H_neutral_clock = K_B * Ti / (m_eff * M_U * g)      # single-fluid barometric
    H_DE = plasma_scale_height_m(alt_km, Te, Ti, m_eff)
    eE = multiplier_readout(alt_km, ne, Te)
    lift = eE / (m_eff * M_U * g)
    lift_static = static_lift_fraction(Te, Ti)
    med = lambda x: float(np.median(np.asarray(x)[sel]))
    return dict(
        window_km=[float(np.asarray(alt_km)[sel][0]),
                   float(np.asarray(alt_km)[sel][-1])],
        n_window=int(sel.sum()),
        H_lift_over_neutral=med(H_meas / H_neutral_clock),
        H_over_static_DE=med(H_meas / H_DE),
        lift_measured=med(lift),
        lift_static=med(lift_static),
        lift_excess=med(lift - lift_static),
        Te_over_Ti=med(Te / Ti),
        profile=dict(alt_km=np.asarray(alt_km)[sel].tolist(),
                     lift=np.asarray(lift)[sel].tolist(),
                     lift_static=np.asarray(lift_static)[sel].tolist()))


def analyze():
    cache = load_cache()
    alt = np.array(cache["_alt_km"], float)
    res = {"conditions": {}}
    for name, cond in cache["conditions"].items():
        res["conditions"][name] = analyze_condition(alt, cond)
    res["verdict"] = _verdict(res)
    return res


def _verdict(res):
    lines = []
    for name, e in res["conditions"].items():
        lines.append(
            f"{name}: H lift x{e['H_lift_over_neutral']:.2f} over the naive "
            f"barometric clock; multiplier lift eE/(m g) = {e['lift_measured']:.2f}"
            f" vs static Te/(Te+Ti) = {e['lift_static']:.2f} "
            f"(excess {e['lift_excess']:+.2f})")
    eq = res["conditions"].get("equator_noon")
    mid = [e for n, e in res["conditions"].items() if n.startswith("midlat")]
    if eq and mid:
        lines.append(
            f"contrast: equatorial column statically supported "
            f"(excess {eq['lift_excess']:+.2f}); high-dip columns carry "
            f"{min(m['lift_excess'] for m in mid):+.2f} to "
            f"{max(m['lift_excess'] for m in mid):+.2f} excess -> "
            f"outflow-driving (polar-wind) configuration")
    return " | ".join(lines)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for name, e in res["conditions"].items():
        p = e["profile"]
        ls = "--" if name == "equator_noon" else "-"
        line, = ax[0].plot(p["lift"], p["alt_km"], ls, lw=1.7, label=name)
        ax[0].plot(p["lift_static"], p["alt_km"], ":", lw=1.1,
                   color=line.get_color())
    ax[0].axvline(1.0, color="k", lw=0.8)
    ax[0].annotate("full weight\nsupport", (1.005, 320), fontsize=8)
    ax[0].set_xlabel(r"ambipolar lift  $eE/(m_i g)$   "
                     r"(dotted: static $T_e/(T_e{+}T_i)$)")
    ax[0].set_ylabel("altitude [km]")
    ax[0].set_title("the multiplier, read off the profile (exact electron\n"
                    "balance): high-dip columns carry an outflow-driving excess",
                    fontsize=10)
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
    names = list(res["conditions"])
    x = np.arange(len(names))
    hlift = [res["conditions"][n]["H_lift_over_neutral"] for n in names]
    hde = [res["conditions"][n]["H_over_static_DE"] for n in names]
    ax[1].bar(x - 0.18, hlift, width=0.34, label="H$_{meas}$ / naive barometric")
    ax[1].bar(x + 0.18, hde, width=0.34, label="H$_{meas}$ / static two-fluid DE")
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax[1].set_ylabel("scale-height ratio")
    ax[1].set_title("the constraint welds electrons+ions into one taller column:\n"
                    "naive clock excluded (x1.5); static DE holds at low dip",
                    fontsize=10)
    ax[1].grid(alpha=0.3, axis="y"); ax[1].legend(fontsize=8)
    fig.suptitle("NR71 — the ambipolar field is the ionosphere's "
                 "divergence-cleaning multiplier (IRI2016)", fontsize=12)
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
    slim = {"verdict": res["verdict"],
            "conditions": {n: {k: v for k, v in e.items() if k != "profile"}
                           for n, e in res["conditions"].items()}}
    with open(a.out, "w") as fh:
        json.dump(slim, fh, indent=2)
    print("VERDICT:", res["verdict"])
    make_figure(res, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
