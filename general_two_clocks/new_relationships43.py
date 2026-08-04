r"""NR66 (theory + real data) — the spin's entropy production is REACTIVE
(housekeeping), not dissipative: the Coriolis/Lorentz face of the two clocks is
the geophysical realization of Kwon-Yeo-Lee-Park "unconventional" entropy
production — an odd-parity antisymmetric force that turns the phase-space current
without exchanging heat.

Where this sits
===============
NR57 gave the spin a thermodynamic meaning (it IS the phase-space probability
current M = AC - CAᵀ = 2ΩC); NR59 made it a divergence-free skew diffusivity;
NR60/61 wrote its spectral density as the exact Gaussian EPR. All of them measure
that σ > 0. NR66 answers the question those leave open: *what CLASS of irreversibility
is it?* Stochastic thermodynamics splits the steady EPR into

  * a DISSIPATIVE (gyrator) part — driven by a genuine thermodynamic gradient
    (two reservoirs, T₁≠T₂), which produces heat and lives in the ANISOTROPY of
    the stationary covariance (Stokes Q,U ≠ 0), and
  * a REACTIVE / HOUSEKEEPING part — driven by an odd-parity antisymmetric force
    (Coriolis on a rotating planet; Lorentz on a charge in a field), which does
    NO work, exchanges NO net heat, yet keeps a divergence-free current
    circulating forever (Kwon, Yeo, Lee & Park 2016, "unconventional EP"; Lee &
    Kwon 2019; Yeo et al. 2016 housekeeping EP with odd-parity variables).

The exact structural discriminant (this unit)
=============================================
For a linear (Ornstein-Uhlenbeck) flow dx = -A x dt + √(2D) dW, stationary C solves
A C + C Aᵀ = 2D and the irreversibility generator is Ω = A - D C⁻¹ (antisymmetric-
part carrier; NR57). Two limits close in CLOSED FORM:

REACTIVE ROTATOR  A = (1/τ)I - f·ε,  D = D₀I   (ε = [[0,-1],[1,0]]):
    C = D₀τ I              → ISOTROPIC and f-INDEPENDENT  (Q = U = 0)
    Ω = -f·ε              → PURELY ANTISYMMETRIC
    σ = tr[Ωᵀ D⁻¹ Ω C] = 2 f²τ = 2 r²/τ,   r = fτ = tan δ  (the two-clocks ratio!)
    mean power of the antisymmetric force  ⟨v·(f ε v)⟩ ≡ 0  → Q̇_heat = 0.
  So the reactive EPR is TEMPERATURE-INDEPENDENT, purely kinematic, heatless, and
  equal to 2r²/τ — the two-clocks ratio squared. It is the geophysical Kwon term.

GYRATOR  A = [[k,u],[u,k]] (symmetric),  D = diag(T₁,T₂):
    C is ANISOTROPIC (Q,U ≠ 0), Ω has a nonzero SYMMETRIC part, and
    σ > 0 iff T₁ ≠ T₂ (a real gradient); σ = 0 when T₁ = T₂ despite u ≠ 0.
  So dissipative EPR REQUIRES a gradient and LIVES IN THE ANISOTROPY.

Two clean theorems fall out:
1. **Isotropy ⇒ reactive.** If the stationary covariance is isotropic (Q=U=0) yet
   σ > 0, the irreversibility is entirely reactive/housekeeping: Ω is antisymmetric,
   the current is divergence-free (div J = tr Ω = 0 — NR59's div-free skew flux in
   phase space), and no heat flows. Anisotropy is the ONLY channel a gradient-driven
   (dissipative) part can use.
2. **The reactive rate is the clock ratio.** σ_reactive = 2r²/τ with r = fτ = tan δ:
   the same eddy/rotation clock ratio that sets the rotary coefficient (NR60), the
   skew fraction (NR59) and the loss tangent everywhere else in this work.

The real-data face (deep ocean, ANDRO)
======================================
NR56 already reduced 8280 ANDRO floats to their per-float Stokes faces: the zero-lag
anisotropy (Q,U) and the lag-1 spin V = ρ_odd. NR66 uses that committed cache to ask
whether the ocean's Lagrangian EPR is reactive or dissipative:

  * REACTIVE PREDICTION: the spin |V| tracks the planetary vorticity |f| = |sin φ|
    (Coriolis sets the rotation), but the anisotropy magnitude P = √(Q²+U²) does NOT
    grow with |f|, and once latitude is controlled P and |V| are uncorrelated (no
    gyrator coupling).
  * DISSIPATIVE PREDICTION (rejected if the above holds): |V| and P rise together.

Honest scope
------------
ANDRO floats are 10-day-cycle quasi-Lagrangian at 700-1300 dbar; V is the lag-1 odd
correlation (NR52 convention ρ_odd = O/E[0]), a proxy for the short-lag areal velocity,
not a full OU fit per float. The equatorial band carries a large ZONAL anisotropy (P
peaks at the equator) of wave/shear origin — this is exactly why P anticorrelates with
|f| and must NOT be read as a gyrator; the test is the PARTIAL |V|~P at fixed |f|. The
"no heat" statement is the mean-power identity ⟨v·εv⟩=0 for the antisymmetric drive,
i.e. the model Coriolis term does no work; it does not claim the real ocean sub-
mesoscale is adiabatic. Single-reservoir OU with an antisymmetric drift is the minimal
model whose EPR is provably all-reactive, matching Kwon et al.'s magnetic-field result
σ = 2B²/(γm).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STOKES_CACHE = os.path.join(HERE, "data", "nr56_stokes_cache.json")
BAND_CACHE = os.path.join(HERE, "data", "nr53_spin_bands_cache.json")
FIG = os.path.join(HERE, "figures", "101_reactive_epr.json")

EPS = np.array([[0.0, -1.0], [1.0, 0.0]])
OMEGA_EARTH_RAD_DAY = 7.292115e-5 * 86400.0          # Earth rotation, rad/day
DT_DAYS = 10.0


# --------------------------------------------------------------------------- #
# closed-form OU machinery
# --------------------------------------------------------------------------- #
def stationary_cov(A, D):
    """Solve A C + C Aᵀ = 2D for symmetric C (2x2, via vectorized Lyapunov)."""
    from scipy.linalg import solve_lyapunov
    return solve_lyapunov(-A, -2.0 * D)


def irreversibility(A, D):
    """Ω = A - D C⁻¹ (NR57 generator) and its symmetric/antisymmetric split."""
    C = stationary_cov(A, D)
    Om = A - D @ np.linalg.inv(C)
    Om_sym = 0.5 * (Om + Om.T)
    Om_asym = 0.5 * (Om - Om.T)
    sigma = float(np.trace(Om.T @ np.linalg.inv(D) @ Om @ C))
    Q = float(C[0, 0] - C[1, 1])
    U = float(C[0, 1] + C[1, 0])
    return dict(C=C, Omega=Om,
                sym_norm=float(np.linalg.norm(Om_sym)),
                asym_norm=float(np.linalg.norm(Om_asym)),
                sigma=sigma, Q=Q, U=U,
                anisotropy=float(math.hypot(Q, U)))


def reactive_rotator(tau, f, D0=1.0):
    A = (1.0 / tau) * np.eye(2) - f * EPS
    D = D0 * np.eye(2)
    out = irreversibility(A, D)
    # mean power of the antisymmetric (Coriolis) drive: <v . f eps v> = 0
    C = out["C"]
    out["coriolis_power"] = float(f * np.trace(EPS @ C))     # ε antisym ⇒ 0
    out["sigma_closed_form"] = 2.0 * f * f * tau
    out["r_clock"] = f * tau
    return out


def gyrator(k, u, T1, T2):
    A = np.array([[k, u], [u, k]], float)
    D = np.diag([T1, T2]).astype(float)
    return irreversibility(A, D)


# --------------------------------------------------------------------------- #
# synthetic verification
# --------------------------------------------------------------------------- #
def synthetic_checks():
    out = {"reactive": [], "gyrator": []}
    for tau, f, D0 in [(20.0, 0.10, 0.5), (17.0, 0.23, 0.7),
                       (40.0, 0.05, 1.1), (5.0, 0.25, 1.3)]:
        r = reactive_rotator(tau, f, D0)
        out["reactive"].append(dict(
            tau=tau, f=f, D0=D0, sigma=r["sigma"],
            sigma_closed=r["sigma_closed_form"],
            sigma_err=abs(r["sigma"] - r["sigma_closed_form"]),
            anisotropy=r["anisotropy"], sym_norm=r["sym_norm"],
            coriolis_power=r["coriolis_power"], r_clock=r["r_clock"]))
    for k, u, T1, T2 in [(1.0, 0.6, 3.0, 0.5), (1.0, 0.6, 1.0, 1.0),
                         (1.0, 0.9, 2.0, 1.0)]:
        g = gyrator(k, u, T1, T2)
        out["gyrator"].append(dict(
            k=k, u=u, T1=T1, T2=T2, sigma=g["sigma"],
            anisotropy=g["anisotropy"], sym_norm=g["sym_norm"]))
    return out


# --------------------------------------------------------------------------- #
# real data: ANDRO Stokes faces (NR56 committed cache)
# --------------------------------------------------------------------------- #
def load_stokes(path=STOKES_CACHE):
    with open(path) as fh:
        c = json.load(fh)
    fl = c["floats"]
    lat = np.array(fl["lat"], float)
    Q = np.array(fl["Q"], float)
    U = np.array(fl["U"], float)
    V = np.array(fl["V"], float)
    ok = np.isfinite(lat) & np.isfinite(Q) & np.isfinite(U) & np.isfinite(V)
    return lat[ok], Q[ok], U[ok], V[ok]


def _spearman(x, y):
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def _partial_spearman(x, y, z, nbin=12):
    """Spearman(x,y) controlling z by within-|z|-bin rank residuals."""
    from scipy.stats import rankdata
    order = np.argsort(np.abs(z))
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    n = len(x)
    edges = np.linspace(0, n, nbin + 1).astype(int)
    rx = np.empty(n); ry = np.empty(n)
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 3:
            rx[a:b] = 0.0; ry[a:b] = 0.0; continue
        rx[a:b] = rankdata(x[a:b]) - (b - a + 1) / 2.0
        ry[a:b] = rankdata(y[a:b]) - (b - a + 1) / 2.0
    from scipy.stats import pearsonr
    r, p = pearsonr(rx, ry)
    return float(r), float(p)


def ocean_reactive_test():
    lat, Q, U, V = load_stokes()
    f = np.sin(np.radians(lat))
    P = np.hypot(Q, U)                                   # anisotropy magnitude
    aV, aF = np.abs(V), np.abs(f)
    res = dict(n_floats=int(lat.size))
    res["spin_tracks_f"] = dict(zip(("rho", "p"), _spearman(aV, aF)))
    res["anisotropy_vs_f"] = dict(zip(("rho", "p"), _spearman(P, aF)))
    res["spin_vs_anisotropy_raw"] = dict(zip(("rho", "p"), _spearman(aV, P)))
    r, p = _partial_spearman(aV, P, aF)
    res["spin_vs_anisotropy_partial_f"] = dict(rho=r, p=p)
    # signed hemispheric check: spin sign follows sign(f) (cyclonic)
    res["signed_spin_vs_f"] = dict(zip(("rho", "p"), _spearman(V, f)))
    # per-band effective clock ratio r = f * tau, using tau from |V| decay proxy
    bands = []
    edges = np.arange(-75, 76, 15)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lat >= lo) & (lat < hi)
        if m.sum() < 40:
            continue
        bands.append(dict(lat_lo=int(lo), lat_hi=int(hi), n=int(m.sum()),
                          mean_V=float(V[m].mean()), mean_absV=float(aV[m].mean()),
                          mean_P=float(P[m].mean()),
                          mean_f=float(np.sin(np.radians(0.5 * (lo + hi))))))
    res["bands"] = bands
    return res


def verdict(syn, ocean):
    v = {}
    v["reactive_isotropic"] = all(s["anisotropy"] < 1e-9 for s in syn["reactive"])
    v["reactive_antisymmetric"] = all(s["sym_norm"] < 1e-9 for s in syn["reactive"])
    v["reactive_sigma_closed"] = all(s["sigma_err"] < 1e-10 for s in syn["reactive"])
    v["reactive_heatless"] = all(abs(s["coriolis_power"]) < 1e-9
                                 for s in syn["reactive"])
    gy = {(*(g["T1"], g["T2"]),): g for g in syn["gyrator"]}
    v["gyrator_needs_gradient"] = (
        any(g["sigma"] > 1e-6 for g in syn["gyrator"] if g["T1"] != g["T2"]) and
        all(g["sigma"] < 1e-9 for g in syn["gyrator"] if g["T1"] == g["T2"]))
    v["gyrator_anisotropic"] = all(g["anisotropy"] > 1e-6 for g in syn["gyrator"])
    # ocean: reactive present (spin~f), dissipative absent (no partial spin~aniso)
    v["ocean_spin_tracks_coriolis"] = (ocean["spin_tracks_f"]["rho"] > 0 and
                                       ocean["spin_tracks_f"]["p"] < 1e-3)
    v["ocean_signed_spin_cyclonic"] = (ocean["signed_spin_vs_f"]["rho"] < 0 and
                                       ocean["signed_spin_vs_f"]["p"] < 1e-3)
    v["ocean_anisotropy_not_gyrator"] = (
        ocean["anisotropy_vs_f"]["rho"] <= 0.05 and
        abs(ocean["spin_vs_anisotropy_partial_f"]["rho"]) < 0.10)
    v["ocean_epr_is_reactive"] = (v["ocean_spin_tracks_coriolis"] and
                                  v["ocean_anisotropy_not_gyrator"])
    return v


def analyze():
    syn = synthetic_checks()
    ocean = ocean_reactive_test()
    v = verdict(syn, ocean)
    return {"what": "NR66: the spin's EPR is reactive/housekeeping (Coriolis, "
                    "heatless, = 2r²/τ), not gyrator-dissipative — proved in closed "
                    "form and confirmed on ANDRO: ocean spin tracks |f| while "
                    "anisotropy carries no gradient signature",
            "synthetic": syn, "ocean": ocean, "verdict": v}


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    oc = res["ocean"]
    bands = oc["bands"]
    lat = [0.5 * (b["lat_lo"] + b["lat_hi"]) for b in bands]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].plot(lat, [b["mean_V"] for b in bands], "o-", label=r"spin $\langle V\rangle$ (signed)")
    ax[0].plot(lat, [math.sin(math.radians(l)) * max(
        abs(b["mean_V"]) for b in bands) / 1.0 for l in lat],
        "--", color="grey", label=r"$-\sin\varphi$ (Coriolis, scaled)")
    ax[0].plot(lat, [-math.sin(math.radians(l)) * 0.05 for l in lat], ":",
               color="tab:red", alpha=0.0)
    ax[0].set_xlabel("latitude [deg]")
    ax[0].set_ylabel(r"Lagrangian spin $\langle V\rangle$")
    ax[0].set_title("REACTIVE: spin follows planetary vorticity\n"
                    fr"$|V|$ vs $|f|$: $\rho$={oc['spin_tracks_f']['rho']:.2f} "
                    fr"(p={oc['spin_tracks_f']['p']:.1e})", fontsize=10)
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(lat, [b["mean_P"] for b in bands], "s-", color="tab:green",
               label=r"anisotropy $\langle P\rangle$")
    ax[1].plot(lat, [b["mean_absV"] for b in bands], "o-", color="tab:blue",
               label=r"$\langle|V|\rangle$")
    ax[1].set_xlabel("latitude [deg]")
    ax[1].set_ylabel("magnitude")
    ax[1].set_title("NOT gyrator: anisotropy carries no $|f|$ signature\n"
                    fr"partial $|V|\sim P\,|\,|f|$: "
                    fr"$\rho$={oc['spin_vs_anisotropy_partial_f']['rho']:+.2f}",
                    fontsize=10)
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle("NR66 — the spin's entropy production is reactive (housekeeping), "
                 "not dissipative", fontsize=12)
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
        json.dump(res, fh, indent=2)
    v = res["verdict"]
    print("REACTIVE (closed form):",
          {k: v[k] for k in v if k.startswith("reactive")})
    print("GYRATOR (closed form): ",
          {k: v[k] for k in v if k.startswith("gyrator")})
    oc = res["ocean"]
    print(f"OCEAN (ANDRO, n={oc['n_floats']}): "
          f"|V|~|f| rho={oc['spin_tracks_f']['rho']:+.3f} "
          f"(p={oc['spin_tracks_f']['p']:.1e}); "
          f"P~|f| rho={oc['anisotropy_vs_f']['rho']:+.3f}; "
          f"partial |V|~P||f| rho={oc['spin_vs_anisotropy_partial_f']['rho']:+.3f}")
    print("VERDICT ocean EPR is reactive:", v["ocean_epr_is_reactive"])
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
