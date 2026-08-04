r"""NR67 (theory + real data) — the Lagrangian velocity field has a POLARIZATION
ENTROPY, and its purity splits into a reversible (even) and an irreversible (odd)
part: p² = p_even² + p_odd², with the odd purity fraction = NR66's reactive share.

Where this sits
===============
NR56 gave the 2-D Lagrangian velocity its Stokes parameters (I, Q, U, V) and the
wave/vortex (linear/circular polarization) split. NR60/61 made the rotary asymmetry
an entropy-production density; NR66 proved the spin's EPR is reactive (heatless).
NR67 supplies the single SCALAR those imply and that polarization optics has used
for 30 years but oceanography has not: the **von Neumann / Cloude–Pottier
polarization entropy** of the coherency (density) matrix, and the clean theorem that
its purity separates by TIME PARITY.

The construction (exact)
========================
Build the 2×2 Hermitian coherency matrix from the normalized Stokes vector
s = (Q, U, V)/I:

    J = ½ (σ₀ + Q σ₃ + U σ₁ + V σ₂) = ½ [[1+Q, U−iV], [U+iV, 1−Q]]

(σ's the Pauli matrices; the standard optics map). tr J = 1, eigenvalues
λ± = (1 ± p)/2 with the **degree of polarization** p = √(Q²+U²+V²). The
**polarization entropy** (Cloude & Pottier 1997; Réfrégier 2005) is the von Neumann
entropy

    S = −λ₊ log₂ λ₊ − λ₋ log₂ λ₋ = h₂((1+p)/2),

a monotone decreasing function of p: S = 1 for a fully depolarized (isotropic,
p = 0) field, S = 0 for a fully polarized (pure, p = 1) one. The
**depolarization index** is D_M = p here (single 2-level system), and Réfrégier's
universal E(D) relation is exactly S = h₂((1+D)/2).

The parity split (the theorem)
==============================
Under time reversal the Stokes components carry the parity of their generators:
Q, U are parity-EVEN (built from equal-time u², v², uv — the reversible stretch face,
NR56/NR39) and V is parity-ODD (the lag-odd spin, NR52; the irreversibility carrier,
NR60/61/66). The Pythagorean purity therefore splits by parity WITH NO CROSS TERM:

    p² = (Q²+U²) + V² =: p_even² + p_odd²

so the **irreversible purity fraction**

    f_odd := p_odd² / p² = V² / (Q²+U²+V²)

is a well-defined, frame-invariant (rotation acts on (Q,U) only) number in [0,1]. Two
theorems:

1. **The polarization entropy is a monotone of a single purity; that purity carries a
   parity-labelled irreversible share.** f_odd = 1 ⇒ purely circular ⇒ the whole
   coherent part is the reactive/housekeeping current (NR66); f_odd = 0 ⇒ purely
   linear ⇒ reversible wave polarization (NR56). This is the polarization-optics face
   of the reversible/irreversible EPR split.
2. **A pure wave and a pure vortex have the SAME entropy (0) but opposite parity.**
   Entropy alone cannot tell them apart — f_odd is the discriminator. So the
   (S, f_odd) pair is the minimal complete polarimetric-thermodynamic label of a 2-D
   flow's second-order structure.

The real-data face (deep ocean, ANDRO)
======================================
On NR56's committed 8280-float Stokes cache, per 15° latitude band, form the
ensemble-mean Stokes vector ⟨s⟩ (the band coherency matrix J_band = ½(σ₀+⟨s⟩·σ)),
its degree of polarization p = |⟨s⟩|, polarization entropy S, and irreversible
fraction f_odd. Predictions from the rest of the work:
  * S is HIGH everywhere (the deep float ensemble is strongly depolarized: eddies are
    mutually incoherent) — the mid-latitude eddy field is a near-ideal depolarizer.
  * The EQUATOR is the least depolarized (lowest S / highest p): NR56's wave-polarized
    equatorial deep jets are the coherent (linear) limit, so f_odd is smallest there.
  * f_odd RISES toward the poles (Coriolis makes the residual coherence circular) —
    the parity face of NR66's |V|~|f|.

Honest scope
------------
The band coherency matrix mixes two depolarization sources: genuine per-float partial
polarization AND ensemble incoherence across floats (float-to-float orientation
scatter). p = |⟨s⟩| is therefore a BAND-COHERENCE degree of polarization, not a single
float's — this is the standard "depolarization by incoherent superposition" of optics
and is exactly why S is high; the equator-vs-pole CONTRASTS (not the absolute S) carry
the physics. V is the lag-1 odd correlation (NR52 proxy). f_odd uses the normalized
per-band Stokes means; a bootstrap over floats gives its CI. This is a second-order
(Gaussian) polarimetric description; higher cumulants are out of scope.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STOKES_CACHE = os.path.join(HERE, "data", "nr56_stokes_cache.json")
FIG = os.path.join(HERE, "figures", "102_polarization_entropy.json")


# --------------------------------------------------------------------------- #
# exact coherency-matrix machinery
# --------------------------------------------------------------------------- #
def coherency_matrix(Q, U, V):
    """J = ½(σ₀ + Qσ₃ + Uσ₁ + Vσ₂), the 2×2 Hermitian coherency/density matrix."""
    return 0.5 * np.array([[1.0 + Q, U - 1j * V],
                           [U + 1j * V, 1.0 - Q]], complex)


def _h2(x):
    """Binary Shannon entropy in bits, safe at 0/1."""
    out = 0.0
    for p in (x, 1.0 - x):
        if p > 0.0:
            out -= p * math.log2(p)
    return out


def polarization_entropy(Q, U, V):
    """von Neumann entropy of J (bits): S = h₂((1+p)/2), p = √(Q²+U²+V²)."""
    p = math.sqrt(Q * Q + U * U + V * V)
    p = min(p, 1.0)
    return _h2((1.0 + p) / 2.0), p


def parity_split(Q, U, V):
    """(p_even, p_odd, f_odd): reversible (Q,U) vs irreversible (V) purity."""
    p_even = math.hypot(Q, U)
    p_odd = abs(V)
    p2 = p_even ** 2 + p_odd ** 2
    f_odd = (p_odd ** 2 / p2) if p2 > 0 else float("nan")
    return p_even, p_odd, f_odd


# --------------------------------------------------------------------------- #
# synthetic verification (exact limits + optics cross-checks)
# --------------------------------------------------------------------------- #
def synthetic_checks():
    out = {}
    # pure states: entropy 0, opposite parity
    S_wave, p_wave = polarization_entropy(1.0, 0.0, 0.0)
    S_vort, p_vort = polarization_entropy(0.0, 0.0, 1.0)
    out["pure_wave"] = dict(S=S_wave, p=p_wave, f_odd=parity_split(1, 0, 0)[2])
    out["pure_vortex"] = dict(S=S_vort, p=p_vort, f_odd=parity_split(0, 0, 1)[2])
    # fully depolarized: entropy 1
    S0, p0 = polarization_entropy(0.0, 0.0, 0.0)
    out["isotropic"] = dict(S=S0, p=p0)
    # eigenvalues of J are (1±p)/2 (optics identity), and S matches -Σλlog₂λ
    checks = []
    for Q, U, V in [(0.3, -0.2, 0.4), (0.1, 0.5, -0.6), (0.7, 0.0, 0.2)]:
        J = coherency_matrix(Q, U, V)
        w = np.linalg.eigvalsh(J)
        p = math.sqrt(Q * Q + U * U + V * V)
        S_vn = float(-sum(l * math.log2(l) for l in w if l > 0))
        S_formula, _ = polarization_entropy(Q, U, V)
        checks.append(dict(
            eig_matches=bool(np.allclose(sorted(w), [(1 - p) / 2, (1 + p) / 2])),
            entropy_matches=bool(abs(S_vn - S_formula) < 1e-12),
            hermitian=bool(np.allclose(J, J.conj().T)),
            trace_one=bool(abs(np.trace(J).real - 1) < 1e-12)))
    out["eig_entropy_checks"] = checks
    # parity split is exact Pythagoras and rotation-invariant on (Q,U)
    Q, U, V = 0.36, -0.15, 0.28
    pe, po, f = parity_split(Q, U, V)
    th = 0.7
    Qr = Q * math.cos(2 * th) + U * math.sin(2 * th)
    Ur = -Q * math.sin(2 * th) + U * math.cos(2 * th)
    per, por, fr = parity_split(Qr, Ur, V)
    out["parity_split"] = dict(
        pythagoras=bool(abs(pe ** 2 + po ** 2
                            - (Q ** 2 + U ** 2 + V ** 2)) < 1e-12),
        rotation_invariant_f_odd=bool(abs(f - fr) < 1e-12),
        f_odd=f)
    return out


# --------------------------------------------------------------------------- #
# real data: banded polarization entropy on the NR56 Stokes cache
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


def _band_entropy(Q, U, V, n_boot=400, seed=0):
    """Ensemble coherency: mean Stokes -> S, p, f_odd, with a float bootstrap CI."""
    sQ, sU, sV = Q.mean(), U.mean(), V.mean()
    S, p = polarization_entropy(sQ, sU, sV)
    _, _, f_odd = parity_split(sQ, sU, sV)
    rng = np.random.default_rng(seed)
    n = Q.size
    Sb, fb, pb = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s2, p2 = polarization_entropy(Q[idx].mean(), U[idx].mean(), V[idx].mean())
        _, _, f2 = parity_split(Q[idx].mean(), U[idx].mean(), V[idx].mean())
        Sb.append(s2); fb.append(f2); pb.append(p2)
    q = lambda a, lo: float(np.percentile(a, lo))
    return dict(S=float(S), p=float(p), f_odd=float(f_odd),
                mean_Q=float(sQ), mean_U=float(sU), mean_V=float(sV),
                S_ci=(q(Sb, 16), q(Sb, 84)), f_odd_ci=(q(fb, 16), q(fb, 84)),
                p_ci=(q(pb, 16), q(pb, 84)))


def ocean_entropy():
    lat, Q, U, V = load_stokes()
    edges = np.arange(-75, 76, 15)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lat >= lo) & (lat < hi)
        if m.sum() < 40:
            continue
        e = _band_entropy(Q[m], U[m], V[m], seed=int(lo) + 1000)
        e.update(lat=float(0.5 * (lo + hi)), lat_lo=int(lo), lat_hi=int(hi),
                 n=int(m.sum()))
        bands.append(e)
    # equator vs off-equator contrast (|lat|<10 vs >30)
    eqm = np.abs(lat) < 10.0
    polm = np.abs(lat) > 40.0
    eq = _band_entropy(Q[eqm], U[eqm], V[eqm], seed=7)
    pol = _band_entropy(Q[polm], U[polm], V[polm], seed=8)
    # f_odd vs |f| trend across bands (reactive parity face)
    f = np.array([math.sin(math.radians(b["lat"])) for b in bands])
    fodd = np.array([b["f_odd"] for b in bands])
    from scipy.stats import spearmanr
    rho, p = spearmanr(np.abs(f), fodd)
    # per-FLOAT read-out (far higher power than the 10-band means)
    absf = np.abs(np.sin(np.radians(lat)))
    p2 = Q ** 2 + U ** 2 + V ** 2
    fodd_pf = V ** 2 / (p2 + 1e-12)
    p_pf = np.sqrt(np.minimum(p2, 0.999))
    lam = (1.0 + p_pf) / 2.0
    S_pf = -lam * np.log2(lam) - (1 - lam) * np.log2(1 - lam)
    rho_f, p_f = spearmanr(absf, fodd_pf)
    rho_S, p_S = spearmanr(np.abs(lat), S_pf)
    per_float = dict(
        n=int(lat.size),
        f_odd_vs_absf=dict(rho=float(rho_f), p=float(p_f)),
        entropy_vs_abslat=dict(rho=float(rho_S), p=float(p_S)),
        equator_mean_f_odd=float(fodd_pf[np.abs(lat) < 10].mean()),
        poleward_mean_f_odd=float(fodd_pf[np.abs(lat) > 40].mean()))
    return dict(n_floats=int(lat.size), bands=bands,
                equator=eq, poleward=pol,
                equator_more_polarized=bool(eq["p"] > pol["p"]),
                equator_lower_entropy=bool(eq["S"] < pol["S"]),
                equator_lower_f_odd=bool(eq["f_odd"] < pol["f_odd"]),
                f_odd_vs_absf=dict(rho=float(rho), p=float(p)),
                per_float=per_float)


def verdict(syn, oc):
    v = {}
    v["pure_states_zero_entropy"] = (syn["pure_wave"]["S"] < 1e-12 and
                                     syn["pure_vortex"]["S"] < 1e-12)
    v["pure_states_opposite_parity"] = (syn["pure_wave"]["f_odd"] == 0.0 and
                                        syn["pure_vortex"]["f_odd"] == 1.0)
    v["isotropic_max_entropy"] = abs(syn["isotropic"]["S"] - 1.0) < 1e-12
    v["eig_and_entropy_exact"] = all(c["eig_matches"] and c["entropy_matches"]
                                     and c["hermitian"] and c["trace_one"]
                                     for c in syn["eig_entropy_checks"])
    v["parity_pythagoras_and_invariant"] = (
        syn["parity_split"]["pythagoras"] and
        syn["parity_split"]["rotation_invariant_f_odd"])
    v["ocean_equator_least_depolarized"] = (oc["equator_more_polarized"] and
                                            oc["equator_lower_entropy"])
    v["ocean_equator_most_reversible"] = oc["equator_lower_f_odd"]
    v["ocean_f_odd_rises_with_coriolis"] = (oc["f_odd_vs_absf"]["rho"] > 0)
    pf = oc["per_float"]
    v["ocean_per_float_f_odd_rises"] = (pf["f_odd_vs_absf"]["rho"] > 0 and
                                       pf["f_odd_vs_absf"]["p"] < 1e-6)
    v["ocean_per_float_entropy_rises_poleward"] = (
        pf["entropy_vs_abslat"]["rho"] > 0 and pf["entropy_vs_abslat"]["p"] < 1e-6)
    return v


def analyze():
    syn = synthetic_checks()
    oc = ocean_entropy()
    return {"what": "NR67: the Lagrangian velocity's polarization entropy "
                    "(Cloude–Pottier/Réfrégier) and the parity split of its purity "
                    "p²=p_even²+p_odd² — the reversible/irreversible faces as "
                    "polarization optics; measured banded on ANDRO",
            "synthetic": syn, "ocean": oc, "verdict": verdict(syn, oc)}


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bands = res["ocean"]["bands"]
    lat = [b["lat"] for b in bands]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    S = [b["S"] for b in bands]
    Slo = [b["S_ci"][0] for b in bands]; Shi = [b["S_ci"][1] for b in bands]
    ax[0].fill_between(lat, Slo, Shi, alpha=0.2, color="tab:blue")
    ax[0].plot(lat, S, "o-", color="tab:blue")
    ax[0].axhline(1.0, color="grey", ls=":", lw=1, label="fully depolarized (S=1)")
    ax[0].set_xlabel("latitude [deg]"); ax[0].set_ylabel("polarization entropy S [bits]")
    ax[0].set_title("the deep float ensemble is a near-ideal depolarizer;\n"
                    "the EQUATOR is the coherent (wave-polarized) minimum", fontsize=10)
    ax[0].set_ylim(0.85, 1.005); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    f_odd = [b["f_odd"] for b in bands]
    flo = [b["f_odd_ci"][0] for b in bands]; fhi = [b["f_odd_ci"][1] for b in bands]
    ax[1].fill_between(lat, flo, fhi, alpha=0.2, color="tab:red")
    ax[1].plot(lat, f_odd, "s-", color="tab:red")
    ax[1].set_xlabel("latitude [deg]")
    ax[1].set_ylabel(r"irreversible purity fraction $f_{odd}=V^2/p^2$")
    ax[1].set_title("the parity face: coherent structure turns circular\n"
                    fr"(reactive) toward the poles ($\rho$="
                    fr"{res['ocean']['f_odd_vs_absf']['rho']:+.2f} vs $|f|$)",
                    fontsize=10)
    ax[1].grid(alpha=0.3)
    fig.suptitle("NR67 — Lagrangian polarization entropy and the parity split of "
                 "purity (ANDRO deep ocean)", fontsize=12)
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
    oc = res["ocean"]
    print(f"OCEAN (ANDRO, n={oc['n_floats']}):")
    print(f"  equator : S={oc['equator']['S']:.3f} p={oc['equator']['p']:.3f} "
          f"f_odd={oc['equator']['f_odd']:.3f}")
    print(f"  poleward: S={oc['poleward']['S']:.3f} p={oc['poleward']['p']:.3f} "
          f"f_odd={oc['poleward']['f_odd']:.3f}")
    print(f"  f_odd vs |f|: rho={oc['f_odd_vs_absf']['rho']:+.2f} "
          f"(p={oc['f_odd_vs_absf']['p']:.3f})")
    pf = oc["per_float"]
    print(f"  per-float f_odd vs |f|: rho={pf['f_odd_vs_absf']['rho']:+.3f} "
          f"(p={pf['f_odd_vs_absf']['p']:.1e}); S vs |lat|: "
          f"rho={pf['entropy_vs_abslat']['rho']:+.3f} "
          f"(p={pf['entropy_vs_abslat']['p']:.1e})")
    print("VERDICT:", {k: v for k, v in res["verdict"].items()})
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
