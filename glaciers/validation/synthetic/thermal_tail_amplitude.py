r"""§G.4 thermal-tail *amplitude* — earn the "subdominant tail" claim.

Context
-------
The §G.4 memory sliding law carries an explicit ice-thermal memory term,

    τ_b(t) = C[N(t) + δp_thermal/g] u_b^{1/m} + ∫₀ᵗ K_ice(t−τ) τ_b(τ) dτ,

and §B.2 derives that kernel in closed form (``ice_kernel_synthetic.py``):

    K_ice(τ) = G(τ)/(ρ_i L),
    G(t) = A[ erfc(√(t/4τ_d)) − 2√(τ_d/πt) e^{−t/4τ_d} ],
    A = k_th θ_far V̄²/(2κ²),   τ_d = κ/V̄².

§G.4 then *reassigns* the lag-setting mechanism to the hydraulic impedance kernel
``K_hydraulic`` and keeps the thermal term "only as a subdominant tail". That
**amplitude** statement was asserted, not quantified. This module earns it.

The deliverable — the thermal weight is the Stefan number [DERIVED]
------------------------------------------------------------------
A memory kernel's influence on the quantity it modifies is set by its DC gain
``∫₀^∞ K dτ`` (the steady response to a unit-step forcing). For the ice kernel the
§B.2 normalisation cross-check gives the DC gain of ``G`` in closed form,
``∫₀^∞ G dτ = H(0) = −ρc θ_far``, so the **dimensionless** weight of the thermal
memory term in the sliding law is, exactly,

    W_thermal ≡ ∫₀^∞ K_ice dτ = −ρc θ_far/(ρ_i L) = c_i|θ_far|/L  ≡  St,      (★)

the **Stefan number** of the far-field basal undercooling ``θ_far`` (using
``ρc = ρ_i c_i`` for ice). This is mechanism-independent: ice carries latent heat
``L`` that dwarfs the sensible heat ``c_i|θ_far|`` available over any physical
undercooling, so ``St ≪ 1``. For ``θ_far`` from 0.1 K (near-temperate ice-stream
bed) to 10 K (a generous cold-margin bound) ``St`` spans only ``6×10⁻⁴ … 6×10⁻²``.
The dominant hydraulic kernel ``K_hydraulic(τ) = (1/RC)e^{−τ/RC}`` is, by
construction, a *unit-DC-gain* memory kernel (``∫₀^∞ K_hydraulic dτ = 1``). Hence

    W_thermal / W_hydraulic = St ≤ 0.06,

i.e. the ice-thermal term is at most a few-percent correction to the basal stress,
**subdominant by construction** — now a derived consequence of ``St ≪ 1`` rather
than an assertion. Three independent facts reinforce this at the *observed* lag
band 0.02–2 yr (§H.2):

  1. **Band-overlapping weight is even smaller.** The fraction of ``St`` that the
     tail deposits inside 0.02–2 yr, ``∫_{0.02yr}^{2yr} K_ice dτ / St``, is
     ``2×10⁻⁴ … 0.2`` across V̄ = 0.001–1 m yr⁻¹, so the *in-band* thermal weight
     is ``≲ 1 %``.
  2. **Skin depth ≪ ice thickness.** ``δ_skin = √(κP/π)`` is 0.5–4.8 m at
     P = 0.02–2 yr (reproducing the §G.4 numbers) ≪ H ≈ 2000 m, confirming the
     semi-infinite (power-law, single-cutoff) reading.
  3. **Wrong shape.** The semi-infinite impulse response is monotone decaying
     (argmax at τ→0⁺, no interior peak), so even at full amplitude it cannot make
     the observed rise-to-a-peak surge — independent of the (small) amplitude.

What this earns (and what it does not)
--------------------------------------
EARNS: the §G.4 "subdominant thermal tail" as a **[DERIVED]** quantitative bound —
the thermal memory weight equals the Stefan number ``St = c_i|θ_far|/L ≤ 0.06``,
vs the unit-gain hydraulic kernel. Rests on the already-verified §B.2 closed-form
kernel (this module integrates *that* kernel; it does not re-plant one).

DOES NOT EARN: a site-specific thermal correction — ``θ_far`` and V̄ are problem
inputs (per §B.2), so ``St`` is bounded, not pinned, without local data. The bound
is what the "subdominant" claim needs.

References
----------
Cuffey & Paterson (2010) (ice ``c_i, k_th, L``); §B.2 / ``ice_kernel_synthetic``
(closed-form kernel + DC-gain cross-check); Stearns et al. (2008); Siegfried et
al. (2016) (observed 0.02–2 yr band).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ice_kernel_synthetic import kernel_G  # noqa: E402  (the §B.2 closed-form kernel)

SEC_PER_YR = 365.25 * 86400.0

# --- ice thermal constants (Cuffey & Paterson 2010) --------------------------
RHO_I = 917.0          # ice density                    [kg m⁻³]
C_ICE = 2009.0         # specific heat (≈ −10 °C)        [J kg⁻¹ K⁻¹]
L_FUSION = 3.34e5      # latent heat of fusion           [J kg⁻¹]
K_TH = 2.1             # ice thermal conductivity        [W m⁻¹ K⁻¹]
RHO_C = RHO_I * C_ICE  # volumetric heat capacity        [J m⁻³ K⁻¹]
KAPPA = K_TH / RHO_C   # thermal diffusivity ≈ 1.14e-6   [m² s⁻¹]

OBS_BAND_YR = (0.02, 2.0)    # observed post-drainage surge-lag band (§H.2)
H_ICE = 2000.0               # representative ice thickness [m] (semi-infinite check)


# --- the derived amplitude (★) -----------------------------------------------
def stefan_weight(theta_far, c_i=C_ICE, L=L_FUSION):
    """Closed-form thermal-memory weight  W_thermal = c_i|θ_far|/L = St  (★).

    Equals ``∫₀^∞ K_ice dτ`` via the §B.2 DC gain ``−ρc θ_far/(ρ_i L)`` with
    ``ρc = ρ_i c_i``. ``θ_far`` is the far-field basal undercooling [K] (sign
    irrelevant — only its magnitude sets the weight).
    """
    return c_i * abs(theta_far) / L


def _trapz(y, x):
    """Version-independent trapezoid (numpy 2.0 dropped ``np.trapz``)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def thermal_kernel_dc_gain(theta_far, Vbar, kappa=KAPPA, rho_c=RHO_C,
                           rho_i=RHO_I, L=L_FUSION, n_oct=24, m=200000):
    """Numerical ``∫₀^∞ K_ice dτ`` of the §B.2 kernel — should equal ``St`` (★).

    Integrates ``G`` out to ``≫ τ_d`` in the ``u=√τ`` variable (cancels the
    integrable ``τ^{−1/2}`` cusp), then divides by ``ρ_i L``. ``Vbar`` in m s⁻¹.
    """
    tau_d = kappa / Vbar**2
    u = np.linspace(0.0, np.sqrt(n_oct * tau_d), m)
    t = u**2
    G = kernel_G(t, kappa, Vbar, theta_far, rho_c)
    integrand = 2.0 * u * G                     # G dτ = G·2u du; finite at u=0
    integrand[0] = 0.0
    return _trapz(integrand, u) / (rho_i * L)


def band_weight(theta_far, Vbar, kappa=KAPPA, rho_c=RHO_C, rho_i=RHO_I,
                L=L_FUSION, band=OBS_BAND_YR, m=300000):
    """In-band thermal weight ``∫_{band} K_ice dτ`` [dimensionless].

    The fraction of the thermal tail that overlaps the observed surge-lag band;
    ``band`` in years. No singularity inside the band (lower limit > 0), so a
    dense linear grid is accurate.
    """
    t = np.linspace(band[0] * SEC_PER_YR, band[1] * SEC_PER_YR, m)
    G = kernel_G(t, kappa, Vbar, theta_far, rho_c)
    return _trapz(G, t) / (rho_i * L)


def skin_depth_m(period_yr, kappa=KAPPA):
    """Thermal skin depth ``δ = √(κP/π)`` [m] for a forcing of period ``P`` [yr]."""
    return float(np.sqrt(kappa * period_yr * SEC_PER_YR / np.pi))


def impulse_is_monotone(theta_far=-1.0, Vbar=0.1 / SEC_PER_YR, kappa=KAPPA,
                        rho_c=RHO_C, t_end_yr=3.0, n=4000):
    """True if the global maximum of |G| sits at the front sample (argmax==0).

    This verifies the physically relevant claim — *no interior peak* — rather than
    strict sample-to-sample monotonicity: it only checks that the largest |G| is at
    t→0⁺, not that the sequence is everywhere decreasing. For the §B.2 semi-infinite
    diffusive kernel (a t^{-1/2} cusp at the front) |G| is in fact monotone-decaying
    over the physical parameter range, so argmax==0 ⇔ no interior peak ⇒ the kernel
    cannot make the observed rise-to-a-peak surge."""
    t = np.linspace(1e-4 * SEC_PER_YR, t_end_yr * SEC_PER_YR, n)
    g = np.abs(kernel_G(t, kappa, Vbar, theta_far, rho_c))
    return bool(np.argmax(g) == 0)


# --- hydraulic kernel weight (the dominant term, for the ratio) --------------
def hydraulic_kernel_dc_gain():
    """DC gain of the §G.4 hydraulic impedance kernel ``K_hydraulic(τ) =
    (1/RC)e^{−τ/RC}``: ``∫₀^∞ (1/RC)e^{−τ/RC} dτ = 1`` exactly, for any ``R, C``.

    So it is a unit-DC-gain memory kernel and ``W_thermal/W_hydraulic = St``.
    """
    return 1.0


# --- literature-range sweep --------------------------------------------------
SWEEP_RANGES = {
    # (lo, hi, "log"/"lin")
    "theta_far_K": (0.1, 10.0, "log"),      # far-field basal undercooling [K]
    "Vbar_m_yr":   (1.0e-3, 1.0, "log"),    # mean basal melt/ablation speed [m yr⁻¹]
    "kappa":       (1.0e-6, 1.4e-6, "lin"),  # ice thermal diffusivity [m² s⁻¹]
}


def _sample(rng, lo, hi, kind):
    if kind == "log":
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def sweep(n=3000, seed=0):
    """Monte-Carlo over literature ranges → distributions of ``St`` and in-band
    thermal weight, and the fraction of parameter space that stays subdominant."""
    rng = np.random.default_rng(seed)
    sts, wbands, ratios = [], [], []
    for _ in range(n):
        th = _sample(rng, *SWEEP_RANGES["theta_far_K"])
        vb = _sample(rng, *SWEEP_RANGES["Vbar_m_yr"]) / SEC_PER_YR
        ka = _sample(rng, *SWEEP_RANGES["kappa"])
        st = stefan_weight(th)
        wb = band_weight(-th, vb, kappa=ka)
        sts.append(st)
        wbands.append(wb)
        ratios.append(wb / st if st > 0 else 0.0)
    sts, wbands, ratios = map(np.array, (sts, wbands, ratios))
    pct = (5, 50, 95)
    return {
        "n": n,
        "St_pct": {q: float(np.percentile(sts, q)) for q in pct},
        "St_max": float(sts.max()),
        "frac_St_below_0.1": float(np.mean(sts < 0.1)),
        "frac_St_below_0.06": float(np.mean(sts < 0.06)),
        "band_weight_pct": {q: float(np.percentile(wbands, q)) for q in pct},
        "band_weight_max": float(wbands.max()),
        "frac_band_below_0.01": float(np.mean(wbands < 0.01)),
        "band_over_St_pct": {q: float(np.percentile(ratios, q)) for q in pct},
    }


def run():
    """Baseline identity check + skin depths + shape + literature sweep."""
    theta_far, Vbar = -1.0, 0.1 / SEC_PER_YR     # representative ice-stream bed
    st = stefan_weight(theta_far)
    dc = thermal_kernel_dc_gain(theta_far, Vbar)
    identity_rel_err = abs(dc - st) / st

    skin = {P: skin_depth_m(P) for P in (0.02, 1.0, 2.0)}
    monotone = impulse_is_monotone(theta_far, Vbar)
    wband = band_weight(theta_far, Vbar)
    sw = sweep()

    ok = bool(
        identity_rel_err < 1e-3                       # (★) holds numerically
        and stefan_weight(-10.0) < 0.1                # St ≤ 0.06 even at 10 K
        and max(skin.values()) < 0.05 * H_ICE         # skin depth ≪ ice thickness
        and monotone                                  # no interior peak
        and sw["St_max"] < 0.1                         # whole sweep subdominant
        and sw["band_weight_max"] < 0.05               # in-band weight ≪ 1
    )
    return {
        "stefan_weight_baseline": st,
        "kernel_dc_gain_numeric": dc,
        "identity_rel_err": identity_rel_err,
        "St_at_10K": stefan_weight(-10.0),
        "hydraulic_dc_gain": hydraulic_kernel_dc_gain(),
        "weight_ratio_thermal_over_hydraulic_at_10K":
            stefan_weight(-10.0) / hydraulic_kernel_dc_gain(),
        "skin_depth_m": skin,
        "ice_thickness_m": H_ICE,
        "band_weight_baseline": wband,
        "impulse_monotone_no_peak": monotone,
        "sweep": sw,
        "obs_band_yr": OBS_BAND_YR,
        "pass": ok,
    }


def main():
    r = run()
    sw = r["sweep"]
    print("=== §G.4 thermal-tail amplitude — earn the 'subdominant tail' claim ===")
    print(f"  W_thermal = ∫K_ice dτ = c_i|θ_far|/L = St  (θ_far=1 K) : {r['stefan_weight_baseline']:.4e}")
    print(f"    numeric ∫ of §B.2 kernel               : {r['kernel_dc_gain_numeric']:.4e}"
          f"   (rel-err {r['identity_rel_err']:.2e})")
    print(f"  St at θ_far=10 K (generous cold bound)   : {r['St_at_10K']:.4e}   (≤ 0.06)")
    print(f"  hydraulic kernel DC gain (unit by constr): {r['hydraulic_dc_gain']:.1f}")
    print(f"  ⇒ W_thermal/W_hydraulic ≤                : {r['weight_ratio_thermal_over_hydraulic_at_10K']:.4e}")
    print("  skin depth √(κP/π) [m] (≪ H=2000 m):")
    for P, d in r["skin_depth_m"].items():
        print(f"      P={P:>5} yr -> {d:5.2f} m")
    print(f"  in-band ∫_(0.02–2yr) K_ice dτ (θ=1K)     : {r['band_weight_baseline']:.3e}")
    print(f"  impulse response monotone (no peak)      : {r['impulse_monotone_no_peak']}")
    print("  --- literature sweep (n={}) ---".format(sw["n"]))
    print(f"    St      p5/p50/p95 : {sw['St_pct'][5]:.2e} / {sw['St_pct'][50]:.2e} / {sw['St_pct'][95]:.2e}"
          f"   (max {sw['St_max']:.2e})")
    print(f"    band wt p5/p50/p95 : {sw['band_weight_pct'][5]:.2e} / {sw['band_weight_pct'][50]:.2e}"
          f" / {sw['band_weight_pct'][95]:.2e}   (max {sw['band_weight_max']:.2e})")
    print(f"    frac St<0.1 / St<0.06 : {sw['frac_St_below_0.1']:.3f} / {sw['frac_St_below_0.06']:.3f}")
    print(f"  PASS : {r['pass']}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
