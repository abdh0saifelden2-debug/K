# NR62 — The reversible reference phase is set by time-parity; entropy production is the departure from it

**Claim.** NR60/61 read irreversibility off the *imaginary* part of a
cross-spectrum. That is correct only when the two channels have the same sign
under time reversal. Observables carry a definite time-parity `εᵢ = ±1`
(positions, intensities, temperatures: even; velocities, currents, fluxes:
odd), and the correct Gaussian time-reversal maps `S(ω) → E S(ω)ᵀ E`,
`E=diag(ε)`. So

```
reversible ⟺ S_ij(ω) = εᵢεⱼ S_ij(ω)*   for all i,j,ω,
carrier of irreversibility = Im S12   (εᵢεⱼ=+1, equal parity; NR60/61)
                           = Re S12   (εᵢεⱼ=−1, opposite parity; this NR)
dσ_ij/dω = (1/π) (carrier-coherency)² / (1 − |Coh|²).
```

The reversible *reference phase* is 0° for equal parity and ±90° for
opposite parity; entropy production is the departure from it.

## Theorems (all verified)

1. **The equilibrium oscillator is reversible — but only with parity.** For
   `v = dx/dt` (x even, v odd) the cross-spectrum is `S_xv(ω)=iω S_xx(ω)`,
   *purely imaginary* for any real x-spectrum. The parity-correct (opposite)
   reading gives Re-coherency = 0 → reversible; the naive equal-parity
   reading sees the 90° x–v phase (Im-coherency ≈ 1) and calls the
   conservative oscillation "maximally irreversible." Quadrature between an
   even and an odd variable is the reversible conservative oscillation, not
   dissipation. (Verified: Re-carrier max 0.00, Im-carrier min 0.98.)
2. **Genuine irreversibility survives the correction.** A two-temperature
   coupled pair (two even positions, T₁≠T₂) has σ = 0.333 (equal parity, Im
   carrier), rising with ΔT, and σ = 0 at T₁=T₂; a non-reciprocally coupled
   oscillator (opposite parity, Re carrier) has σ = 6.0, rising with the
   non-reciprocity. The parity correction removes the reversible reference,
   never a real current. The parity EPR is non-negative on random systems.

## Real solar (committed NR46 cache: 26 y SOHO GOLF × VIRGO-green, I–V, 395 p-mode bins + background + 13 epochs; BiSON × GOLF V–V control)

Intensity I is even, Doppler velocity V is odd → the I–V pair is **opposite
parity**, reversible reference = the adiabatic −90°.

* **The p-modes are nearly reversible; the irreversibility is the
  non-adiabatic departure.** Mode-comb median phase −104° — only 15.8° from
  the −90° reference: the conservative oscillation sits at the reversible
  reference. The parity-correct EPR density (median 0.19) is the small
  non-adiabatic (radiative/convective) departure; the **naive equal-parity
  reading (median 1.62) over-counts the solar irreversibility 8.5×** by
  mistaking the conservative −90° oscillation for dissipation.
* **The non-adiabaticity is stable over two solar cycles.** Across 13 annual
  epochs the mode phase holds to std 2.2° and the irreversibility proxy
  cos²φ to 0.09 ± 0.022 — a fixed material property, not activity-driven.
* **The convective background is the broadband entropy source.** Background
  cross-phase sits far from ±90° (median departure 85°) — large in-phase
  (irreversible) component, exactly where NR46 placed the channel-local
  convective slow clock. Convection, not the modes, is where the Sun pays.
* **Parity control (BiSON × GOLF, V–V, equal parity).** Here the reference is
  0° and the carrier is the *quadrature*: the top-quintile pair has high
  coherence (0.91) with a small −15.7° quadrature = NR48's −10.3 s
  inter-instrument all-pass delay — an *instrumental* irreversibility,
  flagged only by the equal-parity criterion.

## Placement

NR46's −118.8° I–V phase, NR48's −10.3 s V–V delay, and NR60/61's
quad-spectrum EPR are one picture once parity is included: the reversible
reference is a parity choice, and the two solar pairs measure two different
irreversibilities (solar non-adiabaticity vs instrument delay) because they
have two different parity products. This is an important correction to NR61
(whose ImCoh criterion silently assumed equal parity). Mainstream anchors:
Maes / Seifert (even–odd variables); Risken (Fokker–Planck detailed balance
with parity); Weiss 1975; Jiménez et al. 1999.

**Artifacts:** `figures/97_parity_epr.json` (all 10 verdicts true); 13 unit
proofs in `tests/test_parity_epr.py`; offline-safe (committed NR46 cache).
Reproduce: `python general_two_clocks/new_relationships39.py`.
