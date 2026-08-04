# NR67 — the Lagrangian velocity has a polarization entropy, and its purity splits by time parity

**Module** `general_two_clocks/new_relationships44.py` ·
**data** the committed NR56 ANDRO Stokes cache (`data/nr56_stokes_cache.json`, 8280 floats) ·
**figure** `general_two_clocks/figures/102_polarization_entropy.json` + `.png` ·
**tests** `general_two_clocks/tests/test_polarization_entropy.py` (16)

## The relationship

NR56 gave the 2-D Lagrangian velocity its Stokes parameters (I,Q,U,V) and the
wave/vortex split; NR60/61/66 made the spin an entropy-production carrier. NR67 supplies
the single scalar those imply and that polarization optics has used for 30 years while
oceanography has not: the **von Neumann / Cloude–Pottier polarization entropy** of the
coherency matrix.

Build the 2×2 Hermitian coherency (density) matrix from the normalized Stokes vector:

`J = ½(σ₀ + Qσ₃ + Uσ₁ + Vσ₂) = ½[[1+Q, U−iV],[U+iV, 1−Q]]`, tr J = 1,

eigenvalues `λ± = (1±p)/2` with **degree of polarization** `p = √(Q²+U²+V²)`.
The **polarization entropy** (Cloude & Pottier 1997; Réfrégier 2005) is
`S = −λ₊log₂λ₊ − λ₋log₂λ₋ = h₂((1+p)/2)` — S = 1 fully depolarized (isotropic), S = 0
fully polarized (pure). All exact, verified to 1e-12.

## The parity split (the theorem)

Q,U are parity-**even** (equal-time stretch face, reversible; NR56/NR39) and V is
parity-**odd** (lag-odd spin, the irreversibility carrier; NR52/NR60/66). The purity is
Pythagorean **with no cross term**:

`p² = (Q²+U²) + V² = p_even² + p_odd²`,  `f_odd := V²/(Q²+U²+V²) ∈ [0,1]`,

so the **irreversible purity fraction** `f_odd` is a frame-invariant (rotation acts on
(Q,U) only) number. Two theorems:
1. **Entropy is a monotone of one purity; that purity carries a parity-labelled
   irreversible share.** `f_odd = 1` ⇒ pure circular ⇒ the coherent part is the
   reactive/housekeeping current (NR66); `f_odd = 0` ⇒ pure linear ⇒ reversible wave
   polarization (NR56). This is the polarization-optics face of the
   reversible/irreversible EPR split.
2. **A pure wave and a pure vortex have the SAME entropy (0) but opposite parity** —
   entropy alone cannot separate them; `f_odd` is the discriminator. `(S, f_odd)` is the
   minimal complete polarimetric-thermodynamic label of a 2-D flow's second-order
   structure.

## Measured (ANDRO deep ocean, 8280 floats)

| | equator (\|φ\|<10°) | poleward (\|φ\|>40°) |
|---|---|---|
| degree of polarization p | **0.36** | 0.05 |
| polarization entropy S [bits] | **0.90** | 0.998 |
| irreversible fraction f_odd | **0.00** | 0.081 |

Per-float trends (far higher power than the 10-band means):
- **f_odd rises with \|f\| = \|sin φ\|**: Spearman ρ = +0.30, **p = 3×10⁻¹⁷¹**
  (equatorial mean 0.06 → poleward 0.26).
- **entropy rises poleward** (S vs \|φ\|): ρ = +0.36, **p = 8×10⁻²⁵¹**.

## Reading

The deep float ensemble is a **near-ideal depolarizer** (S ≈ 1 everywhere: eddies are
mutually incoherent — the optics statement of a turbulent field). Against that ceiling,
the **equator is the coherent minimum**: NR56's wave-polarized equatorial deep jets are
the linear (reversible, f_odd ≈ 0) pure limit. Moving poleward, the residual coherence
turns **circular** — the parity face of NR66's Coriolis reactive current — so f_odd
climbs and entropy saturates. The whole latitude structure of the ocean's Lagrangian
second-order statistics is one Réfrégier E(D) curve with a parity label: reversible
(wave) coherence at the equator, irreversible (reactive-circular) coherence at the poles.

## Honest scope

The band/ensemble coherency mixes genuine per-float partial polarization with float-to-
float incoherence, so `p = |⟨s⟩|` is a **band-coherence** degree of polarization, not a
single float's — this is standard optics "depolarization by incoherent superposition"
and is exactly why S sits near 1; the equator-vs-pole **contrasts** carry the physics,
not the absolute S. The per-float read-out (ρ, p above) is the decisive test and does
not depend on ensemble averaging. V is the lag-1 odd correlation (NR52 proxy). Second-
order (Gaussian) polarimetry only; higher cumulants out of scope.
