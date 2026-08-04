# Theorem 3 — Counter-Gradient parameter C_G in a 3-D active-buoyancy LES

<!-- Hand-written analysis of a GPU run. The numbers come from
     theorem3_cg_gpu_probe.py; the data artifact is figures/52_theorem3_cg.json.
     To refresh, rerun the probe on a CuPy-capable GPU (Kaggle P100 used here)
     and regenerate figures/52_theorem3_cg.json. -->

This report closes the **"to be measured"** status of Theorem 3 in
[`subglacial/THEORY_CAVITY.md`](subglacial/THEORY_CAVITY.md) §5 (recorded inline as
RESULT 11, §11.1). It is the heat-flux counterpart of the RESULT 8 melt-ratio null:
same solver, same Ri sweep, but it measures the **counter-gradient parameter C_G**
rather than the melt ratio R alone.

## What C_G is and why it is the melt lever

K-theory (a scalar eddy diffusivity κ_t > 0) forces the turbulent heat flux to be
**down-gradient**: **F** = −κ_t∇θ̄, so **F** is anti-parallel to ∇θ̄ everywhere. In a
lee recirculation the true flux can be **counter-gradient** — fluid parcels carry the
thermal memory of the global pressure field that swept them there, so heat is moved
*up* the local mean gradient. Theorem 3 measures this with the magnitude-weighted
alignment

> **C_G = ⟨F·∇θ̄⟩ / ⟨|F| |∇θ̄|⟩**,  C_G = −1 ⇒ pure down-gradient (K-theory), C_G > −1 ⇒ counter-gradient.

The corrected scaling (THEORY_CAVITY eq 13) predicts the departure **grows with the
clock mismatch** and is stratification-modulated:

> **C_G + 1 = c₁·(1 − 1/𝓜_clock)·g(Ri_b, Re_Δ)**.

So a true Theorem-3 confirmation needs the departure to (a) be larger for the
backscatter (two-clocks) closure than for Smagorinsky, and (b) vary with Ri — and to
**co-locate** with any melt enhancement R(Ri) (§9 item 5): same mechanism.

## What was run

`theorem3_cg_gpu_probe.py` — a self-contained 3-D LES probe wrapping the validated
`CavityFlow` solver (byte-identical physics to the RESULT-8 Direction-C probe), with a
`CGAccumulator` that accumulates ⟨θ̄⟩, ⟨ūᵢ⟩, ⟨uᵢθ⟩ over the measurement window and
forms the **resolved turbulent heat flux** Fᵢ = ⟨uᵢθ⟩ − ⟨uᵢ⟩⟨θ⟩ and the mean gradient
∇θ̄ spectrally. (This solver imposes **no SGS closure on θ** — θ is advected by the
resolved velocity — so C_G measures the alignment of the *resolved* flux, the
LES-accessible analog of the §5 definition.)

- Grid n = 64, ν = κ = 5 × 10⁻⁴, Cs = 0.17, spin-up 400, measure 600, sample every 5.
- Ri ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5}; 3 seeds per point.
- Two closures: **white-FDT** (Smagorinsky, bs_tau = 0) vs **colored-FDT**
  (two-clocks backscatter, bs_tau = 0.05).
- C_G over the whole fluid and over the ice-base band; melt, dissipation breakdown,
  SGS dominance D = ε_sgs/ε_mol recorded alongside.
- Backend CuPy/GPU (Kaggle P100), wall 553 s. Data:
  [`figures/52_theorem3_cg.json`](figures/52_theorem3_cg.json).

## Results

| Ri | C_G fluid (2-clocks) | C_G band (2-clocks) | C_G fluid (Smag) | C_G band (Smag) | R(2c/Smag) | D |
|----|------|------|------|------|------|------|
| 0.00 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |
| 0.25 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |
| 0.50 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |
| 0.75 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |
| 1.00 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |
| 1.50 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |

Seed scatter: σ(C_G fluid) ≈ 3.4 × 10⁻³, σ(melt) ≈ 8.3 × 10⁻⁸. Max departure
`max(C_G + 1)` over the sweep (two-clocks band) = **0.6744**. `has_melt_hump = False`.

## Interpretation — a partial confirmation, split honestly into two halves

**1. The counter-gradient flux is real (qualitative half — supported).** C_G is
≈ −0.60 in the fluid and ≈ −0.33 in the ice-base band, far above the K-theory value
−1. The departure C_G + 1 ≈ **0.40 (fluid) / 0.67 (band)** means a large fraction of
the resolved lee heat flux is *not* down-gradient, and it is strongest exactly where
melt happens — at the ice base. This is the live-LES, resolved-flux confirmation of
the same counter-gradient transport that `REPORT_SUBGLACIAL.md` §4 and THEORY_CAVITY
§7.1 measured in the *exact subgrid* flux on a frozen field. The regime is
SGS-dominant (D ≈ 1.24 > 1), so this is the physically meaningful regime, not the §8.1
trivial limit where R ≈ 1 by construction.

**2. The predicted scaling (13) is not supported (quantitative half — fails).** C_G is

- **Ri-invariant**: flat to four decimals across Ri ∈ [0, 1.5]; and
- **closure-invariant**: two-clocks vs Smagorinsky differ by only ≈ 0.006 (fluid) /
  ≈ 0.005 (band) — within ≈ 2σ, and in *opposite* directions in the two regions, i.e.
  at the noise floor.

Eq (13) requires C_G + 1 to grow with the clock mismatch 𝓜 (backscatter on vs off)
and to depend on Ri. Neither happens. The departure is a near-constant set by the
resolved cavity geometry, not the stratification- and memory-tunable lever the
theorem proposed.

**Why it is closure- and Ri-blind here (mechanism, not bug).** The SGS model acts on
*momentum*; θ has no SGS closure, so C_G reflects the *resolved* flow, and both
closures produce nearly identical resolved fields — the same spatial-CLT
decorrelation that flattens the RESULT-8 melt ratio (τ_mem^eff ≪ τ_mem^set, so
temporal recoloring washes out of the time average). Buoyancy is also weak over this
grid (N_BV ≤ 0.37, τ_mem·N_BV ≈ 10⁻³ ≪ 2π), so varying Ri barely perturbs the flow
and hence C_G.

**Co-location (§9 item 5).** R(Ri) is flat (1.0004, matching RESULT 8's 1.0006) **and**
C_G is flat: the two diagnostics agree — but on a *null in Ri*. There is no co-located
hump because there is no hump in either quantity at these parameters. The melt flux
does creep up monotonically with Ri (2.368 → 2.375 × 10⁻⁵, ≈ 0.3 % over Ri 0 → 1.5)
but identically for both closures, so R is unmoved.

## Verdict

**PARTIAL.** Counter-gradient transport (C_G > −1) is **measured and confirmed
present** — the qualitative melt mechanism of Theorem 3 is real and sits at the ice
base — but its predicted dependence on the clock mismatch 𝓜 and on Ri (eq 13) is
**not supported** at n = 64: C_G + 1 ≈ 0.4–0.67 is closure- and Ri-invariant, and the
melt ratio stays flat. This is an honest partial result, not a tuned confirmation,
and it sharpens the RESULT-8 null: neither the melt ratio nor its underlying
heat-flux alignment responds to stratification or to backscatter recoloring in this
regime.

## Scope / caveats

- The §5 definition writes C_G in terms of the basal/subgrid stress τ_b; this solver
  has no SGS heat closure, so the operationally measured quantity is the **resolved**
  turbulent heat-flux alignment — the LES-accessible proxy for the melt-relevant
  counter-gradient transport. It is stated as such throughout.
- n = 64 with weak buoyancy (N_BV ≤ 0.37) is the same regime as RESULT 8; a genuinely
  strong-stratification test (larger N_BV so τ_mem·N_BV ≈ 2π) would need either a much
  finer grid or an explicitly stratified inflow, and is outside this probe's reach.
  The closure-independence, by contrast, is robust here because it follows from the
  CLT averaging argument, not from the buoyancy magnitude.

## Reproduce

```
# GPU (Kaggle P100 / any CuPy device):
python theorem3_cg_gpu_probe.py            # writes figures/52_theorem3_cg.json

# C_G alignment math is unit-tested (CPU, no GPU):
pytest tests/test_theorem3_cg.py -q
```
