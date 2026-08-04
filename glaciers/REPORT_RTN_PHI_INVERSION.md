# RESULT 18 — the §H.1.1 φ-area inversion is a unique, unbiased estimator

**Status:** the §H.1.1 claim that the gauge-RTN *intruded area* is a "calibrated
inverse for the basal-water fraction φ" is now **[VERIFIED — synthetic]** as an
estimator: the area↦φ map is **monotone (so unique)**, the underlying
critical-thickness threshold is **exact**, and a finite mapped survey inverts for
φ **without bias** (spread ∝ 1/√N). Harness
`validation/synthetic/rtn_phi_synthetic.py`, tests in
`tests/test_validation_synthetic.py` (5/5). No external data; no GPU.

## What was open

The gauge fix collapses RTN to an Atwood ratio `RTN = (ρ_w/φρ_i)·(d_base/H)`, so
`RTN>1 ⇔ d_base/H > φ·ρ_i/ρ_w` — `g` and the atmosphere cancel. §H.1.1 (#1) reads
the **grounded-ice area with `RTN>1`** as a monotone inverse for φ, and
`validation/external/rtn_phi_calibration.py` reports that inverse on real Bedmap2
as a steep table (3.85 % of grounded ice at φ=0.70 → 0.11 % at φ=0.98; sensitivity
≈0.13 %-area per +0.01 in φ near φ=0.9). But the *inversion itself* was asserted
from the deterministic table: it was never shown that the area↦φ map is **unique**
(strictly monotone) nor that recovering φ from a **finite** mapped survey is
**unbiased**. Those are estimator properties, and they were untested.

## What this result establishes

The harness reuses the **real driver math** (`build_rtn`, `classify`, `RHO_W`,
`RHO_I` from `run_rtn_bedmap2`), so the equation under test is identical to the
field calibration. It plants a grounded-cell population whose ratio
`r = d_base/H` is lognormal, tuned so the `RTN>1` area spans the same range as
Antarctica (3.83 % → 0.118 % across φ=0.70→0.98):

1. **Monotone & unique.** `A(φ)` is strictly decreasing over the whole φ-grid, so
   a mapped area selects exactly one φ — the inverse is single-valued.
2. **Critical-thickness identity is exact.** `RTN>1` reproduces
   `H < H* = (ρ_w/φρ_i)·d_base` **cell-for-cell (0 mismatch)** at φ ∈
   {0.70…0.98}. This is the φ-parameterised threshold the MISI-margin refinement
   (§H.1.1 #2, `H* = H_flot/φ`) rests on.
3. **Population inverse is faithful.** Inverting the monotone `A(φ)` curve at
   planted φ ∈ {0.74, 0.84, 0.90, 0.96} recovers φ to interpolation precision
   (max abs err ≈ 0).
4. **Finite-sample inverse is unbiased.** With only `N = 4000` sampled cells the
   area estimate has sampling scatter, yet over 200 seeds the recovered φ̂ has
   **mean 0.905 at φ_true = 0.90 (bias 5×10⁻³)** and its spread falls from
   `std = 0.0245` to `0.0163` when `N` doubles — ratio **0.66**, right at the
   `1/√2 ≈ 0.707` law. So a mapped finite survey inverts for φ without bias.
5. **Sensitivity has the right sign.** `dA/dφ < 0` everywhere (a wetter bed ⇒ more
   intrusion), matching the driver's reported `dfrac_dφ`.

## Why this matters

This is the §G.3 analogue of the plant-and-recover guarantees the repo already
has for the **RTN classifier** (`rtn_synthetic`), the **GL-migration
discriminant** (`glmig_synthetic`, RESULT 16) and the **§G.4 lag estimator**
(RESULT 17). Together they certify that each constant-free §G readout is a
faithful estimator before any real-data verdict is quoted. Here specifically: the
headline φ-inversion table is not just monotone *as drawn* — the map is provably
unique and the finite-survey inverse is provably unbiased.

## Honest scope

This validates the **inversion (the estimator)**, not the physics. It does **not**
assert any particular φ for Antarctica, nor that the real intruded-area map is
free of model error (resolution, the `d_base = max(0,−bed)` approximation, the
single continent-effective φ). The synthetic ratio distribution is *calibrated to
span the same area range* as Bedmap2, not to reproduce its interior shape — the
mid-curve values differ from the real table and are not claimed to match. The
guarantee is structural: *given* the gauge-RTN equation, the area↦φ map is a
unique, unbiased, sign-correct readout.
