# RESULT 19 — the §H.1.1 hydrology-corrected MISI margin band, calibrated

**Status:** the §H.1.1 (#2) claim — `RTN=1 ⇔ H < H* = H_flot/φ`, so a pressurized
bed (φ<1) opens a *grounded-but-intrudable* band `H_flot < H < H*` **inland** of
the classical flotation line, of width **∝(1−φ)** — is now **[VERIFIED —
synthetic]** as a structural/scaling statement. Harness
`validation/synthetic/rtn_phi_synthetic.py::run_misi_band`, tests in
`tests/test_validation_synthetic.py` (4/4). No external data; no GPU. This is the
companion to RESULT 18 (§H.1.1 #1, the area↦φ inversion); together they close
both constant-free consequences of the gauge RTN.

## What was open

§H.1.1 (#2) reframes MISI: standard marine-ice-sheet instability keys off
flotation (φ=1, `H_flot = (ρ_w/ρ_i)·d_base`), but the gauge RTN says a
pressurized bed pre-conditions ocean intrusion **earlier**, at the larger local
critical thickness `H* = (ρ_w/φρ_i)·d_base = H_flot/φ`. The driver
`rtn_phi_calibration.py` reports the consequence on real Bedmap2 as point numbers
(at φ=0.9: a **152,500 km²** band, median **6 km** from the GL, reaching **42 km**
inland). But the underlying *structure* — that this band is exactly
`grounded ∧ RTN>1`, that its per-cell width is exactly `(1−φ)/φ`, and that its
**area scales like (1−φ)** (vanishing as φ→1) — was asserted, never calibrated.

## What this result establishes

Reusing the **real** `build_rtn`/`classify` math on the same planted population
as RESULT 18:

1. **Per-cell width is exactly `(1−φ)/φ`.** `(H*−H_flot)/H_flot = 1/φ − 1`
   holds to machine precision (`max abs err ≈ 1.4×10⁻¹⁶`) for every cell with
   `d_base>0`, independent of the cell — the "width ∝(1−φ)" claim made exact.
2. **The RTN=1 line is inland of flotation.** `H* > H_flot` wherever `d_base>0`
   (φ<1), so the band is non-empty and sits *inland* of the classical line.
3. **The band is exactly `grounded(H>H_flot) ∧ (RTN>1 at φ)`**, cell-for-cell
   (zero mismatch) — the band definition and the RTN>1 prediction coincide.
4. **Band area vanishes ∝(1−φ).** The band-area fraction **decreases
   monotonically to 0 as φ→1**, and is **linear in (1−φ)** to leading order: the
   ratio `A_band/(1−φ)` converges as φ→1 (gap between φ=0.99 and 0.98 is 15 %).
   The ratio grows at larger (1−φ) only because more of the thickness
   distribution enters the widening band — the leading-order law is linear.

| φ | 0.99 | 0.98 | 0.96 | 0.94 | 0.90 | 0.85 | 0.80 | 0.70 |
|---|---|---|---|---|---|---|---|---|
| band-area [%] | 0.009 | 0.021 | 0.054 | 0.099 | 0.227 | 0.508 | 1.05 | 3.73 |

## Why this matters

§H.1.1 has two constant-free consequences of the gauge RTN; RESULT 18 certified
the **inversion** (#1, area↦φ is a unique, unbiased estimator) and this certifies
the **MISI-margin geometry** (#2). The hydrology correction to MISI is therefore
not just a mapped number — it is a structurally exact band (`H* = H_flot/φ`,
inland of flotation, per-cell width `(1−φ)/φ`) whose footprint scales linearly
with the bed-wetness deficit `(1−φ)`. With both halves of §H.1.1 calibrated, the
gauge-RTN §G.3 readouts join the repo's plant-and-recover-verified estimator set
(`rtn_synthetic`, `glmig_synthetic` RESULT 16, §G.4 lag estimator RESULT 17).

## Honest scope

This validates the **equation's structure and scaling**, not the physics. The
`H* = H_flot/φ` band is a definitional consequence of the gauge RTN once φ is
fixed; what is *non-trivial and here verified* is the exact per-cell width law,
the band≡`grounded ∧ RTN>1` identity, and the population-level **linear-in-(1−φ)
area scaling** (which depends on the thickness distribution near flotation, not
on the definition alone). It does **not** assert any particular φ or band area
for Antarctica — the planted population is calibrated to the real RTN>1 *area
range* (RESULT 18), not the continent's interior thickness structure — nor that a
single continent-effective φ is physically correct.
