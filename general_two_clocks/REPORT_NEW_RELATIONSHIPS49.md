# NR72 — the F2 peak audited by the two clocks: R(hmF2) classifies what built the layer

**Module** `general_two_clocks/new_relationships49.py` ·
**data** committed matched NRLMSIS 2.1 + IRI2016 cache
(`data/nr72_f2peak_cache.json`, 5 conditions, 150–650 km; builder
`build_nr72_f2peak_cache.py`) ·
**figure** `general_two_clocks/figures/107_f2peak_two_clocks.json` + `.png` ·
**tests** `general_two_clocks/tests/test_f2peak_two_clocks.py` (9)

## The relationship

Classical F2-layer theory (Rishbeth & Garriott 1969; Schunk & Nagy ch. 13) puts
the daytime peak near the level where O+ chemical loss β = k1[N2]+k2[O2] equals
the ambipolar diffusion rate D_a/H². That is verbatim a two-clocks crossover —
τ_chem = 1/β racing τ_diff = H_p²/D_a on NR71's welded plasma column. NR72 turns
it into an audit:

**R(h) = β·H_p²/D_a** (dimensionless clock ratio), evaluated at IRI's own
empirical peak hmF2, with neutrals from NRLMSIS and plasma from IRI at MATCHED
epochs/locations, zero free parameters.

## Measured

| condition | hmF2 | R(hmF2) | R=1 level | verdict |
|---|---|---|---|---|
| midlat equinox noon | 228 km | 17 | 274 km | balance-formed |
| midlat winter noon | 217 km | 27 | 264 km | balance-formed |
| midlat summer noon | 245 km | 5.1 | 272 km | balance-formed |
| midlat equinox midnight | 332 km | 0.0048 | 254 km | **dynamics-formed** |
| equator equinox noon | 359 km | 0.0037 | 274 km | **dynamics-formed** |

- **Classifier gap: 3.0 decades**, no case in between.
- Daytime midlatitude peaks sit at R ≈ 5–27, i.e. 1–2 N₂ scale heights *below*
  the naive R=1 handoff — where full F2-layer solutions put the peak relative to
  the balance level. The chemical clock built these layers.
- The midnight and equatorial peaks sit 78–85 km *above* the handoff at
  R ≈ 4×10⁻³: no chemical crossover builds them. Mainstream dynamics names both:
  nighttime equatorward wind lifting + downward plasmaspheric flux (Rishbeth's
  servo picture), and the equatorial E×B fountain (Appleton anomaly) — a third,
  electrodynamic clock the two-clock audit correctly refuses to absorb.
- Robustness: R falls with a measured e-fold of 15–17 km (β collapses with [N2]
  while D_a ∝ 1/[O] grows), so O(1) convention changes (H_p vs neutral H,
  rate-coefficient factors) move the R=1 level ≤ 25 km — negligible against a
  3-decade gap computed identically across conditions.

## The reading

NR68 read the turbopause as K_zz = D_i; NR72 reads hmF2 as β = D_a/H_p² — the
same handoff one story down. The new element is the *audit* face (Paper 1's
program: a diagnostic, not a universal model): the clock ratio at the empirical
peak tells you WHICH physics built it, and where the two-clock balance fails,
the failure is signed, large, and names the third clock (wind/flux,
electrodynamics) — exactly as the 2-D a-posteriori ceiling in Paper 1 names
what eddy viscosity cannot represent.

## Honest scope

NRLMSIS and IRI are independently constructed mainstream empirical models —
consistency here is a nontrivial cross-check, not a first-principles
measurement. k1, k2 (St-Maurice & Torr 1978) and ν_in (Banks; Schunk & Nagy)
carry ~20–30% uncertainties; Teff uses (Ti+Tn)/2 (drift heating neglected,
quiet midlatitude). Vertical, not field-aligned, gradients (dip > 60° for the
midlat columns). The claim is the classifier and its 3-decade separation, not
an exact hmF2 prediction.

## Verification

`python3 general_two_clocks/build_nr72_f2peak_cache.py` rebuilds the cache
(needs pymsis + gfortran/iri2016; committed JSON replays offline).
`python3 general_two_clocks/new_relationships49.py` regenerates figure/JSON.
`pytest general_two_clocks/tests/test_f2peak_two_clocks.py` (9 passed): rate and
collision-frequency magnitudes/scalings, crossing + e-fold machinery on
synthetics, R = τ_diff/τ_chem identity, balance-formed band (R 3–50, peak 15–70
km below handoff), dynamics-formed band (R < 0.05, peak > 50 km above), gap > 2
decades, stable R=1 levels and e-folds, verdict completeness.
