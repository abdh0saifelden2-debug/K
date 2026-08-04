# New cross-relationship — NR44 (`general_two_clocks/new_relationships21.py`) — literature-data (ledger E8)

Continues the derived-and-verified program (NR1–NR43; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, no downloads (all inputs are published-table transcriptions);
unit-proofs in [`tests/test_bl_bedform_table.py`](tests/test_bl_bedform_table.py)
(11 tests).

```bash
python general_two_clocks/new_relationships21.py   # -> figures/79_bl_bedform_table.json
pytest general_two_clocks/tests/test_bl_bedform_table.py -v
```

---

## NR44 — The `B_L` bedform-drag table: the Blumberg–Curl dissolution constant is **not an any-state universal** but **the attractor of the developed, separation-locked state** — measured on the one field record that co-reports melt-bedform geometry *and* independent drag through time  [§A.2 `z_0` closure × Carey 1966 (USGS PP 550-B) × Blumberg & Curl 1974 × RESULT 14/25 selection state]

### The claim under test (ledger E8, as written)

§A.2 closed the scallop-roughness prefactor with **Blumberg & Curl (1974)**:
`z_0 = L₃₂·e^{−κB_L}` with `B_L = 9.4` ⇒ `z_0 ≈ 0.023·L₃₂ ≈ 0.23a`
(`k_s ≈ 0.70 L₃₂`, `k_s/h ≈ 5.6–8.4` — scalloped walls drag like boulders
relative to their relief). E8 conjectured this is one row of a **universal**
bedform table collapsing drag across melt/dissolution morphologies (ripples,
dunes, flutes, scallops), i.e. one `B_L` for all of them.

### The dataset (transcribed, not digitized)

**Carey (1966), USGS Professional Paper 550-B, B192–B198** — the only field
record found that co-reports, for the *same* melting surface through time:

- **independent drag**: 14 winter discharge measurements (St. Croix River,
  Wis., Dec 1964–Mar 1965) with the ice-cover friction factor `f_I`
  partitioned from the bed by the zero-shear-surface method (Kármán–Prandtl
  rough-pipe resistance + Johnson/Vanoni–Brooks sidewall logic);
- **bedform geometry**: 12 dated photographs of extracted ice blocks —
  ripple/dune wavelength `λ` (0.5–1.0 ft) and trough-to-crest height `h`
  (0.03–0.14 ft), sharp-crested, lee slopes ~1.6:1 (steep, separating).

Table 1's **raw** columns (Q, V, areas, wetted perimeters, energy slope) are
transcribed and every derived column recomputed: `R = A/P` reproduces the
published radii; `f = 8gRS/V²` reproduces the published `f_I` column; Manning
`n_I` reproduces all published values **including the per-photo captions to
≤0.3 %** (`test_manning_nI_reproduces_published_captions`). The one bubbly-ice
outlier block (Feb 24, stratified frazil origin — a different morphology
class, per Carey) is excluded.

Conversion, identical to the §A.2 closure: `u_* = √(gR_I S)` → Rouse
`1/√f = 2·log₁₀(2R/k_s) + 1.74` → `k_s` → `z_0 = k_s/30` →
`B_L = −κ⁻¹ ln(z_0/λ)`. Rows with `k_s⁺ = u_*k_s/ν < 70` are flagged
transitional — for those the fully-rough inversion **overestimates** `k_s`,
i.e. their `B_L` values are **lower bounds**, which is the honest direction
for every verdict below.

### Findings (`figures/79_bl_bedform_table.json`)

| phase | rows | `B_L` | `k_s/h` | `k_s⁺` |
|---|---|---|---|---|
| developing (Dec 30–Feb 11) | 6 | **13.9–19.7** (transitional rows = lower bounds) | 0.12–0.6 | 29–407 (mixed) |
| **developed (Feb 18–Mar 30)** | 5 | **10.3–12.2**, median **11.9**, spread 1.8 | **1.4–4.8** | 10²–10³ (all fully rough) |
| Blumberg–Curl scallops (lab) | — | **9.4** | 5.6–8.4 | fully rough |

1. **Any-state universality: FALSIFIED.** The developing trains sit 4.5–10
   units *above* 9.4 — up to `e^{κΔB_L} ≈ 60×` smoother per wavelength than
   the dissolution constant. Speleological palaeo-discharge practice (Curl
   1974's mean-velocity law with `B_L = 9.4`, used throughout the
   scallop-dominant-discharge literature) inherits an **order-10× `z_0`
   systematic** whenever the train is immature or atypical — tens of % in
   log-law velocity.
2. **The developed state is a tight attractor just above the dissolution
   constant: SUPPORTED.** Five successive fully-rough late-winter
   measurements over six weeks land in `B_L = 10.3–12.2` (spread 1.8),
   approaching 9.4 *from above*; `k_s/h` rises from 0.12 to 1.4–4.8, i.e.
   reaches the scallop "rougher-than-your-relief" class (5.6–8.4) from below,
   within ~2× at closest approach.
3. **Drag is maturity-, not steepness-controlled.** Across the record the
   steepness `h/λ` *falls* (~0.20 → ~0.09) while the drag *rises* 5×
   (`f_I` 0.016 → 0.082); Spearman `ρ(h/λ, f_I) = +0.07`. This is the field
   echo of RESULT 14's `K`-independent conduction smoothing and §G.6's
   amplitude-independent `z_0`: what sets the drag is not "how steep" but
   "which flow state the surface has locked into".
4. **The attractor state is the selection state.** The developed rows sit at
   `Re_* = u_*λ/ν = 3.8–6.1×10³` — inside the Thorsness–Hanratty most-unstable
   band (3100–6300) that also selects scallop wavelength, i.e. the
   **separation-locked regime** in which each bedform hosts a captive lee
   eddy. `B_L ≈ 9–12` reads as the *drag signature of that locked state* —
   the same state whose reattachment quadrature carries the migration
   (RESULT 14, and now RESULT 25 on the raw Bushuk arrays). A young train
   that has not yet locked its separation runs 30–60× smoother.

### Honest scope

- Carey's partition rests on a zero-shear surface and an assumed constant bed
  `k_B = 0.23 ft` (his `n_B` comes out constant at 0.0249–0.0253, as designed);
  he calls the absolute values "provisional". The five-row developed *cluster*
  (spread 1.8) and the 5× *trend* are robust to this; the ~±1 absolute `B_L`
  systematic is carried, same as the `B_L ± 1` already carried for
  Blumberg–Curl in §A.2.
- `L₃₂` (Sauter mean) is taken ≈ `λ` for both rows — exact for a uniform
  train; Carey's trains are quasi-uniform (0.5–1.0 ft spread).
- Two rows, one table: this is a two-point universality test with a
  time-resolved development axis, not a survey. The falsification (young ≠ 9.4)
  and the attractor direction are what two points *can* establish; adding
  dune/flute rows (Ashton–Kennedy lab ripples, karst flutes with independent
  drag) is the natural extension — most published karst `f` values are
  *derived from* `B_L = 9.4` and are therefore circular for this test (checked
  and excluded: Gale 1984's conduit friction factors).

### What this feeds

- **§A.2 / P2 parameterisation**: `z_0 = 0.023 L₃₂` is safe **only for
  developed trains**; a maturity gate (or `k_s⁺` gate) should accompany the
  closure. The developed-state attractor being ~1–2.7 above 9.4 brackets the
  prefactor uncertainty honestly.
- **P4 / karst methods**: a quantified warning for scallop palaeo-discharge
  estimates on immature/atypical trains.
- **Theory**: `B_L(maturity)` is a *development gauge* — a drag-only
  observable of whether a melting interface has entered the separation-locked
  (migrating, RESULT 25) state. Candidate for the same treatment on repeat
  radar `z_0` retrievals over subglacial beds (NR33/NR35 family).
