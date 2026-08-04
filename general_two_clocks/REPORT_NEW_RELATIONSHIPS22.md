# New cross-relationship — NR45 (`general_two_clocks/new_relationships22.py`) — real-data (ledger E11)

Continues the derived-and-verified program (NR1–NR44; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only; offline-safe — every number re-derives from the committed per-float
accumulator cache [`data/nr45_andro_gser_cache.json`](data/nr45_andro_gser_cache.json)
(964 KB; the raw 0.5 GB ANDRO deposit is *not* committed — set
`$ANDRO_DAT`/`$ANDRO_NC` and call `build_cache()` to regenerate bit-for-bit).
Unit-proofs in [`tests/test_andro_gser.py`](tests/test_andro_gser.py) (13 tests).

```bash
python general_two_clocks/new_relationships22.py   # -> figures/80_andro_gser.json
pytest general_two_clocks/tests/test_andro_gser.py -v
```

---

## NR45 — Ocean microrheology: the mesoscale ocean has a measurable complex modulus, its elasticity is carried by the eddy-trapped floats, and the standard record-mean detrend *fakes* it ensemble-wide  [GSER (Mason & Weitz 1995) × ANDRO deep displacements × REPORT_NONMARKOV_ARGO.md × two-clocks]

### The claim under test (ledger E11, as written)

Argo float dispersion, read through the generalized Stokes–Einstein relation,
turns the committed non-Markov ocean-memory result
([`REPORT_NONMARKOV_ARGO.md`](REPORT_NONMARKOV_ARGO.md) — same North-Atlantic
box, same ~1000 dbar level, temperature memory with τ_slow > 60 d) into a
**complex modulus `G*(ω)` of the mesoscale ocean** — a new *measurement
framing*, not a new microphysical law.

### The dataset

**ANDRO** (Ollitrault & Rannou; SEANOE doi:10.17882/47077, CC-BY): per-cycle
park-depth displacements/velocities for the global Argo array. Filtered to
park pressure 700–1300 dbar, valid deep fixes, |u| < 1.5 m/s: **1,383,958
cycles, 9,379 floats**, median cadence 10.0 d. The companion 0.5° gridded
atlas (`mean_u`, `mean_v` in the same deposit) supplies an **independent
Eulerian mean** for residual-velocity construction — the load-bearing control
(below). North-Atlantic box 25–45 °N, 20–65 °W: 301 floats with ≥ 30 cycles
and ≥ 12-cycle regular (7–13 d) segments.

### Method (all standard, composed)

1. Per float: residual velocity `v′ = v_park − v_atlas(x)`; near-regular
   segments; per-float MSD (displacement sums) and VACF accumulators to lag
   300 d; per-float spin `Ω = ⟨u′v′₊₁ − v′u′₊₁⟩/⟨u′²+v′²⟩` (discrete rotation
   measure; **loopers = top |spin| quartile**, the Veneziani/Griffa
   spin-parameter subpopulation).
2. Ensemble MSD(t) → local exponent `α(t) = d ln MSD/d ln t` → Mason-form
   GSER: `|G*(1/t)| ∝ 1/[MSD(t)·Γ(1+α(t))]`, phase `δ = πα/2`. The loss
   tangent and the **elastic fraction `G′/G″ = 1/tan(πα/2) = tan((1−α)π/2)`
   are normalization-free** — no ocean-"kT", no tracer radius, no calibration
   (unit-proofs: exponent recovery, viscous limit, scale invariance).
3. The finite-record bias control: removing the float's OWN record-mean
   velocity (the common detrend) forces MSD to bend down at record-length
   lags — an *apparent* elastic window. Removing the independent atlas mean
   is unbiased. Both are computed; the difference is reported as bias.

### Findings (`figures/80_andro_gser.json`)

| ensemble | n | α[90–140 d] | α[100–190 d] | α[150–190 d] | G′/G″ [100–190 d] |
|---|---|---|---|---|---|
| NA box, atlas mean | 301 | **1.001** | 0.98 | 0.93 | — |
| loopers (top \|spin\| quartile) | 76 | 0.94 | **0.91** (range 0.86–0.95) | 0.88 | **0.09–0.22** |
| non-loopers | 225 | **1.03** | 1.00 | 0.95 | ~0 |
| NA box, float-mean removal (bias) | 301 | 0.97 | 0.93 | **0.86** | (fake) |
| global \|lat\|<60, pooled | 6936 | 1.18 | — | — | (artefact, flagged) |

1. **The mesoscale ocean at ~1000 dbar is a viscoelastic LIQUID with a
   measured terminal-relaxation crossover.** α relaxes from 1.43 (10 d,
   near-ballistic) through the velocity-decorrelation knee (VACF e-fold
   ~15 d, zero-crossing ~50 d) to a clean terminal window **α = 1.00–1.01
   across 90–140 d** with eddy diffusivity `K = MSD/4t ≈ 2.2×10³ m²/s` at
   100 d — the Taylor (1921) picture, now read as rheology: pure loss,
   `δ = π/2`.
2. **The elasticity is subpopulation-carried.** Loopers dip to α = 0.86–0.95
   (median 0.91) over 100–190 d — an elastic fraction **`G′/G″ ≈ 0.09–0.22`
   at 3–6-month forcing periods** — while non-loopers hold α ≈ 1.00 (elastic
   fraction consistent with 0). In GSER language: the eddy-trapped
   subpopulation feels the mesoscale field as a *weak elastic solid* at the
   eddy-coherence timescale; the untrapped majority feels a simple liquid.
   This is the two-clocks split as rheology — fast rotational (trapped)
   clock ⇒ storage G′; slow dispersive clock ⇒ loss G″.
3. **The record-mean detrend fakes ensemble elasticity, and the bias is now
   quantified.** With per-float record-mean removal the whole 301-float
   ensemble drops to α = 0.86 at 150–190 d; the independent atlas-mean
   control holds 0.93 (bias 0.07). Lagrangian "subdiffusion" claims at
   record-length lags need this control before any viscoelastic reading.
4. **Global pooling is flagged, not claimed.** The 6936-float pooled
   ensemble stays superdiffusive (α[90–140 d] = 1.18) — cross-float K
   heterogeneity (a mixture/Richardson-like artefact of pooling), which is
   why the claim is made in a homogeneous box, per-subpopulation.

### What is mainstream here, and what is new

Loopers/spin subpopulations (Veneziani, Griffa et al. 2004; Lumpkin 2016),
Taylor dispersion regimes, and the record-mean pitfall (Davis-style residual
construction) are established. GSER passive microrheology (Mason & Weitz
1995; Squires & Mason 2010) is established *in soft matter*. The composition
is new in three specific places: (i) the **normalization-free** part of GSER
(α → δ, G′/G″) transfers to the ocean with no "kT" analogue needed, turning
float archives into a rheometer for the eddy field; (ii) the elastic signal
is **subpopulation-carried** — an ensemble-mean analysis reports a much
weaker modulus than the trapped population actually feels; (iii) the
float-mean detrend bias is measured *against an independent atlas mean* on
the same floats — 0.07 in α is the size of the fake elasticity, comparable
to the real looper signal, i.e. the control is not optional.

### Limits

Discrete 10-d cadence caps the resolved band at ω ≲ (20 d)⁻¹; the 190–300 d
tail thins (fewer long segments, loopers n=76 noisiest there — the verdict
windows stop at 190 d); `|G*|` is reported in arbitrary units (only its
shape and the phase are claimed); loopers are a *state* label (top-quartile
|spin| over a record), not a per-eddy segmentation; the NA-box result is one
region — the framing is portable, the numbers are not.

### Cross-links

Same box/level as [`REPORT_NONMARKOV_ARGO.md`](REPORT_NONMARKOV_ARGO.md)
(temperature memory ⇒ now velocity-field rheology); the two-clocks
fast/slow structure (trapped vs dispersive) as in NR35/NR43's
episodic-on-secular split; the memory⇒transport chain NR16–NR24 (a
non-Markov velocity kernel is exactly what a nonzero G′ encodes at the
matching band).
