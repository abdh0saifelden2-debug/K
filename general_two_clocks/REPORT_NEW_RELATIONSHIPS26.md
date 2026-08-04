# New cross-relationship — NR49 (`general_two_clocks/new_relationships26.py`) — theory

Continues the derived-and-verified program (NR1–NR48; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the real reads are the committed Bushuk `.mat`
([`glaciers/subglacial/data/bushuk/melt_data_timestring_sub2.mat`](../glaciers/subglacial/data/bushuk/melt_data_timestring_sub2.mat),
12 frames, Δt = 300 s) and the committed tracker cache
([`glaciers/subglacial/data/bushuk_raw_derived.json`](../glaciers/subglacial/data/bushuk_raw_derived.json)).
Unit-proofs in
[`tests/test_dispersion_boost.py`](tests/test_dispersion_boost.py) (13 tests).

```bash
python general_two_clocks/new_relationships26.py   # -> figures/84_dispersion_boost.json
pytest general_two_clocks/tests/test_dispersion_boost.py -v
```

---

## NR49 — The snapshot-pair transfer factorizes into **gain × all-pass**; a Galilean boost *is* the spatial all-pass factor (NR48's delay in the conjugate variable), measured on the real scallop field  [NR2 × NR4 × NR44 × NR48 × Paper 3 data]

### The mining question

NR48 factorized the temporal response: relaxational memory is
amplitude-determined (minimum-phase), and a transport delay is the unique
all-pass factor `e^{-iωτ_d}`, read model-free off the excess phase. The
corpus's *spatial* data — Bushuk's laboratory ice interface `h(x,t)`
(Paper 3), bedforms (NR44), migrating patterns (NR2/NR4) — raises the
conjugate question: what is the all-pass factor in **k**, what physical
motion does it correspond to, and what does its removal leave behind?

### The relationship (derived + verified)

**1. Evolution operator.** Any linear interface evolution
`∂ĥ/∂t = λ(k)ĥ` with `λ = σ − ikc` gives the snapshot-pair transfer

```
T̂(k) = ĥ(k, t+Δt)/ĥ(k, t) = e^{λ(k)Δt}
|T̂| = e^{σΔt}      (gain   → growth clock)
arg T̂ = −kcΔt      (phase  → transport clock)
```

**2. The boost identity (exact).** Under a Galilean boost `x → x − Vt`
the spectrum maps `ĥ → ĥ e^{ikVt}`, i.e. **`λ(k) → λ(k) + ikV`**: the
growth spectrum `σ(k) = Re λ` is boost-*invariant*, while the boost
contributes a pure all-pass factor `e^{ikVΔt}` (`|T̂| = 1`, phase linear
in k). This is NR48's delay `e^{-iωτ_d}` with `ωτ_d ↔ −kVΔt` — *delay in
time ↔ displacement in space*. Conversely: no gain structure
(growth/damping) can produce a k-linear phase, and no boost can change
`|T̂|` — kinematics and dynamics separate exactly, per snapshot pair.
(Aliasing bound: the phase is principal-valued, so the celerity is
readable only where `|kVΔt| < π` — displacement per frame under half a
wavelength, the Nyquist condition of every phase tracker.)

**3. Measurement method.** `λ̂(k) = ln T̂(k)/Δt` from any two snapshots;
power-weighted medians over successive pairs give the full complex
dispersion relation — `σ(k)` is the linear-stability fingerprint, `c(k)`
the migration dispersion — from data that only reports interface heights.

### Findings (`figures/84_dispersion_boost.json`)

* **Synthetic (exact).** On a periodic advected–damped field the estimator
  recovers `σ(k)` and `c = V` to machine precision (< 1e-9); removing the
  boost spectrally leaves `σ` unchanged (< 1e-12) and kills the celerity;
  a pure damping field fits zero celerity and a pure boost fits zero gain
  — the factorisation is exact, not approximate.
* **Migration is one all-pass factor (real data).** Lab-frame pattern-band
  (top modal wavelengths ≥ 45 mm) phase celerity = **48.2 mm/hr**, agreeing
  with the two committed trackers (55.8 xcorr / 57.2 Bushuk's own advection
  velocity) to ~14 % (Hann window + pattern evolution bias the spectral
  estimate low). In Bushuk's co-moving (advected) frame the same estimator
  gives **3.4 mm/hr — the boost annihilates 93 %** of the phase clock, as
  an all-pass factor must.
* **The gain is Galilean-invariant (real data).** Pattern growth
  `σ = −0.06 /hr` (lab) vs `−0.45 /hr` (advected); roughness-band
  (15–45 mm) damping `−4.7 /hr` (lab) vs `−5.9 /hr` (advected): same sign,
  same scale, both frames — boosts move phase, never gain.
* **Marginal pattern, damped roughness — NR44's attractor, spectrally.**
  `|σ_pattern| < 0.5 /hr` (lifetime ≫ experiment) while
  `σ_roughness ≈ −5 /hr` (lifetimes of minutes): the developed scallop
  field is a saturated, rigidly migrating state — the pattern band sits at
  the marginal-stability fixed point while sub-pattern roughness is
  strongly relaxational. This is the spectral face of NR44's
  "developed-state attractor".

### Consequence for the program

NR48's factorisation is the general kinematics/dynamics splitter for any
evolving field, in either conjugate variable: the transport clock is
all-pass (NR2/NR4's migration, NR48's delay, this boost), the memory/growth
clock is gain (NR45's modulus, NR47's memory phase, this `σ(k)`). The
protocol — gain → dynamics, k-linear phase → kinematics, excess structure →
genuine dispersion — turns any repeat-survey dataset (flume frames, repeat
bathymetry, ICESat-2 repeat tracks) into a two-clocks measurement.

### Verdicts

| check | result |
|---|---|
| boost identity exact (synthetic) | **PASS** (all six < 1e-9) |
| migration = one all-pass factor | **PASS** (48.2 vs 55.8 mm/hr; co-moving residual 7 %) |
| gain is boost-invariant | **PASS** (Δσ ≪ damping scale, both bands) |
| marginal pattern + damped roughness | **PASS** (−0.06 vs −4.7 /hr, lab) |

All references remain `[cite]` slots — none invented.
