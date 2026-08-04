# New cross-relationship — NR51 (`general_two_clocks/new_relationships28.py`) — theory

Continues the derived-and-verified program (NR1–NR50; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the real reads are the committed Bushuk `.mat`
(12 frames) and the committed Paper-3 flux cache
([`glaciers/subglacial/data/bushuk_raw_derived.json`](../glaciers/subglacial/data/bushuk_raw_derived.json)).
Unit-proofs in [`tests/test_triad_parity.py`](tests/test_triad_parity.py)
(15 tests).

```bash
python general_two_clocks/new_relationships28.py   # -> figures/86_triad_parity.json
pytest general_two_clocks/tests/test_triad_parity.py -v
```

---

## NR51 — **Closed spectral polygons are shape, open ones are motion**: triads are exactly boost-blind, Im B is the parity-odd shape coordinate, and the real scallops break parity in the *flux clock*, not the stored shape  [NR2 × NR4 × NR49 × Paper 3, and the Elgar–Guza bispectral asymmetry]

### The mining question

NR49 proved the two-point (pair) transfer carries the entire Galilean
boost as an all-pass factor. What do *higher* spectral correlators carry —
and which observable holds the parity breaking that Paper 3 detected in
the melt-flux quadrature `E_sin`?

### The relationship (derived + verified)

**1. Closed polygons are boost-blind (exact).** Under `x → x − Vt` every
Fourier coefficient gains `e^{ikVt}`, so an n-point correlator
`⟨ĥ(k₁)…ĥ(kₙ)⟩` gains `e^{i(k₁+…+kₙ)Vt}` — equal to 1 **iff the
wavevector polygon closes**. The pair transfer (open: k at two times)
carries the whole boost (NR49); the equal-time bispectrum
`B(k₁,k₂) = ⟨ĥ(k₁)ĥ(k₂)ĥ*(k₁+k₂)⟩` carries **none**. Triad phases are
pure *shape*; pair phases are pure *motion*.

**2. Im B is the parity-odd shape coordinate.** Under the mirror
`x → −x`, `ĥ(k) → ĥ*(k)`, so `B → B*`: Re B (skewness family) is
parity-even; Im B (asymmetry family — Elgar–Guza) is parity-odd; the
biphase flips sign exactly. Exact mod-N moment sum rules (machine
precision):

```
mean(h³)     = (1/N³) Σ_closed ĥ₁ĥ₂ĥ₃            (Re face — skewness)
mean(ℋ[h]³)  = (1/N³) Σ_closed i·s₁s₂s₃·ĥ₁ĥ₂ĥ₃    (Im face — asymmetry)
```

Bulk skewness/asymmetry *are* triad sums; a parity-symmetric shape has
all biphases at 0/180°. (Documented caveat, pinned by a test: **linear
detrending is not boost-covariant** on a discrete grid — the ramp
projects on every Fourier mode — so exactness statements use the raw
field; mean removal only touches k = 0, outside every triad.)

**3. Migration does not imply asymmetric shape.** Because the boost is
invisible to every closed polygon, a rigidly migrating pattern can be
perfectly parity-symmetric: pair-level parity breaking (migration, flux
quadrature) leaves *no* triad imprint. Shape asymmetry is strictly a
**nonlinear** (asymmetric triad coupling) effect. Dune-style lee–stoss
asymmetry is a *nonlinearity* meter, not a migration meter; Paper 3's
flux quadrature reads parity at the linear level — where the scallop
parity breaking actually lives.

### Findings (`figures/86_triad_parity.json`)

* **Exactness on data.** Random spectral boosts of every frame move the
  dominant scallop biphase by 4e-15 deg; mirroring flips it exactly
  (2e-14); the moment sum rules hold to ≤ 1e-14 on every frame.
* **Positive control.** The leaning profile `cos θ + 0.3cos(2θ − 90°)`
  gives asymmetry +0.56 (biphase 90°); the phase-aligned control is
  skewed (0.56) but has asymmetry exactly 0 — the detector separates the
  two parities.
* **The scallops: flux breaks parity, shape barely does.** Committed
  flux quadrature (Paper 3): `t = −4.7` over 11 pairs — decisive. Shape:
  per-frame asymmetry `0.17 ± 0.25` (`t ≈ 2.4`, frames correlated),
  dominant-triad biphase 30° with **47° frame-to-frame circular std** —
  weak and unstable, from the very frames whose migration is decisive
  (NR49: 48 mm/hr, 7 % co-moving residual). The parity breaking is
  *dynamic* (transport clock), not *stored* (shape).

### Consequence for the program

The spectral-polygon rule organizes the corpus's phase observables: open
polygons (pair transfer, cross-spectra) hold kinematics and transport
(NR48's delay, NR49's boost); closed polygons (bispectrum) hold intrinsic
shape parity. Parity-break detectors must read the flux/pair level
(Paper 3's `E_sin`) — shape-based (triad) detectors certify asymmetric
nonlinear coupling instead, which these near-sinusoidal scallops barely
have, unlike avalanche-faced dunes.

### Verdicts

| check | result |
|---|---|
| closed polygons boost-blind + parity-odd (exact) | **PASS** (4e-15 deg; sum rules ≤ 1e-14) |
| detector separates parities (controls) | **PASS** (leaning A = +0.56; symmetric A = 0) |
| scallops break parity in flux, not shape | **PASS** (flux t = −4.7 vs shape t = 2.4; biphase circ-std 47°) |

All references remain `[cite]` slots — none invented.
