# New cross-relationship — NR52 (`general_two_clocks/new_relationships29.py`) — theory + real data

Continues the derived-and-verified program (NR1–NR51; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the real reads come from the committed cache
[`data/nr52_odd_spin_cache.json`](data/nr52_odd_spin_cache.json)
(7258 ANDRO deep floats, 700–1300 dbar, 10-day cycles; regenerate with
`build_cache()` from the ANDRO pickle). Unit-proofs in
[`tests/test_odd_spin.py`](tests/test_odd_spin.py) (15 tests).

```bash
python general_two_clocks/new_relationships29.py   # -> figures/87_odd_spin.json
pytest general_two_clocks/tests/test_odd_spin.py -v
```

---

## NR52 — The **odd completion** of the two-clocks response: the Lagrangian spin is the parity-odd face that scalar MSD/GSER cannot see — and the deep ocean's chirality flips sign at the equator, anticyclonic in both hemispheres  [NR39 × NR45 × NR47 × NR51, and the odd-viscosity / Lagrangian-spin mainstream]

### The mining question

NR47–NR51 built the phase anatomy of a *scalar* response: memory phase
(ω-odd, KK), transport phase (all-pass), triad biphase (parity-odd shape).
But the ocean's response is a **tensor** in a rotating frame. What face of
the two-clocks structure does rotation add, which measurements are blind
to it, and can the corpus's own float data measure it?

### The relationship (derived + verified)

**1. Tensor completion.** In a 2D rotating layer, the correlation/response
tensor of an isotropic-but-chiral system decomposes as

```
C(τ) = C_s(τ)·I + C_a(τ)·ε,     ε = z× = [[0,−1],[1,0]]
```

with `C_s` the symmetric (two-clocks) part and `C_a` the **odd** part —
the Lagrangian *spin* correlation `(⟨u(0)v(τ)⟩ − ⟨v(0)u(τ)⟩)/2`. For the
damped rotator GLE `du/dt = −u/τ_L + f ẑ×u + noise` the matrix exponential
factorizes exactly (I and ε commute):

```
C(τ)/C(0) = e^{−τ/τ_L} (cos fτ · I + sin fτ · ε)
```

so `ρ_odd(τ) = sin(fτ)e^{−τ/τ_L}` — positive spin = counterclockwise
(sign anchored by a deterministic circular-motion test).

**2. The chirality-blind spot.** Scalar MSD and every scalar contraction
read only `C_s`: mirroring `y → −y` maps a realization with rotation `+f`
to one with `−f` with the **same scalar MSD sample-by-sample**, while
`C_a` flips sign exactly. Chirality is invisible to GSER — the tensorial
analog of NR39's invisibility (elliptic mixing → no Im) and NR51's parity
hiding. `C_a` is **doubly odd**: under parity *and* under lag reversal
(`C(−τ) = C(τ)ᵀ` by stationarity, and the odd part of the transpose is
exactly `−C_a`). A nonzero spin therefore certifies broken microscopic
time-reversal — a detailed-balance meter, with rotation supplying the
breaking in a rotating fluid.

**3. The measurement.** `ρ_odd(τ) = O(τ)/E(0)` from float displacement
series (float-mean residuals) reads the net eddy handedness of a region
from single-particle statistics — no velocity fields needed.

### Findings (`figures/87_odd_spin.json`)

* **Chirality flips at the equator (≈10σ).** Per-float spin at 10 d lag:
  NH (>5°) mean **−0.023, t = −9.7** (clockwise); SH (<−5°) mean
  **+0.024, t = +13.2** (counterclockwise); equatorial band (|lat| < 5°)
  **+0.003, t = +1.0 — a null exactly where f → 0**. Band structure is
  sign-antisymmetric: (−, −, 0, +, +) from NH-mid to SH-mid.
* **Anticyclonic dominance in both hemispheres.** NH-clockwise and
  SH-counterclockwise are both *anticyclonic* — the deep-looper censuses'
  known preference, recovered here as a two-line statistic from raw
  displacements. The odd memory decays with a 10–30 d half-life — the
  same eddy memory as NR45's scalar VAC.
* **Exactness.** expm vs the analytic factorization ≤ 1e-15; the
  mirrored-realization pairing exact (MSD identical, spin flipped, even
  part untouched); the estimator matches the analytic `sin(fτ)e^{−τ/τ_L}`
  within sampling noise; `O(0) = 0` identically.

### Consequence for the program

The full response anatomy now has four measured faces: memory phase δ
(ω-odd; NR47/48), transport phase −ωτ_d (all-pass; NR48/49), triad
biphase (parity-odd shape; NR51), and the **spin** (parity- and lag-odd
handedness; this NR). Every scalar inversion in the corpus (GSER, MSD
microrheology — NR45) silently projects out the odd face: legitimate for
the modulus, blind to rotation sense. The odd face is not small physics:
in the deep ocean it is a 10σ, hemisphere-antisymmetric,
anticyclone-dominated signal sitting in data the corpus already had.

### Verdicts

| check | result |
|---|---|
| odd completion exact (synthetic) | **PASS** (expm 2e-16; mirror pairing exact; ρ_odd vs analytic < 0.05) |
| MSD is chirality-blind | **PASS** (MSD diff 0 with spin flipped exactly) |
| ocean chirality flips at equator | **PASS** (NH t = −9.7, SH t = +13.2, EQ t = +1.0) |
| anticyclonic dominance both hemispheres | **PASS** (NH −0.023 CW, SH +0.024 CCW) |

All references remain `[cite]` slots — none invented.
