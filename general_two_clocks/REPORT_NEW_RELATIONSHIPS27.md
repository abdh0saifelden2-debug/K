# New cross-relationship — NR50 (`general_two_clocks/new_relationships27.py`) — theory

Continues the derived-and-verified program (NR1–NR49; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the real-data reads come from the committed NR48
cache [`data/nr48_minphase_cache.json`](data/nr48_minphase_cache.json)
(ocean GSER from NR45's ANDRO analysis; BiSON × GOLF delay pair from NR46's
instruments). Unit-proofs in
[`tests/test_phase_area_budget.py`](tests/test_phase_area_budget.py)
(11 tests).

```bash
python general_two_clocks/new_relationships27.py   # -> figures/85_phase_area_budget.json
pytest general_two_clocks/tests/test_phase_area_budget.py -v
```

---

## NR50 — The **phase-area budget**: total memory phase is a conserved quantity fixed by the two clock endpoints (waterbed), and the banded budget classifies gain into transport / causal memory / static weighting  [NR39 × NR45 × NR46 × NR47 × NR48, and Bode's phase-area theorem]

### The mining question

NR48 proved the memory phase is pointwise amplitude-determined
(minimum-phase). What is the *integral* content of that theorem — is there
a conserved quantity, and what does a finite-band phase integral measure
when the response also contains transport (all-pass) or purely
instrumental gain structure?

### The relationship (derived + verified)

**1. Phase-area theorem (exact).** For a minimum-phase response with
finite nonzero endpoint gains — a genuine two-clocks response, instantaneous
clock `χ(∞)` plus relaxed limit `χ(0)`:

```
∫ δ(ω) dln ω = (π/2) [ln|χ(∞)| − ln|χ(0)|]
```

over the full line (Fubini on the Bode integral; the kernel
`ln|coth(u/2)|` has total weight `π²/2`). The **total** memory phase is
fixed by the two clock endpoints alone. Redistribution of relaxation times
moves phase across scales but cannot create or destroy it — control
theory's *waterbed*. Total memory is conserved; only its scale-distribution
is free. "How much memory" is therefore a **two-number measurement**:
`χ(0)` and `χ(∞)`.

**2. The banded budget.** Over a finite band, three distinct physical
objects pay into the measured phase area `A = ∫ φ_meas dln ω`:

| payer | gain | phase area |
|---|---|---|
| transport (all-pass `e^{-iωτ_d}`) | none | `−τ_d(ω₂−ω₁) + φ₀ ln(ω₂/ω₁)` |
| causal memory (minimum-phase) | `Δln\|χ\|` | `(π/2)·Δln\|χ\|` (waterbed-conserved) |
| static weighting (real positive per-frequency factor) | arbitrary | **zero** |

The third row is NR39's invisibility theorem in integral form: a per-mode
sensitivity ratio (instrument response, mode weighting) is a continuum of
elliptic elements — gain without phase. Measuring **both** the phase area
and the endpoint gains therefore *classifies* the gain structure: if
`A ≈ (π/2)Δln|χ|` the gain is causal dynamics; if `A` falls far short, the
gain tilt is static weighting.

### Findings (`figures/85_phase_area_budget.json`)

* **Exact (synthetic).** Lead-lag `(1+iωτ₂)/(1+iωτ₁)`: full-line area
  equals `(π/2)ln(τ₂/τ₁)` to 7e-8. **Waterbed**: a two-pole/two-zero
  system with the same endpoint ratio but different interior structure
  differs pointwise by 0.17 rad yet moves the area by only 9e-9 —
  redistribution without creation.
* **Ocean GSER — the gain is causal memory (real data).** Measured banded
  phase area 6.04 rad·e-fold vs endpoint-gain prediction
  `(π/2)Δln|G*|` = 5.65: **6.4 % closure with zero transport**. The
  mesoscale modulus pays its full Bode phase — the integral, endpoint-only
  face of NR48's pointwise minimum-phase verdict.
* **Solar delay pair — the gain is static weighting (real data).**
  BiSON × GOLF: measured phase area 0.213 is explained by the fitted
  transport term alone (0.207, **2.7 %**, `τ_d = −10.3 s` — NR48's delay
  clock), while a causal minimum-phase reading of the observed gain tilt
  would demand 0.76 — **3.5× more than measured**. The phase *vetoes* the
  causal reading: the instrument-pair gain tilt is per-mode static
  sensitivity (different observation heights/lines of two Doppler
  instruments), not dynamics. The only genuine dynamics between the two
  series is the 10 s time-base transport.

### Consequence for the program

The memory phase is not only pointwise amplitude-determined (NR48) — its
integral is an invariant of the two clock endpoints, robust to everything
in between. And the budget turns any measured (gain, phase) pair into a
three-way classification of the underlying physics: transport / causal
memory / static weighting. Practical protocol: integrate the measured
phase over the resolved band, compare with `(π/2)Δln|χ|` from the two
endpoint amplitudes and with `−τ_dΔω` from the phase slope — the residual
against these two terms is the detector for unmodeled structure.

### Verdicts

| check | result |
|---|---|
| area theorem + waterbed exact (synthetic) | **PASS** (7e-8; waterbed 9e-9, shapes differ 0.17 rad) |
| ocean gain = causal memory | **PASS** (6.04 vs 5.65, 6.4 % closure) |
| solar gain = static weighting | **PASS** (transport closes 2.7 %; causal reading overshoots ×3.5) |

All references remain `[cite]` slots — none invented.
