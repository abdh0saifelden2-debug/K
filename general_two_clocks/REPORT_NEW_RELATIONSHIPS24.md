# New cross-relationship — NR47 (`general_two_clocks/new_relationships24.py`) — theory capstone

Continues the derived-and-verified program (NR1–NR46; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the closed-form invariant is self-contained, and the
real-data reads come from the committed cache
[`data/nr47_memory_phase_cache.json`](data/nr47_memory_phase_cache.json)
(assembled from the committed NR39/NR45/NR46 caches).
Unit-proofs in [`tests/test_memory_phase.py`](tests/test_memory_phase.py)
(14 tests).

```bash
python general_two_clocks/new_relationships24.py   # -> figures/82_memory_phase.json
pytest general_two_clocks/tests/test_memory_phase.py -v
```

---

## NR47 — The **memory phase** `δ = arg χ` is the universal two-clocks coordinate: one number, five faces, two poles  [NR1 × NR2 × NR11 × NR28 × NR39 × NR45 × NR46, and the Bode/Kramers–Kronig backbone]

### The mining question

The four real-data closures shipped 2026-07-08 — ocean microrheology (NR45),
the grid gate (NR42), the solar protocol (NR46), and the real-EEG gate
(NR39) — each measured "how much memory" in a different observable and a
different unit. What single quantity are they all measuring, what is the one
identity that ties it to the whole NR1–NR46 corpus, and what new measurement
method does it imply?

### The relationship (derived + verified)

Every two-clocks instance is a causal linear response `χ(ω) = elliptic
(instantaneous) + parabolic (memory)`. Define the **memory phase**
`δ(ω) = arg χ(ω)` and the **clock ratio** `r = tan δ`.

**(1) One number, five faces — EXACT.** `r = tan δ` is, to machine precision,
all of:

| face | quantity | corpus link |
|---|---|---|
| (i) rheology | loss tangent `G″/G′` | NR45 ocean GSER |
| (ii) cross-spectrum | quadrature ratio `Im/Re` | NR28, NR46, NR39 |
| (iii) migration | index `2πI = tan ψ` | NR2 / NR4 scallop |
| (iv) fractional order | `tan(πα/2)`, α the anomalous-diffusion / Warburg exponent | NR45, NR11 |
| (v) MZ pole | `ω/ω_c` of the slowest memory pole | NR1 / NR28 |

(`test_five_faces_identical`: all five agree to ≤1e-9 for a power-law
response.)

**(2) The Bode backbone.** Gain and phase of a causal minimum-phase response
are a Hilbert pair (Bode 1945). The *local* relation

$$\delta(\omega) \;\approx\; \frac{\pi}{2}\,\frac{d\ln|\chi|}{d\ln\omega}$$

is **exact** whenever `|χ| ~ ω^α` (giving `δ = πα/2`) and **exact at the
log-log inflection** (the 45° crossover). This is the single identity behind
GSER (δ from the MSD slope α), the Warburg half order (α=½ → 45°), and the
migration phase — so a memory phase can be read from an *amplitude slope
alone*, no phase measurement needed. (Honest bound: for a single Debye pole
the local approximation is ~6° median off mid-band and exact at the knee; the
full relation is the Hilbert transform of the log-slope.
`test_bode_local_exact_at_crossover_for_debye`.)

**(3) Two poles bound the axis — the two clocks, made a coordinate.**
`δ = 0` (`r = 0`) is the **pure elliptic / instantaneous** clock — volume
conduction (NR39), K-theory's in-phase real projection (NR2), the atlas-mean
instantaneous mixing (NR45 bias control) — and is *invisible to any
phase/imaginary estimator*. `δ = π/2` (`r → ∞`) is the **pure parabolic /
memory** clock — the terminal-viscous ocean (NR45), the adiabatic solar pole
(NR46), the Warburg DC limit (NR11). Every real coupling lives strictly
between the poles.

**(4) The certification theorem.** A purely instantaneous (elliptic,
real-symmetric) coupling has `δ = 0` identically; therefore **any `δ ≠ 0`
certifies genuine lagged (parabolic) dynamics that no instantaneous confound
can fake**, and the instantaneous clock is the unique Kramers–Kronig-free
real constant `χ(∞)` — the only content the phase cannot carry. This is the
one theorem behind NR39 (ImCoh rejects volume conduction), NR2 (K-theory
zeroes Im), NR28/NR46 (coherence-gated phase), and NR45's normalization-free
elastic fraction.

### Real-data confirmation (`figures/82_memory_phase.json`)

The four closures populate the single `δ ∈ [0°, 90°]` axis exactly where the
theorem places them:

| domain | reading | δ |
|---|---|---|
| **EEG** (NR39) | near-electrode volume-conduction pairs at the **elliptic pole**; distant genuine-coupling pairs off it | **2.1°** (near) vs 6.4° (far) |
| **Ocean** (NR45) | untrapped majority at the **parabolic pole** (pure-loss liquid); eddy-trapped loopers pulled off it (storage `G′/G″`) | rest **90.1°**, loopers **81.8°** (`r=G″/G′=7`) |
| **Solar** (NR46) | coherence-gated I–V mode phase, one-sided and bounded about the −90° loss pole | **−119°…−84°** |
| **Grid** (NR42) | same admittance; the two clocks are the inertia M / damping β split | (structural) |

The instantaneous confound (volume conduction) sits at `δ≈0`, the pure-memory
limits at `δ≈90°`, and the physically interesting signals — the eddy-trapped
loopers' elasticity, the distant EEG coupling, the solar nonadiabaticity —
are exactly the measurable *departures* from a pole.

### What is new (relative to NR2, NR11, NR23, NR28)

NR2 identified the complex admittance and named K-theory as its `Re`-only
projection; NR23 used Kramers–Kronig only for a DC eddy-viscosity sum rule;
NR11/NR28 gave specific phases. NR47 is the **inversion and unification**: it
names `δ = arg χ` as *the* coordinate, proves the five faces are one number,
gives the Bode rule that lets a memory phase be read from an amplitude slope,
identifies the two poles as the two clocks, and states the certification
theorem that explains — in one line — why every phase-based estimator in the
program is artifact-immune. The **new measurement method**: report `δ(ω)`
(dimensionless, normalization-free, artifact-immune) as the domain-portable
memory fingerprint, and recover it from `(π/2)·d ln|χ|/d ln ω` when only an
amplitude spectrum is available.

### Limits

The Bode identity used pointwise is the *local* approximation (exact for
power laws and at the crossover; the full relation is a Hilbert integral) —
NR47 uses it as an exact statement only where the response is locally
power-law, which is precisely the regime the corpus's fractional/GSER/Warburg
faces occupy. The real-data reads are single representative windows/bands from
the upstream caches, not new measurements; NR47 is a re-reading of committed
results through one coordinate, plus the closed-form theory.
