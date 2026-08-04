# New cross-relationship — NR48 (`general_two_clocks/new_relationships25.py`) — theory

Continues the derived-and-verified program (NR1–NR47; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the synthetic checks are self-contained, and the
real-data reads come from the committed cache
[`data/nr48_minphase_cache.json`](data/nr48_minphase_cache.json)
(ocean GSER from the committed NR45 ANDRO cache; delay clock from the
BiSON × GOLF cross-spectrum of the NR46 instrument pair).
Unit-proofs in [`tests/test_minphase_delay.py`](tests/test_minphase_delay.py)
(13 tests).

```bash
python general_two_clocks/new_relationships25.py   # -> figures/83_minphase_delay.json
pytest general_two_clocks/tests/test_minphase_delay.py -v
```

---

## NR48 — The two-clocks response is **minimum-phase** (the memory phase is the Bode transform of the amplitude), and a genuine transport delay is the unique **all-pass** third clock  [NR45 × NR46 × NR47, and the Bode 1945 / minimum-phase–all-pass factorisation backbone]

### The mining question

NR47 promoted the memory phase `δ = arg χ` to the universal two-clocks
coordinate, using the *local* Bode relation `δ ≈ (π/2)·d ln|χ|/d ln ω`
(exact only for power laws). Two questions the corpus leaves open:

1. When is the phase *fully determined* by the amplitude spectrum — i.e.
   when does an amplitude measurement alone suffice to read the memory
   clock? (This is the unstated assumption behind every GSER/microrheology
   inversion the repo uses, NR45 included.)
2. What physical content can a measured phase have that the amplitude does
   **not** determine — and what does it measure?

### The relationship (derived + verified)

**1. A passive relaxational (two-clocks) response is minimum-phase.** A
causal `χ` with no right-half-plane zeros satisfies the exact Bode
gain–phase integral (Bode 1945):

```
δ(ω₀) = (1/π) ∫ (d ln|χ|/du) · ln|coth(|u|/2)| du,   u = ln(ω/ω₀)
```

Relaxational responses built from decaying kernels (sums/continua of
`1/(1+iωτ)` poles with positive weights — every MZ/GLE two-clocks instance
in the corpus, NR1/NR11/NR28) have neither RHP poles *nor* RHP zeros, so
the memory phase carries **no information independent of the amplitude
spectrum**: `δ = Bode[ln|χ|]`. NR47's local slope rule is the first term of
this exact transform (the kernel `ln|coth(u/2)|` has weight π²/2
concentrated at `u=0`).

*Why GSER works:* Mason–Weitz recover the full complex modulus `G*(ω)` from
the amplitude of the MSD alone. That inversion is legitimate **iff** the
modulus is minimum-phase — NR48 states the assumption and verifies it on
the real ocean (below).

**2. A pure transport delay is the unique all-pass factor.** Any causal
response factors uniquely as `χ = χ_min · χ_ap` (minimum-phase × all-pass).
The all-pass factor `e^{-iωτ_d}` has `|χ_ap| = 1` — *no amplitude
signature, zero Bode phase* — and a phase linear in ω. Hence the **excess
phase**

```
φ_excess(ω) = φ_measured(ω) − Bode[ln|χ|](ω)
```

isolates genuine propagation/advection time, and its slope is `τ_d`: a
model-free delay measurement, cleanly separated from relaxational memory.
No relaxation can fake it (relaxation is amplitude-determined); no delay
can hide in the amplitude (it has none). This is the delay counterpart of
NR39's "instantaneous mixing is invisible to the imaginary part".

**3. Three clocks, not two.** The NR47 axis gains a third, orthogonal
element:

| clock | phase | amplitude | signature |
|---|---|---|---|
| elliptic (instantaneous) | `δ = 0` | flat | none in either |
| parabolic (relaxational memory) | `δ = Bode[ln\|χ\|]` | determines δ | amplitude-determined phase |
| **advective (transport delay)** | `−ωτ_d` | `\|χ\| = 1` | **excess phase, linear in ω** |

Any measured response decomposes uniquely into (amplitude → minimum-phase
part) + (all-pass delay). The standard control-theory factorisation becomes
the *measurement protocol* for the two-clocks program.

### Findings (`figures/83_minphase_delay.json`)

* **Synthetic (exact).** For Debye, fractional (`α = 0.6`) and two-pole
  minimum-phase models, the Bode integral reconstructs the phase from
  `ln|χ|` alone to **0.01–0.04° median**. A pure delay `e^{-iωτ}` gives
  zero Bode phase and an excess-phase slope recovering `τ` to machine
  precision (0.3000 vs 0.3000).
* **Ocean GSER is minimum-phase (real data, NR45 ANDRO cache).** The GSER
  phase `δ = πα/2` measured from the MSD slope agrees with the Bode
  transform of the measured `ln|G*|` to **3.0° median** over the interior
  of the resolved 1.5-decade band (the band is extended with its terminal
  power-law slopes before the integral — exactly the Mason–Weitz local
  power-law assumption; raw truncation gives 6.8°, the difference is pure
  edge bias of the finite-band integral). The Kramers–Kronig/minimum-phase
  assumption implicit in every microrheology inversion, verified on the
  mesoscale ocean.
* **A delay clock measured model-free on real data (NR46 instrument
  pair).** BiSON × GOLF velocity–velocity cross-spectrum (two independent
  solar Doppler instruments, 2003–2006, 123 coherent bins in the p-mode
  band): the gain is flat (log-std 0.19 → minimum-phase part ≈ 0) while
  the phase is *linear in frequency*. Under `S_xy = GOLF × conj(BiSON)`
  and the delay convention `χ = e^{-iωτ_d}`, the excess-phase slope gives
  **τ_d = −10.3 s — GOLF leads BiSON by ~10 s**: a pure time-base offset
  between the two instruments' clocks, extracted with no model of the Sun.
  The third clock observed in isolation: no amplitude signature, all
  phase.

### Consequence for the program

The memory phase of NR47 is amplitude-determined for every relaxational
(parabolic) system — an amplitude spectrum suffices to read the memory
clock, which is why single-quantity inversions (GSER, MSD-only
microrheology, |χ|-only spectroscopy) work at all. Conversely a nonzero
excess phase is the unambiguous signature of genuine propagation/advection
time. The protocol: measure `|χ|`, Bode-transform it, subtract, and read
the residual slope — memory and transport separate exactly.

### Verdicts

| check | result |
|---|---|
| minimum-phase reconstruction (3 models) | **PASS** (0.039° worst median) |
| GSER = minimum-phase on real ocean | **PASS** (3.0° median) |
| delay = all-pass excess (synthetic) | **PASS** (τ exact, Bode phase < 1e-6) |
| real delay clock (BiSON × GOLF) | **PASS** (τ_d = −10.3 s, gain log-std 0.19) |

All references remain `[cite]` slots — none invented.
