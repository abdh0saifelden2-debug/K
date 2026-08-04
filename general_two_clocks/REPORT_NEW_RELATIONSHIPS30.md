# New cross-relationship — NR53 (`general_two_clocks/new_relationships30.py`) — theory + real data

Continues the derived-and-verified program (NR1–NR52; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the real reads come from the committed cache
[`data/nr53_spin_bands_cache.json`](data/nr53_spin_bands_cache.json)
(11 latitude bands of the NR52 per-float spin accumulators; ANDRO deep
floats, 700–1300 dbar). Unit-proofs in
[`tests/test_spin_two_clocks.py`](tests/test_spin_two_clocks.py) (13 tests).

```bash
python general_two_clocks/new_relationships30.py   # -> figures/88_spin_two_clocks.json
pytest general_two_clocks/tests/test_spin_two_clocks.py -v
```

---

## NR53 — The **two clocks of the spin**: an oscillatory *eddy clock* (deep vortex rotation, hemisphere-mirrored, T ≈ 2 months) and a persistent *circulation clock* — and the subpolar/polar cyclonic reversal belongs to the circulation clock  [NR45 × NR52, the two-clocks paradigm applied inside the parity-odd face]

### The mining question

NR52 measured one spin number per region. The full lag structure
`ρ_odd(τ)` must distinguish two very different carriers of handedness:
coherent vortices (which rotate floats through full orbits) and
large-scale circulation curvature (gyres, boundary currents). Are they
separable from single-float statistics alone — and which one carries the
polar chirality reversal?

### The relationship (derived + verified)

**The two-clock decomposition of the odd correlation:**

```
ρ_odd(τ) = A sin(f_e τ) e^{−τ/τ_m}   ← eddy clock (damped rotator:
                                        sign-flipping overshoot at half the
                                        rotation period, finite memory)
         + g τ                        ← circulation clock (slow persistent
                                        rotation, period ≫ window: linear
                                        growth, no sign flip)
```

Nonparametric fingerprints: a persistent **zero-crossing** certifies the
eddy clock; a same-sign **long-lag tail** certifies the circulation
clock. (Synthetic mixture: the fit recovers `f_e` to 1.8 % and `g` within
×1.3; a drift-only control has no crossing and a dominant tail.)

### Findings (`figures/88_spin_two_clocks.json`)

* **The deep eddy rotation period, measured and mirrored.** Tropical
  bands (5–20°) oscillate: NH starts clockwise and overshoots at 30 d; SH
  is the exact mirror. Damped-rotator fits: **T_e = 61 d clockwise (NH)
  vs 49 d counterclockwise (SH)** — anticyclonic in both hemispheres,
  |f_e| agreeing to ~20 %: the rotation period of the mean deep vortex,
  from displacement statistics alone.
* **Interior (20–50°)**: anticyclonic eddy clock decaying on the 10–30 d
  NR45 eddy memory, with small same-sign tails (weak curvature).
* **The polar reversal is a circulation-clock feature.** NH 65–85°:
  `ρ₁ = +0.119, t = +9.9` (cyclonic) but with **no zero-crossing and a
  31 % same-sign tail** — persistent rotation, i.e. the cyclonic
  subpolar/polar circulation (subpolar gyres, Nordic Seas rim) seen as
  trajectory curvature, *not* an eddy-population chirality flip. SH
  mirror: 50–65°S flips against 35–50°S (per-float means, t = −2.8 vs
  +12.0). The energy-weighted vs per-float distinction matters in the
  ACC band (jets dominate energy; the typical float is cyclonic) and is
  kept explicit in the cache.
* **The equator is null at every lag** (t = +1.0).

### Consequence for the program

The two-clocks paradigm is scale-free: it reappears *inside* the
parity-odd face. Protocol for any trajectory ensemble (floats, drifters,
tracked particles, cells): read `ρ_odd(τ)`; a sign-flipping damped
oscillation certifies coherent vortices and measures their rotation
period; a linear persistent component certifies large-scale curvature —
separable without maps, velocity fields, or eddy detection. NR52's
anticyclone dominance is now attributed: it belongs to the eddy clock of
the 5–50° interior; the polar cyclonic reversal belongs to the
circulation clock.

### Verdicts

| check | result |
|---|---|
| two spin clocks separable (synthetic) | **PASS** (f_e err 1.8 %, g ratio 1.29; drift control clean) |
| deep eddy rotation measured + mirrored | **PASS** (NH 61 d CW, SH 49 d CCW, crossings at 30 d) |
| polar reversal = circulation clock | **PASS** (NH polar t = +9.9, no crossing, tail 0.31) |
| equatorial null | **PASS** (t = +1.0) |

All references remain `[cite]` slots — none invented.
