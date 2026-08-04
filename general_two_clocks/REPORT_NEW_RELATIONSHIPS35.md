# New cross-relationship — NR58 (`general_two_clocks/new_relationships35.py`) — theory

Continues the derived-and-verified program (NR1–NR57; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe, fully analytic (the B.2 kernel is a closed form).
Unit-proofs in
[`tests/test_ice_kernel_anatomy.py`](tests/test_ice_kernel_anatomy.py)
(11 tests).

```bash
python general_two_clocks/new_relationships35.py   # -> figures/93_ice_kernel_anatomy.json
pytest general_two_clocks/tests/test_ice_kernel_anatomy.py -v
```

---

## NR58 — **The ice memory kernel in the response anatomy**: the B.2 tempered-Warburg kernel is pure minimum-phase memory (no transport, no handedness), and its tempering rate is *exactly* the Kramers–Kronig crossover of its phase  [NR34 × NR47 × NR48 × NR50 × NR55 — bridging cryosphere to the response anatomy]

### The mining question

NR47–NR57 built the four-phase response anatomy almost entirely on ocean/
solar/EEG data. Does it classify the *cryosphere* half of the corpus?
NR34 proved the B.2 interface kernel is a tempered half-derivative — where
does it sit in the anatomy, and what does the anatomy reveal about it?

### The relationship (derived + verified)

The exact B.2 transfer (NR34), with Warburg coefficient `W = −2A√τ_d` and
tempering rate `λ = 1/(4τ_d)`, is `H(s) = A/s + W√(s+λ)/s`, whose
non-trivial factor is the tempered Warburg `χ(s) = √(s+λ)`. In the
anatomy:

1. **P1 memory — minimum-phase, 0 → 45°, crossover at λ.**
   `δ(ω) = arg √(iω+λ) = ½·arctan(ω/λ)`: monotone from 0 (elliptic pole,
   ω ≪ λ) to 45° (Warburg pole, ω ≫ λ), through **exactly 22.5° at
   ω = λ**. The kernel's exponential tempering cutoff (`e^{−t/4τ_d}`) is
   *identically* the KK crossover frequency of its memory phase — one
   number λ governs both the time-domain cutoff and the frequency-domain
   phase transition. The phase-area theorem (NR50) closes exactly.
2. **P2 transport — none.** The excess phase (NR48) has zero slope
   (< 1e-3 τ_d): diffusive ice memory is not a transport delay — the
   opposite corner from the solar all-pass pair.
3. **P4 handedness — none.** `χ` is scalar (1D conduction); the spin face
   is identically zero — a pure symmetric two-clocks response.
4. **Five-faces order parameter (NR47).** As ω ≫ λ the fractional order
   → ½ in *both* faces (`2δ/π` and the Bode amplitude slope
   `d ln|χ|/d ln ω`) — the half-order Caputo/Randles–Warburg contact
   point (NR34 claim 2).

### Findings (`figures/93_ice_kernel_anatomy.json`)

* `δ(λ) = 22.5°` to 1e-6 — the crossover *is* the tempering rate;
  `δ → 0°/45°` at the two poles.
* minimum-phase reconstruction of `δ` from `|χ|` alone (Bode) to 0.00°
  median; phase-area theorem exact (rel err ~ 0).
* excess-phase transport delay `~1e-4 τ_d` (zero); both fractional-order
  faces → 0.500 in the Warburg band.
* placement matrix: ice kernel = **(memory 0→45°, transport 0, spin 0)** —
  a *new corner* distinct from EEG (memory 0), ocean (memory 90° + spin),
  solar (transport), scallops (transport + weak parity).

### Consequence for the program

The response anatomy (NR55) now classifies **both halves** of the corpus —
cryosphere and ocean — on the same axes. The ice kernel occupies the
"tempered half-order memory, no transport, no handedness" corner, and its
single physical parameter `τ_d = κ/V̄²` sets its lone anatomy coordinate
(the KK crossover `λ = 1/4τ_d`). The anatomy is not ocean-specific — it is
the universal chart of the whole two-clocks program.

### Verdicts

| check | result |
|---|---|
| tempering rate = KK crossover | **PASS** (δ(λ) = 22.5°; poles 0°/45°) |
| ice kernel is minimum-phase | **PASS** (recon 0.00°; area exact) |
| no transport clock | **PASS** (excess delay ~1e-4 τ_d) |
| Warburg half-order limit | **PASS** (both faces → 0.500) |

All references remain `[cite]` slots — none invented.
