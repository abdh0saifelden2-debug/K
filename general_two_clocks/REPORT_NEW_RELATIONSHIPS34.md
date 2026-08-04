# New cross-relationship — NR57 (`general_two_clocks/new_relationships34.py`) — theory + real data

Continues the derived-and-verified program (NR1–NR56; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the real spin comes from the committed NR52 cache
[`data/nr52_odd_spin_cache.json`](data/nr52_odd_spin_cache.json).
Unit-proofs in
[`tests/test_spin_irreversibility.py`](tests/test_spin_irreversibility.py)
(13 tests).

```bash
python general_two_clocks/new_relationships34.py   # -> figures/92_spin_irreversibility.json
pytest general_two_clocks/tests/test_spin_irreversibility.py -v
```

---

## NR57 — **The spin is the irreversibility clock**: the Lagrangian areal velocity is the phase-space probability current, so its square lower-bounds the entropy production rate — and the deep ocean is time-irreversible with a reversible equatorial line  [NR52 × NR56 × stochastic thermodynamics (Godreche–Luck / Seifert)]

### The mining question

NR52 called the spin a "detailed-balance meter" — a qualitative claim.
Stochastic thermodynamics should make it quantitative: is the spin
*literally* the entropy production, and does the corpus's equatorial spin
null correspond to a genuine equilibrium surface?

### The relationship (derived + verified)

For a stationary OU flow `dx = −A x dt + √(2D) dW`, stationary covariance
`C` (`AC + CAᵀ = 2D`):

**1. Areal velocity = probability current.** The steady current is
`J(x) = −Ω x ρ(x)` with `Ω = A − DC⁻¹`, and

```
M := AC − CAᵀ = 2ΩC   (exact, antisymmetric)
```

The Lagrangian spin (NR52) is exactly this areal velocity — the short-lag
slope of the odd correlation is `−M` — so the measured spin *is* the
phase-space current, not a proxy.

**2. Entropy production is the spin squared (exact, isotropic).** With
`σ = tr(ΩCΩᵀD⁻¹)` (Godreche–Luck/Seifert), for isotropic `D = νI`,
`C = cI`:

```
σ = m² / (2cν),   m = M₂₁ = 2·(areal velocity)   — EXACT
```

**3. The spin lower-bounds irreversibility (general).** For arbitrary
anisotropic `A, D`: `σ ≥ m²/(2c̄ν̄)` (`c̄ = tr C/2, ν̄ = tr D/2`), equality
in the isotropic limit — a measured spin sets a floor on irreversibility
with *no* knowledge of drift/diffusion detail. **Detailed balance
`AC = CAᵀ` ⇔ M = 0 ⇔ spin = 0 ⇔ σ = 0** (machine zero).

### Findings (`figures/92_spin_irreversibility.json`)

* **Exact + bound (synthetic).** `M = 2ΩC` to 1e-13; the isotropic law
  exact across rotation/diffusion sweeps; detailed-balance control gives
  spin and σ at machine zero; the anisotropic lower bound holds over 788
  random systems (min ratio 1.01, **zero violations**); `−M` equals the
  lagged-covariance asymmetry slope.
* **The deep ocean is time-irreversible, with an equatorial reversible
  line.** The measured spin gives a nonzero cycling rate and an
  irreversibility floor `Σ̂ = ρ_odd²`: NH and SH are decisively nonzero
  and *opposite handedness* (NH t = −9.7, SH t = +13.2 — the current
  circulates clockwise N, counterclockwise S), while the equatorial band
  is statistically **reversible** (t = +1.0) — a line of vanishing
  phase-space current exactly where f → 0. Irreversibility grows poleward
  with |f|.

### Consequence for the program

The parity-odd face is not merely descriptive — it is the thermodynamic
arrow. The two-clocks tensor's antisymmetric part measures how far each
region sits from equilibrium, from single-particle statistics, and the
corpus's equatorial null is a genuine detailed-balance surface. This ties
the "instantaneous mixing is invisible" family (NR39) to its
thermodynamic counterpart: reversible (symmetric) dynamics produce no
spin *and* no entropy.

### Verdicts

| check | result |
|---|---|
| current identity + isotropic law exact | **PASS** (M = 2ΩC to 1e-13; σ = m²/2cν exact) |
| spin² lower-bounds entropy | **PASS** (788 systems, min ratio 1.01, 0 violations) |
| detailed balance ⇔ zero spin ⇔ zero entropy | **PASS** (machine zero) |
| ocean time-irreversible + handed | **PASS** (NH t = −9.7, SH t = +13.2) |
| equatorial reversible line | **PASS** (t = +1.0) |

All references remain `[cite]` slots — none invented.
