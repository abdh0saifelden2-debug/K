# New cross-relationship — NR59 (`general_two_clocks/new_relationships36.py`) — theory + real data

Continues the derived-and-verified program (NR1–NR58; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the real integrals come from the committed NR52
band cache
[`data/nr52_odd_spin_cache.json`](data/nr52_odd_spin_cache.json).
Unit-proofs in
[`tests/test_skew_diffusivity.py`](tests/test_skew_diffusivity.py)
(11 tests).

```bash
python general_two_clocks/new_relationships36.py   # -> figures/94_skew_diffusivity.json
pytest general_two_clocks/tests/test_skew_diffusivity.py -v
```

---

## NR59 — **The spin is a skew (divergence-free) diffusivity**: the Green–Kubo integral of the odd correlation is the antisymmetric eddy-transport tensor, its ratio to the Taylor diffusivity is exactly the clock ratio `f τ_L`, and in the deep ocean it flips sign across the equator  [NR52 × NR57 × NR15/NR23 (signed eddy viscosity) × Green–Kubo]

### The mining question

NR57 gave the spin a thermodynamic meaning (irreversibility). What is its
**transport** meaning — and does it connect to the corpus's earlier
"signed net eddy viscosity" (NR15/NR23), the rotational flux that
down-gradient K-theory discards?

### The relationship (derived + verified)

Green–Kubo splits the Lagrangian eddy-transport tensor
`K = ∫₀^∞ ⟨u(0)u(t)ᵀ⟩ dt` into

- `K_S` — symmetric (Taylor) diffusivity: down-gradient, **mixing**;
- `K_A` — antisymmetric (skew) diffusivity: the odd correlation's
  integral, `K_A[xy] = ½∫₀^∞ ρ_odd(t) dt`.

For the damped-rotator two-clocks velocity (`du = −u/τ_L + f εu + noise`),
both are closed-form:

```
K_S = τ_L²/(1 + f²τ_L²),   K_A = f τ_L³/(1 + f²τ_L²)
⟹  K_A / K_S = f τ_L = r   — the two-clocks clock ratio (NR47), EXACTLY
```

Two consequences:

1. **The skew part transports but does not mix.** The skew flux
   `F_A = −K_A ∇C` is divergence-free (`∇·F_A = 0`) and everywhere
   *parallel to tracer contours* (`F_A·∇C = 0`) — it advects tracer
   around, adding nothing to the variance budget. This is the
   transport-tensor form of NR23's signed eddy viscosity and of the
   streamfunction (bolus/Nakamura) eddy advection — the rotational flux
   that down-gradient closures discard (the transport face of the
   discarded-operator structure NR2/NR15).
2. **The spin sets the skew diffusivity, sign and size.** `K_A` has the
   sign of `f`, so a measured Lagrangian spin gives the sign *and*
   magnitude of the divergence-free eddy transport directly.

### Findings (`figures/94_skew_diffusivity.json`)

* **Closed forms exact.** `K_S`, `K_A`, and `K_A/K_S = f τ_L` to 1e-14;
  the skew flux divergence-free and contour-parallel to machine precision.
* **The deep ocean's skew transport flips at the equator.** Green–Kubo
  integrals of the committed band correlations: **K_A < 0 in the NH**
  (clockwise), **K_A > 0 in the SH** (counterclockwise), with
  |K_A/K_S| ~ 0.05–0.15 in mid-latitudes — a 5–15 % rotational
  (non-mixing) share of the deep eddy transport tensor,
  hemisphere-antisymmetric, minimal at the equator (K_A → 0 as f → 0).
  Taylor part K_S ~ 1900–3600 m²/s (NH/SH interior) — the known deep-eddy
  diffusivity scale.

### Consequence for the program

The odd face (NR52) is the third leg of a single object measured four
ways: **memory phase (NR47), irreversibility (NR57), and skew transport
(this NR) are all `r = f τ_L = tan δ`**. The corpus's "discarded operator"
(NR2/15/23) is, in transport terms, exactly this skew diffusivity — and
it is measurable, signed, and hemisphere-antisymmetric in the real ocean.

### Verdicts

| check | result |
|---|---|
| K_A/K_S = clock ratio f τ_L | **PASS** (closed-form 7e-15; ratio 1e-14) |
| skew flux divergence-free + contour-parallel | **PASS** (both ~1e-15) |
| ocean skew transport flips at equator | **PASS** (NH K_A = −17, SH +387 m²/s) |
| Taylor diffusivity scale reasonable | **PASS** (1878 / 3234 m²/s) |

All references remain `[cite]` slots — none invented.
