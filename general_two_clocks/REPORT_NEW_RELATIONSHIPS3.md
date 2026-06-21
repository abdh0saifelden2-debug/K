# New cross-relationship — NR26 (`general_two_clocks/new_relationships3.py`)

Continues the derived-and-verified program (NR1–NR25; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, deterministic; unit-proofs in
[`tests/test_new_relationships3.py`](tests/test_new_relationships3.py). Run:

```bash
python general_two_clocks/new_relationships3.py        # -> figures/nr26_memory_ews_bias.{json,png}
pytest general_two_clocks/tests/test_new_relationships3.py -v
```

---

## NR26 — The two failure modes are *coupled* at finite distance: bath memory biases the critical-slowing-down early warning by the Deborah number  [P1 `De` × P4 fold]

### The gap this closes

The index organises NR1–NR25 into **two** universal ways the mainstream
local/equilibrium closure fails:

1. **Discarded memory / reactive operator structure**, organised by the Deborah (memory)
   number `De = τ_c/τ_event` (NR1, NR13, NR15);
2. **Criticality at a saddle-node fold** — the `s_N` flotation pole / MISI — with critical
   slowing down (rising variance + lag‑1 autocorrelation, NR3/NR22; Lorentzian corner
   `f_c ∝ (N−N_c)²`, NR8).

The index's organizing note argues these are **separate** because *"the memory ratio
`De = τ_c/τ_relax` **vanishes** at a fold, where `τ_relax → ∞`."* That asymptotic statement
is correct, but it conceals the operationally decisive fact: **at any finite distance from
the fold the leading correction to every CSD early-warning estimator is `O(De)`, it has a
definite sign, and a memoryless (white-noise) warning therefore mis-reads the distance to
tipping.** Every real early warning is read off a slow critical mode driven by a *fast but
finite-memory* bath — which is exactly the Mori–Zwanzig / second-FDT picture the repo
already certifies (NR25). So the two structures are not independent; they meet here.

### Setup (both structures in one minimal, mainstream model)

A slow mode `s` near the fold has a vanishing restoring rate `λ` (`λ → 0` at the fold;
`τ_relax = 1/λ` is the diverging CSD time). By the exact MZ/second-FDT reduction (NR25;
Zwanzig 1973) the eliminated fast bath supplies `s` with friction **and** a random force
that is *not white* but carries the bath's own memory time `τ_c`. The minimal realisation
is a slow linear mode driven by Ornstein–Uhlenbeck (colored) noise (Hänggi & Jung 1995):

> `ṡ = −λ s + η`,  `η̇ = −η/τ_c + (√(2D)/τ_c) ξ(t)`,  ⟨η(t)η(0)⟩ = (D/τ_c) e^{−|t|/τ_c},

so `η → ` white noise of intensity `D` as `τ_c → 0`, and **`De ≡ λτ_c = τ_c/τ_relax`** is
exactly the index's memory ratio.

### Derived consequences (each verified against the simulated process)

**(a) Variance suppression — a signed bias toward false safety.** From the spectrum
`S_s(ω) = 2D / [(ω²+λ²)(1+ω²τ_c²)]`,

> **`Var(s) = D / [ λ (1 + De) ]`.**

The white-noise CSD law is `Var = D/λ`; memory **suppresses** the variance by `1/(1+De)`.
A variance-based EWS that inverts `λ_V = D/Var` therefore reports

> **`λ_V = λ (1 + De) > λ`** — it reads the system as *farther* from the fold (falsely
> **safer**) by the factor `1+De`.

**(b) Bi-exponential ACF ⇒ the two standard precursors bracket the truth.**

> **`C(t)/C(0) = [ e^{−λt} − De · e^{−t/τ_c} ] / (1 − De)`** — a *difference of two
> exponentials* (rates `λ` and `1/τ_c`), not the single-exponential AR(1)/OU that CSD
> theory (Wissel 1984; Dakos 2012) assumes.

Consequently the lag‑1 (AC1) rate `λ_A = −ln AC1/Δt` reads **low** (`λ_A < λ`, false
**alarm**) while the variance rate reads **high** (false **safety**):

> **`λ_A < λ < λ_V`.**

The apparent rate `−ln AC(t)/t` **grows with lag**; its lag-slope is a
**noise-intensity-(`D`)-free** gauge of bath memory — zero only when `De = 0`.

**(c) De-biasable from the record alone.** A blind two-exponential fit of the measured
ACF recovers **both** rates — the true proximity `λ` *and* the bath memory `1/τ_c` —
removing the bias without knowing `D`.

### Numerical verification (`figures/nr26_memory_ews_bias.{json,png}`)

OU-driven slow mode, `τ_c = 1`, `D = 1`, `Δt = 2τ_c`, `n = 4×10⁶`, four distances from the
fold:

| `De = λτ_c` | `Var` sim / analytic | bracket `λ_A < λ < λ_V` | variance bias `λ_V/λ` (pred `1+De`) | lag-slope gauge |
|---|---|---|---|---|
| 0.05 | 19.25 / 19.05 (1.0%) | 0.028 < 0.05 < 0.052 ✓ | 1.039 (1.05) | +0.0091 |
| 0.10 | 9.09 / 9.09 (0.0%) | 0.055 < 0.10 < 0.110 ✓ | 1.100 (1.10) | +0.0183 |
| 0.20 | 4.15 / 4.17 (0.5%) | 0.109 < 0.20 < 0.241 ✓ | 1.206 (1.20) | +0.0368 |
| 0.40 | 1.78 / 1.79 (0.6%) | 0.210 < 0.40 < 0.563 ✓ | 1.408 (1.40) | +0.0724 |

- **Variance suppression `1/(1+De)`** reproduced to ≤1.1%.
- **Bracketing** `λ_A < λ < λ_V` holds at every distance; the variance-EWS false-safety
  bias is exactly `1+De`.
- **Lag-slope gauge** is positive and monotone in `De` (a `D`-free memory read-out).
- **De-bias:** a blind 2-exponential fit at `De=0.4` recovers `λ` to 4.4% and `τ_c` to 7.9%.

### Why it matters

This refines the repo's own central organizing claim. The two failure modes *are*
asymptotically distinct (`De → 0` at the fold), **but** the leading finite-distance
coupling is the Deborah number, and it is a **signed, removable** bias in the
critical-slowing-down early warning:

- a single-estimator (variance) CSD warning under a finite-memory bath **under-states
  tipping risk** by `1+De`;
- the disagreement between the variance and AC1 precursors (NR22's white-noise invariant
  `Var·(1−AC1²)=const` no longer holds) **measures** the memory and **de-biases** the
  proximity — from the time series alone, no noise calibration needed.

It sharpens **Paper 4b**'s tidal-admittance / CSD ungrounding early-warning: where the
basal-water bath has a finite hydraulic memory `τ_c` (the cavity↔channel relaxation
certified in Paper 4a), a memoryless slowing-down fit is biased, and the same velocity
record both flags and corrects it.

**Mainstream tools (cited, not claimed):** Zwanzig (1973) and Hänggi & Jung (1995)
(GLE / colored noise); Scheffer et al. (2009), Dakos et al. (2012), Wissel (1984)
(critical-slowing-down early warning). The contribution is the *coupling* of these to the
repo's memory structure and the resulting signed, removable EWS bias.
