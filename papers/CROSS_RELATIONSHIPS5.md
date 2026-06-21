# New cross-cutting relationship, batch 5 (NR15)

Continuation of `CROSS_RELATIONSHIPS{,2,3,4}.md` (NR1–NR14). Derived here and verified in
`glaciers/validation/synthetic/cross_relationships5.py`
(`test_cross_relationship5_verified` + one targeted test). No external data; CPU only.

| # | Relationship | Links | Status |
|---|---|---|---|
| NR15 | The **signed** time-integral of the memory kernel is the *net* eddy viscosity `ν_eff=∫K`; a backscatter-dominated kernel gives `ν_eff<0` (Kraichnan) → resolved-mode growth, impossible for positive K-theory | P1 ⇄ NR2 ⇄ Kraichnan; MZ 2nd-FDT | `[DERIVED, VERIFIED]` |

---

## NR15 — The backscatter budget is the signed kernel integral `[DERIVED, VERIFIED]`

**Statement.** NR2 showed K-theory keeps only the real, sign-definite (forward) part of
the transport admittance and projects out the backscatter (`Re Z<0`). NR15 is the
*dynamical, energy-budget* version. For a Mori–Zwanzig memory kernel `K(τ)`, the
Markovian (local) limit is a single **net eddy viscosity**

> `ν_eff = ∫₀^∞ K(τ) dτ`  (the DC gain of the kernel).

Splitting `K` into a forward lobe (weight `c_f`) and a delayed negative **backscatter**
lobe (weight `c_b`) gives `ν_eff = c_f − c_b`. So:

- `c_b < c_f` (`ν_eff>0`): net-dissipative — the resolved mode **decays**.
- `c_b > c_f` (`ν_eff<0`): **Kraichnan negative eddy viscosity** — the resolved mode
  **grows**, energy returned from sub-grid scales.

A strictly positive K-theory eddy viscosity can only ever produce decay; it is
*structurally* unable to represent `ν_eff<0`. The backscatter fraction `β=c_b/c_f`
controls the crossover, with the sign flip exactly at `β=1`.

**Verification.** On the exponential-memory GLE (exact Markovian embedding): `∫K = c_f−c_b`
to `<10⁻⁶`; the net-dissipative kernel decays (rate `−1.02`, `x→5×10⁻¹⁸`); the
backscatter-dominated kernel grows (rate `+0.48`, `x→2×10⁸`); and sweeping `β`, the
asymptotic-rate sign flips exactly at `β=1` (`ν_eff=0`). (`nr15_backscatter_budget`.)

**Honest scope.** A two-lobe exponential kernel is a minimal model of the discarded
backscatter structure; the claim is the *sign law* `ν_eff=∫K` and its dynamical
consequence, not a turbulence-closure prediction of `c_f,c_b` (those are pinned
separately, cf. RESULT 12 / `gle_coefficients.py`).
