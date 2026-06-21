# New cross-cutting relationships, batch 4 (NR13–NR14)

Continuation of `CROSS_RELATIONSHIPS{,2,3}.md` (NR1–NR12). Each is **derived** here and
**verified one-by-one** in `glaciers/validation/synthetic/cross_relationships4.py`
(`test_cross_relationship4_verified` + two targeted tests). No external data; CPU only.

| # | Relationship | Links | Status |
|---|---|---|---|
| NR13 | The K-theory → MZ-memory hierarchy is a turbulent **Chapman–Enskog** expansion in a memory Deborah number `De=τ_c/τ_event`; truncation error `~ De^{p+1}` | P1 §G.5 ⇄ Chapman–Enskog / Burnett ⇄ MZ | `[DERIVED, VERIFIED]` |
| NR14 | The §6.2 RTN grounding-line concentration is the normal-CDF front `Φ((x₅₀−x)/w)` with width `w=σ_h/s` and centre `x₅₀=H(1−φ)/s` (from NR10's `h_af`) | P4 §6 ⇄ NR10 ⇄ Schoof flotation | `[DERIVED, VERIFIED]` |

---

## NR13 — Turbulent transport closure is a Chapman–Enskog hierarchy in a memory Deborah number `[DERIVED, VERIFIED]`

**Statement.** The progression *Fickian K-theory → single-`K` + the §G.5 memory
correction → higher memory corrections* is not ad hoc: it is a **Chapman–Enskog
(gradient) expansion** in a memory Deborah/Knudsen number

> `De = τ_c / τ_event`  (closure memory time / forcing timescale),

exactly the structure the `c1` framing invokes (Boltzmann→Navier–Stokes→Burnett via a
Knudsen expansion). A relaxational (Maxwell–Cattaneo) constitutive law for the turbulent
flux, `τ_c ∂_t J + J = −D ∇θ`, has the gradient expansion

> `J = −D [ g − τ_c ġ + τ_c² g̈ − … ]`,  `g = ∇θ`,

whose order-`p` truncation has error `~ De^{p+1}`. So **Fickian K-theory is the 0th
rung** (`O(De)` error), the **single-`K` closure plus the §G.5 term `−CMN·∇·(∂_tK ∇θ)`
is the next rung** — the Navier–Stokes→**Burnett-analogue**, `O(De²)` error — and so on.
The §G.5/§H.3 result ("naive `∝τ_c`, corrected `∝τ_c²`") is the first two rungs of this
ladder; NR13 makes the whole hierarchy and its expansion parameter explicit.

**Verification.** Sweeping `De∈[0.01,0.16]`, the truncation-error exponents are exactly
`1, 2, 3` for `p=0,1,2`, and each higher order is strictly more accurate at the smallest
`De` (errors `10⁻², 10⁻⁴, 10⁻⁶`). (`nr13_chapman_enskog_ladder`.)

**Honest scope.** The Maxwell–Cattaneo law is a single-relaxation-time model of the
memory; the "Burnett" label is a structural analogy (same gradient/Knudsen hierarchy),
not a claim that the coefficients equal kinetic-theory Burnett coefficients.

---

## NR14 — The RTN grounding-line concentration is a derived normal-CDF front `[DERIVED, VERIFIED]`

**Statement.** The §6.2 finding that `RTN>1` cells concentrate near the grounding line
(median distance 6 km vs 221 km) and that the ordering is "robust across `φ`, only the
magnitude scales" follows from NR10's height-above-flotation `h_af`. With `h_af(x)`
rising inland at gradient `s=dh_af/dx` and cell-to-cell variability `σ_h`, the `RTN>1`
fraction is a **normal-CDF front**

> `P(RTN>1 ∣ x) = P(h_af < h_thr) = Φ((x₅₀ − x)/w)`,
> width `w = σ_h/s`,  centre `x₅₀ = h_thr/s = H(1−φ)/s`.

So the concentration **length scale is `σ_h/s`** (variability over flotation gradient),
and raising `φ` (shrinking the buffer `h_thr=H(1−φ)`) **moves the front centre `x₅₀`
toward the grounding line — lowering the near-margin magnitude — at fixed width `w`**.
That is precisely §6.2's "ordering robust across `φ`, only magnitude scales," now derived
rather than observed.

**Verification.** On synthetic transects the probit-linear fit recovers `w = σ_h/s` to
**0.15%** across four `(s,σ_h)` settings (universal), `x₅₀ = H(1−φ)/s` to `<1 km` for
`φ=0.90/0.95/1.00` (centre moves `20→10→0 km` inland) at **fixed** width
(`4.01 km`), the near-GL fraction falls `1.00→0.99→0.51` as `φ` rises, and `RTN>1` cells
sit at median 2 km vs 31 km for the rest. (`nr14_rtn_concentration_length`.)

---

### What is genuinely new vs. a sharpening

- **New:** NR13 (the explicit Chapman–Enskog/Deborah hierarchy with the verified
  `De^{p+1}` error ladder — it places the §G.5 correction as one rung of a kinetic-theory-
  style expansion, directly engaging the `c1` molecular-origin framing), NR14 (the derived
  normal-CDF front, with the concentration length `σ_h/s` and the `φ`-translation-at-fixed-
  width that explains §6.2's magnitude-only `φ` dependence).

### Honest limits

NR13's ladder is for a single-relaxation-time flux model; real closures have a spectrum of
memory times, so the practical gain is the *order* improvement, not a universal prefactor.
NR14 is verified on synthetic transects (the real Bedmap2 concentration is §6.2); it
assumes a locally-linear `h_af(x)` with stationary Gaussian variability. All references
remain `[cite]` slots — none invented.
