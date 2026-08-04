# New cross-cutting relationships, batch 3 (NR11–NR12)

Continuation of `CROSS_RELATIONSHIPS.md` (NR1–NR7) and `CROSS_RELATIONSHIPS2.md`
(NR8–NR10). Each is **derived** here and **verified one-by-one** in
`glaciers/validation/synthetic/cross_relationships3.py`
(`test_cross_relationship3_verified` + two targeted tests). No external data; CPU only.

| # | Relationship | Links | Status |
|---|---|---|---|
| NR11 | The tidal velocity **phase lag** `φ=arctan(ωRC)` measures the hydraulic residence time `RC` from surface velocity alone | P4 §7.3/§K ⇄ NR6 ⇄ Gudmundsson; Bode/KK | `[DERIVED, VERIFIED]` |
| NR12 | One ice clock `τ_d=κ/V̄²` sets **both** the §B.2 kernel cutoff and the §A.1 coupling rolloff; `ω_half·τ_d` is universal, so the rolloff inverts for `V̄` | §B.2 ⇄ §A.1 ⇄ Stefan diffusion | `[DERIVED, VERIFIED]` |

---

## NR11 — The tidal velocity phase lag measures the hydraulic residence time `[DERIVED, VERIFIED]`

**Statement.** NR6 read the sliding-law *curvature* from the tidal `2f/1f` amplitude
ratio. The **phase** of the tidal velocity response is the complementary, untapped
channel, and it measures a quantity §9 lists as an open gate: the hydraulic residence
time `RC` of §7.3/§K. If the subglacial hydrology low-passes the tidal/ocean head with
a lumped impedance `K_hydraulic(t)=(1/RC)e^{−t/RC}` (the §K analogy, *not* a
derivation), then the effective pressure — and hence the surface velocity,
`u' ∝ s_N N'` (phase-preserving) — lags the tide by

> `φ(ω) = arctan(ω RC)`,  gain `= |s_N|/√(1+(ω RC)²)`.

So a **tidal velocity phase-lag spectrum measures `RC` from surface velocity alone**,
without any knowledge of `N` — the phase analogue of NR6's amplitude reading. Gain and
phase are a causal (minimum-phase) pair, the **Bode gain–phase / Kramers–Kronig**
relation of NR9, so the two are not independent.

**Verification.** Integrating the lumped low-pass `RC ẋ = −x + cos ωt` (RK4) and
measuring the steady velocity phase at `ω∈{0.5,…,8}` recovers the planted `RC=0.30` to
**1.7×10⁻⁷** (`RC=tan φ/ω`); measured phase and gain match `arctan(ωRC)` and
`1/√(1+(ωRC)²)` to plotting precision. (`nr11_tidal_phase_hydraulic_rc`.)

**Honest scope.** The lumped `RC` kernel is the §K *analogy*, so NR11 is a falsifiable
*measurement recipe* (“if the hydrology is RC-like, the tidal phase gives `RC`”), not a
claim that the kernel is derived — earning the kernel itself is still §9 gate #2.

---

## NR12 — One ice clock `τ_d` sets the kernel cutoff and the coupling rolloff, and inverts for `V̄` `[DERIVED, VERIFIED]`

**Statement.** Two separate §-results share a single timescale, the ice thermal clock
`τ_d = κ/V̄²`:

- the §B.2 memory kernel `G(t)` (short-time `t^{−1/2}` tail, exponential cutoff at
  `τ_d`), and
- the §A.1 interface-coupling number `Λ(ω)` (half-power rolloff at `ω ∼ 1/τ_d`).

Because the transfer function depends on `s` only through `τ_d s`, the normalized
coupling `Λ(ω)/Λ(0)` is a **universal function of the product `ω τ_d`**. Hence the
half-power frequency obeys `ω_half · τ_d = const`, independent of `V̄`. Measuring the
rolloff (or, equivalently, the kernel cutoff) therefore **inverts for the basal
ablation velocity**

> `V̄ = √(κ / τ_d)`,  with `τ_d = c / ω_half`, `c = ω_half τ_d` (universal).

This is the ice-side analogue of NR11's hydraulic `RC` — a memory time read directly
from a response rolloff.

**Verification.** Across `V̄ = 0.03–1 m/yr`, `ω_half·τ_d = 2.55` is constant to
**7×10⁻¹⁶** (universality), so the calibrated constant recovers `V̄` to
**4×10⁻¹⁶**; the §B.2 kernel keeps its diffusive short-time slope `−0.511 ≈ −1/2`.
(`nr12_ice_clock_inversion`.)

---

### What is genuinely new vs. a sharpening

- **New:** NR11 (the tidal *phase* as an `RC` meter — a different observable from NR6's
  amplitude, and it targets the §9 open hydraulic gate), NR12 (the explicit
  universality of `ω_half·τ_d` that turns the §A.1/§B.2 rolloff into a basal-ablation-
  velocity inversion). Together they pair the two memory clocks the framework carries:
  hydraulic `RC` (NR11) and ice-thermal `τ_d` (NR12).

### Honest limits

NR11's `RC` kernel is the lumped §K analogy (falsifiable recipe, not a derived kernel).
NR12's inversion assumes the §B.2 semi-infinite-ice transfer function (the `H²/κ`
thin-ice replacement changes `τ_d` but not the universality argument). All references
remain `[cite]` slots — none invented.
