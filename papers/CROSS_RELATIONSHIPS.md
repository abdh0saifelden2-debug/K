# New cross-cutting relationships: the four papers, each other, and the mainstream

This note answers a single question asked after the four drafts were finalised:
*what do the findings, taken together, imply — and can we derive new, useful
relationships between them and the existing literature?* Each relationship is
**derived** here and then **verified one-by-one** in
`glaciers/validation/synthetic/cross_relationships.py` (7 parametrised tests in
`glaciers/tests/test_validation_synthetic.py`). No external data; CPU only.

The unifying thread of all four papers is one operator-theoretic statement:

> **A local, memoryless, down-gradient (eddy-diffusivity / "K-theory") closure is
> the real-even-positive, Markovian-delta *projection* of an exact non-local
> operator. Each paper measures a different shadow that the projection throws
> away — a memory time, a migration, a backscatter, a fold.**

The relationships below make that statement quantitative and tie it to mainstream
results (Mori–Zwanzig/t-model, Kraichnan, Schoof MISI, Scheffer/Dakos early
warning, Gilpin/Hanratty, Gudmundsson, rough Rayleigh–Bénard).

| # | Relationship | Links | Status |
|---|---|---|---|
| NR1 | One memory number `Me=τ_fast/τ_slow` governs Markovian-closure breakdown | P1 ⇄ P4 ⇄ MZ/t-model | `[VERIFIED]` |
| NR2 | K-theory = real-even-positive projection of one complex admittance `Z` | P3 ⇄ P1 ⇄ Kraichnan/Hanratty | `[VERIFIED]` |
| NR3 | `s_N(N)` flotation pole = Schoof MISI saddle-node; CSD scaling | P4 ⇄ Schoof / Scheffer–Dakos | `[VERIFIED]` |
| NR4 | `I = |tan ψ|/2π`, ψ the Gilpin/Hanratty flux-topography phase | P3 ⇄ Gilpin/Hanratty | `[VERIFIED]` |
| NR5 | Melt ceiling = rough-RB no-enhancement branch (suppression ↑ steepness) | P2 ⇄ rough-RB | `[VERIFIED]` (decomp.); crossover `[CONJECTURE]` |
| NR6 | Tidal `2f/1f` velocity ratio = curvature of the sliding law | P4 ⇄ Gudmundsson | `[VERIFIED]` |
| NR7 | Intrusion residence `Ro` is a Damköhler number | P4 ⇄ transport theory | `[DERIVED]` |

---

## NR1 — One memory number governs every "local closure fails" verdict `[VERIFIED]`

**Statement.** Paper 1 shows single-`K` turbulence closure is the *Markovian-delta
collapse* of a Mori–Zwanzig (MZ) memory kernel; Paper 4 shows the literal thermal
sliding-lag kernel is wrong and the replacement is an *exact MZ memory kernel*
`K(τ)=M_sq M_qs e^{M_qq τ}` from eliminating the channel. These are the **same
object** — the impedance Green's function of an eliminated fast variable — and the
validity of the *local* (Markovian) closure in both is set by **one** dimensionless
number,

> `Me = τ_fast / τ_slow`   (eliminated-variable relaxation time / resolved timescale).

**Derivation.** For the lumped pair `ẋ = M x`, `M=[[−1/τ₁,−a],[b,−1/τ₂]]`,
eliminating `q` gives the generalized Langevin equation `ṡ = M_ss s + ∫K(t−τ)s dτ`
with `K(τ)=−ab e^{−τ/τ₂}` and DC gain `∫K = M_sq M_qs/(−M_qq)`. Holding the DC gain
fixed (so the Markovian closure `ṡ=(M_ss+∫K)s` is the correct *slow* limit), the
residual is purely the finite-memory correction, which vanishes as `τ₂→0`. This is
the Chorin–Hald–Kupferman / Stinis statement that the Markovian approximation
improves as the timescales separate.

**Verification.** Sweeping `Me∈[0.02,0.5]`, the L∞ closure error rises monotonically
`0.7% → 6.2%`, vanishing as a positive power (`≈Me^0.69`) — the same criterion in
both systems. Paper 1's measured `τ_c≈0.02–0.03` and Paper 4's channel time `τ₂`
are the two instances. (`nr1_memory_number`.)

---

## NR2 — K-theory is the real-even-positive projection of one complex admittance `[VERIFIED]`

**Statement.** A down-gradient closure is a **real, even, sign-definite** transport
operator. Write the exact linear transport response to a sinusoid as one complex
admittance `Z = a_r + i a_i`. Then:

- **Im Z = a_i** is the *quadrature* response: the scallop migration `E_cos`
  (Paper 3) **and** the closure memory phase (`τ_c`, Paper 1).
- **Re Z < 0** is *backscatter* — negative eddy viscosity (Paper 1, Kraichnan 1976).
- **K-theory = the projection** `Z ↦ max(Re Z, 0) + 0·i`. It zeroes the migration
  **and** the backscatter at once.

**Derivation.** For topography `y=a sin(kx)` the flux `e=Re[Z·ŷ·e^{ikx}]` with
`ŷ=−ia` gives `e = a[a_r sin(kx) + a_i cos(kx)]`, so `E_sin = a a_r` (amplitude rate,
Re s) and `E_cos = a a_i` (migration, Im s). A real diffusive flux of a sinusoid is
in-phase → `E_cos=0` (Paper 3's keystone). The transfer sign `T~−Re Z` is backscatter
iff `a_r<0` (Kraichnan's 2-D negative eddy viscosity).

**Verification.** With `Z=−1−0.6i`: exact `E_sin=−1`, `E_cos=−0.6`, migration
`I=0.0955`, backscatter present; the K-theory projection gives `E_cos=0`, `I=0`, no
backscatter. The migration and the backscatter are independent components that vanish
*together* only for a genuine down-gradient closure. (`nr2_admittance_unification`.)

---

## NR3 — The `s_N` flotation pole is the Schoof MISI saddle-node `[VERIFIED]`

**Statement.** Paper 4's effective-pressure master curve `|s_N|(N)=m/(1−(N_c/N)^m)`
has a simple pole at the flotation threshold `N_c`. That pole is the loss of basal-drag
restoring stiffness — i.e. the **Schoof (2007) grounding-line saddle-node** that
underlies marine ice-sheet instability — and the velocity critical-slowing-down
early-warning (Paper 4) is the **Scheffer (2009) / Dakos (2008)** generic precursor of
that fold.

**Derivation.** Near `N=N_c(1+δ)`, `R=(N_c/N)^m ≈ 1−mδ`, so the sliding sensitivity
`|s_N|=m/(1−R) ∝ δ^{−1}` (the pole), while the restoring stiffness
`dτ_b/du ∝ (1−R)²/R ∝ δ^{+2}`. The restoring eigenvalue `λ ∝ δ²` → 0 at the fold;
by the standard CSD result `Var ∝ 1/λ ∝ δ^{−2}` and `AC1 = e^{−λΔt} → 1`. Schoof's
flux `q_g ∝ h_g^{(m+n+3)/(m+1)} ≈ h_g^{4.75}` blows up at the same flotation thickness.

**Verification.** Fitted near-fold exponents: `|s_N|: −0.997`, stiffness `+1.998`,
variance `−1.998`; `AC1: 0.905 (far) → 0.9999 (near)`. An OU realization with `N`
declining toward `N_c` gives rising rolling variance and AC1 (Kendall `τ=0.54, 0.55`)
— the operational early-warning. (`nr3_misi_fold_ews`.)

---

## NR4 — The constant-free field ratio is the Gilpin/Hanratty phase `[VERIFIED]`

**Statement.** Paper 3's `ΔT`-free, constant-free field test
`I = Im(s)/(2π|Re(s)|)` is exactly

> `I = |tan ψ| / (2π)`,   `ψ = arg(s) = atan2(E_cos, E_sin)`,

where ψ is the **Gilpin–Hirata–Cheng (1980) / Hanratty (1981)** heat-flux-to-topography
phase shift. Downstream migration ⇔ `ψ∈(π/2,π)` (Gilpin's criterion) ⇔ damped
(`Re s<0`) and migrating (`Im s≠0`). The K-theory limit is `ψ→0` (`I→0`).

**Verification.** On Paper 3's own measured `(E_sin,E_cos)` table the two expressions
for `I` agree to machine precision (`0.168, 0.197, 0.248, 0.394`), and every point has
`|ψ|∈(90°,180°)` (`−133°…−112°`) — the damped-downstream branch. (`nr4_migration_phase`.)

---

## NR5 — The melt ceiling is the rough-RB no-enhancement branch `[VERIFIED]` decomposition

**Statement.** Paper 2's melt ceiling `Nu/Nu_flat ≤ 1` is the **no-enhancement** branch
of rough Rayleigh–Bénard heat transfer. Writing `Nu/Nu_flat = A·f_loc` with `A` the
wetted-area gain and `f_loc` the local-flux factor, the ceiling *proves* the cold wall
suppresses local flux faster than the geometry adds area.

**Derivation.** For `y=a sin(kx)`, `A=⟨√(1+y'²)⟩ ≈ 1+π²(a/λ)² > 1` always. Since
Paper 2 measures `Nu/Nu_flat ≈ 0.96–0.97 ≤ 1`, the implied `f_loc = (Nu/Nu_flat)/A < 1`
and the suppression `1−f_loc` must **rise** with steepness — exactly cancelling the
area gain. Enhancement (`Nu/Nu_flat>1`) requires `f_loc>1/A`, i.e. the Paper-2 §G.6
lee-flux growth term to overcome the deficit — a regime crossover flagged
`[CONJECTURE]` (it needs the lee/separation data to place).

**Verification.** Across `a/λ∈[0.05,0.20]`: `A: 1.02→1.39`, `Nu/Nu_flat: 0.97→0.96`,
implied `f_loc: 0.95→0.69`, suppression `0.05→0.31` (monotone rising). (`nr5_melt_ceiling_suppression`.)

---

## NR6 — The tidal `2f/1f` ratio measures the curvature of the sliding law `[VERIFIED]`

**Statement.** Under a tide `N(t)=N₀(1+ε cos ωt)`, the surface-velocity second-harmonic
ratio is

> `2f/1f = (ε/4) · |s_N'/s_N − 1 + s_N|`,   `s_N' = d s_N/d ln N`,

i.e. it reads the **curvature** of the sliding law and diverges toward `N_c`. This is
the sliding-law origin of Gudmundsson's observed nonlinear (MSf) tidal response, and
answers the "no knowledge of `N`" objection (it is recovered from velocity alone).

**Derivation.** Taylor-expanding `ln u_b(ln N)` to second order and exponentiating
(velocity, not log-velocity, is the observable — hence the `+s_N` from `exp`) gives the
fundamental amplitude `∝ s_N ε` and the second harmonic `∝ (ε²/4)(s_N'−s_N+s_N²)`.

**Verification.** Synthetic tide + FFT at `N₀/N_c = 20,8,3,1.5`: the measured `2f/1f`
rises monotonically toward flotation and matches the closed-form to ≤15%.
(`nr6_tidal_curvature`.)

---

## NR7 — The intrusion residence number is a Damköhler number `[DERIVED]`

**Statement.** Paper 4's `Ro = v_kin·τ_hyd/ℓ` (with `v_kin = A·dH/dt`) is a **Damköhler
number**: the ratio of the kinematic intrusion-front advance to the hydraulic
relaxation. `Ro=1` at `τ_crit = ℓ/v_kin` separates thinning-paced (`Ro<1`) from
hydraulic-limited (`Ro>1`) intrusion — the transport-vs-process crossover familiar
from reaction–transport theory.

**Verification.** With Paper-4 runaway-tail numbers (`v_kin=1.05 km/yr`, `ℓ=1 km`):
`τ_crit=0.95 yr`; sweeping `τ_hyd` gives `Ro=0.011,0.105,1.05,2.1`, flipping regime at
`Ro=1`. (`nr7_residence_damkohler`.)

---

### What is genuinely new vs. a sharpening

- **New unifications:** NR1 (one memory number across turbulence *and* sliding),
  NR2 (one complex admittance whose two parts are Paper 3 migration and Paper 1
  backscatter), NR3 (the `s_N` pole *is* the Schoof MISI saddle-node, with the CSD
  exponents derived).
- **Sharpenings of connections the papers already gesture at:** NR4 (the explicit
  `I=|tan ψ|/2π` identity to Gilpin/Hanratty), NR6 (the closed-form tidal curvature
  ratio), NR5 (the area/flux decomposition of the melt ceiling), NR7 (naming `Ro` a
  Damköhler number).

### Honest limits

NR1's error exponent is metric-dependent (`≈0.69` in L∞); the claim is the *monotone
vanishing*, not a universal power. NR2's admittance is a linear, single-mode model of
the discarded structure, not a turbulence closure. NR3 identifies the bifurcation and
the local EWS scaling; it does not run a full grounding-line model. NR5's enhancement
crossover is a conjecture pending lee-flux data. NR6 is single-mechanism (tidal `N`
modulation only). All references remain `[cite]` slots — none invented.
