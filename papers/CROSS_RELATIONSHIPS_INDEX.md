# Cross-cutting relationships — index (NR1–NR31)

A running index of the **derived + one-by-one-verified** relationships connecting the
four manuscripts to each other and to the mainstream literature. Batches:

- `CROSS_RELATIONSHIPS.md` — NR1–NR7 (`cross_relationships.py`)
- `CROSS_RELATIONSHIPS2.md` — NR8–NR10 (`cross_relationships2.py`)
- `CROSS_RELATIONSHIPS3.md` — NR11–NR12 (`cross_relationships3.py`)
- `CROSS_RELATIONSHIPS4.md` — NR13–NR14 (`cross_relationships4.py`)
- `CROSS_RELATIONSHIPS5.md` — NR15 (`cross_relationships5.py`)
- `general_two_clocks/REPORT_NEW_RELATIONSHIPS.md` — NR16–NR21 (`general_two_clocks/new_relationships.py`)
- `general_two_clocks/REPORT_NEW_RELATIONSHIPS2.md` — NR22–NR25 (`general_two_clocks/new_relationships2.py`)
- `general_two_clocks/REPORT_NEW_RELATIONSHIPS3.md` — NR26 (`general_two_clocks/new_relationships3.py`)
- `general_two_clocks/REPORT_NEW_RELATIONSHIPS4.md` — NR27 (`general_two_clocks/new_relationships4.py`)
- `general_two_clocks/REPORT_NEW_RELATIONSHIPS5.md` — NR28 (`general_two_clocks/new_relationships5.py`)
- `general_two_clocks/REPORT_NEW_RELATIONSHIPS6.md` — NR29 (`general_two_clocks/new_relationships6.py`)
- `general_two_clocks/REPORT_NEW_RELATIONSHIPS7.md` — NR30 (`general_two_clocks/new_relationships7.py`)
- `general_two_clocks/REPORT_NEW_RELATIONSHIPS8.md` — NR31 (`general_two_clocks/new_relationships8.py`)

All are CPU-only, no external data; each `nrN_*` returns `ok` and is unit-tested in
`glaciers/tests/test_validation_synthetic.py` (`test_cross_relationship{,2,3,4,5}_verified`
+ targeted tests). Run them all:

```bash
for n in "" 2 3 4 5; do python glaciers/validation/synthetic/cross_relationships$n.py; done
pytest glaciers/tests/test_validation_synthetic.py -k cross_relationship
python general_two_clocks/new_relationships.py && python general_two_clocks/new_relationships2.py
pytest general_two_clocks/tests/test_new_relationships.py general_two_clocks/tests/test_new_relationships2.py
python general_two_clocks/new_relationships3.py && python general_two_clocks/new_relationships4.py
pytest general_two_clocks/tests/test_new_relationships3.py general_two_clocks/tests/test_new_relationships4.py
python general_two_clocks/new_relationships5.py
pytest general_two_clocks/tests/test_new_relationships5.py
python general_two_clocks/new_relationships6.py
pytest general_two_clocks/tests/test_new_relationships6.py
python general_two_clocks/new_relationships7.py
pytest general_two_clocks/tests/test_new_relationships7.py
python general_two_clocks/new_relationships8.py
pytest general_two_clocks/tests/test_new_relationships8.py
```

| # | One line | Links | Status |
|---|---|---|---|
| NR1 | One memory number `Me=τ_fast/τ_slow` governs Markovian-closure breakdown | P1⇄P4⇄MZ | `[VERIFIED]` |
| NR2 | K-theory = real-even-positive projection of one complex admittance `Z` | P3⇄P1⇄Kraichnan | `[VERIFIED]` |
| NR3 | `s_N` flotation pole = Schoof MISI saddle-node; CSD exponents | P4⇄Schoof/Scheffer–Dakos | `[VERIFIED]` |
| NR4 | `I=|tan ψ|/2π`, ψ the Gilpin/Hanratty flux–topography phase | P3⇄Gilpin/Hanratty | `[VERIFIED]` |
| NR5 | Melt ceiling = rough-RB no-enhancement branch (area×local-flux) | P2⇄rough-RB | `[VERIFIED]` |
| NR6 | Tidal `2f/1f` ratio = curvature of the sliding law | P4⇄Gudmundsson | `[VERIFIED]` |
| NR7 | Intrusion residence `Ro` is a Damköhler number | P4⇄transport | `[DERIVED]` |
| NR8 | The fold's spectral face: Lorentzian corner `f_c∝(N−N_c)²`; FDT `Var·2πf_c=D` | P4/NR3⇄Bury/Kuehn⇄FDT | `[DERIVED, VERIFIED]` |
| NR9 | Causality (Kramers–Kronig) ties migration to eddy-viscosity dispersion | P3⇄P1⇄NR2⇄KK | `[DERIVED, VERIFIED]` |
| NR10 | Height above flotation `h_af` unifies RTN, the Schoof fold, and the `s_N` pole | P4§6⇄P4§10⇄Schoof | `[DERIVED, VERIFIED]` |
| NR11 | Tidal velocity **phase** lag `φ=arctan(ωRC)` measures the hydraulic residence `RC` | P4§7.3⇄NR6⇄Gudmundsson | `[DERIVED, VERIFIED]` |
| NR12 | One ice clock `τ_d=κ/V̄²` sets kernel cutoff + coupling rolloff; inverts for `V̄` | §B.2⇄§A.1⇄Stefan | `[DERIVED, VERIFIED]` |
| NR13 | K-theory→MZ is a turbulent Chapman–Enskog ladder in `De=τ_c/τ_event`; error `~De^{p+1}` | P1§G.5⇄Chapman–Enskog/Burnett | `[DERIVED, VERIFIED]` |
| NR14 | RTN GL concentration = normal-CDF front `Φ((x₅₀−x)/w)`, width `σ_h/s` | P4§6⇄NR10⇄Schoof | `[DERIVED, VERIFIED]` |
| NR15 | Net eddy viscosity `ν_eff=∫K`; backscatter-dominated kernel → `ν_eff<0` (growth) | P1⇄NR2⇄Kraichnan | `[DERIVED, VERIFIED]` |
| NR16 | Lowen–Teich: power-law bursts are a `1/f^γ` spectrum, `γ=3−α` | P3⇄Lowen–Teich | `[VERIFIED]` |
| NR17 | Power-law spectrum ⇒ growing temporal memory `τ_int` | P3→P4⇄Wiener–Khinchin | `[VERIFIED]` |
| NR18 | Exact: exchange fraction `=√⟨k_x²/k²⟩` (buoyancy-power anisotropy) | P2⇄Helmholtz/Leray | `[VERIFIED]` |
| NR19 | Green–Kubo/Taylor: integrated memory time = eddy diffusivity | P4⇄Taylor/Green–Kubo | `[VERIFIED]` |
| NR20 | Additive diffusivity `D_total=D_fast+D_slow` (multiscale memory) | P4⇄Taylor | `[VERIFIED]` |
| NR21 | Long-range dependence: DFA/Hurst `H=(γ+1)/2` | P3→P4⇄Mandelbrot/Peng | `[VERIFIED]` |
| NR22 | EWS precursors are one rate: `Var·(1−AC1²)=σ_ε²` (+ CSD-vs-noise discriminator) | P4⇄NR3/NR8⇄Scheffer/Dakos | `[DERIVED, VERIFIED]` |
| NR23 | Kramers–Kronig DC sum rule: net eddy viscosity `ν_eff=(2/π)∫(−Im Z)/ω dω` | P1⇄NR9/NR15⇄Kramers–Kronig | `[DERIVED, VERIFIED]` |
| NR24 | Eddy diffusivity is the zero-frequency PSD, `D=½S(0)`; `∝(N−N_c)⁻²` at the fold, `∞` for `1/f^γ` | P4⇄NR17/NR19⇄Green–Kubo | `[DERIVED, VERIFIED]` |
| NR25 | Second FDT: MZ memory kernel = random-force ACF, `⟨F(t)F(0)⟩=k_BT·K(t)` | P4(MZ)⇄NR1⇄Kubo/Zwanzig | `[DERIVED, VERIFIED]` |
| NR26 | The two failure modes **couple** at finite distance: bath memory biases the CSD early warning by `De`. `Var=D/(λ(1+De))` (variance→false safety `1+De`), AC1→false alarm, so `λ_A<λ<λ_V`; bi-exponential ACF is a `D`-free, removable memory gauge | P1(`De`)⇄P4(fold)⇄NR22/NR25⇄Hänggi/Zwanzig | `[DERIVED, VERIFIED]` |
| NR28 | The foundational two-clocks split as one cross-spectrum: elliptic (instantaneous) pressure **leads** parabolic (lagged) temperature by `φ=arctan(ω/ω_c)`; the 45° crossover **is** the parabolic clock `ω_c=κk²` (no calibration); and the p–T coherence **drops** at high `ω` — the small-scale decoupling itself. Phase one-sided & bounded `<90°` (diffusive lag, not a transport delay) | core thesis (elliptic `p`⇄parabolic `θ`)⇄REPORT_RB/THEORY⇄Bendat–Piersol/Leray | `[DERIVED, VERIFIED]` |
| NR29 | The Part-9c backscatter volume fraction is a Gaussian flux law `P(Π<0)=Φ(−⟨Π⟩/σ_Π)` — ~½ by the CLT, just below it because the cascade is net-forward (`μ/σ≈0.05`); the few-×10⁻³ residual is the positive flux skewness; Smagorinsky's one-sided `Π≥0` gives exactly 0 (the `μ/σ→∞` limit). Verified on a CPU DNS + the six P100 GPU runs | P1 closure (Part 9c)⇄CLT/Edgeworth⇄Piomelli/Cerutti–Meneveau | `[DERIVED, VERIFIED]` |
| NR27 | First-FDT/response dual of NR26: the response (tidal admittance) `χ=1/(λ+iω)` measures fold proximity **noise-free** (`\|χ(0)\|=1/λ`), while the FDT effective temperature `T_eff(ω)=ωS/(2\|Im χ\|)=D/(1+ω²τ_c²)` rolls off at `1/τ_c` (PSD slope `−2→−4`) to measure the bath memory — `De` de-biased from response+fluctuation | P4b(admittance)⇄P4a(`τ_c`)⇄NR25/NR26⇄Kubo/Harada–Sasa | `[DERIVED, VERIFIED]` |
| NR30 | The subglacial hydraulic potential `φ` is **not** a Leray pressure: `φ` is the parabolic (finite-time, screened) Darcy head `H_φ=1/(k²+iω/D_h)` of a bed *with storage*, the Leray `p` the elliptic (instantaneous, real) multiplier `H_p=1/k²` *without* storage; they coincide only as `τ_hyd→0`, and at finite storage the head lags by `arctan(ω/D_h k²)` (45° at `ω_c=D_h k²`) — a field-measurable separator | P4§9/P4a§8(`φ`)⇄P1(Leray `p`)⇄NR28/NR11 | `[DERIVED, VERIFIED]` |
| NR31 | Paper 1's 2-D a-posteriori null **is** the inverse cascade: `ν_opt=⟨Π⟩/(2⟨|S|²⟩)`, so the optimal resolved eddy viscosity has the sign of the net inter-scale energy flux; 2-D is inverse (`⟨Π⟩≈0`, `ν_opt≈0` → no positive `ν_t` beats no-model), 3-D is forward (`⟨Π⟩>0`, the NR29 sign, `ν_opt>0` → a closure must help). A falsifiable 2-D-vs-3-D ordering | P1§5b/§8.1⇄NR29⇄Kraichnan/Batchelor | `[DERIVED, VERIFIED]` |

## The organizing observation (a physical reality)

Read together, the first 15 relationships partition into **two universal structures** — the two
ways the mainstream *local / equilibrium* description fails in this work:

1. **Discarded operator structure (memory / reactive part).** NR1, NR2, NR4, NR9, NR11,
   NR12, NR13, NR15. A *local, memoryless, down-gradient* closure is the
   real-even-positive, Markovian-delta **projection** of an exact non-local, causal,
   complex response. What it throws away is a **reactive/quadrature part** (migration
   `Im Z`, backscatter `Re Z<0`, memory time `τ_c`/`τ_d`/`RC`). These are tied together
   by **causality (Kramers–Kronig, NR9)** and organised by a **memory Deborah/Knudsen
   number** whose vanishing is the Markovian limit (NR1/NR13); the correction enters as a
   Chapman–Enskog gradient hierarchy (NR13) and its sign budget is fixed by the *signed*
   kernel integral (NR15). Canonical signature: **closure error `∝ (clock ratio)^{p+1}`**.

2. **Criticality proximity (a fold).** NR3, NR5, NR6, NR7, NR8, NR10, NR14. A single
   state variable — effective pressure `N` / height above flotation `h_af` — controls a
   **saddle-node** (Schoof MISI). Approaching it, the basal-drag restoring stiffness
   `∝(N−N_c)²→0`, so the sliding sensitivity `|s_N|` has a **simple pole** and the system
   shows **critical slowing down** (rising variance/AC1, NR3; a Lorentzian spectral
   corner `f_c∝(N−N_c)²`, NR8). The same `h_af` sets where ocean intrusion (RTN)
   concentrates (NR10/NR14). Canonical signature: **response `∝ (proximity)^{−1}`,
   variance `∝ (proximity)^{−2}`**.

These two are *asymptotically distinct* (the memory ratio `De=τ_c/τ_relax` actually
**vanishes** at a fold, where `τ_relax→∞`, so criticality is not "infinite memory" — they
are separate failure modes, not the same divergence). Every NR1–NR25 above is one instance
of structure (1) or (2); the unifying claim of the four papers is that **mainstream
closures fail in exactly these two ways, and both are quantitatively repairable /
measurable**.

**The finite-distance coupling (NR26).** "Separate" is an *asymptotic* statement. At any
finite distance from the fold the two structures **couple**, and the coupling is exactly
the Deborah number: a slow critical mode (2) driven by a finite-memory bath (1) has its
critical-slowing-down variance **suppressed by `1/(1+De)`**, so a white-noise early warning
is biased toward false safety by `1+De` (variance) while the AC1 precursor reads false
alarm — the two bracket the truth (`λ_A<λ<λ_V`). The bias is `O(De)` (it does vanish at
the fold, consistent with the above) but it is the *leading* finite-distance correction,
it is signed, and it is **removable** from the record (the bi-exponential ACF recovers both
`λ` and `τ_c`). NR26 is thus the explicit bridge between structures (1) and (2) at the
place they were declared independent.

**The response dual (NR27).** NR26 couples the two structures through *correlations*; NR27
supplies the *response* half of the same fluctuation–dissipation statement. The fold mode's
admittance `χ(ω)=1/(λ+iω)` is the bare relaxation — it carries no noise, so the static
admittance `|χ(0)|=1/λ` reads proximity to the fold (structure 2) with no noise calibration
(the response face of Paper 4b's `|s_N|`), while the equilibrium FDT between that response
and the fluctuation spectrum is broken by the bath memory (structure 1) as a
frequency-dependent effective temperature `T_eff(ω)=D/(1+ω²τ_c²)` whose roll-off (and the
PSD slope steepening `−2→−4`) measures `τ_c`. So De-coupling has two field faces: a
*correlation* bias (NR26) and a *response/FDT-violation* read-out (NR27), and the two
together de-bias the critical-slowing-down ungrounding warning from one velocity record.

**The foundational observable (NR28).** NR1–NR27 mine *consequences* of the two-clocks
split (closure memory, the fold, the FDT); NR28 writes the split *itself* as one
cross-spectral signature on a co-located pressure+temperature pair. Because the elliptic
pressure is the **instantaneous** diagnostic of the strain and the parabolic temperature
the **lagged** integrator, their cross-phase is the bounded, one-sided `arctan(ω/ω_c)`
whose 45° crossover **measures the parabolic clock** `ω_c=κk²` with no calibration, while
the p–T **coherence drop** at high `ω` *is* the small-scale decoupling the core thesis
asserts. It grounds the whole program in a measurement on exactly the co-located p+T
records (NEON/ASOS/reanalysis) the repo already uses elsewhere.

**Deriving the keystone number (NR29).** Part 9c's headline — that K-theory represents
0 % of the ~½ of space where 3D turbulence backscatters — gets its *value* in NR29: the
backscatter volume fraction is `Φ(−⟨Π⟩/σ_Π)`, the standard-normal CDF of the (small)
inverse signal-to-noise of the subgrid flux `Π=−τ^d_ij S_ij`. It is ~½ by the CLT, just
below because the cascade is net-forward, tracks `μ/σ` as resolution and filter change
(verified across the six P100 runs), and is identically 0 for a one-sided eddy-viscosity
flux — so K-theory's failure is the degenerate `μ/σ→∞` limit of one law.

**Distinguishing the two pressures (NR30).** A recurring referee-style objection to Paper 4/4a
is that the subglacial hydraulic potential `φ` might just *be* a Leray pressure (both "enforce a
constraint"); both papers flag this as an open caveat. NR30 closes it structurally: the Leray
pressure is the **elliptic, instantaneous** constraint multiplier `H_p=1/k²` of an incompressible
flow with **no storage** (real transfer, zero phase, no memory), whereas `φ` is the **parabolic,
finite-time** Darcy head `H_φ=1/(k²+iω/D_h)` of a bed **with storage `S`** (screened transfer, a
phase lag `arctan(ω/D_h k²)`, a relaxation time `τ_k=1/(D_h k²)`). The storage `S` (hydraulic time
`τ_hyd`) is the order parameter: the two coincide only in the singular `τ_hyd→0` limit a
water-storing bed never reaches, and the distinctness is **field-measurable** as a nonzero
forcing→head phase lag. NR30 places the subglacial `p`/`φ` pair into the same elliptic/parabolic
two-clock structure NR28 gave the core thesis.

**Why the 2-D a-posteriori test is null (NR31).** Paper 1 honestly reports that in pure 2-D no
eddy-viscosity closure beats no-model on the resolved spectrum. NR31 shows this is the **2-D
inverse cascade**, not a closure defect: the optimal resolved eddy viscosity is
`ν_opt=⟨Π⟩/(2⟨|S|²⟩)`, so its sign is the sign of the net inter-scale energy flux at the cutoff.
In 2-D the energy cascade is inverse (Kraichnan 1967), so the forward energy flux at the cutoff is
≈0 (`ν_opt≈0`) — a positive-definite eddy viscosity has no net forward flux to represent and cannot
beat no-model. In 3-D the cascade is forward (`⟨Π⟩>0`, the NR29 sign; `ν_opt>0`), so a structural
closure with the right net dissipation *must* help. NR31 converts P1's main honest a-posteriori
limitation into a derived consequence of dimensionality and a falsifiable 2-D-vs-3-D ordering —
the decisive 3-D test §8.1 defers to.

**Extensions (NR16–NR24).** The later batches slot into the same two structures.
Discarded-operator / memory (1): NR16–NR21 (power-law bursts ⇒ `1/f^γ` spectrum ⇒ long
memory ⇒ eddy diffusivity, with the exact Helmholtz anisotropy identity NR18), NR23 (the
Kramers–Kronig DC sum rule fixing the **signed** net eddy viscosity of NR15), and NR24(iii)
(long memory ⇒ `S(0)=∞` ⇒ no finite down-gradient diffusivity). Criticality / fold (2):
NR22 (the two early-warning precursors are one relaxation rate, with a genuine
slowing-down-vs-noise discriminator) and NR24(ii) (the eddy diffusivity diverges as
`(N−N_c)⁻²` at the fold — the transport face of the variance EWS). NR24 is the explicit
bridge: a memory identity (1) whose fold limit (2) is the transport analog of critical
slowing down. NR25 adds the **fluctuation** side of that MZ certification: in the
exact harmonic-bath instance the certified kernel *is* the random-force
autocorrelation (second FDT), the linear face of NR1 (memory and noise are one
object).

All references remain `[cite]` slots — none invented.
