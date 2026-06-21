# Paper outline — *Two Clocks: operator-structural diagnosis of K-theory and a falsification-driven subglacial sliding law*

> **Status of this document.** Working outline for a hypothesis/methods manuscript.
> The **Abstract** and **§1–§9** are drafted (prose-level); references are still to be
> filled. Every citation is flagged `[cite]` — none
> are invented; the author fills exact references. Claim tags mirror the repo
> discipline used throughout `FUTURE_WORK.md`: `[VERIFIED empirical]`,
> `[VERIFIED qual., this config]`, `[LIT]`, `[HYP]`, `[FALSIFIED]`. Every headline
> number is produced by code in this repo, not asserted.

---

## Abstract *(draft)*

We present an operator-structural framework for turbulent scalar transport and apply
it to subglacial cavity dynamics. The thesis is that **pressure and temperature
cannot be modelled as one coupled scalar**: pressure is a *global, elliptic* field
(a Poisson constraint resolved instantaneously over the whole domain) while
temperature is a *local, parabolic* field (neighbour-to-neighbour diffusion with
memory). Classical single-Prandtl **K-theory** stirs both with one eddy diffusivity
and is therefore structurally unable to represent the distinction; we derive the
exact operator-level repair (Mori–Zwanzig / projected-FDT) and verify it on a
filtered-DNS benchmark `[VERIFIED]`.

Applied to a subglacial cavity, the framework yields several `[HYP]` predictions; we
exercise the two that **open, no-authentication data can settle**. (i) A **Regime
Transition Number** (RTN) predicts where ocean water can intrude grounded ice; on
real BAS Bedmap2 geometry (1.33 M grounded cells) `RTN>1` concentrates at the
grounding line (median distance 6 km vs 221 km) and decays monotonically inland,
robust to the subglacial-water fraction — a `[VERIFIED]` *directional* result (not a
precision/recall score; no gridded intrusion survey exists). (ii) A **non-local
(memory) sliding law** predicts ice-stream surges that lag basal-water forcing by a
thermal-diffusion timescale `τ_ice = H²/κ_ice`; evaluated on 131 catalogued active
subglacial lakes this is ~`10^5` yr — `~8×10^4×` slower than observed surge lags of
0.02–2 yr — and is therefore `[FALSIFIED]` as literally written.

The falsification is *diagnostic*. A thermal skin-depth argument
(`δ_skin = √(κ_ice P/π) ≈ 0.5–5 m` at surge periods) and the *impulse* shape of a
lake drainage (a semi-infinite diffusive response is monotone, with no peak, whereas
observed speed-ups rise to a peak) together **exclude ice thermal diffusion** as the
lag-setting mechanism and relocate the memory into **subglacial hydrology**. We refine
the picture to **two driving potentials** — the Leray pressure constraint (elliptic)
and temperature (parabolic) — plus a **distinct, nonlinear-parabolic hydraulic
subsystem** whose lumped impedance Green's function `K_hydraulic(τ)=(1/RC)e^{−τ/RC}`
is itself a Mori–Zwanzig memory kernel; the ice-thermal term survives only as a weak
`t^{−1/2}` tail. Two gates remain open: a matched lake-drainage→velocity lag test
(blocked on a login-gated drainage-date catalogue) and a *derived* nonlinear hydraulic
kernel (requires a coupled GlaDS/Röthlisberger model). The contribution is the
structural diagnosis, the falsifiable-forecast methodology, and a mechanism
reassignment driven by — not in spite of — a clean falsification on real data.

---

## §1 Introduction *(draft)*

**1.1 The problem.** Eddy-diffusivity (K-theory) closures `[cite]` represent all
turbulent transport with a single scalar diffusivity `K` (optionally a turbulent
Prandtl number `Pr_t`). This conflates two physically distinct fields. We argue the
conflation is not a tuning error but a **structural** one: it cannot be repaired by
re-fitting `K`, because pressure and temperature obey different *operators*.

**1.2 The two clocks.** Temperature obeys an advection–diffusion (parabolic)
equation: local, memory-bearing, torn into filaments by local shear. Pressure in an
incompressible flow obeys a Poisson (elliptic) equation: it is a global constraint,
resolved instantaneously over the whole domain and its boundaries (the Leray
projector enforcing `∇·u = 0`) `[cite]`. A single local diffusivity acting on both is
structurally blind to this elliptic/parabolic split — the "two clocks."

**1.3 Evidence ladder.** We test the thesis at increasing complexity (Parts 1–8):
temporal decoupling in eddy-flux observations `[cite]`; temporal+spatial gradient in
1-min surface data `[cite]`; spatial elliptic/parabolic decoupling in Rayleigh–Bénard
DNS `[cite]`; the K-theory breakdown under stability analysis; hyperbolic→elliptic
crossover and nonlinear compressible demonstrations; and the **projection method as
the two clocks**, reunified by Leray projection `[cite]`.

**1.4 The repair.** A Mori–Zwanzig generalized-Langevin reduction `[cite]` produces a
closure with a *memory kernel* plus a fluctuating force, linked to dissipation by a
projected fluctuation–dissipation relation (projected-FDT). On a filtered-DNS
benchmark this generalizes K-theory and recovers the correct backscatter/solenoidality
structure that a Smagorinsky surrogate misses `[VERIFIED]` (§4/§5 of the repo;
`general_two_clocks/run_closure.py`).

**1.5 Why glaciology.** A subglacial cavity is a natural stress test: a turbulent
water layer between a cold ice ceiling and a rock/till bed, with melt at the
ice–water interface (a Stefan condition `[cite]`) and sliding controlled by basal
water pressure `[cite]`. It couples an elliptic pressure constraint, a parabolic
thermal field, and a separate hydraulic subsystem — exactly the regime where the
two-clocks distinction should bite. We use it to (a) probe candidate melt-enhancement
mechanisms and (b) derive falsifiable field-scale predictions.

**1.6 Contributions.**
1. An operator-structural diagnosis of K-theory's blindness and its MZ/projected-FDT
   repair, verified on a closure benchmark `[VERIFIED]`.
2. A null map of four candidate subglacial melt-enhancement mechanisms, all bounded
   by the conduction-limited wall in the cold-cavity (Type-I) regime `[VERIFIED
   empirical, this config]`.
3. Two field-scale forecasts exercised on open data: the RTN intrusion predictor
   (`[VERIFIED]` directional on Bedmap2) and a non-local sliding law
   (`[FALSIFIED]` as literally written on 131 real lakes).
4. A falsification-driven mechanism reassignment: the surge lag is **hydraulic
   impedance**, not ice thermal diffusion — formalised as two potentials plus a
   nonlinear hydraulic memory kernel `[HYP]`.

---

## §2 The Two-Clocks framework *(draft)*

**2.1 Definitions.** Elliptic pressure operator (Poisson/Leray projector `ℙ`),
parabolic temperature operator (advection–diffusion), and their distinct response
"clocks" (timescales/Green's functions). State the incompressible NS system and the
Leray decomposition `[cite]`.

**2.2 Structural claim and what is verified.** The claim that the two fields cannot
share one diffusivity is *structural*. In this manuscript "verified" means: the
operator distinction is **demonstrated numerically** at each rung of the evidence
ladder, and the MZ/projected-FDT closure **reproduces** the DNS force structure the
single-`K` surrogate cannot. We are explicit that this is `[VERIFIED]` in the
benchmark configuration, not a theorem about all flows.

**2.3 The closure repair.** MZ memory kernel + fluctuating force; projected-FDT tying
backscatter to an effective viscosity through `ℙ`; the Clock-Mismatch Number (CMN) as
the ratio of upstream influence for pressure vs scalars. (Forward-reference §7/§G.5:
the CMN correction `−CMN·∇·(∂_t K_u ∇θ)` vanishes for steady turbulence and is
unit-tested numerically `[VERIFIED identity]`, and is verified operationally in a 2-D
advection + moving-plume proxy — §H.3, `h3_cmn_reduced_model.py`: ~9× transient error cut,
naive ∝τ_c vs corrected ∝τ_c², exact steady null, `+τ_c` the unique corrector.)

---

## §3 Candidate mechanisms — methodology *(draft)*

**3.1 The cavity model.** 2D/3D penalized (Brinkman volume-penalization) DNS/LES of a
turbulent cavity between a cold ice ceiling and a bed, with a passive/active thermal
field and an interfacial melt diagnostic; real BEDMAP1 bed transects as geometry
`[cite]`. State governing equations, penalization, resolution, and the melt-rate /
Nusselt diagnostics.

**3.2 Pre-registration and go/no-go.** Each candidate mechanism is stated as a
*pre-registered* expectation with a falsifiable go/no-go criterion **before** the run,
so that a null is informative rather than a tuning failure. Define the flat-wall
control, the enhancement ratio `R_mean = ⟨melt⟩/⟨melt⟩_flat`, and the
`Nu/Nu_flat` ceiling test.

**3.3 The Type-I regime and the conductive sublayer.** Define the cold-cavity
(Type-I) regime and the hypothesis that mean melt is bounded by the **thermal
conductive sublayer** rather than by momentum stagnation — the criterion the four
candidates are tested against (results in §4–§5).

---

## §4 Results — the null map *(draft)*

Four candidate melt-enhancement mechanisms were pre-registered (§3.2) and run on the
2D penalised cavity. This section covers the **three with no positive mean signal —
Candidates 1, 2 and 4** — which are **null or bounded** in the Type-I (cold-walled,
grounded) regime; Candidate 3 (the scallop), the only candidate with a positive
flow-driven interfacial signal, is treated separately in §5 (where it is also bounded
in the mean, `Nu/Nu_flat < 1`). Each run sweeps the relevant control with three
closures `{none, smagorinsky, backscatter}`; numbers below are the auto-generated
report values (`REPORT_CANDIDATE{1,2,4}.md`). `[VERIFIED empirical, this config]`.

**§4.1 Candidate 1 — intermittent plumes from ice-base roughness.** Pre-registration:
band-limited roughness on the ice base should make interfacial melt *intermittent*
(skew `S>0`, excess kurtosis `K>3`, peak/mean `>2`), humped at intermediate `Ri≈0.3–0.6`.
Result: the thresholds are **not met and the statistics are flat in `Ri`** (relative
spread `<2%`, closure-independent): pooled `peak/mean≈1.8`, `skew≈−0.75`, `kurt≈2.2`.
The variability is **geometric, not plume-driven** — thinner ice columns conduct faster —
and switching the flow off entirely (`f_amp=0`, pure conduction) reproduces `melt_mean`
and `peak/mean` to ~1%. Mean melt is conduction-limited: `melt_mean≈2.17e-4`, constant
to 4 significant figures across all `Ri` and closures. The genuine turbulent signal
lives in the interior flux `F_turb=⟨v'θ'⟩`, not at the interface. `[cite]` roughness /
boundary-layer-plume literature.

**§4.2 Candidate 2 — double-diffusive layering.** Pre-registration: with `Le=100`
(heat diffuses ~100× faster than salt), salt fingers should give a *humped* thermal
Nusselt `Nu_T(R_ρ)` peaking near `R_ρ≈2`. Result: **no hump** — `Nu_T` falls
monotonically from `1.44` (`R_ρ=0.5`) to `0.73` (`R_ρ=10`); `R_ρ=2` is not a maximum.
The differential-diffusion signature does appear elsewhere: `Nu_S ≫ Nu_T` (`Nu_S≈45`
vs `Nu_T≈1.4` near `R_ρ=1`, ~30×), and at strong stabilisation (`R_ρ=10`) both drop
below 1 / go negative (`Nu_S≈−26`) — **counter-gradient transport**, which a positive
eddy diffusivity is structurally unable to represent (the two-clocks theme). The Turner
flux ratio scales as `γ≈1/R_ρ`, confirmed to ~1%. Backscatter resists the collapse
(`Nu_T=0.86` vs `0.73` unclosed at `R_ρ=10`); `none` and Smagorinsky coincide elsewhere.
`[cite]` double-diffusion / salt-finger literature.

**§4.3 Candidate 4 — hydraulic switching.** Pre-registration: under a tidal body force a
wide cavity should switch between *filled* (`H1→1`) and *stratified* (`H1→0.2`) states,
with time-averaged melt peaking at intermediate `Ri` (switching frequency `f_switch`
maximal there). Result: at every `Ri` the cavity is **monostable filled**
(`H1≈0.90±0.01`, `f_switch=0`) — no hump. (An earlier "dead-flow" null was a numerical
under-resolution artefact: `ny=48` let the Brinkman penalty bleed into the ~10-row fluid
gap; resolving the cavity at `ny≥64` restores normal flow, `umax≈0.6`, KE up ~500×, so
the null is physical, not a construction artefact.) `[cite]` subglacial
hydraulic-switching / drainage-state literature.

## §5 Results — the scallop mechanism and the wall ceiling *(draft)*

Candidate 3 is the one candidate with a **positive, flow-driven interfacial signal**,
and it is also where the wall ceiling is established most sharply. Two parts and a
capstone; GPU-confirmed on a Tesla P100 (CuPy 14.0.1).

**§5.1 Part A — the pre-registered runaway is sign-reversed (correct ice physics).**
The roughness-growth hypothesis (thin ice → faster melt → more roughness → more
turbulence → runaway, `Λ>0`) is **falsified with the opposite sign**: the
conduction-pinned base **self-smooths**, `Λ≈−0.265 (<0)`, with melt/ice-height
anticorrelation `corr(m,y_ice)≈−0.95`, closure-independent. This is the *expected*
behaviour for a melting (not eroding) surface — the runaway is a false analogy to
rock/sediment beds. `[VERIFIED empirical, this config]`. (The value was briefly
corrupted to `+0.24` by a spectral-filter bug and restored on the GPU path; provenance
in `glaciers/REPORT_CANDIDATE3.md` Part A, restored by PR #1.)

**§5.2 Part B — the corrected scallop / melting-instability hypothesis passes.** A flat
melting wall is linearly stable, but a finite-amplitude bump larger than the thermal
boundary layer drives **lee-side separation → recirculation → reattachment heat-flux
enhancement** — a stable, self-limiting scallop. The staged go/no-go probe confirms it:
a resolved bump under a mean current gives a coherent, closure-independent enhancement
relative to its own conduction baseline, `R = m_flow/m_cond` with `R_mean ≈ 1.06–1.36`
and lee peaks `R_max ≈ 2–3.3`, **above** the flat-wall control, scaling monotonically
with the mean current (`U0.8→U1.5`). `[VERIFIED qual., this config]`. `[cite]` Curl (1966)
scallops; Dubnick et al. (2020) ripple/melting-instability analogue.

**§5.3 Part C — full `Λ(λ)/a_sat/Nu` sweep (P100).** The wavelength sweep shows an
**interior optimum** at `n_waves=12` (`λ≈2.094`, `λ/dx≈10.7`): a `+12.5%` peak in the
conduction-relative `R_mean` and a peak `Nu/Nu_flat=0.9219`, falling off either side.
The conduction-relative enhancement is tight and reproducible, `R_mean=1.1273±0.0044`.
Crucially, the two ratios that look contradictory are **consistent by construction**:
`R_mean>1` measures bump-melt against its own conduction baseline, while `Nu/Nu_flat<1`
measures the bump against a *flat wall with flow ON*. Flow lifts melt above conduction
on the bump, yet the bump still delivers less mean heat than a flat wall — both true at
once. The exact identity `Nu/Nu_flat = δ_flat·⟨1/δ_T⟩` is definitional; the proposed
"mean-thickening beats convexity" criterion is **[FALSIFIED]** as a clean inequality
(it fails at small `a/λ`), while `Nu/Nu_flat<1` is `[VERIFIED empirical]` at every
amplitude.

**§5.4 Capstone — the basal-heat ceiling is boundary-condition-robust.** Every prior
null used a no-slip bed and a cold-Dirichlet ice wall; two targeted gates show the
ceiling is not an artefact of either (P100, `n=128`, turbulent):
- *Thermal BC (Dirichlet → finite-conductance Robin, `q=h(θ−θ_ice)`):* basal heat
  absorbed is unchanged flow-ON vs flow-OFF to `<0.01%`.
- *Momentum BC (no-slip → Navier slip):* the near-bed tangential speed grows **~100×**
  (`u_tang` 0.027 → 2.68 from locked to free slip), yet `Nu/Nu_flat` stays `0.96–0.97`
  and **never exceeds 1** (it slightly falls); ridge melt holds to within 0.4%, and
  `s=1` reproduces the no-slip trajectory bit-for-bit (including on the GPU CuPy path).

Removing the stagnant momentum layer entirely delivers **no** extra basal heat: the
limit is the **thermal conductive sublayer** set by `κ` and the cold wall, not the
momentum stagnation layer. `[VERIFIED empirical, this config]`. The deliverable is a
**regime map** — Type I (grounded, cold-walled, closed) **no**; Type II (grounding line,
flux BC) **maybe**; Type III (open shelf) **yes** — making explicit that the null is
regime-specific, not a universal claim about ice–water interfaces.

**§5.5 Amplitude dynamics (§G.2).** Under a time-dependent drive the bump amplitude is
**monostable**: a single attractor `a*`, with no overshoot, hysteresis, or resonance.
`[VERIFIED qual., this config]`; the `a*` coefficients and any amplitude memory kernel
remain `[HYP]`. `[cite]` melting/dissolution-instability and ripple-dynamics literature.

**§5.6 The local lee-flux growth law (§G.6) `[MEASURED]`.** The *mean* conductance is
amplitude-flat, but the growth-carrying *local* peak flux `R_max(a/λ)` rises **linearly**
(`R²=0.94`, free exponent 0.69 — not the posited `(a/λ)²`), bounded `1.9–4.3` over
`a/λ∈[0.05,0.30]`, with lee-flow separation onset at `a/λ≈0.11`
(`g6_local_flux_law.py`).

**§5.7 The interface coupling number (§A.1) `[DERIVED]`.** `Λ(ω)=|H(iω)|/(ρ_iL)` measures
when ice participates: `Λ(0)=St≤0.06` (DC), `Λ→0` at high frequency; in the surge band
(0.02–2 yr ≪ ice clock `τ_d`) `Λ<5×10⁻⁵≪St`, so the cold ice wall is a **passive BC to
<1 %**, participating only at millennial forcing (`interface_coupling_number.py`).

## §6 Results — the Regime Transition Number *(draft)*

**§6.1 Definition.** The Regime Transition Number `RTN = (p_ocean − p_atm)/p_w`
predicts where ocean water can intrude grounded ice: overburden `p_i = ρ_i g H`, ocean
head `p_ocean = ρ_w g d_base` (`d_base` = bed depth below sea level), subglacial water
`p_w = φ p_i`. The forecast is **directional**: `RTN>1` should concentrate near
grounding lines and anticorrelate with ice thickness inland.

**§6.2 Test on real Bedmap2 geometry.** Built from BAS Bedmap2 (1 km, decimated to 3 km;
1.33 M grounded cells), the `RTN>1` fraction binned by distance to the grounding line
(`φ=0.9`) decays monotonically from the margin inland:

| dist-to-GL [km] | 0–5 | 5–10 | 10–25 | 25–50 | 50–100 | 100–250 | >250 |
|---|---|---|---|---|---|---|---|
| RTN>1 fraction | 12.2% | 5.2% | 3.2% | 1.7% | 0.6% | 0.0% | 0.0% |

The median distance-to-GL is **6 km** for `RTN>1` cells vs **221 km** for the rest, and
the ordering is robust across `φ ∈ {0.8, 0.9, 0.95}` (overall `RTN>1` fractions
`2.3% / 1.1% / 0.4%` — only the magnitude scales). `[VERIFIED]` directional.

**§6.3 Scope and falsification.** This is **not** a precision/recall score: no gridded
intrusion survey exists, and 1 km cannot resolve the ~1–10 m Röthlisberger channel, so
channel size is implicit in `φ` rather than resolved. The checkable prediction is the
grounding-line *concentration*: uniform inland intrusion would falsify the RTN ordering.
Testable against systematic radar surveys (IceBridge successors, NISAR). `[cite]`
ocean-intrusion observations; Bedmap2 (BAS) `[cite]`.

**§6.4 The intrusion residence number `Ro` `[DERIVED]`.** The RTN=1 surface is a
threshold; the *pace* of front advance is `Ro=v_kin·τ_hyd/ℓ` with regime boundary
`τ_crit=ℓ/v_kin` (`Ro=1` at `τ_hyd=τ_crit`). For runaway-tail cells (`A=0.70 km/m`,
`dH/dt=1.5 m/yr`): `v_kin=1.05 km/yr`, `τ_crit≈1.9 yr`, so ~98 % of the hydraulic
residence band lies below `τ_crit` ⇒ **thinning-paced (`Ro≈1`)**, hydraulic-limited only at
the slow (~2 yr) end. Falsifiable with a DInSAR `v_obs` over the runaway cells
(`intrusion_residence_number.py`). `[cite]` Werder et al. 2013.

## §7 Falsification and correction — the non-local sliding law *(draft)*

**§7.1 The literal kernel, falsified on real thickness.** The §G.4 memory sliding law
predicts ice-stream surges lag basal-water forcing by `τ_ice = H²/κ_ice`. Evaluated on
real Bedmap2 thickness at the **131 Siegfried & Fricker (2018) catalogued active lakes**
(130 with valid thickness; median `H=2282 m`, range 637–3905 m), `τ_ice` has **median
≈ 151,000 yr** (p5–p95: 23,000–323,000 yr) — versus observed post-drainage surge lags of
**0.02–2 yr**, i.e. **~8×10⁴× too slow** at the median lake. The `log₁₀ τ_ice` histogram
is fully disjoint from the observed band: a clean **`[FALSIFIED]`** of the kernel *as
literally written*, on real geometry.

**§7.2 Why the falsification is diagnostic.** Two facts exclude ice thermal diffusion as
the lag-setting mechanism: (i) the thermal skin depth at surge periods is
`δ_skin = √(κ_ice P/π) ≈ 0.5–5 m` (`P=0.02–2 yr`) — the perturbation never penetrates
more than metres of ice, so the full-thickness `H` is the wrong scale; (ii) a drainage
is an *impulse*, whose semi-infinite diffusive response is a **monotone `t^{−1/2}` decay
with no peak**, yet observed speed-ups **rise to a peak**. The lag is therefore
**hydromechanical, not thermal**.

**§7.3 Mechanism reassignment `[HYP]`.** The memory relocates into the **subglacial
hydrology**. We refine the ontology to two driving potentials — the Leray pressure
constraint (elliptic) and temperature (parabolic) — plus a **distinct,
nonlinear-parabolic hydraulic subsystem** (cavity↔channel storage–transport, with `φ`
the hydraulic potential, *not* the Leray pressure). Its lumped impedance Green's function
`K_hydraulic(τ)=(1/RC)e^{−τ/RC}` is itself a Mori–Zwanzig memory kernel; the ice-thermal
`t^{−1/2}` term survives only as a weak subdominant tail. Tuned thermal / water-inertia
kernels are rejected as curve-fitting. The surviving empirical forecast — "surface
velocity rises after lake drainage, with a delay set by cavity/channel geometry and water
flux (a hydraulic residence time), only weakly by ice thickness" — is testable once a
vetted drainage-date catalogue is available (§9). `[cite]` Röthlisberger (1972); Schoof
(2010); GlaDS (Werder et al. 2013); observed surge lags (Stearns et al. 2008; Siegfried
et al. 2016).

**§7.4 The dimensional bridge and roughness closure (§A.3/§A.2) `[DERIVED + bounded]`.**
With `ρ_iL` and literature subglacial inputs the steady R‑channel is metre-scale
(`R*≈2.4 m`, `τ≈0.18 yr`); `ρ_iL` **cancels** in `V_scallop/V_o=0.33` so a scalloped reach
grows channels +33 % in area (calibration-free), the only genuine calibration being the
concentration gain `g∈[0.1,0.9]`. The roughness `z_0=c_z a` feeds the log-law drag
`C_d=[κ/ln(H/z_0)]²`, whose logarithmic dependence buffers a 10× `z_0` uncertainty to
only ~1.7× in `C_d` (`a3_dimensional_bridge.py`, `a2_z0_roughness.py`).

## §8 Discussion *(draft)*

**§8.1 Two pressure fields in the coupled system.** *(drafted)*

> The incompressible Navier–Stokes solver enforces a Leray pressure `p` — a Lagrange
> multiplier that instantaneously satisfies `∇·u = 0`. This is distinct from the
> subglacial hydraulic potential `φ = p_w + ρ_w g z`, which evolves according to
> storage–transport dynamics (Röthlisberger channel opening, GlaDS sheet thickness).
> The two are coupled at the cavity interface but operate on different operators: `p`
> is elliptic, `φ` is parabolic. Conflating them — treating the hydraulic head as if
> it were constrained instantaneously — is the same category error as K-theory's
> pressure–temperature conflation, and it explains why mainstream models miss the
> hydraulic lag.
 
This distinction is the verified, continuum-level core of the framework; it does **not**
depend on any microscopic (molecular/EM) account of pressure, which is deliberately
omitted from the formal argument. Tag: `[VERIFIED structural]` for the operator split;
`[HYP]` for the claim that the conflation is *why* mainstream models miss the lag.
`[cite]` Röthlisberger (1972); GlaDS (Werder et al. 2013); Leray projection.
 
**§8.2 What the falsification buys, and where the closure sits.** *(draft)*
The corrected ontology is **two driving potentials** (elliptic Leray pressure +
parabolic temperature) plus a **distinct hydraulic-impedance subsystem** carrying the
surge memory (§7.3). The §7 falsification is *productive*: it does not discard the
non-local sliding law, it **relocates the memory** from ice thermal diffusion to
hydraulic residence time — turning a wrong number into a sharper, still-falsifiable
mechanism. Operationally this means surge-lag parameterisations should key on
cavity/channel geometry and water flux (effective pressure `N`), not on ice thickness
`H`. Positioned against existing closures, the MZ/projected-FDT repair is the
operator-level generalisation that a single-`K` eddy-diffusivity (and its Smagorinsky
surrogate) cannot reach — it carries a memory kernel + fluctuating force tied by a
projected-FDT, recovering the backscatter/solenoidality structure that RSM / EDMF / LES
closures approximate with explicit transport or mode-splitting. The claim is structural
and benchmark-verified, **not** a blanket "better than mainstream." `[HYP]` for the
hydraulic-memory reassignment; `[cite]` RSM / EDMF / LES closure references.

## §9 Open gates *(draft)*

Three gates remain open, each with a concrete unblock path. `[FUTURE]`.

1. **Matched lake-drainage → velocity lag test.** The response side is already open: the
   ITS_LIVE datacube over lake **Mac1** yields a real surface-speed series of **4505
   image-pair measurements, 1987–2025, median 420 m/yr**, and `sliding_validator.estimate_lag`
   runs on it directly (self-lag ≈ 0 as expected). The only missing piece is the drainage
   *dates* (volume-change time series), which sit behind a USAP-DC login / unreachable
   NSIDC host — so they are not fabricated. Unblock: a vetted drainage-date catalogue.
2. **Derived nonlinear hydraulic kernel.** §7.3 posits the impedance kernel
   `K_hydraulic(τ)=(1/RC)e^{−τ/RC}` phenomenologically; deriving it requires a coupled
   GlaDS/Röthlisberger cavity↔channel model rather than a lumped RC analogue.
3. **RTN precision/recall.** The §6 result is directional only; a quantitative score needs
   a future **gridded intrusion survey** (NISAR / IceBridge successors) to score against.

---

## §10 — A field-measurable sliding law: the `s_N(N)` master curve, three probes, and an ungrounding early-warning *(draft; new)*

The §7 falsification relocated the surge memory into hydrology and effective pressure
`N`. PRs #1–#4 then turned `N` from a tuned coefficient into a **measured** quantity,
and §I (`FUTURE_WORK.md`) closes the loop with a closed-form, invertible law. This is
the constructive capstone of the manuscript's sliding-law arc.

**§10.1 The unification — `s_N(N)` as the common observable.** A lake drainage is an
in-situ effective-pressure *step* whose surge amplitude `du/u = |s_N|·(dN/N)` measures
the sliding-law sensitivity `s_N = d ln u_b/d ln N` (§G.4, `lake_lag_trunk.py`: 3
detections, amplitude rising toward flotation); ocean thermal forcing lowers `N`
*continuously*, so the gating slope `d ln u_*/dTF` measures the *same* `s_N` (§H.1.6,
`efp_gate_direct_n.py`: +0.46→+0.62/°C, interaction `d=−0.035`, direct BedMachine `N`).
`[VERIFIED directional / SUPPORTED, QUALIFIED]`.

**§10.2 The closed-form master curve `[DERIVED, VERIFIED]`.** Solving the
regularized-Coulomb law `τ_b=C N (u/(u+u0))^{1/m}` at `τ_b=τ_d` gives
`|s_N|(N)=m/(1−(N_c/N)^m)`, `N_c=τ_d/C` — `→m` far from flotation, a simple pole
`≈N_c/(N−N_c)` at `N_c`. Verified to `1.4×10⁻⁴` vs the numeric `s_N`
(`sn_master_curve.py`). This *derives* the near-flotation weakening that Joughin,
Smith & Schoof (2019) impose ad hoc (linear `h_af<h_T` ramp + fixed `u0`).

**§10.3 Inversion — measuring `N_c` and `m` `[DERIVED + VERIFIED on synthetic]`.** A
drainage-step population recovers the flotation/Type-III threshold `N_c` to a few
percent (robust to amplitude noise); `m` is degenerate with the per-event drop far
from flotation (`sn_master_curve.py`). Ocean tides give a **third, continuous** probe:
the fundamental velocity admittance equals `|s_N|`, and its second-harmonic ratio
`≈(ε/4)|s_N'/s_N−1|` diverges toward `N_c`, so admittance + harmonics + the known
tidal amplitude recover `m` and the dimensionless flotation proximity `R=(N_c/N)^m`
**from surface velocity alone** (`tidal_admittance_probe.py`) — a sliding-law reading
of the Gudmundsson (2006/2007/2011) nonlinear MSf tidal response.

**§10.4 An ungrounding early-warning `[DERIVED + demonstrated; field test HYP]`.** The
`N_c` fold makes the basal drag stiffness `∂τ_b/∂u∝(1−R)²/R→0` (critical slowing
down), so an ice stream approaching ungrounding should show **rising variance and
lag-1 autocorrelation of its surface speed** — a velocity-based MISI early-warning,
distinct from the Greenland surface-melt CSD of Boers & Rypdal (2021), and made
*operational* by continuous tidal monitoring of `R(t)→1`.

**§10.5 Scope and falsification.** Absolute `s_N` is uncalibrated (per-event `dN/N` is
a lumped-storage lower bound); the robust claims are the sign/shape and the
*threshold* `N_c`. The lag-vs-`N` sign is an open discriminator (n=3 contradicts the
baseline model). Falsifiers: a co-located `dN`+GPS population whose `|s_N(N)|` does not
follow `m/(1−(N_c/N)^m)`; a stream nearing flotation with no variance/AC1 rise; tidal
admittance/harmonics not steepening toward the grounding line. `[cite]` Schoof (2005);
Gagliardini et al. (2007); Joughin, Smith & Schoof (2019); Tulaczyk et al. (2000);
Gudmundsson (2006, 2007, 2011); Minchew et al. (2017); Scheffer et al. (2009);
Dakos et al. (2008); Boers & Rypdal (2021).

**§10.6 The framework on real lakes `[REAL DATA]`.** Read through the §I lens
(`lake_lag_sn_ews.py`), the 3 open-velocity MacAyeal lakes give **0 positive surge
detections** (`|Δv/v|≤2 %`, mixed sign, near the ~1 % noise floor) and **0 critical-
slowing-down precursors** — the master curve places them far from the `N_c` pole, and the
early-warning correctly returns a true negative (unit tests confirm the detectors fire on
planted surges/CSD). No new download; the full 131-lake/ATL15 population stays gated.

---

## §K — Hydraulic memory kernel `[HYP, citable]`

This appendix states the *shape* of the derived-kernel argument that §9 gate #2 leaves
open, without claiming any earned number. It is the honest, citable placeholder for the
mechanism §7.3 currently represents with the lumped `K_hydraulic(τ)=(1/RC)e^{−τ/RC}`
analogy.

> The standard linearized-hydrology treatment (Werder et al. 2013; Hewitt 2013;
> Schoof 2010) couples a distributed sheet (parabolic diffusion) to channelized flow
> (creep closure). The impulse response is the time convolution of the sheet Green's
> function with the channel relaxation kernel. The exact form, coefficients, and
> predicted lag range require numerical computation with regional parameters. Until
> computed, the kernel is `[HYP]` and the lag match to the observed 0.02–2 yr
> surge-lag band is `[NOT EARNED]`.

Accordingly, no closed-form coefficients, timescales, or lag predictions are asserted
here: the `(1/RC)e^{−τ/RC}` of §7.3 is an *analogy*, not a derivation, and earning a
quantitative lag is deferred to the coupled GlaDS/Röthlisberger computation named in
§9 gate #2. `[HYP, citable]`. `[cite]` Röthlisberger (1972); Schoof (2010);
Hewitt (2013); Werder et al. (2013).

---

## Reproduce

```bash
python glaciers/validation/external/run_rtn_bedmap2.py  --stride 3 --phi 0.9   # §6 / §H.1 RTN
python glaciers/validation/external/run_sliding_real.py --with-velocity Mac1   # §7 / §H.2 sliding
 pytest                                                                # math harnesses
```

Provenance and numbers: `glaciers/validation/REAL_DATA_RESULTS.md`; forecasts and the corrected
§G.4: `FUTURE_WORK.md §G.4`, `§H`.

---

## Reference slots (author to fill — do NOT invent)

- `[cite]` K-theory / eddy-diffusivity closure (e.g. Boussinesq; Monin–Obukhov).
- `[cite]` Leray projection / Helmholtz–Hodge decomposition.
- `[cite]` Mori–Zwanzig formalism; fluctuation–dissipation.
- `[cite]` Rayleigh–Bénard DNS reference; NEON / ASOS / NCEP reanalysis datasets.
- `[cite]` Brinkman volume penalization.
- `[cite]` BEDMAP1 / Bedmap2 (BAS); Siegfried & Fricker (2018) lake inventory; ITS_LIVE.
- `[cite]` Stefan moving-boundary melt; Curl (1966) scallops; melting/dissolution instabilities.
- `[cite]` Röthlisberger (1972) channels; Schoof (2010) sliding/effective pressure; GlaDS (Werder et al. 2013); subglacial-lake drainage & surge-lag observations (Stearns et al. 2008; Siegfried et al. 2016).
