# Future Work — Scale Coupling and the Next Phase

This document scopes the research questions opened — but not closed — by the
results in `glaciers/subglacial/THEORY_CAVITY.md` §1–§14, `REPORT_CANDIDATE{1–4}.md`, and
`glaciers/REPORT_CANDIDATE3.md` (Part C). It is **theory-building and targeted
experiments**, not further parameter sweeps on the existing code.

### How to read this document — claim status tags

Every non-trivial statement below carries one of three tags so the reader knows
what is established vs. proposed:

- **[VERIFIED]** — measured in this repo's runs (CPU and/or Tesla P100), with the
  numbers in §13.
- **[LIT]** — a standard closure or result from the glaciology / fluids
  literature, cited and used as a parameterisation.
- **[HYP]** — a hypothesis *this* work proposes for future testing. The
  derivations are genuine, but they are predictions, not results.

Two caveats recur and are flagged inline:
- **Caveat S (sign):** channelisation feedback runs through flow *concentration
  and speed-up*, not flow slowing (the naïve "more drag → slower → more melt"
  loop is self-limiting).
- **Caveat D (data):** the *amplitude* dependence is now **measured** (§G.1
  area-partition and the §G.6 closure test, `figures/59`): the mean-Nu deficit is
  **amplitude-independent** (`Nu/Nu_flat ≈ 0.90` across `a/λ ∈ [0.05, 0.30]`,
  power `p≈0`), so the §G.6 `(1 + ζ(a/λ)²)` roll-off is **falsified** — `Nu<1` is
  near-geometric, not an amplitude effect. What stays uncalibrated is the
  *wavelength*/field amplitude of the roughness length `z_0(λ)` against a real
  scallop train (one field point), which leans on a Nikuradse-type literature
  closure.

---

## §A — The scale-coupling framework: from local scallop to global channel

### A.1 The scale hierarchy

The original "two clocks" (§1–§8) separated two **operator classes**: elliptic
pressure and parabolic temperature. The candidate tests refine this. The verified
picture (see THEORY_CAVITY §14) is **two operator classes operating in two media**
(water and ice), with the resolved inertial flow as an emergent structure:

| Scale | Operator | Timescale | Status |
|---|---|---|---|
| Pressure (water) | Elliptic (Poisson) | τ_p ~ L/c → 0 | [VERIFIED] §1–§7 |
| Temperature (water) | Parabolic (heat) | τ_θ ~ L²/κ_w ~ 10⁴–10⁵ s | [VERIFIED] §1–§7 |
| Resolved flow | Hyperbolic (inertial) | τ_flow ~ λ/u* ~ 10¹–10³ s | [VERIFIED] §13 |
| Temperature (ice) | Parabolic (heat), slow | τ_ice ~ H_ice²/κ_ice ~ 10⁶–10⁸ s | [VERIFIED] §13.2 |

**[HYP]** The interface is then not a boundary condition but a *coupling surface*
where these operators interact. The original framework treated ice as a passive
BC; the ice-side conduction result (§13.2) shows ice is a participating medium
with its own (slow) thermal clock.

### A.2 The scallop → roughness → hydraulic law

**[VERIFIED]** Scallop wavelength is fluid-selected; optimal `n_waves=12`,
`Re_L = Uλ/ν ≈ 2000–2200`, `R_max ≈ 2.5` local, `Nu/Nu_flat ≈ 0.92` mean
(§13.1).

**[LIT]** Curl's criterion: `Re* = u*λ/ν ≈ 2200` (Curl 1966; Blumberg & Curl
1974). Our `Re_L` band is consistent with this.

**[HYP, leans on LIT — Caveat D]** A scallop *field* sets an effective roughness
length parameterised as

> `z_0 ≈ a · f(λ/δ_T)`,  with `δ_T ~ κ_w/u*`,

where `a` is the (ice-memory-stabilised) amplitude and `λ` the fluid-selected
wavelength. The **amplitude** direction of `f` is now measured (§G.6, `figures/59`):
the mean-Nu deficit is amplitude-independent over `a/λ ∈ [0.05, 0.30]`, so a real
scallop train's mean wall flux is set by its *wavelength*/separation geometry,
not by how steep the bumps are. The remaining unknown is the **wavelength**
(field) amplitude of `z_0`, fit from one `(λ)` point; it must lean on a
Nikuradse-type sand-grain closure (`k_s ≈` scallop amplitude; Nikuradse 1933)
until calibrated against a real scallop train.

**[LIT]** The roughness enters the log-law drag coefficient
`C_d = [κ / ln(H/z_0)]²` (κ ≈ 0.41), which sets the hydraulic-potential gradient
in a Röthlisberger/GlaDS channel (Röthlisberger 1972; Werder et al. 2013):

> `dφ/ds = −ρ_w g S_f`,  `φ = p_w + ρ_w g z_b`.

**[HYP]** The closed loop: framework's structural fix (divergence-free
backscatter, §7.1) protects the large-scale pressure field → sets `dφ/ds` → drives
`Q` → sets `u*` → sets `λ` (Curl) → with `a` sets `z_0` → feeds back to `C_d`. This
operates at the **resolved-field + parameterised-boundary** scale, not the
subgrid scale.

### A.3 Channelisation — corrected feedback sign (Caveat S)

**[LIT]** Röthlisberger channel evolution (Röthlisberger 1972; Nye 1976; Spring &
Hutter 1982):

> `∂S/∂t = V_o − V_c`,
> `V_o = (1/ρ_i L) |Q ∂φ/∂s|`  (melt opening from dissipation),
> `V_c = 2 A S (N/n)^n`  (creep closure, `N = p_i − p_w`).

The larger channel has lower water pressure and drains its neighbours → the
"main artery" instability.

**[HYP]** Scallop-coupled extension adds a distributed wall source from
lee-localised heat-flux enhancement:

> `∂S/∂t = V_o − V_c + V_scallop`,
> `V_scallop = (1/ρ_i L) ∫_λ [q_reattach(x) − q_flat] dx`.

**Caveat S made explicit:** the feedback is *flow-concentration*-driven. Incipient
low captures flow → `u_channel > u_sheet` → higher `u*` → more melt → deeper
channel → more concentration. The *wrong* loop ("more drag → slower flow → more
melt") fails because turbulent melt scales with `u*`: slower flow delivers *less*
heat (self-limiting). Real channelisation (Röthlisberger; Dow et al. 2018) is the
concentration loop.

**Test [DONE → §D.1].** This test was run. `glaciers/scallop_channel_feedback.py` adds
`V_scallop` to the R-channel ODE; `glaciers/REPORT_CHANNEL.md` (titled "§A.3 / §D.1",
verified on the P100, `glaciers/tests/test_channel_feedback.py`) shows localised
reattachment melt **does** create preferred nucleation sites — opening-source
sign `V_scallop/V_o = +0.33`, phase-locking `R_phase = 0.95`, deterministic site
selection `R_winner = 1.00` (scallop) vs `0.571` (noise). See §D.1 for the full
result; only the dimensional `ρ_i L` bridge stays [HYP].

---

## §B — Ice-side conduction formalism

### B.1 Two-phase Stefan condition

**[VERIFIED]** We replaced the water-only update `v = q_water/(ρ_i L)` with the
two-phase balance (Stefan 1891; Nye 1953; Crank 1984):

> `ρ_i L v = q_water − q_ice`,  `q_ice = −κ_ice ∂T_ice/∂n`.

The ice field obeys `∂_t T_ice = κ_ice ∇²T_ice` with `T_ice = T_melt` at the
interface. The P100 three-branch run (§13.2) shows this stabilises amplitude
(0.41 vs. 0.21 water-only).

### B.2 The boundary memory kernel [DERIVED closed form + normalisation; amplitude is a problem input]

**[DERIVED]** Solving the ice heat equation with a moving boundary gives an interface
velocity with memory:

> `v(t) = (1/ρ_i L) [ q_water(t) − ∫₀ᵗ K_ice(t−τ) q_water(τ) dτ ]`.

The kernel `K_ice` is no longer schematic. Linearising the ice heat equation in the
interface-attached frame about a steady ablation speed `V̄` (perturbation
`θ' = T − T_melt` with `θ'(0)=0` at the pinned interface, `θ'(∞)=0`),

> `∂_t θ' = κ ∂_ξξ θ' + V̄ ∂_ξ θ' + v'(t)·θ̄'(ξ)`,  `θ̄'(ξ) = θ_far (V̄/κ) e^{−V̄ξ/κ}`,

and Laplace-transforming in time gives the **interface-flux transfer function** in
closed form (decaying root only):

> `H(s) = q_ice'(s)/v'(s) = A·[1 − √(1 + 4 τ_d s)]/s`,  `A = k_th θ_far V̄²/(2κ²)`,  `τ_d = κ/V̄²`,

whose inverse Laplace transform is the **memory kernel**

> `G(t) = A·[ erfc(√(t/4τ_d)) − 2√(τ_d/πt)·e^{−t/4τ_d} ]`.

Four properties follow rigorously (all confirmed in `glaciers/validation/synthetic/ice_kernel_synthetic.py`,
which matches `G`'s step response to a direct PDE solve to `0.2%`):

1. **Power-law short-time tail `∼ τ^{−1/2}`** — the singular term dominates as
   `τ→0`, `G ≈ |A|·2√(τ_d/π)·τ^{−1/2}` (measured log–log slope `−0.515`). This is
   the classic `√t` diffusive response, as anticipated.
2. **Exponential cutoff at the *diffusion time*, decaying at LONG lag** — the cutoff
   is `e^{−τ/4τ_d}` with `τ_d = κ/V̄²` (or `H_ice²/κ` when the ice is thin enough to
   feel its far boundary). This **corrects** the earlier schematic
   `exp(−H_ice²/4κτ)`: that argument is *inverted* — it would suppress *short* lags,
   the opposite of a causal memory tail that is strong at short lag and decays at
   long lag.
3. **Dimensional closure (the deliverable).** `[A] = W·m⁻³` and the bracket is
   dimensionless, so `[G] = W·m⁻³`; the resolvent kernel `K_ice = G/(ρ_i L)` that
   appears above has units `s⁻¹` — a genuine **rate**, resolving the old note that
   `(κ/τ)^{1/2}` was a velocity, not a rate.
4. **Normalisation cross-check.** The DC gain `H(0) = −ρc·θ_far = ∫₀^∞ G dτ` equals
   the quasi-steady sensitivity `d q̄_ice/dV̄` computed independently from the steady
   profile (verified to `3×10⁻⁵`), so the prefactor `A` is correct, not fitted.

The Stefan balance closes the loop implicitly: `ρ_i L v' = q_water' − (G * v')`, a
Volterra equation whose resolvent gives exactly the `K_ice` above (to leading order
`K_ice ≈ G/ρ_i L`). **What is *not* derived:** the per-cavity amplitude, which scales
with the local `θ_far` (far-field ice undercooling) and `V̄` (mean ablation speed) —
these are problem inputs, not universal constants. So the kernel's *shape, units, and
normalisation are now derived*; only the site values of `θ_far, V̄` remain empirical.

**Contrast with the subgrid kernel [VERIFIED null + HYP unification]:** the
subgrid memory→phase-lag was null (Directions A/C) — its kernel is effectively
`K_SGS(τ) ∼ δ(τ) + exp(−τ/τ_mem)` (instantaneous FDT + fast OU). The ice kernel is
diffusive with a power-law tail. A unified multi-scale memory formalism (see §D.4)
would carry both with scale-selective kernels.

---

## §C — The regime equation: retired and replaced

**[VERIFIED retired]** The original `R(D,M) = 1 + cβ(1−1/D)⁺ + γaRi(1−Ri/Ri_c)⁺ +
α(1−1/M)⁺` had a stratification-hump term `γaRi` that **no test verified** —
retired.

**[VERIFIED + HYP replacement]** The honest form separates structural, resolved,
and boundary-memory contributions that do **not** sum to a universal multiplier:

> `R_total = R_struct(D,M) + R_resolved(Re_L, a/λ)·𝟙_{a>a_crit} + R_memory(St_ice)`

| Term | Meaning | Status |
|---|---|---|
| `R_struct = 1 + cβ(1−1/D)⁺ + α(1−1/M)⁺` | structural baseline, ≈1.0–1.5× | [VERIFIED] §8 |
| `R_resolved = (q_reattach/q_flat)·(A_reattach/A_total)` | local redistribution, `Nu/Nu_flat<1` mean | [VERIFIED] §13.1 |
| `R_memory = f(St_ice = κ_ice τ/H_ice²)` | amplitude stabilisation, *not* enhancement | [VERIFIED qualitatively] §13.2 |

The framework's value is **discriminating mechanisms by scale**, not predicting a
single melt factor.

---

## §D — New questions opened by the falsifications

### D.1 Scallop field → channel network [VERIFIED directional / structural; HYP absolute magnitude]
Does parameterised scallop roughness modify Röthlisberger/GlaDS channel stability?
**[LIT]** GlaDS sheet conductivity `k_s ∝ (h_s−h_b)^{3}` (laminar) or
`∝(h_s−h_b)^{5/4}` (turbulent; Hill et al. 2024); the scallop field modifies the
effective bed `h_b`. **Original prediction:** channel nucleation probability is
higher where reattachment zones are dense (Caveat S concentration loop).

**Done — the wrapper exists and earned a positive structural result.**
`glaciers/scallop_channel_feedback.py` + `glaciers/REPORT_CHANNEL.md` (titled "§A.3 / §D.1") wire
the DNS-measured reattachment-flux field into a Röthlisberger/GlaDS
channel-network ODE (`dS/dt = V_o − V_c + V_scallop`). On a Tesla-P100
(`nx=ny=128, n_waves=12, a/λ=0.30`) it earns three dimensionless,
coefficient-robust claims (raw: `glaciers/figures/49_scallop_channel_feedback.json`):
- **opening-source sign:** `V_scallop/V_o = +0.33` — reattachment is a *positive*
  (opening) source, same sign as `V_o` (Caveat S: more melt, not less);
- **phase-locking:** `R_phase = 0.95` (φ_pref ≈ 4.55 rad) — the source is pinned
  to the bedform (twelve sharp peaks, one per wavelength), not white in `x`;
- **deterministic site selection:** `R_winner = 1.00` (scallop) vs `0.571`
  (noise control) — the phase-locked source picks the channel-nucleation site
  on every seed, overriding the random site a noise-seeded flat wall would pick.

**Closure-robust to `z_0` (`glaciers/REPORT_CHANNEL_Z0.md`, `glaciers/scallop_channel_z0_robustness.py`).**
The DNS uses no `z_0`; the roughness closure enters only the reduced channel ODE,
through the concentration gain `g`, the creep rate `k_creep`, and (via `ρ_i L`)
the source strength. Reusing the committed DNS source and sweeping that closure
box (`g ∈ [0.1,0.9]`, `k_creep` over 16×, source strength over a `0.001×–2×`
band — 405 points) leaves the verdict invariant: at the measured strength
`R_winner(scallop) = 1.000` across the whole `g × k_creep` box, always beating
the noise control by margin `≥ 0.33`, with the winning site pinned within
`0.166 rad` of `φ_pref`. The site selection is moreover *magnitude-independent*
— it stays locked at every strength down to `0.001×` (3 decades), because the
concentration-feedback loop amplifies the source's *sign and phase*, not its
size. So the directional claim does not ride on the unverified closure (the
repo's directional pattern, cf. RTN > 1 on Bedmap2).

**Residual [HYP].** Only the *dimensional* bridge stays open — turning the
normalised channel sizes into physical radii via `ρ_i L` and the calibrated
value of the concentration gain `g`. That is a *magnitude*, not a sign, and needs
external calibration data, not a code wrapper; the directional mechanism above is
verified and closure-robust.

### D.2 Scallop → double-diffusion [VERIFIED structural: heterogeneity + phase-lock + regime coexistence]
The DD null (Candidate 2) was for a **smooth** wall. A scalloped wall traps
warm/salty water in the lee (`R_ρ = α_S ΔS/α_T ΔT > 1`, finger-favourable) and
thins cold/fresh meltwater on the stoss face (`R_ρ < 1`, diffusive-convective) —
**both regimes in one geometry**. **Original prediction:** spatially heterogeneous
`Nu_T`, `Nu_S`; any mean `Nu_T` hump would be *geometry-induced regime mixing*,
not a subgrid effect.

**Done — the wrapper exists and confirmed the structural prediction.** The
Candidate-2 double-diffusion solver gained `wall_amp`/`wall_nwaves` knobs
(`ocean/scallop_doublediff.py`, `glaciers/subglacial/candidate2_doublediff.py`); on a Tesla-P100
(`nx=256, ny=96, n_waves=12, a/λ=0.30, R_ρ=1.5, Le=100`) the scalloped wall is
compared to the smooth wall at identical physics (`ocean/REPORT_DOUBLEDIFF.md`;
diagnostics in `ocean/tests/test_doublediff_scallop.py`):
- **heterogeneity:** phase-binned `Nu_T` peak-to-trough (turbulence averaged out)
  rises smooth `5.31` → scallop `67.48` — the wall imposes a structured `Nu_T(x)`;
- **phase-locking:** wall-coherent variance fraction `η²` rises smooth `0.093` →
  scallop `0.891`; the coherent response splits into a fundamental (lee/stoss,
  `0.437`) and a dominant 2nd-harmonic (symmetric crest+trough constriction,
  `0.506`);
- **regime coexistence:** the local Turner ratio `γ(x)` straddles the smooth
  value (`+0.620`) — enhanced fraction `0.93`, suppressed `0.07` — so
  finger-favourable and diffusive-convective regimes coexist in one geometry,
  confirming the mixing is **geometry-induced**, not a subgrid effect.

So the structural claim is earned. (As in §G.1 the *mean* `Nu_T`/`Nu_S` is
distribution-dominated, not a clean enhancement multiplier; the result is *where*
the regimes sit, not a universal hump.)

### D.3 Scallop → hydraulic switching [FALSIFIED]
The switching null (Candidate 4) was a single-mode body force on a **smooth**
cavity. **[LIT]** Composite-Froude control (Armi 1986; Pratt 1986): roughness adds
`ΔG²_rough ∼ C_d(z_0/H)·(L_rough/L_cavity)`. **Original prediction:** a developed
roughness field can push `G²` across the subcritical/supercritical boundary,
enabling the bistability the smooth test missed.

**Falsified.** The hypothesis requires *two* stable states at *one* forcing —
the logical negation of the verified Candidate-4 result (`glaciers/REPORT_CANDIDATE4.md`),
which finds the cavity **monostable filled at every `Ri`**: `H1 ≈ 0.90 ± 0.01`,
`f_switch = 0`. There is no bistable filled⇄stratified regime and therefore no
intermediate-`Ri` melt hump. Adding a rough wall cannot manufacture a second
stable state where the governing dynamics admit only one in the explored drive
envelope, so §D.3 as posed (hysteresis in `Q` from a parameterised rough wall) is
**falsified**, not merely unobserved. (Realising bistability *at all* would need
a regime the body-force drive never reached — stronger/slower tidal pumping or a
genuinely two-layer initial state — which is a different hypothesis from §D.3.)

### D.4 Unified memory formalism [DERIVED structure: additivity + scale-selectivity + Markov limit; coefficients MEASURED/closed-form — RESULT 12]
The subgrid (Mori–Zwanzig) memory and the resolved ice-thermal memory (§B.2) **can**
be folded into one generalised Langevin equation
`dx/dt = −∫₀ᵗ K(t−τ)x(τ)dτ + F(t)`, `⟨F(t)F(t')⟩ = k_BT_eff·K(t−t')` (second FDT),
with **scale-selective kernels**. The three structural claims close; the
coefficients are now pinned (RESULT 12, `general_two_clocks/REPORT_GLE_COEFFICIENTS.md`): the fast bath
time `τ_c` is **measured**, and the slow bath `B, τ_d` are **closed-form** from §B.2.
Validated in `glaciers/validation/synthetic/gle_memory_synthetic.py`.

**(1) Additivity of independent baths [DERIVED].** The MZ memory kernel is the
autocorrelation of the *projected* (orthogonal) force, `K(t)=⟨(QLf)(t)(QLf)(0)⟩`.
If the eliminated degrees of freedom split into two statistically independent
sub-baths — fast subgrid turbulence and slow ice-thermal diffusion — their
orthogonal forces are uncorrelated, so the cross-term vanishes and **both** the
kernel and (by the second FDT) the noise covariance are additive:
`K = K_SGS + K_ice`. (Validator: autocorrelation of a sum of two independent
coloured noises equals the sum of autocorrelations to ~1.5%, cross-correlation ≈0.01.)

**(2) Scale selectivity → two relaxation timescales [DERIVED].** Take
`K_SGS(τ)=(1/τ_c)e^{−τ/τ_c}` (fast OU) and `K_ice(τ)=B·τ^{−1/2}e^{−τ/τ_d}` (the
§B.2 power-law tail, cutoff `τ_d≫τ_c`). The combined kernel has a fast exponential
head and a heavy power-law tail with a crossover lag `τ*` obeying `τ_c<τ*<τ_d`.
Solving the GLE relaxation (`x(0)=1`, `F=0`) shows the **fast kernel sets the early
relaxation** (the combined response and the SGS-only response agree at short lag)
while the **slow kernel adds a long-time tail** absent from the fast-only response
(they differ by ~10¹⁴ at `t≈30`). So the response carries two well-separated
relaxation timescales, each from its own kernel — exactly the §D.4 prediction.
(The heavy-tail kernel makes the relaxation *sub-diffusive*, i.e. non-monotone /
anti-persistent, not a simple exponential — consistent with a fractional GLE.)

**(3) Markovian white-noise limit recovers the existing closure [DERIVED].**
`∫₀^∞ K_SGS dτ = 1` independent of `τ_c`, while the peak `K_SGS(0)=1/τ_c→∞` as
`τ_c→0`: the OU kernel approaches `δ(τ)`. So in the white-noise limit the subgrid
memory collapses to the **instantaneous eddy-diffusivity** closure the model
already uses, recovering the *local* FDT as a special case — the unified GLE is a
strict generalisation, not a competing model.

**Coefficients [MEASURED / closed-form — RESULT 12].** Pinned in
`general_two_clocks/gle_coefficients.py` → `general_two_clocks/figures/53_gle_coefficients.json` (no new data/GPU):
- **Fast bath `τ_c` [MEASURED].** The OU time is the decorrelation time of the SGS
  eddy diffusivity `K_u`, measured at fixed interior probe points as
  `τ_c ≈ 0.02–0.03` (solver units; sign **+**), consistent with the committed
  RESULT-8 SGS-force memory (`τ_mem^set=0.05`, `τ_mem^eff≈9.5×10⁻³`).
- **Slow bath `B, τ_d` [DERIVED closed form, site-input dependent].** From §B.2,
  `τ_d = κ/V̄²` and `B = |A|·2√(τ_d/π)/(ρ_iL)`; for representative subglacial inputs
  `τ_d ≈ 0.3–34 yr` while `τ_c` is seconds-scale, so the fast/slow baths are
  separated by **~10⁶–10⁹ in physical time** — the scale-selectivity with real
  numbers. Only the site values `θ_far, V̄` remain empirical.
- **Bath weights `K_SGS : K_ice = 1 : St` [DERIVED].** A bath's second-FDT weight is
  its DC gain `∫K dτ`. The fast bath is unit-normalized (`∫K_SGS dτ = 1`, point (3)),
  so the slow ice bath's relative weight is the *dimensionless* §B.2 DC gain
  `∫K_ice dτ = −ρc θ_far/(ρ_iL) = c_i |θ_far|/L = St` — the **Stefan number**
  (`≈ 0.013` at `θ_far=2 K` to `0.063` at `10 K`, `∝ θ_far`, `V̄`-independent). The
  slow ice bath is therefore Stefan-suppressed: it adds only the weak long-time tail
  of point (2), never the early relaxation. This is the *same* `St` as the §G.4
  thermal-tail weight (`W_thermal/W_hydraulic`, hydraulic kernel unit-gain) — one
  ice kernel, one number, two derivations (`general_two_clocks/gle_coefficients.py`,
  `general_two_clocks/REPORT_GLE_COEFFICIENTS.md` §D; `general_two_clocks/tests/test_gle_coefficients.py` 7/7).

**`τ_c` is a *turbulence* clock, not a per-scalar clock [VERIFIED — RESULT 13].**
A single fast-bath time `τ_c` can serve *all* buoyancy-active scalars because the
turbulent transport memory is set by the velocity field, not the scalar. Tested in
the double-diffusion solver (`general_two_clocks/scalar_clock_universality.py`,
`general_two_clocks/REPORT_SCALAR_CLOCK.md`): heat and salt — stirred by the same velocity but with a
100× molecular-diffusivity contrast (`Le=100`) — have turbulent-flux memory times
that agree to **0.3%** (`τ_c(salt)/τ_c(heat) = 0.997 ± 0.001`, flux cross-correlation
1.000), while the 100× contrast appears only in the *transport efficiency*
(`Nu_S ≈ 100·Nu_T`). So `K_φ = K_u/Sc_t^(φ)` rescales amplitude, not memory — the
unified GLE legitimately carries one `τ_c` for every passively-stirred scalar.

Pure theory + light CPU; no new data.

### D.5 Critical amplitude band [VERIFIED empirical: a_crit; NULL: shedding edge]
The original sweep used a single amplitude (`a/λ=0.1`). The hypothesis was that two
transitions bound a "useful" band: a lower `a_crit` below which the bump is buried in
the penalty zone (no separation), and an upper amplitude above which steady
recirculation gives way to vortex shedding. **Test:** an amplitude scan at the fixed
optimal wavelength (`n_waves=12`, `λ=4·2π/12`), measuring two physically grounded
diagnostics over the post-spin-up window. Harness: `glaciers/scallop_amplitude_band.py`
(`nx=ny=128`, `U_drive=1.5`, `spinup=3000`, `measure=800`,
`a/λ ∈ {0.01,…,0.60}`); diagnostics validated in
`glaciers/validation/synthetic/` via the unit tests in
`glaciers/tests/test_validation_synthetic.py`. Numbers below were reproduced on an independent
array backend to FFT round-off, so they are numerics, not an artefact of one machine.

**Separation onset `a_crit ≈ 0.02` [VERIFIED empirical].** With a steady forcing
(`f_amp=0.4`) the time-mean **reverse-flow area fraction** (fluid cells with mean
`u<0`, i.e. a standing lee recirculation) rises *smoothly and monotonically* with
amplitude — `0.003, 0.006, 0.012, 0.022, …, 0.113` at `a/λ = 0.01…0.60`. It first
exceeds a `0.5 %`-area threshold at `a/λ = 0.02`, so we report `a_crit ≈ 0.02`. The
transition is **soft, not sharp**: a faint recirculation (`<0.5 %`) already exists at
the smallest amplitude, and the fraction grows continuously rather than switching on.
`a_crit` is therefore the amplitude at which a *coherent* lee cavity appears, not a
bifurcation. (The simple scaling estimate `a_crit ~ √(ν η_pen)` in the original note
remains an order-of-magnitude rationale only — the measured onset is what is earned.)

**The Type-I thermal-wall bound holds across the entire band [VERIFIED empirical].**
`Nu_bump/Nu_flat` stays strictly `< 1` at *every* amplitude (range `≈ 0.82–0.94`,
no monotone trend). Increasing roughness amplitude never lifts normal melt above the
flat-wall value, consistent with the §D.6 / Type-I analysis: separation alone does not
make the rough wall out-transfer a flat one in this conduction-limited regime.

**Upper (shedding) edge: not testable in this solver class → [NULL], now with a
mechanism (`glaciers/REPORT_SHEDDING.md`, `glaciers/scallop_shedding_deterministic.py`).** A first
quasi-laminar scan (small drive `f_amp=0.02`) found the **detrended** kinetic-energy
coefficient of variation essentially *flat* across the whole band (`≈ 3.7–4.0×10⁻³`),
never rising above the small-amplitude floor plus a 3× margin (`1.19×10⁻²`): no
shedding transition up to `a/λ=0.60`. That left an honest caveat — the drive is gated
behind `f_amp>0`, so a small stochastic force is always present, making the absence
*suggestive, not conclusive*. A follow-up **deterministic-limit** probe closes that gap
without a new solver: it drives the cavity at successively smaller `f_amp`
(`10⁻²→3×10⁻⁴`) and checks base-flow stationarity *first* via
`ke_drift = |slope·N|/mean_KE`. The result is unambiguous — `ke_drift ≈ 0.76–0.78` at
**every** amplitude and `f_amp` (the KE changes ~76 % across the window; a steady base
would need `<0.1`), and the residual fluctuation is `f_amp`-**independent** with its
power piled at the lowest mode (`spec_conc ≈ 0.89`, period = full window) — i.e. it is
leftover spin-up drift, not a forced response (∝`f_amp`) or an intrinsic narrow-band
limit cycle. **Mechanism:** a *constant* `U_drive` body force in a periodic spectral box
has no momentum sink (no inflow/outflow, no drag wall), so the driven current never
equilibrates. A vortex-shedding (Hopf) bifurcation is only well posed *about* a steady
base flow, so `a_shed` is **not testable** here — the null is a property of the
configuration, not a missed detection. A clean `a_shed` needs a constant-mass-flux or
inflow/outflow DNS (a different solver class). The lower edge and band-wide `Nu<Nu_flat`
result remain solid. Moderate compute (single-`λ` amplitude × `f_amp` scan).

### D.6 Creep is a [NULL] for cavity heat-transfer enhancement — rigid wall justified
Glen's-law creep neither enhances the roughness nor rescues the Type-I thermal-wall
bound `Nu_TypeI/Nu_flat < 1`. The argument is a displacement–timescale comparison,
not a velocity ratio (the latter, used in an earlier draft, conflated the solver's
seconds-to-hours clock with geological time and produced a misleadingly soft margin).
Validator: `glaciers/validation/synthetic/creep_scaling_synthetic.py`.

**The operative null: creep displacement ≪ roughness over the solver clock [NULL].**
Glen's law gives strain-rate `ε̇ = A σ^n` (`n=3`). The creep wall displacement over a
run of duration `T`, **as a fraction of the roughness amplitude**, is the dimensionless
strain

> `f(N,T) = A N^n T`   (`A` [Pa⁻ⁿ s⁻¹], `N` [Pa], `T` [s]).

At realistic open-cavity effective pressure `N=0.1–1 MPa` over an hour, `f` runs from
`~1×10⁻⁶` (cold ice, `0.1 MPa`) to `~9×10⁻³` (temperate, `1 MPa`) — i.e. **< 1 % of the
scallop height** — and it scales *linearly* in `T`, so the solver's actual (sub-hour)
measurement window is proportionally smaller. The cavity boundary therefore moves
almost entirely by **phase change (the Stefan condition)**, not ice deformation, which
is exactly the assumption the **rigid-wall Brinkman penalization** encodes. Even at an
*unrealistically* high `N=5 MPa` the hour-long displacement is only `~11 %` for cold ice
(`~4 %` for the colder end of the Glen range); the `N=5 MPa`–*temperate* corner, where
`f` would reach `O(1)`, is doubly unphysical for an open cavity (it can sustain neither
that effective pressure nor temperate ice at it). So on the solver clock the ice is
rigid — underwritten by the ~16-order-of-magnitude ice/water viscosity ratio.

**Sign, if creep acted at all [DERIVED].** Over long times creep can only *smooth*: a
corrugated interface under overburden concentrates deviatoric stress at the crests, so
creep drives crests down faster than troughs and a corrugation relaxes monotonically
(no mechanism amplifies it under uniform load). Modelled as the amplitude sink
`ȧ|_creep = −A σ^n a`, the amplitude decays monotonically (validator confirms). So
creep is **never** an enhancement and cannot reverse the melt-set amplitude; at worst
it is a slow same-sign smoothing. The clincher is independent and [LIT]: morphologically
identical scallops form on **non-creeping limestone** (Curl 1966), so creep is not
required to set the amplitude at all.

**Residual [computational, Group B] — now run [VERIFIED, quantified].** The full coupled
Stefan+creep amplitude sweep is no longer deferred: `glaciers/validation/synthetic/creep_stefan_coupled.py`
(report `glaciers/REPORT_CREEP_STEFAN.md`, 7 tests) couples the *already-measured* RESULT 14
melt smoothing rate (`β_melt`, `τ_melt≈3.03 yr` at the Curl anchor — no DNS re-run) to the
Glen-creep sink and integrates the coupled amplitude ODE over the multi-year melt timescale.
The creep↔melt crossover stress is `σ_crit=(β_melt/A)^{1/3}≈0.16 MPa` (temperate) / `0.35 MPa`
(cold), whereas the physical corrugation relief stress `σ_dev=ρ_i g a≈35–283 Pa` (over
`a/λ∈[0.05,0.40]`) is ~2300× below it, so `ρ=r_creep/β_melt≤5×10⁻⁹` and the coupled steady
amplitude `a*_coupled/a*_melt=1/(1+ρ)` departs from 1 by `<10⁻⁸`: the long-time smoothing
correction is **negligible and same-sign** (never enhancement), consistent with limestone
scallops (Curl 1966). Honest caveat surfaced: the §D.6 displacement bound's worst-case stress
`σ=N` is solver-clock-only — extrapolated to years it would give `ρ>1` at temperate/high-`N`,
which is why the bound is (correctly) restricted to the run window.

### D.7 3-D scallop geometry [NULL — closed]
In 3-D the mean current may select streamwise ridges (combed scallops) vs. 2-D
transverse waves (Dubnick et al. 2020). **Original test:** anisotropic 3-D bed.

**Done — closed null.** The 3-D penalised LES solver `glaciers/subglacial/flow3d.py` plus
`glaciers/scallop3d_probe.py` build 3-D beds (streamwise `ridge`, `eggcarton`, and a flat
control of identical mean gap) and feed them through the same solver, reading the
mean basal heat flux delivered to the (flat) ice base — exactly the
feed-inputs-to/from-the-current-solver wrapper. The gate `Nu/Nu_flat > 1` asks
whether channelling between streamwise ridges beats a flat wall. On the P100 it
does **not**: streamwise-ridge channelisation gives `Nu/Nu_flat ≈ 0.96–0.97` —
the flow reorganises (low-drag troughs along `x`) but delivers ~0 % mean basal
heat gain (THEORY_CAVITY §13, §14.4). It was stress-tested both ways: a free-slip
bed (`bed_slip → 0`, near-bed tangential speed ×100: `0.027 → 2.68`) and a
finite-conductance Robin ice wall both leave `Nu/Nu_flat` at 0.96–0.97 and never
above 1 (§14.4 table). So the third dimension genuinely cannot beat the
conduction-limited wall here — not for lack of a solver, but because the hard
limit is the **thermal** conductive sublayer (set by `κ` and the cold-wall
Dirichlet condition), not the **momentum** stagnation layer. The honest scope of
§14.4 applies (Type-I grounded, cold-walled, closed cavity; `Pr ≈ 1`;
penalised-LES `melt_flux` proxy, not a Stefan phase change).

---

## §E — Implementation roadmap

| Phase | What | Type | Rough cost | Status of inputs |
|---|---|---|---|---|
| E1 | Parameterised scallop roughness → 1-D R-channel/GlaDS | theory + existing data + LIT closure | weeks | **done directionally** (§A.3/§D.1); `z_0` magnitude still open (Caveat D) |
| E2 | Two-scalar (T,S) + scalloped wall, 2-D DNS | new compute | weeks | **done** (§D.2, `ocean/REPORT_DOUBLEDIFF.md`) |
| E3 | Rough wall + anisotropic cavity, switching | new compute (3-D) | months | **resolved** — switching [FALSIFIED] (§D.3), 3-D anisotropic [NULL] (§D.7) |
| E4 | Unified multi-scale memory GLE | pure theory | weeks | needs §B.2 derivation |

Phases E1 and D.4 are theory-building that uses existing data + literature
closures. E2 is now done (§D.2) and E3 is resolved (§D.3 falsified, §D.7 null);
the only E1 residual is the `z_0` magnitude closure. E4 remains pure theory.

---

## §F — Narrative arc (for the paper)

> We began with a structural critique — K-theory collapses pressure and
> temperature into one diffusivity, which is wrong because they are different
> operators — and verified it. We then asked what emergent melt mechanisms the
> corrected closure enables. Four candidates showed that **subgrid-driven**
> mechanisms are killed by the conduction-limited wall, while **resolved-flow
> separation** (the scallop) produces genuine local heat-flux enhancement with a
> fluid-selected wavelength and an amplitude stabilised by ice thermal memory.
> This forces a refinement: the "two clocks" are **two operator classes in two
> media**, with a resolved inertial scale emergent from their coupling. The
> framework's value is not a universal melt multiplier but **identifying which
> mechanisms operate at which scales, and where the boundaries lie** — plus the
> structural closure fixes that keep the large-scale fields driving those
> mechanisms uncorrupted.
>
> Finally we stress-tested the one assumption the nulls all shared — the
> boundary conditions themselves. Relaxing the cold-Dirichlet ice wall to a
> finite-conductance Robin wall, and the no-slip bed to a freely-sliding Navier
> bed, **both leave the wall-limited result intact** (THEORY_CAVITY §14.4):
> driving the near-bed tangential speed ~100× changes mean basal heat by < 0.5 %.
> The ceiling is the **thermal conductive sublayer**, not the momentum stagnation
> layer or any single BC pin. The deliverable for the grounded cold-wall regime
> (Type I) is thus a **regime map** — enhancement is bounded here and would
> require a different boundary-condition class (grounding line, ice-shelf cavity)
> to operate — which is why operational subglacial models carry no
> flow-enhanced-melt term.

> **Update — the regime map's missing Type III now has a model
> (`glaciers/validation/synthetic/type_iii_regime.py`).** The map has three basal
> regimes vs effective pressure `N`: **Type I** (grounded cold wall, high `N`, no
> surge — the result above), **Type II** (near-flotation, low `N`, the discrete
> §G.4 cavity-paced surge, now field-detected at `Thw_142`/`Thw_170`/`Rutford_1`,
> `lake_lag_trunk.py`), and **Type III** (fully floating, `N → 0`, *continuous*
> response). Type III was flagged as the one regime with no validated model. It
> follows from two mainstream ingredients with no new free parameters: **(1)** a
> regularized-Coulomb sliding law `τ_b = C N (u/(u+u₀))^{1/m}` (Schoof 2005;
> Joughin 2019) makes the `N`-sensitivity `s_N = ∂ln u_b/∂ln N` diverge as
> `C N → τ_d`, so **below `N_c = τ_d/C ≈ 0.060 MPa` there is no grounded steady
> solution** — the bed cannot support the driving stress and accelerates to
> flotation (Type III); **(2)** the cavity store↔cavity coupling collapses
> (`J21 ∝ N^{1.5} → 0`, cavity fill `h_s/h_r → 1` saturates) as `N → 0`, so the
> discrete-lag mechanism **turns off** and the impulse kernel collapses from the
> Type II **peaked** form (interior peak ≈ 0.80 yr, inside the 0.02–2 yr band) to a
> Type III **monotone single-relaxation** that simply follows the forcing. This
> makes the transition falsifiable: the amplitude law `du/u = |s_N| f` grows toward
> flotation and diverges at `N_c` (the §H.1.6 steepening as a closed law), and the
> three HYP1 field detections overlay on the predicted curve with the largest surge
> (Rutford, 21.7 %) at the lowest `rel`. **Honest scope:** this is an *analytic*
> regime model (RC law + the 2-compartment cavity Jacobian; no GPU); the field
> overlay's `rel → N` placement is approximate, and the measured coupling exponent
> is 1.5 (vs the `n−1 = 2` nominal). Figure/JSON:
> `validation/reports/type_iii_regime.{png,json}`.

> **Synthesis — the three field results measure the *same* quantity: the basal
> sliding-law `N`-sensitivity `s_N(N) = d ln u_b/d ln N`, which models currently tune
> away (`glaciers/validation/synthetic/efp_probe_theory.py`).** This is the new
> physical relationship the repo's three results imply, made falsifiable:
> - **PROBE-1 (a fast `N`-step, HYP1).** A lake drainage is an *in-situ step
>   experiment*: it drops `N` locally and the downstream surge `du/u` is the sliding
>   response, so `s_N = (du/u)/(dN/N)`. The measured amplitude `du/u` **grows toward
>   flotation** (`log du/u` vs `log N` slope ~ -0.23; and -3.5 per unit `rel` in HYP1).
> - **PROBE-2 (a slow `N`-drift, HYP2).** Ocean thermal forcing TF lowers `N`
>   continuously; the gating slope `d ln u_*/dTF` **steepens toward flotation**
>   (0.46 -> 0.62 /degC; interaction `d = -0.035`, CI excludes 0).
> - **The closed form (HYP3).** The regularized-Coulomb law predicts the shape:
>   `|s_N| -> m` far from flotation, diverging at `N_c = tau_d/C`.
>
> **Why this is new.** The modeling community **subsumes `N` into tuned friction
> coefficients** ("no reliable knowledge of basal water pressure", Joughin et al.
> 2019) and applies ad hoc near-flotation weakening; nobody measures `s_N(N)`
> directly. Pairing a *direct* `N` (HYP2, BedMachine ocean-connected) with a drainage
> step (HYP1) turns `s_N` -- otherwise only inverted/tuned -- into a **field-measured**
> quantity, with the ocean gating as an independent cross-check of the same `s_N(N)`.
> **Honest scope:** the per-event *absolute* `s_N` is uncalibrated -- `dN/N` is a
> lumped-storage estimate that overestimates the true step, so these `s_N` are lower
> bounds ~1-2 decades below the RCF curve (we do **not** claim a magnitude match). The
> robust, falsifiable claim is the **sign**: both directly-measured field responses
> grow toward flotation, the sign the RCF `s_N(N)` predicts; a calibrated `s_N` needs
> co-located GPS + a hydrology-constrained `dN`. Figure/JSON:
> `validation/reports/efp_probe_theory.{png,json}`.

---

## §G — Proposed structural relationships (the "derived laws" audit)

This section records a set of proposed structural relationships and the honest
result of auditing them against this repo's runs. **None of §G.1–§G.6 is a
theorem.** The only [VERIFIED] facts are the empirical `Nu/Nu_flat < 1` (§13.1)
and the BC-robustness (§14.4); everything else is [HYP]/[LIT]. Two of the
proposals were checked numerically here and the result *demoted* the claim — that
is reported as a finding, not hidden.

### G.1 Type-I thermal-wall result — empirical, **not** a Jensen bound [VERIFIED empirical + mechanism VERIFIED (exact area-partition); (1+CV²) decomposition FALSIFIED]

**Claim audited (rejected as a theorem).** A proposed proof used Jensen's
inequality on the local conductance `1/δ_T` to argue a *geometric* bound
`Nu_rough ≤ Nu_flat`. This is wrong twice over: (i) `f(x)=1/x` is convex, so
Jensen gives `⟨1/δ_T⟩ ≥ 1/⟨δ_T⟩` — the convexity term is an *enhancement*, it
would push `Nu>1`; and (ii) pure conduction against a corrugated cold isotherm
*raises* through-flux per projected area (added wetted area, gradient crowding),
so geometry alone does not lower the mean flux. There is **no** general bound.

**What is true.** Writing the local interfacial flux as `q(x) = κ ΔT / δ_T(x)`
(δ_T = local thermal-sublayer thickness, ΔT the bulk→melt drop, common to bump
and flat), the cavity-mean Nusselt ratio decomposes *exactly* as

> `Nu/Nu_flat = ⟨q_bump⟩/⟨q_flat⟩ = δ_flat · ⟨1/δ_T⟩`   (exact, definitional)
> ` = (δ_flat/⟨δ_T⟩) · (1 + CV_δ² + …)`   (2nd-order Taylor; CV_δ² = Var δ_T/⟨δ_T⟩²)

with a *thickening* factor `δ_flat/⟨δ_T⟩ ≤ 1` and a *convexity* factor
`1 + CV_δ² ≥ 1`. The tidy story "Nu<1 ⟺ mean-thickening beats the variance boost"
holds **only if `CV_δ ≪ 1`**.

**[VERIFIED — measurement, this PR] The `(1+CV²)` decomposition is FALSIFIED at
scallop amplitudes.** `glaciers/scallop_sublayer_probe.py` measures `δ_T(x) = κΔT/m_n(x)`
from the true local-normal interfacial flux (the κΔT constant cancels in every
dimensionless quantity) at the optimal `n_waves=12` (`λ=2.094`), Part-C config
(`nx=ny=128, U_drive=1.5, f_amp=0.4, spinup=3000, measure=800`; matches the
documented Part-C control `Nu_flat=2.34e-4`, `umax=2.61`):

| a/λ | Nu/Nu_flat (exact) | CV_δ | 1+CV² | ⟨δ_T⟩/δ_flat | pred = (δ_flat/⟨δ_T⟩)(1+CV²) | mean-thicken > convexity? | Nu<1? |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.937 | **1.61** | 3.58 | 2.19 | 1.64 | **No** | Yes |
| 0.20 | 0.897 | 0.96 | 1.92 | 1.75 | 1.10 | **No** | Yes |
| 0.50 | 0.930 | 0.57 | 1.33 | 2.31 | 0.58 | Yes | Yes |

`CV_δ ≈ 0.6–1.6` is **O(1), not ≪1**, so the two-moment truncation is invalid: its
prediction misses the exact ratio by 0.2–0.7 and is not even monotone, and the
proposed criterion `⟨δ_T⟩/δ_flat > 1+CV²` is *False* at a/λ=0.10 and 0.20 even
though `Nu<1` holds at all three. (A secondary effect: 4–6 of 128 lee columns have
reversed flux `m_n≤0`, i.e. no physical sublayer; they are excluded from the δ_T
statistics but kept in `⟨m_n⟩`.)

> **§G.1 statement of record.** The Type-I result is **not** a theorem from
> Jensen's inequality. `Nu/Nu_flat < 1` is **[VERIFIED] empirically** at every
> amplitude; the *exact* identity `Nu/Nu_flat = δ_flat·⟨1/δ_T⟩` is definitional.
> The suppression is **distribution-dominated, not moment-dominated**: stagnant
> lee cores give a large-δ_T majority that dominates the harmonic mean, while thin
> reattachment patches put a fat tail in `1/δ_T` that no `(1+CV²)` truncation
> captures. This is a sharper physical statement (flow-separation structure of the
> thermal boundary layer), not a weaker one. The mechanism is now **[VERIFIED]**
> by the exact area-partition below (no truncation), so it is no longer [HYP].

**[VERIFIED — exact area-partition mechanism, this PR] `Nu<1` is set by the
separated/thickened area, not by a variance boost.** `glaciers/scallop_g1_populations.py`
uses the fact that `Nu/Nu_flat = ⟨m_n,bump⟩/m_flat` is *literally an area mean of
the local flux* and partitions the interface into three populations relative to
the flat-wall conductance `m_flat = ⟨m_n,flat⟩` — **reattachment** (`m_n ≥ m_flat`,
thin sublayer, enhances), **thickened** (`0 < m_n < m_flat`, thick sublayer,
suppresses) and **reversed** (`m_n ≤ 0`, no physical sublayer). Because the
populations tile the interface, their area-weighted flux shares sum to the ratio
*exactly* — no moments, no truncation:

> `Nu/Nu_flat = C_reatt + C_thick + C_rev`,   `C_p = (1/N) Σ_p m_n / m_flat`
> `Nu/Nu_flat − 1 = e_reatt + e_thick + e_rev`,   `e_p = C_p − f_p` (`Σ f_p = 1`)

Both identities hold to machine precision (`|C_sum − Nu/Nu_flat| ≲ 2·10⁻¹⁶`).
Writing `surplus = e_reatt > 0` and `deficit = −(e_thick + e_rev) > 0`, the
mechanism is one line: **`Nu < 1` ⟺ `deficit > surplus`.** Part-C config (same as
above), full amplitude sweep:

| a/λ | Nu/Nu_flat | f_reatt | f_thick | f_rev | surplus | deficit | def>surp | (δ_flat/⟨δ_T⟩)(1+CV²) |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.895 | 0.477 | 0.523 | 0.000 | 0.112 | 0.217 | **Yes** | 0.99 |
| 0.10 | 0.937 | 0.531 | 0.469 | 0.000 | 0.178 | 0.241 | **Yes** | 1.64 |
| 0.15 | 0.888 | 0.406 | 0.555 | 0.039 | 0.180 | 0.292 | **Yes** | 2.37 |
| 0.20 | 0.897 | 0.281 | 0.672 | 0.047 | 0.211 | 0.315 | **Yes** | 1.10 |
| 0.30 | 0.893 | 0.234 | 0.734 | 0.031 | 0.306 | 0.413 | **Yes** | 2.09 |
| 0.50 | 0.930 | 0.195 | 0.773 | 0.031 | 0.418 | 0.488 | **Yes** | 0.58 |

**Mechanism (earned).** At *every* amplitude the reattachment surplus is
outweighed by the thickened+reversed deficit, so `Nu<1`. The `deficit` grows
strictly monotonically with `a/λ` (`0.22→0.49`) and stays above `surplus`
throughout; beyond the separation onset (`a/λ ≳ 0.10`) the suppressed area
`f_thick` climbs (`0.47→0.77`) while the enhanced area `f_reatt` falls
(`0.53→0.20`) — i.e. the cavity progressively fills with stagnant, thick-sublayer
lee fluid. The thin reattachment patches *do* carry a fat tail in `1/δ_T` (the top
conductance decile holds `0.17→0.48` of `⟨1/δ_T⟩` and the conductance skewness
rises `~0→2.2` with amplitude), but they are an area minority that never lifts the
area mean above the flat wall. The final column shows the FALSIFICATION
quantitatively: with `CV_δ ~ O(1)` the `(1+CV²)` truncation swings `0.58–2.37` and
*predicts `Nu>1`* at four of six amplitudes, versus the true `0.89–0.94`. The
mechanism is reproduced under an independent solver run — the partition exactness,
`deficit>surplus`, the monotone `deficit(a)`, and the `(1+CV²)` failure are all
invariant; only the scalar `Nu/Nu_flat` carries the few-% spread already noted for
this forced config.

**Local vs. mean — the two observables are distinct (resolves an apparent
tension).** `R_mean ≈ 1.06–1.36 > 1` (`glaciers/scallop_probe.py` / `glaciers/REPORT_CANDIDATE3.md`)
and `Nu/Nu_flat < 1` (`glaciers/scallop_sweep.py`) are **not** contradictory because they
have different baselines, not because one is "bump-local":
- `R_mean = ⟨m_flow/m_cond⟩` is a *mean of per-column ratios* over the full
  interface, with each column normalised by **its own flow-OFF conduction on the
  same bumpy geometry** — it measures flow-vs-no-flow enhancement on the bump.
- `Nu/Nu_flat = ⟨m_n,bump⟩/⟨m_n,flat⟩` is a *ratio of means* normalised by the
  **flat wall (flow ON)** — it measures bump-vs-flat mean basal heat delivery.

So `R_mean>1` ("flow lifts melt above conduction on the bump") and `Nu/Nu_flat<1`
("the bump delivers less mean heat than a flat wall") coexist by construction; the
growth-driving flux in §G.2/§G.6 is the **local face flux**, not the spatial mean.

### G.2 Scallop-creep amplitude law — **monostable**, creep negligible [VERIFIED monostable + DERIVED structure; magnitudes DERIVED × LIT (RESULT 14); field validation open]

**Claim audited (algebra corrected).** A proposed law
`ρL ȧ = α a^{1/2} − β a − γ a³` (Röthlisberger melt-opening, conduction loss, Glen
creep `n=3`) was said to be *bistable* with a grow→collapse→decay threshold. It is
not. Substituting `y = a^{1/2} > 0`, fixed points solve
`h(y) = α − β y − γ y⁵ = 0` with `h'(y) = −β − 5γ y⁴ < 0`: `h` is strictly
decreasing from `h(0)=α>0`, so there is **exactly one** positive root → **one
interior fixed point** `a*`, with `a=0` unstable and `a*` stable. There is no
second `a**` and no collapse threshold; a first-order autonomous scalar ODE is
monotone in time and **cannot** produce a grow→collapse→decay transient — that
requires time-dependent forcing `α(t)` (discharge events) or feedback through
`N(t)`. At scallop scale the Glen term is negligible (§D.6: over the solver clock
creep deforms the wall by ≪1 % of the roughness amplitude, so it is a [NULL] — and
in any case same-sign smoothing, not required for scallop regulation per the
non-creeping-limestone analogy), leaving the honest two-term law

> `ρL ȧ ≈ α a^{1/2} − β a`,  single stable amplitude  `a* ≈ (α/β)²`.

`α` is the **local** face-flux enhancement coefficient (§G.1 local-vs-mean), [HYP]
in this *pre-RESULT-14* reading from one (a, λ) point (Caveat D, §A.2/§D.5) — but
see the **Update (RESULT 14)** below, which finds **no** autonomous-growth `α` at
all (the driven response is smoothing-only), so this `α` is superseded.

**Direct test against the moving-boundary solver (`glaciers/scallop_forcing_probe.py`).**
The corrected monostable law makes four falsifiable *qualitative* predictions for
the full solver's seeded-mode amplitude `a(t)` when `U_drive` is made
time-dependent (the only new code mutates `s.U_drive` each step — **no solver
change**; `ProbeFlow._forcing` reads it fresh). At the Part-C config
(`nx=ny=128, n_waves=12, a₀/λ=0.20, spinup=3000`, 400 boundary updates; a 3-seed
turbulent noise floor of 1σ ≈ 0.011 in `a`):

- **`U_drive` is the control knob; conduction alone is inert.** Constant runs:
  `U=0` leaves the mode essentially frozen (0.419→0.398, ~5%), `U=1.5`→0.186
  (−56%), `U=3.0`→0.150 (−64%). The seeded mode **monotonically decays** under
  steady flow (no interior `a*` reached in 400 updates; here `a*≲0.15`) — *more*
  drive ⇒ *smaller* `a*`, matching §C.2 of `glaciers/REPORT_CANDIDATE3.md`.
- **Monotone, no overshoot [VERIFIED].** A mid-run step up (1.5→3.0) and step
  down (3.0→1.5) both relax monotonically onto the new decay branch;
  overshoot-dip ≈ 4×10⁻⁴ ≪ 3σ.
- **No hysteresis / no memory [VERIFIED].** After a down-step to `U=1.5`, the
  decay *rate* matches the constant-`U=1.5` rate at the same amplitude (residual
  ≈ the rate itself, within noise) ⇒ `ȧ` is a function of `(a, U)` only — the
  defining property of a first-order autonomous law.
- **No resonance [VERIFIED].** A sinusoidal drive (`U=2.25±0.75`, period 133
  updates < relaxation ~211 updates) gives *bounded* tracking inside the
  `U∈[1.5,3.0]` envelope; the fast wobble (σ≈0.0095) is **below** the
  steady-drive turbulent floor (σ≈0.010–0.011) — no amplification, no
  sub-harmonic.

So the monostable picture is **[VERIFIED] (qualitative, this config)**: the
amplitude is slaved to the *instantaneous* drive with no memory, overshoot or
resonance, and a grow→collapse→decay transient therefore **cannot** be
autonomous — it requires time-dependent forcing `α(t)`, exactly the corrected
claim. **Update (RESULT 14).** The frozen-interface harmonic decomposition has
since superseded this two-term reading: there is **no** autonomous-growth
coefficient `α` (the in-phase excess is *smoothing* at every driven `(a,U)` →
`Re(s)<0`), so the saturation value `a*=(α/β)²` is **moot** — it presumed an
`α>0` that does not exist. The smoothing coefficient `β` is **solver-MEASURED**
(`~K`-independent, `β/a~K^{+0.13}`), and the dimensional magnitudes are no longer
free: they are **[DERIVED dimensionless × LIT constants]** via the St-free Stefan
bridge and the Curl anchor (§G.2 dimensional bridge). What stays open is *field*
validation (the constant-free `I_obs`). The **amplitude** generalisation of the
swept window (`a₀/λ=0.20`; Caveat D, §A.2/§D.5) is now **[VERIFIED — RESULT 22]**:
re-running the harmonic decomposition across `a₀/λ ∈ [0.05, 0.40]`
(`amplitude_generalization_scan`, `figures/56`) reproduces the two structural
verdicts at **every** amplitude — the driven in-phase flow excess is *smoothing*
(`Re(s)<0`, no `+α a^{1/2}` growth; its magnitude grows monotonically with
amplitude, `−1.3×10⁻⁵ → −1.0×10⁻⁴`), and the conduction `β/a~K^p` stays far below
the `K²` curvature ansatz (and Mullins–Sekerka `|k|`) at all amplitudes
(`p ∈ [−0.27, +0.66]`). Strict `K`-independence (`|p|<0.6`) and an amplitude-flat
`β/a` hold in the signal-rich regime (`a₀/λ ≳ 0.2`) but degrade at shallow
amplitude, where the single-wavenumber conduction in-phase signal is at the noise
floor (`β/a` even sign-flips at `a₀/λ=0.05`) — a measurement-precision limit, not
an emergent physical `K`-dependence. The **drive-window** half of Caveat D is closed
in turn by **[VERIFIED — RESULT 23]** (`drive_window_scan`, `figures/56`): pushing
the mean drive to `U=6` (well beyond the swept `U∈[1.5,3.0]`) at the `a₀/λ=0.20,
n_w=12` operating point, the driven in-phase flow excess stays *smoothing* (`<0`) at
every drive — strong-drive lee separation opens **no** autonomous `+α a^{1/2}`
growth channel (a genuine falsification risk, since a recirculating lee eddy could
in principle add in-phase flux that reinforces the corrugation) — and the quadrature
migration keeps its **sub-kinematic** friction-velocity `∝U^{0.5–0.8}` scaling
(`U^{+0.48}`, never accelerating to kinematic `U¹`) while vanishing to noise at `U=0`
(parity control). The migration is *not* monotone: at the strongest drive (`U≳4.5`)
it **saturates/rolls over** as the lee structure reaches a limiting form — which
reinforces the sub-kinematic reading rather than contradicting it.
With both axes generalised, **Caveat D is retired**; only the wavenumber set is
unchanged, and it is already swept `n_w∈{6…20}` in the RESULT 14 base run.

**Symbolic structure of `α, β` [DERIVED forms + fixed-point/stability algebra;
magnitudes now DERIVED × LIT].** The *form, sign, units, and fixed-point/stability
algebra* close without data (validator
`glaciers/validation/synthetic/amplitude_law_synthetic.py`). Per RESULT 14 the
autonomous-growth `α` is absent (smoothing-limited), and the dimensional
magnitudes — previously `[HYP]` — are now fixed by `k_th, ρ_iL, ΔT, λ` through the
Stefan bridge, i.e. **[DERIVED dimensionless × LIT constants]**; only `I_obs`
(field) remains untested.

- **`ρL ȧ` is a latent-heat flux** `[W m⁻²]`. Hence the two terms must carry
  `[α] = W m^{−5/2}` and `[β] = W m⁻³`. The conduction-smoothing coefficient has
  the **closed curvature form** `β = c_β·k_th·ΔT·(2π/λ)²` `[W m⁻³]` (a corrugation
  `a cos(kx)` has crest curvature `∝ a k²`, so conduction melts crests faster than
  troughs → linear `−βa` smoothing, `β > 0`). The melt-opening coefficient `α`
  inherits the Röthlisberger dissipation scale (`V_o = |Q ∂φ/∂s|/ρ_iL`); its
  **`a^{1/2}` (concave, saturating) exponent is the modelling ansatz**, not derived
  from Navier–Stokes — the flow-concentration enhancement saturates as the trough
  fills with separated flow.
- **Fixed point & stability [DERIVED].** `a* = (α/β)²` is the unique positive root;
  `a=0` is unstable (concave growth dominates near zero) and `a*` is stable with
  linear eigenvalue `d(ȧ)/da|_{a*} = −β/(2ρL) < 0` — so **`β` doubles as the
  amplitude relaxation-rate coefficient** (validator matches the measured rate to
  ~2%). Scaling: `a* ∝ (α/β)²`, and with `β ∝ λ⁻²` this predicts `a* ∝ λ⁴` at
  fixed `α`. Notably `ΔT` cancels if `α` is also `∝ ΔT`.
- **Flow-dependence lives in `β`, not `α` [DERIVED constraint / falsification].**
  The naïve "melt-opening grows with drive" closure `α ∝ u_*` would give
  `a* ∝ u_*²` — *increasing* with drive. But the forcing probe above [VERIFIED]
  the **opposite** (more drive → smaller `a*`). So the net flow-dependence cannot
  reside in the growth coefficient `α`; it must enter the smoothing coefficient `β`
  (stronger turbulent drive → more uniform basal melt → larger `β` → smaller `a*`).
  This **falsifies the simplest `α ∝ u_*` closure** and is exactly why `α`'s
  magnitude is left `[HYP]` pending a regional flux closure.

**RESULT 14 — wall-flux harmonic decomposition corrects the two-term form
itself [VERIFIED falsification; solver units].** The decay-watching above only
ever measures the *net* rate `ȧ = (α a^{1/2} − β a)/ρL`, which conflates the two
coefficients (a two-term fit to the moving boundary returns unphysical
both-negative coefficients). To earn `α, β` *separately* we freeze the interface
`y = ȳ + a sin(Kx)`, time-average the per-column melt flux `m(x) = −κ ∂θ/∂y`
(no boundary motion ⇒ no `_smooth121` contamination), and project onto the
corrugation harmonics: the **in-phase** part `2⟨e sin Kx⟩` changes the amplitude
(smoothing if it opposes the shape, growth if it reinforces), the **quadrature**
part `2⟨e cos Kx⟩` is pattern **migration**. (`glaciers/scallop_amplitude_harmonics.py`;
`glaciers/figures/56_scallop_amplitude_harmonics.json`; test
`test_scallop_amplitude_harmonics.py`.) Three findings overturn the assumed form:

- **The `β ∝ λ⁻²` curvature ansatz is FALSIFIED.** Pure-conduction (flow off)
  in-phase smoothing per unit amplitude is essentially **wavelength-independent**
  across `K = 1.5 … 5.0`: `β/a ~ K^{+0.13}` — not the `K²` (=+2) curvature form,
  nor even Mullins–Sekerka `|k|` (=+1). The near-wall flux tracks interface
  *displacement* with a `K`-independent gain, so the closed form
  `β = c_β k_th ΔT (2π/λ)²` does **not** describe this system.

| `K` | `λ` | conduction `β/a` |
|---|---|---|
| 1.5 | 4.19 | 4.5e−5 |
| 2.0 | 3.14 | 8.4e−5 |
| 3.0 | 2.09 | 6.3e−5 |
| 4.0 | 1.57 | 7.0e−5 |
| 5.0 | 1.26 | 6.1e−5 |

- **There is NO autonomous `+α a^{1/2}` growth term.** The in-phase flow-excess
  flux is **negative at every driven `(a, U)`** — flow only ever adds *more*
  smoothing, never reinforcement. So the two-term law with an interior stable
  `a* = (α/β)²` does **not** describe this solver: the scallop mode is
  **smoothing-limited / decay-only**, which mechanistically explains the forcing
  probe's monotone decay. A finite scallop here *requires* an external
  Röthlisberger melt-opening, which is absent from the solver — so `α`-as-growth
  is not merely unmeasured, it is **structurally absent**.

- **The genuine flow channel is MIGRATION, not growth.** Flow induces a
  *quadrature* component (the Curl-1966 reattachment/lee signature) that is ≈0 at
  `U=0` and grows sub-linearly with drive (`migration ~ U^{0.5–0.8}`,
  amplitude-roughly-independent). The in-phase smoothing also **grows with `U`**
  (more drive → smaller `a*`), confirming the "flow-dependence in `β`" sign above.

> **Corrected reduced model.** The scallop mode is a **damped, downstream-migrating
> mode**, not a growth–saturation balance: `s(K, U) = −β(K,U) + i ω_mig(U)` with
> `Re(s) < 0` always (smoothing, `β` ≈ `K`-independent and flow-enhanced) and
> `Im(s) ∝ U^{0.5–0.8}` (the Curl migration). What is *earned in solver units* is
> `β(U,a)` and the migration rate `ω_mig(U)`; the "growth coefficient `α`" is
> reinterpreted as a **migration** coefficient. Absolute physical magnitudes still
> need the dimensional bridge (known ice constants `k_th, ρ_iL, ΔT, u_*`), but the
> *functional form* is now solver-determined rather than `[HYP]`.

**The dimensional bridge — `β` and `ω_mig` in SI [DERIVED, via the physical Stefan
condition].** The solver advances the interface with `y_ice += dt_eff·m/St` using
an **artificially small** `St = 2×10⁻⁴` — a *numerical accelerator* (its docstring
notes "large ⇒ slow melting"). The *physical* latent-to-sensible ratio
`St_phys = ρ_iL/(ρ_w c_p ΔT)` is `O(10²–10³)` for `ΔT ~ 0.1 K`, so the solver melts
`~10⁶×` faster (relative to thermal diffusion) than reality. **We therefore discard
the solver's `St` entirely** and convert the *measured dimensionless wall-flux
response* to an interface velocity with the exact Stefan balance `ρL v = q`, with
`q = −k_th (ΔT/L₀) ∂θ̃/∂ỹ` and the length scale `L₀ = λ_phys/λ_nd` set by anchoring
the corrugation wavelength (`to_physical()` in `glaciers/scallop_amplitude_harmonics.py`).
`κ_nd` cancels analytically. This yields two **St-free** laws, both `∝ k_th ΔT/ρL`:

```
amplitude e-folding rate   r     = (k_th ΔT / (ρL L₀²)) · (β/a)/κ_nd       [1/s]   (~ λ² ΔT⁻¹ for τ=1/r)
downstream migration speed c_mig = (k_th ΔT / (ρL κ_nd L₀)) · (−E_cos)/(a K) [m/s]  (~ λ⁻¹ ΔT)
```

With water-side `k_th = 0.56 W m⁻¹K⁻¹`, `ρ_iL = 3.0×10⁸ J m⁻³` and the measured
coefficients (conduction `β/a` at `n_w=8`; migration `E_cos` at the strongest
drive `U=3`, `n_w=12`):

| `λ_phys` | `ΔT` | amplitude e-fold `τ` | migration `c_mig` |
|---|---|---|---|
| 2 cm | 0.1 K | 24 days | 7.0 cm yr⁻¹ |
| 5 cm | 0.1 K | 0.41 yr | 2.8 cm yr⁻¹ |
| 5 cm | 0.3 K | 0.14 yr | 8.4 cm yr⁻¹ |
| 10 cm | 0.1 K | 1.6 yr | 1.4 cm yr⁻¹ |

So in physical units the corrected scallop mode **smooths its amplitude on a
weeks-to-years e-folding time** (`τ ∝ λ²/ΔT`) while **migrating downstream at a few
cm yr⁻¹** (`c_mig ∝ ΔT/λ`) — both **independent of the solver's accelerator `St`**,
fixed only by `k_th, ρ_iL, ΔT, λ`. This moves the G.2 magnitudes from `[HYP]` to
`[DERIVED dimensionless × LIT constants]`.

**Single anchored operating point — the wavelength is *not* a free choice.** For
melt/dissolution scallops the wavelength is *selected* by the flow: a constant
friction-velocity Reynolds number `Re_* = u_*λ/ν ≈ 2200` (Blumberg & Curl 1974;
recent ice-melting experiments give `2600–3400`), i.e. `λ = Re_*·ν/u_*`. Taking a
representative subglacial R-channel `u_* = 0.05 m s⁻¹` (with `ν = 1.79×10⁻⁶ m² s⁻¹`)
**selects `λ ≈ 7.9 cm`**, and `ΔT = 0.1 K`. Using the *same* solver mode (`n_w=12`)
for both `β` and the migration (so one `L₀` is consistent), `anchored_subglacial()`
collapses the table to a single quote:
 
> **τ ≈ 3.0 yr** (amplitude e-folding) and **c_mig ≈ 1.8 cm yr⁻¹** (downstream
> migration). Sensitivity: `ΔT = 0.03 → 0.3 K` gives `τ ≈ 10 → 1 yr`,
> `c_mig ≈ 0.5 → 5 cm yr⁻¹`.
 
**Regime-match caveat (honest).** The migration is quoted at the solver's strongest
drive (`U_drive=3`), whose mean-flow wavelength-Reynolds is `Re_λ = ū·λ_nd/ν_nd ≈
2.1·2.09/8×10⁻⁴ ≈ 5500` (the solver runs at `Pr=1`). That sits in the turbulent,
lee-separated regime — between Curl's friction-velocity selection (`Re_* ≈ 2200`)
and free-stream selection (`Re ≈ 22500`) — i.e. within a factor of a few of the
natural scallop-selection point, not tuned to it exactly (implied velocity scale
`U₀ ≈ 0.24 m s⁻¹` per nondim unit; Curl free-stream `U ≈ 0.51 m s⁻¹`). So the
*magnitude* `c_mig ≈ 1.8 cm yr⁻¹` is regime-representative to within an `O(1)`
factor; the *scalings* (`τ ∝ λ²/ΔT`, `c_mig ∝ ΔT/λ`, both `St`-free) are exact.

#### Implications of RESULT 14 — what the corrected mode tells us [NEW]

These are second-order consequences of the harmonic decomposition above; they are
not new runs but new *readings* of the same result.

1. **Migration is the morphological fingerprint of broken symmetry — a direct
   falsifier of K-theory on the ice.** The repo's thesis (`general_two_clocks/REPORT_THEORY.md` §1)
   is that single-eddy-viscosity (K-theory) closure crushes the "fast clock": it
   is *local, memoryless, and down-gradient-only*. A down-gradient eddy diffusivity
   acting on a sinusoidal interface is **parity-symmetric**, so it can only produce
   an **in-phase** (real, damping) flux response — `Im(s) = 0`, *no migration*. The
   probe measures `Im(s) = ω_mig ≠ 0`, which **no K-theory closure can produce**; it
   requires the flow-direction-selecting, non-local reattachment asymmetry. So
   `ω_mig ≠ 0` in *morphology* space is the exact analogue of `τ_c ≠ 0` (memory,
   RESULT 12/§G.5) and backscatter (up-gradient transfer) in *flux* space: the ice
   records the broken time/space symmetry that K-theory discards. (Empirically the
   `U=0` control has `E_cos ≈ 0` — `migration_zero_at_U0` in the figure — i.e. no
   quadrature without a mean flow, as the parity argument requires.) This is now a
   **direct measurement**, not just an argument: a positive control
   (`glaciers/scallop_ktheory_control.py`) advances the *same* momentum field at the *same*
   drive but transports heat with a down-gradient eddy-diffusivity closure instead
   of advection. A uniform eddy diffusivity gives `E_cos = O(10⁻¹⁸)` (machine zero,
   ratio `~10⁻¹⁴` to the resolved migration, seed-independent — the scalar literally
   cannot feel the flow direction), and even a *flow-aware* Smagorinsky eddy
   diffusivity gives `E_cos` ~`57×` below the resolved value — while both still
   **smooth** (`E_sin ≠ 0`). K-theory can damp but cannot migrate, exactly as the
   parity argument predicts (see "GPU/solver validation" below).

2. **"Right scaling, wrong operator" — the dimensional `τ ∝ λ²` is a decoy.** The
   e-fold time comes out `τ ∝ λ²/ΔT` (the classic diffusive / Mullins–Sekerka-looking
   law), but that `λ²` is produced **entirely by the conduction length↔time anchoring
   `L₀²/κ`**, *not* by a curvature operator — the measured `β/a` is `K`-independent
   (`K^{+0.13}`, not `K²`). So fitting `τ ∝ λ²` in the field would *wrongly* "confirm"
   the very curvature ansatz the probe falsified. The only discriminator is the
   `K`-exponent of `β/a` (≈0 vs +2), which the dimensional scaling hides: the
   dimensional law is **degenerate**, only the harmonic decomposition resolves
   mechanism.

3. **The sub-linear exponent `U^{0.5–0.8}` says migration is *friction-velocity*
   controlled, not advective.** Pure kinematic advection of a phase pattern gives
   `c_mig ∝ U` (exponent 1). The measured 0.5–0.8 is the signature of a **turbulent
   wall-flux** control (`q ~ u_* ~ U^{1/2}` in rough-wall scaling) at the lee
   reattachment — self-consistent with the anchoring, since the *same* `u_*` sets the
   Curl wavelength selection `Re_* = u_*λ/ν`. Prediction: the exponent → `½` in a
   cleaner rough-wall limit — **now confirmed** on GPU: a higher-fidelity run
   (`nx=160`, `spinup=2000`, 4 seeds, small amplitude) fits `0.52` over `U∈[3,8]`,
   sitting right on the friction-velocity `½` (see "GPU validation" below).

4. **`α` absent ⇒ scallops are a *two-conservation-law* phenomenon.** The lack of a
   `+α a^{1/2}` term is stronger than "unmeasured": **no** momentum+heat+Stefan solver,
   at any resolution or closure, can produce `Re(s) > 0`, because autonomous growth
   lives in a *different* conservation law (subglacial water-mass/pressure → dissipation
   → wall melt; the Röthlisberger opening). Scallop *existence* (amplitude) and scallop
   *kinematics* (migration + smoothing) are therefore governed by decoupled physics —
   which is exactly why the kinematics are cleanly earnable here while the amplitude is
   not.

5. **A constant-free, `ΔT`-free field test (and an inverse problem) [verified].** In
   the *product* `τ · c_mig` the constants `k_th`, `ρ_iL` **and `ΔT` cancel exactly**:

   ```
   I ≡ τ · c_mig / λ_phys  =  (−E_cos) / (λ_nd · (β/a) · a_nd · K_nd)
   ```

   — a pure ratio of the dimensionless migration gain to the smoothing gain
   (`constant_free_ratio()` in `glaciers/scallop_amplitude_harmonics.py`). The solver pins it at
   **`O(0.3–0.9)`** (`I = 0.33, 0.68, 0.88` at `n_w = 8, 12, 16` — mildly
   wavelength-dependent, *not* a single universal constant). Two payoffs: (i) a repeat
   scallop survey measuring (migration speed, amplitude e-fold time, wavelength) tests
   the corrected damped-migrating model **without** knowing the basal `ΔT` or any ice
   thermal constant; (ii) measuring `c_mig` + `λ` alone *pins* `ΔT` (since `c_mig ∝ ΔT/λ`),
   so `τ` becomes a **prediction** — the system is over-determined and falsifiable.

6. **Scallops as a slow paleo-flow recorder.** `τ ~ years` means a scallop field
   low-pass-filters the basal flow over years and its migration *direction* is the
   time-integrated flow direction; with (5) this could invert scallop morphology for
   basal flow with no ice-constant assumptions.

**GPU/solver validation of (1) and (3) [VERIFIED, Tesla P100 / CuPy + CPU].** Three
falsifiable solver checks were run directly on the frozen-interface flow decomposition
(`f_amp=0.4`, `n_w=12`):

- **Parity control — remove the drive (tests #1).** With the mean drive removed (`U=0`)
  the quadrature gain collapses to `E_cos = +2.1×10⁻⁶ ± 4.9×10⁻⁷`, versus
  `−8.4×10⁻⁵ ± 1.2×10⁻⁶` at `U=3` — i.e. **~40× smaller, statistically indistinguishable
  from zero**. No mean flow ⇒ no quadrature ⇒ `Im(s)→0`.
- **Parity control — K-theory positive control (tests #1, the keystone).** Keeping the
  drive but replacing the *advective* heat flux `−u·∇θ` with the down-gradient
  eddy-diffusivity flux `∇·(K_eddy∇θ)` that K-theory actually prescribes
  (`glaciers/scallop_ktheory_control.py`, `nx=128`, `spinup=2000`, 2 seeds): the resolved run
  migrates (`E_cos = −1.14×10⁻⁴ ± 2.8×10⁻⁶`), a **uniform** eddy diffusivity gives
  `E_cos = −1.2×10⁻¹⁸` (machine zero, ratio `~10⁻¹⁴`, `std = 0` exactly — the scalar is
  decoupled from the flow direction), and even a **flow-aware Smagorinsky** eddy
  diffusivity gives `E_cos = +2.0×10⁻⁶` (ratio `0.018`, ~`57×` below resolved). Both
  closures still **smooth** (`E_sin ≠ 0`). This is the direct measurement of the keystone:
  a down-gradient closure damps but cannot migrate; the migration lives entirely in the
  advective, flow-direction-selecting flux K-theory discards.
- **High-`Re`/rough-wall limit (tests #3).** Holding amplitude fixed and pushing the
  drive into the high-`Re_*` regime, the *local* migration exponent decreases toward the
  friction-velocity `½`. A first coarse pass (`nx=128`, `spinup=1500`, 2 seeds,
  `afrac=0.12`) gave a monotonic descent `0.82` (`U∈[1,3]`) → `0.69` (`U∈[2,6]`) →
  `0.59` (`U∈[4,6]`), reaching `½` *from above* as `Re_*` grows. A **higher-fidelity
  re-run** (`nx=160`, `spinup=2000`, **4 seeds**, `afrac=0.10`, `U∈[3,8]`) tightens this
  to **exponent `= 0.52`** (seed scatter now ~`1–2%` of the mean, down from ~`±0.1`) —
  i.e. essentially exactly `½`. (`E_cos` saturates / rolls over at the very top point
  `U=8`, so the strictly-monotonic `U∈[3,7]` window is a touch steeper, ~`0.6`; the
  full-range fit including saturation is `0.52`.) Either way the exponent is firmly
  sub-linear and lands on `½`, confirming the wall-flux (`q∼u_*∼U^{1/2}`) interpretation
  rather than kinematic advection (`∝U`). An amplitude sweep at fixed drive is consistent
  (`1.03→0.95→0.82` as `afrac` shrinks `0.30→0.12`: larger eddies carry a stronger
  advective admixture).

(Caveat: a single exponent still carries scatter from spin-up length, seed count, `U`-grid
density, and end-point saturation, so the robust claims are the three *qualitative* ones —
`Im(s)→0` at `U=0`, strictly sub-linear migration, and exponent`→½` at high `Re_*` — now
backed by a tightened central value of `0.52`.)

**Field-test protocol for the constant-free ratio `I` [proposed].** The ratio
`I ≡ τ·c_mig/λ` (implication 5) needs no `ΔT`, `k_th` or `ρ_iL`, so it can be checked
against a repeat scallop survey with no basal-thermal assumptions. Minimum measurement
set on a single scallop train of wavelength `λ`:

1. **`λ`** — crest-to-crest spacing (one bedform photograph / lidar swath).
2. **`c_mig`** — downstream crest displacement between two visits separated by `Δt`
   (`c_mig = Δx_crest/Δt`); the *sign* must be downstream (with mean flow).
3. **`τ`** — amplitude e-fold time, from the decay of crest height on an isolated /
   abandoned scallop (flow shut off), or from the amplitude spread across a train.

Then form `I_obs = τ·c_mig/λ` (dimensionless) and compare to the solver band:

| outcome | reading |
|---|---|
| `I_obs ∈ [0.2, 1.0]` and `c_mig` **downstream** | **consistent** with the damped-migrating mode `s=−β+iω_mig` |
| `c_mig ≈ 0` (no migration), `Im(s)=0` | parity-symmetric closure — **K-theory not falsified** on the ice (would contradict the solver) |
| `c_mig` **upstream** | neither model — new physics (e.g. depositional, not melt-limited) |
| `I_obs ≫ 1` or `≪ 0.1` | scaling mismatch — re-examine the `λ`↔`u_*` (Curl) anchoring |

Because `I` is `ΔT`-free, a *failure* cannot be excused by an unknown basal `ΔT`: the
test is clean. Two refinements:

- **Over-determination / inverse problem.** Measuring `c_mig` + `λ` alone *pins* `ΔT`
  (since `c_mig ∝ ΔT/λ`); `τ` is then a **prediction**. Measuring all three over-determines
  the model — a genuine falsification, not a fit.
- **`Re_*` self-consistency.** The same survey gives `u_* = Re_* ν / λ` with `Re_*≈2200`
  (Curl); an independent near-bed velocity estimate (borehole / tracer) checks that the
  flow regime matches, closing the loop between *kinematics* (`c_mig`, `τ`) and
  *selection* (`λ`).

**Literature grounding & data status for `I` [LIT survey].** Each ingredient of the triple
`(λ, c_mig, τ)` is independently corroborated, but **no single study co-reports all three**
with the isolation the test needs, so `I_obs` *cannot yet be pinned from the published
record* — the protocol stays a genuine prediction, not a post-hoc fit.

- **`λ` is solid.** The `Re_* = u_*λ/ν` selection is mainstream: Blumberg & Curl (1974)
  `≈2200` (the value anchored above), Thomas (1979) `≈1000` over `50 µm–1 m`, Hsu, Locher &
  Kennedy (1979) `λ=3180 ν/u_*`, Thorsness & Hanratty (1979)/Hanratty (1981) most-unstable
  band `3100–6300 ν/u_*`. Curl's (1966) free-stream `Re≈22 500` and the friction `Re_*≈2200`
  are **mutually consistent** at `u_*/U≈0.1`, so the anchoring is not a free choice.
- **`c_mig` direction & mechanism are solid.** Downstream migration is reported by Curl
  (1966), Ashton & Kennedy (1972), Hsu et al. (1979) and Gilpin, Hirata & Cheng (1980); the
  mechanism — maximum ablation *downstream of the trough at flow reattachment* (Blumberg &
  Curl 1974) — is exactly the quadrature lee signature `E_cos` isolates. Gilpin et al.'s
  linear stability predicts downstream migration when the heat-flux↔topography phase shift
  lies in `(π/2, π)`, i.e. the same statement as `Im(s)≠0`. Note `I = τ·c_mig/λ = Im(s)/(2π
  |Re(s)|)` — `I` is (up to `2π`) the ratio of migration rate to amplitude rate, the
  **`ΔT`-free repackaging of Gilpin's phase shift**.
- **`c_mig` *magnitude* and `τ` are sparse.** Cave surveys report `λ` only; migration speed
  and amplitude decay time are essentially never tabulated together for one train.
- **Sign caveat.** Real ice ripples *grow* (`Re(s)>0`: the Ashton–Kennedy/Gilpin/Hanratty
  moving-boundary instability), whereas this frozen-interface excess is *decay-only*
  (`Re(s)<0`) by construction. The migration `Im(s)` is the shared, K-theory-falsifying
  piece; matching the growing-ripple branch needs the moving-boundary feedback the frozen
  probe excludes (consistent with implication 4). So compare `|I|` to `|Im(s)/Re(s)|/2π`, not
  its sign.
- **Best dataset for a real `I`.** Bushuk et al. (2019, *JFM* 873) track `h(x,t)` at sub-mm/
  15 Hz through a "transition → equilibrium → **adjusting**" sequence; they **explicitly measure
  the mean horizontal crest advection speed `c`** (`h(x,t)→H(x−ct)`) and find max ablation `≈¾λ`
  downstream of the crest (the lee-quadrature `E_cos` signature). It is the one published dataset
  from which a real `c_mig` *and* an amplitude timescale come from the same train.
- **Regime-matched bound on `I_obs` [EST, figure-limited].** Bushuk's **scallop-adjustment** regime
  (exp 1b, drive cut `1.00→0.16 m s⁻¹`) *damps* the crests **and** migrates them downstream — i.e.
  `s=−β+iω_mig` with `Re(s)<0, Im(s)≠0`, the **exact** structure of the frozen probe (so the
  grow-vs-decay sign caveat **does not apply** to this regime; it is a like-for-like comparison). Using
  Bushuk's **measured** advection speed `c_mig=0.11 mm min⁻¹` (`=1.8×10⁻⁶ m s⁻¹`) and **observed**
  `λ≈13 cm` (they argue Curl's `Re_*≈22 500` is *not* the selector, so the observed `λ` is used), with
  the amplitude e-fold `τ≈1–3 h` read by eye off the non-collapsing profiles of their fig 3(o):
  `I_obs=τ·c_mig/λ ≈ 0.05–0.15` (point `≈0.1`; a Curl `λ=5 cm` would give `0.13–0.40`). So
  `I_obs~O(0.05–0.4)`. **Honest read:** the *sign structure matches exactly* (damping + downstream
  migration), and the magnitude is the same order as the solver's `O(0.3–0.9)`, **overlapping its lower
  edge only at the top of the input range** — the point estimate `≈0.1` sits a factor ~2–3 **below** the
  band (a mild tension, *not* a falsification: `I_obs` is `O(0.1–1)`, downstream). This **revises the
  earlier crude `≈1.2`** down ~10× (that used a guessed `c~1 mm min⁻¹`; the measured `c` is ~9× smaller).
  It is a **bound, not a pin** — `τ` is figure-limited (factor ~2) and `λ` is convention-dependent
  (factor ~2.6). **Clean test:** apply the §G.2 harmonic decomposition to Bushuk's raw `h(x,t)` arrays
  (read `Re(s)⇒τ`, `Im(s)⇒c_mig` for the same train) to turn the `O(0.1)` bound into a pin.
  This bound and the pin recipe are now **committed, tested code** in `glaciers/scallop_field_test.py`:
  `bushuk_adjustment_bound()` regenerates the numbers above (`figures/58`), and
  `harmonic_mode_rate(x,t,H)` implements the decomposition and is self-checked to recover a known
  `(Re(s), c_mig, λ, I)` to `<3%` on a synthetic damped-migrating train — so the only missing input
  for a pin is Bushuk's raw `h(x,t)` (supp. / on request).

> The full RESULT 14 write-up (method, dimensional bridge, keystone control, the constant-free
> ratio, and this literature grounding in one place) is **[`glaciers/REPORT_SCALLOP_MIGRATION.md`](glaciers/REPORT_SCALLOP_MIGRATION.md)**.

### G.3 Regime Transition Number (ocean intrusion) [HYP, leans on LIT]

> `RTN = (p_ocean − p_atm) / (ρ_i g H_ice − N_R)`,  with `N_R = p_i − p_w`,
> `p_i = ρ_i g H_ice` ⇒ denominator `= p_w`, so `RTN = (p_ocean − p_atm)/p_w`.

Dimensionally consistent; the qualitative bound (intrusion favoured by thin ice /
large channels / high ocean pressure) matches the 2024 intrusion literature
[LIT]. **Gauge caveat resolved:** RTN is a ratio of two pressures against the
*same* reference, so the atmosphere cancels exactly — at the grounding-line bed
seawater intrudes when `p_atm + ρ_w g d_base > p_atm + p_w` ⇔ `ρ_w g d_base > p_w`,
with **no** `p_atm` term. The validator and the Bedmap2 runner now use **gauge**
pressures throughout (default `p_atm = 0`); the earlier `−p_atm` in the numerator
was a spurious offset whose relative size `p_atm/(ρ_w g d_base)` blew up as
`d_base → 0`, i.e. worst exactly in the shallow grounding zone the prediction
targets (`glaciers/tests/test_validation_synthetic.py::test_rtn_gauge_convention_atmosphere_cancels`).
The one remaining caveat is that the conduction-limited `Q_max` the channel term
relies on rests on the §G.1 *empirical* result, not a bound, so the *threshold
magnitude* is still **[HYP]**; the *direction* (RTN>1 concentrates near grounding
lines) is [VERIFIED] on real Bedmap2 (§H.1).

### G.4 Non-local (memory) sliding law [HYP mechanism + DERIVED kernel shape + DERIVED lag value (order-of-magnitude) + DERIVED thermal-tail amplitude (St bound); data fit deferred]

> `τ_b(t) = C [ N(t) + δp_thermal(t)/g ] u_b(t)^{1/m} + ∫₀ᵗ K_ice(t−τ) τ_b(τ) dτ`

A memory-modified Schoof/Weertman sliding law: past basal heat thermally softens
the bed with a lag, giving rate-weakening-with-memory and predicting surges that
*lag* subglacial-water forcing — a genuinely new, testable mechanism. The ice kernel
`K_ice` it invokes is now derived and dimensionally closed (§B.2: `K_ice=G/ρ_iL` has
units `s⁻¹`), **but that fixes only the units, not the mechanism**: as a *thermal*
lag it is far too slow. The diffusion time `H_ice²/κ_ice` for `H=1000 m` is ~10⁴ yr,
and even the boundary-layer reading gives the wrong response *shape* (see below), so
the thermal reading is rejected as the lag-setting mechanism. **Status: mechanism
reassigned to hydraulics (below); thermal term retained only as a subdominant tail.**

**Corrected reading — the lag is *not* thermal. [DERIVED exclusion]** §H.2 falsifies
`τ=H²/κ` as a lag predictor on real lakes. Two facts then fix the correct
interpretation — and both are now **derived and machine-checked** in
`glaciers/validation/synthetic/thermal_tail_amplitude.py` (the exclusion is no longer an
assertion):

1. **Diffusive memory has no single timescale.** A basal perturbation of period
   `P` penetrates only a thermal skin depth `δ_skin = √(κ_ice P/π)` — **0.48 m at
   `P`=0.02 yr, 3.38 m at 1 yr, 4.79 m at 2 yr** (`skin_depth_m`) — so at observed
   surge periods the ice is effectively a *semi-infinite* medium and the far surface
   (`H≈2300 m`) is never felt (`δ_skin ≪ H`). `K_ice` is a power-law `∼ t^{−1/2}`
   tail, not an exponential lag.
2. **A drainage is an impulse, not a sinusoid.** The impulse response of a
   semi-infinite diffusive medium is *monotone decaying* (maximal at `t=0`, no
   interior peak — `impulse_is_monotone() == True`). Observed post-drainage speed-ups
   instead **rise to a peak** at 0.02–2 yr — the shape the *derived* hydraulic kernel
   produces (peak at `t*`, §G.4 above) but a diffusive-thermal kernel **cannot**. So
   shape *and* amplitude (the `St ≤ 0.06` bound, caveat iii) both **exclude ice
   thermal diffusion** as the lag-setting mechanism — leaving the empirical
   confirmation at a specific lake (USAP-DC-gated drainage dates, §H.2) as the only
   deferred step.

**Where the memory actually lives [HYP].** The peaked, sub-decadal lag is most
parsimoniously **hydromechanical**: drainage → cavity/channel pressurization →
effective-pressure `N` drop → Schoof/Budd sliding → Röthlisberger channel growth →
creep-closure relaxation. The peak ≈ the **hydraulic residence time** of the
cavity/channel system, not a thermal timescale; the thermal term survives only as
a weak `t^{−1/2}` tail at multi-decadal scales. A thermal kernel *tuned* (via a
boundary-layer depth `δ_T,ice` or a lumped water-inertia term `ρ_w c_w H_c/h`) to
land in 0.02–2 yr is **fitting, not deriving** — the implied `δ_T,ice` and heat-
transfer coefficient are unphysical (a steady Fourier balance `δ=k_iΔT/q` gives
hundreds of metres, and realistic `h∼10–10³ W m⁻² K⁻¹` gives `τ_water` of hours)
— so it is rejected here.

**The right ontology — two potentials + a hydraulic impedance kernel. [HYP]**
The lag is *not* a third "clock." There are **two driving potentials** plus a
distinct subsystem whose impedance sets the delay:

| field | what it is | operator | role |
|---|---|---|---|
| Leray pressure `p` | incompressible-flow constraint | elliptic (instantaneous) | enforces `∇·u=0` in the cavity |
| temperature `θ` | heat diffusion | parabolic | thermal boundary / melt rate; weak memory tail |
| hydraulic potential `φ=p_w+ρ_w g z` | subglacial water pressure | **nonlinear parabolic** (storage–transport) | governs storage, transport, **the observed lag** |

The hydraulic potential is **not** the Leray pressure — it is a separately-evolved
field of a distinct, coupled subsystem (schematically, Röthlisberger channel
`∂_t S = V_open − V_close + (Q_in−Q_out)` and GlaDS sheet `∂_t h_s = m/ρ_w −
∇·(k_s(h_s)∇φ)`, with **state-dependent** conductivity `k_s`). Its linearised
lumped Green's function **is** a memory kernel in the exact Mori–Zwanzig sense,

> `K_hydraulic(τ) = (1/RC) e^{−τ/RC}`,  `τ_lag ≈ R·C` (cavity/channel resistance × storage),

so the §G.4 non-local term survives — what changes is *which subsystem's kernel
dominates*: the **hydraulic** impedance kernel, not the ice-thermal one.

> **The "exact Mori–Zwanzig sense" phrase is now [DERIVED] (projection structure).**
> RESULT 21 (`glaciers/validation/synthetic/hydraulic_mz_projection_synthetic.py`) performs the
> actual projection it names, on the *same* coupled model `hydraulic_kernel_synthetic`
> builds. Eliminating the channel variable from the linear `ẋ = M x + F` exactly (no
> Markov approximation) yields a closed **generalized Langevin equation** for the
> resolved store, `ṡ = M_ss s + ∫₀ᵗ K(t−τ) s(τ)dτ + R(t)`, with memory kernel
> `K(τ) = M_sq M_qs e^{M_qq τ}` and Mori residual `R(t)` carried by the eliminated
> initial state. The harness certifies five properties: (A) the projection is **exact**
> — the GLE transfer function equals the full 2×2 resolvent at random complex `s` for
> random stable systems (max err `5×10⁻¹⁶`); (B) `K(τ)` **is** the eliminated channel's
> own Green's function `e^{M_qq τ}` (err `7×10⁻¹⁸`, decays at `1/τ₂`, sign-definite);
> (C) integrating the reduced GLE reproduces the full resolved trajectory (rel-err
> `9×10⁻⁸`); (D) the **memory is what makes the lag** — the full response peaks at the
> coupled-eigenvalue `t*` while the Markovian (adiabatic-elimination) closure is
> monotone with argmax at `t=0`; (E) the Markovian/local closure is recovered as the
> zero-correlation-time limit `K→(∫K)δ(τ)` with DC gain `∫K=M_sq M_qs/(−M_qq)`. This
> closes the *projection* claim (the lumped Green's function genuinely is an MZ memory
> kernel); the *ontology* — that `φ` is a distinct field, not the Leray pressure, and
> that this hydraulic subsystem is the one that dominates the real lag — remains the
> modeling **[HYP]** above, gated on field validation (§H.2, USAP-DC).

**The peaked shape is [DERIVED] — it is generic to a two-compartment linear system.**
The open question in caveat (i) below was whether reproducing the observed
*rise-to-a-peak* requires the full nonlinear model. It does not. Take the linearised
lumped hydraulics as **two coupled storages** — cavity water pressure (charge time
`τ₁`) driving channel cross-section (open time `τ₂`):

> `dx₁/dt = −x₁/τ₁ − a·x₂ + f(t)`,  `dx₂/dt = b·x₁ − x₂/τ₂`.

A *single* storage gives the monotone `g₁(t) = (1/τ)e^{−t/τ}` (peak at `t=0`). The
*downstream* (channel) impulse response of the coupled pair is, in the cascade limit,

> `g₂(t) = (e^{−t/τ₁} − e^{−t/τ₂})/(τ₁ − τ₂)`,  `g₂(0)=0`, interior peak at  `t* = τ₁τ₂ ln(τ₁/τ₂)/(τ₁ − τ₂) > 0`,

which **starts at zero, rises to a peak, then decays** — built entirely from
monotone, overdamped components, with *no* nonlinearity and *no* special pulse shape.
`glaciers/validation/synthetic/hydraulic_kernel_synthetic.py` confirms all three: the
first-order model is monotone (argmax at `t=0`), the cascade peak matches the
analytic `t*` to `<0.1%`, and the full 2×2 matrix model (real eigenvalues) gives a
single interior peak with no oscillation. So the *qualitative signature* — a sub-decadal
peaked lag — is now a derived consequence of two coupled hydraulic storages, closing
caveat (i) as a structural result rather than a conjecture.

**Caveats / status:**
(i) The single first-order `RC` is monotone; the peak needs the **second** storage
(cavity↔channel coupling) — now *derived* above as the minimal mechanism, so this is
no longer an open "needs pulse forcing or feedback" question.
(ii) The lag **value** `t*` (equivalently `τ₁, τ₂, R, C`) is now **[DERIVED — order of
magnitude]** in `glaciers/validation/synthetic/hydraulic_lag_derivation.py`: the lumped
GlaDS-sheet / linked-cavity + Röthlisberger-channel ODEs are written with **named
physical constants and literature parameter ranges** (Werder et al. 2013; Schoof 2010;
Hewitt 2011; Cuffey & Paterson 2010), linearised about the *observed* near-flotation
effective pressure, and `τ₁, τ₂` are read off the **2×2 Jacobian** (analytic-vs-finite-
difference rel-err `≈1.4×10⁻⁵`) instead of planted. The physical clock is the **cavity
opening–closure time** `τ_open = 1/(u_b/l_r + A Nⁿ)` (sliding-opening + Nye creep-
closure), and by (★) `t* ≈ τ_open · ln(τ_store/τ_open)` — set by `τ_open` and only
*logarithmically* by the storage time `τ_store = R·C` — so it lands robustly sub-decadal:
baseline `t* ≈ 0.01 yr`, and a literature-range Monte-Carlo gives median `≈0.012 yr`,
p75 `≈0.027 yr`, p95 `≈0.10 yr`, with **34 % of plausible parameter space inside the
observed 0.02–2 yr band** — i.e. **~6 orders below** the falsified thermal `H²/κ`
(~10⁵ yr, §H.2). The module also **pins the regime**: the literal Röthlisberger
*channel* `(p_w, S)` is structurally unstable (`trace > 0` — the Schoof-2010 /
Kingslake-2015 lake-*oscillation* branch, with no decaying peaked kernel), so the
observed single rise-and-decay surge is **cavity-opening-paced**, not channel-paced;
and a full **nonlinear** ODE integration under an impulse drainage (Step B) confirms the
linearised `t*` survives (peak-time drift `<20 %` even at a 40 %-of-`N` pulse). **What is
*not* earned:** a data-`[VERIFIED]` lag *fit* — the vetted drainage-*date* catalogue is
USAP-DC-gated (§H.2), so an end-to-end observed-vs-predicted match is not runnable here
— and the lumped coefficients are a reduction of the full spatial GlaDS PDE. So this is
a derived *order-of-magnitude* `t*` consistent with the observed band, not a precision
data fit. (iii) The thermal term is the subdominant `K_thermal(τ) ∼ τ^{−1/2}` skin-depth
tail (§B.2) — and its **amplitude** is now **[DERIVED — quantitative bound]** in
`glaciers/validation/synthetic/thermal_tail_amplitude.py`, not just asserted. A memory kernel's
weight in the sliding law is its DC gain `∫₀^∞ K dτ`; integrating the §B.2 closed-form
ice kernel gives, in closed form,
`W_thermal = |∫₀^∞ K_ice dτ| = |−ρc θ_far/(ρ_iL)| = c_i|θ_far|/L ≡ St` — exactly the
**Stefan number** of the far-field basal undercooling `θ_far` (verified: the numerical
`∫` of the §B.2 kernel matches `c_i|θ_far|/L` to rel-err `≈8×10⁻⁵`). Because ice's latent
heat dwarfs the sensible heat available over any physical undercooling, `St ≪ 1`: for
`θ_far` = 0.1 K (near-temperate ice-stream bed) → 10 K (a generous cold-margin bound),
`St` spans only `6×10⁻⁴ … 6×10⁻²`. The dominant hydraulic impedance kernel
`K_hydraulic = (1/RC)e^{−τ/RC}` is a **unit-DC-gain** kernel (`∫₀^∞ = 1` exactly), so
`W_thermal/W_hydraulic = St ≤ 0.06` — the thermal term is **at most a few-percent
correction to the basal stress, subdominant by construction**, independent of the
hydraulic details. Three checks reinforce this at the observed 0.02–2 yr band: the
*in-band* weight `∫_{0.02–2yr}K_ice/St` is only `2×10⁻⁴ … 0.2` across V̄ = 0.001–1 m yr⁻¹
(so `≲1 %` of `St`); the skin depth `√(κP/π)` = 0.48/3.38/4.79 m at P = 0.02/1/2 yr
(reproducing the numbers above) is `≪ H ≈ 2000 m`; and the semi-infinite impulse response
is **monotone decaying** (no interior peak), so even at full amplitude it cannot make the
observed rise-to-a-peak surge. A literature sweep (n=3000 over `θ_far, V̄, κ`) keeps
`St < 0.07` for **100 %** of the space. *Not earned:* a site-specific thermal correction —
`θ_far, V̄` are problem inputs (§B.2), so `St` is bounded, not pinned, without local data;
the bound is what "subdominant" needs. This mechanism reassignment is specific to the
*sliding-lag* problem; it does **not** touch the §10e–10f / Candidate-3 melt-ceiling
result, where the limit is genuinely the thermal conductive sublayer.

### G.5 Clock-mismatch correction to K-theory [DERIVED identity + MEASURED coefficient/sign — the cleanest of the new terms]

> `∂_t θ + u·∇θ = ∇·(K_θ ∇θ) − CMN · ∇·(∂_t K_u ∇θ)`,  `K_θ = K_u/Pr_t`

**The commutator identity [DERIVED, exact].** Write the eddy-diffusion operator as
`D_t[θ] = ∇·(K_u(x,t) ∇θ)`. Spatial and temporal derivatives commute (`∂_t∇ = ∇∂_t`),
so the product rule gives

> `∂_t(∇·(K_u∇θ)) = ∇·((∂_t K_u)∇θ) + ∇·(K_u∇(∂_t θ))`,

and subtracting `D_t[∂_t θ] = ∇·(K_u∇(∂_t θ))` leaves, with no approximation,

> `[∂_t, D_t]θ ≡ ∂_t(D_t[θ]) − D_t[∂_t θ] = ∇·((∂_t K_u)∇θ)`.   (exact identity)

This is the spurious pressure→temperature coupling K-theory hides: when the
momentum-derived `K_u` is itself unsteady, advancing `θ` with a frozen `K_u`
mis-orders the operators by *exactly* this commutator. It **vanishes for steady
turbulence** (`∂_t K_u = 0`) and is maximal in transients (plumes, surges).
`glaciers/validation/synthetic/cmn_synthetic.py` confirms the identity to `3.7e-7` (→0 as
`dt→0`), the steady-state null (machine zero), and linearity in `∂_t K_u`.

**Dimensional closure — the coefficient is a *time*, not an O(1) constant
[DERIVED].** The commutator has units `Θ·T⁻²` (a rate-of-rate), one inverse-time
*more* than the tendency `∂_t θ ~ Θ·T⁻¹`; equivalently its ratio to the diffusion
term scales as `∂_t K_u / K_u ~ T⁻¹`. The same validator shows this directly:
doubling the transient rate `ω` leaves `∇·(K_u∇θ)` unchanged (rate-independent) but
doubles `∇·((∂_t K_u)∇θ)`. Dimensional consistency therefore *forces* the
coefficient to carry units of time — `CMN ≡ τ_c`, a correlation/memory time (the
decorrelation time of `K_u`), **not** the dimensionless `CMN ≈ 1` the original stub
implied. This removes a latent dimensional inconsistency from the heuristic form.

**The coefficient `τ_c`: value, sign, and add/remove-flux — now closed [MEASURED /
DERIVED, RESULT 12].** `general_two_clocks/gle_coefficients.py` (`general_two_clocks/REPORT_GLE_COEFFICIENTS.md`) pins all
three open pieces with no new data:
- **Value & sign.** `τ_c` is the decorrelation time of `K_u`, measured at fixed
  interior probe points as `τ_c ≈ 0.02–0.03` (solver units), consistent with the
  RESULT-8 SGS-force memory. An autocorrelation time is non-negative *by
  construction* and the measured value is strictly positive ⇒ **`sign(τ_c) = +`**.
- **Adds or removes flux?** Neither, on average. To first order
  `∇·(K_u∇θ) − τ_c∇·(∂_tK_u∇θ) = ∇·((K_u − τ_c∂_tK_u)∇θ) ≈ ∇·(K_u(t−τ_c)∇θ)` — a
  **pure time-lag** by `τ_c` (verified: lag rel-err `1.1×10⁻⁴`, scaling as `τ_c²`).
  Over a steady `K_u` cycle (`⟨∂_tK_u⟩=0`) the time-mean correction is machine-zero
  (`|mean|/rms ≈ 7×10⁻¹⁹`): it adds flux while `K_u` grows, removes it while `K_u`
  decays. **This is exactly why RESULT 8/11 measure no mean-melt / `C_G` effect** in
  the statistically steady cavity — §G.5 and that null are the same statement.
So the term is now **structurally derived, dimensionally closed, and coefficient-
pinned** — the soundest of the new proposals, tying directly to the "two clocks"
thesis (§A) and resembling known dynamic / non-equilibrium eddy-diffusivity fixes.
(Cf. the unified memory GLE, §D.4.)

**Operational payoff [VERIFIED — synthetic solver, RESULT 15].** §H.3 below now
runs the term in a transient K-theory thermal solver (`general_two_clocks/cmn_solver_demo.py`): with
`CMN=+τ_c` it reconstructs the lagged-clock truth to `O(τ_c²)`, cutting the
transient error ~15× (naive `∝τ_c`, corrected `∝τ_c²`), is identically zero in
steady turbulence, and the `+` sign is the unique error-reducing one (`−τ_c` is
worse) — so the §G.5 sign is now a *solver* result, not only an autocorrelation
fact.

### G.6 Unified melt rate [HYP form, corrected to local-flux; `(a/λ)²` closure MEASURED → FALSIFIED]

> `v_melt(x) = (1/ρ_i L) [ h_local(u*, a/λ) · (T_bulk − T_melt(N))
>            − ∫₀ᵗ K_ice(t−τ) q_water(τ) dτ ]`

Phenomenology that stitches §G.1–§G.5 together. The growth-driving term must be the
**local face flux** `h_local` (which legitimately exceeds flat; §G.1 local-vs-mean
and §G.2), *not* the spatial-mean conductance — the spatial mean is bounded below
the flat wall by the §G.1 empirical result. It inherits the §G.4 kernel caveat
(§B.2) and the `a/λ` enhancement closure
(Caveat D). The posited closure `δ_T,eff = δ_T,flat·(1 + ζ(a/λ)²)` reproduces
`Nu<1` but its **quadratic amplitude form is now [MEASURED → FALSIFIED]**
(`glaciers/scallop_amplitude_closure.py`, `figures/59`): the frozen-boundary amplitude
sweep at the Part-C config (`nx=128, n_waves=12, U_drive=1.5`, local-normal Nu;
reproduces the §G.1 sweep) gives a mean-Nu deficit `D = 1/(Nu/Nu_flat) − 1` that
is **amplitude-independent** — `D = 0.11 ± 0.03` with **no** power-law in `a/λ`
(`p_free ≈ 0`, `R²_logD ≈ 0`); the pinned quadratic (`R² = −5.3`) and even a
linear (`R² = −2.4`) law are *worse than a constant*. So `Nu<1` is set
near-geometrically (present already at `a/λ = 0.05`, the §G.1 separated-area
mechanism), **not** by an amplitude-thickened boundary layer: the right closure is
`δ_T,eff ≈ δ_T,flat·C` with `C ≈ 1.11` constant over the resolved range, not
`(1 + ζ(a/λ)²)`. (No tension with the §G.1 partition, whose `surplus` *and*
`deficit` shares both grow with `a/λ`: it is their **difference** — the net mean
deficit measured here — that is amplitude-flat, the growing reattachment surplus
tracking the growing thickened/reversed deficit.) The *local* lee flux that drives growth does keep rising with
amplitude (`R_max` grows `1.9 → 4.3` across the sweep), consistent with the
local-vs-mean split above. **[HYP phenomenology; `(a/λ)²` closure FALSIFIED, mean
deficit MEASURED constant].** For `RTN>1` (ocean intrusion, §G.3) `T_bulk` is set by the ocean,
`h_ocean` is 10–100× larger, and this Type-I expression is replaced by an
ocean-controlled one — the Type-II/III branch of the regime map (§F).

### G.7 Summary of the audit

| Proposed law | Verdict | Status |
|---|---|---|
| G.1 Type-I thermal-wall bound | Not a theorem; `Nu<1` empirical, and the suppression mechanism is now **earned** by the *exact area-partition* — `Nu<1` ⟺ the thickened+reversed **deficit** outweighs the reattachment **surplus**, an identity that closes to machine precision at every amplitude (`glaciers/scallop_g1_populations.py`); the older `(1+CV²)` two-moment truncation is **falsified** (`CV_δ~O(1)`, predicts `Nu>1` at 4/6 amplitudes) | [VERIFIED empirical] + [VERIFIED mechanism (exact area-partition)] + [(1+CV²) truncation FALSIFIED] |
| G.2 Scallop-creep amplitude | Monostable (one `a*`), **not** bistable; creep negligible. **RESULT 14** (frozen-interface wall-flux harmonic decomposition) corrects the two-term form: `β` is ~`K`-independent (`K²` curvature ansatz **falsified**), there is **no** `+α a^{1/2}` growth term (in-phase excess is smoothing at every driven `(a,U)` → smoothing-limited), and the real flow channel is a **migration** term `∝ U^{0.5–0.8}`. Corrected model: damped downstream-migrating mode `s=−β+iω_mig`. **Dimensional bridge** (Stefan condition, St-free): amplitude smooths in weeks–years (`τ∝λ²/ΔT`), pattern migrates at a few cm yr⁻¹ (`c_mig∝ΔT/λ`). **Anchored point** (Curl wavelength selection `λ=Re_*ν/u_*`, `u_*=0.05 m/s`→`λ≈7.9 cm`, `ΔT=0.1 K`): `τ≈3.0 yr`, `c_mig≈1.8 cm yr⁻¹`. **Implications**: migration `Im(s)≠0` is a parity-symmetry break that **no K-theory closure can produce** (morphological analogue of memory/backscatter, falsifies K-theory on the ice); the dimensional `τ∝λ²` is a *decoy* (from `L₀²/κ`, not curvature); and the product `τ·c_mig/λ` is a **constant-free, ΔT-free** field test, `I=O(0.3–0.9)` (`constant_free_ratio()`). **GPU/solver checks** (P100/CPU): parity control `E_cos→0` at `U=0` (~40× below `U=3`), a **K-theory positive control** (down-gradient eddy diffusivity at full drive) gives `E_cos`→machine-zero (uniform) / `57×` below resolved (Smagorinsky) — the keystone falsifier *measured*, not argued — and a higher-fidelity high-`Re_*` sweep (`nx=160`, 4 seeds) pins the migration exponent at `0.52` (`→½`). A **constant-free field-test protocol** for `I` is specified. **RESULT 22** (`amplitude_generalization_scan`) generalises the swept window: across `a₀/λ∈[0.05,0.40]` the smoothing-only (no `+α`) verdict and the falsified-`K²` (curvature) verdict hold at **every** amplitude (`p∈[−0.27,+0.66]`, far below `+2`); strict `β`-flatness only holds for `a₀/λ≳0.2` (shallow-amplitude conduction signal is noise-floor-limited). **RESULT 23** (`drive_window_scan`) closes the other half of Caveat D: pushing the drive to `U=6` (beyond the swept `U∈[1.5,3.0]`), the smoothing-only verdict survives at every drive (no `+α` channel from strong-drive lee separation) and the migration keeps its **sub-kinematic** `∝U^{0.5–0.8}` friction-velocity scaling (`U^{+0.48}`, never `U¹`; it even saturates/rolls over at `U≳4.5`) — **Caveat D retired** | [VERIFIED falsification + DERIVED form] + [magnitudes = DERIVED × LIT constants] + [implications: ANALYTIC + VERIFIED ratio + GPU-VERIFIED parity & exponent] + [amplitude + drive-window generalisation VERIFIED — RESULT 22/23; Caveat D retired] |
| G.3 Regime Transition Number | Dimensionally OK (gauge caveat resolved); threshold inherits §G.1; direction [VERIFIED] §H.1 | [HYP / LIT] |
| G.4 Non-local sliding law | Literal thermal kernel [FALSIFIED] on real lakes (§H.2); lag is **hydromechanical**, not thermal. The peaked sub-decadal shape is **derived** as the generic two-compartment (cavity→channel) linear response (peak at `t*=τ₁τ₂ln(τ₁/τ₂)/(τ₁−τ₂)`); the lag **value** is now **derived to order-of-magnitude** from linearised GlaDS-sheet/cavity + Röthlisberger physics over literature ranges (`τ_open=1/(u_b/l_r+ANⁿ)`; baseline `t*≈0.01 yr`, 34 % of swept space in 0.02–2 yr, ~6 orders below thermal `H²/κ`; channel branch is the unstable Kingslake oscillator; nonlinear Step B confirms it) — `hydraulic_lag_derivation.py`. The retained thermal term's **amplitude** is **derived subdominant**: its kernel weight `∫K_ice dτ = c_i|θ_far|/L = St ≤ 0.06` (Stefan number) vs the unit-gain hydraulic kernel — `thermal_tail_amplitude.py`. Not a data fit (drainage dates USAP-DC-gated) | [HYP mechanism] + [DERIVED kernel shape] + [DERIVED lag value (order-of-mag)] + [DERIVED thermal amplitude] |
| G.5 Clock-mismatch correction | Commutator identity `[∂_t,D]θ=∇·((∂_tK_u)∇θ)` **derived exactly**; coefficient forced to be a time `τ_c`; `τ_c` **measured** (`≈0.02–0.03`, sign **+**), correction is a pure net-zero time-lag explaining the RESULT-8/11 null (RESULT 12) | [DERIVED identity + dim. closure] + [MEASURED coeff/sign] |
| G.6 Unified melt rate | OK as phenomenology if growth uses *local* flux. The posited `δ_T,eff = δ_T,flat(1+ζ(a/λ)²)` amplitude closure is **falsified**: the frozen-boundary amplitude sweep (`glaciers/scallop_amplitude_closure.py`, `figures/59`; reproduces the §G.1 sweep) shows the mean-Nu deficit is **amplitude-independent** (`D=0.11±0.03`, `p≈0`; quadratic `R²=−5.3`), i.e. `Nu<1` is near-geometric, while the *local* lee flux `R_max` still grows `1.9→4.3` with `a/λ` | [HYP phenomenology] + [`(a/λ)²` closure MEASURED → FALSIFIED] |

**Bottom line.** *(Re-audited in the lens of the now-complete §G.2 / RESULT 14.)*
There are **two keepers, and they turn out to be the same statement told in two
spaces.** §G.5 is the cleanest *flux-space* result — the clock-mismatch
commutator with a measured `τ_c` — but it explains a **null** (the RESULT-8/11
correction is net-zero). §G.2 is the framework's first **constructive, positive,
falsifiable** result: RESULT 14 *derives* the corrected mode `s=−β+iω_mig`
(it falls out of the harmonic decomposition, it is not an ansatz), measures every
coefficient in the solver, anchors physical units independently via the Curl
selection law, and yields a constant-free, `ΔT`-free field test `I`. Its keystone
— migration `Im(s)≠0` is a parity-symmetry break that **no down-gradient K-theory
closure can produce**, now *measured* by the K-theory positive control
(`E_cos`→machine-zero) rather than only argued — is the **morphology-space
analogue** of §G.5's memory (`τ_c`) and backscatter: both are fingerprints of the
same broken time/space symmetry that a local, memoryless, down-gradient closure
structurally discards. So the two keepers unify: K-theory's blindness leaves a
trace in the *flux* (§G.5) **and** carved into the *ice* (§G.2).

The rest sharpen the framework by **removing wrong things**: §G.1 is real — `Nu<1`
is empirical and its mechanism is now **verified** (distribution-dominated, not
moment-dominated: the exact area-partition, not a `(1+CV²)` bound); §G.4's literal
thermal kernel is [FALSIFIED] on real lakes (lag is hydromechanical; only the
kernel *shape* is derived); §G.3 and §G.6 are reasonable [HYP] scaffolding.

Honest status of G.2 (updating the old "all-[HYP]" verdict): the falsification of
the `K²` curvature ansatz and the corrected *form* are [VERIFIED]/[DERIVED], the
keystone is **GPU-verified**, and the migration exponent is pinned at `0.52`. The
dimensional *magnitudes* are no longer free [HYP] — they are now **[DERIVED
dimensionless × LIT constants]** via the St-free Stefan bridge and the Curl anchor.
The one remaining gap is the important one: the whole result is verified **in the
solver, not yet against field scallops**. Computing `I_obs` from a real scallop
train (§G.2 field-test protocol; Bushuk et al. 2019 is the candidate dataset) is
the single step that would move RESULT 14 from *a discovery in the framework* to
*a discovery about the world*. That distinction **is** the
[VERIFIED]/[LIT]/[HYP] discipline of this document — sharpened, not softened.

---

### References used as closures above
Curl (1966); Blumberg & Curl (1974) — scallop wavelength `Re*`. Nikuradse (1933) —
sand-grain roughness `k_s`. Röthlisberger (1972); Nye (1976); Spring & Hutter
(1982) — channel evolution. Werder et al. (2013); Hill et al. (2024) — GlaDS
sheet/channel conductivity. Dow et al. (2018) — basal channel observations. Armi
(1986); Pratt (1986) — composite-Froude hydraulic control. Stefan (1891); Nye
(1953); Crank (1984) — two-phase Stefan. Dubnick et al. (2020) — basal scallop
field observations.

---

## §H — Falsifiable forecasts for community validation

The §G predictions are **[HYP]**: the solver cannot test them on its own, and no
systematic field survey exists to score them precisely. Rather than end with a
vague "future work: validate with data", we state each as an **exact, testable
forecast** and run whatever part of it *open* data can already settle. The
synthetic harnesses (§V.3) prove the math; the runs below expose the physics to
falsification on real geometry. Reproduce them with:

```
python glaciers/validation/external/run_rtn_bedmap2.py  --stride 3 --phi 0.9   # §H.1
python glaciers/validation/external/run_sliding_real.py --with-velocity Mac1   # §H.2
```

Full numbers, data provenance and caveats: `glaciers/validation/REAL_DATA_RESULTS.md`.

### H.1 RTN prediction — **[VERIFIED directional]** on real Bedmap2

> **Forecast.** Ocean intrusion into subglacial space concentrates where
> `RTN = p_ocean/p_w > 1` (gauge; the atmosphere cancels, §G.3) — i.e. near
> grounding lines, anticorrelated with ice thickness.

**Test run.** RTN built from real BAS Bedmap2 geometry (1 km, decimated to 3 km;
1.33 M grounded cells): overburden `p_i = ρ_i g H`, ocean head `p_ocean = ρ_w g
d_base` (`d_base` = bed depth below sea level), subglacial water `p_w = φ p_i`.
The `RTN>1` fraction of grounded ice, binned by distance to the grounding line
(φ = 0.9):

| dist-to-GL [km] | 0–5 | 5–10 | 10–25 | 25–50 | 50–100 | 100–250 | >250 |
|---|---|---|---|---|---|---|---|
| RTN>1 fraction | 15.1% | 6.0% | 3.6% | 2.1% | 0.8% | 0.0% | 0.0% |

Median distance-to-GL is **6 km** for `RTN>1` cells vs **222 km** for the rest.
The decay is monotone and robust across `φ ∈ {0.8, 0.9, 0.95}` (overall
2.5% / 1.3% / 0.6% — only the magnitude scales). So `RTN>1` concentrates sharply
at the grounding line and anticorrelates with ice thickness inland — exactly the
directional prediction, on real geometry.

**Gauge fix re-run (sharpens the signal).** The table above uses the corrected
gauge RTN (`p_atm = 0`). Re-running the *old* `−p_atm` convention on the identical
grid reproduces the original pre-fix table exactly (`12.2 / 5.2 / 3.2 / 1.7 / 0.6 %`,
overall 1.07%), so the bug **understated** the grounding-line concentration: the
fix flips **2,919** grounded cells to `RTN>1`, clustered at **median 4.2 km from
the GL**. This is the predicted consequence of the gauge analysis — the spurious
offset `p_atm/(ρ_w g d_base)` blows up as `d_base → 0` (the grounding line), so
removing it adds intrusion-favourable cells preferentially there. The §H.1
direction is unchanged and the pre-fix conclusion was conservative.

**What this is *not*.** Not a precision/recall score: no gridded intrusion survey
exists, and 1 km cannot resolve the ~1–10 m channel the Röthlisberger `N_R` term
needs, so the channel size is implicit in `φ` rather than resolved. The result is
*directional*, as the §G.3 caveat demands.

**Falsification.** If intrusion sites were found scattered uniformly inland
(no GL concentration), the RTN ordering would be falsified. The grounding-line
concentration is a real, checkable prediction for systematic radar surveys
(IceBridge successors, NISAR).

#### H.1.1 Two constant-free consequences of the gauge RTN

Because the gauge fix collapses RTN to a pure buoyancy ratio
`RTN = (ρ_w/φρ_i)·(d_base/H)` — an Atwood-style number in which `g` and the
atmosphere both cancel — two quantitative, falsifiable diagnostics drop out with
**no** thermal or pressure constants. Both are computed on real Bedmap2 by
`glaciers/validation/external/rtn_phi_calibration.py` (3 km, 1.33 M grounded cells).

**(1) The intruded area is a calibrated inverse for φ.** `RTN>1 ⇔ d_base/H >
φ·ρ_i/ρ_w`, so the grounded-ice area with `RTN>1` is a monotone readout of the
continent-effective basal water fraction φ:

| φ | 0.70 | 0.80 | 0.90 | 0.94 | 0.98 |
|---|---|---|---|---|---|
| RTN>1 area [% grounded] | 3.85 | 2.51 | 1.29 | 0.75 | 0.11 |
| RTN>1 area [10³ km²] | 461 | 301 | 154 | 90 | 13 |

The curve is steep and monotone — near φ=0.9 the sensitivity is
**≈0.13 %-area (~15,000 km²) per +0.01 in φ** — so a mapped intrusion *extent*
inverts directly (and sensitively) for φ. The intrusion's *inland reach* carries
the same information: the median `RTN>1` distance-to-GL grows from 3 km (φ=0.98)
to 13 km (φ=0.70) as the bed wets. This is the §G.3 analogue of the §G.2
constant-free morphology inversion `I = τ·c_mig/λ` — same "reference cancels →
constant-free inverse," different observable.

> **The inversion is a unique, unbiased estimator [VERIFIED — synthetic].** The
> area↦φ map is calibrated by `glaciers/validation/synthetic/rtn_phi_synthetic.py`
> (RESULT 18, `glaciers/REPORT_RTN_PHI_INVERSION.md`), reusing the **real** `build_rtn`
> math on a planted population whose `RTN>1` area spans the same range as Bedmap2
> (3.83 %→0.118 % across φ=0.70→0.98). It confirms: `A(φ)` is **strictly
> monotone** (so the inverse is single-valued), the critical-thickness threshold
> `RTN>1 ⇔ H < H*=(ρ_w/φρ_i)·d_base` is **exact cell-for-cell**, and a *finite*
> mapped survey recovers φ **without bias** (`N=4000`: mean φ̂=0.905 at φ=0.90,
> spread ∝1/√N). So the headline inversion is unique and unbiased, not merely
> monotone-as-drawn — the §G.3 analogue of `rtn_synthetic` (classifier),
> `glmig_synthetic` (RESULT 16) and the §G.4 lag estimator (RESULT 17).

**(2) A hydrology-corrected MISI margin.** `RTN>1 ⇔ H < H* = (ρ_w/φρ_i)·d_base`,
a *local critical thickness*; the margin `m = (H−H*)/H` is the fraction of present
thickness a cell must lose before intrusion is favoured. Since φ<1,
`H* = H_flot/φ` with `H_flot = (ρ_w/ρ_i)·d_base`, so the RTN=1 line sits **inland
of the classical flotation line** by the grounded-but-intrudable band
`H_flot < H < H*` (width ∝(1−φ)). On Bedmap2 (φ=0.9) that band is **152,500 km²**
(1.27 % of grounded ice), median **6 km** from the GL but reaching **42 km**
inland (p90); **~1.1 %** of grounded ice sits within 10 % thinning of its tipping
threshold, concentrated on the West Antarctic / Peninsula / Siple-Coast grounding
zones. Standard MISI keys off flotation (φ=1); RTN says a pressurized bed
pre-conditions intrusion earlier, by a mapped `(1−φ)` margin — a closure-level
refinement, not a new instability.

> **The band structure & (1−φ) scaling are calibrated [VERIFIED — synthetic].**
> `glaciers/validation/synthetic/rtn_phi_synthetic.py::run_misi_band` (RESULT 19,
> `glaciers/REPORT_RTN_MISI_BAND.md`), on the real `build_rtn` math, confirms: the per-cell
> band width `(H*−H_flot)/H_flot = (1−φ)/φ` **exactly** (machine precision); the
> band `H_flot<H<H*` equals `grounded ∧ (RTN>1)` cell-for-cell; `H*>H_flot` so the
> RTN=1 line is **inland of flotation**; and the band-area fraction **vanishes
> monotonically as φ→1, linear in (1−φ)** to leading order. So the hydrology
> correction is a structurally exact, (1−φ)-scaling band — the companion to the
> RESULT-18 area↦φ inversion that closes the second §H.1.1 consequence.

#### H.1.2 The intrusion clock — kinematics **[VERIFIED]**, hydraulic pacing **[HYP]**

RTN (§H.1) and its `φ`/`H*` corollaries (§H.1.1) are *static* — they say where
intrusion is favoured, not how fast it spreads. The rate companion: the RTN=1
line is the zero level-set of `m = H − H*`, and with the bed (hence `H*`) fixed
while ice thins at `dH/dt`, the front advances inland at the level-set speed
`v_front = (dH/dt)/|∇m|`. So the **geometric amplification** `A = 1/|∇m|`
(km of inland advance per metre of thinning) is a pure-geometry field, mapped
from Bedmap2 by `glaciers/validation/external/rtn_intrusion_clock.py` (φ=0.9, near-tipping
front `0 < m/H < 0.2`, 25.8 k cells):

- **The line is mostly *pinned*, not runaway.** `A` is small over most of the
  margin — median **0.09 km per m** of thinning (p90 0.42) — i.e. steep
  thickness/bed gradients hold the RTN=1 line in place. Runaway (large-`A`) cells
  are a tail concentrated on the **West Antarctic ice plains** (Siple Coast),
  the geometric fingerprint of ice-plain vs pinning-point behaviour.
- **The predicted advance is the right order.** `v_front = A·(dH/dt)` gives
  **0.05 → 0.47 km yr⁻¹** for a thinning band `dH/dt = 0.5 → 5 m yr⁻¹`, squarely
  in the observed grounding-line-retreat range (Thwaites/PIG ~0.1–1 km yr⁻¹) — a
  sanity check that the kinematics are physical, not that they reproduce any
  specific basin.
- **Necessary co-location with short memory.** The near-tipping front sits on ice
  whose §G.4 thermal memory `τ_ice = H²/κ_ice` is **~5.6× shorter** than the
  interior (front median ≈ 29 kyr vs ≈ 161 kyr) — the thin, fast-responding ice
  where a clock could pace the advance.

> **The driver code path is convergent, isotropic and advects the RTN=1 line [VERIFIED — synthetic].**
> The level-set advance law is already certified by `glmig_synthetic` (RESULT 16);
> `glaciers/validation/synthetic/rtn_intrusion_clock_synthetic.py` (RESULT 20,
> `glaciers/REPORT_RTN_INTRUSION_CLOCK.md`) hardens that at the level of the **real driver
> functions** `margin_field` / `amplification` (the ones `analyse` calls — glmig
> reimplements the gradient inline in 1-D and never runs them). On closed-form
> geometries the real `A = 1/|∇m|` is **machine-precision exact** on a planar
> margin, **second-order convergent** on a curved one (error ratio 3.99 ≈ 4 as `dx`
> halves), **isotropic** around a radial RTN=1 ring (axis-vs-diagonal median differs
> 7×10⁻⁵), and **recovers the planted front advance** (zero set of `m − ΔH`) to
> ≤1 cell; the advected object is verified to be the §H.1.1 RTN=1 line itself
> (`H* = H_flot/φ` exact; `margin<0 ⇔ RTN>1`, 0 mismatch). It is the code-path
> companion to RESULT 16, not a replacement, and leaves the hydraulic-pacing
> conjecture (#5) below untouched as `[HYP]`.

**The conjecture (#5, [HYP]).** Thinning sets only the *kinematic* ceiling; the
actual advance is likely **rate-limited by the subglacial hydraulic residence
time** (how fast the newly-exposed bed re-pressurises), which §G.4 ties to the
same `u_*`-controlled wall flux as the §G.2 scallop-migration √-law. This unifies
morphology (§G.2), intrusion (§G.3), and memory (§G.4) under one friction-velocity
closure — all fingerprinted by the parity break K-theory cannot see. **Falsifiable
test:** measure the real front-migration rate (repeat-pass radar / ITS_LIVE) and
compare to `v_front = (dH/dt)/|∇m|`. If observed ≈ kinematic, thinning paces it;
if observed ≪ kinematic, the hydraulic clock is rate-limiting — the interesting
case. Not resolved at 3 km here; stated as the next falsification, not verified.

**Closing the bridge — a constant-free measurement protocol.** Note the tagging:
the §G.2 `u_*` √-law is **[VERIFIED]**; what is **[HYP]** is the *transfer* of that
closure from scallop migration to grounding-line intrusion. The bridge flips to
[VERIFIED] with one ratio — define the **residence number**

    Ro = v_kin / v_obs ,   v_kin = (dH/dt)/|∇m|  (Bedmap2 + altimetry),

with `v_obs` the observed grounding-line migration (DInSAR / repeat radar). `Ro`
is dimensionless and constant-free — the same "reference cancels → constant-free
inverse" pattern as the §H.1.1 φ-inversion. Protocol:

1. **Targets.** `rtn_intrusion_clock.py` flags the runaway tail: **754 cells**
   with `A ≥ 0.42 km m⁻¹` within 50 km of the GL (median `A = 0.70 km m⁻¹`),
   concentrated on the West Antarctic ice plains (Siple Coast / Thwaites).
2. **Feasibility.** There `v_kin = A·(dH/dt) ≈ 0.7–1.4 km yr⁻¹` for
   `dH/dt = 1–2 m yr⁻¹` — well above DInSAR GL-detection precision (~tens of
   m yr⁻¹), so `Ro` is measurable, not marginal.
3. **Verdict.** `Ro ≈ 1` ⇒ thinning-paced (hydraulic clock fast, not limiting);
   `Ro ≫ 1` ⇒ hydraulic-limited, and `Ro` itself *is* the residence-time discount.
4. **√-law discriminant.** Across targets, regress `Ro` against an independent
   `u_*` proxy (basal melt / driving stress / ITS_LIVE sliding speed). If `Ro`
   tracks the **§G.2 scaling**, the same friction-velocity closure governs
   morphology (§G.2), intrusion (§G.3) and memory (§G.4) — the unification. A flat
   or scattered `Ro(u_*)` **falsifies** the transfer, leaving G.2 itself intact.

So G.2 is finished and used as a tool; this protocol tests only whether its `u_*`
law *also* paces the intrusion front — a single repeat-radar campaign over the 754
flagged cells settles it.

#### H.1.3 The bridge, run against real observations — `u_*` pacing **[HYP → disfavored at continental scale]**
 
We executed the §H.1.2 protocol on **open** data (no Earthdata auth), driver
`glaciers/validation/external/rtn_glmig_test.py`, figure `glaciers/validation/reports/rtn_glmig_test.png`:
 
| field | source | role |
|---|---|---|
| `v_obs` grounding-line migration (159,127 pts) | Konrad et al. 2018, *Nat. Geosci.* (CPOM portal) | observed front speed |
| `dH/dt` (10 km SEC, 2010–2016 trend) | Schröder et al. 2019, PANGAEA (CC-BY) | thinning |
| `|∇m|`, surface, `H` (Bedmap2, 2 km) | this repo's loader | kinematics + `u_*` proxy |
| `τ_d = ρ_i g H|∇s|` | derived | friction-velocity (`u_*`) proxy |
 
**Test 1 — kinematic sensitivity (constant-free, no `dH/dt`).** Konrad's headline
observation is ~**110 m of GL retreat per metre of thinning** (`v_obs/(dH/dt) ≈
0.11 km m⁻¹`). The level-set theory predicts this sensitivity *equals*
`A = 1/|∇m|`. Sampled at the GL points, `A` has median **0.035** and **p90 0.110
km m⁻¹** — i.e. the bare geometric kinematics reproduce the observed
retreat-per-thinning to a factor ~3 at the median and **exactly at the p90** (the
fast-stream tail Konrad's number describes). The level-set framing is sound.
 
**Test 2 — residence number `Ro = v_kin/v_obs` and the `u_*` discriminant**
(n = 7,064 GL points with measurable thinning):
- `Ro` median **1.58** (p25 0.70, p75 4.44) — **O(1)**. The front advance is, to
  leading order, **kinematic/thinning-paced**: thinning through the local margin
  geometry already accounts for the bulk of the observed migration.
- `Ro` vs `τ_d` (the `u_*` proxy): log–log slope **−0.11**, correlation
  **r = −0.06** — **flat**. The residual scatter in `Ro` shows **no
  friction-velocity organisation**.
 
**Verdict.** At continental, coarse scale the data **do not support — and mildly
disfavour — the conjecture that the §G.2 `u_*` √-law paces the grounding-line
front.** `Ro ≈ O(1)` says the front is essentially thinning-paced (no large hidden
hydraulic discount), and a `u_*`-paced residence clock would imprint a systematic
`Ro(τ_d)` slope that is absent (`r ≈ −0.06`). **§G.2 itself is untouched** — only
its *transfer* to the intrusion front is disfavoured. The conjecture's remaining
refuge is the **unresolved channel scale**: Bedmap2 (2 km) and SEC (10 km) cannot
see the ~1–10 m Röthlisberger-channel hydraulics where a `u_*` clock would live,
so this is a falsification of *large-scale* `u_*` pacing, not of channel-scale
control. Honest caveats: `τ_d` is a proxy (not measured basal melt); `dH/dt`
sampled at the nearest finite SEC cell (≤30 km) and `A` at the nearest grounded
cell (≤10 km), adding scatter; coastal altimetry is sparse exactly at the GL;
Konrad direction assumption (their option 3).
 
This is the framework's design working as intended: a clean, pre-registered test
(§H.1.2) returned a **falsifying** answer for the speculative bridge while the
verified pieces (level-set kinematics, RTN geometry, §G.2) stand.

> **Update (§H.1.4).** The continental-flat result above is *not* the whole story.
> It is a Simpson's-paradox artefact compounded by a poor proxy: `τ_d` is a
> driving-stress proxy that **anti-correlates with speed in streaming ice**, and
> the continental average mixes opposite-signed regimes. Disaggregating by basin
> and substituting a *direct* `u_*` proxy (ITS_LIVE surface speed) reverses the
> verdict in the marine West-Antarctic sectors. See §H.1.4.

#### H.1.4 The `u_*` discriminant is regime-dependent **[HYP supported in marine West Antarctica]**

Two robustness checks on §H.1.3, driver `glaciers/validation/external/rtn_glmig_basin.py`
(with `glaciers/validation/external/itslive_glmig_sample.py` for the direct proxy), figure
`glaciers/validation/reports/rtn_glmig_basin.png`:

1. **Per-basin split.** Bin the Konrad GL points into Antarctic sectors and re-run
   the discriminant per sector.
2. **Sharper `u_*` proxy.** Replace `τ_d = ρ_i g H|∇s|` with **measured ITS_LIVE
   240 m surface speed** — a direct friction-velocity proxy. (The two proxies
   agree only weakly, log–log `r = 0.23`, precisely because fast ice streams ride
   on *low* basal drag: low `τ_d`, high speed. `τ_d` is blind where `u_*` matters.)

Residence number `Ro = v_kin/v_obs` regressed (log–log) against each proxy; slope
`> 0` ⇒ `u_*`-paced rate-limiting. 95% bootstrap CIs in brackets (paired resample;
note: bootstrap ignores spatial autocorrelation, so effective `N` < nominal):

| region | `Ro` vs `τ_d` (driving stress) | `Ro` vs **ITS_LIVE speed** (direct `u_*`) | n |
|---|---|---|---|
| Amundsen (PIG/Thwaites/Smith) | `+0.82`, r `+0.43` | **`+0.57` [`+0.41`,`+0.76`], r `+0.33`** | 246 |
| Bellingshausen/WAIS-Pacific | `+0.05` (flat) | **`+0.49` [`+0.45`,`+0.53`], r `+0.37`** | 5 126 |
| Ross/Siple Coast | `−0.10` | `−0.03` (flat) | 376 |
| East Antarctica | `−0.41` | **`−0.30` [`−0.39`,`−0.20`], r `−0.19`** | 1 475 |
| **continental** | **`−0.11` [`−0.15`,`−0.06`], r `−0.06`** | **`+0.15` [`+0.12`,`+0.18`], r `+0.13`** | 7 064 |

**Findings.**
- **The proxy was the weak link.** Swapping `τ_d` for measured speed flips the
  *continental* slope from `−0.06` to **`+0.15`** (CI excludes 0).
- **The signal is regime-dependent and alive in the marine West.** Both marine,
  fast-streaming sectors — Amundsen *and* the large Bellingshausen/WAIS-Pacific bin
  that diluted the continental average — show a robust **positive** `Ro`–`u_*`
  organisation (`r ≈ 0.33–0.37`): faster ice ⇒ higher `Ro` ⇒ the observed front
  sits further below its kinematic ceiling, i.e. **`u_*`-dependent rate-limiting**,
  exactly the §G.2 transfer §H.1.2 set out to test. The cold, frozen-bed East
  Antarctic interior shows the **opposite** sign. Averaging opposite regimes through
  a blind proxy manufactured the spurious continental "flat."

**Revised verdict.** §H.1.3's "disfavoured at continental scale" is superseded:
the `u_*` → grounding-line transfer is **regime-dependent — supported in marine
West Antarctica (the sectors that govern sea-level), absent/reversed in the cold
interior.** §G.2 itself is untouched; what changes is that its *transfer* is no
longer falsified wholesale.

**One confound ruled out by sign.** `Ro ∝ 1/v_obs`, so any trivial coupling between
surface speed and the GL-migration speed `v_obs` would drag the slope **negative**;
the observed **positive** marine slope survives despite that downward pull (the
plausible mechanism is dynamic thinning: fast sliding → thinning → larger
`v_kin = A·(dH/dt)`).

**Honest caveats.** Correlations are modest (an *organisation*, not a deterministic
law); bootstrap CIs ignore spatial autocorrelation (true CIs wider); the Amundsen
box is nested inside the Bellingshausen box (not independent — but the disjoint
East sector confirms the sign reversal independently); coastal altimetry/velocity
is sparse at the GL.

**[DONE → ocean analysis — `validation/external/rtn_glmig_ocean.py`].** *Why* the
marine sectors carry a positive `u_*` residence signal is an ocean-coupled question,
now tested directly against **observed** ocean thermal forcing. TF =
`CT − CT_freezing(SA, p)` (TEOS-10) from the **Schmidtko et al. (2014)** Antarctic
shelf-**bottom**-water climatology — the seabed water that ventilates cavities and
reaches grounding lines (warm Circumpolar-Deep-Water shelves off Amundsen/
Bellingshausen reach TF ≈ 2.5–3.5 °C; cold Ross/Weddell/East shelves ≈ 0.4–0.6 °C).
Each Konrad GL point is assigned the nearest shelf-column TF (≤ 150 km), and `Ro`,
the thinning `dH/dt`, and `u_*` (ITS_LIVE speed) are regressed (log-linear) on TF
per sector. n = 7 958 GL points with a shelf match. (dH/dt here is the **ICESat-2
ATL15** 2019– trend, since PANGAEA/Schröder was down at run time; the *slope*
metric is invariant to a uniform `A`-scale and robust to the dH/dt source, so the
sectoral organisation is unaffected.)

**Result — the ocean organises the *velocity/residence* channel, not a local
melt→thinning law:**
- **`u_*` rises with TF in every sector** (significant except Ross): continental
  `+0.39`/°C (r +0.32, CI [+0.37, +0.41]); strongest in the marine West (Amundsen
  `+1.34`, Bellingshausen `+0.56`, r +0.54). Warmer shelf water ⇒ faster GL ice.
- **`Ro` rises with TF in the marine West**: Bellingshausen/WAIS-Pacific `+0.27`/°C
  (r +0.23, CI [+0.24, +0.30], n ≈ 6 000); Amundsen `+0.50` but **not significant**
  (its TF is uniformly high ≈ 2.5 °C, little within-sector range). The cold
  **East/Ross** sectors are flat/negative in `Ro` — the same regime contrast as the
  §H.1.4 `u_*` test.
- **Grounded thinning does *not* track local shelf TF**: flat/negative in the West
  (continental `−0.05`/°C; Amundsen `−0.74`), weakly positive in the cold interior.
  Grounded `dH/dt` is set by upstream dynamic drawdown and buttressing loss, not by a
  local cavity-melt law at the GL point.

**Revised verdict — [supported, qualified].** Ocean thermal forcing organises the
marine-West `u_*`–`Ro` residence signal **through the velocity channel** (warm shelf
⇒ fast ice ⇒ larger residence number), confirming the §H.1.4 signal is ocean-paced
in the sectors that govern sea level. It is **not** a simple local-melt→thinning
relation, and it is **regime-dependent** (absent/reversed in the cold East). **Honest
caveats:** an *organisation*, not a deterministic law (r ≈ 0.2–0.5); coastal proximity
is a partial confound (warm CDW and fast ice are both coastal); Schmidtko is a
1975–2012 climatology and ATL15 dH/dt is 2019– (TF treated as a slowly-varying
boundary condition); nearest-shelf TF is a source-water proxy (median match 22–91 km
by sector); bootstrap CIs ignore spatial autocorrelation (true CIs wider). Figure/JSON:
`validation/reports/rtn_glmig_ocean.{png,json}`.

#### H.1.5 The estimator behind §H.1.3/§H.1.4, calibrated on synthetic ground truth **[VERIFIED — synthetic]**

The §H.1.3/§H.1.4 real-data verdicts rest on two estimator assumptions that had
no synthetic null: (i) the level-set advance law `v_front = |dH/dt|/|∇m|`, and
(ii) that the `log Ro` vs `log u_*` slope faithfully reflects a hydraulic discount
rather than a regression/noise artefact. `glaciers/validation/synthetic/glmig_synthetic.py`
supplies the §V plant-and-recover calibration (`glaciers/REPORT_GLMIG_ESTIMATOR.md`,
RESULT 16; no data, no GPU):

- **Level-set law is exact.** A tracked analytic zero-contour reproduces
  `|dH/dt|/|∇m|` to rel-err `5×10⁻¹⁶` (1-D) and `4×10⁻¹⁴` (a tilted 2-D plane,
  using the gradient *magnitude*); the grid estimator `A=1/|∇m|` recovers `1/b`
  exactly. So §H.1.3's kinematic is the level-set identity, not an approximation.
- **The thinning-paced null is flat by construction.** With no discount
  (`D≡1`) `Ro≡1` to machine zero and the slope is `0.0` — the correct null behind
  the continental "flat slope ⇒ thinning-paced" reading.
- **A planted discount `D=1+γu_*^p` is recovered and sign-faithful.** Fitting
  `log(Ro−1)` recovers `p` to `7×10⁻¹⁶`; the raw `log Ro` slope (the §H.1.4
  statistic) is **positive and monotone in `p`** (`0.11, 0.22, 0.40` for
  `p=0.25,0.5,0.9`) — so a positive observed slope genuinely means `u_*`-paced.
- **Unbiased under noise, null under permutation.** 25 % multiplicative noise
  leaves the slope within `0.002` of noiseless; shuffling `u_*` collapses it to
  `≈0` — the marine-West positive slopes (§H.1.4) are not noise/ordering artefacts.

This is a **methods** result — it does not itself claim pacing (that is the
real-data §H.1.3/§H.1.4 question); it guarantees those slopes are read against the
right null with an unbiased, sign-faithful estimator, the same way
`rtn_synthetic.py`/`sliding_synthetic.py` calibrate the RTN classifier and lag
estimator.

#### H.1.6 Effective-pressure gating of the ocean→velocity coupling **[supported in the marine West — `rtn_ocean_efp_gate.py`]**

A new, falsifiable relationship that *unifies* the two halves of this section — the
RTN intrusion margin (§H.1.1–§H.1.2, *where* the ocean can reach the bed) and the
§H.1.4-ocean velocity coupling (*how hard* it then pushes). The normalized margin
`rel = m/H` (`m = H − H*`) is a dimensionless proxy for the **normalized effective
pressure**, since `N = ρ_i gH − p_w ≈ φρ_i g·m`, so `rel → 0` at flotation
(`N → 0`). A regularized-Coulomb / Budd sliding law makes the basal-speed elasticity
`s_N = ∂ln u_b/∂ln N < 0` *steepen* as `N → 0`; ocean melt lowers `N`; chaining,
`d ln u_*/dTF = s_N·(d ln N/dTF)` with `|s_N|` largest near flotation. **Prediction:
the ocean-thermal-forcing sensitivity of grounding-line speed is largest where the
bed is nearest flotation** — TF and a near-floating bed *multiply*. This is exactly
the `∂u_b/∂N` hydromechanical-state control the §G.4 lake-lag result pointed to, now
made quantitative and testable on open data (Bedmap2 `rel`, ITS_LIVE `u_*`,
Schmidtko TF; the same good-point set as §H.1.4-ocean).

**Result — supported in the marine West, absent in the cold interior.** Fitting
`ln u_* = a + b·TF + c·ln rel + d·(TF·ln rel)`:
- **Marine West (Amundsen + Bellingshausen, n = 5 964):** interaction `d = −0.100`
  (95% CI [−0.122, −0.078], excludes 0) — the TF–speed slope **steepens toward
  flotation**: `d ln u_*/dTF = +0.68`/°C near flotation (rel ≈ 0.15) vs `+0.44`/°C
  well-grounded (rel ≈ 1). The near-flotation tercile is the steepest
  (`+0.67`/°C, r +0.57). r² = 0.31.
- **Continental (n = 7 937):** `d = −0.004` (CI [−0.026, +0.018]) — **no gating**:
  the TF slope is flat across `rel` (≈ +0.41/°C).

The gating is **regime-specific** — it appears only where the ocean actually forces
the bed (the warm marine West), not in the cold-shelf interior where TF is low and
uniform — which strengthens the effective-pressure reading over a spurious global
artefact.

**Verdict — [supported, qualified].** Ocean thermal forcing accelerates grounding
lines preferentially where the bed is near flotation, in the marine West — a
predicted (and falsified-if-flat) consequence of effective-pressure-dependent
sliding, and a concrete unification of the RTN margin with the §H.1.4-ocean coupling.
**Honest caveats:** the robust evidence is the *continuous* interaction term — the
three-tercile pattern is not perfectly monotone (the mid-`rel` bin dips below the
high-`rel` bin); `rel` is a normalized-EP *proxy*, not measured `N`; `u_*` and `rel`
are themselves correlated (fast ice rides near flotation), so the interaction controls
the main effects but not every confound; r² ≈ 0.31 (an *organisation*, not a law); the
good-point set uses the ICESat-2 ATL15 dH/dt (PANGAEA/Schröder down at run time) and
the Schmidtko 1975–2012 climatology; bootstrap CIs ignore spatial autocorrelation.
Figure/JSON: `validation/reports/rtn_ocean_efp_gate.{png,json}`.

> **Update — a more *direct*, φ-free effective pressure confirms the gating
> (`efp_gate_direct_n.py`).** The honest caveat above is that `rel = m/H` is a
> *proxy* (with a free φ=0.9 and the H*=H_flot/φ adjustment) on Bedmap2. We removed
> both weaknesses by recomputing N from the ocean-connected flotation balance on the
> independent, higher-resolution **BedMachine Antarctica v3 (500 m)**:
> `N = ρ_i g (H − H_flot)`, `n̂ = 1 − H_flot/H` — the **physical normalized effective
> pressure, with no tunable φ** (median **N ≈ 0.31 MPa** at the GL good points). On
> the *identical* §H.1.6 good-point set (n=5964 marine-West), the committed `rel`
> proxy is reproduced **exactly** (`d = −0.100`, CI [−0.122, −0.078]) and the direct
> BedMachine N **also gates**: interaction `d = −0.035` (CI [−0.052, −0.019], excludes
> 0), with `d ln u_*/dTF` steepening from **+0.46/°C well-grounded → +0.62/°C near
> flotation** (monotone terciles). So the effective-pressure reading is **robust to a
> more-direct, φ-free, independent-dataset N** — the gating is not an artefact of the
> Bedmap2/φ proxy. The interaction is *weaker* with the direct N (−0.035 vs −0.100),
> i.e. the φ-proxy mildly over-states the gating, which is the honest direction. (Still
> a normalized-N from an ocean-connected-head assumption, not a borehole N; that
> remains the only fully-direct route.) Figure/JSON:
> `validation/reports/efp_gate_direct_n.{png,json}`.

### H.2 Non-local sliding law — kernel **[FALSIFIED as written]** on real thickness

> **Forecast.** Ice-stream surges lag basal-water perturbations by the §G.4
> memory timescale `τ_ice = H²/κ_ice`.

**Test run.** `τ_ice` evaluated on real Bedmap2 thickness at the **131
Siegfried & Fricker (2018) catalogued active lakes** (130 with valid thickness;
median `H = 2282 m`, range 637–3905 m):

- `τ_ice = H²/κ_ice`: **median ≈ 151,000 yr** (p5–p95: 23,000–323,000 yr).
- Observed post-drainage surge lags (Stearns et al. 2008; Siegfried et al. 2016;
  literature): **0.02–2 yr**.
- ⇒ the literal kernel is **~8×10⁴× too slow** at the median lake. The histogram
  of `log₁₀ τ_ice` (10⁴–10⁶ yr) is fully disjoint from the observed surge-lag
  band — a clean falsification of the kernel *as literally written*, on real
  geometry. This makes the §B.2 / §G.4 caveat quantitative: the full-thickness
  diffusive timescale is the wrong scale; only an **empirical** lag (or a kernel
  built on the thin thermal boundary layer, not `H`) is interpretable.

**Response data is real and open.** The matched lag test needs drainage *dates*
(volume-change time series), which sit behind a USAP-DC login / unreachable NSIDC
host — so we do **not** fabricate them. But the velocity half is openly
accessible: the ITS_LIVE datacube over lake **Mac1** yields a real surface-speed
series of **4505 image-pair measurements, 1987–2025, median 420 m/yr**, and the
`sliding_validator.estimate_lag` machinery runs on it directly (self-lag ≈ 0 as
expected). The only missing piece for the end-to-end §V.2 lag test is the gated
drainage-date catalogue.

**The empirical-lag strategy is kernel-shape-safe [VERIFIED — synthetic].** Since
the true kernel is hydromechanical and its *shape* stays `[HYP]`, the §V.2
programme measures only the empirical lag and keeps the kernel generic. That is
sound only if the estimator returns the same lag whatever the kernel shape —
now confirmed: `sliding_synthetic.run_kernel_shapes` plants one target lag with
five markedly different causal kernels (Gamma `k=2/4`, log-normal, bi-exponential,
symmetric raised-cosine) and `estimate_lag` recovers it to `≤10 %` for every
shape, with a delta-kernel control returning `0` (RESULT 17,
`glaciers/REPORT_SLIDING_KERNEL_SHAPE.md`). So a recovered lag is a property of the data,
not an artefact of the assumed Gamma form — the §G.4 analogue of the
`rtn_synthetic`/`glmig_synthetic` estimator calibrations.

**What the falsification buys us (mechanism). [HYP]** The mismatch is *diagnostic*,
not just "wrong number." (i) The thermal skin depth at observed surge periods is
`δ_skin = √(κ_ice P/π) ≈ 0.5–5 m` (`P` = 0.02–2 yr) — the perturbation never reaches
beyond a few metres of ice. (ii) A drainage is an *impulse*, whose semi-infinite
diffusive response is a monotone `t^{−1/2}` decay with **no peak**, yet observed
speed-ups **rise to a peak** at 0.02–2 yr. Both facts **exclude ice thermal
diffusion** as the lag-setting mechanism and relocate the memory into **subglacial
hydrology** (cavity/channel pressurization + evolution, effective-pressure `N`
control of sliding). So the falsified comparison tells us the lag is
**hydromechanical**: the §G.4 non-local term survives, but its dominant kernel is
the **hydraulic impedance** Green's function `K_hydraulic(τ)=(1/RC)e^{−τ/RC}`
(Mori–Zwanzig memory of the cavity↔channel storage–transport subsystem, with `φ` the
hydraulic potential, *not* the Leray pressure), with the ice-thermal `t^{−1/2}` tail
subdominant — the impedance-kernel refinement in §G.4. This is a hypothesis about
*where the memory lives* (and which kernel dominates), not a full nonlinear-kernel
derivation — though the lag *value* it implies is now **derived to order-of-magnitude**
(§G.4; `glaciers/validation/synthetic/hydraulic_lag_derivation.py`), landing in this 0.02–2 yr band.

**Falsification.** Already triggered for the literal `H²/κ` form. The *surviving*
empirical forecast — "surface velocity increases after lake drainage, with the
delay set by cavity/channel geometry and water flux (a hydraulic residence time),
only weakly by ice thickness" — can be tested once a vetted drainage-date catalogue
is paired with the (already-accessible) ITS_LIVE velocities. A clean falsifier of
the *hydromechanical* reading would be a measured response that is impulse-monotone
(no peak) and scales with `H²/κ_ice` rather than with hydrology.

> **Update — the matched test is now run co-temporally in the dense satellite era
> (§V.2d).** Two new runners pair the vetted catalogue with ITS_LIVE directly.
> (1) The 2003–2007 USAP-DC catalogue (`validation/external/lake_lag_itslive_match.py`)
> is **coverage-limited** — 7 events testable, 0 significant — because the active
> lakes sit in slow ice and the fast outlets only have dense ITS_LIVE after ≈2013
> (generalizing the §V.2b 5-lake null to the whole 58-event catalogue). (2) The
> **modern** runner (`validation/external/lake_lag_atl15_itslive.py`) closes the
> co-temporal gap with **ICESat-2 ATL15** quarterly surface-height change (Earthdata)
> for 2019–2026 drainage dates + dense 2019–2026 ITS_LIVE. Of **29** lakes that drain
> and **19** testable events, **18 show no surge** (amplitude bound **~3 %** of trunk
> speed) — so the hydromechanical surge is **not a universal** consequence of
> drainage — but **one Thwaites lake (`Thw_142`, drained 2021.75) steps +8.5 %**,
> peaking at **lag 1.1 yr** inside the derived 0.02–2 yr band, at **4.5σ** and
> surviving a pre-drainage secular-trend control: the **first co-temporal in-band
> field detection** of the §G.4 lag. **Net status:** the lag *value* stays
> `[DERIVED]` with its first field candidate; the *universal* surge form is
> **`[disfavoured]`** (a dynamically-primed bed is needed); the thermal `H²/κ` kernel
> stays `[FALSIFIED]`. The *selectivity* itself is informative — a response present at
> dynamically-active Thwaites but absent at stable MacAyeal/Byrd/Slessor outlets
> points to the lag being set by the **bed's hydromechanical state** (the local
> effective-pressure sensitivity `∂u_b/∂N`), not by a universal thermal or geometric
> clock.

> **Update — extending the probe to the downstream trunk turns the one Thwaites
> point into a 3-detection amplitude calibration (§G.4, `lake_lag_trunk.py`).** The
> §V.2d runner samples ITS_LIVE at the **lake centroid**, where the active lakes sit
> in thick, slow, high-`N` ice, so the sliding response is mostly invisible. This
> runner traces the ITS_LIVE 240 m flow field **downstream** of each draining lake
> and re-runs the identical secular-trend-controlled drainage-response test at fixed
> **5–25 km trunk samples** on the fast bed, where `N` is low and sliding is
> Coulomb/plastic. Across the **29** ATL15-era draining lakes (**99** testable
> sample-events) this yields **three** independent in-band **trunk** detections (vs
> the single centroid point): `Thw_142` (+2.4 % at 25 km, lag 1.38 yr, **11.6σ**
> detrended), `Thw_170` (+9.1 % at 15 km, lag 1.88 yr, 3.0σ), and `Rutford_1`
> (+21.7 % at 15 km, lag 0.13 yr, 2.9σ) — all inside the derived 0.02–2 yr band.
> **Amplitude calibration (the flagged-missing deliverable).** With a 3-lake
> population we can finally regress the surge amplitude `du/u` on the
> effective-pressure proxy `rel = m/H`: the amplitude **rises toward flotation**
> (semilog slope **−3.5 per unit `rel`**, `r = −0.78` — the largest surge, Rutford at
> 21.7 %, sits at the lowest `rel ≈ −0.08`, i.e. at/over flotation) and **grows with
> drained depth** (`log du/u` vs `log Δh` slope **+2.6**, `r = 0.85`), while trunk
> speed alone explains little (slope +0.7, `r = 0.20`). This is the §H.1.6 prediction
> — the sliding-law `N`-sensitivity `|s_N| = |∂ln u_b/∂ln N|` steepening as `N → 0` —
> now seen in the *field amplitude* of natural drainage steps, not only in the ocean
> gating. **Honest scope:** `n = 3` is a calibration, not a robust regression (the
> low-`N` end rests on Rutford alone, so the log–log `rel` fit is null and we report
> the semilog form), but the project's "not enough field detections to calibrate the
> amplitude" gap is now a fitted population with the predicted sign. Figure/JSON:
> `validation/reports/lake_lag_trunk.{png,json}`.

### H.3 Clock-mismatch (CMN) correction — **[VERIFIED in a synthetic K-theory solver; real-model test deferred]**

> **Forecast.** Adding `−CMN · ∇·(∂_t K_u ∇θ)` to an operational temperature
> solver reduces spurious temperature oscillations during transients (plumes,
> surges), and vanishes for steady turbulence (`∂_t K_u = 0`).

**Model test — now run here (RESULT 15).** `general_two_clocks/cmn_solver_demo.py`
(`general_two_clocks/figures/60_cmn_solver_demo.json`, `general_two_clocks/REPORT_CMN_SOLVER.md`) settles the
*model-side* claim in a self-contained 1-D transient K-theory thermal solver — no
external data, no GPU. The §G.5 mechanism is the two-clocks lag: the faithful
("truth") flux uses the **lagged** diffusivity `K_u(t−τ_c)`, the naive closure
freezes it at `K_u(t)`, and since `K_u(t)−τ_c∂_tK_u = K_u(t−τ_c)+O(τ_c²)` the
§G.5 term with `CMN=+τ_c` is exactly its first-order reconstruction. Advancing
truth / naive / corrected with an identical RK4 stepper over one `K_u` transient
cycle (so the common numerical error cancels and only the *clock* error remains):

- **Transient error cut ~15×** (`τ_c=0.05`: time-max rel-err `9.1×10⁻³` → `6.0×10⁻⁴`).
- **Leading-order error removed:** naive scales `∝τ_c¹` (log–log slope `1.03`),
  corrected `∝τ_c²` (slope `2.00`) — the correction is one order higher-accurate.
- **Steady-turbulence null is exact:** at `∂_tK_u=0` the term vanishes *and* the
  lagged/frozen clocks coincide, so naive ≡ corrected ≡ truth (error `0.0`).
- **The `+τ_c` sign is the error-reducing one:** `CMN=−τ_c` is *worse* than naive
  (`1.8×10⁻²`), so §G.5/RESULT-12's positive sign is the unique correcting choice.

So the commutator identity (`glaciers/validation/synthetic/cmn_synthetic.py`, rel-err
~1e-7) *and* its operational payoff (this result) are both verified; what stays
deferred is only the **real-model** test — the same term in ISSM/GlaDS on a real
surge/plume transient — now with a quantified expectation (`τ_c`-order reduction,
steady null, `+` sign).

**Falsification.** If a real K-theory thermal solver shows oscillations unchanged
(or worsened by the `+τ_c` term) during a transient, the spurious commutator
coupling is not the dominant transient error and §G.5 is demoted.

**Summary.** §H.1 is a [VERIFIED] *directional* result on real Bedmap2; §H.2
[FALSIFIES] the literal §G.4 kernel timescale on real lake thicknesses while
showing the response data is real and accessible; §H.3's model-side forecast is
now [VERIFIED] in a self-contained transient K-theory solver (RESULT 15), with the
real-solver (ISSM/GlaDS) test deferred. Precision is not claimed anywhere — the
framework is *exposed to falsification*, which is the point.

---

## §I — The `s_N(N)` master curve, its inversion, and an ungrounding early-warning signal

*(New; `glaciers/validation/synthetic/sn_master_curve.py`,
`glaciers/REPORT_SN_MASTER_CURVE.md`, `glaciers/tests/test_sn_master_curve.py`.
Analytic + a 1-D stochastic ODE; no GPU, no data download.)*

PR #1–#4 / §G.4 / §H.1.6 established **qualitatively** that lake-drainage steps and
ocean-forcing gating are two field probes of the same regularized-Coulomb (RC)
`s_N(N) = d ln u_b/d ln N`, both steepening toward flotation. §I makes that
quantitative and falsifiable.

### I.1 Closed-form master curve **[DERIVED, VERIFIED]**

Solving the RC law `τ_b = C N (u/(u+u0))^(1/m)` at `τ_b = τ_d` gives, exactly,

> `|s_N|(N) = m / ( 1 − (N_c/N)^m )`,  `N_c = τ_d/C`;  near flotation `≈ N_c/(N−N_c)`.

`|s_N| → m` far from flotation; a **simple pole at `N_c`**. Verified to
`1.4×10⁻⁴` vs the repo's numeric `type_iii_regime.s_N`. This **derives** the
near-flotation weakening that Joughin, Smith & Schoof (2019) impose by hand (an ad
hoc `h_af<h_T` ramp + fixed `u0=300 m/yr`). Convention caveat: with `u0=N^mΛ_o` the
well-grounded asymptote becomes `0`, but the `N_c` pole is convention-independent.

### I.2 Inversion method — measure `N_c` from drainage steps **[DERIVED + VERIFIED on synthetic; field application gated on dN]**

A population of drainage steps with known `N` and measured `du/u=|s_N(N)|f`
over-determines `(m, N_c)`. Plant→recover: **`N_c` recovers to 0.2 % / 0.4 % / 1.0 %
at 5 / 10 / 20 % amplitude noise**; `m` is degenerate with the per-event drop `f`
far from flotation (7–28 %). So the **flotation threshold is measurable, not tuned**;
`m` needs near-`N_c` sampling or a co-located hydrology `dN`. Independent
order-of-magnitude cross-check from the ocean-gating TF-slope curvature →
`N_c ≈ 0.036 MPa` (RC default 0.06 MPa).

### I.3 Critical-slowing-down early-warning signal for ungrounding **[DERIVED + demonstrated; field test HYP]**

Near flotation the Coulomb plateau makes basal drag velocity-insensitive: the
restoring **stiffness** `∂τ_b/∂u ∝ (1−R)²/R → 0` at the `N_c` fold (`R=(N_c/N)^m`),
so the restoring rate vanishes — **critical slowing down** (Scheffer 2009; Dakos
2008). An OU velocity perturbation under slowly declining `N` shows **rising variance
+ lag-1 autocorrelation** (Kendall-τ ≈ 0.54; equilibrium variance ↑ ≈ 1900× toward
`N_c`). **Forecast:** an ice stream approaching ungrounding should show rising
variance/AC1 **in its surface speed** before it goes afloat — a velocity-based MISI
early-warning, **distinct** from the Greenland surface-melt CSD of Boers & Rypdal
(2021) (different observable, mechanism, and threshold). Falsifier: a stream observed
to approach flotation with adequate sampling and **no** variance/AC1 rise.

### I.4 Open discriminator (flagged, not claimed): lag vs `N`

Field amplitude rises toward flotation (`corr(rel, ln du/u)=−0.78`, as predicted),
but the baseline cavity model predicts the discrete **lag** also rises toward
flotation while the **n=3** detections trend the opposite way (driven by the marginal
2.86σ Rutford point). The **sign of lag-vs-`N` is unresolved** — a clean discriminator
for the next batch of in-band detections, not a result.

### I.5 Tidal velocity admittance — a continuous third probe of `s_N(N)` **[DERIVED + VERIFIED on synthetic; field test HYP]**

*(`glaciers/validation/synthetic/tidal_admittance_probe.py`,
`glaciers/REPORT_TIDAL_ADMITTANCE.md`, `glaciers/tests/test_tidal_admittance_probe.py`.)*

Ocean tides modulate the grounding-zone effective pressure `N`, so the velocity
response is a **third, continuous, high-cadence** probe of the *same* `s_N(N)` that
the drainage-step (§G.4) and ocean-gating (§H.1.6) probes measure. Casting the
classic tidal-flow nonlinearity (Gudmundsson 2006/2007/2011; exponent ≈3; MSf
amplitude increases toward the GL, Minchew et al. 2017) through the RC `s_N(N)`:

* **Fundamental admittance = `|s_N|`** at the tidal frequency (→ `|s_N|` in the
  small-amplitude limit, verified to <3%).
* **Harmonic fingerprint:** `A2/A1 ≈ (ε/4)|s_N'/s_N − 1|`, `s_N'/s_N = −mR/(1−R)` →
  **diverges as `N → N_c`** (verified vs analytic to <8%; 2f/1f rises 1.3 %→10 %
  from well-grounded to near flotation). A sliding-law explanation for the observed
  toward-GL increase of the nonlinear MSf signal.
* **Tides-only inversion [new method of measurement]:** the fundamental admittance
  `|s_N|` + the 2f/1f ratio + the *known* tidal amplitude `ε` recover the Weertman
  exponent `m` (to ~0 %) and the **dimensionless flotation proximity**
  `R=(N_c/N)^m ∈ [0,1]` (`R→1` at ungrounding) to ~3 % **from surface velocity
  alone — no basal-pressure measurement** (directly answers Joughin 2019's
  "no reliable knowledge of basal water pressure").

Operationally this turns the §I.3 critical-slowing-down early-warning into a
*continuous* signal: monitor `R(t) → 1` (rising admittance + rising harmonics).
Honest scope: quasi-static tidal limit, the single-mechanism (RC-sliding) reading —
hydrology / GL-migration / margin-widening can also generate MSf (Rosier 2014/2015;
Robel 2017); the field test decomposes high-cadence GPS/InSAR admittance + harmonics
by `N`.

### I.6 SPATIAL early-warning — a single-snapshot ungrounding precursor **[DERIVED + VERIFIED on synthetic; field test HYP]**

*(`glaciers/validation/synthetic/spatial_ews.py`, `glaciers/tests/test_spatial_ews.py`.)*

The §I.3 temporal CSD and §I.4 tidal EWS both need a *time series*. The same `N_c`
stiffness collapse (`lambda(N) ∝ (1-R)^2/R -> 0`) has a **spatial** signature needing
only **one velocity snapshot**: along a flowline `N` falls toward the grounding line,
so a longitudinally-coupled velocity field has stationary
`Var(x) ∝ 1/sqrt(D*lambda(x))` and along-flow correlation length `xi(x) ∝ sqrt(D/lambda(x))`,
both **rising toward the GL**. Solved exactly via the Lyapunov equation for the
stationary covariance (no time-stepping): variance rises ×12 and correlation length
×4 toward the GL (Kendall tau 0.95 / 0.97); interior `Var*sqrt(lambda)` is constant to
CV=0.06, confirming the `1/sqrt(D*lambda)` law. **Field test:** bin ITS_LIVE speed
variance + along-flow correlation by distance-to-GL — single-snapshot, no time series.
Completes the early-warning toolkit: temporal (§I.3) + tidal-operational (§I.4) +
spatial (§I.6). `[cite]` Dakos et al. 2010 (spatial EWS); Schoof 2007 (MISI).

### I.7 Unifying §G.3 ocean intrusion (RTN) with the §I sliding divergence **[DERIVED + VERIFIED]**

*(`glaciers/validation/synthetic/rtn_sliding_unification.py`, `glaciers/tests/test_rtn_sliding_unification.py`.)*

In the single normalized effective pressure `n_hat = N/(ρ_i g H)` (= `1 − H_flot/H`, the
`rel`/`m-over-H` proxy), the §G.3 Regime Transition Number and the §I sliding-law
divergence are **the same N→0 condition**: `RTN = (1 − n_hat)/φ` (intrusion at
`n_hat < 1 − φ`), while `|s_N| = m/(1 − (N_c/N)^m)` diverges at the fold
`n_hat_c = N_c/(ρ_i g H)`. Since `n_hat_c ≈ 0.003–0.013 ≪ 1 − φ = 0.1`, the **RTN=1
intrusion line sits inland of the ungrounding fold for all H tested** — ocean intrusion
is an *upstream precursor* of the sliding-instability zone. Falsifiable spatial ordering
along a flowline: Type-II surge band opens → RTN=1 intrusion → sliding fold (Type III).
Committed ocean-gating terciles are consistent (near-flotation bins are RTN>1 with higher
TF-slope; well-grounded bins RTN<1). This partially closes the §G.3 "threshold magnitude"
HYP by tying it to the flotation/`N_c` threshold rather than leaving it free. `[cite]`
Schoof 2007 (MISI); Schoof 2005 / Joughin et al. 2019 (sliding law).

---

## §A.1 (closed) — the interface coupling number: ice as a frequency-dependent participant **[DERIVED + VERIFIED]**

*(`glaciers/validation/synthetic/interface_coupling_number.py`,
`glaciers/tests/test_interface_coupling_number.py`.)*

The §A.1 "coupling-surface" scaffolding is now a quantitative criterion. Writing the
ice flux as the §B.2 linear response `q_ice'(s)=H(s)v'(s)`, the interface velocity is
`v'=q_water'/(rho_i L + H(s))`, so the passive-BC limit is corrected by the
**interface coupling number** `Lambda(omega)=|H(i omega)|/(rho_i L)`:

* DC (slow forcing): `Lambda(0)=c_i|theta_far|/L = St <= 0.06` (ice participates at its
  latent-heat-limited Stefan weight);
* high frequency (`omega tau_d>>1`): `Lambda->0` (ice frozen, passive-adiabatic BC exact);
* crossover at the **ice clock** `tau_d=kappa/Vbar^2 ~ 10^3-10^4 yr`.

Because the observed surge band (0.02-2 yr) is far faster than `tau_d`, in-band
`Lambda < 5e-5 << St` (verified; frozen across a 2000-sample literature sweep): the
interface is a **passive BC to <1%** for the sliding-lag physics, and ice becomes a
participating medium (up to St<=6%) only for millennial forcing (long-term thinning).
Honest scope: `Lambda` bounds the *quantitative* correction; a small `Lambda` can still
flip *stability* (the §B.1 two-phase melt-amplitude stabilisation 0.41 vs 0.21), which
still needs the participating-ice treatment. Converts §A.1 from `[HYP]` scaffolding to a
frequency-resolved coupling criterion `Lambda(omega)=St*|hat H(omega tau_d)|`.

## §H.1.2 (closed) — the intrusion residence number Ro: thinning-paced vs hydraulic-limited **[DERIVED]**

*(`glaciers/validation/synthetic/intrusion_residence_number.py`,
`glaciers/tests/test_intrusion_residence_number.py`.)*

§H.1.2 verifies the *kinematics* of the RTN=1 intrusion front (level-set advance
`v_kin=(dH/dt)/|grad m| = A*dH/dt`, geometric amplification `A=1/|grad m|`) but left
the *pacing* as an unmeasured ratio `Ro=v_kin/v_obs` (>1 hydraulic-limited, ~1
thinning-paced) pending a DInSAR `v_obs`. This module **derives the predicted Ro from
physics** so a future `v_obs` only has to locate the system on the curve.

For the front to advance one smoothing length `ell~H` (~2 km), the newly-near-flotation
bed must re-pressurise to ocean head — the §G.4 hydraulic residence process with
timescale `tau_hyd` (the cavity<->channel band 0.01-2 yr). The maximum hydraulic
speed is `v_hyd=ell/tau_hyd`, so `v_obs=min(v_kin,v_hyd)` and

* `Ro_pred = v_kin*tau_hyd/ell`, with **regime boundary** the critical residence
  `tau_crit = ell/v_kin` (Ro=1 exactly at `tau_hyd=tau_crit`);
* implied hydraulic diffusivity `D_hyd=ell^2/tau_hyd ~ 0.06-12.7 m^2/s` across the band
  (distributed->channelized; Werder et al. 2013, Hewitt 2013).

**Result (runaway tail, A=0.70 km/m, dH/dt=1.5 m/yr):** `v_kin=1.05 km/yr`,
`tau_crit=1.9 yr`. Since ~98% of the §G.4 residence band lies *below* `tau_crit`,
intrusion is predicted **thinning-paced (Ro~1)** for the runaway cells, turning
hydraulic-limited only at the slow (~2 yr) residence end (or for faster thinning).
Falsifiable: a measured `v_obs` over the 754 runaway cells giving `Ro>>1` places
`tau_hyd>tau_crit` (hydraulic limitation); `Ro~1` confirms thinning-pacing. Converts
§H.1.2 from an unmeasured ratio into a derived, falsifiable regime prediction. No GPU,
no download.

## §G.6 (sharpened) — the local lee-flux law: what carries melt growth, and how it scales **[MEASURED]**

*(`glaciers/validation/synthetic/g6_local_flux_law.py`,
`glaciers/tests/test_g6_local_flux_law.py`; consumes the committed
`figures/59_scallop_amplitude_closure.json`.)*

§G.6's unified melt rate `v_melt=(1/ρ_iL)[h_local(u*,a/λ)(T_bulk−T_melt) − ∫K_ice q]`
posited a *mean*-conductance closure `δ_T,eff=δ_T,flat(1+ζ(a/λ)²)`.
`scallop_amplitude_closure.py` already falsified that — the mean-Nu deficit is
**amplitude-flat** (`D=0.108±0.031`, `p_free≈0`). But §G.6's growth term is the
**local** lee flux, not the mean, and the committed sweep recorded `R_max(a/λ)`
without ever fitting it. This module fits it (no re-run of the 535 s solver sweep):

* **Growth law:** the local peak lee flux rises **roughly linearly** with amplitude —
  linear `R²=0.945`, free exponent `p=0.69` (sub-quadratic) — from `R_max≈1.9`
  (`a/λ=0.05`) to `≈4.3` (`a/λ=0.30`). The §G.6 `(a/λ)²` form is **rejected for the
  local term too** (quadratic `R²=0.40 ≪ 0.94`). The origin-respecting,
  steepness-proportional law `R_max≈1+1.7·(2π a/λ)` (correct flat-wall limit
  `R_max→1`) holds to `R²=0.92`; seed-robust (`R_max=1.89±0.03` to `3.98±0.22`).
* **Mean saturation:** `R_mean` rises then saturates (`≈1.28`) — the conduction-limited
  mean cannot escape ~flat, confirming the local/mean split.
* **Separation onset:** `R_min` turns **negative at `a/λ≈0.11`** — reversed,
  recirculating lee flux that the amplitude-flat mean hides.

Honest §G.6 closure: **mean conductance amplitude-flat (`C≈1.11`), melt growth carried
by a linear-in-steepness, bounded local lee flux `R_max∈[1.9,4.3]`, with lee-separation
onset near `a/λ≈0.11`** — phenomenology with measured, bounded coefficients rather than
a quadratic ansatz. For `RTN>1` (ocean intrusion, §G.3) this Type-I expression is
replaced by the ocean-controlled branch. No GPU, no download.

## §A.3 (closed) — the dimensional bridge: derived channel magnitudes, one bounded calibration **[DERIVED + calibration isolated]**

*(`glaciers/validation/synthetic/a3_dimensional_bridge.py`,
`glaciers/tests/test_a3_dimensional_bridge.py`.)*

§A.3/§D.1 verified the *direction* of scallop→channel feedback
(`V_scallop/V_o=+0.33`, phase-lock `R_phase=0.95`, site-selection `R_winner=1.00`,
`g`-robust). The single residual was the **dimensional bridge** — "normalised sizes
→ physical radii via `ρ_iL` and the calibrated gain `g`." This module performs it and
shows **most of it is derivable**:

* **Absolute magnitudes DERIVED (no free fudge).** With `ρ_iL=3.0e8 J/m³`, Glen `A`,
  effective pressure `N`, and literature subglacial inputs (`Q`, `∂φ/∂s`), the steady
  Röthlisberger channel `S*=V_o/k_creep`, `R*=√(2S*/π)`, `τ=1/k_creep` is
  **metre-scale** (central `R*≈2.4 m`, band median ~13 m; 73% of the literature box
  gives 0.1–50 m) with **sub-annual-to-annual** adjustment (central `τ≈0.18 yr`). The
  wide upper tail is the **low-N near-flotation limit** (`k_creep∝N³→0`) — the same
  §G.3/§I flotation fold reappearing in the hydrology.
* **`ρ_iL` CANCELS in the scallop fraction.** `ΔS/S = V_scallop/V_o = 0.33` maps
  straight through, so a scalloped reach grows channels **+33% in area (+15% radius)**
  over a smooth reach — a **calibration-free** prediction.
* **`g` is the one true calibration knob.** The concentration gain enters only the
  network competition `V_o_eff=⟨V_o⟩(1+g(S−⟨S⟩)/⟨S⟩)` — it sets *drainage capture*,
  not local size. Direction is already `g`-invariant; `g∈[0.1,0.9]` bounds the
  competitive margin, pinned by observed channel spacing/size or tracer/borehole
  drainage timing — not by code.

§A.3 moves from "[HYP] dimensional bridge" to **[DERIVED magnitudes over a literature
band] + [calibration-free +33% scallop fraction] + [one bounded, direction-invariant
gain `g`]**. No GPU, no download.

## §A.2 (sharpened) — scallop roughness z_0(λ,a): max tractable theory + the one field point **[DERIVED bound; field point flagged]**

*(`glaciers/validation/synthetic/a2_z0_roughness.py`,
`glaciers/tests/test_a2_z0_roughness.py`.)*

§A.2 closes the loop `dφ/ds→Q→u*→λ(Curl)→z_0(λ,a)→C_d→dφ/ds`. Two legs are settled —
**wavelength** (`λ=Re*ν/u*`, `Re*≈2200`, Curl 1966 [VERIFIED/LIT]) and the **amplitude
direction** ([MEASURED] amplitude-independent, §G.6/figures/59, so `z_0` is
geometry-set not steepness-set). The lone residual is the **wavelength prefactor**
`z_0=c_z·a` (`c_z=α_s/30`, ~10× uncertain over `α_s∈[0.3,3]`), which leans on Nikuradse
until a real scallop train pins it. Rather than fake that field point, this module
answers the tractable theory question — **how much does the 10× prefactor uncertainty
actually matter?**

* **Log-law buffering [DERIVED].** At the Curl anchor (`u*=0.05 m/s`→`λ≈7.9 cm`,
  `a≈7.9 mm`) with `H≈2.4 m` (§A.3), `z_0∈[7.9e-5, 7.9e-4] m` (10× span) gives
  `C_d=[κ/ln(H/z_0)]²∈[1.6e-3, 2.6e-3]` — only **~1.7×**, and the same factor in
  `dφ/ds=ρ_w C_d u²/H`. The log law **compresses the 10× roughness uncertainty ~6×**,
  valid wherever `H/z_0≫1` (here `~10³–10⁴`). So the missing field point is the
  framework's **least damaging** open closure.
* **The one field point (flagged, not faked).** Pinning `c_z` needs a single scallop
  train with BOTH geometry `(λ,a)` AND an independent drag measure on the same train
  (near-wall velocity profile→`z_0`, or measured `u*`+`dφ/ds`). Candidates: Curl 1966 /
  Blumberg & Curl 1974 (limestone & ice scallops); cave-scallop morphometry;
  subglacial-conduit dye-trace+pressure. None exists in-repo — no faked field
  verification. The amplitude leg being settled means the point only has to pin the
  *geometry* prefactor, not an amplitude law.

§A.2 moves from "[HYP, leans on LIT]" to **[two legs settled] + [the open prefactor
bounded and shown to be log-law-buffered to ~1.7×] + [one field point precisely
specified, honestly absent]**. No GPU, no download.

## §H.3 (extended) — CMN correction survives a 2-D advection + moving-plume reduced-model proxy **[VERIFIED in proxy; real ISSM/GlaDS test still deferred]**

*(`glaciers/validation/synthetic/h3_cmn_reduced_model.py`,
`glaciers/tests/test_h3_cmn_reduced_model.py`.)*

RESULT 15 verified the §G.5/§H.3 correction `−CMN·∇·(∂_tK_u∇θ)` (`CMN=+τ_c`) in a
**1-D scalar** solver with one prescribed `K_u(t)` cycle. The deferred piece is the
**real-model** test (ISSM/GlaDS on a real surge/plume), which needs heavy external
solvers unavailable here. **Honest scope:** this module is **not** that test — no ice
dynamics, no real turbulence closure, no real geometry. It is the standalone
intermediate: a **2-D advection-diffusion** solver with a Gaussian plume whose
amplitude ramps and whose **centre propagates** across the domain — the closest
self-contained analogue to a moving plume/surge transient. truth/naive/corrected share
the RK4 stepper and advection; only the diffusion clock differs
(`K(t−τ_c)` / `K(t)` / `K(t)−τ_c∂_tK`), so only the clock error survives.

All four RESULT-15 conclusions carry over to 2-D + advection + spatial structure:
* **transient error cut ~9×** (`τ_c=0.02`: `9.9e-3 → 1.1e-3`);
* **order lifted:** naive `∝τ_c^0.99`, corrected `∝τ_c^2.02` (one order higher-accurate);
* **steady null exact:** no event ⇒ naive≡corrected≡truth to machine zero (`0.0`);
* **`+τ_c` is the unique corrector:** `−τ_c` (err `2.0e-2`) is worse than naive (`9.9e-3`).

This raises confidence that the real-solver (ISSM/GlaDS) test would behave as forecast
— but it **does not replace it**; the real-model test remains the honest open item. No
GPU, no download.

## Real-data lake-lag extension — the §I framework on real CryoSat-2 + ITS_LIVE data **[REAL DATA; cross-framework check]**

*(`glaciers/validation/external/lake_lag_sn_ews.py`,
`glaciers/tests/test_lake_lag_sn_ews.py`; re-analyses the committed
`external/data/lake_lag_matched.json` — no new download.)*

The §H.2 population test already ran (`lake_lag_atl15_itslive.py`): 1 in-band surge /
19 testable lakes (Thw_142), not universal; the full 131-lake catalogue stays
USAP-DC/ATL15-gated (a ~7.6 GB re-download that would mostly reproduce that result).
What was never done is to read the **real** lake velocity series through the **§I
framework built this run**. This module applies two §I probes to the 3 marquee
MacAyeal lakes with OPEN ITS_LIVE velocity + OPEN CryoSat-2 drainage:

* **Drainage-response (§I.1/§I.2).** Across the 3 resolved real drainage events,
  `|Δv/v| ≤ 2%`, mixed sign, at/near the ~1% year-to-year noise floor (with a *positive*
  >2σ test for a genuine §G.4 surge) → **0 surge detections**. Via the master curve
  `Δv/v≈|s_N|·(ΔN/N)`, the bounded response places these trunk lakes **far from the
  `N_c` flotation pole**.
* **Critical-slowing-down EWS (§I.3/§I.6).** No lake shows the *joint* signature
  (rising variance AND high lag-1 AC together) → **0 precursors**. The §I early-warning
  correctly returns a **true negative** on stable trunk ice (unit tests confirm the
  detectors *do* fire on planted surges/CSD, so the null is real, not a dead test).

Both §I probes agree: these MacAyeal lakes are far from ungrounding — consistent with
the 1/19 population non-detection. **Honest limits:** n=8 annual points per lake is weak
for a CSD trend (2 of 5 events lack pre-drainage coverage); a strong test needs the
dense quarterly series and a lake actually approaching flotation, and the full
131-lake/ATL15 population extension remains gated. A genuinely-new cross-framework
result on real data, with no faked download.
