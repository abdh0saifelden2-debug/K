# What single-eddy-diffusivity closure discards: a Mori–Zwanzig audit, a scalar-independent memory time, and a divergence-cleaning design window

---

## Abstract

Eddy-diffusivity ("K-theory") closures represent all unresolved turbulent transport
with a single scalar diffusivity. It is established that such closures arise as the
Markovian (short-memory) limit of a Mori–Zwanzig (MZ) reduction of the resolved dynamics
(Chorin–Hald–Kupferman; Stinis; Parish & Duraisamy 2017), and that the discarded
backscatter can be reinstated with a divergence-free, dissipation-tied stochastic force
(Leith 1990; Mason & Thomson 1992; Schumann 1995). Building on these results, this paper
contributes an operator-by-operator **audit** of exactly what a single diffusivity throws
away, and two concrete, falsifiable measurements. The audit names four simultaneous
simplifications hidden in `ν_t`: (i) the MZ memory kernel collapsed to a delta in time,
(ii) the spectral eddy viscosity frozen constant in wavenumber, (iii) the
fluctuation–dissipation-linked orthogonal force deleted, and (iv) the sign restriction
`ν_t≥0` that forbids backscatter — each mapped to a specific term of the generalized
Langevin equation. Assembling the known repairs (a scale-selective spectral eddy
viscosity, a Leray-projected FDT-tied stochastic force, and an exact Leray projection)
into one closure form, we run an a-priori, frozen-field benchmark in a 256² periodic box
(sharp-spectral filter at `k_c=32`). The **structural** result is *predicted, not fitted*:
the projected force stays divergence-free to `RMS(∇·m)≈1.7×10⁻¹⁴` and carries the
correct sign of energy backscatter, whereas a spectrum-matched random surrogate
reproduces the force spectrum yet breaks incompressibility (`RMS(∇·m)≈12`) and randomises
the transfer direction, and Smagorinsky is purely dissipative (`T(k)≤0` everywhere). The
transfer-magnitude match to correlation **1.000** is a *consistency check on a fitted
quantity* — the backscatter amplitude `Θ(k)` is set to the measured net transfer — and is
reported as such, not as a prediction. Two measurements are genuinely new. First, the
closure's memory time is **scalar-independent**: `τ_c(salt)/τ_c(heat)≈1` across a 100×
molecular-Lewis contrast even though the scalar Nusselt numbers track `Le` strongly — the
memory belongs to the *flow*, as the MZ orthogonal dynamics require — and the implied
clock-mismatch correction cuts a transient K-theory scalar-solver error ~15× while
vanishing identically in steady turbulence. Second, when exact projection is replaced by
hyperbolic (Dedner) divergence cleaning, the constraint residual collapses only inside a
**finite design window** `2≲γτ≲12` and *re-grows* beyond it: pushing the cleaning rate up
to clean fast locally stalls the global low-wavenumber modes through the slow telegrapher
root — an over-cleaning failure absent from the original cleaning analysis. The same
a-priori benchmark, repeated in genuine **3-D** (128³ forced DNS, where vortex stretching
is active), reproduces all three structural verdicts and exposes a failure 2-D cannot see:
the true subgrid flux is up-gradient (backscatter) over **~48 %** of space, which a
positive-definite eddy viscosity represents at **0 %** by construction. Scope is
explicit: a-priori (frozen-field), periodic, in 2-D and 3-D; the clean FDT/projection
commutation is special to the spectral box and does not survive solid walls. A time-integrated **a-posteriori** test of a
predictive version of the closure (§5b) confirms every variant is stable, that Smagorinsky
is markedly over-dissipative, and that the FDT backscatter helps in the predicted direction
— while honestly bounding the benefit: in pure 2-D no eddy-viscosity closure beats no model
on the resolved spectrum, because the resolved scales need near-zero net subgrid dissipation.
The contribution is the audit, the scalar-independent memory time, the cleaning window, and
the a-posteriori stability/over-dissipation bound — not a universal closure.

---

## 1. Introduction

### 1.1 The problem with one diffusivity

Reynolds-averaged and large-eddy closures of geophysical and engineering turbulence
overwhelmingly rest on the Boussinesq eddy-viscosity hypothesis and its Smagorinsky
realisation (Boussinesq 1877; Smagorinsky 1963; Monin–Obukhov): the unresolved
flux is modelled as down-gradient transport with a single positive scalar diffusivity
`K` (optionally a turbulent Prandtl number relating momentum and scalar transport).
This object is cheap and local, and it works where turbulence is near-equilibrium and
the cascade is forward. It fails — systematically and in the same way — wherever the
flow carries long-lived coherent structure, an inverse cascade, or a sharp
pressure/strain geometry: separated wakes, rotating and stratified flows, 2-D and
quasi-2-D turbulence, and boundary-layer transition.

The usual response is to re-fit `K` (dynamic Smagorinsky, stability functions,
wall models). We argue the deficiency cannot be repaired by re-fitting because it is
**structural**: pressure and the transported scalars obey *different operators*, and a
single diffusivity acting on both is blind to the distinction. We make this precise.

### 1.2 The two clocks

Temperature (and any passive/active scalar) obeys an advection–diffusion (parabolic)
equation: local, memory-bearing, torn into filaments by local strain. Pressure in an
incompressible flow obeys a Poisson (elliptic) constraint — the Leray projector
enforcing `∇·u=0` — resolved instantaneously over the whole domain and its boundaries.
A single local diffusivity acting on both is structurally blind to this
elliptic/parabolic split. This "two clocks" reading motivates the operator audit below;
the quantitative content of the paper is the audit (§3), the a-priori structural
benchmark (§5), the scalar-independent memory time (§6), and the divergence-cleaning
window (§7).

### 1.3 Contributions

1. An **operator-by-operator audit** identifying the four simultaneous simplifications by
   which single-eddy-diffusivity closure reduces the MZ generalized Langevin equation,
   each mapped to a named GLE term (§3). The MZ→eddy-viscosity connection itself is
   established (Parish & Duraisamy 2017); the contribution is the explicit term-by-term
   accounting.
2. A **structural a-priori demonstration** that an MZ/FDT/Leray-projected closure form,
   assembled from established components, preserves incompressibility and the backscatter
   sign where a spectrum-matched surrogate fails — the *predicted*, non-fitted
   discriminator being solenoidality, not the (partly fitted) transfer magnitude (§4–§5).
3. The closure's **memory time `τ_c` measured and shown scalar-independent** (heat vs salt,
   100× Lewis), with a sign-pinned clock-mismatch correction verified in 1-D and a 2-D
   advection proxy (§6).
4. A **divergence-cleaning design window** `2≲γτ≲12` with an over-cleaning stall and a
   measured non-local divergence amplification `A≈3.9×`, quantifying how exactly the
   constraint must be enforced in codes lacking exact projection (§7).
5. An **a-posteriori, time-integrated test** of a *predictive* version of the closure
   (§5b): every variant is stable over ~24 eddy-turnovers, Smagorinsky is confirmed
   markedly over-dissipative, the FDT backscatter helps in the predicted direction, and
   the honest 2-D ceiling — no eddy-viscosity closure beats no-model on the resolved
   spectrum — is established and explained.

---

## 2. Governing system and the closure problem

Filter the incompressible NS equations at scale `Δ` (overbar = resolved). With the
Leray projector `ℙ = I − ∇(∇²)⁻¹∇·` and the subgrid force `m = −∇·τ`,
`τ_ij = (overline{u_i u_j} − ū_i ū_j)`,

> `∂ₜū = ℙ[ −(ū·∇)ū + ν∇²ū + m ]`, `∇·ū = 0`. (1)

The closure problem is to model `m` from resolved data. K-theory sets

> `m = ∇·(2 ν_t S̄)`, `ν_t = (Cₛ Δ)²|S̄|`, `S̄ = ½(∇ū + ∇ūᵀ)`. (2)

A single scalar field `ν_t`. **That one object hides four assumptions, each a clock
being crushed:**

1. **Local in space** — `m` depends only on `∇ū` at a point; denies the elliptic,
   non-local pressure clock (`∇²p=f` couples the whole domain).
2. **Instantaneous in time** — no memory; denies that the fast clock carries history.
3. **Down-gradient only** (`ν_t≥0`, purely dissipative) — forbids backscatter. In 2-D
   the inverse energy cascade *is* net up-scale transfer, so this is not a small error.
4. **No structural re-imposition** — the closure acts on `ū` and leaves
   incompressibility entirely to the pressure solve, rather than being
   divergence-consistent itself.

---

## 3. The audit: single-K as the Markovian collapse of the MZ kernel

That eddy viscosity is the Markovian (short-memory) limit of an MZ/optimal-prediction
reduction is established (Chorin–Hald–Kupferman 2000, 2002; Stinis 2015; Parish &
Duraisamy 2017). We restate it only to make the discarded terms explicit, one by one.

Write the semi-discrete NS system as `u̇=R(u)`; let the Koopman/Liouville generator `L`
act on observables, `e^{tL}g(u₀)=g(u(t))`. Choose a projection `P` onto the resolved
variables `â=Pu` (Mori's linear projection, or Chorin–Hald–Kupferman optimal-prediction
conditional expectation), `Q=I−P`. The Dyson identity applied to the resolved dynamics
gives the exact **generalized Langevin equation**:

> `∂ₜâ(t) = PLâ(t) − ∫₀ᵗ K(t−s) â(s) ds + f(t)`, (3)
> `K(τ) = −P L e^{τQL} Q L` (memory kernel), `f(t)=e^{tQL}QLâ(0)` (orthogonal force).

The three terms map one-to-one onto the two clocks, made exact: `PLâ` is the slow,
resolved Markov drift; `−∫K â ds` is the running memory of the unresolved fast clock —
**the term K-theory does not have**; `f(t)` is the honest origin of the "stochastic"
term in stochastic-PDE closures.

**Second fluctuation–dissipation theorem.** Under stationarity and the inner product
`(·,·)` that defines `P`, kernel and noise are two faces of one object:

> `K(τ) = (f(τ),f(0))·(â,â)⁻¹`. (4)

In the periodic spectral box the inner product is the kinetic-energy (`L²`) norm and
`P` is the sharp spectral truncation; both `P` and the Leray projector
`ℙ_k=I−kkᵀ/|k|²` are diagonal in Fourier space, hence commute and are self-adjoint, so
(4) is a relation between well-defined solenoidal correlations. (This clean commutation
is special to the periodic spectral setting; solid walls break the shared eigenbasis —
see §7 and the Scope.)

**The collapse.** If `K` decays fast relative to the evolution of `â` (short-memory /
t-model limit), the convolution localises:

> `∫₀ᵗ K(t−s) â(s) ds ⟶ (∫₀^∞ K(τ)dτ) â(t) ≡ Γ â(t)`. (5)

Taking `Γ` diagonal in wavenumber with `Γ(k)=ν_t|k|²` and **dropping `f`** reduces (3) to

> `∂ₜû(k) = ℙ_k N̂(û)(k) − ν_t|k|² û(k)`, (6)

which is exactly Smagorinsky K-theory (2) in spectral form. Therefore:

> **K-theory is the GLE with (i) the memory kernel replaced by a delta in time,
> (ii) `ν_t` taken constant in `k`, (iii) the FDT-linked noise deleted — plus the
> down-gradient restriction `ν_t≥0` that forbids the backscatter part of the true
> transfer.**

This is the precise statement of why single-K is blind to the fast clock, and which
terms a structurally-consistent closure must restore.

---

## 4. Assembling a structurally-consistent closure form

Each discarded term has an established repair; we assemble them so each named
simplification in §3 is undone by a named, established component (spectral eddy
viscosity: Kraichnan 1976; Chollet & Lesieur 1981; FDT-tied divergence-free backscatter:
Leith 1990; Mason & Thomson 1992; Schumann 1995). Keep the structure of (3) but make each
term computable. In the periodic spectral box:

> `∂ₜû(k,t) = ℙ_k N̂(û)(k,t) − ν_t(k,t)|k|² û(k,t) + f̂(k,t)`. (7)

**(a) Nonlocal (spectral) eddy viscosity `ν_t(k,t)`** — the plateau–cusp spectral eddy
viscosity read off the resolved energy at the cutoff (Kraichnan 1976; Chollet &
Lesieur 1981):

> `ν_t(k) = ν_t^∞ · √(E(k_c)/k_c) · [1 + c(k/k_c)^p]`, (8)

flat at low `k`, rising into a cusp near `k_c`. Scale-selective dissipation instead of
one number. (In 2-D the low-`k` plateau can be *negative* — the inverse cascade — which
is exactly why a single positive `ν_t` is most wrong there.)

**(b) Projected, FDT-linked backscatter `f̂`** — zero-mean, divergence-free by
construction (`ℙ_k f̂=f̂`; in 2-D `f̂=ik^⊥ψ̂`, a random streamfunction), with covariance
tied to the *same* operator that dissipates, via the discrete 2nd FDT:

> `⟨f̂_i(k,t)f̂_j*(k,s)⟩ = 2 ν_t^B(k,t)|k|² Θ(k) ℙ_ij(k) δ(t−s)`, (9)

with `ν_t^B(k)` the backscatter (energy-injecting) part of the transfer and `Θ(k)` set
so the *net* transfer equals the true subgrid transfer. Forward and backward channels
are two signs of one FDT-consistent operator, not a damping plus an unrelated kick
(Leith 1990; Mason & Thomson 1992; Frederiksen & Davies).

**(c) Leray projection wraps everything**, so (7) is incompressible *exactly*; the fast
clock is represented as the **operator** `ℙ`, never approximated by noise.

**Memory generalization.** Dropping the Markovian collapse,
`ν_t(k,t)|k|²û ⟶ ∫₀ᵗ K(k,t−s)û(k,s)ds`, restores time non-locality; (7) is its
short-memory limit. Keeping `ν_t(k)` + projected FDT noise while collapsing the kernel
is a *Markovianized representation of a genuinely non-Markovian process*: it reproduces
the statistical structure of the fast clock (scale-selective net transfer, exact
solenoidality) but assumes the unresolved scales clear memory instantly relative to the
slow clock. The structural constraints (`ℙ`, FDT) are exact; the time-locality is the
approximation at risk, and §5's a-priori test is designed to isolate structure from
time-locality.

**Limits and invariances.** The model satisfies Galilean invariance, exact
incompressibility, a closed energy budget (FDT fixes mean transfer; no spurious
injection), correct net-transfer sign (forward in 3-D, inverse in 2-D), and two clean
limits: `Δ→0 ⇒ (ν_t→0, f̂→0, K→δ)` recovers molecular NS, and the
local/Markovian/no-noise limit (5)–(6) recovers Smagorinsky. It is a strict
generalization, not a rival model. **No flawlessness is claimed**: `K(k,τ)` (or
`ν_t(k), Θ(k)`) still needs modelling choices; what is *guaranteed* is exact
satisfaction of the structural constraints and removal of the specific
energy-without-structure failure mode.

---

## 5. An a-priori structural benchmark

**Setup.** 256² 2-D incompressible DNS (vorticity–streamfunction, forced inverse
cascade), evolved to a developed spectrum, sharp-spectral-filtered at `k_c=32` to give
resolved `ū` and the *exact* subgrid force `m_true = ℙ[N(u)‾ − N(ū)]`. Three models of
`m` are compared: (i) Smagorinsky (2); (ii) a spectrum-matched random field with the
same `E_m(k)` as `m_true` (the phase-randomised surrogate); (iii) the projected-FDT
model (7)–(9). Three diagnostics: the force spectrum `E_m(k)`, the RMS divergence of
the closed field, and the energy-transfer spectrum `T(k)=Re⟨û*·m̂⟩` (resolving forward
dissipation `T<0` from backscatter `T>0`).

**Results (re-run at draft time; Figs. 1–3).**

![Subgrid-force spectrum on the 256² frozen-field benchmark (sharp-spectral filter at cutoff `k_c=32`). The projected-FDT closure (this work) reproduces the exact subgrid-force spectrum, while Smagorinsky and a spectrum-matched surrogate do not.](figures/28_closure_force_spectrum.png)

![Solenoidality of the closed subgrid force, the RMS of its divergence. The projected-FDT model stays divergence-free to about `10⁻¹⁴`, whereas the spectrum-matched surrogate breaks incompressibility by about `10¹`.](figures/29_closure_divergence.png)

![Energy-transfer spectrum `T(k)`: negative is forward dissipation, positive is backscatter. The projected-FDT closure matches the exact transfer including the sign of backscatter (correlation 1.000); Smagorinsky is purely dissipative (correlation 0.071).](figures/30_closure_transfer.png)

| diagnostic | truth | Smagorinsky | spectrum-matched surrogate | projected-FDT |
|---|---|---|---|---|
| `RMS(∇·m)` (rel.) | `7.41×10⁻¹⁴` | `1.77×10⁻¹⁴` | **`1.20×10⁺¹`** | `1.73×10⁻¹⁴` |
| transfer corr. with truth | — | **`0.071`** | `0.907` | **`1.000`** |
| `T(k)≤0` everywhere? | no (has backscatter) | **yes (no backscatter)** | no (randomised sign) | no (correct partition) |

The three diagnostics confirm the §4 prediction exactly:

1. **Smagorinsky** captures gross dissipation but is purely dissipative — `T(k)≤0`
   everywhere (no backscatter) — and mis-sets the near-cutoff cusp. Transfer
   correlation `0.071`.
2. **The spectrum-matched surrogate** matches `E_m(k)` by construction yet **fails
   incompressibility** (`RMS(∇·m)≈12`, twelve orders above truth) and randomises the
   transfer direction — "energy yes, structure no."
3. **The projected-FDT model** stays solenoidal to `1.7×10⁻¹⁴` and reproduces the
   *sign* of the forward/backscatter partition — the predicted, non-fitted structural
   result. Its transfer-magnitude correlation `1.000` is a consistency check on a fitted
   quantity (`Θ(k)` is set to the measured net transfer), not an independent prediction.

The predicted, non-fitted discriminator is therefore **structural** — solenoidality and
the backscatter sign — separating a closure that respects incompressibility from a
spectrum-matched surrogate that does not; the transfer magnitude is fitted and reported
only as a consistency check.

**Scope of the benchmark.** Frozen-field a-priori in a 2-D periodic box: it isolates
*structural correctness at one instant*. Any gap appearing only in a time-integrated
(a-posteriori) run would be the fingerprint of the Markovian delta-in-time
simplification (§4), not of the structural constraints — the benchmark is built to
separate the two, and the a-posteriori test is the natural follow-on (§8).

---

## 5b. The a-posteriori (time-integrated) test

§5 is *a-priori*: it scores closures on a single frozen filtered-DNS snapshot, where
the predicted discriminator is structural (solenoidality, transfer sign). The natural
follow-on (named in §8.1) is the *a-posteriori* test — integrate each closure forward as
a coarse LES and compare its long-time statistics to the spectrally-filtered DNS. We now
run it. Note the §5 projected-FDT operator *measures* its eddy viscosity and backscatter
amplitude from the true transfer, so it is not self-contained; an honest a-posteriori
test requires a **predictive** closure. We therefore integrate a self-contained closure
family that uses only resolved data: a scale-selective spectral eddy viscosity `ν_t(k)`
(plateau + cutoff cusp, Eq. 8), a **cusp-only** variant that concentrates the dissipation
at the cutoff and leaves the resolved interior untouched, and either of those augmented by
an **FDT-tied, Leray-projected stochastic backscatter** whose energy-injection *rate* is
tied to the near-cutoff eddy-viscous dissipation (Leith 1990; Mason & Thomson 1992).

**Setup.** The §5 256² DNS supplies the truth; we sharp-filter it at the cutoff to get the
reference resolved spectrum. A coarse LES (vorticity–streamfunction, 2/3-dealiased to the
cutoff, forced identically to the DNS) is integrated ~24 eddy-turnovers under each closure
and time-averaged over the developed window. Two cutoffs bracket the subgrid load: a *weak*
subgrid (forcing just below the cutoff) and a *strong* subgrid (forcing a decade below the
cutoff, a real unresolved enstrophy range). Diagnostics: resolved kinetic energy and
enstrophy relative to filtered DNS, and the resolved energy-spectrum L2 error. In vorticity
form the velocity is solenoidal *by construction*, so the a-posteriori test scores
**statistical fidelity**, not divergence — divergence was §5's a-priori discriminator.

**Results (strong-subgrid cutoff; Fig. 5).**

![A-posteriori coarse-LES statistics under each predictive closure versus the spectrally-filtered 256² DNS truth (strong-subgrid cutoff). (a) Time-averaged resolved energy spectrum: Smagorinsky (red) over-drains the resolved band, while the cusp eddy viscosity with FDT backscatter (blue) tracks the filtered DNS and nearly matches the no-model spectrum. (b) Resolved-spectrum L2 error and `KE/KE_dns` across closures: every eddy-viscosity closure is stable, Smagorinsky is the most over-dissipative, and the structured cusp+backscatter closure matches (does not beat) no-model — the honest 2-D ceiling.](figures/76_closure_aposteriori.png)


| closure | `KE/KE_dns` | `enstrophy/dns` | spectrum error | stable? |
|---|---|---|---|---|
| no model | 0.69 | 0.70 | 0.46 | yes |
| Smagorinsky | **0.50** | **0.50** | **0.59** | yes |
| spectral eddy viscosity | 0.63 | 0.64 | 0.49 | yes |
| spectral EV + FDT backscatter | 0.64 | 0.64 | 0.48 | yes |
| cusp eddy viscosity | 0.67 | 0.68 | 0.47 | yes |
| **cusp EV + FDT backscatter** | 0.67 | 0.68 | **0.46** | yes |

Three robust, falsifiable findings:

1. **Every predictive closure is stable** over ~24 eddy-turnovers — no drift, no blow-up.
   The §8.1 stability question is answered in the affirmative for this closure family.
2. **Smagorinsky is markedly over-dissipative a-posteriori** — it drains *half* the resolved
   kinetic energy and enstrophy and **worsens** the spectrum relative to doing nothing
   (error `0.59` vs `0.46`). This is the time-integrated counterpart of §5's a-priori
   result that a single positive `ν_t` carries the wrong (purely-dissipative) transfer: the
   error compounds into a large energy deficit. The sign is regime-robust — at the
   weak-subgrid cutoff it is milder but identical in direction (`KE/KE_dns≈0.81`, error
   `0.32` vs `0.19`).
3. **The FDT backscatter helps in the predicted direction, and a cusp-localized eddy
   viscosity nearly removes the spurious dissipation** — the backscatter term consistently
   lowers the spectrum error (`0.49→0.48`, `0.47→0.46`) by returning energy to the resolved
   band, and concentrating dissipation at the cutoff cusp brings the closure to within ~1%
   of the no-model spectrum error while still draining the cutoff enstrophy pile-up.

**An honest ceiling — stated as the scope of the result.** No eddy-viscosity closure
*beats* no-model on the resolved 2-D spectrum; the best (cusp + backscatter) only *matches*
it. This is consistent with — indeed predicted by — the paper's thesis: in 2-D the
resolved-scale energy budget is dominated by the *up-scale* (inverse) cascade, so the net
subgrid dissipation the resolved scales require is nearly zero, and the only a-posteriori
virtue available to a closure is to **not** inject spurious dissipation — which Smagorinsky
badly violates and the structured closure nearly achieves. The a-priori structural
discriminator of §5 is therefore *necessary but, in pure 2-D, not sufficient* to yield an
a-posteriori resolved-spectrum gain over no model; the structural benefit is expected to be
decisive precisely in the 3-D forward-cascade and wall-bounded settings (§5c demonstrates
the 3-D a-priori structural failure directly; §8), where net SGS dissipation is essential and an over/under-dissipative closure cannot be replaced by doing
nothing. The a-posteriori test thus confirms stability and the Smagorinsky over-dissipation,
validates the backscatter direction, and **bounds** the 2-D closure benefit — it does not
claim an a-posteriori win. (`general_two_clocks/closure_aposteriori.py`.)

---

## 5c. The same benchmark in 3-D — with vortex stretching active

Part 5's structural diagnosis is established in **2-D**, where it is clean and
CPU-verifiable but where vortex stretching `ω·∇u` — the engine of the forward cascade and
of singularity formation — is *identically zero*. We repeat the **identical** a-priori
benchmark in genuine **3-D incompressible DNS** (128³, pseudo-spectral, `ν=1.2×10⁻³`,
sharp-spectral filter at `k_c=24`; developed field `KE=3.49×10⁻²`, normalized stretching
production `⟨ω·S·ω⟩*=0.168>0`). The closure machinery — sharp filter, exact SGS force,
Leray projection, FDT-tied backscatter — carries over **verbatim**, because the
Helmholtz/Leray projection is dimension-agnostic (§4).

**The three 2-D verdicts survive (force-spectrum, solenoidality and transfer panels below).** Solenoidality: truth, Smagorinsky and
projected-FDT are divergence-free to machine precision (`RMS(∇·m) ≈ 1.0×10⁻¹⁴`,
`9.1×10⁻¹⁵`, `8.9×10⁻¹⁵`), while the spectrum-matched surrogate breaks it (`9.85`). The
energy transfer now runs **forward** (net `Σ_k T_true(k) = −4.5×10⁸ < 0`, the opposite of
the 2-D inverse cascade); the transfer-spectrum correlation with truth is projected-FDT
**1.000**, surrogate `0.27`, and Smagorinsky **−0.64** — *anti-correlated*: a single
positive `ν_t` removes energy from the very wavenumbers where the true flux deposits it.

![3-D force spectrum `E_m(k)`: Smagorinsky carries roughly the right magnitude but mis-sets the near-cutoff shape; the surrogate matches by construction; projected-FDT fills the correct spectrum.](figures/61_closure3d_force_spectrum.png)

![3-D solenoidality `RMS(∇·m)`: truth, Smagorinsky and projected-FDT are divergence-free to machine precision (all Leray-projected); the spectrum-matched surrogate fails (phase randomisation breaks `∇·m=0`).](figures/62_closure3d_divergence.png)

![3-D energy-transfer spectrum `T(k)`: net forward cascade; projected-FDT tracks the true transfer (corr 1.000), the surrogate is weakly correlated (0.27), and Smagorinsky is anti-correlated (−0.64).](figures/63_closure3d_transfer.png)

**The 3-D-only failure that 2-D cannot see (vortex-stretching panel below).** Three diagnostics have no 2-D
analogue and are why 3-D is scientifically necessary, not merely "2-D with an extra index":

- **Vortex-stretching production** `⟨ω_i S_ij ω_j⟩* = 0.168 > 0` (identically zero in 2-D,
  where vorticity is orthogonal to the strain plane) — the forward-cascade engine.
- **Strain–vorticity alignment** (the Constantin–Fefferman geometric depletion), mean `|cos|`
  over 4×10⁵ points: extensional `e₁=0.479`, **intermediate `e₂=0.654`**, compressive
  `e₃=0.325` — the preferential alignment with the *intermediate* strain eigenvector that
  depletes self-stretching and holds singularity formation off.
- **SGS backscatter volume fraction** (local flux `Π=−τ^d_{ij}S_{ij}<0`): truth **0.480**,
  Smagorinsky **0.000**. A positive-definite eddy viscosity has `Π≥0` *everywhere by
  construction*, so it is structurally incapable of representing the **~half of physical
  space** where 3-D turbulence transfers energy up-scale. In 2-D this is a boundary
  curiosity; in 3-D it is half of space.

![3-D-only physics: vortex-stretching production (`>0`, zero in 2-D), the intermediate-eigenvector strain–vorticity alignment, and the local SGS energy flux `Π` whose backscatter (`Π<0`) fills ~48 % of the volume for the truth and exactly 0 % for Smagorinsky.](figures/64_closure3d_vortex_stretching.png)

**Scope.** A frozen-field **a-priori** test in a 3-D periodic box — the structural-correctness
analogue of §5, now with vortex stretching active. It is **not** a grid-converged production
DNS, **not** an a-posteriori (time-integrated) closed-loop LES, and **not** a claim of 3-D
Navier–Stokes regularity or operational superiority. It establishes that the two-clocks
structural repair, and K-theory's specific structural failure, both carry into 3-D where the
dynamics that matter for the forward cascade and regularity actually live. GPU-verified
(Tesla P100, CuPy). (`general_two_clocks/run_closure3d.py`.)

---

## 6. The memory time is real, scalar-independent, and operationally useful

The collapse in §3 hinges on a finite memory time `τ_c`. We measure it and exercise its
consequences.

**6.1 `τ_c` measured, with sign.** From the autocorrelation of the SGS
eddy-diffusivity in a developed run, `τ_c≈0.02–0.03` (in eddy-turnover units) with a
**definite positive sign** (Mori 1965; Zwanzig 1973).

**6.2 `τ_c` is a turbulence clock, not a scalar clock.** Sweeping the
molecular Lewis number across a 100× contrast (heat vs salt), `τ_c(salt)/τ_c(heat)≈1`
even though the scalar Nusselt numbers track `Le` strongly. The memory belongs to the
*flow*, not the transported field — exactly what (3)–(4) require, since `K` is built
from the orthogonal dynamics of the velocity, independent of which scalar rides on it.

**6.3 The clock-mismatch correction works and has a unique sign.** The
first-order consequence of a lagged diffusivity is the clock-mismatch term
`−τ_c·∇·(∂_t K_u ∇θ)` (a commutator identity `[∂_t, D]θ = ∇·((∂_t K_u)∇θ)`, verified to
`~10⁻⁷`). Run inside a self-contained 1-D transient K-theory thermal solver over one
diffusivity transient (RK4, common numerical error cancelled):

- **Transient error cut ~15×** (`τ_c=0.05`: time-max rel-err `9.1×10⁻³ → 6.0×10⁻⁴`).
- **Leading-order error removed:** naive scales `∝τ_c¹` (slope 1.03), corrected
  `∝τ_c²` (slope 2.00) — one order higher-accurate.
- **Steady-turbulence null is exact:** at `∂_t K_u=0` the term vanishes and
  naive ≡ corrected ≡ truth (error 0.0).
- **The `+τ_c` sign is unique:** `−τ_c` is *worse* than naive — confirming §6.1's
  measured positive sign is the error-reducing choice. (Fig. 4)

![Transient-solver demonstration of the clock-mismatch correction. The `+τ_c`-corrected K-theory solver cuts transient error by about 15× relative to naive K-theory and vanishes identically in steady turbulence; the `−τ_c` sign performs worse than naive, pinning the sign.](figures/60_cmn_solver_demo.png)

This converts `τ_c` from a diagnostic into a deployable, sign-pinned correction; §6.4
brackets it with a 2-D advection proxy, leaving only the full operational solver (§8).

**6.4 The correction survives a 2-D advection + moving-plume proxy.**
The §6.3 demo used a single prescribed `K_u(t)` cycle in 1-D. A self-contained **2-D
advection–diffusion** solver with a Gaussian plume whose amplitude ramps *and* whose
centre **propagates** across the domain (the closest standalone analogue to a moving
plume/surge transient — *not* a real ice-flow solver) reproduces all four 1-D conclusions
under dimensionality + advection + spatial structure: transient error cut **~9×**
(`τ_c=0.02`), naive `∝τ_c^{0.99}` vs corrected `∝τ_c^{2.02}` (order lifted), exact steady
null (no event ⇒ error 0), and `+τ_c` the **unique** corrector (`−τ_c` worse than naive).
This raises confidence in — but does not replace — the deferred real-solver (ISSM/GlaDS)
test (`validation/synthetic/h3_cmn_reduced_model.py`, 7 tests).

---

## 7. When projection is inexact: a finite cleaning design window

§5 enforced `∇·m=0` by an *exact* spectral Leray projection. Real ice-sheet, ocean and
MHD codes instead enforce the constraint approximately (hyperbolic divergence cleaning,
penalization relaxation, a few projection iterations). How fast must cleaning be?

Deploying the Dedner GLM cleaning system (Dedner et al. 2002) as a tunable
approximate projector on a frozen real-bed cavity field, with the single dimensionless
knob `G = γ_clean·τ_adj` (cleaning rate × pressure-adjustment transit), the
constraint-violation residual after one transit collapses through a **finite window**:

| regime | `G = γ_clean·τ_adj` | behaviour |
|---|---|---|
| under-cleaned | `G ≲ 2` | weakly-damped wave; violation persists (residual ≈ 0.7 at `G=0`) |
| **design window** | **`2 ≲ G ≲ 12`** | violation suppressed; knee at `G*≈2.0` (1/e) |
| over-cleaned (stall) | `G ≳ 12` | slow telegrapher root `λ₋ ≈ −c_h²|k|²/γ → 0` freezes the low-`k` (global) modes; residual *re-grows* |

The over-damped upturn at `G_opt≈12` is the result: it is **not** in the original Dedner
analysis. Its mechanism is the slow root of the telegrapher equation
`∂_t²Q + γ_clean ∂_t Q − c_h²∇²Q = 0` — pushing `γ_clean` up to clean fast *locally*
simultaneously stalls the *global* low-wavenumber modes. **Consequence:** exact Leray
projection is a *singular* limit, approached optimally at finite `G_opt≈12` and degraded
beyond it; fast-local-cleaning codes sit *past* the optimum and carry a small,
persistent low-wavenumber divergence bias. Measured at `G=0`, the same elliptic
spreading gives a **non-local amplification `A≈3.9×`**: a divergence error concentrated
in 6 % of the domain corrupts 25 % of the global pressure field. This makes the
solenoidality requirement *prescriptive* — not just that `ℙ` matters, but how exactly it
must be enforced. (Runner: `glaciers/subglacial/theory_tests.py::result_dedner_cleaning`.)

---

## 8. Discussion and limitations

**What this is.** An operator-by-operator audit of single-K closure (the established MZ
Markovian limit, made term-explicit), a structurally-consistent closure form assembled
from known components that satisfies the structural constraints exactly and recovers
Smagorinsky and molecular NS in named limits, an a-priori structural benchmark
(solenoidality the predicted discriminator), a measured and sign-pinned
*scalar-independent* memory time with a working correction (verified in 1-D and a 2-D
advection proxy), and a prescription for inexact projection.

**What this is not.** Not a claim of 3-D regularity, not a universal closure, not a
demonstration of operational superiority in a full weather/climate/engineering model.
The benchmark is periodic-box and a-priori, now in both 2-D (§5) and 3-D (§5c). The FDT commutation that makes (4) clean
is special to the periodic spectral setting; bounded-domain (solid-wall) formulations
break the shared eigenbasis of `P` and `ℙ` and require a separate treatment.

**Natural next steps (in this paper or a sequel).**
1. §5c now runs the a-priori benchmark in **3-D** (forward cascade, vortex stretching
   active): the structural verdicts survive and the ~48 %-of-space backscatter that
   Smagorinsky misses is exposed. §5b runs the a-posteriori test in 2-D (stable; Smagorinsky
   over-dissipates; the structured closure matches but does not beat no-model, because 2-D
   resolved scales need near-zero net SGS dissipation). The remaining open item is the **3-D /
   wall-bounded a-posteriori** (time-integrated) test, where net dissipation is essential and
   the structural advantage should become decisive; if long-time tracking drifts there while the structural diagnostics stay
   correct, restore the finite-memory kernel (the t-model), which §3–§4 already name.
2. A bounded-domain / wall-bounded formulation of (4).
3. Couple §6's measured `τ_c` into an operational eddy-viscosity correction (the
   `−τ_c∇·(∂_t K∇θ)` term) inside a production solver.

**Prior art, honestly attributed.** Mori (1965); Zwanzig (1973);
Chorin–Hald–Kupferman optimal prediction and the t-model (2000, 2002); Stinis;
Parish & Duraisamy (MZ-based LES); Kraichnan (1976); Chollet & Lesieur (1981);
Domaradzki & Adams (nonlocal SGS transfer); Leith (1990); Mason & Thomson (1992);
Frederiksen & Davies; Kraichnan's DIA lineage for FDT; Dedner et al. (2002) for GLM
cleaning. It claims only the *audit* (the term-by-term accounting of which GLE structure single-K
discards), the a-priori structural benchmark (solenoidality as the predicted
discriminator), the measured scalar-independent `τ_c`, and the Dedner over-cleaning
window.

---

## 9. Reproduce

```bash
pip install -r requirements.txt
python general_two_clocks/run_closure.py --out-dir general_two_clocks/figures # §5, Figs 28–30
python general_two_clocks/gle_coefficients.py # §6.1, Fig 53
python general_two_clocks/scalar_clock_universality.py # §6.2, Fig 55
python general_two_clocks/cmn_solver_demo.py # §6.3, Fig 60
python general_two_clocks/closure_aposteriori.py --kc 16 --k-f 10 --n-les 48 # §5b a-posteriori test (Fig 76)
python general_two_clocks/run_closure3d.py --gpu --n 128 --kc 24 --nu 0.0012 --steps 2500 # §5c 3-D benchmark (Figs 61–64, GPU/CuPy)
pytest general_two_clocks/tests -v # structural unit proofs
# §7 Dedner window: glaciers/subglacial/theory_tests.py::result_dedner_cleaning
```

## Data and code availability

The 256² closure benchmark (Figs. 1–3), the transient-solver demonstration (Fig. 4), and the 3-D benchmark (its four panels, `run_closure3d.py`, Tesla P100/CuPy) regenerate from the scripts listed in §9. All underlying data and analysis code are in the public repository at https://github.com/abdh0saifelden2-debug/K.

## References

Boussinesq, J. (1877). Essai sur la théorie des eaux courantes. *Mémoires présentés par divers savants à l'Académie des Sciences* 23, 1–680.

Chollet, J.-P., & Lesieur, M. (1981). Parameterization of small scales of three-dimensional isotropic turbulence using spectral closures. *Journal of the Atmospheric Sciences* 38, 2747–2757.

Chorin, A. J., Hald, O. H., & Kupferman, R. (2000). Optimal prediction and the Mori–Zwanzig representation of irreversible processes. *Proceedings of the National Academy of Sciences* 97, 2968–2973. doi:10.1073/pnas.97.7.2968

Chorin, A. J., Hald, O. H., & Kupferman, R. (2002). Optimal prediction with memory. *Physica D: Nonlinear Phenomena* 166, 239–257. doi:10.1016/S0167-2789(02)00446-3

Dedner, A., Kemm, F., Kröner, D., Munz, C.-D., Schnitzer, T., & Wesenberg, M. (2002). Hyperbolic divergence cleaning for the MHD equations. *Journal of Computational Physics* 175, 645–673. doi:10.1006/jcph.2001.6961

Domaradzki, J. A., & Adams, N. A. (2002). Direct modelling of subgrid scales of turbulence in large eddy simulations. *Journal of Turbulence* 3, N24. doi:10.1088/1468-5248/3/1/024

Frederiksen, J. S., & Davies, A. G. (1997). Eddy viscosity and stochastic backscatter parameterizations on the sphere for atmospheric circulation models. *Journal of the Atmospheric Sciences* 54, 2475–2492.

Kraichnan, R. H. (1959). The structure of isotropic turbulence at very high Reynolds numbers. *Journal of Fluid Mechanics* 5, 497–543. doi:10.1017/S0022112059000362

Kraichnan, R. H. (1976). Eddy viscosity in two and three dimensions. *Journal of the Atmospheric Sciences* 33, 1521–1536.

Leith, C. E. (1990). Stochastic backscatter in a subgrid-scale model: Plane shear mixing layer. *Physics of Fluids A: Fluid Dynamics* 2, 297–299. doi:10.1063/1.857779

Leray, J. (1934). Sur le mouvement d'un liquide visqueux emplissant l'espace. *Acta Mathematica* 63, 193–248. doi:10.1007/BF02547354

Mason, P. J., & Thomson, D. J. (1992). Stochastic backscatter in large-eddy simulations of boundary layers. *Journal of Fluid Mechanics* 242, 51–78. doi:10.1017/S0022112092002271

Monin, A. S., & Obukhov, A. M. (1954). Basic laws of turbulent mixing in the surface layer of the atmosphere. *Trudy Geofizicheskogo Instituta, Akademiya Nauk SSSR* 24(151), 163–187.

Mori, H. (1965). Transport, collective motion, and Brownian motion. *Progress of Theoretical Physics* 33, 423–455. doi:10.1143/PTP.33.423

Parish, E. J., & Duraisamy, K. (2017). A dynamic subgrid scale model for large eddy simulations based on the Mori–Zwanzig formalism. *Journal of Computational Physics* 349, 154–175. doi:10.1016/j.jcp.2017.07.053

Schumann, U. (1995). Stochastic backscatter of turbulence energy and scalar variance by random subgrid-scale fluxes. *Proceedings of the Royal Society A* 451, 293–318. doi:10.1098/rspa.1995.0126

Smagorinsky, J. (1963). General circulation experiments with the primitive equations: I. The basic experiment. *Monthly Weather Review* 91, 99–164.

Stinis, P. (2015). Renormalized Mori–Zwanzig-reduced models for systems without scale separation. *Proceedings of the Royal Society A* 471, 20140446. doi:10.1098/rspa.2014.0446

Zwanzig, R. (1973). Nonlinear generalized Langevin equations. *Journal of Statistical Physics* 9, 215–220. doi:10.1007/BF01008729
