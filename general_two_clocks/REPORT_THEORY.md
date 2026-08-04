# Part 8 — A two-clocks closure for K-theory (Mori–Zwanzig / projected-FDT)
 
This part turns the diagnosis of Parts 1–7 into a *prescription*: a closure that
keeps the cheap, local "slow clock" but restores the non-local, boundary-aware
"fast clock" that single-eddy-diffusivity (K-theory) closure throws away. It is a
**strict generalization** of Smagorinsky K-theory, built so that it reduces to
K-theory in a named limit and to molecular Navier–Stokes (NS) in another.
 
> **Honesty up front.** The ingredients here — Mori–Zwanzig (MZ) optimal
> prediction, spectral (nonlocal) eddy viscosity, and fluctuation–dissipation
> (FDT) stochastic backscatter — are established in the literature (see
> *Prior art*). The contribution of this repo is the **synthesis framed by the
> two clocks**: stating precisely that K-theory's error *is* the Markovian-delta
> collapse of the MZ memory kernel, that the FDT linkage plus the Leray projection
> is exactly the "structure" the Part-6 SPDE experiment proved spectrum-matching
> lacks, and a concrete benchmark (Part 8b) that decides it. "Flawless" closure is
> not claimed — turbulence closure is an open problem; what *is* claimed is exact
> satisfaction of the structural constraints below and elimination of the specific
> failure mode the repo isolated.
 
---
 
## 1. The role K-theory plays, and its four hidden assumptions
 
Filter the incompressible NS equations at scale Δ (overbar = resolved). With the
Leray projector ℙ = I − ∇(∇²)⁻¹∇· and the subgrid force **m** = −∇·τ,
τ_ij = (overline{u_i u_j} − ū_i ū_j),
 
   ∂ₜū = ℙ[ −(ū·∇)ū + ν∇²ū + **m** ],     ∇·ū = 0.            (1)
 
The closure problem is: model **m** from resolved data. K-theory (Boussinesq
hypothesis + Smagorinsky) sets
 
   **m** = ∇·(2 ν_t S̄),   ν_t = (Cₛ Δ)² |S̄|,   S̄ = ½(∇ū + ∇ūᵀ).   (2)
 
A single scalar field ν_t. That one object hides four assumptions — each is a clock
being crushed:
 
1. **Local in space** (gradient diffusion, **m** depends only on ∇ū at a point) —
   denies the elliptic, *non-local* pressure clock (∇²p = f couples the whole domain).
2. **Instantaneous in time** (no memory) — denies that the fast clock carries history.
3. **Down-gradient only** (ν_t ≥ 0; purely dissipative) — forbids **backscatter**
   (energy transfer from unresolved to resolved scales). In 2-D this is not a small
   error: the inverse energy cascade *is* net up-scale transfer.
4. **No structural re-imposition** — the closure acts on ū and leaves
   incompressibility entirely to the pressure solve, rather than being
   divergence-consistent itself.
 
In the language of the repo: K-theory ≈ "the slow Markov drift, plus a dissipative
leak." It is exactly the local, memoryless forecaster Parts 5–6 proved blind to the
fast clock.
 
---
 
## 2. The exact resolved dynamics: the Mori–Zwanzig generalized Langevin equation
 
Write the (semi-discrete) NS system as u̇ = R(u) and let the Koopman/Liouville
generator L act on observables, e^{tL}g(u₀) = g(u(t)). Choose a projection P onto
the resolved variables â = Pu (Mori's linear projection, or Chorin–Hald–Kupferman
*optimal prediction* conditional expectation), and Q = I − P. The Dyson identity
e^{tL} = e^{tQL} + ∫₀ᵗ e^{(t−s)L} P L e^{sQL} ds applied to the resolved dynamics
gives the **generalized Langevin equation (GLE)** — exact, no modeling yet:
 
   ∂ₜâ(t) = **PLâ(t)**  −  **∫₀ᵗ K(t−s) â(s) ds**  +  **f(t)**,          (3)
            └ Markov ┘     └─── memory ───┘          └ noise ┘
 
   K(τ) = −P L e^{τQL} Q L  (memory kernel),   f(t) = e^{tQL} Q L â(0)  (orthogonal "noise").
 
These map one-to-one onto the two clocks, *made exact*:
 
- **PLâ** — the slow, resolved, **deterministic Markov drift** (advection + viscous
  transport of resolved modes): the perfect local generator of Part 6.
- **−∫K(t−s)â ds** — the **non-local-in-time** correction: the running memory of
  what the unresolved fast clock did. This is the term K-theory does not have.
- **f(t)** — the orthogonal-dynamics force living entirely in the unresolved
  subspace: the honest origin of the "stochastic" term in SPDE closures.
 
**Second fluctuation–dissipation theorem (2nd FDT).** Under stationarity and the
inner product (·,·) that defines P, the memory kernel and the noise are *not*
independent — they are two faces of one object:
 
   K(τ) = ( f(τ), f(0) ) · ( â, â )⁻¹.                              (4)
 
**Well-posedness of (4) in the spectral box.** The inner product (·,·) is the
kinetic-energy (L²) norm, and the resolved-mode projection P is the sharp spectral
truncation (keep |k| ≤ k_c). In a periodic box with exact spectral derivatives,
*both* P and the Leray projector ℙ_k = I − kkᵀ/|k|² are orthogonal projections that
are **diagonal in the Fourier basis**, so they commute (Pℙ = ℙP) and each is
self-adjoint w.r.t. (·,·). Hence the orthogonal subspace Q = I − P is ℙ-invariant,
the noise f = e^{tQL}QLâ(0) stays solenoidal, and (4) is a relation between
well-defined solenoidal correlations — no ambiguity in the order of projecting vs.
taking the energy norm. (This clean commutation is special to the periodic spectral
setting; with solid walls the two projections no longer share an eigenbasis.)
 
Equation (4) is the crux. **Dissipation (K) and fluctuation (f) are locked
together.** Any closure that sets the noise covariance independently of the memory
kernel is internally inconsistent — it injects energy at a rate unrelated to the
rate it removes it.
 
---
 
## 3. K-theory = the Markovian-delta collapse of the MZ kernel
 
If the memory kernel decays fast compared with the evolution of â (short-memory /
"t-model" limit), the convolution collapses to a local damping:
 
   ∫₀ᵗ K(t−s) â(s) ds  ⟶  ( ∫₀^∞ K(τ) dτ ) â(t)  ≡  Γ â(t).          (5)
 
Choosing Γ diagonal in wavenumber with Γ(k) = ν_t |k|² and **dropping f
entirely** reduces the exact GLE (3) to
 
   ∂ₜû(k) = ℙ_k N̂(û)(k) − ν_t |k|² û(k),                            (6)
 
which is exactly Smagorinsky K-theory (2) in spectral form. So:
 
> **K-theory is the GLE with (i) the memory kernel replaced by a delta in time,
> (ii) ν_t taken constant in k, and (iii) the FDT-linked noise deleted.**
 
Each of the four sins in §1 is now a specific approximation in (5)–(6): (i)+(ii)
are the delta-in-time, constant-in-k collapse; (iii) is deleting f; the
down-gradient restriction is ν_t ≥ 0, i.e. forbidding the negative/backscatter
part of the true transfer. This is the precise mathematical statement of *why*
single-K is blind to the fast clock.
 
---
 
## 4. The closed model: nonlocal eddy viscosity + projected FDT backscatter
 
Keep the structure of the exact GLE but make each term computable. In the periodic
spectral box (where ℙ_k = I − kkᵀ/|k|² is exact):
 
   ∂ₜû(k,t) = ℙ_k N̂(û)(k,t)  −  ν_t(k,t) |k|² û(k,t)  +  f̂(k,t),     (7)
 
with three repo-native ingredients.
 
**(a) Nonlocal (spectral) eddy viscosity ν_t(k,t).** Not a scalar — the
plateau–cusp spectral eddy viscosity (Kraichnan 1976; Chollet–Lesieur 1981) read
off the resolved energy at the cutoff k_c:
 
   ν_t(k) = ν_t^∞ · √( E(k_c)/k_c ) · [ 1 + c (k/k_c)^p ],            (8)
 
flat at low k, rising into a cusp near k_c. The slow-clock dissipation, done
**scale-selectively** instead of as one number. (In 2-D the low-k plateau can be
*negative* — the inverse cascade — which is why a single positive ν_t is especially
wrong there.)
 
**(b) Structural backscatter f̂(k,t): projected and FDT-linked.** Zero-mean,
**divergence-free by construction**,
 
   ℙ_k f̂ = f̂   (in 2-D: f̂ = ik^⊥ ψ̂, a random streamfunction ψ̂),     (9)
 
with covariance tied to the *same* operator that dissipates, via the discrete 2nd
FDT (4):
 
   ⟨ f̂_i(k,t) f̂_j*(k,s) ⟩ = 2 ν_t^B(k,t) |k|² Θ(k) ℙ_ij(k) δ(t−s),   (10)
 
where ν_t^B(k) is the **backscatter** (energy-injecting) part of the transfer and
Θ(k) is fixed so the *net* spectral transfer (dissipation − backscatter) equals the
true subgrid transfer. The forward (dissipative) and backward (backscatter)
channels are thus two signs of one FDT-consistent operator, not a damping plus an
unrelated random kick.
 
**(c) Leray projection wraps everything**, so (7) is incompressible *exactly*. The
fast clock is represented as the **operator** ℙ — never approximated by noise.
 
The memory generalization (drop the Markovian collapse of §3) is
 
   ν_t(k,t)|k|² û(k,t)  ⟶  ∫₀ᵗ K(k, t−s) û(k,s) ds,                  (11)
 
restoring non-locality *in time*; (7) is the short-memory limit of (11).
 
**What the Markovian (delta-in-time) step does and does not buy.** Keeping the
scale-dependent ν_t(k) and the FDT-linked, projected noise while collapsing the
kernel to a delta is a **Markovianized representation of a genuinely
non-Markovian process**. It reproduces the *statistical structure* of the fast
clock — the scale-selective net transfer and the exact solenoidal constraint — but
it assumes the unresolved scales clear their memory instantly relative to the slow
clock. In 2-D turbulence dominated by long-lived coherent vortices, the subgrid
scales can carry a longer memory than this allows. The structural constraints (ℙ,
FDT) are unaffected; the time-locality is the approximation at risk. The
a-posteriori benchmark (§7) is designed to expose exactly this: if long-time
tracking drifts from DNS truth while the spectrum, divergence, and net transfer
stay correct, the culprit is the delta-in-time collapse — i.e. one should restore
the finite-memory kernel (11) (the t-model / finite-memory MZ), not abandon the
structural repair.
 
---
 
## 5. Why this is exactly the "energy AND structure" fix of Part 6 (Fig 24)
 
Part 6 showed: a phase-randomized surrogate with the **identical power spectrum**
of the true projection correction has ≈0.01 pointwise correlation with it and
leaves RMS ∇·u ≈ 3.8×10⁻² instead of zero — *energy yes, structure no*. The model
(7)–(10) names and repairs both failures:
 
| failure in spectrum-matching | repair here |
|---|---|
| noise covariance set independently of dissipation | **2nd FDT (4)/(10)** locks ⟨f̂f̂⟩ to the kernel |
| forcing not divergence-free | **Leray projection (9)** ⇒ ∇·f̂ = 0 exactly |
| only the spectrum is constrained | **net transfer Θ(k)** matches the true SGS transfer T(k) |
 
So the three Fig-24 panels — spectrum, divergence, structure/transfer — *all* pass,
which is the precise sense in which this closure "compensates for K-theory's role."
 
---
 
## 6. NS-consistency conditions ("works flawlessly with NS" — stated honestly)
 
The model is built to satisfy, exactly:
 
- **Galilean invariance** — acts on k ≠ 0 modes / resolved strain only.
- **Exact incompressibility** — ℙ on both the drift and the noise (9).
- **Energy budget / realizability** — FDT (10) fixes the mean energy transfer; no
  spurious energy injection (∮ net transfer = modeled SGS dissipation).
- **Symmetry & dissipativity** — ℙ_ij(k) symmetric; the resolved-mean (ensemble)
  part of the transfer is sign-correct (net forward in 3-D, net inverse in 2-D).
- **Two clean limits** — Δ → 0 ⇒ (ν_t → 0, f̂ → 0, K → δ) recovers molecular NS;
  the local/Markovian/no-noise limit (5)–(6) recovers Smagorinsky K-theory. So it
  is a strict generalization, not a rival model.
 
**Caveat that will not be papered over.** No closure is universally exact — the
kernel K(k,τ) (or ν_t(k), Θ(k)) still requires modeling choices, and this model is
not claimed to be flawless across all regimes. What it *does* guarantee is exact
satisfaction of the structural constraints above and removal of the specific
failure mode (energy-without-structure) the repo isolated.
 
---
 
## 7. The benchmark that decides it (implemented in Part 8b)
 
A frozen-field **a-priori** test plus a short **a-posteriori** run in the same 2-D
spectral box used in Parts 5–6.
 
1. **Truth.** Evolve a high-resolution divergence-free 2-D field to a developed
   spectrum (DNS truth u). Sharp-spectral-filter at k_c → resolved ū and the
   *exact* subgrid force m_true = ℙ[ N(u)‾ − N(ū) ].
2. **Models of m:** (i) Smagorinsky (2); (ii) spectrum-matched random field with the
   same E_m(k) as m_true (the Part-6 surrogate); (iii) the projected-FDT model
   (8)–(10).
3. **Three diagnostics vs. truth:** the force spectrum E_m(k); the RMS divergence of
   the closed field; and the **energy-transfer spectrum** T(k) = Re⟨ û*·m̂ ⟩,
   resolving forward dissipation (T<0) from backscatter (T>0).
 
**Predicted outcome** (the falsifiable claim): (i) Smagorinsky reproduces gross
dissipation but has T(k) ≤ 0 everywhere — **no backscatter** — and mis-sets the
near-cutoff cusp; (ii) the spectrum-matched surrogate matches E_m(k) but has T(k)
**uncorrelated** with truth and RMS ∇·u = O(1); (iii) the projected-FDT model
matches E_m(k), holds ∇·u ≈ 0, and reproduces the forward/backscatter partition of
T(k). That is Fig 24 promoted to a closure benchmark.
 
The a-priori test isolates *structure* at a frozen instant (no memory yet matters),
so the projected-FDT model should win cleanly there. Any gap that appears only in
the *a-posteriori* (time-integrated) test is the fingerprint of the Markovian
delta-in-time simplification discussed in §4, not of the structural constraints —
the benchmark is built to separate the two.
 
---
 
## Prior art (honest attribution)
 
- **Mori–Zwanzig / GLE:** Mori (1965); Zwanzig (1973). **Optimal prediction & the
  t-model:** Chorin, Hald & Kupferman (2000, 2002); Stinis; Parish & Duraisamy
  (MZ-based LES).
- **Spectral / nonlocal eddy viscosity:** Kraichnan (1976); Chollet & Lesieur
  (1981); Domaradzki & Adams (review of nonlocal SGS transfer).
- **Stochastic backscatter & FDT closures:** Leith (1990); Mason & Thomson (1992);
  Frederiksen & Davies; Schumann. **FDT in turbulence:** Kraichnan's DIA lineage.
 
This repo does not claim these results. It claims the *two-clocks reading* of them —
K-theory as the Markovian-delta collapse of the MZ kernel, and the FDT+projection
pair as the missing "structure" — and provides the decisive 2-D benchmark.
 
## Scope
 
A 2-D, periodic-box formulation and benchmark. It is a structural demonstration
that an MZ/FDT/projected closure repairs the exact deficiencies of single-K closure
that Parts 1–7 isolated; it is **not** a claim of 3-D regularity, of a universal
closure, or of operational superiority in a full weather/climate model.
