# Part 9c — A two-clocks closure for turbulent heat transfer in the subglacial cavity (stratified)

This note derives, from first principles, the closure tested numerically in Part 9
(`subglacial/`) and 9b (`subglacial/flow3d.py`), specialised to the **turbulent
meltwater cavity** between a rough bed and the ice base. It is a glacier-facing
restatement and extension of the general two-clocks closure of **Part 8**
(`REPORT_THEORY.md`): same Mori–Zwanzig / projected-FDT machinery, now written for
the quantity a glaciologist cares about — the **turbulent heat flux delivered to the
ice base**, i.e. the basal/ice-base melt rate — carried through to a **stratification
(Richardson-number) correction**, a **predicted regime diagram**, and an
**operator-level structural theory** whose four theorems collapse to a single
non-commutation that is **verified numerically with no GPU**.

> **Honesty up front.** The ingredients (Mori–Zwanzig optimal prediction, spectral
> nonlocal eddy viscosity, fluctuation–dissipation backscatter, Leray projection)
> are established (see *Prior art*, Part 8 §"Prior art"). What is new here is the
> *specialisation to the subglacial cavity*: (i) the explicit identification of the
> two clocks with the **elliptic pressure-adjustment** time and the **parabolic
> thermal-memory** time; (ii) the recognition that the four structural failures of
> K-theory are one **operator non-commutation** `[𝕃, ℙ] ≠ 0` (§7); (iii) a
> transparent reduced model for the **heat-flux enhancement factor** R and how
> stratification controls it; and (iv) a **falsifiable regime prediction** that turns
> the planned LES into a discriminating test. The four theorems are operator
> statements with named O(1) constants pinned by simulation, **not** first-principles
> melt numbers; this is a closure derivation and a prediction, **not** a validated
> production melt law, and not a claim that turbulence closure is solved.

---

## 1. Axioms and scope

### 1.1 Why this is *not* a relabeling of subglacial hydrology

Subglacial hydrology already has a two-timescale structure the community knows well:
**fast** water-transit / pressure-wave adjustment (hours) versus **slow** cavity and
channel evolution by ice creep (months–years). That split is **hydrology versus ice
mechanics**, and we do not touch it.

Our "two clocks" live one level down, *inside* the fast hydrologic pressure field, in
the turbulent boundary layer of the meltwater itself:

- **Clock A — elliptic pressure adjustment (fast, memoryless at the turbulence
  scale).** Incompressibility makes pressure satisfy ∇²p = −∇·(∇·(uu)); it responds
  *instantaneously and non-locally* to the whole velocity field (the Leray clock).
- **Clock B — parabolic thermal advection–diffusion (slow, memory-bearing).** Heat
  is carried by eddies and diffused; the subgrid heat flux carries a *running memory*
  of recent overturning events. This is the clock that controls melt.

K-theory (Smagorinsky + gradient-diffusion) collapses **both** clocks to a single
local, instantaneous eddy diffusivity. The claim is precise and narrow:

> We do **not** replace the hydrology/ice-mechanics split. We resolve a *substructure
> within* the fast hydrologic pressure field — the separation between the elliptic
> pressure-adjustment time and the thermal-memory time — that single-eddy-diffusivity
> closures discard. The basal melt rate is sensitive to that substructure.

### 1.2 The three operators (axioms)

Let (ū, p̄, b̄) be the filtered fields in a cavity Ω with boundary
∂Ω = Γ_bed ∪ Γ_ice. Everything below is built from three operators:

- **Elliptic clock 𝔼 = −∇²** with Neumann/Dirichlet BCs on Γ_ice/Γ_bed. **Axiom:**
  𝔼⁻¹ is a **non-compact** pseudodifferential operator of order −2; its Green's
  function is the long-ranged G ∼ ln r (2D) / 1/r (3D). *Global, instantaneous.*
- **Parabolic clock ℙ(t) = e^{tA}, A = κ∇² − ū(x)·∇** (the heat–advection semigroup
  with the same wall BCs). **Axiom:** ℙ(t) is a **compact** semigroup for t > 0;
  its kernel is exponentially localized. *Local, memory-bearing.* **ℙ(t) is
  variable-coefficient** (ū depends on x) and **not** a Fourier multiplier — this is
  essential in §6–§7.
- **Constraint projector 𝕃 = 𝕀 − ∇𝔼⁻¹∇·** (Leray). **Axiom:** 𝕃 is a
  **Calderón–Zygmund** singular integral operator; with the symbol
  ℙ_k = 𝕀 − kkᵀ/|k|² in the periodic interior, exact projection onto ∇·u = 0.

The whole structural theory is: **𝕃 (global, elliptic) does not commute with ℙ(t)
(local, parabolic).** The four theorems are four shadows of that one fact (§7).

---

## 2. The filtered cavity system and the two clocks (foundation)

Let the cavity occupy the fluid region between the rough bed y = y_b(x,z) (warm,
θ = 1) and the ice base y = y_i(x,z) (cold, θ = 0). Non-dimensionalise on cavity
height H, bulk along-bed speed U, and bed–ice thermal contrast ΔT. The incompressible
Boussinesq equations are

&nbsp;&nbsp;∂ₜu + (u·∇)u = −∇p + (1/Re)∇²u + Ri_b θ ĵ + F,&nbsp;&nbsp;&nbsp;&nbsp;∇·u = 0,&nbsp;&nbsp;&nbsp;&nbsp;(1)

&nbsp;&nbsp;∂ₜθ + (u·∇)θ = (1/Pe)∇²θ,&nbsp;&nbsp;&nbsp;&nbsp;(2)

with Re = UH/ν, Pe = UH/κ = Re·Pr, and **Ri_b = g'H/U²** the bulk Richardson number
from the reduced gravity g' = gΔρ/ρ₀ of the density anomaly θ carries. Solids are
embedded by volume penalization (Brinkman), so (1)–(2) hold in the cavity and relax
u→0, θ→θ_solid inside rock/ice (`subglacial/flow3d.py`).

**Sign of buoyancy — two physical regimes.** The single scalar θ can represent either
limit, and they predict *opposite* baseline behaviour:

- **(T) Thermally driven, warm bed below cold ice** (subglacial-lake / geothermal
  limit): statically **unstable** ⇒ convection *enhances* melt; control Ra ∝ Ri_b·Re·Pe.
- **(S) Marine meltwater boundary layer** (Thwaites grounding-zone limit): fresh,
  buoyant melt on top ⇒ **stably stratified** ⇒ stratification *suppresses* transfer.

The discriminating physics lives in the **stable regime (S)** — the regime the
cryosphere community cares about — so the headline prediction is made there.

**Ice-base melt BC.** The three-equation melt law (Holland–Jenkins; McPhee) sets the
melt velocity m from the turbulent heat balance at the ice base,

&nbsp;&nbsp;ρ_w c_p γ_T u\* (T_w − T_b) = ρ_i L_f m + (conductive ice flux),&nbsp;&nbsp;&nbsp;&nbsp;(3)

and the product γ_T u\* (T_w − T_b) is *exactly* the near-wall turbulent heat flux
q·ĵ. **So the melt rate is the wall heat flux.** In the solver we report the basal
melt proxy

&nbsp;&nbsp;m ∝ q_ice = ⟨ v θ − (1/Pe) ∂_y θ ⟩ |_{ice band},&nbsp;&nbsp;&nbsp;&nbsp;(4)

and the *ratio* between closures is what we trust — as the community treats γ_T as a
factor-uncertain transfer coefficient.

**Filtered equations.** Apply an LES filter at scale Δ (overbar = resolved). Filtering
(1)–(2) (commutator-error-free spectral filter, Part 8 §2),

&nbsp;&nbsp;∂ₜū + ∇·(ū⊗ū) = −∇p̄ + (1/Re)∇²ū + Ri_b θ̄ ĵ + F − ∇·**τ**,&nbsp;&nbsp;&nbsp;&nbsp;(5)

&nbsp;&nbsp;∂ₜθ̄ + ∇·(ū θ̄) = (1/Pe)∇²θ̄ − ∇·**q**,&nbsp;&nbsp;&nbsp;&nbsp;(6)

&nbsp;&nbsp;**τ** = overline{u⊗u} − ū⊗ū,&nbsp;&nbsp;**q** = overline{uθ} − ū θ̄.&nbsp;&nbsp;&nbsp;&nbsp;(7)

**Pressure is slaved (the elliptic clock made explicit).** Taking ∇· of (5) and using
∇·ū = 0, every single-divergence term drops except pressure and the subgrid force:

&nbsp;&nbsp;**𝔼 p̄ = ∂_y(Ri_b θ̄) − ∇·(∇·τ),**&nbsp;&nbsp;i.e.&nbsp;&nbsp;∇²p̄ = ∂_y(Ri_b θ̄) + ∇·(∇·τ).&nbsp;&nbsp;&nbsp;&nbsp;(8)

This is the operator origin of Theorem 1 (§3): whatever divergence the *modelled*
subgrid force carries is injected, through 𝔼⁻¹, into the global pressure.

**Exact resolved dynamics (Mori–Zwanzig) and the two clocks.** Write the semi-discrete
system as ẇ = R(w), w = (u,θ); project onto resolved modes â = Pw (sharp truncation
|k| ≤ k_c), Q = I − P. The Dyson identity gives the **generalized Langevin equation**
(exact, no modeling):

&nbsp;&nbsp;∂ₜâ = **P L â** − **∫₀ᵗ K(t−s) â(s) ds** + **f(t)**,&nbsp;&nbsp;&nbsp;&nbsp;(9)

with kernel K(τ) = −P L e^{τQL} Q L, orthogonal noise f(t) = e^{tQL} Q L â(0), locked
by the **second fluctuation–dissipation theorem** K(τ) = (f(τ),f(0))·(â,â)⁻¹. The
kernel decays on a memory time τ_mem; the relevant clock ratio is

&nbsp;&nbsp;**𝒞 ≡ τ_mem / τ_turn**&nbsp;&nbsp;(τ_turn = H/U).&nbsp;&nbsp;&nbsp;&nbsp;(10)

𝒞 ≪ 1 ⇒ memory collapses ⇒ K-theory valid; 𝒞 = O(1) ⇒ memory matters ⇒ two-clocks
departs from K-theory.

**K-theory = the Markovian-delta collapse.** If K(τ) decays fast (𝒞 → 0) the
convolution collapses to local damping Γ = ∫₀^∞ K dτ, and **dropping f** gives

&nbsp;&nbsp;**q** = −κ_t ∇θ̄,&nbsp;&nbsp;κ_t = ν_t/Pr_t,&nbsp;&nbsp;ν_t = (C_s Δ)²|S̄| (≥ 0)&nbsp;&nbsp;&nbsp;&nbsp;(11)

— gradient-diffusion / Smagorinsky. Four collapses are baked in: local in space,
instantaneous in time, **down-gradient only** (κ_t ≥ 0: no backscatter), and **no
structural projection** — the curl-free part of the closure force is dumped into the
pressure solve. The two-clocks closure keeps the structure of (9) but makes each term
computable (periodic spectral box, ℙ_k exact):

&nbsp;&nbsp;∂ₜâ(k) = ℙ_k N̂(k) − ν_t(k)|k|² â(k) + f̂(k),&nbsp;&nbsp;&nbsp;&nbsp;(12)

| ingredient | Smagorinsky / K-theory | Two-clocks (projected-FDT) |
|---|---|---|
| eddy viscosity | scalar ν_t = (C_sΔ)²\|S̄\|, k-independent | ν_t(k): Kraichnan plateau + cusp at k_c |
| time structure | instantaneous (Γ = δ-in-time) | memory kernel ∫₀ᵗ K(k,t−s)â ds |
| sign of transfer | down-gradient only (ν_t ≥ 0) | forward **and** backscatter (FDT noise f̂) |
| noise | none (f deleted) | ⟨f̂_i f̂_j*⟩ = 2 ν_t^B\|k\|² Θ ℙ_ij δ(t−s) |
| incompressibility | left to pressure solve (φ leaks into p̄) | ℙ_k on the whole RHS ⇒ ∇·m = 0 exactly |
| heat flux **q** | −κ_t∇θ̄, κ_t ≥ 0 | −κ_t(k)∇θ̄ + projected stochastic flux |

The difference is **operator-level**, not a coefficient. In the limit 𝒞→0, β→0, ℙ
delegated to pressure, (12) reduces *exactly* to (11): the closure is a strict
generalization of Smagorinsky. The next four sections are the structural consequences.

---

## 3. Theorem 1 — the Spurious Pressure Source (verified)

**Statement.** Let m be any subgrid momentum force with ∇·m = Q ≠ 0. The spurious
pressure δp = 𝔼⁻¹Q satisfies the elliptic-regularity bound ‖δp‖_{L²(Ω)} ≥
C(Ω)‖Q‖_{H⁻¹(Ω)}, and for a compact K away from supp Q,
‖δp‖_{L²(K)} ≥ c·dist(K, supp Q)^{2−d}·‖Q‖_{L¹} (logarithmic in 2D). **A local
subgrid error becomes a global pressure corruption** through the Green's function.

**Proof.** From (8), the modelled pressure obeys 𝔼 p̄ = ∂_y(Ri_b θ̄) − ∇·m, so
δp = −𝔼⁻¹(∇·m). The bound is elliptic regularity (𝔼⁻¹: H⁻¹ → H¹) plus the
Green's-function lower bound; image charges from the walls add only bounded
corrections, so the long-ranged singularity dominates at finite distance. ∎

**Smagorinsky vs projected-FDT.** For Smagorinsky m_Smag = −2∇·(ν_t S̄) and
∇·m_Smag ≠ 0 in general. For the two-clocks closure the force is built as
ŵ_i = ℙ_ij ĝ_j with ℙ_ij = δ_ij − k_ik_j/|k|², so k·ŵ = 0, hence ∇·m_FDT = 0: **no
spurious source by construction.** "Spurious relative to what?" — the true filtered
force ∇·τ has its own (physically correct) curl-free part; projected-FDT sets it to
zero, Smagorinsky sets it to something *uncontrolled*. The honest theorem: Smagorinsky's
curl-free part is an uncontrolled error the Green's function spreads globally.

**Glacier consequence (motivation only, deferred).** N = P_ice − P_water drives
sliding; Schoof-type laws give u_b ∝ N⁻ⁿ, n ∼ 3, so a 1% pressure error → ~3% sliding
error everywhere. This is the momentum-side reason solenoidality is not optional.

**Correction A — this is a pressure/N statement, not (in the bulk) a melt statement.**
A Leray-projected solver updates velocity as ū^{n+1} = 𝕃 ū\*, so 𝕃 m_Smag = 𝕃 m_FDT ⇒
**identical bulk velocity**; the spurious δp is *diagnosed*, not dynamical, in the
bulk. It bites melt only where 𝕃 is inexact — the **penalized wall** — where the leak
(𝕀 − 𝕃_pen)m enters the velocity directly. So solenoidality corrupts diagnosed p̄/N
(→ sliding, deferred), and near-wall melt only through the penalization-boundary error.

**Verified (2D frozen DNS, real BEDMAP1 bed, n = 128; `theory_tests.py::result1`).**

| quantity | Smagorinsky | projected-FDT |
|---|---|---|
| ‖∇·m‖/‖m‖ (nondim) | **5.9** | 9 × 10⁻¹⁵ |
| curl-free (dilatational) energy fraction | 4.2 % | 0.00 % |
| spurious pressure rms(δp)/rms(p_dyn) | 0.5 % (globally spread) | 0.00 % |

Correction A confirmed exactly: ‖𝕃(m_Smag,raw) − m_Smag,proj‖ / ‖𝕃(m_Smag,raw)‖ =
**0** (bit-identical bulk tendency). And the dilatational energy sits in the **open
cavity**, only ~2 % within the solid/interface band — so even the penalization-leak
route is small. Net: on this bed, solenoidality is a **pressure/N (sliding)** effect,
not a bulk-melt one.

### 3.1 Operator universality: the York–Leray common Hodge structure

Theorem 1's *global* pressure corruption from a *local* source is not a quirk of the
Navier–Stokes equations — it is the fluid instance of a structure shared by every
field theory with a differential gauge constraint enforced by an elliptic inverse.

In the ADM 3+1 formulation of general relativity, the extrinsic curvature K_ij is
split into trace and trace-free parts A_ij. The momentum constraint is

  ∇^j A_ij − (2/3) ∇_i K = 0.

York (1973) decomposed A_ij = (LW)_ij + A_ij^TT, where (LW)_ij is the conformal
Killing operator and A_ij^TT is transverse-traceless. The TT projector is

  ℙ^York = 𝕀 − 𝕃(∇·𝕃)⁻¹ ∇· .

Both ℙ^York and the fluid Leray projector 𝕃 = 𝕀 − ∇(∇²)⁻¹∇· are **Hodge-type
orthogonal projectors built by the same algorithm**: (1) define a longitudinal
subspace as the range of a gradient-like operator, (2) take the transverse subspace
as its L²-orthogonal complement, (3) project via the inverse of the elliptic operator
that maps potentials to divergences. They are **members of one class, not isomorphic
structures**: they act on different spaces (vector fields vs. symmetric trace-free
2-tensors) through different elliptic operators (the scalar Laplacian ∇² vs. the
conformal-vector Laplacian ∇·𝕃). Dedner GLM cleaning (§3.2) is a third member.

**Corollary (the algebraic tail is universal).** For any elliptic projector of this
class in d ≥ 2, the response to a localized source Q has an algebraic tail. Labelling
which object carries which exponent (the field is one derivative *above* the
potential, the projector kernel one more):

| object | scaling | 3D | 2D |
|---|---|---|---|
| potential  δφ = 𝔼⁻¹Q | r^{2−d} | r⁻¹ | ln r |
| field  δu = ∇δφ | r^{1−d} | r⁻² | r⁻¹ |
| projector kernel (Calderón–Zygmund) | r^{−d} | r⁻³ | r⁻² |

This is consistent with §1.2/§3, which carry the potential as 1/r in 3D (ln r in 2D).
**Referee defense.** The nonlocality of Theorem 1 is *not* "just a fluid artifact":
it is the same elliptic action-at-a-distance that enforces the ADM momentum
constraint in numerical relativity, and it is unavoidable for any differential gauge
constraint enforced by an elliptic inverse. (The word is "common Hodge structure,"
not "isomorphism" — they are the same *construction*, not a structure-preserving
bijection.)

### 3.2 Dedner GLM cleaning: the approximate-projection threshold (verified)

The exact-Leray solver keeps ∇·ū = 0 every step regardless of the closure
(Correction A), so the "spurious source" has teeth only where projection is
**inexact** — compressible, penalized, or iterative/approximate projection. MHD meets
the identical problem (∇·B = 0 is a gauge constraint, but discretization creates
∇·B = Q ≠ 0) and solves it with the Dedner et al. (2002) mixed GLM system:

  ∂_t B + ∇×(B×u) = −∇ψ,
  ∂_t ψ + c_h² ∇·B = −(c_h²/σ) ψ,

with cleaning wave speed c_h and damping σ. Taking ∇· and eliminating ψ (the curl
term drops since ∇·(∇×·) = 0) gives the **telegrapher equation** for Q = ∇·B:

  ∂_t² Q + γ_clean ∂_t Q − c_h² ∇² Q = 0,   γ_clean = c_h²/σ.

Correspondence with this framework:

| MHD Dedner | this framework |
|---|---|
| c_h (cleaning wave speed) | c (sound speed, Part 4) |
| σ (damping parameter) | τ_corr (memory decorrelation time) |
| Q = ∇·B | Q = ∇·m (subgrid divergence) |
| ψ (cleaning potential) | δp (spurious pressure potential) |
| γ_clean = c_h²/σ | → ∞ (exact Leray), finite (approximate) |

**Theorem (approximate-projection design criterion).** For any solver enforcing
incompressibility by hyperbolic cleaning, penalization relaxation, or iterative
projection rather than an exact Leray step, the cleaning rate γ_clean and the
pressure-adjustment time τ_adj = L/c_eff must satisfy

  γ_clean · τ_adj ≳ O(1).

*Proof sketch.* When γ_clean·τ_adj ≪ 1 the damping is slower than one wave transit:
the violation propagates as an undamped wave and accumulates. When γ_clean·τ_adj ≫ 1
the telegrapher collapses to the parabolic ∂_t Q ≈ (c_h²/γ_clean)∇²Q, suppressing Q
locally before it spreads globally via 𝔼⁻¹. The O(1) crossover is the threshold. ∎

This is the **dimensionally clean, correctly oriented** form of the GR-motivated
damping bound: it is dimensionless (1/T·T), exact Leray (γ_clean → ∞) is the *safest*
limit, and it bites only where 𝕃 is inexact — *not* the broken K(0)·τ_adj > 1/2,
which has units 1/T and would mislabel exact projection as the worst case. There is
**no universal "1/2"** to transfer from the Bona–Massó / Z4 (κ₁, κ₂) analysis (a
second-order-in-time constraint subsystem, structurally unlike a first-order MZ
relaxation); the constant is O(1) and must be *measured per method/geometry*.

**Corollary (timestep separation).** To keep the two clocks separated numerically the
step must resolve the cleaning: Δt ≲ 1/γ_clean. If Δt ≫ 1/γ_clean the fast
cleaning/pressure clock aliases into the slow advective clock in a single step and
the operator separation collapses.

| closure | γ_clean·τ_adj | mechanism | verdict |
|---|---|---|---|
| exact Leray projection | ∞ | instantaneous elliptic solve | satisfies bound; ∇·m = 0 by construction |
| Dedner hyperbolic cleaning | c_h L/σ_eff | finite-speed wave damping | satisfies bound if tuned (σ_eff ≲ c_h L) |
| Smagorinsky, no cleaning | 0 | no constraint enforcement | violates bound; divergence accumulates, spreads via 𝔼⁻¹ |
| spectrum-matched surrogate | ∼ 0 | noise without damping | violates bound; energy right, divergence channel undamped |

**Critical caveat (Correction A).** In an *exact* Leray-projected solver the velocity
update is ū^{n+1} = 𝕃 ū\*, so ∇·ū = 0 regardless of the subgrid model — Smagorinsky
corrupts only the *diagnosed* pressure p̄, not the velocity. The "catastrophic
violation" rows apply only to compressible/penalized/approximate-projection solvers
where 𝕃 is inexact — precisely the regime Dedner cleaning lives in, and where this
bound is operationally meaningful.

**Verified (2D, GPU-free; `theory_tests.py::result_dedner_cleaning`).** Our solver
uses exact spectral Leray, so we deploy the GLM telegrapher *as* the tunable
approximate projector: freeze a BEDMAP1 DNS field, inject the raw-Smagorinsky
divergence (ū\* = ū + Δt·m_Smag ⇒ ∇·ū\* = Δt·∇·m), clean for one transit τ_adj = L/c_h,
and sweep the single dimensionless knob G = γ_clean·τ_adj. The closed-form per-mode
factor is checked against direct RK4 to machine precision. Over one transit the only
free group besides G is the per-mode phase |k|L, so the knee is essentially geometry-
robust (G\*= 2.00 at both n = 96 and n = 128).

| G = γ_clean·τ_adj | 0 | 1 | **2** | 4 | 6 | 12 | 24 |
|---|---|---|---|---|---|---|---|
| ‖∇·ū‖/‖∇·ū\*‖ | 0.72 | 0.44 | **0.27** | 0.10 | 0.036 | 0.0018 | 4e−5 |
| ‖δp‖/‖δp\*‖ (low-k) | 0.75 | 0.45 | 0.27 | 0.10 | 0.037 | 0.0020 | 0.022 |

- **Measured knee G\* ≈ 2.0** (divergence residual at 1/e of its uncleaned value) —
  squarely O(1), confirming the criterion. *This is our number for this bed, not a
  transplanted GR "1/2".*
- **Over-damped optimum G_opt ≈ 12**: beyond it the low-k (pressure) residual rises
  again as the telegrapher's slow over-damped root stalls *global* cleaning — the
  classic Dedner trade-off, so the design target is G ∈ [~2, ~12], not G → ∞.
- **G = 0 spurious-pressure global-tail fraction ≈ 0.25**: with no cleaning, divergence
  sourced in only ~6 % of the area produces a pressure spread globally — the
  Theorem-1 / §3.1 universal tail, measured. As a **nonlocal amplification factor**,
  A = (tail fraction)/(area fraction) ≈ **3.9×**: *a local divergence error in 6 % of
  the domain corrupts 25 % of the global pressure field.* (A = 1 would be a purely
  local response; A > 1 is the elliptic Green's-function spreading, quantified.)

### 3.3 The design window for approximate projection (measured)

The criterion of §3.2 is one-sided (γ_clean·τ_adj ≳ O(1)), but the measurement reveals
a **finite window**, because the telegrapher equation has two roots:

  λ± = −γ/2 ± √( γ²/4 − c_h²|k|² ).

- **Under-cleaned (G ≲ 2).** For small γ the discriminant is negative, λ± = −γ/2 ± iω
  are an under-damped oscillation: the violation propagates as a weakly-damped wave and
  is not removed within one transit (residual ≈ 0.7 at G = 0).
- **Over-cleaned (G ≳ 12).** For large γ and small |k| the discriminant is positive and
  the **slow root λ₋ ≈ −c_h²|k|²/γ → 0** as γ → ∞. The global / low-wavenumber modes are
  not cleaned — they are *frozen*. This is the measured re-growth of the low-k pressure
  residual past G_opt.

**Theorem (approximate-projection design window, measured).** For the BEDMAP1 cavity,

  2.0 ≲ G = γ_clean·τ_adj ≲ 12,

verified by direct measurement on frozen turbulent fields at n = 96 and n = 128
(convergent), with the closed-form per-mode factor matching RK4 to machine precision
(`theory_tests.py::result_dedner_cleaning`). ∎

**Design implication.** Exact Leray projection is **not** the G → ∞ limit of hyperbolic
cleaning; it is a *singular* limit, approached optimally at finite G_opt ≈ 12 and
**degraded beyond it**. At any finite G the low-k residual is non-zero, with a best value
at G_opt. This is the structural reason iterative / penalized projection in ice-sheet and
ocean codes carries a small, persistent low-wavenumber divergence bias: tuned for fast
local cleaning (large γ), they sit *past* the optimum, in the over-damped stall regime,
where the global modes are frozen rather than projected.

---

## 4. Theorem 2 — the Clock Mismatch Number 𝓜_clock (verified)

**Statement.** For a localized source and a field φ, define the **upstream-influence
fraction** Λ_φ = (∫_{x<x₀−ℓ} |Δφ| dx)/(∫_all |Δφ| dx) — the fraction of the response
located *upstream* of the source (streamwise axis; mean flow +x). Define the **Clock
Mismatch Number** 𝓜_clock = Λ_p / Λ_θ. Then 𝓜_clock > 1 because the elliptic pressure
responds in all directions (action-at-a-distance, including upstream) while the
advected scalar is screened upstream (causal, downstream cone).

**Correction B — this is high-Péclet advective screening, not elliptic-vs-parabolic.**
The *steady* scalar also obeys a Laplacian (∇²θ ≈ source), so without advection it
would be as nonlocal as pressure. It is advection that screens it. The discriminator
is **upstream** influence, and the screening length is **longitudinal**,
ℓ_scr = κ/U ∝ Pe⁻¹ — *not* the transverse boundary-layer thickness λ·Pe⁻¹/². Hence
𝓜_clock has **no universal power law in Pe**: it is an empirical diagnostic that grows
modestly then **saturates** once ℓ_scr drops below the grid/cavity scale. A single
local eddy diffusivity cannot carry the elliptic upstream response — that is the
structural mismatch.

**Verified (real BEDMAP1 bed, `theory_tests.py::result2`):**

| Pe | Λ_p | Λ_θ | 𝓜_clock |
|---|---|---|---|
| 1 | 0.41 | 0.31 | 1.31 |
| 10 | 0.41 | 0.26 | 1.57 |
| 100 | 0.41 | 0.27 | 1.52 |
| 300 | 0.41 | 0.27 | 1.51 |
| **real bed (Pe=100)** | 0.41 | 0.27 | **1.52** |

Pressure reaches ~1.5× farther upstream than the advected scalar — robust in sign,
modest in size. **Honest consequence:** 𝓜_bare ≈ 1.5 is *below* the geometry threshold
for a 2–3× factor (§7–§8). On this bed the enhancement is **not** primarily geometric;
it must come from stratification + SGS dominance. The large 𝓜 ≈ 8 seen in a confined
cavity figure is a property of confinement, not of the bare operators. 𝓜_clock is a
**pre-simulation** diagnostic computable from any bed DEM.

---

## 5. Theorem 3 — the Counter-Gradient Parameter C_G (the melt mechanism; measured — partial)

**Statement.** Unlike Theorem 1, *this* is the lever on melt. Define
C_G = ⟨**τ**_b·∇θ̄⟩ / ⟨|**τ**_b||∇θ̄|⟩. A scalar eddy diffusivity forces C_G = −1
(down-gradient by construction); the true flux in a lee recirculation can be
counter-gradient, C_G > −1, because parcels carry the memory of the global pressure
field that swept them there (the commutator of §7 maps thermal memory into
recirculation). K-theory deletes the recirculation by over-dissipating, predicting
C_G = −1 by construction.

**Corrected scaling.** Tying the departure to the mismatch and stratification,

&nbsp;&nbsp;**C_G + 1 = c₁·(1 − 1/𝓜_clock)·g(Ri_b, Re_Δ)**.&nbsp;&nbsp;&nbsp;&nbsp;(13)

At 𝓜_clock = 1 (no operator mismatch) C_G = −1, K-theory valid; as 𝓜_clock → ∞,
C_G + 1 → c₁·g, the maximum counter-gradient set by stratification and Re_Δ. The
departure from K-theory therefore **grows** with the mismatch (this corrects an
earlier inverse form). This is exactly what the §2 backscatter term supplies and
Smagorinsky cannot.

**Measured (RESULT 11, §11.1, partial).** The 3-D active-buoyancy LES confirms the
*qualitative* half: the resolved heat flux is genuinely counter-gradient,
C_G ≈ −0.60 (fluid) / −0.33 (ice-base band), i.e. a departure C_G + 1 ≈ 0.4–0.67
far from the K-theory value −1. But the *quantitative* scaling (13) is **not
supported**: C_G + 1 is invariant in both Ri (flat across [0, 1.5]) and the clock
mismatch (two-clocks vs Smagorinsky agree to ≈ 0.006), and the melt ratio R(Ri)
stays flat at 1.0004 (RESULT 8). The counter-gradient flux is real but is set by
the resolved cavity geometry, not the stratification/memory lever the corrected
scaling proposed. Full analysis: [`REPORT_CG_BUOYANCY.md`](../REPORT_CG_BUOYANCY.md),
data [`figures/52_theorem3_cg.json`](../figures/52_theorem3_cg.json).

---

## 6. Theorem 4 — the pressure–buoyancy cross-spectrum (a memory fingerprint; RB-targeted)

**Statement.** In Fourier space resolved pressure responds to buoyancy through
H_bp(k) = −i k_y/|k|² — instantaneous, no frequency dependence: pressure is an
elliptic *constraint*, no memory. Temperature, by (9), carries an MZ memory kernel.
Hence the cross-spectrum S_bp(ω,k) carries phase coherence at finite frequency that
K-theory (which collapses the temperature memory to a delta) destroys: K-theory gives
S_bp white in ω at each k; the truth has structure at the thermal memory rate. So
S_bp is a **structural test of the single-clock assumption from time series alone**,
no full LES.

**Correction C.** The relevant memory rate is the **thermal** one, κ|k|² (diffusive)
or U|k| (advective) — *distinct* from the buoyancy frequency N_BV, which adds a
*separate* spectral line under stratification.

**Correction D — the non-commutation is not a symbol identity.** As Fourier
multipliers, 𝕃 (symbol 𝕀 − kkᵀ/|k|²) and a *constant-coefficient* heat semigroup
commute exactly — their symbol commutator vanishes identically. The obstruction
`[𝕃, ℙ(t)] ≠ 0` comes entirely from (i) the **spatially varying advection** ū(x)·∇
(so ℙ is not a multiplier) and (ii) the **wall BCs** (so 𝔼⁻¹ is not
translation-invariant), via the Duhamel identity of §7. **Honest risk:** in a
compressible cavity (Part 4) pressure acquires an acoustic memory at ω ∼ c|k| and the
two clocks become three; the incompressible split here is the singular c → ∞ limit.
Best measured on the time-resolved Rayleigh–Bénard fields of Part 2, not the cavity.

---

## 7. The Master Identity `[𝕃, ℙ] ≠ 0` (verified) — why the four theorems are one

Theorems 1–4 are not four claims; they are one **operator non-commutation** seen from
four sides. The Leray projector 𝕃 (global, elliptic) does not commute with the
heat–advection semigroup ℙ(t) (local, parabolic, memory-bearing). By Duhamel,

&nbsp;&nbsp;**[𝕃, ℙ(t)] = ∫₀ᵗ ℙ(t−s)·[𝕃, A]·ℙ(s) ds,&nbsp;&nbsp;A = κ∇² − ū(x)·∇,&nbsp;&nbsp;[𝕃, A] = −[𝕃, ū·∇] ≠ 0.**&nbsp;&nbsp;&nbsp;&nbsp;(14)

- **Theorem 1** = 𝕃 not commuting with the *local subgrid force* (curl-free leak into
  the global pressure).
- **Theorem 2** = the *norm* of the obstruction (elliptic upstream reach vs advective
  screening).
- **Theorem 3** = the obstruction mapping thermal memory into recirculation
  (counter-gradient flux); collapsing ℙ to a delta (K-theory) deletes it.
- **Theorem 4** = its frequency content (the memory line K-theory zeroes out).

**Verified directly (n = 128, real bed; `theory_tests.py::result3_commutator`).** For a
divergence-free probe v₀, since 𝕃v₀ = v₀, the commutator action is exactly the
divergence v₀ *acquires* under advection:
‖[𝕃, ℙ(t)]v₀‖/‖ℙ(t)v₀‖ = ‖dilatational part of ℙ(t)v₀‖/‖ℙ(t)v₀‖.

| advection time t | uniform ū | real cavity ū |
|---|---|---|
| 0.04 | 1 × 10⁻¹⁵ | 0.007 |
| 0.16 | 3 × 10⁻¹⁵ | 0.028 |
| 0.64 | 8 × 10⁻¹⁵ | **0.59** |
| 1.28 | 2 × 10⁻¹⁴ | 0.56 (saturated) |

Uniform flow gives **machine zero** (translation preserves divergence-free, so 𝕃 and
ℙ commute); the real BEDMAP cavity grows ∝ t (the small-t limit of (14)) to an **O(1)
plateau ≈ 0.6** by one eddy-turnover. So `[𝕃, ℙ] ≠ 0` is real and **O(1)**, driven by
**shear + walls** exactly as (14) says — not by the (identically vanishing)
constant-coefficient symbol. The master identity is the spine; the regime equation
(§8) is its quantitative shadow on the melt rate.

### 7.1 The information cost of non-commutation: a Girsanov / Entropy–Pressure identity

It is tempting to dress §7 in the language of conditional expectation and a *failed
tower property* (project-then-evolve ≠ evolve-then-project). **That framing is wrong**
and must be abandoned: the heat–advection semigroup ℙ(t) = exp(t(κ∇² − ū·∇)) **is** a
bona-fide Markov / conditional-expectation operator (positivity-preserving, unital, the
transition density of the parcel SDE dX = ū dt + √(2κ) dW), but the Leray projector 𝕃
is an L²-**orthogonal** projection that is **not** a conditional expectation (neither
positivity-preserving nor unital). There is no tower property to violate — the only
obstruction is the plain non-commutation `[𝕃, ℙ] ≠ 0` of (14). The correct
probabilistic home for that obstruction is not a KL divergence of densities but a
**Girsanov entropy on diffusion paths**, and it turns out to coincide *exactly* with
the Theorem-1 spurious-pressure energy.

**Girsanov entropy of the splitting order.** Compare two candidate effective parcel
drifts that differ only in the order of constraint and transport:

- **Path A (project, then transport):** `b_A = ℙ(t) 𝕃 u₀`
- **Path B (transport, then project):** `b_B = 𝕃 ℙ(t) u₀`

For two diffusions with the same noise √(2κ), the path-measure relative entropy is the
standard Girsanov functional (no conditional expectation, no tower property, no
positivity assumption):

&nbsp;&nbsp;**D_KL(P^A ‖ P^B) = (1/4κ) 𝔼 ∫₀^τ |b_A − b_B|² ds.**&nbsp;&nbsp;&nbsp;&nbsp;(14a)

To leading order in the splitting lag `t`, the drift difference is the **same
commutator measured in RESULT 3**, `b_A − b_B = −t·[𝕃, −ū·∇]u₀ + O(t²)`, so the
entropy **rate** (per unit path-time τ) is

&nbsp;&nbsp;**dD_KL/dτ = (t²/4κ) ‖[𝕃, −ū·∇]u₀‖² + O(t³).**&nbsp;&nbsp;&nbsp;&nbsp;(14b)

(Note the bookkeeping: `t` is the operator-splitting lag and `τ` the path horizon —
written as a flat `t²` the formula would silently drop the `τ/4κ` prefactor and be
dimensionally wrong. It is an entropy *rate*, not a one-shot divergence.)

**The Entropy–Pressure identity (exact).** For a divergence-free field (𝕃u₀ = u₀) the
commutator is *exactly* the curl-free part of the advection — i.e. the Theorem-1
spurious-pressure gradient:

&nbsp;&nbsp;**[𝕃, −ū·∇]u₀ = (𝕀 − 𝕃)(ū·∇u₀) = ∇δp_spur,&nbsp;&nbsp;∇²δp_spur = ∇·(ū·∇u₀).**&nbsp;&nbsp;&nbsp;&nbsp;(14c)

By the Hodge orthogonality `∫|(𝕀−𝕃)w|² = ∫|∇φ|²` with `∇²φ = ∇·w`, the Girsanov
entropy rate is therefore the **H¹ (Dirichlet) energy of the spurious pressure**:

&nbsp;&nbsp;**‖[𝕃, ℙ(t)]u₀‖² = ∫_Ω |∇δp_spur|² dx = −∫_Ω (∇·u)·δp_spur dx.**&nbsp;&nbsp;&nbsp;&nbsp;(14d)

This is the payoff: **information cost = mechanical work of the spurious pressure =
corruption of the effective pressure N** (and hence of the Schoof/Tsai sliding law
u_b ∝ N^{−n}). It is an identity between path-information, operator theory, and glacier
mechanics — *not* an analogy — exact to O(t²) for the linearized path measure.

**Verified (`theory_tests.py::result_commutator_entropy`, n = 128, real bed).**

| check | measured | meaning |
|---|---|---|
| (a) `‖[𝕃,−ū·∇]u₀ − (𝕀−𝕃)(ū·∇u₀)‖/‖·‖` | machine zero (≈ 0) | (14c): commutator **is** the curl-free part of the advection |
| (b) `‖([ℙℒ−ℒℙ]u₀)/t − C‖/‖C‖`, t = 0.02→0.005 | 0.280 → 0.143 → 0.070 | difference quotient → commutator, **linear in t** (validates the Girsanov integrand) |
| (c) `⟨\|C\|²⟩` vs `⟨\|∇δp_spur\|²⟩` | 0.5816 vs 0.5816, **ratio 1.000000** | (14d): path-entropy = H¹ spurious-pressure energy, exact |
| dilatational fraction `‖C‖/‖ū·∇u₀‖` | 0.428 | the curl-free leak; cf. the RESULT-3 plateau ≈ 0.6 |

**Counter-gradient link — a conjecture, kept honest (`≲`).** The lee counter-gradient
flux of §5 is *conjectured* to be the thermal dissipation of this frozen mechanical
energy, bounded by the global entropy cost:

&nbsp;&nbsp;**C_G + 1 ≲ 𝓔_rec / 𝓔_total ∼ (∫_lee u_rec·∇δp_spur) / (∫_cavity |∇δp_spur|²).**

This is **not** a theorem — it needs a minimum-entropy-production variational principle
we have not derived, so the "≲" is empirical and "minimum entropy production" is not
claimed. Consistent measured support on the same frozen field: lee counter-gradient
fraction **≈ 0.49** (reproduces the ~50% of Part 9), with the up-gradient flux
correlating with the parcel-current compressibility |ū·∇θ̄| at **r ≈ 0.49** (and with
(ū·∇θ̄)² at r ≈ 0.41) — a partial mechanism (r² ≈ ¼ of the variance), supportive but
not deterministic. K-theory collapses ℙ to a delta, zeroes this memory, and so
represents none of it.

### 7.2 Symplectic structure and the constraint class (why 𝕃 is nonlocal)

§7.1 reads the non-commutation as an entropy; this subsection reads the *operator*
𝕃 itself. It answers a structural question — *why* must the constraint inverse be
the nonlocal `|k|⁻²` Green's function rather than a local algebraic coefficient? —
and turns the answer into a falsifiable spectral measurement.

**The first-class kernel is unavoidable (exact).** In the Hamiltonian formulation of
ideal fluids (Arnold 1966; Marsden–Weinstein 1974), incompressibility `∇·u = 0` is a
**first-class constraint**: with the naive canonical bracket the constraint matrix is
the *differential operator* `{C(x),C(y)} = ∇²δ(x−y)`, which is singular (its zero mode
is the constants), so the constraint generates a gauge symmetry (fluid relabeling) and
the pressure is its Lagrange multiplier. Elimination is by **symplectic reduction**, and
the "inverse" that appears is the nonlocal Green's function `(∇²)⁻¹` with Fourier symbol
`|k|⁻²`. The Leray projector **𝕃 = 𝕀 − ∇(∇²)⁻¹∇·** is the geometric realization of the
projection onto this constraint manifold — not an algebraic convenience but the
reduction projector itself. (The boxed "Fourier symbol = `|k|⁻²`" is the *definition*
of the Laplacian Green's function — an observation, not a theorem.)

**The local surrogate is a null hypothesis with the wrong symbol (measured).** A
*second-class* caricature would replace the differential constraint matrix by a local,
invertible one — i.e. replace `(∇²)⁻¹` by multiplication by a constant, a **flat `|k|⁰`
symbol** that decouples scales. We falsify this on the frozen BEDMAP cavity by driving
the Poisson kernel with the genuine resolved dynamic-pressure source
`Q_dyn = ∇·(ū·∇ū)` (`p_dyn = (∇²)⁻¹Q_dyn` the true first-class pressure) and comparing
the exact response to the local surrogate `p_local = c·Q_dyn` (variance-matched at a
mid shell):

| quantity (n=128, frozen BEDMAP) | value |
|---|---|
| symbol sanity `‖p̂_dyn + Q̂_dyn/\|k\|²‖/‖·‖` | **3.5×10⁻¹⁶** (machine zero) |
| spectral-ratio slope `d ln(\|p̂_dyn\|²/\|p̂_local\|²)/d ln k` | **−3.95** (≈ −4) |
| true `p_dyn` low-k energy fraction (`\|k\|<16`) | **0.990** |
| local surrogate low-k energy fraction (`\|k\|<16`) | **0.384** |

The slope **−3.95** is the exact `|k|⁻²` operator signature in amplitude-squared space;
a local operator has a flat `|k|⁰` symbol and **cannot** produce it (this number is
convention-free — independent of any pressure sign choice). The low-k contrast is the
kernel's doing: the true response concentrates ~99% of its energy at large scales, while
the local surrogate inherits only the source's own ~38%. This is a **falsified null
hypothesis** — the subgrid constraint response is *not* local.

**Where the corruption actually comes from (measured).** The physically correct subgrid
force in this framework is Leray-projected, hence **divergence-free by construction**
(dilatational fraction `0.000`): its spurious pressure is machine zero — the correct
closure lives *on* the constraint manifold and does **not** corrupt `N`. The corruption
is entirely **model error**: Smagorinsky's raw force `m_Smag = ∇·(2ν_t S̄)` carries a
small dilatational part that the *same unavoidable* `|k|⁻²` kernel spreads globally,
`δp_Smag = (∇²)⁻¹∇·m_Smag`:

| field | rms / p_dyn | phase corr w/ p_dyn | low-k frac (`\|k\|<16`) | dilatational frac of source |
|---|---|---|---|---|
| `p_dyn` (resolved, true) | 1.000 | 1.000 | 0.990 | — |
| exact subgrid force | ~10⁻¹⁵ (solenoidal **by construction**) | — | — | 0.000 |
| `δp_Smag` (spurious) | **0.0046** | **−0.008** | **0.602** | 0.042 |

Smagorinsky's spurious divergence (~4% of its force) is injected into the global kernel
and emerges as a pressure ~0.5% of `p_dyn` but **phase-decorrelated** (corr ≈ 0) and
**large-scale** (60% low-k) — a phantom, wrong-pattern `N` field corrupting sliding
everywhere. This is the spectral face of the Theorem-1 amplification `A ≈ 3.9×`: a
*local* model error, forced through the unavoidable first-class kernel, produces a
pressure perturbation whose footprint is ~4× the source.

**What K-theory actually does.** It does **not** "use a local pressure kernel" — the
solver's pressure solve still applies `(∇²)⁻¹` with symbol `|k|⁻²` to the *resolved*
field, identically to the exact dynamics. What it does is **manufacture a spurious
divergence source at the subgrid level** that the true (solenoidal) subgrid dynamics
never produces, then force that source through the unavoidable global kernel. The
"second-class caricature" is therefore strictly an **operator-level analogy at the
subgrid scale** (eddy viscosity is a local algebraic stress with no subgrid Poisson
coupling) — not a claim that practitioners use Dirac brackets, and not a statement about
the resolved-pressure solve, which is the same `|k|⁻²` inverse in both.

| claim | status |
|---|---|
| `∇·u=0` is first-class in Hamiltonian fluid mechanics | exact (Arnold; Marsden–Weinstein) |
| 𝕃 is the constraint-manifold (symplectic-reduction) projector | exact |
| the Poisson inverse `(∇²)⁻¹` has symbol `\|k\|⁻²` and a global tail | exact (definition of the Green's function) |
| a local surrogate has flat `\|k\|⁰`; the true response has measured slope **−3.95** | **measured** (BEDMAP cavity) |
| Smagorinsky's spurious divergence is the dominant corruption of `N`; the exact subgrid force is solenoidal by construction | **measured** + definitional |
| K-theory's subgrid operator content is the "second-class caricature" | structural analogy (operator-level, subgrid only) |

(Reproducible via `python -m subglacial.theory_tests --n 128`, RESULT 6.)

---

### 7.3 Theorem 8 — the Roughness–Scale Separation criterion (measured)

The non-commutation `[𝕃, ℙ] ≠ 0` has a purely **spectral** corollary that needs *no
DNS at all* — only the bed DEM and the operator structure. Pressure and the
thermal/form-drag response read **different moments** of the same bed spectrum
`E_h(k) = |ĥ(k)|²`, so a single local diffusivity (K-theory) cannot serve both.

**The claim.** Let the bed have a power-law spectrum `E_h(k) ∝ k^{-α}`. Then

- **Pressure responds to elevation** through the elliptic kernel. Its symbol `|k|⁻²`
  reddens the response: the pressure-felt fraction
  `F_p(>λ) = (Σ_{k<2π/λ} |k|⁻² E_h) / (Σ |k|⁻² E_h)` concentrates at the **largest**
  (cavity) scales.
- **Form drag and heat flux respond to bed slope**, whose variance spectrum
  `k² E_h ∝ k^{2-α}` is nearly flat / blue and is concentrated at the **smallest**
  scales. The thermal boundary layer screens scales below `ℓ_scr = κ/U`, so the
  slope variance below `ℓ_scr` is a **blind zone**: roughness pressure feels globally
  but the thermal field never reaches.

**Measurement (real BEDMAP1 transect, 4029 pts, 220 km, 1757 m relief).** Using the
repo's mirror-periodic embedding (`bedmap._embed_periodic`) to suppress the seam
leakage of this very red bed:

| quantity | value | reading |
|---|---|---|
| elevation slope `α` (log-binned) | **≈ 2.2** (estimator range 2.0–2.5) | natural red bedrock |
| slope-variance exponent `k² E_h` | `∝ k^{-0.2}` (nearly flat) | slope roughness lives at small scales |
| pressure-felt `F_p(>50 km)` | **0.996** | pressure is a cavity-scale quantity |
| pressure-felt `F_p(>20 km)` | **1.000** | the canyon, not the cobbles |
| **slope blind zone** `λ<300 m` | **≈ 0.17** | form-drag roughness pressure feels but thermal screens |
| **slope blind zone** `λ<1 km` | **≈ 0.41** | …and it grows monotonically toward fine scales |
| slope blind zone `λ<5 km` | **≈ 0.85** | |
| elevation blind zone `λ<20 km` (control) | **≈ 0.02** | red spectrum ⇒ elevation blind zone is tiny |

**The decoupling, in one line.** *Pressure sees the canyon; drag sees the cobbles.*
The elliptic pressure field is dominated by the cavity-scale topography (>99 % above
50 km), while the form-drag/heat-flux response is dominated by sub-kilometre slope
roughness (~41 % below 1 km). K-theory's single local diffusivity assumes the two
fields see the same bed — Theorem 8 measures that they do not.

**Crossover.** For **molecular** κ the screening length `ℓ_scr = κ/U` (∼µm) is far
below every bed feature, so both fields are dominated by the cavity scale and the
crossover is `λ* ≈ L ≈ 220 km`.

**Honest limit (resolution).** For realistic **eddy** κ (`κ ∼ 10⁻⁴–10⁻² m² s⁻¹`,
`U ∼ 0.1 m s⁻¹`), `ℓ_scr ∼ 1 mm–0.1 m` — and even the predicted `λ* ∼ 1–10 m`
crossover sits **below the DEM Nyquist (~150–175 m)**. The blind-zone fractions above
are therefore **measured at resolvable scales (λ > Nyquist)**; the sub-Nyquist
1–10 m regime is reached only by **extrapolating** the `α ≈ 2.2` power law (the slope
spectrum does not roll off, so the blind zone only *grows* below Nyquist). The
fractions are also estimator-dependent (un-mirrored FFT inflates them to ~0.46/0.78
via leakage; the mirror-periodic values ~0.17/0.41 are the reported, leakage-free
numbers). The qualitative ordering — slope blind zone ≫ elevation blind zone, and
pressure large-scale vs drag small-scale — is robust across all estimators.

**Felt-roughness ratio (not a drag-coefficient ratio).** The natural temptation is to
turn the blind zone into a ratio of two effective drag coefficients,
`A_drag = C_d^(p)/C_d^(θ)`, with `C_d^(p)` weighting `C_d(k)` by the pressure spectrum
`k⁻²E_h` and `C_d^(θ)` by the screened spectrum. **This collapses to O(1) for a red bed
and must be retired.** Both are *normalized* averages, so for a constant `C_d(k)` the
ratio is identically 1 (algebra), and for `C_d(k)∝k^β` with `α≈2.2` both integrals are
IR-dominated by the cavity scale `k_L`, giving `A_drag ≈ 0.87–1.0` regardless of the
small-scale structure. (The `≫1` scaling `(L/λ*)^{α+β−3}` only holds for a near-white
bed, `α≲1`.) The honest quantity is the **un-normalized** felt roughness — *which*
slope variance each process samples, not a ratio of mean coefficients:

  𝓕_p = ∫ k²E_h dk  (form drag / pressure: all scales up to the cutoff),
  𝓕_θ = ∫_{k<2π/ℓ_scr} k²E_h dk  (thermal: only λ>ℓ_scr),
  **ℛ_felt = 𝓕_p / 𝓕_θ ≥ 1.**

| cutoff | `λ*=5 km` | `λ*=1 km` | `λ*=10 m` |
|---|---|---|---|
| **measured** (cutoff = Nyquist) | **6.5** | **1.7** | — (sub-Nyquist) |
| extrapolated (η = 1 m power-law) | — | 32 | 5.1 |

Pressure / form drag feel **100 %** of the slope variance; the thermal field feels only
**15 % (λ*=5 km) – 59 % (λ*=1 km)** at resolvable scales, and `ℛ_felt` only grows toward
finer `η` (`ℛ_felt ∝ η^{-(3-α)} ≈ η^{-0.78}`, unbounded as `η→0`). This is the structural
form of the Schoof–Hewitt puzzle (drag from sliding over-predicting basal melt by
2–5×): the discrepancy is a *sampling* mismatch between two operators, not a tuning
error in a single `C_d`.

(Reproducible via `python -m subglacial.theory_tests`, RESULT 7 — DEM-only, no DNS.)

---

## 8. The regime equation: bare R₀(D), stratified R(Ri), and the additive master form

Define the **enhancement factor** (the melt factor we report):
**R ≡ q_ice(two-clocks) / q_ice(Smagorinsky)**.

### 8.1 Bare case (Ri_b = 0) — and why it explains our null result

The only structural difference acting on **q** in the unstratified limit is that
backscatter sustains lee-wake TKE that Smagorinsky drains. Its size is set by the
**SGS dominance ratio** D ≡ ε_sgs/ε_mol (`dissipation_breakdown()`). A transparent
reduced model — backscatter returns a fraction β of the part the SGS actually
controls, ∝ (1 − 1/D)₊:

&nbsp;&nbsp;**R₀(D, β) ≈ 1 + c·β·(1 − 1/D)₊**,&nbsp;&nbsp;c = O(1).&nbsp;&nbsp;&nbsp;&nbsp;(15)

> **When the grid resolves the flow, D → 1 and R₀ → 1.** A resolved simulation has
> nothing for the closure to do — independent of how turbulent the flow is.

This is exactly our negative result: a-posteriori runs at n = 48 (3D) and n = 192 (2D,
fully turbulent) gave R ≈ 1.00–1.02 because D ≈ 1 there. **The theory predicts its own
null.** The 2–3× does not come from the bare case.

### 8.2 Stratified correction (Ri_b > 0): the second clock

Stratification adds a buoyancy oscillator N_BV² = Ri_b·(∂_y θ̄ / scale) and the
**gradient Richardson number Ri = N_BV²/|S̄|²** — a genuine second physical clock that
makes K(τ) oscillatory and partitions FDT energy between mixing and internal waves.
The two closures respond differently:

- **Smagorinsky:** stratification enters only through a stability function,
  κ_t → κ_t·f_h(Ri), f_h ≈ 1/(1 + a·Ri) monotone decreasing (Mellor–Yamada /
  Munk–Anderson). Mixing shuts off **smoothly, monotonically**; no overturning events.
  q_K(Ri) ≈ q₀/(1 + a·Ri).
- **Two-clocks:** FDT backscatter injects the TKE to **intermittently overturn**
  against buoyancy — the counter-gradient transport a positive-definite κ_t cannot
  produce — recovering a fraction γ of the discarded mixing until a cutoff Ri_c:
  q_2c(Ri) ≈ q₀·[ f_h + γ(1 − f_h)(1 − Ri/Ri_c)₊ ]. Forming the ratio,

&nbsp;&nbsp;**R(Ri) ≈ R₀ + γ·a·Ri·(1 − Ri/Ri_c)₊**,&nbsp;&nbsp;&nbsp;&nbsp;(16)

which **peaks at Ri\* = Ri_c/2** with R_max ≈ R₀ + γ·a·Ri_c/4. For representative
a = 8–10, γ = 0.6–0.7, Ri_c ≈ 1: **R_max ≈ 2.5–2.8 at Ri ≈ 0.5**; R → 1 as Ri → 0 and
again as Ri ≳ Ri_c (buoyancy shuts mixing in both closures).

### 8.3 The unified additive regime equation

Each theorem contributes an independent, bounded, non-negative term that vanishes at
its own K-theory limit:

&nbsp;&nbsp;**R(D, Ri, 𝓜) = 1 + c·β·(1 − 1/D)₊ + γ·a·Ri·(1 − Ri/Ri_c)₊ + α·(1 − 1/𝓜)**.&nbsp;&nbsp;&nbsp;&nbsp;(17)

| term | vanishes when | meaning (theorem) |
|---|---|---|
| c·β·(1 − 1/D)₊ | D → 1 (resolved) | SGS dominance (§8.1; explains our null) |
| γ·a·Ri·(1 − Ri/Ri_c)₊ | Ri → 0 or Ri ≥ Ri_c | stratification hump (§8.2; peaks at Ri_c/2) |
| α·(1 − 1/𝓜) | 𝓜 → 1 (no mismatch) | operator/geometry effect (Theorems 2–3) |

The additive form is the **only** one consistent with the three independent R → 1
limits (D → 1, Ri → 0, 𝓜 → 1). It also exposes the honest accounting for **this** bed:
with the verified 𝓜 ≈ 1.5 the operator term is only α·(1 − 1/1.5) = α/3, so unless
α ≳ 3 the 2–3× must come from the **stratification and SGS** terms — i.e. the
two-clocks closure wins here not because the geometry is nonlocal, but because it
preserves the pressure well that stratification intermittently overturns.

### 8.4 The predicted regime diagram and the falsifiable sentence

```
   R(Ri)
  3 |                  . - .            <- two-clocks, D >> 1 (under-resolved LES)
    |               .'       '.
  2 |             .'           '.
    |           .'               '.
 1.5|        . '                   ' .
    |     .'                          ' . _ _ _   <- two-clocks, D ~ 2 (marginal)
  1 |__.'________________________________________  <- both closures, D -> 1 (resolved)
    +----+----+----+----+----+----+----+----+--- Ri
      0  0.1  0.2 0.3  0.4  0.5  0.6  0.8  1.0
```

> **Pre-registered, falsifiable (before any GPU run):** for an under-resolved LES with
> D ≳ 5 (Re_Δ ≳ 10³), Pe_Δ ≳ 10², and a stably stratified ice-base layer at
> Ri ≈ 0.2–0.5, the two-clocks closure is predicted to enhance basal melt by R ≈ 2–3
> relative to Smagorinsky; for Ri → 0, R → 1 (consistent with our resolved tests); for
> Ri ≳ Ri_c ≈ 1, R → 1 again.

A run landing **on** this curve confirms the mechanism; a large factor at Ri = 0 (⇒
dynamic 𝓜 ≫ 𝓜_bare) or a flat R(Ri) refutes it. Either outcome is reportable.

---

## 9. Dimensionless groups, scope boundary, and what the simulation must do

**Controlling groups:**

| group | definition | role |
|---|---|---|
| Re_Δ | ūΔ/ν | sets SGS dominance D; need ≳10³ for D≫1 |
| Pe_Δ | ūΔ/κ = Re_Δ·Pr | heat-carrying scale; O(10²) in cavity flow |
| **Ri** | N_BV²/\|S̄\|² | stratification kill-switch; factor peaks at Ri≈Ri_c/2 |
| **D** | ε_sgs/ε_mol | LES "under-resolved" proxy; R₀→1 as D→1 |
| **𝓜** | Λ_p/Λ_θ | clock mismatch (Theorem 2); operator term in (17) |
| **𝒞** | τ_mem/τ_turn | two-clock separation; memory matters when O(1) |
| β, γ | backscatter / recovery fraction | amplitude of enhancement |
| St | ρ_i L_f/(ρ_w c_p ΔT) | Stefan — **excluded** (no phase change) |
| ε_r | bed relief / cavity height | roughness — **smooth-wall limit assumed** |

**Cascading couplings — handled vs deferred:**

| coupling | in this paper? | how |
|---|---|---|
| **Stratification / buoyancy** | **Yes** | §8.2, the Ri-perturbation |
| **Effective pressure → sliding → frictional heat** | **Motivation only** | §3: ∇·m = 0 ⇒ clean N. Not simulated. |
| **Moving boundary / Stefan feedback** | **Deferred** | fixed cavity; next coupling layer |
| **Multiscale roughness / form drag** | **Deferred** | smooth-wall limit; roughness may parameterize some backscatter |
| **Double diffusion / salinity** | **Deferred** | single buoyancy scalar; true marine BL is thermohaline |

Naming and deferring these honestly is deliberate: the community trusts a stated scope
boundary more than a silent one.

**What the LES (`run_subglacial3d.py`, GPU via CuPy) must do to *test*, not tune:**

1. **Reach the under-resolved regime:** report D and turbulence intensity with every
   melt number; trust R only where D ≫ 1 (else §8.1 gives R≈1 trivially).
2. **Implement active stabilising buoyancy** at the ice base (regime S); the
   passive-scalar runs are the Ri = 0 point and *must* give R ≈ 1.
3. **Sweep Ri** to trace R(Ri) and check for the hump at Ri ≈ Ri_c/2 — not a point.
4. **Measure 𝒞** (Lagrangian memory/turnover) to confirm the clocks are separated
   where the factor appears.
5. **Report C_G** (Theorem 3) alongside R(Ri): the melt enhancement and the departure
   C_G > −1 should co-locate (in Ri and in the lee wake) — they are the same mechanism.

Confirmation = measured R(Ri) matches (16) in shape and locates the 2–3× near Ri ≈ 0.5.
Refutation = flat R(Ri), or large R at Ri = 0. Either is reportable and honest.

---

## 10. Numerical verification summary (GPU-free, reproducible)

All structural numbers above come from `subglacial/theory_tests.py` — a frozen-field
DNS over the real BEDMAP1 transect (n = 128, spin-up 400 steps), using only the
existing 2D solvers, no turbulence model active, no GPU. Run:

```
python -m subglacial.theory_tests --n 128 --spinup 400
```

| structural prediction | status | measured value |
|---|---|---|
| **Theorem 1** Spurious Pressure | **verified** | ‖∇·m‖/‖m‖ = 5.9 (Smag) vs 9e-15 (FDT); δp = 0.5 % of p_dyn; bulk-inert diff = 0; nonlocal amplification A ≈ 3.9× (6 % of area → 25 % of global δp) |
| **Theorem 2** Clock Mismatch 𝓜 | **verified** | 𝓜 ≈ 1.5 (real bed); empirical, saturating (κ/U ∝ Pe⁻¹; no power-law) |
| **Master identity [𝕃,ℙ]≠0** | **verified** | ‖[𝕃,ℙ]v₀‖/‖ℙv₀‖ ≈ 0.6 (cavity) vs 1e-15 (uniform) — O(1), shear/wall-driven |
| **§3.2–3.3 Dedner cleaning window** | **verified** | design window 2.0 ≲ G = γ_clean·τ_adj ≲ 12 (knee at 1/e; over-damped stall past G_opt ≈ 12); the measured O(1), not GR's "1/2" |
| **§7.1 Girsanov / Entropy–Pressure** | **verified** | ‖[𝕃,ℙ]u₀‖² = ∫\|∇δp_spur\|² **exactly** (ratio 1.000000; commutator = curl-free advection to machine zero); path-entropy = H¹ spurious-pressure energy = corruption of N. Counter-gradient link kept conjectural (≲): lee frac ≈ 0.49, corr ≈ 0.49 |
| **§7.2 Constraint class (first-class)** | **verified** | local surrogate falsified: spectral-ratio slope ≈ −3.95 (the `\|k\|⁻²` symbol; a local `\|k\|⁰` op cannot fake it); true p_dyn 99 % low-k vs local 38 %. Corruption is model error: exact subgrid force solenoidal (δp ≈ 0), δp_Smag ≈ 0.5 % of p_dyn, phase-decorrelated (corr ≈ 0), 60 % low-k — phantom N via the same kernel (cf. Thm-1 A ≈ 3.9×) |
| **§7.3 / Theorem 8** Roughness–scale separation | **verified** | DEM-only: pressure-felt `F_p(>50 km) ≈ 0.996` (cavity-scale) vs bed-**slope** blind zone ≈ 0.17 (λ<300 m) / 0.41 (λ<1 km) / 0.85 (λ<5 km); elevation blind zone ≈ 0.02 (control). α ≈ 2.2. Felt-roughness ratio ℛ_felt = slope-var(form drag)/slope-var(thermal) = 1.7 (λ*=1 km) / 6.5 (λ*=5 km) measured, ≈ 32 (λ*=1 km) / 5.1 (λ*=10 m) extrapolated to η=1 m; the *normalized* drag-coeff ratio A_drag is retired (≡ O(1) for a red bed). Eddy-κ crossover λ*∼1–10 m is sub-Nyquist (extrapolated) |
| **Theorem 3** Counter-Gradient C_G | **measured — partial** | C_G ≈ −0.60 (fluid) / −0.33 (band) ⇒ counter-gradient present (C_G+1 ≈ 0.4–0.67 ≫ 0); but Ri- **and** closure-invariant, so scaling (13) C_G+1 ∝ (1−1/𝓜)·g(Ri) **not supported** (3-D buoyancy LES, RESULT 11, §11.1) |
| **Theorem 4** p–b cross-spectrum | prediction | memory line at κ\|k\|²/U\|k\|; best on Part-2 Rayleigh–Bénard time series |
| **Regime factor R(D,Ri,𝓜)** eq (17) | **partial null** | R₀ ≈ 1 verified; hump at Ri ≈ 0.5 **not supported** — R flat at 1.0006 across Ri ∈ [0,1.5] (Direction C, RESULT 8) |

The point of putting these numbers next to the theorems is that a closure paper that
*predicts and then measures* its own structural diagnostics is much harder to dismiss
as tuning than one that reports a single melt ratio.

---

## 11. Stratification resonance test (Direction C, null — RESULT 8)

**Hypothesis:** The regime equation predicts a melt enhancement hump at
Ri* = Ri_c/2 due to resonant coupling between the subgrid memory time
τ_mem and the buoyancy period N_BV⁻¹. When τ_mem · N_BV ≈ 2π,
stochastic backscatter injects energy at the buoyancy frequency,
resonantly amplifying mixing.

**Test:** 3D cavity DNS with active Boussinesq buoyancy (g′ = Ri ·
(θ − θ_ref), θ_ref = mean cavity temperature), Ri ∈ [0, 0.25, 0.5,
0.75, 1.0, 1.5], white-FDT (bs_tau=0) vs. colored-FDT (bs_tau=0.05),
n=64, 3 seeds per point. Backend: CuPy/GPU (Kaggle T4 × 2).

**Result:** R(colored/white) = 1.0006 ± 0.0003 flat across all Ri. No
hump in either closure.

| Ri | R(colored/white) | τ_mem (measured) | N_BV | τ·N | ≈2π? |
|----|-----------------|-------------------|------|------|------|
| 0.00 | 1.0006 | 3.16e-3 | 0.00 | 0.000 | no |
| 0.25 | 1.0006 | 3.17e-3 | 0.15 | 0.000 | no |
| 0.50 | 1.0006 | 3.17e-3 | 0.22 | 0.001 | no |
| 0.75 | 1.0006 | 3.17e-3 | 0.26 | 0.001 | no |
| 1.00 | 1.0006 | 3.17e-3 | 0.30 | 0.001 | no |
| 1.50 | 1.0006 | 3.17e-3 | 0.37 | 0.001 | no |

**Mechanism:** Spatial averaging of N³ = 262,144 independent OU processes
decorrelates the mean-field force with τ_mem^(eff) ≈ 3.17 × 10⁻³ ≪
τ_mem^(set) = 0.05 (a 16× CLT effect). The FDT variance constraint pins
instantaneous forcing amplitude identically for both closures, so temporal
recoloring washes out of the time-averaged melt flux. The product
τ_mem^(eff) · N_BV ≈ 0.001 ≪ 2π, so the resonance condition is never
approached.

> **Diagnostic correction (post-run).** The τ_mem column above was produced
> by an earlier version of the Direction-C diagnostic that converted the
> autocorrelation lag to physical time using `dt` instead of the true sample
> spacing `RECORD_EVERY · dt` (RECORD_EVERY = 3 in `direction_c_gpu_probe.py`),
> and whose `record_sgs_force` advanced the OU/RNG state an extra time on every
> recorded step. Both bugs are now fixed. The sampling-interval fix alone
> rescales τ_mem by ×RECORD_EVERY (≈ 3), giving τ_mem^(eff) ≈ 9.5 × 10⁻³ — an
> ≈ 5× (not 16×) CLT shortening relative to τ_mem^(set) = 0.05. This is still
> τ_mem^(eff) ≪ τ_mem^(set) and τ_mem^(eff) · N_BV ≈ 0.003 ≪ 2π, so the null
> conclusion is unchanged. The exact post-fix values should be regenerated on
> GPU; the table is retained as the original run artifact.

**Falsification:** The regime equation hump is not produced by this
mechanism at these parameters. If the hump exists, it must come from a
different physical process (intermittent convective plumes,
double-diffusive layering, or ice-base roughness feedback).

**Implication:** The colored-FDT closure is structurally correct
(divergence-free, FDT-consistent, backscatter-capable) but its temporal
memory does not couple to buoyancy oscillations in a way that amplifies
melt. The memory kernel’s role is energy conservation and spectral
fidelity, not stratification resonance.

The stratification term γ a Ri(1 − Ri/Ri_c)_+ in the regime equation
remains a heuristic scaling argument; the DNS does not support the
proposed resonance mechanism. The additive form without the hump:

> R = 1 + cβ(1 − 1/D)_+ + α(1 − 1/𝓜)

is the supported prediction. Any observed stratification enhancement would
require a different physical explanation.

(Reproducible via `direction_c_gpu_probe.py` on any CuPy-capable GPU, or
NumPy fallback on CPU at ~30 min wall time.)

### 11.1 Counter-Gradient parameter C_G across the Ri sweep (Theorem 3, RESULT 11)

**Hypothesis (Theorem 3, §5).** The melt lever is the departure of the resolved
turbulent heat flux from down-gradient (K-theory) transport, measured by
C_G = ⟨**F**·∇θ̄⟩ / ⟨|**F**||∇θ̄|⟩ with **F** the resolved turbulent heat flux
**F** = ⟨**u**′θ′⟩ (this solver imposes no SGS closure on θ, so **F** is the
LES-resolved flux and ∇θ̄ the time-mean gradient; C_G = −1 is pure down-gradient,
C_G > −1 is counter-gradient). Eq (13) predicts the departure **grows with the
clock mismatch** and is stratification-modulated: C_G + 1 = c₁·(1 − 1/𝓜)·g(Ri, Re_Δ).
If true, the counter-gradient departure and any melt enhancement R(Ri) should
**co-locate** (§9 item 5) — same mechanism.

**Test.** Same 3-D active-buoyancy LES and Ri grid as RESULT 8
(`theorem3_cg_gpu_probe.py`, n = 64, spin-up 400, measure 600, sample every 5,
3 seeds), accumulating ⟨θ̄⟩, ⟨ūᵢ⟩ and ⟨uᵢθ⟩ over the window to form
**F**ᵢ = ⟨uᵢθ⟩ − ⟨uᵢ⟩⟨θ⟩ and ∇θ̄ spectrally, for **white-FDT** (Smagorinsky,
bs_tau = 0) vs **colored-FDT** (two-clocks, bs_tau = 0.05). C_G is reported over
the whole fluid and over the ice-base band. Backend: CuPy/GPU (Kaggle P100, wall
553 s). Data: [`figures/52_theorem3_cg.json`](../figures/52_theorem3_cg.json).

| Ri | C_G fluid (2-clocks) | C_G band (2-clocks) | C_G fluid (Smag) | C_G band (Smag) | R(2c/Smag) | D = ε_sgs/ε_mol |
|----|------|------|------|------|------|------|
| 0.00 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |
| 0.25 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |
| 0.50 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |
| 0.75 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |
| 1.00 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |
| 1.50 | −0.5980 | −0.3256 | −0.6038 | −0.3208 | 1.0004 | 1.24 |

(seed scatter σ(C_G fluid) ≈ 3.4 × 10⁻³, σ(melt) ≈ 8.3 × 10⁻⁸.)

**Result — a *partial* confirmation, honestly split into two halves.**

1. **The counter-gradient flux is real and substantial (the qualitative half is
   supported).** C_G sits at ≈ −0.60 in the fluid and ≈ −0.33 in the ice-base band,
   far above the K-theory value −1: the departure is C_G + 1 ≈ 0.40 (fluid) / **0.67
   (band)**. The resolved lee heat flux is genuinely *not* down-gradient — the same
   mechanism §7.1 and `REPORT_SUBGLACIAL.md` §4 see in the *exact subgrid* flux on a
   frozen field is now confirmed in the *resolved* flux of a live 3-D buoyant LES,
   and it is strongest exactly where it matters, at the ice base. The regime is
   SGS-dominant (D ≈ 1.24 > 1), so this is the meaningful regime, not the §8.1 trivial
   limit.

2. **The predicted *scaling* (13) is not supported (the quantitative half fails).**
   C_G is **Ri-invariant** (flat to four decimals across Ri ∈ [0, 1.5]) and
   **closure-invariant** (two-clocks vs Smagorinsky differ by only ≈ 0.006 in the
   fluid and ≈ 0.005 in the band — within ≈ 2σ, and in *opposite* directions in the
   two regions, i.e. noise-level). Eq (13) requires C_G + 1 to **grow with the clock
   mismatch 𝓜** (backscatter on vs off) and to depend on Ri; neither happens. The
   departure is a near-constant set by the resolved cavity geometry, not the
   stratification- and memory-tunable lever the theorem proposed.

**Mechanism (why it is closure- and Ri-blind here).** The SGS model acts on
*momentum*; θ is advected by the resolved velocity with no SGS heat closure, so
C_G measures the alignment of the *resolved* flux. Both closures produce nearly
identical resolved flow fields for the same reason RESULT 8 gives a flat melt
ratio: spatial CLT averaging over N³ cells decorrelates the mean-field backscatter
force (τ_mem^eff ≪ τ_mem^set), so temporal recoloring washes out of the
time-averaged transport. And the buoyancy stays weak over this grid
(N_BV ≤ 0.37, τ_mem·N_BV ≈ 10⁻³ ≪ 2π, RESULT 8), so changing Ri barely perturbs
the flow and hence C_G.

**Co-location verdict (§9 item 5).** The melt ratio R(Ri) is flat (1.0004, matching
RESULT 8's 1.0006) **and** C_G is flat: the two diagnostics agree, but they agree
on a *null in Ri*. There is no co-located hump because there is no hump in either
quantity at these parameters — consistent with, and a sharper statement of,
RESULT 8. The melt flux does creep up monotonically with Ri (2.368 → 2.375 × 10⁻⁵,
≈ 0.3 % over Ri 0 → 1.5) but identically for both closures, so R is unmoved.

**Verdict: PARTIAL.** Counter-gradient transport (C_G > −1) is **measured and
confirmed present** — the qualitative melt mechanism of Theorem 3 is real — but its
predicted dependence on 𝓜_clock and Ri (eq 13) is **not supported** at n = 64:
C_G + 1 ≈ 0.4–0.67 is closure- and Ri-invariant. This closes the "to be measured"
status of Theorem 3 with an honest partial result rather than a tuned confirmation.
Full analysis: [`REPORT_CG_BUOYANCY.md`](../REPORT_CG_BUOYANCY.md).

(Reproducible via `theorem3_cg_gpu_probe.py` on any CuPy-capable GPU, or NumPy
fallback on CPU; the C_G alignment math is unit-tested in
`tests/test_theorem3_cg.py`.)


---

## 12. Minimal Stefan prototype (Option-3 scope, RESULT 9)

§1–§8 are structural properties of the operator split in a **fixed-geometry**,
single-phase, incompressible flow. The melt-driven *roughness feedback*
(Candidate 3) cannot exist there: a flat, fixed ice base has no protrusion to
melt differentially. §12 records a minimal, falsifiable test of the feedback
loop — melt → boundary motion → flow change → melt — built on the existing
Brinkman penalty by letting the ice-base level evolve.

**Implementation** (`subglacial/stefan_prototype.py`). 2-D periodic spectral
cavity, wall at `y=0` (bed), ice base at `y = H_c(x,t)`. The penalty mask is a
level set `χ_ice = ½[1 + tanh((y − H_c(x))/δ)]`, rebuilt every `N_mask` steps.
The boundary moves by the explicit Stefan condition

> ∂H_c/∂t = m(x,t)/St,  m(x,t) = −κ ∂θ/∂y|_{H_c},  St = ρ_i L_f /(ρ_w c_p ΔT),

melting (m>0) raising the base so the cavity grows. The flow is driven by a body
force `f_body(t) = f₀ + df·sin(ω_tide t)` only — **the mass-flux controller is
disabled** — so stratification/geometry are free to reshape the profile. The
rigorous <2 % validation of the underlying 1-D enthalpy core against the Neumann
similarity solution is in `subglacial/moving_boundary/` (10 tests).

**What it shows (RESULT 9).**

| Test | Setup | Result |
|------|-------|--------|
| A | flat, no forcing | recovers the Stefan `s(t) ∝ √t` law; `s²`–`t` linear (R² > 0.999), similarity slope within ~15–40 % of analytic Neumann (diffuse-interface accuracy, resolution dependent) |
| B | flat, steady forcing | melt(forced)/melt(no-flow) ≈ **1.00** — gentle horizontal flow does **not** enhance conduction-limited melt |
| C | wavy, steady forcing | melt rate is **differential** and tracks the geometry, corr(m, base) ≈ −0.9; roughness `σ_h` evolves |
| D | wavy, tidal forcing | `σ_h(t)` oscillates with the tide; a finite melt-vs-forcing phase lag is measurable |

**Honest scope and the central finding.** The prototype demonstrates the
**melt → geometry** leg cleanly (Tests C/D): over a wavy base, differential
*conduction* through the variable gap reshapes `σ_h`. The **flow → melt** leg is
**not** activated by body-force-driven *horizontal* flow at accessible Re/Pe
(Test B null): melt is set by the *vertical* interfacial gradient, which weak
horizontal laminar flow does not change. Verified across n = 128–256 (clean
penalty core) and steep topography (ε up to 0.2) — the null is robust, not a
mask artifact. Activating flow → melt requires genuine **vertical** transport:
buoyant convection, which develops on the diffusive timescale τ ~ L²/κ
(~10⁵ steps here), or resolved turbulence. There is also a numerical trilemma —
vigorous melt needs a *thin warm* cavity, but a thin cavity makes the diffuse
penalty masks overlap and damp the flow, while a clean flowing core needs a
*thick* (cold-based, slow-melting) cavity; only higher resolution relieves it.

**Update — corrected Candidate 3 (scallop / melting-instability), `REPORT_CANDIDATE3.md`.**
Test B's null is specifically a *flat* base under *gentle* horizontal flow. The
roughness-growth framing of Candidate 3 was a false analogy to eroding rock beds;
for a *melting* ice surface the right question is a **stability** one — does a
finite-amplitude perturbation organise the flow into a heat-transfer-enhancing
scallop? A staged go/no-go probe (`scallop_battery.py`) re-runs Test B with a
**resolved** finite-amplitude bump (`a/λ=0.1`, `a ≈ 2.8×` the interface width)
and a **stronger mean current** (steady `U_drive`). There the flow→melt leg
**does** activate: the bumpy wall passes a coherent, bump-locked lee/reattachment
heat flux *above* both its conduction baseline and a flat-wall control —
`R_mean = m_flow/m_cond ≈ 1.06–1.36` (peaks `2–3.3×`), growing with the mean
current and **closure-independent** (a *resolved-scale* effect, not subgrid).
Verified on CPU and a Tesla P100. This does not overturn the Test B / Candidate
1/4 conduction limit — that limit bounds *subgrid wall* mechanisms; the scallop
is a *resolved-flow* enhancement that sits alongside it. The quantitative
`Λ(λ)/a_sat/Nu(λ)` sweep and the corrected (rigorous local-normal) numbers are
in **§13**; the scale-separation picture they imply is in **§14**.

This prototype does **not** include ice viscous creep, pressure-dependent
melting (regelation), 3-D geometry or overhangs, a sharp interface (the penalty
zone is 3–5 cells), or implicit flow–geometry coupling (the update is explicit
with a CFL-like constraint). It is a **proof-of-concept** for the feedback
mechanism and a correct moving-boundary foundation for the convective/turbulent
runs that would test flow → melt — not an operational cavity model. The verified
structural results (§1–§8) are independent of it.

(Reproducible on CPU via `python run_stefan_sweep.py`; unit tests in
`tests/test_stefan.py`.)

## 13. Scallop melting instability — resolved-flow verification (RESULT 10)

§12's corrected Candidate 3 tested whether a finite-amplitude sinusoidal
perturbation on the ice base organises flow into a heat-flux-enhancing scallop.
The full quantitative sweep (`scallop_sweep.py`, `REPORT_CANDIDATE3.md`) confirms
it does — at a single wavelength, and only locally.

### 13.1 Wavelength selection (Part C, frozen boundary)

Grid `nx=ny=128` (`Lx=8π`), `U_drive=1.5`, `Ri=0`, `a/λ=0.1`, local-normal
gradient. Flat-wall control `Nu_flat = 2.377e-4`.

| n_waves | λ | λ/dx | R_mean | R_max | Nu/Nu_flat | corr(excess,slope) |
|---|---|---|---|---|---|---|
| 4 | 6.283 | 32.0 | 1.001 | 1.51 | 0.827 | −0.62 |
| 6 | 4.189 | 21.3 | 1.060 | 2.23 | 0.831 | −0.67 |
| 8 | 3.142 | 16.0 | 1.078 | 1.96 | 0.837 | −0.84 |
| 10 | 2.513 | 12.8 | 1.108 | 2.30 | 0.900 | −0.78 |
| **12** | **2.094** | **10.7** | **1.125** | **2.487** | **0.922** | −0.80 |
| 16 | 1.571 | 8.0 | 1.063 | 1.84 | 0.859 | −0.86 |
| 20 | 1.257 | 6.4 | 1.097 | 2.06 | 0.881 | −0.82 |
| 24 | 1.047 | 5.3 | 1.031 | 2.24 | 0.916 | −0.61 |

**Optimal `n_waves = 12`** (`λ ≈ 2.094`, `λ/dx ≈ 10.7`): an interior peak in
both `R_mean` (+12.5 %) and `Nu/Nu_flat` (0.922), falling off on either side —
**band-selective**, the signature of a preferred scallop wavelength. The selection
is **fluid-determined**: `Re_L = U λ / ν ≈ 2000–2200` at the optimal λ, matching
the empirical Curl criterion (Curl, 1966; 1974) for scallop wavelength on ice and
limestone.

`corr(excess, slope) < 0` throughout confirms the excess flux is lee-concentrated
(reattachment), not turbulent noise. Seed ensemble (4 seeds) gives
`R_mean = 1.127 ± 0.004` — tight.

**Closure independence.** `none/smagorinsky ≈ 0.1 %`; `backscatter ≈ 2.8 %` — a
**resolved-scale** effect, not a subgrid closure prediction. The two-clocks
framework's role is structural: by preserving divergence-free backscatter (§7.1)
it ensures the large-scale pressure field driving the separation is not corrupted.

**Mean heat flux (honest).** With the rigorous local-normal gradient,
`Nu/Nu_flat = 0.83–0.92` at **every** tested wavelength: the spatially-averaged
normal flux on the bumpy wall stays **below** the flat wall. The scallop
redistributes heat flux (concentrating it at reattachment, `R_max ≈ 2.5×`;
depleting it on the stoss face, `R_min ≈ 0.16–0.6`) but does **not** increase
the net mean.

### 13.2 Amplitude regulation — ice-side conduction and the implicit solver

The Stefan condition `ρ L · v = q_water − q_ice` was originally integrated with
an explicit sub-cycler that only enforced the diffusion CFL, ignoring the
advection CFL from `v_int = m/St`. On the GPU (P100) this overflowed to `NaN` in
stage 2b. The fix replaces the explicit integrator with an **implicit
backward-Euler tridiagonal (Thomas) solve** + sign-aware upwind advection:

> `(I + h·A) T^{n+1} = T^n + BC`,

where `A` is the M-matrix from diffusion + upwind advection (diagonally dominant).
This is **unconditionally stable** — no CFL constraint, no NaN overflow regardless
of `v_int` magnitude.

**Three-branch comparison on Tesla P100** (`a0 = 0.20λ`, `n_waves=12`, 4000
boundary updates, `seed=0`):

| Branch | ice_side | amp_final | finite? |
|---|---|---|---|
| `main` (explicit) | True | `nan` | False — overflow |
| fix (implicit) | False (water-only) | **0.2101** | True — decay |
| fix (implicit) | True | **0.4091** | True — saturation |

**Physics.** Without ice thermal memory, scallop amplitude decays (geometric
self-smoothing: thin ice melts faster, raises the base). With ice-side conduction
(`q_ice`), cold ice conducts heat away from the interface, opposing melt and
**stabilising amplitude** near its initial value. This is a genuine thermal-memory
effect at the resolved boundary, not a subgrid closure property.

### 13.3 What the scallop is (and is not)

**Is:** a resolved-flow instability whose wavelength is fluid-selected (Curl's
`Re_L`), whose local heat flux enhancement (`R_max ≈ 2.5×`) is real, and whose
amplitude is stabilised by ice-side conduction against geometric self-smoothing.

**Is not:** a mean melt-enhancement mechanism at this amplitude (`Nu/Nu_flat < 1`)
or a subgrid closure effect. The ~12.5 % conduction-relative gain on the same
geometry (`R_mean = 1.125`) is distinct from the flat-wall comparison
(`Nu/Nu_flat = 0.922`).

**Corrected framing.** Earlier text (§12 update) quoted probe-era numbers
(`R_mean ≈ 1.06–1.36`, peaks `2–3.3×`) from a vertical-gradient proxy. The
rigorous local-normal sweep yields `R_mean = 1.125`, `R_max = 2.487`,
`Nu/Nu_flat = 0.922` at the optimal wavelength — lower and more honest.

The scallop wavelength is **not** determined by clock-matching between water thermal
and ice creep timescales. Ice creep is also a smoothing process; scallops form
identically on limestone (no creep, no melting into solid; Curl 1966). The verified
"clocks" are two: water pressure (instantaneous) and water temperature
(memory-bearing). Ice thermal inertia adds a resolved-scale memory, not a new
operator-class clock.

(CPU: 17 unit tests pass. GPU: P100 three-branch `a_sat`; recorded in session.)

---

## 14. From two clocks to three scales

The original framework (§1–§8) separated **two operator classes**: elliptic
pressure (instantaneous constraint) and parabolic temperature (memory-bearing
diffusion). The candidate tests (§11–§13) force a refinement: the dynamically
relevant structure has **three scales in the water** plus **ice thermal inertia**.

| Scale | What | Timescale | Operator | Role |
|---|---|---|---|---|
| Pressure | Incompressibility constraint | τ_p ~ L/c → 0 | Elliptic (Poisson) | Instantaneous; global |
| Water temperature | Molecular diffusion | τ_θ ~ L²/κ ~ 10⁴–10⁵ s | Parabolic (heat) | Memory-bearing; FDT-linked |
| Resolved flow | Separation, recirculation | τ_flow ~ λ/u* ~ 10¹–10³ s | Hyperbolic (inertial) | DNS physics; closure-free |
| Ice temperature | Conduction in solid | τ_ice ~ H²/κ_ice ~ 10⁶–10⁸ s | Parabolic (heat), slow | Boundary memory |

More precisely, this is not "four clocks": it is the **same two operator classes
(elliptic pressure, parabolic temperature) operating in two media** (water and
ice), with the resolved inertial flow emergent from their coupling. The ice-side
conduction result (§13.2) is what promotes ice from a passive boundary condition
to a participating medium with its own (very slow) thermal clock — so the
interface is a *coupling surface*, not a wall.

### 14.1 Why subgrid-driven mechanisms fail

Every candidate that required the **subgrid** to drive emergent wall behaviour
produced a null:

- Direction A/B (memory kernel): FDT renormalisation pins variance
  instantaneously; memory recolours force, no phase lag.
- Direction C (stratification resonance): CLT decorrelation washes out N³ OU
  temporal correlation.
- Candidate 1 (plumes): conduction-pinned melt; geometric intermittency.
- Candidate 2 (double-diffusion): `Nu_T` monotonic; `Nu_S ≫ Nu_T`.
- Candidate 4 (hydraulic switching): monostable-filled.

The subgrid wall is a **hard boundary**: the penalised no-slip Brinkman interface
sets melt by conduction through a thin diffuse zone. Turbulence cannot reach
the interface at the energies accessible here.

This last sentence was the framework's one remaining *untested* assertion — it
could have been an artifact of the **no-slip** momentum boundary layer rather
than a thermal limit. §14.4 closes that gap directly: relaxing no-slip to free
slip (so the bed *slides* and the stagnant momentum layer is removed) raises the
near-bed tangential speed ~100× yet leaves basal heat delivery unchanged. The
bottleneck is therefore the **thermal conductive sublayer**, not the momentum
stagnation layer — the hard-boundary claim is a property of the cold-wall
thermal BC, not of any one velocity BC.

### 14.2 Why resolved-flow mechanisms work

The scallop (§13) lives at the **resolved-flow** scale — it is not
pressure-driven (too fast), not temperature-diffusion-driven (too slow), and not
subgrid-closure-driven (closure-independent to < 3 %). It is **inertial**: the flow
separates because of its own momentum, and the separation length is set by fluid
mechanics (Re_L ≈ 2000–2200).

The ice thermal memory (§13.2) operates at the **ice temperature** scale —
orders of magnitude slower than water temperature, essentially frozen on flow
timescales. It enters only through the boundary condition (`q_ice` in the Stefan
balance), not through the closure.

### 14.3 The regime equation, revised

The additive form (§8.3) becomes:

> R = R_cond(D, M) + R_scallop(λ/δ_T, Re_L) · 𝟙_{a > a_crit} + R_ice(St_ice)

| Term | What | Limit behaviour |
|---|---|---|
| R_cond | Structural baseline (§8) | ≈ 1 at accessible D |
| R_scallop | Resolved-flow redistribution | R_max ≈ 2.5× local; Nu/Nu_flat < 1 mean |
| R_ice | Amplitude stabilisation by ice conduction | Saturates a near a₀; does not change mean Nu |

The spatially-averaged melt may never exceed the flat wall in the mean, but the
*local* enhancement at reattachment points (`R_max ≈ 2.5×`) could drive
larger-scale processes: channelisation, moulin initiation, focused basal melting.
The framework distinguishes **spatially-averaged melt** (what the regime equation
predicts) from **locally-enhanced melt** (what drives geometry evolution).

### 14.4 The basal-heat ceiling is boundary-condition-robust (capstone)

Every prior null used a fixed pair of boundary conditions: a **no-slip** bed for
momentum and a **cold-Dirichlet** ice wall for temperature. The obvious objection
is that the wall-limited result is an *artifact* of those two pins — relax either
and the mechanism might reappear. Two targeted gates close this objection.

**Thermal-BC gate (Robin / finite conductance).** Replacing the infinite-conductance
Dirichlet pin with a finite-conductance Robin ice wall (`q = h(θ−θ_ice)`) leaves
basal heat absorbed unchanged flow-ON vs flow-OFF to `< 0.01 %`, at `n=128`,
turbulent (`umax ≈ 12`), streamwise ridges (P100). The result is **not** an
artifact of the infinite-conductance assumption. *(See PR #2 / `wall_flux.py`.)*

**Momentum-BC gate (Navier slip / bed sliding).** Replacing no-slip with a Navier
tangential-slip bed (`bed_slip = s`; `s=1` no-slip, `s→0` free slip — the
volume-penalisation analogue of Weertman/regelation sliding) at the *same*
turbulent forcing (P100, `n=128`, `a=0.30`, `kz=4.0`):

| slip `s` | `Nu/Nu_flat` | ridge/ridge(no-slip) | `u_tang(ridge)` |
|---|---|---|---|
| 1.00 (no-slip)  | 0.9679 | 1.0000 | 2.71e‑2 |
| 0.30            | 0.9681 | 1.0002 | 2.30e‑2 |
| 0.10            | 0.9683 | 1.0004 | 2.71e‑2 |
| 0.00 (free slip)| 0.9638 | 0.9958 | 2.68e+0 |

The near-bed tangential speed grows **~100×** (0.027 → 2.68) as the bed goes from
locked to freely sliding, yet `Nu/Nu_flat` stays **0.96–0.97 and never exceeds 1**
(it slightly *falls*), and ridge melt vs the no-slip ridge holds to within 0.4 %.
Flat-bed melt is invariant across slip to 5 digits. Removing the stagnant momentum
layer entirely delivers **no** extra basal heat: the limit is the **thermal**
conductive sublayer set by `κ` and the cold-wall Dirichlet condition, not the
**momentum** stagnation layer. (`s=1` reproduces the no-slip trajectory
bit-for-bit, including on the GPU CuPy path, so the gate is a controlled test, not
a re-tuned model.)

**The closed-gate ledger.** Across every axis the framework can reach, the cavity
is wall-limited and the flow-enhanced-melt mechanism is **absent in the mean**:

| Gate | Knob varied | Result | Status |
|---|---|---|---|
| 2D scallop | geometry | local `R_max≈2.5×`, mean `Nu/Nu_flat<1` | closed (§13.1) |
| 3D channelisation | dimensionality | flow reorganises, ~0 % mean heat gain | closed (§13) |
| Amplitude scan | `a/λ` | monotone, no optimum | closed (FUTURE_WORK D.5) |
| Ice memory | `St_ice` | stabilises amplitude, no mean enhancement | closed (§13.2) |
| Thermal BC | Dirichlet → Robin | identical to `< 0.01 %` | closed (PR #2) |
| Momentum BC | no-slip → free slip | `Nu/Nu_flat` flat while `u_tang ×100` | **closed (§14.4)** |

**Regime map — where the ceiling does and does not bind.** The result is specific
to the **grounded, cold-walled, closed** cavity; it is not a universal statement
about ice–water interfaces:

| Type | System | Interface | Flow-enhanced mean melt? |
|---|---|---|---|
| I | Subglacial cavity (grounded) | closed; cold ice wall; conduction-limited | **No** — proven here |
| II | Grounding line | hybrid; flux-type; ocean-pressure-coupled | Maybe — needs a flux BC |
| III | Ice-shelf cavity | open; ocean-driven; warm ambient water | Yes — ocean heat advection |

The framework's deliverable for Type I is therefore a **regime classification, not
a melt multiplier**: it explains *why* enhancement is bounded (the conductive
sublayer is the hard limit) and *where* the boundary-condition class would have to
change for it to operate (Types II–III). This is consistent with operational
subglacial models (Röthlisberger; GlaDS) carrying no flow-enhanced-melt term.

**Scope (unchanged honesty).** These gates are run at `Pr ≈ 1` (`κ ≈ ν`) with
constant fluid properties in a penalised LES demonstration, and the observable is
the solver's advective+diffusive `melt_flux` proxy, not a Stefan phase change. The
conclusion is a statement about *this regime and closure class*, not a calibrated
glaciological prediction; and in the flow-independent-conductance Robin mode the
only channel for a flow effect is near-wall heat advection, which the (now
slip-relaxed) wall still does not open.

---

## 15. Positioning: vs RSM / EDMF / LES

This work is **not** a new turbulence closure. It is a **structural framework** —
a regime discriminator — that says where any closure of a given class is dominated
by the boundary-condition regime rather than by the subgrid model. We used
existing closures as *instruments* (§9–§13 run None/Smagorinsky/backscatter), not
as the product. The contrast with the three standard families:

| Method | What it optimises | What it does not separate | Where our critique bites |
|---|---|---|---|
| **RSM** (Reynolds-stress / second-moment) | anisotropic momentum stresses via the pressure–strain term `Φ_ij` | the **operator class** of the *scalar* field: even a scalar-flux transport equation for `u'_iθ'` models `Φ_{iθ}` as a fast return-to-isotropy | it has no **slow conduction clock** — it cannot represent a memory-bearing parabolic field pinned by a cold solid wall (the §14 limit) |
| **EDMF** (eddy-diffusivity mass-flux) | **spatial** locality: local diffusion + non-local plume mass flux | **operator** structure: both environment and updraft scalars are still transported quantities | the environment scalar is still gradient-diffusion; the elliptic-pressure / parabolic-temperature *timescale mismatch* is orthogonal to the spatial split |
| **LES** (Smagorinsky / dynamic / backscatter) | accuracy of the resolved large scales; choice of subgrid model | whether the subgrid model **matters** in a given regime | in the Type-I cavity the three subgrid choices agree to **< 3 %** — the wall, not the subgrid, is the bottleneck |

**The sharp RSM point (so it survives the most sophisticated variant).** It is *not*
that "RSM uses K-theory for temperature" — second-moment codes can carry a
scalar-flux transport equation (or GGDH/AFM algebraic flux models) that already go
beyond `−K_T∂Θ`. The durable critique is that the pressure-scalar scrambling term
`Φ_{iθ}` is modelled as a **fast** (instantaneous, return-to-isotropy) process,
so even full RSM does not carry the **slow parabolic conduction clock** of a
cold-walled solid boundary. RSM is a *better closure within the same operator
class*; our framework is a *map of which operator/boundary class governs where*.

**What is genuinely new — and what is not.** RSM fits anisotropic stresses; EDMF
separates local/non-local transport; LES resolves large scales. None of them asks:
*given the operator structure and the boundary-condition class, where can an
emergent wall mechanism operate and where is it killed?* That diagnostic map is the
contribution. The honest limitation: it does **not** give a better model — it tells
you when your existing model is wasted effort and why.

**Claim status (kept bright).**
- **[VERIFIED]** In the Type-I (grounded, cold-wall, closed) cavity the mean basal
  heat flux is closure-independent to `< 3 %` (§13) and boundary-condition-robust
  to both a Robin thermal wall and a free-slip bed (§14.4). "Don't pay for an
  expensive turbulence model in this regime" is a *result*.
- **[HYP]** The Type-II (grounding line) and Type-III (ice-shelf cavity) rows of the
  §14.4 regime map are *predictions* — physically motivated by the boundary-condition
  class changing (flux-type / open-warm), but **not tested in this repo**. The map's
  value holds with these tagged as hypotheses, consistent with the
  [VERIFIED]/[LIT]/[HYP] discipline of `FUTURE_WORK.md`.

**One line.** RSM, EDMF, and LES are better hammers; this framework is the map that
tells you which wall needs a hammer and which wall needs a screwdriver.

---

## Prior art

- Mori (1965); Zwanzig (1973) — generalized Langevin equation, projection operators.
- Chorin, Hald, Kupferman (2000, 2002) — optimal prediction, t-model, MZ for PDEs.
- Kraichnan (1976); Chollet & Lesieur (1981) — spectral (nonlocal) eddy viscosity, cusp.
- Leith (1990); Mason & Thomson (1992) — stochastic backscatter in LES.
- Holland & Jenkins (1999); McPhee (1992, 2008) — three-equation ice-ocean melt law,
  turbulent transfer coefficients, friction-velocity scaling.
- Schoof (2005); Tsai et al. (2015) — effective-pressure-dependent sliding laws.
- Curl (1966, 1974) — fluid-selected scallop wavelength via `Re_L ≈ 2000–2200` on
  limestone and ice (§13.1); the wavelength is fluid, not creep, selected.
- Nikuradse (1933) — sand-grain equivalent roughness `k_s`; the closure route for
  parameterising a scallop field as hydraulic roughness `z_0` (FUTURE_WORK §1).
- Mellor & Yamada (1982); Munk & Anderson (1948) — stratified stability functions.
- York (1973) — conformal/TT decomposition of the ADM momentum constraint (§3.1).
- Dedner et al. (2002) — hyperbolic/GLM divergence cleaning for MHD (§3.2).
- Gundlach, Calabrese, Hinder, Martín-García (2005) — Z4 constraint-damping (κ₁,κ₂);
  the source of the (rejected) universal "1/2", motivating only the O(1) form (§3.2).
- Part 8 (`REPORT_THEORY.md`) — the general two-clocks closure this note specialises.

> **Scope, once more.** This is a closure derivation and a *prediction*. The four
> theorems are operator-level statements; their amplitudes — the reduced-model
> constants {c, a, γ, Ri_c, α, c₁} in (13),(15)–(17) — are transparent scaling
> arguments with named O(1) constants the LES must pin, not first-principles theorems
> for the melt factor. The verified items (Theorems 1, 2, the master identity, and the
> R₀ → 1 null) are structural and GPU-free; the 2–3× factor itself is the prediction
> the active-buoyancy LES will test. None of this is a validated production glacier
> melt law. The value is that it converts the planned simulation into a discriminating
> test with a pre-registered, falsifiable curve.
