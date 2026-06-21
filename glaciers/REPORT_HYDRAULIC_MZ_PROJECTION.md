# RESULT 21 — the §G.4 hydraulic lag kernel is a memory kernel in the *exact* Mori–Zwanzig sense

**Status:** §G.4 reassigns the surge-lag memory to the hydraulic cavity↔channel
subsystem and asserts (ontology table, "two potentials + a hydraulic impedance
kernel") that *"its linearised lumped Green's function **is** a memory kernel in
the exact Mori–Zwanzig sense."* The peaked **shape** of that kernel is already
[DERIVED] by `hydraulic_kernel_synthetic` (RESULT, §V.5) and its **value** by
`hydraulic_lag_derivation` (RESULT, §V.5/(ii)). What was never demonstrated — and
what makes the quoted phrase a theorem rather than a metaphor — is the **projection
itself**: that eliminating the channel variable from the coupled system *exactly*
produces a closed generalized Langevin equation whose convolution kernel is the
eliminated subsystem's Green's function. RESULT 21
(`validation/synthetic/hydraulic_mz_projection_synthetic.py`) performs that
projection on the **same** linear model `hydraulic_kernel_synthetic.coupled_response`
builds, and certifies five properties. Tests in
`tests/test_validation_synthetic.py` (6/6). No external data; no GPU.

## What was already covered, and what was open

- `hydraulic_kernel_synthetic` (§V.5) shows the *downstream observable* of the
  coupled 2×2 cavity↔channel system rises to an interior peak `t*` and that a single
  first-order `RC` store cannot — the **shape** of the lag.
- `hydraulic_lag_derivation` (§V.5/(ii)) reads `τ₁, τ₂` off the named-constant
  GlaDS/Röthlisberger Jacobian — the **magnitude** of the lag.
- `gle_memory_synthetic` (§V.7, §D.4) validates Mori–Zwanzig **additivity** and
  scale-selectivity of two *prescribed* baths (OU + power-law) for the unified GLE.

None of these performs the **projection** the §G.4 phrase names. `coupled_response`
builds `M` and exponentiates it; it never eliminates a variable, never exhibits a
*memory integral*, and so never shows the reduced dynamics is non-Markovian with a
kernel equal to the eliminated channel's Green's function. `gle_memory_synthetic`
starts from kernels that are *assumed*, not *derived from a resolved subsystem*. So
the central structural claim — that the §G.4 non-local convolution term is a genuine
Mori–Zwanzig memory kernel obtained by projecting out the channel — had no test.
That gap is what this result closes.

## The exact projection (linear Nakajima–Zwanzig / Mori, no Markov approximation)

The lumped linearisation is `ẋ = M x + F` with `x = (s, q)` — `s` the resolved
store (cavity water pressure `p_w`), `q` the eliminated channel variable — and
`M = [[−1/τ₁, −a],[b, −1/τ₂]]`. Solving the `q` row by variation of parameters and
substituting into the `s` row gives a **non-local** (memory) term:

> `ṡ(t) = M_ss s(t) + ∫₀ᵗ K(t−τ) s(τ) dτ + R(t)`,
> `K(τ) = M_sq M_qs e^{M_qq τ}`  (the memory kernel),
> `R(t) = M_sq e^{M_qq t} q(0) + F_s(t) + M_sq ∫₀ᵗ e^{M_qq(t−τ)} F_q(τ) dτ`  (the Mori residual force).

`K(τ) = −ab·e^{−τ/τ₂}` is exactly the channel subsystem's own Green's function
`e^{M_qq τ}` weighted by the up/down couplings `M_sq M_qs`; `R(t)` is carried by the
eliminated initial state `q(0)` and is orthogonal to the resolved variable's own
history. This is the precise content of "memory kernel in the exact Mori–Zwanzig
sense."

## What this result establishes

The harness reuses the **same coupled model** (`hydraulic_kernel_synthetic.coupled_response`,
`cascade_peak_time`) so the system under test is the §G.4 lumped hydrology, not an
ad-hoc system:

1. **The projection is exact (machine precision).** In Laplace space the GLE transfer
   function `ŝ(s) = [s₀(s−M_qq) + M_sq q₀] / [(s−M_ss)(s−M_qq) − M_sq M_qs]` equals the
   full 2×2 resolvent's (1,1) component term-for-term, verified at random complex `s`
   for 400 random stable overdamped systems — **max abs err `5.0×10⁻¹⁶`**. So `K` is
   the *exact* kernel, not a closure approximation.
2. **The kernel IS the eliminated subsystem's Green's function.** Probing the channel
   alone (`q̇ = M_qq q`, `q(0)=M_qs`) and reading `M_sq·q(τ)` reproduces `K(τ)` to
   **`6.9×10⁻¹⁸`**; the kernel decays at exactly `1/τ₂` and is sign-definite — it is the
   channel impedance response, not a fit.
3. **The reduced GLE reproduces the resolved trajectory.** Time-stepping the GLE
   (trapezoidal memory + the MZ residual force) matches the exact `expm` solution of the
   full 2×2 system to **rel-err `9.1×10⁻⁸`** over the transient — the reduced
   non-Markovian model *is* the resolved dynamics.
4. **Memory is necessary for the lag/peak.** The full (memory-carrying) downstream
   response peaks at the interior time set by the **coupled eigenvalues** (`t*=0.911`,
   analytic `0.911`, rel-err `5.4×10⁻⁴`), whereas the **Markovian** adiabatic-elimination
   closure (drop the memory: slave `q = −M_qq⁻¹ M_qs s`) collapses to a first-order
   response that is **monotone with argmax at `t=0`**. The surge lag is a property of the
   *memory kernel*, exactly §G.4's claim.
5. **Markovian limit = adiabatic elimination (DC gain).** `∫₀^∞ K dτ = M_sq M_qs/(−M_qq)`
   (numeric-vs-analytic err `7.5×10⁻⁹`), and shrinking the channel time `τ₂→0` at fixed DC
   gain collapses the kernel toward `(∫K)·δ(τ)` (half-widths `0.347 → 0.070 → 0.014`),
   recovering the instantaneous local closure `ṡ = (M_ss + ∫K) s` the model uses when the
   eliminated subsystem is fast — the §D.4 FDT-Markov limit, here **derived from the
   projection** rather than assumed.

## Why it matters

The §G.4 ontology rests on calling the hydraulic lag term a Mori–Zwanzig memory
kernel. RESULT 21 turns that from an assertion into a verified consequence of the
linear algebra: the projection is exact, the kernel is the channel's Green's
function, the reduced GLE reproduces the dynamics, the memory is what produces the
peak, and the local closure the model uses elsewhere is the kernel's fast-subsystem
limit. Combined with `hydraulic_kernel_synthetic` (shape) and
`hydraulic_lag_derivation` (magnitude), the **structure** of the §G.4 hydraulic
memory term is now fully certified.

## Honest scope

This validates the **projection structure** of the *linearised lumped* model — that
the §G.4 hydraulic term is a genuine MZ memory kernel and how it reduces to the
local closure. It does **not** validate (a) the physical identity of `φ` as distinct
from the Leray pressure, (b) that this hydraulic subsystem (rather than another)
dominates a real surge, or (c) any field lag value — those remain the modeling
**[HYP]** in §G.4 and are gated on real data (§H.2, USAP-DC). It is the projection
companion to RESULT (§V.5) shape/magnitude work, not a replacement.

## Distributed (spatially-resolved) extension — RESULT 24 (§V.5d, paper4a §4.5)

`hydraulic_mz_spatial.py` answers the *spatial* half of the "it is just a 2×2
identity" objection (the other half — nonlinearity — is `hydraulic_nonlinear_kernel.py`,
§V.5c). Resolve the flowline: two coupled **fields** on a 1-D periodic line, the
cavity store `s(x,t)` and the channel `q(x,t)` that *routes water downstream*
(advection `U`, spreading `D_q`, relaxation `τ₂`). Project out the channel **field**
exactly (linear Nakajima–Zwanzig) ⇒ a generalized Langevin equation for `s` whose
kernel `𝒦(τ)=A_sq e^{A_qq τ} A_qs` is now a **matrix-valued, spatially non-local**
operator. On `N=24` nodes (6/6 tests):

1. **Projection exact (operator Schur complement).** The GLE transfer operator equals
   the full resolvent's resolved block at random complex `s` — max rel-err **7.4×10⁻¹⁶**.
2. **Kernel == channel operator Green's function** (vs `expm`, **2.7×10⁻¹⁷**), spatially
   non-local with a resolved along-flow memory range `ℓ_mem ≈ 0.31 ≫ Δx`.
3. **Reduced field-GLE reproduces the full PDE** trajectory (slow channel `τ₂=1.5`) to
   rel-err **1.4×10⁻⁵**.
4. **Memory necessity is a dose-response.** The memoryless local closure
   `ṡ=(A_ss+𝒦_DC)s` (`𝒦_DC=−A_sq A_qq⁻¹ A_qs=∫₀^∞𝒦`) has a trajectory error that grows
   monotonically with channel slowness (`0.01 % → 1.0 %` over `τ₂=0.05…3.0`) and → 0 as
   `τ₂→0` — the Markovian limit, now derived **in the distributed system**.
5. **The lumped 2×2 kernel is the no-transport, single-node limit.** With `D_q=U=0` the
   operator kernel is **exactly diagonal** and its diagonal equals the committed lumped
   `−ab e^{−τ/τ₂}` to **6.9×10⁻¹⁸**; switching transport on moves the kernel mass
   off-diagonal (off-diagonal fraction `0 → 0.99`), giving the surge-lag memory a finite
   **along-flow footprint** with closed-form decay length `ℓ_mem=2D_q/(√(U²+4D_q/τ₂)−U)` (numeric steady-influence operator vs analytic to `4×10⁻⁴`; ballistic limit `→Uτ₂`, diffusive limit `→√(D_qτ₂)`) — a field-measurable observable.

So the §V.5b 2×2 result is the spatially-local limit of an exact projection of a
genuine distributed (PDE) operator, and the distributed system makes a new falsifiable
structural prediction: the lagged velocity response to a drainage is spread along-flow
over `ℓ_mem`, which a memoryless (local) closure sets to zero. **Honest scope:** still a
*linearised* distributed model on a periodic flowline; the fully nonlinear
spatially-resolved coupled GlaDS/Röthlisberger PDE remains the open gate. Artifacts:
`validation/reports/hydraulic_mz_spatial.json`, `figures/78_hydraulic_mz_spatial.png`.
