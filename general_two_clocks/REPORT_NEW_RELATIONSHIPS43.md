# NR66 — the spin's entropy production is reactive (housekeeping), not dissipative

**Module** `general_two_clocks/new_relationships43.py` ·
**data** the committed NR56 ANDRO Stokes cache (`data/nr56_stokes_cache.json`, 8280 floats) ·
**figure** `general_two_clocks/figures/101_reactive_epr.json` + `.png` ·
**tests** `general_two_clocks/tests/test_reactive_epr.py` (12)

## The relationship

NR57–NR61 established that the Lagrangian spin **is** the phase-space probability
current and gave its entropy-production rate σ > 0. NR66 answers what those left open:
**which class of irreversibility is it?** Stochastic thermodynamics with odd-parity
variables (Kwon, Yeo, Lee & Park 2016, *unconventional entropy production*; Yeo et al.
2016, *housekeeping EP with odd-parity variables*; Lee & Kwon 2019) splits the steady
EPR into a **dissipative** part driven by a thermodynamic gradient (a gyrator, two
reservoirs T₁≠T₂, produces heat) and a **reactive/housekeeping** part driven by an
odd-parity antisymmetric force (Coriolis / Lorentz) that does **no work**, exchanges
**no net heat**, yet keeps a divergence-free current circulating forever. The spin is
the reactive one — the geophysical realization of Kwon's magnetic-field term σ = 2B²/γm.

## The exact structural discriminant (closed form, verified to 1e-10)

For a linear OU flow `dx = -A x dt + √(2D) dW`, stationary `A C + C Aᵀ = 2D`, generator
`Ω = A - D C⁻¹` (NR57):

**Reactive rotator** `A = (1/τ)I − f·ε`, `D = D₀I`:
- `C = D₀τ·I` — **isotropic and temperature/f-independent** (Stokes Q = U = 0)
- `Ω = −f·ε` — **purely antisymmetric** (no symmetric part ⇒ no dissipative channel)
- `σ = 2f²τ = 2r²/τ`, `r = fτ = tan δ` — the **two-clocks ratio squared**
- mean Coriolis power `⟨v·(fεv)⟩ ≡ 0` — **heatless**

**Gyrator** `A = [[k,u],[u,k]]`, `D = diag(T₁,T₂)`:
- `C` is **anisotropic** (Q,U ≠ 0); `Ω` has a **nonzero symmetric part**
- `σ > 0` **iff** T₁ ≠ T₂ (σ = 0 when T₁ = T₂ despite coupling u ≠ 0)

Two theorems:
1. **Isotropy ⇒ reactive.** Isotropic stationary covariance with σ > 0 forces Ω
   antisymmetric, the current divergence-free (`div J = tr Ω = 0`, NR59's div-free
   skew flux in phase space), no heat. Anisotropy is the *only* channel a gradient can use.
2. **The reactive rate is the clock ratio.** `σ_reactive·τ/2 = r² = (fτ)² = tan²δ` —
   the same eddy/rotation ratio that sets the rotary coefficient (NR60), the skew
   fraction (NR59) and every loss tangent in this work.

## The real ocean is reactive (ANDRO, 8280 floats)

Using NR56's committed per-float Stokes faces — anisotropy `P = √(Q²+U²)` and spin
`V = ρ_odd` (lag-1):

| test | reactive prediction | measured | verdict |
|---|---|---|---|
| \|V\| vs \|f\|=\|sin φ\| | rises (Coriolis sets spin) | ρ = **+0.15**, p = 3.5×10⁻⁴⁴ | yes |
| signed V vs f | cyclonic (sign follows f) | ρ = −0.13, p = 1.0×10⁻³⁰ | yes |
| P vs \|f\| | flat / no gyrator growth | ρ = **−0.40** (P peaks at equator) | yes (no gradient signature) |
| partial \|V\|~P given \|f\| | ≈ 0 (no gyrator coupling) | ρ = **−0.06** | yes |

The spin tracks planetary vorticity while the anisotropy does the opposite of what a
gyrator would require — and once latitude is controlled, spin and anisotropy are
uncorrelated. **The deep ocean's Lagrangian irreversibility is reactive/housekeeping**:
Coriolis turns the phase-space current without a thermodynamic gradient, exactly the
odd-parity "unconventional" EP class, measured on real floats.

## Reading

The two clocks are not just eddy-vs-rotation kinematically (NR47) and skew-vs-Taylor in
transport (NR59) — they are **dissipative-vs-reactive thermodynamically**. Everything
this work has called "the spin/circulation clock" is the heatless, gradient-free,
divergence-free housekeeping face; everything it has called "the eddy/mixing clock" is
the dissipative face. The rate of the reactive face is `2r²/τ`, closing the circle with
the loss tangent r = tan δ that appears in every other NR.

## Honest scope

ANDRO floats are 10-day quasi-Lagrangian at 700–1300 dbar; V is the lag-1 odd
correlation (NR52 proxy for short-lag areal velocity), not a per-float OU fit. The
equatorial band carries a large **zonal** anisotropy (P peaks at the equator, wave/shear
origin) — this is *why* P anticorrelates with |f| and must not be read as a gyrator; the
decisive test is the partial |V|~P at fixed |f| (≈ 0). "No heat" is the mean-power
identity ⟨v·εv⟩ = 0 for the model antisymmetric drive; it does not claim the real
sub-mesoscale is adiabatic. Single-reservoir OU with an antisymmetric drift is the
minimal system whose EPR is provably all-reactive (Kwon et al. 2016, σ = 2B²/γm).
