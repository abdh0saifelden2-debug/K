# K — Two Clocks: Diagnosing and Repairing K-Theory's Structural Blindness

[![Tests](https://img.shields.io/badge/tests-410+%20passing-brightgreen)](#tests)
[![Parts](https://img.shields.io/badge/parts-1--10-blue)](#roadmap)

---

> **Layout:** the repo is organized into four domain folders —
> [`general_two_clocks/`](general_two_clocks/README.md) (the observation, its
> implications, and the verified proofs), [`atmosphere/`](atmosphere/README.md),
> [`glaciers/`](glaciers/README.md), [`ocean/`](ocean/README.md) — plus
> [`working_notes/`](working_notes/README.md). Each domain folder has its own
> README (found so far / still [HYP] / pending), `figures/` and `tests/`.
> See [Repo Layout](#repo-layout).

## Quick Start (for reviewers)

```bash
pip install -r requirements.txt

# The decisive benchmark (Part 8b) — no data download needed:
python general_two_clocks/run_closure.py --out-dir general_two_clocks/figures --report general_two_clocks/REPORT_CLOSURE.md

# Run all tests:
pytest -v
```

**If you only read one thing:** [`general_two_clocks/REPORT_THEORY.md`](general_two_clocks/REPORT_THEORY.md) (the derivation)
followed by [`general_two_clocks/REPORT_CLOSURE.md`](general_two_clocks/REPORT_CLOSURE.md) (the benchmark that decides it).

---

## Table of Contents

- [Core Thesis](#core-thesis)
- [Roadmap](#roadmap)
- [Part 1 — Temporal decoupling](#part-1--temporal-decoupling-the-two-clocks)
- [Part 2 — Spatial decoupling](#part-2--spatial-decoupling-elliptic-vs-parabolic)
- [Part 3 — K-theory breakdown](#part-3--the-breakdown-of-k-theory)
- [Part 4 — Hyperbolic → elliptic crossover](#part-4--the-hyperbolic--elliptic-crossover-demonstration)
- [Part 5 — Nonlinear compressible two clocks](#part-5--nonlinear-compressible-two-clocks-demonstration)
- [Part 6 — Synthesis via projection](#part-6--synthesis-the-two-clocks-reunified-by-projection-demonstration)
- [Part 7 — Real reanalysis winds](#part-7--the-two-clocks-in-real-reanalysis-winds)
- [Part 8 — MZ/FDT closure (theory + benchmark)](#part-8--a-two-clocks-closure-that-generalizes-k-theory)
- [Part 9 — Subglacial cavity application](#part-9--the-two-clocks-in-a-subglacial-cavity-application-benchmark)
- [Part 10 — Goldshtik–Sorokin vortex levitation](#part-10--the-two-clocks-at-a-turbulent-vortex-goldshtiksorokin-levitation)
- [Scope and Honesty](#scope-and-honesty)
- [Getting the Data](#getting-the-data)
- [Running](#running)
- [Repo Layout](#repo-layout)
- [Tests](#tests)
- [Figures](#figures)

---

## Core Thesis

> Pressure and temperature **cannot be modelled as one coupled scalar**. They
> operate on fundamentally different **temporal clocks** *and* different
> **spatial architectures**. Temperature is a *local, parabolic* field — it
> diffuses neighbour-to-neighbour (the heat equation) and is torn into sharp
> filaments by local shear. Pressure is a *global, elliptic* field — it solves a
> Poisson equation that respects the whole domain's geometry and boundaries at
> once. Any model that stirs both with one local eddy diffusivity (classical
> single-Prandtl **K-theory**) is structurally unable to represent this.

This repo tests that thesis against independent public datasets at increasing
complexity, derives the exact operator-level repair (Mori–Zwanzig / projected-FDT),
and validates it with a concrete benchmark. Every headline number is computed by
the code in this repo, not asserted.

---

## Roadmap

| # | Part | claim | dataset | report | key figure |
|---|------|-------|---------|--------|------------|
| 1 | Observational | Temporal two clocks | NEON eddy-flux (WREF, 30-min) | [`general_two_clocks/REPORT.md`](general_two_clocks/REPORT.md) | Fig 1–5 |
| 2 | Observational | Temporal + spatial gradient | ASOS 1-min (3 stations, 25–42°N) | [`general_two_clocks/REPORT_ASOS.md`](general_two_clocks/REPORT_ASOS.md) | Fig 9–12 |
| 3 | DNS | Spatial two clocks (elliptic vs parabolic) | Rayleigh–Bénard DNS (Ra=10⁶) | [`general_two_clocks/REPORT_RB.md`](general_two_clocks/REPORT_RB.md) | Fig 13–15 |
| 4 | Closure test | K-theory breakdown | NEON stability analysis | [`general_two_clocks/REPORT_STABILITY.md`](general_two_clocks/REPORT_STABILITY.md) | Fig 6–8 |
| 5 | Demo | Hyperbolic→elliptic crossover | 2D linear-acoustics solver | [`general_two_clocks/REPORT_COMPRESSIBLE.md`](general_two_clocks/REPORT_COMPRESSIBLE.md) | Fig 16–18 |
| 6 | Demo | Nonlinear compressible two clocks | 2D isothermal compressible NS | [`general_two_clocks/REPORT_NS.md`](general_two_clocks/REPORT_NS.md) | Fig 19–21 |
| 7 | Synthesis | Projection method *is* the two clocks; SPDE limit | 2D Boussinesq convection | (below) | Fig 22–24 |
| 8 | Reanalysis | Two clocks in **real** winds | NCEP/NCAR Reanalysis 1 (OPeNDAP) | (below) | Fig 25–27 |
| 9 | **Theory** | MZ/FDT closure generalizes K-theory | Derivation + 2D benchmark | [`general_two_clocks/REPORT_THEORY.md`](general_two_clocks/REPORT_THEORY.md) | — |
| 9b | **Benchmark** | Decisive test: Smag vs surrogate vs FDT | 256² filtered DNS | [`general_two_clocks/REPORT_CLOSURE.md`](general_two_clocks/REPORT_CLOSURE.md) | **Fig 28–30** |
| 9c | **Benchmark (3D)** | The vortex-stretching extension of 9b: same verdicts **plus** the 3D-only physics (`ω·∇u` production, intermediate-eigenvector alignment, ~½ backscatter volume that Smagorinsky has at 0%) | forced 3D DNS, filtered (GPU-ready, CuPy) | [`general_two_clocks/REPORT_CLOSURE3D.md`](general_two_clocks/REPORT_CLOSURE3D.md) | **Fig 61–64** |
| 10 | **Application** | Same test in a subglacial cavity (Part 9), synthetic + real BEDMAP1 bed | 128² penalized DNS | [`glaciers/REPORT_SUBGLACIAL.md`](glaciers/REPORT_SUBGLACIAL.md), [`…_REAL.md`](glaciers/REPORT_SUBGLACIAL_REAL.md) | **Fig 31–38** |
| 10b | **Application** | A-posteriori 3D cavity LES: closure-dependent basal **melt rate** over real BEDMAP1 beds (Pr=1 two-clocks theory) | 3D penalized LES (GPU-ready) | [`glaciers/subglacial/THEORY_CAVITY.md`](glaciers/subglacial/THEORY_CAVITY.md) (`glaciers/run_subglacial3d.py`) | **Fig 60** |
| 10c | **Probe** | Mechanism probe on the 2D cavity: double-diffusive layering (salt + temperature, Candidate 2). Honest finding: no salt-finger hump in `Nu_T(R_rho)`; the diffusivity contrast shows as `Nu_S ≫ Nu_T` and counter-gradient flux at strong stabilisation | 2D penalized DNS, two scalars | [`ocean/REPORT_CANDIDATE2.md`](ocean/REPORT_CANDIDATE2.md) (`ocean/run_candidate2.py`) | — |
| 10d | **Probe** | Mechanism probes on the 2D cavity: intermittent plumes from ice-base roughness (Candidate 1). Honest finding: interfacial melt is conduction-limited (geometric, flat in Ri); the flow signal lives in `Fturb=⟨v'θ'⟩` | 2D penalized DNS | [`glaciers/REPORT_CANDIDATE1.md`](glaciers/REPORT_CANDIDATE1.md) (`glaciers/run_candidate1.py`) | — |
| 10e | **Probe** | Mechanism probe on the 2D cavity: roughness feedback (Candidate 3). The pre-registered runaway is sign-reversed — the conduction-limited base **self-smooths** (`Λ≈−0.27`), the correct ice physics; the **corrected** scallop / melting-instability hypothesis then **passes** a go/no-go probe — a resolved finite-amplitude bump under a mean current gives a coherent, closure-independent lee heat-flux enhancement (`R_mean` 1.06–1.36, peaks 2–3.3×) above the flat-wall control | 2D penalized DNS (CPU + P100) | [`glaciers/REPORT_CANDIDATE3.md`](glaciers/REPORT_CANDIDATE3.md) (`glaciers/scallop_battery.py`) | — |
| 10f | **Capstone** | The basal-heat ceiling is **boundary-condition-robust**: relaxing the cold-Dirichlet ice wall (→Robin, finite conductance) and the no-slip bed (→Navier free slip, `u_tang ×100`) both leave mean melt unchanged (`Nu/Nu_flat` flat, never >1). The limit is the **thermal conductive sublayer**, not momentum stagnation. Closes the wall-limited result and frames the deliverable as a **regime map** (Type I no / II maybe / III yes), plus a positioning vs **RSM / EDMF / LES** | 3D penalized LES (P100) | [`THEORY_CAVITY.md` §14.4, §15](glaciers/subglacial/THEORY_CAVITY.md) (`glaciers/subglacial/slip_gate.py`, `glaciers/subglacial/wall_flux.py`) | — |
| 10g | **Result** | **RESULT 14** — wall-flux harmonic decomposition of the scallop amplitude law: conduction `β/a` is `K`-independent (`K²` curvature ansatz **falsified**), no `+α a^{1/2}` growth (smoothing-limited), and the flow channel is a **quadrature migration** term `∝U^{0.5–0.8}`. Corrected mode `s=−β+iω_mig`; migration is a **parity-symmetry break no K-theory closure can produce** (keystone control measured on P100). Constant-free, `ΔT`-free field test `I=τ·c_mig/λ`. **RESULT 22** generalises the swept window: across `a₀/λ∈[0.05,0.40]` the smoothing-only (no `+α`) and falsified-`K²` verdicts hold at **every** amplitude (`amplitude_generalization_scan`); **RESULT 23** closes the other half of Caveat D — out to drive `U=6` (beyond the swept `U∈[1.5,3.0]`) the smoothing-only verdict survives and migration keeps its **sub-kinematic** `∝U^{0.5–0.8}` scaling (`U^{+0.48}`, never `U¹`; saturates at `U≳4.5`) (`drive_window_scan`) | 2D penalized DNS (CPU + P100) | [`glaciers/REPORT_SCALLOP_MIGRATION.md`](glaciers/REPORT_SCALLOP_MIGRATION.md) (`glaciers/scallop_amplitude_harmonics.py`, `glaciers/scallop_ktheory_control.py`) | **Fig 56** |
| 11 | **Application** | Same test at a turbulent vortex (Goldshtik–Sorokin levitation), synthetic + real Hurricane Otis | 128² swirl DNS | [`atmosphere/REPORT_SWIRL.md`](atmosphere/REPORT_SWIRL.md), [`…_REAL.md`](atmosphere/REPORT_SWIRL_REAL.md) | **Fig 40–47** |
| V | **Validation** | Turn the §G subglacial **[HYP]** predictions into falsifiable forecasts (§H). Synthetic harnesses prove the math; real open data exposes the physics: **RTN>1 concentrates at the grounding line** on Bedmap2 (median 6 km vs 221 km; [VERIFIED] directional), and the literal §G.4 lag `H²/κ ≈ 1.5×10⁵ yr` on 131 real lakes is ~8×10⁴× too slow ([FALSIFIED] as written) | BAS Bedmap2 + Siegfried&Fricker 2018 + ITS_LIVE | [`glaciers/validation/README.md`](glaciers/validation/README.md), [`glaciers/validation/REAL_DATA_RESULTS.md`](glaciers/validation/REAL_DATA_RESULTS.md), `FUTURE_WORK.md §H` | — |

**Reading order for the impatient:**  Part 3 → Part 6 → Part 8/8b → general_two_clocks/REPORT_THEORY.md

---

## Part 1 — Temporal decoupling (the "two clocks")

**NEON eddy-covariance, WREF (Wind River old-growth forest, WA), Jan 2020, 30-min.**

- **Solar source → heating cycle:** incoming shortwave leads air temperature by
  ~2 h and the sensible heat flux by ~3 h; daytime heat flux is upward (buoyant).
- **Two clocks:** temperature is overwhelmingly a **24 h** (diurnal) signal,
  while barometric pressure is dominated by multi-day **synoptic** weather
  variability plus a strong **semidiurnal (12 h) atmospheric tide** — a
  qualitatively different spectral fingerprint.
- **Not "equal":** bulk pressure and temperature are essentially decoupled on the
  daily scale (Pearson r ≈ +0.07, R² ≈ 0.005), nowhere near the steep
  constant-density line a fixed-parcel ideal-gas lock predicts. The mismatch is
  absorbed by small (~1%) density changes — the gas law still holds *locally*;
  it is just wrong to read it as "bulk P tracks T".

**ASOS 1-minute, three stations spanning 25–42°N (MIA, DFW, DSM), Q1 2020.**
Higher temporal resolution and, crucially, a **latitude gradient**:

| station | lat | S₁ temp (24 h) | S₂ pressure tide (12 h) |
|---|---|---|---|
| MIA | 25.8°N | 2.73 °C | **1.08 hPa** |
| DFW | 32.9°N | 3.28 °C | 0.96 hPa |
| DSM | 41.5°N | 2.73 °C | **0.63 hPa** |

- Temperature is a clean diurnal peak at **every** latitude — it is set by *local*
  solar heating.
- The **12 h pressure tide weakens monotonically equator→pole**. That latitude
  dependence is the signature of a *global, planetary-scale* resonance (the solar
  atmospheric tide), not local heating. Pressure "knows about" the geometry of
  the whole atmosphere; temperature only knows about the sun overhead.

---

## Part 2 — Spatial decoupling (elliptic vs parabolic)

**Rayleigh–Bénard convection DNS (The Well / polymathic-ai, Ra=10⁶, Pr=1,
512×128, periodic in x, no-slip walls).** Here pressure, buoyancy and velocity
are *all resolved on the same grid*, so the spatial version of the thesis can be
measured directly — something no single-point sensor can do.

The decoupling is **not** about different dominant sizes — both fields share the
same convection-roll wavelength, so their integral lengths are nearly equal
(ratio ≈ 1.06×). It lives at **small scales**:

| metric | buoyancy (temperature) | pressure | pressure / buoyancy |
|---|---|---|---|
| integral length (large scale) | 0.195 | 0.207 | 1.06× |
| **Taylor microscale** (small scale) | 0.253 | 0.607 | **2.40×** |
| high-k spectral power (k≈10) | — | — | buoyancy ~**10⁴×** larger |

- **Pressure is smooth / blobby (elliptic).** The Poisson equation ∇²p = f is, in
  Fourier space, p̂ = −f̂/k². Dividing by k² is a **global low-pass filter**: it
  annihilates small-scale content, which is exactly why the pressure spectrum
  plummets ~10⁴× below buoyancy at high wavenumber. Pressure simply doesn't exist
  at the local scale the way temperature does.
- **Buoyancy is sharp / filamentary (parabolic).** It diffuses locally and is
  advected into thin plumes, retaining intense small-scale gradients (small
  Taylor microscale).

This is the spatial fingerprint of the thesis: same grid, same coupling through
the Boussinesq equations, two fundamentally different geometries of propagation.

### A note on acoustic waves (elliptic vs hyperbolic)

The incompressible/Boussinesq picture above (∇²p = f, elliptic, infinite signal
speed, no waves) is the **low-Mach limit** of the real compressible system, whose
pressure obeys the **hyperbolic** wave equation (1/c²)∂²p/∂t² − ∇²p = 0 at finite
sound speed c. A local compression then propagates as a sound wave and attenuates
by geometric spreading + viscous/thermal damping. As Mach → 0, c → ∞ and the wave
equation degenerates into the Poisson constraint — the waves become infinitely
fast and vanish, leaving only the instantaneous global field. The two are the
same physics at different compressibility. The RB DNS above is incompressible
(it shows only the elliptic side); the crossover itself is demonstrated
separately with a small compressible solver — see Part 4.

---

## Part 3 — The breakdown of K-theory

**NEON stability analysis (same WREF data, 602 usable 30-min periods).**
Classical first-order closure assumes momentum and heat are stirred by the same
eddies with a single constant turbulent Prandtl number (K_M ≈ Pr_t⁻¹ K_H). If
true, the ratio of momentum to heat transport efficiency should be constant
across conditions. Binning by Monin–Obukhov stability (ζ = (z−d)/L):

| stability class | n | median r_uw (momentum) | median \|r_wT\| (heat) | momentum/heat ratio |
|---|---|---|---|---|
| strongly unstable | 95 | 0.182 | 0.209 | 0.87 |
| unstable | 73 | 0.324 | 0.150 | 2.17 |
| near-neutral | 121 | 0.312 | 0.095 | 3.28 |
| stable | 153 | 0.225 | 0.200 | 1.13 |
| strongly stable | 160 | 0.156 | 0.203 | 0.77 |

Momentum efficiency peaks near-neutral (shear-driven), heat efficiency peaks
under strong convection — they move **oppositely**, and the transport ratio
swings **~4.3×** (0.77 → 3.28). A single fixed-Prandtl closure assumes that ratio
is constant, so it is structurally unable to reproduce the swing.

**Why:** heat is a local/parabolic field and momentum is shaped by the
global/elliptic pressure field — different spatial *and* temporal footprints. A
closure built on one local diffusivity cannot represent two processes with
different geometries. This is the empirically testable core of the critique
("single-timescale K-theory is blind").

---

## Part 4 — The hyperbolic → elliptic crossover (demonstration)

**A minimal 2D linear-acoustics solver** (`general_two_clocks/compressible/`, `general_two_clocks/run_compressible.py`)
makes the elliptic-vs-hyperbolic point concrete rather than asserted. In a
compressible fluid the pressure obeys a **wave equation** (hyperbolic, finite
sound speed c); incompressible pressure obeys the **Poisson equation**
(elliptic, instantaneous). The solver shows these are the same physics at two
compressibilities:

- **Radiating pulse:** an initial pressure bump propagates outward as a ring at
  speed c and fades by geometric spreading — the finite-speed, wave-like
  behaviour (figure 16).
- **Crossover:** under a fixed forcing, the compressible pressure relaxes to the
  *exact same* field as the elliptic Poisson solve (final relative error ~2×10⁻³
  for every c), but the time to get there scales as **t₉₀ ≈ 2.3·(L/c)**.
  Plotting error vs the rescaled time c·t/L collapses all sound speeds onto one
  curve — c's only role is to set the clock. As c → ∞ the adjustment becomes
  instantaneous: the elliptic limit (figures 17–18).

This is why water is "more fiercely elliptic" than air: c ≈ 1500 m/s vs ~340 m/s
pushes it closer to the instantaneous-global limit. **Scope:** this *demonstrates*
the crossover; it does **not** test 3D regularity / Beale–Kato–Majda or the
wave-radiation-damping argument (those need rigorous analysis, not a linear demo).
Part 5 takes the same crossover into a *nonlinear* flow.

---

## Part 5 — Nonlinear compressible two clocks (demonstration)

**A fully nonlinear 2D isothermal compressible Navier–Stokes solver**
(`general_two_clocks/compressible/ns.py`, `general_two_clocks/run_compressible_ns.py`; pseudo-spectral, periodic,
constant viscosity, 2/3-rule dealiasing) takes the Part-4 crossover off the
*linear* toy model and into a real nonlinear flow. The isothermal closure
p = c²ρ keeps the sound speed c an explicit knob, so the Mach number M = U/c is
swept directly while retaining full nonlinear advection. From a Taylor–Green
vortex (Re ≈ 600) the velocity is Helmholtz-split into a **solenoidal/vortical**
part (the slow advective clock, τ_adv ~ L/U) and a **dilatational/acoustic** part
(the fast clock, τ_p ~ L/c):

| Mach M | ⟨KE_dil⟩/⟨KE_sol⟩ | avg-pressure residual vs elliptic |
|---|---|---|
| 0.05 | 1.5×10⁻⁴ | 8.1×10⁻⁴ |
| 0.10 | 5.9×10⁻⁴ | 3.2×10⁻³ |
| 0.20 | 2.3×10⁻³ | 1.4×10⁻² |
| 0.40 | 9.3×10⁻³ | 1.0×10⁻¹ |

- **Two clocks coexist in one nonlinear flow.** The vortical energy decays slowly
  while the acoustic energy oscillates on the fast acoustic period (figure 20),
  and the dilatational energy fraction scales as **~M²** → 0 — the fast clock's
  energy vanishes in the incompressible limit.
- **Pressure → the elliptic Poisson field.** The *instantaneous* compressible
  pressure is dominated by the standing acoustic wave launched by the uniform
  initial density, so it is first **averaged over the fast acoustic clock** (the
  quasi-steady second half of each run) — a low-pass filter that scrubs τ_p and
  leaves the slow, balanced field. That acoustic-averaged pressure matches the
  elliptic incompressible Poisson pressure (correlation ≈ 1), with the residual
  itself shrinking as **~M²** toward the incompressible limit (figure 21).

The nonlinear confirmation of the structural picture: pressure carries a fast,
global, wave-like adjustment distinct from the slow vortical/advective evolution,
and as M → 0 it collapses onto the purely elliptic, instantaneous global field —
the same regime as the RB DNS (Part 2) and the linear crossover (Part 4).
**Scope:** demonstrates scale separation and the elliptic limit in a nonlinear
compressible flow; it does **not** prove 3D regularity / Beale–Kato–Majda or
wave-radiation damping (those need rigorous analysis, not a 2D demo).
Part 6 puts the two clocks back together.


---

## Part 6 — Synthesis: the two clocks reunified by projection (demonstration)
 
Parts 1–5 take the two clocks apart; this part puts them back together in one
nonlinear flow and shows that the standard way to do so — the **projection
(fractional-step) method** — *is* the two-clocks structure made literal. A 2D
incompressible **Boussinesq convection** solver (`general_two_clocks/boussinesq/solver.py`,
`general_two_clocks/run_boussinesq.py`; same pseudo-spectral periodic box as Part 5) advances each
step in two pieces:
 
```
u*    = uⁿ + Δt·[ −(u·∇)u + ν∇²u + (b−⟨b⟩) ŷ ]   (slow drift — the temperature clock)
uⁿ⁺¹  = ℙ u*                                       (one elliptic Leray projection — the pressure clock)
bⁿ⁺¹  = bⁿ + Δt·[ −(u·∇)b + κ∇²b ]                (parabolic temperature transport)
```
 
- The slow **drift** is the deterministic generator of the local **Markov / heat
  semigroup** (its heat kernel *is* the Gaussian transition density) — the
  *perfect* local, memoryless forecaster, with no ML/statistical noise. It is
  exactly what single-K closure and statistical predictors see.
- The **Leray projector** ℙ = I − ∇(∇²)⁻¹∇· is the divergence-free part of the
  Helmholtz split (Part 5's `helmholtz()`): one global Poisson solve that feels
  the whole domain at once.
 
Three results:
 
- **The cycle reappears from first principles (figure 22).** Warm fluid lifts as
  plumes, the pressure clock pulls cooler fluid inward (entrainment), and the
  shear between rising and inflowing streams rolls up into vortices — the exact
  mesoscale convection cell, with both clocks running inside it.
- **The local drift is divergent; the elliptic clock fixes it instantly
  (figure 23).** The intermediate `u*` from the slow drift alone violates mass
  conservation (RMS ∇·u\* ≈ 2.7×10⁻²) — the blind spot of any local, memoryless
  model. A single Leray projection drops the divergence to machine zero
  (≈2×10⁻¹⁵) through the global elliptic potential φ (∇²φ = ∇·u\*).
- **The SPDE limit — energy yes, structure no (figure 24).** Stochastic-climate /
  SPDE closures fake the unresolved fast clock with random forcing of the right
  *spectrum*. Replacing the deterministic projection correction with a
  phase-randomized field of **identical power spectrum** (Parseval-exact) yields
  a pointwise correlation of ≈0.01 with the true correction and leaves
  RMS ∇·u ≈ 3.8×10⁻² instead of zero. Matching a spectrum does **not**
  reconstruct a boundary-aware, globally-coupled elliptic operator.
 
This is why Markov chains and noise-injection SPDE schemes capture
slowly-diffusing anomalies yet stay blind to events where the fast pressure clock
rewrites the system's geometry. **Scope:** a 2D demonstration that the projection
method *is* the two-clocks synthesis and that spectrum-matching has a structural
ceiling; it does **not** prove 3D regularity / Beale–Kato–Majda, and the "Markov
chain" here is the advection–diffusion operator, not a data-trained transition
matrix.
 
---
 
## Part 7 — the two clocks in real reanalysis winds
 
Parts 5–6 split a *synthetic* general_two_clocks/compressible/Boussinesq flow into a slow
solenoidal part and a fast dilatational part. Part 7 runs the same split on
**real** horizontal winds. **NCEP/NCAR Reanalysis 1** (2.5°) is fetched live from
NOAA PSL over OPeNDAP (`general_two_clocks/reanalysis/ncep.py`, `general_two_clocks/run_reanalysis.py`; no credentials)
and Helmholtz-decomposed on the sphere — via the spherical-harmonic inversions
∇²ψ = ζ, ∇²χ = δ — into a **rotational** (non-divergent, balanced) wind and a
**divergent** wind.
 
> **Honesty note.** Horizontal wind divergence is *not* a model artifact: by 3-D
> mass continuity ∂u/∂x + ∂v/∂y = −∂w/∂z, it is the genuine footprint of vertical
> motion. The diagnostic is therefore the *Helmholtz split* — how much kinetic
> energy lives in the divergent (fast) vs rotational (slow) part — not the raw
> divergence.
 
| level | KE_div / KE_rot (daily mean) |
|---|---|
| 850 hPa | 4.0% |
| 500 hPa | 0.7% |
| 250 hPa | 1.0% |
 
- **The weather is rotational; the fast clock is a weak, structured residual
  (figure 25).** The rotational wind *is* the synoptic highs/lows and jets; the
  divergent wind is ~100× weaker (in energy) and concentrated in
  convergence/overturning zones (tropical convection, storm inflow) — the
  real-atmosphere analogue of the dilatational velocity.
- **Divergent ≪ rotational at every scale (figure 26).** Splitting the kinetic
  energy by spherical wavenumber *l* (from the SH power of ζ and δ, weighted by
  1/(l(l+1))), the divergent branch sits ~1–2 decades below the rotational branch
  for all *l*. Integrated, **KE_div/KE_rot ≈ 0.04** at 850 hPa — the real-data
  echo of KE_dil/KE_sol ~ M² ≪ 1 from Parts 5–6.
- **Vertical structure + a low-pass-filter check (figure 27).** The divergent
  fraction is *smallest near 500 hPa* — the classical **level of non-divergence**
  — and larger in the boundary layer and the upper-level jet outflow: physics,
  not tuning, sets the V-shape. And the **6-hourly** ratio at 500 hPa (1.1%)
  exceeds the **daily-mean** ratio (0.7%): time-averaging is a low-pass filter
  that scrubs the fast divergent clock — the same move that recovered the elliptic
  pressure field in Part 5 (averaging over the acoustic period).
 
**Scope:** reanalysis is a model-assimilated product, not raw observation, and
2.5° resolves only large scales (l ≤ 35). This is a real-data *confirmation* that
the wind's energy is overwhelmingly in the slow rotational clock with a weak,
physically structured fast divergent clock — not a turbulence-closure or 3-D
regularity claim.
 
---
 
## Part 8 — a two-clocks closure that generalizes K-theory
 
Parts 1–7 *diagnose* why single-eddy-diffusivity (K-theory) closure is blind to the
fast clock. Part 8 turns that into a *prescription*. Full derivation in
[`general_two_clocks/REPORT_THEORY.md`](general_two_clocks/REPORT_THEORY.md); the essentials:
 
- The **exact** resolved-scale dynamics is the Mori–Zwanzig generalized Langevin
  equation, ∂ₜâ = PℒÂ − ∫₀ᵗ K(t−s)â(s)ds + f(t): a slow Markov **drift**, a
  non-local-in-time **memory kernel**, and an orthogonal **noise** — with the
  *second fluctuation–dissipation theorem* locking the noise covariance to the
  kernel, K(τ) = (f(τ),f(0))·(â,â)⁻¹.
- **K-theory is exactly this equation with the memory kernel collapsed to a delta
  in time, ν_t taken constant in k, and the FDT-linked noise deleted** — the precise
  statement of its four sins (local, instantaneous, down-gradient-only, not
  divergence-consistent).
- The proposed closure keeps the structure but makes each term computable: a
  **nonlocal spectral eddy viscosity** ν_t(k) (Kraichnan cusp), a **Leray-projected,
  FDT-linked stochastic backscatter** ⟨f̂ᵢf̂ⱼ⟩ = 2ν_tᴮ|k|²Θ(k)ℙ_ij(k)δ(t−s), all
  wrapped in the projector ℙ so it is divergence-free *by construction*. It reduces
  to molecular NS as Δ→0 and to Smagorinsky in the local/no-noise limit — a strict
  generalization.
- This is the operator-level repair of the Part-6 SPDE failure: spectrum-matching
  set the noise independently of the dissipation and skipped the projection; the
  FDT linkage + ℙ restore *both* energy and structure.

### Part 8b — The decisive benchmark

A frozen-field a-priori test (`general_two_clocks/run_closure.py`): 256² DNS → sharp spectral filter at
k_c=32 → exact subgrid force m_true → score three models:

| model | E_m(k) | RMS(∇·m) | corr(T(k)) |
|---|---|---|---|
| **truth** | — | 7.4×10⁻¹⁴ | 1.000 |
| Smagorinsky (K-theory) | over-dissipates near k_c | 1.8×10⁻¹⁴ | **0.071** |
| spectrum-matched surrogate | matches | **12.0** | 0.907 |
| **projected-FDT** | matches | 1.7×10⁻¹⁴ | **1.000** |

Three decisive results:
1. **Smagorinsky** is purely dissipative — T(k)≤0 everywhere, no backscatter (Fig 30).
2. **Surrogate** matches spectrum but catastrophically fails div=0 — "energy yes, structure no."
3. **Projected-FDT** passes all three: correct spectrum, machine-zero divergence, perfect
   transfer correlation including backscatter.

Full results: [`general_two_clocks/REPORT_CLOSURE.md`](general_two_clocks/REPORT_CLOSURE.md) | Figures: 28–30

### Part 9c — the same benchmark in 3D (where vortex stretching lives)

Part 8b/9b is clean and CPU-verifiable but **2D**, where vortex stretching `ω·∇u` is
*identically zero* — the term that drives the forward cascade and singularity formation.
Part 9c (`general_two_clocks/run_closure3d.py`) re-runs the identical benchmark logic in
genuine **3D incompressible DNS** (velocity form, pseudo-spectral). The closure machinery
generalizes verbatim — the Leray projection is dimension-agnostic (`closure/spectral3d.py`,
`project3d`).

The 2D verdicts survive (Smagorinsky purely dissipative; surrogate breaks `div=0`;
projected-FDT matches spectrum + divergence + transfer, `corr(T)`≈1.0). The new payload is
the physics 2D structurally cannot see:

| 3D-only diagnostic | finding |
|---|---|
| vortex-stretching production `⟨ωᵢSᵢⱼωⱼ⟩` | **> 0** (forward-cascade engine; `=0` in 2D) |
| net transfer `Σₖ T(k)` | **< 0** — forward cascade (opposite sign to 2D's inverse cascade) |
| strain/vorticity alignment | Constantin–Fefferman intermediate-eigenvector geometry (sharpens with Re) |
| SGS backscatter **volume fraction** (`Π=-τ:S<0`) | **truth ≈ ½, Smagorinsky = 0 exactly** — K-theory's positive-definite eddy viscosity cannot represent the half of space that backscatters |

**Backend-agnostic** (`xp = numpy` or `cupy`): a CPU smoke test runs at `n=48`; the
developed-turbulence headline numbers come from a `--gpu` run at `n≥128` on a Tesla P100.
Full results: [`general_two_clocks/REPORT_CLOSURE3D.md`](general_two_clocks/REPORT_CLOSURE3D.md) | Figures: 61–64

**Scope:** a frozen-field a-priori test in a 3D periodic box — *not* a grid-converged
production DNS, an a-posteriori closed-loop LES, or a claim of 3D regularity. It establishes
that the structural repair and K-theory's structural failure both carry into 3D, where the
dynamics that matter for regularity actually live.
 
**Scope & honesty:** MZ optimal prediction, spectral eddy viscosity, and FDT
backscatter are established (see `general_two_clocks/REPORT_THEORY.md` *Prior art*); the contribution
is the two-clocks synthesis and the decisive 2D benchmark, not the invention of
those tools. No claim of a universal or "flawless" closure — turbulence closure is
open; the claim is exact structural consistency and removal of the specific failure
mode the repo isolated.
 
---

## Part 9 — the two clocks in a subglacial cavity (application benchmark)

Part 8b's frozen-field test, re-run in a glacier-relevant geometry: a 128²
penalized pseudo-spectral DNS of turbulent meltwater in a cavity between a bumpy
rock bed (warm) and a flat ice base (cold). The developed field is sharp-filtered
at k_c = 20 and the exact subgrid momentum force and heat flux are scored against
Smagorinsky (K-theory) and projected-FDT (`glaciers/run_subglacial.py`):

| model | rel. RMS(∇·m) | transfer corr. with truth | counter-gradient heat |
|---|---|---|---|
| **truth** | 1.1×10⁻¹⁴ | 1.000 | 50% of wake band |
| Smagorinsky (K-theory) | 8.8×10⁻¹⁵ | **−0.69** (anti-correlated) | 0% (by construction) |
| spectrum-matched surrogate | **9.7** (div fails) | 0.10 | — |
| **projected-FDT** | 8.0×10⁻¹⁵ | **+0.36** | — |

The same structural failure as Part 8b, now with a physical reading:

1. **Wake / basal drag:** Smagorinsky's transfer is T(k) ≤ 0 at every scale (no
   backscatter) and anti-correlated with the truth — a purely dissipative LES
   drifts toward a "dead wake". Projected-FDT restores the lee backscatter
   (positively correlated with the truth's scale-resolved transfer).
2. **Effective pressure:** the spectrum-matched surrogate breaks ∇·m = 0
   (a spurious bed pressure); the Leray-projected closures stay solenoidal.
3. **Basal melt:** half the exact subgrid heat flux in the lee is *counter-gradient*
   — invisible to a positive eddy diffusivity, so K-theory under-predicts the heat
   held against the bed.

Full results: [`glaciers/REPORT_SUBGLACIAL.md`](glaciers/REPORT_SUBGLACIAL.md) | Figures: 31–34

**Scope & honesty:** a **2D, a-priori (frozen-field) mechanism demonstration**, not
an operational glacier model. The bed/ice are embedded by volume **penalization**
(a true solid wall breaks the clean Leray-projection commutation the periodic
spectral method assumes); "melt" is a **heat-flux proxy**, not a Stefan phase
change; there is no ice rheology, real bed DEM, or coupled hydrology/sliding law.
This upgrades the subglacial application (a hand-written extrapolation in
`general_two_clocks/REPORT_CLOSURE.md`) from a qualitative analogy to a computed 2D structural claim.

### Real-bed variant (BEDMAP1 transect)

The same benchmark, but with the synthetic sinusoidal bed replaced by a **real
measured Antarctic bedrock transect** — a 220 km, 1757 m-relief airborne-radar
flight line from BEDMAP1 (British Antarctic Survey / SCAR Bedmap, CC-BY-4.0,
DOI:10.5285/f64815ec-4077-4432-9f55-0ce230f46029), mirrored for periodicity and
non-dimensionalised to the cavity (`glaciers/run_subglacial.py --bed real`):

| model | rel. RMS(∇·m) | transfer corr. with truth | counter-gradient heat |
|---|---|---|---|
| **truth** | 9.1×10⁻¹⁵ | 1.000 | 50% of wake band |
| Smagorinsky (K-theory) | 8.4×10⁻¹⁵ | **−0.71** (anti-correlated) | 0% (by construction) |
| spectrum-matched surrogate | **6.7** (div fails) | 0.70 | — |
| **projected-FDT** | 8.0×10⁻¹⁵ | **+0.45** | — |

The K-theory failure is **geometry-independent**: on real, asymmetric relief the
scorecard is the same as the idealized bed (Smagorinsky anti-correlated and purely
dissipative; surrogate breaks solenoidality; half the lee heat flux counter-gradient).
Full results: [`glaciers/REPORT_SUBGLACIAL_REAL.md`](glaciers/REPORT_SUBGLACIAL_REAL.md) | Figures: 35–38.
The turbulence is still simulated (no instrument resolves subglacial DNS); only the
**bed geometry** is now data-grounded.

### Why geometry → pressure → K-theory fails (figure 39)

A single computed picture for *why* the bed geometry breaks K-theory. Add one
**localized bump** to the bed and read off the localized continuity source it
creates, `q = −∇·(bump·u)`. Apply that *same* source to the two operators the
thesis separates (both functions of the Laplacian):

| clock | operator | response to a localized geometry source |
|---|---|---|
| pressure (elliptic) | `∇⁻²` (steady Poisson, Green's fn ~ln r) | **global** — **79%** of \|Δp\| lies beyond one bump-width; it spans the cavity |
| temperature (parabolic) | `e^{t∇²}` (heat kernel, Gaussian) | **local** — only **10%** beyond one bump-width; a confined blob |

K-theory models the subgrid force as `ν_t∇²ū` — a parabolic, neighbour-to-neighbour
operator, structurally in the *local* row. But the pressure that enforces
incompressibility is the *global* `∇⁻²` of the geometry-induced source. No local
diffusivity can reproduce a global inverse-Laplacian — that is the blindness, in one
source and two operators. Full note: [`general_two_clocks/REPORT_ELLIPTIC_PRESSURE.md`](general_two_clocks/REPORT_ELLIPTIC_PRESSURE.md)
(`general_two_clocks/run_elliptic_demo.py`) | Figure 39.

---

## Part 10 — the two clocks at a turbulent vortex (Goldshtik–Sorokin levitation)

The same frozen-field test, re-run at a **turbulent vortex**. A 128² pseudo-spectral
DNS sustains a coherent central vortex (peak azimuthal speed ≈1.4, core radius ≈0.7)
in a turbulent skirt. The incompressible pressure digs a deep, *global* (elliptic)
low-pressure well at the core — the structure that holds a heavy particle up against
gravity in the **Goldshtik–Sorokin effect**. The developed field is sharp-filtered at
k_c = 20 and the exact subgrid force scored against Smagorinsky and projected-FDT
(`atmosphere/run_swirl.py`):

| model | rel. RMS(∇·m) | net resolved SGS power ⟨ū·m⟩ | suspension margin | verdict |
|---|---|---|---|---|
| **truth** | 2.9×10⁻¹⁴ | −2.1×10⁻⁵ | 1.000 (calibration) | suspended |
| Smagorinsky (K-theory) | 8.4×10⁻¹⁵ | **−6.2×10⁻³** (≈290× truth) | **0.36** | **particle falls** |
| spectrum-matched surrogate | **8.1** (div fails) | +9.9×10⁻⁶ | 1.00 | spurious core pressure |
| **projected-FDT** | 8.0×10⁻¹⁵ | −1.6×10⁻⁵ | 1.00 | particle stays up |

The **levitation force scales with the core-well depth, which scales with the resolved
swirl energy** (p ~ ρ|u|²). A purely dissipative eddy viscosity has no backscatter, so
it drains the swirl ≈290× faster than the truth; over one core turnover the well
shoals and the **suspension margin drops below 1 — the particle falls**. Projected-FDT
tracks the truth's net drain and keeps the margin ≈1 — **the particle stays up**. The
surrogate doesn't over-drain, but it breaks ∇·m = 0, injecting a spurious pressure on
the very well it would need to preserve.

Full results: [`atmosphere/REPORT_SWIRL.md`](atmosphere/REPORT_SWIRL.md) | Figures: 40–43.

### Real-vortex variant (Hurricane Otis 2023)

The same test, with the swirl's target profile **calibrated to a real hurricane**
(mirroring Part 9's real-bed run). From the NHC best track (HURDAT2, bundled in
`atmosphere/swirl/data/otis_besttrack.csv`) we read Hurricane Otis's peak-intensity vortex —
the record-breaking Cat-5 that hit Acapulco on 25 Oct 2023: minimum pressure
**922 mb**, Vmax **145 kt**, a **5 nmi pinhole eye**. A Holland (1980) profile is
fitted to that wind–pressure relation (shape parameter **B ≈ 2.05**) and handed to
the solver as its sustained vortex. The verdict survives real-vortex calibration:

| model | net resolved SGS power ⟨ū·m⟩ | suspension margin | verdict |
|---|---|---|---|
| **truth** | −4.3×10⁻⁵ | 1.000 (calibration) | suspended |
| Smagorinsky (K-theory) | **−7.4×10⁻³** (≈170× truth) | **0.69** | **particle falls** |
| spectrum-matched surrogate | −9.0×10⁻⁸ (div fails, ∇·m≈9.4) | 1.00 | spurious core pressure |
| **projected-FDT** | −3.2×10⁻⁵ | 1.00 | particle stays up |

The real data calibrate only the **mean vortex geometry** (Otis's Holland shape and
deficit-to-wind relation), not the closure under test; the ≈9 km pinhole eye is
sub-grid at 128², so it is non-dimensionalised to the box exactly as Part 9's bed is.
Run it with `python atmosphere/run_swirl.py --swirl otis`.
Full results: [`atmosphere/REPORT_SWIRL_REAL.md`](atmosphere/REPORT_SWIRL_REAL.md) | Figures: 44–47.

**Scope & honesty:** a **2D, a-priori (frozen-field) mechanism demonstration**, not an
operational model. The real Goldshtik–Sorokin effect is **3D axisymmetric** with vortex
stretching (absent in 2D); the "particle" is a **force-balance proxy** (the suspension
margin is a one-turnover extrapolation of the measured SGS power, not an integrated
Lagrangian trajectory), calibrated so the truth marginally suspends it, so only the
*relative* ordering of closures is meaningful. It demonstrates the operator-level
mechanism — a global elliptic well that a local eddy viscosity drains away.

---

## Scope and Honesty

What this repo **does** test: the temporal two clocks (Part 1), the spatial
elliptic-vs-parabolic structure (Part 2), the insufficiency of single-Prandtl
K-theory (Part 3), and — in **real** reanalysis winds — that the kinetic energy is
overwhelmingly rotational (slow clock) with only a weak divergent (fast clock)
fraction (Part 7).

Parts 4–6 *demonstrate* the hyperbolic→elliptic crossover and its synthesis —
Part 4 with a linear-acoustics toy model (the change of PDE type), Part 5 with a
nonlinear 2D compressible Navier–Stokes Mach sweep (scale separation + the
elliptic limit in an actual nonlinear flow), and Part 6 by reuniting the two
clocks via the projection method in a 2D Boussinesq flow (and exposing the
structural limit of spectrum-matching SPDE surrogates). None of *those* is a real
3D turbulent flow — Part 7 brings the same energy split to real reanalysis data.

What the repo **does not** test: rigorous wave-radiation-damping /
Beale–Kato–Majda 3D-regularity arguments. (Part 6 uses the projection method only
as a 2D structural demonstration, not as a regularity proof.) Those require
rigorous PDE analysis or a full nonlinear 3D compressible simulation — not a flux
tower, a 2D incompressible slice, or a 2D general_two_clocks/compressible/Boussinesq demo. Claims
there are flagged as out of scope rather than asserted.

> **On 20 Hz data:** the NEON DP4.00200 bundle only contains aggregated
> statistics (1, 2, 3, 6, 9, 30-min). The raw 20 Hz turbulence series is a
> separate, much larger product, so seconds-scale dynamics cannot be resolved
> from this file.

---

## Getting the Data

None of the raw data is committed (sizes range from ~130 MB to ~1 GB).

| dataset | source | local path | download needed? |
|---|---|---|---|
| NEON DP4.00200 | eddy-flux ZIP | `data/` | yes |
| ASOS 1-min | IEM archive CSVs | `data_asos/` | yes |
| Rayleigh–Bénard DNS | The Well (polymathic-ai) | `data_rb/` | yes |
| NCEP/NCAR Reanalysis 1 | NOAA PSL OPeNDAP | `data_reanalysis/` (auto-cached) | **no** |
| Parts 4–6, 8b, 9, 10 | self-generated | — | **no** |
| Part 9 real bed | BEDMAP1 transect (BAS, CC-BY-4.0) | `glaciers/subglacial/data/` (bundled) | **no** |
| Part 10 real vortex | Hurricane Otis best track (NHC HURDAT2, public domain) | `atmosphere/swirl/data/` (bundled) | **no** |

---

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# === Parts requiring downloaded data ===
python general_two_clocks/run_analysis.py   --data-dir data      --out-dir general_two_clocks/figures --report general_two_clocks/REPORT.md
python general_two_clocks/run_stability.py  --data-dir data      --out-dir general_two_clocks/figures --report general_two_clocks/REPORT_STABILITY.md
python general_two_clocks/run_asos.py       --data-dir data_asos --out-dir general_two_clocks/figures --report general_two_clocks/REPORT_ASOS.md
python general_two_clocks/run_rb.py         --data data_rb/rb_Ra1e6_Pr1.hdf5 --out-dir general_two_clocks/figures --report general_two_clocks/REPORT_RB.md

# === Self-contained (no data download) ===
python general_two_clocks/run_compressible.py     --out-dir general_two_clocks/figures --report general_two_clocks/REPORT_COMPRESSIBLE.md
python general_two_clocks/run_compressible_ns.py  --out-dir general_two_clocks/figures --report general_two_clocks/REPORT_NS.md
python general_two_clocks/run_boussinesq.py       --out-dir general_two_clocks/figures --report REPORT_BOUSSINESQ.md
python general_two_clocks/run_reanalysis.py       --out-dir general_two_clocks/figures --report REPORT_REANALYSIS.md  # streams from NOAA
python general_two_clocks/run_closure.py          --out-dir general_two_clocks/figures --report general_two_clocks/REPORT_CLOSURE.md  # ← THE BENCHMARK (2D)
python general_two_clocks/run_closure3d.py        --out-dir general_two_clocks/figures --report general_two_clocks/REPORT_CLOSURE3D.md  # ← Part 9c (3D, CPU smoke test)
python general_two_clocks/run_closure3d.py --gpu --n 128 --kc 24 --steps 2500 --out-dir general_two_clocks/figures --report general_two_clocks/REPORT_CLOSURE3D.md  # ← Part 9c (3D, Tesla P100)
python glaciers/run_subglacial.py       --out-dir glaciers/figures   # ← Part 9 (idealized bed, Fig 31–34)
python glaciers/run_subglacial.py --bed real --out-dir glaciers/figures  # ← Part 9 (real BEDMAP1 bed, Fig 35–38)
python general_two_clocks/run_elliptic_demo.py    --out-dir general_two_clocks/figures   # ← Part 9 (geometry→global pressure, Fig 39)
python atmosphere/run_swirl.py            --out-dir atmosphere/figures   # ← Part 10 (Goldshtik–Sorokin swirl, Fig 40–43)
python atmosphere/run_swirl.py --swirl otis --out-dir atmosphere/figures # ← Part 10 (real Hurricane Otis vortex, Fig 44–47)
```

Each command regenerates its figures in `figures/` and rewrites its report with
freshly computed numbers.

---

## Mainstream-model corrections (drop-in)

Two of the framework's falsified-mainstream results are now packaged as **drop-in
corrections** an operational model can call directly. Both are CPU-only with
deterministic tests, and both reduce to the original local model in the appropriate
limit (the safe default):

- **Gradient-diffusion / K-theory** —
  [`general_two_clocks/nonlocal_flux_correction.py`](general_two_clocks/nonlocal_flux_correction.py).
  THEOREM (proved + tested): any local closure with a non-negative eddy diffusivity
  `F = −κ(x)∇C`, `κ ≥ 0`, has `F·∇C ≤ 0` pointwise, so the counter-gradient alignment
  `C_G = ⟨F·∇C⟩/⟨|F||∇C|⟩` is pinned at `−1` and cannot represent the counter-gradient
  flux measured in RESULT 11 ([`glaciers/REPORT_CG_BUOYANCY.md`](glaciers/REPORT_CG_BUOYANCY.md)).
  The fix is a scale-dependent (nonlocal) diffusivity `κ̂(k)` that admits backscatter
  and reduces to local Fick when constant.
- **GlaDS-type subglacial hydrology → sliding** —
  [`glaciers/subglacial/glads_memory_correction.py`](glaciers/subglacial/glads_memory_correction.py).
  Eliminating the channel from the linearised cavity↔channel system is an exact
  Mori–Zwanzig projection (RESULT 21); the mainstream *local* effective-pressure
  closure is its fast-channel limit, which drops the post-drainage surge lag.
  `apply_memory_kernel(...)` convolves the channel memory back in.

---

## Repo Layout

The repository is organized into four domain folders plus a working-notes
folder. Each domain folder has its own `README.md` (status: found / [HYP] /
pending), its reports, its runner scripts and libraries, and its own
`figures/` and `tests/` directories.

```
K/
├── README.md                      ← you are here (thesis, roadmap, history)
├── FUTURE_WORK.md                 ← master status ledger (§A–§H, claim tags)
├── PAPER_OUTLINE.md               ← paper skeleton
├── CONTRIBUTING.md                ← repo rules
├── make_presentation.py           ← slide-deck builder (resolves figures across folders)
│
├── general_two_clocks/            ← THE OBSERVATION + implications + verified proofs
│   ├── README.md                  │  what we observed, what it implies, what is proven
│   ├── REPORT*.md                 │  Parts 1–8 reports + RESULT 12/13/15 closure results
│   ├── neon_pt/ asos/ reanalysis/ │  observational libraries (NEON, ASOS, NCEP)
│   ├── compressible/ boussinesq/  │  demo solvers (Parts 4–6)
│   ├── closure/                   │  Part 8b/9b MZ/FDT benchmark (dns2d, sgs) + 9c 3D (dns3d, sgs3d, spectral3d)
│   ├── run_*.py                   │  Part runners (run_closure.py = THE BENCHMARK)
│   ├── gle_coefficients.py        │  RESULT 12 (τ_c measured)
│   ├── scalar_clock_universality.py │ RESULT 13 (τ_c is a turbulence clock)
│   ├── cmn_solver_demo.py         │  RESULT 15 (CMN correction in a K-theory solver)
│   ├── nonlocal_flux_correction.py │ K-theory → nonlocal eddy-diffusivity correction (drop-in)
│   ├── figures/                   │  Figs 01–30, 39, 53, 55, 60
│   └── tests/
│
├── atmosphere/                    ← atmospheric application (turbulent vortex)
│   ├── README.md                  │  found / HYP / pending
│   ├── swirl/                     │  swirl DNS + Hurricane Otis best-track calibration
│   ├── run_swirl.py               │  Part 10 runner (idealized + Otis)
│   ├── REPORT_SWIRL.md, REPORT_SWIRL_REAL.md
│   ├── figures/                   │  Figs 40–47
│   └── tests/
│
├── glaciers/                      ← subglacial cavity + ice-sheet validation
│   ├── README.md                  │  found / HYP / pending
│   ├── subglacial/                │  cavity DNS/LES library, candidates, Stefan, THEORY_CAVITY.md
│   │                              │  + glads_memory_correction.py (GlaDS-type mainstream retrofit)
│   ├── validation/                │  §V/§H pipeline (Bedmap2, BedMachine, lakes, ITS_LIVE)
│   ├── run_subglacial*.py, run_candidate{1,3,4}.py, run_stefan_sweep.py
│   ├── scallop_*.py               │  scallop instability / migration / channelisation family
│   ├── theorem3_cg_gpu_probe.py   │  RESULT 11 (counter-gradient C_G, GPU)
│   ├── REPORT_*.md                │  subglacial, candidates, scallops, RTN/GL-migration, sliding
│   ├── figures/                   │  Figs 31–38, 49–52, 56–59, candidate JSONs
│   └── tests/
│
├── ocean/                         ← ice–ocean interface (salt + temperature)
│   ├── README.md                  │  found / HYP / pending
│   ├── run_candidate2.py          │  Candidate 2 double-diffusion sweep
│   ├── scallop_doublediff.py      │  §D.2 scalloped-wall double-diffusion regime mixing
│   ├── REPORT_CANDIDATE2.md, REPORT_DOUBLEDIFF.md
│   ├── figures/                   │  Figs 48, 50
│   └── tests/
│
├── working_notes/                 ← scratch probes / plans that fit no single domain
│   └── README.md                  │  what each note is and how to run it
│
├── requirements.txt
└── pyproject.toml                 ← pytest wiring (testpaths + pythonpath for all folders)
```

Shared physics libraries are imported across folders (e.g. `atmosphere/run_swirl.py`
uses `closure/` from `general_two_clocks/` and `subglacial/` diagnostics from
`glaciers/`); the runner scripts bootstrap `sys.path` accordingly and `pytest`
is wired up via `pyproject.toml`, so everything runs from the repo root.

## Tests

```bash
pytest -v
```

308 tests covering signal processing, compressible solver accuracy, stability
classification, closure structural properties (solenoidality, backscatter,
filtering), the subglacial cavity (penalty masks, counter-gradient heat flux), the
3D a-posteriori cavity LES (multi-site BEDMAP1 beds, Leray projection, advective
melt flux, SGS dissipation breakdown), the moving-boundary Stefan prototype (1D
Neumann similarity reference + 2D melt feedback), the THEORY_CAVITY.md commutator
checks, the Goldshtik–Sorokin swirl (core pressure
well, suspension margin), and the real Hurricane Otis calibration (best-track
loader, Holland fit, real-vortex margin). No
downloaded data required — tests use synthetic signals, the bundled Otis best track,
and short in-process spinups.

---

## Figures

All figures are numbered sequentially and generated by the corresponding runner script. They now live in each domain's `figures/` folder: 01–30, 39, 53, 55, 60 in `general_two_clocks/figures/`; 40–47 in `atmosphere/figures/`; 31–38, 49–52, 56–59 + candidate 1/4 JSONs in `glaciers/figures/`; 48 and 50 in `ocean/figures/`:

| # | file | content | part |
|---|------|---------|------|
| 1 | `01_diurnal_composites.png` | T/P diurnal composites | 1 |
| 2 | `02_lead_lag.png` | solar → T → H lead-lag | 1 |
| 3 | `03_spectra_two_clocks.png` | T (24h) vs P (12h) spectra | 1 |
| 4 | `04_pressure_temperature_coupling.png` | P–T scatter (r≈0.07) | 1 |
| 5 | `05_shear_turbulence.png` | shear-driven turbulence | 1 |
| 6 | `06_stability_distribution.png` | stability class histogram | 3 |
| 7 | `07_transport_decoupling.png` | momentum/heat ratio swing | 3 |
| 8 | `08_flux_variance_similarity.png` | flux-variance similarity | 3 |
| 9 | `09_asos_spectra.png` | ASOS multi-station spectra | 1b |
| 10 | `10_asos_diurnal.png` | ASOS diurnal composites | 1b |
| 11 | `11_tide_vs_latitude.png` | S₂ tide vs latitude | 1b |
| 12 | `12_asos_coupling.png` | ASOS P–T coupling | 1b |
| 13 | `13_rb_fields.png` | RB DNS fields (p, b, ω) | 2 |
| 14 | `14_rb_autocorrelation.png` | spatial autocorrelation | 2 |
| 15 | `15_rb_spectra.png` | pressure vs buoyancy spectra | 2 |
| 16 | `16_acoustic_pulse.png` | radiating acoustic pulse | 4 |
| 17 | `17_crossover.png` | hyperbolic→elliptic crossover | 4 |
| 18 | `18_field_match.png` | compressible→Poisson match | 4 |
| 19 | `19_ns_structure.png` | NS sol/dil decomposition | 5 |
| 20 | `20_ns_two_clocks.png` | KE time series (fast/slow) | 5 |
| 21 | `21_ns_mach_sweep.png` | Mach sweep KE_dil/KE_sol | 5 |
| 22 | `22_boussinesq_convection.png` | Boussinesq convection cell | 6 |
| 23 | `23_projection_step.png` | Leray projection: div→0 | 6 |
| 24 | `24_spde_limit.png` | SPDE limit: E yes, struct no | 6 |
| 25 | `25_reanalysis_two_clocks.png` | rotational vs divergent wind | 7 |
| 26 | `26_reanalysis_ke_spectrum.png` | KE(l) rot vs div | 7 |
| 27 | `27_reanalysis_levels.png` | KE_div/KE_rot vs level | 7 |
| 28 | `28_closure_force_spectrum.png` | E_m(k) — force spectrum | 8b |
| 29 | `29_closure_divergence.png` | RMS(∇·m) — solenoidality | 8b |
| 30 | `30_closure_transfer.png` | T(k) — energy transfer | 8b |
| 31 | `31_subglacial_fields.png` | cavity DNS: speed, ω, heat, pressure | 9 |
| 32 | `32_subglacial_transfer.png` | T(k) backscatter + solenoidality scorecard | 9 |
| 33 | `33_subglacial_backscatter_map.png` | spatial backscatter Π in the lee | 9 |
| 34 | `34_subglacial_heatflux.png` | subgrid heat flux: exact vs K-theory | 9 |
| 35 | `35_subglacial_fields.png` | real-bed (BEDMAP1) cavity DNS: speed, ω, heat, pressure | 9 |
| 36 | `36_subglacial_transfer.png` | real-bed T(k) backscatter + solenoidality scorecard | 9 |
| 37 | `37_subglacial_backscatter_map.png` | real-bed spatial backscatter Π in the lee | 9 |
| 38 | `38_subglacial_heatflux.png` | real-bed subgrid heat flux: exact vs K-theory | 9 |
| 39 | `39_elliptic_pressure.png` | geometry bump → global Δp (elliptic) vs local Δθ (parabolic) | 9 |
| 40 | `40_swirl_fields.png` | turbulent swirl DNS: speed, ω, pressure well + radial profiles | 10 |
| 41 | `41_swirl_transfer.png` | T(k) energy transfer + solenoidality scorecard | 10 |
| 42 | `42_swirl_transfer_map.png` | spatial transfer Π = ū·m around the core | 10 |
| 43 | `43_swirl_levitation.png` | suspension margin — particle stays up vs falls | 10 |
| 44 | `44_otis_calibration.png` | Hurricane Otis best track + fitted Holland vortex | 10 |
| 45 | `45_swirl_fields.png` | Otis-calibrated swirl DNS: speed, ω, core well + profiles | 10 |
| 46 | `46_swirl_transfer.png` | Otis-calibrated T(k) transfer + solenoidality scorecard | 10 |
| 47 | `47_swirl_levitation.png` | Otis-calibrated suspension margin verdict | 10 |
| 48 | `48_candidate2_doublediff.png` | double-diffusive Nu_T/Nu_S vs R_rho (Candidate 2 probe) | 10c |
| 49 | `49_scallop_channel_feedback.png` | scallop reattachment as a phase-locked channel-nucleation source | §A.3/§D.1 |
| 50 | `50_scallop_doublediff.png` | scalloped wall → spatially heterogeneous double-diffusive regimes (`Nu_T(x)`, `Nu_S(x)`) | §D.2 |
| 50b | `50b_scallop_channel_z0_robustness.json` | channel site-selection invariant to the `z_0` roughness closure | §D.1 |
| 51 | `51_shedding_deterministic.json` | deterministic-limit test of the vortex-shedding onset `a_shed` ([NULL]) | §D.5 |
| 52 | `52_theorem3_cg.json` | counter-gradient parameter `C_G` on a 3D active-buoyancy LES (partial) | Thm 3 |
| 53 | `53_gle_coefficients.json` | unified-memory GLE coefficients — `τ_c` measured, §G.5 net-zero time-lag, slow:fast bath weight `= St` | RESULT 12 (§D.4/§G.5) |
| 54 | `54_scalar_clock_universality.json` | `τ_c` is a turbulence clock, not a scalar clock (fixed `Le=100`) — *artifact not committed* | RESULT 13 |
| 55 | `55_le_sweep_clock_vs_amplitude.json` | Le-sweep: `τ_c(salt)/τ_c(heat)`≈1 while `Nu_S/Nu_T` tracks Le | RESULT 13 |
| 56 | `56_scallop_amplitude_harmonics.json` | scallop wall-flux harmonic decomposition: `s=−β+iω_mig` | 10g |
| 57 | `57_rtn_phi_calibration.json` | gauge-RTN on Bedmap2: RTN>1 area inverts for basal water fraction `φ`; critical-thickness `H*` / pre-flotation MISI margin | §G.3/§H.1.1 |
| 58 | `58_scallop_field_test.json` | RESULT 14 constant-free field test: `I=τ·c_mig/λ` (harmonic-pin self-check + committed Bushuk adjustment-regime bound vs solver band) | 10g/§G.2 |
| 59 | `59_scallop_amplitude_closure.json` | §G.6 amplitude sweep (`a/λ∈[0.05,0.30]`): mean-Nu deficit is amplitude-independent (`p≈0`) → `(1+ζ(a/λ)²)` closure **falsified**; local lee `R_max` still grows | §G.6 |
| 60 | `60_cmn_solver_demo.json` | §G.5/§H.3 clock-mismatch correction in a transient K-theory thermal solver: `CMN=+τ_c` cuts transient error ~15× (`∝τ_c²` vs naive `∝τ_c`), null in steady turbulence, `+` sign is error-reducing | §G.5/§H.3 |

> **Numbering notes.** Figure 50 has two companion probes distinguished by a sub-label: `50_scallop_doublediff` (§D.2) and `50b_scallop_channel_z0_robustness` (§D.1). Figure 54 (`general_two_clocks/scalar_clock_universality.py` default fixed-`Le` run) is defined but its JSON artifact is not committed — regenerate it with `python general_two_clocks/scalar_clock_universality.py`. Several later figures are JSON-only data artifacts (no `.png`).
