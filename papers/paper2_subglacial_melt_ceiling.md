# Why a scalloped cold-grounded ice base melts less on average: an exact area-partition below the conductive sublayer

---

## Abstract

Melt at an ice–water interface is rate-limited by transfer through a thin thermal/haline
sublayer — the basis of the standard three-equation melt parameterization (Holland &
Jenkins 1999; McPhee 2008; Kader & Yaglom 1972). For rough or scalloped ice, the
prevailing expectation is that geometry *enhances* this transfer and raises melt (Gilpin
et al. 1980; Claudin et al. 2017; Bushuk et al. 2019). Using penalized (Brinkman volume-penalization) DNS/LES of a turbulent cavity
between a cold ice ceiling and a bed, we report the **opposite** in the grounded,
cold-walled regime, and give the exact mechanism. Our central result is an **exact
area-partition identity**: the mean basal heat a scalloped wall delivers, relative to a
flat wall, decomposes to machine precision as `Nu/Nu_flat = C_reatt + C_thick + C_rev`
(reattachment-thin, thickened, and reversed populations). Although lee-side reattachment
lifts *local* melt `2–3.3×` above its own conduction baseline, the *mean* stays **below** a
flat wall (`Nu/Nu_flat<1`, peak `0.922` at the reference resolution) at every amplitude, because the
thickened+reversed lee deficit outweighs the thin-reattachment surplus; a two-moment
`(1+CV²)` truncation is falsified (`CV_δ~O(1)`), so the suppression is
distribution-dominated, not moment-dominated. The ceiling is **boundary-condition
robust** — relaxing the cold-Dirichlet ice wall to a finite-conductance Robin wall, and
the no-slip bed to a Navier slip that raises near-bed speed `~100×`, both leave the mean
unchanged — which pins the limit to the thermal conductive sublayer rather than momentum
transport, consistent with the three-equation picture. Two further candidate mechanisms
testable in 2-D (intermittent roughness plumes; tidal hydraulic switching) also produce
no positive mean signal across three subgrid closures `{none, Smagorinsky, backscatter}`;
the double-diffusive case, being intrinsically three-dimensional, is reported here as
**untested in 2-D**, not as a null. We frame the deliverable as a **regime map**: Type I
(grounded, cold-walled) — no mean enhancement; Type II (grounding line, flux boundary) —
possible; Type III (open shelf) — yes. Scope is explicit: the candidate sweeps are 2-D (3-D
only for the capstone and the counter-gradient flux probe) with Brinkman penalization. A
three-seed wall-normal convergence and penalization (`η`) study shows the suppression
*sign* is robust — `Nu/Nu_flat<1` at every resolution (`ny=48–192`), penalization
strength (4× `η`), and forcing seed — while its *magnitude* is not grid-converged (it
varies non-monotonically across resolution, `0.74–0.99`, with per-grid seed scatter
`±0.005`). We therefore claim only the resolution-robust results: the *sign* of the
ceiling and the *exact* area-partition identity, which holds to machine precision at every
resolution. In this regime, scallop geometry suppresses rather than enhances mean basal melt.

---

## 1. Introduction

Basal melt and the effective pressure it modulates set ice-stream sliding and
grounding-line stability (Schoof 2010). Melt at an ice–water interface is rate-limited by
molecular transfer through a thin thermal/haline sublayer — the basis of the standard
three-equation parameterization (Mellor et al. 1986; Holland & Jenkins 1999; McPhee 2008;
Kader & Yaglom 1972). For rough or scalloped ice the prevailing expectation, from
laboratory and theory, is that geometry *enhances* this transfer and raises melt (Gilpin
et al. 1980; Claudin et al. 2017; Bushuk et al. 2019). Operational subglacial models, by
contrast, carry no
flow-enhanced-melt term in the grounded regime (Röthlisberger 1972; Werder et al. 2013).
Whether *resolved cavity turbulence* adds melt above the conduction limit at a cold,
grounded ice base is rarely tested with resolved simulation.

We test it with resolved penalized DNS/LES under a pre-stated discipline: each candidate
mechanism is stated as a falsifiable expectation **before** the run, so a null is
informative rather than a tuning failure. The result runs against the
roughness-enhancement expectation: in the grounded, cold-walled regime, scallop geometry
*suppresses* mean melt, and we give the exact mechanism (an area-partition identity) and a
regime map for where it applies.

**Contributions.**
1. An **exact area-partition identity** `Nu/Nu_flat = C_reatt + C_thick + C_rev`
   explaining why a scalloped cold-grounded wall melts *less* on the mean despite `2–3×`
   local lee hot spots — opposite to the roughness-enhancement expectation (§4.3).
2. **Falsification of the `(1+CV²)` two-moment truncation**: the suppression is
   distribution-dominated, not moment-dominated (§4.3).
3. **Boundary-condition robustness** (Robin wall; Navier slip ×100) pinning the mean
   ceiling to the thermal conductive sublayer rather than momentum transport (§5).
4. A **pre-stated, closure-swept null map** of candidate enhancement mechanisms in the
   cold-grounded (Type-I) cavity, with double diffusion identified as requiring 3-D and
   reported as untested here (§3).
5. A **regime map** (Type I/II/III) scoping the result (§6).
6. The growth-carrying **local lee-flux law** `R_max(a/λ)` — linear, not quadratic,
   bounded `1.9–4.3`, with separation onset at `a/λ≈0.11` (§4.4).

---

## 2. Methods

**2.1 Cavity model.** 2-D/3-D penalized (Brinkman volume-penalization) incompressible
DNS/LES of a turbulent cavity between a cold ice ceiling and a bed, with a thermal field
and an interfacial melt diagnostic; illustrative real bed transects (Lythe & Vaughan 2001)
supply geometry realism (the area-partition result is geometry-general and does not depend
on the specific bed dataset). The interfacial melt obeys a Stefan condition `ρ_i L v = q`,
`q = −k_th ∂θ/∂n` (Stefan 1891; Nye 1953; Crank 1984). We define the flat-wall
control, the conduction-relative enhancement `R_mean = ⟨m_flow/m_cond⟩` (each column
normalised by its **own** flow-off conduction on the same geometry), and the
flat-baseline Nusselt ratio `Nu/Nu_flat = ⟨m_bump⟩/⟨m_flat⟩` (normalised by the flat
wall with flow **on**). These two ratios have **different baselines** and are not
interchangeable — a distinction that resolves an apparent tension in §4.

**2.2 Pre-stated go/no-go criteria.** Each candidate's expected signature and its
falsifiable threshold are fixed before the run (stated per candidate below). Each is
swept across three closures `{none, Smagorinsky, backscatter}` so a verdict that is
closure-independent cannot be a closure artefact.

**2.3 Type-I regime.** "Type I" denotes the grounded, cold-walled, closed cavity:
cold-Dirichlet ice ceiling, bounded water layer. The hypothesis under test is that mean
melt here is bounded by the **thermal conductive sublayer** rather than by momentum
stagnation. §5 stress-tests this against the boundary conditions themselves; §6
generalises to other regimes.

**2.4 Provenance.** CPU runs plus Tesla P100 (CuPy 14.0.1) confirmations where noted.
GPU-only numbers (counter-gradient `C_G`; double-diffusion variance amplification) are
tagged as such.

---

## 3. The null map — three subgrid-driven mechanisms

**3.1 Candidate 1 — intermittent plumes from ice-base roughness.**
*Pre-stated criterion:* band-limited roughness on the ice base should make interfacial melt
*intermittent* (skew `S>0`, excess kurtosis `K>3`, peak/mean `>2`), humped at
intermediate `Ri≈0.3–0.6`.
*Result:* thresholds **not met** and the statistics are **flat in `Ri`** (relative
spread `<2%`, closure-independent): pooled `peak/mean≈1.8`, `skew≈−0.75`, `kurt≈2.2`.
The variability is **geometric, not plume-driven** — thinner ice columns conduct faster
— and switching the flow off entirely (`f_amp=0`, pure conduction) reproduces
`melt_mean` and `peak/mean` to ~1 %. Mean melt is conduction-limited:
`melt_mean≈2.17×10⁻⁴`, constant to four significant figures across all `Ri` and
closures. The genuine turbulent signal lives in the interior flux `F_turb=⟨v'θ'⟩`, not
at the interface. (roughness / boundary-layer-plume literature)

**3.2 Candidate 2 — double-diffusive layering (untested in 2-D).**
*Pre-stated criterion:* with `Le=100` (heat diffuses ~100× faster than salt), salt fingers
should give a *humped* thermal Nusselt `Nu_T(R_ρ)` peaking near `R_ρ≈2`.
*Result, and why it is not a null:* the 2-D runs show **no hump** (`Nu_T` falls
monotonically `1.44→0.73`), but salt fingering and double-diffusive staircases are
intrinsically three-dimensional, so a 2-D penalized forced run cannot represent them. We
therefore report this mechanism as **untested here**, not as evidence against
double-diffusive enhancement; a 3-D unforced double-diffusive run is required (§7). What
the 2-D runs *do* show is a real differential-diffusion signature — `Nu_S ≫ Nu_T` (`~30×`
near `R_ρ=1`) and counter-gradient salt flux at strong stabilisation (`Nu_S≈−26`,
`R_ρ=10`), which a positive eddy diffusivity cannot represent — and the Turner flux ratio
`γ≈1/R_ρ` (to ~1 %). (Fig. 1)

![Double-diffusive heat and salt Nusselt numbers (`Nu_T`, `Nu_S`) versus the density ratio `R_ρ` in the grounded subglacial cavity. The backscatter-resolving closure resists the conductive-sublayer collapse (`Nu_T=0.86` versus `0.73` unclosed at `R_ρ=10`); the cold-wall mean-melt ceiling is nonetheless unchanged.](figures/48_candidate2_doublediff.png)

**3.3 Candidate 4 — tidal hydraulic switching.**
*Pre-stated criterion:* under a tidal body force a wide cavity should switch between *filled*
(`H1→1`) and *stratified* (`H1→0.2`) states, with time-averaged melt peaking at
intermediate `Ri` (switching frequency `f_switch` maximal there).
*Result:* at every `Ri` the cavity is **monostable filled** (`H1≈0.90±0.01`,
`f_switch=0`) — no hump. (An earlier "dead-flow" null was a numerical under-resolution
artefact: `ny=48` let the Brinkman penalty bleed into the ~10-row fluid gap; resolving
at `ny≥64` restores normal flow, `umax≈0.6`, KE up ~500×, so the null is physical, not a
construction artefact.) (subglacial hydraulic-switching / drainage-state lit.)

> **Null-map summary.** Neither 2-D-testable subgrid-driven mechanism (roughness plumes,
> tidal switching) lifts the *mean* interfacial melt above conduction in the Type-I cavity,
> across all three closures; double diffusion is untested in 2-D (§3.2). The flow does
> carry real interior structure (counter-gradient flux, `F_turb`), but it does not reach
> the cold wall as net mean melt.

---

## 4. The scallop — a positive local signal that is mean-bounded

Candidate 3 (the scallop / melting instability) is the one mechanism with a positive,
flow-driven *interfacial* signal, and the cleanest place to establish the ceiling.
GPU-confirmed (Tesla P100).

**4.1 The pre-stated runaway is sign-reversed — the correct ice physics
.** The roughness-growth hypothesis (thin ice → faster melt → more roughness
→ runaway, `Λ>0`) is falsified *with the opposite sign*: the conduction-pinned base
**self-smooths**, `Λ≈−0.265 (<0)`, melt/ice-height anticorrelation
`corr(m,y_ice)≈−0.95`, closure-independent. This is the expected behaviour of a *melting*
(not eroding) surface — the runaway was a false analogy to rock/sediment beds. (The
value was briefly corrupted to `+0.24` by a spectral-filter bug and restored on the GPU
path; provenance in `glaciers/REPORT_CANDIDATE3.md`.)

**4.2 The corrected scallop passes its go/no-go, locally.** A flat melting wall is linearly stable, but a finite-amplitude bump larger
than the thermal boundary layer drives lee-side separation → recirculation →
reattachment heat-flux enhancement — a stable, self-limiting scallop. The staged probe
confirms a coherent, closure-independent enhancement relative to its *own* conduction
baseline: `R_mean≈1.06–1.36`, lee peaks `R_max≈2–3.3`, scaling monotonically with the
mean current (`U^{0.8}–U^{1.5}`). The wavelength sweep shows an interior optimum at
`n_waves=12` (`λ≈2.094`, `λ/dx≈10.7`): a `+12.5%` peak in conduction-relative `R_mean`
and a peak `Nu/Nu_flat=0.9219`, falling off either side; the conduction-relative
enhancement is tight (`R_mean=1.1273±0.0044`). (Curl 1966)

**4.3 …but the mean is still below a flat wall, and we prove why.** The two
ratios look contradictory but are consistent **by construction**: `R_mean>1` measures
bump-melt against its own conduction baseline; `Nu/Nu_flat<1` measures the bump against a
*flat wall with flow on*. Flow lifts melt above conduction on the bump, yet the bump
still delivers less mean heat than a flat wall — both true at once.

The suppression obeys an **exact area-partition identity** (no moments, no truncation).
Partition the interface relative to the flat-wall conductance `m_flat=⟨m_n,flat⟩` into
*reattachment* (`m_n≥m_flat`, thin sublayer), *thickened* (`0<m_n<m_flat`), and
*reversed* (`m_n≤0`, no physical sublayer). Because the populations tile the interface,

> `Nu/Nu_flat = C_reatt + C_thick + C_rev`, `C_p = (1/N) Σ_p m_n/m_flat`,

holding to machine precision (`|C_sum − Nu/Nu_flat| ≲ 2×10⁻¹⁶`). With
`surplus = e_reatt > 0`, `deficit = −(e_thick+e_rev) > 0` (`e_p = C_p − f_p`), the
mechanism is one line: **`Nu<1` ⟺ `deficit > surplus`.** Full amplitude sweep
(Part-C config):

| `a/λ` | `Nu/Nu_flat` | `f_reatt` | `f_thick` | `f_rev` | surplus | deficit | def>surp | `(δ_flat/⟨δ_T⟩)(1+CV²)` |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.895 | 0.477 | 0.523 | 0.000 | 0.112 | 0.217 | **Yes** | 0.99 |
| 0.10 | 0.937 | 0.531 | 0.469 | 0.000 | 0.178 | 0.241 | **Yes** | 1.64 |
| 0.15 | 0.888 | 0.406 | 0.555 | 0.039 | 0.180 | 0.292 | **Yes** | 2.37 |
| 0.20 | 0.897 | 0.281 | 0.672 | 0.047 | 0.211 | 0.315 | **Yes** | 1.10 |
| 0.30 | 0.893 | 0.234 | 0.734 | 0.031 | 0.306 | 0.413 | **Yes** | 2.09 |
| 0.50 | 0.930 | 0.195 | 0.773 | 0.031 | 0.418 | 0.488 | **Yes** | 0.58 |

At every amplitude the reattachment surplus is outweighed by the thickened+reversed
deficit, so `Nu<1`; the deficit grows monotonically (`0.22→0.49`) as the cavity fills
with stagnant lee fluid (`f_thick: 0.47→0.77`). The thin reattachment patches carry a
fat `1/δ_T` tail (top conductance decile holds `0.17→0.48` of `⟨1/δ_T⟩`) but are an area
minority. The last column shows the **falsification** of the two-moment `(1+CV²)`
truncation: with `CV_δ~O(1)` it swings `0.58–2.37` and wrongly predicts `Nu>1` at 4 of 6
amplitudes. The suppression is therefore **distribution-dominated, not
moment-dominated** — a sharper physical statement (flow-separation structure of the
thermal boundary layer), not a weaker one. (`glaciers/scallop_g1_populations.py`,
`glaciers/scallop_sublayer_probe.py`.)

**4.4 The growth-carrying local flux, quantified.** The *mean* is bounded,
but melt **growth** is carried by the local lee flux, which the area-partition does not
fit. From the committed amplitude sweep, the local peak ratio `R_max(a/λ)`
rises **roughly linearly** with bump steepness — linear `R²=0.945`, free exponent `0.69`
(sub-quadratic), from `1.9` (`a/λ=0.05`) to `4.3` (`a/λ=0.30`); the `(a/λ)²` closure is
rejected for the *local* term too (quadratic `R²=0.40≪0.94`), and the origin-respecting
steepness law `R_max≈1+1.7·(2π a/λ)` holds to `R²=0.92` (seed-robust). The mean
conductance, by contrast, is amplitude-**flat** (`R_mean` saturates ~1.28; the
`Nu/Nu_flat` deficit is constant, `C≈1.11`), and the lee flux **reverses** (`R_min<0`) at
`a/λ≈0.11` — a separation onset the amplitude-flat mean hides. The honest closure for a
unified melt rate is therefore: mean amplitude-flat, growth carried by a bounded,
linear-in-steepness local lee flux — not a quadratic ansatz
(`validation/synthetic/g6_local_flux_law.py`, 6 tests).

---

## 5. The ceiling is boundary-condition robust

Every null in §3–§4 used a no-slip bed and a cold-Dirichlet ice wall. Two targeted gates
(P100, `n=128`, turbulent) show the ceiling is not an artefact of either:

- **Thermal BC (Dirichlet → finite-conductance Robin, `q=h(θ−θ_ice)`):** basal heat
  absorbed is unchanged flow-on vs flow-off to **`<0.01%`**.
- **Momentum BC (no-slip → Navier slip):** the near-bed tangential speed grows **~100×**
  (`u_tang` `0.027 → 2.68` from locked to free slip), yet `Nu/Nu_flat` stays `0.96–0.97`
  and **never exceeds 1** (it slightly falls); ridge melt holds to within 0.4 %, and the
  slip parameter `s=1` reproduces the no-slip trajectory bit-for-bit (including on the
  GPU CuPy path).

Removing the stagnant momentum layer entirely delivers **no** extra basal heat: the
limit is the **thermal conductive sublayer** set by `κ` and the cold wall, not the
momentum stagnation layer. (`glaciers/subglacial/slip_gate.py`,
`glaciers/subglacial/wall_flux.py`; `THEORY_CAVITY.md` §14.4.)

**Resolution and penalization robustness of the ceiling.** Because the cavity is
Brinkman-penalized, the suppression could in principle be a near-wall penalization-smearing
or under-resolution artefact. We test this directly with a wall-normal resolution ladder
`ny∈{48,64,96,128,192}` (sublayer spacing `dy=L_y/ny`, `L_y=2π`) at fixed `η=5×10⁻⁵`,
and a penalization sweep `η∈{2.5,5,10}×10⁻⁵` at `ny=128`, each grid point **ensembled over
three forcing seeds** so that turbulent-realization scatter is separated from genuine grid
dependence (single-mode bump `a/λ=0.20`, `n_x=128`, the same local-normal flux diagnostic
as §4.3, with a fixed spinup/measure window held constant across the sweep). Two facts emerge:

- **The sign is robust.** `Nu/Nu_flat<1` at *every* seed and *every* grid point (21
  configurations): the mean-melt suppression survives a 4× refinement of the wall-normal
  grid, a 4× change in penalization strength, and the forcing realization. Weakening the
  penalty shifts the ratio only mildly and monotonically (`η: 2.5→5→10×10⁻⁵` gives
  `Nu/Nu_flat: 0.783→0.807→0.836`, spread `0.053`), in the expected direction — a leakier
  wall passes slightly more flux — so the ceiling is not a penalization-strength artefact.
- **The magnitude is not grid-converged.** Across the resolution ladder `Nu/Nu_flat` varies
  non-monotonically — `0.786, 0.740, 0.989, 0.807, 0.869` for `ny=48…192` — with a per-grid
  seed scatter of only `±0.002–0.007`. Because that scatter is `~30×` smaller than the
  `0.18` spread across resolution, the variation is a real, reproducible grid dependence,
  not turbulent noise; `ny=96` robustly nearly erases the suppression (`0.989`). The precise
  suppression magnitude is therefore resolution-sensitive and we do **not** certify it as
  converged.

We accordingly claim only what is resolution-robust: the *sign* of the ceiling
(`Nu/Nu_flat<1`, melts less on the mean) and the *exact* area-partition identity (§4.3),
which holds to machine precision at every resolution. Certifying the magnitude would
require a body-fitted (penalization-free) near-wall grid or a uniform near-wall
refinement, which we leave open. (`glaciers/scallop_sublayer_convergence.py`,
`glaciers/subglacial/sublayer_convergence.json`.)

**Independent corroboration of the flux structure.** A 3-D active-buoyancy LES measures
**counter-gradient** subgrid heat flux (`C_G > −1`, growing with stratification): the
flux carries the global pressure field's memory, which a down-gradient closure cannot
represent (a GPU counter-gradient probe). This is consistent with the null map — the flow has real
non-local flux structure, but it does not raise the cold-wall mean melt.

**Interface coupling number — when the passive-BC limit is exact.** The gates
above show the *mean* ceiling is BC-robust empirically; a derived **interface coupling
number** says *why* the cold-Dirichlet wall is the correct BC at surge timescales. Writing
the ice flux as the linear response `q_ice'(s)=H(s)v'(s)`, the interface velocity is
`v'=q_water'/(ρ_iL+H(s))`, so the passive-BC limit is corrected by
`Λ(ω)=|H(iω)|/(ρ_iL)`: `Λ(0)=St≤0.06` (DC — ice participates at its latent-heat-limited
Stefan weight), `Λ→0` at high frequency (ice frozen, passive-adiabatic BC exact),
crossing over at the ice clock `τ_d=κ/V̄²~10³–10⁴ yr`. Because the surge band (0.02–2 yr)
is far faster than `τ_d`, in-band `Λ<5×10⁻⁵≪St`: the interface is a **passive BC to <1 %**
for the sliding/melt physics, and ice becomes a participating medium only for millennial
forcing (`validation/synthetic/interface_coupling_number.py`, 3 tests). (Stefan
1891; Crank 1984)

---

## 6. The deliverable is a regime map

The null is **regime-specific**, and stating the regime is the deliverable:

| Regime | Setting | Boundary conditions | Mean melt enhancement? |
|---|---|---|---|
| **Type I** | grounded, cold-walled, closed cavity | cold Dirichlet ice + bounded water | **No** — conduction-sublayer ceiling (`Nu/Nu_flat<1`), BC-robust |
| **Type II** | grounding line | flux / open boundary | **Maybe** — the closed-cavity assumption is relaxed |
| **Type III** | open shelf / ice-shelf cavity | warm inflow, free interface | **Yes** — different BC class, outside this study |

This explains why operational subglacial models carry no flow-enhanced-melt term in the
grounded regime, and makes explicit the boundary-condition class (Type II/III) a future
enhancement would require. It is a map of *where mechanisms operate and where the
boundaries lie*, not a universal melt multiplier.

---

## 7. Discussion and limitations

**What this is.** A pre-stated, closure-swept, reproducible demonstration that mean
basal melt in the cold-grounded cavity is conduction-sublayer-limited and
boundary-condition robust, with an exact mechanism (area-partition) and a regime map.

**What this is not.** 2-D for the candidate sweeps (3-D LES for the capstone and the
counter-gradient flux probe); penalized (Brinkman) rather than body-fitted geometry;
idealised forcing. Double diffusion (§3.2) is untested here: salt fingering is
intrinsically 3-D, so the 2-D runs neither confirm nor exclude double-diffusive
enhancement. The conductive-sublayer ceiling has been stress-tested for resolution and
penalization (§5): its *sign* is robust (`Nu/Nu_flat<1` across `ny=48–192`, 4× `η`, and
three seeds), but its *magnitude* is not grid-converged (non-monotone `0.74–0.99`), so we
report a sign-robust ceiling and an exact area-partition identity, **not** a certified
suppression magnitude. The scallop local enhancement is qualitative; its
amplitude coefficients are a companion study. Type II/III are not simulated.

**Open follow-ons.** A penalization-free (body-fitted) or uniformly wall-refined grid to
grid-converge the suppression *magnitude* (the sign is already resolution-robust, §5); 3-D
unforced double diffusion; a Type-II grounding-line flux-BC cavity to test the "maybe";
coupling the `Nu_S≫Nu_T` asymmetry into a melt-rate phenomenology.

---

## 8. Reproduce

```bash
pip install -r requirements.txt
python glaciers/run_subglacial.py --bed real --out-dir glaciers/figures # §2 cavity (Figs 31–38)
python glaciers/run_candidate1.py # §3.1 roughness plumes
python ocean/run_candidate2.py # §3.2 double diffusion (Fig 48)
python glaciers/run_candidate4.py # §3.3 hydraulic switching
python glaciers/scallop_battery.py # §4.1–4.2 scallop go/no-go (P100)
python glaciers/scallop_sweep.py # §4.2 wavelength sweep
python glaciers/scallop_g1_populations.py # §4.3 exact area-partition mechanism
python glaciers/scallop_sublayer_probe.py # §4.3 (1+CV²) falsification
python glaciers/scallop_sublayer_convergence.py # §5 resolution + penalization robustness (3 seeds)
# §5 BC robustness: glaciers/subglacial/slip_gate.py, wall_flux.py (P100)
python glaciers/theorem3_cg_gpu_probe.py # §5 counter-gradient C_G (Fig 52, P100)
pytest glaciers/tests ocean/tests -v
```

## Data and code availability

The double-diffusive transport result (Fig. 1), the cavity-field, transfer and counter-gradient figure set, and the area-partition tables regenerate from the scripts in the reproduce section. All data and analysis code are in the public repository at https://github.com/abdh0saifelden2-debug/K.

## References

Angot, P., Bruneau, C.-H., & Fabrie, P. (1999). A penalization method to take into account obstacles in incompressible viscous flows. *Numerische Mathematik* 81, 497–520. doi:10.1007/s002110050401

Brinkman, H. C. (1949). A calculation of the viscous force exerted by a flowing fluid on a dense swarm of particles. *Applied Scientific Research* A1, 27–34. doi:10.1007/BF02120313

Bushuk, M., Holland, D. M., Stanton, T. P., Stern, A., & Gray, C. (2019). Ice scallops: a laboratory investigation of the ice–water interface. *Journal of Fluid Mechanics* 873, 942–976. doi:10.1017/jfm.2019.398

Claudin, P., Durán, O., & Andreotti, B. (2017). Dissolution instability and roughening transition. *Journal of Fluid Mechanics* 832, R2. doi:10.1017/jfm.2017.711

Crank, J. (1984). *Free and Moving Boundary Problems.* Oxford University Press.

Curl, R. L. (1966). Scallops and flutes. *Transactions of the Cave Research Group of Great Britain* 7, 121–160.

Gilpin, R. R., Hirata, T., & Cheng, K. C. (1980). Wave formation and heat transfer at an ice-water interface in the presence of a turbulent flow. *Journal of Fluid Mechanics* 99, 619–640. doi:10.1017/S0022112080000791

Holland, D. M., & Jenkins, A. (1999). Modeling thermodynamic ice–ocean interactions at the base of an ice shelf. *Journal of Physical Oceanography* 29, 1787–1800. doi:10.1175/1520-0485(1999)029<1787:MTIOIA>2.0.CO;2

Kader, B. A., & Yaglom, A. M. (1972). Heat and mass transfer laws for fully turbulent wall flows. *International Journal of Heat and Mass Transfer* 15, 2329–2351. doi:10.1016/0017-9310(72)90131-7

Lythe, M. B., & Vaughan, D. G. (2001). BEDMAP: A new ice thickness and subglacial topographic model of Antarctica. *Journal of Geophysical Research: Solid Earth* 106, 11335–11351. doi:10.1029/2000JB900449

McPhee, M. G. (2008). *Air-Ice-Ocean Interaction: Turbulent Ocean Boundary Layer Exchange Processes.* Springer. doi:10.1007/978-0-387-78335-2

Mellor, G. L., McPhee, M. G., & Steele, M. (1986). Ice-seawater turbulent boundary layer interaction with melting or freezing. *Journal of Physical Oceanography* 16, 1829–1846. doi:10.1175/1520-0485(1986)016<1829:ISTBLI>2.0.CO;2

Nye, J. F. (1953). The flow law of ice from measurements in glacier tunnels, laboratory experiments and the Jungfraufirn borehole experiment. *Proceedings of the Royal Society A* 219, 477–489. doi:10.1098/rspa.1953.0161

Radko, T. (2013). *Double-Diffusive Convection.* Cambridge University Press.

Röthlisberger, H. (1972). Water pressure in intra- and subglacial channels. *Journal of Glaciology* 11, 177–203. doi:10.3189/S0022143000022188

Schoof, C. (2010). Ice-sheet acceleration driven by melt supply variability. *Nature* 468, 803–806. doi:10.1038/nature09618

Stefan, J. (1891). Über die Theorie der Eisbildung, insbesondere über die Eisbildung im Polarmeere. *Annalen der Physik* 278, 269–286.

Turner, J. S. (1973). *Buoyancy Effects in Fluids.* Cambridge University Press.

Werder, M. A., Hewitt, I. J., Schoof, C. G., & Flowers, G. E. (2013). Modeling channelized and distributed subglacial drainage in two dimensions. *Journal of Geophysical Research: Earth Surface* 118, 2140–2158. doi:10.1002/jgrf.20146
