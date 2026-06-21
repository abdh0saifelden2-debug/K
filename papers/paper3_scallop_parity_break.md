# A temperature-free morphological observable that distinguishes non-local from down-gradient heat transport at a melting interface

---

## Abstract

Ice and rock scallops migrate downstream as they evolve, a motion classically tied to the
phase lag between the wall heat flux and the interface topography: a melting surface
migrates downstream when the flux maximum lies downstream of the trough, i.e. when the
flux–topography phase lies in `(π/2, π)` (Ashton & Kennedy 1972; Gilpin et al. 1980;
Hanratty 1981). A local, memoryless, down-gradient ("K-theory") closure
carries no such lag. Freezing a sinusoidal interface `y(x)=ȳ+a sin(Kx)` and projecting the
time-averaged wall flux onto in-phase (amplitude rate `β`) and quadrature (migration
`ω_mig`) harmonics, every coefficient is solver-measured: the conduction amplitude rate is
wavelength-independent (`β/a ~ K^{+0.13}`), falsifying the `K²` and Mullins–Sekerka `|k|`
ansätze; the in-phase excess is negative at every amplitude and drive (smoothing-limited);
and the quadrature grows sub-kinematically
(`E_cos ∝ U^{≈0.5}`, friction-velocity controlled), giving a damped, downstream-migrating
mode `s = −β + iω_mig`. Swapping only the heat-transport operator at fixed momentum drives
the quadrature to machine zero (`−1.2×10⁻¹⁸` for a uniform eddy diffusivity) against
`−1.1×10⁻⁴` for resolved advection. We then construct a temperature-free observable
`I ≡ τ·c_mig/λ = |Im(s)|/(2π|Re(s)|)` in which conductivity, latent heat, and basal
temperature drop cancel, computable from interface morphology alone; the solver
pins `I = O(0.3–0.9)`, and a regime-matched reading of Bushuk et al. (2019) gives
`I_obs ≈ 0.05–0.4` with the predicted downstream sign. We give a field-test protocol and
a self-validated reanalysis harness (recovers `I` to `<3%`); the field test is proposed, not
performed here, as the raw `h(x,t)` arrays are not openly available.

---

## 1. Introduction

Scalloped erosion/melt patterns are ubiquitous on soluble and meltable surfaces — cave
walls, glacier soffits, ice-shelf bases, meteorites, and lab ice (Curl 1966;
Blumberg & Curl 1974; Ashton & Kennedy 1972; Bushuk et al. 2019). Their wavelength is
fluid-selected through a friction-Reynolds criterion `Re_*=u_*λ/ν≈2200`. A robust but
under-exploited observation is that the pattern **migrates downstream** as it evolves —
maximum ablation sits downstream of the trough, at flow reattachment (Blumberg &
Curl 1974; Gilpin, Hirata & Cheng 1980).

The migration direction and its phase-lag origin are established (Gilpin et al. 1980;
Hanratty 1981): downstream migration corresponds to a flux–topography phase in `(π/2,π)`,
and a local, memoryless, down-gradient closure carries no such phase. What is **new here**
is not that fact but three things built on it: a phase-separated harmonic decomposition
that returns the amplitude and migration rates as independent solver measurements; a
direct operator-swap experiment that quantifies how completely a down-gradient closure
discards the migration-carrying asymmetry; and a temperature-free morphological observable
that makes the phase lag measurable from field geometry alone, without knowing the basal
temperature drop or any material constant. The non-local, direction-selecting structure
the migration requires is exactly the structure (memory, backscatter) that a local,
down-gradient closure cannot represent.

**Contributions.**
1. A **phase-separated** wall-flux harmonic decomposition that returns the corrected mode
   `s=−β+iω_mig` as an *output*, not an ansatz, with the conduction amplitude rate
   measured wavelength-independent — falsifying the `K²`/`|k|` curvature-smoothing ansätze
   for the conduction baseline, generalised across amplitude and drive (§3–§4).
2. A **direct operator-swap measurement** quantifying how far down-gradient closures fall
   short of the known phase-lag requirement: swapping only the heat-transport operator
   collapses the migration signal to machine zero (§5).
3. A **temperature-free, constant-free morphological observable** `I=τ·c_mig/λ`, with an
   implemented, self-validated reanalysis harness and a regime-matched literature bound,
   that makes the phase lag measurable from field morphology alone (§6–§8).

---

## 2. Method — harmonic decomposition of the wall flux

Freeze `y(x)=ȳ+a sin(Kx)` (`K=2πn_w/L_x`), time-average the per-column melt flux
`m(x)=−κ ∂θ/∂y`, and project onto the two corrugation harmonics:

> `E_sin = 2⟨e(x) sin(Kx)⟩` (in-phase with the **shape** → amplitude change, `β`)
> `E_cos = 2⟨e(x) cos(Kx)⟩` (in **quadrature** → pattern **migration**, `ω_mig`)

`E_cos` is the Curl-1966 reattachment/lee signature: zero by symmetry unless the flow
carries a direction. Two decompositions are run: the conduction baseline `m_cond` (flow
off) for the `β ~ K^p` scaling, and the flow-induced excess `e = m_flow − m_cond` versus
drive `U`. Freezing the interface removes boundary motion so the flux is uncontaminated;
this is *why* the two coefficients separate cleanly, where a moving-boundary fit conflates
them and returns unphysical (both-negative) coefficients.

---

## 3. Findings

| quantity | measured | ansatz it kills |
|---|---|---|
| conduction `β/a ~ K^{+0.13}` | wavelength-**independent** | `K²` curvature and Mullins–Sekerka `|k|` **falsified** |
| in-phase flow excess `E_sin` | **negative at every driven `(a,U)`** | no autonomous `+α a^{1/2}` growth — **smoothing-limited** |
| quadrature `E_cos` | `≈0` at `U=0`, grows `∝ U^{0.5–0.8}` | migration is real and **friction-velocity** (`u_*~U^{1/2}`) controlled, not kinematic (`∝U`) |

**Corrected mode:** `s(K,U) = −β(K,U) + i·ω_mig(U)`, `Re(s)<0` always, `Im(s)∝U^{1/2}` —
a damped, downstream-migrating pattern, not a growth–saturation balance.

**Why this counts as a result, not a check.** The corrected mode *fell out of* the
decomposition; every coefficient is solver-measured; physical units are pinned
independently by the Curl selection law without choosing `λ` by hand; and the final
ratio is constant-free and falsifiable.

---

## 4. Generalisation across amplitude and drive

**4.1 Amplitude (`a₀/λ ∈ [0.05, 0.40]`).** The two structural verdicts are not
amplitude artefacts: the driven in-phase excess is **negative at every amplitude**
(smoothing throughout, magnitude growing monotonically), and the conduction exponent
stays `p∈[−0.27,+0.66]`, far below the `K²` (`+2`) and `|k|` (`+1`) ansätze at every
amplitude. *Honest caveat:* strict `K`-independence (`|p|<0.6`) and amplitude-flat `β/a`
hold only in the signal-rich regime `a₀/λ≳0.2`; at shallow amplitude the
single-wavenumber conduction signal is at the noise floor (the K²/|k| falsification is
robust regardless). (`amplitude_generalization_scan`.)

**4.2 Drive (`U` to 6, beyond the swept `[1.5,3.0]`).** The genuine falsification risk
is that strong-drive lee separation adds an *in-phase* component (an emergent growth
term the weak-drive sweep would miss). It does not: `E_sin` is **negative at every `U`**
out to `U=6`, and the migration follows a **sub-kinematic** `E_cos ∝ U^{+0.48}` (≈½),
far below the kinematic-advection alternative `U¹`. *Honest caveat:* migration is
non-monotone — it peaks near `U≈4.5` and rolls over by `U=6` (the lee structure reaches
a limiting form), which *reinforces* the sub-kinematic reading. (`drive_window_scan`.)

| `U` | in-phase `E_sin` | quadrature `E_cos` |
|---|---|---|
| 0.0 | `−5.17×10⁻⁶` (smoothing) | `+4.9×10⁻⁸` (≈0, parity control) |
| 1.5 | `−6.23×10⁻⁵` | `−6.59×10⁻⁵` |
| 3.0 | `−9.19×10⁻⁵` | `−1.14×10⁻⁴` |
| 4.5 | `−8.86×10⁻⁵` | `−1.38×10⁻⁴` |
| 6.0 | `−4.81×10⁻⁵` | `−1.19×10⁻⁴` |

---

## 5. A direct operator-swap measurement of the closure shortfall

A local, memoryless, down-gradient eddy diffusivity acting on a sinusoid is symmetric
under `x→−x`: it carries no flow direction and can only return an in-phase (real,
damping) response — `E_cos≡0`. This is the closure-space statement of the classical
phase-lag requirement (Gilpin et al. 1980; Hanratty 1981). The resolved advective flux
measured `E_cos≠0`, so migration requires the flow-direction-selecting, non-local
reattachment asymmetry that K-theory discards. The contribution here is to make the
shortfall **quantitative**: advancing the **same** momentum field at the **same** drive
(`U=3`, same interface) but transporting heat with the K-theory flux instead of resolved
advection measures exactly how far each closure falls below the resolved migration:

| heat transport (same `U=3`, same interface) | `E_cos` | reading |
|---|---|---|
| resolved advective `−u·∇θ` | `−1.14×10⁻⁴` | migrates |
| **uniform** eddy diffusivity `∇·(K∇θ)` | `−1.2×10⁻¹⁸` | machine zero (`~10⁻¹⁴` of resolved, `std=0`) |
| **flow-aware** Smagorinsky `K_eddy(|S|)` | `+2.0×10⁻⁶` | `~57×` below resolved |

Both closures still **smooth** (`E_sin≠0`): K-theory can damp but cannot migrate.
GPU/CuPy controls (Tesla P100) confirm (a) parity — `E_cos→+2.1×10⁻⁶` at `U=0` vs
`−8.4×10⁻⁵` at `U=3` (~40× smaller); and (b) the high-`Re_*` migration exponent tightens
to **`0.52`** (`nx=160`, four seeds, scatter ~1–2 %), on the friction-velocity `½`.
(`scallop_ktheory_control.py`.)

This is the morphological analogue of the memory and backscatter signatures that a
down-gradient closure omits in *flux* space: the closure's blindness does not
merely lose information in the model — it leaves a fingerprint carved into the ice.

---

## 6. A constant-free, ΔT-free field test

A Stefan-number-free dimensional bridge (`ρL v = q`, `q=−k_th(ΔT/L₀)∂θ/∂y`, length
anchored by the corrugation wavelength) gives `τ ~ λ²/ΔT` (amplitude e-fold) and
`c_mig ~ ΔT/λ` (migration speed). In the product the constants `k_th`, `ρ_iL` **and `ΔT`
cancel exactly**:

> `I ≡ τ·c_mig/λ = Im(s)/(2π|Re(s)|)` — the dimensionless ratio of migration rate to
> amplitude rate, basal-`ΔT`-free.

The solver pins `I=O(0.3–0.9)` (`0.33, 0.68, 0.88` at `n_w=8,12,16` — wavelength-dependent
by a factor `≈2.7`, an `O(1)` morphological number rather than a universal constant).

**Decoy warning (a methodological result).** `τ ∝ λ²/ΔT` *looks* like classic
curvature/Mullins–Sekerka smoothing, but the `λ²` comes entirely from the conduction
length↔time anchoring `L₀²/κ`, **not** from a curvature operator — `β/a` is
`K`-independent (§3). Fitting `τ∝λ²` in the field would *wrongly confirm* the very
curvature ansatz the harmonic probe falsified; the only discriminator is the
`K`-exponent of `β/a` (`≈0` here vs `+2` for curvature), which the dimensional scaling
hides. Anyone analysing field scallops should measure the `K`-exponent, not fit `τ∝λ²`.

**Field-test protocol.** On one scallop train measure `λ` (crest spacing), `c_mig`
(downstream crest displacement between two visits), and `τ` (amplitude e-fold). Then:

| outcome | reading |
|---|---|
| `I_obs ∈ [0.2,1.0]`, `c_mig` downstream | consistent with `s=−β+iω_mig` |
| `c_mig ≈ 0` (`Im(s)=0`) | parity-symmetric closure — K-theory **not** falsified on the ice (contradicts the solver) |
| `c_mig` upstream | new physics (e.g. depositional, not melt-limited) |
| `I_obs ≫ 1` or `≪ 0.1` | `λ ↔ u_*` anchoring mismatch |

Because `I` is `ΔT`-free, a failure cannot be excused by an unknown basal `ΔT`; measuring
`c_mig + λ` alone pins `ΔT`, so `τ` becomes a prediction and the system is
over-determined.

---

## 7. Literature grounding

- **`λ` (wavelength) — solid.** `Re_*=u_*λ/ν` selection is mainstream: Blumberg & Curl
  (1974) `≈2200`, Thomas (1979) `≈1000`, Hsu–Locher–Kennedy (1979) `λ=3180 ν/u_*`,
  Thorsness–Hanratty (1979)/Hanratty (1981) band `3100–6300 ν/u_*`.
- **`c_mig` direction & mechanism — solid.** Downstream migration with maximum ablation
  downstream of the trough at reattachment: Curl (1966), Ashton & Kennedy (1972), Hsu et
  al. (1979), Gilpin et al. (1980). Gilpin et al.'s linear stability predicts downstream
  migration when the heat-flux/topography phase shift lies in `(π/2, π)` — the same
  statement as `Im(s)≠0`.
- **`c_mig` magnitude and `τ` — sparse.** Cave-scallop surveys report `λ` only; migration
  speed and amplitude decay time are essentially never tabulated together for one train.

---

## 8. The one open gate — and a regime-matched bound

**Sign caveat.** Real ice ripples *grow* (`Re(s)>0`, the Ashton–Kennedy/Gilpin/Hanratty
moving-boundary instability), whereas the frozen-interface excess here is decay-only
(`Re(s)<0`) by construction. The migration `Im(s)` is the robustly shared,
K-theory-falsifying piece; matching the *growing* branch needs an interface-growth
mechanism (a Röthlisberger opening) this smoothing-only solver does not contain.

**The migration transfers to a moving boundary.** We close the part of this gate the
solver *can* reach. Seeding a single mode on the interface and advancing it under the
Stefan melt feedback (`ρL v=q`), the co-evolving interface tracks a **coherent
damped-migrating eigenmode**: `log|Z|` and `arg(Z)` of the fundamental shape mode are both
linear in time (`R²≥0.91` for amplitude and phase, two seeds × two Stefan numbers). It
**decays** (`Re(s)<0` at every `St` — the decay-only branch is *physical*, not an artefact
of freezing) and **migrates downstream**, with `Im(s)` tracking the frozen-probe quadrature
`E_cos` (predicted `Im=E_cos/(aSt)≈−0.43` vs measured `−0.40` at `St=10⁻³`, ~5–10%). The
observable `I=|Im|/(2π|Re|)≈0.15–0.17` is **`St`-invariant** (the rate scales as `1/St`, so
`Re·St` and `Im·St` are constant to ~3%), confirming `I` is a well-defined,
normalization-free property of the mode. The migration — the closure-distinguishing
quadrature — therefore **survives interface motion**. The total moving-interface decay runs
`~1.75×` faster than the flow-off conduction estimate (turbulence adds effective smoothing),
so this motion-consistent `I≈0.16` sits *below* the frozen flow-off band and *closer to* the
Bushuk field bound below. The *growing* branch (`Re(s)>0`) is still out of reach here.
(`scallop_moving_boundary_check.py`, `subglacial/moving_boundary_check.json`.)

**A regime-matched real-data bound.** Bushuk et al. (2019, *JFM* 873)
track an evolving ice–water interface `h(x,t)` at sub-mm/15 Hz through
transition→equilibrium→**adjustment**. Their adjustment regime (after the drive was cut
`U=1.00→0.16 m s⁻¹`) **damps and migrates downstream simultaneously** — the exact
analogue of the frozen probe (`s=−β+iω_mig`, `Re(s)<0`, `Im(s)≠0`), so the sign caveat
does not apply. Reading their measured kinematics (`c_mig=0.11 mm min⁻¹=1.83×10⁻⁶ m
s⁻¹`, observed `λ≈13 cm`, amplitude e-fold `τ≈1–3 h` by eye):

> `I_obs ≈ 0.05–0.15` (point `≈0.10`) at the observed `λ`; `O(0.05–0.4)` overall;
> **downstream sign**. Measured the *same way* as the field data — a moving boundary with
> total (turbulent) smoothing, via the identical `Re(s)=d/dt ln a_k`, `Im(s)=dφ_k/dt`
> protocol — the solver gives the motion-consistent `I≈0.16` (above), a factor `≈1.6` above
> the point estimate, versus the factor `≈3.3` against the frozen flow-off band `[0.33,0.88]`.
> The motion-consistent comparison thus **reduces the tension to mild** — `I_obs` is
> `O(0.1–1)`, neither `≈0` nor `≫1`, with the predicted downstream sign — a consistency
> check, not a falsification.

**The pin.** The one missing input is Bushuk's underlying `h(x,t)` arrays, available from
the authors on request (`mitchell.bushuk@noaa.gov`); the published JFM supplement is a
parameter table only, not the interface time series, and no open data deposit exists. The §6 ratio is
computed from morphology alone, and the reanalysis harness is **already implemented and
self-validated**: `harmonic_mode_rate(x,t,H)` rFFTs each frame, tracks the dominant mode
`a_k(t)e^{iφ_k(t)}`, fits `Re(s)=d/dt ln a_k`, `Im(s)=dφ_k/dt`, and returns `I` plus the
downstream sign; on a synthetic damped, downstream-migrating train with **known**
`(Re(s), c_mig, λ)` it recovers all three to `<2%` and `I` to `<3%`. It ingests Bushuk's
arrays directly — no further code, just the data. (`scallop_field_test.py`.)

---

## 9. Discussion and scope

This is a solver-verified, quantitative characterization of the closure: the corrected
mode, the operator-swap shortfall, and the temperature-free ratio are all self-consistent
and GPU-reproducible. It is **not yet validated against field scallops** — §8 is precisely
why that test is cheap (no `ΔT` needed). The banked claims are the qualitative three —
`Im(s)→0` at `U=0`, strictly sub-linear migration, exponent `→½` at high `Re_*` (central
value `0.52`) — robust to the ~±0.1 scatter from spin-up/seed/`U`-grid. The solver-side
moving-boundary measurement (§8) is now done — the migration transfers to an actually-moving
interface and `I` is motion-invariant — so what a field dataset would still pin is the
migration exponent's precise value and the field `I_obs` from real `h(x,t)`.

---

## 10. Reproduce

```bash
python glaciers/scallop_amplitude_harmonics.py # §3–§4; writes figures/56_*.json
python glaciers/scallop_ktheory_control.py # §5 keystone control (CPU; xp=cupy for GPU)
python glaciers/scallop_field_test.py # §6/§8; figures/58_*.json (harness self-check + Bushuk bound)
python glaciers/scallop_moving_boundary_check.py # §8 moving-boundary migration transfer + I(St-invariance)
pytest glaciers/tests/test_scallop_amplitude_harmonics.py \
       glaciers/tests/test_scallop_ktheory_control.py \
       glaciers/tests/test_scallop_field_test.py
```

## Data and code availability

All scallop-migration and harmonic-decomposition results regenerate from the scripts in the reproduce section; the field-test harness and its synthetic validation data are in the public repository at https://github.com/abdh0saifelden2-debug/K.

## References

Ashton, G. D., & Kennedy, J. F. (1972). Ripples on underside of river ice covers. *Journal of the Hydraulics Division (ASCE)* 98, 1603–1624. doi:10.1061/JYCEAJ.0003407

Blumberg, P. N., & Curl, R. L. (1974). Experimental and theoretical studies of dissolution roughness. *Journal of Fluid Mechanics* 65, 735–751. doi:10.1017/S0022112074001625

Bushuk, M., Holland, D. M., Stanton, T. P., Stern, A., & Gray, C. (2019). Ice scallops: a laboratory investigation of the ice–water interface. *Journal of Fluid Mechanics* 873, 942–976. doi:10.1017/jfm.2019.398

Curl, R. L. (1966). Scallops and flutes. *Transactions of the Cave Research Group of Great Britain* 7, 121–160.

Gilpin, R. R., Hirata, T., & Cheng, K. C. (1980). Wave formation and heat transfer at an ice-water interface in the presence of a turbulent flow. *Journal of Fluid Mechanics* 99, 619–640. doi:10.1017/S0022112080000791

Hanratty, T. J. (1981). Stability of surfaces that are dissolving or being formed by convective diffusion. *Annual Review of Fluid Mechanics* 13, 231–252. doi:10.1146/annurev.fl.13.010181.001311

Hsu, K.-H., Locher, F. A., & Kennedy, J. F. (1979). Forced-convection heat transfer from irregular melting wavy boundaries. *Journal of Heat Transfer* 101, 598–602. doi:10.1115/1.3451043

Mullins, W. W., & Sekerka, R. F. (1964). Stability of a planar interface during solidification of a dilute binary alloy. *Journal of Applied Physics* 35, 444–451. doi:10.1063/1.1713333

Thomas, R. M. (1979). Size of scallops and ripples formed by flowing water. *Nature* 277, 281–283. doi:10.1038/277281a0

Thorsness, C. B., & Hanratty, T. J. (1979). Mass transfer between a flowing fluid and a solid wavy surface. *AIChE Journal* 25, 686–697. doi:10.1002/aic.690250415
