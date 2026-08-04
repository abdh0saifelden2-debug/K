# Falsifiable forecasts for subglacial hydrology: an exact Mori–Zwanzig memory kernel, a motivated thermal-lag falsification, and an honest co-temporal null

> **Note (provenance).** This is the combined working manuscript. Per the referee's
> primary recommendation (one thesis per paper), it has been **split for submission**
> into two single-thesis papers, both built from the content below:
> [`paper4a_hydraulic_memory_kernel.md`](paper4a_hydraulic_memory_kernel.md) (the
> hydraulic Mori–Zwanzig kernel + thermal-lag falsification + matched-lag null + the
> RTN/variable-`φ` screen) and
> [`paper4b_sliding_law_field_test.md`](paper4b_sliding_law_field_test.md) (the
> field-measurable `s_N(N)` sliding law + `N_c` inversion + tidal-admittance
> early-warning). Submit 4a/4b; this omnibus is retained for provenance only.

---

## Abstract

Subglacial hydrology controls ice-stream sliding through the effective pressure
`N=p_i−p_w` (Schoof 2010; Röthlisberger 1972), and the central modelling difficulty is
that the cavity↔channel storage system has *memory*. We make three honest claims on open
Antarctic data, reporting a structural certification, a motivated falsification, and a
null with equal weight. **The central positive result is structural:** the lumped,
linearised cavity↔channel hydraulic subsystem's Green's function is an *exact*
Mori–Zwanzig memory kernel — projecting out the channel variable yields a generalized
Langevin equation whose kernel equals the eliminated subsystem's own Green's function
(verified to `7×10⁻¹⁸`), with the projection exact to `5×10⁻¹⁶`, the reduced model
reproducing the full trajectory to `9×10⁻⁸`, and the Markovian (adiabatic-elimination)
limit recovered as the channel time vanishes. This certifies *why* surge response carries
memory and how it degenerates to a local closure; it is a structural result, not a
field-validated lag value. **A motivated null:** a literal thermal sliding-lag kernel
`τ_ice=H²/κ_ice` — surge lag set by the *full-thickness* ice thermal-diffusion time — is
implausible a priori (the thermal skin depth at surge periods is metres, and a drainage
impulse gives a peakless `t^{−1/2}` response), and is duly falsified on real thickness at
the 131 Siegfried & Fricker (2018) active lakes by ~5 orders of magnitude (median
`τ_ice≈1.5×10⁵ yr` vs observed 0.02–2 yr; robust on the independent Arthern et al. (2015) thickness inversion,
`r=0.970`). We report it as a motivated null that relocates the memory into the hydraulic
subsystem above, not as a headline empirical result. **An honest co-temporal null:** a
matched lake-drainage→velocity test (CryoSat-2 drainage dates + ITS_LIVE velocities)
returns a null (peak `+0.56σ`, 0/5 significant), limited not by record precision (~1–3 %)
but by temporal aliasing and an amplitude bound, with unblock routes via sub-annual GPS or
fast-outlet trunk velocity. We also expose two field-scale predictors, positioned honestly
against their established counterparts. A dimensionless intrusion screen
`RTN=ρ_w g d_base/(φ ρ_i g H)` is, by construction, the flotation/effective-pressure ratio
(`RTN>1 ⟺` ocean head exceeds the `φ`-fraction of overburden, i.e. `N→0`), so its
concentration within ~6 km of the grounding line on Bedmap2 and BedMachine restates known
flotation geometry rather than adding skill over a thickness-above-flotation predictor; we
present it as a continental screen, not a verified forecast, paired with a residence
number `Ro`. And a regularized-Coulomb sliding law gives a closed-form master curve
`|s_N|(N)=m/(1−(N_c/N)^m)` that *derives* the near-flotation weakening imposed by hand in
Joughin, Smith & Schoof (2019), with an `N_c` inversion and a velocity/tidal-admittance
early-warning that **operationalizes**, for ungrounding, the critical-slowing-down
indicators already established for marine-ice-sheet tipping (Scheffer et al. 2009; Dakos
et al. 2008; Rosier et al. 2021). Synthetic plant-and-recover harnesses calibrate every
estimator. The contributions are the exact MZ certification, the falsifiable-forecast
discipline, and a field-measurable sliding law — not a verified intrusion forecast or a new
early-warning.

---

## 1. Introduction

Subglacial hydrology controls ice-stream sliding through the effective pressure
`N=p_i−p_w`, the difference between ice overburden and basal water pressure (Cuffey &
Paterson 2010; Schoof 2010; Röthlisberger 1972). Near the grounding line `N→0` as the ice
approaches flotation (Schoof 2007; Leguy et al. 2014), and seawater intrusion beneath
grounded ice is an active question (Robel et al. 2022). The
cavity↔channel drainage system carries *memory*, so its surge response is non-local
(Werder et al. 2013; Hewitt 2013). Because these are hard to observe at continental scale,
testable *falsifiable* statements that key on open geometry and velocity datasets are
valuable even when they return null or negative results.

We expose a structural prediction (a non-local hydraulic memory kernel) and two
field-scale predictors (an intrusion screen and a sliding law) to open data (Bedmap2,
BedMachine, the Siegfried & Fricker active-lake inventory, ITS_LIVE velocities, CryoSat-2
elevation), reporting each against its established counterpart: synthetic plant-and-recover
harnesses prove the estimators while the real-data runs expose the physics to
falsification.

**Contributions.**
1. The **exact Mori–Zwanzig structure** of the cavity↔channel hydraulic memory kernel —
   a structural certification (§5).
2. A **motivated falsification** of the literal `H²/κ` thermal sliding-lag kernel on 131
   real lakes, relocating the memory into hydrology (§4).
3. An **honest matched-lag null** with a precise account of *why* it is null (aliasing +
   amplitude bound, not noise) and the two routes to a verified lag (§6).
4. A **field-measurable sliding law** — the closed-form `s_N(N)` master curve, an `N_c`
   inversion, and a velocity/tidal-admittance early-warning that operationalizes
   established critical-slowing-down theory for ungrounding (§7).
5. A nondimensional **intrusion screen** `RTN`, shown to be the flotation/effective-
   pressure ratio, applied at continental scale and paired with a residence number `Ro` —
   presented as a screen, not a verified forecast (§3).
6. **Calibrated estimators** via synthetic plant-and-recover (§8).

---

## 2. Data and methods

| Dataset | What | Access |
|---|---|---|
| BAS **Bedmap2** | thickness, bed, surface, grounded/shelf mask (1 km) | open, no auth |
| **BedMachine** (Antarctica) | independent thickness/bed | NSIDC/USAP-DC |
| **Siegfried & Fricker (2018)** | 131 active subglacial-lake outlines | open (GitHub mirror) |
| **ITS_LIVE** | surface-speed datacubes (image-pair `v`) | open (anonymous S3) |
| **CryoSat-2** in-lake elevation | drainage *dates* | open (Siegfried 2021-GRL mirror) |
| **USAP-DC 601439** | ICESat lake volume-change (drainage volumes) | open |

Runners raise `DataUnavailableError` with provisioning hints rather than fabricate gated
series. Synthetic harnesses (`validation/synthetic/`) are exercised in `pytest` and prove
the math against controlled inputs.

---

## 3. The Regime Transition Number

**3.1 Definition.** `RTN = (p_ocean − p_atm)/p_w` on grounded ice (`icemask==0`):
overburden `p_i=ρ_i g H`, ocean head at the bed `p_ocean=ρ_w g·d_base`
(`d_base=max(0,−bed)`), subglacial water `p_w=φ p_i` (so `N_eff=(1−φ)p_i`), gauge-corrected
so the atmosphere cancels (`p_atm=0`). **`RTN` is, by construction, the flotation ratio.**
With `p_atm=0`, `RTN = ρ_w d_base/(φ ρ_i H)`, so `RTN>1 ⟺ ρ_w d_base/(ρ_i H) > φ` — the ice
is within a factor `φ` of flotation (`ρ_w d_base ≥ ρ_i H`), equivalently `N→0` (Schoof 2007;
Leguy et al. 2014; Cuffey & Paterson 2010). `RTN` is therefore a nondimensional form of the
effective-pressure/flotation criterion, not an independent predictor. The checkable *screen*
is **directional**: because `N→0` at the grounding line, `RTN>1` necessarily concentrates
there; the question (§3.4) is whether it adds skill beyond that geometry.

**3.2 Result on Bedmap2** (1 km decimated to 3 km, 1,329,743 grounded cells, `φ=0.9`):

| dist-to-GL [km] | 0–5 | 5–10 | 10–25 | 25–50 | 50–100 | 100–250 | >250 |
|---|---|---|---|---|---|---|---|
| `RTN>1` fraction | 15.1% | 6.0% | 3.6% | 2.1% | 0.8% | 0.0% | 0.0% |
| n cells | 50532 | 52094 | 87318 | 99425 | 143492 | 285743 | 611139 |

- Median distance-to-GL: **`RTN>1` cells = 6 km** vs **`RTN<1` cells = 222 km**.
- Robust across `φ`: overall `RTN>1` fraction `2.5% / 1.3% / 0.6%` for
  `φ=0.8/0.9/0.95` — only the magnitude scales; the monotone grounding-line decay holds.
- *Gauge fix.* The corrected gauge (`p_atm=0`) sharpens, not weakens, the signal: it
  flips 2,919 grounded cells into `RTN>1`, clustered at median 4.2 km from the
  grounding line (the spurious offset was largest where `d_base→0`, i.e. at the
  grounding line). The earlier `−p_atm` convention *understated* the concentration.

**3.3 Independent cross-check.** The directional concentration survives on **BedMachine
v4** (NSIDC-0756; an independent thickness/bed inversion to Bedmap2), decimated to 2 km
from its 500 m native posting, with the same monotone grounding-line decay: overall
`RTN>1` fractions `2.5% / 1.4% / 0.8%` for `φ=0.8/0.9/0.95`, and a median distance-to-GL
of **6 km** for `RTN>1` cells (at `φ=0.9`) versus ≈200 km for the rest
(`validation/external/rtn_corollaries_bedmachine.py`,
`validation/reports/rtn_bedmachine.json`, Figs. 1–2).

![Regime Transition Number on real Bedmap2 geometry (1.33 M grounded cells). The fraction of cells with `RTN>1` (ocean intrusion favoured) decays monotonically from 15.1% within 5 km of the grounding line to 0% beyond 100 km; the median distance to the grounding line is 6 km for `RTN>1` cells versus 222 km for the rest.](figures/rtn_bedmap2.png)

![Independent BedMachine Antarctica v4 cross-check at 2 km resolution. The directional result — `RTN>1` concentrated near the grounding line — reproduces the Bedmap2 finding on an independent thickness inversion (median distance-to-grounding-line 6 km for `RTN>1` cells versus ≈200 km for the rest).](figures/rtn_bedmachine.png)

**3.4 Scope, and the baseline question.** This is **not** a precision/recall score: no
gridded intrusion survey exists, and 1 km cannot resolve the ~1–10 m Röthlisberger channel
(channel size is absorbed into `φ`, not resolved). More fundamentally, because `RTN` is a
monotone function of the flotation ratio `ρ_w d_base/(ρ_i H)` (§3.1), its grounding-line
concentration largely *restates* known flotation geometry; to be informative as a screen it
must be shown to add skill over the trivial predictors it is built from — thickness-above-
flotation, bed-depth-below-sea-level, or distance-to-grounding-line. A direct baseline test
on real Bedmap2 geometry confirms it does not: over 478,767 grounded cells, `{RTN>1}` is
*identical* to the thickness-above-flotation threshold `{H_af/H<1−φ}` (agreement
**100.0000 %**, 0 disagreeing cells, for `φ=0.8/0.9/0.95`), and `RTN` is a strictly
monotone function of the flotation fraction (`Spearman(RTN, H_af/H)=−1.000`). So `RTN`
adds **no skill** over thickness-above-flotation — it *is* that standard quantity,
nondimensionalised by `φ`. It is, however, not reducible to distance-to-grounding-line
alone (`Spearman=−0.23`). We therefore present `RTN` as a compact continental-scale screen
and a nondimensional packaging of the flotation/effective-pressure criterion, **not** a
verified forecast; a future gridded intrusion survey (IceBridge successors, NISAR) would be
needed to test any predictor against actual intrusion
(`validation/external/rtn_baseline_skill.py`).

**3.4b The variable-`φ` escape — the only way `RTN` can add skill, and its open data
step.** The zero-skill result above is specific to a spatially-uniform `φ`: with one
continent-wide `φ`, `RTN=(1−H_af/H)/φ` is a monotone function of the flotation fraction, so
the identity is algebraically forced. `RTN`'s only possible source of information beyond
flotation is therefore `φ` *itself* — a spatially-varying, data-derived basal-connectivity
fraction `φ(x,y)` makes `RTN=ρ_w d_base/(φ(x,y) ρ_i H)` no longer a function of the flotation
fraction alone. A synthetic plant-and-recover (`validation/synthetic/rtn_variable_phi_skill.py`,
5 tests) establishes the mechanism and its size: against a planted truth in which intrusion
depends on *both* flotation proximity and connectivity, a connectivity-informed `φ(x,y)` adds
genuine skill over thickness-above-flotation (near-flotation-band `F1` 0.78→0.89; `ROC-AUC`
0.995→0.999), a connectivity-blind random `φ` of the same distribution adds none, and the gain
rises monotonically with `φ`'s informativeness — so the skill is connectivity-sourced, not a
variance artifact. On real Bedmap2 a calibration-free `φ(x,y)` from subglacial
hydraulic-potential routing (Shreve 1972: priority-flood fill + D8 flow accumulation)
reproduces the constant-`φ` anchor exactly and then reorders the intrusion classification on
~10³ grounded cells, concentrated at the grounding line (median 10 km), with the routed `φ`
nearly rank-independent of the flotation fraction in the near-flotation band
(`Spearman(RTN, H_af/H)=−0.99` vs `−1.00` at constant `φ`;
`validation/external/rtn_variable_phi_real.py`). The reordering is therefore real and
non-trivial — but its *correctness* is unvalidated, because the routed `φ` is a model field
derived from the same geometry that sets the flotation fraction. The independent-observable
step is now **executed on real data**: ICECAP radar bed-echo **specularity content** (Young
et al. 2020; 432 East Antarctic transects, 3.2×10⁶ points, 80–170°E incl. Totten/Aurora/Byrd)
binned to the same decimated Bedmap2 grid gives a measured `φ` driver on 14,126 covered
grounded cells that is nearly rank-independent of the flotation fraction in-band
(`Spearman=±0.17`). Under the pressure reading of specularity (high specularity = distributed
water at high pressure; Schroeder et al. 2013; Dow et al. 2020) the intrusion class changes
on **162** covered cells (all newly flagged, median 10 km from the grounding line); under the
opposite connectivity reading, 89 cells (all dropped, median 5 km) — the two
physically-arguable spec→`p_w` sign conventions move the near-grounding-line classification
in *opposite directions*. On the same cells the Shreve-routed connectivity is **uncorrelated
with the measured specularity** (`Spearman≈0.04`), so the routed-`φ` reordering should not be
read as a validated basal-water proxy (`validation/external/rtn_specularity_real.py`). The
sharpened open steps are the spec→`p_w` sign (co-located borehole pressure) and a gridded
intrusion survey to score against. We thus state variable-`φ` `RTN` as a falsifiable screen
whose driving observable is now measured, not modelled — still not a verified forecast.

**3.5 The intrusion residence number `Ro` — thinning-paced vs hydraulic-limited
.** The RTN=1 surface is a *threshold*; the *pace* at which the intrusion
front advances is a separate, testable quantity. The level-set advance speed is
`v_kin=(dH/dt)/|∇m|=A·dH/dt` (geometric amplification `A=1/|∇m|`); the newly
near-flotation bed must then re-pressurise to ocean head over a hydraulic residence time
`τ_hyd` (the §5 cavity↔channel band, 0.01–2 yr), so the front moves at
`v_obs=min(v_kin, ℓ/τ_hyd)` and the **residence number** `Ro=v_kin·τ_hyd/ℓ` has a regime
boundary at the critical residence `τ_crit=ℓ/v_kin` (`Ro=1` exactly at `τ_hyd=τ_crit`).
For the runaway-tail cells (`A=0.70 km/m`, `dH/dt=1.5 m/yr`): `v_kin=1.05 km/yr`,
`τ_crit≈1.9 yr`, so ~98 % of the residence band lies below `τ_crit` ⇒ intrusion is
predicted **thinning-paced (`Ro≈1`)**, turning hydraulic-limited only at the slow (~2 yr)
end. The implied hydraulic diffusivity `D_hyd=ℓ²/τ_hyd≈0.06–12.7 m²/s` spans the
distributed→channelized range. Falsifier: a DInSAR `v_obs` over the runaway cells with
`Ro≫1` would place `τ_hyd>τ_crit` (hydraulic limitation); `Ro≈1` confirms thinning-pacing
(`validation/synthetic/intrusion_residence_number.py`, 7/7 tests). (Werder et al.
2013; Hewitt 2013)

---

## 4. The thermal sliding-lag kernel, falsified

**4.1 The hypothesis (a motivated null).** No established sliding theory predicts that
surge lag equals the *full-thickness* ice thermal-diffusion time; we state
`τ_ice = H²/κ_ice` (`κ_ice=1.09×10⁻⁶ m² s⁻¹`) as an explicit, falsifiable straw hypothesis
precisely so the data can reject it.

**4.2 The test.** On real Bedmap2 thickness at the 131 Siegfried & Fricker (2018)
catalogued active lakes (130 with valid thickness; median `H=2282 m`, range 637–3905 m):

- `τ_ice = H²/κ_ice`: **median ≈ 1.5×10⁵ yr** (p5 ≈ 2×10⁴; p95 ≈ 3×10⁵ yr).
- Observed post-drainage surge lags: **0.02–2 yr** (Stearns et al. 2008;
  Siegfried et al. 2016).
- ⇒ the literal kernel is **~8×10⁴× too slow** at the median lake; the `log₁₀ τ_ice`
  histogram (10⁴–10⁶ yr) is fully disjoint from the observed band.

**4.3 Robust to the thickness dataset.** Recomputing on an independent ice-thickness
dataset (USAP-DC 601470, Stubblefield et al. 2021; ice thickness after Arthern et al.
2015): median thickness 2356 m, `τ_ice` median **≈1.55×10⁵ yr**; the 601470-vs-Bedmap2
thickness at the same centroids agrees at **`r=0.970`**, so the
falsification is not an artefact of one sampling.

**4.4 Why the falsification is diagnostic.** Two facts exclude ice thermal diffusion as
the lag-setting mechanism: (i) the thermal skin depth at surge periods is
`δ_skin=√(κ_ice P/π)≈0.5–5 m` (`P=0.02–2 yr`) — the perturbation never penetrates beyond
metres of ice, so the full-thickness `H` is the wrong scale; (ii) a drainage is an
*impulse*, whose semi-infinite diffusive response is a monotone `t^{−1/2}` decay with
**no peak**, yet observed speed-ups **rise to a peak**. The lag is therefore
**hydromechanical**, not thermal — the memory relocates into subglacial hydrology
(cavity/channel pressurisation, effective-pressure control of sliding), with the
ice-thermal `t^{−1/2}` term subdominant.

---

## 5. The replacement kernel is an exact Mori–Zwanzig memory kernel

We refine the ontology to **two driving potentials** — the Leray pressure constraint
(elliptic) and temperature (parabolic) — plus a **distinct, nonlinear-parabolic
hydraulic subsystem** (cavity↔channel storage–transport, with `φ` the hydraulic
potential, *not* the Leray pressure). Its lumped linearisation `ẋ=Mx+F`, `x=(s,q)` (`s`
the resolved cavity store, `q` the eliminated channel variable,
`M=[[−1/τ₁,−a],[b,−1/τ₂]]`), reduces by exact projection (linear Nakajima–Zwanzig / Mori,
no Markov approximation) to

> `ṡ(t) = M_ss s(t) + ∫₀ᵗ K(t−τ) s(τ) dτ + R(t)`, `K(τ) = M_sq M_qs e^{M_qq τ}`,

a non-local (memory) equation whose kernel `K(τ)=−ab·e^{−τ/τ₂}` is exactly the channel
subsystem's own Green's function weighted by the up/down couplings. The harness reuses the
**same coupled model** as the lag-shape/lag-value derivations and certifies five
properties (`validation/synthetic/hydraulic_mz_projection_synthetic.py`, 6/6 tests):

1. **Projection exact to machine precision.** The GLE Laplace transfer function equals
   the full 2×2 resolvent's (1,1) component for 400 random stable overdamped systems —
   max abs err **`5.0×10⁻¹⁶`**.
2. **The kernel IS the eliminated subsystem's Green's function** — reproduced to
   **`6.9×10⁻¹⁸`**, decaying at exactly `1/τ₂`, sign-definite.
3. **The reduced GLE reproduces the resolved trajectory** to rel-err **`9.1×10⁻⁸`** over
   the transient.
4. **Memory is necessary for the lag/peak.** The full response peaks at the interior time
   set by the coupled eigenvalues (`t*=0.911`, analytic `0.911`); the Markovian
   adiabatic-elimination closure collapses to a monotone response with argmax at `t=0`.
5. **Markovian limit = adiabatic elimination (DC gain).** `∫₀^∞ K dτ = M_sq M_qs/(−M_qq)`
   (err `7.5×10⁻⁹`); shrinking the channel time `τ₂→0` at fixed DC gain collapses the
   kernel toward `(∫K)·δ(τ)`, recovering the local closure — the FDT-Markov limit, here
   *derived from the projection* rather than assumed.

This certifies the *structure* of the replacement mechanism (that the surge-lag memory is
a genuine MZ kernel obtained by projecting out the channel, and how it degenerates to a
local closure). (Werder et al. 2013; Hewitt 2013; Schoof 2010; Röthlisberger
1972)

**Honest scope.** This validates the **projection structure** of the *linearised lumped*
model. It does **not** validate the physical identity of `φ` as distinct from the Leray
pressure, that this subsystem dominates a real surge, or any field lag value — those
remain, gated on real co-temporal velocity data (§6).

**5.3 Physical magnitudes — the dimensional bridge and the roughness closure.** The reduced kernel is dimensionless; the bridge to metres uses only known
ice constants. With `ρ_iL=3.0×10⁸ J m⁻³`, Glen `A`, effective pressure `N`, and literature
subglacial inputs (`Q`, `∂φ/∂s`), the steady R‑channel `S*=V_o/k_creep`, `R*=√(2S*/π)`,
`τ=1/k_creep` is **metre-scale** (central `R*≈2.4 m`, `τ≈0.18 yr`; 73 % of the literature
box gives 0.1–50 m) — the R‑channel regime, no free fudge factor; the wide upper tail is
the low-`N` near-flotation limit (`k_creep∝N³→0`, the §3 fold reappearing in the
hydrology). Crucially `ρ_iL` **cancels** in the scallop fraction `ΔS/S=V_scallop/V_o=0.33`,
so a scalloped reach grows channels **+33 % in area** (calibration-free); the only genuine
calibration is the concentration gain `g∈[0.1,0.9]` (network competition, direction-
invariant). The roughness leg closes the `φ` loop: `z_0=c_z·a` feeds the log-law drag
`C_d=[κ/ln(H/z_0)]²`, and although the Nikuradse prefactor `c_z` is ~10× uncertain, the
log law **buffers** it — at the Curl anchor a 10× `z_0` uncertainty propagates to only
~1.7× in `C_d` (and `∂φ/∂s`), the least damaging open closure
(`validation/synthetic/a3_dimensional_bridge.py`, `a2_z0_roughness.py`; 6+5 tests).
(Röthlisberger 1972; Werder et al. 2013; Cuffey & Paterson 2010; Nikuradse 1933;
Curl 1966)

---

## 6. The matched lag test — an honest null

Both halves of a matched test are open: drainage **dates** from CryoSat-2 in-lake
elevation (`detect_drainages` extracts 7 fill→drain events across 5 lakes, including the
documented Mercer ≈2012.5 drainage), and velocity **response** from ITS_LIVE box-mean
surface speed.

| lake | base speed (m/yr) | drainages | annual noise floor | quarterly scatter | peak post-drainage |
|---|---|---|---|---|---|
| Mac1 | 422 | 3 | 1.0% | 2.3% | **+0.56σ** |
| Mac2 | 402 | 1 | 1.1% | 2.1% | +0.27σ |
| Mac3 | 394 | 1 | 1.1% | 3.4% | −0.12σ |

**Result.** No matched lake shows a post-drainage surge above the 2σ floor (peak
`+0.56σ`, 0/5 significant). The lag value stays, *not* promoted to.

**Why null is a *stronger* statement than "too noisy".** The modern ITS_LIVE record is
well resolved (annual ≈1 %, quarterly ≈2–3 %), so the limit is not a noise budget. Two
distinct, quantifiable limits remain: (1) **temporal aliasing** — the derived lag is
sub-annual (`t*` baseline ≈0.01 yr, p95 ≈0.1 yr), finer than the finest *robust* ITS_LIVE
bin (quarterly, 0.25 yr); a brief transient is averaged out regardless of scatter (a
Nyquist limit); (2) **amplitude bound** — a *sustained* speed-up would survive aliasing as
a step, yet the peak anomaly is only `+0.56σ`, bounding any sustained surge to ≲10 m/yr on
~400 m/yr. These map onto the two routes to: **sub-annual GPS/GNSS**
(EarthScope/POLENET; beats aliasing) **or fast-outlet trunk velocity** (e.g. Byrd,
Stearns et al. 2008; steep `v(N)` sensitivity beats the amplitude floor), paired with that
site's drainage dates. The thermal `H²/κ` kernel stays regardless.

**Vetted forcing obtained.** USAP-DC 601439 (Smith et al. 2009 ICESat) yields **58
drainage events across 52 lakes** (median drained volume 0.128 km³, max 2.92 km³;
implied water flux `q_water` median ≈2.5 m³/s, p95 ≈31 m³/s — squarely in the literature
subglacial-flood range), anchoring the §4/§5 forcing magnitude to real observations.

---

## 7. A field-measurable sliding law: the `s_N(N)` master curve, three probes, and an ungrounding early-warning

The §4 falsification relocated the surge memory into effective pressure `N`. This section
turns `N` from a tuned coefficient into a **measured** quantity with a closed-form,
invertible sliding law and an operational precursor — the constructive capstone of the
paper's sliding-law arc (`validation/synthetic/sn_master_curve.py`,
`tidal_admittance_probe.py`, `spatial_ews.py`; 17 tests).

**7.1 `s_N(N)` as the common observable.** A lake drainage is an in-situ effective-
pressure *step* whose surge amplitude `du/u=|s_N|·(dN/N)` measures the sliding sensitivity
`s_N=d ln u_b/d ln N`; ocean thermal forcing lowers `N` *continuously*, so the gating
slope `d ln u_*/dTF` measures the **same** `s_N` (`+0.46→+0.62/°C`, direct
BedMachine `N`); ocean tides oscillate `N` at known amplitude, a third probe (7.3). All
three read one curve.

**7.2 The closed-form master curve.** Solving the
regularized-Coulomb law `τ_b=C N (u/(u+u0))^{1/m}` at `τ_b=τ_d` gives, exactly,
`|s_N|(N)=m/(1−(N_c/N)^m)`, `N_c=τ_d/C` — `→m` far from flotation, a **simple pole** at
`N_c` (`≈N_c/(N−N_c)`). Verified to `1.4×10⁻⁴` against the numeric `s_N`. This *derives*
the near-flotation weakening that Joughin, Smith & Schoof (2019) impose by hand (ad hoc
`h_af<h_T` ramp + fixed `u0`).

**7.3 Inversion — measuring `N_c` and `m`.** A
drainage-step population recovers the flotation/Type-III threshold `N_c` to **0.2 / 0.4 /
1.0 %** at 5 / 10 / 20 % amplitude noise; `m` is degenerate with the per-event drop far
from flotation. Ocean **tides** give a continuous third probe: the fundamental velocity
admittance equals `|s_N|` and the 2f/1f harmonic ratio diverges toward `N_c`, so
admittance + harmonics + the known tidal amplitude recover `m` and the dimensionless
flotation proximity `R=(N_c/N)^m` **from surface velocity alone** — a sliding-law reading
of the Gudmundsson nonlinear MSf response, answering the "no knowledge of `N`" objection.
The tidal forcing amplitude is **measured**, not assumed: sampling the CATS2008 tide model
(USAP-DC 601235) across the BedMachine grounding zone (within 10 km of the line; `n=1108`
valid cells) gives a tidal ocean-pressure swing with median **0.30 % of ice overburden**
(median tide `η≈0.84 m`), rising monotonically from 0.21 % at 8–10 km to 0.45 % at 2–4 km
from the grounding line (`validation/reports/tidal_forcing_gz.json`).

**7.4 An ungrounding early-warning (operationalizing established CSD theory).** The `N_c`
fold makes the basal drag stiffness `∂τ_b/∂u∝(1−R)²/R→0` (critical slowing down), so an
ice stream approaching ungrounding should show **rising variance and lag-1 autocorrelation
of its surface speed** (Kendall-τ≈0.54 in an OU demonstration with `N` declining toward
`N_c`). Critical-slowing-down early-warning is established (Scheffer et al. 2009; Dakos et
al. 2008) and has already been applied to marine-ice-sheet tipping — Rosier et al. (2021)
detected rising lag-1 autocorrelation and variance in grounding-line flux ahead of Pine
Island tipping, and Boers & Rypdal (2021) in Greenland surface melt. The contribution here
is not the early-warning concept but a specific **operational route**: reading
`R(t)=(N_c/N)^m→1` from the continuous tidal admittance of surface velocity. A
single-snapshot **spatial** analogue (variance + along-flow correlation length rising
toward the grounding line) is also derived (exact Lyapunov solve).

**7.5 The framework on the same real lakes.** Reading the §6 matched data
(open CryoSat-2 drainage + ITS_LIVE velocity, 3 MacAyeal lakes) through this lens
(`validation/external/lake_lag_sn_ews.py`, 5 tests): across the 3 resolved drainage
events `|Δv/v|≤2 %`, mixed sign, near the ~1 % year-to-year noise floor — **0 positive
>2σ surge detections** — so via the master curve these trunk lakes sit **far from the
`N_c` pole**; and **no** lake shows the joint critical-slowing-down signature (rising
variance *and* high AC1), so the early-warning correctly returns a **true negative** on
stable ice (unit tests confirm the detectors *do* fire on planted surges/CSD, so the null
is real). Both probes agree: far from ungrounding — consistent with the 1/19 population
non-detection.

**7.6 Scope and falsification.** Absolute `s_N` is uncalibrated (per-event `dN/N` is a
lumped-storage lower bound); the robust claims are the sign/shape and the *threshold*
`N_c`. With n=8 annual points per lake the CSD test is weak; a strong test needs the dense
quarterly/GPS series and a stream actually nearing flotation. Falsifiers: a co-located
`dN`+GPS population whose `|s_N(N)|` does not follow `m/(1−(N_c/N)^m)`; a stream nearing
flotation with no variance/AC1 rise; tidal admittance/harmonics not steepening toward the
grounding line. (Schoof 2005; Gagliardini et al. 2007; Joughin, Smith & Schoof
2019; Tulaczyk et al. 2000; Gudmundsson 2011; Scheffer et al. 2009; Dakos et
al. 2008; Boers & Rypdal 2021)

---

## 8. Estimator calibration (synthetic plant-and-recover)

Every real-data estimator has a synthetic null/recovery harness exercised in `pytest`:

- **RTN classifier** — plant `H*`, recover `RTN>1` region (`rtn_synthetic`).
- **Lag estimator is kernel-shape-generic** — plant one target lag with five markedly
  different causal kernels (Gamma `k=2/4`, log-normal, bi-exponential, raised-cosine);
  `estimate_lag` recovers it to ≤10 % for every shape, delta-kernel control returns 0.
  So a recovered lag is a property of the data, not the assumed Gamma form.
- **Grounding-line migration level-set law is exact** — `|dH/dt|/|∇m|` reproduced to
  rel-err `5×10⁻¹⁶` (1-D) / `4×10⁻¹⁴` (tilted 2-D); the thinning-paced null is flat by
  construction; a planted `u_*` discount is recovered sign-faithfully and is unbiased
  under 25 % noise / null under permutation.
- **`φ`-area inversion, MISI band, intrusion-clock drivers** — calibrated on synthetic
  ground truth.
- **CMN commutator identity** — to ~`10⁻⁷` (`cmn_synthetic`).

These guarantee the §3–§6 verdicts are read against the right null with unbiased,
sign-faithful estimators.

---

## 9. Discussion and limitations

**What this is.** A machine-precision certification of the cavity↔channel kernel's exact
MZ structure (the central positive result); a motivated null on a literal thermal sliding
kernel; an honest matched-lag null with a precise diagnosis and a concrete unblock path; a
flotation-ratio intrusion *screen* with an independent cross-check; and a field-measurable
sliding law (the `s_N(N)` master curve, an `N_c` inversion, and an early-warning that
operationalizes established CSD theory) read honestly against the same real lakes.

**What this is not.** No RTN precision/recall (no gridded survey). No verified surge-lag
*value* (the matched test is null; the value is to order of magnitude only).
No nonlinear hydraulic-kernel derivation (the §5 result is the *linearised lumped*
projection; a coupled GlaDS/Röthlisberger derivation is future work). The physical
distinctness of `φ` from the Leray pressure remains unproven.

**The open gates, each with a concrete unblock.**
1. **Matched lag →:** sub-annual GPS/GNSS (EarthScope/POLENET) or
   fast-outlet trunk velocity, paired with drainage dates.
2. **Derived nonlinear hydraulic kernel:** a coupled GlaDS/Röthlisberger cavity↔channel
   model (replace the lumped RC analogue).
3. **RTN precision/recall:** a future gridded intrusion survey (NISAR / IceBridge
   successors) to score against.
4. **`s_N` law / early-warning →:** a co-located `dN`+GPS (or tidal-
   admittance) population at a stream nearing flotation, to fit `m/(1−(N_c/N)^m)` and
   test the variance/AC1 rise; and a DInSAR `v_obs` over runaway intrusion cells to place
   `Ro` on the thinning-paced vs hydraulic-limited curve (§3.5).

---

## 10. Reproduce

```bash
pip install -r requirements.txt
python glaciers/validation/external/run_rtn_bedmap2.py --stride 3 --phi 0.9 # §3 RTN (Fig 57 family)
# §3.3 BedMachine cross-check: glaciers/validation/external/rtn_corollaries_bedmachine.py (GPU)
python glaciers/validation/external/rtn_baseline_skill.py --stride 5 # §3.4 RTN==thickness-above-flotation (zero skill at constant phi)
python glaciers/validation/synthetic/rtn_variable_phi_skill.py # §3.4b variable-phi is the only escape (plant-and-recover, 5 tests)
python glaciers/validation/external/rtn_variable_phi_real.py --stride 10 # §3.4b hydraulic-routing phi(x,y) reorders RTN on real Bedmap2
python glaciers/validation/external/rtn_specularity_real.py --stride 5 --min-pts 3 # §3.4b MEASURED phi from ICECAP radar specularity (USAP-DC 601371)
python glaciers/validation/external/run_sliding_real.py --with-velocity Mac1 # §4 τ_ice + ITS_LIVE
python -m glaciers.validation.external.lag_fit_real # §6 matched-lag null
python glaciers/validation/external/run_usapdc_lakes.py # §6 vetted forcing (58 events)
pytest glaciers/tests/test_validation_synthetic.py -v # §8 estimator calibration + §5 MZ projection (6/6)
python glaciers/validation/synthetic/intrusion_residence_number.py # §3.5 intrusion residence number Ro
python glaciers/validation/synthetic/sn_master_curve.py # §7 s_N(N) master curve + inversion
python glaciers/validation/synthetic/tidal_admittance_probe.py # §7.3 tidal third probe
python glaciers/validation/external/lake_lag_sn_ews.py # §7.5 §I framework on real lakes
```

## Data and code availability

The intrusion result (Figs. 1–2) uses open datasets — Bedmap2, BedMachine Antarctica v4, CryoSat-2 lake-drainage dates, and ITS_LIVE velocities — and regenerates from the scripts in the reproduce section. All analysis code and derived products are in the public repository at https://github.com/abdh0saifelden2-debug/K.

## References

Arthern, R. J., Hindmarsh, R. C. A., & Williams, C. R. (2015). Flow speed within the Antarctic ice sheet and its controls inferred from satellite observations. *Journal of Geophysical Research: Earth Surface* 120(7), 1171–1188. doi:10.1002/2014JF003239

Boers, N., & Rypdal, M. (2021). Critical slowing down suggests that the western Greenland Ice Sheet is close to a tipping point. *Proceedings of the National Academy of Sciences* 118(21), e2024192118. doi:10.1073/pnas.2024192118

Cuffey, K. M., & Paterson, W. S. B. (2010). *The Physics of Glaciers*, 4th ed. Academic Press, Amsterdam.

Curl, R. L. (1966). Scallops and flutes. *Transactions of the Cave Research Group of Great Britain* 7, 121–160.

Dakos, V., Scheffer, M., van Nes, E. H., Brovkin, V., Petoukhov, V., & Held, H. (2008). Slowing down as an early warning signal for abrupt climate change. *Proceedings of the National Academy of Sciences* 105(38), 14308–14312. doi:10.1073/pnas.0802430105

Dow, C. F., McCormack, F. S., Young, D. A., Greenbaum, J. S., Roberts, J. L., & Blankenship, D. D. (2020). Totten Glacier subglacial hydrology determined from geophysics and modeling. *Earth and Planetary Science Letters* 531, 115961. doi:10.1016/j.epsl.2019.115961

Fretwell, P., Pritchard, H. D., Vaughan, D. G., et al. (2013). Bedmap2: improved ice bed, surface and thickness datasets for Antarctica. *The Cryosphere* 7, 375–393. doi:10.5194/tc-7-375-2013

Gardner, A. S., Moholdt, G., Scambos, T., et al. (2018). Increased West Antarctic and unchanged East Antarctic ice discharge over the last 7 years. *The Cryosphere* 12, 521–547. doi:10.5194/tc-12-521-2018

Gardner, A., Fahnestock, M., Greene, C. A., Kennedy, J. H., Liukis, M., Lopez, L., & Scambos, T. (2025). MEaSUREs ITS_LIVE Regional Glacier and Ice Sheet Surface Velocities, Version 2 (NSIDC-0776) [Data set]. *NASA NSIDC DAAC.* doi:10.5067/JQ6337239C96

Gagliardini, O., Cohen, D., Råback, P., & Zwinger, T. (2007). Finite-element modeling of subglacial cavities and related friction law. *Journal of Geophysical Research: Earth Surface* 112, F02027. doi:10.1029/2006JF000576

Gudmundsson, G. H. (2011). Ice-stream response to ocean tides and the form of the basal sliding law. *The Cryosphere* 5, 259–270. doi:10.5194/tc-5-259-2011

Hewitt, I. J. (2013). Seasonal changes in ice sheet motion due to melt water lubrication. *Earth and Planetary Science Letters* 371–372, 16–25. doi:10.1016/j.epsl.2013.04.022

Joughin, I., Smith, B. E., & Schoof, C. G. (2019). Regularized Coulomb friction laws for ice sheet sliding: Application to Pine Island Glacier, Antarctica. *Geophysical Research Letters* 46, 4764–4771. doi:10.1029/2019GL082526

Leguy, G. R., Asay-Davis, X. S., & Lipscomb, W. H. (2014). Parameterization of basal friction near grounding lines in a one-dimensional ice sheet model. *The Cryosphere* 8, 1239–1259. doi:10.5194/tc-8-1239-2014

Mori, H. (1965). Transport, collective motion, and Brownian motion. *Progress of Theoretical Physics* 33, 423–455. doi:10.1143/PTP.33.423

Morlighem, M., Rignot, E., Binder, T., et al. (2020). Deep glacial troughs and stabilizing ridges unveiled beneath the margins of the Antarctic ice sheet. *Nature Geoscience* 13, 132–137. doi:10.1038/s41561-019-0510-8

Nakajima, S. (1958). On quantum theory of transport phenomena. *Progress of Theoretical Physics* 20, 948–959. doi:10.1143/PTP.20.948

Nikuradse, J. (1933). Strömungsgesetze in rauhen Rohren. *VDI-Forschungsheft* 361. (English transl. NACA TM 1292, 1950.)

Robel, A. A., Wilson, E., & Seroussi, H. (2022). Layered seawater intrusion and melt under grounded ice. *The Cryosphere* 16, 451–469. doi:10.5194/tc-16-451-2022

Rosier, S. H. R., Reese, R., Donges, J. F., De Rydt, J., Gudmundsson, G. H., & Winkelmann, R. (2021). The tipping points and early warning indicators for Pine Island Glacier, West Antarctica. *The Cryosphere* 15, 1501–1516. doi:10.5194/tc-15-1501-2021

Röthlisberger, H. (1972). Water pressure in intra- and subglacial channels. *Journal of Glaciology* 11, 177–203. doi:10.3189/S0022143000022188

Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H., van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early-warning signals for critical transitions. *Nature* 461, 53–59. doi:10.1038/nature08227

Schoof, C. (2005). The effect of cavitation on glacier sliding. *Proceedings of the Royal Society A* 461, 609–627. doi:10.1098/rspa.2004.1350

Schoof, C. (2007). Ice sheet grounding line dynamics: Steady states, stability, and hysteresis. *Journal of Geophysical Research: Earth Surface* 112, F03S28. doi:10.1029/2006JF000664

Schoof, C. (2010). Ice-sheet acceleration driven by melt supply variability. *Nature* 468, 803–806. doi:10.1038/nature09618

Schroeder, D. M., Blankenship, D. D., & Young, D. A. (2013). Evidence for a water system transition beneath Thwaites Glacier, West Antarctica. *Proceedings of the National Academy of Sciences* 110(30), 12225–12228. doi:10.1073/pnas.1302828110

Schroeder, D. M., Blankenship, D. D., Raney, R. K., & Grima, C. (2015). Estimating subglacial water geometry using radar bed echo specularity: Application to Thwaites Glacier, West Antarctica. *IEEE Geoscience and Remote Sensing Letters* 12(3), 443–447. doi:10.1109/LGRS.2014.2337878

Shreve, R. L. (1972). Movement of water in glaciers. *Journal of Glaciology* 11, 205–214. doi:10.3189/S002214300002219X

Siegfried, M. R., Fricker, H. A., Carter, S. P., & Tulaczyk, S. (2016). Episodic ice velocity fluctuations triggered by a subglacial flood in West Antarctica. *Geophysical Research Letters* 43, 2640–2648. doi:10.1002/2016GL067758

Siegfried, M. R., & Fricker, H. A. (2018). Thirteen years of subglacial lake activity in Antarctica from multi-mission satellite altimetry. *Annals of Glaciology* 59, 42–55. doi:10.1017/aog.2017.36

Siegfried, M. R., & Fricker, H. A. (2021). Illuminating active subglacial lake processes with ICESat-2 laser altimetry. *Geophysical Research Letters* 48, e2020GL091089. doi:10.1029/2020GL091089

Smith, B. E., Fricker, H. A., Joughin, I. R., & Tulaczyk, S. (2009). An inventory of active subglacial lakes in Antarctica detected by ICESat (2003–2008). *Journal of Glaciology* 55, 573–595. doi:10.3189/002214309789470879

Stearns, L. A., Smith, B. E., & Hamilton, G. S. (2008). Increased flow speed on a large East Antarctic outlet glacier caused by subglacial floods. *Nature Geoscience* 1, 827–831. doi:10.1038/ngeo356

Stubblefield, A. G., Arthern, R. J., Kingslake, J., & Siegfried, M. R. (2021). Antarctic ice thickness, slipperiness, and subglacial lake locations. *U.S. Antarctic Program Data Center.* doi:10.15784/601470

Tulaczyk, S., Kamb, W. B., & Engelhardt, H. F. (2000). Basal mechanics of Ice Stream B, West Antarctica: 1. Till mechanics. *Journal of Geophysical Research: Solid Earth* 105(B1), 463–481. doi:10.1029/1999JB900329

Werder, M. A., Hewitt, I. J., Schoof, C. G., & Flowers, G. E. (2013). Modeling channelized and distributed subglacial drainage in two dimensions. *Journal of Geophysical Research: Earth Surface* 118, 2140–2158. doi:10.1002/jgrf.20146

Wingham, D. J., Francis, C. R., Baker, S., et al. (2006). CryoSat: A mission to determine the fluctuations in Earth's land and marine ice fields. *Advances in Space Research* 37, 841–871. doi:10.1016/j.asr.2005.07.027

Young, D. A., Blankenship, D. D., Greenbaum, J., Roberts, J., Schroeder, D., Siegert, M., & van Ommen, T. (2020). ICECAP Basal Interface Specularity Content Profiles: IPY and OIB [Data set]. *U.S. Antarctic Program (USAP) Data Center.* doi:10.15784/601371

Zwanzig, R. (1960). Ensemble method in the theory of irreversibility. *The Journal of Chemical Physics* 33, 1338–1341. doi:10.1063/1.1731409

Zwanzig, R. (1973). Nonlinear generalized Langevin equations. *Journal of Statistical Physics* 9, 215–220. doi:10.1007/BF01008729
