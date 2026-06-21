# A Mori–Zwanzig memory kernel for subglacial drainage: the surge-lag memory is hydraulic, not ice-thermal

*Abdelrahman Saifelden (AIET) — Abdulrahman.Saifelden22011411@aiet.edu.eg*

---

## Abstract

Post-drainage ice-stream speed-ups lag their subglacial trigger by months to a
couple of years, and the lag carries *memory*: the response rises to a peak rather
than decaying monotonically. We make one claim and defend it three ways. **The
lumped, linearised cavity↔channel storage system's surge response is an exact
Mori–Zwanzig memory kernel**: projecting out the channel variable yields a
generalized Langevin equation whose kernel equals the eliminated subsystem's own
Green's function (verified to `7×10⁻¹⁸`, projection exact to `5×10⁻¹⁶`, reduced
model reproducing the full trajectory to `9×10⁻⁸`), with the Markovian
(adiabatic-elimination) local closure recovered exactly as the channel time
vanishes. This certifies *why* surge response carries memory and *how* it
degenerates to a local closure — a structural result, not a field-validated lag
value. We motivate it by **falsifying the only zeroth-order alternative**: a literal
ice-thermal lag `τ_ice=H²/κ_ice` is rejected by ~5 orders of magnitude on the 131
Siegfried & Fricker (2018) active lakes (median `τ_ice≈1.5×10⁵ yr` vs observed
0.02–2 yr; robust on the independent Arthern et al. (2015) thickness inversion, `r=0.970`), and on first
principles (the thermal skin depth at surge periods is metres, and an impulse gives
a peakless `t^{−1/2}` response while observations peak). We then expose the kernel to
the one open co-temporal dataset — a matched lake-drainage→velocity test (CryoSat-2
drainage dates + ITS_LIVE speeds) — which returns an **honest null** (peak `+0.56σ`,
0/5 significant), limited by temporal aliasing and an amplitude bound, not record
precision, with two concrete unblock routes. Finally we include, as a continental
*screen* rather than a forecast, the Regime Transition Number `RTN`, show it is by
construction the flotation/effective-pressure ratio (so a constant-`φ` `RTN` adds no
skill over thickness-above-flotation), and identify its only genuine degree of
freedom — a spatially-varying basal-connectivity fraction `φ(x,y)` — with a
plant-and-recover proof and a real-Bedmap2 hydraulic-routing demonstration. The
contributions are the exact MZ certification (with two extensions that retire the "it is
just a 2×2 linear-algebra identity" objection: a *nonlinear* extension — the linear kernel
is the small-amplitude limit of the Röthlisberger system, which predicts that bigger floods
lag longer; and a *distributed* extension — the kernel is an exact, spatially non-local
operator projection of the linearised flowline PDE whose no-transport, single-node limit is
exactly the lumped 2×2, and which predicts a finite along-flow memory footprint), the
honest-falsifier discipline, and the diagnosis of the matched-lag null — not a verified
surge-lag value.

---

## 1. Introduction

Subglacial hydrology controls ice-stream sliding through the effective pressure
`N=p_i−p_w`, the difference between ice overburden and basal water pressure (Cuffey &
Paterson 2010; Schoof 2010; Röthlisberger 1972). Drainage events — subglacial-lake
floods — perturb `N` and trigger transient speed-ups (Stearns et al. 2008; Siegfried
et al. 2016), but the speed-up *lags* the drainage and the lagged response is
non-monotone (it peaks), implying the cavity↔channel system carries memory (Werder et
al. 2013; Hewitt 2013). The modelling question is what sets that memory.

The mainstream tools are established and we use, not claim, them: the
Mori–Zwanzig/Nakajima–Zwanzig projection and the generalized Langevin equation (Mori
1965; Nakajima 1958; Zwanzig 1960, 1973), the Röthlisberger channel and
GlaDS-type cavity↔channel drainage (Röthlisberger 1972; Werder et al. 2013; Hewitt
2013), and the flotation/effective-pressure criterion (Schoof 2007; Leguy et al.
2014). What is new here is not a mechanism but an **exact structural certification**
and the **discipline applied to its alternatives**.

**What is new, stated against the closest prior work.** Werder/Hewitt model the
cavity↔channel system forward; we instead *reduce* its linearisation by exact
projection and show the surge-lag memory kernel **is** the eliminated channel
subsystem's Green's function, with the local (Markovian) closure recovered as a
limit — a closed-form statement of *why* the lag has memory and *when* a local sliding
closure is admissible. No established theory predicts the literal `H²/κ` thermal lag;
we state it as an explicit straw hypothesis precisely so the active-lake record can
reject it, and report that rejection as a *motivated null* rather than a discovery.

**Contributions.**
1. A **motivated falsification** of the literal `H²/κ` ice-thermal sliding-lag kernel
   on 131 real lakes, relocating the memory into hydrology (§3).
2. The **exact Mori–Zwanzig structure** of the cavity↔channel hydraulic memory kernel
   — a structural certification (§4), with a **nonlinear extension** showing the linear
   kernel is the small-amplitude limit of the physical Röthlisberger system and predicting
   an amplitude-dependent (bigger-floods-lag-longer) surge lag (§4.4), and a **distributed
   (spatially-resolved) extension** showing the kernel is an exact operator projection of
   the linearised flowline PDE — the lumped 2×2 is its no-transport, single-node limit, and
   channel transport gives the surge-lag memory a finite along-flow footprint (§4.5).
3. An **honest matched-lag null** with a precise account of *why* it is null (aliasing
   + amplitude bound, not noise) and two routes to a verified lag (§5).
4. A nondimensional **intrusion screen** `RTN`, shown to equal the flotation ratio,
   with its only genuine degree of freedom (a data-derived `φ(x,y)`) isolated and
   tested (§6) — presented as a screen, not a verified forecast.
5. **Calibrated estimators** via synthetic plant-and-recover (§7).

---

## 2. Data and methods

| Dataset | What | Access |
|---|---|---|
| BAS **Bedmap2** | thickness, bed, surface, grounded mask (1 km) | open, no auth |
| **USAP-DC 601470** (Stubblefield et al. 2021) | independent thickness (Arthern et al. 2015) + 131 lake stats | open (USAP-DC) |
| **BedMachine v4** (NSIDC-0756) | independent continental thickness (RTN screen) | NSIDC (Earthdata) |
| **Siegfried & Fricker (2018)** | 131 active subglacial-lake outlines | open (GitHub mirror) |
| **ITS_LIVE** | surface-speed datacubes (image-pair `v`) | open (anonymous S3) |
| **CryoSat-2** in-lake elevation | drainage *dates* | open (Siegfried 2021-GRL mirror) |
| **USAP-DC 601439** | ICESat lake volume-change (drainage volumes) | open |

Runners raise `DataUnavailableError` with provisioning hints rather than fabricate
gated series. Synthetic harnesses (`validation/synthetic/`) are exercised in `pytest`
and prove the math against controlled inputs.

---

## 3. The thermal sliding-lag kernel, falsified

**3.1 The hypothesis (a motivated null).** No established sliding theory predicts that
surge lag equals the *full-thickness* ice thermal-diffusion time; we state
`τ_ice = H²/κ_ice` (`κ_ice=1.09×10⁻⁶ m² s⁻¹`) as an explicit, falsifiable straw
hypothesis precisely so the data can reject it.

**3.2 The test.** On real Bedmap2 thickness at the 131 Siegfried & Fricker (2018)
catalogued active lakes (130 with valid thickness; median `H=2282 m`, range
637–3905 m):

- `τ_ice = H²/κ_ice`: **median ≈ 1.5×10⁵ yr** (p5 ≈ 2×10⁴; p95 ≈ 3×10⁵ yr).
- Observed post-drainage surge lags: **0.02–2 yr** (Stearns et al. 2008;
  Siegfried et al. 2016).
- ⇒ the literal kernel is **~8×10⁴× too slow** at the median lake; the `log₁₀ τ_ice`
  histogram (10⁴–10⁶ yr) is fully disjoint from the observed band.

**3.3 Robust to the thickness dataset.** Recomputing on an independent ice-thickness
dataset (USAP-DC 601470, Stubblefield et al. 2021; ice thickness after Arthern et al.
2015): median thickness 2356 m, `τ_ice` median **≈1.55×10⁵ yr**; the 601470-vs-Bedmap2
thickness at the same centroids agrees at **`r=0.970`**, so the falsification is not an
artefact of one sampling.

**3.4 Why the falsification is diagnostic.** Two facts exclude ice thermal diffusion as
the lag-setting mechanism: (i) the thermal skin depth at surge periods is
`δ_skin=√(κ_ice P/π)≈0.5–5 m` (`P=0.02–2 yr`) — the perturbation never penetrates
beyond metres of ice, so the full-thickness `H` is the wrong scale; (ii) a drainage is
an *impulse*, whose semi-infinite diffusive response is a monotone `t^{−1/2}` decay
with **no peak**, yet observed speed-ups **rise to a peak**. The lag is therefore
**hydromechanical**, not thermal — the memory relocates into subglacial hydrology
(cavity/channel pressurisation, effective-pressure control of sliding), with the
ice-thermal `t^{−1/2}` term subdominant.

---

## 4. The replacement kernel is an exact Mori–Zwanzig memory kernel

We model the **hydraulic subsystem** as cavity↔channel storage–transport (with `φ` the
hydraulic potential, *not* the Leray pressure). Its lumped linearisation `ẋ=Mx+F`,
`x=(s,q)` (`s` the resolved cavity store, `q` the eliminated channel variable,
`M=[[−1/τ₁,−a],[b,−1/τ₂]]`), reduces by exact projection (linear Nakajima–Zwanzig /
Mori, no Markov approximation) to

> `ṡ(t) = M_ss s(t) + ∫₀ᵗ K(t−τ) s(τ) dτ + R(t)`, `K(τ) = M_sq M_qs e^{M_qq τ}`,

a non-local (memory) equation whose kernel `K(τ)=−ab·e^{−τ/τ₂}` is exactly the channel
subsystem's own Green's function weighted by the up/down couplings. The harness
certifies five properties
(`validation/synthetic/hydraulic_mz_projection_synthetic.py`, 6/6 tests):

1. **Projection exact to machine precision.** The GLE Laplace transfer function equals
   the full 2×2 resolvent's (1,1) component for 400 random stable overdamped systems —
   max abs err **`5.0×10⁻¹⁶`**.
2. **The kernel IS the eliminated subsystem's Green's function** — reproduced to
   **`6.9×10⁻¹⁸`**, decaying at exactly `1/τ₂`, sign-definite.
3. **The reduced GLE reproduces the resolved trajectory** to rel-err **`9.1×10⁻⁸`** over
   the transient.
4. **Memory is necessary for the lag/peak.** The full response peaks at the interior
   time set by the coupled eigenvalues (`t*=0.911`, analytic `0.911`); the Markovian
   adiabatic-elimination closure collapses to a monotone response with argmax at `t=0`.
5. **Markovian limit = adiabatic elimination (DC gain).** `∫₀^∞ K dτ = M_sq M_qs/(−M_qq)`
   (err `7.5×10⁻⁹`); shrinking the channel time `τ₂→0` at fixed DC gain collapses the
   kernel toward `(∫K)·δ(τ)`, recovering the local closure — the FDT-Markov limit, here
   *derived from the projection* rather than assumed.

This certifies the *structure* of the replacement mechanism (that the surge-lag memory
is a genuine MZ kernel obtained by projecting out the channel, and how it degenerates
to a local closure). (Werder et al. 2013; Hewitt 2013; Schoof 2010; Röthlisberger
1972)

**Honest scope.** This validates the **projection structure** of the *linearised
lumped* model. It does **not** validate the physical identity of `φ`, that this
subsystem dominates a real surge, or any field lag value — those remain gated on real
co-temporal velocity data (§5).

**4.3 Physical magnitudes.** The reduced kernel is dimensionless; the bridge to metres
uses only known ice constants. With `ρ_iL=3.0×10⁸ J m⁻³`, Glen `A`, effective pressure
`N`, and literature subglacial inputs (`Q`, `∂φ/∂s`), the steady R-channel
`S*=V_o/k_creep`, `R*=√(2S*/π)`, `τ=1/k_creep` is **metre-scale** (central `R*≈2.4 m`,
`τ≈0.18 yr`; 73 % of the literature box gives 0.1–50 m) — the R-channel regime, no free
fudge factor; the wide upper tail is the low-`N` near-flotation limit (`k_creep∝N³→0`).
The roughness leg uses `z_0=c_z·a` in the log-law drag `C_d=[κ/ln(H/z_0)]²`; although
the Nikuradse prefactor `c_z` is ~10× uncertain, the log law **buffers** it — a 10× `z_0`
uncertainty propagates to only ~1.7× in `C_d`
(`validation/synthetic/a3_dimensional_bridge.py`, `a2_z0_roughness.py`; 6+5 tests).
(Röthlisberger 1972; Werder et al. 2013; Cuffey & Paterson 2010; Nikuradse 1933;
Curl 1966)

**4.4 The linear kernel is the small-amplitude limit of the nonlinear system — and the
nonlinearity is itself a falsifiable prediction.** The §4 projection is exact but *linear*,
and a fair objection is that eliminating one variable of a 2×2 system is a linear-algebra
identity. We therefore embed it in the **physically nonlinear** Röthlisberger cavity↔channel
model — turbulent `√p` channel discharge and the Glen-law `N³` creep closure (Röthlisberger
1972; Schoof 2010; Hewitt 2013) — and linearise at its steady state. The linearisation
recovers *exactly* the overdamped `M` of §4 (real, negative eigenvalues), so **the linear MZ
kernel `K(τ)=−ab·e^{−τ/τ₂}` is the small-amplitude limit**: a plant-and-recover sweep shows
the nonlinear surge-lag `t*` converging to the linear value as the drainage amplitude → 0
(relative gap `<0.5%` at a 2 %-of-baseflux flood, rising smoothly with amplitude — Fig. 1).
The leading nonlinear correction is not cosmetic; it is driven by the `N³` creep term and
yields a falsifiable geophysical prediction. A larger drainage drives water pressure up,
`N=p_i−p_w` down, collapses the `N³` creep closure, keeps the channel open longer, and
**lengthens the lag**: the model gives `t*≈0.10 yr` at small floods, rising ~14 % to
`≈0.11 yr` at an 80 %-of-baseflux flood (`d t*/d(flood fraction) ≈ 0.018 yr`). So *bigger
floods should have longer surge lags* — a statement the linear kernel cannot make, testable
against a drainage-volume-stratified lag population (the USAP-DC 601439 volumes of §5 already
span 0.13–2.9 km³). This converts the exact-but-linear certification into the leading term of
a controlled, physically-grounded expansion with a testable next-order signature. The
prediction is **not a parameter artefact**: across a physical sweep of steady effective
pressure, storage and melt-opening efficiency, bigger-floods-lag-longer holds for *every*
overdamped configuration and **strengthens toward flotation** (the lag–flood slope rises from
`≈0.010 yr` at `N*=0.55` to `≈0.032 yr` at `N*=0.25`), i.e. the effect is largest exactly in
the low-`N` near-flotation regime where surges matter most.
(`validation/synthetic/hydraulic_nonlinear_kernel.py`, 2 tests.)

![Surge lag `t*` versus drainage flood size in the nonlinear Röthlisberger cavity↔channel model. The linear Mori–Zwanzig kernel (dashed) is amplitude-independent and equals the nonlinear lag in the small-flood limit; the Glen-law `N³` creep closure makes the nonlinear lag (circles) rise with flood size — bigger floods lag longer, a falsifiable prediction the linear kernel cannot make.](figures/77_hydraulic_nonlinear_kernel.png)

**4.5 The kernel survives spatial resolution: a distributed operator projection whose
no-transport limit is the lumped 2×2.** The other half of the "2×2 identity" objection is
*spatial*: a lumped two-box model has no geometry, so is the memory kernel an artefact of
collapsing the drainage line to a point? We resolve the flowline. Take the linearised
cavity↔channel system as two coupled *fields* on a 1-D flowline — `∂ₜs = D_s ∂ₓₓs − s/τ₁ −
a q` (cavity store `s`) and `∂ₜq = D_q ∂ₓₓq − U ∂ₓq + b s − q/τ₂` (the channel `q`, which
*routes water downstream* at speed `U` and spreads it at `D_q`; Röthlisberger 1972; Werder
et al. 2013) — and project out the channel *field* exactly (linear Nakajima–Zwanzig). The
resolved store obeys a generalized Langevin equation `ṡ = A_ss s + ∫₀ᵗ 𝒦(t−τ) s(τ) dτ + R`
whose kernel `𝒦(τ) = A_sq e^{A_qq τ} A_qs` is now a *matrix-valued, spatially non-local*
operator. On `N=24` flowline nodes
(`validation/synthetic/hydraulic_mz_spatial.py`, 6 tests): (i) the projection is **exact** —
the GLE transfer operator equals the full resolvent's resolved block at random complex `s`,
max rel-err `7×10⁻¹⁶`; (ii) the kernel **is** the eliminated channel operator's Green's
function (`3×10⁻¹⁷` vs `expm`) and is spatially non-local with a resolved along-flow range
`ℓ_mem≈0.31 ≫ Δx`; (iii) the channel-eliminated field-GLE reproduces the full PDE
trajectory (`1×10⁻⁵`); and (iv) memory necessity is a **dose-response** — the memoryless
local closure's trajectory error grows monotonically with the channel time `τ₂` and
vanishes as `τ₂→0` (the Markovian limit, now derived *in the distributed system*).
Crucially, **the committed lumped 2×2 kernel is exactly this projection's no-transport,
single-node limit**: with `D_q=U=0` the operator kernel `𝒦(τ)` is exactly diagonal and its
diagonal equals `−ab e^{−τ/τ₂}` to `7×10⁻¹⁸`. Switching channel transport on moves the
kernel mass off-diagonal (off-diagonal fraction `0→0.99`), so the surge-lag memory acquires
a finite **along-flow footprint** with the closed-form decay length
`ℓ_mem = 2D_q/(√(U²+4D_q/τ₂)−U)` (verified against the numeric steady-influence operator to
`4×10⁻⁴`; *ballistic* limit `ℓ_mem→Uτ₂` for fast channels, *diffusive* limit
`ℓ_mem→√(D_qτ₂)` for `U→0`) — a new, falsifiable, **field-measurable** structural
prediction: a drainage's lagged velocity response is spread along the flowline over `ℓ_mem`,
which a memoryless (local) closure sets to zero. This converts the 2×2 "identity" into the
spatially-local limit of an exact projection of a genuine distributed (PDE) operator. Scope:
still a *linearised* distributed model on a periodic flowline; the fully nonlinear
spatially-resolved coupled PDE remains future work.

![The distributed cavity↔channel memory kernel `𝒦(τ=τ₂)` (left) is spatially non-local — the off-diagonal band is the downstream channel transport that the lumped 2×2 omits; its along-flow influence profile (centre) has a finite memory range `ℓ_mem≈0.31` versus the lumped kernel's zero-range spike; and the memoryless local-closure trajectory error (right) grows with the channel relaxation time `τ₂`, vanishing in the Markovian (`τ₂→0`) limit.](figures/78_hydraulic_mz_spatial.png)

---

## 5. The matched lag test — an honest null

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
`+0.56σ`, 0/5 significant). The lag value stays *not* promoted to.

**Why null is a *stronger* statement than "too noisy".** The modern ITS_LIVE record is
well resolved (annual ≈1 %, quarterly ≈2–3 %), so the limit is not a noise budget. Two
distinct, quantifiable limits remain: (1) **temporal aliasing** — the derived lag is
sub-annual (`t*` baseline ≈0.01 yr, p95 ≈0.1 yr), finer than the finest *robust*
ITS_LIVE bin (quarterly, 0.25 yr); a brief transient is averaged out regardless of
scatter (a Nyquist limit); (2) **amplitude bound** — a *sustained* speed-up would
survive aliasing as a step, yet the peak anomaly is only `+0.56σ`, bounding any
sustained surge to ≲10 m/yr on ~400 m/yr. These map onto the two routes to a verified
lag: **sub-annual GPS/GNSS** (EarthScope/POLENET; beats aliasing) **or fast-outlet
trunk velocity** (e.g. Byrd, Stearns et al. 2008; steep `v(N)` sensitivity beats the
amplitude floor), paired with that site's drainage dates.

**Vetted forcing obtained.** USAP-DC 601439 (Smith et al. 2009 ICESat) yields **58
drainage events across 52 lakes** (median drained volume 0.128 km³, max 2.92 km³;
implied water flux `q_water` median ≈2.5 m³/s, p95 ≈31 m³/s — squarely in the literature
subglacial-flood range), anchoring the §3/§4 forcing magnitude to real observations.

---

## 6. A continental intrusion screen: RTN is the flotation ratio, and its only escape

**6.1 Definition.** `RTN = (p_ocean − p_atm)/p_w` on grounded ice: overburden
`p_i=ρ_i g H`, ocean head at the bed `p_ocean=ρ_w g·d_base` (`d_base=max(0,−bed)`),
subglacial water `p_w=φ p_i` (so `N_eff=(1−φ)p_i`), gauge-corrected so the atmosphere
cancels (`p_atm=0`). Then `RTN = ρ_w d_base/(φ ρ_i H)`, so `RTN>1 ⟺ ρ_w d_base/(ρ_i H)
> φ` — the ice is within a factor `φ` of flotation (`N→0`; Schoof 2007; Leguy et al.
2014; Cuffey & Paterson 2010). **`RTN` is, by construction, a nondimensional form of the
flotation/effective-pressure criterion, not an independent predictor.**

**6.2 Result on Bedmap2.** Over 1.33 M grounded cells (1 km decimated to 3 km, `φ=0.9`),
the `RTN>1` fraction decays monotonically with distance to the grounding line (15.1 %
within 5 km → 0 % beyond 100 km; median distance 6 km for `RTN>1` cells vs 222 km for
the rest), and the directional result reproduces on the independent **BedMachine v4**
(NSIDC-0756) inversion (median 6 km vs 200 km to the grounding line).

**6.3 No skill over thickness-above-flotation (constant `φ`).** A direct baseline test
on 478,767 grounded Bedmap2 cells confirms `{RTN>1}` is *identical* to the
thickness-above-flotation threshold `{H_af/H<1−φ}` (agreement **100.0000 %**, 0
disagreeing cells, `φ=0.8/0.9/0.95`), with `Spearman(RTN, H_af/H)=−1.000` — so at
constant `φ`, `RTN` **adds no skill**; it *is* that standard quantity, nondimensionalised
by `φ` (`validation/external/rtn_baseline_skill.py`).

**6.4 The only escape — a data-derived `φ(x,y)`.** Because `RTN=(1−H_af/H)/φ` is
algebraically forced to track the flotation fraction at constant `φ`, `RTN`'s only
possible independent signal is `φ` *itself*. A synthetic plant-and-recover
(`validation/synthetic/rtn_variable_phi_skill.py`, 5 tests) shows a
connectivity-informed `φ(x,y)` adds genuine skill over thickness-above-flotation
(near-flotation-band `F1` 0.78→0.89; `AUC` 0.995→0.999), a random `φ` of the same
distribution adds none, and the gain rises monotonically with `φ`'s informativeness. On
real Bedmap2, a calibration-free `φ(x,y)` from subglacial hydraulic-potential routing
(Shreve 1972: priority-flood fill + D8 flow accumulation) reproduces the constant-`φ`
anchor exactly and then reorders the intrusion class on ~10³ near-grounding-line cells
(near-flotation-band `Spearman` `−0.99` vs `−1.00`), nearly rank-independent of the
flotation fraction in-band (`validation/external/rtn_variable_phi_real.py`). The
reordering is real but its *correctness* is unvalidated — the routed `φ` is a model
field derived from the same geometry — so the decisive test needs a `φ` from an
**independent** observation of basal water (radar bed specularity; Schroeder et al.
2013, 2015), scored against a future gridded intrusion survey. We thus present
variable-`φ` `RTN` as a falsifiable screen with one clearly-stated open data step, not a
verified forecast.

**6.5 Intrusion residence number `Ro`.** The level-set advance speed of the `RTN=1`
front is `v_kin=A·dH/dt`; the residence number `Ro=v_kin·τ_hyd/ℓ` (regime boundary
`Ro=1` at the critical residence `τ_crit=ℓ/v_kin`) predicts whether intrusion is
thinning-paced (`Ro≈1`) or hydraulic-limited (`Ro≫1`). For the runaway-tail cells
(`A=0.70 km/m`, `dH/dt=1.5 m/yr`) ~98 % of the §4 residence band lies below `τ_crit≈1.9
yr` ⇒ predicted **thinning-paced**; a DInSAR `v_obs` would place it on the curve
(`validation/synthetic/intrusion_residence_number.py`, 7/7 tests).

---

## 7. Estimator calibration (synthetic plant-and-recover)

Every real-data estimator has a synthetic null/recovery harness exercised in `pytest`:

- **Lag estimator is kernel-shape-generic** — plant one target lag with five markedly
  different causal kernels (Gamma `k=2/4`, log-normal, bi-exponential, raised-cosine);
  `estimate_lag` recovers it to ≤10 % for every shape; a delta-kernel control returns 0.
- **RTN classifier / `φ`-area inversion / variable-`φ` skill** — planted thresholds and
  connectivity recovered without bias (`rtn_synthetic`, `rtn_phi_synthetic`,
  `rtn_variable_phi_skill`).
- **Grounding-line migration level-set law is exact** — `|dH/dt|/|∇m|` reproduced to
  rel-err `5×10⁻¹⁶` (1-D) / `4×10⁻¹⁴` (tilted 2-D).
- **MZ projection** — the five §4 properties to machine precision
  (`hydraulic_mz_projection_synthetic`, 6/6).

---

## 8. Discussion and limitations

**What this is.** A machine-precision certification of the cavity↔channel kernel's exact
MZ structure (the central positive result); a motivated null on a literal thermal
sliding kernel; an honest matched-lag null with a precise diagnosis and a concrete
unblock path; and a flotation-ratio intrusion *screen* whose only genuine degree of
freedom (a data-derived `φ(x,y)`) is isolated and partially tested.

**What this is not.** No verified surge-lag *value* (the matched test is null; the value
is to order of magnitude only). The §4.4 nonlinear extension shows the linear kernel is the
small-amplitude limit of the nonlinear *lumped* Röthlisberger system and predicts an
amplitude-dependent lag, and the §4.5 distributed extension certifies the kernel as an exact
spatially non-local operator projection of the linearised flowline PDE; only the fully
*nonlinear spatially-resolved* coupled GlaDS/Röthlisberger PDE remains future work. No RTN
precision/recall (no gridded survey). The physical distinctness of `φ` from the Leray
pressure remains unproven.

**Open gates, each with a concrete unblock.** (1) Matched lag → sub-annual GPS/GNSS or
fast-outlet trunk velocity, paired with drainage dates. (2) The *linear* distributed
projection (§4.5) and the *lumped* nonlinear extension (§4.4) are done; the remaining gate is
the fully *nonlinear spatially-resolved* coupled GlaDS/Röthlisberger PDE. (3) RTN skill → a
`φ(x,y)` from an
independent basal-water observable (specularity) + a gridded intrusion survey.

---

## 9. Reproduce

```bash
pip install -r requirements.txt
python glaciers/validation/external/run_sliding_real.py --with-velocity Mac1 # §3 τ_ice + ITS_LIVE
python -m glaciers.validation.external.lag_fit_real                          # §5 matched-lag null
python glaciers/validation/external/run_usapdc_lakes.py                      # §5 vetted forcing (58 events)
pytest glaciers/tests/test_validation_synthetic.py -v                        # §4 MZ projection (6/6) + §7
python glaciers/validation/external/rtn_baseline_skill.py --stride 5         # §6.3 RTN==flotation (zero skill)
python glaciers/validation/synthetic/rtn_variable_phi_skill.py               # §6.4 variable-φ is the only escape
python glaciers/validation/external/rtn_variable_phi_real.py --stride 10     # §6.4 hydraulic-routing φ(x,y)
python glaciers/validation/synthetic/intrusion_residence_number.py           # §6.5 residence number Ro
python glaciers/validation/synthetic/hydraulic_nonlinear_kernel.py            # §4.4 nonlinear kernel (linear = small-amplitude limit)
python glaciers/validation/synthetic/hydraulic_mz_spatial.py                  # §4.5 distributed projection (lumped 2×2 = no-transport limit)
```

## Data and code availability

All analysis code and derived products are in the public repository at
https://github.com/abdh0saifelden2-debug/K, and the results regenerate from the scripts
above. The study uses the following open datasets:

- Bedmap2 ice thickness (Fretwell et al., 2013, https://doi.org/10.5194/tc-7-375-2013).
- Independent ice thickness and active-lake inventory: USAP-DC 601470 (Stubblefield et al.,
  2021, https://doi.org/10.15784/601470; ice thickness after Arthern et al., 2015; active
  lakes after Siegfried & Fricker, 2018).
- Active subglacial-lake drainage volumes: USAP-DC 601439 (Smith et al., 2012,
  https://doi.org/10.15784/601439; ICESat inventory of Smith et al., 2009).
- BedMachine Antarctica v4 (Morlighem, 2025, https://doi.org/10.5067/POJQI54A45HX; see
  Morlighem et al., 2020) for the independent continental RTN screen.
- ITS_LIVE surface velocities (Gardner et al., 2025, https://doi.org/10.5067/JQ6337239C96;
  see Gardner et al., 2018) and CryoSat-2 lake-drainage dates.

USAP-DC data were obtained from the U.S. Antarctic Program Data Center
(https://www.usap-dc.org); BedMachine and ITS_LIVE from the NASA NSIDC DAAC.

## References

Arthern, R. J., Hindmarsh, R. C. A., & Williams, C. R. (2015). Flow speed within the Antarctic ice sheet and its controls inferred from satellite observations. *Journal of Geophysical Research: Earth Surface* 120(7), 1171–1188. doi:10.1002/2014JF003239

Cuffey, K. M., & Paterson, W. S. B. (2010). *The Physics of Glaciers*, 4th ed. Academic Press, Amsterdam.

Curl, R. L. (1966). Scallops and flutes. *Transactions of the Cave Research Group of Great Britain* 7, 121–160.

Fretwell, P., Pritchard, H. D., Vaughan, D. G., et al. (2013). Bedmap2: improved ice bed, surface and thickness datasets for Antarctica. *The Cryosphere* 7, 375–393. doi:10.5194/tc-7-375-2013

Gardner, A. S., Moholdt, G., Scambos, T., et al. (2018). Increased West Antarctic and unchanged East Antarctic ice discharge over the last 7 years. *The Cryosphere* 12, 521–547. doi:10.5194/tc-12-521-2018

Gardner, A., Fahnestock, M., Greene, C. A., Kennedy, J. H., Liukis, M., Lopez, L., & Scambos, T. (2025). MEaSUREs ITS_LIVE Regional Glacier and Ice Sheet Surface Velocities, Version 2 (NSIDC-0776) [Data set]. *NASA NSIDC DAAC.* doi:10.5067/JQ6337239C96

Hewitt, I. J. (2013). Seasonal changes in ice sheet motion due to melt water lubrication. *Earth and Planetary Science Letters* 371–372, 16–25. doi:10.1016/j.epsl.2013.04.022

Leguy, G. R., Asay-Davis, X. S., & Lipscomb, W. H. (2014). Parameterization of basal friction near grounding lines in a one-dimensional ice sheet model. *The Cryosphere* 8, 1239–1259. doi:10.5194/tc-8-1239-2014

Morlighem, M., Rignot, E., Binder, T., et al. (2020). Deep glacial troughs and stabilizing ridges unveiled beneath the margins of the Antarctic ice sheet. *Nature Geoscience* 13, 132–137. doi:10.1038/s41561-019-0510-8

Morlighem, M. (2025). MEaSUREs BedMachine Antarctica, Version 4 (NSIDC-0756) [Data set]. *NASA National Snow and Ice Data Center DAAC.* doi:10.5067/POJQI54A45HX

Mori, H. (1965). Transport, collective motion, and Brownian motion. *Progress of Theoretical Physics* 33, 423–455. doi:10.1143/PTP.33.423

Nakajima, S. (1958). On quantum theory of transport phenomena. *Progress of Theoretical Physics* 20, 948–959. doi:10.1143/PTP.20.948

Nikuradse, J. (1933). Strömungsgesetze in rauhen Rohren. *VDI-Forschungsheft* 361. (English transl. NACA TM 1292, 1950.)

Röthlisberger, H. (1972). Water pressure in intra- and subglacial channels. *Journal of Glaciology* 11, 177–203. doi:10.3189/S0022143000022188

Schoof, C. (2007). Ice sheet grounding line dynamics: Steady states, stability, and hysteresis. *Journal of Geophysical Research: Earth Surface* 112, F03S28. doi:10.1029/2006JF000664

Schoof, C. (2010). Ice-sheet acceleration driven by melt supply variability. *Nature* 468, 803–806. doi:10.1038/nature09618

Schroeder, D. M., Blankenship, D. D., & Young, D. A. (2013). Evidence for a water system transition beneath Thwaites Glacier, West Antarctica. *Proceedings of the National Academy of Sciences* 110(30), 12225–12228. doi:10.1073/pnas.1302828110

Schroeder, D. M., Blankenship, D. D., Raney, R. K., & Grima, C. (2015). Estimating subglacial water geometry using radar bed echo specularity: Application to Thwaites Glacier, West Antarctica. *IEEE Geoscience and Remote Sensing Letters* 12(3), 443–447. doi:10.1109/LGRS.2014.2337878

Shreve, R. L. (1972). Movement of water in glaciers. *Journal of Glaciology* 11, 205–214. doi:10.3189/S002214300002219X

Siegfried, M. R., Fricker, H. A., Carter, S. P., & Tulaczyk, S. (2016). Episodic ice velocity fluctuations triggered by a subglacial flood in West Antarctica. *Geophysical Research Letters* 43, 2640–2648. doi:10.1002/2016GL067758

Siegfried, M. R., & Fricker, H. A. (2018). Thirteen years of subglacial lake activity in Antarctica from multi-mission satellite altimetry. *Annals of Glaciology* 59, 42–55. doi:10.1017/aog.2017.36

Siegfried, M. R., & Fricker, H. A. (2021). Illuminating active subglacial lake processes with ICESat-2 laser altimetry. *Geophysical Research Letters* 48, e2020GL091089. doi:10.1029/2020GL091089

Smith, B. E., Fricker, H. A., Joughin, I. R., & Tulaczyk, S. (2009). An inventory of active subglacial lakes in Antarctica detected by ICESat (2003–2008). *Journal of Glaciology* 55, 573–595. doi:10.3189/002214309789470879

Smith, B., Fricker, H., Joughin, I., & Tulaczyk, S. (2012). Antarctic Active Subglacial Lake Inventory from ICESat Altimetry. *U.S. Antarctic Program Data Center.* doi:10.15784/601439

Stearns, L. A., Smith, B. E., & Hamilton, G. S. (2008). Increased flow speed on a large East Antarctic outlet glacier caused by subglacial floods. *Nature Geoscience* 1, 827–831. doi:10.1038/ngeo356

Stubblefield, A. G., Arthern, R. J., Kingslake, J., & Siegfried, M. R. (2021). Antarctic ice thickness, slipperiness, and subglacial lake locations. *U.S. Antarctic Program Data Center.* doi:10.15784/601470

Werder, M. A., Hewitt, I. J., Schoof, C. G., & Flowers, G. E. (2013). Modeling channelized and distributed subglacial drainage in two dimensions. *Journal of Geophysical Research: Earth Surface* 118, 2140–2158. doi:10.1002/jgrf.20146

Wingham, D. J., Francis, C. R., Baker, S., et al. (2006). CryoSat: A mission to determine the fluctuations in Earth's land and marine ice fields. *Advances in Space Research* 37, 841–871. doi:10.1016/j.asr.2005.07.027

Zwanzig, R. (1960). Ensemble method in the theory of irreversibility. *The Journal of Chemical Physics* 33, 1338–1341. doi:10.1063/1.1731409

Zwanzig, R. (1973). Nonlinear generalized Langevin equations. *Journal of Statistical Physics* 9, 215–220. doi:10.1007/BF01008729
