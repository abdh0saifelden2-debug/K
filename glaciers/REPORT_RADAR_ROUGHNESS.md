# §V.1e — The specularity gauge in metres (`validation/external/e1_radar_roughness.py`)

**Claim.** The ICECAP HiCARS bed-echo *specularity content* (USAP-DC 601371) is not just a
[0,1] wetness index: under the Kirchhoff approximation for a Gaussian-height interface it
is the coherent power fraction `s = exp(−g)`, `g = (4π σ cosθ / λ_ice)²` (Ament 1953;
Beckmann & Spizzichino 1963), so it inverts **in closed form, with zero free parameters**,
to an RMS interface roughness in metres:

    σ(s) = K √(−ln s),      K = λ_ice / (4π cosθ) ≈ 0.223 m   (HiCARS, nadir)

with `λ_ice = c/(f·n_ice) ≈ 2.81 m` (60 MHz; `n_ice = √3.17`, Fujita et al. 2000 — the wave
strikes the bed *while propagating in ice*, so the in-ice wavelength is the right one, not
the 5 m vacuum value). This executes horizon-ledger item **E1** and gives the P4a §6.4
spec→φ driver physical units; with NR33 it yields the first joint `(σ, φ)` bed-state
constraints in metres and pascals.

Status: **[VERIFIED on real data]** — 14 unit proofs (`tests/test_e1_radar_roughness.py`,
including a first-principles Monte-Carlo of the power convention) + full-survey inversion
(3.2 M points, 432 transects, 14 126 covered 5-km Bedmap2 cells) with the lake anchor
transferring rank-exactly. Artifacts: `validation/reports/e1_radar_roughness.{json,png}`.

---

## 1. Three closed-form corollaries (all unit-proved)

1. **Gauge window.** Trusting specularity only in `[ε, 1−ε]` bounds the measurable
   roughness to `σ ∈ [K√(−ln(1−ε)), K√(−ln ε)]` — for ε = 0.05: **5.1 cm … 38.7 cm**.
   HiCARS specularity is a *decimetre* gauge; smoother beds saturate toward s→1, rougher
   beds are censored at the specularity noise floor (here s < 0.01 ⇒ only `σ > 0.48 m`).
2. **Sweet spot.** Noise amplification `|dσ/ds| = K/(2s√(−ln s))` is minimised at exactly
   `s* = e^{−1/2} ≈ 0.607`, where `σ* = K/√2 ≈ 15.8 cm`. The gauge is most precise right
   at the water/rock discrimination scale.
3. **The Schroeder-2015 water criterion drops out.** Schroeder, Blankenship, Raney & Grima
   (2015, IEEE GRSL 12(3); online 2014) estimated — from independent scattering/attenuation/cross-section
   models of Thwaites specularity — that distributed-water interfaces have **RMS roughness
   ≲ 15 cm**. Our inversion turns that bound into a specularity threshold with no tuning:
   `σ ≤ 0.15 m ⇔ s ≥ exp(−(0.15/K)²) = 0.637`. "Specular water" in the established
   qualitative sense *is* the σ ≤ 15 cm class of this gauge.

## 2. What the East Antarctic data say (601371 × Bedmap2 × 601470)

| quantity | value |
|---|---|
| point-level median σ (censoring-aware) | **0.387 m** |
| point-level p10 | 0.220 m (p90 censored: > 0.48 m; 21.8 % of points below spec floor) |
| covered grounded 5-km cells | 14 126 (2 073 censored) |
| cell-level median σ | 0.385 m |
| water-like cells (σ ≤ 15 cm ⇔ s ≥ 0.637) | 61 cells; 5.3 % of raw points |
| **lake anchor (metres)**: median σ, 74 active-lake cells | **0.333 m** |
| — vs flotation-matched controls (13 410 cells) | **0.387 m** (Δ = 5.4 cm smoother) |
| — rank-exact transfer of §V.1d line 1 | P[σ_lake < σ_ctrl] = 0.692, stratified-permutation p = 5.0 × 10⁻⁴ |

The covered East Antarctic bed reads σ ≈ 0.2–0.5 m at the sub-Fresnel horizontal scale
(first Fresnel radius `√(λ_ice h/2)` ≈ 50–80 m for 2–4 km ice) — decimetre relief, i.e.
*hydraulically* rough but much smoother than the 10²–10⁴ m-wavelength "total roughness"
of the FFT traverse literature (Bingham & Siegert 2007-class analyses), which measures a
different (longer-wavelength) part of the spectrum. Active-lake footprints are smoother
than matched controls with the *identical* significance as §V.1d (the inversion is
strictly monotone, so every rank statistic transfers verbatim) — what is new is the
*metric* statement: **ponded-water cells sit ~5 cm smoother in RMS at 65-m scales**.

**Where the smoothest cells are.** The 61 σ ≤ 15 cm cells contain **zero** active-lake
footprint cells and sit *deep inland* (median 620 km from the grounding line vs 209 km
for the background). This is the expected physics, not a failure: altimetry-active lakes
(fill–drain features under ice streams) and *interior distributed canal systems* are
different hydrological objects — specularity highs concentrate in the interior
(Young et al. 2016; Dow et al. 2019 for Totten), and specularity *drops* toward margins
as water channelises (the Thwaites transition of Schroeder et al. 2013). The gauge reads
the interior distributed-water class at σ ≲ 15 cm exactly where the mainstream picture
puts it; active lakes appear as a *distributional* smoothing at 5-km binning (epoch
mismatch 2003–2016 outlines vs 2008–2012 flights + ~10-km lakes in 5-km cells dilute
their contrast — both directions conservative, as in §V.1d).

## 3. NR33 joint bed state — (σ, φ) in physical units

For the 7 two-epoch persistent-specular cells (s ≥ 0.2 in 2008/09 **and** 2011/12;
NR33/§V.1d set), combining this gauge (smoothness) with the creep-persistence floor
(pressure; N_max = 3.55 bar at t_obs = 4 yr):

| cells | σ estimate [m] | φ floor | H [m] | dist to GL [km] |
|---|---|---|---|---|
| 5 deep-interior cells | 0.19–0.26 | ≥ 0.990 | ≈ 3 870–3 945 | 396–421 |
| 2 mid-catchment cells | 0.21–0.28 | ≥ 0.981 | ≈ 2 030–2 130 | 115–120 |

**Statement (the P4a §6.4 driver in units):** persistently specular East Antarctic cells
are simultaneously *smooth at the ~0.2–0.3 m RMS level over ~65 m scales* and
*pressurised to within ≈ 3.5 bar of overburden* (φ ≥ 0.98). Any process model for these
cells must reproduce both numbers at once; a borehole or gridded intrusion survey can
falsify either axis independently.

## 4. Honest scope

* **Model assumptions**: Gaussian heights, Kirchhoff validity (curvature ≫ λ, moderate
  slopes), roughness stationary within the ~1-km L2 smoothing, specularity content read
  as the coherent power fraction (its design intent — the along-track angular
  decomposition of Schroeder et al. 2013, 2015). Self-affine beds (Jordan et al. 2017)
  make σ scale-dependent; this is the leading-order single-scale reading at the Fresnel
  footprint, and σ here ≠ hydraulic `z₀` of §A.2/E8 (different moments of the spectrum —
  connecting them needs a correlation-length/Hurst assumption, deliberately not made).
* **Two opposing bias directions, both derived and unit-proved**: within-cell
  heterogeneity (Jensen on the convex `s(σ²)`) makes binned-cell σ a *lower* bound on
  cell RMS; extraneous decoherence (englacial scattering, clutter, epoch mixing) *raises*
  σ. Absolute values carry this bracket; the lake–control *contrast* and all rank
  statistics are immune to any common-mode monotone distortion.
* **Censoring is handled exactly**: below-floor specularity enters every percentile as a
  +inf rank (a censored `σ > 0.48 m`), never dropped — dropping would bias the survey
  smooth by construction (unit-proved against the naive estimator).
* The dielectric constant (3.17 ± 0.02) moves K by < 0.4 %; incidence off-nadir by 5°
  moves it by < 0.4 %. Parameter risk is negligible next to the model assumptions.

## 5. Payoff hooks

* **P4a §6.4**: the φ driver's observable now has units — "spec = 0.2–0.5" reads
  "σ ≈ 0.28–0.17 m"; joint (σ, φ) constraints above.
* **NR33**: persistence bound (pressure) + this gauge (geometry) = the two-clocks split
  read *simultaneously* from one instrument.
* **E5b (ledger)**: along- vs across-track specularity anisotropy → directional σ —
  channel azimuth in metres, the routing-escape upgrade.
* **Wavelength-transfer prediction (falsifiable)**: at MCoRDS 195 MHz (λ_ice ≈ 0.86 m,
  K ≈ 6.9 cm) the same beds must read `s' = s^{(λ₁/λ₂)²} = s^{10.6}` — e.g. our lake-cell
  median s = 0.108 predicts near-zero 195-MHz specularity, while a σ = 7 cm patch
  (s = 0.91 at HiCARS) predicts s' ≈ 0.36. Dual-frequency surveys over the same lines
  can kill or confirm the Gaussian-Kirchhoff reading outright.
