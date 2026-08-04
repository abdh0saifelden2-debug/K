# NR65 — the phase face of the tidal EIS: the bed's complex wavenumber and its 45° loss angle

**Module** `general_two_clocks/new_relationships42.py` ·
**data** the committed §I.5 harmonic cache, REBUILT with absolute-epoch phases (`glaciers/.../tidal_admittance_field_cache.json`) ·
**figure** `general_two_clocks/figures/100_tidal_phase_dispersion.json` + `.png` ·
**tests** `general_two_clocks/tests/test_tidal_phase_dispersion.py` (11)

## The relationship

NR64 measured **|Z|** — the upstream attenuation length δ_amp = 1/k_i of each tidal
constituent. A diffusive (parabolic) bed transmits exp(iωt − (1+i)d/δ): amplitude decay
and phase lag advance at the **same rate**,

k_r = k_i = √(ω/2K)  (loss angle 45°),  c = ω/k_r = √(2ωK),  K_φ = ω/2k_r².

So the MSf **phase-lag profile φ(d)** is a second, *independent* in-situ measurement of
the same hydraulic diffusivity — the two faces must agree with **no free parameter**.
The competing readings decouple them: in-situ generation by the nonlinear sliding law
alone predicts MSf phase *"almost constant"* upstream (Rosier, Gudmundsson & Green 2014,
Fig. 6b: k_r ≈ 0 while k_i > 0), and purely elastic transmission likewise screens
amplitude without accumulating travel time. The measured **loss tangent k_i/k_r** is
therefore a sharper two-clock discriminant than NR64's underpowered Mm/MSf ratio.

## Enabling step (methods)

The §I.5 harmonic LSQ referenced phases to each station's first sample — useless across
stations. The cache is rebuilt with all records on **one absolute epoch** (hours since
1970-01-01; MATLAB datenum − 719529), `y = A cos(ωt + φ)`; amplitudes are bit-identical
(worst relative change 4.6×10⁻⁹; all 21 pre-existing §I.5/NR64 tests pass unchanged).
Cross-epoch phases are legitimate because MSf is the M2×S2 difference frequency —
astronomically phase-locked. **Empirical control:** truly-afloat neighbour stations
(separation < 60 km, |d_GL| ≥ 5 km) deployed in different years agree in M2 height
phase to **0.034 rad** (E2B–E4B, E4B–E5) — the cross-epoch phase floor.

## Measured (MSf, along-flow displacement, grounded stations)

| branch | k_r [rad/km] | ordering (exact 1-sided) | δ_φ [km] | c | K_φ [m²/s] | k_i/k_r | K_amp |
|---|---|---|---|---|---|---|---|
| Foundation (n=5) | 0.0144 ± 0.0094 (jk) | 5/5 monotone, p=0.008 | 69 | **29 km/d** (0.34 m/s) | 1.2×10⁴ | **1.45 ± 1.05** | 5.6×10³ |
| Evans (n=4) | 0.0096 ± 0.0032 | 4/4 monotone, p=0.042 | 104 | 44 km/d (0.51 m/s) | 2.7×10⁴ | 4.7 ± 6.1 | 1.2×10³ |
| Talutis (2pt) | 0.0348 | — | 29 | 12 km/d | 2.0×10³ | 2.6 | 2.9×10² |
| Rutford (2pt, wrap-risk) | 0.0244 | — | 41 | 17 km/d (alt branch: 4 km/d) | 4.1×10³ | 1.9 | 1.2×10³ |
| Evans_XX (2pt) | sign violation | — | — | — | — | — | — |

Sign concordance: **4/5 branches accumulate lag upstream** — the registered direction of
a wave entering at the grounding line.

## Reading

1. **The MSf response travels — it is not a standing, locally-generated pattern.**
   On Foundation the phase drops 0.84 rad over 43 km, strictly monotone across 5
   stations (exact p = 0.008). Rosier-2014's generation-only prediction (constant
   phase) is excluded where the data have power.
2. **The loss angle is consistent with 45°.** Foundation k_i/k_r = 1.45 ± 1.05
   (parabolic prediction: exactly 1); the two independent diffusivities agree within a
   factor 2 (K_φ = 1.2×10⁴ vs K_amp = 5.6×10³ m²/s; geometric mean 8.2×10³).
   One transport process carries both faces of the complex wavenumber.
3. **Mainstream cross-check, same number from independent methods:** Foundation
   c = 29 km/d equals Minchew et al. 2017's InSAR propagation rate on *Rutford*
   (29 km/d), and sits at Rosier-2014's model value for the fortnightly band
   (0.27 m/s = 23 km/d). Gudmundsson-2006's 1–2 m/s is the *semidiurnal-band*
   modulation speed — the √ω dispersion (1.45/0.27 m/s in Rosier-2014's own model)
   explains the order-of-magnitude gap between the two bands' literature values.
4. **What the mainstream leaves on the table:** Rosier et al. 2017 (ESSD, this very
   archive) note qualitatively that "the speed at which the Msf signal propagates
   upstream shows more variation" — the joint (k_r, k_i) read-out, its loss tangent,
   and the K_φ = K_amp closure are not published. NR65 turns their data release into
   a two-parameter impedance spectrometer.

## Honest scope

Formal per-station σ_φ (~10⁻³ rad) is unrealistic against seasonal/epoch systematics —
branch errors are jackknife over stations. Foundation's slope is 1.5σ_jk as a *slope*
(the profile is quasi-linear with local structure H01→H10 flat, H10→H02 steep) but the
*ordering* is exact-permutation significant. MSf is partly generated along the path
(Gudmundsson 2007/2011), so k_r, like δ_amp, is an effective linear-response quantity —
distributed generation biases both faces together, which is why their ratio is the
robust discriminant. Two-point branches carry a 2π ambiguity (Rutford flagged; the
nearest branch is the one compatible with Minchew's independent 29 km/d). Loss tangents
> 1 on the weak branches (Evans 4.7 ± 6.1) are unresolved, not evidence.
