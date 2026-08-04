# NR64 — the tidal constituent ladder is an in-situ EIS sweep of the subglacial bed

**Module** `general_two_clocks/new_relationships41.py` ·
**data** the committed §I.5 harmonic cache (`glaciers/.../tidal_admittance_field_cache.json` — offline replay) ·
**figure** `general_two_clocks/figures/99_tidal_eis_ladder.json` + `.png` ·
**tests** `general_two_clocks/tests/test_tidal_eis_ladder.py` (11)

## The relationship

NR30 split the two constraint pressures — Leray (elliptic, instantaneous, storage-free)
vs Darcy head (parabolic, storage, finite-time) — and NR38 read the bed as a Randles
circuit whose EIS sweep the tides perform for free. NR64 executes that sweep
**spatially**: each tidal constituent probes the bed at its own frequency ω, and its
upstream attenuation length is the corresponding **skin depth**. The parabolic clock
predicts, with **no free parameters**,

δ(ω) = √(2K/ω) ⟹ δ_Mm/δ_MSf = √(27.55/14.77) = **1.366**, δ_MSf/δ_M2 = **5.34**.

Mainstream anchors: Rosier, Gudmundsson & Green (2015) model tidal head as exactly this
diffusion (their Eq. 12; decay scale √(2K/ω); K *"poorly constrained… treated as an
unknown"*); Rosier & Gudmundsson (2016): hydrology dominates MSf while flexure/damming
dominate the semidiurnal band; Rosier et al. (2014): the Maxwell (viscoelastic)
alternative — elastic screening at short periods, **viscous saturation at long periods**,
i.e. δ_Mm/δ_MSf ≈ 1.00. The Mm/MSf ratio is therefore a clean two-clock discriminant:
**1.366 (parabolic) vs 1.00 (Maxwell-saturated)**.

## Measured on the real network (committed cache; grounded stations)

| branch | δ_MSf [km] | δ_Mm [km] | δ_Mm/δ_MSf | K_MSf [m²/s] |
|---|---|---|---|---|
| Foundation (n=5) | 48 ± 15 | 62 ± 21 | **1.30** | **5.6 × 10³** |
| Evans (n=4) | 22 ± 27 | 22 ± 15 | **1.00** | **1.2 × 10³** |
| Talutis (2-pt) | 11 | 12 | 1.09 | — |
| Rutford (2-pt, mixed epochs) | 22 | 39 | 1.76 | — |

- **A new method of measurement:** the GPS tidal ladder returns the **in-situ subglacial
  hydraulic diffusivity K ≈ 1.2–5.6 × 10³ m²/s** — the exact parameter the Rosier-2015
  model must leave free and requires to be "highly conductive". No borehole, no radar:
  harmonic amplitudes at 3–5 GPS stations suffice.
- **The two bands ride different clocks (NR30's field face).** The directly-forced
  semidiurnal band is at the few-mm OTL/noise floor by the FIRST grounded station on
  every branch (δ_M2 ≤ 1.4–8.7 km, taking the last floating station as the forcing
  scale), while MSf/Mm penetrate 10–60 km. The short/long ratio ≥ 5–30 **exceeds** the
  single-diffusion prediction 5.34 on most branches — semidiurnal screening is
  elastic/flexural (with the Holdsworth stress-reversal node), long-period penetration
  is hydraulic. One transport law cannot carry both — the program's two-clocks thesis,
  measured on ice.
- **The long-band discriminant is genuinely open at current n:** measured δ_Mm/δ_MSf =
  {1.00, 1.09, 1.30, 1.76}, mean 1.29 ± 0.17 (se) — between Maxwell saturation (1.00)
  and the parabolic prediction (1.366), leaning parabolic but not separable at 3–4
  branches. **Registered decisive measurement:** two more multi-season stations at
  40–70 km on Foundation Ice Stream (where δ_MSf = 48 km gives the longest lever and
  the smallest relative error) would resolve 1.37 vs 1.00 at ≳2σ.

## Honest scope

MSf/Mm on grounded ice are partly nonlinearity-*generated* along the path (Gudmundsson
2007/2011: MSf = M2×S2 intermodulation through m>1 sliding), so δ mixes generation and
transmission, and m>1 itself stretches decay scales (Rosier 2015, Fig. 8) — K is the
*effective linear-response* diffusivity, exactly the object their model parameterizes.
The diurnal band is excluded (K1 = GPS orbit-repeat artifact; S2 carries solar aliases).
Rutford's two stations mix epochs (2004–07 vs 2010s). Evans' jackknife error is honest
about EGPS's 45-day span. Absolute δ values inherit the MEaSUREs GL position (±1–2 km).

## Chain

NR30 (two constraint pressures) → NR38 (bed as Randles/EIS) → §I.5 field unit (the
harmonic cache) → **NR64** (the spatial EIS sweep: skin depths, in-situ K, and the
parabolic-vs-Maxwell discriminant for the long band).
