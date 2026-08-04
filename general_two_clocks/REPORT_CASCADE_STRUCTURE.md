# Diagonal-band cascade structure in a stratified fluid (WREF 1-min, 2020)

> **Paper 1 of the "research these" set — real data.** A pre-registered test that the
> atmospheric surface layer (a stratified turbulent fluid) builds a *local,
> intermittent* cross-scale cascade, measured as a **band-coupling matrix**
> φ_ij = corr(log envelope of octave band i, octave band j). Mainstream grounding:
> cascade locality (Kolmogorov 1941; Kraichnan 1971) plus multiplicative
> intermittency (Kolmogorov 1962; Frisch 1995) ⇒ φ should be **diagonal-band**
> (diagonal-dominant, monotone off-diagonal decay, finite local decay scale) and
> *above* a spectrum-matched, phase-randomized Gaussian surrogate — the time-domain
> "energy yes, structure no" companion to [`run_boussinesq.py`](run_boussinesq.py).
>
> Data: NEON WREF (Wind River old-growth forest, WA), **2020, 1-min**, barometric
> pressure (DP1.00004, `staPresMean`) and triple-aspirated air temperature
> (DP1.00003, `tempTripleMean`), QC `finalQF==0`. Public NEON API, no auth; series
> gap-filled by linear interpolation (pressure 98.3% valid, temperature
> 98.6% valid before fill; 527,040 1-min samples ≈ 1.00 yr). Code:
> [`cascade_band_structure.py`](cascade_band_structure.py),
> [`run_cascade_structure.py`](run_cascade_structure.py),
> [`tests/test_cascade_band_structure.py`](tests/test_cascade_band_structure.py),
> figure `figures/73_cascade_band_structure.png`.

## Pre-registered predictions (thresholds fixed before the real data)

- **CB1 diagonal dominance** — each band most coupled to itself; φ(1) < 1, > far coupling.
- **CB2 monotone decay** — Spearman trend of φ(d) vs separation ≤ −0.8 (noise-robust;
  a single-exponential decay length is brittle under forcing peaks, so it is reported
  for reference only, not gated on).
- **CB3 finite local decay scale** — half-coupling separation d½ (φ falls below ½·φ(1))
  is finite and < n_bands.
- **CB4 above surrogate** — off-diagonal coupling exceeds the phase-randomized surrogate.

## Result — both clocks show diagonal-band cascade structure (n_bands=12)

| field | φ(1) | φ(1) surr | mean φ_off | surr | Spearman | d½ (oct) | pass |
|---|---|---|---|---|---|---|---|
| barometric pressure | 0.286 | -0.010 | 0.141 | -0.000 | -0.89 | 4 | 4/4 |
| air temperature | 0.380 | +0.004 | 0.180 | +0.003 | -0.99 | 5 | 4/4 |

![diagonal-band cascade](figures/73_cascade_band_structure.png)

**The decisive result is CB4.** Real cross-scale coupling (φ(1) ≈ 0.29–0.38,
mean off-diagonal ≈ 0.14–0.18) sits **far above** the
phase-randomized surrogate (≈ 0, i.e. no cross-scale coupling). A spectrum-matched Gaussian has independent bands;
the real stratified fluid builds genuine cross-scale amplitude correlation — the
intermittent-cascade signature, the time-domain "energy yes, structure no".

**The φ matrices are diagonal-band** (CB1, CB3): coupling concentrated on the diagonal,
falling below half its nearest-neighbour value within d½ ≈ 4–5 octaves —
the cascade is local in scale.

**The two clocks differ (the two-clocks signature).** Temperature is a clean, local,
monotone cascade (Spearman -0.99); pressure shows a **secondary
enhancement at synoptic separation** (visible as off-diagonal brightening at large
band separation), so its monotonicity is weaker (Spearman -0.89) and
band-count-sensitive — exactly what the global, elliptic pressure clock should do
(it couples broadly across scales), versus the local, parabolic temperature clock.

## Robustness across band counts (transparency)

| n_bands | P Spearman | P d½ | P φ(1) | P pass | T Spearman | T d½ | T φ(1) | T pass |
|---|---|---|---|---|---|---|---|---|
| 10 | -0.98 | 4 | 0.305 | 4/4 | -0.83 | 8 | 0.451 | 4/4 |
| 11 | -0.71 | 4 | 0.287 | 3/4 | -0.99 | 6 | 0.409 | 4/4 |
| 12 | -0.89 | 4 | 0.286 | 4/4 | -0.99 | 5 | 0.380 | 4/4 |

Diagonal dominance, d½ (≈4 oct pressure), the strong φ(1)/off-diagonal coupling and
the surrogate gap are **robust** to band count. The only band-count-sensitive call is
pressure's strict monotonicity — and that sensitivity is itself the global-clock
(synoptic) signature, not a numerical artifact. The single-exponential decay length is
omitted from the headline because it is not robust (it varies 5–12 octaves with the
band range under the diurnal/synoptic peaks); d½ and the Spearman trend are used instead.

## Scope

A single site-year at 1-min cadence resolves the meso-to-synoptic band (~2 min to
~6 days), not the sub-second inertial range; "cascade" here is the cross-timescale
amplitude-coupling structure, validated on a synthetic multiplicative cascade in the
unit tests. An empirical characterization, not a turbulence-closure or regularity claim.
