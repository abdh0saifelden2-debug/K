# §I.3 FIELD TEST — temporal CSD ungrounding early-warning on real ASE trunks (2014–2025)

**Module** `glaciers/validation/external/temporal_csd_field.py` ·
**cache** `glaciers/validation/external/data/temporal_csd_field_cache.json` (committed) ·
**report** `glaciers/validation/reports/temporal_csd_field.json` + `.png` ·
**tests** `glaciers/tests/test_temporal_csd_field.py` (8)

## What was registered

§I.3: near the flotation fold the restoring rate `λ ∝ (1−R)²/R → 0`, so a stream
approaching ungrounding shows **rising variance + rising lag-1 autocorrelation in its
surface speed** (velocity-based MISI early-warning, distinct from Boers & Rypdal 2021).
Registered falsifier: *"a stream observed to approach flotation with adequate sampling
and NO variance/AC1 rise."* The lake-lag §I re-analysis flagged that annual n=8 series
were too weak and a strong test needs *dense subannual* series — this unit is that test.

## Data and method

Quarterly ITS_LIVE v2 medians (pairs ≤ 60 d baseline, ≥5 pairs/quarter, MAD/√n SEs) at
the committed §I.6 flowline points: near-GL segments (≤40 km) and upstream bands
(80–140 km) of Thwaites/Pine Island/Smith (thinning → N declining through the record)
vs Rutford control. Per point (≥32 filled quarters of 48): joint season+trend removal →
12-quarter rolling **noise-corrected** variance (−window SE²) and AC1 → Kendall τ vs
time (Dakos 2008) → one-sided p from 200 **Fourier-phase surrogates** (stationary,
autocorrelation-preserving). Sampling-drift controls per point: τ of quarterly pair
counts and of SE² (the L8→+S2→+L9/S1 sensor history changes sampling density through
the record and must not be read as CSD).

## Result — REGISTERED NULL (with the falsifier's own scope condition examined)

| group | n | median τ_var | median τ_ac1 | joint-sig frac (p<0.1 both) |
|---|---|---|---|---|
| trunk near-GL | 31 | **−0.15** | **−0.18** | **0.00** |
| trunk upstream | 32 | +0.12 | −0.05 | 0.03 |
| control near-GL | 14 | +0.47 | +0.22 | 0.00 |
| control upstream | 9 | −0.07 | +0.21 | 0.00 |

- **No CSD signature on the near-flotation trunks**: median variance AND AC1 trends are
  *negative*; not one of 31 near-GL trunk points passes the joint surrogate criterion.
- The only positive variance-trend group is the **control** (+0.47, 0/14 significant) —
  and it carries the largest sampling-density drift (τ_npairs +0.49), exactly the
  artifact direction the control columns exist to flag.
- Amplitude ratio (late/early rolling σ): trunks ≈ 1.04 — fluctuation levels are flat
  through the decade.

## Reading it honestly (why this null does not kill §I.3, and what it does bound)

The §I.6 spatial unit on the *same corridors* found the fold-proximity variance gradient
**in space** (σ_rel rising toward the GL on the SNR≥10 segments). The two results are
mutually consistent under the §I.1 master curve: moving 4→40 km along-flow sweeps a
large range of `N`, while 12 years of ASE thinning moves `N` at a fixed point by only a
small fraction of its distance to `N_c` — the *temporal* λ-drift over 2014–2025 is far
smaller than the *spatial* λ-contrast the flowline samples. So the honest statement is a
**bound**: at ITS_LIVE quarterly precision (relative noise ~0.3–1 % near the GL), no ASE
trunk point moved measurably toward the fold in 2014–2025 — the registered *decadal*
early-warning would fire only on a stream much closer to ungrounding than today's
Thwaites/PIG trunk points (or on multi-decade records). The registered falsifier's
"observed to approach flotation" premise is only weakly satisfied on this window; the
§I.3 forecast survives, but now with a measured sensitivity floor attached.

## Honest limits

- 48 quarters × 12-window rolling stats: per-point trend power is modest (the planted-CSD
  unit proof fires at the population level, not per-realization — matching how the
  read-out is used).
- Phase surrogates on gap-masked series (gaps re-imposed) slightly deflate surrogate
  variance for very gappy points; MIN 32/48 quarters limits this.
- Quarterly medians average out any sub-90-day dynamics (stick-slip, tidal bands —
  §I.5's regime, not §I.3's).
- Sampling-density drift is flagged per point, not corrected beyond the SE² window floor.

## Data citations

ITS_LIVE v2 (Gardner et al., NASA MEaSUREs); Dakos et al. 2008 (rolling-EWS methodology);
Scheffer et al. 2009; Boers & Rypdal 2021 (PNAS, Greenland CSD — different observable);
Joughin, Smith & Schoof 2019 (RC sliding law).
