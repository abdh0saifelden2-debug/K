# §I.6 FIELD TEST — the single-snapshot spatial early-warning on real Antarctic data

**Module** `glaciers/validation/external/spatial_ews_field.py` ·
**cache** `glaciers/validation/external/data/spatial_ews_field_cache.json` (committed; replays offline) ·
**report** `glaciers/validation/reports/spatial_ews_field.json` + `.png` ·
**tests** `glaciers/tests/test_spatial_ews_field.py` (15)

## What was registered

§I.6 (synthetic `spatial_ews.py`, exact Lyapunov solution) derived that near the
regularized-Coulomb flotation fold the restoring stiffness `λ(N) ∝ (1−R)²/R → 0`, so a
longitudinally-coupled stochastic velocity field has `Var ∝ 1/√(Dλ)` and correlation
length `ξ ∝ √(D/λ)` — both **rising toward the grounding line**. Registered field test:
*"bin ITS_LIVE speed variance + along-flow correlation by distance-to-GL."*

## Real data used (no synthetic anywhere in the result)

| ingredient | source |
|---|---|
| velocity fluctuations | ITS_LIVE **v2 datacubes**, full image-pair series (~17k pairs/point on the ASE trunks, 2013–2025), pairs with baseline ≤ 120 d |
| flow directions / corridors | ITS_LIVE v2 **static mosaic** (120 m), remote byte-range subsets (no 21 GB download) |
| grounding line | **MEaSUREs NSIDC-0498-derived NSIDC-0709** InSAR grounding line (shapefile); the product ships the grounded sheet as polygons — the GL is its boundary |
| stream identity | NSIDC-0709 IceBoundaries polygons (`TYPE == GR`) |

Flowlines are seeded at the fastest grounded trunk pixel 4–15 km upstream of the GL and
integrated **upstream** (RK2 on bilinear vx,vy), sampled every 2 km:
Thwaites (121 pts, 4–226 km), Pine Island (121 pts, 4–142 km), Smith (52 pts, 4–99 km),
control Rutford (82 pts, 14–143 km).

**Estimator.** Per point: calendar-year median speeds (≥8 pairs/yr, ≥8 yr) with
MAD/√n standard errors → detrend → residual variance **minus the noise floor**
(mean SE²) → `σ_rel = √max(Var−noise,0)/v̄`. `ξ(s)` from pooled pair-correlations of the
relative residual profiles in ±20 km windows (1/e crossing). Kendall τ vs distance with a
**circular-shift null** (≥25 km shifts, support enumerated **exhaustively** — the p-floor
1/(K+1) is explicit; naive iid permutation would over-reject on autocorrelated profiles,
and a randomly-sampled shift null can fake p=1/2001 on a 7-member support).

## Results

1. **Raw full-corridor gradients are NOT interpretable.** On all four streams σ_rel
   *falls* toward the GL (τ = +0.40…+0.71, shift p ≈ 0.01–0.03) — but the **noise-control
   column** (τ of the SE-based noise floor vs distance) shows the identical gradient
   (τ ≈ +0.78…+0.82): slow upstream ice has large *relative* errors, and the MAD/√n
   correction under-removes correlated-pair noise. The raw gradient is the noise
   gradient. (This is why the registered test as literally written — "bin ITS_LIVE speed
   variance by distance" — would mislead in either direction without the noise column.)
2. **Within-stream signal where the measurement is real.** Restricting to points with
   `Var_raw ≥ 10× noise` (the fast trunks; selection floor *falls* toward the GL, i.e.
   the cut biases *against* the prediction) and pooling the near-flotation trunks
   (Thwaites n=19, Pine Island n=25; Smith has only 4 such points):
   **σ_rel rises toward the GL** — weighted Kendall **τ = −0.337**, one-sided exhaustive
   shift-null **p = 0.0123** (80-combination support; the observed value beats *every*
   null combination). **Quadratic-detrend robustness** (the trunks' speed-up is curved;
   smooth acceleration-rate change must not count as fluctuation): **τ = −0.398,
   p = 0.0196 — survives.** The grounded control has the **opposite sign**
   (Rutford τ_snr10 = +0.58; its own segment is support-limited, descriptive).
3. **Correlation-length face NOT detected.** ξ ≈ 10–18 km with no distance gradient on
   any stream (|τ| ≤ 0.16, p ≥ 0.43) — even though upstream noise *biases ξ down*, i.e.
   toward faking the predicted rise. At 2 km sampling / ±20 km windows and n≈12 annual
   anomalies, the ξ estimator has no power beyond ~20 km; the derived ξ divergence at the
   fold stays untested at ITS_LIVE precision.
4. **Levels (a real measurement either way).** Within 30 km of the GL the detrended
   interannual relative fluctuation is 1.5 % (Thwaites, SNR≈21), 1.6 % (Pine Island,
   SNR≈30), 2.0 % (Smith), vs 3.1 % on slow Rutford (390 m/yr; plausibly aliased tidal
   modulation) — ~30–41 m/yr absolute on the ASE trunks. **No cross-stream
   fold-proximity ordering in σ_rel** — the within-stream gradient, not the absolute
   level, carries the §I.6 signature.

**Verdict (both faces, honestly):** the **variance face is supported on the measurable
segments** — the two near-flotation trunks show a genuine, detrend-robust rise of relative
speed fluctuation toward the GL that the grounded control does not show — while the
**correlation-length face is not detected**, and the naive full-corridor version of the
registered test is noise-confounded in the *opposite* direction. This is a *partial*
confirmation with the confound mapped, not a clean sweep.

## Honest limits

- **Whillans ice plain** — the §I ice-plain target nearest flotation — is **not testable
  with ITS_LIVE**: it sits in the optical polar hole (corridor mosaic 25 % finite; the
  datacube holds 15 usable pairs total, 2016–2022, vs the ≥8/yr × ≥8 yr floor). Ice-plain
  §I.6/§I.5 testing needs InSAR/GPS (queued: §I.5 tidal admittance on USAP-DC GPS).
  Smith replaced it (392 pairs).
- Corridors start ≥4 km upstream of the GL (mosaic GL-zone/floating mask); a fold zone
  narrower than ~4 km would be invisible here.
- The pooled p sits at its explicit exhaustive floor (0.0123/0.0196): stronger
  certification needs more independent trunks (Smith-class streams with denser pair
  coverage) rather than more surrogates.
- σ_rel is the scale-invariant reading of the registered "speed variance" (the mean-speed
  scale varies ~30× along the corridors; absolute variance would trivially fake the
  signal on every stream including the control).
- n≈12 annual medians per point: AC1 stored in the cache but underpowered for the §I.3
  temporal face (separate unit, subannual medians).

## Data citations

Gardner, A. S. et al. (ITS_LIVE v2 velocity mosaics & datacubes, NASA MEaSUREs);
Rignot/Mouginot/Scheuchl NSIDC-0709 Antarctic Boundaries (grounding line from DInSAR);
Dakos et al. 2010 (spatial EWS); Schoof 2007 (MISI); Joughin, Smith & Schoof 2019 (RC law).
