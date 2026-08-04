# §I.5 FIELD TEST — tidal velocity admittance on the real Filchner–Ronne GPS network

**Module** `glaciers/validation/external/tidal_admittance_field.py` ·
**cache** `glaciers/validation/external/data/tidal_admittance_field_cache.json` (committed, 64 KB; raw 283 MB of 30-s positions NOT committed — open licence, re-fetchable with `--fetch`) ·
**report** `glaciers/validation/reports/tidal_admittance_field.json` + `.png` ·
**tests** `glaciers/tests/test_tidal_admittance_field.py` (10)

## What was registered (§I.5)

Tides modulate grounding-zone effective pressure `N`; through the RC law the velocity
response carries (i) a **fundamental admittance** reading `|s_N|` and (ii) a **harmonic
fingerprint** `A2/A1=(ε/4)|s_N'/s_N−1|` — both **rising toward the GL** as `N→N_c`.
Registered field test: *"decompose high-cadence GPS/InSAR admittance + harmonics by N."*
This unit also closes the §I.6 gap: ice-plain/GZ physics needs GPS where ITS_LIVE cannot go.

## Real data

- **Gudmundsson, Fenney & Rosier 2017** (BAS PDC, OGL): 29 stations, 30-s Bernese PPP,
  five streams (Evans/XX, Foundation H, Institute IIS, Talutis/Carlson T/C, Rutford
  R145) + adjoining shelf sites; several filenames encode km-from-GL.
- **Smith, Murray & King 2020** (BAS PDC, OGL): the 750-day Rutford record (2004–2007).
- Distance-to-GL + grounded/floating from the committed MEaSUREs NSIDC-0709 boundary;
  each station also carries its own flotation coordinate = vertical semidiurnal
  admittance (grounded ≈ 0 m; shelf ≈ 1.2–1.4 m).

**Method.** Per station: EPSG:3031 → along-flow displacement + height → 5-min bins →
**joint harmonic LSQ across all ≥10-d segments** (per-segment mean+trend+quadratic
nuisance; the gaps *extend* the effective span, so multi-season stations pass the
183-d MSf/Mf Rayleigh limit), adaptive constituent set (O1 K1 [Q1] M2 S2 N2 MSf [Mf Mm]
M4 MS4). Unit-proofs plant gapped multi-season signals and recover MSf vs Mf.

**Methods cross-validation on the literature record:** the 744-d Rutford fit gives
**MSf/Mf = 4.8** — Murray et al. 2007 report "MSf about five times larger than Mf" on
the same data. ε_v(MSf) = 5.3 % at 21 km upstream matches the Gudmundsson 2006 scale.

## Results (17 grounded stations, 6 streams)

1. **Fundamental fortnightly admittance rises toward the GL — AS REGISTERED.**
   ε_v(MSf) = A_v(MSf)/v̄ vs distance: pooled Kendall **τ = −0.49, exact-permutation
   p = 0.007 (n = 17)**; within-stream concordance **23/27 pairs (sign-test
   p = 0.0002)** — Foundation is strictly monotone over 4 stations (3.6→47 km:
   4.0 → 1.7 %), Evans spans 6.0 % (8 km) → 0.9 % (35 km), Talutis 5.3 % (17 km) →
   1.7 % (34 km).
2. **The response is nonlinearity-generated (the mechanism behind the §I.5 harmonic
   face):** grounded stations are **fortnightly-dominated** — MSf displacement 40–230 mm
   vs M2 at 1–6 mm (>0.8 of significant grounded stations have MSf > M2; astronomical
   forcing would give Mf > MSf ≈ 12:1, observed MSf/Mf ≈ 4.8 → the fortnightly line is
   the M2×S2 *intermodulation product* of the sliding nonlinearity, Gudmundsson 2006).
3. **The literal 2f/1f curvature ratio is NOT field-resolvable here, for two mapped
   reasons:** (i) M4 (the true 2f of M2) sits at ≤ few mm ≈ the GPS/OTL noise floor on
   grounded ice; (ii) the natural surrogate MSf/M2 is confounded by **differential
   constituent screening** — M2 velocity response decays within ~10 km of the GL while
   MSf penetrates 20–100 km (R145 still shows 3.7 mm MSf at ~100 km) — so the ratio
   measures relative *penetration*, not local curvature. τ(MSf/M2 vs d) = −0.20,
   p = 0.31: not resolved.
4. The vertical-admittance ordering test has no resolving power *within* the grounded
   set (all h-admittances ≈ 0.00–0.02 m — every station is fully grounded; the
   coordinate only separates grounded from shelf).

## Honest scope

- The observed rise of ε_v(MSf) toward the GL is **consistent with the registered
  |s_N| steepening but not exclusive**: upstream *stress-transmission decay*
  (viscoelastic screening; Rosier & Gudmundsson 2015–2020) predicts the same ordering.
  Separating "forcing decays upstream" from "sensitivity rises near flotation" needs an
  independent ΔN(x) scale — which is also why the registered **tides-only (m, R)
  inversion is deferred**: ΔN/N is not directly measured by GPS alone.
- Positions lack OTL/IB corrections (metadata): grounded-station semidiurnal amplitudes
  (1–6 mm) are treated as a noise/OTL floor, not physics.
- Nodal corrections ignored (<4 % amplitude bias; irrelevant to ordering tests).
- n = 17 stations: exact-permutation p-values only.

## Data citations

Gudmundsson, Fenney & Rosier 2017 (doi:10.5285/4fe11286-0e53-4a03-854c-a79a44d1e356, OGL);
Smith, Murray & King 2020 (doi:10.5285/dac20505-a56e-4beb-97ba-077eecd587c0, OGL);
Gudmundsson 2006 (Nature), 2007, 2011; Murray et al. 2007; King et al. 2010;
Rosier & Gudmundsson 2015–2020; Minchew et al. 2017; MEaSUREs NSIDC-0709.
