# Reproduction log — independent re-check of the framework's claims (2026-06-14)

An end-to-end re-run of the repository's headline `[HYP]`/`[VERIFIED]` claims on a
fresh machine, with the external datasets downloaded from source and the GPU paths
exercised on a real CUDA device. Every number below was **recomputed**, not quoted,
and is compared against the value committed in the repo. Artifacts land in
`glaciers/validation/reports/` (checked in via `.gitignore` exceptions).

Environment: Python 3.12, numpy 2.4.6 / scipy 1.17.1 (CPU); Kaggle **Tesla
P100-PCIE-16GB**, cupy 14.0.1 (GPU). Full `pytest`: **354 passed**.

| Claim | Runner | Data (provenance) | Reproduced | Committed | Status |
|---|---|---|---|---|---|
| §V.1/§H.1 RTN directional | `run_rtn_bedmap2.py --stride 3 --phi 0.9` | BAS Bedmap2 (`secure.antarctica.ac.uk`, open) | RTN>1 median **6 km** vs **222 km** to GL; φ-robust **2.5/1.3/0.6 %** | 6 vs 222 km; 2.5/1.3/0.6 % | [PASS] exact |
| §V.2c §G.4 lake forcing | `run_usapdc_lakes.py` | USAP-DC **601439**+**601470** (reCAPTCHA, no login) + Bedmap2 | **58** events / 52 lakes; vol median **0.128**, max **2.916** km³ (Cook_E2); q_water med **2.5** m³/s | 58 / 52; 0.128 / 2.92; ~2.5 | [PASS] exact |
| §V.2 thermal kernel | `run_usapdc_lakes.py` (R2) | 601470 BedMachine H + Bedmap2 | τ_ice median **1.55×10⁵ yr** → 7.8×10⁴× too slow; BedMachine-vs-Bedmap2 **r=0.970** | 1.55×10⁵; r=0.97 | [PASS] FALSIFIED (as written) |
| §V.2b matched lag (open) | `lag_fit_real.py` | committed CryoSat-2/ITS_LIVE artifact | NULL: 0/5 events significant (peak +0.56σ); κ-kernel FALSIFIED | identical | [PASS] NULL; lag stays [DERIVED] |
| §G.4 hydraulic lag derivation | `hydraulic_lag_derivation.py` | none | t* median **0.012 yr**; **34 %** in 0.02–2 yr band; nonlinear peak 3.9–4.7 d | 0.012; 34 %; 3.9–4.7 d | [PASS] exact |
| Part 8b decisive closure | `run_closure.py` | none (synthetic DNS) | transfer corr Smag **0.071** / surrogate **0.907** / FDT **1.000**; FDT RMS(div) **1.7×10⁻¹⁴**; Smag T(k)≤0 ∀k; truth has backscatter | 0.071 / 0.907 / 1.000 | [PASS] exact |
| GPU CuPy family (pending item) | `gpu_family_sweep.py --gpu` | none | **13/13 PASS** on Tesla P100 (cupy 14.0.1), no device→host errors | skill: 13/13 on P100 | [PASS] exact |
| Part 9c 3D closure | `run_closure3d.py --gpu` | none (3D DNS, P100) | KE **0.0349**, ⟨ωSω⟩* **0.168**, div **1e-14**, FDT corr **1.000**, backscatter vol truth **0.48** / smag **0.00** | KE 0.035, 0.168, 1.05e-14 | [PASS] exact |
| RESULT 14 field test | `scallop_field_test.py` | none (published Bushuk scalars) | solver band **0.331–0.879**; harmonic I_rec **0.101**; Bushuk I_obs ~**0.10** (downstream, not falsified) | bit-identical Fig 58 | [PASS] exact |
| RESULT 11 counter-gradient | `theorem3_cg_gpu_probe.py` | none (3D LES, P100) | C_G(fl) **−0.598**, C_G(band) **−0.326**, departure **+0.674**, no melt hump (PARTIAL) | C_G −0.598, dep 0.674, no hump | [PASS] exact |
| §V.2 sliding + ITS_LIVE | `run_sliding_real.py --with-velocity Mac1` | Bedmap2 + **live ITS_LIVE S3** | 130 lakes τ_ice median **151,458 yr**; Mac1 **n=4505** 1987→2025, median **420 m/yr** | 151,458; n=4505, 420 | [PASS] exact |

---

## §V.1 RTN on real Bedmap2 — [VERIFIED directional], reproduced exactly

`python glaciers/validation/external/run_rtn_bedmap2.py --bin-dir <bedmap2_bin> --stride 3 --phi 0.9`

stride 3 km → 2223² grid, **1,329,743 grounded cells**. φ=0.9 RTN>1 fraction by
distance-to-GL: **15.1 / 6.0 / 3.5 / 2.1 / 0.8 / 0 / 0 %** over 0–5…>250 km;
median distance **6 km (RTN>1)** vs **221.8 km (RTN<1)**. Overall RTN>1 fraction
**2.5 / 1.3 / 0.6 %** for φ = 0.8 / 0.9 / 0.95. Matches `REAL_DATA_RESULTS.md §V.1`.
Artifact: `reports/rtn_bedmap2.json` + `.png`.

## §V.2c USAP-DC vetted catalogues — forcing obtained; kernel [FALSIFIED]

Data downloaded fresh from USAP-DC (a reCAPTCHA bot-check, **no login**):
**601439** (Smith 2009 ICESat, 124 `Volume_history.csv`) and **601470**
(Stubblefield 2021 `active_lake_statistics.dat`, 131 lakes, BedMachine H).

- **R1 forcing:** 58 fill→drain events over 52/124 lakes; drained volume median
  0.128 km³, max 2.916 km³ (Cook_E2 > Recovery_7 > Whillans_1 > Byrd_2);
  q_water median ~2.5 m³/s, p95 ~31 m³/s (literature subglacial-flood range).
- **R2 kernel:** τ_ice = H²/κ median 1.55×10⁵ yr → 7.8×10⁴× slower than the
  observed 0.02–2 yr surge band ⇒ **[FALSIFIED as written]**; the BedMachine
  (601470) vs Bedmap2 thickness at the same 131 centroids agrees at **r=0.970**,
  so the falsification is robust to the thickness source.

Honest scope unchanged: the *forcing* half is now obtained; the lag *value* is
**not** promoted to [VERIFIED] (still needs co-temporal sub-annual velocity).
Artifacts: `reports/usapdc_lakes.{json,png}` + `usapdc_lakes_events.csv` (58-event
catalogue). Matches `REAL_DATA_RESULTS.md §V.2c`.

## GPU CuPy family — 13/13 PASS on Tesla P100

`python glaciers/gpu_family_sweep.py --gpu` on Kaggle P100 (cupy 14.0.1): all 13
backend-agnostic solvers (`scallop_*` family, `scallop3d_probe`, slip/wall_flux
gates, `candidate4` anchor, `scallop_battery`) return rc=0 with **no**
`Implicit conversion to a NumPy array` device→host `TypeError` — the GPU-only bug
class the CPU `pytest` suite structurally cannot catch. Closes the "GPU re-runs of
the full CuPy family" pending item. Artifact: `reports/gpu_family_sweep_p100.json`.

## Part 9c — 3D closure benchmark on P100, reproduced

`run_closure3d.py --gpu --n 128 --kc 24 --nu 0.0012 --steps 2500` on Tesla P100
(cupy 14.0.1). Developed KE **0.0349** (committed 0.0349). All Part-9b verdicts
hold in 3D **plus** the 3D-only physics:

- forward cascade: net Σₖ T_true = −4.52×10⁸ (<0); div truth **1.0×10⁻¹⁴** vs
  surrogate **9.9** (Leray solenoidality);
- transfer corr with truth: **FDT 1.000**, Smagorinsky **−0.641** (purely
  dissipative, T(k)≤0 ∀k), surrogate 0.268;
- vortex-stretching production **⟨ωSω⟩\* = 0.168** (committed 0.168; identically 0 in 2D);
- strain–vorticity alignment |cos|: e1 0.479, **e2(intermediate) 0.654**, e3 0.325
  — the intermediate eigenvector preferentially aligns, the canonical 3D signature;
- backscatter volume fraction: **truth 0.480 vs Smagorinsky 0.000** — Smagorinsky
  structurally cannot represent the ~half-volume up-scale transfer.

Matches `REPORT_CLOSURE3D.md` (committed P100 run, commit f0e63fd). Artifact:
`reports/closure3d_p100.json`.

## RESULT 14 — scallop field test (§G.2), reproduced exactly

`python glaciers/scallop_field_test.py` (CPU; needs no external data — the Bushuk
et al. 2019 *adjustment*-regime estimate uses only published scalar kinematics
`c=1.833e-6, λ=0.13`). Regenerates `figures/58_scallop_field_test.json`
**bit-identically**: solver band (0.331, 0.879); harmonic-pin self-check
I_rec 0.101 vs I_true 0.102 (recovered=True, downstream=True); Bushuk exp-1b bound
I_obs (0.051, 0.152), point ~0.10, sign downstream, `falsified=False`. The
constant-free field test `I = τ·c_mig/λ` recovers the known mode and is the same
order as the solver band — RESULT 14 stands. (A true *pin* still needs Bushuk's
raw h(x,t) arrays; `harmonic_mode_rate()` is ready to ingest them.)

## RESULT 11 — counter-gradient C_G on a 3D buoyant LES (P100), reproduced

`python glaciers/theorem3_cg_gpu_probe.py` on Tesla P100 (cupy, n=64, nseeds=3,
6-point Ri sweep, wall 532 s). Two-clocks **C_G(fluid) = −0.598**, **C_G(band) =
−0.326** (Smagorinsky −0.604 / −0.321; R(2c/sm)=1.0004). The counter-gradient
departure **max(C_G+1) = +0.674**: heat in the lee band is carried *counter*-gradient
(C_G > −1), which a down-gradient eddy diffusivity (C_G = −1 by construction)
structurally cannot produce. But it does **not** co-locate with a basal melt hump
(peak R 1.0004), so the verdict is the honest **PARTIAL**, matching committed Fig 52.
Artifact: `reports/theorem3_cg_p100.json`.

## §V.2 sliding-law on real Bedmap2 + live ITS_LIVE velocity

`python glaciers/validation/external/run_sliding_real.py --with-velocity Mac1`.
τ_ice = H²/κ on real Bedmap2 at the 131 Siegfried & Fricker lakes (130 with valid
thickness): median H **2282 m**, τ_ice median **151,458 yr** (p5 22,678; p95 322,790)
— ~8×10⁴× too slow than the 0.02–2 yr surge band, **kernel FALSIFIED as written**.
Live **ITS_LIVE** datacube pull (anon S3, zarr) at lake Mac1: **n=4505** image-pair
speeds, 1987.0→2025.2, median **420 m/yr** (109–788); `estimate_lag` self-lag 0.00.
Confirms the §V.2 velocity-response side is real + openly accessible (only the
drainage-*date* catalogue was gated — now obtained via USAP-DC, §V.2c). Matches
`REAL_DATA_RESULTS.md §V.2`.

## What was NOT re-derived here

- Auth-gated data (BedMachine v3 via Earthdata, CATS2008 tides) remains
  unreachable without credentials; those loaders still raise `DataUnavailableError`.
- The §G.4 lag **value** stays `[DERIVED]` (matched test returns a NULL on open
  satellite velocity); the thermal `H²/κ` kernel stays `[FALSIFIED]`.
- All other probes (ocean Candidate 2, atmosphere swirl, scallop family, Stefan,
  CMN solver, RTN corollaries) are exercised by the **354-passing** `pytest` suite.

---

## 2026-06-15 addendum — Earthdata provisioned; BedMachine v4 §V.1 cross-check closed

The one open auth-gated item from the §V.1 column — *independent-dataset* RTN on
NSIDC BedMachine — is now closed. A free NASA Earthdata login (`earthaccess`,
`short_name='NSIDC-0756'`) fetched **BedMachine Antarctica v4**
(`BedMachineAntarctica_…_V04.1.nc`, 1.2 GB, 13333² @ 500 m). Running the
independent §V.1 RTN-vs-distance test (`run_rtn_bedmachine.py`, stride 4 → **2 km**,
*finer* than the 3 km Bedmap2 run; 2,999,854 grounded cells, `mask==2`):

| Claim | Runner | Reproduced (BedMachine v4) | Bedmap2 (committed) | Status |
|---|---|---|---|---|
| §V.1 RTN directional | `run_rtn_bedmachine.py --stride 4 --phi 0.9` | median **6 km** (RTN>1) vs **200 km** (RTN<1); overall RTN>1 **2.5 / 1.4 / 0.8 %** for φ=0.8/0.9/0.95; monotone GL decay 14.0→0 % | 6 vs 222 km; 2.5 / 1.3 / 0.6 % | [PASS] reproduces on an independent inversion |

BedMachine (NSIDC, mass-conserving) and Bedmap2 (BAS, kriged) are independent
thickness products, so this is a true robustness cross-check. Two enabling changes
landed with it: `bedmachine_loader.load_fields` now reads the **strided slice
directly** from the NetCDF (was: materialise the full 1.4 GB/field grid then
decimate — OOMs a small-RAM box), so the native grid runs CPU-only at stride ≥ 4;
and `run_rtn_bedmachine.py` grew a `--json` emitter mirroring `rtn_bedmap2.json`.
Artifacts checked in: `reports/rtn_bedmachine.{json,png}`. The 10 BedMachine loader
/ run tests still pass.

**CATS2008 tides — also closed (no gated §V data items remain).** The other
auth-gated item is done: `CATS2008.zip` (582 MB, MD5 `008a30cd…`) fetched from
USAP-DC (reCAPTCHA bot-check, no login); `tide_loader.load_tide` now predicts real
tidal elevation via `pyTMD` (verified: Filchner-Ronne cavity range **2.70 m**,
Ross **1.65 m** — the known circum-Antarctic pattern); and `tidal_forcing_gz.py`
measures the §I.5 forcing at the *real* grounding zone (CATS2008 × BedMachine):
tidal ocean-pressure swing median **0.30 % of ice overburden** (a hard lower bound
on the admittance forcing `ε`, since `N ≤ p_i`), **growing 0.21 % → 0.45 % toward
the GL** over 1,108 grounding-zone cells — the toward-GL growth §I.5 predicts.
Artifacts: `reports/tidal_forcing_gz.{json,png}`; 5 tide-loader tests pass.
