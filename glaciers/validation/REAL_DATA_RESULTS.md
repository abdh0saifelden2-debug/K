# §V real-data validation — results on open public datasets

This file records the **real-data** runs of the §V external-validation pipeline,
using only datasets that are openly reachable (no NSIDC/Earthdata login). It
backs the falsifiable-forecast section `FUTURE_WORK.md §H`. The synthetic
harnesses (`validation/synthetic/`, exercised in `pytest`) prove the math; the
runs here expose the §G **[HYP]** physics to falsification on real geometry.

All numbers below were produced by the two runner scripts; figures land in the
git-ignored `validation/reports/`.

## Data sources (all open, no auth)

| Dataset | What | Source | Local path expected by runners |
|---|---|---|---|
| BAS **Bedmap2** | ice `thickness`, `bed`, `surface`, grounded/shelf `icemask` (1 km, 6667²) | `https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_bin.zip` (~222 MB) | `/home/ubuntu/data_bedmap/bedmap2_bin/` |
| **Siegfried & Fricker (2018)** active-lake outlines | 131 named lake polygons (x/y/lon/lat, EPSG:3031); cites Smith et al. 2009 | GitHub `mrsiegfried/Siegfried2021-GRL` (`data/outlines/SiegfriedFricker2018-outlines.h5`, 954 KB) | `/home/ubuntu/data_lakes/SiegfriedFricker2018-outlines.h5` |
| **ITS_LIVE** velocity datacubes | surface-speed time series (image-pair `v`), anonymous S3 | `s3://its-live-data/datacubes/...` via `catalog_v02.json` | (streamed; needs `xarray`+`zarr`+`s3fs`) |

What needs a (free) NASA Earthdata login: **BedMachine Antarctica** (NSIDC-0756),
**now provisioned** via `earthaccess` (`short_name='NSIDC-0756'`) and driving the
§V.1 RTN cross-check below. The **CATS2008** tide model (USAP-DC
doi:10.15784/601235) is gated by a reCAPTCHA bot-check (no login) — also **now
obtained**, wired into `tide_loader` with `pyTMD`, and driving the grounding-zone
tidal-forcing probe (`tidal_forcing_gz.py`, below). The vetted lake
**volume-change / drainage-date** series was treated as USAP-DC-gated but is in fact
reachable (bot-check, no login; §V.2c). §V.2d additionally uses **ICESat-2 ATL15**
(NSIDC, Earthdata-authenticated via `earthaccess`) for modern drainage dates. The
open-pipeline runners (§V.1/§V.2/§V.2b) still raise `DataUnavailableError` with
provisioning hints rather than fabricate anything.

## Reproduce

```bash
pip install -r requirements.txt xarray zarr s3fs # xarray/zarr/s3fs only needed for ITS_LIVE
# Bedmap2
curl -L -o bedmap2_bin.zip https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_bin.zip
unzip bedmap2_bin.zip -d /home/ubuntu/data_bedmap
# Lake outlines
curl -L -o /home/ubuntu/data_lakes/SiegfriedFricker2018-outlines.h5 \
  https://raw.githubusercontent.com/mrsiegfried/Siegfried2021-GRL/main/data/outlines/SiegfriedFricker2018-outlines.h5

python validation/external/run_rtn_bedmap2.py  --stride 3 --phi 0.9   # §V.1 / §H.1
# §V.1 independent cross-check on NSIDC BedMachine v4 (needs a free Earthdata login:
#   python -c "import earthaccess; earthaccess.login()"; earthaccess.download(
#   earthaccess.search_data(short_name='NSIDC-0756'), '.') -> BedMachineAntarctica_*_V04.1.nc):
python validation/external/run_rtn_bedmachine.py --path <V04.1.nc> --stride 4 --phi 0.9 \
  --json reports/rtn_bedmachine.json
python validation/external/run_sliding_real.py --with-velocity Mac1   # §V.2 / §H.2
python -m validation.external.lag_fit_real                            # §V.2b matched-data lag test
```

The §V.2b matched-data test is deterministic and offline: it reads the small
committed artifact `validation/external/data/lake_lag_matched.json` (drainage
dates + annual-binned velocity + bootstrap CIs). Regenerating that artifact from
the live open sources needs outbound network + `zarr`/`s3fs` (see §V.2b).

---

## §V.1 RTN on real Bedmap2 — [VERIFIED directional]

RTN is built from real geometry on grounded ice (icemask==0), 1 km decimated to
3 km (1,329,743 grounded cells):

- overburden `p_i = ρ_i g H`
- ocean head at the bed `p_ocean = ρ_w g · d_base`, `d_base = max(0, −bed)`
- subglacial water `p_w = φ · p_i` (so `N_eff = (1−φ) p_i`)

**Result.** `RTN>1` fraction of grounded ice, binned by distance to the nearest
grounding line, for `φ = 0.9`:

| dist-to-GL [km] | 0–5 | 5–10 | 10–25 | 25–50 | 50–100 | 100–250 | >250 |
|---|---|---|---|---|---|---|---|
| RTN>1 fraction | 15.1% | 6.0% | 3.6% | 2.1% | 0.8% | 0.0% | 0.0% |
| n cells | 50532 | 52094 | 87318 | 99425 | 143492 | 285743 | 611139 |

- Median distance-to-GL: **`RTN>1` cells = 6 km** vs **`RTN<1` cells = 222 km**.
- Robust across `φ`: overall `RTN>1` fraction = **2.5% / 1.3% / 0.6%** for
  `φ = 0.8 / 0.9 / 0.95` — only the magnitude scales; the monotone GL decay holds.

**Gauge fix sharpened the signal.** These numbers use the corrected gauge RTN
(`p_atm = 0`; see §G.3). Re-running the *old* `−p_atm` convention on the identical
grid gives the pre-fix table (`12.2 / 5.2 / 3.2 / 1.7 / 0.6 %`, overall 1.07%) — i.e.
the bug *understated* the grounding-line concentration. The fix flips **2,919**
grounded cells into `RTN>1`, and they cluster at **median 4.2 km from the GL**: the
spurious offset `p_atm/(ρ_w g d_base)` was largest exactly where `d_base → 0` (the
grounding line), so removing it adds intrusion-favourable cells preferentially there.
The directional result is unchanged but stronger; the original §H.1 conclusion was
conservative.

`RTN>1` concentrates sharply at the grounding line and anticorrelates with ice
thickness inland — the directional §G.3 prediction, on real geometry.

**Caveats (carried from §G.3).** Not a precision/recall score — no gridded
intrusion survey exists. The 1 km posting cannot resolve the ~1–10 m subglacial
channel the Röthlisberger `N_R` term needs, so channel size is absorbed into `φ`
rather than resolved. The earlier numerator/denominator gauge mismatch (`p_atm`)
is now **resolved** — RTN is the gauge ratio in which the atmosphere cancels (§G.3).

Figure: `validation/reports/rtn_bedmap2.png` (left: grounded ice with `RTN>1` in
red, hugging the coastal grounding zones; right: the monotone GL-distance bars).

### §V.1 cross-check on independent NSIDC BedMachine v4 — [VERIFIED directional], reproduces Bedmap2

The same directional test on a *completely independent* thickness/bed product —
**BedMachine Antarctica v4** (NSIDC-0756, Morlighem et al. 2020; 500 m native,
mass-conserving inversion; fetched via `earthaccess`) — reproduces the Bedmap2
result. Run on grounded ice (BedMachine `mask==2`, **2,999,854 cells**) decimated
500 m → **2 km** (stride 4, *finer* than the 3 km Bedmap2 run), for `φ = 0.9`:

| dist-to-GL [km] | 0–5 | 5–10 | 10–25 | 25–50 | 50–100 | 100–250 | >250 |
|---|---|---|---|---|---|---|---|
| RTN>1 fraction | 14.0% | 5.3% | 3.6% | 2.4% | 1.0% | 0.1% | 0.0% |
| n cells | 138817 | 97703 | 215918 | 235841 | 343865 | 700007 | 1267703 |

- Median distance-to-GL: **`RTN>1` = 6 km** vs **`RTN<1` = 200 km** (Bedmap2: 6 vs 222 km).
- Across `φ`: overall `RTN>1` fraction **2.5% / 1.4% / 0.8%** for `φ = 0.8 / 0.9 / 0.95`
  (Bedmap2: 2.5% / 1.3% / 0.6%) — the same monotone grounding-line decay, on an
  independent dataset, at finer resolution.

BedMachine (NSIDC, mass-conserving) and Bedmap2 (BAS, kriged) are independent
thickness inversions, so this is a genuine robustness cross-check rather than a
re-run of the same grid — yet the §G.3/§H.1 directional prediction holds on both.
The loader reads the strided slice **directly** from the NetCDF, so the native
13333² grid runs on a small-RAM CPU box (no GPU needed at stride ≥ 4).

Provenance: `earthaccess.search_data(short_name='NSIDC-0756')` →
`BedMachineAntarctica_…_V04.1.nc`. Reproduce:
`python validation/external/run_rtn_bedmachine.py --path <nc> --stride 4 --phi 0.9
--json reports/rtn_bedmachine.json`. Artifacts:
`validation/reports/rtn_bedmachine.{json,png}`.

---

## §I.5-real tidal forcing at the grounding zone — CATS2008 × BedMachine

The §I.5 tidal-admittance probe (`tidal_admittance_probe.py`) is analytic; it needs
a *real* tidal forcing amplitude `ε` (the fractional modulation of basal effective
pressure `N`). `tidal_forcing_gz.py` **measures** it by combining two independent
real datasets — **CATS2008** tidal elevation (`pyTMD`) and **BedMachine v4** ice
overburden — sampled at the *actual* grounding zone.

Grounded BedMachine cells within 10 km of the grounding line (2 km grid; 1,108 of
254,055 with valid CATS2008 tides), 30-day hourly prediction:

- **Tidal amplitude** `η_amp` (half peak-to-peak): median **0.84 m** (p5 0.51,
  p95 2.61). Verification points: Filchner-Ronne cavity range **2.70 m**, Ross
  **1.65 m** — the known circum-Antarctic pattern.
- **Tidal pressure swing / ice overburden** `Δp/p_i = ρ_w g η_amp /(ρ_i g H)`:
  median **0.30 %** (p95 49.7 %, the thin-ice flotation tail). Because `N ≤ p_i`,
  this is a hard **lower bound on the §I.5 forcing `ε`**, so the admittance
  `A1 ≈ |s_N|·ε` rests on a *measured* floor, not an assumption.
- It **grows toward the grounding line**: median `Δp/p_i` rises **0.21 % → 0.45 %**
  from 8–10 km to 2–4 km of the GL — the same toward-GL growth of the nonlinear
  tidal response (Minchew et al. 2017) that §I.5 reads as `s_N(N)` steepening
  toward flotation.
- Connectivity-limited estimate `ε_flot = Δp/(p_i − ρ_w g·d_base)` (assumes the
  ocean-pressure tide reaches the bed): median **0.008**, p90 **0.149** — `[HYP]`,
  diverging as `N → 0`.

Provenance: `CATS2008.zip` from USAP-DC (reCAPTCHA, no login; MD5 `008a30cd…`),
unzipped to `<dir>/CATS2008/`; `pip install pyTMD dask pyproj`. Reproduce:
`python validation/external/tidal_forcing_gz.py --bedmachine <V04.1.nc> --tide-dir
<dir> --stride 4 --json reports/tidal_forcing_gz.json`. Artifacts:
`validation/reports/tidal_forcing_gz.{json,png}`.

---

## §V.2 sliding law on real lake thicknesses — kernel [FALSIFIED as written]

The matched lag test needs drainage **dates** (gated). What open data *can*
settle is the literal §G.4 memory timescale `τ_ice = H²/κ_ice`
(`κ_ice = 1.09e-6 m² s⁻¹`), evaluated on real Bedmap2 thickness at the 131
catalogued lakes (130 with valid thickness):

- ice thickness at lakes: **median `H = 2282 m`** (min 637, max 3905).
- `τ_ice = H²/κ_ice`: **median ≈ 151,458 yr** (p5 = 22,678; p95 = 322,790 yr).
- observed post-drainage surge lags: **0.02–2 yr** (Stearns et al. 2008;
  Siegfried et al. 2016; literature).
- ⇒ literal kernel is **~8×10⁴× too slow** at the median lake. The `log₁₀ τ_ice`
  histogram (10⁴–10⁶ yr) is fully disjoint from the observed surge-lag band.

This **falsifies the §G.4 kernel as literally written** on real geometry and
makes the §B.2 caveat quantitative: the full-thickness diffusive timescale is
the wrong scale. The mismatch is *diagnostic* — the thermal skin depth at
observed surge periods is only `√(κ_ice P/π) ≈ 0.5–5 m`, and a drainage *impulse*
gives a monotone `t^{−1/2}` response with **no peak**, whereas observations peak
at 0.02–2 yr. Both exclude ice thermal diffusion as the lag mechanism: the lag is
**hydromechanical** (subglacial hydrology / effective-pressure control), with the
thermal term a weak multi-decadal tail. A boundary-layer or water-inertia kernel
*tuned* to the observed lag would be fitting, not deriving (see FUTURE_WORK.md
§G.4 "corrected reading" and §H.2). Only an **empirical** lag is interpretable.

**Response data is real and accessible.** The ITS_LIVE datacube over lake
**Mac1** (MacAyeal) yields a genuine surface-speed series:

- **n = 4505 image-pair measurements**, spanning **1987.0 → 2025.2**,
  median **420 m/yr** (min 109, max 788).
- `validators.sliding_validator.estimate_lag` runs on it directly (self-lag ≈ 0,
  as expected for an identical pair) — the §V.2 lag machinery is functional on
  real satellite observations.

Both the thickness (forcing-side geometry) and velocity (response-side) halves
are real and open. **Update (§V.2b below):** the drainage-*date* half is in fact
open too — vetted *volume-change* catalogues are USAP-DC-gated, but the CryoSat-2
*in-lake elevation* series resolve the drainage dates directly — so the
end-to-end matched lag test **is** runnable on open data. It returns an honest
null (no surge), which keeps the lag value `[DERIVED]`.

Figure: `validation/reports/sliding_tau_ice_real.png` (histogram of `log₁₀ τ_ice`
vs the observed surge-lag band).

---

## §V.2b matched-data lag test on OPEN data — [DERIVED retained; honest null]

`lag_fit_real.py` runs the §G.4/§H.2 matched lag test end-to-end using **only**
open data, superseding the §V.2 remark that the test was un-runnable. Both halves
are open:

- **Drainage *dates* (forcing).** The earlier "USAP-DC-gated" blocker applies to
  the vetted *volume-change* catalogue; the drainage *dates* themselves drop out
  of the open CryoSat-2 *in-lake mean-elevation* series (Siegfried 2021-GRL
  mirror, monthly 2010.6–2020.5). `detect_drainages` extracts **7 fill→drain
  events across 5 lakes**, including the documented **Mercer ≈2012.5** drainage
  (peak 2012.54, drop 5.98 m).
- **Velocity *response*.** ITS_LIVE box-mean surface speed at each lake centroid.
  Three MacAyeal lakes have a usable series; Mercer/Conway return ~0–1 finite
  samples (slow ice plain).

| lake | base speed (m/yr) | drainages | annual noise floor | quarterly scatter | peak post-drainage |
|---|---|---|---|---|---|
| Mac1 | 422 | 3 | 1.0% | 2.3% | **+0.56σ** |
| Mac2 | 402 | 1 | 1.1% | 2.1% | +0.27σ |
| Mac3 | 394 | 1 | 1.1% | 3.4% | −0.12σ |

**Result — honest null.** No matched lake shows a post-drainage surge above the
2σ noise floor: peak anomaly **+0.56σ**, **0/5** events significant. The lag
**value stays `[DERIVED]`** (it is *not* promoted to `[VERIFIED]` here).

**Why the lag is unverifiable from satellite velocity — and why that's a *stronger*
statement than "too noisy".** With the dense modern ITS_LIVE catalogue the record
is **well resolved** (annual ≈1%, quarterly ≈2–3%), so the limit is *not* a noise
budget. Two distinct, quantifiable limits remain:

1. **Temporal aliasing (a *sampling*-resolution limit).** The derived lag is
   sub-annual — `t*` baseline ≈0.01 yr (~4 days), p95 ≈0.1 yr (~5 weeks) — finer
   than the finest *robust* ITS_LIVE bin (**quarterly, 0.25 yr**). A brief
   post-drainage transient is averaged out by the velocity estimator's
   integration window *regardless of how small the scatter is* — a Nyquist limit.
2. **Amplitude upper bound (the empirical null).** A *sustained* speedup would
   survive aliasing as a step in the time-integrated series, yet the peak anomaly
   is only +0.56σ — *bounding* any sustained surge to a few percent (≲10 m/yr on
   ~400 m/yr), a real constraint rather than a "can't tell".

These two limits map onto the two routes to `[VERIFIED]`: **sub-annual GPS/GNSS**
(EarthScope/POLENET; sub-daily cadence beats the aliasing) **or fast-outlet
*trunk* velocity** (e.g. Byrd, Stearns et al. 2008; steep `v(N)` sensitivity
beats the amplitude floor), paired with that site's drainage dates. The thermal
`H²/κ` kernel remains **[FALSIFIED]** (§V.2) regardless.

**Regeneration.** `run()` is deterministic from the committed artifact
`validation/external/data/lake_lag_matched.json`. To rebuild it from the live
open sources (CryoSat-2 elevation → drainage dates via `detect_drainages`;
ITS_LIVE datacubes → annual/quarterly box-mean velocity with 16/84 bootstrap CIs;
`κ_ice` thermal `τ` for the falsification cross-check), install `zarr`/`s3fs` and
pull the per-lake `*_elevs.dat` (Siegfried `data/cs2/proc_out/`) and the ITS_LIVE
cubes at each lake centroid — `build_artifact_live` documents the exact steps and
source URLs. The ~7 KB JSON is committed (with a `.gitignore` exception) so CI
needs no network.

---

## §V.2c USAP-DC vetted catalogues — forcing obtained; kernel [FALSIFIED] confirmed

§V.2 and §V.2b both flag the same residual gap: the *vetted* lake
**volume-change** catalogue (the §G.4 forcing `q_water` — drainage **dates and
volumes**) was treated as USAP-DC-login-gated, so the forcing side was only
*approximated* (CryoSat-2 elevation proxies for 5 lakes). The USAP-DC datasets
are in fact open (a bot-check, not a login), so `run_usapdc_lakes.py` now uses
the real catalogues directly.

**R1 — real §G.4 drainage forcing (USAP-DC 601439, Smith et al. 2009 ICESat).**
Parsing the per-lake `Volume_history.csv` series (124 lakes, 2003–2009) and
extracting discrete fill→drain events (peak-to-trough, ≥0.05 km³) yields **58
drainage events across 52 lakes** — the vetted forcing the earlier runners could
not obtain:

- drained volume per event: **median 0.128 km³**, p95 ≈ 0.99 km³, **max 2.92 km³**
  (Cook_E2); largest events at Cook_E2, Recovery, Byrd, Whillans, Totten, Lambert.
- implied water flux `q_water = ΔV / Δt`: **median ≈ 2.5 m³/s**, p95 ≈ 31 m³/s —
  squarely in the literature subglacial-flood range O(10–100) m³/s, so the §G.4
  forcing magnitude is now anchored to real observations rather than synthetic.

The event catalogue is written to `validation/reports/usapdc_lakes_events.csv`
with a `t` column, directly consumable by `lake_catalog_loader.load_drainage_events`.

**R2 — thermal kernel on independent BedMachine thickness (USAP-DC 601470).**
Recomputing `τ_ice = H²/κ_ice` on the 131 lakes' **BedMachine** thickness (an
independent source to the Bedmap2 sampling in §V.2):

- thickness median **2356 m**; `τ_ice` **median ≈ 1.55×10⁵ yr** (p5 2.6×10⁴,
  p95 3.4×10⁵) — **~7.8×10⁴× slower** than the observed 0.02–2 yr surge band.
- **independently reproduces §V.2's Bedmap2 result** (1.51×10⁵ yr): the
  BedMachine-vs-Bedmap2 thickness at the same centroids agrees at **r = 0.970**,
  so the falsification is robust to the thickness dataset, not an artifact of one.

**Honest scope.** This promotes the **forcing** half from "gated/approximated"
to "obtained and characterised on the vetted catalogue". It does **not** promote
the lag *value* to `[VERIFIED]`: that still needs a *co-temporal velocity
response* for the 2003–2009 ICESat era (sub-annual GPS/GNSS, EarthScope-gated),
which 601439 does not contain. The §G.4 thermal kernel stays **[FALSIFIED]**.

Figure: `validation/reports/usapdc_lakes.png` (left: `log₁₀ τ_ice` vs the surge-lag
band; middle: BedMachine-vs-Bedmap2 thickness 1:1; right: drained-volume histogram).
JSON: `validation/reports/usapdc_lakes.json`.

---

## §V.2d Modern co-temporal matched lag — ICESat-2 ATL15 dates + ITS_LIVE — [DERIVED retained + first in-band field detection; surge is NOT universal]

§V.2b (5 open CryoSat-2 lakes) and §V.2c (58-event vetted forcing, 2003–2009)
closed on the same gate: the matched lag needs a velocity *response* dense enough
to (a) clear the amplitude floor and (b) resolve a 0.02–2 yr lag, and named the
two routes to it — sub-annual GPS, or **fast-outlet trunk velocity**. This section
runs the fast-outlet route end-to-end with two new runners.

**(i) The 2003–2007 catalogue against ITS_LIVE — coverage-limited null**
(`lake_lag_itslive_match.py`). Matching the 58 USAP-DC events to ITS_LIVE at the
31 lakes with SF2018 coordinates: only **7 events are testable** (dense enough
co-temporal pre/post coverage), **0 significant**, median local amplitude bound
**~29 %**. Two structural reasons, now quantified across the whole catalogue
(generalizing the §V.2b 5-lake null): the active lakes sit in **slow ice** at
their centroids (Byrd_2 50, Cook_E2 5, Totten_1 12 m yr⁻¹ → 20–100 % velocity
noise floors), and the genuinely fast outlets (Lambert 277, Slessor 424 m yr⁻¹;
noise floor 0.5–1.3 %) only have dense ITS_LIVE **after ≈2013**, so the 2003–2007
events have **no co-temporal baseline**. (A tempting false positive — every Byrd
lake "spikes" in the 2005 annual bin — is an early-Landsat sampling artefact: that
bin is 3–67 noisy image pairs with IQR [90, 307], and it vanishes under a local
two-sample noise model.)

**(ii) Modern, fully co-temporal — ICESat-2 ATL15 + dense ITS_LIVE**
(`lake_lag_atl15_itslive.py`; ATL15 v005 via **Earthdata** `earthaccess`, ~9.7 GB).
Both halves sit in the 2019–2026 dense era:

- **forcing:** ICESat-2 **ATL15** quarterly gridded surface-height change `delta_h`
  (Smith et al., NSIDC, 1 km, EPSG:3031). Averaging `delta_h` over each SF2018 lake
  outline, detrending, and detecting sustained drawdowns (≥1 m) finds **29 of 131
  lakes with a modern drainage**, including fast outlets (Slessor, Rutford,
  Thwaites, David, Mercer, MacAyeal).
- **response:** ITS_LIVE box-mean speed 2019–2026, quarterly-binned with a *local*
  robust two-sample significance test that is also **controlled against the
  pre-drainage secular trend** (so an accelerating trunk cannot masquerade as a
  surge).

Result: **13 drained lakes are well-resolved** (modern quarterly noise floor <5 %),
**19 events testable**.

- **18/19 show no post-drainage surge** — median amplitude bound **~3 %** of trunk
  speed. At fast, dynamically-relevant lakes (Mac1–4 ≈400, Slessor_23 297, Thw_70
  406, Rutford 383 m yr⁻¹) a drainage does **not** produce a detectable sliding
  response: velocity is flat across the drainage to within a few percent.
- **One in-band detection — Thw_142** (a Thwaites active lake; Smith et al. 2017)
  drained at **2021.75** (1.0 m drawdown); surface speed steps ~234→~250 m yr⁻¹
  (**+8.5 %**), **lag-to-peak 1.125 yr** (inside the derived 0.02–2 yr band), peak
  **4.5σ**, surviving the secular-trend control (detrend 6.3σ, sustained-detrend
  12.9σ). This is the **first in-band field detection** of the §G.4 hydraulic-lag
  signature.

**Honest verdict.** The §G.4 hydromechanical surge is **not a universal consequence
of lake drainage** — most drained, well-resolved lakes show no response (≤~3 %). It
*is* present at one dynamically-active outlet (Thwaites `Thw_142`) with a lag in the
derived band — a single, statistically-robust, secular-trend-controlled candidate.
So the lag **value** stays **[DERIVED]** but now has its **first co-temporal in-band
field detection**, while the *universal* form of the forecast is **disfavoured**:
the response needs a dynamically-primed bed, not merely a drainage. The literal
thermal `H²/κ` kernel remains **[FALSIFIED]** regardless. Figures/JSON:
`validation/reports/lake_lag_itslive.{json,png}`, `lake_lag_atl15_itslive.{json,png}`.

---

## §V.3 synthetic — [VERIFIED] (math)

Unchanged and exercised in `pytest` (`tests/test_validation_synthetic.py`):
`rtn_synthetic` (plant `H*` → recover `RTN>1` region), `sliding_synthetic`
(plant lag → recover via cross-correlation), `cmn_synthetic` (commutator identity
to ~1e-7). These validate the equations against controlled inputs; §V.1/§V.2
above are the real-data exposure.

## Bottom line

- **§H.1 RTN**: [VERIFIED] directional on real Bedmap2 — `RTN>1` hugs the
  grounding line, anticorrelates with thickness, robust in `φ`. **Reproduced on an
  independent NSIDC BedMachine v4 inversion** (§V.1 cross-check, 2 km: median 6 km
  vs 200 km to GL; overall `RTN>1` 2.5 / 1.4 / 0.8% for `φ = 0.8 / 0.9 / 0.95`,
  vs Bedmap2 2.5 / 1.3 / 0.6%).
- **§H.2 sliding**: literal `H²/κ` kernel [FALSIFIED] on 131 real lakes —
  confirmed on independent BedMachine thickness (§V.2c, `τ_ice` median 1.55×10⁵ yr,
  BedMachine-vs-Bedmap2 r=0.97). The vetted USAP-DC forcing is now obtained
  (§V.2c): 58 real drainage events (median 0.13 km³; `q_water` median ~2.5 m³/s).
  The matched lag test runs **end-to-end on open data** (§V.2b): CryoSat-2
  drainage dates + ITS_LIVE velocity give an honest **null** (peak +0.56σ, 0/5
  significant), so the lag value stays [DERIVED]. The record is well resolved
  (~1–3%); the lag is unverifiable here by *temporal aliasing* (sub-annual `t*` <
  quarterly bin) plus an *amplitude bound*, not by noise — pointing to sub-annual
  GPS or fast-outlet trunk velocity for [VERIFIED]. **§V.2d takes the fast-outlet
  route fully into the dense satellite era** (ICESat-2 ATL15 drainage dates +
  ITS_LIVE, both 2019–2026): the surge is **not universal** — 18/19 testable
  drained lakes show no response (amplitude bound ~3% of trunk speed) — but a
  **first in-band field detection** appears at Thwaites `Thw_142` (drained 2021.75,
  +8.5%, lag-to-peak 1.1 yr, 4.5σ, secular-trend-controlled). Lag value stays
  [DERIVED] with its first co-temporal field candidate; the *universal* surge form
  is disfavoured.
- **§I.5 tidal forcing**: real CATS2008 × BedMachine grounding-zone forcing now
  *measured* — the tidal ocean-pressure swing is median 0.30% of ice overburden (a
  hard lower bound on the admittance forcing `ε`), growing ~2× toward the grounding
  line; the §I.5 tidal-admittance probe's forcing input is data-backed, not assumed.
- Precision is not claimed; the framework is *exposed to falsification*.
