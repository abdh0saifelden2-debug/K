# §G.4 / §H.2 population data fit UN-GATED — ATL15-dated drainage events × ITS_LIVE response

**Module** `glaciers/validation/external/lake_drainage_response_atl15.py` ·
**cache** `glaciers/validation/external/data/lake_drainage_response_cache.json` (committed) ·
**report** `glaciers/validation/reports/lake_drainage_response.json` ·
**tests** `glaciers/tests/test_lake_drainage_response.py` (11)

## What was gated, and what un-gated it

§G.4's closing line was *"Not a data fit (drainage dates USAP-DC-gated)"*: the derived
two-compartment hydraulic lag kernel (peak `t* ~ 0.01–2 yr`) had no dated-event
population to be fit against. The gate dissolves with data already in-repo: the
committed NR43 **ATL15 cache** (quarterly 10-km Δh at all 131 Siegfried–Fricker
outlines, 2019–2026) contains the drainage dates directly. This unit crosses those
dates with **monthly-binned ITS_LIVE v2** speed at each lake centroid — a dated-event
population test that is *independent* of the CryoSat/ICESat event lists used by the
earlier §H.2 run (`lake_lag_atl15_itslive.py`, 1 in-band detection / 19 lakes) and of
the 3-lake §I re-analysis (`lake_lag_sn_ews.py`).

## Event set (from the committed Δh cache alone)

Sustained-drop detector (±4-quarter medians, −3σ_quad, 5 cm floor, ≥6-quarter
separation): **27 dated events on 23 lakes**, 2019–2026 — including Thw_70 (−2.0 m,
2021.5), Thw_124 (−1.45 m, 2023.25), Slessor_23 (−6.3 m, 2024.0),
EngelhardtSubglacialLake (−3.5 m, 2023.5), Totten_2 (−0.85 m, 2023.25), six Byrd-system
lakes. Monthly velocity binning (pairs ≤90 d, ≥3/month) is essential: Antarctic optical
pairs cluster Oct–Mar and leave half of all calendar *quarters* empty (the quarterly
version of this test loses 6 of 9 testable events).

## Result — POPULATION NULL with a quantified background

| quantity | value |
|---|---|
| events testable (coverage + ≥20 m/yr site) | 9 / 27 (reasons tallied: 8 pre-coverage, 7 post-coverage, 2 too-slow) |
| nominal detections (2 consecutive windows, \|Δv/v\| > max(2σ, 2 %)) | 3/9 (Thw_124 +8.5 %, Institute_W2 +4.9 %, Byrd_s6 −13 %) |
| **same detector on pre-event windows of the same lakes** | **5/9 fire** |
| same detector on quiet-lake pseudo-events (n=175) | 17 % fire |
| binomial vs quiet background | p = 0.19 |
| population response bound | median max\|Δv/v\| 8.5 %, p90 23 % (0.25–2 yr band) |

The detector fires *more often on the pre-event side of the same lakes* than after the
dated drainages, and the post-event rate is indistinguishable from the quiet-lake
background: **drainage timing adds nothing at this precision.** The nominal detections
are mixed-sign (2 up, 1 down) — not the uniform §G.4 surge direction (and Byrd_s6's
−13 % is the *opposite* of Stearns et al. 2008's +10 % Byrd speed-up template).

## What this settles / bounds

- The **literal §G.4 "drainage step → in-band sliding surge" coupling is population-null
  on an independent, ATL15-dated 2019–2026 event set** — confirming and extending the
  1/19 CryoSat-era result with denser (monthly) response resolution and explicit
  same-site background calibration.
- Via the §I.1 master curve `Δv/v = |s_N|·ΔN/N`, the 8.5 % (median) response bound keeps
  these trunk-lake sites **far from the `N_c` flotation fold** unless per-event ΔN/N is
  tiny — the same far-from-fold conclusion the 3-lake §I re-analysis reached, now at
  population scale.
- NR32's band-limited-transmission reading survives untouched: the rare genuine
  responders (CryoSat-era Thw_142; possibly Thw_124 here) remain the exception that a
  distributed→channelized transition window selects — universality was *never* the
  two-clocks prediction.

## Honest limits

10-km ATL15 pixels smear small lakes (drop amplitudes are lower bounds; small-lake dates
±1 quarter). Centroid velocity is a single point. The fast corner of the derived band
(`t* ~ 0.01 yr ≈ 4 d`) is invisible at monthly binning. 18/27 events untestable —
dominated by the polar-hole/coverage floor (Siple Coast + Foundation), not by physics.
Pre-event calibration n=9 (record-length limited).

## Data citations

Smith, Fricker et al. ICESat-2 **ATL15** v5 (via committed NR43 cache); Siegfried &
Fricker 2018 outlines; ITS_LIVE v2 (Gardner et al.); Stearns, Smith & Hamilton 2008
(Byrd template); Siegfried et al. 2016 (Whillans GPS template).
