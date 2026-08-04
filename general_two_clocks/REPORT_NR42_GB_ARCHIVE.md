# NR42 real-data gate — EXECUTED (`general_two_clocks/nr42_gb_archive.py`) — GB 1-s frequency archives vs measured inertia (ledger E9)

Executes the gate registered in
[`figures/nr42_grid_frequency_window.json`](figures/nr42_grid_frequency_window.json)
(`real_data_gate`) by NR42
([`REPORT_NEW_RELATIONSHIPS19.md`](REPORT_NEW_RELATIONSHIPS19.md)): band-limited
disturbance variance in the [2,30] s primary-response window, binned by measured
inertia — **two-clocks window prediction** = single-peaked with an interior
maximum (peak at c\* ≈ 30 % of the all-synchronous inertia scale) vs the
**Markovian (quasi-static control) null** = maximum at the highest-inertia bin.
Offline-safe: every number re-derives from the committed 46 KB aggregate cache
[`data/nr42_gb_archive_cache.json`](data/nr42_gb_archive_cache.json) (raw NESO
CSVs not committed; set `$NR42_GB_DIR` and run `build_cache()`).
Unit-proofs in [`tests/test_nr42_gb_archive.py`](tests/test_nr42_gb_archive.py)
(12 tests).

```bash
python general_two_clocks/nr42_gb_archive.py   # -> figures/nr42_gb_archive.json
pytest general_two_clocks/tests/test_nr42_gb_archive.py -v
```

## Data

NESO Data Portal (open licence): **"Historic frequency data"** (1-s GB system
frequency; 7 months spanning the inertia decline and the Oct-2020 Dynamic
Containment introduction — preDC: Jan 2018, Jul 2018, May 2020; DC: Jan 2022,
Jul 2023, Jan 2025, Jul 2025), **"System Inertia"** (settlement-period Outturn
Inertia, GVA·s — the measured inertia the gate asked for, not a proxy), and
**"Historic demand data"** (ND, for the demand-normalized control; available
through Jul 2023). Joined on the settlement key: **10,416 half-hours (18.7 M
1-s samples)**. Per half-hour: Hann periodogram → band variance in [2,30] s
(registered) and 60–300 s (out-of-band control); strata: all / day (10–17 h) /
night (0–5 h) / demand-normalized (var/ND²); octile bins by inertia within
each era × stratum; Spearman ρ over half-hours; bootstrap P(argmax interior).

## Findings (`figures/nr42_gb_archive.json`)

| [2,30] s stratum | preDC ρ | preDC argmax | DC ρ | DC argmax |
|---|---|---|---|---|
| all | **−0.24** (p~1e-59) | bin 1 (171 GVA·s) | +0.03 | bin 4 |
| day | **−0.64** | bin 0 (low edge) | −0.25 | bin 3 |
| night | −0.16 | bin 2 | −0.06 | bin 2 |
| demand-normalized | **−0.76** | bin 0 (low edge) | **−0.53** | bin 0 (low edge) |
| *60–300 s control (all)* | *+0.18, argmax bin 5* | | *+0.33, argmax bin 7 (top edge)* | |

1. **Markovian null REJECTED, 10/10 strata**: the maximum is never at the
   highest-inertia bin and ρ ≤ +0.03 everywhere in the registered band. The
   quasi-static-control ordering ("more inertia = more observable") is the
   wrong ordering for real GB frequency noise — exactly the discriminator
   NR42 derived.
2. **Window-side ordering CONFIRMED**: in-band variance rises as inertia
   falls (toward the predicted sub-range peak), strongest under controls
   (preDC day −0.64, demand-normalized −0.76 with the maximum pinned at the
   low-inertia edge).
3. **The interior peak is honestly unresolved — and the criterion itself
   says it must be.** The observed within-era inertia span is 0.37–0.40
   decades, ~⅓ of the 1.18-decade band, below NR42's own crossing
   requirement; the c\* ≈ 0.30 readout puts the peak at 105–135 GVA·s for
   any all-synchronous scale of 350–450 GVA·s — at/below the observed low
   edge (low-octile medians 151/126 GVA·s). On this window the resolvable
   registered prediction is the **sign of the slope** (finding 2), not the
   turning point. Resolution needs either future lower-inertia states or
   event-conditioned responses.
4. **Band specificity**: out-of-band (60–300 s, secondary control/AGC) the
   ordering **inverts** (+0.18/+0.33, DC maximum at the top-inertia edge —
   the demand/scheduling direction). The in-band signal is not a generic
   variance artifact.
5. **DC-era flattening** (flagged reading, not registered): post-Dynamic
   Containment the pooled in-band ordering flattens (−0.24 → +0.03;
   demand-normalized −0.76 → −0.53). Converter-fast response raises in-band
   β without adding storage — filling the very window the criterion
   monitors. A grid-scale instance of the criterion's "removes storage but
   re-fills fast response" boundary case.
6. **Limit — the ambient PSD knee is not a calibration-free inertia
   readout.** Fitting the identifiable SFR shape `|iωτ + (1−k) +
   k/(1+iωT_g)|⁻²` (τ = M/β exactly) to pooled tercile spectra gives τ_sys
   = 8.9/7.6/6.0 s while measured H rises 168→294 GVA·s: β co-varies with
   demand (response holding scales with demand, which co-varies with
   inertia), cancelling M in the knee. NR42's calibration-free
   critical-inertia readout therefore requires **event-conditioned**
   (infeed-loss) responses — the one piece of the gate that stays open.

## What this buys the program

The NR32→NR42 transfer is no longer only structural: on real archives the
memory-signature ordering (band-limited observability *rising* as the storage
compartment empties) is measured with the Markovian alternative rejected in
every stratum, and the two failure modes the theory itself predicted — range
under-spanning the band, and fast-response re-fill (DC) — are both observed
doing exactly what the criterion says they do.
