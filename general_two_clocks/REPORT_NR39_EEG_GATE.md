# NR39 real-data gate — EXECUTED (`general_two_clocks/nr39_eeg_gate.py`) — the imaginary-coherence volume-conduction lemma on real EEG (ledger E7)

Closes the real-EEG consistency gate registered by NR39
([`REPORT_NEW_RELATIONSHIPS16.md`](REPORT_NEW_RELATIONSHIPS16.md);
`figures/nr39_imaginary_coherence.json` → `real_eeg: available=false`).
NR39 proved that P0's phase-surrogate ceiling and EEG's imaginary-coherency
artifact rejection (Nolte et al. 2004) are **two corollaries of one lemma**:
instantaneous linear mixing `x = A s` of a real-spectrum source set gives a
real-symmetric cross-spectrum, so `Im S_x = 0` — volume conduction (an
instantaneous Poisson solve) cannot manufacture imaginary coherency. This
runs that lemma on real resting EEG. Offline-safe: every number re-derives
from the committed 84 KB cache
[`data/nr39_eeg_gate_cache.json`](data/nr39_eeg_gate_cache.json) (raw EDF not
committed; set `$NR39_EDF` and run `build_cache()`).
Unit-proofs in [`tests/test_nr39_eeg_gate.py`](tests/test_nr39_eeg_gate.py)
(9 tests).

```bash
python general_two_clocks/nr39_eeg_gate.py   # -> figures/nr39_eeg_gate.json
pytest general_two_clocks/tests/test_nr39_eeg_gate.py -v
```

## Data

**OpenNeuro ds003775** (Hatlestad-Hall et al., SRM resting-state EEG, CC0),
subject sub-001 ses-t1 task-resteyesc: 64-channel BioSemi ActiveTwo (10-10),
4 min eyes-closed, 1024 Hz. Alpha-band (8–13 Hz) magnitude-squared coherency
`C_ij` over all 2016 pairs from 118 Hann windows (4 s, 50 % overlap). Ceiling:
whole-series phase randomization (preserves each channel's PSD, destroys
cross-phase), 200 surrogates, per-pair 95th percentile. Electrode distances
from the biosemi64 montage.

## Findings (`figures/nr39_eeg_gate.json`)

1. **Volume conduction dominates coherency — in the real part.** Median
   `|Re C| = 0.53` vs `|Im C| = 0.064`, **Re/Im = 8.2**. The
   instantaneous-mixing field is large and, as the lemma requires, carried
   by Re.
2. **The decisive real-data signature: mixing feeds Re, not Im.** Across
   pairs `Spearman(|Re C|, |Im C|) = −0.25` (p = 3×10⁻³⁰) — more mixing means
   *less* imaginary coherency. The top-mixing decile (near-adjacent
   electrodes, strongest instantaneous mixing) has **Re/Im = 28** with
   `|Im C| = 0.031`, *below* the global median. Where an artifact account
   predicts the largest imaginary coherency, the lemma predicts (and the data
   show) the smallest.
3. **Imaginary coherency is inter-regional; mixing is local.**
   `Spearman(dist, |Re C|) = −0.15` (mixing falls with distance) while
   `Spearman(dist, |Im C|) = +0.15` (lagged coupling rises with distance);
   nearest-decile pairs `|Re C| = 0.83, |Im C| = 0.028`.
4. **The surrogate ceiling behaves exactly as P0 §6 predicts.** Null floor
   (surrogate median `|Im C|`) = 0.016; 62.6 % of pairs clear their per-pair
   p95 (genuine alpha-band lagged coupling is widespread) — but only 33 % of
   the top-mixing decile clear it. The most volume-conduction-contaminated
   pairs are the *least* likely to show significant imaginary coherency, the
   direct inverse of an artifact.

## What this closes

E7's registered gate asked for "any OpenNeuro resting EEG" as a consistency
demo (`Re ≫ Im`). The real data deliver more than consistency: the *sign* of
the mixing→coherency map is measured (mixing loads Re; the strongest-mixed
pairs have the smallest Im and clear the ceiling least), which is the lemma's
content — instantaneous mixing is real and therefore invisible to Im and to
the phase-surrogate ceiling. P0's ceiling and Nolte's ImCoh rejection are one
theorem, confirmed on the Sun-as-a-star scale (NR46) and now on the human
cortex.
