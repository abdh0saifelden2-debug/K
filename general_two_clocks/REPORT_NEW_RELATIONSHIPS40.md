# NR63 — Active subglacial lake systems are gain-clock synchronized, not transport-clock cascades

**Claim.** The NR49 gain/all-pass decomposition, applied to the real ICESat-2
active-lake network, shows the network runs on the **gain clock**
(common-mode fill/drain), not the transport (directed-cascade) clock. For
two lake height series the coupling splits into `gain` = zero-lag |corr|
(common mode) and `all-pass` = lead-lag asymmetry (directed propagation).
Because lake height is even under time reversal, the lead-lag asymmetry is
exactly NR61's equal-parity irreversibility carrier.

## Real network (committed ATL15 cache: ICESat-2 v005 Δh, 131 Siegfried–Fricker 2018 active-lake outlines, 29 quarterly epochs 2019–2026, 22 multi-lake systems)

* **Connected lakes are synchronized far above chance.** Within-system
  median |corr| = **0.442**, versus an independent phase-randomization
  surrogate (each lake's power spectrum preserved, cross-phase destroyed) of
  0.218 (q95 0.235; p < 1/300), and versus the between-system baseline of
  0.305 (Mann–Whitney p ≈ 5×10⁻²¹). The excess over *both* nulls is the gain
  clock: connected lakes fill and drain together, a system-coherent common
  mode above the continental baseline.
* **The gain clock dominates the transport clock.** The zero-lag (gain)
  coupling is robust and surrogate-significant; the lead-lag (all-pass)
  asymmetry does not align with along-flow geometry for the large systems
  (|ρ(net-lead, along-axis)| = 0.02–0.10 for Byrd_s n=15, Foundation n=16,
  Kamb n=12) — directed flood routing is below the ICESat-2 quarterly/decadal
  resolution floor. (Small n=3 systems give trivially high |ρ|; not
  interpreted. One 6-lake system, Slessor, shows |ρ|=0.71 but does not
  survive multiple-comparison caution.)
* **Regime map.** Systems rank from tightly synchronous (Bindschadler, gain
  0.92 — a hydraulically-locked reservoir group) to loosely coupled (Byrd_s
  0.35). None of the large systems is a clean all-pass cascade.

## Method validation (synthetic — shows the decomposition *can* see a cascade, so the real all-pass null is meaningful)

* Synchronous system (one common driver + independent per-lake noise): gain
  recovered high (0.57 > surrogate q95 0.38), net lead ≈ 0.
* Directed cascade (`h_{k+1}=a·h_k(t−τ)+noise`): net-lead ordering recovers
  the true chain order (Spearman = 1.0) and the correct direction; all-pass
  exceeds the synchronous case >1.5×.
* Phase-randomization surrogate calibrated.

## Physical reading

At ICESat-2 scales the active-lake network is a gain-clock system: regional
common-mode hydraulic forcing (basal-melt/storage variability shared across a
connected system) dominates the gridded-altimetry observable, while
sequential flood routing (the transport clock, seen in dedicated event
studies) averages below the floor. This sharpens NR43 (the lakes are the
episodic fast clock) with a new statement — within a connected system the
lakes are *synchronized* — and is the NR49 gain/all-pass split measured on a
real cryosphere network. Mainstream anchors: Siegfried & Fricker 2018
(active-lake inventory); Smith et al. / Fricker (ICESat-2 lake altimetry);
Livingstone et al. 2022 review (subglacial lake drainage connectivity).

**Artifacts:** `figures/98_lake_gain_allpass.json` (all 6 verdicts true); 11
unit proofs in `tests/test_lake_gain_allpass.py`; offline-safe (committed
ATL15 cache). Reproduce: `python general_two_clocks/new_relationships40.py`.
