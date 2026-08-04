# New cross-relationship — NR35 (`general_two_clocks/new_relationships12.py`)

Continues the derived-and-verified program (NR1–NR34; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only; unit-proofs in
[`tests/test_new_relationships12.py`](tests/test_new_relationships12.py) (9 tests,
2 exercising the local datasets). Run:

```bash
python general_two_clocks/new_relationships12.py   # -> figures/nr35_two_sided_bracket.{json,png}
pytest general_two_clocks/tests/test_new_relationships12.py -v
```

---

## NR35 — The two-sided persistence bracket: repeat radar is a **threshold gauge** for effective pressure, and a dated drainage **measures** its correction  [NR33 Nye e-fold × USAP-DC 601439 dated volume histories × ICECAP two-epoch specularity × §V.1e σ-gauge; executes horizon-ledger E5a]

### The gap this closes

NR33 proved one direction: a water body specular in surveys separated by `Δt` survived
Nye closure, so `N ≤ N*(Δt) = n(2AΔt)^{−1/n}` — a pressure **floor**, resting on the
*assumption* that melt-opening ≪ closure. Two things were missing: the **other side**
(what a *disappearing* specular cell says), and a way to replace the quiescence
assumption with a measurement. Both close here.

### The statements

**(1) Vanish ceiling (new; melt-robust).** A cell specular at epoch A and dark at
epoch B lost at least one net closure e-fold: `∫(c−m)dt ≥ 1` with opening `m ≥ 0`
forces `c·Δt ≥ 1`, i.e.

    N ≥ N*(Δt)      ⇔      φ ≤ 1 − N*(Δt)/p_i .

Melt only *strengthens* this direction (a fortiori — unit-proved by explicit survival
integrals). The systematic alternatives (roughening, water spreading below detection,
track-sampling flicker) are handled by a deadband classifier (specular ≥ 0.2, dark
< 0.1, ≥ 3 pts/cell/epoch) and stated as the attribution caveat.

**(2) Melt-corrected floor.** Survival with *measured* mean relative opening `m̄`
weakens NR33's floor by exactly the measured amount:

    N ≤ n((1/Δt + m̄)/(2A))^{1/n}     (reduces to N*(Δt) at m̄ = 0).

**(3) Threshold-gauge corollary.** Both directions share the same e-fold constant:
one repeat pair splits every twice-seen specular cell into `N ≤ N*` (persist) vs
`N ≥ N*` (vanish) — **repeat radar is a binary effective-pressure classifier** whose
threshold is set only by ice rheology and revisit time, tunable as `N* ∝ Δt^{−1/n}`
(NR33's archive dividend, now two-sided). Conservative spans per direction (floor:
3 yr midpoints; ceiling: 4 yr envelope) give thresholds **3.90 / 3.55 bar** with the
honest deadband between them.

### Applied to ICECAP 2008/09 × 2011/12 (grounded Bedmap2 cells)

| class | count | reading |
|---|---|---|
| twice-seen | 259 | (matches NR33) |
| persist | **7** | `N ≤ 3.90 bar`, `φ ≥ 0.98` (NR33 floor, now with company) |
| **vanish** | **11** | `N ≥ 3.55 bar` — firmly **not** at flotation: new information no single survey carries |
| appear | 2 | cavity opening (no creep bound; logged) |
| dark | 207 | uninformative |

Deadband sensitivity: persist/vanish = 9/18 at (0.15, 0.075), 7/11 at (0.2, 0.1),
2/7 at (0.3, 0.15) — the vanish class is robustly non-empty. Each ledger row carries
the §V.1e roughness gauge per epoch (`σ_A, σ_B`): vanishing cells cross the gauge
ceiling — the *geometric* face of the same transition.

### The dated anchor: Totten_2 (601439 × both radar epochs)

Centroids for all 124 volume-history lakes were parsed from their z-grid GeoTIFF tags
(PIL; CRS key asserted EPSG:3031) and cross-validated against the independent 601470
outline catalogue: **median nearest-neighbour 0.52 km**. Exactly one dated lake sits in
the twice-seen set: **Totten_2** — which is also the *strongest persistent-specular
cell in the census* (spec 0.49 → 0.32) and §V.1e's top joint-(σ,φ) row (H = 3886 m).
Its record gives two dated drainages (troughs 2005.4, 2007.8; 0.60, 0.32 km³), the
lake low and slowly refilling at record end. Three instruments interlock:

* **altimetry** (601439): water present, roof deflated since `t_d = 2007.79`, refill
  rate `dV/dt = 0.089 km³/yr` measured;
* **radar**: survival over the *dated* span `t_d →` epoch B: `Δt = 4.21 yr` — measured,
  not assumed;
* **creep clock**: dated floor `N ≤ 3.49 bar`, melt-corrected to
  `N ≤ 4.10–5.70 bar` (`φ ≥ 0.984–0.988`), where the bracket spans the two defensible
  normalisations of `m̄` (refill velocity over full range: 0.148 yr⁻¹; over current
  deflated depth: 0.80 yr⁻¹) — corrections of **+18 % to +63 %**, *measured*, replacing
  NR33's "melt ≪ closure" assumption at the one cell where the datasets overlap.

### Why this is one of ours

The two-clocks program keeps finding that a slow clock *reads* a fast field. NR33 made
persistence a one-sided gauge; NR35 makes the pair of epochs a **two-sided instrument**
— and shows the gauge's leading systematic (melt-opening) is itself measurable when an
altimetry clock (dated drainage + refill) runs alongside the radar clock. Survey-design
dividend: with ICESat-2-era repeat radar (`Δt` 1–15 yr), `N*` sweeps 2.5–6 bar — a
tunable effective-pressure CT scan of the bed from archives alone.

### Honest scope

The ceiling's creep attribution competes with non-creep dimming channels (stated
above; the deadband and min-pts control flicker, not physics). The floor keeps NR33's
caveats where undated. `m̄`'s normalisation for a lake-mean volume series is a modelling
choice — carried as an explicit bracket, never a point value. Glen `A` (temperate) is
the dominant rheological systematic, entering all bounds as `A^{−1/n}`. Epoch spans
use conservative direction-specific choices (3 yr floor / 4 yr ceiling).

### Artifacts

`figures/nr35_two_sided_bracket.{json,png}`; 9 unit proofs
(`tests/test_new_relationships12.py`): e-fold root, shared-threshold + archive
dividend, melt-robustness of the ceiling (explicit survival integrals), monotone
melt-corrected floor + exact 1/3-power structure, φ-bound directions, deadband
classifier logic, class disjointness, GeoTIFF-centroid cross-validation (< 2 km),
Totten_2 dated arithmetic.
