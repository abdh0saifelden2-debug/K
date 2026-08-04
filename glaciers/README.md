# Glaciers — subglacial cavities, scallops, sliding, and ice-sheet validation

The largest application folder: the two-clocks closure tested in a subglacial
cavity (Parts 9–10g), the scallop/channelisation instability family, and the §V/§H
validation pipeline that takes the framework's [HYP] predictions to **real
Antarctic data** (Bedmap2, BedMachine, active lakes, ITS_LIVE velocities).
The master claim-status ledger is [`../FUTURE_WORK.md`](../FUTURE_WORK.md)
(§A–§H); statuses below are quoted from it.

## What was found so far

**Cavity benchmark and melt**
- **[VERIFIED]** The Part-8b closure verdict transfers to a penalized cavity DNS
  over synthetic *and real BEDMAP1* beds: K-theory kills the wake/backscatter
  structure that controls basal heat flux ([`REPORT_SUBGLACIAL.md`](REPORT_SUBGLACIAL.md),
  [`REPORT_SUBGLACIAL_REAL.md`](REPORT_SUBGLACIAL_REAL.md), Figs 31–38), and the
  3-D a-posteriori LES makes the basal **melt rate closure-dependent**
  (`run_subglacial3d.py`, [`subglacial/THEORY_CAVITY.md`](subglacial/THEORY_CAVITY.md)).
- **[VERIFIED, RESULT 11]** Counter-gradient heat flux measured: `C_G > −1`
  grows with stratification — flux carries the global pressure field's memory,
  which down-gradient K-theory cannot represent
  ([`REPORT_CG_BUOYANCY.md`](REPORT_CG_BUOYANCY.md), Fig 52, GPU).
- **[VERIFIED + mechanism]** §G.1 thermal-wall ceiling: scalloped walls give
  `Nu/Nu_flat < 1` (conduction-limited, boundary-condition-robust — Robin ice
  wall and Navier slip change nothing); mechanism earned via the exact
  area-partition identity (`scallop_g1_populations.py`); the `(1+CV²)` truncation
  is **FALSIFIED**.

**Scallops (the constructive result)**
- **[VERIFIED/DERIVED, RESULT 14 + 22 + 23]** The scallop is a *damped,
  downstream-migrating* mode `s = −β + i·ω_mig`: no autonomous growth
  (smoothing-limited), `K²` curvature ansatz falsified, and migration
  `∝ U^{0.5–0.8}` is a **parity-symmetry break no K-theory closure can produce**
  (keystone control measured on GPU). Verdicts generalized across amplitude
  `a₀/λ ∈ [0.05, 0.40]` and drive up to `U = 6`.
  [`REPORT_SCALLOP_MIGRATION.md`](REPORT_SCALLOP_MIGRATION.md), Fig 56.
- **[VERIFIED directional]** Scallop reattachment melt is a phase-locked opening
  source that deterministically selects R-channel nucleation sites
  ([`REPORT_CHANNEL.md`](REPORT_CHANNEL.md), [`REPORT_CHANNEL_Z0.md`](REPORT_CHANNEL_Z0.md),
  Figs 49, 50b); shedding-edge **[NULL]** made conclusive
  ([`REPORT_SHEDDING.md`](REPORT_SHEDDING.md), Fig 51); creep **[NULL]** (§D.6;
  long-time correction quantified, [`REPORT_CREEP_STEFAN.md`](REPORT_CREEP_STEFAN.md));
  3-D scallop geometry **[NULL — closed]** (§D.7); `(a/λ)²` amplitude closure
  **FALSIFIED** (Fig 59).

**Mainstream-model correction (drop-in)**
- **[DERIVED → code]** [`subglacial/glads_memory_correction.py`](subglacial/glads_memory_correction.py)
  turns the RESULT 21 Mori–Zwanzig projection into a retrofit for operational
  GlaDS-type closures. Large-scale models slave basal sliding to an *instantaneous*
  effective-pressure law, dropping the eliminated channel's memory; the exact
  reduced (cavity-only) dynamics is a generalized Langevin equation with kernel
  `K(τ) = −a·b·e^{−τ/τ₂}` (the eliminated channel's Green's function). The mainstream
  *local* closure is its fast-channel limit `τ₂ → 0`, where `K` collapses to its DC
  gain `∫K = −a·b·τ₂` and the post-drainage **surge lag is lost**. `apply_memory_kernel(...)`
  convolves that memory back in, so the corrected downstream response recovers the
  interior surge peak (the 0.02–2 yr lag of §H.2 / Thwaites `Thw_142`) that the
  memoryless closure — monotone from t=0 — cannot produce, while `τ₂ → 0` returns
  the original local model. Verified deterministically: corrected closure matches
  the resolved cavity↔channel truth to <1e-4 and the local closure misses the
  interior peak ([`tests/test_glads_memory_correction.py`](tests/test_glads_memory_correction.py)).
  This is the analytic/linearised retrofit; the real ISSM/GlaDS run remains pending
  (below).

**Real-data validation (§H)**
- **[VERIFIED directional]** RTN > 1 concentrates at the grounding line on
  Bedmap2 (median 6 km vs 221 km) and survives an independent BedMachine v4
  cross-check at native 500 m ([`REPORT_RTN_MISI_BAND.md`](REPORT_RTN_MISI_BAND.md),
  `validation/REAL_DATA_RESULTS.md`, Fig 57).
- **[FALSIFIED as written]** the §G.4 literal thermal sliding kernel: the
  `H²/κ ≈ 1.5×10⁵ yr` lag is ~8×10⁴× too slow on 131 real lakes — the lag is
  **hydromechanical**; the peaked kernel *shape* and order-of-magnitude *value*
  are **[DERIVED]** from cavity↔channel physics, and the projection is an exact
  Mori–Zwanzig GLE ([`REPORT_HYDRAULIC_MZ_PROJECTION.md`](REPORT_HYDRAULIC_MZ_PROJECTION.md),
  RESULT 21).
- **[VERIFIED — synthetic]** estimator calibrations: GL-migration level-set law
  (RESULT 16, [`REPORT_GLMIG_ESTIMATOR.md`](REPORT_GLMIG_ESTIMATOR.md)),
  kernel-shape-generic lag estimator (RESULT 17,
  [`REPORT_SLIDING_KERNEL_SHAPE.md`](REPORT_SLIDING_KERNEL_SHAPE.md)),
  φ-area inversion (RESULT 18, [`REPORT_RTN_PHI_INVERSION.md`](REPORT_RTN_PHI_INVERSION.md)),
  hydrology-corrected MISI band (RESULT 19), intrusion-clock driver
  (RESULT 20, [`REPORT_RTN_INTRUSION_CLOCK.md`](REPORT_RTN_INTRUSION_CLOCK.md)).

## What remains a HYP

- **§G.3** RTN as the *threshold* for ocean-intrusion regime change ([HYP / LIT];
  only the directional concentration is verified).
- **§G.4** the hydromechanical lag *mechanism* on real glaciers ([HYP mechanism];
  shape/value derived, not data-fit — drainage dates are USAP-DC-gated).
- **§H.1.2/§H.1.3** intrusion-clock *hydraulic pacing* and `u_*` pacing
  ([HYP → disfavored at continental scale]; **[HYP supported]** in marine West
  Antarctica, §H.1.4 — regime-dependent).
- **§G.6** unified melt-rate phenomenology (OK only with *local* flux).

## What is pending

- **The field test of RESULT 14**: compute `I_obs = τ·c_mig/λ` on a real scallop
  train (Bushuk et al. 2019 is the candidate dataset) — the single step that
  moves RESULT 14 from a discovery *in the framework* to a discovery *about the
  world* (`scallop_field_test.py` is ready, Fig 58).
- Gated data: USAP-DC lake drainage-date series (§V.2 full lag fit), BedMachine
  v3 via Earthdata, CATS2008 tides (`validation/external/tide_loader.py`).
- §H.3 CMN correction inside a *real* ice-sheet model (ISSM/GlaDS) — the
  synthetic-solver half is done (RESULT 15, general folder).
- GPU re-runs of the full CuPy family on a CUDA device (see `gpu_family_sweep.py`).

## Contents

- `subglacial/` — cavity DNS/LES library: `flow.py`, `flow3d.py`, candidates
  1/3/4 probes (candidate 2 lives here too but its *runs* are in `../ocean/`),
  Stefan/moving-boundary, slip/wall-flux gates,
  `glads_memory_correction.py` (GlaDS-type mainstream memory-kernel retrofit),
  `THEORY_CAVITY.md`, bundled BEDMAP1 transect (`data/`).
- `validation/` — §V pipeline: `synthetic/` harnesses, `validators/`,
  `external/` real-data loaders + runners, `REAL_DATA_RESULTS.md`.
- Runners: `run_subglacial.py`, `run_subglacial3d.py`, `run_candidate1.py`,
  `run_candidate3.py`, `run_candidate4.py`, `run_stefan_sweep.py`.
- Scallop family: `scallop_probe.py`, `scallop_battery.py`, `scallop_sweep.py`,
  `scallop_amplitude_*.py`, `scallop_channel_*.py`, `scallop_field_test.py`,
  `scallop_ktheory_control.py`, `scallop_g1_populations.py`,
  `scallop_shedding_deterministic.py`, `scallop_sublayer_probe.py`,
  `scallop_forcing_probe.py`, `scallop3d_probe.py`.
- GPU probe: `theorem3_cg_gpu_probe.py` (Fig 52).
- `figures/` — Figs 31–38, 49–52, 56–59 + candidate JSONs. `tests/` — 20 test
  modules.

Run from the repo root, e.g.
`python glaciers/run_subglacial.py --bed real --out-dir glaciers/figures`;
`python glaciers/validation/external/run_rtn_bedmap2.py --stride 3 --phi 0.9`;
`pytest glaciers/tests -v`.
