# Ocean — double diffusion at the ice–ocean interface (salt + temperature)

Where the two-clocks framework meets **seawater**: cold-fresh meltwater over
warm-salty ocean, two buoyancy-active scalars with a 100× molecular-diffusivity
contrast (`Le = 100`), and what a scalloped ice wall does to that system.

Closely related ocean-facing work that runs on *ice-sheet* geometry is kept with
its data in `../glaciers/`: the §G.3 **ocean-intrusion** RTN (verified
directional at the grounding line) and the §H.1.2 intrusion clock. The ocean
*tide* loader (CATS2008, auth-gated) is `../glaciers/validation/external/tide_loader.py`.

## What was found so far

- **[VERIFIED, honest negative]** Candidate 2 — the pre-registered salt-finger
  hump is **absent**: `Nu_T(R_ρ)` decreases monotonically (1.44 → 0.73 across
  `R_ρ = 0.5 → 10`); `R_ρ ≈ 2` is *not* a maximum in this 2-D forced penalized
  regime. [`REPORT_CANDIDATE2.md`](REPORT_CANDIDATE2.md), Fig 48
  (`run_candidate2.py`).
- **[VERIFIED]** The diffusivity contrast *does* show up robustly: `Nu_S ≫ Nu_T`
  (salt transported far more efficiently than heat) and **counter-gradient** heat
  flux at strong stabilisation — another flux-space signature down-gradient
  K-theory cannot produce.
- **[VERIFIED structural, §D.2]** A scalloped wall on the same double-diffusion
  solver: phase-binned `Nu_T` heterogeneity amplified (5.3 → 67 peak-to-trough),
  wall-coherent variance fraction `η²` 0.09 → **0.89** (phase-locked, with a
  dominant 2nd harmonic), and *both* double-diffusive regimes (enhanced 93% /
  suppressed 7% of the wall) **coexist in one geometry** — local Turner-ratio
  mixing. [`REPORT_DOUBLEDIFF.md`](REPORT_DOUBLEDIFF.md), Fig 50 (GPU).
- **[VERIFIED, RESULT 13 — kept in `../general_two_clocks/`]** the transport
  memory time `τ_c` is identical for heat and salt despite `Le = 100`: the clock
  belongs to the turbulence, not the scalar.

## What remains a HYP

- **Salt fingers in realistic regimes [HYP]** — the absence of the `Nu_T` hump
  may be an artifact of 2-D + penalization + forcing; 3-D unforced
  double-diffusive staircases remain untested here.
- **Real ice-shelf relevance [HYP]** — whether scallop-locked regime mixing
  (§D.2) shapes melt under actual ice shelves (e.g. staircase observations under
  George VI) is not tested against ocean data.
- **§G.3 threshold [HYP / LIT]** — RTN as the quantitative criterion for warm
  ocean water intruding under grounded ice leans on literature closures (see
  `../glaciers/README.md`).

## What is pending

- A 3-D double-diffusion run (the solver family is GPU-ready; see
  `../.agents/skills/testing-gpu-cupy/`).
- Testing the §D.2 regime-coexistence prediction against oceanographic
  microstructure / ice-shelf cavity observations.
- Coupling the double-diffusive `Nu_S ≫ Nu_T` asymmetry into the §G.6 melt-rate
  phenomenology (currently local-flux-only).

## Contents

- `run_candidate2.py` — Candidate 2 `R_ρ × closure` sweep (Fig 48).
- `scallop_doublediff.py` — §D.2 scalloped-wall double-diffusion regime mixing
  (Fig 50).
- Reports: [`REPORT_CANDIDATE2.md`](REPORT_CANDIDATE2.md),
  [`REPORT_DOUBLEDIFF.md`](REPORT_DOUBLEDIFF.md).
- `figures/` — Figs 48, 50. `tests/` — `test_candidate2.py`,
  `test_doublediff_scallop.py`.

The solver itself (`candidate2_doublediff.py`) lives in the shared cavity
library `../glaciers/subglacial/` (both scripts bootstrap the path). Run from
the repo root: `python ocean/run_candidate2.py`; `pytest ocean/tests -v`.
