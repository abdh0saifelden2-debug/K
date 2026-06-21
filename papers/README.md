# Papers extractable from this repository

This folder contains **submission-ready manuscript drafts** built *only* from the
results this repository tags `[VERIFIED]`, `[DERIVED]`, or productively `[FALSIFIED]`,
plus the literature scaffolding (`[LIT]`) needed to frame them. Every headline number
in every draft is produced by code in this repo and was re-confirmed before drafting
(see **Verification status** below). Open hypotheses (`[HYP]`) are carried only as
clearly-labelled future work, never as claims.

The drafts follow the same claim-tag discipline used in `../FUTURE_WORK.md` and
`../PAPER_OUTLINE.md`. References are flagged `[cite: ...]` — **none are invented**;
the author fills exact citations from the named prior art.

---

## Why four papers, not one

`../PAPER_OUTLINE.md` already sketches a single omnibus manuscript. That is the right
move for a *thesis-style* preprint, but it bundles four results with **different
audiences, different methods sections, and different acceptance bars** into one paper,
which weakens each. The verified material cleanly separates into four self-contained
contributions, ordered from most general / most defensible to most application-specific:

| # | Draft | Core verified claim | Primary audience | Strength of the floor |
|---|-------|---------------------|------------------|-----------------------|
| 1 | [`paper1_closure_theory.md`](paper1_closure_theory.md) | K-theory = the Markovian-delta collapse of the Mori–Zwanzig kernel; a projected-FDT closure restores backscatter + solenoidality (256² benchmark, transfer corr 1.000 vs Smagorinsky 0.071) | turbulence modelling, CFD, applied math | **Strongest.** Pure math + reproducible benchmark; no field data needed. |
| 2 | [`paper2_subglacial_melt_ceiling.md`](paper2_subglacial_melt_ceiling.md) | In grounded (Type-I) cavities, mean basal melt is ceilinged by the thermal conductive sublayer, *boundary-condition-robustly* (Robin wall + Navier slip ×100 leave `Nu/Nu_flat<1`); the suppression mechanism is an exact area-partition identity; the growth-carrying local lee flux is linear (not quadratic) and the cold-wall BC is certified by an interface coupling number `Λ≪St` at surge timescales | glaciology, ice–ocean, geophysical fluids | **Strong.** Reproducible DNS/LES; an honest, falsifiable *negative*. |
| 3 | [`paper3_scallop_parity_break.md`](paper3_scallop_parity_break.md) | Downstream scallop migration `Im(s)≠0` is a parity-symmetry break no eddy-diffusivity closure can produce (measured: `E_cos`→machine-zero under K-theory); constant-free field test `I=τ·c_mig/λ` | morphodynamics, ice/cave/dissolution patterns, turbulence | **Strong-in-solver.** One open gate to become "about the world" (Bushuk data). |
| 4 | [`paper4_subglacial_hydrology_forecasts.md`](paper4_subglacial_hydrology_forecasts.md) | A grounding-line intrusion number (RTN) verified *directionally* on Bedmap2 + BedMachine; the literal thermal sliding-lag kernel `H²/κ` **falsified** on 131 real lakes; memory reassigned to an exact MZ hydraulic kernel; a field-measurable `s_N(N)` master curve + three probes + `N_c` inversion + ungrounding early-warning, read honestly against the same real lakes; an intrusion residence number `Ro` | glaciology, remote sensing, sea-level | **Solid + honest.** Directional + a clean falsification + derived structure + a constructive, field-measurable law. |

Papers 2 and 3 can be merged into a single "subglacial cavity" paper if a journal
prefers one larger submission; the drafts are written so that merge is a copy-paste,
and each notes the seam.

See [`CONTACTS_AND_OUTREACH.md`](CONTACTS_AND_OUTREACH.md) for who to contact, in what
order, what to ask each person, the journal targets, and how to capture value
(authorship, collaboration, data unblocks, funding).

---

## Verification status (re-confirmed at draft time)

- **`pytest`: 410 passed, 4 warnings, in 468 s** (no GPU, synthetic + bundled data;
  includes the §I sliding-law cluster and this run's six new modules — intrusion
  residence number, the §G.6 local lee-flux law, the §A.3 dimensional bridge, the §A.2
  roughness closure, the §H.3 2-D CMN proxy, and the real-data §I lake test).
  Note: the README badge ("284") and CONTRIBUTING ("308") undercount — the live suite
  is **410**. Fix the badges before submission.
- **Closure benchmark `run_closure.py` (256², k_c=32) reproduces exactly:**
  - `RMS(∇·m)`: truth `7.41e-14`, Smagorinsky `1.77e-14`, **surrogate `1.20e+01`**
    (phase-randomisation breaks `∇·m=0`), projected-FDT `1.73e-14`.
  - transfer correlation with truth: Smagorinsky `0.071`, surrogate `0.907`,
    projected-FDT `1.000`.
  - Smagorinsky `T(k)≤0` everywhere (no backscatter); truth has backscatter (`T>0`).
- GPU-only results (RESULT 11 counter-gradient `C_G`; scallop migration exponent
  `0.52`; double-diffusion `η²` amplification) are tagged GPU-verified in the repo
  (Tesla P100, CuPy) and are reported as such; they were **not** re-run on this CPU box.
  Re-run them on a P100 via `../.agents/skills/testing-gpu-cupy/` before the camera-ready.

---

## What is deliberately NOT claimed in any draft

- No 3-D turbulence-regularity or universal-closure claim (Paper 1 is a 2-D periodic-box
  benchmark — a strict generalization in a named limit, not a "flawless" closure).
- No operational weather/climate/ice-sheet forecast skill (the Otis and hurricane-RI
  framings are motivation, explicitly `[HYP]`, and are quarantined out of Papers 1–4's
  result claims).
- No field-validated scallop discovery yet (Paper 3 is "verified in the solver"; the
  Bushuk `h(x,t)` reanalysis is the single step to make it "about the world").
- No RTN precision/recall, and no verified surge-lag value (Paper 4 is directional +
  a falsification + derived structure; the matched-lag test returns an honest null).
- No absolute `s_N` calibration (Paper 4 §7: the master-curve *shape* and the flotation
  threshold `N_c` are the robust claims; per-event `dN/N` is a lumped lower bound), and
  the ungrounding early-warning is demonstrated, not field-confirmed.
- No real-solver (ISSM/GlaDS) clock-mismatch test (Paper 1 §6.4 is a 2-D advection +
  moving-plume *proxy*; the real-solver test is deferred), and no field-calibrated
  scallop-roughness prefactor (Paper 4 §5.3 bounds it and shows the log law buffers a
  10× `z_0` uncertainty to ~1.7× in drag).
