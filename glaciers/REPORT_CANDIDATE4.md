# Candidate 4 — Hydraulic Switching (2D anisotropic, body-force tidal drive)

## Hypothesis (pre-registered)

A wide, shallow subglacial cavity (`Lx = A·2π`, `Ly = 2π`) under a tidal body
force can sit in two states: **filled** (flow reaches the ice base, the active
layer `H1 → 1`, high interfacial transport) and **stratified** (a thin basal
layer with a stagnant upper region, `H1 → 0.2`, low transport). The prediction
was that the *time-averaged melt* should peak at an **intermediate** buoyancy
coupling `Ri`, where the cavity switches most often between the two states
(switching frequency `f_switch` peaks at intermediate `Ri`), and that
Smagorinsky K-theory should over-dissipate the transitions and suppress the
peak.

## What the run actually shows

Sweep: `nx=128, ny=64, A=4`, `Ri ∈ {0, 0.25, 0.5, 0.75, 1.0, 2.0}`,
closures `{none, smagorinsky, backscatter}`, spinup 1500 + measure 1500 steps
(CPU/NumPy). Figure: `figures/candidate4_hydraulic_switch.png`; raw numbers in
`figures/candidate4_hydraulic_switch.json`.

1. **The dead-flow bug was numerical, not physical.** The earlier
   configuration under-resolved the cavity (`ny=48`, ~10 fluid rows), so the
   Brinkman penalty bled into the fluid gap and divided the velocity by ~1.8
   every step (`umax ≈ 0.006`). With the cavity resolved (`ny≥64`, cavity
   filling most of `Ly`) the flow develops normally (`umax ≈ 0.6`, KE up
   ~500×). Candidate 4 is fixable, not null-by-construction.

2. **No switching / no hump.** At every `Ri` the cavity stays **monostable
   filled**: `H1 ≈ 0.90 ± 0.01`, `f_switch = 0`. The bistable filled⇄stratified
   regime — and therefore the predicted intermediate-`Ri` melt hump — **did not
   appear** in the explored drive parameters. This is the honest null for the
   *switching* hypothesis as posed; realizing bistability would need a regime
   the body-force drive did not reach here (stronger/slower tidal pumping or a
   genuinely two-layer initial state), and is not demonstrated.

3. **Interfacial melt is conduction-limited (flat).** The interfacial melt
   `m = ⟨-κ ∂θ/∂y⟩` is `2.144e-4` to four figures across *all* `Ri` and *all*
   closures. The no-slip Brinkman wall pins transport at the interface to
   molecular conduction, so the bulk flow cannot move the wall gradient. This
   is the same scope boundary documented for the Stefan prototype — the
   flow→melt leg is inactive when the observable is the conductive wall flux.

4. **The flow-dependent observable is the turbulent heat flux**
   `Fturb = ⟨v'θ'⟩` over the cavity interior (added as a first-class
   diagnostic). Unlike the wall melt, `Fturb`:
   - **rises monotonically with `Ri`** (buoyancy drives vertical transport):
     for the unclosed run `4.89e-4 → 5.71e-4` as `Ri: 0 → 2`;
   - **is systematically suppressed by Smagorinsky** relative to the unclosed
     run at every `Ri` (≈1.6–1.7% lower) — the K-theory over-dissipation
     signature the project predicts.

   | closure | Fturb @ Ri=0 | Fturb @ Ri=2 |
   |---|---|---|
   | none (unclosed) | 4.887e-4 | 5.705e-4 |
   | smagorinsky | 4.808e-4 | 5.610e-4 |
   | backscatter | 4.476e-4 | 5.263e-4 |

## Honest scope

- The closure separation here is **small (~1.7%)** because at `nx=128, ny=64`
  with this viscosity the flow is only weakly under-resolved, so the SGS model
  is not the dominant dissipation. The project's own criterion is that closures
  decide the answer only in a genuine under-resolved LES (`n ≥ 128`, low
  molecular viscosity). The **direction** matches the two-clocks prediction
  (K-theory transports less); the **magnitude** is a higher-resolution GPU job,
  not settled here.
- In this 2D, modestly-resolved, monostable regime the **backscatter** closure
  suppresses `Fturb` *further* than Smagorinsky (its net spectral eddy
  viscosity outweighs the stochastic injection). This is reported as-is; the
  "backscatter restores transport" claim from the a-priori 8b benchmark is
  **not** reproduced in this a-posteriori 2D cavity and should not be inferred
  from this run.
- This is a mechanism probe, not a validated melt law. Sliding, Stefan
  feedback, salinity, and 3D are all deferred to the other candidates / future
  work.

## Reproduce

```bash
# CPU (this report):
python run_candidate4.py --nx 128 --ny 64 \
    --ri 0.0,0.25,0.5,0.75,1.0,2.0 --aspect 4 \
    --closures none,smagorinsky,backscatter \
    --spinup 1500 --measure 1500 --sample-every 10 \
    --out-dir figures --report REPORT_CANDIDATE4.md

# GPU LES (decisive closure separation; auto-detects CuPy on Kaggle/Colab):
python run_candidate4.py --nx 384 --ny 96 \
    --ri 0.0,0.25,0.5,0.75,1.0 --aspect 1,4 \
    --closures none,smagorinsky,backscatter \
    --spinup 4000 --measure 6000 --out-dir figures
```

Unit tests: `pytest tests/test_candidate4.py -q` (8 CPU tests).
