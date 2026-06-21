# Candidate 2 — Double-diffusive layering (salt + temperature)

## Hypothesis (pre-registered)

Heat diffuses ~100× faster than salt. With cold-fresh meltwater over warm-salty
ocean, the differential diffusion drives **salt fingers** that enhance heat
transport. The thermal Nusselt number `Nu_T` was predicted to be **humped**,
peaking near the density ratio `R_rho ≈ 2` (the finger sweet spot) and falling
to ~1 for `R_rho < 1` (diffusive convection) and `R_rho ≳ Le` (stable), with
Smagorinsky over-dissipating the fingers and lowering the peak.

## What the run shows

Sweep: `nx=128, ny=64, A=4`, `Le=100` (`kappa_S=kappa_T/100`), `Ri_T=2`,
`f_amp=0.2`, `R_rho ∈ {0.5,1,1.5,2,2.5,3,5,10}`, closures `{none, smagorinsky,
backscatter}`, spinup 1500 + measure 800 (CPU). Figure
`figures/48_candidate2_doublediff.png`, raw `figures/48_candidate2_doublediff.json`.

1. **No finger hump — `Nu_T` decreases monotonically with `R_rho`.** Across the
   whole sweep `Nu_T` falls smoothly from `1.44` at `R_rho=0.5` to `0.73` at
   `R_rho=10`; `R_rho=2` is **not** a maximum. Stabilising salinity simply
   suppresses convective heat transport. The predicted salt-finger sweet spot
   does not appear in this 2-D forced penalised regime.

2. **The diffusivity contrast shows up robustly: `Nu_S >> Nu_T`.** Because salt
   diffuses ~100× slower, its transport is almost entirely advective, so the
   haline Nusselt number is ~30× the thermal one (`Nu_S ≈ 45` vs `Nu_T ≈ 1.4`
   near `R_rho=1`). This is the genuine double-diffusive signature the cavity
   *does* reproduce.

3. **Counter-gradient transport at strong stabilisation (K-theory-invisible).**
   At `R_rho=10` both `Nu_T` and `Nu_S` drop below 1 / go negative
   (`Nu_S ≈ −26`): the buoyancy-arrested flow carries heat/salt *against* the
   mean gradient — exactly the counter-gradient flux a positive eddy diffusivity
   cannot represent (the recurring two-clocks theme). Backscatter resists this
   collapse (`Nu_T=0.86` vs `0.73` unclosed at `R_rho=10`); `none` and
   Smagorinsky are essentially identical elsewhere.

4. **The Turner flux ratio scales as `gamma ≈ 1/R_rho`.** Since the same
   velocity advects two near-identical scalar ramps, `F_T ≈ F_S`, so
   `gamma = F_T/(R_rho F_S) ≈ 1/R_rho` (`1.98, 0.99, 0.49, …`), confirmed to
   ~1%.

| closure | Nu_T(0.5→10) | Nu_S(0.5→10) | verdict |
|---|---|---|---|
| none | 1.440 → 0.728 | 45.6 → −26.5 | monotonic, no hump |
| smagorinsky | 1.438 → 0.732 | 45.4 → −26.1 | monotonic, no hump |
| backscatter | 1.435 → 0.862 | 45.1 → −12.9 | monotonic, no hump |

## Honest scope

- **The finger-hump hypothesis is not confirmed here.** Salt fingers grow from a
  *small-scale differential-diffusion instability* of an otherwise stably
  stratified column. In this setup both scalars are seeded with the same ramp,
  so the net stratification is stable for `R_rho>1`; the finger mode is weak and
  is swamped by the forcing-driven turbulence that advects both scalars
  together. Resolving fingers needs the finger scale
  `~(ν κ_T / g α dT/dy)^{1/4}` (a few cells), **low external forcing** (so the
  buoyancy drives the flow), and a **long** integration for the fingers to
  organise — a high-resolution GPU job (the runner auto-detects CuPy; see the
  GPU recipe below).
- What the CPU sweep *does* establish honestly: the diffusivity-contrast
  signature (`Nu_S >> Nu_T`), the monotonic suppression of heat transport by
  stabilising salt, the counter-gradient regime at large `R_rho`, and the
  `gamma ≈ 1/R_rho` Turner scaling.
- This is a mechanism probe, not a melt/mixing law.

## Reproduce

```bash
# CPU (this report):
python run_candidate2.py --nx 128 --ny 64 \
    --rrho 0.5,1.0,1.5,2.0,2.5,3.0,5.0,10.0 --aspect 4 \
    --ri-t 2.0 --f-amp 0.2 --closures none,smagorinsky,backscatter \
    --spinup 1500 --measure 800 --out-dir figures --report REPORT_CANDIDATE2.md

# GPU (finger-resolved attempt: high res, low forcing, long spinup):
python run_candidate2.py --nx 384 --ny 96 \
    --rrho 0.5,1,1.5,2,2.5,3,5,10 --ri-t 3.0 --f-amp 0.05 \
    --closures none,smagorinsky,backscatter \
    --spinup 6000 --measure 4000 --out-dir figures
```

Unit tests: `pytest tests/test_candidate2.py -q` (5 CPU tests).
