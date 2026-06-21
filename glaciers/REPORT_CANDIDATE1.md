# Candidate 1 — Intermittent plumes from ice-base roughness

## Hypothesis (pre-registered)

Bumps on the ice base protrude through the thermal boundary layer; where a crest
dips into warm water a localised plume of enhanced melt forms. Over a rough base
the interfacial-melt field was predicted to be **intermittent** — skewness
`S > 0`, excess kurtosis `K > 3`, peak-to-mean `> 2` — **humped at intermediate**
`Ri ~ 0.3–0.6` (plumes frequent-but-weak at low `Ri`, suppressed at high `Ri`),
with Smagorinsky over-dissipating the bursts and lowering peak/mean.

## What the run shows

Sweep: `nx=128, ny=64, A=4`, rough ice base `y_ice(x)=5.5+h(x)` with band-limited
`a_k ~ k^-1.25` roughness (`sigma_h=0.30`, modes 2–8), `Ri ∈ {0,0.25,0.5,0.75,1,1.5}`,
closures `{none, smagorinsky, backscatter}`, spinup 1200 + measure 1200 (CPU).
Figure `figures/candidate1_plumes.png`, raw `figures/candidate1_plumes.json`.

1. **The interfacial-melt intermittency is geometric, not plume-driven, and the
   prediction's thresholds are not met.** Pooled over space and the measurement
   window the melt field has `peak/mean ≈ 1.8` (not `> 2`), `skew ≈ −0.75` (not
   `> 0`), `kurt ≈ 2.2` (not `> 3`). Crucially these statistics are **flat in
   `Ri`** (relative spread `< 2%` across the whole sweep) and **closure-
   independent** — there is no hump. The variability that does exist is set by
   ice-thickness geometry (thinner columns conduct faster), not by convective
   plumes: switching the flow **off entirely** (`f_amp=0`, pure conduction)
   reproduces the same `melt_mean` and `peak/mean` to ~1%.

2. **The melt is conduction-limited.** `melt_mean ≈ 2.17e-4` is constant across
   all `Ri` and all closures (4 significant figures). The no-slip Brinkman wall
   pins the interfacial flux to molecular conduction, so bulk plumes cannot move
   the near-wall gradient — the same scope boundary documented for the Stefan
   prototype (THEORY_CAVITY.md §12) and Candidate 4.

   > Diagnostic note: the melt is read with a *local* finite difference in a
   > window starting 3 cells below each column's local ice base (clear of the
   > penalty). An earlier global-spectral `∂θ/∂y` read produced a spurious
   > `peak/mean ≈ 20` and a sign-flipping skew — Gibbs ringing against the sharp
   > penalty interface, not physics. The local FD removes that artifact.

3. **The flow-dependent, closure-sensitive observable is `Fturb = ⟨v'θ'⟩`.** It
   rises monotonically with `Ri` (`5.1e-4 → 8.9e-4` as `Ri: 0 → 1.5`) and
   **Smagorinsky sits just below the unclosed run at every `Ri`** — the K-theory
   over-dissipation signature (same direction as Candidate 4).

   | closure | Fturb @ Ri=0 | Fturb @ Ri=1.5 | peak/mean (flat) |
   |---|---|---|---|
   | none | 5.123e-4 | 8.905e-4 | 1.80 |
   | smagorinsky | 5.051e-4 | 8.823e-4 | 1.80 |
   | backscatter | 5.204e-4 | 8.940e-4 | 1.88 |

## Honest scope

- **The plume hypothesis is not confirmed by the interfacial melt** in this 2-D
  penalised, conduction-limited regime: no positive skew, no `peak/mean > 2`, no
  intermediate-`Ri` hump. Genuine plume intermittency needs the wall flux to be
  turbulence-controlled (`q ~ u_* ΔT`, `Re ≫ 1`) rather than conduction-pinned —
  i.e. a resolved high-`Re` rough-wall boundary layer, which this regime (and the
  Brinkman no-slip wall) does not provide.
- The `Fturb` separation between closures is small (~1.5%) for the same reason as
  Candidate 4: at `nx=128` the flow is only weakly under-resolved. The
  *direction* matches the two-clocks prediction; the *magnitude* is a
  higher-resolution GPU job (the runner auto-detects CuPy).
- This is a mechanism probe, not a melt law.

## Reproduce

```bash
# CPU (this report):
python run_candidate1.py --nx 128 --ny 64 \
    --ri 0.0,0.25,0.5,0.75,1.0,1.5 --aspect 4 \
    --closures none,smagorinsky,backscatter \
    --spinup 1200 --measure 1200 --sample-every 10 \
    --out-dir figures --report REPORT_CANDIDATE1.md

# GPU (higher resolution; auto-detects CuPy):
python run_candidate1.py --nx 384 --ny 96 \
    --ri 0.0,0.25,0.5,0.75,1.0,1.5 \
    --closures none,smagorinsky,backscatter \
    --spinup 2000 --measure 4000 --out-dir figures
```

Unit tests: `pytest tests/test_candidate1.py -q` (7 CPU tests).
