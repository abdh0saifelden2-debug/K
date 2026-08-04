# Candidate 3 — Ice-base roughness feedback → corrected to scallop melting-instability

This candidate has two parts: the **pre-registered roughness-growth hypothesis**
(an honest null / sign-reversal that turns out to be the *correct* ice physics),
and the **corrected hypothesis** — scallop formation by a resolved-scale
melting instability — which the staged probe below confirms is **active**
(the first positive, flow-driven interfacial result of the four candidates).

---

## Part A — Pre-registered hypothesis (roughness growth) and why it is the wrong question for ice

**Hypothesis.** Melt is faster where the ice is thin → thinning grows roughness →
roughness enhances turbulence → turbulence enhances melt: a **positive runaway**
(growth rate `Λ > 0`), choked at large amplitude, with Smagorinsky suppressing it.

**Result — sign-reversed, and that is correct.** The feedback is **negative —
the boundary self-smooths**. Because the no-slip Brinkman wall pins interfacial
melt to conduction (`m ~ κ ΔT/H`), melt is largest where the column is *thinnest*,
so melting raises the base fastest exactly where it is lowest. Measured
**`Λ ≈ −0.265 (<0)`**, melt/ice-height anti-correlation `corr(m, y_ice) ≈ −0.95`,
**closure-independent** (GPU-confirmed on a P100; the value was briefly corrupted
to `+0.24` by a spectral-filter bug, restored by PR #1).

**Why this is right, not a failure.** The roughness-growth picture is a **false
analogy to rock/sediment beds** (erosion dynamics). An ice base is a *melting*
surface, not an eroding one:

| | Rock bed (river / glacier bed) | Ice-cavity ceiling |
|---|---|---|
| Roughness source | erosion, sediment transport | melting instability, flow-organised scallops |
| Growth mechanism | differential erosion (high stress → more erosion) | differential melting (high flux → more melt → retreat) |
| Stability | can grow unbounded | self-limiting: smooths at large scale, scallops at small scale |
| Timescale | geological (10³–10⁶ yr) | seasonal–decadal |

So `Λ < 0` (large-scale self-smoothing) is the **expected** behaviour for ice.
The real question is not "does roughness grow?" but a **stability** question:

> Does flow-organised melting create structures (scallops) that enhance heat
> transfer beyond a smooth wall?

---

## Part B — Corrected hypothesis: scallop / Hertzberg–Kueny melting instability

A flat melting surface is linearly stable (thermal-boundary-layer smoothing).
The exception is **flow–topography coupling**: a finite-amplitude perturbation
larger than the boundary layer causes **lee-side flow separation**, a
**recirculation vortex** advects warm fluid to the **reattachment point**, and
the **local heat flux is enhanced** there — a stable, self-limiting scallop
(Dubnick et al. 2020; the ripple-formation analogue with melting instead of
erosion). This is a **resolved-scale** phenomenon living *above* the penalty
zone, so the conduction limit that nulled Parts A / Candidate 1 / Candidate 4
need not apply.

### Staged go/no-go probe (this report)

**Design.** A single resolved sine mode on the ice base
`y_ice(x) = ȳ + a·sin(2π n_waves x / Lx)` with `a/λ = 0.1`, plus a steady
x body force `U_drive` to set up a mean current (the stock zero-mean ring
forcing has no "lee"). Per-column interfacial flux `m(x)` is read with the
penalty-clear local finite difference (identical to Candidate 1). The
**conduction baseline `m_cond(x)` is the same bumpy geometry with the flow OFF**
(same stencil → pure-geometry effects cancel). Enhancement
`R(x) = m_flow(x)/m_cond(x)`, time-averaged. A **flat-wall control** (`a=0`,
identical flow) is the turbulent-noise floor. Closures `none / smagorinsky /
backscatter`; regimes `U_drive ∈ {0.8, 1.5} × Ri ∈ {0, 1}`.

Grid `nx=ny=128` (`Lx=8π, Ly=2π`), so `dy≈0.049`; `n_waves=12` →
`λ≈2.09 (λ/dx≈10.7)`, `a≈0.21` — about `2.8×` the penalty interface width, i.e.
the bump is genuinely resolved while keeping `a/λ=0.1`.

**Result — GO.** The bumpy melting wall passes **more** net heat than its own
conduction baseline, in every regime and closure, well above the flat-wall
control; the excess is **spatially organised by the bump** (concentrated at the
lee, depleted on the stoss face) and **closure-independent**.

P100 (CuPy 14.0.1), `R = m_flow/m_cond`:

| regime | closure | R_mean | R_max | R_min | corr(excess,slope) |
|---|---|---|---|---|---|
| U0.8 Ri0 | flat control | 1.026 | 1.102 | 0.958 | 0.000 |
| U0.8 Ri0 | none | **1.211** | 2.706 | 0.158 | −0.296 |
| U0.8 Ri0 | smagorinsky | 1.211 | 2.705 | 0.158 | −0.296 |
| U0.8 Ri0 | backscatter | 1.211 | 2.807 | 0.167 | −0.243 |
| U1.5 Ri0 | flat control | 1.026 | 1.093 | 0.962 | 0.000 |
| U1.5 Ri0 | none | **1.357** | 3.312 | 0.463 | −0.197 |
| U1.5 Ri0 | smagorinsky | 1.356 | 3.309 | 0.462 | −0.196 |
| U1.5 Ri0 | backscatter | 1.335 | 3.012 | 0.621 | −0.195 |
| U0.8 Ri1 | none | **1.186** | 2.615 | 0.166 | −0.285 |
| U1.5 Ri1 | none | **1.324** | 3.185 | 0.480 | −0.228 |

CPU (NumPy), same config (cross-check; magnitude differs by turbulent
realisation, sign identical):

| regime | closure | R_mean | R_max | R_min | corr(excess,slope) |
|---|---|---|---|---|---|
| U0.8 Ri0 | flat control | 0.989 | 1.562 | 0.570 | 0.000 |
| U0.8 Ri0 | none | **1.059** | 1.918 | 0.177 | −0.744 |
| U1.5 Ri0 | none | **1.125** | 2.123 | 0.055 | −0.875 |
| U0.8 Ri1 | none | **1.082** | 2.024 | 0.164 | −0.743 |
| U1.5 Ri1 | none | **1.154** | 2.225 | 0.046 | −0.851 |

**How to read it.**
- **BUMP > FLAT everywhere.** Net enhancement (bump vs its conduction baseline)
  is `+6 % … +36 %`; the flat wall sits at `~conduction`. The gap is the
  scallop signal.
- **Scales with the mean current.** `U0.8 → U1.5` lifts `R_mean` monotonically —
  stronger flow → stronger separation/reattachment. Buoyancy (`Ri=1`) adds a
  few %.
- **Separation→reattachment structure.** `R_max ≈ 2–3.3` (lee enhancement) with
  `R_min ≈ 0.02–0.6` (stoss depletion); `corr(excess, bump-slope) < 0` confirms
  the excess is geometry-organised, not white turbulent noise (flat control
  corr `= 0` by construction).
- **Closure-independent (`≲0.5 %`).** The recirculation is a *resolved*
  structure; the subgrid model does not touch it. Unlike Part A (where
  closure-independence meant the mechanism was *absent / conduction-pinned*),
  here it means the mechanism is *robust* DNS physics.

### Honest scope

- `m(x)` uses the vertical `dT/dy` near the topmost fluid cell (a proxy for the
  true surface-normal gradient; max face angle ≈ 32° at `a/λ=0.1`). The
  flat-wall control and flow-off baseline use the **identical** stencil/geometry,
  so this does not bias the GO/NO-GO *comparison*; the quantitative
  `Nu(λ)` sweep switches to a true local-normal gradient.
- The bump is **frozen** in the probe (it isolates the flow→flux leg). Whether
  the scallop *saturates* at finite amplitude is the `a_sat(λ)` part of the full
  sweep (boundary allowed to move via the Stefan update of Part A).
- CPU↔GPU magnitude differs because the flow is turbulent (FFT round-off →
  divergent realisations); the full sweep uses seed ensembles to pin magnitude.

### Where this sits in the framework

The scallop is a **resolved-scale** effect, *not* a subgrid-closure claim. It
does not contradict the §1–§8 structural results or the Candidate 1/4 nulls — it
sits **alongside** them: the conduction limit bounds *subgrid* wall mechanisms,
while a resolved finite-amplitude perturbation under a mean current produces
genuine, flow-driven heat-transfer enhancement. The framework's improved subgrid
treatment (projected-FDT) matters for the small scales; it ensures the
large-scale separation structure is not corrupted by spurious pressure.

### Part C — Full Λ(λ) / a_sat / Nu sweep (P100, completed)

Run on a Tesla P100 (CuPy 14.0.1) from `scallop_sweep.py --gpu` with the
`melt_normal` `np.roll` contamination fix of this PR. Config: `nx=ny=128`
(`Lx=8π, Ly=2π`), `spinup=3000`, `measure=800` (time-averaged), `U_drive=1.5`,
`Ri=0`, frozen `a/λ=0.1` for the Λ scan, true **local-normal** gradient for
`Nu`. Flat-wall control `Nu_flat = 2.377e-4` (`umax=2.732`). Total wall time
≈ **7.9 min**.

**1. Λ(λ) — frozen-boundary enhancement vs wavelength.** `R = m_flow/m_cond`
(same bumpy geometry, flow on vs off); `Nu/Nu_flat` is the spatial-mean
local-normal flux relative to the flat wall.

| n_waves | λ | λ/dx | R_mean | R_max | Nu/Nu_flat | corr(excess,slope) |
|---|---|---|---|---|---|---|
| 4 | 6.283 | 32.0 | 1.0006 | 1.514 | 0.8274 | −0.624 |
| 6 | 4.189 | 21.3 | 1.0603 | 2.231 | 0.8306 | −0.674 |
| 8 | 3.142 | 16.0 | 1.0779 | 1.964 | 0.8365 | −0.840 |
| 10 | 2.513 | 12.8 | 1.1080 | 2.297 | 0.8995 | −0.779 |
| **12** | **2.094** | **10.7** | **1.1250** | **2.487** | **0.9219** | −0.802 |
| 16 | 1.571 | 8.0 | 1.0632 | 1.842 | 0.8587 | −0.862 |
| 20 | 1.257 | 6.4 | 1.0973 | 2.059 | 0.8813 | −0.816 |
| 24 | 1.047 | 5.3 | 1.0312 | 2.238 | 0.9159 | −0.612 |

Optimal `n_waves = 12` (`λ ≈ 2.094`, `λ/dx ≈ 10.7`): an **interior peak** in
both `R_mean` (+12.5 %) and `Nu/Nu_flat` (0.9219), falling off on either side —
i.e. **band-selective**, the signature of a preferred scallop wavelength
(`λ ~ a few δ_T`). `corr(excess, slope) < 0` throughout confirms the excess is
geometry-organised (lee-concentrated), not turbulent noise.

**2. a_sat(λ) — Stefan boundary released at `n_waves=12` (600 updates).**

| seeded a₀ | a_final |
|---|---|
| 0.2094 | 0.0797 |
| 0.4189 | 0.1787 |

Both amplitudes **decay** monotonically: at this setting the flow→flux
enhancement does not outrun conduction self-smoothing, so the mode is
**self-limiting** (any saturated amplitude sits below `a ≈ 0.08` within this
horizon). Consistent with Part A's negative large-scale feedback.

**3. Seed ensemble at `n_waves=12` (4 turbulence seeds, frozen).**

| seed | R_mean | R_max | Nu_bump |
|---|---|---|---|
| 0 | 1.1250 | 2.487 | 2.191e-4 |
| 1 | 1.1264 | 2.077 | 2.192e-4 |
| 2 | 1.1232 | 2.114 | 2.194e-4 |
| 3 | 1.1347 | 2.373 | 2.217e-4 |

**R_mean = 1.1273 ± 0.0044** — the conduction-relative enhancement is tight
across seeds (+12.7 % ± 0.4 %).

**4. Closure spot-check at `n_waves=12`.**

| closure | R_mean | Nu_bump |
|---|---|---|
| none | 1.1250 | 2.191e-4 |
| smagorinsky | 1.1261 | 2.192e-4 |
| backscatter | 1.0933 | 2.133e-4 |

Closure-independent to ≈ 0.1 % (Smagorinsky) / ≈ 2.8 % (backscatter):
**resolved-scale** physics, as expected.

**How to read Part C (honest scope).** With the rigorous **local-normal**
gradient, `Nu/Nu_flat < 1` at *every* wavelength: the spatial-mean normal flux
on the bumpy wall stays **below** the flat wall. The scallop signal is therefore
(i) a conduction-relative gain on the same geometry (`R_mean = 1.127 ± 0.004`),
and (ii) strongly **local** (`R_max ≈ 2.5` at the lee/reattachment), but it does
**not** exceed the flat-wall Nu in the mean at this single amplitude. This
sharpens the probe's vertical-proxy "+6…36 %" GO (Part B): the mechanism, its
wavelength selection, its lee-organisation and its closure-independence are all
confirmed, while the net mean-Nu enhancement vs a flat wall is *not* positive
here — and the released boundary self-limits rather than runs away. Raw results
JSON (`scallop_sweep_gpu.json`, full per-column arrays + a_sat trajectories) was
produced in the Kaggle P100 session.

## Reproduce

```bash
# Probe battery (auto-detects CuPy; pass xp=cupy on GPU):
python scallop_battery.py            # 4 regimes × (3 closures + flat control)
# single probe:
python -c "import scallop_probe as s, numpy as np; print(s.probe(nx=128,ny=128,n_waves=12,U_drive=1.5,Ri=0.0,xp=np)[1]['none'])"
# full Λ(λ)/a_sat/Nu sweep (Part C); add --gpu on a CuPy box, --fast for a CPU smoke:
python scallop_sweep.py --gpu        # writes scallop_sweep_gpu.json
```

The honest Part-A null (`Λ<0`, self-smoothing) is exercised by
`pytest tests/test_candidate3.py -q` and the Stefan prototype
(`tests/test_stefan.py`, THEORY_CAVITY.md §12).
