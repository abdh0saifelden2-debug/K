# RESULT 20 — the §H.1.2 driver estimator is convergent, isotropic, and advects the RTN=1 line

**Status:** the §H.1.2 level-set advance law `v_front = (dH/dt)/|∇m|`,
`A = 1/|∇m|`, was already certified synthetically by `glmig_synthetic`
(RESULT 16). This result hardens that certification at the level of the **real
driver code path**: `validation/synthetic/rtn_intrusion_clock_synthetic.py`
imports and exercises the *actual* driver functions `margin_field` /
`amplification` from `rtn_intrusion_clock.py` (the ones `analyse` calls) on
planted geometries with closed-form kinematics, and shows the discrete `A` map is
**machine-precision exact** on a planar margin, **second-order convergent** on a
curved one, **isotropic** around a curved (radial) front, **recovers the planted
front advance**, and **advects exactly the §H.1.1 RTN=1 line**. Tests in
`tests/test_validation_synthetic.py` (6/6). No external data; no GPU.

## What was already covered, and what was open

`glmig_synthetic` (RESULT 16) validates the §H.1.2 *advance law itself*: a 1-D
contour-tracking speed `r/b`, a 2-D tilted-plane normal speed
`r/√(bx²+by²)`, and the `Ro = v_kin/v_obs` discriminant the §H.1.3/§H.1.4
real-data slopes are read against. That harness, however, **reimplements the
gradient inline in 1-D** (`np.gradient(b*x, x).mean()`); it never runs the *actual
2-D driver functions* that execute on Bedmap2. So a regression in the real code
path — a broken `dx`-in-km scaling, a changed gradient stencil, or mishandled
grounded-mask `NaN`s — would pass `glmig_synthetic` untouched. And three numerical
properties of the 2-D estimator were never checked anywhere: its **convergence
order**, its **isotropy** on a curved front, and the **identity of the advected
object with the §H.1.1 RTN=1 line**. Those gaps are what this result closes.

## What this result establishes

The harness reuses the **real driver math** (`margin_field`, `amplification` from
`rtn_intrusion_clock` — the inline gradient block in `analyse` was refactored into
`amplification` with no behaviour change, so the field driver and this calibration
are now provably one function):

1. **Exact on a planar margin.** For `m = s·x` (slope `s` m thickness/km) the
   central-difference `|∇m|` is exact, so `A = 1/s` and `v_front = (dH/dt)/s` come
   back with **0.0 max error** and equal the analytic front speed
   `dx_f/dt = (dH/dt)/s` cell-for-cell across `dH/dt = 0.5–5 m yr⁻¹` — confirming
   the real function's `dx`-in-km scaling.
2. **Second-order convergent on a curved margin.** For a smooth non-polynomial
   margin `m = s·x + b·sin(kx)` the max error in `A` falls **5.5×10⁻⁴ → 1.4×10⁻⁴
   (ratio 3.99 ≈ 4)** when the grid spacing halves (1 → 0.5 km) — the textbook
   second-order rate toward the analytic `1/|s + bk·cos(kx)|`. (glmig checks 1-D
   exactness at a single resolution; the *rate* was untested.)
3. **Isotropic on a radial front.** For `m = s·(r − r₀)` the RTN=1 line is a circle
   with analytic `|∇m| = s` everywhere. Around the ring (1516 cells) the recovered
   `A` has **median 0.2000 = 1/s** (rel. err ≤ 2×10⁻⁴), coefficient of variation
   **2.5×10⁻⁵**, and the axis-vs-diagonal median differs by **6.6×10⁻⁵** — the
   Cartesian stencil introduces no directional bias on a curved front (glmig's 2-D
   test is analytic with constant gradient — no grid, no curvature).
4. **Recovers the planted front advance.** Thinning by `ΔH` moves the analytic
   RTN=1 line to the zero set of `m − ΔH`; advancing each front cell inland by the
   estimated displacement `A·ΔH` lands on that new line with **0.0 km** position
   error (planar, exact), and the thinned-margin mask boundary sits within **one
   cell** of the analytic new front.
5. **The advected object is the §H.1.1 RTN=1 line.** On a planted `(H, bed)`
   population `margin_field` gives `H* = H_flot/φ` to **2×10⁻¹³** and
   `margin < 0 ⇔ RTN > 1` (`classify(build_rtn)`) with **0 mismatched cells** at
   φ ∈ {0.80, 0.90, 0.96} — so the clock advances exactly the line the static
   φ-area / MISI-band results key off. (glmig never connects to the real RTN
   classifier.)

## Why this matters

`glmig_synthetic` certifies the *law*; this certifies the *implementation* that
runs on real data, plus the discretization behaviour (convergence, isotropy) and
the RTN-line identity. Together they mean that when the [HYP] residence-number test
`Ro = v_kin/v_obs` is read off Bedmap2 + altimetry (§H.1.3/§H.1.4), a departure of
`Ro` from 1 can be attributed to physics (hydraulic pacing) rather than to a biased
or mis-scaled `v_kin` from the driver. This is the code-path-level companion to the
RESULT-16 law calibration and the RESULT-18/19 φ-area calibration.

## Honest scope

This validates the **driver estimator (the kinematics)**, not the physics, and it
does **not** supersede `glmig_synthetic` — it complements it. It does not test the
#5 hydraulic-pacing conjecture (`Ro ≫ 1`), which still requires observed
front-migration rates, nor does it assert any real advance speed for Antarctica.
The planted geometries are analytic and noise-free: they certify discretization
behaviour (exactness, second-order convergence, isotropy, advance recovery) and the
RTN-line identity, not robustness to data gaps, sub-grid bed roughness, or the
`d_base = max(0, −bed)` approximation in the real driver. The guarantee is
structural: *given* the level-set definition, the real driver's `A = 1/|∇m|` and
`v_front = A·(dH/dt)` are a convergent, isotropic, unbiased readout acting on the
§H.1.1 RTN=1 line.
