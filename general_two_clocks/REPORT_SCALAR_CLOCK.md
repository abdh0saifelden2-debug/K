# RESULT 13 — `τ_c` is a turbulence clock, not a scalar clock
 
**Status:** **[VERIFIED, falsifiable]** — direct follow-up to RESULT 12 (§G.5/§D.4).
Harness `scalar_clock_universality.py` → `figures/54_scalar_clock_universality.json`
(fixed `Le=100`) and `figures/55_le_sweep_clock_vs_amplitude.json` (the `Le`-sweep);
test `tests/test_scalar_clock_universality.py` (4/4). CPU only; no new data, no GPU.
 
## The claim being tested
 
RESULT 12 measured `τ_c` as the decorrelation time of the SGS eddy diffusivity
`K_u = (c_s Δ)²|S̄|`. Because `K_u` is built from the **velocity** strain, the
structural prediction (§D.4, §G.5) is that **every passively-stirred scalar
inherits the same turbulent transport memory**: the eddy diffusivity a scalar `φ`
sees is `K_φ = K_u/Sc_t^(φ)`, and the turbulent Prandtl/Schmidt number only
*rescales the amplitude*, never the memory time. So `τ_c` should be a property of
the turbulence, identical across scalars — not a property of the individual scalar.
 
This is **falsifiable** in the double-diffusion solver
(`../glaciers/subglacial/candidate2_doublediff.py`), which advects two buoyancy-active scalars
with the *same* velocity field but a 100× molecular-diffusivity contrast:
 
- temperature `θ`, diffusivity `κ_T`;
- salinity `S`, diffusivity `κ_S = κ_T/Le`, `Le = 100`.
 
> **Prediction:** `τ_c(heat flux) ≈ τ_c(salt flux) ≈ τ_c(K_u)`.
> **Falsifier:** a systematic `τ_c(salt)/τ_c(heat) ≠ 1` scaling with the 100× `Le`.
 
**Design (why it isn't circular).** We measure the memory time of the **resolved
turbulent vertical flux** `F_φ(t) = ⟨v'φ'⟩` for heat and salt *separately*. This
depends on each scalar's *own* advected field, so equality is a non-trivial
outcome — it is **not** `K_φ = K_u/Sc_t`, whose memory time would be identical by
construction. `τ_c(K_u)` is also recorded in the same run to tie back to RESULT 12.
 
## Result
 
Three seeds, `Le = 100`, backscatter closure, salt-finger regime (`R_ρ = 2`):
 
| seed | `τ_c(heat)` | `τ_c(salt)` | salt/heat | `τ_c(K_u)` | xcorr(F_T,F_S) | `Nu_T` | `Nu_S` |
|---|---|---|---|---|---|---|---|
| 1 | 4.61×10⁻¹ | 4.60×10⁻¹ | 0.999 | 5.13×10⁻¹ | 1.000 | −11.2 | −1250.9 |
| 2 | 5.29×10⁻¹ | 5.27×10⁻¹ | 0.996 | 5.48×10⁻¹ | 1.000 | −9.0 | −1055.3 |
| 3 | 4.61×10⁻¹ | 4.60×10⁻¹ | 0.998 | 5.11×10⁻¹ | 1.000 | −4.6 | −586.0 |
 
**Verdict: `τ_c` is a turbulence clock.**
- **salt/heat memory ratio = 0.997 ± 0.001** — the two scalars share the same flux
  memory time to 0.3%, even though their molecular diffusivities differ by 100×.
- **flux cross-correlation = 1.000** — `F_T(t)` and `F_S(t)` co-vary essentially
  perfectly: the energy-containing turbulent flux is set by the velocity field,
  which is common to both scalars.
- **`Nu_S` ≈ 100·`Nu_T`** — the 100× contrast *is* physically present, but it shows
  up in the **transport-efficiency normalization** (`Nu_φ = 1 + F_φ/(κ_φ ∂_yC)`,
  and `κ_S = κ_T/100`), *not* in the memory time. The molecular diffusivity governs
  the fine-scale scalar structure and the Nusselt normalization; the velocity field
  governs the flux memory. (The negative `Nu` here is the counter-gradient /
  diffusive-convection sign of this long-spinup regime — RESULT 11's `C_G < −1`
  story — and does not affect the memory-time conclusion: the times match in the
  counter-gradient regime too.)
 
## Interpretation
 
The near-perfect agreement is the physics, not an artifact: at the energy-containing
scales the two scalars are stirred *identically* by one velocity field, so their
turbulent fluxes — and the memory of those fluxes — are the same. The 100× molecular
contrast acts only at the dissipation scales (and through the `κ` normalization in
`Nu`). Hence:
 
- the §G.5 commutator coefficient `τ_c` (the `K_u` autocorrelation time) and the
  §D.4 fast-bath time are **scalar-agnostic**: one value applies to heat, salt,
  suspended sediment, dissolved gas — any passively-stirred, buoyancy-active scalar,
  rescaled per scalar only by `Sc_t^(φ)`;
- this is why the unified-memory formalism can carry a *single* fast-bath time `τ_c`
  shared across all the scalars that feed the buoyancy/pressure coupling (§D.4),
  rather than one per scalar.
 
**Scope / honesty.** The *absolute* `τ_c ≈ 0.46–0.55` here is longer than RESULT 12's
`≈ 0.02–0.03`; that is expected — this is a different solver and regime (2-D
anisotropic finger cavity, long spinup, large slow overturning structures), and the
flux memory tracks the large-eddy turnover of *this* flow. The cross-solver absolute
value is **not** the claim; the claim is the **scalar-independence** of the memory
time within a run, which holds to 0.3%. The check is one regime (`Le=100`, `R_ρ=2`,
backscatter); the mechanism (one velocity field stirs all scalars) is regime-generic.
 
## Hardening: the `Le`-sweep separates the clock from the amplitude

The fixed-`Le=100` run above can be read two ways: τ_c really is scalar-independent,
or `F_T ≈ F_S` *trivially* because heat and salt start from the same ramp. The
decisive test is to **sweep the contrast `Le` itself** and watch two quantities that
must behave *oppositely* if the clock/amplitude split is real:

- the **clock** `τ_c(salt)/τ_c(heat)` must stay pinned at ~1 (it is the shared
  velocity field's), while
- the **amplitude** `Nu_S/Nu_T` must scale with `Le` (the contrast lives in the
  `κ`-normalized transport efficiency, `κ_S = κ_T/Le`).

Three decades of `Le`, 2 seeds each (`scalar_clock_universality.py --le-sweep`):

| `Le` | salt/heat `τ_c` | `Nu_S/Nu_T` | xcorr(`F_T,F_S`) |
|---|---|---|---|
| 1 | 1.000 ± 0.000 | 1.00 | 1.000 |
| 10 | 0.997 ± 0.001 | 11.3 | 1.000 |
| 100 | 0.997 ± 0.001 | 114.7 | 1.000 |
| 1000 | 0.997 ± 0.001 | 1148.1 | 1.000 |

**The clock is flat; the amplitude tracks `Le` almost exactly** (`Nu_S/Nu_T ≈ Le`
over all three decades). This rules out the triviality objection: `Le=1` is the
control (heat ≡ salt, both ratios = 1); as `Le` grows the scalars genuinely diverge
at small scales — `Nu_S/Nu_T` swings by ~1000× — yet the large-eddy flux *clock* is
untouched. If `τ_c` were a property of the scalar, salt's up-to-1000× lower
diffusivity would have to lengthen its memory and the ratio would trend with `Le`;
it does not. Amplitude side = molecular `κ` (and `Sc_t`); clock side = the shared
velocity field.

## Reproduce
 
```bash
python scalar_clock_universality.py            # ~3 min CPU -> figures/54_scalar_clock_universality.json
python scalar_clock_universality.py --le-sweep # ~5 min CPU -> figures/55_le_sweep_clock_vs_amplitude.json
python -m pytest tests/test_scalar_clock_universality.py -q
```
