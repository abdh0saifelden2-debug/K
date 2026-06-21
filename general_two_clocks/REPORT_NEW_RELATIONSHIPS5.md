# New cross-relationship — NR28 (`general_two_clocks/new_relationships5.py`)

Continues the derived-and-verified program (NR1–NR27; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, deterministic; unit-proofs in
[`tests/test_new_relationships5.py`](tests/test_new_relationships5.py). Run:

```bash
python general_two_clocks/new_relationships5.py    # -> figures/nr28_two_clocks_cross_spectrum.{json,png}
pytest general_two_clocks/tests/test_new_relationships5.py -v
```

---

## NR28 — The two clocks in one cross-spectrum: elliptic (instantaneous) pressure *leads* parabolic (lagged) temperature by `arctan(ω/ω_c)`, the 45° crossover *is* the parabolic clock `ω_c=κk²`, and the p–T coherence *drops* at high frequency — the decoupling itself  [the foundational two-clocks thesis as a co-located-sensor measurement]

### The gap this closes

The repo's **core thesis** (README "Core Thesis"; `REPORT_RB`, `REPORT_THEORY`,
`REPORT_TIMESCALE_SEPARATION`) is that pressure and temperature run on *different clocks*
and obey *different spatial operators*: incompressible pressure is a **global, elliptic**
field — the Leray/Poisson solve `p ~ −Δ⁻¹∂ᵢ∂ⱼ(uᵢuⱼ)` is an *instantaneous* diagnostic of
the strain — whereas temperature is a **local, parabolic** field — the prognostic heat
equation carries *memory* of past straining through a diffusive integral. The whole NR
program (NR1–NR27) has mined the **closure** (memory) and **fold** (criticality)
consequences of this split, but never wrote down its most direct **observable**: the
cross-spectrum between a co-located pressure proxy and temperature proxy. NR28 supplies it,
turning the foundational claim into a single, calibration-free measurement on the kind of
co-located p+T time series NEON/ASOS/reanalysis already provide.

### Setup (one representative mode `k`, both fields driven by the same straining)

A diagnostic (elliptic) pressure responds **instantaneously** to the common turbulent
driver `d(t)`; the prognostic (parabolic) temperature responds through a **first-order
diffusive lag** with corner `ω_c = κk²`, and additionally carries its **own** small-scale
filamentation that the global pressure does not (`REPORT_RB`: "temperature is torn into
filaments"):

> `p(t)      = d(t) + n_p`              (elliptic: instantaneous diagnostic)
> `θ̇(t)      = −ω_c θ + d(t)`, `+ n_θ`   (parabolic: lagged + independent filaments)

so the transfer is `H_θ(ω) = 1/(ω_c + iω)` and the two observables of the **same** sensor
pair are the cross-spectrum `S_{p,θ}(ω)` and the magnitude-squared coherence `γ²(ω)`.

### Derived consequences (each verified against the simulated processes)

**(a) The cross-spectral phase IS the parabolic clock — read with no calibration.**
Because `p` tracks the driver instantaneously and `θ` lags it through `H_θ`,

> **`φ(ω) = arg S_{p,θ}(ω) = arctan(ω/ω_c)`**,

rising `0 → 90°` with the **45° crossover EXACTLY at `ω = ω_c = κk²`**. A co-located
`(p, θ)` pair therefore **measures the parabolic clock from the phase crossover alone** —
no diffusivity calibration, no amplitude calibration, no noise model.

**(b) The coherence DROPS at high frequency → the clocks decouple at small scales.** The
common-driver coherence

> `γ²(ω) = |S_{p,θ}|² / (S_pp S_θθ)`

is `≈ 1` at low `ω` (both fields quasi-statically track the driver: clocks coupled) and
**falls** at high `ω`, because the parabolic low-pass shrinks temperature's common content
(`|H_θ|² ~ ω⁻²`) until its independent filamentation dominates. The coherence half-fall is
a **second, independent** read of the decoupling scale — and the drop *is* the
small-scale decoupling the core thesis asserts.

**(c) One-sided, bounded causality distinguishes a diffusive clock from a transport delay.**
The phase is **positive for all `ω`** (pressure leads, never lags: the elliptic field is
the instantaneous diagnostic, the parabolic field the laggard) and is **bounded by 90°** —
the signature of a pure first-order lag, distinguishing the diffusive clock from a pure
transport delay (which would give an unbounded linear phase `ωτ`).

### Numerical verification (`figures/nr28_two_clocks_cross_spectrum.{json,png}`)

Common white driver `d`; instantaneous `p = d + n_p`; first-order-lagged `θ` (exact sampled
ODE, `a=e^{−ω_c dt}`) + filament noise; `ω_c = 0.5`, `dt = 0.02`, `n = 4×10⁶`, Welch
`nperseg = 16384`. Cross-spectral phase `φ = −arg S_{p,θ}` (scipy conjugate convention) and
coherence on the same grid. The continuous-time law `arctan(ω/ω_c)` is validated over the
**coherent band** (`0.1 ω_c < ω < 8 ω_c`, `γ² > 0.1`) where the sampled lag still tracks
continuous time (Bendat & Piersol: phase estimates are meaningful only where coherence is
non-negligible).

| Claim | Predicted | Measured |
|---|---|---|
| (a) phase law `φ(ω)=arctan(ω/ω_c)` | exact | coherence-weighted RMS **0.035 rad** (≈2°) over the band |
| (a) parabolic clock from the 45° crossover | `ω_c=0.5` | **0.514** (2.8 %), interpolated crossover |
| (a) clock from the phase fit `tan φ = ω/ω_c` | `ω_c=0.5` | **0.506** (1.2 %), from the phase alone |
| (b) coherence at large scales (low `ω`) | `0.90` | **0.89** (clocks coupled) |
| (b) coherence at small scales (high `ω`) | `0.06` | **0.05** (clocks decoupled) |
| (b) coherence drops | yes | `γ²_high < 0.6 γ²_low` ✓ |
| (c) one-sided (pressure leads ∀ω) | yes | **100 %** of coherent bins `φ>0` |
| (c) bounded by 90° (diffusive, not transport) | `<90°` | p98 `φ` = **83°** `< 90°` ✓ |

All five check groups pass (`ok = true`).

### Why it matters

NR28 is the **foundational** entry of the program, arriving late on purpose: every earlier
NR mined a *consequence* of the two-clocks split (closure memory, the fold, the FDT); NR28
writes the split itself as **one cross-spectral signature** that an experimentalist can
read off two co-located sensors:

- a **bounded, one-sided phase** `arctan(ω/ω_c)` whose **45° crossover measures the
  parabolic clock** `ω_c=κk²` with no calibration — the elliptic field is the phase
  reference because it is the instantaneous diagnostic;
- a **high-frequency coherence drop** that *is* the small-scale decoupling of the two
  clocks — temperature surrenders its common content to its own filamentation while
  pressure stays globally slaved to the strain.

It connects the abstract operator split (elliptic Poisson vs parabolic heat) to the
**field-spectroscopy** language the repo already uses for ice (`s_N` admittance, CSD early
warning) and for turbulence (NR8 Lorentzian, NR27 FDT), and it does so on exactly the
co-located pressure+temperature records that NEON/ASOS/reanalysis already publish.

**Mainstream tools (cited, not claimed):** Bendat & Piersol (*Random Data*, cross-spectral
phase & coherence); Leray (1934, the pressure Poisson/Leray projection — elliptic,
instantaneous); the heat equation (parabolic, memory-carrying). The contribution is the
*identification* of the repo's two-clocks split with the bounded one-sided phase
`arctan(ω/ω_c)` and the coherence drop on a single co-located sensor pair.
