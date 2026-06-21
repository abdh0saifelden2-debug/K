# New derived cross-relationships NR16–NR21

> Continues the cross-relationship program (NR1–NR15) using the four new
> observational papers as anchors. Each relationship is derived from mainstream
> theory + a repo result and **numerically verified**:
> [`new_relationships.py`](new_relationships.py),
> [`tests/test_new_relationships.py`](tests/test_new_relationships.py) (11 tests).

## NR16 — Lowen–Teich: power-law bursts are a 1/f^γ spectrum, γ = 3 − α  [Paper 3]

A telegraph/on–off process whose sojourn times are power-law `p(τ) ∝ τ^{-α}`
(1<α<3) has a low-frequency power-law spectrum `S(f) ∝ f^{-γ}` with **γ = 3 − α**
(Lowen & Teich, fractal point processes). So Paper 3's scale-free *pressure* bursts
(α ≈ 2.0) are the time-domain face of a **1/f** (γ ≈ 1) spectrum.

Verified (averaged PSD of a synthetic power-law telegraph):

| α | measured γ | predicted 3−α |
|---|---|---|
| 1.5 | 1.44 | 1.50 |
| 2.0 | 1.09 | 1.00 |
| 2.5 | 0.76 | 0.50 |

γ tracks 3−α (max abs error 0.26 over the band), confirming the burst-statistics ↔
spectrum link.

## NR17 — Power-law spectrum ⇒ growing temporal memory  [Paper 3 → Paper 4]

By Wiener–Khinchin a `1/f^γ` spectrum (γ>0) has a slowly-decaying (non-summable for
γ≥1) autocorrelation, so the **integrated memory time** `τ_int = 1 + 2 Σ C(τ)` grows
monotonically with γ, while white noise (γ=0, Markov) has `τ_int ≈ 1`. Chained with
NR16 this links Paper 3's bursts to Paper 4's non-Markovian memory: *scale-free bursts
⇒ power-law spectrum ⇒ long temporal memory.*

| γ | recovered slope | τ_int |
|---|---|---|
| 0.0 | 0.06 | 1.06 |
| 0.5 | 0.56 | 8.14 |
| 1.0 | 1.06 | 38.4 |
| 1.5 | 1.56 | 59.1 |

Spearman(γ, τ_int) = **1.00**.

**Honesty note.** A first attempt used the lag-2 Chapman–Kolmogorov excess
`C(2)−C(1)²` as the memory measure; it is **non-monotone** in γ (it vanishes for both
white noise *and* very strong memory, because C(1)→1 forces C(1)²→C(1)≈C(2)). The
test caught this (Spearman −0.4), so the relationship was reformulated with `τ_int`,
which is the correct monotone memory measure. An earlier framing ("off-diagonal
cascade coupling ⇔ ACF excess") was likewise found false — amplitude/variance coupling
and signal autocorrelation are distinct — and dropped rather than forced.

## NR18 — Exact identity: exchange fraction = √⟨k_x²/k²⟩  [Paper 2]

For the buoyancy force `F = (0, b')`, the Leray projection removes the gradient part
`∇φ` with `Δφ = ∂_y b'`. In Fourier, `φ̂ = −i k_y b̂ / k²`, so the removed field is
`∇φ̂ = k·k_y b̂ / k²` with `|∇φ̂|² = (k_y²/k²)|b̂|²`. Hence the **surviving
(motion-driving) fraction** is exactly

```
‖F_sol‖ / ‖F‖  =  √( Σ (k_x²/k²)|b̂|²  /  Σ |b̂|² )  =  √⟨k_x²/k²⟩_b,
```

the buoyancy-power-weighted spectral anisotropy. Pure vertical stratification (b̂ at
`k_x=0`) ⇒ 0 (held in place — Paper 2's PR2); horizontal `b'(x)` ⇒ 1 (full exchange);
isotropic ⇒ 1/√2. Verified against Paper 2's physical-space `solenoidal_fraction`:

| buoyancy field | solenoidal_fraction | √⟨k_x²/k²⟩ |
|---|---|---|
| vertical b'(y) | 0.000000 | 0.000000 |
| horizontal b'(x) | 1.000000 | 1.000000 |
| isotropic random | 0.719408 | 0.719408 |
| anisotropic | 0.964341 | 0.964341 |

Agreement to **3×10⁻¹⁶** (machine precision) — an exact identity that recasts Paper 2's
"in-place vs exchange" split as the spectral anisotropy of the buoyancy field.

## NR19 — Green–Kubo / Taylor: integrated memory time *is* an eddy diffusivity  [Paper 4]

Taylor (1922) / Green–Kubo: a fluctuating velocity's transport diffusivity equals the
time-integral of its own autocovariance,

```
D = ½ C(0) + Σ_{k≥1} C(k)   (= σ_v² · τ_int, the variance times the integrated memory time).
```

So Paper 4's **integrated memory time is literally an eddy diffusivity** — the
long-memory (slow) clock transports more. Verified by computing `D` **two independent
ways from the same ensemble**: single-particle dispersion `⟨x(T)²⟩/2T` versus the
Green–Kubo autocovariance integral.

| case | D_dispersion | D_GreenKubo | ratio |
|---|---|---|---|
| AR(1) φ=0.8 | 12.53 | 12.33 | 1.016 |
| AR(1) φ=0.95 | 199.2 | 196.4 | 1.014 |
| two-timescale | 8.51 | 8.60 | 0.990 |

The two estimators agree to **1.6%**, and both AR(1) rows match the *exact* random-walk
value `D = 0.5/(1−φ)²` (12.5, 200) to <1%.

**Honesty note.** A first estimator subtracted each trajectory's **own sample mean**
before forming the autocovariance; this injects an `O(maxlag/T)` negative bias into the
Green–Kubo tail (the sample-mean-subtracted biased ACF sums to exactly zero over all
lags), pulling `D_GreenKubo` ~19% **below** theory — a discrepancy that only "passed" a
loose 20% threshold. The fix removes the *known* (zero) process mean **globally**
(`_acov_rows`), after which the identity holds to ~1.6% and the threshold is tightened to
8%. The bias was diagnosed, not tuned away.

## NR20 — Additive diffusivity for multiscale memory: D_total = D_fast + D_slow  [Paper 4]

Because dispersion is linear in the velocity (`x = ∫v`) and Paper 4's fast/slow channels
are statistically independent, the total eddy diffusivity is the **sum** of the
per-channel diffusivities (the cross-term `⟨x_f x_s⟩ → 0`). Built from independent
fast (φ=0.5) and slow (φ=0.97) AR(1) channels at **equal energy** (Var = 0.25 each):

| | D_fast | D_slow | D_fast+D_slow | D_total (measured) | rel. err |
|---|---|---|---|---|---|
| two-timescale | 0.38 | 8.14 | 8.51 | 8.51 | <0.1% |

Additivity holds to **<0.1%**. Note the slow clock carries **~21×** the diffusivity of
the fast clock *at equal variance*: transport weights the **memory time**, not the
energy — the quantitative core of the two-clocks picture for turbulent dispersion.

## NR21 — Long-range dependence: DFA / Hurst exponent = (γ+1)/2  [Paper 3 → Paper 4]

Detrended fluctuation analysis (Peng et al. 1994) of a `1/f^γ` process recovers a Hurst
exponent `H = α_DFA = (γ+1)/2` — the standard fGn/fBm spectral↔self-similarity link
(Mandelbrot; Beran). Chained with NR16 (`γ = 3−α_burst`), **Paper 3's burst exponent
predicts Paper 4's long-range-dependence Hurst exponent**: `H = (4−α_burst)/2`.

| γ | α_DFA (measured) | (γ+1)/2 |
|---|---|---|
| 0.0 | 0.516 | 0.500 |
| 0.4 | 0.714 | 0.700 |
| 0.8 | 0.917 | 0.900 |
| 1.2 | 1.119 | 1.100 |
| 1.6 | 1.321 | 1.300 |

`α_DFA` tracks `(γ+1)/2` to **~0.02** (the small, consistent positive offset is the
well-documented finite-scale DFA bias). White noise (γ=0) ⇒ `H=½` (no LRD); the
pressure burst exponent `α_burst ≈ 2` ⇒ `γ ≈ 1` ⇒ `H ≈ 1` (strong persistence).

**Honesty note.** A companion super-diffusion law `⟨x²⟩ ∝ T^{2H}` was implemented and
**dropped**: the FFT-filtered `1/f^γ` generator distorts the low-frequency power that
drives the asymptotic scaling, so the measured dispersion exponent **saturated** (−0.30
below `2H` at γ=1.2). The relationship is true asymptotically but not cleanly
demonstrable with this generator, so it was removed rather than shipped with a tuned
tolerance.

## Scope

NR16/NR17 are verified on synthetic generators (power-law telegraph; 1/f^γ noise);
NR18 is an exact analytic identity verified to round-off; NR19/NR20 are verified on
AR(1) / two-timescale ensembles against both an independent estimator and the exact
AR(1) theory; NR21 is verified by DFA against the spectral prediction. They connect the
four new papers to each other and to mainstream point-process / spectral / Helmholtz /
Taylor–Green–Kubo / long-range-dependence theory; they are derived structural
relationships, not new empirical claims about the field.
