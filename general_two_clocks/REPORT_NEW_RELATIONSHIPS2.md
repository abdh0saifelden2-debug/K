# New derived cross-relationships NR22–NR25

> Continues the cross-relationship program (NR1–NR21) with the same discipline:
> each relationship is **derived** from mainstream theory + a repo result and
> **numerically verified** — [`new_relationships2.py`](new_relationships2.py),
> [`tests/test_new_relationships2.py`](tests/test_new_relationships2.py) (11 tests).
> CPU-only, deterministic.

Run:

```bash
python general_two_clocks/new_relationships2.py
pytest general_two_clocks/tests/test_new_relationships2.py -v
```

## NR22 — Early-warning indicators are one rate: `Var·(1−AC1²) = σ_ε²`  [Paper 4 → NR3/NR8]

The two canonical critical-slowing-down precursors of the early-warning-signal
literature (Scheffer et al. 2009; Dakos et al. 2012) — **rising variance** and
**rising lag-1 autocorrelation** — are *not independent*. For the OU / AR(1)
reduction of a system near the saddle-node that controls this work (the `s_N`
flotation pole, NR3; its Lorentzian spectral corner `f_c∝(N−N_c)²`, NR8) one has,
exactly,

```
AC1 = a = e^{−Δt/τ},     Var = σ_ε² / (1 − a²),
```

so the product

```
I  ≡  Var · (1 − AC1²)  =  σ_ε²
```

is **invariant** as the fold is approached (`a→1`, `AC1→1`, `Var→∞`). Both
precursors are two reads of the single relaxation rate `τ`, and `Var ∝ τ` — i.e.
the variance EWS *is* the memory time (cf. NR19, memory time = diffusivity). The
invariant doubles as a **discriminator**: genuine slowing-down (`τ` grows, `σ_ε`
fixed) holds `I` flat while `AC1` rises, whereas mere forcing inflation (`σ_ε`
grows, `τ` fixed) raises `I` while `AC1` stays put — so `I` separates a true
approach-to-threshold from a louder driver.

Verified on AR(1) ensembles (`σ_ε=1`), approach-the-fold sweep:

| a | Var | AC1 | Var·(1−AC1²) |
|---|---|---|---|
| 0.50 | 1.334 | 0.499 | 1.0016 |
| 0.80 | 2.773 | 0.799 | 1.0016 |
| 0.90 | 5.250 | 0.900 | 1.0016 |
| 0.95 | 10.218 | 0.950 | 1.0016 |
| 0.98 | 25.067 | 0.980 | 1.0016 |

Invariant constant to **0.16 %** while Var rises ~19× and AC1→0.98. Discriminator
(a=0.8 fixed, σ_ε up): `I` = 1.00, 2.25, 4.00, 6.25 (= σ_ε²) while AC1 stays at
0.7995 — the invariant correctly attributes the variance rise to the driver, not
to slowing-down. Mainstream: Wissel (1984); Scheffer et al. (2009); Dakos et al.
(2012).

## NR23 — Kramers–Kronig DC sum rule: net eddy viscosity = migration-spectrum integral  [Paper 1 → NR9/NR15]

A causal eddy-response kernel `K(t≥0)` has admittance `Z(ω)=∫₀^∞ K(t)e^{−iωt}dt`
whose real and imaginary parts are Hilbert pairs (Kramers–Kronig, NR9). Evaluating
the once-subtracted KK relation at `ω=0` gives the **sum rule**

```
Z(0) − Z(∞)  =  (2/π) ∫₀^∞ [−Im Z(ω)] / ω  dω .
```

Here `Z(0)=∫₀^∞ K dt` is exactly NR15's **net eddy viscosity `ν_eff`** (and
`Z(∞)=0` for an integrable kernel), while `−Im Z` is the migration / quadrature
(reactive) spectrum. So the **sign and magnitude** of `ν_eff` — in particular the
backscatter branch `ν_eff<0` (growth) of NR15 — is fixed by a single weighted
integral of the migration spectrum: a one-number bridge from NR15 (sign budget)
to NR9 (causality). Verified on multi-Lorentzian kernels `K=Σ a_j e^{−t/τ_j}`:

| kernel | `Z(0)=ν_eff` | `(2/π)∫(−Im Z)/ω dω` | rel. err |
|---|---|---|---|
| dissipative (`ν_eff>0`) | +11.500 | +11.494 | 5.3×10⁻⁴ |
| two-scale | +21.400 | +21.394 | 2.6×10⁻⁴ |
| **backscatter (`ν_eff<0`)** | **−13.900** | **−13.897** | 1.8×10⁻⁴ |

Agreement to ~5×10⁻⁴ (set by the quadrature grid, not the physics), and the sign
tracks through a genuinely net-negative kernel. Mainstream: Kramers (1927);
Kronig (1926).

## NR24 — Eddy diffusivity is the zero-frequency PSD: `D = ½ S(0)`  [Paper 4 → NR17/NR19]

Green–Kubo (NR19) gives `D = ½C(0) + Σ_{k≥1} C(k)`; Wiener–Khinchin gives the
PSD `S(0) = Σ_{k=−∞}^∞ C(k) = C(0) + 2Σ_{k≥1} C(k)`. Hence

```
D  =  ½ S(0)   —   the eddy diffusivity is half the zero-frequency power.
```

Three consequences are verified.

**(i) Independent spectral route.** On AR(1) ensembles, a Welch-periodogram
estimate of `S(0)` reproduces the dispersion diffusivity `⟨x(T)²⟩/2T`:

| | `D_dispersion` | `½ S(0)` (Welch) | rel. err |
|---|---|---|---|
| a=0.80 | 12.64 | 12.50 | 1.1 % |
| a=0.90 | 50.50 | 49.95 | 1.1 % |
| a=0.95 | 201.6 | 199.1 | 1.2 % |

**(ii) Transport criticality.** Approaching the fold (`a→1`, critical slowing
down, NR22/NR3), `D = σ_ε²/(2(1−a)²)` **diverges as `D ∝ (1−a)⁻² ∝ (N−N_c)⁻²`** —
the transport face of the variance EWS (fitted exponent −2.000; empirical check
2.2 %).

**(iii) Long memory has no finite diffusivity.** A `1/f^γ` spectrum (γ>0,
NR16/NR17) has `S(0)→∞`, so **no finite eddy diffusivity exists** — the
quantitative reason the local down-gradient closure fails for Paper 4's long
memory. The ensemble-averaged window diffusivity `D_W` is flat for white noise
(slope −0.09) but **grows** for `1/f` (slope +0.55), confirming non-convergence.

Mainstream: Taylor (1922); Green–Kubo; Wiener–Khinchin.

## NR25 — Second fluctuation–dissipation theorem: memory kernel = random-force autocorrelation  [Paper 4 MZ → NR1]

In the exact harmonic-bath (Caldeira–Leggett / Zwanzig) reduction — the *linear*
instance of Paper 4's certified Mori–Zwanzig structure — eliminating the bath gives
a generalized Langevin equation whose friction (memory) kernel is

```
K(t) = Σ_i (c_i²/ω_i²) cos(ω_i t),
```

and whose random force `F(t)`, with the bath drawn from Gibbs at temperature `T`,
obeys the **second FDT**

```
⟨F(t) F(0)⟩ = k_B T · K(t).
```

Paper 4 certifies the *dissipation* side (kernel = the eliminated channel's Green's
function); this adds the *fluctuation* side — the **same** kernel is the noise
autocorrelation, so memory and noise are one object (the linear face of NR1).
Verified by computing the two sides by genuinely **independent** routes — the
friction kernel as a deterministic bath sum versus the Gibbs–Monte-Carlo
random-force autocorrelation (`M=1.5×10⁵` samples):

| t | `K(t)` (friction, deterministic) | `⟨F(t)F(0)⟩/k_BT` (sampled) |
|---|---|---|
| 0.0 | +8.170 | +8.186 |
| 1.0 | +2.453 | +2.480 |
| 2.0 | +0.826 | +0.845 |
| 3.0 | −0.909 | −0.909 |
| 4.0 | −0.783 | −0.792 |

Agreement to **0.35 % of peak** across the full oscillatory kernel (including the
sign changes), at sampling precision. Mainstream: Kubo (1966); Zwanzig (1973);
Caldeira & Leggett (1983).

## Scope and honesty

NR22 is verified empirically on AR(1) ensembles (the OU reduction near a
saddle-node); the invariant is the exact stationary identity and the test checks
both that it holds along the approach-to-fold and that it *fails* (tracks σ_ε²)
under pure noise inflation, so it is a genuine discriminator and not a tautology.
NR23 is an exact analytic identity verified to quadrature precision, including a
genuinely net-negative (`ν_eff<0`) kernel so the sign-tracking is actually
exercised. NR24(i)–(ii) are verified to ~1 % against an independent estimator and
the exact AR(1) value.

NR24(iii) is reported as a **qualitative** demonstration: the window-diffusivity
slope for `1/f` (+0.55) is *not* the asymptotic exponent, because the FFT-filtered
`1/f^γ` generator distorts the low-frequency power that drives the scaling — the
same generator artifact documented in the NR21 honesty note (where a precise
super-diffusion exponent was *dropped* rather than shipped with a tuned
tolerance). The claim here is only the robust, sign-definite fact that `D_W` grows
without bound for long memory while staying flat for white noise (`S(0)<∞` vs
`S(0)=∞`); no exponent is claimed.

These four connect the four papers to mainstream early-warning-signal,
Kramers–Kronig sum-rule, Taylor–Green–Kubo / Wiener–Khinchin, and
fluctuation–dissipation (Kubo / Zwanzig / Caldeira–Leggett) theory; they are
derived structural relationships, not new empirical claims about the field.

NR25 closes the loop with Paper 4's Mori–Zwanzig certification: that work proves the
dissipation side (kernel = eliminated Green's function) to round-off; NR25 verifies
the fluctuation side (kernel = random-force ACF) on the exact harmonic-bath
instance, where both sides are explicit. It is the second FDT in its textbook
linear form, not a new claim about the kernel — the contribution is exhibiting that
the repo's certified kernel *is* the noise autocorrelation, the linear face of NR1
(one memory number governing the breakdown of a memoryless closure).
