# New cross-relationship — NR27 (`general_two_clocks/new_relationships4.py`)

Continues the derived-and-verified program (NR1–NR26; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, deterministic; unit-proofs in
[`tests/test_new_relationships4.py`](tests/test_new_relationships4.py). Run:

```bash
python general_two_clocks/new_relationships4.py     # -> figures/nr27_fluctuation_response_fdt.{json,png}
pytest general_two_clocks/tests/test_new_relationships4.py -v
```

---

## NR27 — The fluctuation–response (first-FDT) face of the fold: the tidal-admittance *response* measures proximity to ungrounding noise-free, and the fold mode's *FDT violation* is a frequency-resolved effective temperature whose roll-off measures the bath memory  [P4b admittance × P4a hydraulic memory]

### The gap this closes

NR25 certified the **second** fluctuation–dissipation theorem for the Mori–Zwanzig
reduction (the certified memory kernel *is* the random-force autocorrelation,
`⟨F(t)F(0)⟩ = k_BT·K(t)`). NR26 then showed that a slow critical fold mode driven by that
finite-memory bath has its critical-slowing-down **correlations** biased by the Deborah
number `De = λτ_c` (variance suppressed `1/(1+De)`; bi-exponential ACF; the two precursors
bracket the truth). NR26 is a statement about **correlations only**.

Its missing dual is the **response** side — the **first** FDT — which the repo already
measures *in the field* as Paper 4b's tidal velocity admittance. NR27 supplies it, and in
doing so turns the abstract De-coupling of NR26 into **two concrete field read-outs from
one velocity record**: a noise-free proximity gauge (the admittance) and a memory gauge
(the FDT violation between admittance and fluctuation spectrum).

### Setup (the same minimal model as NR26, now also forced)

A slow mode `s` near the fold has a vanishing restoring rate `λ` (`λ → 0` at the fold), is
driven by the colored Ornstein–Uhlenbeck bath of NR25/NR26 (memory `τ_c`, white-intensity
`D`), and is probed by a small external (tidal) force `f(t)`:

> `ṡ = −λ s + η + f(t)`,  `η = OU(τ_c)`,  `⟨η(t)η(0)⟩ = (D/τ_c) e^{−|t|/τ_c}`,  `De ≡ λτ_c`.

Two observables of the **same** record:

> **response** `χ(ω) = 1/(λ + iω)`  (admittance to the tidal force);
> **fluctuation** `S_s(ω) = 2D / [(λ²+ω²)(1+ω²τ_c²)]`  (the NR26 double-Lorentzian spectrum).

### Derived consequences (each verified against the simulated process)

**(a) The response is noise-free → a calibration-free proximity gauge.** `χ(ω)=1/(λ+iω)`
contains neither `D` nor `τ_c`: it is the **bare deterministic relaxation**. Hence the
static admittance

> **`|χ(0)| = 1/λ`** diverges at the fold,

and from a measured `χ(ω)`, `λ = Re(1/χ)` — **the distance to ungrounding read off the
admittance with no knowledge of the noise.** This is the response face of Paper 4b's
`|s_N|`.

**(b) The fluctuation spectrum is a *double* Lorentzian → a model-free memory signature.**
The high-frequency PSD rolls off as `ω⁻²` for a **white** bath (the standard CSD
assumption, NR8) but as **`ω⁻⁴`** once the bath has memory; the slope steepens from `−2`
to `−4` and the crossover is exactly at `ω = 1/τ_c`. The high-frequency log-log slope of
the velocity spectrum is therefore a calibration-free (`D`- and `λ`-free) test of *whether
the early-warning bath has memory at all*.

**(c) The FDT is violated by a frequency-dependent effective temperature.** The first-FDT
ratio (Kubo 1966; Cugliandolo–Kurchan–Peliti 1997; Harada–Sasa 2005) is, exactly,

> **`T_eff(ω) ≡ ω S_s(ω) / (2|Im χ(ω)|) = D / (1 + ω²τ_c²)`.**

In equilibrium (white bath, `De→0`) this is a **constant `= D`** and the FDT holds. With
bath memory it **rolls off** as a Lorentzian: plateau `D` at low `ω` (the equilibrium
temperature) → `0` at high `ω` (FDT maximally violated), half-value at `ω = 1/τ_c`. So the
**roll-off measures `τ_c`** and the **plateau measures `D`**, and `De = λτ_c` is recovered
from **response + fluctuation** — a frequency-domain de-bias of NR26's time-domain
bi-exponential-ACF fit.

### Numerical verification (`figures/nr27_fluctuation_response_fdt.{json,png}`)

OU-driven, externally-forced slow mode; `λ = 0.10`, `τ_c = 1`, `D = 1`, `De = 0.10`;
response by lock-in detection (noiseless run = exact `χ`; noisy run recovers it),
spectrum + effective temperature by Welch PSD, `n = 3×10⁶`.

| Claim | Predicted | Measured |
|---|---|---|
| (a) response = bare relaxation `1/(λ+iω)` | exact, `D`-free | lock-in matches to **0.4 %**; noisy record recovers it to **2.8 %** |
| (a) proximity from admittance `λ=Re(1/χ)` | `0.100` | **0.101** (0.8 %), no noise calibration |
| (b) high-`f` PSD slope, white bath | `−2` | **−1.98** |
| (b) high-`f` PSD slope, memory bath | `−4` | **−3.93** |
| (c) `T_eff(ω)` shape | `D/(1+ω²τ_c²)` | white-bath flat ratio **1.01** (FDT holds); memory drop ratio **0.03** (FDT violated) |
| (c) memory + temperature from the roll-off | `τ_c=1`, `D=1` | `τ_c` → **0.99** (1 %), `D` → **0.993** (0.7 %) |
| de-bias `De` from response × FDT roll-off | `0.10` | **0.0998** (0.2 %) |

All ten checks pass (`ok = true`).

### Why it matters

NR26 said *the warning is biased by `De`*; NR27 says *here are the two field measurements
that flag and remove it from one record*:

- the **tidal admittance** (Paper 4b) is a pure response, so it reads the distance to
  ungrounding **without any noise model** — `|χ(0)|=1/λ`;
- comparing that admittance to the **velocity-fluctuation spectrum** tests the FDT; a
  frequency-dependent effective temperature `T_eff(ω)=D/(1+ω²τ_c²)` exposes the **hydraulic
  bath memory** `τ_c` (the cavity↔channel relaxation certified in Paper 4a) and de-biases
  the CSD proximity — and even the bare PSD **slope** (`−2` vs `−4`) flags whether the
  bath has memory before any closure is assumed.

NR27 is thus the **first-FDT / response dual** of NR26's correlation-only result: the fold
has *one* fluctuation–dissipation structure with *two* measurable faces (response and
fluctuation), and the equilibrium relation between them holds only in the memoryless limit
— its breaking *is* the De-coupling of the repo's two failure modes, now read directly off
the spectrum and the admittance.

**Mainstream tools (cited, not claimed):** Kubo (1966, the fluctuation–dissipation
theorem); Cugliandolo, Kurchan & Peliti (1997, effective temperature); Harada & Sasa
(2005, FDT-violation = dissipation); Zwanzig (1973) and Hänggi & Jung (1995) (GLE /
colored noise); Scheffer et al. (2009), Dakos et al. (2012) (critical-slowing-down early
warning); Gudmundsson (2011) (ice-stream tidal admittance). The contribution is the
*coupling* of these to the repo's fold + memory structure and the resulting pair of
field read-outs (noise-free proximity; memory-measuring FDT violation).
