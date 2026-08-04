# NR68 — the atmosphere's vertical structure IS the two-clocks competition, and the turbopause is the clock handoff

**Module** `general_two_clocks/new_relationships45.py` ·
**data** committed NRLMSIS 2.1 cache (`data/nr68_msis_cache.json`, 4 geophysical
conditions × 181 altitudes × 11 columns, driven by real Celestrak F10.7/Ap) ·
**figure** `general_two_clocks/figures/103_atmosphere_two_clocks.json` + `.png` ·
**tests** `general_two_clocks/tests/test_atmosphere_two_clocks.py` (8)

## The relationship

Paper 1 measured, in turbulence, that the eddy-mixing memory time is
**scalar-independent** — τ_c(salt)/τ_c(heat) ≈ 1 across a 100× molecular-Lewis
contrast — because the memory belongs to the FLOW, not the substance. NR68 finds the
same statement written **vertically in the real atmosphere**, where aeronomy has
measured it for 60 years without reading it as a clock competition:

* **flow clock** — turbulent eddy mixing K_zz (gravity-wave breaking). Scalar-BLIND:
  moves every species identically, forcing one shared scale height set by the mean
  air mass, H̄ = kT/(m̄g). Constant mixing ratio = Paper-1 scalar-independence.
* **molecular clock** — species diffusion D_i ∝ m_i^(−1/2)/n. Scalar-SELECTIVE: in
  diffusive equilibrium each species takes its OWN H_i = kT/(m_i g).

The mainstream **turbopause/homopause** is defined as exactly D_i = K_zz
(Banks & Kockarts 1973; Vlasov & Kelley 2014) — the two clocks' crossover, with
altitude as the control parameter. Homosphere = flow-clock regime; heterosphere =
molecular-clock regime ("cream separates from milk").

## The discriminant (measured, no fit)

Read the per-species density scale height H_i(z) = −(d ln n_i/dz)⁻¹ for the inert
tracers He, Ar, N2 (+ slow O) straight off the NRLMSIS 2.1 profiles. The ratio
**H_He/H_N2 is the two-clocks discriminant**: 1 = mixed (flow clock), m_N2/m_He =
28/4 = 7.00 = separated (molecular clock).

Result (all 4 conditions — equatorial noon/midnight, midlatitude equinox,
high-latitude winter):

| quantity | prediction | measured |
|---|---|---|
| H_He/H_N2 at 85 km (homosphere) | → 1 | **1.09–1.10** |
| H_He/H_N2 at 450 km (heterosphere) | 28.0134/4.0026 = 6.999 | **7.00** (<0.1%) |
| H_Ar/H_N2 at 450 km | 28.0134/39.948 = 0.701 | **0.701** |
| H_O/H_N2 at 450 km | 28.0134/15.999 = 1.751 | **1.751** |
| departure onset (discriminant > 1.25) | ~turbopause | **89–90 km** |
| geometric-mean crossover (> √7) | homopause band | **99–132 km** |

The onset-to-crossover band (≈90–130 km) brackets the conventional D_i = K_zz
homopause (95–120 km in the literature; species/season-dependent, Garcia 2014). The
transition is gradual — we report the band, not a fake sharp level. Above it He
follows its own diffusive kT/(m_He g) (figure, right panel): the molecular clock in
possession.

## Jeans escape — the same competition at the exobase

At the top the two clocks stop being about transport and become about *escape*: the
gravity constraint (v_esc) races the thermal Maxwell tail (v_p ∝ m^(−1/2)). The Jeans
parameter λ = v_esc²/v_p² = m g r/(kT) is literally the clock ratio, and the escape
flux factor (1+λ)e^(−λ) makes the selectivity astronomical. From the cache at 500 km
(exobase proxy, midlatitude):

λ_H = 8.9 < λ_He = 35.2 < λ_O = 140.8 < λ_N2 = 246.5, flux factors
1.4e−3 ≫ 1.9e−14 ≫ 1.1e−59 ≫ 2.4e−105 — H and He leak, N2/O stay bound for the age
of the solar system. The same m-vs-thermal selectivity that separates the
heterosphere, taken to the altitude where the constraint clock loses the light tail
entirely. (Classical thermal estimate only; non-thermal channels — charge exchange,
polar wind — are out of scope.)

## Two theorems

1. **The turbopause is a two-clocks crossover, not a substance boundary.** D_i = K_zz
   is the Paper-1 scalar-blind vs scalar-selective race with altitude as control
   parameter; the discriminant H_He/H_N2 rides 1 → 7.00 exactly as the handoff
   completes.
2. **Jeans escape is the crossover's top boundary.** λ = (m g r)/(kT) is the
   gravity-clock/thermal-clock ratio; the atmosphere retains its heavy species
   because the constraint clock wins everywhere except the light, fast tail.

## Honest scope

NRLMSIS 2.1 is the community-standard *empirical model* (mass-spectrometer, ISR and
satellite-drag constrained; Emmert et al. 2021), not a single instrument — the
He/Ar/N2 heterospheric barometric structure it encodes is observational, but wording
stays "reference atmosphere", not "raw measurement". Chemically reactive species
(NO, O3) are excluded (sources/sinks, not clocks, set them). Thermal diffusion
(α_T, few-% shift) neglected. This is a reading of established aeronomy in the
two-clocks language plus a quantitative discriminant — not a new atmospheric
measurement, and no new closure is claimed.

## Verification

`python3 general_two_clocks/new_relationships45.py` regenerates the figure/JSON from
the committed cache. `pytest general_two_clocks/tests/test_atmosphere_two_clocks.py`
(8 passed): exponential-profile scale-height recovery, kT/mg textbook value + exact
1/m scaling, inverse-square gravity, λ linear-in-m/inverse-in-T, flux-factor
monotonicity + >1e30 light/heavy contrast, v_esc²/v_p² identity, cache schema, and
the end-to-end verdict (homosphere ≈ 1, heterosphere = mass ratios within 5%,
turbopause band ordering, Jeans ordering).
