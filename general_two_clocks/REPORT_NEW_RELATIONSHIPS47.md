# NR70 — the escape ladder's clock algebra: series takes the slowest clock, parallel takes the fastest

**Module** `general_two_clocks/new_relationships47.py` ·
**data** the committed NR68 NRLMSIS 2.1 cache (via NR69's exobase locator) ·
**figure** `general_two_clocks/figures/105_escape_flux_algebra.json` + `.png` ·
**tests** `general_two_clocks/tests/test_escape_flux_algebra.py` (9)

## The relationship

NR69 located the ladder's rungs; NR70 computes the FLUX the ladder carries and
finds the two-clocks composition law — the resistor algebra of transport:

- **series** (one species through stacked stages): 1/g_tot = Σ 1/g_k (harmonic) —
  the *slowest* clock rules. Proved exactly in-module on a two-slab steady
  diffusion problem (<1e−12).
- **parallel** (one reservoir, side-by-side channels): g_tot = Σ g_k — the
  *fastest* clock rules.

Which clock is rate-determining is set by circuit topology, not by the clocks
alone. Earth's upper atmosphere provides one measured example of each.

## Hydrogen — the series face (diffusion-limited escape, Hunten 1973)

| flux (cm⁻²s⁻¹) | value | source |
|---|---|---|
| diffusive pipe ceiling Φ_lim = 2.5×10¹³·f_T | 2.5×10⁸ | Hunten 1973, f_T≈10⁻⁵ (H2O+CH4+H2; Catling & Kasting 2017) |
| observed TOTAL escape | ~1×10⁸ (~3 kg/s) | standard satellite-era value |
| thermal (Jeans) valve — **this cache** | **1.0–3.2×10⁷** (0.1–0.3 kg/s) | computed at NR69's exobase, 4 conditions |

The observed loss sits at the **pipe**, not the valve: the thermal valve carries
only ~13% at this epoch (charge exchange and polar wind — parallel channels —
carry the rest of the supplied flux). Series verdict: the slowest stage (the
rung-1→2 diffusive pipe) sets Earth's hydrogen loss — the mainstream reason H
escape is insensitive to exospheric temperature.

## Helium — the parallel face (the classical helium problem)

Crustal α-decay outgasses ⁴He at ≈10⁶ cm⁻²s⁻¹, which must leave in steady state
(Nicolet 1957). The thermal valve from the cache: **≤1.4×10⁻³ cm⁻²s⁻¹ — nine-plus
decades short** (λ_He = 36–43; e^(−λ) annihilates it). Mainstream resolution
(Axford 1968): the **polar wind** — ion outflow along open field lines — a
parallel channel whose clock is electromagnetic (the constraint clock), not
thermal. Parallel verdict: the fastest channel rules, and for He it is not in the
neutral ladder at all. (Sets up the ionospheric constraint-clock unit.)

## The exponential rectifier

Φ_J ∝ (1+λ)e^(−λ), λ ∝ 1/T_exo: across just one epoch's four conditions the H
valve swings ×3.1 and the He valve ×2067. Escape happens at hot excursions — the
time-average is solar-max-dominated (Hunten & Donahue 1976) — while the diffusive
pipe, algebraic in K_zz/D, is steady. Another face of "the slow clock is the
steady one."

## Honest scope

f_T ≈ 10⁻⁵, the He outgassing 10⁶ cm⁻²s⁻¹, and the observed total ~10⁸ cm⁻²s⁻¹
are cited literature constants, not derived. The cache is one low-activity epoch:
valve numbers are epoch-specific (that is the rectifier point); the He "≥9
decades" is for this epoch — cycle-averaged estimates still fall orders short,
which is why the polar wind is the accepted resolution. Classical Jeans formula;
kinetic ×2–2.5 corrections (Volkov 2017) move no decade-scale conclusion.

## Verification

`python3 general_two_clocks/new_relationships47.py` regenerates figure/JSON from
the committed cache. `pytest general_two_clocks/tests/test_escape_flux_algebra.py`
(9 passed): two-slab exact = harmonic series (<1e−12), slowest/fastest pinning,
exact composition identities, Jeans-flux linearity in n and exponential collapse
in λ, H valve band + under-pipe + observed-at-pipe, He ≥8-decade shortfall, O
bound (<1e−40), rectifier swings (He ≫ H), NR69 exobase reuse + verdict strings.
