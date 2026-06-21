# Referee report — Paper 4

**Manuscript:** *Falsifiable forecasts for subglacial hydrology: a grounding-line
intrusion number verified on Bedmap2/BedMachine, and a thermal sliding-lag kernel
falsified on 131 active lakes.*

**Suggested venue assumed:** *The Cryosphere* / *Journal of Glaciology* / *GRL*.

**Recommendation: Reject in present form — resubmit as two or three focused papers
(equivalently, major revision contingent on restructuring).**

---

## Summary of the submission

The paper exposes two predictions of an "operator-structural turbulence framework" to open
Antarctic data: (1) a dimensionless Regime Transition Number (RTN) for where ocean water
can intrude under grounded ice, reported as directionally verified on Bedmap2 and
cross-checked on BedMachine; and (2) a thermal sliding-lag kernel `τ_ice=H²/κ_ice`, which
is falsified by ~5 orders of magnitude on 131 active lakes. Around these it adds a
machine-precision Mori–Zwanzig certification of a replacement (hydraulic) memory kernel, an
honest null matched-lag test, a closed-form `s_N(N)` sliding-law master curve with three
probes and an `N_c` inversion, a velocity-based ungrounding early-warning, and a residence
number `Ro`. Synthetic plant-and-recover harnesses calibrate every estimator.

## Strengths

1. **Exemplary scientific honesty.** Verifications, falsifications, and nulls are reported
   with equal weight; the matched-lag null is diagnosed precisely (aliasing + amplitude
   bound, *not* noise) rather than buried. This is exactly the discipline one wants.
2. **The RTN result is concrete and reproducible**, and the independent BedMachine
   cross-check (`r=0.970` thickness agreement) is the right way to guard against a
   single-dataset artifact.
3. **The MZ projection certification (§5)** is mathematically clean and verified to machine
   precision, with an explicit Markovian/adiabatic-elimination limit.
4. **Strong estimator discipline** — synthetic plant-and-recover harnesses and a
   kernel-shape-generic lag estimator ensure verdicts are read against the right null.

## Major concerns

1. **This is three or four papers, not one — the single most serious issue.** The
   manuscript carries: (a) the RTN intrusion number + residence number `Ro`; (b) the
   thermal-kernel falsification; (c) the MZ certification of a replacement kernel; (d) the
   matched-lag null; (e) the `s_N(N)` master curve + three probes + `N_c` inversion; (f) a
   critical-slowing-down ungrounding early-warning; (g) a dimensional-bridge/roughness
   closure. No single thesis sentence can carry these. The abstract alone advances ~7
   distinct results. A reader cannot state what the paper *claims*. It must be split — e.g.
   (I) the RTN intrusion forecast; (II) the sliding-lag falsification + MZ replacement
   structure; (III) the field-measurable sliding law + early-warning — each with one clear
   claim. As one submission it is not publishable regardless of the quality of the parts.

2. **The RTN "verification" lacks a baseline and may be near-geometric.** `RTN>1` requires
   ocean head `ρ_w g·d_base` to exceed `φ·ρ_i g H`. Near the grounding line the bed is deep
   (large `d_base`) and the ice is thin (small `H`), so the ratio *necessarily* tends to
   exceed 1 there. The claim "`RTN>1` concentrates near the grounding line" therefore risks
   restating known geometry. To establish that RTN is informative, the paper **must compare
   against a null/baseline predictor** — does RTN add skill over simply "distance to
   grounding line" or "bed depth below sea level"? Without that, the directional result is
   internal consistency, not a validated forecast, and the word "verified" in the title
   overstates it.

3. **The falsified kernel is a straw target.** No established sliding theory predicts surge
   lag equals the *full-thickness* thermal diffusion time; the thermal skin depth at
   surge periods is metres (as the paper itself shows), so `H²/κ` is a priori implausible.
   Falsifying a hypothesis the framework itself proposed is legitimate self-correction but
   should not be framed as a headline empirical result; its significance is limited and the
   framing ("a clean falsification on real geometry") overstates a foregone conclusion.

4. **The positive empirical content, after restructuring, is thin.** The one
   co-temporal real-data test (matched lag, §6) is **null**; the `s_N`/early-warning
   machinery (§7) is validated on synthetic data and, applied to real lakes, returns *true
   negatives* (far from flotation, no precursor). So the empirically *positive*, validated
   results reduce to: a directional geometric number (RTN, needing a baseline) and a
   falsification of an implausible kernel. The MZ certification and master curve are
   internally exact but not tested against field observations. The paper should calibrate
   its claims to this reality rather than to the framework's ambition.

5. **The "operator-structural turbulence framework" framing is unmotivated for a
   subglacial-hydrology audience.** Re-using the Paper 1 MZ formalism for cavity↔channel
   storage is fine, but to a glaciology referee it reads as imported machinery; the paper
   should motivate why MZ is the right lens here on hydrology's own terms, not by analogy to
   turbulence closure.

6. **Incomplete reference list — a hard editorial blocker.** The text cites many works
   absent from the reference list: Tsai et al., Cuffey & Paterson (2010), Nikuradse (1933),
   Joughin, Smith & Schoof (2019), Gagliardini et al. (2007), Schoof (2005), Tulaczyk et
   al. (2000), Gudmundsson (2006/2007/2011), Scheffer et al. (2009), Dakos et al. (2008),
   Boers & Rypdal (2021). An editor will return this immediately; every cited work must
   appear with a complete entry.

## Minor concerns

- The abstract (~45 lines) is far too long; it must state one or two claims, not seven.
- Pervasive internal scaffolding (RESULT 16–20, §H.1.6, §G.6, candidate numbering) leaks
  project structure into the manuscript; remove for an external reader.
- Define `φ`, `N`, `N_c`, `s_N`, `R`, `Ro`, `RTN` at first use; symbol load is very high.
- Several magnitudes are quoted to suspiciously many significant figures (e.g. median
  `τ_ice ≈ 151,458 yr`); round to the precision the inputs justify.

## Assessment against the journal criteria

- *Significance:* the honest-falsification methodology is valuable; the individual results
  are mixed (one directional, one null, one self-correcting falsification).
- *Scientific quality:* the synthetic/MZ machinery is rigorous; the real-data validation is
  thin and partly negative.
- *Focus/clarity:* **fails** — far too many claims in one paper.
- *References:* **incomplete** — multiple cited works missing.

## What would move this toward acceptance

Split into focused papers. For the RTN paper, add a baseline-skill comparison
(RTN vs. distance-to-GL / bed-depth) and drop "verified" unless it beats the baseline. For
the falsification/MZ paper, reframe the thermal kernel as a motivated null and foreground
the MZ structure honestly as a *structural* (not field-validated) result. Complete the
reference list. Each resulting paper would be defensible; the present omnibus is not.
