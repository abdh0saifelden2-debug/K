# Referee report — Paper 1

**Manuscript:** *Single-eddy-diffusivity closure is the Markovian-delta collapse of a
Mori–Zwanzig memory kernel: a projected fluctuation–dissipation repair.*

**Suggested venue assumed:** *Physical Review Fluids* / *Journal of Fluid Mechanics*
(theoretical / methods contribution).

**Recommendation: Major revision (revise & resubmit).**

---

## Summary of the submission

The paper argues that the single-scalar eddy-diffusivity ("K-theory"/Smagorinsky) closure
is not merely under-tuned but *structurally* deficient, and makes this precise by
identifying it as the Mori–Zwanzig (MZ) generalized Langevin equation with three named
simplifications: (i) the memory kernel collapsed to a delta in time, (ii) the spectral
eddy viscosity frozen constant in wavenumber, (iii) the fluctuation–dissipation-linked
orthogonal force deleted — plus the sign-definiteness that forbids backscatter. It then
proposes a "projected-FDT repair" (scale-dependent spectral eddy viscosity + a
divergence-free FDT-tied stochastic force + an exact Leray projection) and tests all three
against an exact subgrid force on a frozen 256² 2-D field. It reports the repair matching
the energy-transfer spectrum (correlation 1.000) and staying divergence-free, where
Smagorinsky is purely dissipative and a spectrum-matched random surrogate violates
incompressibility. A measured memory time `τ_c`, a clock-mismatch correction, and a Dedner
divergence-cleaning "design window" round out the paper.

## Strengths

1. **The central reframing is clean and correct.** Writing single-K closure explicitly as
   the Markovian-delta collapse of the MZ kernel, with the four assumptions named
   one-by-one, is pedagogically sharp and a genuinely useful way to *organize* the failure
   modes of eddy-viscosity models.
2. **The "structure vs. spectrum" benchmark is well-designed.** The three-model ×
   three-diagnostic table makes a real point that is often muddled in the SGS literature:
   matching the force *spectrum* (the surrogate) is not sufficient — it destroys
   incompressibility and randomizes the transfer direction. The contrast is instructive.
3. **Unusually honest scoping.** The "what this is / what this is not" section
   pre-empts overclaiming (no 3-D regularity, no operational superiority, a-priori only).
   This is commendable and rare.
4. **Reproducible.** Scripts, unit tests, and a public repository are provided — meeting
   the transparency bar of both target journals.
5. **The Dedner "design window" (§7)** is a concrete, falsifiable, and apparently new
   practical observation (an over-cleaning stall from the slow telegrapher root).

## Major concerns

1. **Novelty relative to existing MZ–LES work is not established.** The identification of
   eddy viscosity / Smagorinsky as a Markovian limit of an MZ/optimal-prediction reduction
   is *not new*: it is the conceptual basis of the Chorin–Hald–Kupferman optimal-prediction
   program, the Stinis renormalized t-model, and especially Parish & Duraisamy (2017) and
   the subsequent MZ-based LES literature, all of which the paper cites. The manuscript
   must state explicitly and defensibly what is new **beyond** these — currently §3 reads
   as a re-derivation of an established connection. If the novelty is the *specific* "four
   sins" decomposition plus the structure-vs-spectrum benchmark, the paper should say so
   and trim the framing that implies the MZ→eddy-viscosity link is itself the contribution.

2. **The headline "transfer correlation = 1.000" is at least partly built in, not
   predicted.** Equation (9) sets `Θ(k)` "so the net transfer equals the true subgrid
   transfer." If the model's free function is tuned to reproduce the measured transfer,
   then reproducing it is not an independent validation. The paper must (a) state precisely
   which quantities are *predicted* and which are *fit*, (b) report the number of free
   functions/parameters in (8)–(9), and (c) demote or re-caption the "1.000" result
   accordingly. As written, "the benchmark that decides it" overstates what an a-priori,
   partially-fitted match can decide. The solenoidality result, by contrast, is genuinely
   structural and should be foregrounded instead.

3. **No a-posteriori test.** The paper concedes that the time-integrated test is deferred.
   For a closure paper at this level, the absence of *any* a-posteriori demonstration (does
   the repair actually improve a running LES?) leaves the practical claim unsupported.
   Either add a minimal a-posteriori run (even 2-D), or reframe the paper explicitly as a
   structural/methods note and soften "repair" to "structurally-consistent closure form."

4. **The clean result lives only where the problem is easiest.** The exact FDT/projection
   commutation holds only in the periodic spectral setting; the manuscript states walls
   break it. But the motivating failures (separated wakes, wall-bounded, stratified flows)
   are exactly the wall-bounded cases. The gap between "where it's proven" and "where it
   matters" should be addressed head-on, not deferred.

5. **Scope creep weakens focus.** The §1.2 observational "two clocks" ladder
   (NEON/ASOS/Rayleigh–Bénard) is described as motivation-only with an unavailable
   companion report, and §7's Dedner study uses an ice-sheet cavity field. Both read as
   imported from adjacent projects. Referees will ask that the paper either integrate these
   properly or remove them; citing an unprovided companion is not acceptable.

## Minor concerns

- The abstract is very long and uses strong adjectives ("decisive," "the benchmark that
  decides it") that the a-priori 2-D evidence does not earn. Tone down.
- Define `τ_adj`, `c_h`, `G`, and the "1/e knee" before use in §7; the telegrapher
  argument needs one explicit equation, not just a clause.
- §6.1 "the slow:fast bath weight closes to the Stefan number" is cryptic — a Stefan
  number is a melting parameter; its appearance here needs explanation or is a mis-statement.
- Per JFM transparency policy, the complete set of equations and boundary conditions for
  the 256² DNS (forcing, dealiasing, dissipation) should be in the paper, not only the repo.
- "256² 2-D DNS … forced inverse cascade" — give `Re`, forcing band, and resolution
  adequacy (is `k_c=32` in the inertial range?).

## Assessment against the journal criteria

- *Novel, important flow physics (PRFluids):* **partially** — the organizing insight is
  valuable but its novelty over MZ-LES prior art is unproven as written.
- *Substantive, comprehensive (PRFluids):* **not yet** — a-priori-only and 2-D-only.
- *Replication/verification (JFM):* **strong** — repo, tests, but in-paper equations are
  incomplete.
- *Clarity:* dense; the core message is good but buried under breadth.

## What would move this to acceptance

Separate predicted from fitted in §5 and re-caption honestly; sharpen the novelty claim
against Parish & Duraisamy / Stinis; add at least a minimal a-posteriori run *or* reframe
as a structural methods note; and either integrate or cut the §1.2 ladder and the §7
ice-sheet study. The skeleton of a publishable, useful paper is clearly here.
