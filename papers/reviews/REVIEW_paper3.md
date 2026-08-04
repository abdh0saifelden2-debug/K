# Referee report — Paper 3

**Manuscript:** *Downstream scallop migration is a parity-symmetry signature that
eddy-diffusivity closures cannot reproduce.*

**Suggested venue assumed:** *Journal of Fluid Mechanics* / *Physical Review Fluids*.

**Recommendation: Major revision (revise & resubmit)** — but the closest of the four to
publishable, conditional on a real-data step.

---

## Summary of the submission

The paper studies why flow-carved scallops migrate downstream. Freezing a sinusoidal
melting interface and projecting the time-averaged wall heat flux onto the in-phase
(`E_sin`, amplitude rate) and quadrature (`E_cos`, migration) harmonics, it argues
migration is a parity-symmetry break that a local, memoryless, down-gradient
eddy-diffusivity closure cannot produce. The keystone is a numerical control: transporting
heat with a uniform eddy diffusivity drives the quadrature signal to machine zero
(`E_cos≈−1.2×10⁻¹⁸`), and Smagorinsky to ~57× below resolved, whereas resolved advection
gives a clear nonzero migration. A constant- and ΔT-free ratio `I=Im(s)/(2π|Re(s)|)` is
proposed as a cheap field test, with a self-validated reanalysis harness and a
regime-matched comparison to Bushuk et al. (2019).

## Strengths

1. **The harmonic phase-separation is elegant and the keystone control is excellent.**
   Turning "a symmetric operator cannot produce an asymmetric (migrating) response" into a
   direct measurement — the same momentum field, same drive, only the heat-transport
   operator swapped, with `E_cos` collapsing to machine zero — is the single cleanest,
   most convincing demonstration across the four papers.
2. **The constant-free, ΔT-free ratio `I`** is a genuinely clever construction: by
   cancelling `k_th`, `ρ_iL`, and `ΔT`, it yields a morphology-only observable, which is
   exactly what makes a field test feasible. This is the paper's most novel contribution.
3. **Honest caveats**, especially the decay-vs-growth sign caveat and the "mild tension,
   not falsification" framing of the Bushuk comparison.
4. **Solid literature grounding** on wavelength selection and migration direction, and a
   reanalysis harness validated on synthetic data to <3%.

## Major concerns

1. **The headline "K-theory cannot reproduce migration" is close to a known result.** That
   downstream migration arises from the phase shift between wall heat flux and topography is
   established — the paper itself notes Gilpin et al. (1980) predict downstream migration
   when that phase lies in `(π/2,π)`, "the same statement as `Im(s)≠0`." A
   parity-symmetric operator producing no quadrature response is nearly tautological. The
   manuscript must therefore reframe its novelty: the *new* contribution is the
   constant-free field-test ratio and the quantitative phase-separated solver measurement —
   not the (expected) fact that down-gradient closures lack the phase lag. As written, the
   abstract oversells the parity statement as the discovery.

2. **The central prediction is never tested against real data.** The paper is explicitly
   "verified in the solver … not yet validated against field scallops." The one real-world
   touchpoint (Bushuk et al. 2019, adjustment regime) uses eyeballed numbers (amplitude
   e-fold "`τ≈1–3 h` by eye") to get `I_obs≈0.05–0.4` against a solver band `0.33–0.88` — a
   factor-~3 tension from rough estimates. Since the underlying `h(x,t)` arrays are
   available on request and the harness is ready, a referee will reasonably require that the
   authors **obtain and analyze that data** before publication. Without it the paper is a
   well-designed *proposal* for a test, not the test.

3. **The frozen-interface probe is in a different dynamical regime than the phenomenon.**
   The frozen probe is decay-only (`Re(s)<0`); real ripples grow (`Re(s)>0`). The paper
   argues `Im(s)` is the shared, transferable piece, but whether the frozen-boundary
   quadrature maps quantitatively onto the growing-boundary migration is not established —
   it is asserted. This is the key physical risk to the field-test calibration and deserves
   a moving-boundary check, at least in the solver.

4. **Strong idealization:** 2-D, single-harmonic, finite-amplitude sinusoid. Real scallops
   are 3-D and multi-harmonic; the transfer of the single-mode result to natural patterns
   should be discussed.

## Minor concerns

- "Constant-free ratio" is free of *material constants* but is wavelength-dependent
  (`0.33→0.88`); the paper says so, but the abstract's framing could mislead — state plainly
  that `I` is an `O(1)` morphological number, not a universal constant.
- The migration exponent drifts across the paper (`0.48–0.8`, "tightened to `0.52`");
  present a single best estimate with its uncertainty rather than several values.
- Dense internal notation and cross-references to the "companion closure study"; make the
  paper self-contained.
- The §6 "decoy warning" (don't fit `τ∝λ²`) is a nice methodological point but is buried;
  promote it.

## Assessment against the journal criteria

- *Novelty:* the field-test ratio is novel; the parity statement is largely known — re-weight.
- *Validity:* the solver evidence is clean and convincing; the real-data link is missing.
- *Significance:* good if the field test is actually executed; otherwise modest.
- *Reproducibility:* strong (harness, tests, synthetic validation).

## What would move this toward acceptance

Run the harness on the Bushuk (or any) real `h(x,t)` dataset and report the result;
reframe the contribution around the constant-free field test rather than the parity
tautology; and add a moving-boundary (growing) solver check to justify transferring `Im(s)`
from the frozen probe. With the real-data step done, this is a clean, citable result.
