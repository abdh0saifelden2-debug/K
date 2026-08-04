# Referee report — Paper 2

**Manuscript:** *A boundary-condition-robust ceiling on turbulent basal melt in grounded
subglacial cavities: the thermal conductive sublayer, not momentum transport.*

**Suggested venue assumed:** *The Cryosphere* / *Journal of Glaciology* (with a fluids
referee).

**Recommendation: Major revision (revise & resubmit).**

---

## Summary of the submission

Using penalized (Brinkman volume-penalization) DNS/LES of a water-filled cavity between a
cold ice ceiling and a bed, the paper tests whether resolved cavity turbulence can raise
basal melt above the local conduction limit at a grounded, cold-walled ice base. Four melt
mechanisms (roughness plumes, double diffusion, tidal switching, scallop instability) are
stated with falsifiable go/no-go criteria and swept across three subgrid closures. Three
produce no positive mean-melt signal; the scallop produces a positive *local* lee-side
signal but a *mean* still below a flat wall, which the paper explains with an exact
area-partition identity and uses to falsify a two-moment `(1+CV²)` truncation. Boundary-
condition stress tests (Robin wall; Navier slip raising near-bed speed ~100×) leave mean
melt unchanged, pinning the ceiling to the thermal conductive sublayer. A Type I/II/III
regime map scopes the claim.

## Strengths

1. **Falsifiable, pre-stated criteria swept across closures.** Fixing each mechanism's
   go/no-go threshold before the run, and demanding closure-independence, is exemplary
   methodology that makes the nulls informative rather than tuning failures. Reviewers will
   value this.
2. **The exact area-partition identity** (`Nu/Nu_flat = C_reatt + C_thick + C_rev` to
   machine precision) is a clean, rigorous, and genuinely illuminating decomposition of why
   a scalloped wall melts *less* on average despite local hot spots.
3. **The `(1+CV²)` falsification** — showing the suppression is distribution-dominated, not
   moment-dominated — is a sharp, specific, and useful negative result.
4. **Boundary-condition robustness** (Robin wall; ×100 slip) is a strong, well-targeted
   stress test of the central claim.
5. **Honest, regime-scoped framing** and full reproducibility (closure sweeps, tests, repo).

## Major concerns

1. **The double-diffusion null cannot be claimed from 2-D penalized forced runs.** Salt
   fingering and double-diffusive staircases are intrinsically three-dimensional; the paper
   itself concedes the absence "may partly reflect 2-D + penalization + forcing." That
   concession undercuts a headline component of the null map. As it stands, "no
   double-diffusive melt enhancement" is not supported — it must be demonstrated in 3-D
   unforced double-diffusive runs, or the claim must be withdrawn for that mechanism.

2. **The central result hinges on resolving the conductive sublayer — but resolution
   adequacy is not demonstrated.** The whole thesis is "the ceiling is the thermal
   conductive sublayer." The interfacial flux `q=−k_th ∂θ/∂n` is exquisitely sensitive to
   near-wall resolution and to the Brinkman penalization parameter, and the paper reports a
   real resolution artifact (the `ny=48` penalty-bleed). A referee will require: (a) an
   explicit near-wall convergence study (`δ_T/Δx`), (b) sensitivity to the penalization
   parameter `η`, and (c) evidence the sublayer is resolved, not penalization-smeared.
   Without this, the ceiling could be a discretization artifact.

3. **Missing the core ice–ocean melt literature; novelty not established.** Melt at an
   ice–water interface being rate-limited by transfer through a thin thermal/haline
   sublayer is the basis of the standard three-equation melt parameterization (Holland &
   Jenkins 1999) and of ice–ocean boundary-layer theory (McPhee; Kader–Yaglom transfer
   laws). None are cited. A glaciology/ocean referee will immediately ask how the
   "conductive sublayer ceiling" differs from this established picture. The paper must
   engage this literature and state precisely what is new (plausibly: the *closure-swept,
   pre-registered* confirmation + the area-partition mechanism, not the ceiling itself).

4. **"Pre-registration" needs substantiation or rewording.** For a simulation study the
   term implies a timestamped, independently lodged protocol. If that exists, cite it; if
   not, soften to "pre-stated falsifiable criteria," because the strong word invites
   skeptical scrutiny it doesn't need.

5. **Dated geometry.** BEDMAP1 (2001) is two generations old. Reviewers will ask why not
   Bedmap2/3 or BedMachine, especially since the companion Paper 4 uses them.

## Minor concerns

- Heavy internal scaffolding leaks into the text (Part-C config, §G.6, candidate
  numbering, "pending in the repo"); clean these for an external reader.
- In-text mention of a bug that "briefly corrupted" a value, while admirably honest, belongs
  in a supplement; in the main text it invites doubt about code reliability.
- The interface-coupling-number argument (§5) is elegant but compressed; give the response
  function `H(s)` and the crossover derivation explicitly.
- Define `Ri`, `R_ρ`, `Le`, `CV_δ`, `St` at first use; the abstract is symbol-dense.

## Assessment against the journal criteria

- *Significance:* good and policy-relevant (why grounded melt models omit a flow term).
- *Scientific quality:* the area-partition and BC tests are strong; the double-diffusion
  null and sublayer-resolution claims are not yet defensible.
- *Reproducibility:* strong.
- *Literature/context:* **insufficient** — the key melt-parameterization references are absent.

## What would move this toward acceptance

A near-wall resolution + penalization-parameter convergence study; 3-D double-diffusion (or
withdraw that null); explicit positioning against Holland & Jenkins / McPhee with a crisp
novelty statement; and substantiation or softening of "pre-registered." The area-partition
result alone is a publishable nugget; the broader ceiling claim needs the convergence
evidence to stand.
