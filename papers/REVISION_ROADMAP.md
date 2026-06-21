# Revision Roadmap — repositioning Papers 1–4 to use mainstream theory as *tools* and claim only the genuine new findings

**Purpose.** The referee reports (in `papers/reviews/`) keep hitting the same nerve: each paper's
*headline mechanism* is already mainstream, so as written the papers risk claiming to rediscover
known physics. This document is the fix. For every paper it separates three things, with verified
citations:

1. **Tools** — the established theory the paper *uses* and must *cite, not claim*.
2. **The delta** — the genuinely new contribution to foreground (usually narrower than the current framing).
3. **Demote / drop** — claims that overstate novelty or aren't supported, and what to do with them.

Then it gives concrete, ordered edit steps (title → abstract → novelty paragraph → body) and the
validation each paper needs before it is honestly submittable.

**The core principle (applies to all four).** *Cite the mechanism, claim the instrument.* In every
paper the physical mechanism (eddy-viscosity-as-Markovian-limit; sublayer-limited melt; phase-lag
migration; flotation-controlled intrusion; critical-slowing-down warning) is **someone else's
result**. What is yours is a sharper **instrument** built on top of it: an exact decomposition, a
constant-free observable, a structural certification, a closure-swept null, a pre-stated falsifier.
The paper becomes honest *and stronger* when the mechanism is demoted to "the tool we stand on" and
the instrument is the headline.

---

## Status (execution)

All four honesty-repositioning passes are committed on `bedmachine-v4-rtn-crosscheck`:
P3 `d11fecd`, P1 `8d5e50a`, P2 `8b7668d`, P4 `76f29b1`. Each demotes the mainstream
mechanism to a cited tool, foregrounds only the genuine delta, removes fitted/tuned results
from the findings (P1's `1.000` re-captioned as a fitted consistency check), drops
rediscoveries (P4 `RTN`=flotation ratio; P4 early-warning=Rosier 2021 operationalization),
completes the tool references (P2, P4), and strips internal series scaffolding. No
"previously claimed" notes appear in any draft.

**Remaining (heavier; user-gated):** P2 near-wall convergence + penalization study (needs
re-running the cavity DNS); P3 real-data Bushuk reanalysis (data not public) + a
moving-boundary solver check; P4 the formal 2–3-way split into focused papers + the RTN
baseline-skill test (needs re-downloading Bedmap2/BedMachine). These are validations and
structural work, not framing — the manuscripts now state each honestly as the open step.

---

## 0. How arXiv/journal papers actually signal novelty (the discipline to copy)

Strong papers in these venues do four things we are currently not doing consistently:

- **An explicit "what is new here" paragraph** near the end of the introduction, written *against*
  the closest prior work by name ("Parish & Duraisamy derive the MZ closure; we instead …").
  Reviewers look for this paragraph first. If they can't find the one-sentence delta, they assume
  there isn't one.
- **A Related Work / Background section that is generous, not defensive.** Citing Holland & Jenkins
  or Gilpin prominently does not weaken the paper — it shows command of the field and makes the
  narrow new claim *credible*. Hiding the prior art is what gets a paper desk-rejected or branded as
  reinvention.
- **One thesis sentence per paper.** A reader must be able to state the single claim. Paper 4 fails
  this outright (≈7 claims); the others are close but over-broad in the abstract.
- **Scope stated as a feature.** "A-priori, 2-D, periodic" / "directional, not precision-recall" /
  "verified in the solver, not yet on field data" — said up front, these read as rigor. Said only in
  a buried caveat, they read as something the authors hoped you'd miss.

The repo's honest-falsification discipline already matches the *best* version of this culture. The
job now is to make the **framing** match the honesty that's already in the methods.

---

## 1. The tools-vs-delta map (one line per paper)

| Paper | Mainstream tool (cite, don't claim) | The genuine delta (foreground) | Status |
|---|---|---|---|
| **P1 Closure** | MZ→eddy-viscosity as the Markovian limit (Chorin–Hald–Kupferman; Stinis; Parish & Duraisamy 2017); divergence-free FDT backscatter (Leith 1990; Mason & Thomson 1992; Schumann 1995) | The **four-named-sins diagnostic**, the **scalar-independent memory time** `τ_c` (heat vs salt, 100× Lewis), and the **Dedner over-cleaning "design window"** | Reframe + separate predicted-vs-fit |
| **P2 Melt ceiling** | Sublayer-limited melt = three-equation theory (Holland & Jenkins 1999; McPhee; Kader–Yaglom) | The **exact area-partition identity** showing a scalloped wall melts *less* on the mean — *against* the "roughness enhances melt" consensus (Gilpin; Wettlaufer; Claudin 2017) — plus the closure-swept null map | Add convergence study; cite melt lit |
| **P3 Scallops** | Phase-lag → downstream migration (Gilpin 1980; Hanratty 1981; Bushuk 2019) | The **constant-free, ΔT-free morphological ratio `I`** as a field test, and the clean operator-swap parity *measurement* | Retitle; run/flag real-data step |
| **P4 Hydrology** | Flotation / effective-pressure criterion (Schoof 2007; Leguy et al. 2014; Cuffey & Paterson); CSD early-warning on grounding-line flux (Rosier et al. 2021; Scheffer; Dakos) | After a split: the **exact MZ certification of the hydraulic memory kernel**, the **honest-falsifier methodology**, and (if it beats baseline) RTN as a nondimensional intrusion screen | Split into 2–3; add RTN baseline |

The recurring move is the same every row: **the mechanism is the tool; the instrument is the claim.**

---

## 2. Paper 1 — closure theory

**Current framing risk (highest of the four).** The abstract leads with "single-K closure is
*precisely* the [Markovian] limit … of the MZ kernel" and a "minimal projected-FDT repair." Both are
established: Parish & Duraisamy (2017, *Phys. Rev. Fluids* 2, 014604; *J. Comput. Phys.* 349, 154)
derive LES closures from MZ and show eddy-viscosity is the Markovian/finite-memory limit; and the
*divergence-free, dissipation-tied stochastic backscatter force* — which the paper calls its repair —
is exactly Mason & Thomson (1992, *JFM* 242, 51), who build the force from the **curl of a random
potential to guarantee non-divergence**, with Leith (1990) and Schumann (1995, *Proc. R. Soc. A*)
fixing the FDT-tied amplitude. So three of the paper's four "repair" ingredients are prior art.

**(A) Tools to cite, not claim.**
- MZ / generalized Langevin equation and optimal prediction: Mori 1965; Zwanzig 1973; Chorin–Hald–Kupferman 2000, 2002; Stinis 2015.
- MZ-based LES and the eddy-viscosity-as-Markovian-limit: **Parish & Duraisamy 2017** (the single most important citation to position against).
- Spectral eddy viscosity / cusp: Kraichnan 1976; Chollet & Lesieur 1981.
- Divergence-free, FDT-tied stochastic backscatter: **Leith 1990; Mason & Thomson 1992; Schumann 1995**; Frederiksen & Davies 1997.

**(B) The genuine delta — foreground these three, nothing else as "new":**
1. **The four-named-sins diagnostic.** Writing single-K as the *simultaneous* collapse of (i) memory→δ, (ii) `ν_t(k)`→const, (iii) FDT force deleted, (iv) `ν_t≥0` (no backscatter) — as a single organizing table that maps each failure mode to a named operator surgery. This is a **pedagogical/diagnostic synthesis**, genuinely useful, but must be sold as *synthesis*, not discovery.
2. **The scalar-independent memory time.** `τ_c(salt)/τ_c(heat) ≈ 1` across a 100× molecular-Lewis contrast while scalar Nusselt numbers track `Le` strongly — i.e. the memory belongs to the *flow*, not the transported field. This is a concrete, falsifiable, apparently-new numerical result. **Strongest novel nugget #1.**
3. **The Dedner over-cleaning "design window."** The `2 ≲ γτ ≲ 12` window with an over-cleaning *stall* from the slow telegrapher root — the referee explicitly flagged this as "apparently new." **Strongest novel nugget #2.** Promote it from §7 to a co-headline.

**(C) Demote / drop.**
- **Demote** "the benchmark that decides it" / "decisive": the transfer-correlation `1.000` is partly *built in* — Eq. (9) tunes `Θ(k)` so net transfer matches truth. Re-caption: the **solenoidality** result (`∇·m ≈ 10⁻¹⁴` projected vs `≈12` for the spectrum-matched surrogate) is the genuinely *predicted* structural discriminator; the `1.000` is a consistency check on a fitted quantity.
- **Demote** the "repair" language → "structurally-consistent closure form assembled from known parts," with Mason–Thomson/Leith/Schumann credited in the same sentence.
- **Cut or spin off** the §1.2 NEON/ASOS/Rayleigh–Bénard "two clocks" ladder that cites an *unprovided* companion report — referees will not accept a citation to an unavailable work. Either fold one concrete panel in, or remove and keep "two clocks" as a one-line motivation.

**(D) Concrete edit steps.**
1. **Retitle** to claim the diagnostic + instruments, not the MZ link. e.g. *"What single-eddy-diffusivity closure throws away: an operator-by-operator audit of the Mori–Zwanzig kernel, a scalar-independent memory time, and a divergence-cleaning design window."*
2. **Rewrite the abstract** in three moves: (1) "It is *known* that eddy-viscosity is the Markovian limit of an MZ reduction [Parish & Duraisamy] and that FDT-tied divergence-free backscatter can be added [Mason & Thomson; Leith]." (2) "We contribute three things on top of this: a unified four-operator audit, a measured scalar-independent memory time, and a finite divergence-cleaning window with an over-cleaning stall." (3) scope: a-priori, 2-D periodic.
3. **Add the explicit "what's new vs Parish & Duraisamy / Mason–Thomson" paragraph** at the end of §1.
4. **Re-caption §5** to separate *predicted* (solenoidality, transfer sign partition) from *fitted* (`Θ(k)`→`1.000`).
5. **Promote §7 (Dedner window)** into the contributions list and abstract.

**(E) Validation before submittable.**
- Re-run §5 reporting, per quantity, predicted vs fitted, and the number of free functions in (8)–(9).
- *Either* add a minimal a-posteriori (time-integrated) 2-D run, *or* explicitly label the paper a **structural/methods note** (the cleaner, faster path given the genuine deltas are diagnostic).

**Net.** P1's honest contribution is real but *narrow and largely diagnostic*. Sold as "we audited a
known reduction and found two new concrete facts (scalar-independent `τ_c`; the cleaning window)," it
is a clean methods paper. Sold as "we discovered single-K is the MZ Markovian limit," it will be
rejected as reinvention.

---

## 3. Paper 2 — basal melt ceiling

**Current framing risk.** The thesis "mean basal melt is limited by the thermal conductive sublayer"
**is** the standard three-equation ice–ocean melt picture (Holland & Jenkins 1999), where molecular
diffusion through a thin sublayer rate-limits transfer (Mellor 1986; McPhee 1987; Kader & Yaglom
1972). None of these are currently cited. A glaciology/ocean referee will read the ceiling as a
restatement of textbook boundary-layer melt theory.

**(A) Tools to cite, not claim.**
- Sublayer-limited melt / three-equation parameterization: **Holland & Jenkins 1999** (*JPO* 29, 1787); McPhee 1987/1992/2008; Kader & Yaglom 1972; Hellmer & Olbers 1989; Jenkins 1991.
- The competing consensus your result pushes against: scallops/roughness **increase** turbulent heat transfer and melt — Gilpin et al. 1980; Seki, Fukusako & Younan 1984; Wettlaufer 1991; Feltham, Worster & Wettlaufer 2002; Claudin, Durán & Andreotti 2017; Bushuk et al. 2019; Couston et al. 2021.

**(B) The genuine delta — foreground:**
1. **The exact area-partition identity** `Nu/Nu_flat = C_reatt + C_thick + C_rev` (machine precision), proving a scalloped wall delivers *less* mean basal heat than a flat wall despite 2–3× local lee hot spots. This is interesting **precisely because it runs against the "roughness enhances melt" consensus** — say so explicitly; that tension is the paper's reason to exist.
2. **The `(1+CV²)` two-moment falsification** — the suppression is *distribution*-dominated, not moment-dominated. A sharp, specific negative result.
3. **The closure-swept, pre-stated null map** in the cold-grounded (Type-I) regime, with BC-robustness (Robin wall; ×100 slip) — the *methodology* is a contribution.

**(C) Demote / drop.**
- **Withdraw or 3-D-ify the double-diffusion null.** Salt fingers are intrinsically 3-D; the paper already concedes the 2-D+penalization caveat. As written, "no double-diffusive enhancement" is unsupported. Either run a 3-D unforced double-diffusive case or remove that mechanism from the null map (keep it as "not testable in 2-D").
- **Demote "pre-registered" → "pre-stated falsifiable criteria"** unless a timestamped protocol exists. The strong word invites scrutiny it doesn't need.
- **Replace BEDMAP1 (2001) with Bedmap2/BedMachine** (the companion P4 already uses them) — a free credibility fix.
- **Reframe the ceiling itself** from "our finding" to "we confirm, with a closure-swept resolved-DNS test and an exact mechanism, the sublayer ceiling that three-equation theory assumes — and show geometry *suppresses* rather than enhances it in this regime."

**(D) Concrete edit steps.**
1. **Retitle** around the counter-consensus mechanism: e.g. *"Why a scalloped cold-grounded ice base melts *less* on average: an exact area-partition ceiling below the conductive sublayer."*
2. **Add a Background paragraph** stating the Holland–Jenkins sublayer picture *and* the roughness-enhancement consensus, then: "we show that in the grounded, cold-walled regime the sign flips, via an exact identity."
3. **Abstract:** lead with the area-partition identity and the against-consensus mean suppression; present the null map as method; state the regime (Type I) as scope.
4. **Move the double-diffusion claim** to "untested in 2-D" unless 3-D is run.

**(E) Validation before submittable.**
- **Near-wall convergence study** (`δ_T/Δx`) **+ penalization-parameter (`η`) sensitivity**, to prove the conductive sublayer is *resolved*, not penalization-smeared (the `ny=48` penalty-bleed artifact makes this mandatory). This is the load-bearing validation — without it the ceiling can be dismissed as a discretization artifact.
- 3-D unforced double diffusion *or* withdraw that null.

**Net.** P2 has the single cleanest *exact* result of the four (the area-partition identity) and a
genuinely counter-consensus headline. It needs the convergence study and the melt-literature
positioning to land.

---

## 4. Paper 3 — scallop parity break (closest to publishable)

**Current framing risk.** The abstract's headline — "migration is a parity-symmetry break that no
K-theory closure can produce" — is essentially Hanratty (1981) and Gilpin et al. (1980): instability
and downstream migration arise from the heat-flux/topography **phase lag** (max transfer downstream
of the trough), and a symmetric, memoryless operator producing no quadrature response is nearly
tautological. The paper *already knows this* (§7 says Gilpin's phase condition "is the same statement
as `Im(s)≠0`"). So the parity statement is a known consequence, not the discovery.

**(A) Tools to cite, not claim.**
- Wavelength selection: Curl 1966; Blumberg & Curl 1974; Thomas 1979; Hsu et al. 1979; Thorsness & Hanratty 1979.
- Phase-lag → downstream migration (the mechanism): **Gilpin, Hirata & Cheng 1980; Ashton & Kennedy 1972; Hanratty 1981**; Bushuk et al. 2019.
- Modern linear-stability / saturation: Camporeale & Ridolfi 2012; Claudin, Durán & Andreotti 2017.
- Interface smoothing: Mullins & Sekerka 1964.

**(B) The genuine delta — foreground:**
1. **The constant-free, ΔT-free morphological ratio** `I = Im(s)/(2π|Re(s)|)` — by cancelling `k_th`, `ρ_iL`, and `ΔT` it becomes a **field-measurable observable computable from morphology alone**. This is the paper's real, novel contribution. Make it the title.
2. **The operator-swap parity *measurement*** — same momentum field, same drive, swap only the heat-transport operator, watch `E_cos` collapse to machine zero (`10⁻¹⁸`). Frame this not as "we discovered parity matters" but as "a clean numerical *isolation* that converts the known phase-lag argument into a direct measurement, and quantifies exactly how far each closure falls short."

**(C) Demote / drop.**
- **Demote the parity statement** from "the discovery" to "a known consequence (Gilpin/Hanratty) that we exploit as an exact control."
- **Present a single migration exponent** with uncertainty (`0.52 ± …`) instead of the drifting `0.48–0.8`.
- **State plainly that `I` is an O(1) morphological number, not a universal constant** (it runs 0.33→0.88) — the abstract currently risks implying universality.

**(D) Concrete edit steps.**
1. **Retitle** around the instrument: e.g. *"A constant-free morphological field test for non-local heat transport at melting interfaces."*
2. **Abstract:** open with the field-test ratio; introduce the parity collapse as the *control that validates the closure claim*; cite Gilpin/Hanratty in the same breath as the phase-lag mechanism.
3. **Promote the §6 "decoy warning"** (don't fit `τ∝λ²`) — it's a genuinely useful methodological point currently buried.

**(E) Validation before submittable.**
- **Run the harmonic reanalysis on real `h(x,t)` data.** The harness is built and self-validated (<3%); the Bushuk et al. (2019) arrays are available on request (`mitchell.bushuk@noaa.gov`). A referee *will* require the test be run, not just proposed. This is the single step that turns P3 from "a well-designed proposal" into "a result." If the data can't be obtained, retitle to "a proposed field test" and own that scope.
- **Add a moving-boundary (growing, `Re(s)>0`) solver check** to justify transferring `Im(s)` from the frozen (decaying) probe to real growing ripples.

**Net.** P3 is the strongest core result and the easiest to make submittable: reframe the title/abstract
around `I`, and execute the one real-data analysis the harness is already waiting for.

---

## 5. Paper 4 — subglacial hydrology (split required)

**Current framing risk (structural, not just framing).** This is 3–4 papers in one (the referee's
single most serious point): RTN + residence number + thermal-kernel falsification + MZ certification +
matched-lag null + `s_N` master curve + early-warning. No reader can state one thesis. **It must be
split before it is reviewable at all.** Beyond the split, two specific over-claims:

- **RTN is essentially the flotation criterion.** `RTN>1` ⟺ `ρ_w g d_base > φ ρ_i g H` ⟺ effective
  pressure `N=(1−φ)p_i` driven toward zero — i.e. *thickness approaching flotation*. The paper even
  writes `N_eff=(1−φ)p_i`. That `N→0` at the grounding line is textbook (Schoof 2007; Leguy et al.
  2014, *TC* 8, 1239 — the `N(p)=ρ_i g H(1−p H_f/H)` connectivity law; Cuffey & Paterson 2010). So
  "`RTN>1` concentrates near the grounding line" risks restating known geometry, exactly as the
  referee says.
- **The velocity/CSD ungrounding early-warning is anticipated.** Critical slowing down (rising
  variance + lag-1 autocorrelation) applied to **grounding-line flux for marine-ice-sheet-instability
  tipping** was done by **Rosier et al. (2021, *The Cryosphere* 15, 1501)** on Pine Island — using
  exactly lag-1 AC + variance + Kendall-τ. The paper cites Boers & Rypdal (2021, Greenland surface
  melt) but **not** Rosier — the closest prior art. This must be cited and positioned against.

**(A) Tools to cite, not claim.**
- Flotation / effective pressure / hydraulic potential: Shreve 1972; Schoof 2007; **Leguy et al. 2014**; Cuffey & Paterson 2010; Tulaczyk et al. 2000.
- Seawater intrusion at the grounding line (the live debate your RTN sits in): Walker et al. 2013; Sayag & Worster 2013; **Robel et al. 2022** (*TC* 16, 451); Bradley & Hewitt 2024; Rignot et al. 2024 (grounding *zone*).
- Sliding laws: Schoof 2005; Gagliardini et al. 2007; Joughin, Smith & Schoof 2019; Gudmundsson 2006/2007/2011.
- Critical-slowing-down EWS, *including the ice-sheet application*: Scheffer et al. 2009; Dakos et al. 2008; **Rosier et al. 2021**; Boers & Rypdal 2021.

**(B) The split and the genuine delta per child paper.**
- **P4a — The RTN intrusion screen.** *Delta only if it beats a baseline.* Foreground RTN as a
  **nondimensionalization of the flotation criterion that adds a connectivity fraction `φ`**, and the
  **independent BedMachine cross-check**. Honest contribution = a cheap continental screen *plus* the
  residence number `Ro` (thinning-paced vs hydraulic-limited), positioned inside the Robel/Bradley
  intrusion literature.
- **P4b — The thermal kernel falsified + the exact MZ replacement structure.** Foreground the
  **machine-precision MZ certification** of the cavity↔channel hydraulic memory kernel (a clean
  *structural* result) and the **honest-falsifier methodology**. Reframe the `H²/κ` thermal kernel as
  a *motivated null* (skin depth is metres), not a headline empirical result.
- **P4c (optional) — The field-measurable sliding law + early-warning.** The `s_N(N)=m/(1−(N_c/N)^m)`
  master curve, the `N_c` inversion, and the tidal-admittance probe, positioned *against* Rosier 2021
  as "a velocity/tidal-admittance operationalization of CSD for ungrounding," not a new EWS.

**(C) Demote / drop.**
- **Drop "verified" from the RTN title** unless RTN beats the baseline predictor (below).
- **Reframe the thermal-kernel falsification** as a motivated null, not "a clean falsification on real geometry" (it's a foregone conclusion — no established theory predicts surge lag = full-thickness diffusion time).
- **Demote the early-warning** to an operationalization of Rosier 2021, not a discovery.
- **Round suspicious significant figures** (`τ_ice ≈ 151,458 yr` → `≈1.5×10⁵ yr`).

**(D) Concrete edit steps.** Execute the split first (three child drafts with one thesis each), then
per child apply the tool/delta/demote edits above, then complete the reference list (every cited work
must appear — Tsai, Cuffey & Paterson, Nikuradse, Joughin/Smith/Schoof, Gagliardini, Schoof 2005,
Tulaczyk, Gudmundsson, Scheffer, Dakos, Boers & Rypdal, **Rosier**).

**(E) Validation before submittable.**
- **RTN baseline-skill test (load-bearing).** Show RTN adds skill over the trivial predictors it might
  be restating: "distance-to-grounding-line," "bed-depth-below-sea-level," and "thickness-above-
  flotation." Concretely: compute each scalar field on the same Bedmap2/BedMachine grids, and test
  whether `RTN` reorders intrusion-favourability beyond what those give. If it doesn't beat them, RTN
  is a repackaging (still publishable as a *clean nondimensional screen*, but not "verified").
- **Complete reference list** (hard editorial blocker).

**Net.** P4 contains good parts trapped in an unreviewable omnibus. The structural fix (split) plus
the RTN baseline and the Rosier citation convert it into 2–3 defensible papers.

---

## 6. Cross-cutting edits (apply to all four)

1. **Strip internal scaffolding** from manuscript text: `[VERIFIED]/[HYP]` residue, candidate numbering, `RESULT 16–20`, `§G.6`, "pending in the repo," "companion study." Keep these in the repo, not the papers.
2. **Add the explicit "what's new vs [closest prior work]" paragraph** to every introduction.
3. **One thesis sentence per paper**, stated in the first three abstract lines.
4. **Scope-as-feature** sentence in every abstract (a-priori / 2-D / directional / solver-only).
5. **Reference hygiene** — every in-text citation appears in the list with title, journal, DOI (P1/P3 already done; P2 needs the melt-theory refs; P4 needs the full set incl. Rosier).
6. **Tone down superlatives** ("decisive," "the benchmark that decides it," "verified") to what a-priori/directional evidence earns.

---

## 7. Recommended sequence (ship order) and readiness

| Order | Paper | Why | Gating work before submit |
|---|---|---|---|
| **1st** | **P3 scallops** | Cleanest keystone + a genuinely novel instrument (`I`); smallest gap | Run Bushuk `h(x,t)` reanalysis (harness ready) **or** retitle as proposed test; moving-boundary check; retitle around `I` |
| **2nd** | **P1 closure** | Reframable now; two real nuggets (`τ_c` scalar-independence, Dedner window) | Separate predicted-vs-fit; reframe as diagnostic/methods note; cut unavailable-companion ladder |
| **3rd** | **P2 melt** | One exact result + counter-consensus hook | Near-wall convergence + `η` study; cite Holland–Jenkins/roughness lit; 3-D or drop double-diffusion |
| **4th** | **P4 hydrology** | Needs structural surgery first | Split into 2–3; RTN baseline-skill test; cite Rosier 2021; complete refs |

---

## 8. What I will do next (gated on your go)

This document is the plan. On your word I will execute it paper-by-paper, in the order above. For each
paper, one focused pass:

1. **Reposition** — retitle, rewrite abstract + intro novelty paragraph, demote the mainstream
   mechanism to a cited tool, foreground the delta (edits to `papers/paperN_*.md`).
2. **Validate** — run the one load-bearing check that paper needs (P3 real-data reanalysis; P1
   predicted-vs-fit re-run; P2 near-wall convergence; P4 RTN baseline), committing the code + numbers.
3. **Reference + scaffolding cleanup**, regenerate TeX/PDF, and push to PR #1.

Tell me **"start with P3"** (my recommendation) or name another, and I'll do that full pass and report
back. If you'd rather I execute all four repositioning passes back-to-back without stopping between,
say so and I will.
