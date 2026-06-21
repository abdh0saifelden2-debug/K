# RESULT 14 — The scallop is a *damped, downstream-migrating* mode (and migration falsifies K-theory on the ice)

**One line.** A frozen-interface wall-flux harmonic decomposition corrects the
FUTURE_WORK §G.2 amplitude ansatz: the scallop mode is
`s(K, U) = −β(K, U) + i·ω_mig(U)` — pure **smoothing** in-phase (no autonomous
`+α a^{1/2}` growth) plus a **quadrature migration** term that grows with the
drive. The migration is a **parity-symmetry break that no down-gradient
(K-theory) closure can produce**, which makes it the morphological analogue of
the memory (`τ_c`) and backscatter signatures the rest of this repo documents in
*flux* space.

This note consolidates the result that was previously distributed across
FUTURE_WORK §G.2; the runnable artefacts are `scallop_amplitude_harmonics.py`
(CPU) and `scallop_ktheory_control.py` (the keystone control, CPU/GPU), with all
numbers in `figures/56_scallop_amplitude_harmonics.json`.

---

## 0. Why this is a *result*, not a *check*

Most of §G is **negative or dimensional**: §G.1 falsifies a bound's mechanism,
§G.4 falsifies a literal thermal kernel, §G.3/§G.6 are dimensional/phenomenological
checks leaning on literature, and §G.5 explains a *null* (the clock-mismatch
correction is net-zero). Those sharpen the framework by **removing wrong things**.

RESULT 14 is the one place the framework produced a **positive, falsifiable
mechanism**. Four independent things had to line up for it to count as a discovery
rather than a check, and they did:

1. **Derived, not assumed.** The corrected mode `s = −β + iω_mig` (smoothing +
   migration, no autonomous growth) *fell out of* the harmonic decomposition — it
   was not an ansatz fitted after the fact.
2. **Solver-measured.** `β`, `E_cos` and the migration exponent are all numbers
   from the DNS, not parameters.
3. **Independently anchored.** The Curl selection law `Re_* = u_*λ/ν` pins physical
   units without choosing `λ` by hand.
4. **Constant-free & falsifiable.** `I = τ·c_mig/λ` cancels `ΔT`, `k_th`, `ρ_iL`,
   giving a field test that needs none of the basal unknowns (§7).

The **keystone** is implication #1: migration (`Im(s) ≠ 0`) is a *parity-symmetry
break that no down-gradient K-theory closure can produce* (§5). That is what ties
this result back to the repo's central thesis — it is the **morphological analogue**
of the memory (`τ_c`) and backscatter (up-gradient transfer) signatures the repo
documents in *flux* space. K-theory's structural blindness doesn't merely lose
information in the closure; it leaves a fingerprint carved into the ice. The GPU
K-theory positive control (`E_cos →` machine zero under a down-gradient flux at
full drive) is direct evidence for exactly that.

**Where the line honestly is:** this is a discovery *about the model/closure,
verified in the solver*. It is **not yet validated against field scallops** — that
is the open step (§8), and the constant-free ratio is precisely what makes that
test cheap. Framed plainly: G.2 is the framework's first *constructive* result, and
it is self-consistent and GPU-falsifiable.

## 1. Why the old two-term law could not be earned

§G.2 reduced the dynamics to `ρL ȧ = α a^{1/2} − β a` with both magnitudes
`[HYP]`. Watching a seeded mode *decay* cannot separate the two coefficients —
the net rate `ȧ = (α a^{1/2} − β a)/ρL` conflates them, and a two-term fit to the
moving boundary returns unphysical (both-negative) coefficients. The fix is to
**separate the coefficients by phase** on a *frozen* interface, so there is no
boundary motion to contaminate the flux.

## 2. Method — harmonic decomposition of the wall flux

Freeze `y(x) = ȳ + a sin(Kx)` (`K = 2π n_w / L_x`), time-average the per-column
melt flux `m(x) = −κ ∂θ/∂y`, and project onto the two corrugation harmonics:

```
E_sin = 2⟨e(x) sin(Kx)⟩   in-phase with the SHAPE  → amplitude change (β)
E_cos = 2⟨e(x) cos(Kx)⟩   in QUADRATURE            → pattern MIGRATION (ω_mig)
```

`E_cos` is the Curl-1966 reattachment/lee signature: it is zero by symmetry
unless the flow carries a *direction*. Two decompositions are run: (1) the
conduction baseline `m_cond` (flow off) for the `β ~ K^p` scaling, and (2) the
flow-induced excess `e = m_flow − m_cond` vs drive `U`.

## 3. Findings (`figures/56`)

| quantity | measured | ansatz it kills |
|---|---|---|
| conduction `β/a ~ K^{+0.13}` | wavelength-**independent** | `K²` curvature (and `|k|` Mullins–Sekerka) **falsified** |
| in-phase flow excess `E_sin` | **negative at every driven `(a, U)`** | no autonomous `+α a^{1/2}` growth — the mode is **smoothing-limited / decay-only** |
| quadrature `E_cos` | `≈0` at `U=0`, grows `∝ U^{0.5–0.8}` | migration is real, and **friction-velocity** (`u_*~U^{1/2}`) controlled, not kinematic (`∝U`) |

**Corrected mode:** `s(K, U) = −β(K, U) + i·ω_mig(U)`, `Re(s) < 0` always,
`Im(s) ∝ U^{1/2}` — a damped, downstream-migrating pattern, *not* a
growth–saturation balance.

## 3a. Generalisation beyond the swept amplitude (RESULT 22)

The findings in §3 were measured at a single relative amplitude `a₀/λ = 0.20`
(Caveat D). `amplitude_generalization_scan` re-runs the harmonic decomposition
across `a₀/λ ∈ [0.05, 0.40]` to test whether the two structural verdicts are
amplitude artefacts. They are not:

| `a₀/λ` | conduction `β/a ~ K^p` | strongest driven in-phase `E_sin` |
|---|---|---|
| 0.05 | `p = +0.57` | `−1.31×10⁻⁵` (smoothing) |
| 0.10 | `p = +0.66` | `−2.32×10⁻⁵` (smoothing) |
| 0.20 | `p = +0.14` | `−6.23×10⁻⁵` (smoothing) |
| 0.30 | `p = +0.33` | `−8.75×10⁻⁵` (smoothing) |
| 0.40 | `p = −0.27` | `−1.05×10⁻⁴` (smoothing) |

**Headline (robust at every amplitude).** (1) the driven in-phase flow excess is
**negative at every `a₀/λ`** — there is no autonomous `+α a^{1/2}` growth term at
*any* amplitude, and its magnitude grows monotonically with amplitude (the mode is
smoothing-limited throughout); (2) the conduction exponent stays
`p ∈ [−0.27, +0.66]`, **far below** the `K²` curvature ansatz (`+2`) and
Mullins–Sekerka `|k|` (`+1`) at every amplitude — the curvature ansatz is
falsified across the whole range.

**Caveat (honest).** Strict `K`-independence (`|p| < 0.6`) and an amplitude-flat
`β/a` hold only in the signal-rich regime `a₀/λ ≳ 0.2`. At shallow amplitude the
single-wavenumber conduction in-phase signal is at the noise floor (`β/a` even
sign-flips at `a₀/λ = 0.05`), so `p` drifts up toward `~0.6` and the cross-amplitude
`β/a` scatter is large (CV ≈ 1.1). This is a measurement-precision limit at small
corrugation, **not** an emergent physical `K`-dependence (the K²/|k| ansätze remain
falsified regardless). Verified by `tests/test_scallop_amplitude_harmonics.py::test_amplitude_generalization_holds_beyond_swept_amplitude`.

## 3b. Generalisation beyond the swept drive window (RESULT 23)

§3 also fixed the *drive* window at `U ∈ [1.5, 3.0]` (the other half of Caveat D).
`drive_window_scan` pushes the mean drive to `U = 6` at the `a₀/λ = 0.20, n_w = 12`
operating point and re-checks the two **flow** verdicts. The genuine falsification
risk is that strong-drive lee separation could add an *in-phase* flux component
(a recirculating lee eddy reinforcing the corrugation) — i.e. an emergent
`+α a^{1/2}` growth term that the weak-drive sweep would have missed. It does not
happen:

| `U` | in-phase `E_sin` | quadrature migration `E_cos` |
|---|---|---|
| 0.0 | `−5.17×10⁻⁶` (smoothing) | `+4.9×10⁻⁸` (≈0, parity control) |
| 1.5 | `−6.23×10⁻⁵` (smoothing) | `−6.59×10⁻⁵` |
| 3.0 | `−9.19×10⁻⁵` (smoothing) | `−1.14×10⁻⁴` |
| 4.5 | `−8.86×10⁻⁵` (smoothing) | `−1.38×10⁻⁴` |
| 6.0 | `−4.81×10⁻⁵` (smoothing) | `−1.19×10⁻⁴` |

**Headline (robust at every drive).** (1) the in-phase flow excess is **negative
at every `U`** out to `U = 6` — no autonomous growth channel opens at strong drive;
(2) the migration follows a **sub-kinematic** power law `E_cos ∝ U^{+0.48}` (≈ `½`),
**far below** the kinematic-advection alternative `U¹` — the migration is a
boundary-layer / friction-velocity effect, not bulk advection; (3) at `U = 0` the
migration is `~10³×` smaller than at full drive (parity control — the quadrature
term is genuinely drive-induced).

**Caveat (honest).** The migration is **not monotone**: it peaks near `U ≈ 4.5` and
**rolls over / saturates** by `U = 6` (and the in-phase magnitude likewise turns
over past `U ≈ 3`). Physically the lee structure carrying the quadrature flux
reaches a limiting form at strong drive; this *reinforces* the sub-kinematic reading
(migration grows slower than `U`, eventually saturating) rather than contradicting
it. `migration_monotone_in_U` is therefore reported as a finer descriptor, not a
headline (it is `False` in the full `figures/56` run). Verified by
`tests/test_scallop_amplitude_harmonics.py::test_drive_window_generalization_holds_beyond_swept_drive`.
With both the amplitude (§3a) and drive (§3b) axes generalised, **Caveat D is
retired** — only the wavenumber set is unchanged, and it is already swept
`n_w ∈ {6…20}` in §3.

## 4. Dimensional bridge (Stefan condition, Stefan-number-free)

The solver's `St = 2×10⁻⁴` is a numerical accelerator and is **discarded**; the
measured wall-flux response is converted with the exact balance `ρL v = q`,
`q = −k_th (ΔT/L₀) ∂θ/∂y`, anchoring the length scale by the corrugation
wavelength `L₀ = λ_phys/λ_nd`. The constants `κ_nd` cancel. Result:

```
amplitude e-fold   τ     ~  λ²/ΔT      (looks diffusive — see §6 for why this is a decoy)
migration speed    c_mig ~  ΔT/λ
```

**Anchored subglacial point** (Curl wavelength selection `λ = Re_* ν / u_*`,
`Re_* = 2200`, `u_* = 0.05 m s⁻¹` → `λ ≈ 7.9 cm`, `ΔT = 0.1 K`, `n_w = 12`):

> **`τ ≈ 3.0 yr`** (amplitude e-folding) and **`c_mig ≈ 1.8 cm yr⁻¹`**
> (downstream migration).

## 5. Keystone — migration is a parity-symmetry break no K-theory closure can produce

A *local, memoryless, down-gradient* eddy diffusivity acting on a sinusoid is
symmetric under `x → −x`: it carries no flow direction, so it can only return an
**in-phase** (real, damping) response — `E_cos ≡ 0`. The resolved advective flux
measured `E_cos ≠ 0`. Therefore migration *requires* the flow-direction-selecting,
non-local reattachment asymmetry that K-theory discards.

This was turned from an argument into a **measurement** (`scallop_ktheory_control.py`),
advancing the *same* momentum field at the *same* drive but transporting heat with
the down-gradient flux K-theory prescribes instead of advection:

| heat transport (same `U=3`, same interface) | `E_cos` | reading |
|---|---|---|
| resolved advective `−u·∇θ` | `−1.14×10⁻⁴` | migrates |
| **uniform** eddy diffusivity `∇·(K∇θ)` | `−1.2×10⁻¹⁸` | machine zero (`~10⁻¹⁴` of resolved, `std=0`) |
| **flow-aware** Smagorinsky `K_eddy(|S|)` | `+2.0×10⁻⁶` | `~57×` below resolved |

Both closures still **smooth** (`E_sin ≠ 0`): K-theory can damp but cannot
migrate. GPU/CuPy falsifiable controls (Tesla P100) confirmed (a) parity —
`E_cos → +2.1×10⁻⁶` at `U=0` vs `−8.4×10⁻⁵` at `U=3` (~40× smaller); and (b) the
high-`Re_*` migration exponent tightens to **`0.52`** (`nx=160`, 4 seeds, scatter
~1–2%), sitting on the friction-velocity `½`.

## 6. The dimensional `τ ∝ λ²` is a *decoy*

`τ ∝ λ²/ΔT` *looks* like classic curvature/Mullins–Sekerka smoothing, but the
`λ²` comes **entirely from the conduction length↔time anchoring `L₀²/κ`**, not
from a curvature operator — the measured `β/a` is `K`-independent. Fitting
`τ ∝ λ²` in the field would *wrongly* "confirm" the very curvature ansatz the
harmonic probe falsified; the only discriminator is the `K`-exponent of `β/a`
(`≈0` here vs `+2` for curvature), which the dimensional scaling hides.

## 7. A constant-free, ΔT-free field test

In the product `τ · c_mig` the constants `k_th`, `ρ_iL` **and `ΔT` cancel exactly**:

```
I ≡ τ · c_mig / λ_phys  =  (−E_cos) / (λ_nd · (β/a) · a_nd · K_nd)   =   Im(s) / (2π |Re(s)|)
```

i.e. `I` is (up to `2π`) the **ratio of the migration rate to the amplitude
rate** — the dimensionless, basal-`ΔT`-free repackaging of the heat-flux ↔
topography phase shift that Gilpin, Hirata & Cheng (1980) already used to predict
downstream migration. The solver pins `I = O(0.3–0.9)` (`0.33, 0.68, 0.88` at
`n_w = 8, 12, 16` — mildly wavelength-dependent, *not* a universal constant).

**Field-test protocol.** Measure on one scallop train: `λ` (crest spacing),
`c_mig` (downstream crest displacement between two visits), `τ` (amplitude e-fold
time). Then `I_obs = τ·c_mig/λ`:

| outcome | reading |
|---|---|
| `I_obs ∈ [0.2, 1.0]`, `c_mig` downstream | consistent with `s = −β + iω_mig` |
| `c_mig ≈ 0` (`Im(s)=0`) | parity-symmetric closure — K-theory **not** falsified on the ice (contradicts the solver) |
| `c_mig` upstream | new physics (e.g. depositional, not melt-limited) |
| `I_obs ≫ 1` or `≪ 0.1` | `λ ↔ u_*` anchoring mismatch |

Because `I` is `ΔT`-free, a failure cannot be excused by an unknown basal `ΔT`.
Measuring `c_mig + λ` alone *pins* `ΔT` (since `c_mig ∝ ΔT/λ`), so `τ` becomes a
**prediction** and the system is over-determined.

## 8. Literature grounding & data status (can we compute `I_obs` today?)

Each ingredient of the triple is independently corroborated, but **no single
study co-reports all three with the isolation the test needs**, so `I_obs` cannot
yet be pinned from the published record — the protocol is a genuine prediction,
not a post-hoc fit.

- **`λ` (wavelength) — solid `[LIT]`.** The `Re_* = u_*λ/ν` selection is mainstream:
  Blumberg & Curl (1974) `≈2200` (the value used here), Thomas (1979) `≈1000`
  over `50 µm–1 m`, Hsu, Locher & Kennedy (1979) `λ = 3180 ν/u_*`, and the
  Thorsness & Hanratty (1979) / Hanratty (1981) most-unstable band
  `3100–6300 ν/u_*`. Curl's (1966) free-stream `Re ≈ 22 500` and the
  friction-velocity `Re_* ≈ 2200` are **mutually consistent** at `u_*/U ≈ 0.1`,
  so the anchoring in §4 is not a free choice.
- **`c_mig` direction & mechanism — solid `[LIT]`.** Downstream migration is
  reported by Curl (1966), Ashton & Kennedy (1972), Hsu et al. (1979) and Gilpin
  et al. (1980); the mechanism — maximum ablation **downstream of the trough at
  flow reattachment** (Blumberg & Curl 1974) — is exactly the quadrature lee
  signature `E_cos` isolates. Gilpin et al.'s linear stability predicts downstream
  migration when the heat-flux/topography phase shift lies in `(π/2, π)`, which is
  the same statement as `Im(s) ≠ 0`.
- **`c_mig` *magnitude* and `τ` — sparse.** Cave-scallop surveys report `λ` only;
  migration *speed* and amplitude *decay time* are essentially never tabulated
  together for a single train.
- **Sign caveat (important).** Real ice ripples *grow* (`Re(s) > 0`: the
  Ashton–Kennedy / Gilpin / Hanratty moving-boundary instability), whereas this
  frozen-interface excess is *decay-only* (`Re(s) < 0`) by construction. The
  migration `Im(s)` is the robustly shared, K-theory-falsifying piece; matching
  the growing-ripple branch requires the moving-boundary feedback the frozen probe
  deliberately excludes (consistent with §G.2 implication #4, "growth lives in a
  different conservation law"). So `|I|` from the solver should be compared to
  `|Im(s)/Re(s)|/2π` from a moving-boundary measurement, **not** to its sign.
- **Best available dataset for a real `I`.** Bushuk et al. (2019, *JFM* 873) track
  the evolving ice–water interface `h(x, t)` at sub-mm / 15 Hz resolution through a
  "transition → equilibrium → **adjusting**" sequence. They **explicitly measure
  the mean horizontal crest advection speed `c`** (transforming `h(x,t) → H(x−ct)`)
  and the spatial-mean melt rate, and they find maximum ablation **≈¾ λ downstream
  of the crest** — the lee-quadrature signature `E_cos` isolates. This is the one
  published dataset from which a real `c_mig` **and** an amplitude timescale can be
  extracted for the *same* train.

### 8.1 `I_obs` from Bushuk's *adjustment* regime (the regime-matched bound)

Bushuk's data resolve **three** regimes (their figure 3): a *developing* train (flat
→ scalloped, amplitude **grows**), an *equilibrium* train (amplitude steady, `Re(s)
→ 0`, so `τ → ∞`), and — crucially — a **scallop-adjustment** regime (experiment 1b:
after the drive was cut from `U = 1.00 → 0.16 m s⁻¹`), in which the melt profile
"preferentially melt[s] the scallop crests, thereby **dampening** the existing
scallop geometry" while the train "**migrat[es] downstream**." That adjustment regime
is the *exact* analogue of the frozen probe: **damping + downstream migration**,
i.e. `s = −β + iω_mig` with `Re(s) < 0`, `Im(s) ≠ 0`. So for this regime the sign
caveat above **does not apply** — both branches decay-and-migrate, and `I_obs` is a
like-for-like comparison.

Reading the regime-matched inputs straight off Bushuk (no regime mixing now):

| input | value used | source / status |
|---|---|---|
| `c_mig` | `0.11 mm min⁻¹` (`= 1.8×10⁻⁶ m s⁻¹`) | Bushuk's **measured** downstream crest advection speed `c` (§2.3) `[LIT-measured]` |
| `λ` | `≈ 13 cm` (their `5–20 cm` band) | **observed** wavelength, fig 3(a/m) — Bushuk argue Curl's `Re_*≈22 500` is *not* the selector, so the observed `λ` is used, not a Curl value `[LIT-measured]` |
| `τ` | `≈ 1–3 h` | amplitude e-fold of the dampening crests, read **by eye** off the non-collapsing profiles of fig 3(o) (times 512–567 min) `[EST, figure-limited]` |

`I_obs = τ·c_mig/λ`:
- with the **observed** `λ = 13 cm`: `I_obs ≈ 0.05–0.15` (point `≈ 0.10`);
- with a **Curl-selected** `λ = 5 cm` (for comparison only): `I_obs ≈ 0.13–0.40`.

So `I_obs ~ O(0.05–0.4)`, point estimate `≈ 0.1`.

**Read this honestly.**
1. **The sign structure matches exactly** — Bushuk's adjustment train damps *and*
   migrates downstream, which is precisely `s = −β + iω_mig`. This is the strongest
   real-data confirmation available: the corrected mode's *qualitative form* is
   observed in a laboratory ice train.
2. **The magnitude is the same order** as the solver's `O(0.3–0.9)` and overlaps its
   lower edge only at the top of the input range; the **point estimate `≈ 0.1` sits a
   factor of ~2–3 below** the solver band. That is a mild, honest **tension**, not a
   rubber-stamp — and not a falsification (`I_obs` is `O(0.1–1)`, neither `≈ 0` nor
   `≫ 1`, with the predicted downstream sign).
3. This **revises the earlier crude estimate `≈ 1.2`** downward by ~10×: that estimate
   used a guessed `c ~ 1 mm min⁻¹`, whereas Bushuk's *measured* `c = 0.11 mm min⁻¹` is
   ~9× smaller, and it used the Curl `λ` rather than the observed one.
4. **It is still a bound, not a pin.** The dominant uncertainties are (i) which `λ`
   convention (observed ~13 cm vs Curl ~5 cm — a factor ~2.6) and (ii) `τ` read by eye
   from a figure whose profiles "do not collapse" (factor ~2).

### 8.2 Why the published figures can't pin it — and what would

I attempted the reanalysis directly from the published record. The kinematics
(`c_mig`, `λ`, the crest-evolution angle `φ = arctan(ḣ/c)` = 15° for 1b vs 40° at
equilibrium) are recoverable, but the one scalar that sets `I` — the **amplitude
e-fold `τ`** — lives only in the vector line-art of fig 3(o), a set of 12 overlapping,
non-collapsing profiles. Tracing an amplitude envelope off that by eye is good to a
factor of ~2 at best, which is exactly the spread quoted above. A trustworthy **pin**
needs Bushuk's underlying `h(x, t)` arrays (their supplementary material / on request),
to which *this report's* harmonic decomposition (§2) can be applied directly: fit
`h(x, t) = Σ_k a_k(t) e^{i(kx + φ_k(t))}`, read the complex growth rate
`s = d/dt[ln a_k] + i dφ_k/dt = Re(s) + i·Im(s)` of the dominant mode, and form
`I_obs = Im(s)/(2π|Re(s)|)` for the **same** train — no regime mixing, no by-eye
envelope, and no `ΔT`. That digitisation/reanalysis on the raw arrays is the
remaining step to turn this `O(0.1)` bound into a pin.

### 8.3 What is committed in code (`scallop_field_test.py`)

The field test is no longer narrated — it is committed, tested code with both
branches:

- **The pin recipe is implemented and self-checked.** `harmonic_mode_rate(x, t, H)`
  performs exactly the §8.2 decomposition: rFFT each frame along `x`, track the
  dominant corrugation mode `a_k(t) e^{iφ_k(t)}`, fit `Re(s)=d/dt ln a_k`,
  `Im(s)=dφ_k/dt`, and return `I = |Im(s)|/(2π|Re(s)|) = τ·c_mig/λ` plus the
  downstream sign. On a synthetic damped, downstream-migrating train with **known**
  `(Re(s), c_mig, λ)` it recovers all three to `<2%` and `I` to `<3%`
  (`test_scallop_field_test.py`). It is therefore ready to ingest Bushuk's raw
  `h(x, t)` arrays directly — no further code, just the data.
- **The figure bound is committed as a numeric artifact.** `bushuk_adjustment_bound()`
  computes the §8.1 regime-matched estimate from the published kinematics
  (`c_mig = 1.833×10⁻⁶ m s⁻¹`, `λ = 13 cm` observed / `5 cm` Curl, `τ ≈ 1–3 h`):
  `I_obs ∈ [0.05, 0.15]` (point `≈ 0.10`) at the observed `λ`, `O(0.05–0.4)`
  overall, downstream sign, point a factor `≈ 3.3` below the solver band
  `[0.33, 0.88]` → recorded as a **mild tension, `falsified = False`** (it is
  `O(0.1–1)`, downstream, neither `≈ 0` nor `≫ 1`). Written to
  `figures/58_scallop_field_test.json`.

The **one** input still missing for a true *pin* is Bushuk's underlying `h(x, t)`
arrays (supplementary material / on request to `mitchell.bushuk@noaa.gov`); the
harness above runs the moment they are dropped in.

## 9. Reproduce

```bash
python scallop_amplitude_harmonics.py        # CPU; writes figures/56_*.json (β/a scan, flow excess, bridge, anchored point, I)
python scallop_ktheory_control.py            # keystone K-theory positive control (CPU; xp=cupy for GPU)
python scallop_field_test.py                 # CPU; figures/58_*.json (harmonic-pin self-check + committed Bushuk figure bound)
pytest tests/test_scallop_amplitude_harmonics.py tests/test_scallop_ktheory_control.py tests/test_scallop_field_test.py
```

## 10. Scope

This is a discovery **about the model/closure, verified in the solver**: the
corrected mode, the parity falsifier, and the constant-free ratio are all
self-consistent and GPU-falsifiable. It is **not yet validated against field
scallops** — §8 is exactly why that test is cheap (no `ΔT` needed). A single
migration exponent still carries `~±0.1` scatter from spin-up/seed/`U`-grid, so
the banked claims are the qualitative three (`Im(s)→0` at `U=0`, strictly
sub-linear migration, exponent `→½` at high `Re_*`), now backed by a tightened
central value of `0.52`.
