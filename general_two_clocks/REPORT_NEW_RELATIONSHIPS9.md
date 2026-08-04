# New cross-relationship — NR32 (`general_two_clocks/new_relationships9.py`)

Continues the derived-and-verified program (NR1–NR31; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, deterministic; unit-proofs in
[`tests/test_new_relationships9.py`](tests/test_new_relationships9.py) (9 tests). Run:

```bash
python general_two_clocks/new_relationships9.py   # -> figures/nr32_drainage_response_window.{json,png}
pytest general_two_clocks/tests/test_new_relationships9.py -v
```

---

## NR32 — The drainage-response window: which lakes can surge is a **band-limited hydraulic transmission** that peaks at the cavity↔channel transition **iff channelization removes storage**; the Markovian (δ-kernel) collapse destroys the peak and predicts the *wrong ordering*; radar bed-echo specularity is the field map of the window  [P4a MZ kernel × §V.2d population result × Schroeder 2013]

### The gap this closes

The §V.2d modern matched-lag test (ATL15 × ITS_LIVE) found the §G.4 post-drainage surge is
**not universal**: 18/19 well-resolved drained lakes show no velocity response (≤ ~3 %), and
exactly one — `Thw_142`, a Thwaites active lake — shows a +8.5 %, 4.5σ, secular-trend-robust
step with lag-to-peak **1.125 yr**, inside the derived 0.02–2 yr band. The honest verdict
said the response "needs a dynamically-primed bed" — but left *primed* undefined. NR32
derives what primed means, why the population splits ~18:1, and which observable maps it.

### Derivation (mainstream pieces; the composition is the contribution)

1. **Lumped linear hydrology** (the same cavity↔channel linearisation whose Green's function
   this repo certified as an exact Mori–Zwanzig kernel; physics per Werder et al. 2013,
   Hewitt 2013, Kamb 1987, Röthlisberger 1972): a drainage impulse `ΔV` charges local
   storage over an input time `τ_in` (the ATL15 drawdown duration, ~a quarter), then relaxes
   through the drainage system with `τ_sys(c) = R(c)·C(c)`, where the evacuation resistance
   `R` **and** the storage capacitance `C` both fall with the channelization state
   `c ∈ [0,1]`. The effective-pressure response to the impulse is the two-stage cascade
   `Δp̂(ω) = ΔV·R(c)·H(ω)`, `H = 1/((1+iωτ_in)(1+iωτ_sys))`, with lag-to-peak
   `t* = τ_in τ_sys ln(τ_in/τ_sys)/(τ_in−τ_sys)`.
2. **The observing window.** A sliding response is *detectable* only inside the derived
   surge band `T ∈ [0.02, 2] yr` (faster is unresolved/aliased, slower hides in the secular
   trend — the exact §V.2d detection design). The detectable signal is the **band-limited
   transmission** `𝒯(c) = (ΔV R)² · (1/π)∫_{ω₁}^{ω₂} |H|² dω` — closed form via partial
   fractions (verified against quadrature to `5×10⁻⁸`).
3. **The dichotomy (the sharp new statement).** Log-slopes: `r = −dlnR/dc`,
   `s = −dlnτ_sys/dc = r + (−dlnC/dc)`. While `τ_sys` is above the window,
   `dln𝒯/dc = 2(s−r) = −2·dlnC/dc`; once below it, `dln𝒯/dc = −2r < 0`. Hence **an interior
   maximum exists iff channelization removes storage** (`decades_C > 0`): if channels merely
   *conducted* (fixed storage), the most distributed bed would respond most; because
   channelization *drains* storage (sheet/cavity area → channel volume), the response is
   **peaked at the transition**, where `τ_sys(c)` crosses the window. Verified: the criterion
   predicts the peak topology in **108/108** parameter combinations spanning
   `τ_dist ∈ [10,50] yr`, `decades_R ∈ [2,4]`, `decades_C ∈ [0,2]`, `τ_in ∈ [0.05,0.25] yr`;
   in all **81/81** interior cases the lag-to-peak at the transmission maximum falls
   **inside** the 0.02–2 yr window.
4. **The Markovian control (the two-clocks tie-in).** Collapsing the memory kernel to a
   δ with the same DC gain — the adiabatic-elimination closure that the repo's MZ
   certification degenerates to as the channel time vanishes — gives
   `𝒯_M(c) ∝ R(c)²·(ω₂−ω₁)`: **monotone**, maximal at the *most distributed* bed, no
   interior peak (verified). So the *observed selectivity* — nulls at distributed
   Siple-coast/interior lakes **and** at fast channelized outlets, detection at the Thwaites
   transition — is a **memory signature**: an adiabatic (K-theory-style) hydrology closure
   predicts the wrong ordering, not merely a smaller effect.
5. **The one detection, read honestly (two-sided).** `Thw_142`'s **lag** (1.125 yr, not
   same-quarter) excludes mature-channel states `τ_sys ≲ 0.25 yr`; its **detected amplitude**
   (8.5 % against the ~3 % population floor) excludes the deep-distributed end
   (`𝒯 < 0.35·𝒯_peak`); jointly this selects `c ∈ [0.14, 0.48]` on the central map — the
   transition neighbourhood. That is where Schroeder et al. (2013, PNAS) mapped Thwaites'
   distributed→channelized transition with radar specularity. (The lag *alone* is
   log-insensitive above the floor — `t* ≈ τ_in ln(τ_sys/τ_in)` — which is *why* the
   amplitude side of the window is needed; stated explicitly to avoid over-claiming.)
6. **The field map + forward predictions.** Specularity content measures the
   distributed(high)→channelized(low) axis (Schroeder 2013, 2015; Young et al. 2016; Dow et
   al. 2020). NR32 therefore predicts: (i) **population**: drainage-response detectability is
   a *single-peaked* function of upstream specularity (stackable with NISAR-era velocity +
   existing radar archives); (ii) **per-event**: within detections, lag-to-peak falls with
   channelization; (iii) **registered now, decidable later**: the East Antarctic
   ATL15-drained lakes in ICECAP coverage — `Byrd_1/2/s10` (local mean specularity
   0.085–0.102, ~2× the 0.05 coverage median on 3.5–7.2 k points each, i.e. wet-distributed
   side) — should **stay null** once post-2026 velocity accrues, *unless* gauged on the
   fast trunk their floods transit (the Stearns et al. 2008 Byrd-trunk speed-up after an
   upstream lake flood is the literature's in-band case of exactly that gauge-position
   corollary). Greenland's seasonal self-limiting speedups (Bartholomew et al. 2010; Sundal
   et al. 2011; Schoof 2010) are the same window statement evaluated where distributed
   `τ_sys` already sits in-band: the monotone limit of the peaked law.

### What is new vs the cited mainstream

The mainstream knows channelization buffers velocity response (Greenland seasonal
self-regulation; Schoof 2010's variability argument). New here: (i) the **band-limited
transmission form** and the **storage-removal dichotomy** (`decades_C > 0` ⟺ interior peak)
tying "dynamically primed" to `τ_sys` inside the observing window; (ii) the **Markovian
control** showing the peak — and hence *which* lakes respond — is carried by the memory
kernel (the same δ-collapse diagnosed across this repo); (iii) **specularity as the
remote-sensing coordinate** of the window, making the population prediction stackable from
archives; (iv) the **two-sided Thw_142 read** and the dated Byrd/David null predictions.

### Verified numbers (this run)

| check | value |
|---|---|
| closed-form band integral vs quadrature | rel. err `5.2×10⁻⁸` |
| criterion predicts peak topology | **108/108** combos |
| interior peaks with `t*` inside 0.02–2 yr | **81/81** |
| Markovian control | monotone, argmax at `c=0`, no interior peak |
| `Thw_142` lag floor | excludes `τ_sys < 0.251 yr` |
| `Thw_142` joint interval (central map) | `c ∈ [0.14, 0.48]` |
| Byrd_1/2/s10 local mean specularity | 0.098 / 0.085 / 0.102 vs coverage median 0.05 |

Artifacts: `figures/nr32_drainage_response_window.{json,png}`. The specularity anchor
recomputes only when the local USAP-DC 601371 download is present (see
`glaciers/validation/external/rtn_specularity_real.py` for provisioning); the committed JSON
carries the values from this run.

### Honest scope

The transmission model is lumped and linear (the same class the MZ certification covers);
`c`-maps are log-linear with literature-spanned decades, and the *existence* results are
map-robust (the 108-combo sweep) while the quoted `c` intervals are central-map readings.
No gridded intrusion/response survey exists; n=1 in-band detection means the population
prediction (i) is *registered*, not confirmed. Specularity is an anisotropic, geometry-
sensitive observable (Schroeder et al. 2015) and ICECAP coverage is regional (East
Antarctica) — the Byrd/David anchors are along-track means, not basin grids.
