# New cross-relationship — NR36 (`general_two_clocks/new_relationships13.py`)

Continues the derived-and-verified program (NR1–NR35; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only; unit-proofs in
[`tests/test_new_relationships13.py`](tests/test_new_relationships13.py) (9 tests,
one exercising the local NWIS downloads). Run:

```bash
python general_two_clocks/new_relationships13.py   # -> figures/nr36_well_barometric_kernel.{json,png}
pytest general_two_clocks/tests/test_new_relationships13.py -v
```

---

## NR36 — The well-barometric response **is** a two-clocks kernel, and open USGS records **select which one**: the drainage (lead) clock wins by ΔAICc > 270 over the RC and diffusive (lag) clocks  [Rojstaczer 1988 / Hsieh 1987 × NR11/NR30/NR32 kernel family × P4a identification grammar; executes horizon-ledger E6]

### The gap this closes

E6's claim was that the classic three-band well-to-barometer response is the
two-clocks kernel family with renamed constants. The execution goes further than the
claim: with nothing but *open* USGS NWIS records, the P4a identification machinery
(complex coherence-weighted fits + information criterion + null controls) not only
*fits* a family member — it **discriminates between family members**, selecting the
lead-type drainage element and rejecting the lag-type elements by decisive margins.

### The kernel family (every element already in the repo)

With `b` barometric head [m H₂O], `d` depth-to-water [m]:

| element | Laplace/Fourier form | repo identity |
|---|---|---|
| static plateau | `BE` | barometric efficiency |
| wellbore RC | `1/(1+iωτ_w)` — phase **−arctan(ωτ_w)** (lag) | **NR11** element |
| water-table drainage | `iωτ_l/(1+iωτ_l)` — phase **+90°−arctan(ωτ_l)** (lead); screens DC | **NR32** two-compartment face; DC screening = **NR30** operator analogue |
| vadose diffusion | `exp(−(1+i)√(ωτ_v/2))` (lag, √ω) | **B.2/G.4** diffusive `t^{−1/2}` family |
| two-path difference | `A₁ − A₂·V(ω)` | the program's operator split (instant elastic path − delayed diffusive path) |

Candidates M0 (static), M1 (RC), M2 (drainage×RC), M3 (vadose×RC), M4 (vadose),
M5 (two-path), M6 (two-path×RC), M7 (pure drainage) are fit to the measured
`W(f) = S_bd/S_bb` with weights `coh²/(1−coh²)` (Bendat–Piersol), earth-tide lines
excluded, AICc refereeing.

### Data and discovery

Programmatic NWIS scan (6 states) for **co-located** instantaneous 72019 (depth to
water) + 00025 (on-site barometric pressure): 92 candidate sites; the two longest
overlapping pairs are Modoc Plateau (CA) wells `415546121205401` and
`415104121232901` (2-hourly; 2021–2024 slices; 13 140 joint samples each; coverage
1.0). No auth, single `curl` per chunk (provisioning in the module docstring).

### Results

**Responsive well (415546121205401):** mid-band coherence 0.55, |W| ≈ 0.54.

| model | ΔAICc |
|---|---:|
| **M7 drainage** `BE·iωτ_l/(1+iωτ_l)` | **0** |
| M2 drainage×RC (τ_w → bound: RC unresolvable) | +2.6 |
| M5 two-path | +6.9 |
| M0 static | +270 |
| M1 wellbore RC (lag) | +273 |
| M4/M3 vadose (lag) | +294/+297 |

Parameters: **BE = 0.518** (68 % CI 0.513–0.542), **τ_l = 0.554 d** (CI 0.527–0.590;
segment-spectra bootstrap, no concatenation seams). The measured response *leads*
the barometer and dies toward DC — an unconfined/semi-confined system equilibrating
with its water table on a half-day clock, exactly Rojstaczer's water-table regime,
now stated as: **the data select the lead clock and reject both lag clocks at
ΔAICc > 270.**

**Null control (415104121232901):** coherence 0.05, |W| ≈ 0.02 — physically
non-responsive; no composition earns meaningful evidence (best-vs-static gap 30 vs
the responsive well's 270, at 25× smaller amplitude). The machinery does not
hallucinate structure where there is none.

### The convention guard (a bug class worth exporting)

`scipy.signal.csd(x, y)` returns `⟨conj(X)·Y⟩`, so the transfer of *d responding to
b* is `csd(b, d)/welch(b)` — **argument order first-input-first**. With the arguments
flipped every phase conjugates, lead becomes lag, and the model competition silently
selects the wrong kernel family (we caught this live: the conjugated spectrum
"preferred" the vadose lag family that the true spectrum rejects at +294). The suite
now carries a pure-delay regression test that fails on the flipped convention.

### Honest scope

Two wells, one responsive — this is an *identification demonstration* on open data,
not a hydrogeological survey; τ_l's aquifer meaning (specific yield × drainage
geometry) is not decomposed without well-construction metadata. The high-frequency
tail carries instrument quantisation (down-weighted by coherence, stated). S1/S2
radiational lines are excluded conservatively; the synthetic proof shows exactly why
(S2 lives coherently in both series with a non-BRF phase — weighting alone cannot
remove it; the mask can). The estimator itself is validated end-to-end on synthetic
truth (10 % parameter recovery; lag families rejected).

### Payoff hooks

- **P4a §4/§6 grammar validated off-glacier**: same fit-compete-null discipline, on a
  system with thousands more stations available (the 6-state scan found 92) — the
  promised "fully instrumented mainstream twin" of the subglacial identification.
- **E3 bridge**: the model set *is* the EIS element library (Warburg = M4's parent,
  RC = Randles branch); a tidal-constituent sweep on the same records would complete
  the bed-as-Randles program on an accessible system first.
- **NR32**: the drainage element's victory is the two-compartment face measured in a
  mainstream archive.

### Artifacts

`figures/nr36_well_barometric_kernel.{json,png}`; 9 unit proofs
(`tests/test_new_relationships13.py`): convention guard (pure delay must lag),
element phase identities (NR11 arctan / drainage lead / vadose √ω), tide-mask
geometry, end-to-end synthetic recovery + decisive rejection of wrong families,
S2-contamination proof that the mask is load-bearing, AICc parsimony on nested truth,
gap-accounting, chunk-safe parser (column-order permutation), real-well
identification regression (data-gated).
