# RESULT 12 — §D.4 / §G.5 unified-memory GLE coefficient closure

**Status:** `τ_c` **[MEASURED]** (value + sign); §G.5 add/remove-flux question
**[DERIVED]**; ice-bath `B, τ_d` **[DERIVED closed form, site-input dependent]**;
slow:fast **bath weight [DERIVED]** — it is the Stefan number `St = c_i·|θ_far|/L`.
Harness `gle_coefficients.py` → `figures/53_gle_coefficients.json`,
tests `tests/test_gle_coefficients.py` (7/7). No new data; no GPU.

## What was open

The *structure* of the unified memory formalism (§D.4 — additivity,
scale-selectivity, Markov limit) and of the clock-mismatch correction (§G.5 — the
exact commutator identity `[∂_t, D]θ = ∇·((∂_t K_u)∇θ)` and its dimensional
signature) was already derived and unit-tested. What stayed `[HYP]` were the
**coefficients**:

| coefficient | section | meaning | prior status |
|---|---|---|---|
| `τ_c` | §G.5, §D.4 fast bath | autocorrelation/memory time of the SGS eddy diffusivity `K_u` | [HYP] value **and sign** |
| (add/remove flux) | §G.5 | whether the correction is a source or a sink | [HYP] |
| `B`, `τ_d` | §D.4 slow bath | ice-kernel amplitude / cutoff (§B.2) | [HYP] amplitude |
| bath weights | §D.4 | relative `K_SGS : K_ice` contribution | [HYP] |

## What this result establishes

### A. `τ_c` is measured, and its sign is positive

`τ_c` is the decorrelation time of the SGS eddy diffusivity
`K_u = (c_s Δ)² |S̄|`. It enters **both** §G.5 (the *local* coefficient of the
commutator term) and §D.4 (the *fast* OU bath time). We measure it directly from
the solver (`direction_c_gpu_probe.CavityFlow`, n=24, CPU): record `K_u` at 64
fixed interior probe points over the measurement window and take the
autocorrelation memory time.

| closure | `τ_c` local (e-fold) | `τ_c` local (integral) | `τ_mem` mean-field | sign |
|---|---|---|---|---|
| Smagorinsky (white-FDT) | 2.67×10⁻² | 2.32×10⁻² | 2.49×10⁻² | **+** |
| two-clocks (`bs_tau=0.05`) | 3.24×10⁻² | 2.82×10⁻² | 2.20×10⁻² | **+** |

- **Value:** `τ_c ≈ 0.02–0.03` (solver time units) for the *local* `K_u`. This
  is consistent with the committed RESULT-8 SGS-force memory time
  (`τ_mem^set = 0.05`, `τ_mem^eff ≈ 9.5×10⁻³`): the local (un-averaged) time sits
  between the set and the CLT-shortened mean-field values, exactly as expected —
  the spatial-mean force decorrelates *faster* than the local field because it
  averages N³ quasi-independent cells. The local `τ_c` is the N-independent
  quantity that the *local* §G.5 PDE term and the §D.4 fast bath actually see.
- **Sign:** an autocorrelation time is non-negative **by construction**, and the
  measured value is strictly positive in every case. So `sign(τ_c) = +`. This
  half of the §G.5 `[HYP]` is now settled definitively (definition + measurement).
- Dimensionless clock `𝒞 = τ_c/τ_turn ≈ 0.018` (`τ_turn = H_cav/U₀ = 1.5`) — the
  two-clock separation is small but non-zero, consistent with the RESULT-8 finding
  that temporal recoloring is a weak effect on this grid.

### B. The §G.5 correction adds *no net* flux — it is a pure time-lag

Whether the correction "adds or removes flux" has a clean answer. Writing the
modeled flux divergence with the correction folded in,

```
∇·(K_u ∇θ) − τ_c ∇·(∂_t K_u ∇θ) = ∇·((K_u − τ_c ∂_t K_u) ∇θ) ≈ ∇·(K_u(t−τ_c) ∇θ)
```

to first order in `τ_c`. **The correction is a time lag by `τ_c`:** with `τ_c > 0`
the diffusion responds to the eddy diffusivity evaluated a memory time *in the
past*. The numerical check (synthetic spectral field, `K_u(t)` oscillating at rate
`ω`):

- **pure lag:** `‖corrected − lagged‖ / ‖frozen‖ = 1.1×10⁻⁴` (and this error scales
  as `τ_c²` — halving `τ_c` cuts it ≈4×, the Taylor signature);
- **net-zero in steady state:** averaged over a full `K_u(t)` cycle (so
  `⟨∂_t K_u⟩ = 0`), the time-mean of the correction is `|mean|/rms = 7×10⁻¹⁹`
  (machine zero), while its instantaneous RMS is O(`τ_c ω`) and sign-indefinite.

So the correction **adds** flux while `K_u` grows and **removes** it while `K_u`
decays, **netting to zero in statistically steady turbulence**. This is not a new
free parameter's worth of melt — it is a phase shift. **This quantitatively
explains the RESULT-8 / RESULT-11 null:** in the steady cavity `⟨∂_t K_u⟩ ≈ 0`, so
the clock-mismatch correction contributes ≈0 to the *mean* melt flux and to the
time-averaged `C_G` — exactly what those two measurements found (`R = 1.0004`,
`C_G` flat). §G.5 and the RESULT-8/11 null are now the same statement.

### C. Ice (slow) bath `B, τ_d` from the §B.2 closed form

§B.2 gives the ice memory kernel in closed form; only the *site* values
`θ_far` (far-field undercooling) and `V̄` (mean ablation speed) are empirical:

```
A   = k_th θ_far V̄² / (2κ²)               τ_d = κ / V̄²
G(τ) ≈ |A|·2√(τ_d/π)·τ^{−1/2}  (short-time) ⇒  B = |A|·2√(τ_d/π) / (ρ_i L)
```

Evaluated for representative subglacial inputs (`κ_ice = 1.09×10⁻⁶ m²/s`,
`k_th = 2.1`, `ρ_i = 917`, `L = 3.34×10⁵`, `θ_far = 2 K`):

| `V̄` | `τ_d` | `B` | fast/slow separation `τ_d/τ_SGS` |
|---|---|---|---|
| ~1 m/yr (3.2×10⁻⁸ m/s) | 1.07×10⁹ s (≈34 yr) | 2.2×10⁻⁷ s⁻¹ᐟ² | ~1.1×10⁸ |
| ~10 m/yr (3.2×10⁻⁷ m/s) | 1.07×10⁷ s (≈0.34 yr) | 2.2×10⁻⁶ s⁻¹ᐟ² | ~1.1×10⁶ |

(`τ_SGS ≈ 10 s` = representative basal-water eddy turnover.) The fast (SGS,
seconds) and slow (ice, months–decades) baths are separated by **~10⁶–10⁹ in
physical time** — the §D.4 scale-selectivity with real numbers in place of the
validator's placeholder `0.2`-vs-`60`. `B` and `τ_d` are *closed-form in material
constants × site inputs*, not universal numbers and not free fits; `τ_d ∝ V̄⁻²`.

### D. Bath weights `K_SGS : K_ice` = `1 : St` — the slow ice bath is Stefan-suppressed

The last open coefficient was the *relative weight* of the two baths. In the
Mori–Zwanzig / second-FDT picture each bath's weight is its zero-frequency (DC)
gain `∫K‪dτ`. The fast SGS/OU bath is unit-normalized by construction
(`∫K_SGS‪dτ = 1`, §D.4-(3); the OU kernel `(1/τ_c)e^{−τ/τ_c}` integrates to 1
independent of `τ_c`). The slow ice bath's relative weight is therefore the
*dimensionless* ice DC gain, which the §B.2 closed form fixes exactly:

```
∫K_ice dτ = (∫G dτ)/(ρ_i L) = −ρc·θ_far/(ρ_i L) = −c_i·θ_far / L      (signed; ∫G dτ = −ρc·θ_far)
St ≡ |∫K_ice dτ| = c_i·|θ_far| / L                                   (Stefan number, the bath weight)
```

so **`K_SGS : K_ice = 1 : St`**. For the representative undercoolings this is
`St ≈ 0.013` (`θ_far = 2 K`) to `0.063` (`10 K`) — the slow ice bath carries only a
*Stefan-number fraction* of the memory and is subdominant everywhere up to 10 K.
The weight is set entirely by `θ_far` (`∝ θ_far`, independent of `V̄`).

This is the **same Stefan number** as the §G.4 thermal-tail weight
(`thermal_tail_amplitude.py`: `W_thermal/W_hydraulic = St` against a unit-DC-gain
hydraulic kernel) — the two subsystems share the one §B.2 ice kernel, so the
slow-bath weight and the thermal-tail weight are literally the same number,
derived two independent ways. `test_ice_bath_weight_matches_thermal_tail_stefan_number`
pins the cross-module agreement (`St/c_i = θ_far/L` exact; values within the
`c_i` reference-temperature choice, 2100 vs 2009 J kg⁻¹ K⁻¹).

## Verdict

- `τ_c`: **MEASURED** (`≈0.02–0.03` solver units; sign **+**), consistent with
  RESULT-8 `τ_mem`. → §G.5 / §D.4 fast bath.
- §G.5 add/remove flux: **DERIVED** — a pure time-lag, **net-zero** mean in steady
  turbulence; explains the RESULT-8/11 null.
- `B`, `τ_d`: **DERIVED closed form** (§B.2), site-input dependent; fast/slow bath
  separation ~10⁶–10⁹.
- bath weights: **DERIVED** — `K_SGS : K_ice = 1 : St` with the fast bath
  unit-normalized (`∫K_SGS dτ = 1`) and the slow ice bath's relative weight equal
  to the magnitude of the dimensionless §B.2 DC gain — the signed gain is
  `∫K_ice dτ = −ρc θ_far/(ρ_i L) = −c_i θ_far/L`, so the weight is
  `St ≡ |∫K_ice dτ| = c_i |θ_far|/L`
  (the **Stefan number**, `≈ 0.013`–`0.063` for `θ_far = 2`–`10 K`). Site-input
  dependent (`∝ θ_far`) but always `≪ 1`; identical to the §G.4 thermal-tail weight
  `W_thermal/W_hydraulic`, so the slow ice bath is Stefan-suppressed.

## Reproduce

```bash
python gle_coefficients.py                 # ~2 min CPU -> figures/53_gle_coefficients.json
python -m pytest tests/test_gle_coefficients.py -q
python validation/synthetic/gle_memory_synthetic.py   # structure (unchanged, PASS)
python validation/synthetic/cmn_synthetic.py          # identity (unchanged, PASS)
```
