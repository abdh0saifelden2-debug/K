# New cross-relationship — NR31 (`general_two_clocks/new_relationships8.py`)

Continues the derived-and-verified program (NR1–NR30; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, deterministic; unit-proofs in
[`tests/test_new_relationships8.py`](tests/test_new_relationships8.py). Run:

```bash
python general_two_clocks/new_relationships8.py   # -> figures/nr31_2d_aposteriori_inverse_cascade.{json,png}
pytest general_two_clocks/tests/test_new_relationships8.py -v
```

---

## NR31 — Paper 1's 2-D a-posteriori null **is** the inverse cascade, not a closure failure: the optimal resolved eddy viscosity is `ν_opt = ⟨Π⟩/(2⟨|S|²⟩)`, so its sign is the sign of the net inter-scale energy flux; in 2-D that flux is ≈0 (energy cascades **up**-scale), so no positive-definite eddy viscosity can beat no-model — while in 3-D it is `>0` (forward, the NR29 sign) and a closure must help  [P1 §5b/§8.1 × NR29 × Kraichnan 2-D inverse cascade]

### The gap this closes

Paper 1 reports — in the abstract, §5b and §8 — that a-posteriori in **pure 2-D** "no
eddy-viscosity closure beats no model on the resolved spectrum, because the resolved scales need
near-zero net subgrid dissipation." Stated as a bound, it reads like a weakness of the closure.
NR31 shows it is instead a **prediction of 2-D physics**: the 2-D dual cascade (energy up,
enstrophy down; Kraichnan 1967, Batchelor 1969) forces the net resolved→subgrid **energy** flux to
≈0, so the truth wants a non-positive eddy viscosity and any positive `ν_t` over-drains the
resolved scales. The same logic **predicts the opposite ordering in 3-D**, which is exactly the
decisive a-posteriori test §8.1 defers to.

### Derivation

Model the resolved deviatoric SGS stress by a scalar eddy viscosity, `τ^d_ij ≈ −2 ν_t S_ij`, and
least-squares fit `ν_t` to the true stress:

> `ν_opt = argmin_ν ⟨|τ^d_ij + 2 ν S_ij|²⟩ = −⟨τ^d_ij S_ij⟩ / (2⟨S_ij S_ij⟩) = ⟨Π⟩/(2⟨|S|²⟩)`,  with  `Π = −τ^d_ij S_ij`.

So `sign(ν_opt) = sign(⟨Π⟩)`, and `⟨Π⟩` is the net resolved→subgrid energy flux at the cutoff
(`= Π_E(k_c)`). Cascade direction therefore fixes the sign:

- **2-D:** the inverse energy cascade carries energy to **large** scales, so `Π_E(k) < 0` for
  `k < k_f` and the *forward* energy flux at the cutoff is ≈0 → `ν_opt ≈ 0`. A positive-definite
  eddy viscosity (Smagorinsky, spectral-EV) removes resolved energy at rate `2 ν_t ⟨|S|²⟩ > 0` —
  there is no net forward flux for it to represent — so it cannot beat no-model (`ν_t=0`); the best
  non-negative choice is `ν_t = 0`. **(Paper 1's 2-D a-posteriori null.)**
- **3-D:** the forward energy cascade gives `⟨Π⟩ > 0` (NR29: the SGS flux is net forward,
  `μ = ⟨Π⟩ > 0`), so `ν_opt > 0` — a positive eddy viscosity is the right sign and a tuned /
  structural closure reduces the budget error below no-model. **(The decisive 3-D test.)**

**A-posteriori statement.** With `B(ν_t) = |2 ν_t ⟨|S|²⟩ − ⟨Π⟩|` the resolved-energy-budget
mismatch, `argmin_{ν_t≥0} B = 0` in 2-D (no-model optimal) but `= ν_opt > 0` in 3-D (a positive
closure strictly beats no-model).

### Numerical verification (`figures/nr31_2d_aposteriori_inverse_cascade.{json,png}`)

Reuses the repo's own solvers: 2-D forced DNS (`closure.dns2d.Vorticity2D`, `n=128`, `k_f=24`,
snapshot-averaged) and the 3-D closure package (`closure.dns3d`+`closure.sgs3d`, the NR29
machinery). All five checks pass (`ok = true`):

| check | result |
|---|---|
| 2-D spectral energy flux below `k_f` | `Π_E(k)` for `k=16…22`: `−0.08, −0.12, −0.16, −0.28, −0.39, −0.63, −1.0` — **negative, steepening toward `k_f`** (inverse cascade) |
| 2-D net SGS flux `⟨Π⟩/σ_Π` per cutoff | `k_c=16: −0.022`, `k_c=24: +0.006`, `k_c=32: +0.001` — **sign-indefinite, ≈0**; forward part `≤ 0.006` |
| 2-D optimal eddy viscosity | `ν_opt = +8.5×10⁻⁸ ≈ 0` (no positive `ν_t` to represent) |
| 3-D net SGS flux (live CPU) | `⟨Π⟩/σ_Π = +0.035 > 0`, `ν_opt = +4.8×10⁻⁵ > 0`, backscatter fraction `0.490 < ½` |
| 3-D forward, robust (six committed P100 runs) | `⟨Π⟩/σ_Π ∈ [0.039, 0.099]`, **all positive** across resolution and filter |
| a-posteriori argmin over `ν≥0` | 2-D `= 8.5×10⁻⁸ ≈ 0` (no-model), 3-D `= 4.8×10⁻⁵ > 0` (positive closure wins) |

### Why it matters

- **It reframes the flagship paper's main honest a-posteriori limitation** as a *consequence of
  2-D physics*, not a defect of the closure: in 2-D the resolved scales carry no net forward energy
  flux (the inverse cascade), so the optimal eddy viscosity is ≈0 and **no positive-definite
  closure can beat no-model** — exactly what P1 measured.
- **It is falsifiable as a 2-D-vs-3-D ordering:** the sign of the optimal resolved eddy viscosity
  must flip from ≈0 (2-D) to `>0` (3-D) because the net inter-scale energy flux does. A 2-D a-priori
  field showing a robust positive `⟨Π⟩` at the cutoff, or a 3-D field showing `⟨Π⟩≤0`, would
  refute it.
- **It sharpens the decisive test P1 §8.1 defers to:** because `⟨Π⟩>0` in 3-D, a structural
  closure with the right net dissipation *must* improve the resolved spectrum a-posteriori in 3-D
  — the prediction to check on the next GPU run.

**Mainstream tools (cited, not claimed):** Kraichnan (1967) and Batchelor (1969) 2-D dual cascade;
Leith (1968) / Kraichnan (1976) eddy viscosity; the a-priori least-squares ("optimal") eddy
viscosity (Clark/Bardina-style); NR29 for the 3-D forward-flux sign. The contribution is tying
Paper 1's 2-D a-posteriori null to the sign of the inter-scale energy flux and turning it into a
falsifiable 2-D-vs-3-D ordering prediction.
