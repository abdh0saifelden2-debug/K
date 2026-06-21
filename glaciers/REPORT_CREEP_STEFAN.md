# §D.6 (completion) — coupled Stefan + Glen-creep amplitude sweep

*Runner:* [`validation/synthetic/creep_stefan_coupled.py`](validation/synthetic/creep_stefan_coupled.py)
· *Test:* [`tests/test_creep_stefan_coupled.py`](tests/test_creep_stefan_coupled.py)
· *Artifact:* `validation/reports/creep_stefan_coupled.{json,png}`

```bash
python glaciers/validation/synthetic/creep_stefan_coupled.py
pytest glaciers/tests/test_creep_stefan_coupled.py -v
```

## What this closes

`FUTURE_WORK.md` §D.6 established Glen's-law creep as a **[NULL]** for the scallop
amplitude — over the *solver clock* (seconds–hours) the creep wall displacement is
<1 % of the roughness amplitude (`creep_scaling_synthetic.py`), justifying the
rigid-wall Brinkman penalization. §D.6 explicitly left **one residual un-run**:

> "The full coupled Stefan+creep simulation across the amplitude sweep is not run
> here … but would quantify the (small) long-time smoothing correction."

This module runs that residual — **without re-running any DNS**. The melt (Stefan)
amplitude-smoothing rate is taken from the *already-measured*, committed RESULT 14
harmonic decomposition (`figures/56_scallop_amplitude_harmonics.json`,
`β/a` per mode → the dimensional bridge `τ_melt ~ λ²/ΔT`); the Glen-creep sink is
added analytically and the **coupled** amplitude ODE is integrated over the physical
multi-year melt timescale.

## Model

Corrected RESULT 14 amplitude dynamics is smoothing-limited, `ȧ|_melt = −β_melt a`
(`s = −β + iω_mig`). Creep adds a same-sign sink `ȧ|_creep = −r_creep a` (§D.6 sign
argument), so the coupled equation is `ȧ = −(β_melt + r_creep) a` and

> `a*_coupled / a*_melt = β_melt / (β_melt + r_creep) = 1/(1+ρ)`,  `ρ ≡ r_creep/β_melt`.

`r_creep = A σ_dev³` (Glen, `n=3`; Cuffey & Paterson 2010). The only modelling choice
is the deviatoric stress `σ_dev` that relaxes a corrugation. The robust,
choice-free quantity is the **critical stress** where creep matches melt,
`σ_crit = (β_melt/A)^{1/3}`, compared with the physical relief stress `ρ_i g a`.

## Result (Curl anchor `λ≈7.9 cm`, `ΔT=0.1 K`, `n_w=12`)

- **Melt clock:** `β_melt = 1.05×10⁻⁸ s⁻¹` → `τ_melt = 3.03 yr` (matches the committed
  RESULT 14 anchor).
- **Crossover stress:** `σ_crit ≈ 0.16 MPa` (temperate) / `0.35 MPa` (cold).
- **Physical relief stress** `σ_dev = ρ_i g a` runs `35→283 Pa` across
  `a/λ∈[0.05,0.40]` — **~2300× below** `σ_crit` — so:

| `a/λ` | `σ_topo` [Pa] | `ρ` (temperate) | `a*_coupled/a*_melt` |
|---|---|---|---|
| 0.05 | 35 | 1.0×10⁻¹¹ | 1.000000000 |
| 0.10 | 71 | 8.2×10⁻¹¹ | 0.99999999992 |
| 0.20 | 142 | 6.5×10⁻¹⁰ | 0.99999999935 |
| 0.40 | 283 | 5.2×10⁻⁹ | 0.99999999477 |

The long-time creep smoothing correction is **quantified and negligible** —
`ρ ≤ 5×10⁻⁹`, i.e. creep is **8–11 orders of magnitude** slower than melt smoothing,
the coupled steady amplitude departs from the melt-only value by `<10⁻⁸`, and the
sign is always smoothing (never enhancement). This is consistent with the [LIT]
clincher that morphologically identical scallops form on **non-creeping limestone**
(Curl 1966) — creep is not needed to set the amplitude.

## The honest caveat (why §D.6 used the solver clock)

The §D.6 displacement bound uses the *worst-case* stress `σ_dev = N` (the full
effective pressure). That is a deliberate upper bound, and §D.6 only ever evaluates
it over the **solver clock** (1 hr), where it gives <1 %. Extrapolating `σ=N` to the
multi-year melt timescale is **not physical**: it would give `ρ>1` at temperate ice +
high `N` (e.g. `ρ≈230` at `N=1 MPa` temperate) — "creep dominating" — because `N` is
not the deviatoric stress that relaxes a small corrugation. The relief stress
`ρ_i g a` is the right driver, and under it the null is robust over **all** timescales.
This is exactly why §D.6 (correctly) restricted the `σ=N` displacement bound to the
solver run window. **Status: [VERIFIED — quantified NULL; residual retired].**
