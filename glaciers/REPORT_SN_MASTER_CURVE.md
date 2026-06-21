# REPORT — the s_N(N) master curve, an inversion method, and a critical-slowing-down early-warning signal

**Module:** `glaciers/validation/synthetic/sn_master_curve.py`
**Outputs:** `glaciers/validation/reports/sn_master_curve.{json,png}`
**Test:** `glaciers/tests/test_sn_master_curve.py` (6 passed)
**Compute:** analytic + a 1-D stochastic ODE; **no GPU, no data download** (optionally
reads the committed `lake_lag_trunk.json` and `efp_gate_direct_n.json`).

This report extends the §G.4 / §H.1.6 / §F effective-pressure-sensitivity work
(PR #1–#4) from a **qualitative** statement (*both field probes' response grows
toward flotation, matching a regularized-Coulomb shape*) to three **quantitative,
falsifiable** results.

---

## Background (what was already established)

A subglacial lake drainage is a natural *step experiment* whose surface-speed surge
`du/u` measures the sliding-law sensitivity `s_N = d ln u_b / d ln N`; ocean thermal
forcing measures the same `s_N(N)` continuously. Both grow toward flotation
(`lake_lag_trunk.py`: amplitude semilog slope −3.5/rel; `efp_gate_direct_n.py`:
TF→speed slope +0.46/°C well-grounded → +0.62/°C near flotation). The proposed
closed form is the regularized-Coulomb (RC) law (Schoof 2005; Joughin, Smith &
Schoof 2019):

> `τ_b = C N (u_b/(u_b+u_0))^(1/m)`,  with a grounded steady state only for `N > N_c = τ_d/C`.

Joughin et al. (2019) note there is *"no reliable knowledge of basal water
pressure"*, so models **subsume `N` into a tuned friction coefficient** and apply an
**ad hoc** linear near-flotation weakening (a `h_af < h_T` ramp) with a **fixed**
`u_0 = 300 m/yr`. The three results below replace those ad hoc choices with derived,
measurable physics.

---

## Result 1 — the closed-form `s_N(N)` master curve  [DERIVED, VERIFIED]

Solving the RC law at fixed driving stress (`τ_b = τ_d`, so `u_b = u_0 R/(1−R)`,
`R = (N_c/N)^m`) and differentiating gives, **in closed form**,

> **`|s_N|(N) = m / ( 1 − (N_c/N)^m )`**,   `N_c = τ_d/C`.

* Far from flotation `|s_N| → m` (Weertman exponent; here `m=3`, `|s_N|→3.00`).
* Near flotation it is a **simple pole**: `|s_N| ≈ N_c/(N − N_c)`.

This is a *single two-parameter* curve `(m, N_c)`. It **derives** the near-flotation
weakening that Joughin (2019) imposes by hand. Verified against the repo's numeric
`type_iii_regime.s_N` to **max rel-diff 1.4×10⁻⁴**; the pole approximation is
leading-order correct to **9%** inside `1.1 N_c`.

**Convention caveat (`WELL_GROUNDED_NOTE`):** for the fixed-`u_0` form (repo /
Joughin practical choice) `|s_N|→m` far from flotation. For the cavitation form
`u_0 = N^m Λ_o`, instead `s_N = −mR/(1−R) → 0` far from flotation — but the `N_c`
pole is **convention-independent** (the Coulomb bound `C N < τ_d` below `N_c`
removes the grounded branch regardless of `u_0`).

## Result 2 — an inversion method: measure `N_c` from drainage steps  [DERIVED + VERIFIED on synthetic]

Because the master curve has only `(m, N_c)`, a *population* of drainage steps with
known `N` and measured `du/u = |s_N(N)| f` **over-determines the sliding law** — so
the flotation/Type-III threshold can be **measured, not tuned**. Plant→recover on
synthetic populations (`inversion_robustness`):

| amplitude noise | `N_c` recovery | `m` recovery |
|---|---|---|
| 5 % | **0.2 %** | 7 % |
| 10 % | **0.4 %** | 14 % |
| 20 % | **1.0 %** | 28 % |

**`N_c` (the MISI-relevant threshold) is robustly invertible to a few percent.**
`m` is only weakly constrained by amplitude alone — it is **degenerate with the
per-event fractional drop `f`** wherever `du/u ≈ f·m` (well-grounded). Pinning `m`
needs near-`N_c` sampling or a co-located hydrology `dN` (the project's stated next
step). An independent order-of-magnitude cross-check from the **ocean-gating**
curvature (`gating_Nc_consistency`: the +0.62/+0.46 TF-slope ratio) implies
`N_c ≈ 0.036 MPa` — same order as the RC default **0.06 MPa**.

## Result 3 — a critical-slowing-down early-warning signal for ungrounding  [DERIVED + demonstrated; field test HYP]

The pole has a *dynamical* consequence. Near flotation the basal drag saturates onto
its Coulomb plateau, so the velocity-restoring **stiffness**

> `∂τ_b/∂u ∝ (1−R)² / R  → 0`  as `N → N_c`,

i.e. the grounded steady branch ends in a **fold** and the restoring rate vanishes —
textbook **critical slowing down** (Scheffer et al. 2009; Dakos et al. 2008). An
Ornstein–Uhlenbeck velocity perturbation driven while `N` slowly declines shows
**rising variance and lag-1 autocorrelation** (Kendall-τ var = 0.54, AC1 = 0.55;
equilibrium variance grows ≈ **1900×** toward `N_c`). 

So an ice stream drifting toward ungrounding should show, *before* it goes afloat, a
rising variance and AC1 **in its surface speed** — an early-warning signal for marine
grounding-line retreat / MISI onset. This is **distinct** from the Greenland
surface-melt CSD of Boers & Rypdal (2021): different observable (**ice velocity**),
mechanism (**basal-sliding `s_N` divergence**), and threshold (**ungrounding**).

**Honest limit (shown, not hidden):** very close to `N_c` the restoring time
`1/λ` exceeds the time the system spends there under a fast sweep, so the *realized*
variance lags the equilibrium prediction — the EWS is clearest in the resolvable
`N ≳ 2 N_c` pre-tipping window.

---

## An open discriminator (flagged, not claimed): lag vs `N`

The field amplitude rises toward flotation (`corr(rel, ln du/u) = −0.78`), as the
master curve predicts. But the baseline cavity model predicts the discrete **lag**
*also rises* toward flotation, whereas the current **n = 3** detections trend the
opposite way (`corr(rel, ln lag) = +1.0`), driven by the marginal 2.86σ Rutford
point. So **the sign of lag-vs-`N` is unresolved** and is a clean discriminator for
the next batch of in-band detections — *not* a result claimed here.

## Falsification

* If a drainage-step population with co-located `dN` returns `|s_N(N)|` that does
  **not** follow `m/(1−(N_c/N)^m)` (e.g. no upturn toward flotation, or a different
  exponent), the RC master curve is wrong.
* If an ice stream observed to approach flotation shows **no** rise in speed
  variance / AC1 (with adequate sampling), the critical-slowing-down prediction is
  falsified.
* If the inverted `N_c` from drainage amplitudes is inconsistent with the
  independently mapped flotation `N_c`, the unification fails.

## Reproduce

```bash
PYTHONPATH=glaciers/validation/synthetic python3 glaciers/validation/synthetic/sn_master_curve.py
python3 -m pytest glaciers/tests/test_sn_master_curve.py -q
```

## References

Schoof (2005) *Proc. R. Soc. A*; Gagliardini et al. (2007) *JGR*; Tsai et al. (2015)
*J. Glaciol.*; Joughin, Smith & Schoof (2019) *GRL* 46, 4764; Tulaczyk et al. (2000)
*JGR*; Scheffer et al. (2009) *Nature* 461, 53; Dakos et al. (2008) *PNAS*; Boers &
Rypdal (2021) *PNAS* 118, e2024192118.
