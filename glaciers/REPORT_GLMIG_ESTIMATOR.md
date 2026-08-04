# RESULT 16 — §H.1.2 grounding-line-migration estimator, calibrated on synthetic ground truth

**Status:** the §H.1.2 **level-set advance law** `v_front = |dH/dt|/|∇m|` and the
**residence-number discriminant** `Ro = v_kin/v_obs` (regressed against a `u_*`
proxy) are **[VERIFIED — synthetic plant-and-recover]** as *unbiased estimators*.
This is a methods/calibration result: it supplies the synthetic null and
exponent-recovery check that the **real-data** verdicts of §H.1.3 / §H.1.4 are
read against — it does **not** itself make a physical pacing claim. Harness
`validation/synthetic/glmig_synthetic.py`, tests in
`tests/test_validation_synthetic.py` (5/5). No external data; no GPU.

## What was open

§H.1.3 / §H.1.4 ran the §H.1.2 protocol on real open data and reported residence-
number slopes against a `u_*` proxy:

| region | `Ro` vs `u_*` (ITS_LIVE) slope | reading |
|---|---|---|
| continental (`τ_d` proxy) | `≈ −0.06` (flat) | thinning-paced |
| Amundsen | `+0.57` | `u_*`-paced |
| Bellingshausen/WAIS-Pacific | `+0.49` | `u_*`-paced |
| East Antarctica | `−0.30` | anti / different regime |

Those readings hinge on two **unvalidated estimator assumptions**: (i) that the
level-set law `v=|dH/dt|/|∇m|` is the correct kinematic, and (ii) that the *sign
and size* of the `log Ro` vs `log u_*` slope faithfully reflect a hydraulic
discount `D(u_*)` rather than being a regression/noise artefact. Neither had a
synthetic null — exactly the §V "plant a known structure, check recovery"
discipline the rest of the repo applies. This result fills that gap.

## What this result establishes

### A. The level-set advance law is exact
For an analytic margin with constant `|∇m| = b` thinning at rate `r`, the zero
contour is `x_c(t) = r t / b`; numerically tracking the contour gives speed
`5.000000` vs the law `r/b = 5.000000` (**rel-err `5×10⁻¹⁶`**). For a tilted 2-D
plane `m = b_x x + b_y y − r t`, the contour normal speed uses the gradient
*magnitude*: `r/√(b_x²+b_y²) = 3.000000`, recovered to `4×10⁻¹⁴`. The grid
estimator `A = 1/|∇m|` recovers `1/b` exactly. So the kinematic §H.1.3 relied on
is not an approximation — it is the level-set identity `v_n = −m_t/|∇m|`.

### B. The thinning-paced null is flat by construction
With no hydraulic discount (`γ=0 ⇒ D≡1`), `Ro ≡ 1` to machine precision and the
`log Ro` vs `log u_*` slope is `0.0` exactly. So the §H.1.3 continental "flat
slope (`r≈−0.06`) ⇒ thinning-paced" reading is reading against the **correct
null**: a genuinely thinning-paced front *does* produce a zero slope here.

### C. A planted discount exponent is recovered, and the slope is sign-faithful
Planting `D(u_*) = 1 + γ u_*^p` (so `Ro = D` exactly):

- **Identifiability:** fitting `log(Ro−1)` vs `log u_*` recovers the planted
  `p=0.5` to **rel-err `7×10⁻¹⁶`**.
- **Sign + monotonicity:** the raw `log Ro` vs `log u_*` slope — the statistic
  §H.1.4 actually reports — is **positive** and **increases monotonically** with
  the planted `p` (`0.111, 0.223, 0.404` for `p = 0.25, 0.5, 0.9`). A positive
  observed slope therefore genuinely means `u_*`-paced, and a larger slope means
  a steeper discount. (The raw slope is *compressed* relative to `p` because
  `Ro = 1 + γu^p` is affine-in-`u^p`, not a pure power law — so the real-data
  `+0.5`-ish slopes are consistent with discount exponents of order `p≈1`.)

### D. The slope is unbiased under noise, and null under permutation
Adding 25 % multiplicative noise to `v_obs` leaves the slope at `0.221` — positive
and within `0.002` of the noiseless `0.223` (small bias). Shuffling `u_*` against
the data collapses the slope to `−4×10⁻³` (≈0). So an observed positive slope is
**not** a noise or ordering artefact: the discriminant has a calibrated zero and
a small, sign-preserving bias.

## Honest scope

This validates the **estimator and the discriminant**, not any physical pacing
claim. The physics question — whether the `u_*` √-law actually paces the
grounding line — is settled *against real data* in §H.1.3 (disfavoured
continentally) and §H.1.4 (supported in marine West Antarctica). What was missing,
and is supplied here, is the synthetic guarantee that those real-data slopes are
read against the right null with an unbiased, sign-faithful estimator — the same
plant-and-recover guarantee `rtn_synthetic.py` gives the RTN classifier and
`sliding_synthetic.py` gives the lag estimator.
