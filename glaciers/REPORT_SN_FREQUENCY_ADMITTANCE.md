# REPORT — frequency-resolved s_N: the bed as a hydromechanical low-pass filter

**Module:** `glaciers/validation/synthetic/sn_frequency_admittance.py`
**Outputs:** `glaciers/validation/reports/sn_frequency_admittance.{json,png}`
**Test:** `glaciers/tests/test_sn_frequency_admittance.py` (6 passed)
**Compute:** analytic + light numerics; **no GPU, no data download.** Reuses the committed `sn_master_curve.s_N_closed` / `restoring_rate`, so it is consistent with the existing master curve by construction.

## The gap this closes

The repo has three **independent** field probes of the regularized-Coulomb (RC) sliding sensitivity `s_N = d ln u_b/d ln N`, and treats all three as quasi-static measurements of the *same* `s_N(N)`:

| probe | mechanism | forcing frequency |
|---|---|---|
| lake-drainage step (`lake_lag_trunk`) | impulse `N`-drop | ~1/yr |
| ocean thermal-forcing drift (`efp_gate_direct_n`) | secular `N`-decline | decadal+ (≈ quasi-static) |
| tidal modulation (`tidal_admittance_probe`) | periodic `N`-swing | 1/(12 h) – 1/(14 d) |

But `sn_master_curve.py` Result 3 shows the **same** `N_c` flotation fold that makes `s_N0` diverge also makes the velocity-restoring rate vanish — `λ(N) ∝ (1−R)²/R → 0` at `N_c` (critical slowing down). So a probe that forces `N` at angular frequency `ω` does **not** read `s_N`; it reads the linear-response admittance of that first-order relaxation:

> **`A(ω, N) = s_N0(N) / (1 + i ω/λ(N))`** — a single-pole **low-pass** with corner `λ(N)`.

`(ADM)` is the exact linear response of the very OU model `sn_master_curve.ews_realization` already integrates. Results are stated in the **calibration-free** ratio `ω/λ(N)`; the absolute corner `λ_ref` is the repo's normalization, and the spectroscopy below is what *measures* it.

## Result 1 — the three probes diverge near flotation [VERIFIED]

As `N → N_c`: `s_N0 ~ N_c/(N−N_c) → ∞` but `λ ~ (N−N_c)² → 0`, so for **any fixed `ω>0`** the gain `|A| → s_N0·λ/ω ~ (N−N_c)/ω → 0`. The **quasi-static** (`ω→0`) probe **diverges** at `N_c`; every **periodic** probe's gain **vanishes**. They coincide only well-grounded. The quasi-static limit recovers the master curve to **1.5×10⁻¹⁶**; the M2-tidal gain near flotation is **8×10⁻⁹ × s_N0**.

This qualifies `tidal_admittance_probe.py` Result 1 ("`A1=|s_N|` steepens toward flotation") — true only while `ω ≪ λ(N)`. **No tension with observations:** the observed tidal *modulation* is `ε(x)·|A(ω,N)|`, and the forcing `ε` grows toward the grounding line (`REAL_DATA_RESULTS §I.5`: `Δp/p_i` 0.21→0.45 %), so raw tidal amplitude can still grow toward flotation (Gudmundsson 2011; Minchew 2017). The admittance rolloff is read in the **frequency dependence** — constituent ratios and phase — not raw amplitude.

## Result 2 — the corner sweeps down the forcing band (ordered ungrounding precursor) [VERIFIED]

`N_corner(ω)` (where `λ(N)=ω`) increases monotonically with `ω` (repo's `λ` normalization):

| probe | secular | decadal | lake-step | MSf | M2 |
|---|---|---|---|---|---|
| `N_corner` [MPa] | 0.25 | 0.43 | 0.92 | 2.69 | 8.21 |
| (× `N_c`) | 4.2 | 7.1 | 15.4 | 44.7 | 136.8 |

So as a site drifts toward `N_c` it crosses the **high-frequency corners first**: M2/S2 roll off (still far from flotation), then fortnightly MSf, then decadal, then secular. A site approaching ungrounding becomes **progressively more low-pass** — an ordered precursor in the *relative* admittance of tidal constituents. The ordering is calibration-free; only the absolute `N_corner` carries the `λ_ref` normalization.

## Result 3 — admittance spectroscopy: separate sensitivity from memory [DERIVED + verified on synthetic]

`1/|A|² = 1/s_N0² + (1/(s_N0²λ²))·ω²` is **linear in `ω²`**. Measuring `|A|` at frequencies that **straddle** the corner (some `ω<λ`, some `>`) recovers `s_N0` (intercept) and `λ` (slope) **separately** — disentangling the **sliding sensitivity** (→ proximity to `N_c`) from the **basal relaxation time** `τ_h = 1/λ` (the hydromechanical memory), which is confounded in any single-frequency probe. Plant→recover on a corner-straddling set:

| amplitude noise | `s_N0` (→ N) | `N` | `λ` (→ τ_h) |
|---|---|---|---|
| 3 % | 7.9 % | **3.6 %** | 10.7 % |
| 5 % | 12.3 % | **6.1 %** | 17.7 % |
| 10 % | 23.1 % | 11.5 % | 34.7 % |

**A new method of measurement:** pair a slow probe (ocean drift, `ω<λ`) with a fast one (tidal, `ω>λ`) at one site → read off both the bed's position on the master curve *and* its relaxation time. The phase `−arctan(ω/λ) → 90°` at `N_c` gives `λ` independently (73°→90° from well-grounded to flotation for MSf).

## Physical reality

The `N_c` fold makes the bed **simultaneously infinitely sensitive** (`s_N0→∞`) **and infinitely sluggish** (`λ→0`). Low-frequency probes see the sensitivity; high-frequency probes see the sluggishness; their product `s_N0·λ → 0` is a finite, *vanishing* diagnostic. This is the repo's two-clocks picture localized at the fold: the sliding clock `1/λ` diverges to swallow any forcing clock `1/ω`.

## Falsification

- If a site's admittance does **not** roll off as a low-pass in `ω` (e.g. tidal/decadal/secular gains all equal), `(ADM)` and the single relaxation are wrong.
- If multi-frequency admittance does **not** linearize `1/|A|²` in `ω²`, a single relaxation time is insufficient (a hydraulic second clock dominates — itself informative).
- If a site observed approaching flotation shows its **higher** constituents rolling off **after** its lower ones (reversed ordering), `λ(N)→0` at `N_c` is falsified.

## Honest scope

`(ADM)` is the exact linear response of the repo's RC-sliding OU caricature (same `s_N0`, `λ`). Real ice adds viscoelastic stress transmission and a **second, hydraulic clock** (the §G.4 Mori–Zwanzig channel kernel), so a real admittance is a **product** of low-passes, not one pole; the robust, calibration-free predictions are the corner-sweep ordering, the `1/|A|²`-vs-`ω²` linearity, and the phase→90°. The absolute `λ_ref` is uncertain — and Result 3 is precisely what measures it, turning the unobserved basal relaxation time into a field quantity.

## Reproduce

```bash
PYTHONPATH=glaciers/validation/synthetic python3 glaciers/validation/synthetic/sn_frequency_admittance.py
python3 -m pytest glaciers/tests/test_sn_frequency_admittance.py -q
```

## References

Gudmundsson (2011) *The Cryosphere* 5, 259; Rosier et al. (2014, 2015); Robel et al. (2017) *JGR*; Minchew et al. (2017) *ESSD* 9, 849; Schoof (2005); Joughin, Smith & Schoof (2019) *GRL* 46, 4764; Scheffer et al. (2009) *Nature* 461, 53; Dakos et al. (2008) *PNAS*.
