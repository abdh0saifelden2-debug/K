# `validation/` — external-validation pipeline (§V)

Turns the §G **[HYP]** predictions into *falsifiable, reproducible* forecasts.
Three of the five §G hypotheses are field/observational claims that the solver
cannot test on its own; this module implements the equations + scoring harnesses
so they can be exercised on **synthetic** inputs now and on **real** public
datasets once those are provided locally.

```
validation/
├── synthetic/      # §V.3 unit tests: plant known parameters, check recovery (no network)
│   ├── rtn_synthetic.py        # plant H* threshold -> recover RTN>1 region + scoring
│   ├── rtn_phi_synthetic.py    # calibrate §H.1.1 area->φ inversion + MISI-margin band (1-φ) scaling
│   ├── sliding_synthetic.py    # plant forcing->response lag -> recover tau via x-corr (+ kernel-shape genericity)
│   ├── cmn_synthetic.py        # verify clock-mismatch commutator identity numerically
│   ├── glmig_synthetic.py      # calibrate §H.1.2 level-set v_front + Ro discriminant (plant-and-recover)
│   ├── rtn_intrusion_clock_synthetic.py  # harden §H.1.2 *driver* A=1/|∇m|: convergence order, isotropy, RTN-line identity
│   └── hydraulic_mz_projection_synthetic.py  # §G.4 project out the channel -> the lag kernel is the exact Mori-Zwanzig memory
├── validators/     # §V.1/§V.2 core equations + scoring (no external-data dependency)
│   ├── rtn_validator.py        # §G.3 RTN = (p_ocean-p_atm)/p_w + precision/recall
│   └── sliding_validator.py    # §G.4 lag-detection (estimator kernel-shape-generic [VERIFIED]; physical kernel [HYP])
├── external/       # real-data loaders — honest stubs (VM has no NSIDC/Earthdata access)
│   ├── bedmachine_loader.py    # ice thickness  (NSIDC-0756 / IDBMG4)
│   ├── its_live_loader.py      # surface velocity (ITS_LIVE)
│   ├── tide_loader.py          # ocean pressure  (CATS2008 / pyTMD)
│   └── lake_catalog_loader.py  # drainage events + GPS u_b(t)
└── reports/        # output figures / CSVs (gitignored except .gitkeep)
```

## What is verified now (synthetic, in CI/pytest)

| Test | Plants | Recovers | Status |
|------|--------|----------|--------|
| `rtn_synthetic`     | thickness threshold `H*` | `RTN>1` region == `H<H*` (0-cell mismatch); F1==1 on a complete survey; recall==1 on a sparse survey | **passes** |
| `sliding_synthetic` | forcing→response lag `τ` | `τ` via cross-correlation (within tol); memoryless control → ~0 lag | **passes** |
| `cmn_synthetic`     | time-dependent `K_u(t)` | commutator identity `[∂t,D]θ = ∇·((∂tK)∇θ)` (rel-err ~1e-7); term→0 for steady `K`; linear in `∂tK` | **passes** |
| `glmig_synthetic`   | level-set margin `m(x,t)` + residence ratio `Ro` | §H.1.2 front speed `v_n = r/|∇m|` (incl. 2-D tilt); `Ro` discriminant recovers planted migrating/standing class | **passes** |
| `rtn_phi_synthetic` | porosity `φ` via `RTN>1` area on a planted population | §H.1.1 area→φ inverse (unique, unbiased) + MISI-margin band width `(1−φ)/φ` exact | **passes** |
| `rtn_intrusion_clock_synthetic` | analytic margin geometries through the **real driver** `amplification`/`margin_field` | §H.1.2 `A=1/\|∇m\|` exact (planar), 2nd-order convergent (curved), isotropic (radial ring), advect recovers planted front + is the §H.1.1 RTN=1 line | **passes** |
| `hydraulic_mz_projection_synthetic` | the §G.4 coupled cavity↔channel `M` (same model as `hydraulic_kernel_synthetic`) | projection is exact (`5e-16`); kernel `K(τ)=M_sq M_qs e^{M_qq τ}` IS the channel Green's fn; reduced GLE reproduces `s(t)`; memory makes the peak (Markovian closure is monotone); Markovian limit = `(∫K)δ(τ)` | **passes** |

These validate the **math against controlled inputs**. They do *not* claim the
physics is correct against reality — that is §V.1/§V.2 on real data.

## Real data (§V.1 / §V.2) — run on open datasets

A subset of §V.1/§V.2 **runs on real, openly reachable data** (no Earthdata
login): BAS **Bedmap2** ice thickness (RTN geometry), **Siegfried & Fricker
(2018)** lake outlines, and **ITS_LIVE** surface velocity (anon S3). See
[`REAL_DATA_RESULTS.md`](REAL_DATA_RESULTS.md) for the verified numbers,
provenance and caveats, and `FUTURE_WORK.md §H` for the falsifiable-forecast
framing. Reproduce:

```bash
python validation/external/run_rtn_bedmap2.py  --stride 3 --phi 0.9   # §V.1 / §H.1
python validation/external/run_sliding_real.py --with-velocity Mac1   # §V.2 / §H.2
# §V.2d matched lake-drainage -> ITS_LIVE velocity lag (the §G.4 fast-outlet route)
python validation/external/lake_lag_itslive_match.py                  # 2003-07 USAP-DC catalogue (coverage-limited null)
python validation/external/lake_lag_atl15_itslive.py                  # MODERN: ICESat-2 ATL15 dates + dense ITS_LIVE (Earthdata)
# §H.1.4-ocean grounding-line residence Ro vs observed ocean thermal forcing
python validation/external/rtn_glmig_ocean.py                         # Ro / u_* / thinning vs Schmidtko-2014 shelf TF, per sector
# §H.1.6 effective-pressure gating: is the ocean→u_* coupling steepest near flotation?
python validation/external/rtn_ocean_efp_gate.py                      # TF-speed sensitivity vs normalized margin rel=m/H
```

Headline results: `RTN>1` concentrates at the grounding line (median 6 km vs
221 km from the GL; robust in `φ`) — a [VERIFIED] *directional* test; the literal
§G.4 lag timescale `H²/κ ≈ 1.5×10⁵ yr` on 131 real lakes is ~8×10⁴× longer than
observed surge lags (0.02–2 yr) — [FALSIFIED] as written.

The **auth-gated** sources (BedMachine v3, CATS2008, vetted lake drainage-date
catalogue) remain unreachable here; those `external/` loaders still raise
`DataUnavailableError` with copy-pasteable provisioning hints instead of
fabricating data. Once dropped locally (see each loader's docstring), the
validators run unchanged:

```python
from validation.external.bedmachine_loader import load_thickness
from validation.external.tide_loader import ocean_pressure_from_draft
from validation.validators.rtn_validator import rtn, classify, precision_recall
H = load_thickness("/data/BedMachineAntarctica-v3.nc")["thickness"]
# p_ocean from CATS2008 draft+tide; N_eff from a hydrology estimate
mask = classify(rtn(H, p_ocean, N_eff))
scores = precision_recall(mask, observed_intrusion_mask)
```

## Honest caveats (carried from §G)

- **RTN** has a gauge mismatch (`p_atm` in the numerator only) and rests on the
  *empirical* §G.1 result — it is a **[HYP]** diagnostic, not a proven bound.
  With sparse point observations, **recall** is the interpretable metric;
  precision collapses to survey sparsity.
- **Sliding law**: the literal §G.4 kernel is **not dimensionally closed** and
  the `H²/κ` lag is ~10⁴ yr (too long). Only the *empirical* lag `τ_lag` is
  interpreted; the kernel shape is a generic placeholder.
