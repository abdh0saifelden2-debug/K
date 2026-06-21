# REPORT — tidal velocity admittance: a continuous third probe of `s_N(N)` and an ungrounding early-warning

**Module:** `glaciers/validation/synthetic/tidal_admittance_probe.py`
**Outputs:** `glaciers/validation/reports/tidal_admittance_probe.{json,png}`
**Test:** `glaciers/tests/test_tidal_admittance_probe.py` (4 passed)
**Compute:** analytic + a forced 1-D map; **no GPU, no download.**

This extends §I (the `s_N(N)` master curve) with a **third independent field probe**
of the regularized-Coulomb (RC) sliding sensitivity, alongside lake-drainage steps
(§G.4) and ocean thermal-forcing gating (§H.1.6).

## The idea

Ocean tides raise/lower sea level at the grounding zone, modulating the basal
effective pressure `N`, hence the basal speed through the *same* `s_N(N)`. So the
tidal velocity response measures `s_N` — continuously and at high cadence.

**Mainstream anchor.** Gudmundsson (2006, *Nature*) found Rutford Ice Stream's speed
varies ~20 % at the fortnightly MSf period; Gudmundsson (2007, 2011) showed this
needs a *non-linear* sliding law (the M2–S2 beat rectified to MSf; a linear law gives
no MSf), with an exponent ≈ 3; Minchew et al. (2017) found MSf amplitude *grows
toward the grounding line*. Other mechanisms (hydrology, GL migration, margin
widening) can also produce MSf — this report is the **sliding-law reading**.

## Results

**1. Tidal admittance = `|s_N|` (third probe).** The fundamental admittance
`A1 = |d ln u_b/d ln N|` equals `|s_N(N)|` in the small-amplitude limit (verified to
**< 3 %**) and so steepens toward flotation like the other two probes.

**2. Harmonic-generation fingerprint.** Because `s_N(N)` is curved, sinusoidal tidal
`N` forcing makes velocity **harmonics**:

> `A2/A1 ≈ (ε/4)·|s_N'/s_N − 1|`,  `s_N'/s_N = −mR/(1−R)`,  `R = (N_c/N)^m`,

which **diverges as `N → N_c`**. Verified against the analytic law to **< 8 %**; the
2f/1f ratio rises from **1.3 %** (well-grounded) to **10 %** (near flotation). This
is a sliding-law explanation for the observed toward-GL growth of the nonlinear MSf
signal.

**3. Tides-only inversion (new method of measurement).** The fundamental admittance
`|s_N|`, the 2f/1f ratio, and the *known* tidal amplitude `ε` over-determine `(m, R)`:

> `R = (4·(A2/A1)/ε − 1)/|s_N|`,  `m = |s_N|·(1 − R)`.

This recovers the Weertman exponent `m` (to **~0 %**) and the **dimensionless
flotation proximity** `R ∈ [0,1]` (`R → 1` at ungrounding) to **~3 %** — *from
surface velocity alone, with no basal-pressure measurement.* It directly answers
Joughin et al. (2019)'s "no reliable knowledge of basal water pressure": the tide
self-calibrates the bed's position on the `s_N` curve.

**4. Operational early-warning.** Continuous tidal monitoring tracks `R(t) → 1`
(rising admittance + rising harmonics, ×3 and ×7 over the swept approach) — turning
the §I.3 critical-slowing-down precursor into a *continuous* ungrounding early-warning.

## Field consistency (n=1, qualitative)

Rutford is a low-`N`, near-flotation site — it is *also* this repo's largest drainage
surge (`du/u = 21.7 %`, `lake_lag_trunk` `Rutford_1`). Its strong, nonlinear,
toward-GL MSf response and tidally-inferred exponent ≈ 3 (= our `m`) are consistent
with `s_N(N)` curvature steepening toward flotation — three probes pointing at the
same place.

## Real tidal forcing — CATS2008 × BedMachine (`tidal_forcing_gz.py`)

This probe is analytic; its one external input is the tidal forcing amplitude `ε`
(the fractional modulation of basal `N`). That input is **now measured**, not
assumed, by combining two independent real datasets — **CATS2008** tidal elevation
(via `pyTMD`) and **BedMachine v4** ice overburden — sampled at the *actual*
grounding zone (`glaciers/validation/external/tidal_forcing_gz.py`,
`REAL_DATA_RESULTS.md §I.5-real`):

* Tidal amplitude `η_amp` at grounded cells within 10 km of the GL: median **0.84 m**
  (verification: Filchner-Ronne range 2.70 m, Ross 1.65 m).
* Tidal ocean-pressure swing over ice overburden `Δp/p_i = ρ_w g η_amp/(ρ_i g H)`:
  median **0.30 %** — and, since `N ≤ p_i`, a hard **lower bound on `ε`**, so the
  fundamental admittance `A1 ≈ |s_N|·ε` rests on a *measured* floor.
* It **grows toward the grounding line** (median `Δp/p_i` 0.21 % → 0.45 % from
  8–10 km to 2–4 km), the same toward-GL growth of the nonlinear tidal response
  this report attributes to `s_N(N)` steepening toward flotation.

This converts the probe's forcing from an assumed scalar to a data-backed,
spatially-resolved field; the admittance-vs-`N` and 2f/1f-vs-`N` predictions remain
the falsifiable claims (below), now anchored to a real `ε(x)`.

## Falsification

* If decomposing observed GPS/InSAR tidal admittance + harmonics by `N` does **not**
  show the 2f/1f ratio rising toward flotation, the RC `s_N(N)` reading of the MSf
  nonlinearity is wrong (and hydrology / GL-migration / margin mechanisms dominate).
* If the tides-only inverted `R`/`m` disagree with independently mapped flotation
  proximity, the probe is invalid.

## Reproduce

```bash
PYTHONPATH=glaciers/validation/synthetic python3 glaciers/validation/synthetic/tidal_admittance_probe.py
python3 -m pytest glaciers/tests/test_tidal_admittance_probe.py -q

# real CATS2008 × BedMachine forcing (needs: pip install pyTMD dask pyproj; CATS2008
# from USAP-DC reCAPTCHA -> <dir>/CATS2008/; BedMachine v4 via earthaccess):
python3 glaciers/validation/external/tidal_forcing_gz.py \
  --bedmachine <BedMachineAntarctica_*_V04.1.nc> --tide-dir <dir> --stride 4 \
  --json glaciers/validation/reports/tidal_forcing_gz.json
python3 -m pytest glaciers/tests/test_tide_loader.py -q
```

## References

Gudmundsson (2006) *Nature* 444, 1063; Gudmundsson (2007) *JGR* 112, F04007;
Gudmundsson (2011) *The Cryosphere* 5, 259; Minchew et al. (2017) *ESSD* 9, 849;
Rosier et al. (2014, 2015); Robel et al. (2017); Joughin, Smith & Schoof (2019)
*GRL* 46, 4764; Schoof (2005); Weertman (1957).
