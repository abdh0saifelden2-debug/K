# RESULT 17 — the §G.4 lag estimator is kernel-shape-generic

**Status:** the §G.4 / §V.2 lag-recovery estimator (`estimate_lag`,
cross-correlation peak) is **[VERIFIED — synthetic]** to recover the planted
forcing→response lag *independent of the memory-kernel shape*. This closes the
"keep the kernel shape generic" assumption *for the estimator*; it does **not**
close the physical kernel (§H.2 falsified the literal thermal kernel `H²/κ_ice`;
the true kernel is hydromechanical and its form stays `[HYP]`). Harness
`validation/synthetic/sliding_synthetic.py::run_kernel_shapes`, tests in
`tests/test_validation_synthetic.py` (4/4). No external data; no GPU.

## What was open

§G.4 / §H.2 deliberately **measure only the empirical lag** and "keep the kernel
shape generic and explicitly labelled `[HYP, kernel not closed]`"
(`validation/validators/sliding_validator.py`). That strategy is only sound if
the lag estimator returns the same answer **whatever the kernel shape** — else a
recovered lag would be an artefact of the assumed Gamma form rather than a
property of the data. The existing synthetic test
(`sliding_synthetic.run`) planted a **single** kernel (Gamma, `k=2`), so the
shape-genericity itself was untested.

## What this result establishes

`run_kernel_shapes` plants the **same** target lag `τ=40` samples with **five
markedly different causal kernels**, all normalised, all peaking at `τ` but
differing in skew, symmetry and tail:

| kernel | character | recovered lag (err vs τ) |
|---|---|---|
| Gamma `k=2` | right-skewed, heavy tail | 42 (2) |
| Gamma `k=4` | less skewed, peakier | 40 (0) |
| log-normal `σ=0.5` | heavy right tail | 38 (2) |
| bi-exponential (rise→decay) | finite rise, exp decay | 36 (4) |
| raised-cosine on `[0,2τ]` | **symmetric, zero skew** | 41 (1) |

- **Every shape recovers the planted lag** to within `≤ 4` samples (`≤ 10 %` of
  `τ`; tolerance `0.12τ+1 ≈ 5.8`). The worst case is the bi-exponential (the most
  asymmetric rise), exactly where the cross-correlation peak is most pulled by the
  tail — and even it lands inside tolerance.
- **All five kernels peak at the target** (`worst_mode_offset ≤ 2` samples), so
  the test compares like with like: identical mode, different shape.
- **Memoryless control stays null.** A delta kernel returns `lag = 0`, so the
  estimator does not manufacture a lag from shape alone.
- **Robust across lags.** Genericity holds at `τ = 25` and `τ = 60` too
  (parametrised test).

## Why this matters

§H.2 showed the *literal* §G.4 kernel (`τ_ice = H²/κ_ice`) is falsified on real
lake thickness (~8×10⁴× too slow) and relocated the memory into subglacial
**hydromechanics**, whose Green's function shape is un-closed `[HYP]`. The whole
§V.2 programme therefore rests on measuring the **empirical lag** while staying
agnostic about the kernel. This result certifies that agnosticism is safe: the
estimator reads the lag, not the shape. It is the §G.4 analogue of the
plant-and-recover guarantees `rtn_synthetic.py` gives the RTN classifier and
`glmig_synthetic.py` (RESULT 16) gives the migration discriminant.

## Honest scope

This validates the **estimator**, not the physics: it does not assert any
particular kernel is correct, nor that a real ice stream's lag is set by §G.4
rather than by the mainstream effective-pressure sliding law. It guarantees only
that, *given* a delayed response, the recovered lag is a faithful, shape-robust
readout — the precondition for the real-data §V.2 test once the gated
drainage-date catalogue (§H.2) becomes available.
