# GPU Test Plan — candidate4 CuPy diagnostics fix (PR #1)
 
## Environment
- Kaggle notebook, T4 GPU, `cupy 14.0.1` (confirmed), internet enabled (clone confirmed).
- Repo cloned to `/kaggle/working/K`. Branches present: `main`, `devin/1780580363-fix-flagged-issues`.
 
## What is being verified
Fix #1: `subglacial/candidate4_hydraulic_switch.py` used `np.asarray(cupy_array)` in
`umean_profile`, `active_layer_height`, `melt_flux`, `run_case`. In CuPy >= 10 this raises
`TypeError: Implicit conversion to a NumPy array is not allowed`. The fix routes device arrays
through a `_to_host()` helper (`xp.asnumpy` on CuPy, `np.asarray` on NumPy).
 
Trigger path: `run_case(cfg, spinup, measure, xp=cupy)` → measurement loop calls
`active_layer_height()` (→ `umean_profile()` → `np.asarray`) and `melt_flux()`.
 
Common config: `HydraulicConfig(nx=128, ny=64, A=4.0, f_amp=0.5, Ri=0.5)`, small `spinup`/`measure`.
 
## Tests
 
### Test 1 — Bug reproduction on `main` (GPU)
Run `run_case(cfg, spinup=50, measure=50, xp=cupy)` on `main` in a fresh subprocess.
- PASS criterion: raises `TypeError` mentioning implicit conversion to NumPy (proves the bug exists on the GPU path pre-fix).
- FAIL: completes without error (would mean bug not reproducible / test invalid).
 
### Test 2 — Fix on PR branch (GPU)
Same call on `devin/1780580363-fix-flagged-issues` in a fresh subprocess.
- PASS: completes with NO exception; prints finite `H1_mean`, `melt_mean`, `f_switch`, `umax` (all `isfinite`).
- FAIL: any exception, or non-finite outputs.
 
### Test 3 — CPU vs GPU numerical sanity (PR branch)
On PR branch, run the same cfg with `xp=np` and `xp=cupy` (same nx/ny/seed defaults).
- PASS: both finish; outputs same order of magnitude / qualitatively consistent (FFT round-off and
  GPU nondeterminism mean we do NOT require bit-identical; check `isfinite` and rough agreement).
- This is a sanity check, not a strict equality assertion.
 
## Evidence
- Screenshot Test 1 traceback (TypeError) and Test 2 success output.
- Record the Kaggle GPU run; annotate bug repro vs fixed.
- Post one summary comment on PR #1.
