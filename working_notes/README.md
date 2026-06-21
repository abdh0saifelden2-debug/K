# Working Notes — scratch probes and plans

Files that are explicitly scratch/exploratory and don't belong to a single
domain's verified record. Nothing here is deleted history — each file is kept
exactly as it was, and several fed results that *are* in the domain folders.

| File | What it is | Status |
|---|---|---|
| `_validate_sub.py` | Smoke check of the subglacial cavity solver (KE stationarity, melt flux) | scratch |
| `_probe_aposteriori.py` | "Does a coarse a-posteriori LES show K-theory killing the wake?" — early probe that motivated `glaciers/run_subglacial3d.py` | scratch |
| `tidal_probe.py` | Tidal melt phase-lag probe v3 in the incompressible cavity (marked "scratch, not committed" in its own docstring) — the incompressible test was null (instantaneous Leray projection ⇒ no pressure clock to beat) | scratch, null |
| `compress_probe.py` | Direction A: the same tidal phase-lag probe in the FINITE sound-speed compressible solver, where pressure has a real adjustment clock `τ_adj = L/c` | scratch |
| `stratification_probe.py` | Direction C: stratification-resonance test of the (since **retired**, §C of `../FUTURE_WORK.md`) regime equation — does the melt hump come from `τ_mem ≈ T_BV` resonance? | exploratory; regime equation retired |
| `TEST_PLAN_gpu_candidate4.md` | Pre-registered GPU test plan for Candidate 4 (hydraulic switch) — the run itself is recorded in `../glaciers/REPORT_CANDIDATE4.md` | plan note |

Note: `direction_c_gpu_probe.py` (the GPU twin of `stratification_probe.py`)
lives in `../general_two_clocks/` because `gle_coefficients.py` (RESULT 12)
imports its embedded cavity solver.

The Python probes import solvers from the domain folders; they have a built-in
path bootstrap, so they run from anywhere, e.g.
`python working_notes/tidal_probe.py`.
