<!-- Resolution-convergence companion to REPORT_CLOSURE3D.md (Part 9c).
     Generated from a Kaggle GPU sweep (Tesla P100-PCIE-16GB, cupy 14.0.1) of
     run_closure3d.py --gpu at n = 128, 160, 192, kc=24, nu=0.0012, f_amp=1.2, seed=42,
     steps=2500. Raw data: figures/closure3d_convergence_summary.json. -->

# Part 9c — robustness of the 3D closure verdicts (grid resolution + filter width)

`REPORT_CLOSURE3D.md` establishes the 3D two-clocks closure benchmark at a single
resolution (`128³`) and is explicit in its Scope that it is *"not a grid-converged
production DNS."* This companion closes that specific caveat for the **structural
diagnostics**: it re-runs the identical a-priori benchmark at `n = 128, 160, 192` with
**everything else held fixed** (forcing `f_amp=1.2`, viscosity `ν=1.2e-3`, sharp filter
`k_c=24`, seed `42`, `2500` steps), so the only variable is DNS resolution. The verdicts
do not move.

**Provenance.** GPU sweep on a **Tesla P100-PCIE-16GB** (cupy 14.0.1) via Kaggle — the
same hardware/stack class behind the committed headline (PAPER_OUTLINE.md). Wall-clock:
`131 s` (`128³`), `235 s` (`160³`), `394 s` (`192³`). Raw per-run JSON +
`convergence_summary.json` are in `figures/`.

## The numbers (fixed `f_amp, ν, k_c=24, seed`; only `n` varies)

| diagnostic | `128³` | `160³` | `192³` | committed (`128³`) |
|---|---|---|---|---|
| **backscatter vol-fraction, truth** (`Π<0`) | **0.4802** | **0.4808** | **0.4807** | 0.480 |
| **backscatter vol-fraction, Smagorinsky** | **0.000** | **0.000** | **0.000** | 0.000 |
| **transfer corr. with truth, projected-FDT** | **0.9997** | **0.9997** | **0.9997** | 1.000 |
| transfer corr. with truth, Smagorinsky | −0.641 | −0.639 | −0.637 | −0.641 |
| transfer corr. with truth, surrogate | 0.268 | −0.111 | −0.276 | 0.268 |
| **vortex-stretching production `⟨ωSω⟩*`** | **0.168** | **0.169** | **0.168** | 0.168 |
| strain–vorticity align, **intermediate `e₂`** | **0.654** | **0.649** | **0.647** | 0.654 |
| strain–vorticity align, extensional `e₁` | 0.479 | 0.482 | 0.484 | 0.479 |
| strain–vorticity align, compressive `e₃` | 0.325 | 0.325 | 0.330 | 0.325 |
| RMS div(m), truth | 1.0e-14 | 1.4e-14 | 1.7e-14 | 1.05e-14 |
| RMS div(m), projected-FDT | 8.9e-15 | 1.2e-14 | 1.4e-14 | 8.85e-15 |
| RMS div(m), surrogate | 9.85 | 10.21 | 9.76 | 9.85 |
| net transfer `Σ_k T_true` (extensive) | −4.5e8 | −1.9e9 | −4.7e9 | −4.5e8 |

![convergence](65_closure3d_convergence.png)

## What this shows

1. **Reproducibility.** The `128³` column reproduces the committed `REPORT_CLOSURE3D.md`
   headline to all reported digits (KE `0.0349`, `⟨ωSω⟩*` `0.168`, backscatter `0.480`,
   transfer corr projected-FDT `≈1.0`, Smagorinsky `−0.641`, surrogate divergence `9.85`)
   on a fresh P100 — the cupy result is deterministic and portable.

2. **The structural verdicts are resolution-invariant** from `128³` to `192³`:
   - **Backscatter volume fraction is `0.4802 ± 0.0004`** while **Smagorinsky stays exactly
     `0.000` at every resolution** — the keystone K-theory failure (a positive-definite
     eddy viscosity has `Π≥0` everywhere, so it *structurally* cannot represent the ~½ of
     space that transfers energy up-scale) is not a grid artifact.
   - **The projected-FDT transfer correlation is `0.9997` at every `n`**, vs Smagorinsky's
     stable anti-correlation `≈−0.64`.
   - **Intermediate-eigenvector alignment dominates** (`e₂≈0.65 > e₁≈0.48 > e₃≈0.33`) at
     every resolution — the Constantin–Fefferman geometric depletion is robust.
   - **Solenoidality** holds to machine precision (`~10⁻¹⁴`) for truth and projected-FDT at
     every `n`; the phase-randomized **surrogate fails identically** (`≈10`).

3. **What legitimately varies, and why it is not non-convergence.** The net transfer
   `Σ_k T_true` is an *extensive* quantity (it sums over an `n`-dependent number of modes),
   so its magnitude grows with `n` — but its **sign stays negative at every resolution**
   (3D is a net forward cascade, the claim of record). The single-snapshot KE shows mild
   spin-up variation (`0.035–0.039`); the *structural ratios* above do not.

## Filter-width robustness (fixed `192³`, vary `k_c`)

A second P100 sweep holds the resolution at `192³` and varies the **LES sharp-filter
cutoff** `k_c ∈ {16, 24, 32}` (everything else fixed) — an independent robustness axis.
Because the DNS truth field is identical across the three, the *field* properties are
filter-independent by construction (KE `0.0351`, `⟨ωSω⟩* = 0.168`, intermediate alignment
`e₂ = 0.647` at every `k_c`); the *filter-dependent* closure scores are:

| diagnostic | `k_c=16` | `k_c=24` | `k_c=32` |
|---|---|---|---|
| backscatter vol-fraction, truth | 0.465 | 0.481 | 0.489 |
| backscatter vol-fraction, Smagorinsky | **0.000** | **0.000** | **0.000** |
| transfer corr. with truth, projected-FDT | 0.9992 | 0.9997 | 0.9999 |
| transfer corr. with truth, Smagorinsky | −0.631 | −0.637 | −0.553 |
| RMS div(m), projected-FDT | 1.4e-14 | 1.4e-14 | 1.5e-14 |
| RMS div(m), surrogate | 6.5 | 9.8 | 13.1 |

![filter sweep](66_closure3d_filtersweep.png)

The verdicts hold at every cutoff: **Smagorinsky carries exactly zero backscatter** while
truth carries ~½, the **projected-FDT transfer correlation stays `≥0.999`**, and the
phase-randomized **surrogate breaks solenoidality** (`≈6–13`). The one honest
`k_c`-dependence is the truth backscatter fraction, which rises mildly `0.465 → 0.489` as
the filter widens (a wider resolved band admits more of the backscatter-rich near-cutoff
scales) — a *trend in the number*, not in the *verdict*: it is ~½ and Smagorinsky is zero
at every cutoff. Wall-clock on the P100: ~400 s per run.

## Scope (unchanged caveats)

This is still an **a-priori, frozen-field** test in a periodic box: it certifies that the
*structural* closure verdicts — Smagorinsky's zero backscatter, projected-FDT's transfer
fidelity, the surrogate's broken solenoidality, and the intermediate-eigenvector geometry
— are **independent of DNS resolution** (128³→192³) and **robust to the LES filter
width** (`k_c`=16→32) at fixed physics. It is **not** an
a-posteriori (time-integrated, closed-loop) LES, and it makes **no** claim of 3D
Navier–Stokes regularity or of operational superiority. It removes exactly one stated
limitation of `REPORT_CLOSURE3D.md` (single-resolution, single-filter), and nothing more.
