# Part 9e — The a-posteriori (time-integrated) closure test in 3-D

**One line.** The 2-D a-posteriori test (`closure_aposteriori.py`,
`REPORT`/paper1 §5b) found, and explained, an honest ceiling: *no* eddy-viscosity
closure beats no-model on the resolved 2-D spectrum, because the 2-D resolved-scale
energy budget is dominated by the **up-scale** cascade and needs near-zero net
subgrid dissipation. paper1 §5b predicted the reversal would come in 3-D, where the
cascade is **forward**. This run does that test on a Tesla P100 — and the ceiling
is broken: a closure beats no-model in 3-D.

Artefacts: `closure_aposteriori3d.py` (backend-agnostic; `--gpu` CuPy/P100),
`figures/77_closure_aposteriori3d.json` / `.png` / `.npz`. Kernel
`abduk5/closure3d-aposteriori`, Tesla P100-PCIE-16GB, cupy 14.0.1, 2218 s.

## Setup

A genuinely **predictive**, self-contained coarse LES (uses only resolved data,
never the DNS truth) is integrated forward in time under each closure and its
long-time resolved statistics are compared to the spectrally-filtered 3-D DNS truth
— the test paper1 §8.1 defers to.

- **Truth:** `ForcedNS3D` at **192³** (nu=3.5e-3, low-k solenoidal forcing), spun
  up 4000 steps, 30 snapshots, sharp-filtered at **k_c=24**.
- **LES:** velocity-form integrating-factor Heun at **n=72** (k_max=24=k_c), forced
  identically, 6000 steps, time-averaged over the developed half. The spectral eddy
  viscosity is folded **semi-implicitly** into the viscous integrating factor
  (unconditionally stable at the cutoff); nonlinear term, forcing and the FDT
  backscatter are explicit.
- **Closures:** `none` (molecular only), `smag` (Smagorinsky, positive-definite
  eddy viscosity), `specEV` (scale-selective spectral eddy viscosity, plateau+cusp),
  `specEV_bs` (+FDT-tied Leray backscatter), `cuspEV` / `cuspEV_bs` (cusp-only).
- **Metric:** resolved-spectrum L2 error vs filtered DNS over `[1, k_c]`, plus
  resolved KE and enstrophy ratios. "Beats no-model" = lower spectrum error than
  `none`.

## Result — the 2-D ceiling is broken in 3-D

| closure | KE/KE_dns | ens/ens_dns | specErr [1,kc] | beats no-model |
|---|---|---|---|---|
| none | 1.023 | 0.994 | **0.0733** | — (baseline) |
| **smag** | 0.979 | 0.875 | **0.0690** | **YES** (−5.9%) |
| specEV | 1.019 | 0.982 | 0.0724 | yes (−1.3%) |
| specEV_bs | 1.019 | 0.982 | 0.0724 | yes (−1.3%) |
| cuspEV | 1.023 | 0.993 | 0.0734 | no (≈none) |
| cuspEV_bs | 1.023 | 0.994 | 0.0734 | no (≈none) |

`any_closure_beats_none = true`. Every closure is stable.

**The sharpest signature is the Smagorinsky sign reversal.** In 2-D Smagorinsky was
the *worst* closure a-posteriori — it over-drained half the resolved energy and
**worsened** the spectrum relative to doing nothing (error 0.59 vs 0.46 no-model).
In 3-D the *same* positive eddy viscosity becomes the **best** closure: it removes
the cutoff energy pile-up the forward cascade now produces and lowers the resolved
spectrum error below no-model (0.069 vs 0.073). The optimal eddy viscosity has
flipped sign with the cascade — exactly the variational result
`nu_opt = <Pi> / (2 <|S|^2>)` reading `<Pi> ≈ 0` in 2-D (up-scale) vs `<Pi> > 0` in
3-D (forward; the n=128–192 a-priori runs measure a net forward mean flux).

## Honest scope

- **It is the *direction* that is decisive, not the size.** The forcing here is
  low-k (k_f=2.5) and k_c=24 captures most of the energy, so the subgrid load is
  mild and the margins are correspondingly small (~6% for Smagorinsky, ~1% for the
  structured eddy viscosity). A forcing band nearer the cutoff (the 2-D test used
  k_f=24, k_c=32) would widen the margins; that is a sharper follow-on, not a
  different conclusion.
- **The FDT backscatter adds no a-posteriori spectral gain here** (`specEV_bs` ≈
  `specEV`, `cuspEV_bs` ≈ `cuspEV`). This is consistent with the rest of the paper:
  the backscatter is an **a-priori / structural** repair (it fixes the transfer
  *sign* and the solenoidality at one instant, §5/§5c), not a resolved-spectrum
  a-posteriori effect at mild subgrid load.
- Still a-posteriori, periodic, forced isotropic 3-D; no wall-bounded a-posteriori
  (the §5d commutator result is a-priori). This **closes paper1 §8.1** in both
  regimes: 2-D (bounded — no a-posteriori win) and 3-D (reversal confirmed — a
  closure beats no-model).

## Forcing-band sweep (follow-up): the margin tracks the subgrid load, not proximity to the cutoff

An earlier draft of §5e speculated that *forcing nearer the cutoff would widen the
margin*. A controlled sweep at fixed `k_c=24`, `dns_n=192`, `n_les=72` (all else
identical to the headline run; `figures/77_ap3d_kfsweep.json`) **refutes that**:

| `k_f` | DNS `<KE>` | no-model err | best closure | margin | beats no-model? |
|------:|----------:|-------------:|:-------------|-------:|:---------------:|
| 2.5   | 0.0332    | 0.07335      | smag 0.06904 | +5.88% | **yes** |
| 10.0  | 0.0050    | 0.01083      | cuspEV_bs 0.01089 | −0.54% | no |
| 17.0  | 0.0018    | 0.00644      | cuspEV_bs 0.01221 | −89.61% | no |

`k_f=2.5` reproduces the committed headline to all digits. As `k_f → k_c` the cascade
range collapses, the DNS kinetic energy falls ~18×, the **subgrid load disappears**
(the resolved forced scales carry the energy), no-model becomes very accurate, and a
positive eddy viscosity only over-damps the resolved scales — so the closure advantage
**vanishes (k_f=10) then reverses (k_f=17)**. The closure-beats-no-model result is
therefore the **sign of the optimal eddy viscosity in the physically relevant
scale-separated regime**, not a universal a-posteriori superiority; widening the margin
needs *more* scale separation (lower ν / higher resolution), the opposite of the
original speculation. §5e text corrected accordingly.

## Seed-robustness ensemble (2026-07-03): the headline verdict is seed-fragile; the *structure* is robust

`closure_aposteriori3d.py` now exposes `--seed` (DNS uses `seed`, LES `seed+1`;
default 0 = the committed headline). Three fresh base seeds were run at the exact
headline config (192³ DNS, k_c=24, 72³ LES, ν=3.5e-3, k_f=2.5, 6000 LES steps) on a
Tesla P100 (kernel `abduk5/gpu-batch-202607` v4; ~35 min/seed; all rc=0;
`figures/77_closure_aposteriori3d_seed{11,23,37}.json`):

| seed | no-model err | smag err (vs none) | best closure | best margin | `any_closure_beats_none` |
|-----:|-------------:|-------------------:|:-------------|------------:|:---:|
| 0 (committed) | 0.0733 | **0.0690 (−5.9%)** | smag | −5.9% | true |
| 11 | 0.1439 | 0.1492 (**+3.7% worse**) | cuspEV_bs | −0.05% | true |
| 23 | 0.0853 | 0.0970 (**+13.7% worse**) | cuspEV | −0.02% | true |
| 37 | 0.0886 | 0.1101 (**+24.3% worse**) | cuspEV | −0.007% | true |

**Honest verdict.**

- **Robust across 4/4 seeds:** every closure is stable; Smagorinsky is the most
  dissipative (lowest KE and enstrophy ratios in every seed); the cusp spectral-EV
  closures track no-model to ≤0.05%; the FDT backscatter variants ≈ their base
  closures (no a-posteriori spectral effect at this load) — all structural claims
  unchanged.
- **NOT robust:** the seed-0 headline that *Smagorinsky beats no-model by −5.9%*.
  In 3/3 new seeds smag is *worse* than no-model (+3.7% to +24.3%), and the
  no-model baseline error itself scatters 0.073–0.144 across seeds (2×) — the
  seed-0 margin is comparable to the realization noise of the metric. The binary
  `any_closure_beats_none` flag stays true in all seeds but only via cusp closures
  at ≤0.05% margins, i.e. within noise: **the "2-D ceiling is broken in 3-D by a
  positive eddy viscosity" narrative is not established at this config** and should
  not be quoted without this caveat.
- **Interpretation.** With low-k forcing at mild subgrid load, the long-time
  resolved-spectrum error is dominated by realization variability of the forced
  large scales (30 DNS snapshots; single-realization LES average), not by closure
  physics. The k_f sweep above showed the margin tracks the subgrid load; this
  ensemble shows the sign at k_f=2.5 is not yet resolved out of the noise.
- **What would settle it:** ensemble-averaged spectra (≥8 seeds) and/or ≥3×
  snapshots per run at the same config; or a lower-ν / higher-resolution run where
  the subgrid load (and hence the expected margin) is larger than the realization
  scatter. Queued as follow-on GPU work.

## Ensemble extension (2026-07-04): 7 seeds — the sign question is SETTLED (null)

Three more seeds (51/67/83; same exact config; P100 kernel `abduk5/ap3d-ensemble-ext`,
~35 min/seed, all rc=0; `figures/77_closure_aposteriori3d_seed{51,67,83}.json`):

| seed | no-model err | smag margin | best closure (margin) |
|-----:|-------------:|------------:|:----------------------|
| 0    | 0.0733 | **−5.9%** | smag (−5.9%) |
| 11   | 0.1439 | +3.7% | cuspEV_bs (−0.05%) |
| 23   | 0.0853 | +13.7% | cuspEV (−0.02%) |
| 37   | 0.0886 | +24.3% | cuspEV (−0.007%) |
| 51   | 0.0474 | **−18.9%** | smag (−18.9%) |
| 67   | 0.1773 | **−2.6%** | smag (−2.6%) |
| 83   | 0.1001 | +7.1% | cuspEV (−0.014%) |

Paired statistics over n = 7 (log-ratio smag/none, same-seed pairing):
**mean +0.022 ± 0.052 (SE), t(6) = 0.42, two-sided p = 0.69; sign test 3/7, p = 1.0.**
The no-model error itself scatters 0.047–0.177 (~4×) across seeds while the smag
margin swings −19% … +24%.

**Resolved verdict.** At this configuration (192³→72³, k_c=24, k_f=2.5, ν=3.5e-3,
mild subgrid load) the a-posteriori spectral margin of Smagorinsky over no-model is
**statistically null** — the seed-0 headline was a realization-noise fluctuation, as
the 4-seed caveat suspected. What IS robust in 7/7 seeds: every closure stable;
Smagorinsky always the most dissipative (KE/enstrophy ratios lowest); cusp spectral-EV
closures track no-model to ≤0.05%; FDT backscatter variants ≈ base closures. Quote the
structural claims; do NOT quote a closure-superiority margin at this config. The
remaining open path to a *positive* margin is unchanged: higher subgrid load
(lower ν / higher k_f / higher resolution), where the k_f sweep already shows the
margin growing with load.
- **Paper impact.** paper1 §5e (frozen at submission) quotes the seed-0 result; the
  ensemble caveat is recorded in `working_notes/P1_POST_SUBMISSION_NOTES.md` for
  the referee-response / proof stage. No other paper quotes §5e.
