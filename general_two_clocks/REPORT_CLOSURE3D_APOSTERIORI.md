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
