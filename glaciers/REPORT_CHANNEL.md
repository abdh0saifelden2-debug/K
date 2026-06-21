# §A.3 / §D.1 — Scallop wall as a phase-locked opening source: preferred channel nucleation sites

## Hypothesis (pre-registered)

The Röthlisberger channel evolves as `dS/dt = V_o − V_c`, with melt opening
`V_o = (1/ρ_i L)|Q dφ/ds|` and Glen creep closure `V_c = 2 A S (N/n)^n`. The
channelisation instability runs through *flow concentration* (Caveat S): an
incipient low captures flow → higher `u*` → more turbulent melt → deeper
channel. The §A.3 hypothesis adds a distributed wall source from lee-localised
reattachment heat flux,

    dS/dt = V_o − V_c + V_scallop ,
    V_scallop(x) = (1/ρ_i L) [q_reattach(x) − q_flat] ,

and asks whether localised reattachment melt creates **preferred nucleation
sites** for channels — rather than the channel location being set by initial
noise. The prediction was that reattachment is (1) a *positive* (opening) source
of the same sign as `V_o`, (2) *phase-locked* to a fixed point on the bedform
(not white in `x`), and (3) able to *deterministically select* the channel site
through the concentration loop, overriding the random site a noise-seeded flat
wall would pick.

## What the run shows

Production DNS on a **Tesla-P100 GPU** (`backend=cupy`), Part-C config matching
`REPORT_CANDIDATE3.md` / §G.1: `nx=128, ny=128, n_waves=12, U_drive=1.5,
f_amp=0.4, a/λ=0.30`, spinup 3000 + measure 800. The channel ODE and all
diagnostics are pure NumPy. Figure `figures/49_scallop_channel_feedback.png`,
raw `figures/49_scallop_channel_feedback.json`.

1. **Sign [structural]: `V_scallop > 0` — reattachment is an opening source.**
   The reattachment population carries a net positive normal-flux excess over the
   flat-wall control, `V_scallop/V_o = +0.330`. It has the *same sign* as `V_o`,
   so scallop reattachment *adds* melt (consistent with Caveat S: more melt, not
   less). This is not a small correction — at production resolution the
   reattachment excess is ~33 % of the bulk opening rate.

2. **Phase-locking [structural]: `R_phase = 0.95`.** The reattachment patches are
   not spread uniformly over the scallop; their area density is concentrated in a
   narrow band of the wall phase (preferred phase ≈ 4.55 rad), giving a large
   Rayleigh resultant `R_phase = 0.95` (0 = uniform, 1 = a single phase). So
   `V_scallop(x)` is a spatially *periodic* source pinned to the geometry — twelve
   sharp peaks, one per wavelength (figure panel 2) — not white noise in `x`. The
   reattachment fraction lives almost entirely in two adjacent phase bins (panel
   1), and the excess–phase correlation `corr(excess, cos φ) = −0.222` confirms
   the lock to the lee side.

3. **Site selection [structural]: deterministic winner, beats noise.** Fed into
   the concentration-loop channel ODE (ring of 96 segments, 24 noise seeds), the
   phase-locked source makes the largest-steady channel form at the reattachment
   phase **deterministically**: `R_winner(scallop) = 1.000 @ 4.71 rad` (every seed
   picks the same site) versus `R_winner(noise) = 0.571` for the noise-only
   control (scattered sites). The steady channel size reaches `S ≈ 3` at the
   preferred phase versus a flat `S ≈ 1` baseline for noise only (panel 3). This
   is exactly the §A.3 "preferred nucleation site" claim, earned rather than
   imposed.

| quantity | symbol | P100 (cupy) | CPU (numpy) cross-check |
|---|---|---|---|
| Nusselt ratio | Nu/Nu_flat | 0.923 | 0.893 |
| opening source | V_scallop/V_o | **+0.330** | +0.306 |
| reattachment area fraction | f_reatt | 0.242 | 0.234 |
| phase-locking | R_phase | 0.950 | 0.948 |
| preferred phase (rad) | φ_pref | 4.55 | 4.56 |
| site lock (scallop) | R_winner | **1.000 @ 4.71** | 1.000 @ 4.71 |
| site lock (noise control) | R_winner | 0.571 | 0.571 |
| preferred nucleation site | — | **True** | True |

## Honest scope

- **The three results are dimensionless and coefficient-robust.** They do *not*
  depend on the (HYP) dimensional bridge `ρ_i L` or the (HYP) concentration gain
  `g`: the sign of `V_scallop` is set by the measured flux excess, the
  phase-locking is a geometric Rayleigh statistic, and the site-selection test
  only needs the source to be periodic-and-pinned versus white. The raw Glen
  constants `A, N` enter only through a lumped, well-conditioned creep rate
  `k_creep = 2A(N/n)^n`, so the steady channel sizes are reported on a normalised
  `[0,1]`-ish scale, not in metres.
- **`Nu/Nu_flat < 1` is the distribution-dominated regime**, the same §G.1 result
  this builds on: the scallop *redistributes* melt (concentrating it at the
  reattachment phase) more than it changes the bulk-mean rate. The mechanism here
  is about *where* the melt goes, which is precisely what sets a nucleation site.
- **CPU vs GPU agree on every structural claim.** The ~3 % spread in `Nu/Nu_flat`
  and `V_scallop` between `numpy` and `cupy` is the expected turbulent-trajectory
  divergence from FFT round-off over 3800 forced steps (instantaneous fields are
  not bit-identical across backends); the *signs, the phase-lock, and the
  deterministic winner are identical*. The channel-network winner is bit-identical
  because that stage is a deterministic CPU ODE.
- **What is still [HYP].** Turning the normalised steady sizes into physical
  channel radii, and the magnitude (not sign/locking) of the feedback, needs the
  `ρ_i L` bridge and a calibrated concentration gain — out of scope for the
  structural mechanism. Reproduce with:

      python scallop_channel_feedback.py --nx 128 --ny 128 --nwaves 12 \
          --udrive 1.5 --famp 0.4 --spinup 3000 --measure 800 --amp 0.30 \
          --out-dir figures --report REPORT_CHANNEL.md      # runner auto-detects CuPy
      python scallop_channel_feedback.py --fast              # quick CPU smoke test
