# NR61 — Imaginary coherency is the pairwise entropy-production spectral density

**Claim.** NR60's Gaussian entropy-production density is completely general
for any bivariate record, and its canonical application is neuroscience, not
oceanography: the two "channels" are two EEG/MEG sensors, and the
antisymmetric quad-spectrum is **Nolte's (2004) imaginary coherency**. For
two jointly-stationary Gaussian channels with 2×2 cross-spectral matrix S,

```
tr[S^{-T}S − I] = 4 (Im S12)² / det S = 4 ImCoh² / (1 − |Coh|²)
⇒  dσ_ij/dω = (1/π) ImCoh(ω)² / (1 − |Coh(ω)|²)
```

So imaginary coherency, the standard volume-conduction rejector, *is* the
pairwise irreversibility spectral density. Nolte's heuristic ("real
interaction has phase lag; volume conduction is instantaneous") is the second
law: instantaneous = reversible, lagged = irreversible.

## Theorems — each a known neuroscience heuristic made exact (all verified)

1. **Irreversibility needs two channels (Qian 2001).** A 1-D stationary
   Gaussian process has a real scalar spectrum → `tr[S^{-T}S−I]=0`
   identically (machine zero over random auto-spectra). Irreversibility is
   *relational* — it lives in the between-channel quad-spectrum, never in any
   single channel's power. This is the thermodynamic reason directionality
   lives in connectivity, not power spectra (Qian's 1-D vs n>1
   contradistinction; Weiss 1975).
2. **Volume conduction is reversible; only phase lag dissipates.** Any real
   instantaneous mixing of independent sources, `S = M diag M^T` with M real,
   gives a real symmetric cross-spectrum → ImCoh=0 → σ=0, for *any* mixing
   strength (verified to <1e−10 across mixing angles). A lagged copy gives
   Im S12≠0 → σ>0. Nolte's two assumptions are exactly "instantaneous =
   reversible, lagged = irreversible."
3. **The all-pass (transport) clock is the maximal irreversibility.** At
   fixed coherence r, `dσ/dω = (1/π) r² sin²φ/(1−r²)` in the coherency phase
   φ: zero at φ=0 (instantaneous / volume conduction, NR39) and *maximal* at
   φ=π/2 (pure quadrature = a pure inter-channel delay = NR48's all-pass
   transport clock). The transport clock carries the irreversibility; the
   elliptic clock (NR39) is the reversible null. Verified: phase sweep peaks
   exactly at π/2, zero at 0 and π.

Synthetic backbone (all exact): 2×2 closed form = general trace = NR60
Stokes form to 1e−12; a pure delay is irreversible with ImCoh sign giving
direction and σ magnitude delay-sign-invariant; unidirectional VAR(1)
coupling irreversible while the uncoupled diagonal is reversible; the
reciprocal-symmetric OU reversible (imported NR60 current formula = 0) vs the
antisymmetric OU irreversible.

## Real EEG (committed NR39 cache: OpenNeuro ds003775, 64-ch resting, alpha 8–13 Hz, 2016 pairs, 200 phase-randomized surrogates)

* **The most coherent pairs are the least irreversible.** The top-decile
  |ReCoh| pairs (maximal instantaneous mixing = volume conduction) have the
  *highest* coherence (median |Coh| 0.86 vs 0.54 overall) yet a *below-median*
  alpha EPR density (0.0044 vs 0.0068) and only 33% clear their own
  phase-randomized reversibility null vs 63% overall. High coherence + low
  irreversibility = reversible volume conduction — exactly Nolte's target.
* **Irreversibility rises with electrode distance; volume conduction falls.**
  Spearman vs 3-D sensor distance: EPR +0.10 and ImCoh +0.15 rise, |ReCoh|
  falls −0.15. A volume-conduction artifact must decay with distance; the
  pairwise EPR does the opposite — genuine lagged interaction, not field
  spread. (NR39's mixing lemma recovered: ρ(|ReCoh|,|ImCoh|)=−0.25.)
* **The alpha network is majority-irreversible.** 63% of pairs exceed their
  reversibility surrogate; a conservative reversible-null share of the summed
  alpha EPR density is 15%, i.e. ~85% of the measured alpha-band connectivity
  irreversibility sits above the time-symmetric floor.

## Placement

NR39 (imaginary coherence = projection invariance), NR48 (transport delay =
all-pass), NR57/NR60 (spin/rotary = irreversibility) are one statement:
imaginary coherency IS pairwise irreversibility, volume conduction is its
reversible (elliptic) null, and a pure delay is its extremal (all-pass)
source. NR60 found the object in oceanography (rotary spectra); NR61 shows
the identical object is the everyday connectivity tool of neuroscience, and
adds the exact thermodynamic meaning it was missing. Mainstream anchors:
Nolte et al. 2004; Qian 2001; Weiss 1975; Lebowitz–Spohn.

**Artifacts:** `figures/96_pairwise_epr.json` (all 12 verdicts true);
13 unit proofs in `tests/test_pairwise_epr.py`; offline-safe (committed NR39
cache). Reproduce: `python general_two_clocks/new_relationships38.py`.
