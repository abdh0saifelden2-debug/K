# NR60 — Gonella's rotary coefficient is an entropy-production spectral density

**Claim.** For a stationary Gaussian process the entropy production rate
(KL rate between the process and its time reversal) is
`σ = (1/4π)∫ tr[S(ω)^{-T} S(ω) − I] dω`, and for a 2-D velocity record the
integrand **closes exactly**:

```
tr[S^{-T}S − I] = 4 (Im S_uv)² / det S            (any frame; exact)
              = 4 V² / (I² − Q² − U² − V²)        (spectral Stokes, NR56)
              = (S₊ − S₋)²/(S₊S₋) = 4 C_R²/(1 − C_R²)   (rotary; Q=U=0)
```

with `C_R` **Gonella's (1972) rotary coefficient** — the standard tool of
ocean current analysis. The frequency-resolved second law of the two-clocks
program: a band is second-order reversible iff its rotary spectra balance.
Fifty years of rotary-coefficient plots have been entropy-production spectral
densities up to the monotone map `C²/(1−C²)`.

## Theorems (all verified)

1. **Rotary asymmetry IS irreversibility** — the closed forms above, machine
   precision (≤1.4e−14) on random Hermitian spectral matrices; invariant
   under frame rotation, scaling, mirror, and time reversal (EPR is
   parity-even although V is parity-odd: V² enters).
2. **Linear polarization is reversible, circular is dissipative** — `V=0`
   kills the density regardless of Q,U anisotropy: any superposition of
   rectilinear (wave) oscillations has identically zero density (machine),
   while a single circularly-polarized mode is strictly positive. NR56's
   wave/vortex polarimetry is a reversible/irreversible decomposition
   frequency-by-frequency. (In this reading the purely-circular inertial
   peak of surface drifters — Elipot & Lumpkin 2008 — is the most
   irreversible band of the surface ocean.)
3. **The OU rotator closes on NR57 exactly** — for `du = −νu + f εu + √2 dW`
   the rotary spectra are Lorentzians at ±f and the integral evaluates (by
   residues) to `σ = 2f²/ν`, exactly NR57's phase-space-current entropy
   production; quadrature agrees to 1e−6 relative. For **arbitrary** OU
   systems (anisotropic 2-D and 3-D, random A, D) the spectral KL equals the
   Lyapunov/current formula `tr(ΩCΩᵀD⁻¹)` to 1e−4.
4. **Guards.** (i) Sampling is a deterministic map ⇒ the sampled-record EPR
   is a lower bound, monotone in dt (verified: rates 0.20→2.90 all below the
   continuous 3.03 as dt→0). (ii) The Gaussian-spectral EPR lower-bounds the
   true EPR of non-Gaussian processes: biased 3-state ring σ_gauss = 0.552 <
   σ_true = (p−q)ln(p/q) = 0.644, both zero for the fair ring — the
   second-order record certifies a floor, not the whole budget. (iii)
   Reversible OU (symmetric A): both formulas zero.

## The real deep ocean (ANDRO 700–1300 dbar, committed NR53 band cache)

Per-band per-10-day-step Gaussian EPR from the pooled odd/even lag
correlations (Bartlett taper; no determinant clipping anywhere; |C_R|<1):

| band | lat | σ (nats/step) | peak period | frac < 120 d |
|---|---|---|---|---|
| nh_polar | 70.7 | 8.2e−2 | 206 d | 0.43 |
| nh_subpolar | 55.9 | 1.8e−3 | 253 d | 0.42 |
| nh_mid | 41.6 | 1.8e−3 | 171 d | 0.35 |
| nh_sub | 27.1 | 4.7e−3 | 85 d | 0.17 |
| nh_trop | 12.0 | 1.3e−2 | 38 d | 0.00 |
| eq | 0.2 | **8.9e−5** | — | — |
| sh_trop | −12.5 | 8.9e−3 | 27 d | 0.00 |
| sh_sub | −28.0 | 4.8e−3 | 72 d | 0.17 |
| sh_mid | −42.3 | 1.1e−2 | 185 d | 0.38 |
| sh_subpolar | −55.9 | 3.4e−4 | 52 d | 0.09 |
| sh_polar | −67.1 | 4.2e−2 | 30 d | 0.28 |

* **The equator is the reversible line twice over** — f→0 kills V (NR52) and
  the band is the wave-polarized corner (NR56, Q/I=+0.44): σ_eq = 8.9e−5 is
  the smallest of all bands and sits **inside** its matched reversible
  surrogate null (q95 = 1.6e−4), while nh_trop exceeds its own null by
  ~120×.
* **The tropics pay at the eddy clock** — EPR density peaks at 38 d (NH) and
  27 d (SH), interior on every taper, with 0% of the entropy below the
  120-d window edge: the NR53 eddy-rotation clock carries the
  irreversibility, not the mean flow.
* **The two polar oceans split by clock** — the Arctic band's density is
  DC-side (peak 206 d, 43% below 120 d): irreversibility of a steady
  cyclonic *circulation* (NR53's no-flip clock). The Antarctic band pays at
  the *eddy* clock (30 d): the ACC/Weddell eddy field. The Arctic is
  irreversible the way a gyre is; the Antarctic the way an eddy field is.
* **Hemispheric mirror** — S_a flips sign across the equator but the density
  is sign-blind: NH/SH tropical σ agree within 1.4×.

## Placement

NR57 (spin = total EPR) + NR53 (two clocks of the spin) + NR56 (Stokes
faces) meet in one object: `dσ/dω = (1/π) V²(ω)/(I²−Q²−U²−V²)`. The odd
face's thermodynamic meaning is now *frequency-resolved*, and the mainstream
bridge is direct: every published rotary-coefficient spectrum is re-readable
as a second-law density. Mainstream anchors: Gonella 1972 (rotary
coefficient); Weiss 1975 (multivariate Gaussian reversibility ⇔ S = Sᵀ);
Lebowitz–Spohn EPR; Elipot & Lumpkin 2008 (drifter rotary spectra).

**Artifacts:** `figures/95_spectral_epr.json` (all 12 verdicts true);
17 unit proofs in `tests/test_spectral_epr.py`; offline-safe (committed
NR53 cache). Reproduce: `python general_two_clocks/new_relationships37.py`.
