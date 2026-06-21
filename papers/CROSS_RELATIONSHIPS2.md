# New cross-cutting relationships, batch 2 (NR8–NR10)

Continuation of `CROSS_RELATIONSHIPS.md` (NR1–NR7). Same question — *what do the
findings, taken together, imply, and can we derive new, useful relationships to the
mainstream?* — pushed three steps further. Each is **derived** here and **verified
one-by-one** in `glaciers/validation/synthetic/cross_relationships2.py`
(`test_cross_relationship2_verified` + three targeted tests in
`glaciers/tests/test_validation_synthetic.py`). No external data; CPU only.

| # | Relationship | Links | Status |
|---|---|---|---|
| NR8 | The flotation fold has a **spectral** face: velocity-noise PSD is Lorentzian, corner `f_c=λ/2π ~ (N−N_c)²`; FDT identity `Var·2πf_c = D` | P4 / NR3 ⇄ Kuehn / Bury (spectral EWS) ⇄ fluctuation–dissipation | `[DERIVED, VERIFIED]` |
| NR9 | **Causality (Kramers–Kronig)** ties scallop migration to eddy-viscosity dispersion; a scale-dependent eddy viscosity with zero migration is acausal | P3 ⇄ P1 ⇄ NR2 ⇄ Kramers–Kronig / Kraichnan / Hanratty | `[DERIVED, VERIFIED]` |
| NR10 | **Height above flotation `h_af`** is the single state variable unifying the RTN intrusion surface, the Schoof MISI fold, and the `s_N` pole | P4 §6 ⇄ P4 §10 / NR3 ⇄ Schoof 2007 | `[DERIVED, VERIFIED]` |

---

## NR8 — The flotation fold has a spectral face, closed by fluctuation–dissipation `[DERIVED, VERIFIED]`

**Statement.** NR3 gave the *time-domain* early-warning (rising variance + lag-1
autocorrelation) at the `s_N(N)` flotation pole `N_c`. The same pole has a
*frequency-domain* signature that the variance/AC1 pair only summarises. Near the
fold the velocity perturbation is an Ornstein–Uhlenbeck process

> `dx = −λ(N) x dt + √(2D) dW`,  `λ(N) ∝ (1−R)²/R ~ (N−N_c)²`  (`R=(N_c/N)^m`),

whose stationary power spectrum is a **Lorentzian**

> `S(ω) = 2D/(λ² + ω²)`,  corner `f_c = λ/2π → 0`  as `(N−N_c)²`.

So the velocity spectrum **reddens** (power migrates below any fixed frequency) as the
stream approaches ungrounding — the spectral early-warning of Kuehn (2011) and Bury,
Bauch & Anand (2020), here with the fold exponent *derived* rather than assumed. The
one rate `λ` sets **both** the fluctuation spectrum (corner `f_c`, variance `D/λ`) and
the deterministic response relaxation, giving a **fluctuation–dissipation identity**

> `Var · (2π f_c) = D`,  constant in `N`.

This makes the noise corner a **calibration-free flotation-proximity gauge**: `f_c`
falls as `(N−N_c)²` independent of the (unknown) noise strength `D`, and combined with a
response measurement (the tidal admittance `|s_N|`, NR6) it returns `D` itself.

**Verification.** `exp(f_c)=+1.998`, `exp(Var)=−1.998` (matching NR3's `±2`); the
low-frequency power fraction rises monotonically toward `N_c` (reddening); the FDT
product `Var·2πf_c=D` is constant to `5×10⁻¹⁷`; and a **simulated** OU PSD recovers the
corner `f_c=λ/2π` to **3.1%** (Welch + Lorentzian fit) — the corner is measurable, not
just analytic. (`nr8_spectral_fdt_ews`.)

---

## NR9 — Causality (Kramers–Kronig) couples migration to eddy-viscosity dispersion `[DERIVED, VERIFIED]`

**Statement.** NR2 wrote the structure a down-gradient closure discards as one complex
transport admittance `Z = a_r + i a_i` (real part = amplitude-rate / backscatter,
imaginary part = scallop migration) **at a single mode**. The deeper fact is that the
flux response in time is **causal**, so `Z(ω)` is analytic in a half-plane and its real
and imaginary parts are **Kramers–Kronig (Hilbert) conjugates**. Consequently the two
"shadows" are *not independent across scale*:

- A **scale-dependent** eddy viscosity — `Re Z` varying with wavenumber `k`, as every
  real closure (spectral eddy viscosity, Smagorinsky `∝|S|`) is — **forces a nonzero
  migration** `Im Z` by Kramers–Kronig.
- K-theory keeps that scale-dependent `Re Z` but sets `Im Z = 0`. That combination
  **violates Kramers–Kronig**: it is *acausal*. The only causal zero-migration response
  is a frequency-independent real constant — pure instantaneous diffusion.
- Equivalently, the **migration-vs-`k` spectrum is the Kramers–Kronig transform of the
  eddy-viscosity-vs-`k` (backscatter) spectrum** — measure one, predict the other.

**Derivation.** For a causal relaxation kernel `K(t)=(g/τ)e^{−t/τ}Θ(t)` (the MZ
impedance / the `K_hydraulic` form of §7.3) the admittance is
`Z(ω)=g/(1+iωτ)`, i.e. `Re Z = g/(1+(ωτ)²)`, `Im Z = −gωτ/(1+(ωτ)²)`, a Kramers–Kronig
pair. The migration ratio `I = |Im Z|/(2π|Re Z|) = |ωτ|/2π` is nonzero for any memory
`τ>0` and `→0` only as `τ→0`.

**Verification.** The discrete KK/Hilbert transform of `Re Z` reconstructs the analytic
`Im Z` to **5.3%** (interior, edge-trimmed). The K-theory projection (same `Re Z`,
claimed `Im=0`) carries an **O(1)** KK residual (`=1.0`, total) — the projection cannot
be causal. The migration ratio increases with `ωτ` and vanishes as `τ→0`.
(`nr9_kramers_kronig_migration`.)

---

## NR10 — Height above flotation unifies RTN, the Schoof fold, and the `s_N` pole `[DERIVED, VERIFIED]`

**Statement.** Three thresholds the manuscripts treat in separate sections are all
functions of **one state variable**, the height above flotation
`h_af = H − H_f`, `H_f = (ρ_w/ρ_i) d_base` (Schoof 2007 flotation condition):

- **Effective pressure (ocean-connected).** `N = ρ_i g h_af` *exactly*, so `N=0 ⇔
  h_af=0 ⇔` flotation.
- **RTN ocean-intrusion (§6).** `RTN(φ) = H_f/(φ H)`, so `RTN=1 ⇔ h_af = H(1−φ)`, which
  `→ 0` as `φ→1`: the `RTN=1` intrusion surface *is* the flotation surface once the
  subglacial water reaches overburden.
- **Sliding-law fold / NR3 MISI saddle-node.** The `|s_N|` pole at `N_c` sits at
  `h_af^c = N_c/(ρ_i g) > 0` — a few **metres above** geometric flotation.

So a thinning stream hits the **drag-side fold first** (basal-drag weakening, the
NR3/NR8 velocity early-warning) and only later **ungrounds** (`h_af→0`, `RTN→1`). The
sliding-law early-warning therefore precedes geometric flotation — a buffer set by
`N_c`.

**Verification.** On a synthetic thinning sweep at `d_base=800 m` (`H_f=896.8 m`): the
identity `N=ρ_i g h_af` holds to machine zero; `RTN=1` occurs at `h_af=H(1−φ)` for every
`φ` and coincides with flotation (`h_af=0`) at `φ=1`; the `s_N` pole sits at
`h_af^c=6.67 m > 0`, with `|s_N|` rising toward it from the grounded side.
(`nr10_flotation_unifier`.)

---

### What is genuinely new vs. a sharpening

- **New:** NR8 (the spectral corner + FDT identity is a *different, calibration-free*
  observable from NR3's variance/AC1), NR9 (causality forces migration whenever the
  eddy viscosity is scale-dependent — a structural impossibility result for K-theory,
  not just a measured gap), NR10 (the explicit `h_af` unification of the RTN intrusion
  predictor with the `s_N`/Schoof fold, including the ordering "fold precedes
  ungrounding").

### Honest limits

NR8's OU reduction is the linearised near-fold limit; the spectral corner is resolvable
only while `λ` stays above the inverse record length (the same sweep-rate caveat as NR3).
NR9's `Z(ω)` is a single-pole causal model of the discarded structure, not a turbulence
closure; the claim is the KK *coupling* of `Re` and `Im`, demonstrated on that model.
NR10's `N=ρ_i g h_af` identity assumes ocean-connected basal water; the `φ<1` case gives
`RTN=1` inland of flotation by the margin `H(1−φ)`, as stated. All references remain
`[cite]` slots — none invented.
