# NR69 — the atmosphere is a ladder of clock crossovers, and the exobase is where closure dies

**Module** `general_two_clocks/new_relationships46.py` ·
**data** the committed NR68 NRLMSIS 2.1 cache (4 real-driver conditions, 80–600 km) ·
**figure** `general_two_clocks/figures/104_clock_ladder.json` + `.png` ·
**tests** `general_two_clocks/tests/test_clock_ladder.py` (10)

## The relationship

NR68 measured rung 1 (turbopause = flow-clock/molecular-clock composition handoff).
NR69 shows the same clock structure repeats twice more overhead, and that the three
mainstream upper-atmosphere boundaries are three crossings of ONE pair structure:

| rung | boundary | crossover | what hands off |
|---|---|---|---|
| 1 | turbopause | K_zz = D(z) | composition: scalar-blind → scalar-selective (NR68) |
| 2 | exobase | l = H ⟺ Kn = 1 | *closure itself*: fluid/diffusive → ballistic |
| 3 | escape | λ vs λ_c = 2γ | retention: Jeans leak (λ>λ_c) vs bulk blow-off |

Mainstream anchors: exobase DEFINED by mean free path = scale height, Kn ≈ 1
(Chamberlain 1963; Bauer & Lammer 2004; Lammer 2022); sharp hydrodynamic/Jeans
transition at λ_c = 2γ ≈ 2.8–3.3 (Gruzinov 2011; Volkov et al. 2011 ApJL); K_zz at
the turbopause 100–1000 m²/s (Colegrove 1965: 4×10⁶ cm²/s; Kelley 2003;
Vlasov & Kelley 2014 — who infer K_zz from the same MSIS He/Ar/N2 profiles NR68 reads).

## The one-curve/two-ceilings theorem

Hard-sphere kinetics gives exactly (same v̄ throughout):

**Kn(z) = l/H = 3D(z)/(v̄H) = τ_coll/τ_transit**   (verified < 1e−9 on all profiles)

so the rising molecular diffusivity D ∝ 1/n meets TWO ceilings: the **eddy ceiling**
K_zz (~10²–10³ m²/s → rung 1) and the **ballistic ceiling** v̄H/3 (~10⁵–10⁶ m²/s →
rung 2, because D = v̄H/3 ⟺ Kn = 1). The rung ORDER z_turbo < z_exo is therefore
forced iff K_zz < v̄H/3 — the atmosphere is *subsonically stirred*. Measured margin:
the ceilings are **3100–3400× apart** (3.5 decades) at the turbopause. The ladder
order is a theorem about subsonically-stirred atmospheres, not a contingency.

## Measured rungs (committed cache, offline, no fit)

| quantity | measured | mainstream |
|---|---|---|
| rung 1: D = K_zz (K_zz 100–1000) | 107–119 km (400: 114 km) | turbopause ~100–115 km |
| — vs NR68 composition band | inside 99–132 km | two independent estimates agree |
| rung 2: Kn = 1 exobase | 334–398 km, T_exo 660–782 K | 350–700 km over solar cycle; low-activity epoch → low exobase, textbook |
| rung 3: λ at exobase | H 9.2 < He 36 < O 143 < N2 250 | Earth-H λ ≈ 9 standard |
| H blow-off critical T = m_H g r/(λ_c k) | 2170–2560 K | actual 660–780 K → factor ~3 below blow-off |

Every species — including H — sits above λ_c: Earth leaks molecule-by-molecule
(Jeans + non-thermal), never in bulk. That factor-3 temperature margin is why Earth
keeps its water on Gyr timescales.

## The Paper-1 reading: where closure dies

Kn = τ_coll/τ_transit is the GLE **memory-time/system-time ratio**. Below the
exobase the collision kernel is short (Kn ≪ 1) and a local Markovian closure
exists — scalar-blind eddy diffusion below rung 1, scalar-selective molecular
diffusion between rungs 1 and 2 (the two closures NR68 distinguished). At rung 2
the kernel support reaches the system's own evolution time: **no local closure of
any kind survives**, and transport goes exact-ballistic — collisionless streaming
with nothing left to model. Paper 1's "what closure discards" has a spatial
completion: the atmosphere exhibits, in one vertical column, the full sequence
scalar-blind closure → scalar-selective closure → **no closure** — and each
boundary between regimes is a measured clock crossover.

## Honest scope

Single generic σ = 3×10⁻¹⁹ m²: a factor-2 change moves z_exo by ~one scale height
(~40 km) — the exobase is a band in the literature too (350–700 km, solar cycle).
D = v̄l/3 with the mean mass is a rung locator, not species-resolved
Chapman–Enskog — rung-1 precision belongs to NR68's composition discriminant,
which needs no σ at all. True escape is ≈2–2.5× the classical Jeans rate near
Kn ~ 0.2 (Volkov 2017) — nothing here uses absolute rates, only orderings and
crossings. NRLMSIS 2.1 is the mainstream empirical reference atmosphere, not a raw
instrument record. λ_c bracket 2.8–3.3 immaterial (Earth clears it by ≥2.7× in λ).

## Verification

`python3 general_two_clocks/new_relationships46.py` regenerates figure/JSON from the
committed cache. `pytest general_two_clocks/tests/test_clock_ladder.py` (10 passed):
mfp 1/(nσ) scalings, v̄ textbook value + √T/√m scalings, the exact
Kn = τ_coll/τ_transit = 3D/(v̄H) identity on synthetic and real profiles (<1e−9),
log-interp crossing finder, rung-1 band vs mainstream + NR68, rung-2 band + rung
separation, subsonic forcing margin >100×, rung-3 Jeans ordering with H closest to
but above λ_c, blow-off temperature margin, verdict completeness.
