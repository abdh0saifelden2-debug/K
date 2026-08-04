# New cross-relationship — NR46 (`general_two_clocks/new_relationships23.py`) — real-data (ledger E12)

Continues the derived-and-verified program (NR1–NR45; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only; offline-safe — every number re-derives from the committed 34 KB
aggregate cache [`data/nr46_solar_cache.json`](data/nr46_solar_cache.json)
(raw archives are the public SOHO mission-long bundles + BiSON open-data
portal; regeneration via `build_cache()` with env paths).
Unit-proofs in [`tests/test_solar_two_clocks.py`](tests/test_solar_two_clocks.py)
(13 tests).

```bash
python general_two_clocks/new_relationships23.py   # -> figures/81_solar_two_clocks.json
pytest general_two_clocks/tests/test_solar_two_clocks.py -v
```

---

## NR46 — The Sun's two clocks in one cross-spectrum: cross-channel coherence is a **resonance detector**, the convective slow clock is **channel-local**, and the coherence-gated I–V phase is the **nonadiabaticity meter** — NR28's protocol at stellar scale  [NR28 × GOLF × VIRGO/SPM × BiSON × Jiménez et al. 1999]

### The claim under test (ledger E12, as written)

NR28's coherence-drop two-clocks fingerprint on BiSON/GONG-class p-mode +
granulation series — *pure protocol demonstration at stellar scale; no new
equation*.

### Data

26 years of simultaneous Sun-as-a-star series: **GOLF** calibrated velocity
(20 s, 1996–2022, Appourchaux et al. 2018 mission-long bundle), **VIRGO/SPM**
green + blue irradiance (60 s, level 2, 1996–2023), **BiSON** network
velocity residuals (40 s; Davies et al. 2014, Hale et al. 2016) as the
ground-based control. 602 gap-gated 11.4-day Hann windows (≥99 % joint
validity, df = 1.02 µHz — modes resolved from background).

Timing: the two mission-long products carry different stamping conventions;
the relative offset is self-calibrated by the **phase-slope method the
literature itself prescribes** (Jiménez et al. 1999 §2.4), anchored to one
number (their raw ⟨all⟩ I–V = −121.3°): dt = +55 s, consistent with the
documented VIRGO 30-s daily-pulse convention plus sample centering. All
coherence magnitudes and the mode-vs-background phase *separation* are
dt-invariant.

### Findings (`figures/81_solar_two_clocks.json`; I–V phase, V downward-positive, adiabatic −90°)

| band | V×I (GOLF×green) | V×V (BiSON×GOLF) | I×I (green×blue) |
|---|---|---|---|
| granulation 300–1200 µHz | **0.065** | **0.035** | **0.988** |
| p-band mode bins (γ²≥0.4) | **0.65 med, 0.96 max** | 0.91 (top quintile) | 0.974 |
| above cutoff 5.8–8 mHz | **0.014** | **0.032** | **0.845** |

1. **Cross-channel coherence is a resonance detector.** Only the global
   standing p-modes are coherent across observables (V×I 395 gated bins,
   peaks 0.96; two independent velocity instruments 0.91). The fast clock is
   channel-global.
2. **The convective slow clock is channel-local.** V×I and V×V decohere in
   the granulation band while the same-signal control I×I holds 0.988 —
   granulation is one physical signal *per channel* (different lines,
   heights, weightings), not one global signal. The NR28 "coherence drop =
   decoupling" transfers with its axis rotated: decoupling is
   cross-*channel* at low frequency, not small-scale at high frequency.
3. **The above-cutoff collapse is a cross-channel property, not a band
   property** (I×I keeps 0.85 where V×I falls to 0.014).
4. **The coherence-gated mode phase is one-sided, bounded, and lands 29°
   below adiabatic** — −118.8° at 2.9–3.3 mHz (Jiménez −121.3 ± 1.2; IPHIR
   −119 ± 3), central dip rising to −84° by 4.3 mHz (their shape). The
   departure is on the side pure Newtonian radiative cooling **cannot
   produce** (Marmolino–Severino model curves sit above −90°): the phase
   meter selects the convection-coupled nonadiabatic models (Houdek et al.
   1995), which is Jiménez's model-3 conclusion — now standing on 26 years
   instead of 2 months.
5. **The inter-mode background is a second phase branch in the same band**,
   140° from the mode branch at 2.6 mHz and converging monotonically
   (139.6 → 104.8 → 59.6 → 35.5 → 15.5°) into it by 4.2 mHz as linewidths
   blend — the Severino/Jiménez background-interference structure, measured
   as a branch pair separated bin-by-bin by the coherence gate.
6. **Protocol stability at stellar scale**: 13 two-year epochs across two
   full activity cycles, mode-phase circular std 2.2° — the fingerprint is a
   property of the star, not the epoch (extends Jiménez 2002).
7. Controls: I×I zero-phase to −0.6° (published green–blue +1.1); V×V
   +15.7° ≈ 0 within timing conventions (~14 s at 3 mHz).

### What transfers from NR28, and what does not

Transfers verbatim: coherence gating for phase validity; one-sided bounded
phase inside the coupled band; coherence structure as the clock separator.
Does **not** transfer: the `arctan(ω/ω_c)` first-order-lag law — the solar
mode-phase departure has the **opposite sign** (convective, not
relaxational), and the decoupling axis is channel-locality rather than
high-frequency noise. E12 asked for a protocol demonstration with no new
equation; the demonstration *and its boundary* are the result: the protocol
is portable, the constitutive lag law is system-specific.

### Limits

The absolute phase level rests on a one-number timing anchor from the
literature (everything else is measured independently of it); GOLF single-wing
epochs carry a known ±8° wing-dependent leak (Pallé et al. 1999) absorbed in
the epoch scatter; BiSON×GOLF timing (UT vs TAI conventions) is only
controlled to ~15 s; ℓ = 0–3 modes are blended in Sun-as-a-star bins (the
literature ℓ-splitting is not reproduced here).
