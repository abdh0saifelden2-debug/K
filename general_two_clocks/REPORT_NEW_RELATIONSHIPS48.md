# NR71 — the ambipolar field is the ionosphere's divergence-cleaning multiplier

**Module** `general_two_clocks/new_relationships48.py` ·
**data** committed IRI2016 cache (`data/nr71_iri_cache.json`, 4 conditions,
150–650 km; builder `build_nr71_iri_cache.py`) ·
**figure** `general_two_clocks/figures/106_ambipolar_multiplier.json` + `.png` ·
**tests** `general_two_clocks/tests/test_ambipolar_multiplier.py` (10)

## The relationship

Paper 1's elliptic face: incompressibility is enforced instantaneously by a
constraint field (pressure) with no dynamics of its own; discard it and closure
fails. The ionosphere repeats the structure with charge instead of volume:

| | Paper 1 | ionosphere |
|---|---|---|
| constraint | div u = 0 | n_e = n_i (quasineutrality) |
| ellipticity | Poisson / instantaneous | Debye ~cm ≪ gradients ~100 km |
| multiplier | pressure | ambipolar potential/field E |
| read-out | p from the projection | eE = −(1/n_e) d(n_e kT_e)/dz — exact |
| discard cost | closure collapse | barometric clock fails ×1.5 |

## Theorems (exact, verified <2e−3 on synthetic columns)

1. **Multiplier read-out.** Inertialess electrons: eE = −kT_e dln n_e/dz − k dT_e/dz.
   The multiplier is fixed by the constraint + the observed profile; no ion
   dynamics enters (Schunk & Nagy 2009 ch. 5).
2. **The doubled clock.** Adding the ion balance: H_p = k(T_e+T_i)/(m_i g) — the
   barometric clock with the SUM of temperatures (T_e=T_i → exactly ×2; general
   lift 1+T_e/T_i), static lift fraction eE/(m_i g) = T_e/(T_e+T_i). The light
   species' thermal clock is handed to the heavy species through the constraint —
   the inverse of NR68: there the flow clock gave every species ONE scale height;
   here the constraint welds two species back into one taller column.
3. **Excess = outflow driver.** Steady field-aligned flux steepens the profile, so
   the read-out lift exceeds T_e/(T_e+T_i); lift > 1 = net upward force on O+ —
   the polar-wind configuration (Banks & Holzer 1968; Axford 1968 — the channel
   NR70's helium problem required).

## Measured (committed IRI2016 cache, topside O+ window)

| condition | H lift over naive clock | lift eE/(m g) | static T_e/(T_e+T_i) | excess |
|---|---|---|---|---|
| midlat noon (60N) | ×1.60 | 0.95 | 0.63 | **+0.33** |
| midlat midnight | ×1.45 | 0.79 | 0.53 | **+0.26** |
| midlat winter noon | ×1.52 | **1.05** | 0.66 | **+0.39** |
| equator noon | ×1.55 | 0.67 | 0.55 | +0.12 |

- The naive single-fluid barometric clock is excluded everywhere (×1.45–1.60).
- The equatorial column sits **on** the static prediction (excess +0.12; and
  H_meas/H_static-DE = 0.92): statically supported.
- High-dip columns carry a persistent excess +0.26…+0.39 — winter noon exceeds
  FULL weight support (lift 1.05 > 1): the multiplier is doing outflow work, not
  just holding the column up. The same parallel plasma channel that NR70 needed
  for helium shows up here as the measured excess of the constraint multiplier.

## Honest scope

IRI2016 is the mainstream empirical ionosphere (Bilitza et al.), not raw ISR
data; its topside is an empirical shape, so the high-dip excess is IRI's encoding
of the observed steepened topside — attribution to field-aligned outflow is the
mainstream reading, not proven from IRI alone. The 0N,0E control is not a
field-aligned column (low dip, fountain geometry); it is quoted only for the
excess contrast. For straight field lines the dip angle cancels in both the
vertical scale height and the lift fraction. m_eff from IRI's own composition
(O+ > 85% windows). No absolute outflow flux is claimed — only the sign and size
of the multiplier excess.

## Verification

`python3 general_two_clocks/build_nr71_iri_cache.py` rebuilds the cache (needs
gfortran; the committed JSON replays offline). `python3
general_two_clocks/new_relationships48.py` regenerates figure/JSON.
`pytest general_two_clocks/tests/test_ambipolar_multiplier.py` (10 passed):
read-out recovers static lift on exact synthetic DE columns (2e−3), pure
thermo-electric term exact, ×2 doubling at T_e=T_i and 1+T_e/T_i generally,
lift-fraction/effective-mass identities, naive-clock exclusion band, measured
lift bands, high-dip excess vs equatorial static support, DE-ratio split,
verdict completeness.
