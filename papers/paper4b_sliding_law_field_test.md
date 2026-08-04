# A field-measurable regularized-Coulomb sliding law: the s_N(N) master curve, an N_c inversion, and a tidal-admittance ungrounding early-warning

*Abdelrahman Saifelden (AIET) — Abdulrahman.Saifelden22011411@aiet.edu.eg*

---

## Abstract

Ice-stream sliding weakens as the bed approaches flotation, a behaviour large-scale
models impose by hand (Joughin, Smith & Schoof 2019 add an ad-hoc near-flotation ramp
to a regularized-Coulomb law). We show this weakening is a **closed-form pole** of the
regularized-Coulomb law itself and, crucially, that it is **measurable from surface
velocity alone**. Solving `τ_b=C N (u/(u+u₀))^{1/m}` at the driving stress gives, exactly,
the sliding-sensitivity master curve `|s_N|(N)=m/(1−(N_c/N)^m)` with `N_c=τ_d/C` — a
simple pole at the flotation threshold `N_c`, verified to `1.4×10⁻⁴` against the
numeric `s_N`. The curve makes three independent surface observables read the **same**
law: a lake-drainage step (`du/u=|s_N|·dN/N`), a continuous ocean-thermal-forcing gating
slope, and the **tidal admittance** of velocity (whose 2f/1f harmonic ratio diverges
toward `N_c`). A drainage-step population inverts for `N_c` to **0.2/0.4/1.0 %** at
5/10/20 % amplitude noise; the tidal probe recovers `m` and the dimensionless flotation
proximity `R=(N_c/N)^m` from velocity alone (answering the "no knowledge of `N`"
objection), with the tidal forcing amplitude *measured*, not assumed (CATS2008 across
the BedMachine grounding zone: median tidal pressure swing 0.30 % of overburden, rising
to 0.45 % within 2–4 km of the line). Because the basal-drag stiffness
`∂τ_b/∂u∝(1−R)²/R→0` at `N_c`, an ice stream nearing ungrounding should show rising
variance and lag-1 autocorrelation of surface speed; we **operationalize** the
established critical-slowing-down early-warning (Scheffer et al. 2009; Dakos et al.
2008; Rosier et al. 2021) as the continuous reading `R(t)=(N_c/N)^m→1` from tidal
admittance. Read through the same lens, three real MacAyeal-lake drainages sit far from
the `N_c` pole and show no critical-slowing-down signature — a correct **true negative**
on stable ice (the detectors fire on planted surges/CSD). The contribution is the
field-measurable law and its inversion, not a new early-warning concept; absolute `s_N`
remains uncalibrated and the strong test awaits a stream actually nearing flotation.

---

## 1. Introduction

The basal sliding law sets how fast an ice stream flows for a given driving stress and
effective pressure `N=p_i−p_w` (Schoof 2005; Gagliardini et al. 2007). Near the
grounding line `N→0` and sliding becomes Coulomb-limited; regularized-Coulomb laws
(Joughin, Smith & Schoof 2019; Tulaczyk et al. 2000; Gudmundsson 2011) capture this but
introduce the near-flotation weakening through an *ad hoc* ramp and a fixed velocity
scale `u₀`. The difficulty is that `N` is rarely known: it depends on the unobserved
subglacial water pressure.

The tools we use, not claim, are the regularized-Coulomb friction law (Schoof 2005;
Joughin et al. 2019), the nonlinear tidal (MSf) velocity response of ice streams
(Gudmundsson 2011), and critical-slowing-down early-warning theory, *including its
ice-sheet application* (Scheffer et al. 2009; Dakos et al. 2008; Rosier et al. 2021;
Boers & Rypdal 2021).

**What is new, stated against the closest prior work.** Joughin, Smith & Schoof (2019)
*impose* near-flotation weakening with a hand-tuned ramp; we *derive* it as a closed-form
pole `|s_N|=m/(1−(N_c/N)^m)` of the regularized-Coulomb law and turn `N_c` into a
**measured** quantity. Rosier et al. (2021) established critical-slowing-down indicators
(lag-1 autocorrelation, variance, Kendall-τ) for grounding-line flux ahead of
marine-ice-sheet tipping; we do not re-claim the early-warning but give a specific
**operational route** — reading `R(t)=(N_c/N)^m→1` from the continuous tidal admittance
of *surface velocity*, which needs no knowledge of `N`.

---

## 2. The `s_N(N)` master curve

**2.1 `s_N(N)` as the common observable.** A lake drainage is an in-situ effective-
pressure *step* whose surge amplitude `du/u=|s_N|·(dN/N)` measures the sliding
sensitivity `s_N=d ln u_b/d ln N`; ocean thermal forcing lowers `N` *continuously*, so
the gating slope `d ln u_*/dTF` measures the **same** `s_N` (`+0.46→+0.62/°C`, direct
BedMachine `N`); ocean tides oscillate `N` at known amplitude, a third probe (§4). All
three read one curve.

**2.2 The closed-form master curve.** Solving the regularized-Coulomb law
`τ_b=C N (u/(u+u₀))^{1/m}` at `τ_b=τ_d` gives, exactly,
`|s_N|(N)=m/(1−(N_c/N)^m)`, `N_c=τ_d/C` — `→m` far from flotation, a **simple pole** at
`N_c` (`≈N_c/(N−N_c)`). Verified to `1.4×10⁻⁴` against the numeric `s_N`. This *derives*
the near-flotation weakening that Joughin, Smith & Schoof (2019) impose by hand (ad hoc
`h_af<h_T` ramp + fixed `u₀`). (`validation/synthetic/sn_master_curve.py`)

---

## 3. Inversion — measuring `N_c` and `m`

A drainage-step population recovers the flotation threshold `N_c` to
**0.2 / 0.4 / 1.0 %** at 5 / 10 / 20 % amplitude noise; `m` is degenerate with the
per-event drop far from flotation. The recovery is unbiased and sign-faithful under
noise and null under permutation (synthetic plant-and-recover).

---

## 4. The tidal third probe — `N_c`, `m`, and `R` from velocity alone

Ocean **tides** give a continuous probe: the fundamental velocity admittance equals
`|s_N|` and the 2f/1f harmonic ratio diverges toward `N_c`, so admittance + harmonics +
the known tidal amplitude recover `m` and the dimensionless flotation proximity
`R=(N_c/N)^m` **from surface velocity alone** — a sliding-law reading of the Gudmundsson
nonlinear MSf response, answering the "no knowledge of `N`" objection. The tidal forcing
amplitude is **measured**, not assumed: sampling the CATS2008 tide model (USAP-DC 601235)
across the BedMachine grounding zone (within 10 km of the line; `n=1108` valid cells)
gives a tidal ocean-pressure swing with median **0.30 % of ice overburden** (median tide
`η≈0.84 m`), rising monotonically from 0.21 % at 8–10 km to 0.45 % at 2–4 km from the
grounding line (`validation/synthetic/tidal_admittance_probe.py`,
`validation/reports/tidal_forcing_gz.json`).

---

## 5. An ungrounding early-warning (operationalizing established CSD theory)

The `N_c` fold makes the basal drag stiffness `∂τ_b/∂u∝(1−R)²/R→0` (critical slowing
down), so an ice stream approaching ungrounding should show **rising variance and lag-1
autocorrelation of its surface speed** (Kendall-τ≈0.54 in an OU demonstration with `N`
declining toward `N_c`). Critical-slowing-down early-warning is established (Scheffer et
al. 2009; Dakos et al. 2008) and has already been applied to marine-ice-sheet tipping —
Rosier et al. (2021) detected rising lag-1 autocorrelation and variance in
grounding-line flux ahead of Pine Island tipping, and Boers & Rypdal (2021) in Greenland
surface melt. The contribution here is **not** the early-warning concept but a specific
**operational route**: reading `R(t)=(N_c/N)^m→1` from the continuous tidal admittance of
surface velocity. A single-snapshot **spatial** analogue (variance + along-flow
correlation length rising toward the grounding line) is also derived (exact Lyapunov
solve). (`validation/synthetic/spatial_ews.py`)

---

## 6. The framework on the same real lakes

Reading open CryoSat-2 drainage + ITS_LIVE velocity (3 MacAyeal lakes) through this lens
(`validation/external/lake_lag_sn_ews.py`, 5 tests): across the 3 resolved drainage
events `|Δv/v|≤2 %`, mixed sign, near the ~1 % year-to-year noise floor — **0 positive
>2σ surge detections** — so via the master curve these trunk lakes sit **far from the
`N_c` pole**; and **no** lake shows the joint critical-slowing-down signature (rising
variance *and* high AC1), so the early-warning correctly returns a **true negative** on
stable ice (unit tests confirm the detectors *do* fire on planted surges/CSD, so the null
is real). Both probes agree: far from ungrounding.

---

## 7. Estimator calibration, scope, and falsification

**7.1 Calibration.** Every estimator has a synthetic plant-and-recover harness in
`pytest`: the master-curve closed form (`1.4×10⁻⁴`), the `N_c` inversion
(0.2/0.4/1.0 %), the tidal admittance/harmonic recovery, and the CSD detectors
(fire on planted surges/CSD, null on stable OU).

**7.2 Scope and falsification.** Absolute `s_N` is uncalibrated (per-event `dN/N` is a
lumped-storage lower bound); the robust claims are the sign/shape and the *threshold*
`N_c`. With n≈8 annual points per lake the CSD test is weak; a strong test needs the
dense quarterly/GPS series and a stream actually nearing flotation. Falsifiers: a
co-located `dN`+GPS population whose `|s_N(N)|` does not follow `m/(1−(N_c/N)^m)`; a
stream nearing flotation with no variance/AC1 rise; tidal admittance/harmonics not
steepening toward the grounding line.

---

## 8. Reproduce

```bash
pip install -r requirements.txt
python glaciers/validation/synthetic/sn_master_curve.py        # §2 s_N(N) master curve + inversion
python glaciers/validation/synthetic/tidal_admittance_probe.py # §4 tidal third probe
python glaciers/validation/synthetic/spatial_ews.py            # §5 spatial early-warning
python glaciers/validation/external/lake_lag_sn_ews.py         # §6 framework on real lakes
pytest glaciers/tests/test_validation_synthetic.py -v          # §7 estimator calibration
```

## Data and code availability

All analysis code and derived products are in the public repository at
https://github.com/abdh0saifelden2-debug/K, and the results regenerate from the scripts
above. The study uses the following open datasets:

- BedMachine Antarctica v4 (Morlighem, 2025, https://doi.org/10.5067/POJQI54A45HX; see
  Morlighem et al., 2020).
- CATS2008 circum-Antarctic tide model: USAP-DC 601235 (Howard et al., 2019,
  https://doi.org/10.15784/601235).
- ITS_LIVE surface velocities (Gardner et al., 2025, https://doi.org/10.5067/JQ6337239C96;
  see Gardner et al., 2018) and CryoSat-2 lake-drainage dates.

BedMachine and ITS_LIVE were obtained from the NASA NSIDC DAAC; CATS2008 from the
U.S. Antarctic Program Data Center (https://www.usap-dc.org).

## References

Boers, N., & Rypdal, M. (2021). Critical slowing down suggests that the western Greenland Ice Sheet is close to a tipping point. *Proceedings of the National Academy of Sciences* 118(21), e2024192118. doi:10.1073/pnas.2024192118

Dakos, V., Scheffer, M., van Nes, E. H., Brovkin, V., Petoukhov, V., & Held, H. (2008). Slowing down as an early warning signal for abrupt climate change. *Proceedings of the National Academy of Sciences* 105(38), 14308–14312. doi:10.1073/pnas.0802430105

Gagliardini, O., Cohen, D., Råback, P., & Zwinger, T. (2007). Finite-element modeling of subglacial cavities and related friction law. *Journal of Geophysical Research: Earth Surface* 112, F02027. doi:10.1029/2006JF000576

Gardner, A. S., Moholdt, G., Scambos, T., Fahnestock, M., Ligtenberg, S., van den Broeke, M., & Nilsson, J. (2018). Increased West Antarctic and unchanged East Antarctic ice discharge over the last 7 years. *The Cryosphere* 12, 521–547. doi:10.5194/tc-12-521-2018

Gardner, A., Fahnestock, M., Greene, C. A., Kennedy, J. H., Liukis, M., Lopez, L., & Scambos, T. (2025). MEaSUREs ITS_LIVE Regional Glacier and Ice Sheet Surface Velocities, Version 2 (NSIDC-0776) [Data set]. *NASA NSIDC DAAC.* doi:10.5067/JQ6337239C96

Gudmundsson, G. H. (2011). Ice-stream response to ocean tides and the form of the basal sliding law. *The Cryosphere* 5, 259–270. doi:10.5194/tc-5-259-2011

Howard, S. L., Erofeeva, S., & Padman, L. (2019). CATS2008: Circum-Antarctic Tidal Simulation version 2008. *U.S. Antarctic Program Data Center.* doi:10.15784/601235

Joughin, I., Smith, B. E., & Schoof, C. G. (2019). Regularized Coulomb friction laws for ice sheet sliding: Application to Pine Island Glacier, Antarctica. *Geophysical Research Letters* 46, 4764–4771. doi:10.1029/2019GL082526

Morlighem, M., Rignot, E., Binder, T., Blankenship, D. D., Drews, R., Eagles, G., et al. (2020). Deep glacial troughs and stabilizing ridges unveiled beneath the margins of the Antarctic ice sheet. *Nature Geoscience* 13, 132–137. doi:10.1038/s41561-019-0510-8

Morlighem, M. (2025). MEaSUREs BedMachine Antarctica, Version 4 (NSIDC-0756) [Data set]. *NASA National Snow and Ice Data Center DAAC.* doi:10.5067/POJQI54A45HX

Padman, L., Fricker, H. A., Coleman, R., Howard, S., & Erofeeva, L. (2002). A new tide model for the Antarctic ice shelves and seas. *Annals of Glaciology* 34, 247–254. doi:10.3189/172756402781817752

Rosier, S. H. R., Reese, R., Donges, J. F., De Rydt, J., Gudmundsson, G. H., & Winkelmann, R. (2021). The tipping points and early warning indicators for Pine Island Glacier, West Antarctica. *The Cryosphere* 15, 1501–1516. doi:10.5194/tc-15-1501-2021

Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H., van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early-warning signals for critical transitions. *Nature* 461, 53–59. doi:10.1038/nature08227

Schoof, C. (2005). The effect of cavitation on glacier sliding. *Proceedings of the Royal Society A* 461, 609–627. doi:10.1098/rspa.2004.1350

Siegfried, M. R., & Fricker, H. A. (2021). Illuminating active subglacial lake processes with ICESat-2 laser altimetry. *Geophysical Research Letters* 48, e2020GL091089. doi:10.1029/2020GL091089

Tulaczyk, S., Kamb, W. B., & Engelhardt, H. F. (2000). Basal mechanics of Ice Stream B, West Antarctica: 1. Till mechanics. *Journal of Geophysical Research: Solid Earth* 105(B1), 463–481. doi:10.1029/1999JB900329
