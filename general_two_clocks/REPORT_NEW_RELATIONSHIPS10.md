# New cross-relationship — NR33 (`general_two_clocks/new_relationships10.py`)

Continues the derived-and-verified program (NR1–NR32; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, deterministic; unit-proofs in
[`tests/test_new_relationships10.py`](tests/test_new_relationships10.py) (8 tests). Run:

```bash
python general_two_clocks/new_relationships10.py   # -> figures/nr33_persistence_pressure_bound.{json,png}
pytest general_two_clocks/tests/test_new_relationships10.py -v
```

---

## NR33 — The persistence–pressure bound: radar persistence of distributed basal water is a **pressure gauge** — the slow (geometry) clock reads the level of the fast (pressure) field  [Nye 1953 closure × Schroeder 2013/2015 specularity × the two-clocks operator split; sharpens P4a §6.4 via §V.1d]

### The gap this closes

Radar bed-echo specularity content maps distributed basal water (Schroeder et al. 2013,
2015; Dow et al. 2020) and NR32 made it the field coordinate of the drainage-response
window — but it has carried no *quantitative pressure meaning*: the two physically-arguable
spec→`p_w` sign conventions moved the §V.1c intrusion classification in opposite
directions (162 flagged vs 89 dropped), and the stated unblock (co-located borehole
pressure) does not exist in East Antarctica. NR33 derives the missing quantitative link
from **persistence**.

### The statement

A basal water body observed specular in repeat surveys separated by `t_obs` carries a
**pointwise lower bound on the water-pressure fraction** `φ = p_w/p_i`:

> `N ≤ N_max(t_obs) = n·(2 A t_obs)^(−1/n)`  ⇒  `φ ≥ 1 − N_max/(ρ_i g H)`,

from Nye creep closure (`V_c/S = 2A(N/n)^n`): a cavity at effective pressure `N` has
e-folded shut within `t_obs` unless `N ≤ N_max`. Numbers: **3.5 bar at 4 yr** (temperate
`A`), i.e. `φ ≥ 0.98–0.99` under 2–4 km of East Antarctic ice. Specular reflectors are
*wide flat* water bodies; a thin sheet (gap `h`, half-width `w ≫ h`) sags faster by
`~(n^n/2)(w/h)`, sharpening the bound to `N_max_sheet = (h/(A t_obs w))^(1/n)` —
**~0.1 bar** at `h=0.1 m, w=500 m` — persistent distributed specular water sits essentially
**at flotation**, consistent with borehole `N` of 0.1–1.6 bar over distributed West
Antarctic beds (Blankenship et al. 1987; Engelhardt & Kamb 1997) and with
distributed-drainage theory operating near overburden (Walder 1986; Kamb 1987; Hewitt
2011; Werder et al. 2013). The cylinder form is the rigorous-conservative anchor; the
sheet form is a scaling (O(1) prefactor).

**Two-clocks reading.** Everywhere else in this program the fast (elliptic, instantaneous)
pressure field constrains the slow (parabolic/creep, memory-carrying) partner's
statistics. NR33 is the converse: a **time integral of the slow clock (geometric
persistence) measures the level of the fast field**. Memory is not just a correction — it
is a *gauge*: the bed remembers its pressure history in its geometry, and repeat radar
reads that memory out as a pressure floor.

### Verified (unit-proofs, deterministic)

- **e-fold criterion**: explicit ODE integration of `dS/dt = −2A(N_max/n)³S` over `t_obs`
  lands on `e⁻¹` to `9×10⁻⁷` (closed form exact by construction).
- **Sheet/cylinder ratio**: `N_max_sheet/N_max_cyl = (2h/(n^n w))^(1/n)` matches the
  implementation to machine precision; strictly tighter for every wide sheet (`w ≥ h`);
  coincidence only at the unphysical tall-slot aspect `w/h = 2/n^n`.
- **Archive dividend**: `N_max ∝ t_obs^(−1/3)` exactly (log-log slope −1/3): deeper
  archives tighten the floor with no new instrument.
- **Cold-ice systematic**: colder ice (lower `A`) raises `N_max` as `A^(−1/n)` — one
  decade in `A` costs only 2.15×; carried as the dominant systematic.
- **Magnitudes**: 12-row (t_obs × H) table; `φ_floor(cyl) > 0.97` for `H ≥ 2 km` at 4 yr;
  sheet floors `> 0.999`.

### Applied to real data (ICECAP two-epoch persistence, USAP-DC 601371)

Binning the 2008/09 and 2011/12 ICECAP seasons separately onto the 5-km Bedmap2 grid:
**259 grounded cells** are sampled (≥3 points) in *both* epochs; **7** are persistent-specular
(`spec ≥ 0.2` in both; 14/9/7/2 across thresholds 0.10/0.15/0.20/0.30). Their floors:
median `φ ≥ 0.9898` (cylinder, rigorous) / `0.99975` (sheet scaling); median `H = 3878 m`
(five of seven sit in the deep Aurora-basin interior). Small-n is honest: crossover
coverage between epochs is sparse flight-line geometry, but seven pointwise physical
bounds is seven more than existed before, and the threshold sensitivity is flat.

### Registered falsifiable corollaries

- **(C1) Model constraint** — any hydrology model (GlaDS-type, Shreve routing) assigning
  `N > 3.55 bar` on the §V.1d persistent set is falsified at those cells (registered
  2026-07-04; decidable against any published modelled-`N` map of the Aurora/Totten
  region, e.g. Dow et al. GlaDS runs).
- **(C2) Sign corollary → §V.1d** — the floor constrains only the *high-spec* (persistent)
  cells, so it fixes the **orientation** of any monotone spec→`φ` map: an increasing map
  keeps a free low end; a decreasing map is squeezed entirely above the floor (admissible
  range `≤ 1−φ_floor ≈ 0.02`), so the §V.1c "connectivity" convention (claimed range 0.17)
  is **inadmissible**. Executed with the lake-anchor and lubrication lines in
  `glaciers/validation/external/rtn_spec_sign_pin.py`.
- **(C3) Archive dividend** — joining ICECAP with newer UTIG/OIB East Antarctic reflights
  must tighten the persistent-set floor as `(t₂/t₁)^(1/3)` — a dated survey-design
  prediction testable from archive geometry alone.

### Honest limits

One-sided only (disappearance of specularity does not bound `N` from below — drainage
mimics closure); assumes quiescence (melt-opening ≪ closure), valid for
low-through-flux distributed/ponded water — a steady R-channel evades the bound but is a
*low*-specularity object at km footprints (Schroeder 2013, 2015), so the evasion does not
contaminate the specular population; temperate-`A` assumption is the dominant systematic
(direction and magnitude quantified above); the sheet form is a scaling, not a theorem.
