# New cross-relationship — NR30 (`general_two_clocks/new_relationships7.py`)

Continues the derived-and-verified program (NR1–NR29; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, deterministic; unit-proofs in
[`tests/test_new_relationships7.py`](tests/test_new_relationships7.py). Run:

```bash
python general_two_clocks/new_relationships7.py   # -> figures/nr30_phi_not_leray_pressure.{json,png}
pytest general_two_clocks/tests/test_new_relationships7.py -v
```

---

## NR30 — The subglacial hydraulic potential `φ` is **not** a Leray pressure: it is the *parabolic* (finite-time, screened) Darcy head of a bed *with storage*, whereas the Leray pressure is the *elliptic* (instantaneous, bare-Poisson) constraint multiplier of an incompressible flow *without* storage. They coincide only in the singular no-storage limit `τ_hyd → 0`  [P4/P4a (`φ`) × P1 (Leray `p`) × NR28]

### The gap this closes

Paper 4 (§9) and Paper 4a (§8) carry the **identical** open caveat, verbatim: *"the physical
distinctness of `φ` from the Leray pressure remains unproven."* The worry is structural — both
`φ` and the turbulent pressure `p` are "pressures that enforce a constraint," so a referee can
reasonably ask whether the subglacial hydraulic potential is just a Leray pressure wearing a
glaciological hat. NR30 settles it: they are governed by **different PDE types**, have
**different Green's functions**, and **coincide only in an unphysical singular limit**. It is the
one open caveat shared by two of the manuscripts, so closing it strengthens both at once.

### Derivation

Two different conservation statements produce two different operators.

**Leray pressure (Paper 1; REPORT_THEORY §6).** Incompressible Navier–Stokes enforces the
*instantaneous kinematic* constraint `∇·u = 0` at every instant. Taking the divergence of the
momentum equation yields an **elliptic** Poisson problem with **no time derivative** and **no
material parameter**:

> `∇²p = −∂_i ∂_j (u_i u_j)`,  so in Fourier  `p̂(k) = (k_i k_j / k²)·(uu)^`,  transfer  **`H_p(k) = 1/k²`** — real and ω-independent.

The pressure is a Lagrange multiplier slaved to the velocity: no storage, no relaxation time, no
memory. It adjusts instantaneously and non-locally (bare-Poisson Green's function `G_p ∼ 1/r` in
3-D, `∼ log r` in 2-D).

**Hydraulic potential (Paper 4/4a; Röthlisberger 1972; Werder et al. 2013).** Subglacial water
mass conservation **with storage `S`** is `S ∂_t φ = ∇·(k_h ∇φ) + m`, i.e. the **parabolic**
(Darcy diffusion) equation

> `∂_t φ = D_h ∇²φ + m/S`,  hydraulic diffusivity `D_h = k_h/S`,  harmonic transfer  **`H_φ(k,ω) = 1/(k² + i ω/D_h)`**.

This has a time derivative (storage), a material diffusivity, hence a **finite hydraulic time**
`τ_k = 1/(D_h k²)` and **memory**; its Green's function is **screened** (Helmholtz/Yukawa), not
bare-Poisson.

**The distinction, made sharp.** The order parameter is the **storage `S`** (equivalently the
hydraulic time `τ_hyd = ℓ²/D_h`):

1. `H_φ(k,ω) → H_p(k)` **iff** `ω/D_h → 0` — either steady state (`ω=0`) or the singular
   no-storage limit `D_h → ∞` (`τ_hyd → 0`). Only then is `φ` a Leray-type instantaneous
   elliptic field. Real subglacial beds store water (`S>0`), so this limit is unphysical and `φ`
   is *always* distinct in practice.
2. At any finite `τ_hyd` and `ω>0`: `|H_φ| = 1/√(k⁴+(ω/D_h)²) < 1/k² = |H_p|` (**screening**),
   and the head **lags** the forcing by `φ_lag = arctan(ω/(D_h k²)) ≠ 0`, passing through **45°
   exactly at the mode's hydraulic clock** `ω_c = D_h k²` (the NR28 elliptic/parabolic crossover,
   here for the `p`/`φ` pair).
3. The Leray pressure carries **no intrinsic relaxation** (free-decay time 0 — slaved to its
   source); the hydraulic mode freely decays as `e^{−t/τ_k}`, `τ_k = 1/(D_h k²)`.

So `φ` is a genuine **dynamical field with storage and memory**; the Leray pressure is a
**memoryless kinematic multiplier**. The **falsifiable separator is field-measurable**: a nonzero
hydraulic response time / phase lag between a forcing (melt input, tidal or surface load) and the
hydraulic head.

### Numerical verification (`figures/nr30_phi_not_leray_pressure.{json,png}`)

Pure-operator test on a 2-D periodic bed patch (`n=64`, `L=2 km`, `D_h=1 m²/s`, in the repo's own
subglacial range `0.06–12.7 m²/s`, Paper 4 §3.5). Forcing set to the dominant-mode hydraulic
clock `ω_c = D_h k²_min` so the separation lands at the 45° crossover. All seven checks pass
(`ok = true`):

| check | result |
|---|---|
| transfer numeric vs closed form (rel-err) | `0.0` (both `H_p`, `H_φ`) |
| Leray phase / ω-dependence | max phase `0.0`, `‖H_p(ω)−H_p(5ω)‖ = 0` — **real, instantaneous** |
| hydraulic phase = `arctan(ω/D_h k²)` (rel-err) | `2.8×10⁻¹⁷`; dominant-mode lag `0.7854 rad` = **45° at ω_c** |
| screening `|H_φ| < |H_p|` + magnitude law | holds ∀ modes; mag rel-err `2.0×10⁻¹⁶` |
| coincidence limit (rel-gap as `D_h×{1,10,10²,10³,10⁴}`) | `0.707 → 0.0995 → 0.010 → 0.0010 → 1.0×10⁻⁴` (**monotone → 0**) |
| driven separator (time-domain lock-in) | head lag `0.7853` (closed `0.7854`); **Leray lag `−2.0×10⁻⁶` (none)** |
| memory time `τ_k` measured vs `1/(D_h k²)` | `1.0132×10⁵ s` vs `1.0132×10⁵ s` (`τ_hyd ≈ 1.17 d`) |

The driven test is the operational statement: a sinusoidal forcing produces a hydraulic head that
**lags by 45°** at the crossover frequency, while the Leray pressure stays exactly in phase. The
coincidence sweep shows the two operators merge **only** as `τ_hyd → 0`.

### Why it matters

- **It removes a stated open caveat from two manuscripts** (P4 §9, P4a §8) by proving `φ ≠ p`
  structurally — elliptic/instantaneous vs parabolic/finite-time — not by assertion.
- **The order parameter is named:** the storage `S` (hydraulic time `τ_hyd`). `φ` reduces to a
  Leray pressure only in the singular `S→0` limit, which a real water-storing bed never reaches.
- **It is falsifiable with field data:** a measured nonzero hydraulic response time, or a nonzero
  forcing→head phase lag (DInSAR / tidal-admittance / melt-pulse timing), confirms `φ` is the
  parabolic head and *not* the instantaneous Leray pressure. A genuinely zero lag at all
  frequencies would be the (unphysical) signature of a storage-free, Leray-type potential.
- It ties the subglacial `p`/`φ` pair into the same elliptic/parabolic two-clock structure that
  NR28 established for the core thesis, and reuses the linear-response phase lag of NR11.

**Mainstream tools (cited, not claimed):** the Leray–Hodge projection (Leray 1934; REPORT_THEORY
§6); Darcy / Röthlisberger subglacial hydrology (Röthlisberger 1972; Werder et al. 2013; Hewitt
2013); the screened (Helmholtz/Yukawa) vs bare-Poisson Green's function; the linear-response
phase lag `arctan(ωτ)` (NR11, NR28). The contribution is the identification of `φ` with the
parabolic screened operator, the storage `S` as the order parameter of its distinctness from the
Leray pressure, and the field-measurable response-time separator.
