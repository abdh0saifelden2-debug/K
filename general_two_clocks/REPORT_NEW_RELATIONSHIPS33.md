# New cross-relationship — NR56 (`general_two_clocks/new_relationships33.py`) — theory + real data

Continues the derived-and-verified program (NR1–NR55; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the real reads come from the committed cache
[`data/nr56_stokes_cache.json`](data/nr56_stokes_cache.json)
(8280 ANDRO deep floats: per-float zero-lag Q, U and lag-1 V; regenerate
with `build_cache()`). Unit-proofs in
[`tests/test_stokes_faces.py`](tests/test_stokes_faces.py) (11 tests).

```bash
python general_two_clocks/new_relationships33.py   # -> figures/91_stokes_faces.json
pytest general_two_clocks/tests/test_stokes_faces.py -v
```

---

## NR56 — **The Lagrangian velocity has Stokes parameters**: waves are linearly polarized, vortices circularly — and the deep ocean crosses from wave-polarized at the equator to vortex-polarized poleward  [NR52 × NR55 × rotary-spectra/polarimetry × the wave–vortex decomposition]

### The mining question

NR55 closed the *phase* anatomy. The 2×2 correlation tensor has one face
left: the deviatoric (anisotropy) part. What physics does it read, and
how does it partner with the spin?

### The relationship (derived + verified)

**1. Stokes decomposition.** The Lagrangian velocity correlation carries

```
I = ⟨u²+v²⟩            energy
Q = ⟨u²−v²⟩            stretch (zonal–meridional)
U = 2⟨uv⟩              stretch (diagonal tilt)
V = ⟨u(0)v(δt)−v(0)u(δt)⟩   spin (lag-odd; NR52)
```

— the Stokes parameters of a 2D oscillation. (Q,U) is the spin-2 face:
parity-even, rotating by 2θ under frame rotation (exact covariance
test); V is the parity-odd face. Under mirror: Q fixed, U and V flip
(machine precision). Oceanographic rotary spectra are the
frequency-resolved version.

**2. The polarization theorem.** A plane wave `ψ = cos(kx+ly−ωt)` gives
`u ∝ l·osc, v ∝ −k·osc` — the same oscillating factor: **rectilinear
polarization** along (−l,k), maximal (Q,U) with the orientation encoding
the wavevector, and *exactly zero spin* (V < 1e-12; angle exact; 2θ
covariance exact). A monopolar vortex orbits floats circularly: **V(δt)
= sin(fδt) exactly**, (Q,U) → 0. The stretch face reads wave content and
orientation; the spin face reads vortex content — a two-moment
Lagrangian wave/vortex decomposition, no track spectra needed.

### Findings (`figures/91_stokes_faces.json`)

* **The equator is wave-polarized.** Zonal stretch Q/I = **+0.44
  (t = +48)** in |lat| < 5° — the equatorial deep jets as the extreme
  rectilinear limit — while the spin there is smallest (0.054).
* **Poleward the ocean vortex-polarizes.** Q/I collapses to ~+0.04 by
  20–35° while |V| grows monotonically 0.054 → 0.077 → 0.091 → **0.107**
  (65–85°): the wave-to-vortex crossover sits at ~20–35°, measured from
  two single-float moments.
* **Polar flip.** At 65–85° the stretch turns **meridional**
  (Q/I = −0.032, t = −2.3): boundary-current/topographic steering
  replaces β as the anisotropy source.

### Consequence for the program

With NR55's four phases plus this polarization face, the full two-point
content of a 2D trajectory ensemble reads: energy (I), memory phase,
transport, stretch (Q,U), spin (V) — the two-clocks tensor is the
**coherency matrix** of the flow, and wave/vortex partition is
polarimetry. Protocol: two moments per float — no spectra, no maps, no
eddy detection. Combined with NR54, the polarization state even predicts
drift.

### Verdicts

| check | result |
|---|---|
| polarization theorems exact | **PASS** (wave V < 1e-12, angle exact, 2θ exact; vortex V = sin(fδt) to 1e-6; mirror exact) |
| equator wave-polarized | **PASS** (Q/I = +0.44, t = +48) |
| vortex polarization grows poleward | **PASS** (0.054 → 0.107, monotone ends) |
| wave/vortex crossover at 20–35° | **PASS** |
| polar meridional flip | **PASS** (Q/I = −0.032, t = −2.3) |

All references remain `[cite]` slots — none invented.
