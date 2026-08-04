# New cross-relationship — NR55 (`general_two_clocks/new_relationships32.py`) — theory capstone

Closes the NR47–NR54 arc (see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — reads only the committed NR47–NR54 figure
artifacts. Unit-proofs in
[`tests/test_response_anatomy.py`](tests/test_response_anatomy.py) (8 tests).

```bash
python general_two_clocks/new_relationships32.py   # -> figures/90_response_anatomy.json
pytest general_two_clocks/tests/test_response_anatomy.py -v
```

---

## NR55 — **The response anatomy is complete**: four independent phases — memory, transport, shape parity, handedness — four guard theorems, and the corpus's systems occupy distinct corners with every zero theorem-enforced  [capstone of NR47 × NR48 × NR49 × NR50 × NR51 × NR52 × NR53 × NR54]

### The four phases and their guards

| # | phase | odd under | guard theorem (the blindness that isolates it) |
|---|---|---|---|
| P1 | memory `δ = arg χ_min` | frequency | **minimum-phase** (NR48): δ is *determined* by \|χ\| (Bode); integrally conserved (waterbed, NR50) |
| P2 | transport `−ωτ_d` / boost `−kVΔt` | — (all-pass) | **all-pass factorisation** (NR48/49): the only phase with *no amplitude signature*; gain can never fake it |
| P3 | shape parity `arg B` (closed triads) | parity | **polygon closure** (NR51): exactly boost-blind — kinematics cannot reach it; only nonlinear asymmetric coupling writes into it |
| P4 | handedness (spin, tensor-odd) | parity *and* lag | **scalar contraction** (NR52): every scalar observable (MSD, GSER, \|χ\|) is chirality-blind |

Pair open phase = P1+P2; pair tensor-odd = P4; closed triad = P3; higher
polygons repeat P3's parity content — **no fifth phase** at the two- and
three-point level.

### The classification matrix (every value from a committed artifact)

| system | P1 memory | P2 transport | P3 shape | P4 spin |
|---|---|---|---|---|
| EEG (NR39/47) | **~0** (2.1°) — elliptic pole, theorem-enforced | — | — | — |
| ocean (NR45/48/50/52–54) | **~90°** parabolic; min-phase to 3.0°; area budget 6% | 0 (GSER closes with no delay) | — | **10σ**, hemisphere-mirrored; eddy clock 49–61 d; β-drift compass t=5.2 |
| solar (NR46/48/50) | ~0 — gain tilt vetoed ×3.5 as *static weighting* | **τ_d = −10.3 s** pure all-pass delay clock | — | — |
| scallops (Paper 3 / NR49/51) | σ ≈ 0 (marginal pattern) | **48 mm/hr**, 93% boost-removable | weak (t≈2.4) vs flux parity (t=−4.7) | — |
| grid (NR42) | band-limited memory window | — | — | — |

Every dash and every ~0 is enforced by the matching guard theorem; every
nonzero is a measured, tested number. The corpus realizes all four
phases, each in the system where its physics lives.

### The protocol (for any new system)

1. Measure `|χ(ω)|` → P1 comes free (Bode transform; check with the
   phase-area budget).
2. Fit the excess-phase slope → P2 (delay/boost), model-free.
3. Close a spectral triad → P3 (parity of the stored shape).
4. Antisymmetrize the correlation tensor → P4 (handedness, and via NR54
   its drift compass).

Four numbers, four theorems, no redundancy.

### Verdicts

| check | result |
|---|---|
| EEG at elliptic pole | **PASS** (2.1°) |
| ocean parabolic + odd | **PASS** (88°; spin −9.7/+13.2/+1.0; compass 5.2) |
| solar pure transport | **PASS** (−10.3 s; veto ×3.5) |
| scallops transport + weak shape parity | **PASS** (48 mm/hr; t 2.4 vs −4.7) |
| four corners distinct | **PASS** |

Cross-face consistency pinned by tests: NR48's τ_d = NR50's independent
fit; NR52's spin signs = NR53's tropical chirality = NR54's compass
hemispheres.

All references remain `[cite]` slots — none invented.
