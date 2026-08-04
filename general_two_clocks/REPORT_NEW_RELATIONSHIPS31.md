# New cross-relationship — NR54 (`general_two_clocks/new_relationships31.py`) — theory + real data

Continues the derived-and-verified program (NR1–NR53; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, offline-safe — the real reads come from the committed cache
[`data/nr54_spin_drift_cache.json`](data/nr54_spin_drift_cache.json)
(8275 ANDRO deep floats: per-float spin, atlas-residual drift, latitude;
regenerate with `build_cache()` from the ANDRO pickle). Unit-proofs in
[`tests/test_spin_drift.py`](tests/test_spin_drift.py) (11 tests).

```bash
python general_two_clocks/new_relationships31.py   # -> figures/89_spin_drift.json
pytest general_two_clocks/tests/test_spin_drift.py -v
```

---

## NR54 — **The spin sets the drift compass**: β-drift read from single-float statistics — strong spinners drift westward; cyclonic spinners poleward, anticyclonic equatorward, at 700–1300 dbar  [NR52 × NR53 × β-plane vortex dynamics × the altimetric eddy-drift law]

### The mining question

NR52/53 established the spin as the parity-odd face and separated its
eddy clock from the circulation clock. Does the odd face *predict
transport*? A coherent vortex on the β-plane self-propagates — westward
at O(βR²), with cyclones drifting poleward and anticyclones equatorward
(the classic vortex–β interaction, observed at the surface by eddy
tracking in altimetry). A float trapped in a vortex inherits this
translation — so the float's spin statistic should forecast its drift.

### The relationship (derived + verified)

**The compass protocol.** For each float: spin `s` = lag-1 odd
correlation (NR52) and drift `(du, dv)` = mean velocity minus a 2°×2°
time-mean atlas. Predictions, in both hemispheres:

- `|s|` high → `du` more westward than `|s|` low (monotone in `|s|`);
- among strong spinners, poleward drift (`v·sign(lat)`) is larger for
  cyclonic spin (`s·sign(f) > 0`) than anticyclonic.

No eddy detection, no tracking, no fields beyond the mean atlas: the
parity-odd single-float statistic is the compass needle. (Synthetic
validation: prescribed spin–drift coupling recovered at |t| > 5 in both
hemispheres; shuffling spin labels kills it, |t| < 0.5.)

### Findings (`figures/89_spin_drift.json`)

* **Westward with spin, monotone.** u-drift by |spin| quintile (km/d):
  NH `+0.06, +0.05, +0.03, −0.05, −0.07` — top-vs-bottom shift
  **−0.135 km/d (t = −2.8)**; SH ends `+0.01 → −0.09`, shift
  **−0.099 km/d (t = −2.5)**.
* **Cyclones poleward, anticyclones equatorward.** Among top-quintile
  spinners: SH poleward(cyc) − poleward(anti) = **+0.24 km/d (t = 5.8)**;
  NH **+0.07 km/d (t = 1.3)**, same sign; **pooled +0.17 km/d
  (t = 5.2)** — the Morrow-type meridional splitting, at depth, from spin
  alone.
* **Magnitude.** O(0.1 km/d) — an order below bare βR_d² (~2–3 km/d), as
  expected: floats are only partially trapped, deep eddies are
  weaker/smaller than surface altimetric eddies, and quintile populations
  dilute the trapped fraction. The compass reads *direction* far more
  robustly than speed.

### Consequence for the program

The response anatomy's odd face is not just a stored signature — it
predicts transport. Spin (closed-orbit, parity-odd) and drift (open-path
translation) are locked by β-plane dynamics: the two-clocks tensor's
antisymmetric part doubles as a dynamical compass. The protocol exports
to any trajectory ensemble on a rotating sphere (drifters, balloons,
icebergs), and it recovers the altimetric eddy-drift law in a depth range
altimetry cannot see.

### Verdicts

| check | result |
|---|---|
| compass estimator validated (synthetic + null) | **PASS** (couplings at \|t\| > 5; shuffled null \|t\| < 0.5) |
| westward drift increases with spin | **PASS** (NH −0.135 km/d t = −2.8; SH −0.099 t = −2.5; monotone ends) |
| cyclones poleward / anticyclones equatorward | **PASS** (pooled +0.17 km/d, t = 5.2; SH alone t = 5.8) |
| magnitude order consistent | **PASS** (O(0.1 km/d), within the diluted β-drift window) |

All references remain `[cite]` slots — none invented.
