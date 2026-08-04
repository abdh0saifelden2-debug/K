# New cross-relationship — NR34 (`general_two_clocks/new_relationships11.py`)

Continues the derived-and-verified program (NR1–NR33; see
[`papers/CROSS_RELATIONSHIPS_INDEX.md`](../papers/CROSS_RELATIONSHIPS_INDEX.md)).
CPU-only, deterministic; unit-proofs in
[`tests/test_new_relationships11.py`](tests/test_new_relationships11.py) (13 tests). Run:

```bash
python general_two_clocks/new_relationships11.py   # -> figures/nr34_fractional_ice_kernel.{json,png}
pytest general_two_clocks/tests/test_new_relationships11.py -v
```

---

## NR34 — The ice kernel **is** a tempered half-order fractional derivative: Warburg interface, Mittag-Leffler relaxation, and the `Ste²` visibility window  [B.2 exact kernel × tempered fractional calculus (Meerschaert & Sabzikar 2014) × Warburg 1899/Randles 1947 × Mainardi/Metzler–Klafter fractional relaxation; executes horizon-ledger E2]

### The gap this closes

P4a §B.2's interface memory kernel (`glaciers/validation/synthetic/ice_kernel_synthetic.py`,
PDE-validated at §V.4) has been used as a *numerical* convolution object: a `t^{−1/2}`
tail with an `exp(−t/4τ_d)` cutoff and a certified DC gain. What family of operators it
belongs to — and when its fractional character is actually *observable* in the coupled
melt problem — was open. Both close in closed form.

### The statements

**(1) Exact operator identity (not asymptotic).** With `A = k_th θ_far V̄²/(2κ²) < 0`,
`τ_d = κ/V̄²`, `W ≡ −2A√τ_d > 0`, and `λ ≡ 1/(4τ_d)`:

    H(s) = A(1 − √(1+4τ_d s))/s  ≡  A/s + W·√(s+λ)/s        (all s)

because `√(1+4τ_d s) = 2√τ_d·√(s+λ)` identically. `√(s+λ)` is the Laplace symbol of the
**exponentially tempered half-derivative**; the kernel's `exp(−t/4τ_d)` cutoff *is* the
tempering factor, with tempering rate exactly `1/(4τ_d)`. Verified to `2.8×10⁻¹⁴` over a
complex-`s` grid, dimensional parameters included.

**(2) Warburg/Caputo-½ limit with the classic coefficient.** For `|s|τ_d ≫ 1`,
`H(s) → W s^{−1/2}` (relative error `½(sτ_d)^{−1/2}`, rate unit-proved), i.e.
`q_ice = W·I^{1/2}v` — the flux is the half-order fractional **integral** of interface
velocity (equivalently the half-*derivative* of interface displacement; the horizon
ledger's `d^{1/2}v/dt^{1/2}` phrasing is corrected here). On the imaginary axis the
phase → **−45° exactly**: the ice side of the interface is a **Warburg element**, the
constant-phase workhorse of electrochemical impedance — the E3 (bed-as-Randles-circuit)
contact point. The coefficient closes a loop with the textbook half-space law:
`W ≡ (k_th/√κ)·|θ̄′(0)|` identically, where `θ̄′(0) = θ_far V̄/κ` is the advected base
gradient (50-random-parameter unit proof).

**(3) The coupled interface relaxes as a Mittag-Leffler function — in closed form.**
Closing the loop with the Stefan condition `ρ_i L_f v = q_w − G∗v` gives, in the Warburg
window, the fractional (Langevin-type) equation

    ρ_i L_f v(t) + W·I^{1/2}v(t) = q_w(t)

whose step response is exactly `v(t) = v₀·E_{1/2}(−√(t/τ_f)) = v₀·e^{t/τ_f}erfc(√(t/τ_f))`
with `τ_f = (ρ_i L_f/W)²` — machine-evaluable as `scipy.special.erfcx`, replacing
convolution numerics outright. Asymptotics: `1 − 2√(t/πτ_f)` short-time; heavy
**algebraic tail** `√(τ_f/πt)` long-time (no exponential stage — the fractional-relaxation
signature of Mainardi / Metzler & Klafter 2000). Proved by quadrature to solve the
fractional integral equation and against a product-integration Volterra solver for the
**full** kernel (solver itself validated to `6×10⁻⁴` against the exact `E_{1/2}` answer on
the pure-Warburg kernel).

**(4) One dimensionless group decides whether an interface is fractional or classical.**

    τ_f / τ_d = Ste⁻²,        Ste = c|θ_far|/L_f

(algebraic identity, random-parameter unit proof). The Mittag-Leffler stage is visible
only if it fits inside the Warburg window (`τ_f ≪ τ_d`), i.e. **iff `Ste ≫ 1`** — and the
full-kernel deviation from the fractional limit falls as `O(1/Ste)` (measured: max rel
err 0.250 / 0.124 / 0.062 at Ste = 5/10/20 — the predicted halving). At glacial
subcooling (`Ste ≈ 6.3×10⁻³` per kelvin), `τ_f ≈ 2.5×10⁴ τ_d`: tempering cuts the
fractional stage off and the coupled response collapses onto the quasi-steady limit
`v_∞ = v₀/(1+Ste)` fixed by the DC gain `H(0) = −ρcθ_far` (verified to `2×10⁻⁷`), moving
only `O(Ste)` from `v₀` with a brief `√t` transient. **Glacial interfaces are classical;
strongly subcooled/dissolution-type interfaces (`Ste ≳ 5`) are measurably fractional.**
A falsifiable design rule for where half-order interface dynamics can be observed at all.

### Why this is one of ours

The two-clocks program's central object is a memory kernel produced by eliminating a fast
diffusive field. NR34 shows the flagship instance sits exactly at the junction of three
mainstream formalisms — tempered fractional calculus (the cutoff = tempering), EIS (the
Warburg element; the E3 bridge), and fractional relaxation (Mittag-Leffler, here in its
rare fully-elementary `erfcx` case) — and extracts a *new* portable number from the
junction: the `Ste²` window. NR12's clock inversion gains the analytic transfer function
`T(s) = 1/(1 + H(s)/ρLf)` (CPE form `1/(1+(τ_f s)^{−1/2})` in-window) with all constants
identified.

### Honest scope

Linearised kernel (perturbations about mean ablation `V̄`); the Warburg identification
of specific systems inherits B.2's assumptions (semi-infinite ice, interface pinned at
`T_melt`). `Ste ≫ 1` is not reachable in temperate glacial settings — the observable
targets are lab ablation/dissolution analogues and the *formal* structure (impedance
fits, E3) where the Warburg element enters regardless of Ste. The Volterra solver is
first-order accurate at the singular endpoint (product integration); all quantitative
claims are made at resolutions where its validated error (`6×10⁻⁴`) is negligible.

### Artifacts

`figures/nr34_fractional_ice_kernel.{json,png}`; 13 unit proofs
(`tests/test_new_relationships11.py`): tempered identity ×2 (unit + dimensional),
classic-coefficient identity, Warburg error rate, −45° phase, `E_{1/2}` solves the FLE,
`E_{1/2}` series check, ML asymptotics + non-exponential tail, solver-vs-exact,
`1/Ste` convergence, glacial quasi-steady collapse, `Ste⁻²` window law, DC-gain limit.
