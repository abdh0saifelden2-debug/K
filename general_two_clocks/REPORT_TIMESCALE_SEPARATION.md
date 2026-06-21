# Timescale separation = the singular-perturbation parameter

> Addresses two checklist titles at once, because they are the same statement:
> **#1 — "two clocks: timescale separation between elliptic pressure constraints
> and parabolic scalar transport"**, and **#4 — "a singular perturbation theory of
> mesoscale transport decoupling."**
> Code: [`timescale_separation.py`](timescale_separation.py);
> tests: [`tests/test_timescale_separation.py`](tests/test_timescale_separation.py)
> (3 tests). CPU, deterministic. Companion to
> [`REPORT_ELLIPTIC_PRESSURE.md`](REPORT_ELLIPTIC_PRESSURE.md) (space) and
> [`REPORT_MACH_REGULARITY.md`](REPORT_MACH_REGULARITY.md) / [`REPORT_THEORY.md`](REPORT_THEORY.md).

## The claim

The incompressible Boussinesq system runs two clocks with **separated** relaxation
times:

| clock | operator | relaxation time |
|---|---|---|
| **fast** — elliptic pressure constraint `∇²φ = ∇·u*` | Leray projection `ℙ = I − ∇Δ⁻¹∇·` | `τ_fast → 0` (instantaneous) |
| **slow** — parabolic scalar transport `∂ₜb = κΔb` | heat semigroup `e^{tκΔ}` | `τ_slow = 1/(κk²)` (finite) |

The separation `ε = τ_fast/τ_slow → 0` is not incidental: it **is** the
singular-perturbation small parameter of the incompressible limit. Regularising the
constraint with a finite sound speed `c_s` gives the pressure clock a finite
acoustic time `τ_fast = 1/(c_s k)`, and then

```
ε(k) = τ_fast / τ_slow = κ k / c_s   →  0   as  c_s → ∞   (Mach → 0).
```

Incompressible flow is the `ε → 0` **singular** limit of compressible flow; the
elliptic projection is the fast clock squeezed to zero relaxation time. "Mesoscale
transport decoupling" is the slow `κk²` transport decoupling from the
infinitely-fast constraint.

## Measured (all code-produced)

**Fast clock — instantaneous.** A smooth field with a genuine divergent part is
projected once:

```
‖∇·u‖ : 1.69  →  9.7×10⁻¹⁵   in ONE Leray projection   (drop ×1.7×10¹⁴)
```

The constraint is satisfied to machine precision in a single step, independent of
the time step — `τ_fast = 0`.

**Slow clock — finite κk².** Forward-integrating `∂ₜb = κΔb` (κ=0.02) and recovering
each mode's decay rate:

| k | τ_measured | τ_theory = 1/(κk²) |
|---|---|---|
| 1 | 50.00 | 50.00 |
| 2 | 12.50 | 12.50 |
| 4 | 3.12 | 3.12 |

Measured rate matches `κk²` to <0.1%; `τ_slow ∝ k⁻²` is finite and resolution-set.

**Singular limit — ε ∝ c_s⁻¹.** The linearised acoustic mode oscillates at `ω = c_s k`
(measured ratio 1.000), so `τ_fast = 1/(c_s k)` and:

| c_s | ω_measured | c_s·k | ε = τ_fast/τ_slow | κk/c_s |
|---|---|---|---|---|
| 5  | 5.00  | 5.00  | 0.0040 | 0.0040 |
| 10 | 10.00 | 10.00 | 0.0020 | 0.0020 |
| 20 | 20.00 | 20.00 | 0.0010 | 0.0010 |
| 40 | 40.00 | 40.00 | 0.0005 | 0.0005 |

Fitted `ε ∝ c_s^(−1.000)` — exactly the `c_s → ∞` (Mach → 0) singular scaling.

## Honest scope

This is a *demonstration* on pseudo-spectral solvers, not a new theorem. The
singular-perturbation structure of low-Mach flow is classical — Majda (1984), Klein
(1995); the rigorous incompressible limit of compressible Navier–Stokes is
Lions–Masmoudi (1998) and Schochet. What the module contributes is the **unified
measurement**: the same `ε = κk/c_s` is read off as (a) the instantaneous-vs-finite
clock-time ratio of #1 and (b) the singular-perturbation parameter of #4 — one number,
two titles. It does not claim anything about global regularity (that thread, and its
honest open core, live in `REPORT_CLAY_REGULARITY.md`).
