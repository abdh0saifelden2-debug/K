# GETTING_STARTED — plugging the two-clocks closure into a subglacial channel model

**Audience:** a glacier modeller who already has a GlaDS- or Röthlisberger-style
subglacial hydrology model and wants to know what the `general_two_clocks/closure/`
benchmark actually *does to a prediction* — concretely, to the post-drainage
**surge lag**.

This file does exactly three things:

1. [Import the closure](#1-import-the-closure) from `general_two_clocks/closure/`.
2. [Swap it into a GlaDS / Röthlisberger channel model](#2-swap-it-into-a-glads--röthlisberger-channel-model) (pseudocode).
3. [See what changes in a real surge-lag prediction](#3-what-changes-in-a-real-surge-lag-prediction) (runnable).

If you read nothing else, read this one sentence, which is the repository's whole
thesis applied to ice:

> **K-theory is the memoryless, down-gradient, no-backscatter truncation of the
> Mori–Zwanzig / projected-FDT closure** (`general_two_clocks/README.md` §3). A
> subglacial surge lag *is* a memory effect. Delete the memory and the model
> predicts the lag away.

---

## Background in 60 seconds

The closure module is a frozen-field **a-priori** benchmark: it builds a 2-D
incompressible turbulence truth field, sharp-filters it at a cutoff `kc`, forms
the **exact** subgrid force `m_true`, and scores three models of it
(`general_two_clocks/REPORT_CLOSURE.md`, `figures/28–30`).

| model | eddy viscosity | backscatter `T(k)>0` | divergence-free | transfer corr w/ truth |
|---|---|---|---|---|
| **Smagorinsky (K-theory)** | `ν_t = (Cs Δ)² |S| ≥ 0` (positive-definite) | **no** (`T(k) ≤ 0` everywhere) | yes | **0.071** |
| spectrum-matched surrogate | — (phase-randomised) | partial | **no** (`div ≈ 1.2e+01`) | 0.907 |
| **projected-FDT (two clocks)** | `ν_t(k)` scale-dependent, **may go negative** | **yes** | yes (`div ≈ 1.7e-14`) | **1.000** |

The one structural fact you will reuse below: `projected_fdt_force` returns a
per-shell eddy viscosity `nu_t_shell` whose entries are **allowed to be negative**
(energy flowing *up*-scale = backscatter = the system's *memory* of the resolved
field), while Smagorinsky's `ν_t` is sign-definite and instantaneous. Verify it
yourself in [step 1](#1-import-the-closure).

The bridge from that 2-D turbulence statement to a 1-D channel surge is the
Mori–Zwanzig (MZ) projection of the lumped cavity↔channel hydrology, certified to
machine precision in `validation/synthetic/hydraulic_mz_projection_synthetic.py`
(RESULT 21, `REPORT_HYDRAULIC_MZ_PROJECTION.md`). Eliminating the channel variable
from the coupled store↔drainage system leaves a **generalized Langevin equation**

```
ṡ(t) = M_ss s(t) + ∫₀ᵗ K(t−τ) s(τ) dτ + R(t),     K(τ) = M_sq M_qs e^{M_qq τ}
```

`K(τ)` is the eliminated channel's own Green's function. Keeping it = the
projected-FDT clock (memory). Dropping it (adiabatic elimination, `q` slaved
instantly to `p_w`) = the K-theory clock (Markovian, no memory). The surge lag
lives entirely in `K`.

---

## 1. Import the closure

`closure` and its dependency `compressible` are top-level packages **inside**
`general_two_clocks/`, so put that directory on the path. (`pytest` already does
this via `pyproject.toml`'s `pythonpath`; for a standalone script do it by hand.)

```python
import sys, pathlib
import numpy as np

REPO = pathlib.Path("/path/to/K")                      # repo root
sys.path.insert(0, str(REPO / "general_two_clocks"))   # makes `closure`, `compressible` importable

from closure.dns2d import Vorticity2D
from closure.sgs import (
    exact_sgs_force,          # m_true = P[(ū·∇)ū − filt((u·∇)u)]
    smagorinsky_force,        # the K-theory closure (positive-definite ν_t)
    projected_fdt_force,      # the two-clocks closure (ν_t(k) may go negative)
    transfer_spectrum,        # T(k) = Re⟨û*·m̂⟩ : <0 dissipation, >0 backscatter
)

# manufacture a turbulence truth field and form the exact subgrid force
dns = Vorticity2D(n=128, seed=42)
u, v = dns.field(steps=1500)
sp, kc = dns.sp, 16
ub, vb, mx, my = exact_sgs_force(sp, u, v, kc)

# the two closures
mx_K,  my_K          = smagorinsky_force(sp, ub, vb, kc)              # K-theory
mx_F,  my_F, nu_shell = projected_fdt_force(sp, ub, vb, mx, my, kc)   # projected-FDT

_, T_true = transfer_spectrum(sp, ub, vb, mx,   my,   kc)
_, T_K    = transfer_spectrum(sp, ub, vb, mx_K, my_K, kc)

print("projected-FDT ν_t(k) has negative shells (backscatter)?",
      bool((nu_shell < 0).any()))                 # -> True  (memory: energy goes up-scale)
print("K-theory T(k) ≤ 0 everywhere (no backscatter)?",
      bool(np.all(T_K[1:] <= 1e-12)))             # -> True  (purely dissipative)
print("truth has backscatter T(k) > 0 somewhere?",
      bool(np.any(T_true[1:] > 0)))               # -> True
```

Running this prints `True / True / True`: the projected-FDT closure carries
backscatter that K-theory structurally cannot. That asymmetry is the entire story
for the surge lag.

---

## 2. Swap it into a GlaDS / Röthlisberger channel model

Here is a lumped GlaDS/Röthlisberger model with the closure made explicit. The
state is `x = (p_w, q)`: water pressure `p_w` (the resolved englacial **store**,
charge time `τ₁`) coupled to a drainage element `q` (channel cross-section `S`, or
linked-cavity opening `h_s`; opening time `τ₂`). `N = p_i − p_w` is effective
pressure. See `validation/synthetic/hydraulic_lag_derivation.py` for the
fully-named-constant version.

```python
# ----- the channel/cavity model (lumped GlaDS continuity + Röthlisberger opening) -----
#   Σ dp_w/dt = Q_in − Q_out(p_w, q)                         # store  (τ₁ = R·C)
#     dq/dt   = OPEN(p_w, q) − CLOSE(N)                      # drainage element (τ₂)
#
# OPEN(...) is the turbulent melt-opening term.  For a Röthlisberger channel
#   OPEN = Ξ/(ρ_i L),   Ξ = (turbulent dissipation in the channel)  ~ | Q · dφ/ds |
# Its basal heat flux Ξ is a SUBGRID quantity — and *that* is the closure slot
# `general_two_clocks/closure/` fills.  Two ways to fill it:
```

### (a) The closure you almost certainly have now — K-theory (Markovian)

```python
def opening_flux__Ktheory(p_w, q):
    # down-gradient, positive-definite eddy viscosity; instantaneous function of NOW
    nu_t = (Cs * Delta)**2 * abs(strain_rate(p_w, q))        # smagorinsky_force(): ν_t ≥ 0
    return down_gradient_dissipation(nu_t, p_w, q)           # no dependence on the past

# Because the flux is memoryless, the fast drainage variable q can be slaved to p_w
# (adiabatic elimination):  q ≈ −M_qq⁻¹ M_qs p_w.  The store then obeys a FIRST-ORDER
# equation with NO memory integral:
#     ṗ_w = (M_ss + ∫K dτ) p_w        with the kernel collapsed to (∫K)·δ(τ)
```

### (b) The swap — projected-FDT (keep the memory kernel)

```python
from closure.sgs import projected_fdt_force        # the two-clocks closure

def opening_flux__projectedFDT(p_w, q, history):
    # scale-dependent ν_t(k) ALLOWED TO GO NEGATIVE (backscatter) + projected noise.
    # In the lumped model this is exactly "do NOT slave q": retain the channel's
    # own relaxation as a Mori–Zwanzig memory kernel.
    _, _, nu_t_shell = projected_fdt_force(sp, ub, vb, m_true, kc)   # has negative shells
    return dissipation_with_memory(nu_t_shell, p_w, q, history)      # depends on the PAST

# Keeping the memory means integrating the GLE instead of the slaved first-order law:
#     ṗ_w = M_ss p_w + ∫₀ᵗ K(t−τ) p_w(τ) dτ + R(t)
#     K(τ) = M_sq M_qs e^{M_qq τ}      # the eliminated drainage element's Green's function
```

The substitution is one line at the call site:

```python
# OPEN = opening_flux__Ktheory(p_w, q)            # before: memoryless, slaved q
  OPEN = opening_flux__projectedFDT(p_w, q, hist) # after:  memory kernel retained
```

**This swap already exists, executable, in two places in this repo** — you do not
have to wire the pseudocode yourself to reproduce the effect:

- **Turbulence-closure layer (where Ξ comes from).**
  `subglacial/candidate4_hydraulic_switch.py` carries the closure as a config knob,
  `sgs ∈ {"none", "smagorinsky", "backscatter"}` (see `_sgs_force`). `"smagorinsky"`
  is K-theory; `"backscatter"` adds the FDT-linked up-scale injection. The
  closure-sensitive observable is `turb_heat_flux` (`⟨v′θ′⟩`) — the flow-dependent
  basal flux that sets melt, as opposed to the conduction-pinned interfacial read.
- **Lumped surge layer (where the lag comes from).**
  `validation/synthetic/hydraulic_lag_derivation.py` (Jacobian → `τ₁,τ₂,a,b`) and
  `hydraulic_mz_projection_synthetic.py` (memory vs. Markovian) are the closed
  cavity↔channel system. Step 3 drives them directly.

> **Honest caveat on "Röthlisberger channel."** The *literal* R-channel drainage
> element (`q = S`, melt-opening vs. Nye creep) is linearly **unstable** for
> ice-stream parameters — `trace(J) > 0`, the Kingslake-2015 lake-oscillation
> regime — so it has *no* decaying peaked kernel and cannot be the source of a
> single rise-and-decay surge (`hydraulic_lag_derivation.channel_regime`). The
> stable, peaked surge lag is **cavity-opening-paced** (`q = h_s`). The closure
> swap is identical in form for both; the cavity element is the one that produces
> the observed lag.

---

## 3. What changes in a real surge-lag prediction

A subglacial lake drains and dumps water into the store at `t = 0`. The basal
sliding speed tracks the drainage-element opening, so the **observed surge lag is
the argmax of the downstream (opening) response.** The only thing that changes
between the two predictions is the closure — memory vs. no memory.

Save as `surge_lag_demo.py` and run from anywhere (set `REPO`):

```python
import sys, pathlib, numpy as np
REPO = pathlib.Path("/path/to/K")
sys.path.insert(0, str(REPO / "glaciers" / "validation" / "synthetic"))

import hydraulic_lag_derivation as hld     # named-constant GlaDS/Röthlisberger Jacobian
import hydraulic_kernel_synthetic as hks   # coupled cavity↔channel impulse response

p   = hld.baseline_params()                # West-Antarctic ice-stream lake catchment
lag = hld.derive_lag(p)                    # reads τ₁, τ₂, a, b off the *physical* Jacobian
M   = hld.jacobian_cavity(p) * hld.SEC_PER_YR     # per-year units
t   = np.linspace(0, 1.0, 200_001)

# (1) projected-FDT closure  → keep the MZ memory kernel → full coupled response
G_mem, _ = hks.coupled_response(t, lag["tau1_yr"], lag["tau2_yr"], a=-M[0,1], b=M[1,0])
k_mem = int(np.argmax(G_mem))

# (2) K-theory closure       → slave q to p_w (adiabatic) → first-order, memoryless
Mss, Msq, Mqs, Mqq = M[0,0], M[0,1], M[1,0], M[1,1]
r_eff   = Mss + Msq*Mqs/(-Mqq)
q_slave = (Mqs/(-Mqq)) * np.exp(r_eff*t)
k_mark  = int(np.argmax(np.abs(q_slave)))

print(f"DERIVED clocks  : τ₁(store)={lag['tau1_yr']:.3f} yr  τ₂(cavity)={lag['tau2_yr']:.3f} yr")
print(f"(1) projected-FDT (memory) : surge PEAKS at t* = {t[k_mem]*365.25:.2f} d  interior={0<k_mem<len(t)-1}")
print(f"(2) K-theory (Markovian)   : argmax idx = {k_mark}  → peak at t=0 (NO lag, monotone)")
print(f"analytic DERIVED t*        : {lag['tstar_yr']*365.25:.2f} d ({lag['tstar_yr']:.4f} yr)")
print(f"observed band              : {hld.OBS_BAND_YR} yr  (Stearns 2008; Siegfried 2016)")
```

Output:

```
DERIVED clocks  : τ₁(store)=0.054 yr  τ₂(cavity)=0.037 yr
(1) projected-FDT (memory) : surge PEAKS at t* = 3.90 d  interior=True
(2) K-theory (Markovian)   : argmax idx = 0  → peak at t=0 (NO lag, monotone)
analytic DERIVED t*        : 3.90 d (0.0107 yr)
observed band              : (0.02, 2.0) yr  (Stearns 2008; Siegfried 2016)
```

**The prediction that changes:**

| | K-theory closure (no backscatter / no memory) | projected-FDT closure (backscatter / memory) |
|---|---|---|
| post-drainage response | **monotone, argmax at `t = 0`** | rises from 0 to an **interior peak** |
| predicted surge lag | **none** (instantaneous speed-up) | `t* ≈ 0.011 yr ≈ 3.9 d` (baseline) |
| literature parameter sweep | always "no lag" | median `t* ≈ 0.012 yr`, ~**34 %** of the literature-plausible space lands in the observed **0.02–2 yr** band |
| survives full nonlinear ODE? | n/a | yes — nonlinear peak `3.9–4.7 d` vs. linear `3.9 d` across impulse amplitudes `0.02–0.4 N` |

So a K-theory subglacial model, asked "how long after the lake drains does the ice
speed up?", answers **"immediately"** — it has thrown away the very memory term the
lag is made of. Swapping in the two-clocks closure restores the kernel and the
model predicts a *delayed* surge whose timescale is set by the cavity-opening clock
`τ₂` and only logarithmically by the store-charge clock `τ₁`
(`t* ≈ τ₂·ln(τ₁/τ₂)`), landing in the observed band.

Why the kernel is the *right* object, not a fit (RESULT 21, machine-precision):

```
cd glaciers/validation/synthetic && python hydraulic_mz_projection_synthetic.py
# (A) projection exact (Laplace):           max abs err 5.0e-16
# (B) kernel == channel Green's fn:          err 6.9e-18, decays at exactly 1/τ₂
# (C) reduced GLE reproduces full trajectory: rel-err 9.1e-08
# (D) memory ⇒ peak; Markovian ⇒ monotone, argmax at t=0
# (E) τ₂→0 collapses K(τ) → (∫K)·δ(τ): the K-theory local limit, *derived* not assumed
```

---

## Reproduce

```bash
pip install numpy scipy matplotlib                         # the only deps these scripts need

# step 1 — the closure benchmark (figures 28–30 + REPORT_CLOSURE.md)
python general_two_clocks/run_closure.py --out-dir general_two_clocks/figures

# step 3 — the surge-lag derivation and the memory-vs-Markovian projection
python glaciers/validation/synthetic/hydraulic_lag_derivation.py
python glaciers/validation/synthetic/hydraulic_kernel_synthetic.py
python glaciers/validation/synthetic/hydraulic_mz_projection_synthetic.py

# the in-repo closure swap at the turbulence layer (CPU smoke test)
python glaciers/subglacial/candidate4_hydraulic_switch.py

pytest glaciers/tests/test_validation_synthetic.py -q      # the synthetic checks above
```

## Honest scope (read before quoting a number)

- The closure benchmark (`REPORT_CLOSURE.md`) is a **frozen-field a-priori** 2-D
  test: it certifies *structural* correctness at one instant (spectrum, div = 0,
  transfer sign incl. backscatter). It is not a time-integrated ice-sheet run.
- The surge-lag value `t*` is **[DERIVED]**, order-of-magnitude — read off named
  GlaDS/Röthlisberger/cavity constants at literature values, **not** fit to drainage
  dates. An end-to-end observed-vs-predicted lag is **USAP-DC-gated** (§H.2;
  `validation/external/lag_fit_real.py`, `run_usapdc_lakes.py`) and is **[HYP]** as
  a *field mechanism*.
- What *is* exact is the **structure**: the §G.4 hydraulic term is a genuine
  Mori–Zwanzig memory kernel (RESULT 21), and K-theory is its zero-memory limit.
  The claim this file demonstrates is "K-theory cannot represent a surge lag,
  because a surge lag is a memory effect," not "the lag is exactly 3.9 days."
- The literal Röthlisberger **channel** element is unstable (no peaked kernel); the
  stable peaked surge is **cavity-opening-paced**. Use `q = h_s` for the lag.

## Where to look next

- `general_two_clocks/REPORT_CLOSURE.md` — the three-diagnostic benchmark + the
  "subglacial cavity flow" physical-mapping section.
- `general_two_clocks/REPORT_THEORY.md` — K-theory as the memoryless/no-backscatter
  truncation of the projected-FDT/MZ closure.
- `glaciers/REPORT_HYDRAULIC_MZ_PROJECTION.md` — RESULT 21, the exact projection.
- `glaciers/validation/synthetic/hydraulic_lag_derivation.py` — the named-constant
  Jacobian, the channel-vs-cavity regime split, and the literature sweep.
- `glaciers/subglacial/candidate4_hydraulic_switch.py` — the closure as a runnable
  `sgs` knob inside a cavity solver.
- `FUTURE_WORK.md` §G.4 / §H.2 — the master claim-status ledger for the lag.
