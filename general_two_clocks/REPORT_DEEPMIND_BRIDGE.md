# DeepMind unstable singularities ↔ this repo's NS-regularity take

**Module:** `deepmind_singularity_bridge.py` · **Tests:** `tests/test_deepmind_singularity_bridge.py` (7, all pass) · CPU-only, no data.

## What DeepMind actually found (Wang et al., arXiv:2509.14185, Sept 2025)

Google DeepMind + Brown/NYU/Stanford (Wang, Buckmaster, Gómez-Serrano, Lai, Georgiev, Jiang et al.) used **physics-informed neural networks (PINNs)** with a **high-precision Gauss–Newton optimizer** to make the *first systematic discovery of families of **unstable** self-similar singularities* for several fluid models: the **incompressible porous media (IPM)** equation, the **2D Boussinesq** equation, and the **3D Euler equation *with boundary*** (plus the 1D Córdoba–Córdoba–Fontelos model as a high-precision testbed).

Two points matter for us:

- **(D1) The singular solutions are *unstable*** — they "require initial conditions tuned with infinite precision," and "infinitesimal perturbations immediately divert the solution from its blow-up trajectory." This is the relevant regime precisely because **no *stable* singularities are expected for the boundary-free 3D Euler / Navier–Stokes** problem.
- **(D2) An empirical ladder** — the blow-up rate λ obeys a *reciprocal* law in the **order of instability** *n* (number of unstable directions of the renormalized self-similar operator): λₙ ≈ 1/(1.4187·n + 1.0863) + 1 for Boussinesq/Euler, λₙ ≈ 1/(1.1459·n + 0.9723) for IPM-with-boundary. Clearest for IPM and Boussinesq.

**Honest scope.** Near-double-float precision was reached for *specific* solutions (CCF stable + 1st-unstable) — high enough to *attempt* computer-assisted proofs, but **none of the new candidates is proved yet**, and the work **does not treat boundary-free 3D Euler / Navier–Stokes**, the actual Clay target.

## What the bridge computes (the verifiable connection)

DeepMind's organizing object is the renormalized self-similar operator and its **instability order**. We compute exactly that object for this repo's restricted-Euler (Vieillefosse) caricature:

Restricted-Euler invariants Q = −½ tr(A²), R = −⅓ tr(A³) obey Q′=−3R, R′=⅔Q². With the self-similar ansatz Q=(t\*−t)⁻²q, R=(t\*−t)⁻³r, s=−ln(t\*−t), the renormalized system is

> q′ = −2q − 3r,  r′ = ⅔q² − 3r,

with nontrivial fixed point **(q\*, r\*) = (−3, 2)** = the Vieillefosse profile. Its Jacobian is [[−2,−3],[−4,−3]] (trace −5, det −6), eigenvalues **{−6, +1}**:

- the **−6** mode is contraction onto the self-similar Vieillefosse *shape* (generic data is pulled onto it);
- the **+1** mode is the **universal blow-up-time (time-translation) freedom** every self-similar blow-up carries — under s=−ln(t\*−t) a shift of t\* grows like eˢ, i.e. eigenvalue +1. DeepMind quotients exactly this mode out.

**⇒ Modulo the trivial time-shift mode, the restricted-Euler self-similar blow-up has instability order 0** — it is DeepMind's *stable* singularity type (the kind classical numerics find readily), **not** a rung of their new order-≥1 unstable ladder. A fixed-exponent quadratic VGT model cannot contain that ladder (its scaling exponent is fixed by homogeneity, so there is no λ-ladder).

## Verdict: does it benefit our take?

**Yes, conceptually and methodologically — but it does not close anything.**

- **[+] Conceptual support.** The repo's open core conjectures that blow-up, if any, is a non-generic, "conspiratorial" event. DeepMind's independent, proof-grade evidence that the singular solutions are *unstable / infinitely fine-tuned* is the same statement from the blow-up side: the depletion geometry is generic; defeating it is not.
- **[+] Sharpened target.** The nonlocal pressure Hessian must deplete not merely the single stable Vieillefosse profile (which the VGT model captures) but the **whole ladder** of unstable self-similar profiles DeepMind exhibits — a concrete, harder open core.
- **[+] A method.** The PINN + high-precision Gauss–Newton + computer-assisted-proof playbook is exactly the tooling our depletion inequality (`REPORT_CLAY_REGULARITY.md` §6) would need to become a rigorous bound.
- **[−] No direct NS gain.** IPM / Boussinesq / Euler-with-boundary ≠ boundary-free 3D Navier–Stokes; nothing here closes our open core, and DeepMind's candidates are not yet proved.
- **[−] Caricature limit.** The VGT model reproduces the *stable* profile and the "blow-up = renormalized fixed point" framing, but not the unstable ladder or the λ-vs-order law.
