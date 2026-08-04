# Part 6 -- Synthesis: the two clocks reunified by projection

The two clocks of the manuscript -- a slow, local, parabolic *temperature/advection* clock and a fast, global, elliptic *pressure* clock -- are reunited in one nonlinear flow using the **projection (fractional-step) method**, the canonical incompressible algorithm. The key point: this algorithm is not an arbitrary numerical trick, it is the two-clocks structure made literal.

## The unified update

```
u*       = u^n + dt * [ -(u.grad)u + nu*lap(u) + (b-<b>) e_y ]   (slow drift)
u^{n+1}  = P u*                                                  (fast projection)
b^{n+1}  = b^n + dt * [ -(u.grad)b + kappa*lap(b) ]              (temperature clock)
```

The drift is the **deterministic generator of the local Markov / heat semigroup** (its heat kernel is the Gaussian transition density) -- the *perfect* localized, memoryless forecaster, with no machine-learning or statistical noise to muddy the argument. The Leray projector `P = I - grad (lap)^-1 div` is exactly the divergence-free part of the Helmholtz split (`compressible/ns.py`).

## 1. The mesoscale convection cycle (figure 22)

From warm buoyancy blobs the flow reproduces the full mesoscale cell: heat diffuses, buoyant fluid lifts as plumes, the pressure field drives cooler fluid inward to fill the space (entrainment), and the shear between rising and inflowing streams rolls up into vortices.

## 2. The slow drift is divergent; one Poisson solve fixes it (figure 23)

The intermediate velocity `u*` from the slow drift alone does **not** conserve mass: RMS div(u*) = 2.75e-02. This is the blind spot of any purely local, memoryless model -- it moves fluid based on local heat and history without knowing the global incompressibility constraint. A single elliptic Leray projection (the fast clock, which instantly 'feels' the whole domain) drops the divergence to machine zero: RMS div(P u*) = 2.3e-15. The projection potential `phi` (lap phi = div u*) is the global elliptic pressure that does the work.

## 3. The SPDE limit -- energy yes, structure no (figure 24)

Modern stochastic-climate / SPDE closures replace the unresolved fast clock with random forcing tuned to the right *spectrum*. We mimic this by replacing the deterministic correction (the gradient field `P` removes) with a phase-randomized surrogate of **identical power spectrum**. The result is decisive:

- **Spectrum:** identical by construction (Parseval) -- the surrogate carries exactly the right energy at every wavenumber.
- **Structure:** pointwise correlation with the true correction is 0.012 ~ 0 -- the field is geometrically unrelated.
- **Constraint:** the deterministic projection gives RMS div = 2.3e-15 (= 0); the spectrum-matched surrogate leaves RMS div = 3.76e-02 -- it does **not** enforce incompressibility.
- **Ensemble (not an anecdote):** over 100 independent phase-randomized surrogates, |corr| has median 0.0697 and 95th percentile 0.1981; residual RMS div is 3.69e-02 [95% 3.48e-02, 3.84e-02] (`figures/24_spde_limit_ensemble.json`).

Matching a spectrum is not the same as capturing the physics: the fast elliptic clock is a *boundary-aware, globally-coupled* operator, and no amount of correctly-coloured local noise reconstructs it. This is, concretely, why statistical anomaly predictors (Markov chains, HMMs) and noise-injection SPDE closures remain blind to events where the pressure clock instantly rewrites the system's geometry.

## Scope

A 2D pedagogical demonstration of how the projection method *is* the two-clocks synthesis, and of the structural ceiling of spectrum-matching surrogates. It does **not** prove 3D regularity / Beale-Kato-Majda, and the 'Markov chain' is the advection-diffusion operator, not a data-trained transition matrix.

## Figures

![22_boussinesq_convection](22_boussinesq_convection.png)

![23_projection_step](23_projection_step.png)

![24_spde_limit](24_spde_limit.png)
