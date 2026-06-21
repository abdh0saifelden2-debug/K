# Two clocks in a nonlinear compressible flow

Fully nonlinear 2D **isothermal compressible Navier-Stokes** (pseudo-spectral, periodic, constant viscosity, 2/3 dealiasing). Isothermal closure `p = c² ρ` keeps the sound speed `c` an explicit knob, so the Mach number `M = U/c` is swept directly while retaining full nonlinear advection.

Initial condition: Taylor-Green vortex, Re ≈ 628. Showcase run at Mach 0.15.

## 1. Two clocks, measured

Helmholtz-decomposing the velocity separates a **solenoidal / vortical** component (the slow advective clock, τ_adv ~ L/U) from a **dilatational / acoustic** component (the fast clock, τ_p ~ L/c). Figure 20 shows the vortical energy decaying slowly while the acoustic energy oscillates on the fast acoustic period — two clocks coexisting in a single nonlinear flow. Their ratio is set by the Mach number, which is exactly the timescale-separation parameter ε = τ_p/τ_adv = M.

## 2. The incompressible (elliptic) limit

Sweeping the Mach number:

| Mach M | ⟨KE_dil⟩/⟨KE_sol⟩ | avg-pressure residual vs elliptic |
|---|---|---|
| 0.05 | 1.466e-04 | 8.068e-04 |
| 0.1 | 5.856e-04 | 3.169e-03 |
| 0.2 | 2.343e-03 | 1.373e-02 |
| 0.4 | 9.328e-03 | 9.956e-02 |

The acoustic (dilatational) kinetic-energy fraction decreases cleanly as **~ M²** toward zero (figure 21, left) — the fast clock's energy vanishes in the incompressible limit. For the pressure field itself, the *instantaneous* compressible pressure is dominated by the persistent standing acoustic wave launched by the uniform-density start, so it is first **averaged over the fast acoustic clock** (the quasi-steady second half of each run). This acoustic-averaged compressible pressure matches the **elliptic incompressible Poisson pressure** at low Mach (figure 21, right; correlation ≈ 1), the residual itself shrinking as **~ M²** toward the incompressible limit and growing with compressibility toward M = 0.4. In the incompressible limit the fast acoustic clock disappears and the pressure is governed entirely by the instantaneous, global Poisson solve — the elliptic regime of the Rayleigh-Bénard DNS (`REPORT_RB.md`) and the linear crossover (`REPORT_COMPRESSIBLE.md`).

## Interpretation

This is the nonlinear confirmation of the structural picture: pressure carries a fast, global, wave-like adjustment (the elliptic/acoustic channel) that is distinct from the slow vortical/advective evolution of the flow. The separation is controlled by the Mach number; as M → 0 the pressure becomes the purely elliptic, instantaneous global field.

## Scope

Demonstrates scale separation and the elliptic limit in a nonlinear compressible flow. Does **not** prove 3D Navier-Stokes regularity / Beale-Kato-Majda, nor 'wave-radiation damping'. Those require rigorous PDE analysis, not a 2D demonstration solver.

## Figures

![19_ns_structure](19_ns_structure.png)
![20_ns_two_clocks](20_ns_two_clocks.png)
![21_ns_mach_sweep](21_ns_mach_sweep.png)