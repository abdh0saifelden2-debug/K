# §D.2 Scallop → double-diffusion: regime mixing in one geometry

A scalloped ice wall (`wall_amp`, `wall_nwaves` on the Candidate-2 double-diffusion solver) is compared to the smooth wall at identical physics. Per-column turbulent fluxes give `Nu_T(x)`, `Nu_S(x)` and the local Turner ratio `gamma(x) = F_T(x)/(R_rho F_S(x))`.

- Grid `nx=256`, `ny=96`, aspect `A=4.0`; `R_rho=1.5`, `Ri_T=2.0`, `Le=100.0`, `f_amp=0.05`; wall `a/λ=0.3`, `n_waves=12`; spinup `6000`, measure `4000`.

## Result

- **Heterogeneity** — phase-binned `Nu_T` peak-to-trough (turbulence averaged out) smooth `5.3110` → scallop `67.4758` (**amplified: True**).
- **Phase-locking** — total wall-coherent variance fraction `η²` (all harmonics, correlation ratio) smooth `0.093` → scallop `0.891` (**locked: True**). The coherent response splits into a *fundamental* (lee/stoss directional) fraction `0.437` and a dominant *2nd-harmonic* (symmetric constriction, peaks at every crest **and** trough) fraction `0.506`; the fundamental-only index `f_lock=0.437` @ `4.68` rad undercounts the lock because the corrugation response is symmetric.
- **Regime mixing** — local `gamma(x)` range `0.994`, enhanced frac `0.93`, suppressed frac `0.07` about the smooth value `+0.620` (**both regimes coexist: True**).
- Global means (sanity vs Candidate 2): `Nu_T` smooth `+0.996` / scallop `-5.781`; `Nu_S` smooth `-0.30` / scallop `-704.91`.

- Backend: `cupy(GPU)`.
