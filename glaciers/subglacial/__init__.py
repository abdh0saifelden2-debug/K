"""Part 9 -- subglacial cavity flow: a computed test of the fast-clock claim.
 
A 2D penalized pseudo-spectral solver for turbulent meltwater flow in a cavity
between a bumpy rock bed (bottom) and the ice base (top), carrying a heat tracer.
Flow separates behind bedrock bumps into lee-side wake eddies -- the fast-clock
structures that single-eddy-diffusivity (K-theory) closure damps away.
 
The module reuses the repo's spectral machinery (`compressible.ns.Spectral2D`
and its Leray projection via `helmholtz`) and the two-clocks closure philosophy
of Part 8: a positive-definite Smagorinsky eddy viscosity (K-theory) versus the
same dissipation augmented with FDT-linked, Leray-projected stochastic
backscatter.  We then ask the glacier-relevant question a-posteriori: does the
closure sustain the lee wake and the heat it traps and delivers to the ice
(the melt proxy), or does it produce a "dead wake" and under-predict melt?
"""
