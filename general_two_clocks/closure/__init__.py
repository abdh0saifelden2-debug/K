"""Part 8b -- the two-clocks closure benchmark.
 
Generates a 2D incompressible DNS truth field, sharp-spectral-filters it, forms the
*exact* subgrid force, and scores three closures (Smagorinsky K-theory, a
spectrum-matched surrogate, and the projected-FDT model of Part 8) on the force
spectrum, the divergence, and the energy-transfer spectrum (forward dissipation vs
backscatter).
"""
