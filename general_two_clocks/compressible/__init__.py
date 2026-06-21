"""Minimal 2D linear-acoustics solver + elliptic (Poisson) reference.

Used to demonstrate the hyperbolic -> elliptic crossover: a compressible
fluid carries pressure information at finite sound speed c (the wave equation,
hyperbolic), and as c -> infinity (the low-Mach / incompressible limit) that
behaviour collapses onto the instantaneous, global Poisson constraint
(elliptic). This is a pedagogical demonstration, not a test of regularity.
"""
