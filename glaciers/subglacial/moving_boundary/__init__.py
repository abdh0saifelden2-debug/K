"""Moving-boundary (free-surface) machinery for the subglacial cavity solver.

The fixed-geometry penalized solvers (``subglacial.flow`` / ``subglacial.flow3d``)
treat the ice base as a stationary Brinkman solid.  A real cavity is a *living
geometry*: melt thins the ice, the changed geometry changes the flow, and the
flow changes the melt.  This subpackage adds that missing Stefan feedback.

Build order (each stage is validated before the next is trusted):

1. ``stefan`` -- one-phase Stefan problem.  Analytic Neumann similarity solution
   plus a fixed-grid enthalpy-method solver, validated to reproduce
   ``X(t) = 2 lambda sqrt(alpha t)``.  This is the moving-front core.
2. (next) 2-D enthalpy melting coupled to an advecting flow.
3. (next) full coupling to the penalized cavity flow with a pressure-gradient
   drive and interface-flux melt diagnostics.
"""

from subglacial.moving_boundary.stefan import (
    Stefan1D,
    Stefan1DConfig,
    neumann_front,
    neumann_lambda,
    neumann_temperature,
)

__all__ = [
    "Stefan1D",
    "Stefan1DConfig",
    "neumann_front",
    "neumann_lambda",
    "neumann_temperature",
]
