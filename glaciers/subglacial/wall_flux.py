r"""Ice-side thermal wall model for the 3D subglacial cavity solver.
 
Replaces the *infinite-conductance* Dirichlet ice-base pin (theta -> 0) with a
*finite-conductance* ice wall, implemented as a clean one-knob modification of
the solver's own Brinkman thermal penalty.
 
**Motivation.** With the default Dirichlet ice BC the near-wall conductive
sublayer is the controlling thermal resistance, and the flow (even strong
turbulence / 3D channelization) cannot raise the net basal heat flux above flat.
A finite-conductance wall decouples the ice thermal boundary from the no-slip
velocity wall, letting the flow influence the delivered heat.
 
**Implementation.** The default solver applies, after the diffusion sub-step,
 
    t1 = (t1 + pen * theta_solid) / (1 + pen),   pen = dt * chi / eta,
    chi = clip(chi_rock + chi_ice, 0, 1),  theta_solid = chi_rock * 1 + chi_ice * 0.
 
This module keeps that update *and its target* ``theta_solid`` intact, and scales
only the ice share of the penalty *strength* by the conductance ratio:
 
    chi_eff = clip(chi_rock + cond_ratio * chi_ice, 0, 1),  pen = dt * chi_eff / eta.
 
Because the target is unchanged and ``chi_ice ~ 0`` throughout the rock and the
rock-fluid transition, the rock pin and that transition are reproduced exactly.
 
``cond_ratio = 1`` reproduces the original Dirichlet wall *bit-for-bit*;
``cond_ratio < 1`` weakens the ice pull -> finite-conductance (Robin-like) sink;
``cond_ratio -> 0`` makes the ice adiabatic.  In ``robin_ustar`` mode the local
conductance is scaled by the friction velocity u*(x,z) (the Reynolds analogy).
 
N.B. an earlier draft split the update into independent rock/ice pulls
``(t1 + s_r*1 + s_i*0)/(1 + s_r + s_i)``; that is *not* algebraically equal to
the solver's update in the transition zones (it pulls the target toward 1 rather
than toward ``chi_rock``), so ``cond_ratio = 1`` did not reduce to the Dirichlet
wall.  The ``chi_eff`` form above is the correct, provably-reducing one.
 
**Design.** Feeds back into ``flow3d.py`` via the single
``Subglacial3DConfig.thermal_wall`` hook; the solver delegates its post-diffusion
thermal update to ``ThermalWall.apply(flow, t1)``.  All physics lives here.
 
Observable: ``wall_flux(flow)`` returns the heat actually absorbed by the ice
penalty per unit wall area per unit time -- the basal-melt heat flux.
 
Usage:
    from subglacial.wall_flux import ThermalWall
    tw = ThermalWall(cond_ratio=0.05, mode='robin_const')
    cfg = Subglacial3DConfig(..., thermal_wall=tw)
    flow = Subglacial3DFlow(cfg); flow.run(1000)
    print(tw.wall_flux(flow))
"""
 
from __future__ import annotations
 
 
 
class ThermalWall:
    """Finite-conductance ice-side thermal boundary condition.
 
    Parameters
    ----------
    cond_ratio : float
        Ice conductance relative to the (infinite) Dirichlet limit, in (0, 1].
        1.0 -> original Dirichlet wall; < 1 -> finite conductance; -> 0 adiabatic.
    mode : str
        'robin_const' -- uniform cond_ratio over the ice wall.
        'robin_ustar' -- local cond_ratio scaled by u*(x,z)/mean(u*) (Reynolds
                         analogy: heat transfer tracks wall friction).
    T_melt : float
        Ice melting temperature (default 0.0 in solver units).
    """
 
    def __init__(
        self,
        cond_ratio: float = 0.05,
        mode: str = "robin_const",
        T_melt: float = 0.0,
    ):
        if not (0.0 < cond_ratio <= 1.0):
            raise ValueError("cond_ratio must be in (0, 1]")
        self.cond_ratio = cond_ratio
        self.mode = mode
        self.T_melt = T_melt
        self._last_flux = None     # realized mean wall flux (heat into ice)
        self._last_ustar = None
 
    # ------------------------------------------------------------------ #
    # core: called by flow3d.step() via cfg.thermal_wall.apply(flow, t1)
    # ------------------------------------------------------------------ #
    def apply(self, flow, t1):
        """Replace the ice Dirichlet pin with a finite-conductance sink.

        The solver's own penalty update ``t1 = (t1 + pen * theta_solid) /
        (1 + pen)`` with ``pen = dt * chi / eta`` and ``chi = clip(chi_rock +
        chi_ice, 0, 1)`` is kept intact; only the *ice* share of the penalty
        strength is scaled by the local conductance ratio:

            chi_eff = clip(chi_rock + ratio * chi_ice, 0, 1),  pen = dt*chi_eff/eta.

        The target ``theta_solid`` (= ``chi_rock``) is unchanged, so the rock pin
        and the rock/fluid transition behave exactly as in the bare solver and
        ``ratio == 1`` reproduces the Dirichlet wall bit-for-bit. ``ratio < 1``
        weakens the ice pull (finite conductance); ``ratio -> 0`` is adiabatic.
        Returns the modified t1.
        """
        xp = flow.xp
        cfg = flow.cfg

        ratio = self._cond_field(flow)                  # scalar or (n,n,n)
        # weaken ONLY the ice contribution to the Brinkman penalty strength;
        # rock + rock/fluid transition (chi_ice ~ 0 there) are preserved exactly.
        chi_eff = xp.clip(flow.chi_rock + ratio * flow.chi_ice, 0.0, 1.0)
        pen = cfg.dt * chi_eff / cfg.eta

        t1_pre = t1
        t1 = (t1_pre + pen * flow.theta_solid) / (1.0 + pen)

        # heat absorbed by the *ice* sink this step, per unit wall area per time.
        # Isolate the ice share of the penalty (s_i) from the combined strength:
        #   Q = sum_cells [ s_i/(1+pen) * (theta_pre - T_melt) ] * dV / (A_wall * dt)
        # dV = dx^3, A_wall = (n*dx)^2 = L^2  ->  dV/A_wall = dx^3 / L^2,
        # and the per-step removal is divided by dt to get a rate.
        s_i = cfg.dt * ratio * flow.chi_ice / cfg.eta
        removal = (s_i / (1.0 + pen)) * (t1_pre - self.T_melt)
        dV_over_A = flow.dx ** 3 / (flow.sp.L ** 2)
        self._last_flux = float(xp.sum(removal) * dV_over_A / cfg.dt)

        return t1
 
    # ------------------------------------------------------------------ #
    def _cond_field(self, flow):
        """Local conductance ratio (scalar for const, (n,n,n) for ustar)."""
        if self.mode == "robin_const":
            return self.cond_ratio
        if self.mode == "robin_ustar":
            return self._cond_from_ustar(flow)
        raise ValueError(f"unknown mode {self.mode!r}")
 
    def _cond_from_ustar(self, flow):
        """cond_ratio scaled by local u* (Reynolds analogy), capped at 1."""
        xp = flow.xp
        cfg = flow.cfg
        band = self._ice_band(flow)
        ustar = self._ustar_field(flow)
        ustar_mean = float(xp.mean(ustar[band])) if band.any() else 1.0
        ustar_mean = max(ustar_mean, 1e-12)
        ratio = self.cond_ratio * ustar / ustar_mean
        return xp.clip(ratio, 0.0, 1.0)
 
    def _ustar_field(self, flow):
        xp = flow.xp
        cfg = flow.cfg
        s = xp.sqrt(flow.u ** 2 + flow.w ** 2)
        delta = xp.clip(cfg.ice_base - flow.Y, flow.dx, None)
        return xp.sqrt(cfg.nu * s / delta)
 
    def _ice_band(self, flow):
        cfg = flow.cfg
        thk = 3.0 * flow.dx
        return flow.fluid & (flow.Y > cfg.ice_base - thk) & (flow.Y < cfg.ice_base)
 
    # ------------------------------------------------------------------ #
    # diagnostics (called by probe scripts)
    # ------------------------------------------------------------------ #
    def wall_flux(self, flow):
        """Mean heat flux absorbed by the ice = basal-melt heat flux."""
        if self._last_flux is not None:
            return self._last_flux
        return 0.0
 
    def friction_velocity(self, flow):
        """Area-averaged friction velocity u* in the near-ice band."""
        xp = flow.xp
        band = self._ice_band(flow)
        ustar = self._ustar_field(flow)
        return float(xp.mean(ustar[band])) if band.any() else 0.0
