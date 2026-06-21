r"""RESULT 14 -- keystone positive control: a K-theory (down-gradient
eddy-diffusivity) closure produces NO migration (``E_cos -> 0``), while the
resolved advective flux produces the migration (``E_cos != 0``) on the *same*
frozen corrugated interface and the *same* mean drive.

Why this matters
----------------
Implication #1 of RESULT 14 (FUTURE_WORK Sec.G.2) is the keystone: the migration
term ``Im(s) = omega_mig != 0`` is a **parity-symmetry break that no K-theory
closure can produce**.  The argument is analytic -- a *local, memoryless,
down-gradient* eddy-diffusivity acting on a sinusoidal interface
``y = a sin(Kx)`` is symmetric under ``x -> -x`` (it carries no flow *direction*),
so it can only return an **in-phase** (real, damping) flux response: the
quadrature gain ``E_cos`` (the migration channel) is identically zero.  The
earlier ``U = 0`` parity control removed the *drive*; this control instead keeps
the drive and removes the *advection*, replacing it with the down-gradient
closure K-theory actually prescribes.  It therefore turns the keystone from an
argument into a **measurement**.

Construction
------------
The momentum field is advanced **identically** to the resolved migration run
(same mean drive ``U``, same turbulent forcing, same seed), so the velocity
field is the same.  Only the *scalar* (temperature) transport is swapped:

    resolved   :  dtheta/dt = -(u . grad theta)  + kappa lap theta   (advective)
    K-theory   :  dtheta/dt =  div(K_eddy grad theta) + kappa lap theta

i.e. the advective heat flux ``-u.grad theta`` -- the only term that carries the
signed flow *direction* -- is replaced by a down-gradient eddy flux
``-K_eddy grad theta``.  Two choices of the (non-negative, down-gradient)
``K_eddy`` are tested, the second deliberately the *most generous* to K-theory:

  * ``uniform``     : ``K_eddy = const`` (canonical eddy-diffusivity hypothesis;
                      geometry/flow-blind) -- the rigorous parity case,
                      ``E_cos`` must vanish to numerical precision.
  * ``smagorinsky`` : ``K_eddy = (cs*Delta)^2 |S|`` from the resolved mean strain
                      (Pr_t = 1) -- a *flow-aware* down-gradient closure; it
                      "sees" the flow magnitude but still carries no direction,
                      so ``E_cos`` must stay ~0 even though ``E_sin`` (smoothing)
                      may grow.

The flux excess ``e = m_model - m_cond`` is projected onto the corrugation
harmonics exactly as in ``scallop_amplitude_harmonics`` (``E_cos`` = quadrature
= migration, ``E_sin`` = in-phase = amplitude change).

Backend-agnostic (``xp=cupy`` for GPU); CPU-runnable at small resolution.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_amplitude_harmonics import LX, project  # noqa: E402
from scallop_probe import ProbeFlow, _run  # noqa: E402
from subglacial.candidate3_roughness_feedback import (  # noqa: E402
    Candidate3Config,
    project2d,
)


class KTheoryProbeFlow(ProbeFlow):
    """``ProbeFlow`` whose *scalar* transport is a down-gradient eddy-diffusivity
    closure (no advection of ``theta``).  Momentum is advanced unchanged, so the
    velocity field is identical to the resolved run."""

    def __init__(self, cfg, U_drive=0.0, eddy="uniform", ke_uniform=None,
                 xp=np):
        super().__init__(cfg, U_drive=U_drive, xp=xp)
        self.eddy = str(eddy)
        # default uniform eddy diffusivity: a few x the molecular kappa, i.e. a
        # turbulent Peclet ~O(1-10), comfortably in the "K-theory matters" range
        self.ke_uniform = (10.0 * cfg.kappa if ke_uniform is None
                           else float(ke_uniform))

    def _eddy_diffusivity(self, u, v):
        """Non-negative, down-gradient eddy diffusivity ``K_eddy(x, y)``."""
        cfg, sp, xp = self.cfg, self.sp, self.xp
        if self.eddy == "uniform":
            return self.ke_uniform * self.fluid
        if self.eddy != "smagorinsky":
            raise ValueError(
                f"unknown eddy closure {self.eddy!r}; expected "
                "'uniform' or 'smagorinsky'")
        # flow-aware Smagorinsky scalar diffusivity (Pr_t = 1) -- still a pure
        # magnitude, carries no flow direction
        delta = float(np.sqrt(sp.dx * sp.dy))
        ux, uy, vx, vy = sp.ddx(u), sp.ddy(u), sp.ddx(v), sp.ddy(v)
        s11, s22, s12 = ux, vy, 0.5 * (uy + vx)
        smag = xp.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2))
        return (cfg.cs * delta) ** 2 * smag * self.fluid

    def step(self):
        cfg, sp, xp = self.cfg, self.sp, self.xp
        u, v, theta = self.u, self.v, self.theta

        Nu = self._advect(u, v, u)
        Nv = self._advect(u, v, v)
        # --- K-theory scalar transport: down-gradient eddy flux, NO advection ---
        Ke = self._eddy_diffusivity(u, v)
        Nt = sp.ddx(Ke * sp.ddx(theta)) + sp.ddy(Ke * sp.ddy(theta))
        # -----------------------------------------------------------------------
        mx, my = self._sgs_force(u, v)
        Nu, Nv = Nu + mx, Nv + my
        if cfg.f_amp > 0.0:
            fx, fy = self._forcing()
            Nu, Nv = Nu + fx, Nv + fy
        if cfg.Ri != 0.0:
            theta_ref = float(xp.mean(theta * self.fluid) / xp.mean(self.fluid))
            Nv = Nv + cfg.Ri * (theta - theta_ref) * self.fluid

        uh = self.specfilt * self.visc_u * (sp.fft(u) + cfg.dt * sp.fft(Nu) * sp.dealias)
        vh = self.specfilt * self.visc_u * (sp.fft(v) + cfg.dt * sp.fft(Nv) * sp.dealias)
        th = self.specfilt * self.visc_t * (sp.fft(theta) + cfg.dt * sp.fft(Nt) * sp.dealias)
        u1, v1, t1 = sp.ifft(uh), sp.ifft(vh), sp.ifft(th)

        u1 = u1 / (1.0 + self.pen)
        v1 = v1 / (1.0 + self.pen)
        t1 = (t1 + self.pen * self.theta_solid) / (1.0 + self.pen)
        u1, v1 = project2d(sp, u1, v1)

        self.u, self.v, self.theta = u1, v1, t1
        self.t += cfg.dt
        self.step_count += 1


def _run_ktheory(cfg, a, n_waves, U_drive, spinup, measure, eddy, ke_uniform=None,
                 xp=np):
    """Spin up + time-average the per-column melt for the K-theory closure."""
    s = KTheoryProbeFlow(cfg, U_drive=U_drive, eddy=eddy, ke_uniform=ke_uniform,
                         xp=xp)
    yb = s.set_single_mode(a, n_waves)
    for _ in range(spinup):
        s.step()
    acc = np.zeros(cfg.nx)
    cnt = np.zeros(cfg.nx)
    for _ in range(max(measure, 1)):
        s.step()
        mi = s.melt_field()
        ok = np.isfinite(mi)
        acc[ok] += mi[ok]
        cnt[ok] += 1.0
    m = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
    return yb, m


def ktheory_control(nw=12, U=3.0, *, seeds=(0, 1), nx=128, ny=128, afrac=0.20,
                    spinup=2000, measure=400, f_amp=0.4, ke_uniform=None,
                    eddies=("uniform", "smagorinsky"), xp=np):
    r"""Compare the quadrature (migration) gain ``E_cos`` of the resolved
    advective flux against down-gradient K-theory closures, on the same
    interface and drive.

    Returns a dict with, per model, the seed-averaged ``(E_cos, E_sin)`` of the
    flux excess ``m_model - m_cond``.  The keystone prediction: ``|E_cos|`` for
    every K-theory closure is far below the resolved ``|E_cos|`` (``~0``), while
    ``E_sin`` (smoothing) may be comparable -- K-theory can damp but not migrate.
    """
    lam = LX / nw
    K = 2.0 * np.pi * nw / LX
    a = afrac * lam
    res = {m: {"Ecos": [], "Esin": []} for m in (("resolved",) + tuple(eddies))}
    for sd in seeds:
        cfg0 = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.0,
                                Ri=0.0, seed=sd)
        _, m_cond, _, _ = _run(cfg0, a, nw, U_drive=0.0, spinup=spinup, xp=xp)

        cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=f_amp,
                               Ri=0.0, seed=sd)
        _, m_res, _, _ = _run(cfg, a, nw, U_drive=U, spinup=spinup,
                              measure=measure, xp=xp)
        Ec, Es = project(m_res - m_cond, K, nx)
        res["resolved"]["Ecos"].append(Ec)
        res["resolved"]["Esin"].append(Es)

        for eddy in eddies:
            cfgk = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=f_amp,
                                    Ri=0.0, seed=sd)
            _, m_k = _run_ktheory(cfgk, a, nw, U_drive=U, spinup=spinup,
                                  measure=measure, eddy=eddy,
                                  ke_uniform=ke_uniform, xp=xp)
            Ec, Es = project(m_k - m_cond, K, nx)
            res[eddy]["Ecos"].append(Ec)
            res[eddy]["Esin"].append(Es)

    out = dict(nw=int(nw), U=float(U), lam=float(lam), K=float(K), a=float(a),
               nx=nx, ny=ny, spinup=spinup, measure=measure, f_amp=f_amp,
               seeds=list(seeds), models={})
    for m, d in res.items():
        ec = np.array(d["Ecos"])
        es = np.array(d["Esin"])
        out["models"][m] = dict(Ecos_mean=float(ec.mean()),
                                Ecos_std=float(ec.std()),
                                Esin_mean=float(es.mean()),
                                Esin_std=float(es.std()))
    rc = abs(out["models"]["resolved"]["Ecos_mean"]) + 1e-30
    for m in out["models"]:
        out["models"][m]["Ecos_ratio_to_resolved"] = (
            abs(out["models"][m]["Ecos_mean"]) / rc)
    return out


if __name__ == "__main__":
    import json
    r = ktheory_control()
    print(json.dumps(r, indent=2))
