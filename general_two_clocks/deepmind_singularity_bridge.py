r"""Bridge: DeepMind's "Discovery of Unstable Singularities" (Wang et al., arXiv:2509.14185,
Sept 2025) vs this repo's nonlocal-pressure-Hessian regularity take.

What DeepMind found (mainstream, 2025)
--------------------------------------
Using physics-informed neural networks (PINNs) + a high-precision Gauss-Newton optimizer,
Wang, Buckmaster, Gomez-Serrano, Lai et al. (Google DeepMind + Brown/NYU/Stanford) made the
"first systematic discovery of new FAMILIES of UNSTABLE self-similar singularities" across
fluid models: the Incompressible Porous Media (IPM) equation, the 2D Boussinesq equation, and
the 3D Euler equation WITH boundary (plus the 1D Cordoba-Cordoba-Fontelos model as a
high-precision testbed). Two headline points matter for us:

  (D1) The relevant singularities are *unstable*: they "require initial conditions tuned with
       infinite precision", and "infinitesimal perturbations immediately divert the solution
       from its blow-up trajectory". Boundary-free Euler/Navier-Stokes are expected to have NO
       stable singularities, so the unstable ones are the only candidates.
  (D2) An empirical asymptotic *ladder*: the self-similar blow-up rate lambda_n follows a
       RECIPROCAL law in the ORDER OF INSTABILITY n (the number of unstable directions of the
       renormalized self-similar operator) -- lambda_n ~ 1/(1.4187 n + 1.0863) + 1 for
       Boussinesq/Euler and lambda_n ~ 1/(1.1459 n + 0.9723) for IPM-with-boundary, i.e.
       1/(lambda_n - c) is linear in n. The pattern is clearest for IPM and Boussinesq.
  For specific solutions (the Cordoba-Cordoba-Fontelos stable and 1st-unstable profiles) they
  reach near-double-float precision -- enough to *attempt* computer-assisted proofs (none of the
  new candidates is proved yet) -- but they do NOT treat boundary-free 3D Euler/Navier-Stokes
  (the actual Clay target).

This repo's take (REPORT_CLAY_REGULARITY.md)
--------------------------------------------
The nonlocal (anisotropic) pressure Hessian is the regularizing operator. Dropping it
(restricted Euler / Vieillefosse) gives finite-time blow-up |A|~(t*-t)^-1 (Theorem 1,
RIGOROUS). The open core (= the Clay problem) is that the nonlocal-Hessian depletion of vortex
stretching is *self-sustaining for all data and time* -- i.e. that no admissible flow can
"conspire" onto a measure-zero alignment that defeats it.

The bridge this module computes (the genuine, verifiable connection)
--------------------------------------------------------------------
DeepMind's organizing object is the renormalized self-similar operator and its *instability
order*. We compute exactly that object for the repo's restricted-Euler caricature and show
where it lands on DeepMind's scale -- honestly, including what the caricature cannot capture.

Restricted-Euler invariants Q=-1/2 tr(A^2), R=-1/3 tr(A^3) obey  Q'=-3R, R'=(2/3)Q^2. The
blow-up is self-similar with  Q=(t*-t)^-2 q,  R=(t*-t)^-3 r,  s=-ln(t*-t), giving the
renormalized 2-D system

    q' = -2q - 3r ,     r' = (2/3) q^2 - 3r ,

whose nontrivial fixed point  (q*, r*) = (-3, 2)  is exactly the Vieillefosse profile
(Q~-3(t*-t)^-2, R~2(t*-t)^-3). Its Jacobian eigenvalues are {+1, -6} (analytic; trace -5,
det -6). So:

  * the -6 (stable) mode is the attraction onto the self-similar Vieillefosse *shape*
    (the tail r^2 + (4/27)q^3 = 0 through (-3,2)); generic data is pulled onto it;
  * the +1 (unstable) mode is the UNIVERSAL blow-up-time / amplitude freedom that every
    self-similar blow-up carries (choice of t*). It is the trivial mode DeepMind quotients out.

=> Modulo the trivial time-translation mode, the restricted-Euler self-similar blow-up has
   instability order ZERO -- it is DeepMind's *stable* singularity type (the kind classical
   numerics find readily). DeepMind's new content -- a LADDER of order-1,2,3,... UNSTABLE
   profiles -- is exactly what a purely quadratic VGT caricature CANNOT contain (its scaling
   exponent is fixed by homogeneity, so there is no lambda-ladder).

Benefit to our take (the honest verdict)
----------------------------------------
  [+] Conceptual support. The repo's open core conjectures blow-up (if any) is a non-generic,
      "conspiratorial" event. DeepMind's independent, proof-grade evidence that the singular
      solutions are *unstable / infinitely fine-tuned* is the same statement from the blow-up
      side: the depletion geometry is generic, defeating it is not.
  [+] Sharpened open core. The nonlocal pressure Hessian must deplete not merely the single
      stable Vieillefosse profile (which the VGT model captures) but the WHOLE ladder of
      unstable self-similar profiles DeepMind exhibits. That is a concrete, harder target.
  [+] A method to attempt the open core. The PINN + high-precision Gauss-Newton +
      computer-assisted-proof playbook is exactly the tooling the repo's depletion inequality
      (REPORT_CLAY_REGULARITY sec6) would need to become a rigorous bound.
  [-] No direct NS gain. DeepMind treats IPM / Boussinesq / Euler-with-boundary, NOT
      boundary-free 3D Navier-Stokes; nothing here closes the repo's open core.
  [-] Caricature limit. The VGT model reproduces the *stable* profile and the
      "blow-up = renormalized fixed point" framing, but not the unstable ladder or the
      lambda-vs-order law (fixed quadratic scaling).

All CPU, no data. Verified in tests/test_deepmind_singularity_bridge.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import restricted_euler_regularity as rer  # noqa: E402


# --------------------------------------------------------------------------- #
# renormalized (self-similar) restricted-Euler invariant system
# --------------------------------------------------------------------------- #
def renorm_rhs(y):
    r"""Renormalized restricted-Euler invariants: q'=-2q-3r, r'=(2/3)q^2-3r.

    Derived from Q'=-3R, R'=(2/3)Q^2 with Q=(t*-t)^-2 q, R=(t*-t)^-3 r, s=-ln(t*-t).
    The nontrivial fixed point is the Vieillefosse self-similar profile (q*,r*)=(-3,2).
    """
    q, r = float(y[0]), float(y[1])
    return np.array([-2.0 * q - 3.0 * r, (2.0 / 3.0) * q * q - 3.0 * r])


def jacobian(y, eps=1e-6):
    J = np.zeros((2, 2))
    for j in range(2):
        yp = np.array(y, float); yp[j] += eps
        ym = np.array(y, float); ym[j] -= eps
        J[:, j] = (renorm_rhs(yp) - renorm_rhs(ym)) / (2 * eps)
    return J


def self_similar_instability():
    """The renormalized restricted-Euler self-similar fixed point and its instability order.

    Returns the fixed point, its (numerical + analytic) Jacobian eigenvalues, the instability
    order (# eigenvalues with Re>0), and the instability order AFTER removing the universal
    time-translation mode (the +1 eigenvalue) -- the quantity comparable to DeepMind's scale.
    """
    fp = np.array([-3.0, 2.0])                       # Vieillefosse self-similar profile
    residual = float(np.linalg.norm(renorm_rhs(fp)))  # must be ~0
    J = jacobian(fp)
    ev = np.sort(np.linalg.eigvals(J).real)          # {-6, +1}
    # analytic: trace=-5, det=-6 -> eigenvalues (-5 +/- 7)/2 = {1, -6}
    tr, det = float(np.trace(J)), float(np.linalg.det(J))
    ev_analytic = sorted([(tr + np.sqrt(tr * tr - 4 * det)) / 2,
                          (tr - np.sqrt(tr * tr - 4 * det)) / 2])
    n_unstable = int(np.sum(ev > 1e-9))
    # the single +1 mode is the universal blow-up-time freedom (present for ANY self-similar
    # blow-up); DeepMind quotients it out. Remove exactly one trivial +1 mode:
    trivial_timeshift_mode = bool(np.any(np.isclose(ev, 1.0, atol=1e-6)))
    order_mod_trivial = max(0, n_unstable - (1 if trivial_timeshift_mode else 0))
    return dict(
        fixed_point=fp.tolist(), residual=residual,
        jacobian=J.tolist(), eigenvalues=ev.tolist(), eigenvalues_analytic=ev_analytic,
        trace=tr, det=det, instability_order_raw=n_unstable,
        has_trivial_timeshift_mode=trivial_timeshift_mode,
        instability_order_mod_trivial=order_mod_trivial,
        deepmind_class=("stable singularity (order 0 mod time-shift) -- the classically-found "
                        "type; DeepMind's NEW unstable ladder (order>=1) is not present in this "
                        "fixed-exponent quadratic caricature"),
        ok=bool(residual < 1e-9 and np.allclose(ev, [-6.0, 1.0], atol=1e-4)
                and order_mod_trivial == 0))


def _integrate_renorm(y0, ds=1e-3, s_max=8.0, big=1e3):
    """Integrate the renormalized system (RK4) from y0; report whether it returns to the
    fixed point (stable-shape direction) or escapes (the time-shift/amplitude direction)."""
    y = np.array(y0, float); s = 0.0
    while s < s_max and np.linalg.norm(y) < big:
        k1 = renorm_rhs(y); k2 = renorm_rhs(y + 0.5 * ds * k1)
        k3 = renorm_rhs(y + 0.5 * ds * k2); k4 = renorm_rhs(y + ds * k3)
        y = y + ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        s += ds
    return y, float(np.linalg.norm(y - np.array([-3.0, 2.0])))


def saddle_character(delta=1e-2):
    """Confirm the fixed point is a saddle: perturb along each Jacobian eigenvector; the
    stable (-6) direction relaxes back, the unstable (+1) direction departs."""
    fp = np.array([-3.0, 2.0])
    J = jacobian(fp)
    w, V = np.linalg.eig(J)
    idx_stable = int(np.argmin(w.real)); idx_unstable = int(np.argmax(w.real))
    vs = V[:, idx_stable].real; vs /= np.linalg.norm(vs)
    vu = V[:, idx_unstable].real; vu /= np.linalg.norm(vu)
    # along stable eigenvector: distance to FP should shrink relative to the initial offset
    _, d_stable = _integrate_renorm(fp + delta * vs, s_max=1.0)
    _, d_unstable = _integrate_renorm(fp + delta * vu, s_max=1.0)
    return dict(eig_stable=float(w[idx_stable].real), eig_unstable=float(w[idx_unstable].real),
                dist_after_stable_perturb=d_stable, dist_after_unstable_perturb=d_unstable,
                ok=bool(d_stable < delta and d_unstable > delta))


def full_model_context():
    """Place the caricature inside the full nonlocal-Hessian model: blow-up is *generic* in
    the truncated (restricted-Euler) model but *non-generic* once the nonlocal pressure Hessian
    is restored -- the embodiment of 'blow-up is the fine-tuned case' in the repo's own VGT
    closure (reuses restricted_euler_regularity)."""
    local_blow = rer.generic_ic_blowup_fraction(n_ic=24)      # tau=0: ~all blow up
    nonlocal_reg = rer.rfd_regularizes(seeds=(0, 1, 2), tau=0.12)  # tau>0: all bounded
    return dict(
        restricted_euler_blowup_fraction=local_blow["blowup_fraction"],
        nonlocal_hessian_bounded=f"{nonlocal_reg['n_bounded']}/{nonlocal_reg['n_seeds']}",
        ok=bool(local_blow["blowup_fraction"] > 0.8 and nonlocal_reg["ok"]))


def run():
    return dict(self_similar=self_similar_instability(),
                saddle=saddle_character(),
                full_model=full_model_context())


def summary():
    r = run()
    r["all_ok"] = bool(r["self_similar"]["ok"] and r["saddle"]["ok"] and r["full_model"]["ok"])
    return r


if __name__ == "__main__":
    r = summary()
    ss = r["self_similar"]; sd = r["saddle"]; fm = r["full_model"]
    print("DeepMind unstable-singularities bridge (arXiv:2509.14185) <-> repo NS-regularity take")
    print("=" * 84)
    print(f"\n[1] Renormalized restricted-Euler self-similar profile (q*,r*)={ss['fixed_point']}")
    print(f"    residual={ss['residual']:.1e}; Jacobian eigenvalues={ss['eigenvalues']} "
          f"(analytic {ss['eigenvalues_analytic']}; trace {ss['trace']:.1f}, det {ss['det']:.1f})")
    print(f"    raw instability order={ss['instability_order_raw']}; the +1 mode is the universal")
    print(f"    blow-up-time freedom -> order mod time-shift = {ss['instability_order_mod_trivial']}")
    print(f"    => {ss['deepmind_class']}")
    print(f"\n[2] Saddle check: stable eig={sd['eig_stable']:.2f} (shape attracts: dist "
          f"{sd['dist_after_stable_perturb']:.2e}), unstable eig={sd['eig_unstable']:.2f} "
          f"(time-shift departs: dist {sd['dist_after_unstable_perturb']:.2e})")
    print(f"\n[3] Full-model context: restricted-Euler blow-up fraction="
          f"{fm['restricted_euler_blowup_fraction']:.0%} (generic), nonlocal-Hessian bounded="
          f"{fm['nonlocal_hessian_bounded']} (blow-up non-generic once the Hessian is restored)")
    print("\n" + "=" * 84)
    print("VERIFIED" if r["all_ok"] else "SOME CHECKS FAILED")
    sys.exit(0 if r["all_ok"] else 1)
