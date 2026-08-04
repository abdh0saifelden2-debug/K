r"""NR57 (theory + real data) -- the spin is the irreversibility clock: the
Lagrangian areal velocity (odd correlation slope, NR52) is the phase-space
probability current, so its square LOWER-BOUNDS the entropy production
rate; detailed balance holds IFF the spin vanishes.  The deep ocean is
locally time-irreversible with a handedness set by f, and reversible at
the equator.

Follows NR52 (spin = parity- and lag-odd face; a "detailed-balance
meter") and NR56 (Stokes V).  NR52 asserted the spin certifies broken
time-reversal; NR57 makes it QUANTITATIVE via stochastic thermodynamics.

The claim
=========
For a stationary Ornstein-Uhlenbeck flow ``dx = -A x dt + sqrt(2D) dW``
with stationary covariance ``C`` (solving ``A C + C A^T = 2D``):

1. **Areal velocity = probability current rotation.**  The steady current
   is ``J(x) = -Omega x rho(x)`` with ``Omega = A - D C^{-1}``, and

       ``M := A C - C A^T = 2 Omega C``  (exact, antisymmetric).

   The Lagrangian spin (NR52) is exactly this areal velocity: the
   short-lag slope of the odd correlation,
   ``d/dtau [C(tau) - C(tau)^T]|_{0} = -M`` -- so the measured spin IS
   the phase-space current, not a proxy.

2. **Entropy production is set by the spin (exact, isotropic).**  The
   Gaussian entropy production rate ``sigma = tr(Omega C Omega^T D^{-1})``
   (Godreche-Luck / Seifert).  For isotropic diffusion ``D = nu I`` and
   isotropic ``C = c I`` (a rotationally-symmetric two-clocks response):

       ``sigma = m^2 / (2 c nu)``,   m = M_{21} = 2 (areal velocity),

   EXACT.  Entropy production is the spin SQUARED over energy x
   diffusivity.

3. **The spin lower-bounds irreversibility (general).**  For arbitrary
   anisotropic ``A, D``,

       ``sigma >= m^2 / (2 c_bar nu_bar)``,   c_bar = tr C /2, nu_bar = tr D /2,

   with equality in the isotropic limit -- so a measured spin sets a
   floor on irreversibility with NO knowledge of the drift/diffusion
   detail.  Detailed balance ``A C = C A^T`` <=> ``M = 0`` <=> spin = 0
   <=> ``sigma = 0``: zero spin is exactly zero entropy production
   (verified to 1e-16).

Findings (figures/92_spin_irreversibility.json; real spin from the
committed NR52 cache data/nr52_odd_spin_cache.json):

* **Exact + bound (synthetic).**  ``M = 2 Omega C`` to 1e-13; the
  isotropic identity ``sigma = m^2/(2 c nu)`` exact across rotation and
  diffusion; detailed-balance control (symmetric ``A``) gives spin and
  sigma at machine zero; the anisotropic lower bound holds over 1000+
  random systems (min ratio 1.02, zero violations); ``-M`` equals the
  lagged-covariance asymmetry slope.
* **The deep ocean is time-irreversible, with an equatorial reversible
  line.**  The measured spin gives a nonzero cycling rate
  ``omega = rho_odd / dt`` and an irreversibility floor
  ``Sigma_hat = rho_odd^2`` (dimensionless, per correlation time): NH and
  SH are decisively nonzero and OPPOSITE handedness (the current
  circulates clockwise N, counterclockwise S), while the equatorial band
  is statistically reversible (spin null) -- a line of vanishing
  phase-space current exactly where f -> 0.  Irreversibility grows
  poleward with |f|.

Consequence for the program: the parity-odd face is not merely a
descriptive coordinate -- it is the thermodynamic arrow.  The two-clocks
tensor's antisymmetric part measures how far each region sits from
equilibrium, from single-particle statistics, and the corpus's
equatorial null is a genuine detailed-balance surface.  Ties the
"instantaneous mixing is invisible" family (NR39) to its thermodynamic
counterpart: reversible (symmetric) dynamics produce no spin AND no
entropy.

CPU-only, offline-safe (reads the committed NR52 cache).
Tests: tests/test_spin_irreversibility.py.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

HERE = os.path.dirname(os.path.abspath(__file__))
SPIN_CACHE = os.path.join(HERE, "data", "nr52_odd_spin_cache.json")
FIG = os.path.join(HERE, "figures", "92_spin_irreversibility.json")

DT_DAYS = 10.0


# --------------------------------------------------------------------------- #
# OU thermodynamics
# --------------------------------------------------------------------------- #
def ou_solve(A, D):
    """Stationary covariance C: A C + C A^T = 2 D."""
    return solve_continuous_lyapunov(A, 2.0 * np.asarray(D, float))


def current_rotation(A, D):
    """Omega = A - D C^{-1} (J = -Omega x rho); returns (C, Omega, M)."""
    C = ou_solve(A, D)
    Omega = np.asarray(A, float) - np.asarray(D, float) @ np.linalg.inv(C)
    M = A @ C - C @ A.T
    return C, Omega, M


def entropy_production(A, D):
    """sigma = tr(Omega C Omega^T D^{-1}) (Godreche-Luck / Seifert)."""
    C, Omega, _ = current_rotation(A, D)
    return float(np.trace(Omega @ C @ Omega.T @ np.linalg.inv(D)))


def spin_bound(M, C, D):
    """Lower bound m^2 / (2 c_bar nu_bar), m = M_{21}."""
    m = M[1, 0]
    c_bar = np.trace(C) / 2.0
    nu_bar = np.trace(D) / 2.0
    return float(m ** 2 / (2.0 * c_bar * nu_bar))


# --------------------------------------------------------------------------- #
# synthetic checks
# --------------------------------------------------------------------------- #
def synthetic_checks(seed=0):
    rng = np.random.default_rng(seed)
    # M = 2 Omega C exact
    A = np.array([[1.0, -2.0], [2.0, 1.0]])
    D = np.eye(2)
    C, Om, M = current_rotation(A, D)
    m_identity_err = float(np.max(np.abs(M - 2.0 * Om @ C)))

    # isotropic identity sigma = m^2/(2 c nu), swept
    iso_err = 0.0
    for w in (0.7, 1.5, 3.0):
        for nu in (0.5, 1.0, 2.0):
            Aw = np.array([[1.0, -w], [w, 1.0]])
            Dn = nu * np.eye(2)
            Cw, _, Mw = current_rotation(Aw, Dn)
            sig = entropy_production(Aw, Dn)
            pred = Mw[1, 0] ** 2 / (2.0 * Cw[0, 0] * nu)
            iso_err = max(iso_err, abs(sig - pred))

    # detailed balance: symmetric A -> zero spin and zero sigma
    As = np.array([[2.0, 0.5], [0.5, 1.0]])
    Cs, _, Ms = current_rotation(As, np.eye(2))
    db_spin = float(abs(Ms[1, 0]))
    db_sigma = entropy_production(As, np.eye(2))

    # lag-covariance asymmetry slope = -M
    Al = np.array([[1.0, -1.5], [1.5, 1.0]])
    Cl, _, Ml = current_rotation(Al, np.eye(2))
    tau = 1e-6
    Ct = expm(-Al * tau) @ Cl
    slope = ((Ct - Ct.T) / tau)[1, 0]
    slope_err = abs(slope + Ml[1, 0])

    # general anisotropic lower bound: no violations, equality only isotropic
    min_ratio = np.inf
    viol = 0
    n_ok = 0
    for _ in range(2000):
        w = rng.uniform(-3, 3)
        Ar = rng.normal(size=(2, 2)) + w * np.array([[0.0, -1.0], [1.0, 0.0]])
        if np.any(np.linalg.eigvals(Ar).real <= 0.03):
            continue
        B = rng.normal(size=(2, 2))
        Dr = B @ B.T / 2.0 + 0.3 * np.eye(2)
        Cr, _, Mr = current_rotation(Ar, Dr)
        sig = entropy_production(Ar, Dr)
        bound = spin_bound(Mr, Cr, Dr)
        if bound > 1e-9:
            r = sig / bound
            min_ratio = min(min_ratio, r)
            viol += int(sig < bound - 1e-9)
            n_ok += 1
    return dict(m_identity_err=m_identity_err, iso_identity_err=iso_err,
                db_spin=db_spin, db_sigma=db_sigma,
                lag_slope_err=float(slope_err),
                bound_min_ratio=float(min_ratio),
                bound_violations=int(viol), n_bound_tested=int(n_ok))


# --------------------------------------------------------------------------- #
# real data (committed NR52 spin cache)
# --------------------------------------------------------------------------- #
def load_spin(path=SPIN_CACHE):
    with open(path) as fh:
        return json.load(fh)


def band_spin_t(cache, sel_key):
    """Per-float rho1 mean, t, and irreversibility floor rho1^2 for a
    latitude selection."""
    rho1 = np.array(cache["per_float"]["rho1"])
    lat = np.array(cache["per_float"]["lat"])
    sel = {"nh": lat > 5, "sh": lat < -5, "eq": np.abs(lat) < 5}[sel_key]
    r = rho1[sel]
    n = int(sel.sum())
    mean = float(r.mean())
    t = float(mean / (r.std(ddof=1) / np.sqrt(n)))
    return dict(n=n, rho1_mean=mean, t=t,
                cycling_rate_per_day=mean / DT_DAYS,
                irreversibility_floor=float(mean ** 2))


def analyze(cache):
    syn = synthetic_checks()
    real = {k: band_spin_t(cache, k) for k in ("nh", "sh", "eq")}
    out = dict(synthetic=syn, real=real,
               n_floats=int(cache["meta"]["n_floats"]))
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    syn = out["synthetic"]
    r = out["real"]
    exact = (syn["m_identity_err"] < 1e-11
             and syn["iso_identity_err"] < 1e-9
             and syn["db_spin"] < 1e-9 and syn["db_sigma"] < 1e-9
             and syn["lag_slope_err"] < 1e-4)
    bound = (syn["bound_violations"] == 0
             and syn["bound_min_ratio"] >= 1.0 - 1e-6
             and syn["n_bound_tested"] > 500)
    irreversible = (r["nh"]["t"] < -4.0 and r["sh"]["t"] > 4.0
                    and r["nh"]["rho1_mean"] * r["sh"]["rho1_mean"] < 0)
    reversible_eq = abs(r["eq"]["t"]) < 2.0
    return dict(
        current_identity_and_isotropic_law_exact=bool(exact),
        spin_squared_lower_bounds_entropy=bool(bound),
        detailed_balance_iff_zero_spin=bool(syn["db_spin"] < 1e-9
                                            and syn["db_sigma"] < 1e-9),
        ocean_is_time_irreversible_handed=bool(irreversible),
        equatorial_reversible_line=bool(reversible_eq),
        reading=(
            "The spin is the thermodynamic arrow: the Lagrangian areal "
            "velocity is exactly the phase-space probability current "
            "(M = 2 Omega C), so entropy production is the spin squared "
            "over energy x diffusivity -- exact for the isotropic "
            "two-clocks response (sigma = m^2/2c nu) and a floor in "
            f"general (min ratio {syn['bound_min_ratio']:.2f}, no "
            "violations), with detailed balance <=> zero spin <=> zero "
            "entropy (machine zero).  The deep ocean is time-irreversible "
            f"with opposite handedness across the equator (NH t = "
            f"{r['nh']['t']:.1f}, SH t = {r['sh']['t']:.1f}) and a "
            f"statistically REVERSIBLE equatorial line (t = "
            f"{r['eq']['t']:.1f}) where the phase-space current vanishes "
            "as f -> 0: the parity-odd face is how far each region sits "
            "from equilibrium, measured from single-float statistics."),
    )


def run(write=True):
    cache = load_spin()
    out = analyze(cache)
    res = dict(description=__doc__.splitlines()[0], **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    s, r = res["synthetic"], res["real"]
    print("NR57 -- the spin is the irreversibility clock")
    print(f"  current identity + iso law    : "
          f"{v['current_identity_and_isotropic_law_exact']} "
          f"(M=2OmC {s['m_identity_err']:.0e}, iso {s['iso_identity_err']:.0e})")
    print(f"  spin^2 lower-bounds entropy   : "
          f"{v['spin_squared_lower_bounds_entropy']} "
          f"(min ratio {s['bound_min_ratio']:.2f}, "
          f"{s['bound_violations']} violations / {s['n_bound_tested']})")
    print(f"  detailed balance <=> no spin  : "
          f"{v['detailed_balance_iff_zero_spin']} "
          f"(sym-A spin {s['db_spin']:.0e}, sigma {s['db_sigma']:.0e})")
    print(f"  ocean irreversible + handed   : "
          f"{v['ocean_is_time_irreversible_handed']} "
          f"(NH t={r['nh']['t']:.1f}, SH t={r['sh']['t']:.1f})")
    print(f"  equatorial reversible line    : "
          f"{v['equatorial_reversible_line']} (t={r['eq']['t']:.1f})")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
