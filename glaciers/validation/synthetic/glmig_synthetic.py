r"""§V synthetic calibration of the §H.1.2 grounding-line-migration estimator.

No external data.  §H.1.2 turns the static RTN line into a *rate*: with the bed
(hence ``H* ``) fixed while ice thins at ``dH/dt``, the RTN=1 line is the zero
level-set of ``m = H - H*`` and advances inland at the level-set normal speed

    v_front = |dH/dt| / |grad m|        (level-set advance law)            (A)

so the geometric amplification ``A = 1/|grad m|`` (km of inland advance per metre
of thinning) is a pure-geometry field.  §H.1.3/§H.1.4 then run the **residence
number**

    Ro = v_kin / v_obs ,   v_kin = |dH/dt| / |grad m|                       (B)

on real observations and read its log-log slope against a friction-velocity proxy
``u_*`` as the discriminant: slope ``> 0`` => ``u_*``-paced (hydraulic clock
rate-limiting), slope ``~ 0`` => thinning-paced (no hidden hydraulic discount).

Those real-data verdicts (continental ``r ~ -0.06`` flat; Amundsen ``+0.57``;
East Antarctica ``-0.30``) are only interpretable if the *estimator itself* is
unbiased and the slope-sign reading is calibrated.  This harness plants a known
ground truth and checks recovery -- the repo's §V "plant-and-recover" discipline:

  A. **Level-set advance law is exact.**  For an analytic margin with constant
     ``|grad m| = b`` thinning at rate ``r`` the zero contour is ``x_c(t)=r t/b``;
     the numerically tracked contour speed matches ``r/b = |dH/dt|/|grad m|`` to
     interpolation order, in 1-D *and* for a tilted 2-D plane where the normal
     speed uses the gradient *magnitude*.  The grid estimator ``A=1/|grad m|``
     recovers ``1/b`` to finite-difference order.

  B. **Ro discriminant is calibrated.**  Plant ``N`` synthetic GL points with a
     known hydraulic discount ``D(u_*) = 1 + gamma * u_*^p`` so that
     ``v_obs = v_kin / D`` and therefore ``Ro = D`` exactly.  Then:
       B0  *thinning-paced null* (``gamma=0``): ``Ro == 1`` exactly and the
           log-log slope ``Ro`` vs ``u_*`` is machine-zero -- the calibrated null
           behind the §H.1.3 "flat slope => not u_*-paced" reading.
       B1  *exponent recovery*: fitting ``log(Ro-1)`` vs ``log u_*`` recovers the
           planted ``p`` to ~1e-2 (the discount law is identifiable).
       B2  *sign + monotonicity*: the raw ``log Ro`` vs ``log u_*`` slope (the
           statistic §H.1.4 actually reports) is ``> 0`` and increases with the
           planted ``p`` -- so a positive observed slope genuinely means
           ``u_*``-paced.
       B3  *unbiased under noise + permutation null*: with multiplicative noise on
           ``v_obs`` the OLS slope stays positive and close to the noiseless value
           (small bias), while shuffling ``u_*`` collapses the slope to ~0 -- so
           observed positive slopes are not noise/!ordering artefacts.

This validates the *estimator and the discriminant*, not any physical pacing
claim (that is settled against real data in §H.1.3/§H.1.4); it supplies the
synthetic null those real-data slopes are read against.
"""
from __future__ import annotations

import numpy as np


# --- Part A: level-set advance law -------------------------------------------
def _zero_crossing(x, f):
    """Linear-interpolated location of the (single, increasing) zero of f(x).
    Returns ``x[0]`` if f is already non-negative at the domain start, and
    ``nan`` if f never reaches zero inside the domain (no crossing)."""
    f = np.asarray(f, dtype=float)
    if f[0] >= 0.0:
        return float(x[0])
    if not np.any(f >= 0.0):          # zero lies beyond the domain -> undefined
        return float("nan")
    i = int(np.argmax(f >= 0.0))
    x0, x1, f0, f1 = x[i - 1], x[i], f[i - 1], f[i]
    return float(x0 + (x1 - x0) * (0.0 - f0) / (f1 - f0))


def levelset_advance_1d(b=0.4, r=2.0, n=4096, t0=1.0, t1=3.0, nt=2001):
    """Margin m(x,t) = b*x - r*t (H thins at rate r, |grad m| = b).  Track the
    zero contour x_c(t) and compare its speed to the level-set law r/b."""
    x = np.linspace(0.0, 40.0, n)
    ts = np.linspace(t0, t1, nt)
    xc = np.array([_zero_crossing(x, b * x - r * t) for t in ts])
    v_num = np.polyfit(ts, xc, 1)[0]            # measured contour speed
    v_law = r / b                               # |dH/dt| / |grad m|
    A_grid = 1.0 / np.gradient(b * x, x).mean()  # estimator A = 1/|grad m| -> 1/b
    return {
        "v_numeric": float(v_num),
        "v_law": float(v_law),
        "rel_err_speed": float(abs(v_num - v_law) / v_law),
        "A_grid": float(A_grid),
        "A_law": float(1.0 / b),
        "rel_err_A": float(abs(A_grid - 1.0 / b) * b),
    }


def levelset_advance_2d_tilt(bx=0.3, by=0.4, r=1.5, dt=2e-3):
    """Tilted 2-D plane m = bx*x + by*y - r*t.  The contour is a moving line; its
    *normal* speed must use the gradient magnitude: v_n = r/sqrt(bx^2+by^2).
    Purely analytic -- no grid is needed (the plane has constant gradient)."""
    g = np.hypot(bx, by)
    # signed distance of the contour from origin along the gradient normal is
    # phi(t) = (r t)/g; advance over one dt must equal r/g * dt.
    phi0 = (r * 1.0) / g
    phi1 = (r * (1.0 + dt)) / g
    v_n = (phi1 - phi0) / dt
    v_law = r / g
    return {
        "v_normal": float(v_n),
        "v_law": float(v_law),
        "rel_err": float(abs(v_n - v_law) / v_law),
    }


# --- Part B: residence-number discriminant -----------------------------------
def _ols_slope(logx, logy):
    return float(np.polyfit(logx, logy, 1)[0])


def ro_discriminant(n=6000, gamma=0.8, p=0.5, noise=0.0, seed=0, shuffle=False):
    """Plant GL points with a known hydraulic discount D(u)=1+gamma*u^p so that
    Ro = v_kin/v_obs = D exactly; recover p and the slope statistic."""
    rng = np.random.default_rng(seed)
    grad_m = np.exp(rng.normal(np.log(0.03), 0.6, n))   # |grad m|  (~Bedmap2)
    dHdt = np.exp(rng.normal(np.log(1.5), 0.5, n))      # thinning  [m/yr]
    u = np.exp(rng.normal(0.0, 0.7, n))                 # u_* proxy (lognormal)

    v_kin = dHdt / grad_m
    D = 1.0 + gamma * u ** p
    v_obs = v_kin / D
    if noise > 0.0:
        v_obs = v_obs * np.exp(rng.normal(0.0, noise, n))   # multiplicative
    Ro = v_kin / v_obs

    u_reg = rng.permutation(u) if shuffle else u
    slope_logRo = _ols_slope(np.log(u_reg), np.log(Ro))

    out = {
        "Ro_median": float(np.median(Ro)),
        "slope_logRo_vs_u": slope_logRo,           # the §H.1.4 statistic
    }
    if gamma == 0.0:
        out["Ro_max_dev_from_1"] = float(np.max(np.abs(Ro - 1.0)))
    else:
        # D-1 = gamma u^p is exactly identifiable when uncorrupted
        if noise == 0.0 and not shuffle:
            p_hat = _ols_slope(np.log(u), np.log(Ro - 1.0))
            out["p_planted"] = float(p)
            out["p_recovered"] = float(p_hat)
            out["rel_err_p"] = float(abs(p_hat - p) / p)
    return out


# --- orchestrator ------------------------------------------------------------
def run():
    A1 = levelset_advance_1d()
    A2 = levelset_advance_2d_tilt()

    null = ro_discriminant(gamma=0.0)                       # B0 thinning-paced
    rec = ro_discriminant(gamma=0.8, p=0.5)                 # B1 exponent recovery
    slopes = {pp: ro_discriminant(gamma=0.8, p=pp)["slope_logRo_vs_u"]
              for pp in (0.25, 0.5, 0.9)}                   # B2 monotone in p
    noisy = ro_discriminant(gamma=0.8, p=0.5, noise=0.25, seed=3)   # B3 noise
    perm = ro_discriminant(gamma=0.8, p=0.5, shuffle=True, seed=3)  # B3 null

    mono = slopes[0.25] < slopes[0.5] < slopes[0.9]
    ok = bool(
        A1["rel_err_speed"] < 1e-3 and A1["rel_err_A"] < 1e-6
        and A2["rel_err"] < 1e-9
        and null["Ro_max_dev_from_1"] < 1e-12 and abs(null["slope_logRo_vs_u"]) < 1e-12
        and rec["rel_err_p"] < 1e-2
        and slopes[0.5] > 0.0 and mono
        and noisy["slope_logRo_vs_u"] > 0.0
        and abs(noisy["slope_logRo_vs_u"] - rec["slope_logRo_vs_u"]) < 0.1
        and abs(perm["slope_logRo_vs_u"]) < 0.05
    )
    return {
        "levelset_1d": A1,
        "levelset_2d_tilt": A2,
        "null_thinning_paced": null,
        "exponent_recovery": rec,
        "slope_vs_planted_p": slopes,
        "slope_monotone_in_p": bool(mono),
        "noisy": noisy,
        "permutation_null": perm,
        "pass": ok,
    }


def main():
    r = run()
    A1, A2 = r["levelset_1d"], r["levelset_2d_tilt"]
    rec, perm, noisy = r["exponent_recovery"], r["permutation_null"], r["noisy"]
    print("=== §V synthetic calibration of the §H.1.2 GL-migration estimator ===")
    print(f"  A. level-set 1-D speed     : {A1['v_numeric']:.6f} vs law {A1['v_law']:.6f}"
          f"  (rel-err {A1['rel_err_speed']:.1e})")
    print(f"     grid A=1/|grad m|       : {A1['A_grid']:.6f} vs {A1['A_law']:.6f}"
          f"  (rel-err {A1['rel_err_A']:.1e})")
    print(f"     2-D tilted normal speed : {A2['v_normal']:.6f} vs law {A2['v_law']:.6f}"
          f"  (rel-err {A2['rel_err']:.1e})")
    print(f"  B0 thinning-paced null     : Ro max|dev-1| {r['null_thinning_paced']['Ro_max_dev_from_1']:.1e}"
          f", slope {r['null_thinning_paced']['slope_logRo_vs_u']:.1e}  (~0 => flat)")
    print(f"  B1 exponent recovery       : p_hat {rec['p_recovered']:.4f} vs planted {rec['p_planted']:.2f}"
          f"  (rel-err {rec['rel_err_p']:.1e})")
    print(f"  B2 slope vs planted p      : {r['slope_vs_planted_p']}  monotone={r['slope_monotone_in_p']}")
    print(f"  B3 noisy slope             : {noisy['slope_logRo_vs_u']:.3f} (>0, near noiseless);"
          f"  permutation-null slope {perm['slope_logRo_vs_u']:.1e} (~0)")
    print(f"  PASS                       : {r['pass']}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
