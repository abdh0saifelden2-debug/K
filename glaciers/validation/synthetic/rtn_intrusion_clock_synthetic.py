r"""§V.3 synthetic calibration of the §H.1.2 intrusion-clock **driver code path**.

No external data; CPU only.  §H.1.2 (`validation/external/rtn_intrusion_clock.py`)
maps the rate companion to the static §H.1 / §H.1.1 RTN results: the RTN=1 line is
the zero level-set of the margin ``m = H − H*`` (``H* = (ρ_w/φρ_i)·d_base``), so as
grounded ice thins at ``dH/dt`` (bed, hence ``H*``, fixed) the front advances inland
at the **level-set speed**

    v_front = (dH/dt) / |∇m| ,      A = 1/|∇m|   [km inland advance per m thinning]

The level-set *advance law itself* is already certified synthetically by
``glmig_synthetic`` (RESULT 16): 1-D contour tracking, a 2-D tilted-plane normal
speed, and the ``Ro = v_kin/v_obs`` discriminant.  But that harness reimplements the
gradient **inline in 1-D** — it never runs the *actual 2-D driver functions*
(`margin_field` / `amplification`) that execute on Bedmap2, so a regression in the
real code path (a broken ``dx`` scaling, gradient scheme or grounded-mask handling)
would slip past it.  This harness closes that gap: it imports and exercises the
**real driver functions** (the ones `analyse` calls) on analytic geometries whose
front kinematics are known in closed form, and adds three numerical properties
glmig's analytic-only Part A does not cover — discretization **convergence order**,
**isotropy** on a curved front, and the **identity of the advected object with the
§H.1.1 RTN=1 line**.  The dynamical #5 conjecture (advance rate-limited by hydraulic
residence time) stays **[HYP]** — settled against real data in §H.1.3/§H.1.4, not
here.  Checks:

  1. **Exact on a planar margin.** For ``m = s·x`` the central-difference ``|∇m|``
     is exact, so ``A = 1/s`` and ``v_front = (dH/dt)/s`` are recovered to machine
     precision — and equal the analytic front speed ``dx_f/dt`` cell-for-cell.
  2. **Second-order convergent on a curved margin.** For a smooth non-polynomial
     margin ``m = s·x + b·sin(kx)`` the estimator error in ``A`` falls like the grid
     spacing squared (≈4× when the grid is halved) toward the analytic
     ``1/|s + bk·cos(kx)|`` — so it is consistent, not just exact on the trivial case.
  3. **Isotropic on a curved (radial) front.** For ``m = s·(r − r₀)`` the RTN=1 line
     is a circle with analytic ``|∇m| = s`` everywhere; the estimator recovers
     ``A = 1/s`` around the ring with no axis-vs-diagonal directional bias.
  4. **Recovers the planted front advance.** Thinning by ``ΔH`` moves the analytic
     RTN=1 line to the zero set of ``m − ΔH``; advancing each front cell inland by
     the estimated displacement ``A·ΔH`` lands on that new line (exactly for the
     planar case, to discretization for the curved one) — the level-set advance is
     a faithful inverse.
  5. **The advected object is the §H.1.1 RTN=1 line.** On a planted ``(H, bed)``
     population the ``margin_field`` zero set equals ``classify(build_rtn)`` (the
     RTN>1 boundary) cell-for-cell, and ``H* = H_flot/φ`` exactly — so the clock
     advances the *same* line the static φ/MISI results key off.

This validates the **driver estimator (the kinematics)**, not the physics: it makes
no claim about real front-migration rates or the [HYP] hydraulic-pacing conjecture.
The guarantee is structural — *given* the level-set definition, the real driver's
``A = 1/|∇m|`` and ``v_front = A·(dH/dt)`` are a convergent, isotropic readout of the
true front kinematics, acting on exactly the §H.1.1 RTN=1 line.  It is the
code-path-level companion to ``glmig_synthetic`` (RESULT 16), not a replacement.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external.rtn_intrusion_clock import (  # noqa: E402
    margin_field, amplification, DHDT_BAND,
)
from external.run_rtn_bedmap2 import build_rtn, classify, RHO_W, RHO_I, G  # noqa: E402
from validators.rtn_validator import RTN_PW_EPS_PA as _RTN_EPS_PA  # noqa: E402

# margin slope ~ a few m thickness per km, matching the real driver's A band
# (median A≈0.09 km/m ⇒ |∇m|≈11; p90 A≈0.42 ⇒ |∇m|≈2.4): s=5 sits inside it.
_SLOPE = 5.0          # [m thickness / km]
_DX_KM = 1.0
# ``_RTN_EPS_PA`` is the p_w floor of validators.rtn_validator.rtn (imported, not
# re-hardcoded): at/below it RTN is forced to +inf, so the geometric margin can
# legitimately disagree there.  Importing keeps the tie-in test in lock-step if
# the validator's floor ever changes.


def _coords(nx, ny, dx_km):
    """x, y coordinate grids [km] with x varying along axis 1 (columns)."""
    x = (np.arange(nx) * dx_km)[None, :] * np.ones((ny, 1))
    y = (np.arange(ny) * dx_km)[:, None] * np.ones((1, nx))
    return x, y


def _interior(shape, pad=3):
    """Boundary-excluding mask (np.gradient is one-sided/1st-order at edges)."""
    m = np.zeros(shape, dtype=bool)
    m[pad:-pad, pad:-pad] = True
    return m


def planar_margin(nx=200, ny=160, dx_km=_DX_KM, slope=_SLOPE, x0_km=100.0):
    """``m = slope·(x − x0)`` (constant in y).  Analytic ``|∇m| = slope``, RTN=1
    line at ``x = x0``, front speed under thinning ``dH/dt`` is ``dH/dt / slope``."""
    x, _ = _coords(nx, ny, dx_km)
    margin = slope * (x - x0_km)
    return margin, np.full(margin.shape, slope), x, x0_km


def sinusoidal_margin(nx, ny, dx_km, slope=_SLOPE, b=10.0, wavelength_km=40.0,
                      x0_km=None):
    """``m = slope·x + b·sin(k x)`` with ``k = 2π/λ`` and ``slope > b·k`` (so ``m``
    is monotone in x and ``|∇m|>0``).  Analytic ``|∇m| = |slope + b·k·cos(k x)|``."""
    if x0_km is None:
        x0_km = 0.0
    k = 2.0 * np.pi / wavelength_km
    assert slope > b * k, "need slope > b*k for a monotone, single-valued front"
    x, _ = _coords(nx, ny, dx_km)
    xs = x - x0_km
    margin = slope * xs + b * np.sin(k * xs)
    grad_analytic = np.abs(slope + b * k * np.cos(k * xs))
    return margin, grad_analytic, x


def radial_margin(nx=200, ny=200, dx_km=_DX_KM, slope=_SLOPE, r0_km=60.0):
    """``m = slope·(r − r0)`` about the grid centre.  Analytic ``|∇m| = slope``
    everywhere (r>0); the RTN=1 line is the circle ``r = r0``."""
    x, y = _coords(nx, ny, dx_km)
    cx = (nx - 1) * dx_km / 2.0
    cy = (ny - 1) * dx_km / 2.0
    r = np.hypot(x - cx, y - cy)
    margin = slope * (r - r0_km)
    return margin, r, (cx, cy)


def run_planar(slope=_SLOPE, dx_km=_DX_KM):
    """(1) Exact recovery of A=1/s and v=(dH/dt)/s on a planar margin."""
    margin, grad_true, x, x0 = planar_margin(slope=slope, dx_km=dx_km)
    grounded = np.ones(margin.shape, dtype=bool)
    grad, amp = amplification(margin, grounded, dx_km)
    intr = _interior(margin.shape)

    grad_err = float(np.max(np.abs(grad[intr] - grad_true[intr])))
    amp_err = float(np.max(np.abs(amp[intr] - 1.0 / slope)))
    # level-set front speed v = A·(dH/dt) vs analytic dx_f/dt = dH/dt / slope
    v_err = 0.0
    for r in DHDT_BAND:
        v_est = amp[intr] * r
        v_err = max(v_err, float(np.max(np.abs(v_est - r / slope))))
    ok = grad_err <= 1e-9 and amp_err <= 1e-12 and v_err <= 1e-12
    return {
        "slope_m_per_km": slope, "A_analytic_km_per_m": 1.0 / slope,
        "grad_max_abs_err": grad_err, "amp_max_abs_err": amp_err,
        "v_front_max_abs_err": v_err,
        "v_front_km_per_yr": {f"dHdt_{r}_m_yr": r / slope for r in DHDT_BAND},
        "pass": bool(ok),
    }


def run_convergence(slope=_SLOPE, b=10.0, wavelength_km=40.0):
    """(2) Estimator error in A falls ~2nd order (≈4×) as the grid is halved."""
    L = 200.0  # physical domain [km]
    errs, dxs = [], []
    for n in (200, 400):
        dx = L / n
        margin, grad_true, _x = sinusoidal_margin(
            n, 24, dx, slope=slope, b=b, wavelength_km=wavelength_km)
        grounded = np.ones(margin.shape, dtype=bool)
        _grad, amp = amplification(margin, grounded, dx)
        intr = _interior(margin.shape)
        amp_true = 1.0 / grad_true
        errs.append(float(np.max(np.abs(amp[intr] - amp_true[intr]))))
        dxs.append(dx)
    ratio = errs[0] / errs[1] if errs[1] > 0 else float("inf")
    ok = errs[1] < errs[0] and ratio >= 3.0 and errs[1] < 5e-3
    return {
        "dx_km": dxs, "amp_max_abs_err": errs,
        "error_ratio_coarse_over_fine": float(ratio),
        "expected_ratio_2nd_order": 4.0, "pass": bool(ok),
    }


def run_isotropy(slope=_SLOPE, dx_km=_DX_KM, r0_km=60.0, band_km=2.0):
    """(3) On the radial RTN=1 ring, A=1/s is recovered with no directional bias."""
    margin, r, (cx, cy) = radial_margin(slope=slope, dx_km=dx_km, r0_km=r0_km)
    grounded = np.ones(margin.shape, dtype=bool)
    _grad, amp = amplification(margin, grounded, dx_km)
    intr = _interior(margin.shape, pad=4)

    ny, nx = margin.shape
    x, y = _coords(nx, ny, dx_km)
    ring = intr & (np.abs(r - r0_km) < band_km) & np.isfinite(amp)
    a_ring = amp[ring]
    med = float(np.median(a_ring))
    rel_med_err = abs(med - 1.0 / slope) * slope
    cv = float(np.std(a_ring) / np.mean(a_ring))

    # isotropy: median A near the axes vs near the diagonals around the ring
    ang = np.arctan2(y - cy, x - cx)
    a4 = np.abs(((ang % (np.pi / 2)) / (np.pi / 2)) - 0.5)  # 0.5 at axis, 0 at diag
    axis = ring & (a4 > 0.4)
    diag = ring & (a4 < 0.1)
    med_axis = float(np.median(amp[axis]))
    med_diag = float(np.median(amp[diag]))
    aniso = abs(med_axis - med_diag) / (0.5 * (med_axis + med_diag))

    ok = rel_med_err <= 0.02 and cv <= 0.03 and aniso <= 0.02
    return {
        "n_ring_cells": int(ring.sum()),
        "A_median_km_per_m": med, "A_analytic_km_per_m": 1.0 / slope,
        "rel_median_err": float(rel_med_err), "ring_cv": cv,
        "A_median_axis": med_axis, "A_median_diag": med_diag,
        "anisotropy": float(aniso), "pass": bool(ok),
    }


def run_advance_recovery(slope=_SLOPE, dx_km=_DX_KM, dH_list=(20.0, 50.0)):
    """(4) Advancing the front by the estimated A·ΔH lands on the analytic new
    RTN=1 line (zero set of ``m − ΔH``).  Planar ⇒ exact."""
    margin, _grad_true, x, x0 = planar_margin(slope=slope, dx_km=dx_km)
    grounded = np.ones(margin.shape, dtype=bool)
    _grad, amp = amplification(margin, grounded, dx_km)
    intr = _interior(margin.shape)
    A = float(np.median(amp[intr]))   # = 1/slope

    pos_err = 0.0
    rows = []
    for dH in dH_list:
        x_new_analytic = x0 + dH / slope         # zero set of m - dH
        x_new_est = x0 + A * dH                   # advance front by estimated A·ΔH
        e = abs(x_new_est - x_new_analytic)
        pos_err = max(pos_err, e)
        # the thinned-margin RTN=1 boundary sits at x_new_analytic to ≤1 cell
        thinned = margin - dH
        # leftmost grounded-safe column index (margin>=0), per the constant-in-y field
        col = int(np.argmax(thinned[thinned.shape[0] // 2, :] >= 0.0))
        boundary_x = col * dx_km
        rows.append({"dH_m": dH, "x_new_analytic_km": x_new_analytic,
                     "x_new_estimated_km": x_new_est,
                     "boundary_mask_x_km": boundary_x,
                     "mask_err_km": abs(boundary_x - x_new_analytic)})
    mask_err = max(r["mask_err_km"] for r in rows)
    ok = pos_err <= 1e-9 and mask_err <= dx_km
    return {"A_km_per_m": A, "front_pos_max_abs_err_km": float(pos_err),
            "mask_boundary_max_err_km": float(mask_err), "rows": rows,
            "pass": bool(ok)}


def run_rtn_tie_in(phi=0.9, n=300, seed=0):
    """(5) The advected margin zero set IS the §H.1.1 RTN=1 line: ``H*=H_flot/φ``
    exactly and ``margin>=0 ⇔ ¬(RTN>1)`` cell-for-cell on a planted population.

    The equivalence is exact wherever the subglacial water pressure is above the
    validator's numerical floor (``p_w = φ·ρ_i g H > eps``, where ``eps`` is the
    ``_RTN_EPS_PA`` constant from ``validators.rtn_validator.rtn``, currently
    1 Pa).  At/below that floor the validator forces
    RTN=+inf (intrusion-favoured) regardless of the geometric margin sign — a
    deliberate, safe over-prediction for ice within ~0.1 mm of flotation.  We
    therefore score the cell-for-cell identity over cells above the floor, so the
    guarantee is seed-independent instead of relying on the planted population
    happening to land above ``eps`` (it would otherwise be a conditional claim).
    """
    rng = np.random.default_rng(seed)
    d_base = rng.uniform(0.0, 1200.0, size=(n,))          # bed depth below sea level [m]
    bed = -d_base
    h_flot = (RHO_W / RHO_I) * d_base
    # thickness straddling H* = h_flot/phi so both classes are populated
    H = (h_flot / phi) * rng.uniform(0.5, 1.5, size=(n,))

    margin, H_star, db = margin_field(H, bed, phi)
    hstar_err = float(np.max(np.abs(H_star - h_flot / phi)))
    # RTN>1 boundary from the real classifier
    rtn_gt1 = classify(build_rtn(H, bed, phi))
    # exclude sub-eps (≈flotation) cells where the validator's +inf floor can
    # legitimately disagree with the pure-geometric margin sign (see docstring).
    above_floor = (phi * RHO_I * G * H) > _RTN_EPS_PA
    # margin<0 (below critical thickness) must equal RTN>1, cell-for-cell
    mismatch = int(np.sum((margin < 0.0)[above_floor] != rtn_gt1[above_floor]))
    n_floored = int(np.sum(~above_floor))
    ok = hstar_err <= 1e-6 and mismatch == 0
    return {"phi": phi, "n": n, "H_star_vs_Hflot_over_phi_max_err": hstar_err,
            "margin_sign_vs_rtn_mismatch_cells": mismatch,
            "n_subeps_floored_cells": n_floored, "pass": bool(ok)}


def run():
    """Assemble the intrusion-clock estimator calibration.  Top-level ``pass``."""
    planar = run_planar()
    conv = run_convergence()
    iso = run_isotropy()
    adv = run_advance_recovery()
    tie = run_rtn_tie_in()
    ok = all(d["pass"] for d in (planar, conv, iso, adv, tie))
    return {"planar": planar, "convergence": conv, "isotropy": iso,
            "advance_recovery": adv, "rtn_tie_in": tie, "pass": bool(ok)}


def main():
    out = run()
    p = out["planar"]
    print("§H.1.2 intrusion-clock estimator — synthetic calibration "
          "(real margin_field/amplification math)")
    print(f"(1) planar  s={p['slope_m_per_km']:g} m/km: A=1/|∇m| "
          f"err={p['amp_max_abs_err']:.1e} km/m, grad err={p['grad_max_abs_err']:.1e}, "
          f"v_front err={p['v_front_max_abs_err']:.1e}")
    print("    v_front=A·dH/dt [km/yr]:",
          {k: round(v, 3) for k, v in p["v_front_km_per_yr"].items()})
    c = out["convergence"]
    print(f"(2) curved (sinusoid): A err {c['amp_max_abs_err'][0]:.2e}"
          f"->{c['amp_max_abs_err'][1]:.2e} as dx {c['dx_km'][0]:g}->{c['dx_km'][1]:g} km, "
          f"ratio={c['error_ratio_coarse_over_fine']:.2f} (2nd order ⇒ ~4)")
    i = out["isotropy"]
    print(f"(3) radial ring ({i['n_ring_cells']} cells): A median={i['A_median_km_per_m']:.4f} "
          f"(analytic {i['A_analytic_km_per_m']:.4f}), CV={i['ring_cv']:.2e}, "
          f"anisotropy axis-vs-diag={i['anisotropy']:.2e}")
    a = out["advance_recovery"]
    print(f"(4) advance recover: front-position err={a['front_pos_max_abs_err_km']:.1e} km, "
          f"thinned-mask boundary err≤{a['mask_boundary_max_err_km']:.2g} km")
    t = out["rtn_tie_in"]
    print(f"(5) RTN tie-in: H*=H_flot/φ err={t['H_star_vs_Hflot_over_phi_max_err']:.1e}, "
          f"margin-sign vs RTN>1 mismatch={t['margin_sign_vs_rtn_mismatch_cells']} cells")
    print(f"PASS={out['pass']}")


if __name__ == "__main__":
    main()
