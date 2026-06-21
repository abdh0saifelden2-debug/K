r"""§H.1.2 — the *intrusion clock*: how fast the RTN=1 line advances inland as
grounded ice thins (the rate companion to the static §H.1 / §H.1.1 results).

Motivation (FUTURE_WORK §G.3 → §G.2 → §G.4).  RTN says *where* intrusion is
favoured; it has no time in it.  The RTN=1 boundary is the zero level-set of the
margin ``m = H − H*`` with ``H* = (ρ_w/φρ_i)·d_base`` (the local critical
thickness, §H.1.1).  As ice thins at rate ``dH/dt`` (with the bed, hence ``H*``,
fixed on these timescales) the front moves inland at the **level-set speed**

    v_front = (dH/dt) / |∇m|            [m yr⁻¹ horizontal]

so the *geometric amplification* ``A = 1/|∇m|`` (km of inland advance per metre
of thinning) is a pure-geometry field we can map from Bedmap2.  Two regimes:
flat margins (small ``|∇m|`` → large ``A``) run away on small thinning, while
steep margins (bed/thickness rolls, ``|∇m|`` large → small ``A``) **pin** the
line — the geometric fingerprint of ice-plain vs pinning-point behaviour.

We also map the §G.4 thermal memory ``τ_ice = H²/κ_ice`` to test the *necessary*
co-location: the runaway (high-A) front sits on thin ice with the **shortest**
memory, exactly where a hydraulic clock could pace it.

**What this does and does not settle.**  The kinematic field ``A`` and the
predicted ``v_front`` for a literature thinning band are computed here (data:
Bedmap2 only).  The *dynamical* claim of #5 — that the advance is rate-limited by
the subglacial hydraulic residence time, itself governed by the same
``u_*``-controlled wall flux as the §G.2 scallop-migration √-law — needs a
channel-resolved solver run or an observed front-migration rate (repeat radar /
ITS_LIVE), neither resolved at 3 km.  It is stated as the falsifiable conjecture,
not verified here.

Run::

    python validation/external/rtn_intrusion_clock.py --stride 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from validators.rtn_validator import RHO_I  # noqa: E402
from external.run_rtn_bedmap2 import RHO_W, distance_to_groundingline_km  # noqa: E402

KAPPA_ICE = 1.09e-6        # ice thermal diffusivity [m^2 s^-1] (k_ice/(ρ_i·cp))
SEC_PER_YR = 3.1557e7
# literature grounded-ice thinning band (WAIS/Amundsen altimetry, m yr^-1)
DHDT_BAND = (0.5, 1.0, 2.0, 5.0)


def margin_field(H, bed, phi):
    r"""§H.1.1 margin ``m = H − H*`` whose zero set is the RTN=1 line.

    ``H* = (ρ_w/φρ_i)·d_base`` with ``d_base = max(0, −bed)`` (the bed depth below
    sea level).  ``m>0`` is grounded-safe; ``m=0`` is the RTN=1 boundary.  Returns
    ``(margin, H_star, d_base)``.  Identical math to the §H.1.1 critical thickness
    (``H* = H_flot/φ``) so the field driver and the synthetic calibration share
    one equation.
    """
    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    H_star = (RHO_W / (phi * RHO_I)) * d_base       # RTN=1 critical thickness [m]
    return H - H_star, H_star, d_base


def amplification(margin, grounded, dx_km):
    r"""Geometric amplification ``A = 1/|∇m|`` [km inland advance per m thinning].

    The RTN=1 line is the zero level-set of ``m``; under uniform thinning at
    ``dH/dt`` (bed, hence ``H*``, fixed) it advances normal to itself at the
    level-set speed ``v_front = (dH/dt)/|∇m|``, so ``A = 1/|∇m|`` is a pure-geometry
    field.  ``|∇m|`` is in [m thickness / km horizontal] (``np.gradient`` spacing in
    km), masked to grounded ice.  Returns ``(grad, amp)``.
    """
    gy, gx = np.gradient(np.where(grounded, margin, np.nan), dx_km, dx_km)
    grad = np.hypot(gx, gy)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = 1.0 / grad
    return grad, amp


def analyse(bin_dir, stride=3, phi=0.9, near_tip=0.20):
    from external.bedmap2_loader import load_fields
    d = load_fields(bin_dir, stride=stride)
    H, bed, mask = d["thickness"], d["bed"], d["icemask_grounded_and_shelves"]
    dx_km = d["_meta"]["cellsize"] / 1000.0

    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)
    dist = distance_to_groundingline_km(grounded, d["_meta"]["cellsize"])

    margin, H_star, d_base = margin_field(H, bed, phi)
    grad, amp = amplification(margin, grounded, dx_km)

    # the advancing front = grounded cells close to tipping (0 < m/H < near_tip)
    rel = margin / H
    front = grounded & (rel > 0) & (rel < near_tip) & np.isfinite(amp)
    interior = grounded & (rel >= near_tip)

    tau_yr = (H ** 2 / KAPPA_ICE) / SEC_PER_YR       # G.4 thermal memory [yr]

    def pct(a, p):
        return float(np.nanpercentile(a, p)) if np.isfinite(a).any() else float("nan")

    amp_front = amp[front]
    A_med = float(np.nanmedian(amp_front))
    summary = {
        "stride_km": dx_km, "phi": phi, "near_tip_rel_margin": near_tip,
        "n_front_cells": int(front.sum()),
        "front_dist_to_GL_km": {"median": float(np.median(dist[front])),
                                 "p90": float(np.percentile(dist[front], 90))},
        "amplification_km_per_m": {                  # A = 1/|grad m|
            "p25": pct(amp_front, 25), "median": A_med, "p75": pct(amp_front, 75),
            "p90": pct(amp_front, 90)},
        "front_advance_km_per_yr": {                 # v = A * dH/dt, median A
            f"dHdt_{r}_m_yr": A_med * r for r in DHDT_BAND},
        "tau_ice_yr": {
            "front_median": float(np.nanmedian(tau_yr[front])),
            "interior_median": float(np.nanmedian(tau_yr[interior])),
            "ratio_interior_over_front":
                float(np.nanmedian(tau_yr[interior]) / np.nanmedian(tau_yr[front]))},
        # co-location: share of high-amplification ("runaway") cells near the GL
        "runaway_frac_within_25km":
            float((front & (amp > A_med) & (dist < 25)).sum()
                  / max(1, (front & (amp > A_med)).sum())),
    }

    # (b) concrete test targets for the residence-number measurement:
    # high-amplification (runaway, A>=p90) front cells within 50 km of the GL
    A_p90 = summary["amplification_km_per_m"]["p90"]
    tgt = front & (amp >= A_p90) & (dist < 50)
    nt = int(tgt.sum())
    A_tgt = float(np.nanmedian(amp[tgt])) if nt else float("nan")
    summary["test_targets"] = {
        "A_threshold_km_per_m": A_p90,
        "n_targets": nt,
        "median_A_km_per_m": A_tgt,
        "median_dist_to_GL_km": float(np.median(dist[tgt])) if nt else float("nan"),
        # kinematic ceiling v_kin = A * dH/dt at these targets [km/yr]
        "v_kin_km_per_yr": {f"dHdt_{r}_m_yr": A_tgt * r for r in (1.0, 2.0)},
        "note": ("Survey these cells: measure dH/dt (altimetry) and grounding-line "
                 "migration v_obs (DInSAR). Ro = v_kin/v_obs is the residence number "
                 "(Ro~1 thinning-paced; Ro>>1 hydraulic-limited)."),
    }
    return summary, (H, grounded, dist, margin, rel, amp, tau_yr, front, near_tip, phi)


def make_figure(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, grounded, dist, margin, rel, amp, tau_yr, front, near_tip, phi = payload

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # left: amplification A = 1/|grad m| on the near-tipping front (log km/m)
    a = np.where(front, amp, np.nan)
    la = np.ma.masked_invalid(np.log10(np.clip(a, 1e-3, 1e3)))
    im = axL.imshow(la, cmap="magma", origin="upper", vmin=-1, vmax=2)
    axL.set_title(f"(#5) advance amplification  A=1/|∇m|  [km per m thinning]\n"
                  f"near-tipping front (0<m/H<{near_tip}), φ={phi}; bright = runaway")
    axL.set_xticks([]); axL.set_yticks([])
    cb = fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04)
    cb.set_label("log₁₀ A  [km / m]")

    # right: G.4 thermal memory tau_ice = H^2/kappa on grounded ice (log yr)
    t = np.ma.masked_invalid(np.where(grounded, np.log10(tau_yr), np.nan))
    im2 = axR.imshow(t, cmap="viridis", origin="upper")
    axR.contour(front.astype(float), levels=[0.5], colors="r", linewidths=0.3)
    axR.set_title("§G.4 thermal memory  τ_ice = H²/κ  [yr]\n"
                  "red = near-tipping front (shortest-memory thin ice)")
    axR.set_xticks([]); axR.set_yticks([])
    cb2 = fig.colorbar(im2, ax=axR, fraction=0.046, pad=0.04)
    cb2.set_label("log₁₀ τ_ice  [yr]")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin-dir", default="/home/data_bedmap/bedmap2_bin")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--phi", type=float, default=0.9)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    reports = os.path.normpath(os.path.join(here, "..", "reports"))
    fig = args.fig or os.path.join(reports, "rtn_intrusion_clock.png")
    js = args.json or os.path.join(reports, "rtn_intrusion_clock.json")

    summary, payload = analyse(args.bin_dir, stride=args.stride, phi=args.phi)
    s = summary
    print(f"=== §H.1.2 intrusion clock (φ={s['phi']}, "
          f"near-tipping front = {s['n_front_cells']} cells) ===")
    A = s["amplification_km_per_m"]
    print("  geometric amplification A=1/|∇m| [km advance per m thinning]:")
    print(f"    p25={A['p25']:.2f}  median={A['median']:.2f}  "
          f"p75={A['p75']:.2f}  p90={A['p90']:.2f}")
    print(f"  front distance-to-GL: median={s['front_dist_to_GL_km']['median']:.1f} km  "
          f"p90={s['front_dist_to_GL_km']['p90']:.1f} km")
    print("  predicted RTN=1-line advance v=A·(dH/dt) [km/yr] (median A):")
    for k, v in s["front_advance_km_per_yr"].items():
        print(f"    {k}: {v:.2f} km/yr")
    t = s["tau_ice_yr"]
    print(f"  §G.4 memory τ_ice: front median={t['front_median']:.0f} yr  "
          f"interior median={t['interior_median']:.0f} yr  "
          f"(interior/front = {t['ratio_interior_over_front']:.1f}×)")
    print(f"  runaway (A>median) cells within 25 km of GL: "
          f"{100*s['runaway_frac_within_25km']:.0f}%")
    tt = s["test_targets"]
    print(f"  (b) residence-number test targets (A>={tt['A_threshold_km_per_m']:.2f} "
          f"km/m, <50 km GL): {tt['n_targets']} cells, "
          f"median A={tt['median_A_km_per_m']:.2f} km/m")
    for k, v in tt["v_kin_km_per_yr"].items():
        print(f"    v_kin({k}) = {v:.2f} km/yr  -> compare to DInSAR v_obs for Ro")

    make_figure(payload, fig)
    with open(js, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"json   -> {js}")


if __name__ == "__main__":
    main()
