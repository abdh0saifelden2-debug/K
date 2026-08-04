r"""§H.1.1 + §H.1.2 RTN corollaries on the independent NSIDC BedMachine 500 m product.

Independent-dataset companion to two Bedmap2 (3 km, BAS) runners:

  - ``rtn_phi_calibration.py``  — §H.1.1: the ``RTN>1`` intruded *area* is a
    monotone, constant-free inverse for the basal water fraction ``φ``; and
    ``H* = (ρ_w/φρ_i)·d_base`` defines a hydrology-corrected MISI margin (the
    grounded-but-intrudable band ``H_flot < H < H*`` of width ∝(1−φ)).
  - ``rtn_intrusion_clock.py``  — §H.1.2: the RTN=1 line is the zero level-set of
    ``m = H − H*`` and advances inland at ``v_front = (dH/dt)/|∇m|``, so the
    geometric amplification ``A = 1/|∇m|`` (km of advance per m of thinning) is a
    pure-geometry field; co-located with the §G.4 thermal memory ``τ_ice = H²/κ``.

Both corollaries are **pure geometry** — they need only ``thickness``, ``bed`` and
the integer ``mask`` (no velocity / melt / drainage data) — so they reproduce on
BedMachine (Morlighem et al. 2020) exactly as on Bedmap2.  BedMachine is a
*separate* thickness/bed product at **2× the Bedmap2 resolution**, so agreement of
the φ-inversion curve, the (1−φ) band scaling and the amplification statistics is a
genuine robustness cross-check: it tells us the published §H.1.1/§H.1.2 numbers are
geometry, not a Bedmap2 grid/resolution artefact.

Backend
-------
The native 500 m grid (13333×13333) makes the Euclidean distance transform and the
``|∇m|`` gradient the memory bottleneck (each full-grid float array is ≈0.7 GB in
float32).  ``--gpu`` runs the RTN map, the EDT and the gradient on the GPU via
CuPy / ``cupyx.scipy.ndimage`` (a 16 GB card handles full resolution); the inline
RTN formula is cross-checked against the canonical ``validators.rtn_validator.rtn``
at load time so the two backends agree.

Run::

    # native 500 m on a GPU box (e.g. Kaggle Tesla P100, 16 GB)
    python validation/external/rtn_corollaries_bedmachine.py --gpu \
        --path /kaggle/working/bm/NSIDC-0756_BedMachineAntarctica_*_V04.1.nc
    # 1 km decimation on CPU (stride=2)
    python validation/external/rtn_corollaries_bedmachine.py --stride 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from validators.rtn_validator import RHO_I  # noqa: E402
from external.run_rtn_bedmachine import (  # noqa: E402
    RHO_W, build_rtn, distance_to_groundingline_km, _assert_matches_validator,
)

KAPPA_ICE = 1.09e-6        # ice thermal diffusivity [m^2 s^-1] (k_ice/(ρ_i·cp))
SEC_PER_YR = 3.1557e7
# literature grounded-ice thinning band (WAIS/Amundsen altimetry, m yr^-1)
DHDT_BAND = (0.5, 1.0, 2.0, 5.0)


def _to_float(a):
    return float(a)


def _nanmedian(a, xp):
    return _to_float(xp.nanmedian(a)) if a.size else float("nan")


def _nanpct(a, p, xp):
    if not a.size:
        return float("nan")
    # CuPy has no ``nanpercentile``; drop non-finite then use ``percentile``
    # (present on both NumPy and CuPy), which is equivalent to ignoring NaNs.
    a = a[xp.isfinite(a)]
    if not a.size:
        return float("nan")
    v = xp.percentile(a, p)
    return _to_float(v)


def _gradient_mag(field2d, dx_km, xp):
    """|∇field| with NaN-aware np.gradient semantics (NaN propagates to neighbours).

    ``xp.gradient`` mirrors ``numpy.gradient``; spacing is in km so the magnitude is
    in [field-unit per km].
    """
    gy, gx = xp.gradient(field2d, dx_km, dx_km)
    return xp.hypot(gx, gy)


def _free_gpu(xp):
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()


def analyse(path, stride=1, phi_ref=0.9, phis=None, near_tip=0.20, use_gpu=False):
    """Compute the §H.1.1 φ-inversion + MISI band and §H.1.2 intrusion clock.

    Returns ``(summary_dict, figure_payload)``.  ``figure_payload`` holds host
    (numpy) arrays for plotting.
    """
    from external.bedmachine_loader import load_fields, MASK_GROUNDED
    if phis is None:
        phis = np.round(np.arange(0.70, 0.981, 0.02), 2)
    phis = np.asarray(phis, dtype="float64")

    d = load_fields(path, fields=("thickness", "bed", "mask"), stride=stride)
    H = d["thickness"]; bed = d["bed"]; mask = d["mask"]
    cellsize_m = d["_meta"]["cellsize"]
    dx_km = cellsize_m / 1000.0
    cell_area = dx_km ** 2  # km^2 per cell

    # Cheap dataset-independent guard: inline (xp-generic) RTN == canonical validator
    # on a decimated slice (numpy, float64) before any GPU work.
    sl = (slice(None, None, max(1, H.shape[0] // 200)),
          slice(None, None, max(1, H.shape[1] // 200)))
    _assert_matches_validator(H[sl], bed[sl], phi_ref)

    if use_gpu:
        import cupy as cp
        xp = cp
        # float32 on the GPU: halves the per-array footprint (≈0.7 GB at 13333²)
        # so EDT + gradient + RTN temporaries fit a 16 GB card at native res.
        H = cp.asarray(H, dtype=cp.float32)
        bed = cp.asarray(bed, dtype=cp.float32)
        mask = cp.asarray(mask)
    else:
        xp = np

    grounded = (mask == MASK_GROUNDED) & xp.isfinite(H) & (H > 0)
    ng = int(grounded.sum())
    dist = distance_to_groundingline_km(grounded, cellsize_m, xp=xp)

    # ---- §H.1.1 (1): φ-calibration — RTN>1 intruded area vs φ ----------------
    curve = []
    for phi in phis:
        pred = (build_rtn(H, bed, float(phi), xp=xp) > 1.0) & grounded
        n = int(pred.sum())
        med = _nanmedian(dist[pred], xp) if n else float("nan")
        curve.append({"phi": float(phi), "frac": (n / ng) if ng else float("nan"),
                      "area_km2": n * cell_area, "median_dist_km": med})
        del pred
        _free_gpu(xp)
    fr = np.array([c["frac"] for c in curve])
    # local sensitivity dA/dφ needs ≥2 φ samples; degenerate single-φ → nan
    dfdphi = np.gradient(fr, phis) if fr.size >= 2 else np.full(fr.shape, np.nan)
    for c, s in zip(curve, dfdphi):
        c["dfrac_dphi_per_0p01"] = float(s * 0.01)
    monotone = bool(np.all(np.diff(fr) <= 1e-12)) if fr.size >= 2 else True

    # ---- §H.1.1 (2): hydrology-corrected MISI band at φ_ref ------------------
    d_base = xp.where(xp.isfinite(bed), xp.maximum(0.0, -bed), xp.nan)
    H_flot = (RHO_W / RHO_I) * d_base          # flotation thickness  [m]
    H_star = H_flot / phi_ref                   # RTN=1 critical thickness  [m]
    with np.errstate(invalid="ignore", divide="ignore"):
        margin_rel = (H - H_star) / H           # thinning fraction to tip
    band = grounded & (H > H_flot) & (H < H_star)
    nb = int(band.sum())
    # band-width identity  (H*-H_flot)/H_flot == (1-φ)/φ  (exact at every cell
    # with H_flot>0; see the ``consistency_checks`` note below).  Sample the
    # median over all such cells rather than a single fixed corner: where bed>=0
    # (land above sea level) H_flot==0 and the ratio is 0/0 = NaN, so cell [0,0]
    # is unsafe on e.g. a Greenland file or a cropped above-sea-level domain.
    valid = xp.isfinite(H_flot) & (H_flot > 0)
    band_width_ratio = _nanmedian(
        (H_star[valid] - H_flot[valid]) / H_flot[valid], xp
    ) if int(valid.sum()) else float("nan")
    band_width_expected = (1.0 - phi_ref) / phi_ref
    margin_band_g = xp.where(grounded, margin_rel, xp.nan)
    misi = {
        "phi_ref": phi_ref,
        "band_cells": nb,
        "band_area_km2": nb * cell_area,
        "band_frac_grounded": (nb / ng) if ng else float("nan"),
        "band_median_dist_km": _nanmedian(dist[band], xp) if nb else float("nan"),
        "band_p90_dist_km": _nanpct(dist[band], 90, xp) if nb else float("nan"),
        "median_margin_grounded": _nanmedian(margin_band_g, xp),
        "frac_within_10pct_of_tipping": float(
            int((grounded & (margin_rel > 0) & (margin_rel < 0.10)).sum()) / ng)
        if ng else float("nan"),
        # ``consistency_checks`` are construction-guaranteed identities, NOT
        # empirical measurements of the ice sheet.  With H* := H_flot/phi_ref
        # and 0<phi_ref<1, every H_flot>0 cell satisfies
        # (H*-H_flot)/H_flot == (1-phi)/phi  and  H* > H_flot  exactly, so these
        # only verify that H*, H_flot and phi are wired together consistently
        # across the grid -- they do not measure the ice sheet.  The empirical
        # content of the band lives in band_cells / band_area_km2 /
        # band_*_dist_km above.  Both checks are restricted to H_flot>0: where
        # bed>=0, H_flot==H_star==0 makes 0/0 and 0>0 degenerate (NaN / False),
        # which on a Greenland file or a cropped above-sea-level domain would
        # otherwise corrupt them.
        "consistency_checks": {
            "band_width_ratio_measured": band_width_ratio,
            "band_width_ratio_expected_1mphi_over_phi": band_width_expected,
            "H_star_inland_of_flotation": bool(
                _to_float((H_star[valid] > H_flot[valid]).all())
            ) if int(valid.sum()) else False,
        },
    }

    # ---- §H.1.2: intrusion clock — A = 1/|∇m| and τ_ice co-location ----------
    margin_abs = H - H_star                      # [m]; zero contour is the RTN=1 line
    grad = _gradient_mag(xp.where(grounded, margin_abs, xp.nan), dx_km, xp)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = 1.0 / grad                          # km advance per m thinning
        rel = margin_abs / H
    front = grounded & (rel > 0) & (rel < near_tip) & xp.isfinite(amp)
    interior = grounded & (rel >= near_tip)
    tau_yr = (H ** 2 / KAPPA_ICE) / SEC_PER_YR    # §G.4 thermal memory [yr]

    amp_front = amp[front]
    A_med = _nanmedian(amp_front, xp)
    A_p90 = _nanpct(amp_front, 90, xp)
    tau_front = _nanmedian(tau_yr[front], xp)
    tau_interior = _nanmedian(tau_yr[interior], xp)
    runaway = front & (amp > A_med)
    n_runaway = int(runaway.sum())
    clock = {
        "near_tip_rel_margin": near_tip,
        "n_front_cells": int(front.sum()),
        "front_dist_to_GL_km": {
            "median": _nanmedian(dist[front], xp),
            "p90": _nanpct(dist[front], 90, xp)},
        "amplification_km_per_m": {
            "p25": _nanpct(amp_front, 25, xp), "median": A_med,
            "p75": _nanpct(amp_front, 75, xp), "p90": A_p90},
        "front_advance_km_per_yr": {f"dHdt_{r}_m_yr": A_med * r for r in DHDT_BAND},
        "tau_ice_yr": {
            "front_median": tau_front, "interior_median": tau_interior,
            "ratio_interior_over_front": (tau_interior / tau_front)
            if tau_front else float("nan")},
        "runaway_frac_within_25km": (
            int((runaway & (dist < 25)).sum()) / max(1, n_runaway)),
    }

    summary = {
        "dataset": "NSIDC BedMachine Antarctica (Morlighem 2020)",
        "backend": "gpu" if use_gpu else "cpu",
        "grid": tuple(int(s) for s in H.shape),
        "cellsize_km": dx_km,
        "n_grounded": ng,
        "rho_ratio_w_i": RHO_W / RHO_I,
        "phi_inversion": {
            "monotone_decreasing_in_phi": monotone,
            "calibration": curve,
            "sensitivity_note": ("dfrac_dphi_per_0p01 = change in RTN>1 area "
                                 "fraction per +0.01 in phi"),
        },
        "misi_band": misi,
        "intrusion_clock": clock,
    }

    # host arrays for plotting (small derived maps only)
    if use_gpu:
        margin_band_g = xp.asnumpy(margin_band_g)
        band_h = xp.asnumpy(band)
        amp_front_map = xp.asnumpy(xp.where(front, amp, xp.nan))
        tau_map = xp.asnumpy(xp.where(grounded, tau_yr, xp.nan))
        front_h = xp.asnumpy(front)
    else:
        band_h = band
        amp_front_map = np.where(front, amp, np.nan)
        tau_map = np.where(grounded, tau_yr, np.nan)
        front_h = front
    payload = (phis, fr, dfdphi, phi_ref, margin_band_g, band_h,
               amp_front_map, tau_map, front_h, near_tip)
    return summary, payload


def make_figure(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (phis, fr, dfdphi, phi_ref, margin_band_g, band,
     amp_front_map, tau_map, front, near_tip) = payload

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))

    # (1) φ-calibration curve + sensitivity
    axL = axs[0]
    axL.plot(phis, 100 * fr, "o-", color="#1f77b4", label="RTN>1 area")
    axL.axvline(phi_ref, ls=":", color="grey")
    axL.set_xlabel("subglacial water fraction  φ")
    axL.set_ylabel("RTN>1 area  [% of grounded ice]", color="#1f77b4")
    axL.tick_params(axis="y", labelcolor="#1f77b4")
    axL.invert_xaxis()
    ax2 = axL.twinx()
    ax2.plot(phis, -100 * dfdphi * 0.01, "s--", color="#d62728", alpha=0.7)
    ax2.set_ylabel("|dA/dφ|  [%-area per +0.01 φ]", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    axL.set_title("(§H.1.1#1) φ-calibration: RTN>1 area inverts for φ\n"
                  "(BedMachine 500 m, independent of Bedmap2)")

    # (2) thinning-margin map + MISI band contour
    axM = axs[1]
    m = np.ma.masked_invalid(margin_band_g)
    im = axM.imshow(np.clip(m, -0.2, 0.6), cmap="RdYlBu", origin="upper",
                    vmin=-0.2, vmax=0.6)
    axM.contour(band.astype(float), levels=[0.5], colors="k", linewidths=0.3)
    axM.set_title(f"(§H.1.1#2) thinning margin (H−H*)/H at φ={phi_ref}\n"
                  "blue→red = closer to RTN=1 tipping; black = (1−φ) band")
    axM.set_xticks([]); axM.set_yticks([])
    fig.colorbar(im, ax=axM, fraction=0.046, pad=0.04, label="thinning margin")

    # (3) intrusion-clock amplification on the near-tipping front
    axR = axs[2]
    la = np.ma.masked_invalid(np.log10(np.clip(amp_front_map, 1e-3, 1e3)))
    im2 = axR.imshow(la, cmap="magma", origin="upper", vmin=-1, vmax=2)
    axR.set_title(f"(§H.1.2) advance amplification A=1/|∇m| [km per m]\n"
                  f"near-tipping front (0<m/H<{near_tip}); bright = runaway")
    axR.set_xticks([]); axR.set_yticks([])
    cb = fig.colorbar(im2, ax=axR, fraction=0.046, pad=0.04)
    cb.set_label("log₁₀ A  [km / m]")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"figure -> {os.path.normpath(path)}")


def _print_summary(s):
    print(f"=== §H.1.1/§H.1.2 RTN corollaries on {s['dataset']} ===")
    print(f"grid={s['grid']} @ {s['cellsize_km']:g} km [{s['backend']}], "
          f"grounded cells={s['n_grounded']}")
    print("\n-- §H.1.1#1 φ-calibration (RTN>1 area vs φ) "
          f"[monotone={s['phi_inversion']['monotone_decreasing_in_phi']}] --")
    for c in s["phi_inversion"]["calibration"]:
        print(f"  φ={c['phi']:.2f}: {100*c['frac']:5.2f}% "
              f"({c['area_km2']:9.0f} km²)  med dist-to-GL={c['median_dist_km']:.1f} km")
    mb = s["misi_band"]
    print(f"\n-- §H.1.1#2 hydrology-corrected MISI band (φ={mb['phi_ref']}) --")
    print(f"  grounded-but-intrudable band: {mb['band_cells']} cells "
          f"({mb['band_area_km2']:.0f} km², {100*mb['band_frac_grounded']:.2f}% of grounded)")
    print(f"  band dist-to-GL: median={mb['band_median_dist_km']:.1f} km  "
          f"p90={mb['band_p90_dist_km']:.1f} km")
    print(f"  within 10% thinning of tipping: {100*mb['frac_within_10pct_of_tipping']:.2f}%")
    cc = mb["consistency_checks"]
    print(f"  [consistency checks] band-width ratio (H*-H_flot)/H_flot = "
          f"{cc['band_width_ratio_measured']:.6f} "
          f"(expected (1-φ)/φ = {cc['band_width_ratio_expected_1mphi_over_phi']:.6f}); "
          f"H* inland of flotation = {cc['H_star_inland_of_flotation']}")
    cl = s["intrusion_clock"]
    A = cl["amplification_km_per_m"]
    print(f"\n-- §H.1.2 intrusion clock (near-tipping front = {cl['n_front_cells']} cells) --")
    print(f"  amplification A=1/|∇m| [km per m]: p25={A['p25']:.2f} "
          f"median={A['median']:.2f} p75={A['p75']:.2f} p90={A['p90']:.2f}")
    print(f"  front dist-to-GL: median={cl['front_dist_to_GL_km']['median']:.1f} km "
          f"p90={cl['front_dist_to_GL_km']['p90']:.1f} km")
    print("  predicted advance v=A·(dH/dt) [km/yr] (median A):")
    for k, v in cl["front_advance_km_per_yr"].items():
        print(f"    {k}: {v:.2f}")
    t = cl["tau_ice_yr"]
    print(f"  §G.4 memory τ_ice: front median={t['front_median']:.0f} yr  "
          f"interior median={t['interior_median']:.0f} yr "
          f"(interior/front={t['ratio_interior_over_front']:.1f}×)")
    print(f"  runaway (A>median) cells within 25 km of GL: "
          f"{100*cl['runaway_frac_within_25km']:.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path",
                    default=os.path.expanduser("~/data_bedmachine/BedMachineAntarctica-v3.nc"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--phi-ref", type=float, default=0.9)
    ap.add_argument("--near-tip", type=float, default=0.20)
    ap.add_argument("--gpu", action="store_true", help="run RTN/EDT/gradient on GPU (CuPy)")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    reports = os.path.normpath(os.path.join(here, "..", "reports"))
    fig = args.fig or os.path.join(reports, "rtn_corollaries_bedmachine.png")
    js = args.json or os.path.join(reports, "rtn_corollaries_bedmachine.json")

    summary, payload = analyse(args.path, stride=args.stride, phi_ref=args.phi_ref,
                               near_tip=args.near_tip, use_gpu=args.gpu)
    _print_summary(summary)
    make_figure(payload, fig)
    with open(js, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"json   -> {os.path.normpath(js)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
