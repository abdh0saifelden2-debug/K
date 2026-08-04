r"""§V.1e — the specularity gauge in meters: Kirchhoff inversion of ICECAP
bed-echo specularity content to RMS interface roughness (horizon ledger E1).

Context (what this upgrades)
----------------------------
§V.1c (``rtn_specularity_real.py``) treated the ICECAP HiCARS specularity
content (USAP-DC 601371) as a *dimensionless* [0,1] index and mapped it
linearly to a φ scale whose endpoints (PHI_LO/PHI_HI) are modelling choices;
§V.1d pinned the map's *sign*.  This module derives what the observable
actually measures at leading order: under the Kirchhoff (physical-optics)
approximation for a Gaussian-height interface, the **coherent (specular)
power fraction** of a reflection is the Rayleigh roughness factor

    s  =  exp(-g),        g = (4 pi sigma cos(theta) / lambda_ice)^2,

(Ament 1953; Beckmann & Spizzichino 1963, ch. 5) where ``sigma`` is the RMS
interface height within the pulse-limited/Fresnel footprint, ``theta`` the
incidence angle at the bed (nadir sounding: 0), and ``lambda_ice`` the radar
wavelength IN ICE — the wave hits the bed while propagating in ice, so the
phase variance uses the in-ice wavenumber.  Reading the L2 "specularity
content" (fraction of bed-echo energy in the angularly-narrow specular
component; Schroeder et al. 2013, 2015) as that coherent fraction inverts in
closed form, per cell, with **zero free parameters**:

    sigma(s) = (lambda_ice / (4 pi cos(theta))) * sqrt(-ln s).

For HiCARS (60 MHz VHF; Peters et al. 2005; Young et al. 2011),
``lambda_vac = 5.0 m`` and ``n_ice = sqrt(3.17) ~ 1.78`` (polar ice
permittivity 3.17 +/- 0.02; Fujita et al. 2000) give ``lambda_ice ~ 2.81 m``
and a gauge constant ``K = lambda_ice/(4 pi) ~ 0.224 m``.

Three closed-form corollaries (all unit-proved)
-----------------------------------------------
1. **Gauge window.**  With a specularity noise floor/ceiling ``eps``, the
   measurable roughness window is
   ``sigma in [K sqrt(-ln(1-eps)), K sqrt(-ln eps)]`` — for eps = 0.05 that is
   **5.1 cm .. 38.7 cm**: HiCARS specularity is a *decimetre* roughness gauge.
   Cells at s below the floor carry only the censored bound sigma >~ 0.39 m.
2. **Sweet spot.**  For additive specularity noise the error
   ``|d sigma/d s| = K / (2 s sqrt(-ln s))`` is minimised at exactly
   ``s* = e^{-1/2} ~ 0.607``, where ``sigma* = K/sqrt(2) ~ 15.8 cm``.
3. **The Schroeder-water criterion drops out.**  Schroeder et al. (2015,
   IEEE GRSL) estimated — from independent scattering/attenuation/cross-section
   modelling of the same instrument class — that the distributed-water
   interfaces producing specular Thwaites returns have **RMS roughness
   <~ 15 cm**.  The zero-parameter inversion reproduces that number as a
   specularity threshold: ``sigma <= 0.15 m  <=>  s >= exp(-(0.15/K)^2) ~ 0.64``
   — i.e. "specular water" cells in the established qualitative sense are
   exactly the cells this gauge places at sigma <~ 15 cm.

What is validated against data (East Antarctica, USAP-DC 601371)
----------------------------------------------------------------
* **Point level** (3.2 M along-track samples, 1-km L2 smoothing): the sigma
  distribution over the survey, with the censored fraction reported.
* **Cell level** (5-km Bedmap2 bins, >= 3 pts): sigma map over covered
  grounded cells.
* **Lake anchor in metres** (rank-exact transfer of §V.1d line 1): active-lake
  footprint cells (USAP-DC 601470 outlines; Stubblefield et al. 2021 after
  Siegfried & Fricker 2018) must sit LOW on the sigma scale against
  flotation-matched controls.  Because ``sigma(s)`` is strictly decreasing,
  the §V.1d one-sided stratified Mann-Whitney transfers *exactly*
  (P[sigma_lake < sigma_ctrl] = P[spec_lake > spec_ctrl]); what is new is the
  metric statement: median sigma over lake cells vs controls, in metres.
* **Water-like enrichment**: cells with ``s >= 0.64`` (sigma <= 15 cm) —
  count, lake-footprint enrichment odds ratio, distance-to-GL profile.
* **NR33 joint bed state (sigma, phi) in physical units**: the two-epoch
  persistent-specular set of §V.1d/NR33 (spec >= 0.2 in 2008/09 AND 2011/12)
  gets, per cell, the Kirchhoff roughness estimate range
  [sigma(max epoch spec), sigma(min epoch spec)] *and* the creep-persistence
  pressure floor phi >= 1 - N_max/p_i — the P4a §6.4 driver acquires metres
  and pascals: "persistently specular = smooth at the decimetre scale AND
  pressurised to within ~3.5 bar of overburden".

Honest scope (biases derived, not hand-waved)
---------------------------------------------
* **Model**: Gaussian heights + Kirchhoff validity (curvature radii >> lambda,
  moderate slopes) at sub-Fresnel horizontal scales (first Fresnel radius
  ``sqrt(lambda_ice h / 2)`` ~ 50-80 m for 2-4 km ice): sigma is RMS relief at
  ~10-100 m horizontal wavelengths — NOT the km-window FFT "total roughness"
  of the Bingham/Siegert traverse literature, which lives at 10^2-10^4 m
  wavelengths and metres-to-hundreds-of-metres amplitudes.  Self-affine beds
  (Jordan et al. 2017) make sigma scale-dependent; the single-scale inversion
  is the leading-order reading at the decorrelation footprint.
* **Two opposing bias directions, both conservative for the lake contrast**:
  (i) within-cell heterogeneity: Jensen on the convex map s(sigma^2) gives
  ``sigma(mean s) <= sqrt(mean sigma^2)`` — binned-cell values UNDER-estimate
  the cell's true RMS; (ii) any extraneous decoherence (englacial volume
  scattering, off-nadir clutter, epoch mixing) depresses s and OVER-estimates
  sigma.  The distributional *contrast* (lakes vs controls) is rank-exact and
  immune to (i)+(ii) insofar as they act as common-mode; absolute values carry
  the bracket.
* The 601470 outlines epoch (2003-2016) vs ICECAP epoch (2008-2012) mismatch
  and 5-km binning of ~10-km lakes dilute the lake contrast (conservative).

Run:
    python e1_radar_roughness.py --bin-dir /home/K/_data_bedmap2/bedmap2_bin \
        --spec-dir /home/K/_data_specularity \
        --lake-stats /home/data_usapdc/601470/data/active_lake_statistics.dat
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external import DataUnavailableError  # noqa: E402
from external.run_rtn_bedmap2 import (  # noqa: E402
    RHO_I, RHO_W, distance_to_groundingline_km)
from external.rtn_specularity_real import (  # noqa: E402
    bin_to_bedmap2, load_specularity_points)
from external.rtn_spec_sign_pin import (  # noqa: E402
    GLEN_A, GLEN_N, YR, phi_floor, creep_nmax, load_lake_stats,
    lake_cell_masks, stratified_mw, per_epoch_binning)
from external.rtn_variable_phi_real import spearman  # noqa: E402

G = 9.81

# ------------------------------------------------------------- instrument --
C_VAC = 299792458.0                 # m/s
F_HICARS = 60.0e6                   # Hz, HiCARS/MARFA VHF centre frequency
EPS_ICE = 3.17                      # polar-ice permittivity (Fujita et al. 2000)
N_ICE = float(np.sqrt(EPS_ICE))     # ~1.780
LAMBDA_VAC = C_VAC / F_HICARS       # ~5.00 m
LAMBDA_ICE = LAMBDA_VAC / N_ICE     # ~2.81 m
SCHROEDER15_SIGMA_MAX = 0.15        # m; water-interface RMS bound, S. et al. 2015


def gauge_constant(lam=LAMBDA_ICE, theta=0.0):
    """K = lambda / (4 pi cos(theta)) [m] — the whole instrument in one number."""
    return lam / (4.0 * np.pi * np.cos(theta))


# ---------------------------------------------------------------- physics --
def rayleigh_g(sigma, lam=LAMBDA_ICE, theta=0.0):
    r"""Rayleigh roughness parameter ``g = (4 pi sigma cos(theta)/lambda)^2``."""
    return (4.0 * np.pi * np.asarray(sigma, float) * np.cos(theta) / lam) ** 2


def spec_from_sigma(sigma, lam=LAMBDA_ICE, theta=0.0):
    r"""Coherent POWER fraction ``exp(-g)`` of a Gaussian rough interface.

    The field factor is ``exp(-g/2)`` (``<e^{i phi}> = e^{-<phi^2>/2}`` with
    ``phi = 2 k sigma cos(theta) xi``, ``xi ~ N(0,1)``); power is its square.
    """
    return np.exp(-rayleigh_g(sigma, lam, theta))


def sigma_from_spec(spec, lam=LAMBDA_ICE, theta=0.0):
    r"""Invert ``s = exp(-g)`` -> RMS roughness [m].

    ``s <= 0`` -> NaN (censored: rougher than the gauge ceiling);
    ``s >= 1`` -> 0 (mirror-smooth at this wavelength).
    """
    s = np.asarray(spec, float)
    out = np.full(s.shape if s.ndim else (), np.nan, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = gauge_constant(lam, theta) * np.sqrt(-np.log(s))
    out = np.where(s >= 1.0, 0.0, np.where(s > 0.0, val, np.nan))
    return out if out.ndim else float(out)


def gauge_window(eps=0.05, lam=LAMBDA_ICE, theta=0.0):
    """(sigma_min, sigma_max) resolvable when spec is trusted in [eps, 1-eps]."""
    K = gauge_constant(lam, theta)
    return (K * np.sqrt(-np.log(1.0 - eps)), K * np.sqrt(-np.log(eps)))


def sweet_spot(lam=LAMBDA_ICE, theta=0.0):
    r"""(s*, sigma*) minimising ``|d sigma / d s| = K/(2 s sqrt(-ln s))``.

    ``d/ds [s sqrt(-ln s)] = 0  =>  -ln s = 1/2  =>  s* = e^{-1/2}``,
    ``sigma* = K/sqrt(2)``.
    """
    K = gauge_constant(lam, theta)
    return float(np.exp(-0.5)), float(K / np.sqrt(2.0))


def spec_water_threshold(sigma_max=SCHROEDER15_SIGMA_MAX, lam=LAMBDA_ICE,
                         theta=0.0):
    """Specularity above which the gauge reads sigma <= sigma_max."""
    return float(spec_from_sigma(sigma_max, lam, theta))


def fresnel_radius(h_ice, lam=LAMBDA_ICE):
    """First Fresnel-zone radius sqrt(lambda h / 2) [m] at ice thickness h."""
    return np.sqrt(lam * np.asarray(h_ice, float) / 2.0)


def censored_percentile(sig_cens, q):
    """Percentile over a sample whose censored members are +inf.

    Censored values (spec below the noise floor) are ROUGHER than the gauge
    ceiling, so they occupy the top ranks exactly; any percentile that falls
    inside the censored mass is itself only a bound -> returns None.
    """
    with np.errstate(invalid="ignore"):
        v = float(np.percentile(sig_cens, q))
    # interpolation between two +inf order statistics yields inf-inf = NaN;
    # both mean the percentile is only bounded, not measured
    return None if not np.isfinite(v) else v


# ------------------------------------------------------------------- run --
def run(bin_dir, spec_dir, lake_stats=None, stride=5, min_pts=3,
        spec_floor=0.01, eps_window=0.05, persist_thresh=0.2,
        t_obs_yr=4.0, seed=0):
    from external.bedmap2_loader import load_fields

    rng = np.random.default_rng(seed)
    d = load_fields(bin_dir, stride=stride)
    H, bed, mask = d["thickness"], d["bed"], d["icemask_grounded_and_shelves"]
    meta = d["_meta"]
    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)

    lon, lat, sp, nfiles = load_specularity_points(spec_dir)
    spec_mean, spec_cnt = bin_to_bedmap2(lon, lat, sp, meta)
    cov = grounded & (spec_cnt >= min_pts) & np.isfinite(spec_mean)

    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    f_af = np.where(H > 0, (H - (RHO_W / RHO_I) * d_base) / H, np.nan)
    dist = distance_to_groundingline_km(grounded, meta["cellsize"])

    K = gauge_constant()
    w_lo, w_hi = gauge_window(eps_window)
    s_star, sig_star = sweet_spot()
    s_water = spec_water_threshold()

    # ---- point level (1-km L2 smoothing; no 5-km Jensen dilution) ----------
    # censoring-aware ranks: spec below the floor means sigma ABOVE the gauge
    # ceiling -> +inf rank, so percentiles are exact, never biased smooth
    pt_cens = sp < spec_floor
    sig_pt_cens = np.where(pt_cens, np.inf,
                           sigma_from_spec(np.maximum(sp, spec_floor)))
    sig_bound = float(sigma_from_spec(spec_floor))
    point_level = {
        "n_points": int(sp.size),
        "censored_frac_below_floor": float(pt_cens.mean()),
        "sigma_bound_censored_m": sig_bound,
        "sigma_median_m": censored_percentile(sig_pt_cens, 50),
        "sigma_p10_p90_m": [censored_percentile(sig_pt_cens, 10),
                            censored_percentile(sig_pt_cens, 90)],
        "frac_water_like": float((sp >= s_water).mean()),
        "note": "percentiles censoring-aware (censored -> +inf rank); None "
                "means the percentile sits in the censored mass, i.e. "
                f"> {sig_bound:.3f} m",
    }

    # ---- cell level ---------------------------------------------------------
    cell_cens = cov & (spec_mean < spec_floor)
    sig_cell = np.where(cov & ~cell_cens,
                        sigma_from_spec(np.clip(spec_mean, spec_floor, 1.0)),
                        np.nan)
    sig_cell_cens = np.where(cell_cens, np.inf, sig_cell)  # NaN outside cov
    cc = sig_cell_cens[cov]
    cell_level = {
        "n_covered_grounded": int(cov.sum()),
        "n_censored_cells": int(cell_cens.sum()),
        "censored_frac": float(cell_cens.sum() / max(cov.sum(), 1)),
        "sigma_median_m": censored_percentile(cc, 50),
        "sigma_p10_p90_m": [censored_percentile(cc, 10),
                            censored_percentile(cc, 90)],
        "n_water_like": int((cov & (spec_mean >= s_water)).sum()),
        "note": "cell sigma from binned-mean spec: Jensen => lower bound on "
                "cell RMS (see docstring); point level carries the "
                "distribution; censored cells (mean spec below floor) enter "
                "percentiles as +inf",
    }

    # ---- lake anchor in metres ---------------------------------------------
    lake_anchor = {"available": False}
    water_like = {}
    if lake_stats is not None and os.path.exists(lake_stats):
        lakes = load_lake_stats(lake_stats)
        lake_m, halo_m, lake_id, _ = lake_cell_masks(lakes, meta, H.shape)
        lake_cov = lake_m & cov
        ctrl_cov = cov & ~halo_m
        mw = stratified_mw(spec_mean, lake_cov, ctrl_cov, f_af, rng)
        sig_lake = sig_cell_cens[lake_cov]
        sig_ctrl = sig_cell_cens[ctrl_cov]
        lake_anchor = {
            "available": True,
            "rank_exact_transfer": "P[sigma_lake<sigma_ctrl] = "
                                   "P[spec_lake>spec_ctrl] (sigma(s) strictly "
                                   "decreasing); MW/permutation from SS V.1d "
                                   "line 1 apply verbatim",
            "n_lake_cells": int(lake_cov.sum()),
            "n_ctrl_cells": int(ctrl_cov.sum()),
            "sigma_median_lake_m": censored_percentile(sig_lake, 50),
            "sigma_median_ctrl_m": censored_percentile(sig_ctrl, 50),
            "sigma_p10_lake_m": censored_percentile(sig_lake, 10),
            "sigma_p10_ctrl_m": censored_percentile(sig_ctrl, 10),
            "delta_median_m": (
                censored_percentile(sig_ctrl, 50)
                - censored_percentile(sig_lake, 50)
                if censored_percentile(sig_ctrl, 50) is not None
                and censored_percentile(sig_lake, 50) is not None else None),
            "prob_sigma_lake_lt_ctrl": mw["prob_lake_gt_ctrl"],
            "p_perm_stratified": mw.get("p_perm_stratified"),
            "mw_p_onesided_iid": mw.get("mw_p_onesided_iid"),
        }
        # enrichment of water-like cells inside lake footprints
        wl = cov & (spec_mean >= s_water)
        a = int((wl & lake_m).sum()); b = int((wl & ~lake_m).sum())
        c = int((~wl & cov & lake_m).sum()); e = int((~wl & cov & ~lake_m).sum())
        odds = (a * e) / (b * c) if b * c > 0 else float("inf")
        water_like = {
            "s_threshold": s_water,
            "sigma_threshold_m": SCHROEDER15_SIGMA_MAX,
            "n_water_like_cells": int(wl.sum()),
            "n_in_lake_footprint": a,
            "enrichment_odds_ratio_lake": float(odds),
            "median_dist_km_water_like": float(np.nanmedian(dist[wl]))
            if wl.any() else float("nan"),
            "median_dist_km_background": float(np.nanmedian(dist[cov & ~wl])),
        }

    # ---- NR33 joint (sigma, phi) bed state ---------------------------------
    joint = {"available": False}
    (sA, cA), (sB, cB) = per_epoch_binning(spec_dir, meta)
    if sA is not None and sB is not None:
        seen = cov & (cA >= min_pts) & (cB >= min_pts) \
            & np.isfinite(sA) & np.isfinite(sB)
        persist = seen & (sA >= persist_thresh) & (sB >= persist_thresh)
        t_obs = t_obs_yr * YR
        phi_fl = phi_floor(H, t_obs)
        rows = []
        for i, j in zip(*np.where(persist)):
            smax, smin = max(sA[i, j], sB[i, j]), min(sA[i, j], sB[i, j])
            rows.append({
                "spec_2008_09": float(sA[i, j]), "spec_2011_12": float(sB[i, j]),
                "sigma_est_lo_m": float(sigma_from_spec(smax)),
                "sigma_est_hi_m": float(sigma_from_spec(smin)),
                "H_m": float(H[i, j]),
                "phi_floor": float(phi_fl[i, j]),
                "fresnel_radius_m": float(fresnel_radius(H[i, j])),
                "dist_gl_km": float(dist[i, j]),
            })
        joint = {
            "available": True,
            "persist_thresh": persist_thresh, "t_obs_yr": t_obs_yr,
            "n_seen_both_epochs": int(seen.sum()),
            "n_persistent": int(persist.sum()),
            "sigma_ceiling_at_thresh_m": float(sigma_from_spec(persist_thresh)),
            "N_max_bar": float(creep_nmax(t_obs) / 1e5),
            "cells": rows,
            "statement": "persistently specular cells are simultaneously "
                         "smooth (sigma <~ 0.28 m at the 0.2 threshold; "
                         "per-cell estimates below) and pressurised "
                         "(phi >= phi_floor ~ 0.98): the P4a SS6.4 driver in "
                         "metres and pascals",
        }

    out = {
        "inputs": {
            "spec_dataset": "USAP-DC 601371 (Young et al. 2020) ICECAP IRSPC2",
            "n_profiles": nfiles, "stride_km": meta["cellsize"] / 1e3,
            "min_pts_per_cell": min_pts, "spec_floor": spec_floor,
            "lake_dataset": "USAP-DC 601470 (Stubblefield et al. 2021)"
            if lake_anchor.get("available") else None,
        },
        "physics": {
            "f_hicars_hz": F_HICARS, "eps_ice": EPS_ICE, "n_ice": N_ICE,
            "lambda_vac_m": LAMBDA_VAC, "lambda_ice_m": LAMBDA_ICE,
            "gauge_constant_K_m": K,
            "inversion": "sigma = K sqrt(-ln s), K = lambda_ice/(4 pi cos "
                         "theta), theta = 0 (nadir)",
            "gauge_window_m": [w_lo, w_hi], "eps_window": eps_window,
            "sweet_spot_spec": s_star, "sweet_spot_sigma_m": sig_star,
            "schroeder2015_water_sigma_max_m": SCHROEDER15_SIGMA_MAX,
            "spec_water_threshold": s_water,
        },
        "point_level": point_level,
        "cell_level": cell_level,
        "lake_anchor_metres": lake_anchor,
        "water_like_cells": water_like,
        "nr33_joint_bed_state": joint,
    }
    return out, (spec_mean, sig_cell, cov, meta)


# ------------------------------------------------------------------ plots --
def make_figure(out, grids, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec_mean, sig_cell, cov, meta = grids
    K = out["physics"]["gauge_constant_K_m"]
    w_lo, w_hi = out["physics"]["gauge_window_m"]
    s_star = out["physics"]["sweet_spot_spec"]
    sig_star = out["physics"]["sweet_spot_sigma_m"]
    s_water = out["physics"]["spec_water_threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    s = np.linspace(1e-3, 0.999, 600)
    ax.plot(s, sigma_from_spec(s), lw=2)
    ax.axhspan(w_lo, w_hi, color="tab:green", alpha=0.15,
               label=f"gauge window {w_lo*100:.0f}-{w_hi*100:.0f} cm")
    ax.axhline(0.15, color="tab:red", ls="--", lw=1,
               label="Schroeder+15 water RMS 15 cm")
    ax.plot([s_star], [sig_star], "ko", ms=5,
            label=f"sweet spot s*=e^-1/2, sigma*={sig_star*100:.0f} cm")
    ax.axvline(s_water, color="tab:red", ls=":", lw=1)
    ax.set_xlabel("specularity content s")
    ax.set_ylabel("RMS roughness sigma  [m]")
    ax.set_title("sigma = K sqrt(-ln s),  K = lambda_ice/4pi")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    sig = sig_cell[np.isfinite(sig_cell)]
    ax.hist(sig, bins=60, color="tab:blue", alpha=0.75)
    n_cens = out["cell_level"]["n_censored_cells"]
    bound = out["point_level"]["sigma_bound_censored_m"]
    ax.annotate(f"+{n_cens} censored cells\n(sigma > {bound:.2f} m)",
                xy=(0.97, 0.55), xycoords="axes fraction", ha="right",
                fontsize=7, color="tab:red")
    la = out.get("lake_anchor_metres", {})
    if la.get("available"):
        ax.axvline(la["sigma_median_lake_m"], color="tab:cyan", lw=2,
                   label=f"lake median {la['sigma_median_lake_m']:.2f} m")
        ax.axvline(la["sigma_median_ctrl_m"], color="tab:orange", lw=2,
                   label=f"control median {la['sigma_median_ctrl_m']:.2f} m")
    ax.axvline(0.15, color="tab:red", ls="--", lw=1)
    ax.set_xlabel("cell sigma  [m]")
    ax.set_ylabel("covered grounded cells")
    ax.set_title("East Antarctic bed roughness (ICECAP coverage)")
    ax.legend(fontsize=7)

    ax = axes[2]
    shown = cov & np.isfinite(sig_cell)
    ii, jj = np.where(shown)
    cs = meta["cellsize"] / 1e3
    sc = ax.scatter(jj * cs, -ii * cs, c=sig_cell[shown], s=1.5,
                    cmap="viridis_r", vmin=0.0, vmax=w_hi)
    plt.colorbar(sc, ax=ax, label="sigma [m]  (bright = smooth)")
    ax.set_title("specularity gauge, metres (5-km bins)")
    ax.set_xlabel("grid x [km]"); ax.set_ylabel("grid y [km]")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bin-dir", required=True)
    p.add_argument("--spec-dir", required=True)
    p.add_argument("--lake-stats", default=None)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--min-pts", type=int, default=3)
    p.add_argument("--out-json", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports",
        "e1_radar_roughness.json"))
    p.add_argument("--out-png", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports",
        "e1_radar_roughness.png"))
    a = p.parse_args(argv)

    out, grids = run(a.bin_dir, a.spec_dir, lake_stats=a.lake_stats,
                     stride=a.stride, min_pts=a.min_pts)
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, grids, a.out_png)
    print(json.dumps({k: out[k] for k in
                      ("physics", "point_level", "cell_level",
                       "lake_anchor_metres", "water_like_cells")}, indent=1))
    jj = out["nr33_joint_bed_state"]
    if jj.get("available"):
        print(f"NR33 joint bed state: {jj['n_persistent']} persistent cells, "
              f"sigma ceiling {jj['sigma_ceiling_at_thresh_m']:.3f} m, "
              f"N_max {jj['N_max_bar']:.2f} bar")
    print(f"wrote {a.out_json} and {a.out_png}")
    return out


if __name__ == "__main__":
    main()
