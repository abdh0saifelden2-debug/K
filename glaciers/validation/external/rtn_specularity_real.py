r"""§H.1.1c — the P4a open data step, executed: a *measured*, independent
connectivity observable (ICECAP radar bed-echo **specularity content**) drives the
variable-phi Regime Transition Number on real Bedmap2 geometry.

Context (what this closes)
--------------------------
``rtn_baseline_skill.py`` proved that at constant ``phi`` RTN *is* the
thickness-above-flotation threshold (zero added skill; Spearman(RTN, f) = -1.000
exactly).  ``rtn_variable_phi_real.py`` showed a *model* connectivity field
(Shreve-potential routing) breaks that degeneracy, but the routed ``phi`` is
derived from the same geometry that sets the flotation fraction, so the papers
(P4 §3.4b / P4a §6.4) state the single open step: "a ``phi`` field from an
**independent** observation of basal water (radar bed specularity; Schroeder et
al. 2013, 2015)".

This script supplies exactly that observable: the **ICECAP HiCARS basal-interface
specularity content** profiles (Young et al. 2020, USAP-DC doi:10.15784/601371;
five seasons, IPY + Operation IceBridge, East Antarctica ~80-170E incl. the
Totten / Aurora / Byrd systems).  Specularity content is an established proxy for
distributed subglacial water and its pressure state (Schroeder et al. 2013 PNAS;
Schroeder et al. 2015 IEEE GRSL; Dow et al. 2019 EPSL; Young et al. 2016 PTRSA).

Method
------
1. Parse the L2 along-track profiles (lon, lat, specularity fraction; ~3.2 M
   points, 432 transects), project WGS-84 -> EPSG:3031 (pyproj), and bin the
   mean specularity onto the (decimated) Bedmap2 grid.
2. Restrict all statistics to **covered grounded cells** (>= ``--min-pts``
   points per cell); report the coverage honestly (this is a *regional* East
   Antarctic test, not continental).
3. Map specularity -> phi with BOTH physically-arguable sign conventions, and
   report both (the degeneracy-break metric is sign-independent; the
   classification shifts are not):
     - ``pressure`` (primary; Schroeder 2013 / Dow 2019 reading): high
       specularity = distributed water at high pressure -> **high** phi
       (``phi = PHI_LO + (PHI_HI-PHI_LO) * spec``).
     - ``connectivity`` (the routing-harness convention, docstring of
       ``rtn_variable_phi_real.py``): high specularity = wet/connected ->
       **low** phi (``phi = PHI_HI - (PHI_HI-PHI_LO) * spec``).
4. Metrics per convention, within coverage: Spearman(RTN_spec, f) globally and
   in the near-flotation band (f_af < 0.7, matching §H.1.1b), the anchor
   Spearman(RTN_const, f) = -1 check, Spearman(phi_spec, f) (is the *observable*
   rank-independent of flotation?  -- this is what "independent" buys), and the
   RTN>1 classification changes vs the constant-phi flotation threshold (count +
   median distance-to-grounding-line).
5. **Bonus cross-test (routing vs radar):** Spearman between the Shreve-routing
   connectivity field of §H.1.1b and the measured specularity on the same cells
   -- the first direct check of the geometry-routed phi against a real
   basal-water observable in this repo.

Honest scope
------------
No gridded intrusion survey exists, so this still cannot be precision/recall
against actual seawater intrusion; what it upgrades is the *provenance* of the
degeneracy break (measured radar observable vs model field).  Coverage is East
Antarctica along flight lines (a few % of grounded cells at 5 km); the
grounding-line-proximal band is sampled by the coastal transects.  The 1 km
smoothing of the L2 product and the 2008-2012 epoch mismatch with Bedmap2 are
inherited caveats.

Run:
    python rtn_specularity_real.py --bin-dir /home/K/_data_bedmap2/bedmap2_bin \
        --spec-dir /home/K/_data_specularity --stride 5 --min-pts 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external import DataUnavailableError  # noqa: E402
from external.run_rtn_bedmap2 import (  # noqa: E402
    build_rtn, distance_to_groundingline_km, RHO_W, RHO_I)
from external.rtn_variable_phi_real import (  # noqa: E402
    PHI_LO, PHI_HI, shreve_potential, priority_flood_route, flow_accumulation,
    phi_from_flux, spearman)
from validators.rtn_validator import classify, G  # noqa: E402


def load_specularity_points(spec_dir):
    """Parse every ``*_spec.txt`` ICECAP L2 profile under ``spec_dir``.

    Returns (lon, lat, spec) float64 arrays with non-finite spec dropped.
    Columns: year, doy, sod, lon, lat, specularity (README_601371 / in-file header).
    """
    files = sorted(glob.glob(os.path.join(spec_dir, "**", "*_spec.txt"),
                             recursive=True))
    if not files:
        raise DataUnavailableError(
            f"no ICECAP specularity profiles under {spec_dir}\n"
            "Provision (bot-check, no login): https://www.usap-dc.org/view/dataset/601371\n"
            "unzip ICECAP_specularity_content_data.zip; tar -xzf each season .tgz")
    lons, lats, specs = [], [], []
    for f in files:
        a = np.loadtxt(f, comments="#", usecols=(3, 4, 5), ndmin=2)
        if a.size == 0:
            continue
        lons.append(a[:, 0]); lats.append(a[:, 1]); specs.append(a[:, 2])
    lon = np.concatenate(lons); lat = np.concatenate(lats); sp = np.concatenate(specs)
    m = np.isfinite(sp) & np.isfinite(lon) & np.isfinite(lat)
    return lon[m], lat[m], np.clip(sp[m], 0.0, 1.0), len(files)


def bin_to_bedmap2(lon, lat, spec, meta):
    """Mean specularity + point count per (decimated) Bedmap2 cell."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x, y = tr.transform(lon, lat)
    cs = meta["cellsize"]                    # effective (decimated) cell size [m]
    nr, nc = meta["nrows"], meta["ncols"]
    xll, yll = meta["xll"], meta["yll"]
    # full-resolution top edge (north-anchored rows), robust to odd grid + stride
    ytop = yll + meta["nrows_full"] * meta["cellsize_full"]
    j = np.floor((np.asarray(x) - xll) / cs).astype(np.int64)
    i = np.floor((ytop - np.asarray(y)) / cs).astype(np.int64)
    ok = (i >= 0) & (i < nr) & (j >= 0) & (j < nc)
    i, j, s = i[ok], j[ok], spec[ok]
    flat = i * nc + j
    ssum = np.zeros(nr * nc); cnt = np.zeros(nr * nc)
    np.add.at(ssum, flat, s)
    np.add.at(cnt, flat, 1.0)
    with np.errstate(invalid="ignore"):
        mean = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
    return mean.reshape(nr, nc), cnt.reshape(nr, nc)


def run(bin_dir, spec_dir, stride=5, phi_const=0.9, min_pts=3, seed=0):
    from external.bedmap2_loader import load_fields
    rng = np.random.default_rng(seed)
    d = load_fields(bin_dir, stride=stride)
    H, bed, mask = d["thickness"], d["bed"], d["icemask_grounded_and_shelves"]
    meta = d["_meta"]
    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)

    lon, lat, sp, nfiles = load_specularity_points(spec_dir)
    spec_mean, spec_cnt = bin_to_bedmap2(lon, lat, sp, meta)
    cov = grounded & (spec_cnt >= min_pts) & np.isfinite(spec_mean)

    # geometry
    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    h_af = H - (RHO_W / RHO_I) * d_base            # thickness above flotation
    f_af = np.where(H > 0, h_af / H, np.nan)       # flotation fraction H_af/H
    dist = distance_to_groundingline_km(grounded, meta["cellsize"])

    rtn_c = build_rtn(H, bed, phi_const)
    pred_c = classify(rtn_c) & grounded

    def _cov(a):
        return np.where(cov, a, np.nan)

    band = cov & (f_af < 0.7)
    out = {
        "inputs": {
            "spec_dataset": "USAP-DC 601371 (Young et al. 2020) ICECAP IRSPC2",
            "n_profiles": nfiles, "n_points": int(sp.size),
            "stride_km": meta["cellsize"] / 1000.0, "min_pts_per_cell": min_pts,
            "phi_const": phi_const, "PHI_LO": PHI_LO, "PHI_HI": PHI_HI,
        },
        "coverage": {
            "n_grounded": int(grounded.sum()),
            "n_covered_grounded": int(cov.sum()),
            "covered_frac_of_grounded": float(cov.sum() / max(grounded.sum(), 1)),
            "n_nearflot_band_covered": int(band.sum()),
            "median_dist_km_covered": float(np.nanmedian(dist[cov])) if cov.any() else float("nan"),
            "spec_mean_covered": float(np.nanmean(spec_mean[cov])),
            "spec_p10_p90": [float(np.nanpercentile(spec_mean[cov], 10)),
                             float(np.nanpercentile(spec_mean[cov], 90))],
        },
        "anchor_const_phi": {
            "spearman_rtn_flotfrac_coverage": spearman(_cov(rtn_c), _cov(f_af), rng),
        },
        "conventions": {},
    }

    for name, phi_spec_full in (
        ("pressure_highspec_highphi",
         PHI_LO + (PHI_HI - PHI_LO) * np.clip(spec_mean, 0, 1)),
        ("connectivity_highspec_lowphi",
         PHI_HI - (PHI_HI - PHI_LO) * np.clip(spec_mean, 0, 1)),
    ):
        phi_s = np.where(cov, phi_spec_full, np.nan)
        rtn_s = build_rtn(H, bed, phi_s)
        pred_s = classify(rtn_s) & cov
        pred_c_cov = pred_c & cov
        changed = (pred_s ^ pred_c_cov) & cov
        newly_on = int((pred_s & ~pred_c_cov).sum())
        newly_off = int((~pred_s & pred_c_cov).sum())
        out["conventions"][name] = {
            "spearman_rtn_flotfrac_coverage": spearman(rtn_s, _cov(f_af), rng),
            "spearman_rtn_flotfrac_band": spearman(
                np.where(band, rtn_s, np.nan), np.where(band, f_af, np.nan), rng),
            "spearman_phiSpec_vs_flotfrac_band": spearman(
                np.where(band, phi_s, np.nan), np.where(band, f_af, np.nan), rng),
            "degeneracy_broken": bool(abs(spearman(rtn_s, _cov(f_af), rng)) < 0.999),
            "n_rtn_gt1_specphi": int(pred_s.sum()),
            "n_rtn_gt1_constphi_coverage": int(pred_c_cov.sum()),
            "n_classification_changed": int(changed.sum()),
            "newly_flagged": newly_on, "newly_dropped": newly_off,
            "median_dist_km_changed": float(np.nanmedian(dist[changed])) if changed.any() else float("nan"),
        }

    # ---- bonus: does Shreve routing predict the *measured* specularity? ----
    pot = shreve_potential(np.where(np.isfinite(bed), bed, 0.0),
                           np.where(np.isfinite(H), H, 0.0))
    _, receiver, pop = priority_flood_route(np.where(grounded, pot, np.inf), grounded)
    acc = flow_accumulation(receiver, pop, grounded)
    phi_routed = phi_from_flux(acc, grounded)
    la = np.log10(acc.reshape(grounded.shape) + 1.0)
    out["routing_vs_radar"] = {
        "spearman_spec_vs_phiRouted_coverage": spearman(_cov(spec_mean), _cov(phi_routed), rng),
        "spearman_spec_vs_logflux_coverage": spearman(_cov(spec_mean), _cov(la), rng),
        "spearman_spec_vs_logflux_band": spearman(
            np.where(band, spec_mean, np.nan), np.where(band, la, np.nan), rng),
        "note": "routing phi is DEcreasing in flux; a negative spec-vs-phiRouted "
                "Spearman = specular (wet) cells sit on high-flux routed axes",
    }
    # apples-to-apples: the routed phi's reordering strength on the SAME covered cells
    rtn_r = build_rtn(H, bed, np.where(cov, phi_routed, np.nan))
    pred_r = classify(rtn_r) & cov
    out["routed_phi_same_coverage"] = {
        "spearman_rtn_flotfrac_coverage": spearman(rtn_r, _cov(f_af), rng),
        "spearman_rtn_flotfrac_band": spearman(
            np.where(band, rtn_r, np.nan), np.where(band, f_af, np.nan), rng),
        "spearman_phiRouted_vs_flotfrac_band": spearman(
            np.where(band, phi_routed, np.nan), np.where(band, f_af, np.nan), rng),
        "n_classification_changed": int(((pred_r ^ (pred_c & cov)) & cov).sum()),
    }
    arrays = dict(grounded=grounded, cov=cov, spec_mean=spec_mean, dist=dist,
                  f_af=f_af, cnt=spec_cnt)
    return out, arrays


def make_figure(arr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g, cov, spec = arr["grounded"], arr["cov"], arr["spec_mean"]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    base = np.where(g, 0.15, np.nan)
    ax[0].imshow(base, cmap="Greys", vmin=0, vmax=1)
    ii, jj = np.where(cov)
    ax[0].scatter(jj, ii, s=0.3, c="crimson", linewidths=0)
    ax[0].set_title("(a) ICECAP specularity coverage\n(covered grounded cells, East Antarctica)")
    sm = np.where(cov, spec, np.nan)
    im1 = ax[1].imshow(sm, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    ax[1].set_title("(b) mean basal specularity content\n(1 = specular / distributed water)")
    hb = ax[2].hexbin(np.clip(arr["f_af"][cov], 0, 1), np.clip(spec[cov], 0, 1),
                      gridsize=40, cmap="magma", mincnt=1)
    fig.colorbar(hb, ax=ax[2], fraction=0.046)
    ax[2].set_xlabel("flotation fraction  H_af/H"); ax[2].set_ylabel("specularity")
    ax[2].set_title("(c) the observable vs flotation\n(rank-independence is the point)")
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("§H.1.1c RTN variable-phi from MEASURED radar specularity (USAP-DC 601371)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="/home/K/_data_bedmap2/bedmap2_bin")
    ap.add_argument("--spec-dir", default="/home/K/_data_specularity")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--phi-const", type=float, default=0.9)
    ap.add_argument("--min-pts", type=int, default=3)
    ap.add_argument("--out", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports",
        "rtn_specularity_real.json")))
    a = ap.parse_args()
    out, arrays = run(a.bin_dir, a.spec_dir, stride=a.stride,
                      phi_const=a.phi_const, min_pts=a.min_pts)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    make_figure(arrays, a.out.replace(".json", ".png"))


if __name__ == "__main__":
    main()
