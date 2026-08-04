r"""Real tidal forcing at the Antarctic grounding zone (CATS2008 x BedMachine).

This closes the last auth-gated §V data item by combining two *independent* real
datasets:

* **BedMachine Antarctica v4** (NSIDC-0756) -- ice ``thickness`` + ``bed`` +
  grounded ``mask`` (the same product used by the §V.1 RTN cross-check), to locate
  the grounding zone and the ice overburden ``p_i = rho_i g H`` there; and
* **CATS2008** (USAP-DC doi:10.15784/601235, via ``pyTMD``) -- the circum-Antarctic
  tide model, to predict the tidal sea-level amplitude ``eta`` at those same
  grounding-zone points.

It turns the §I.5 tidal-admittance probe's *assumed* forcing amplitude ``eps``
(the fractional tidal modulation of basal effective pressure ``N``) into a
**measured, population-level** quantity over the real grounding line:

    Delta p  = rho_w g eta_amp                 (tidal ocean-pressure swing [Pa])
    p_i      = rho_i g H                        (ice overburden [Pa])
    ratio    = Delta p / p_i                    (clean, connectivity-free)
    N_flot   = p_i - rho_w g d_base             (flotation effective pressure)
    eps_flot = Delta p / N_flot                 ([HYP] -- assumes ocean-pressure
                                                 reaches the bed; grows toward
                                                 flotation as N_flot -> 0)

``ratio = Delta p / p_i`` is a hard **lower bound** on the §I.5 ``eps`` (because
``N <= p_i`` always), so the §I.5 admittance/harmonic signal is at least
``A1 ~ |s_N|`` times this -- a measured floor, not an assumption.  ``eps_flot`` is
the connectivity-limited estimate and is reported as ``[HYP]``.

No GPU.  Needs ``pip install pyTMD dask pyproj`` and the two datasets locally.
Run::

    python validation/external/tidal_forcing_gz.py \
        --bedmachine <BedMachineAntarctica_*_V04.1.nc> \
        --tide-dir <dir containing CATS2008/> \
        --json reports/tidal_forcing_gz.json --fig reports/tidal_forcing_gz.png
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

RHO_I = 917.0    # ice density [kg m^-3] (matches validators.rtn_validator.RHO_I)
RHO_W = 1028.0   # seawater density [kg m^-3] (matches run_rtn_bedmachine.RHO_W)
G = 9.81


def _percentiles(a, qs=(5, 25, 50, 75, 90, 95)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def sample_grounding_zone(bedmachine_path, stride=8, gl_dist_km=10.0, n_sample=1500):
    """Pick grounded BedMachine cells within ``gl_dist_km`` of the grounding line.

    Returns a dict of 1-D arrays (lon, lat, H, bed, dist_km) for up to
    ``n_sample`` evenly-subsampled near-grounding-line grounded cells, plus the
    cell size used.
    """
    from external.bedmachine_loader import load_fields, MASK_GROUNDED
    from external.run_rtn_bedmachine import distance_to_groundingline_km
    import pyproj

    d = load_fields(bedmachine_path, fields=("thickness", "bed", "mask"), stride=stride)
    H = d["thickness"]; bed = d["bed"]; mask = d["mask"]
    x = d["x"]; y = d["y"]; cellsize = d["_meta"]["cellsize"]

    grounded = (mask == MASK_GROUNDED) & np.isfinite(H) & (H > 0) & np.isfinite(bed)
    dist = distance_to_groundingline_km(grounded, cellsize, xp=np)
    sel = grounded & (dist > 0) & (dist <= gl_dist_km)
    rows, cols = np.where(sel)
    if rows.size == 0:
        raise RuntimeError("no grounding-zone cells selected")

    # deterministic even subsample
    if rows.size > n_sample:
        idx = np.linspace(0, rows.size - 1, n_sample).round().astype(int)
        rows, cols = rows[idx], cols[idx]

    xs = x[cols]; ys = y[rows]
    # BedMachine grid is EPSG:3031 (Antarctic Polar Stereographic) in metres
    tr = pyproj.Transformer.from_crs(3031, 4326, always_xy=True)
    lon, lat = tr.transform(xs, ys)
    return {
        "lon": np.asarray(lon, float), "lat": np.asarray(lat, float),
        "H": H[rows, cols].astype(float), "bed": bed[rows, cols].astype(float),
        "dist_km": dist[rows, cols].astype(float), "cellsize_km": cellsize / 1000.0,
        "n_groundingzone_total": int(np.count_nonzero(sel)),
    }


def tidal_amplitude(model_dir, lon, lat, days=30, dt_hours=1.0,
                    start=(2020, 1, 1), cutoff_km=25.0):
    """Half peak-to-peak CATS2008 tidal amplitude ``eta_amp`` [m] per point."""
    from external.tide_loader import load_tide
    n = int(round(days * 24 / dt_hours))
    t0 = datetime.datetime(*start)
    times = [t0 + datetime.timedelta(hours=i * dt_hours) for i in range(n)]
    eta = load_tide(model_dir, x=lon, y=lat, times=times,
                    extrapolate=True, cutoff=cutoff_km)  # (npts, ntime)
    amp = 0.5 * (np.nanmax(eta, axis=1) - np.nanmin(eta, axis=1))
    return amp


def analyse(bedmachine_path, tide_dir, stride=8, gl_dist_km=10.0, n_sample=1500,
            days=30):
    gz = sample_grounding_zone(bedmachine_path, stride=stride,
                               gl_dist_km=gl_dist_km, n_sample=n_sample)
    amp = tidal_amplitude(tide_dir, gz["lon"], gz["lat"], days=days)

    H = gz["H"]; bed = gz["bed"]
    d_base = np.maximum(0.0, -bed)
    p_i = RHO_I * G * H
    dp = RHO_W * G * amp                      # tidal ocean-pressure swing [Pa]
    N_flot = p_i - RHO_W * G * d_base         # flotation effective pressure [Pa]
    ratio = dp / p_i                          # lower bound on eps (connectivity-free)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_flot = np.where(N_flot > 1.0, dp / N_flot, np.nan)  # [HYP] connectivity-limited

    valid = np.isfinite(amp) & (amp >= 0)
    n_valid = int(np.count_nonzero(valid))

    def stats(a):
        return _percentiles(a[valid])

    # how the lower-bound forcing grows toward the grounding line
    bins = [0, 2, 4, 6, 8, 10]
    by_dist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = valid & (gz["dist_km"] >= lo) & (gz["dist_km"] < hi)
        by_dist.append({"lo_km": lo, "hi_km": hi, "n": int(np.count_nonzero(m)),
                        "eta_amp_med_m": float(np.nanmedian(amp[m])) if m.any() else None,
                        "ratio_med": float(np.nanmedian(ratio[m])) if m.any() else None})

    summary = {
        "dataset": "CATS2008 (USAP-DC 601235) x BedMachine Antarctica v4 (NSIDC-0756)",
        "test": ("real tidal forcing at the Antarctic grounding zone: tidal "
                 "ocean-pressure swing vs ice overburden, and the implied §I.5 "
                 "effective-pressure modulation eps"),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat().replace("+00:00", "Z"),
        "cellsize_km": gz["cellsize_km"], "gl_dist_km": gl_dist_km,
        "n_groundingzone_total": gz["n_groundingzone_total"],
        "n_sampled": int(amp.size), "n_valid": n_valid, "tide_days": days,
        "eta_amp_m": stats(amp),
        "tidal_pressure_swing_kPa": _percentiles(dp[valid] / 1e3),
        "ratio_dp_over_pi": stats(ratio),                 # lower bound on eps
        "eps_flot_HYP": _percentiles(eps_flot[valid]),    # connectivity-limited
        "by_distance_to_gl": by_dist,
    }
    arrays = {"lon": gz["lon"], "lat": gz["lat"], "dist_km": gz["dist_km"],
              "eta_amp": amp, "ratio": ratio, "eps_flot": eps_flot, "valid": valid}
    return summary, arrays


def make_figure(summary, arrays, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = arrays["valid"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    sc = ax[0].scatter(arrays["lon"][v], arrays["lat"][v], c=arrays["eta_amp"][v],
                       s=6, cmap="viridis")
    ax[0].set_title("CATS2008 tidal amplitude at GZ cells [m]")
    ax[0].set_xlabel("lon"); ax[0].set_ylabel("lat")
    fig.colorbar(sc, ax=ax[0], shrink=0.8)

    ax[1].hist(100 * arrays["ratio"][v][np.isfinite(arrays["ratio"][v])], bins=40,
               color="#2c7fb8")
    ax[1].set_xlabel(r"$\Delta p / p_i$ [%]  (lower bound on $\epsilon$)")
    ax[1].set_ylabel("GZ cells")
    ax[1].set_title("Tidal pressure swing / ice overburden")

    bd = summary["by_distance_to_gl"]
    xs = np.arange(len(bd))
    ax[2].bar(xs, [100 * (b["ratio_med"] or 0) for b in bd], color="#c0392b")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([f"{b['lo_km']}-{b['hi_km']}" for b in bd])
    ax[2].set_xlabel("distance to grounding line [km]")
    ax[2].set_ylabel(r"median $\Delta p/p_i$ [%]")
    ax[2].set_title("Forcing grows toward the GL")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bedmachine", required=True)
    ap.add_argument("--tide-dir", required=True,
                    help="pyTMD parent dir containing CATS2008/")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--gl-dist-km", type=float, default=10.0)
    ap.add_argument("--n-sample", type=int, default=1500)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    summary, arrays = analyse(args.bedmachine, args.tide_dir, stride=args.stride,
                              gl_dist_km=args.gl_dist_km, n_sample=args.n_sample,
                              days=args.days)
    print("=== real tidal forcing at the Antarctic grounding zone ===")
    print(f"  sampled {summary['n_valid']}/{summary['n_sampled']} GZ cells "
          f"(of {summary['n_groundingzone_total']} within {summary['gl_dist_km']:g} km, "
          f"@ {summary['cellsize_km']:g} km)")
    e = summary["eta_amp_m"]; r = summary["ratio_dp_over_pi"]; ef = summary["eps_flot_HYP"]
    print(f"  tidal amplitude eta_amp [m]      : median {e['p50']:.3f}  (p5 {e['p5']:.3f}, p95 {e['p95']:.3f})")
    print(f"  Delta p / p_i  [%] (eps floor)   : median {100*r['p50']:.2f}  (p5 {100*r['p5']:.2f}, p95 {100*r['p95']:.2f})")
    print(f"  eps_flot [HYP, connectivity-lim] : median {ef['p50']:.3f}  (p90 {ef['p90']:.3f})")
    print("  median Delta p/p_i by distance-to-GL:")
    for b in summary["by_distance_to_gl"]:
        rm = "nan" if b["ratio_med"] is None else f"{100*b['ratio_med']:.2f}%"
        print(f"    {b['lo_km']}-{b['hi_km']} km : {rm}  (n={b['n']})")

    if args.fig:
        make_figure(summary, arrays, args.fig)
        print(f"figure -> {os.path.normpath(args.fig)}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        print(f"json -> {os.path.normpath(args.json_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
