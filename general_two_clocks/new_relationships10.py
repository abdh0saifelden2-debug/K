r"""New derived cross-relationship NR33, continuing the program (NR1-NR32) with the same
discipline: derive from mainstream pieces, verify numerically, register falsifiable
corollaries.  CPU-only, deterministic.  Unit-proofs in
``tests/test_new_relationships10.py``.

NR33 -- The persistence-pressure bound: radar persistence of distributed basal
water is a *pressure gauge* -- the slow (geometry) clock reads the level of the
fast (pressure) field
==============================================================================
[Nye 1953 creep closure x Schroeder 2013/2015 specularity x the repo's
 two-clocks operator split; sharpens P4a §6.4 via §V.1d]

The gap this closes
-------------------
Radar bed-echo specularity content is the established *map* of distributed basal
water (Schroeder et al. 2013, 2015; Young et al. 2016; Dow et al. 2020), and NR32
made it the field coordinate of the drainage-response window.  But specularity has
carried no *quantitative pressure meaning* -- the two physically-arguable
spec->p_w sign conventions moved the §V.1c intrusion classification in opposite
directions, and the stated unblock (co-located borehole pressure) does not exist
in East Antarctica.  NR33 derives the missing quantitative link from persistence:

    A basal water body observed specular in repeat surveys separated by t_obs
    carries a POINTWISE LOWER BOUND on the water-pressure fraction phi = p_w/p_i.

Derivation (mainstream pieces; the composition is the contribution)
-------------------------------------------------------------------
1. **Creep closure (Nye 1953; Röthlisberger 1972; Evatt 2006).**  A cavity in ice
   at effective pressure ``N = p_i - p_w`` closes at the Glen-law rate
   ``V_c/S = 2A (N/n)^n`` (cylindrical geometry; ``A ~ 2.4e-24 Pa^-3 s^-1``
   temperate, ``n = 3``).  Absent opening terms the cross-section decays as
   ``exp(-2A(N/n)^n t)``.
2. **The persistence bound (cylinder, rigorous).**  Radar persistence over
   ``t_obs`` requires the closure to have e-folded at most ~once:

       N  <=  N_max(t_obs) = n * (2 A t_obs)^{-1/n},

   i.e. ``phi >= phi_floor = 1 - N_max/p_i``.  Numbers: ``N_max = 3.5 bar`` at
   ``t_obs = 4 yr`` -> ``phi_floor ~ 0.98-0.99`` under 2-4 km of East Antarctic
   ice.  The bound *tightens with archive depth* as ``t^{-1/n}`` (the "archive
   dividend": a 25-yr archive -> 1.9 bar).
3. **Geometry sharpening (sheet, scaling).**  Specular reflectors are *wide flat*
   water bodies (that is what specularity measures).  A wide thin sheet
   (half-width ``w``, gap ``h``, ``w >> h``) presents the load ``N`` over a region
   of depth ``~w``, so the roof sags at ``~A N^n w`` and the gap closes in
   ``t ~ h/(A N^n w)`` -- FASTER than the cylinder by ``~(n^n/2)(w/h)``.  The
   persistence bound sharpens to

       N_max_sheet = (h / (A t_obs w))^{1/n} = N_max_cyl * (2h/(n^n w))^{1/n},

   ~0.1 bar for ``h = 0.1 m, w = 500 m, t_obs = 4 yr`` -- persistent distributed
   specular water sits essentially AT flotation (``phi >~ 0.999``), consistent
   with borehole effective pressures of 0.1-1.6 bar over distributed West
   Antarctic beds (Blankenship et al. 1987; Engelhardt & Kamb 1997) and with
   distributed-system theory operating near overburden (Walder 1986; Kamb 1987;
   Hewitt 2011; Werder et al. 2013).  The cylinder bound stays the conservative
   rigorous anchor; the sheet form is a scaling (prefactor O(1)).
4. **Honest limits.**  (i) The bound assumes quiescence (melt-opening << closure)
   -- valid for low-through-flux distributed/ponded water; a steady R-channel
   evades it by construction, but concentrated channels are LOW-specularity
   objects at km footprints (Schroeder 2013, 2015), so the evasion does not
   contaminate the specular population.  (ii) One-sided only: DISAPPEARANCE of
   specularity does not bound N from below (drainage mimics closure).  (iii)
   ``A`` is the temperate value; cold basal ice lowers ``A`` and *raises*
   ``N_max`` as ``A^{-1/n}`` (a factor ~2 for one decade in ``A``) -- carried as
   the dominant systematic.

Two-clocks reading (the physical reality this exposes)
------------------------------------------------------
In the repo's operator split the basal pressure field is the FAST (elliptic,
instantaneously-adjusting) partner and interface geometry is the SLOW
(parabolic/creep, memory-carrying) partner.  Everywhere else in this program the
fast field constrains the slow one's statistics; NR33 is the converse: a time
integral of the slow clock (geometric persistence) MEASURES the level of the
fast field.  Memory is not just a correction term -- it is a *gauge*: the bed
remembers its pressure history in its geometry, and repeat radar reads that
memory out as a pressure floor.

Registered falsifiable corollaries
----------------------------------
(C1, model constraint) Any subglacial hydrology model (GlaDS-type, Shreve
routing) that assigns ``N > N_max(t_obs)`` on cells that stay specular across
surveys separated by ``t_obs`` is falsified at those cells.  This turns repeat
radar archives into a data-assimilation prior: modelled-N products for the
ICECAP region (e.g. GlaDS Aurora/Totten runs) must satisfy ``N <= 3.5 bar`` on
the §V.1d persistent set (registered 2026-07-04, decidable against any published
modelled-N map).
(C2, sign corollary -> §V.1d) The floor pins the ORIENTATION of any monotone
spec->phi map: an increasing map keeps a free low end, a decreasing map is
squeezed entirely above the floor (admissible range <= 1-phi_floor ~ 0.02), so
the §V.1c "connectivity" convention (claimed range 0.17) is inadmissible --
executed on real data in ``glaciers/validation/external/rtn_spec_sign_pin.py``.
(C3, archive dividend) ``N_max prop t_obs^{-1/3}``: joining ICECAP (2008-2012)
with newer UTIG/OIB East Antarctic reflights should tighten the floor by
``(t2/t1)^{1/3}`` on the persistent set -- a concrete, dated survey-design
prediction (the bound's scaling is testable purely from archive geometry).

Run:
    python general_two_clocks/new_relationships10.py
      -> figures/nr33_persistence_pressure_bound.{json,png}
    pytest general_two_clocks/tests/test_new_relationships10.py -v
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

RHO_I = 917.0
G = 9.81
GLEN_A = 2.4e-24          # Pa^-3 s^-1 (temperate; Cuffey & Paterson 2010)
GLEN_N = 3.0
YR = 365.25 * 24 * 3600.0


# ------------------------------------------------------------- closed forms --
def nmax_cylinder(t_obs_s, A=GLEN_A, n=GLEN_N):
    r"""``N_max = n (2 A t)^{-1/n}`` -- one closure e-fold in ``t_obs``."""
    return n * (2.0 * A * t_obs_s) ** (-1.0 / n)


def nmax_sheet(t_obs_s, h=0.1, w=500.0, A=GLEN_A, n=GLEN_N):
    r"""Sheet-geometry sharpening ``(h/(A t w))^{1/n}`` (scaling, O(1) prefactor)."""
    return (h / (A * t_obs_s * w)) ** (1.0 / n)


def phi_floor(H, t_obs_s, geometry="cylinder", **kw):
    """Lower bound on ``phi = p_w/p_i`` for radar-persistent quiescent water."""
    nmax = (nmax_cylinder(t_obs_s, **kw) if geometry == "cylinder"
            else nmax_sheet(t_obs_s, **kw))
    p_i = RHO_I * G * np.asarray(H, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p_i > 0, 1.0 - nmax / p_i, np.nan)


# ------------------------------------------------------ numerical unit-proofs --
def verify_efold(t_obs_yr=4.0, nstep=200_000):
    """Explicit integration of dS/dt = -2A(N/n)^n S at N = N_max reaches 1/e."""
    t = t_obs_yr * YR
    nmax = nmax_cylinder(t)
    rate = 2.0 * GLEN_A * (nmax / GLEN_N) ** GLEN_N
    s, dt = 1.0, t / nstep
    for _ in range(nstep):
        s *= (1.0 - rate * dt)
    return {"survival_at_Nmax": float(s), "target": float(np.exp(-1.0)),
            "abs_err": float(abs(s - np.exp(-1.0)))}


def verify_sheet_crossover():
    """Sheet and cylinder bounds coincide where ``2h/(n^n w) = 1`` (a tall slot,
    ``w/h = 2/n^n`` -- outside the sheet regime); for every wide flat sheet
    (``w >= h``) the sheet bound is strictly tighter than the cylinder."""
    t = 4.0 * YR
    out = []
    for w_over_h in (2.0 / GLEN_N ** GLEN_N, 1.0, 1e2, 1e3, 1e4):
        h = 0.1
        w = w_over_h * h
        ratio = nmax_sheet(t, h=h, w=w) / nmax_cylinder(t)
        out.append({"w_over_h": w_over_h, "nmax_ratio_sheet_over_cyl": float(ratio)})
    # analytic: ratio = (2h/(n^n w))^{1/n}
    for row in out:
        anal = (2.0 / (GLEN_N ** GLEN_N * row["w_over_h"])) ** (1.0 / GLEN_N)
        row["analytic"] = float(anal)
        row["match"] = bool(abs(row["nmax_ratio_sheet_over_cyl"] - anal) < 1e-12)
    return out


def magnitude_table():
    rows = []
    for t_yr in (1.0, 4.0, 10.0, 25.0):
        for H in (2000.0, 3000.0, 4000.0):
            rows.append({
                "t_obs_yr": t_yr, "H_m": H,
                "Nmax_cyl_bar": float(nmax_cylinder(t_yr * YR) / 1e5),
                "phi_floor_cyl": float(phi_floor(H, t_yr * YR)),
                "Nmax_sheet_bar": float(nmax_sheet(t_yr * YR) / 1e5),
                "phi_floor_sheet": float(phi_floor(H, t_yr * YR,
                                                   geometry="sheet")),
            })
    return rows


def verify_archive_dividend():
    """``N_max(t) prop t^{-1/n}`` exactly (log-log slope -1/3)."""
    ts = np.array([1.0, 2.0, 5.0, 10.0, 25.0]) * YR
    ns = np.array([nmax_cylinder(t) for t in ts])
    slope = np.polyfit(np.log(ts), np.log(ns), 1)[0]
    return {"loglog_slope": float(slope), "target": -1.0 / GLEN_N,
            "abs_err": float(abs(slope + 1.0 / GLEN_N))}


# ----------------------------------------------- data-gated ICECAP application --
def icecap_floor_map(spec_dir="/home/K/_data_specularity",
                     bin_dir="/home/K/_data_bedmap2/bedmap2_bin",
                     stride=5, min_pts=3, spec_thresh=0.2,
                     epochs=(("2008", "2009"), ("2011", "2012")),
                     t_obs_yr=4.0):
    """phi-floor map on the ICECAP two-epoch persistent-specular set (optional:
    returns availability hints instead of fabricating data)."""
    files = glob.glob(os.path.join(spec_dir, "**", "*_spec.txt"), recursive=True)
    if not files:
        return {"available": False,
                "hint": "USAP-DC 601371 profiles not present; see "
                        "glaciers/validation/external/rtn_specularity_real.py"}
    try:
        from pyproj import Transformer  # noqa: F401
    except Exception:
        return {"available": False, "hint": "pyproj not installed"}
    import sys
    vroot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "glaciers", "validation")
    sys.path.insert(0, vroot)
    try:
        from external.bedmap2_loader import load_fields
        from external.rtn_specularity_real import bin_to_bedmap2
    except Exception as ex:
        return {"available": False, "hint": f"validation imports failed: {ex}"}
    if not os.path.isdir(bin_dir):
        return {"available": False, "hint": f"Bedmap2 not present: {bin_dir}"}

    d = load_fields(bin_dir, stride=stride)
    H, mask = d["thickness"], d["icemask_grounded_and_shelves"]
    meta = d["_meta"]
    grounded = np.isfinite(mask) & (mask == 0) & np.isfinite(H) & (H > 0)

    def _epoch(group):
        lons, lats, sps = [], [], []
        for f in files:
            m = re.search(r"(20\d\d)", os.path.relpath(f, spec_dir))
            if m and m.group(1) in group:
                a = np.loadtxt(f, comments="#", usecols=(3, 4, 5), ndmin=2)
                if a.size:
                    lons.append(a[:, 0]); lats.append(a[:, 1]); sps.append(a[:, 2])
        if not lons:
            return None, None
        lon = np.concatenate(lons); lat = np.concatenate(lats)
        sp = np.concatenate(sps)
        good = np.isfinite(sp) & np.isfinite(lon) & np.isfinite(lat)
        return bin_to_bedmap2(lon[good], lat[good], np.clip(sp[good], 0, 1), meta)

    (sA, cA), (sB, cB) = _epoch(epochs[0]), _epoch(epochs[1])
    if sA is None or sB is None:
        return {"available": False, "hint": "one of the epochs has no profiles"}
    seen = grounded & (cA >= min_pts) & (cB >= min_pts)
    persist = seen & (sA >= spec_thresh) & (sB >= spec_thresh)
    fl_cyl = phi_floor(H, t_obs_yr * YR)
    fl_sheet = phi_floor(H, t_obs_yr * YR, geometry="sheet")
    res = {
        "available": True, "stride_km": meta["cellsize"] / 1e3,
        "min_pts": min_pts, "spec_thresh": spec_thresh, "t_obs_yr": t_obs_yr,
        "n_cells_seen_both_epochs": int(seen.sum()),
        "n_persistent_specular": int(persist.sum()),
        "median_H_persistent": float(np.nanmedian(H[persist]))
        if persist.any() else float("nan"),
        "median_phi_floor_cyl": float(np.nanmedian(fl_cyl[persist]))
        if persist.any() else float("nan"),
        "median_phi_floor_sheet": float(np.nanmedian(fl_sheet[persist]))
        if persist.any() else float("nan"),
        "median_Nmax_cyl_bar": float(nmax_cylinder(t_obs_yr * YR) / 1e5),
        "median_Nmax_sheet_bar": float(nmax_sheet(t_obs_yr * YR) / 1e5),
        "registered_C1": ("modelled-N products must satisfy N <= "
                          f"{nmax_cylinder(t_obs_yr * YR) / 1e5:.2f} bar on "
                          "these cells (registered 2026-07-04)"),
    }
    return res


# ----------------------------------------------------------------------- run --
def run():
    out = {
        "relationship": "NR33 persistence-pressure bound",
        "statement": "radar persistence of distributed specular basal water over "
                     "t_obs bounds phi = p_w/p_i >= 1 - n(2A t_obs)^{-1/n}/(rho_i g H) "
                     "(cylinder, rigorous-conservative); sheet geometry sharpens "
                     "N_max by (2h/(n^n w))^{1/n}",
        "two_clocks_reading": "the slow (creep/geometry) clock's time integral "
                              "measures the level of the fast (elliptic pressure) "
                              "field: memory as a gauge, the converse of the "
                              "usual fast-constrains-slow direction",
        "verify_efold": verify_efold(),
        "verify_sheet_crossover": verify_sheet_crossover(),
        "verify_archive_dividend": verify_archive_dividend(),
        "magnitude_table": magnitude_table(),
        "icecap_application": icecap_floor_map(),
    }
    return out


def make_figure(out, path_png):                              # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    t = np.logspace(-0.5, 1.7, 100) * YR
    a = ax[0]
    a.loglog(t / YR, [nmax_cylinder(x) / 1e5 for x in t], label="cylinder (rigorous)")
    a.loglog(t / YR, [nmax_sheet(x) / 1e5 for x in t], "--",
             label="sheet h=0.1m, w=500m (scaling)")
    a.set_xlabel("persistence t_obs [yr]"); a.set_ylabel("N_max [bar]")
    a.set_title("NR33: persistence bounds effective pressure\n(slope -1/n, the archive dividend)")
    a.legend(fontsize=8)
    b = ax[1]
    H = np.linspace(1000, 4500, 200)
    for tt, c in ((1, "C0"), (4, "C1"), (25, "C2")):
        b.plot(H, phi_floor(H, tt * YR), c, label=f"cyl, t={tt} yr")
        b.plot(H, phi_floor(H, tt * YR, geometry="sheet"), c + "--", alpha=0.6)
    b.axhline(0.97, color="k", lw=0.7, ls=":", label="PHI_HI=0.97 (V.1c cap)")
    b.set_xlabel("ice thickness H [m]"); b.set_ylabel("phi floor")
    b.set_ylim(0.9, 1.001); b.legend(fontsize=7)
    b.set_title("phi floors (dashed: sheet scaling)")
    c = ax[2]
    app = out["icecap_application"]
    c.axis("off")
    txt = json.dumps({k: v for k, v in app.items() if k != "registered_C1"},
                     indent=1)[:900]
    c.text(0.0, 1.0, "ICECAP two-epoch application\n" + txt, va="top",
           fontsize=7, family="monospace")
    fig.tight_layout()
    fig.savefig(path_png, dpi=140)
    plt.close(fig)


def main():                                                  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=os.path.join(
        FIGDIR, "nr33_persistence_pressure_bound.json"))
    ap.add_argument("--png-out", default=os.path.join(
        FIGDIR, "nr33_persistence_pressure_bound.png"))
    a = ap.parse_args()
    out = run()
    os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
    with open(a.json_out, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, a.png_out)
    print(json.dumps({"efold": out["verify_efold"],
                      "dividend": out["verify_archive_dividend"],
                      "icecap": out["icecap_application"]}, indent=1))
    print("wrote", a.json_out, "and", a.png_out)


if __name__ == "__main__":
    main()
