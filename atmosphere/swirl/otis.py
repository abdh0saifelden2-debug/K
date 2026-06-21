r"""Calibrate the Part-10 swirl to a real hurricane vortex (Hurricane Otis, 2023).

Hurricane Otis (EP182023) underwent record-fast rapid intensification into a
category-5 landfall near Acapulco on 25 October 2023, with an extremely tight,
deep eye.  From the NHC best track (HURDAT2, bundled in ``swirl/data``) we read
its peak-intensity vortex parameters -- minimum central pressure ``pc``, maximum
sustained wind ``Vmax``, and radius of maximum wind ``RMW`` -- and fit a Holland
(1980) parametric vortex.

The Holland *shape* parameter B is set by the real wind-pressure relation,

    B = rho * e * Vmax^2 / dp ,     dp = p_env - pc ,

and the gradient-balanced tangential profile (Coriolis neglected for this small,
intense core) is

    V(r)/Vmax = sqrt( (RMW/r)^B * exp(1 - (RMW/r)^B) ) ,
    p(r)      = pc + dp * exp(-(RMW/r)^B) .

We feed that radial *shape* to the swirl solver as the target tangential profile it
sustains.  Only the shape (radial structure / peakedness) and the
pressure-deficit-to-swirl relation come from data; absolute size and speed are
non-dimensionalised to the resolvable spectral box -- the real ~5 nmi pinhole eye is
sub-grid at n=128 -- exactly as Part 9's real bed is non-dimensionalised to the
cavity.  This grounds the vortex *geometry* in measurements; it is not a hurricane
model (see the report's scope section).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

KT_TO_MS = 0.514444          # knots -> m/s
NMI_TO_KM = 1.852            # nautical miles -> km
MB_TO_PA = 100.0             # millibar -> pascal
RHO_AIR = 1.15               # kg/m^3, near-surface tropical air
ENV_PRESSURE_MB = 1007.0     # Otis far-environment pressure (HURDAT2 outer track)


@dataclass
class OtisVortex:
    """Peak-intensity vortex parameters read from the Otis best track, plus the
    fitted Holland shape.  Lengths in km, speed in m/s, pressure in mb."""
    iso_time: str
    vmax_kt: float
    vmax_ms: float
    pc_mb: float
    penv_mb: float
    dp_mb: float
    rmw_nmi: float
    rmw_km: float
    r34_nmi: float               # mean 34-kt wind radius (vortex footprint)
    holland_B: float


def load_otis_track(path: str):
    """Read the bundled Otis best-track CSV; return a list of per-row dicts.

    Missing/​unanalysed fields (blank in the CSV) become ``float('nan')``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Otis best-track CSV not found: {path!r}"
        )
    rows = []
    header = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if header is None:
                header = parts
                continue
            rec = {}
            for key, val in zip(header, parts):
                if key in ("iso_time", "status"):
                    rec[key] = val
                else:
                    rec[key] = float(val) if val != "" else float("nan")
            rows.append(rec)
    if not rows:
        raise ValueError(f"No data rows found in {path!r}")
    return rows


def _quadrant_mean(rec, prefix):
    vals = [rec[f"{prefix}_{q}_nmi"] for q in ("ne", "se", "sw", "nw")]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    return float(np.mean(vals)) if vals else float("nan")


def holland_B(vmax_ms: float, dp_pa: float, rho: float = RHO_AIR) -> float:
    """Holland (1980) shape parameter from the wind-pressure relation, clamped to
    the physically observed envelope B in [1.0, 2.5]."""
    if dp_pa <= 0:
        return 1.5
    B = rho * np.e * vmax_ms ** 2 / dp_pa
    return float(np.clip(B, 1.0, 2.5))


def peak_vortex(path: str) -> OtisVortex:
    """Find the peak-intensity record (minimum central pressure) and fit Holland B."""
    rows = load_otis_track(path)
    valid = [r for r in rows if np.isfinite(r["mslp_mb"])]
    if not valid:
        raise ValueError(
            f"No records with finite mslp_mb in {path!r}; "
            "cannot determine peak-intensity vortex"
        )
    rec = min(valid, key=lambda r: r["mslp_mb"])
    pc = rec["mslp_mb"]
    penv = ENV_PRESSURE_MB
    dp = penv - pc
    vmax_ms = rec["vmax_kt"] * KT_TO_MS
    B = holland_B(vmax_ms, dp * MB_TO_PA)
    return OtisVortex(
        iso_time=rec["iso_time"],
        vmax_kt=rec["vmax_kt"],
        vmax_ms=vmax_ms,
        pc_mb=pc,
        penv_mb=penv,
        dp_mb=dp,
        rmw_nmi=rec["rmw_nmi"],
        rmw_km=rec["rmw_nmi"] * NMI_TO_KM,
        r34_nmi=_quadrant_mean(rec, "r34"),
        holland_B=B,
    )


def holland_speed_shape(r, rmw, B):
    """Normalised Holland tangential speed V(r)/Vmax (peaks at r = rmw)."""
    r = np.asarray(r, dtype=float)
    x = (rmw / np.maximum(r, 1e-12)) ** B          # (RMW/r)^B
    return np.sqrt(np.clip(x * np.exp(1.0 - x), 0.0, None))


def holland_pressure_deficit(r, rmw, B):
    """Normalised pressure deficit (p_env - p(r)) / dp  (1 at core, 0 far away)."""
    r = np.asarray(r, dtype=float)
    x = (rmw / np.maximum(r, 1e-12)) ** B
    return 1.0 - np.exp(-x)


def target_uth_factory(vortex: OtisVortex, U_swirl: float, r_core_box: float):
    """Return a callable ``uth(r)`` giving the target azimuthal speed on the box.

    The real Holland *shape* (parameter B, the peakedness set by Otis's Vmax-dp
    relation) is preserved; the radius of maximum wind is mapped to ``r_core_box``
    and the peak speed to ``U_swirl`` (non-dimensional box units).  An outer
    Gaussian taper keeps the vortex localized and periodic in the spectral box.
    """
    B = vortex.holland_B

    def uth(r):
        shape = holland_speed_shape(r, r_core_box, B)
        taper = np.exp(-(r ** 2) / (2.0 * (1.8 * r_core_box) ** 2))
        return U_swirl * shape * taper

    return uth
