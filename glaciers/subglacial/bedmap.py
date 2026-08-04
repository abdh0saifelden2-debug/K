r"""Load a real glacier bedrock transect and map it onto the cavity geometry.

The Part-9 solver builds its rock bed from a 1-D height profile ``ybed(x)``.  By
default that profile is a synthetic sum of sines; this module instead supplies a
profile taken from a *real, measured* Antarctic bedrock transect (BEDMAP1
airborne-radar standardised data points, British Antarctic Survey / SCAR Bedmap,
DOI:10.5285/f64815ec-4077-4432-9f55-0ce230f46029, CC-BY-4.0).

A measured transect is neither periodic nor non-dimensional, so to embed it in the
doubly-periodic spectral box we:

  1. mirror it (segment + its reverse) so the profile is continuous and periodic
     across x = 0 .. 2*pi (no spurious spectral jump at the seam);
  2. resample to the grid resolution n;
  3. non-dimensionalise: recentre to the cavity's mean bed height and rescale the
     peak-to-peak relief to 2*bed_amp, so the real *shape* drives the geometry on
     the same amplitude scale as the idealized run (keeping the two comparable).

The real relief structure (asymmetric bumps, overdeepenings, varying wavelengths)
is preserved; only the absolute units are mapped into the cavity.
"""

from __future__ import annotations

import os
import warnings

import numpy as np


def load_bed_transect(path: str):
    """Read the standardised transect CSV; return (along_track_m, bedrock_m)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Bedrock transect file not found: {path!r}"
        )
    rows = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.lower().startswith("along_track"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                skipped += 1
                continue
    if not rows:
        raise ValueError(
            f"No valid (along_track, bedrock) rows in {path!r}"
        )
    if skipped:
        warnings.warn(
            f"Skipped {skipped} malformed (non-numeric) row(s) while parsing "
            f"{path!r}; kept {len(rows)}. A large skip count may indicate a "
            f"wrong delimiter or encoding.",
            stacklevel=2,
        )
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def clean_sorted_transect(path: str):
    """Load the transect, drop fill values, and sort by along-track distance.

    Returns (dist, bed) as monotone-in-distance finite arrays.
    """
    dist, bed = load_bed_transect(path)
    # drop any residual fill values and enforce monotone along-track distance
    good = np.isfinite(dist) & np.isfinite(bed) & (bed > -9000.0)
    dist, bed = dist[good], bed[good]
    order = np.argsort(dist)
    return dist[order], bed[order]


# backward-compatible private alias
_clean_sorted_transect = clean_sorted_transect


def _embed_periodic(seg, n: int, bed_mean: float, bed_amp: float):
    """Map a 1-D bedrock segment onto a periodic, non-dimensional ``ybed(x)``.

    Mirror the segment (segment + its reverse) so it is continuous and periodic
    across the seam, resample to ``n`` columns, then non-dimensionalise so the
    mean is ``bed_mean`` and the peak-to-peak relief is ``2*bed_amp``.
    """
    mirrored = np.concatenate([seg, seg[::-1]])
    xp = np.linspace(0.0, 1.0, len(mirrored), endpoint=False)
    xq = np.linspace(0.0, 1.0, n, endpoint=False)
    h = np.interp(xq, xp, mirrored)
    h = h - h.mean()
    ptp = float(h.max() - h.min())
    if ptp > 0:
        h = h / ptp * (2.0 * bed_amp)
    return h + bed_mean


def bed_profile_from_transect(path: str, n: int, bed_mean: float,
                              bed_amp: float):
    """Map a real bedrock transect onto a periodic ``ybed(x)`` of length ``n``.

    Returns (ybed, meta) where ybed is the non-dimensional bed-top height at each
    of the n grid columns and meta carries provenance for plotting/reporting.
    """
    dist, bed = _clean_sorted_transect(path)

    relief_m = float(bed.max() - bed.min())
    length_km = float((dist.max() - dist.min()) / 1000.0)

    h = _embed_periodic(bed, n, bed_mean, bed_amp)

    meta = {
        "relief_m": relief_m,
        "length_km": length_km,
        "n_points": int(bed.size),
        "raw_dist_m": dist,
        "raw_bed_m": bed,
    }
    return h, meta


def bed_window_profile(path: str, n: int, bed_mean: float, bed_amp: float,
                       frac_lo: float, frac_hi: float):
    """Map a *sub-window* of the measured transect onto a periodic ``ybed(x)``.

    The single BEDMAP1 flight line is 220 km long.  Taking distinct along-track
    windows ``[frac_lo, frac_hi]`` (fractions of the line) yields several
    independent *real* bed segments -- each with its own measured relief
    structure -- so a multi-site melt-rate comparison uses genuinely different
    measured topography rather than synthetic variants.  This is not a set of
    named glaciers (Thwaites/PIG would need their own datasets); it is multiple
    real sub-transects of one Antarctic flight line.
    """
    dist, bed = _clean_sorted_transect(path)

    m = bed.size
    i0 = int(np.clip(round(frac_lo * m), 0, m - 2))
    i1 = int(np.clip(round(frac_hi * m), i0 + 2, m))
    seg = bed[i0:i1]
    seg_dist = dist[i0:i1]

    relief_m = float(seg.max() - seg.min())
    length_km = float((seg_dist.max() - seg_dist.min()) / 1000.0)

    h = _embed_periodic(seg, n, bed_mean, bed_amp)

    meta = {
        "relief_m": relief_m,
        "length_km": length_km,
        "n_points": int(seg.size),
        "frac_lo": float(frac_lo),
        "frac_hi": float(frac_hi),
    }
    return h, meta
