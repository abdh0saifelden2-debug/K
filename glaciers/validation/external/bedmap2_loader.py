r"""BAS Bedmap2 loader for the §V.1 RTN validator (REAL data, no auth).
 
Unlike the NSIDC BedMachine loader (Earthdata-gated, unreachable from the dev
VM), **Bedmap2 is openly downloadable** from the British Antarctic Survey with
no authentication::
 
    curl -L -o bedmap2_bin.zip \
      https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_bin.zip
    unzip bedmap2_bin.zip            # -> bedmap2_bin/bedmap2_*.flt (+ .hdr)
 
Each field is an ESRI flat float32 raster (``*.flt``) with an ArcInfo ``*.hdr``
(ncols/nrows/cellsize/NODATA_value/byteorder).  Bedmap2 is a 1 km, 6667x6667
polar-stereographic grid (Fretwell et al., 2013).  No netCDF/GDAL needed -- we
read the raw float32 with numpy.
 
Fields used by the RTN validator: ``thickness`` (ice overburden), ``bed`` (ocean
pressure at the bed via depth below sea level) and
``icemask_grounded_and_shelves`` (to restrict to grounded ice and locate the
grounding line).
 
Resolution caveat (carried into the report): 1 km cannot resolve the ~1-10 m
subglacial channel scale that the Roethlisberger ``N_R`` term needs, so the RTN
run is a **directional** spatial test (does ``RTN>1`` concentrate near grounding
lines?), not a channel-resolved precision/recall score.
"""
from __future__ import annotations
 
import os
 
import numpy as np
 
from . import DataUnavailableError
 
# Bedmap2 polar-stereographic grid constants (from the .hdr files)
BEDMAP2_NODATA = -9999.0
 
 
def _read_hdr(hdr_path):
    meta = {}
    with open(hdr_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                meta[parts[0].lower()] = parts[1]
    return {
        "ncols": int(meta["ncols"]),
        "nrows": int(meta["nrows"]),
        "xll": float(meta["xllcorner"]),
        "yll": float(meta["yllcorner"]),
        "cellsize": float(meta["cellsize"]),
        "nodata": float(meta.get("nodata_value", meta.get("nodata", BEDMAP2_NODATA))),
        "byteorder": meta.get("byteorder", "LSBFIRST"),
    }
 
 
def read_flt(flt_path):
    """Read one Bedmap2 ``*.flt`` raster + its ``*.hdr`` -> (array, meta).
 
    Rows run north->south (top row = +y).  NODATA cells become ``np.nan``.
    """
    hdr_path = os.path.splitext(flt_path)[0] + ".hdr"
    if not (os.path.exists(flt_path) and os.path.exists(hdr_path)):
        raise DataUnavailableError(
            f"Bedmap2 raster not found: {flt_path} (+ .hdr).\n"
            "Provision (open, no auth):\n"
            "  curl -L -o bedmap2_bin.zip "
            "https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_bin.zip\n"
            "  unzip bedmap2_bin.zip\n"
        )
    m = _read_hdr(hdr_path)
    dt = np.dtype("<f4") if m["byteorder"].upper() == "LSBFIRST" else np.dtype(">f4")
    a = np.fromfile(flt_path, dtype=dt).astype("float64")
    a = a.reshape(m["nrows"], m["ncols"])
    a[a == m["nodata"]] = np.nan
    return a, m
 
 
def load_fields(bin_dir, fields=("thickness", "bed", "surface",
                                 "icemask_grounded_and_shelves"), stride=1):
    """Load named Bedmap2 fields from ``bin_dir`` (the unzipped ``bedmap2_bin/``).
 
    ``stride`` block-subsamples by simple decimation (stride=5 -> 5 km) to keep
    the 6667^2 grid tractable; the grid metadata is updated to match.
    """
    if not os.path.isdir(bin_dir):
        raise DataUnavailableError(
            f"Bedmap2 directory not found: {bin_dir}\n"
            "Download + unzip bedmap2_bin.zip (open, no auth) first."
        )
    out = {}
    meta = None
    for f in fields:
        a, m = read_flt(os.path.join(bin_dir, f"bedmap2_{f}.flt"))
        if stride > 1:
            a = a[::stride, ::stride]
        out[f] = a
        meta = m
    if meta is not None:
        # Record the *original* full-resolution grid (cellsize/nrows/ncols) before
        # decimation so coordinate reconstruction is exact even for odd grid sizes
        # (Bedmap2 is 6667x6667: axis_len*stride loses the odd cell). The plain
        # cellsize/nrows/ncols keys remain the *effective* (decimated) values that
        # the rest of the pipeline uses for grid spacing. Only ``nrows_full`` is
        # actually consumed downstream (the north-anchored y axis counts *down* from
        # the top row); ``ncols_full`` is stored for completeness/symmetry, since the
        # x axis counts *up* from ``xll`` and so was already exact without it.
        meta = dict(meta, cellsize_full=meta["cellsize"],
                    nrows_full=meta["nrows"], ncols_full=meta["ncols"], stride=stride)
        if stride > 1:
            meta = dict(meta, cellsize=meta["cellsize_full"] * stride,
                        ncols=out[fields[0]].shape[1], nrows=out[fields[0]].shape[0])
    out["_meta"] = meta
    return out
