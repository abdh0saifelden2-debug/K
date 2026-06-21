r"""BedMachine ice-thickness loader for the §V.1 RTN validator (stub).

Data source
-----------
BedMachine Antarctica v3 / Greenland v5 (Morlighem et al., 2020),
distributed by NSIDC: https://nsidc.org/data/nsidc-0756 (Antarctica),
https://nsidc.org/data/idbmg4 (Greenland).  NetCDF, ~500 m posting.

Auth / access
-------------
Requires a (free) NASA Earthdata login.  Typical fetch with ``earthaccess``::

    pip install earthaccess
    python -c "import earthaccess; earthaccess.login()"   # stores ~/.netrc
    # then earthaccess.search_data(short_name='NSIDC-0756', ...)

The development VM has **no outbound access to NSIDC**, so this loader does not
download anything.  Drop the NetCDF locally and point ``path`` at it.

Returned fields (what the validator needs)
------------------------------------------
``thickness`` [m] on a regular grid, plus ``x``/``y`` coordinates.  The RTN
validator only consumes ``thickness`` (-> ice overburden ``rho_i g H``); bed
elevation is optional context.
"""
from __future__ import annotations

import os

from . import DataUnavailableError


def load_thickness(path=None, var="thickness"):
    """Load BedMachine ice thickness ``H_ice(x, y)`` [m] from a local NetCDF.

    Parameters
    ----------
    path : str
        Path to a local BedMachine NetCDF file.
    var : str
        Thickness variable name (BedMachine uses ``"thickness"``).

    Returns
    -------
    dict with keys ``thickness`` (2-D ndarray, m), ``x``, ``y`` (1-D ndarrays).

    Raises
    ------
    DataUnavailableError
        If ``path`` is missing -- with a provisioning hint.
    """
    if path is None or not os.path.exists(path):
        raise DataUnavailableError(
            "BedMachine NetCDF not found locally. This VM cannot reach NSIDC.\n"
            "Provision it with a NASA Earthdata login, e.g.:\n"
            "  pip install earthaccess netCDF4\n"
            "  python -c \"import earthaccess; earthaccess.login()\"\n"
            "  # download NSIDC-0756 (Antarctica) or IDBMG4 (Greenland)\n"
            "then call load_thickness(path='/path/to/BedMachineAntarctica-v3.nc')."
        )
    import netCDF4  # local import: optional dependency, only for real runs

    with netCDF4.Dataset(path) as ds:  # pragma: no cover - needs real data
        H = ds.variables[var][:].astype("float64")
        x = ds.variables["x"][:].astype("float64")
        y = ds.variables["y"][:].astype("float64")
    return {"thickness": H, "x": x, "y": y}


# BedMachine Antarctica ``mask`` codes (Morlighem et al. 2020, NSIDC-0756):
#   0 ocean, 1 ice-free land, 2 grounded ice, 3 floating ice (shelf),
#   4 Lake Vostok.  Only ``mask == 2`` is grounded ice; the §V.1 directional
#   RTN test measures distance from each grounded cell to the nearest
#   non-grounded cell (grounding line / ice margin / coast).
MASK_GROUNDED = 2


def load_fields(path=None, fields=("thickness", "bed", "surface", "mask"),
                stride=1):
    """Load named BedMachine fields from a local NetCDF (real-data §V.1 run).

    Mirrors ``bedmap2_loader.load_fields`` so ``run_rtn_bedmachine`` can reuse
    the Bedmap2 RTN pipeline on this *independent* (NSIDC BedMachine) thickness
    source.  ``thickness``/``bed``/``surface`` are returned as ``float64`` with
    NODATA -> ``nan``; ``mask`` stays integer (codes above).  ``stride`` block-
    decimates (stride=2 on the 500 m grid -> 1 km) and the recorded
    ``_meta['cellsize']`` is scaled to match.

    Returns a dict with the requested ``fields`` plus ``x``, ``y`` (1-D, m) and
    ``_meta`` (``cellsize`` [m], ``nrows``, ``ncols``, ``stride``).

    Raises
    ------
    DataUnavailableError
        If ``path`` is missing -- with the Earthdata provisioning hint.
    """
    if path is None or not os.path.exists(path):
        raise DataUnavailableError(
            "BedMachine NetCDF not found locally. This VM cannot reach NSIDC.\n"
            "Provision it with a NASA Earthdata login, e.g.:\n"
            "  pip install earthaccess netCDF4\n"
            "  python -c \"import earthaccess; earthaccess.login()\"\n"
            "  # download NSIDC-0756 (Antarctica), then call\n"
            "  load_fields(path='/path/to/BedMachineAntarctica-v3.nc')."
        )
    import numpy as np
    import netCDF4  # local import: optional dependency, only for real runs

    out = {}
    with netCDF4.Dataset(path) as ds:  # pragma: no cover - needs real data
        x = ds.variables["x"][:].astype("float64")
        y = ds.variables["y"][:].astype("float64")
        cellsize = float(abs(x[1] - x[0]))  # native posting (computed pre-stride)
        for f in fields:
            v = ds.variables[f]
            # Read the strided slice *directly* from the NetCDF rather than
            # materialising the full native grid (13333^2 float64 ~ 1.4 GB per
            # field) and decimating in RAM.  ``v[::stride, ::stride]`` is
            # value-identical to ``v[:][::stride, ::stride]`` (netCDF4 applies
            # the same auto-masking + scale_factor/add_offset on the strided
            # read) but bounds peak memory by ~stride^2, so the real-data §V.1
            # run fits a small-RAM CPU box (previously GPU / large-RAM only).
            sub = v[::stride, ::stride] if stride > 1 else v[:]
            if f == "mask":
                # netCDF4 auto-masks _FillValue; fill any masked cells with an
                # explicit non-grounded sentinel (-1) instead of letting the raw
                # integer fill value (e.g. -128 for int8) pass through.  -1 is
                # distinct from the BedMachine codes 0-4, so a masked mask cell
                # can never be misread as grounded (== MASK_GROUNDED).
                a = np.ma.filled(sub, -1).astype("int16")
            else:
                # netCDF4 auto-masks _FillValue (and applies any
                # scale_factor/add_offset) by default, so the slice is a masked
                # array with NODATA masked in *scaled* space.  Filling the
                # mask with NaN is robust to scaling, unlike comparing the
                # scaled data against the unscaled _FillValue attribute.
                a = np.ma.filled(sub.astype("float64"), np.nan)
            out[f] = a
        if stride > 1:
            x = x[::stride]
            y = y[::stride]
    out["x"] = x
    out["y"] = y
    ref = out[fields[0]]
    out["_meta"] = {
        "cellsize": cellsize * stride,
        "cellsize_full": cellsize,
        "nrows": ref.shape[0],
        "ncols": ref.shape[1],
        "stride": stride,
    }
    return out
