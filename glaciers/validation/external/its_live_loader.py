r"""ITS_LIVE surface-velocity loader for §V.1 / §V.2 (stub).

Data source
-----------
ITS_LIVE ice-surface velocity mosaics & time series (Gardner et al., 2019),
https://its-live.jpl.nasa.gov/ .  Zarr / NetCDF, ~120 m, 1985-present.

Auth / access
-------------
ITS_LIVE is open (AWS S3, ``s3://its-live-data``) but still needs outbound
network -- unavailable on this VM.  Typical access::

    pip install xarray zarr s3fs
    import xarray as xr
    ds = xr.open_dataset("s3://its-live-data/.../*.zarr", engine="zarr",
                         storage_options={"anon": True})

Returned fields
---------------
``vx``, ``vy`` [m yr^-1] surface velocity on a grid (for RTN's discharge
estimate via mass continuity), or a ``u_s(t)`` time series at a point (for the
§V.2 sliding-law lag test).
"""
from __future__ import annotations

import os

from . import DataUnavailableError


def load_velocity(path=None):
    """Load ITS_LIVE ``vx``/``vy`` [m yr^-1] from a local Zarr/NetCDF.

    Raises
    ------
    DataUnavailableError
        If ``path`` is missing -- with a provisioning hint.
    """
    if path is None or not os.path.exists(path):
        raise DataUnavailableError(
            "ITS_LIVE data not found locally. This VM cannot reach the S3 bucket.\n"
            "Provision it with:\n"
            "  pip install xarray zarr s3fs\n"
            "  # open s3://its-live-data/... (anon=True) and save a local subset,\n"
            "  # then call load_velocity(path='/path/to/its_live_subset.nc').\n"
        )
    import xarray as xr  # pragma: no cover - needs real data

    ds = xr.open_dataset(path)  # pragma: no cover
    return {"vx": ds["vx"].values, "vy": ds["vy"].values}
