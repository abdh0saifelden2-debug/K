r"""CATS2008 ocean-tide loader for the §V.1 RTN validator (stub).

Data source
-----------
CATS2008 circum-Antarctic tide model (Padman et al., 2008; Howard et al., 2019),
distributed via USAP-DC / the ``pyTMD`` tooling.  Gives tidal elevation
``eta(x, y, t)``; the validator converts to ocean pressure at the grounding line
via ``p_ocean = rho_w g (draft + eta)``.

Auth / access
-------------
CATS2008 is distributed via USAP-DC (Howard et al., 2019; doi:10.15784/601235)
behind a **reCAPTCHA bot-check** (no login/account).  There is no programmatic
fetch (pyTMD has no CATS2008 downloader); download ``CATS2008.zip`` in a browser
and unzip so that ``<model_dir>/CATS2008/{grid_CATS2008,hf.CATS2008.out}`` exist.
Prediction is then done locally with ``pyTMD`` (``pip install pyTMD dask``)::

    eta = load_tide(model_dir, x=lon, y=lat, times=times)  # -> eta(x,y,t) [m]

This loader does not fetch the model; provide the model directory locally.

Returned fields
---------------
``p_ocean`` [Pa] on the validator's grid (or a scalar/array of ocean pressure
samples), matching the shape of the BedMachine thickness grid.
"""
from __future__ import annotations

import os

from . import DataUnavailableError

RHO_W = 1027.0   # seawater density [kg m^-3]
G = 9.81


def ocean_pressure_from_draft(draft_m, eta_m=0.0):
    """``p_ocean = rho_w g (draft + eta)`` [Pa].  Pure helper, no I/O.

    ``draft_m`` is the ice draft below sea level [m]; ``eta_m`` is the tidal
    anomaly [m] (from CATS2008).  Usable directly in tests / synthetic runs.
    """
    import numpy as np
    return RHO_W * G * (np.asarray(draft_m, float) + np.asarray(eta_m, float))


def _times_to_delta_seconds(times, epoch):
    """Convert ``times`` to seconds since ``epoch`` (a ``(Y,M,D,h,m,s)`` tuple).

    Accepts a 1-D array of python ``datetime`` / ``numpy.datetime64`` / ISO
    strings, or already-numeric seconds since ``epoch`` (passed through).
    """
    import numpy as np
    import datetime as _dt
    arr = np.atleast_1d(np.asarray(times))
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype("float64")
    if arr.dtype == object:
        arr = np.array([np.datetime64(t) for t in arr], dtype="datetime64[s]")
    else:
        arr = arr.astype("datetime64[s]")
    ep = np.datetime64(_dt.datetime(*epoch), "s")
    return (arr - ep) / np.timedelta64(1, "s")


def load_tide(model_dir=None, x=None, y=None, times=None, *, model="CATS2008",
              crs=4326, epoch=(1992, 1, 1, 0, 0, 0), method="linear",
              extrapolate=True, cutoff=10.0):
    """Predict CATS2008 tidal elevation ``eta(x, y, t)`` [m] from a local model dir.

    Parameters
    ----------
    model_dir : str
        pyTMD *parent* directory that contains the ``CATS2008/`` subfolder
        (``grid_CATS2008``, ``hf.CATS2008.out``).
    x, y : array_like
        Point coordinates -- longitude/latitude by default (``crs=4326``); pass a
        projected CRS via ``crs`` to use model/native coordinates.
    times : array_like
        1-D ``datetime`` / ``numpy.datetime64`` (or numeric seconds since ``epoch``).

    Returns
    -------
    numpy.ndarray, shape ``(n_points, n_times)``
        Tidal elevation [m]; ``NaN`` where a point is outside the model's wet
        domain and cannot be extrapolated within ``cutoff`` km.

    Raises
    ------
    DataUnavailableError
        If ``model_dir`` is missing -- with a provisioning hint.
    """
    if model_dir is None or not os.path.isdir(model_dir):
        raise DataUnavailableError(
            "CATS2008 model directory not found locally.\n"
            "CATS2008 is distributed via USAP-DC (doi:10.15784/601235) behind a\n"
            "reCAPTCHA bot-check (no login). Download CATS2008.zip, unzip so that\n"
            "<model_dir>/CATS2008/{grid_CATS2008,hf.CATS2008.out} exist, then call\n"
            "  load_tide(model_dir=..., x=lon, y=lat, times=...).\n"
            "Needs: pip install pyTMD dask."
        )
    import numpy as np
    import pyTMD.compute  # optional dep: only for real tide runs
    if x is None or y is None or times is None:
        raise ValueError("load_tide requires x, y and times")
    x = np.atleast_1d(np.asarray(x, dtype="float64"))
    y = np.atleast_1d(np.asarray(y, dtype="float64"))
    delta = _times_to_delta_seconds(times, epoch)
    h = pyTMD.compute.tide_elevations(
        x, y, delta, directory=model_dir, model=model, epoch=epoch,
        type="time series", crs=crs, method=method,
        extrapolate=extrapolate, cutoff=cutoff,
    )
    return np.ma.filled(np.ma.asarray(h, dtype="float64"), np.nan)
