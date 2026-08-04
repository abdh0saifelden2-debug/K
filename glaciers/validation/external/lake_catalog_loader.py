r"""Subglacial-lake-drainage & GPS loader for the §V.2 sliding-law test (stub).

Data sources
------------
* Active subglacial lake drainage catalogue (Smith et al., 2009; Siegfried &
  Fricker, 2018) -- timing & volume of drainage events (the forcing ``q_water``
  step times for the §G.4 lag test).
* GPS / GNSS basal-velocity time series at ice-stream stations (UNAVCO / POLENET,
  e.g. Whillans/Kamb Ice Stream) -- the response ``u_b(t)``.

Auth / access
-------------
GPS time series are open via the EarthScope/UNAVCO data archive but require
outbound network (unavailable here).  The lake catalogue is published as
supplementary tables; drop a CSV locally.

Returned fields
---------------
* ``event_times`` : 1-D array of drainage-event times (same units as the GPS
  time axis) -- consumed by ``sliding_validator.validate_lags``.
* ``t``, ``u_b`` : GPS time axis and basal velocity -- consumed by
  ``sliding_validator.estimate_lag`` / ``validate_lags``.
"""
from __future__ import annotations

import os

from . import DataUnavailableError


def load_drainage_events(csv_path=None, time_col="t", **kw):
    """Load lake-drainage event times from a local CSV.

    Raises
    ------
    DataUnavailableError
        If ``csv_path`` is missing -- with a provisioning hint.
    """
    if csv_path is None or not os.path.exists(csv_path):
        raise DataUnavailableError(
            "Lake-drainage catalogue CSV not found locally.\n"
            "Provision it from the published supplementary tables "
            "(Siegfried & Fricker 2018; Smith et al. 2009), save as CSV with a\n"
            f"'{time_col}' column, then call load_drainage_events(csv_path=...).\n"
        )
    import numpy as np  # pragma: no cover - needs real data

    import csv  # pragma: no cover
    times = []  # pragma: no cover
    with open(csv_path, newline="") as fh:  # pragma: no cover
        for row in csv.DictReader(fh):
            times.append(float(row[time_col]))
    return np.asarray(times, float)  # pragma: no cover


def load_gps_velocity(csv_path=None, time_col="t", vel_col="u_b"):
    """Load a GPS basal-velocity time series ``(t, u_b)`` from a local CSV.

    Raises
    ------
    DataUnavailableError
        If ``csv_path`` is missing -- with a provisioning hint.
    """
    if csv_path is None or not os.path.exists(csv_path):
        raise DataUnavailableError(
            "GPS velocity CSV not found locally. This VM has no network.\n"
            "Provision it from the EarthScope/UNAVCO archive (e.g. POLENET\n"
            "Whillans/Kamb stations), save as CSV, then call\n"
            "load_gps_velocity(csv_path=...).\n"
        )
    import numpy as np  # pragma: no cover - needs real data

    import csv  # pragma: no cover
    t, u = [], []  # pragma: no cover
    with open(csv_path, newline="") as fh:  # pragma: no cover
        for row in csv.DictReader(fh):
            t.append(float(row[time_col]))
            u.append(float(row[vel_col]))
    return np.asarray(t, float), np.asarray(u, float)
