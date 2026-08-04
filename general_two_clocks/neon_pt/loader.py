"""Load NEON bundled eddy-covariance (DP4.00200) HDF5 files into tidy series.

The NEON Surface-Atmosphere Exchange (NSAE) HDF5 bundle stores each variable as a
compound dataset with ``timeBgn`` / ``timeEnd`` (ISO-8601 UTC byte strings) plus
statistics such as ``mean``, ``min``, ``max`` and ``vari``. This module flattens
the handful of variables we need for the pressure/temperature analysis into a
single time-indexed :class:`pandas.DataFrame`.

Nothing here is specific to a particular site/month; the site code is read from
the top-level group, and the UTC->local offset is read from the file metadata.
"""

from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd


def find_h5(data_dir: str) -> str:
    """Return the single NSAE ``.h5`` file under ``data_dir`` (searched recursively)."""
    matches = sorted(glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"No NEON .h5 file found under {data_dir!r}. "
            "Download/unzip the NEON eddy-flux bundle there first."
        )
    # Prefer the net surface-atmosphere exchange ('nsae') bundle if several exist.
    nsae = [m for m in matches if "nsae" in os.path.basename(m).lower()]
    return (nsae or matches)[0]


def _site_code(f: h5py.File) -> str:
    # Top level holds the 4-letter site group alongside metadata groups
    # ('objDesc', 'readMe'); pick the data group.
    meta_groups = {"objDesc", "readMe"}
    sites = [k for k in f.keys() if k not in meta_groups]
    if len(sites) != 1:
        raise ValueError(f"Expected a single site group, found: {sites}")
    return sites[0]


def _to_datetime(byte_iso: np.ndarray) -> pd.DatetimeIndex:
    """Convert an array of ISO-8601 byte strings to a tz-naive UTC DatetimeIndex."""
    return pd.to_datetime([b.decode() for b in byte_iso], utc=True).tz_localize(None)


def _read_stat(f: h5py.File, path: str, field: str = "mean") -> pd.Series:
    ds = f[path][:]
    idx = _to_datetime(ds["timeBgn"])
    return pd.Series(np.asarray(ds[field], dtype=float), index=idx)


@dataclass
class SiteMeta:
    site: str
    lat: float
    lon: float
    elevation_m: float
    utc_to_local_hours: int
    time_zone: str
    canopy_height_m: float
    ecosystem: str
    displacement_m: float
    meas_height_m: float

    @property
    def z_minus_d(self) -> float:
        """Aerodynamic measurement height above the zero-plane displacement."""
        return self.meas_height_m - self.displacement_m


def read_meta(f: h5py.File, site: str) -> SiteMeta:
    a = f[site].attrs

    def scalar(key, cast):
        v = a[key]
        v = v[0] if isinstance(v, np.ndarray) else v
        if isinstance(v, bytes):
            v = v.decode()
        return cast(v)

    eco = a.get("TypeEco")
    if isinstance(eco, np.ndarray):
        eco = ", ".join(x.decode() if isinstance(x, bytes) else str(x) for x in eco)

    # Tower-top measurement height = highest measurement level on the tower.
    levels = a.get("DistZaxsLvlMeasTow")
    if isinstance(levels, np.ndarray):
        meas_height = max(float(x.decode() if isinstance(x, bytes) else x) for x in levels)
    else:
        meas_height = scalar("DistZaxsTow", float)

    return SiteMeta(
        site=site,
        lat=scalar("LatTow", float),
        lon=scalar("LonTow", float),
        elevation_m=scalar("ElevRefeTow", float),
        utc_to_local_hours=scalar("TimeDiffUtcLt", int),
        time_zone=scalar("ZoneTime", str),
        canopy_height_m=scalar("DistZaxsCnpy", float),
        ecosystem=str(eco),
        displacement_m=scalar("DistZaxsDisp", float),
        meas_height_m=meas_height,
    )


# Map of output column -> (hdf5 path suffix under <site>, compound field).
# Paths use the 30-minute (dp01) products to align with the dp04 turbulent fluxes.
_VARS_30M = {
    "pres_kpa": ("dp01/data/presBaro/000_000_30m/presAtm", "mean"),
    "pres_vari": ("dp01/data/presBaro/000_000_30m/presAtm", "vari"),
    "temp_air_c": ("dp01/data/tempAirTop/000_080_30m/temp", "mean"),
    "temp_air_vari": ("dp01/data/tempAirTop/000_080_30m/temp", "vari"),
    "sw_in_wm2": ("dp01/data/radiNet/000_080_30m/radiSwIn", "mean"),
    "lw_in_wm2": ("dp01/data/radiNet/000_080_30m/radiLwIn", "mean"),
    "wind_u_ms": ("dp01/data/soni/000_080_30m/veloXaxsErth", "mean"),
    "wind_v_ms": ("dp01/data/soni/000_080_30m/veloYaxsErth", "mean"),
    "wind_w_ms": ("dp01/data/soni/000_080_30m/veloZaxsErth", "mean"),
    "wind_u_vari": ("dp01/data/soni/000_080_30m/veloXaxsErth", "vari"),
    "wind_v_vari": ("dp01/data/soni/000_080_30m/veloYaxsErth", "vari"),
    "wind_w_vari": ("dp01/data/soni/000_080_30m/veloZaxsErth", "vari"),
    "wind_dir_deg": ("dp01/data/soni/000_080_30m/angZaxsErth", "mean"),
    "temp_soni_c": ("dp01/data/soni/000_080_30m/tempSoni", "mean"),
    "temp_soni_vari": ("dp01/data/soni/000_080_30m/tempSoni", "vari"),
}

# dp04 turbulent fluxes have the layout <var>/turb (and nsae/stor) with field 'flux'.
_FLUX_30M = {
    "H_sensible_wm2": ("dp04/data/fluxTemp/turb", "flux"),
    "LE_latent_wm2": ("dp04/data/fluxH2o/turb", "flux"),
    "fco2_umol": ("dp04/data/fluxCo2/turb", "flux"),
    "ustar_ms": ("dp04/data/fluxMome/turb", "veloFric"),
}


def load_dataframe(data_dir: str) -> tuple[pd.DataFrame, SiteMeta]:
    """Load the 30-minute analysis variables into one DataFrame plus site metadata.

    The returned DataFrame is indexed by UTC time and also carries a
    ``local_time`` column and a ``hour_local`` column (decimal local hour) used by
    the diurnal-composite analyses.
    """
    path = find_h5(data_dir)
    with h5py.File(path, "r") as f:
        site = _site_code(f)
        meta = read_meta(f, site)

        cols = {}
        missing = []
        for name, (suffix, field) in {**_VARS_30M, **_FLUX_30M}.items():
            full = f"{site}/{suffix}"
            if full in f:
                cols[name] = _read_stat(f, full, field)
            else:
                missing.append(name)
        if missing:
            warnings.warn(
                f"NEON HDF5 missing {len(missing)} variable(s) "
                f"(first 5: {missing[:5]}); they will be absent from the DataFrame",
                stacklevel=2,
            )

    if not cols:
        raise ValueError(
            f"No recognized variables found in {path!r} under site group {site!r}"
        )
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "time_utc"

    # Wind speed (horizontal) from earth-frame components.
    if {"wind_u_ms", "wind_v_ms"}.issubset(df.columns):
        df["wind_speed_ms"] = np.hypot(df["wind_u_ms"], df["wind_v_ms"])

    # Absolute temperature (Kelvin) for ideal-gas tests.
    if "temp_air_c" in df:
        df["temp_air_k"] = df["temp_air_c"] + 273.15

    local = df.index + pd.to_timedelta(meta.utc_to_local_hours, unit="h")
    df["local_time"] = local
    df["hour_local"] = local.hour + local.minute / 60.0
    return df, meta


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "data"
    frame, m = load_dataframe(d)
    print(m)
    print(frame.describe().T)
