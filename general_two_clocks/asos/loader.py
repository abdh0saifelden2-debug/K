"""Load IEM ASOS 1-minute CSV files into tidy DataFrames.

Each CSV (from the Iowa Environmental Mesonet 1-min ASOS archive) has columns:
  station, station_name, lat, lon, valid(UTC), tmpf, pres1

We convert to metric and add a local solar time column for diurnal composites.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class StationData:
    station_id: str
    station_name: str
    lat: float
    lon: float
    df: pd.DataFrame  # DatetimeIndex (UTC), columns: temp_c, pres_hpa, hour_local


def load_station(path: str | Path) -> StationData:
    """Read a single ASOS 1-min CSV and return cleaned StationData."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ASOS CSV not found: {path}")
    raw = pd.read_csv(path, low_memory=False)

    # Parse the UTC timestamp.
    raw["valid(UTC)"] = pd.to_datetime(raw["valid(UTC)"], errors="coerce")
    raw = raw.dropna(subset=["valid(UTC)"])
    if raw.empty:
        raise ValueError(
            f"No valid timestamps found in {path}; cannot build StationData"
        )
    raw = raw.set_index("valid(UTC)").sort_index()

    # Extract station metadata from first row.
    station_id = str(raw["station"].iloc[0]).strip()
    station_name = str(raw["station_name"].iloc[0]).strip()
    lat = float(raw["lat"].iloc[0])
    lon = float(raw["lon"].iloc[0])

    # Convert temperature: °F → °C.
    raw["tmpf"] = pd.to_numeric(raw["tmpf"], errors="coerce")
    temp_c = (raw["tmpf"] - 32.0) * 5.0 / 9.0

    # Convert station pressure: inHg → hPa.
    raw["pres1"] = pd.to_numeric(raw["pres1"], errors="coerce")
    pres_hpa = raw["pres1"] * 33.8639

    # Build clean DataFrame.
    df = pd.DataFrame({"temp_c": temp_c, "pres_hpa": pres_hpa})
    df.index.name = "time_utc"

    # QC: physical range.
    df.loc[(df["temp_c"] < -50) | (df["temp_c"] > 60), "temp_c"] = np.nan
    df.loc[(df["pres_hpa"] < 850) | (df["pres_hpa"] > 1100), "pres_hpa"] = np.nan

    # Local solar time (hours) from longitude: LST = UTC_hour + lon/15.
    utc_hours = df.index.hour + df.index.minute / 60.0
    df["hour_local"] = (utc_hours + lon / 15.0) % 24.0

    return StationData(
        station_id=station_id,
        station_name=station_name,
        lat=lat,
        lon=lon,
        df=df,
    )
