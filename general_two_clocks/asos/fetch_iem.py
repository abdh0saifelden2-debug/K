"""Fetch IEM ASOS 1-minute data for the two-clocks station set (P0-R2).

Downloads Jan-Mar 2020 ("2020Q1", the window of the committed 3-station
analysis) for an extended latitude ladder of stations, writing
``data_asos/<ID>_2020Q1.csv`` in exactly the schema ``asos/loader.py`` reads
(station, station_name, lat, lon, valid(UTC), tmpf, pres1).

Source: Iowa Environmental Mesonet 1-minute ASOS archive
        https://mesonet.agron.iastate.edu/request/asos/1min.phtml

Usage:  python3 asos/fetch_iem.py [--out-dir data_asos]
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request

BASE = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
        "?station={st}&vars=tmpf,pres1&sts={sts}&ets={ets}"
        "&sample=1min&what=download&delim=comma&gis=yes")

# latitude ladder, tropical -> subarctic, coastal + interior mix
STATIONS = (
    "EYW",   # Key West FL         24.6 N  subtropical marine (SJU/HNL are
             #                              not in the IEM 1-min archive)
    "MIA",   # Miami FL            25.8 N  (committed set)
    "MSY",   # New Orleans LA      30.0 N  Gulf coast
    "DFW",   # Dallas-Ft Worth TX  32.9 N  (committed set)
    "PHX",   # Phoenix AZ          33.4 N  interior SW
    "DSM",   # Des Moines IA       41.5 N  (committed set)
    "CAR",   # Caribou ME          46.9 N  NE continental
    "SEA",   # Seattle WA          47.4 N  Pacific coast
)

WINDOW = ("2020-01-01T00:00Z", "2020-04-01T00:00Z")


def fetch(station: str, out_dir: str, sts: str = WINDOW[0],
          ets: str = WINDOW[1], tag: str = "2020Q1") -> str:
    url = BASE.format(st=station, sts=sts, ets=ets)
    path = os.path.join(out_dir, f"{station}_{tag}.csv")
    with urllib.request.urlopen(url, timeout=300) as resp:
        data = resp.read()
    if not data.startswith(b"station,"):
        raise RuntimeError(f"{station}: unexpected response "
                           f"({data[:80]!r})")
    with open(path, "wb") as fh:
        fh.write(data)
    n = data.count(b"\n") - 1
    print(f"  {station}: {n} rows -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_asos"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"IEM ASOS 1-min fetch: {len(STATIONS)} stations, {WINDOW[0]}..{WINDOW[1]}")
    for st in STATIONS:
        for attempt in (1, 2, 3):
            try:
                fetch(st, args.out_dir)
                break
            except Exception as exc:
                print(f"  {st}: attempt {attempt} failed ({exc}); retrying")
                time.sleep(10 * attempt)
        time.sleep(3)          # be polite to IEM


if __name__ == "__main__":
    main()
