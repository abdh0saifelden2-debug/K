"""Build the committed NR71 IRI2016 cache: topside ionosphere profiles
(n_e, T_e, T_i, ion composition) for the ambipolar-multiplier measurement.
IRI2016 is the mainstream empirical ionosphere (Bilitza et al.); runs offline
after the fortran build, and NR71 replays from the committed JSON only.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import iri2016

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "nr71_iri_cache.json")

ALT = (150.0, 650.0, 5.0)   # km: start, stop, step

CONDITIONS = {
    # high-dip midlatitude (dip ~73 deg at 60N,0E): diffusive equilibrium along
    # near-vertical field lines -- the clean constraint-multiplier regime
    "midlat_noon":     dict(time=datetime(2019, 3, 20, 12), lat=60.0, lon=0.0),
    "midlat_midnight": dict(time=datetime(2019, 3, 20, 0),  lat=60.0, lon=0.0),
    "midlat_winter_noon": dict(time=datetime(2019, 12, 21, 12), lat=60.0, lon=0.0),
    # equatorial control: fountain/ExB breaks diffusive equilibrium (known
    # deviation regime, honest negative control)
    "equator_noon":    dict(time=datetime(2019, 3, 20, 12), lat=0.0, lon=0.0),
}

VARS = ("ne", "Tn", "Ti", "Te", "nO+", "nH+", "nHe+", "nO2+", "nNO+")


def main():
    cache = {"_model": "IRI2016 (Bilitza et al.), via iri2016 python driver",
             "_vars": list(VARS), "conditions": {}}
    for name, c in CONDITIONS.items():
        sim = iri2016.IRI(c["time"], ALT, c["lat"], c["lon"])
        if "_alt_km" not in cache:
            cache["_alt_km"] = [float(x) for x in
                                np.asarray(sim["alt_km"].values).ravel()]
        prof = {v: [float(x) for x in np.asarray(sim[v].values).ravel()]
                for v in VARS}
        cache["conditions"][name] = dict(
            time=c["time"].isoformat(), lat=c["lat"], lon=c["lon"],
            hmF2_km=float(np.asarray(sim["hmF2"].values).ravel()[0]),
            NmF2=float(np.asarray(sim["NmF2"].values).ravel()[0]),
            profile=prof)
        print(f"{name}: hmF2={cache['conditions'][name]['hmF2_km']:.0f} km, "
              f"NmF2={cache['conditions'][name]['NmF2']:.2e}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(cache, fh)
    print(f"cache -> {OUT} ({os.path.getsize(OUT)/1024:.0f} kB)")


if __name__ == "__main__":
    main()
