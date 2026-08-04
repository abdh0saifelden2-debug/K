"""Build the committed NR72 cache: MATCHED neutral (NRLMSIS 2.1 via pymsis) +
ionosphere (IRI2016) profiles at identical epochs/locations, for the F2-peak
chemistry-clock/diffusion-clock crossover measurement. Both are the mainstream
empirical reference models; NR72 replays offline from the committed JSON.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "nr72_f2peak_cache.json")

ALT = (150.0, 650.0, 5.0)

CONDITIONS = {
    "midlat_equinox_noon":     dict(time=datetime(2019, 3, 20, 12), lat=45.0, lon=0.0),
    "midlat_equinox_midnight": dict(time=datetime(2019, 3, 20, 0),  lat=45.0, lon=0.0),
    "midlat_winter_noon":      dict(time=datetime(2019, 12, 21, 12), lat=45.0, lon=0.0),
    "midlat_summer_noon":      dict(time=datetime(2019, 6, 21, 12), lat=45.0, lon=0.0),
    "equator_equinox_noon":    dict(time=datetime(2019, 3, 20, 12), lat=0.0, lon=0.0),
}

IRI_VARS = ("ne", "Te", "Ti")


def main():
    import iri2016
    import pymsis

    alts = np.arange(ALT[0], ALT[1] + ALT[2] / 2, ALT[2])
    cache = {"_models": "NRLMSIS 2.1 (pymsis) + IRI2016 (iri2016), matched",
             "_alt_km": [float(a) for a in alts], "conditions": {}}
    for name, c in CONDITIONS.items():
        t = np.datetime64(c["time"].isoformat())
        out = pymsis.calculate(t, c["lon"], c["lat"], alts)
        out = np.squeeze(out)   # (nalt, 11)
        msis = dict(
            N2=[float(x) for x in out[:, pymsis.Variable.N2]],
            O2=[float(x) for x in out[:, pymsis.Variable.O2]],
            O=[float(x) for x in out[:, pymsis.Variable.O]],
            Tn=[float(x) for x in out[:, pymsis.Variable.TEMPERATURE]])
        sim = iri2016.IRI(c["time"], ALT, c["lat"], c["lon"])
        alt_iri = np.asarray(sim["alt_km"].values).ravel()
        iri = {v: [float(x) for x in
                   np.interp(alts, alt_iri,
                             np.asarray(sim[v].values).ravel())]
               for v in IRI_VARS}
        cache["conditions"][name] = dict(
            time=c["time"].isoformat(), lat=c["lat"], lon=c["lon"],
            hmF2_km=float(np.asarray(sim["hmF2"].values).ravel()[0]),
            NmF2=float(np.asarray(sim["NmF2"].values).ravel()[0]),
            msis=msis, iri=iri)
        print(f"{name}: hmF2={cache['conditions'][name]['hmF2_km']:.0f} km")
    with open(OUT, "w") as fh:
        json.dump(cache, fh)
    print(f"cache -> {OUT} ({os.path.getsize(OUT)/1024:.0f} kB)")


if __name__ == "__main__":
    main()
