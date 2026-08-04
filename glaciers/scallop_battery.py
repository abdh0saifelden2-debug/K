r"""Decisive go/no-go battery for the corrected Candidate 3 (scallop mechanism).

Gives the mechanism its best shot across regimes: stronger mean current (higher
Re -> real separation) and buoyancy (Ri>0 brings warm bottom water toward the
ice). For each regime, probe() returns the flat-wall control and the 3 closures.

Decision (per co-thinker): GO if some closure shows a coherent, bump-locked
net flux enhancement clearly above the flat-wall turbulent floor
(R_mean and R_max meaningfully > flat control). Otherwise NO-GO: the penalised
conduction wall caps the mean interfacial flux.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scallop_probe as sp  # noqa: E402


def main(xp=np, nx=128, ny=128, n_waves=12, spinup=3000, measure=800):
    regimes = [
        {"name": "U0.8_Ri0.0", "U_drive": 0.8, "f_amp": 0.4, "Ri": 0.0},
        {"name": "U1.5_Ri0.0", "U_drive": 1.5, "f_amp": 0.4, "Ri": 0.0},
        {"name": "U0.8_Ri1.0", "U_drive": 0.8, "f_amp": 0.4, "Ri": 1.0},
        {"name": "U1.5_Ri1.0", "U_drive": 1.5, "f_amp": 0.4, "Ri": 1.0},
    ]
    out = {}
    for rg in regimes:
        t0 = time.time()
        meta, res = sp.probe(nx=nx, ny=ny, a=None, n_waves=n_waves,
                             U_drive=rg["U_drive"], f_amp=rg["f_amp"],
                             spinup=spinup, measure=measure, Ri=rg["Ri"], xp=xp)
        out[rg["name"]] = {"meta": meta, "res": res}
        dt = time.time() - t0
        flat, none = res["_flat_control"], res["none"]
        print(f"[{rg['name']}] ({dt:.0f}s) umean={none['umean']:.3f} "
              f"umax={none['umax']:.3f} | FLAT R_mean={flat['R_mean']:.4f} "
              f"R_max={flat['R_max']:.3f} | BUMP(none) R_mean={none['R_mean']:.4f} "
              f"R_max={none['R_max']:.3f} corr_slope={none['corr_excess_slope']:+.3f}",
              flush=True)
    return out


if __name__ == "__main__":
    use_gpu = "--gpu" in sys.argv
    if use_gpu:
        import cupy as cp
        xp = cp
        tag = "gpu"
    else:
        xp = np
        tag = "cpu"
    nx = ny = 128
    spinup, measure = 3000, 800
    if "--hires" in sys.argv:
        nx, ny, spinup, measure = 192, 192, 4000, 1000
    out = main(xp=xp, nx=nx, ny=ny, spinup=spinup, measure=measure)
    path = os.path.join(os.getcwd(), f"scallop_battery_{tag}.json")
    out = sp._json_safe(out)
    # Only OSError (permissions, disk full) is recoverable by echoing to stdout:
    # _json_safe already guarantees finite-only data so allow_nan=False never
    # raises ValueError, and catching it would be a trap -- the fallback
    # json.dumps reuses the same data and allow_nan=False, re-raising it uncaught.
    try:
        with open(path, "w") as f:
            json.dump(out, f, indent=2, allow_nan=False)
        print("WROTE " + path)
    except OSError:
        print("BATTERY_JSON " + json.dumps(out, allow_nan=False))
