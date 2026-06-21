r"""Sample ITS_LIVE 240 m surface speed at the §H.1.3/§H.1.4 grounding-line points.

ITS_LIVE provides an open, EPSG:3031 Antarctic velocity mosaic
(ANT_G0240_0000.nc, ~5.4 GB, netCDF4/HDF5, 64x64 chunks). We never download it
whole: the GL points are grouped by the file's HDF5 chunk grid and each needed
chunk is read once over HTTPS (block cache), then points are sampled. For the
~7 k Konrad GL points this touches ~140 chunks and finishes in seconds.

Output: an npz keyed to rtn_glmig_basin.sample()'s good-point order, carrying
`speed` plus `lon/lat/Ro/tau_d`, consumed by `rtn_glmig_basin.py --itslive-npz`.

Extra deps beyond requirements.txt (external driver, install ad hoc):
    pip install pyproj fsspec aiohttp     # h5py is already a core dependency
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtn_glmig_basin import sample  # noqa: E402

URL = ("https://its-live-data.s3.amazonaws.com/velocity_mosaic/landsat/v00.0/"
       "static/ANT_G0240_0000.nc")


def sample_speed(lon, lat, url=URL, block_mb=2):
    import fsspec
    import h5py
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    px, py = tr.transform(lon, lat)

    fs = fsspec.filesystem("https")
    f = h = None
    try:
        # Both opens are inside the try so that if h5py.File() fails (e.g. a
        # corrupt/partial remote read) the already-open fsspec handle still gets
        # closed by the finally block below, rather than leaking.
        f = fs.open(url, block_size=block_mb * 1024 * 1024, cache_type="blockcache")
        h = h5py.File(f, "r")
        x, y, v = h["x"][:], h["y"][:], h["v"]
        cy, cx = v.chunks
        x0, dx = x[0], x[1] - x[0]
        y0, dy = y[0], y[1] - y[0]
        ix = np.clip(np.round((px - x0) / dx).astype(int), 0, x.size - 1)
        iy = np.clip(np.round((py - y0) / dy).astype(int), 0, y.size - 1)

        spd = np.full(lon.shape, np.nan, dtype="f8")
        keys = (iy // cy) * 10**6 + (ix // cx)
        order = np.argsort(keys)
        uniq, starts = np.unique(keys[order], return_index=True)
        starts = list(starts) + [len(order)]
        t0 = time.time()
        for n, _ in enumerate(uniq):
            rows = order[starts[n]:starts[n + 1]]
            r0 = (iy[rows].min() // cy) * cy
            c0 = (ix[rows].min() // cx) * cx
            bb = np.asarray(v[r0:r0 + cy, c0:c0 + cx], dtype="f8")
            bb[(bb < 0) | (bb > 1e5) | ~np.isfinite(bb)] = np.nan
            spd[rows] = bb[iy[rows] - r0, ix[rows] - c0]
    finally:
        if h is not None:
            h.close()
        if f is not None:
            f.close()
    print(f"sampled {uniq.size} chunks in {time.time()-t0:.0f}s; "
          f"valid speed at {np.isfinite(spd).sum()}/{spd.size} points")
    return spd


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default=os.path.expanduser("~/data_bedmap/bedmap2_bin"))
    ap.add_argument("--konrad", default=os.path.expanduser("~/data_glmig/konrad2018_glmig.txt"))
    ap.add_argument("--dhdt-npz", default=os.path.expanduser("~/data_glmig/dhdt_at_konrad.npz"))
    ap.add_argument("--out", default=os.path.expanduser("~/data_glmig/itslive_speed_at_konrad_good.npz"))
    a = ap.parse_args()

    r = sample(a.bin_dir, a.konrad, a.dhdt_npz)
    spd = sample_speed(r["lon"], r["lat"])
    np.savez(a.out, speed=spd, lon=r["lon"], lat=r["lat"], Ro=r["Ro"], tau_d=r["tau_d"])
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
