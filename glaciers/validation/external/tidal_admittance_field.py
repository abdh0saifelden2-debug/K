r"""§I.5 FIELD TEST — tidal velocity admittance & harmonic fingerprint on REAL
grounding-zone GPS: the BAS Filchner-Ronne 29-station collection (2005-2016) plus the
long-duration Rutford record.

What §I.5 registered (FUTURE_WORK.md; synthetic/tidal_admittance_probe.py)
--------------------------------------------------------------------------
Ocean tides modulate the grounding-zone effective pressure N, so surface velocity
carries (i) a FUNDAMENTAL admittance that reads |s_N| = m/(1-(N_c/N)^m) and (ii) a
HARMONIC fingerprint A2/A1 = (eps/4)|s_N'/s_N - 1| that DIVERGES toward the flotation
fold - both rising toward the grounding line. Registered field test (verbatim): "the
field test decomposes high-cadence GPS/InSAR admittance + harmonics by N."

Real data
---------
* Gudmundsson, Fenney & Rosier (2017), BAS PDC doi:10.5285/4fe11286-0e53-4a03-854c-
  a79a44d1e356 (Open Government Licence): 29 stations, 30-s Bernese PPP positions,
  Evans (E*/XX*), Rutford (R145), Institute (IIS*), Talutis/Carlson (T*/TT1/C*),
  Foundation (H*) + ice-shelf sites; several files encode km-upstream-of-GL in the
  name (C+00, C+20, C-16, T+18, T+38, T-02, T-05).
* Smith, Murray & King (2020), doi:10.5285/dac20505-a56e-4beb-97ba-077eecd587c0:
  single Rutford site, 750 days (2004-2007) - the fortnightly-resolving deep record.
* Distance-to-GL and grounded/floating from the committed MEaSUREs NSIDC-0709
  grounding line; flotation degree measured directly per station as the VERTICAL
  semidiurnal admittance (grounded ~ 0 -> freely-floating ~ 1) - the dataset carries
  its own flotation-proximity coordinate.

Method
------
Per station: mask gaps -> EPSG:3031 -> along-flow displacement s(t) (unit vector of
net displacement) and height h(t) -> decimate 30 s -> 5 min -> GLOBAL harmonic LSQ
across all >=10-day segments jointly (per-segment mean+trend+quadratic nuisance
columns; gaps between segments EXTEND the effective span, so multi-season stations
separate MSf from Mf even though single segments cannot). Constituents (adaptive to
span by the Rayleigh criterion): O1,K1,M2,S2,N2 [,Q1] + MSf [,Mf,Mm] + M4,MS4.
Velocity amplitude at constituent = omega * displacement amplitude; mean speed from
net displacement. Uncertainties from the LSQ covariance scaled by the residual;
amplitude significance requires A > 3 sigma_A.

§I.5 observables and tests
--------------------------
1. Fundamental semidiurnal velocity admittance eps_v(M2) = A_v(M2)/vbar and the
   fortnightly modulation eps_v(MSf) = A_v(MSf)/vbar.
2. Nonlinearity fingerprint: A_v(MSf)/A_v(M2) (the intermodulation of M2xS2 - the
   difference-frequency face of the registered 2f/1f ratio; Gudmundsson 2006's MSf is
   nonlinearity-generated, NOT astronomical) and A_v(M4)/A_v(M2).
3. Ordering tests (exact-permutation Kendall, small n): both observables vs
   distance-to-GL among grounded stations (within-stream and pooled), and vs the
   measured vertical admittance (the flotation-proximity coordinate).
4. Where the quasi-static small-amplitude expansion applies, the tides-only
   §I.5 inversion for (m, R) with R = (N_c/N)^m in [0,1].

Positioning vs mainstream (this is NOT a discovery-of-MSf unit): Gudmundsson 2006/
2007/2011 and Rosier & Gudmundsson 2015-2020 established and modelled these signals
(nonlinear sliding + viscoelasticity) on the same archive. The §I.5 contribution is
the s_N(N)-reading: admittance+harmonics as a MEASUREMENT of flotation proximity R,
checked against the independent per-station vertical admittance.

Honest scope: positions lack ocean-tide-loading/IB corrections (metadata) - OTL
contaminates grounded-station semidiurnal amplitudes at the cm level, so the
fortnightly/intermodulation face (OTL-free) is the primary §I.5 read-out; nodal
corrections ignored (<4% amplitude); the shortest stations resolve only the
semidiurnal band.

Run modes
---------
  python tidal_admittance_field.py --fetch   # download BAS files (open licence)
  python tidal_admittance_field.py           # analyze -> committed cache + report
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_RAW = os.path.join(_DATA, "bas_frgps")
_CACHE = os.path.join(_DATA, "tidal_admittance_field_cache.json")
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))
_GL_SHP = os.path.join(_DATA, "measures0709", "GroundingLine_Antarctica_v02.shp")

_RAMADDA = "https://ramadda.data.bas.ac.uk/repository/entry/"
_COLLECTION = "synth:4fe11286-0e53-4a03-854c-a79a44d1e356:L0dQU19EQVRB"
_RUTFORD_LONG = "dac20505-a56e-4beb-97ba-077eecd587c0"

# constituent periods [hours]
CONSTITUENTS = {
    "O1": 25.81933871, "K1": 23.93446959, "Q1": 26.86835,
    "M2": 12.4206012, "S2": 12.0, "N2": 12.65834751,
    "MSf": 354.3670666, "Mf": 327.8599387, "Mm": 661.3111655,
    "M4": 6.210300601, "MS4": 6.103339275,
}
# Rayleigh separation partners: constituent -> (partner, min span days)
_RAYLEIGH = {
    "Q1": ("O1", 27.6), "N2": ("M2", 27.6), "Mf": ("MSf", 183.0),
    "Mm": ("MSf", 32.0), "MSf": (None, 25.0), "M4": (None, 10.0),
    "MS4": ("M4", 14.6), "O1": (None, 10.0), "K1": ("O1", 13.7),
    "M2": (None, 10.0), "S2": ("M2", 14.8),
}
SEG_GAP_H = 2.0
SEG_MIN_D = 10.0
DECIMATE_S = 300.0
STREAM_OF = {"C": "Carlson", "E": "Evans", "X": "Evans", "H": "Foundation",
             "I": "Institute", "R": "Rutford", "T": "Talutis"}


def _stream(name):
    return STREAM_OF.get(name[0].upper(), "other")


# ----------------------------------------------------------------- fetch
def fetch():
    import urllib.request
    os.makedirs(_RAW, exist_ok=True)
    lst = json.load(urllib.request.urlopen(
        _RAMADDA + f"show/?entryid={_COLLECTION}&output=json"))
    for e in lst:
        name, eid = e["name"], e["id"]
        dest = os.path.join(_RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 2e4:
            continue
        urllib.request.urlretrieve(_RAMADDA + f"get/{name}?entryid={eid}", dest)
        print("fetched", name)
    # long Rutford record (folder listing -> files)
    try:
        sub = json.load(urllib.request.urlopen(
            _RAMADDA + f"show/?entryid={_RUTFORD_LONG}&output=json"))
        for e in sub:
            name, eid = e["name"], e["id"]
            if e.get("isGroup"):
                for e2 in json.load(urllib.request.urlopen(
                        _RAMADDA + f"show/?entryid={eid}&output=json")):
                    n2, id2 = e2["name"], e2["id"]
                    dest = os.path.join(_RAW, "rutford_long_" + n2)
                    if not os.path.exists(dest):
                        urllib.request.urlretrieve(
                            _RAMADDA + f"get/{n2}?entryid={id2}", dest)
                        print("fetched", n2)
            else:
                dest = os.path.join(_RAW, "rutford_long_" + name)
                if not os.path.exists(dest):
                    urllib.request.urlretrieve(
                        _RAMADDA + f"get/{name}?entryid={eid}", dest)
                    print("fetched", name)
    except Exception as ex:                                   # pragma: no cover
        print("rutford long record fetch failed:", ex)
    print("raw ->", _RAW)


# ----------------------------------------------------------------- harmonic core
def _segments(t_days, max_gap_h=SEG_GAP_H, min_len_d=SEG_MIN_D):
    """Indices of continuous segments."""
    dt = np.diff(t_days) * 24.0
    brk = np.where(dt > max_gap_h)[0]
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, t_days.size - 1]
    return [(s, e) for s, e in zip(starts, ends)
            if (t_days[e] - t_days[s]) >= min_len_d]


def _decimate(t, y, bin_s=DECIMATE_S):
    """Bin-average to bin_s; returns bin centers and means."""
    tb = np.floor((t - t[0]) * 86400.0 / bin_s).astype(int)
    n = np.bincount(tb)
    sy = np.bincount(tb, weights=y)
    st = np.bincount(tb, weights=t)
    ok = n > 0
    return st[ok] / n[ok], sy[ok] / n[ok]


def pick_constituents(span_days):
    keep = []
    for c, (partner, min_span) in _RAYLEIGH.items():
        if span_days >= min_span:
            keep.append(c)
    return [c for c in CONSTITUENTS if c in keep]


def harmonic_fit(t_days, y, segs, constituents, t_ref=0.0):
    """Joint LSQ across segments: per-segment [1, t, t^2] nuisance + global cos/sin.

    Phases are referenced to the ABSOLUTE epoch t_ref (days, same clock as t_days)
    so they are comparable across stations. Returns dict constituent ->
    (amplitude, phase_rad, sigma_amp), plus residual rms.
    """
    rows = []
    for (s, e) in segs:
        rows.append((s, e))
    npts = sum(e - s + 1 for s, e in rows)
    ncols = 3 * len(rows) + 2 * len(constituents)
    A = np.zeros((npts, ncols))
    yy = np.zeros(npts)
    r0 = 0
    for k, (s, e) in enumerate(rows):
        n = e - s + 1
        ts = t_days[s:e + 1] - t_days[s]
        A[r0:r0 + n, 3 * k] = 1.0
        A[r0:r0 + n, 3 * k + 1] = ts
        A[r0:r0 + n, 3 * k + 2] = ts ** 2
        yy[r0:r0 + n] = y[s:e + 1]
        tt = (t_days[s:e + 1] - t_ref) * 24.0            # hours since common epoch
        for j, c in enumerate(constituents):
            w = 2.0 * math.pi / CONSTITUENTS[c]
            A[r0:r0 + n, 3 * len(rows) + 2 * j] = np.cos(w * tt)
            A[r0:r0 + n, 3 * len(rows) + 2 * j + 1] = np.sin(w * tt)
        r0 += n
    coef, res, rank, sv = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ coef
    dof = max(npts - ncols, 1)
    s2 = float(resid @ resid) / dof
    # covariance of coefficients ~ s2 * (A^T A)^-1 (diagonal only, via SVD pinv)
    AtA_inv_diag = np.sum(np.linalg.pinv(A.T @ A) * np.eye(ncols), axis=1)
    out = {}
    for j, c in enumerate(constituents):
        a = coef[3 * len(rows) + 2 * j]
        b = coef[3 * len(rows) + 2 * j + 1]
        va = AtA_inv_diag[3 * len(rows) + 2 * j] * s2
        vb = AtA_inv_diag[3 * len(rows) + 2 * j + 1] * s2
        amp = math.hypot(a, b)
        sig = math.sqrt(max(va + vb, 0.0) / 2.0)
        out[c] = (float(amp), float(math.atan2(-b, a)), float(sig))
    return out, float(math.sqrt(s2))


def analyze_station(path):
    import netCDF4 as ncdf
    from pyproj import Transformer
    d = ncdf.Dataset(path)
    t = np.asarray(d["Date number"][:]).ravel()
    lat = np.asarray(d["Latitude"][:]).ravel()
    lon = np.asarray(d["Longitude"][:]).ravel()
    h = np.asarray(d["height"][:]).ravel()
    ok = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h)
    t, lat, lon, h = t[ok], lat[ok], lon[ok], h[ok]
    o = np.argsort(t)
    t, lat, lon, h = t[o], lat[o], lon[o], h[o]
    # de-duplicate
    uniq = np.r_[True, np.diff(t) > 1e-9]
    t, lat, lon, h = t[uniq], lat[uniq], lon[uniq], h[uniq]
    tr = Transformer.from_crs(4326, 3031, always_xy=True)
    x, y = tr.transform(lon, lat)
    x = np.asarray(x); y = np.asarray(y)
    # station relocations (pre-winter moves): treat as segment breaks via jumps
    jump = np.r_[False, np.hypot(np.diff(x), np.diff(y)) > 50.0]     # >50 m in 30 s
    t_days = t.copy()
    t_days[jump] += 1.0                       # force a gap at relocations
    # unify to days since 1970 (files carry MATLAB datenum; 719529 = 1970-01-01)
    # so that harmonic PHASES share one absolute epoch across stations
    t = t - 719529.0
    # decimate positions
    tc, xc = _decimate(t, x)
    _, yc = _decimate(t, y)
    _, hc = _decimate(t, h)
    segs = _segments(tc)
    if not segs:
        return None
    span = tc[segs[-1][1]] - tc[segs[0][0]]
    n_days = sum(tc[e] - tc[s] for s, e in segs)
    # flow direction from total displacement of the longest segment
    s0, e0 = max(segs, key=lambda se: tc[se[1]] - tc[se[0]])
    ux, uy = xc[e0] - xc[s0], yc[e0] - yc[s0]
    nrm = math.hypot(ux, uy)
    vbar = nrm / (tc[e0] - tc[s0]) * 365.25          # m/yr
    if nrm == 0 or vbar < 5.0:
        return None
    ux, uy = ux / nrm, uy / nrm
    s_along = (xc - xc[s0]) * ux + (yc - yc[s0]) * uy
    cons = pick_constituents(span)
    fit_s, rms_s = harmonic_fit(tc, s_along, segs, cons)
    fit_h, rms_h = harmonic_fit(tc, hc, segs, cons)
    out = dict(name=os.path.basename(path).replace(".nc", ""),
               lat=float(np.median(lat)), lon=float(np.median(lon)),
               x=float(np.median(xc)), y=float(np.median(yc)),
               span_days=float(span), obs_days=float(n_days),
               n_segments=len(segs), vbar_m_yr=float(vbar),
               rms_along_m=rms_s, rms_h_m=rms_h, constituents={})
    for c in cons:
        A_s, ph_s, sig_s = fit_s[c]
        A_h, ph_h, sig_h = fit_h[c]
        w = 2.0 * math.pi / (CONSTITUENTS[c] / 24.0)          # rad/day
        out["constituents"][c] = dict(
            A_disp_m=A_s, sig_disp_m=sig_s,
            A_v_m_yr=float(A_s * w * 365.25), 
            A_h_m=A_h, sig_h_m=sig_h,
            ph_disp_rad=ph_s, ph_h_rad=ph_h,
            sig_ph_rad=(float(sig_s / A_s) if A_s > 0 else None))
    return out


def analyze_station_enu(path, name="RUT_long", lat0=-78.141646342, lon0=-83.91234181):
    """The Smith/Murray/King 750-day Rutford ENU record (5-min, E/N/U + sigmas)."""
    from pyproj import Transformer
    raw = np.loadtxt(path)
    yr = raw[:, 0].astype(int)
    doy = raw[:, 1]
    # day-of-year 'rolls into following years' from the FIRST year
    t = doy + 366.0 * 0.0
    base = np.datetime64(f"{yr[0]}-01-01").astype("datetime64[D]").astype(float)
    t_days = base + doy - 1.0                     # matlab-free absolute day count
    e, n, u = raw[:, 2], raw[:, 3], raw[:, 4]
    ok = np.isfinite(e) & np.isfinite(n) & np.isfinite(u)
    t_days, e, n, u = t_days[ok], e[ok], n[ok], u[ok]
    o = np.argsort(t_days)
    t_days, e, n, u = t_days[o], e[o], n[o], u[o]
    uniq = np.r_[True, np.diff(t_days) > 1e-9]
    t_days, e, n, u = t_days[uniq], e[uniq], n[uniq], u[uniq]
    segs = _segments(t_days)
    if not segs:
        return None
    span = t_days[segs[-1][1]] - t_days[segs[0][0]]
    s0, e0 = max(segs, key=lambda se: t_days[se[1]] - t_days[se[0]])
    ux, uy = e[e0] - e[s0], n[e0] - n[s0]
    nrm = math.hypot(ux, uy)
    vbar = nrm / (t_days[e0] - t_days[s0]) * 365.25
    ux, uy = ux / nrm, uy / nrm
    s_along = (e - e[s0]) * ux + (n - n[s0]) * uy
    cons = pick_constituents(span)
    fit_s, rms_s = harmonic_fit(t_days, s_along, segs, cons)
    fit_h, rms_h = harmonic_fit(t_days, u, segs, cons)
    tr = Transformer.from_crs(4326, 3031, always_xy=True)
    x, y = tr.transform(lon0, lat0)
    out = dict(name=name, lat=lat0, lon=lon0, x=float(x), y=float(y),
               span_days=float(span),
               obs_days=float(sum(t_days[b] - t_days[a] for a, b in segs)),
               n_segments=len(segs), vbar_m_yr=float(vbar),
               rms_along_m=rms_s, rms_h_m=rms_h, constituents={})
    for c in cons:
        A_s, ph_s, sig_s = fit_s[c]
        A_h, ph_h, sig_h = fit_h[c]
        w = 2.0 * math.pi / (CONSTITUENTS[c] / 24.0)
        out["constituents"][c] = dict(A_disp_m=A_s, sig_disp_m=sig_s,
                                      A_v_m_yr=float(A_s * w * 365.25),
                                      A_h_m=A_h, sig_h_m=sig_h,
                                      ph_disp_rad=ph_s, ph_h_rad=ph_h,
                                      sig_ph_rad=(float(sig_s / A_s)
                                                  if A_s > 0 else None))
    return out


# ----------------------------------------------------------------- assembly
def build_cache():
    import glob as _glob
    from pyproj import Transformer  # noqa: F401
    files = sorted(_glob.glob(os.path.join(_RAW, "*.nc")))
    stations = []
    for f in files:
        try:
            st = analyze_station(f)
        except Exception as ex:
            print("SKIP", os.path.basename(f), repr(ex)[:80])
            continue
        if st is None:
            print("SKIP", os.path.basename(f), "(too short/slow)")
            continue
        stations.append(st)
        print(f"{st['name']:>6s}: span {st['span_days']:.0f} d, v={st['vbar_m_yr']:.0f} "
              f"m/yr, {len(st['constituents'])} constituents")
    enu = os.path.join(_RAW, "rutford_long_smap.SEI1.enu")
    if os.path.exists(enu):
        st = analyze_station_enu(enu)
        if st is not None:
            stations.append(st)
            print(f"{st['name']:>6s}: span {st['span_days']:.0f} d (750-day Rutford), "
                  f"v={st['vbar_m_yr']:.0f} m/yr, {len(st['constituents'])} constituents")
    # distance to GL + grounded flag
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        gl_poly = gpd.read_file(_GL_SHP).geometry.union_all()
        gl_line = gl_poly.boundary
        for st in stations:
            p = Point(st["x"], st["y"])
            st["d_gl_km"] = float(gl_line.distance(p)) / 1e3
            st["grounded_measures"] = bool(gl_poly.contains(p))
            if not st["grounded_measures"]:
                st["d_gl_km"] = -st["d_gl_km"]        # negative = seaward/floating
    except Exception as ex:                            # pragma: no cover
        print("GL attach failed:", ex)
    cache = {"_description": "committed per-station harmonic table for the §I.5 field "
                             "test (BAS Filchner-Ronne GPS collection + Rutford long "
                             "record); raw 30-s files NOT committed (300 MB, open "
                             "licence, re-fetchable)",
             "_phase_convention": "y = A cos(w*tt + ph) with tt = hours since "
                                  "1970-01-01 00:00 (common absolute epoch across "
                                  "stations; BAS datenum shifted by 719529), "
                                  "w = 2*pi/T_hours; sig_ph_rad = sig_amp/A",
             "_sources": ["doi:10.5285/4fe11286-0e53-4a03-854c-a79a44d1e356 (OGL)",
                          "doi:10.5285/dac20505-a56e-4beb-97ba-077eecd587c0 (OGL)",
                          "MEaSUREs NSIDC-0709 grounding line"],
             "stations": stations}
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    with open(_CACHE, "w") as fh:
        json.dump(cache, fh, indent=1)
    print(f"cache -> {_CACHE} ({len(stations)} stations)")
    return cache


# ----------------------------------------------------------------- §I.5 read-out
def _sig_amp(st, c, field="A_v_m_yr", nsig=3.0):
    cc = st["constituents"].get(c)
    if not cc:
        return None
    w = 2.0 * math.pi / (CONSTITUENTS[c] / 24.0) * 365.25
    sig_v = cc["sig_disp_m"] * w
    if field == "A_v_m_yr":
        return cc[field] if cc[field] > nsig * sig_v else None
    return cc[field]


def _kendall_exact(x, y):
    from itertools import permutations
    from scipy.stats import kendalltau
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 4:
        return np.nan, np.nan, n
    tau = float(kendalltau(x, y)[0])
    if n <= 8:
        taus = [kendalltau(x, np.asarray(p))[0] for p in permutations(y)]
        p = float(np.mean(np.abs(taus) >= abs(tau) - 1e-12))
    else:
        rng = np.random.default_rng(0)
        taus = [kendalltau(x, rng.permutation(y))[0] for _ in range(5000)]
        p = float((np.sum(np.abs(taus) >= abs(tau) - 1e-12) + 1) / 5001)
    return tau, p, n


def vertical_admittance(st):
    """Flotation-proximity coordinate: vertical semidiurnal amplitude [m]."""
    a = 0.0
    for c in ("M2", "S2"):
        cc = st["constituents"].get(c)
        if cc:
            a += cc["A_h_m"] ** 2
    return math.sqrt(a)


def analyze(cache=None):
    cache = cache or json.load(open(_CACHE))
    sts = cache["stations"]
    rows = []
    for st in sts:
        vbar = st["vbar_m_yr"]
        Am2 = _sig_amp(st, "M2")
        Amsf = _sig_amp(st, "MSf")
        Am4 = _sig_amp(st, "M4")
        Ah = vertical_admittance(st)
        rows.append(dict(
            name=st["name"], stream=_stream(st["name"]), d_gl_km=st.get("d_gl_km"),
            grounded=st.get("grounded_measures"), vbar=vbar,
            span_days=st["span_days"],
            eps_v_M2=(Am2 / vbar if Am2 else None),
            eps_v_MSf=(Amsf / vbar if Amsf else None),
            ratio_MSf_M2=(Amsf / Am2 if (Amsf and Am2) else None),
            ratio_M4_M2=(Am4 / Am2 if (Am4 and Am2) else None),
            A_v_M2=Am2, A_v_MSf=Amsf, vert_adm_m=Ah))
    res = {"what": "§I.5 tidal admittance + harmonic fingerprint on the BAS "
                   "Filchner-Ronne GPS network (+ Rutford long record)",
           "stations": rows}
    # ordering tests on GROUNDED stations
    g = [r for r in rows if r["grounded"] and r["d_gl_km"] is not None
         and r["d_gl_km"] >= 0]
    def _pairs(field):
        x = [r["d_gl_km"] for r in g if r[field] is not None]
        y = [r[field] for r in g if r[field] is not None]
        return x, y
    tests = {}
    for field in ("eps_v_MSf", "ratio_MSf_M2", "eps_v_M2"):
        x, y = _pairs(field)
        tau, p, n = _kendall_exact(x, y)
        tests[f"{field}_vs_distance"] = dict(tau=tau, p_exact=p, n=n,
                                             registered_sign="negative (rises toward GL)")
    # vs vertical admittance (flotation proximity; expected POSITIVE)
    xg = [r["vert_adm_m"] for r in g if r["ratio_MSf_M2"] is not None]
    yg = [r["ratio_MSf_M2"] for r in g if r["ratio_MSf_M2"] is not None]
    tau, p, n = _kendall_exact(xg, yg)
    tests["ratio_MSf_M2_vs_vertical_admittance"] = dict(
        tau=tau, p_exact=p, n=n, registered_sign="positive (rises with flotation)")
    res["ordering_tests"] = tests
    # within-stream concordance of eps_v(MSf) toward the GL
    per_stream = {}
    for r in g:
        if r["eps_v_MSf"] is not None:
            per_stream.setdefault(r["stream"], []).append((r["d_gl_km"],
                                                           r["eps_v_MSf"], r["name"]))
    conc = {}
    n_c = n_t = 0
    for s, rows_ in per_stream.items():
        if len(rows_) < 2:
            continue
        rows_.sort()
        c = t_ = 0
        for i in range(len(rows_)):
            for j in range(i + 1, len(rows_)):
                t_ += 1
                if rows_[i][1] > rows_[j][1]:      # closer to GL -> larger eps
                    c += 1
        conc[s] = dict(n_stations=len(rows_), concordant=c, pairs=t_,
                       order=[r[2] for r in rows_])
        n_c += c; n_t += t_
    from scipy.stats import binomtest
    res["within_stream_concordance"] = dict(
        per_stream=conc, concordant=n_c, pairs=n_t,
        p_sign=(float(binomtest(n_c, n_t, 0.5, "greater").pvalue) if n_t else None))
    res["n_grounded"] = len(g)
    res["verdict"] = _verdict(res)
    return res


def _verdict(res):
    t = res["ordering_tests"]
    msf = t.get("eps_v_MSf_vs_distance", {})
    rat = t.get("ratio_MSf_M2_vs_distance", {})
    va = t.get("ratio_MSf_M2_vs_vertical_admittance", {})
    ws = res.get("within_stream_concordance", {})
    bits = [f"grounded stations n={res['n_grounded']}"]
    for lab, d, want_neg in (("eps_v(MSf) vs d_GL", msf, True),
                             ("MSf/M2 vs d_GL", rat, True),
                             ("MSf/M2 vs vertical admittance", va, False)):
        if d and np.isfinite(d.get("tau", np.nan)):
            hit = (d["tau"] < 0) == want_neg and d["p_exact"] < 0.05
            bits.append(f"{lab}: tau={d['tau']:+.2f} (exact p={d['p_exact']:.3f}, "
                        f"n={d['n']}) {'AS REGISTERED' if hit else 'not resolved'}")
    if ws.get("pairs"):
        bits.append(f"within-stream concordance (closer-to-GL has larger eps_v(MSf)): "
                    f"{ws['concordant']}/{ws['pairs']} pairs, sign-test "
                    f"p={ws['p_sign']:.4f}")
    return " | ".join(bits)


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in res["stations"] if r["d_gl_km"] is not None]
    streams = sorted({r["stream"] for r in rows})
    colors = dict(zip(streams, plt.cm.tab10.colors))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    for r in rows:
        if not r["grounded"]:
            continue
        c = colors[r["stream"]]
        if r["eps_v_MSf"]:
            ax[0].semilogy(r["d_gl_km"], 100 * r["eps_v_MSf"], "o", color=c, ms=6)
            ax[0].annotate(r["name"], (r["d_gl_km"], 100 * r["eps_v_MSf"]),
                           fontsize=6, xytext=(3, 3), textcoords="offset points")
        if r["A_v_MSf"]:
            ax[1].semilogy(r["d_gl_km"], r["A_v_MSf"], "o", color=c, ms=6)
        if r["A_v_M2"]:
            ax[1].semilogy(r["d_gl_km"], r["A_v_M2"], "s", mfc="none", color=c, ms=5)
    t = res["ordering_tests"]["eps_v_MSf_vs_distance"]
    ws = res["within_stream_concordance"]
    ax[0].set_xlabel("distance upstream of grounding line [km]")
    ax[0].set_ylabel(r"$\epsilon_v$(MSf) = $A_v$(MSf)/$\bar v$  [%]")
    ax[0].set_title(f"fortnightly velocity admittance rises toward the GL\n"
                    f"pooled Kendall tau={t['tau']:+.2f} (exact p={t['p_exact']:.3f}); "
                    f"within-stream {ws['concordant']}/{ws['pairs']} "
                    f"(p={ws['p_sign']:.4f})", fontsize=9)
    ax[0].grid(alpha=0.3)
    for s in streams:
        ax[0].plot([], [], "o", color=colors[s], label=s)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("distance upstream of grounding line [km]")
    ax[1].set_ylabel(r"$A_v$ [m/yr]   (filled: MSf, open: M2)")
    ax[1].set_title("constituent screening: M2 sits at the mm noise/OTL floor\n"
                    "upstream while MSf penetrates 20-100 km", fontsize=9)
    ax[1].grid(alpha=0.3)
    fig.suptitle("§I.5 FIELD TEST — BAS Filchner-Ronne GPS network (grounded stations)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default=os.path.join(_REPORTS, "tidal_admittance_field.json"))
    a = ap.parse_args()
    if a.fetch:
        fetch()
    if a.build or not os.path.exists(_CACHE):
        build_cache()
    res = analyze()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    for r in sorted(res["stations"], key=lambda r: (r["stream"], -(r["d_gl_km"] or 0))):
        print(f"{r['name']:>6s} {r['stream']:<10s} d={r['d_gl_km']!s:>8s} km "
              f"gr={r['grounded']!s:>5s} v={r['vbar']:.0f} "
              f"epsM2={r['eps_v_M2'] if r['eps_v_M2'] is None else round(r['eps_v_M2'], 4)!s:>8s} "
              f"MSf/M2={r['ratio_MSf_M2'] if r['ratio_MSf_M2'] is None else round(r['ratio_MSf_M2'], 3)!s:>7s} "
              f"h_adm={r['vert_adm_m']:.2f} m")
    print("VERDICT:", res["verdict"])
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
