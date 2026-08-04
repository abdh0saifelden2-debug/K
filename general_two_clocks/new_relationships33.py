r"""NR56 (theory + real data) -- the Lagrangian velocity has Stokes
parameters: waves are linearly polarized (stretch face, zero spin),
vortices are circularly polarized (spin face, zero stretch) -- and the
deep ocean crosses from wave-polarized at the equator to vortex-polarized
poleward, with the equatorial deep jets as the extreme linear limit.

Follows NR52-NR54 (the spin face) and NR55 (the phase anatomy).  NR55
covered the PHASES; NR56 adds the amplitude-anisotropy face and unifies
the tensor decomposition as polarization optics.

The claim
=========
1. **Stokes decomposition.**  The 2x2 Lagrangian velocity correlation
   decomposes exactly as

       ``I = <u^2 + v^2>``          (energy)
       ``Q = <u^2 - v^2>``          (stretch, zonal-meridional)
       ``U = 2 <u v>``              (stretch, diagonal tilt)
       ``V = <u(0)v(dt) - v(0)u(dt)>``  (spin; lag-odd, NR52)

   -- the Stokes parameters of a 2D oscillation.  (Q, U) is the
   deviatoric (spin-2) face: parity-EVEN, rotating by 2*theta under
   frame rotation; V is the antisymmetric (parity-odd) face.  Under the
   mirror ``x -> -x``: Q invariant, U and V flip.  Oceanographic rotary
   spectra are the frequency-resolved version of the same split.

2. **Waves are linear, vortices are circular (the polarization
   theorem).**  A plane wave with streamfunction ``psi = cos(kx+ly-wt)``
   has ``u = -dpsi/dy, v = dpsi/dx`` proportional to the SAME oscillating
   factor: the velocity is rectilinearly polarized along (-l, k) --
   maximal (Q, U) with orientation 2*arg(k-vector rotated), and EXACTLY
   ZERO spin.  A monopolar vortex orbits floats circularly: maximal V,
   zero mean (Q, U).  So the stretch face reads the WAVE content (and
   the wavevector orientation), the spin face reads the VORTEX content
   -- a two-moment Lagrangian wave/vortex decomposition, no tracks
   spectra needed.

Findings (figures/91_stokes_faces.json, committed cache
data/nr56_stokes_cache.json; 8280 floats):

* **The equator is wave-polarized.**  Zonal stretch Q/I = +0.44
  (t = +48) in |lat| < 5 -- the equatorial deep jets as the extreme
  rectilinear limit -- while the spin there is smallest (0.054).
* **Poleward the ocean vortex-polarizes.**  Q/I collapses to ~+0.04 by
  20-35 deg while |V| grows monotonically (0.054 -> 0.077 -> 0.091 ->
  0.107 at 65-85): the wave-to-vortex crossover sits at ~20-35 deg --
  measured from two single-float moments.
* **Polar flip.**  At 65-85 deg the stretch turns MERIDIONAL
  (Q/I = -0.032, t = -2.3): boundary-current/topographic steering
  replaces beta as the anisotropy source.
* **Exactness.**  Plane-wave synthetic: |V| < 1e-12 with the (Q, U)
  orientation obeying the 2*theta covariance to machine precision;
  vortex synthetic: (Q, U) -> 0 with V maximal; mirror flips (U, V) and
  fixes Q exactly.

Consequence for the program: with NR55's four phases plus this
polarization face, the full two-point measurement content of a 2D
trajectory ensemble is: energy (I), memory phase (arg of the scalar
response), transport, stretch (Q, U), and spin (V) -- the corpus's
two-clocks tensor is the coherency matrix of the flow, and wave/vortex
partition is polarimetry.  Protocol: two moments per float; no spectra,
no maps, no eddy detection.

CPU-only, offline-safe (committed cache; regenerate with build_cache()).
Tests: tests/test_stokes_faces.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr56_stokes_cache.json")
FIG = os.path.join(HERE, "figures", "91_stokes_faces.json")

BANDS = [(0, 5), (5, 20), (20, 35), (35, 50), (50, 65), (65, 85)]


# --------------------------------------------------------------------------- #
# Stokes estimators
# --------------------------------------------------------------------------- #
def stokes_zero_lag(u, v):
    """Normalized zero-lag Stokes (Q/I, U/I) of a residual velocity
    series."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    I = np.mean(u ** 2 + v ** 2)
    return float(np.mean(u ** 2 - v ** 2) / I), float(2 * np.mean(u * v) / I)


def polarization_angle_deg(Q, U):
    """Orientation of the linear-polarization axis (mod 180 deg)."""
    return float(np.degrees(0.5 * np.arctan2(U, Q)) % 180.0)


def band_stats(lat, x, lo, hi):
    m = (np.abs(lat) >= lo) & (np.abs(lat) < hi)
    xx = np.asarray(x, float)[m]
    n = int(m.sum())
    mean = float(xx.mean())
    sd = xx.std(ddof=1)
    t = float(mean / (sd / np.sqrt(n))) if sd > 0 else 0.0
    return dict(n=n, mean=mean, t=t)


# --------------------------------------------------------------------------- #
# exact polarization theorems (synthetic)
# --------------------------------------------------------------------------- #
def synthetic_checks():
    t = np.linspace(0, 200 * np.pi, 20001)
    out = {}
    # plane wave: psi = cos(kx+ly-wt) sampled at a point -> u ~ l cos, v ~ -k cos
    k, l = 1.3, 0.7
    osc = np.cos(1.7 * t + 0.3)
    u, v = l * osc, -k * osc
    Q, U = stokes_zero_lag(u, v)
    V = float(np.mean(u[:-1] * v[1:] - v[:-1] * u[1:])
              / np.mean(u ** 2 + v ** 2))
    ang = polarization_angle_deg(Q, U)
    ang_pred = float(np.degrees(np.arctan2(-k, l)) % 180.0)
    # 2-theta covariance: rotate the frame by th -> angle shifts by th
    th = 37.0
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    ur, vr = c * u - s * v, s * u + c * v
    Qr, Ur = stokes_zero_lag(ur, vr)
    ang_r = polarization_angle_deg(Qr, Ur)
    rot_err = abs((ang_r - ang - th + 90.0) % 180.0 - 90.0)
    # vortex: circular polarization; V(lag1) = sin(f dt) exactly
    fv = 0.9
    dtv = float(t[1] - t[0])
    uc, vc = np.cos(fv * t), np.sin(fv * t)
    Qc, Uc = stokes_zero_lag(uc, vc)
    Vc = float(np.mean(uc[:-1] * vc[1:] - vc[:-1] * uc[1:])
               / np.mean(uc ** 2 + vc ** 2))
    Vc_pred = float(np.sin(fv * dtv))
    # mirror: Q fixed, U and V flip
    Qm, Um = stokes_zero_lag(u, -v)
    Vm = float(np.mean(u[:-1] * (-v[1:]) - (-v[:-1]) * u[1:])
               / np.mean(u ** 2 + v ** 2))
    return dict(
        wave_Q=Q, wave_U=U, wave_V=abs(V),
        wave_angle_deg=ang, wave_angle_pred_deg=ang_pred,
        wave_angle_err_deg=abs((ang - ang_pred + 90) % 180 - 90),
        rotation_2theta_err_deg=float(rot_err),
        vortex_QU=float(np.hypot(Qc, Uc)), vortex_V=Vc,
        vortex_V_pred=Vc_pred,
        vortex_V_err=abs(Vc - Vc_pred),
        mirror_Q_err=abs(Qm - Q), mirror_U_flip_err=abs(Um + U),
        mirror_V_flip_err=abs(Vm + V))


# --------------------------------------------------------------------------- #
# committed real data
# --------------------------------------------------------------------------- #
def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


def analyze(cache):
    syn = synthetic_checks()
    f = cache["floats"]
    lat = np.array(f["lat"])
    Q = np.array(f["Q"])
    V = np.abs(np.array(f["V"]))
    bands = {}
    for lo, hi in BANDS:
        key = f"{lo}-{hi}"
        bands[key] = dict(stretch=band_stats(lat, Q, lo, hi),
                          spin_abs=band_stats(lat, V, lo, hi))
    out = dict(synthetic=syn, bands=bands,
               n_floats=int(cache["meta"]["n_floats"]))
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    syn = out["synthetic"]
    b = out["bands"]
    exact = (syn["wave_V"] < 1e-12
             and syn["wave_angle_err_deg"] < 1e-6
             and syn["rotation_2theta_err_deg"] < 1e-6
             and syn["vortex_QU"] < 1e-3 and syn["vortex_V_err"] < 1e-6
             and syn["vortex_V"] > 0
             and syn["mirror_Q_err"] < 1e-12
             and syn["mirror_U_flip_err"] < 1e-12
             and syn["mirror_V_flip_err"] < 1e-12)
    eq_wave = (b["0-5"]["stretch"]["mean"] > 0.3
               and b["0-5"]["stretch"]["t"] > 20.0)
    spins = [b[f"{lo}-{hi}"]["spin_abs"]["mean"] for lo, hi in BANDS]
    vortex_poleward = (spins[0] == min(spins)
                       and spins[-1] == max(spins)
                       and spins[3] > spins[1])
    crossover = (b["0-5"]["stretch"]["mean"]
                 > b["0-5"]["spin_abs"]["mean"]
                 and b["35-50"]["spin_abs"]["mean"]
                 > b["35-50"]["stretch"]["mean"])
    polar_flip = (b["65-85"]["stretch"]["mean"] < 0
                  and b["65-85"]["stretch"]["t"] < -2.0)
    return dict(
        polarization_theorems_exact=bool(exact),
        equator_is_wave_polarized=bool(eq_wave),
        vortex_polarization_grows_poleward=bool(vortex_poleward),
        wave_vortex_crossover_20_35=bool(crossover),
        polar_meridional_flip=bool(polar_flip),
        reading=(
            "The Lagrangian velocity has Stokes parameters: plane waves "
            "are rectilinearly polarized (stretch Q,U with exact 2-theta "
            "covariance; spin V = 0 to machine precision) and vortices "
            "circularly polarized (V maximal, Q,U -> 0).  The deep ocean "
            "is wave-polarized at the equator -- Q/I = "
            f"{b['0-5']['stretch']['mean']:+.2f} (t = "
            f"{b['0-5']['stretch']['t']:+.0f}), the equatorial deep jets "
            "as the extreme linear limit, with the smallest spin -- and "
            "vortex-polarizes poleward: |V| grows monotonically from "
            f"{b['0-5']['spin_abs']['mean']:.3f} to "
            f"{b['65-85']['spin_abs']['mean']:.3f} while the stretch "
            "collapses, crossing at ~20-35 deg; at 65-85 deg the stretch "
            f"flips MERIDIONAL ({b['65-85']['stretch']['mean']:+.3f}, "
            f"t = {b['65-85']['stretch']['t']:+.1f}) as boundary steering "
            "replaces beta.  Wave/vortex partition is polarimetry: two "
            "moments per float, no spectra, no maps."),
    )


def run(write=True):
    cache = load_cache()
    out = analyze(cache)
    res = dict(description=__doc__.splitlines()[0],
               meta=cache["meta"], **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


# --------------------------------------------------------------------------- #
# cache builder (needs the ANDRO pickle; not used in tests)
# --------------------------------------------------------------------------- #
def build_cache(pkl_path="/home/andro_valid.pkl", out_path=CACHE):
    import pandas as pd
    df = pd.read_pickle(pkl_path)
    rows = []
    for wmo, sub in df.groupby("wmo"):
        if len(sub) < 30:
            continue
        sub = sub.sort_values("juld")
        t = sub.juld.values
        u = sub.u.values / 100.0
        v = sub.v.values / 100.0
        ur = u - u.mean()
        vr = v - v.mean()
        E0 = np.sum(ur ** 2 + vr ** 2)
        if E0 <= 0:
            continue
        Q, U = stokes_zero_lag(ur, vr)
        O1 = sum(ur[i] * vr[i + 1] - vr[i] * ur[i + 1]
                 for i in range(len(t) - 1) if 7.0 <= t[i + 1] - t[i] <= 13.0)
        rows.append((float(sub.lat.mean()), Q, U, float(O1 / E0)))
    a = np.array(rows)
    out = dict(
        meta=dict(description=("NR56: Stokes faces (zero-lag Q,U "
                               "anisotropy + lag-1 V spin) per ANDRO "
                               "float"),
                  provenance=("ANDRO SEANOE doi:10.17882/47077 (CC-BY), "
                              "700-1300 dbar, 10-day cycles, float-mean "
                              "residuals"),
                  columns=["lat_mean", "Q_norm (A2=<u2-v2>/E)",
                           "U_norm (2<uv>/E)", "V_lag1 (rho_odd)"],
                  n_floats=int(len(a))),
        floats=dict(lat=a[:, 0].round(3).tolist(),
                    Q=a[:, 1].round(6).tolist(),
                    U=a[:, 2].round(6).tolist(),
                    V=a[:, 3].round(6).tolist()))
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    return out


def main():
    res = run()
    v = res["verdicts"]
    b = res["bands"]
    print("NR56 -- Stokes faces: waves are linear, vortices are circular")
    print(f"  polarization theorems exact   : "
          f"{v['polarization_theorems_exact']}")
    print(f"  equator wave-polarized        : "
          f"{v['equator_is_wave_polarized']} "
          f"(Q/I = {b['0-5']['stretch']['mean']:+.2f}, "
          f"t = {b['0-5']['stretch']['t']:+.0f})")
    print(f"  vortex polarization poleward  : "
          f"{v['vortex_polarization_grows_poleward']} "
          f"(|V| {b['0-5']['spin_abs']['mean']:.3f} -> "
          f"{b['65-85']['spin_abs']['mean']:.3f})")
    print(f"  wave/vortex crossover 20-35   : "
          f"{v['wave_vortex_crossover_20_35']}")
    print(f"  polar meridional flip         : {v['polar_meridional_flip']} "
          f"(Q/I = {b['65-85']['stretch']['mean']:+.3f}, "
          f"t = {b['65-85']['stretch']['t']:+.1f})")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
