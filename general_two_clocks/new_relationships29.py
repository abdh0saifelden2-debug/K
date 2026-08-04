r"""NR52 (theory) -- the odd completion of the two-clocks response: rotation
adds an antisymmetric (gyroscopic) part that scalar MSD/GSER cannot see,
and its correlation face -- the Lagrangian spin -- measures handedness.
Real ANDRO floats: deep-eddy chirality flips sign across the equator
(anticyclonic in BOTH hemispheres) with a clean equatorial null.

Follows NR47 (memory phase delta), NR48 (delay = all-pass), NR51 (Im B =
parity-odd shape).  NR52 completes the PHASE ANATOMY of a linear response:

    - memory phase  delta(omega)   : odd in frequency lag  (KK, NR47/48)
    - transport phase -omega tau_d : all-pass              (NR48/49)
    - handedness   theta           : odd under PARITY      (this NR)

The claim
=========
1. **Tensor completion.**  In 2D (a rotating layer), the response/
   correlation tensor of an isotropic-but-chiral system decomposes as

       ``C(tau) = C_s(tau) I + C_a(tau) eps``,   eps = [[0,1],[-1,0]],

   symmetric part ``C_s = (<u u'> + <v v'>)/2`` and ODD part
   ``C_a = (<u(0)v(tau)> - <v(0)u(tau)>)/2`` (the Lagrangian *spin*
   correlation).  For the damped rotator GLE ``du/dt = -u/tau_L +
   f eps u + noise`` the matrix exponential factorizes EXACTLY
   (I and eps commute):

       ``C(tau)/C(0) = e^{-tau/tau_L} (cos f.tau I + sin f.tau eps)``.

2. **Invisibility (the chirality-blind spot).**  Scalar MSD and every
   scalar contraction (trace) read ONLY ``C_s``: mirroring ``y -> -y``
   maps a realization with rotation ``+f`` to one with ``-f`` with the
   SAME scalar MSD sample-by-sample, while ``C_a`` flips sign exactly.
   Chirality is invisible to GSER -- the tensorial analog of NR39
   (elliptic mixing invisible to Im) and NR51 (parity hides from even
   faces).  ``C_a`` is doubly odd: under parity AND under lag reversal
   (``C_a(-tau) = -C_a(tau)``) -- it exists only with broken
   time-reversal, i.e. it is a detailed-balance meter, and rotation is
   what breaks the balance in a rotating fluid.

3. **The measurement.**  ``rho_odd(tau) = O(tau)/E(0)`` from float
   displacement series (10-day ANDRO cycles, float-mean residuals) reads
   the net eddy handedness per region -- a single-particle measurement of
   the sign of vorticity dominance, no fields needed.

Findings (figures/87_odd_spin.json, committed cache
data/nr52_odd_spin_cache.json; 7258 floats, 700-1300 dbar):

* **Chirality flips at the equator (10 sigma).**  Per-float spin
  ``rho_odd(10 d)``: NH (>5 deg) mean -0.023, t = -9.7 (clockwise); SH
  (<-5 deg) mean +0.024, t = +13.2 (counterclockwise); equatorial band
  (|lat|<5 deg) +0.003, t = +1.0 -- a null exactly where f -> 0.  Band
  structure is sign-antisymmetric: (-,-,0,+,+) from NH-mid to SH-mid.
* **Anticyclonic dominance in BOTH hemispheres.**  NH clockwise and SH
  counterclockwise are both ANTICYCLONIC: the deep-looper censuses'
  known preference, here recovered as a two-line statistic from raw
  displacements.  The odd memory decays on ~10-30 d, the same eddy
  memory as NR45's scalar VAC.
* **Exactness.**  expm vs the analytic factorization to machine
  precision; the mirrored-realization pairing (MSD identical, spin
  flipped) exact; ``C_a(-tau) = -C_a(tau)`` exact.

Consequence for the program: the two-clocks response is not two numbers
but a symmetric pair PLUS a handedness: (elliptic, parabolic; odd).  Any
scalar inversion (GSER, MSD microrheology, NR45) silently projects out
the odd part -- legitimate for the modulus, blind to rotation sense.  The
spin correlation is the missing face, and in the ocean it is decisively
nonzero, hemisphere-antisymmetric, and anticyclone-dominated at depth.

CPU-only, offline-safe (committed cache; regenerate with build_cache()
from the ANDRO pickle).  Tests: tests/test_odd_spin.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr52_odd_spin_cache.json")
FIG = os.path.join(HERE, "figures", "87_odd_spin.json")

DT_DAYS = 10.0
MAXLAG = 12


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def odd_even_accumulate(u, v, maxlag=MAXLAG):
    """Odd (spin) and even numerators per lag for one residual series."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    n = len(u)
    O = np.zeros(maxlag + 1)
    E = np.zeros(maxlag + 1)
    C = np.zeros(maxlag + 1)
    for m in range(0, min(maxlag, n - 1) + 1):
        a, b = u[:n - m], u[m:]
        p, q = v[:n - m], v[m:]
        O[m] += np.sum(a * q - p * b)
        E[m] += np.sum(a * b + p * q)
        C[m] += n - m
    return O, E, C


def spin_stats(rho1, sel):
    r = np.asarray(rho1, float)[sel]
    n = int(sel.sum())
    mean = float(r.mean())
    t = float(mean / (r.std(ddof=1) / np.sqrt(n)))
    return dict(n_floats=n, rho1_mean=mean, t_stat=t)


# --------------------------------------------------------------------------- #
# exact damped-rotator theory
# --------------------------------------------------------------------------- #
EPS = np.array([[0.0, -1.0], [1.0, 0.0]])   # z-cross: f>0 = counterclockwise


def rotator_correlation(tau, f, tau_L):
    """Analytic C(tau)/C(0), tau >= 0, for du/dt = -u/tau_L + f (z x u) +
    noise; f > 0 rotates counterclockwise, so rho_odd = +sin(f tau) e^{-tau/
    tau_L} (positive spin = CCW).  Lag reversal is the transpose:
    C(-tau) = C(tau)^T by stationarity, so the odd part is odd in tau."""
    return np.exp(-tau / tau_L) * (np.cos(f * tau) * np.eye(2)
                                   + np.sin(f * tau) * EPS)


def odd_part(Cmat):
    return float((Cmat[1, 0] - Cmat[0, 1]) / 2.0)


def synthetic_checks(seed=0):
    from scipy.linalg import expm
    f, tau_L = 0.31, 3.7
    M = -np.eye(2) / tau_L + f * EPS
    expm_err = max(float(np.max(np.abs(
        expm(M * t) - rotator_correlation(t, f, tau_L))))
        for t in (0.3, 1.0, 4.2))
    # sign anchor: deterministic CCW circular motion has POSITIVE spin
    tt = np.arange(400) * 0.05
    Od, Ed, _ = odd_even_accumulate(np.cos(f * tt), np.sin(f * tt),
                                    maxlag=4)
    sign_anchor = float(Od[1] / Ed[0])
    # simulate one realization; mirror y -> -y maps f -> -f exactly
    rng = np.random.default_rng(seed)
    dt, nst = 0.02, 150000
    A = expm(M * dt)
    u = np.zeros((nst, 2))
    x = np.zeros(2)
    noise = rng.standard_normal((nst, 2))
    for i in range(1, nst):
        x = A @ x + np.sqrt(dt) * noise[i]
        u[i] = x
    um = u * np.array([1.0, -1.0])           # the mirrored realization
    # scalar MSD of the velocity series is identical sample-by-sample
    msd_diff = float(np.max(np.abs(np.sum(u ** 2, 1) - np.sum(um ** 2, 1))))
    stride = int(round(1.0 / dt))
    us, vs = u[::stride, 0], u[::stride, 1]
    O, E, _ = odd_even_accumulate(us, vs, maxlag=8)
    Om, Em, _ = odd_even_accumulate(um[::stride, 0], um[::stride, 1],
                                    maxlag=8)
    spin_flip_err = float(np.max(np.abs(O + Om)))
    even_same_err = float(np.max(np.abs(E - Em)))
    # rho_odd tracks the analytic +sin(f tau) e^{-tau/tau_L}
    taus = np.arange(1, 9) * 1.0
    pred = np.sin(f * taus) * np.exp(-taus / tau_L)
    meas = (O / E[0])[1:9]
    rho_err = float(np.max(np.abs(meas - pred)))
    # doubly odd: lag reversal is the transpose (stationarity), and the
    # odd part of the transpose is exactly minus the odd part
    lag_odd_err = max(float(abs(
        odd_part(rotator_correlation(t, f, tau_L).T)
        + odd_part(rotator_correlation(t, f, tau_L))))
        for t in (0.5, 2.0))
    return dict(expm_factorization_err=expm_err,
                sign_anchor_ccw_positive=sign_anchor,
                mirrored_msd_diff=msd_diff,
                mirrored_spin_flip_err=spin_flip_err,
                mirrored_even_same_err=even_same_err,
                rho_odd_vs_analytic_err=rho_err,
                lag_reversal_err=lag_odd_err,
                f_true=f, tau_L_true=tau_L)


# --------------------------------------------------------------------------- #
# committed real data
# --------------------------------------------------------------------------- #
def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


def analyze(cache):
    syn = synthetic_checks()
    rho1 = np.array(cache["per_float"]["rho1"])
    lat = np.array(cache["per_float"]["lat"])
    stats = dict(
        nh_all=spin_stats(rho1, lat > 5),
        sh_all=spin_stats(rho1, lat < -5),
        eq=spin_stats(rho1, np.abs(lat) < 5),
        nh_mid=spin_stats(rho1, (lat > 20) & (lat < 60)),
        nh_low=spin_stats(rho1, (lat > 5) & (lat <= 20)),
        sh_low=spin_stats(rho1, (lat < -5) & (lat >= -20)),
        sh_mid=spin_stats(rho1, (lat < -20) & (lat > -60)))
    # ensemble odd-correlation decay (memory of the odd part)
    decay = {}
    for hemi in ("nh_all", "sh_all"):
        b = cache["bands"][hemi]
        o = np.array(b["O"])
        e = np.array(b["E"])
        rho = o / e[0]
        m = np.argmax(np.abs(rho[1:]) < 0.5 * abs(rho[1])) + 1
        decay[hemi] = dict(rho=rho.tolist(),
                           halflife_days=float(m * DT_DAYS))
    out = dict(synthetic=syn, stats=stats, decay=decay,
               n_floats=int(cache["meta"]["n_floats"]))
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    syn = out["synthetic"]
    st = out["stats"]
    exact = (syn["expm_factorization_err"] < 1e-12
             and syn["sign_anchor_ccw_positive"] > 0.0
             and syn["mirrored_msd_diff"] < 1e-12
             and syn["mirrored_spin_flip_err"] < 1e-12
             and syn["mirrored_even_same_err"] < 1e-12
             and syn["lag_reversal_err"] < 1e-12
             and syn["rho_odd_vs_analytic_err"] < 0.05)
    flips = (st["nh_all"]["t_stat"] < -4.0 and st["sh_all"]["t_stat"] > 4.0
             and abs(st["eq"]["t_stat"]) < 2.0
             and st["nh_all"]["rho1_mean"] * st["sh_all"]["rho1_mean"] < 0)
    anticyclonic = (st["nh_all"]["rho1_mean"] < 0    # NH clockwise
                    and st["sh_all"]["rho1_mean"] > 0  # SH counterclockwise
                    and st["nh_mid"]["rho1_mean"] < 0
                    and st["sh_mid"]["rho1_mean"] > 0)
    return dict(
        odd_completion_exact=bool(exact),
        msd_is_chirality_blind=bool(syn["mirrored_msd_diff"] < 1e-12
                                    and syn["mirrored_spin_flip_err"]
                                    < 1e-12),
        ocean_chirality_flips_at_equator=bool(flips),
        anticyclonic_dominance_both_hemispheres=bool(anticyclonic),
        reading=(
            "The two-clocks response has a third, parity-odd face: the "
            "antisymmetric (spin) correlation, exactly "
            "e^(-tau/tau_L) sin(f tau) for the damped rotator, invisible "
            "to every scalar contraction (mirroring a realization "
            "preserves the MSD sample-by-sample while flipping the spin "
            "exactly).  The deep ocean measures it decisively: "
            f"NH rho_odd(10 d) = {st['nh_all']['rho1_mean']:+.3f} "
            f"(t = {st['nh_all']['t_stat']:+.1f}, clockwise), SH "
            f"{st['sh_all']['rho1_mean']:+.3f} "
            f"(t = {st['sh_all']['t_stat']:+.1f}, counterclockwise), "
            f"equatorial null {st['eq']['rho1_mean']:+.3f} "
            f"(t = {st['eq']['t_stat']:+.1f}) -- chirality flips with the "
            "sign of f, and both hemispheres are ANTICYCLONE-dominated "
            "at depth, with the odd memory decaying on the same 10-30 d "
            "eddy timescale as NR45's scalar VAC."),
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
    per_rho, per_lat, bands_acc = [], [], {}
    all_O, all_E, all_C, lats = [], [], [], []
    for wmo, sub in df.groupby("wmo"):
        if len(sub) < 30:
            continue
        sub = sub.sort_values("juld")
        t = sub.juld.values
        u = sub.u.values / 100.0 - sub.u.values.mean() / 100.0
        v = sub.v.values / 100.0 - sub.v.values.mean() / 100.0
        segs, cur = [], [0]
        for i in range(1, len(t)):
            if 7.0 <= t[i] - t[i - 1] <= 13.0:
                cur.append(i)
            else:
                if len(cur) >= 12:
                    segs.append(np.array(cur))
                cur = [i]
        if len(cur) >= 12:
            segs.append(np.array(cur))
        if not segs:
            continue
        O = np.zeros(MAXLAG + 1)
        E = np.zeros(MAXLAG + 1)
        C = np.zeros(MAXLAG + 1)
        for s in segs:
            o, e, c = odd_even_accumulate(u[s], v[s])
            O += o
            E += e
            C += c
        if E[0] <= 0:
            continue
        all_O.append(O)
        all_E.append(E)
        all_C.append(C)
        lats.append(float(sub.lat.mean()))
    all_O = np.array(all_O)
    all_E = np.array(all_E)
    all_C = np.array(all_C)
    lats = np.array(lats)
    bands = {"nh_mid": (lats > 20) & (lats < 60),
             "nh_low": (lats > 5) & (lats <= 20),
             "eq": np.abs(lats) < 5,
             "sh_low": (lats < -5) & (lats >= -20),
             "sh_mid": (lats < -20) & (lats > -60),
             "nh_all": lats > 5, "sh_all": lats < -5}
    out = dict(
        meta=dict(description=("NR52: Lagrangian odd (spin) correlations "
                               "from ANDRO deep displacements"),
                  provenance=("ANDRO deep displacements, SEANOE "
                              "doi:10.17882/47077 (CC-BY); 700-1300 dbar, "
                              "10-day cycles, float-mean residuals, "
                              "segments >=12 cycles (gap 7-13 d)"),
                  dt_days=DT_DAYS, maxlag=MAXLAG, n_floats=int(len(lats)),
                  odd_def=("O(m)=sum u_i v_{i+m} - v_i u_{i+m}; "
                           "E(m)=sum u_i u_{i+m}+v_i v_{i+m}; "
                           "rho_odd=O/E[0]")),
        per_float=dict(rho1=(all_O[:, 1] / all_E[:, 0]).round(6).tolist(),
                       lat=lats.round(3).tolist()),
        bands={name: dict(n_floats=int(sel.sum()),
                          O=all_O[sel].sum(0).tolist(),
                          E=all_E[sel].sum(0).tolist(),
                          n_pairs=all_C[sel].sum(0).tolist())
               for name, sel in bands.items()})
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    return out


def main():
    res = run()
    v = res["verdicts"]
    st = res["stats"]
    print("NR52 -- the odd completion: Lagrangian spin, the parity-odd face")
    print(f"  odd completion exact (synth)  : {v['odd_completion_exact']}")
    print(f"  MSD is chirality-blind        : {v['msd_is_chirality_blind']}")
    print(f"  chirality flips at equator    : "
          f"{v['ocean_chirality_flips_at_equator']} "
          f"(NH t={st['nh_all']['t_stat']:+.1f}, SH "
          f"t={st['sh_all']['t_stat']:+.1f}, EQ "
          f"t={st['eq']['t_stat']:+.1f})")
    print(f"  anticyclonic both hemispheres : "
          f"{v['anticyclonic_dominance_both_hemispheres']} "
          f"(NH {st['nh_all']['rho1_mean']:+.3f} CW, SH "
          f"{st['sh_all']['rho1_mean']:+.3f} CCW)")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
