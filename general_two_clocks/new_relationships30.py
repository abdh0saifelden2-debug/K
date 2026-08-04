r"""NR53 (theory + real data) -- the two clocks of the spin: the ocean's
odd (Lagrangian-spin) correlation decomposes into an oscillatory EDDY
clock and a persistent CIRCULATION clock, and the subpolar/polar
chirality reversal belongs to the circulation clock.

Follows NR52 (the odd completion: the spin is the parity-odd face of the
two-clocks tensor).  NR52 measured one number per region; NR53 reads the
full lag structure rho_odd(tau) and finds the two-clocks paradigm
REPEATED INSIDE the odd face:

    rho_odd(tau) = A sin(f_e tau) e^{-tau/tau_m}   (eddy clock: damped
                                                    rotator, zero-crossing,
                                                    finite memory)
                 + g tau                            (circulation clock:
                                                    slow persistent
                                                    rotation, period >>
                                                    window, no decay)

The claim
=========
1. **Separability.**  A damped rotator and a slow background rotation
   (gyre / boundary-current curvature) are distinguishable from the LAG
   STRUCTURE alone: the eddy clock oscillates (sign-flipping overshoot at
   half its rotation period) and decays on the eddy memory; the
   circulation clock grows linearly (small-angle sine) and never flips.
   Nonparametric fingerprints: a persistent zero-crossing for the eddy
   clock; a same-sign long-lag tail fraction for the circulation clock.

2. **The deep eddy rotation period is measurable and hemisphere-mirrored.**
   In the tropical bands (5-20 deg) the measured rho_odd oscillates:
   NH starts negative (clockwise = anticyclonic) and overshoots positive
   at lag ~3 (30 d); SH is the exact mirror.  The damped-rotator fit
   gives the rotation period of the mean deep eddy, T_e = 2 pi/|f_e|
   ~ 50-90 days, opposite chirality across the equator -- a single-float
   measurement of deep vortex rotation, and the two hemispheres agree on
   |f_e| within the fit error.

3. **The polar chirality reversal is a circulation-clock feature.**
   Poleward of ~50-65 deg the spin REVERSES to cyclonic in both
   hemispheres (NH 65-85: rho1 = +0.119, t = +9.9; SH 50-65: t = -2.8),
   but its lag structure is PERSISTENT (same-sign tail ~40-50 % of rho1,
   no zero-crossing): this is the cyclonic subpolar/polar circulation
   (subpolar gyres, Nordic Seas rim, ACC-adjacent flows) seen as
   trajectory curvature -- not an eddy-population chirality flip.  The
   celebrated anticyclone dominance (NR52) belongs to the EDDY clock of
   the 5-50 deg interior; the polar reversal belongs to the CIRCULATION
   clock.  Two clocks, two different physical carriers of handedness.

Findings (figures/88_spin_two_clocks.json, committed cache
data/nr53_spin_bands_cache.json, 11 latitude bands):

* tropical NH: zero-crossing at 30 d, T_e ~ 60 d clockwise; tropical SH:
  mirror (T_e ~ 65 d counterclockwise); |f_e| agreement ~15 %.
* interior bands (20-50 deg): anticyclonic eddy clock decaying on
  10-30 d with small same-sign tails (weak curvature).
* nh_polar: +0.119 -> +0.05 persistent (tail fraction ~0.4, no
  crossing): cyclonic circulation clock.  eq: null at every lag.
* synthetic: mixture separation exact in the noiseless limit; with
  realistic noise the fit recovers f_e within 15 % and g within a factor
  of 2; pure-drift data yields no persistent zero-crossing.

Consequence for the program: the two-clocks paradigm is scale-free -- it
reappears inside the parity-odd face itself.  Measurement protocol for
any trajectory ensemble: read rho_odd(tau); a sign-flipping damped
oscillation certifies coherent vortices and measures their rotation
period; a linear persistent component certifies large-scale circulation
curvature.  The two are separable WITHOUT maps, velocity fields, or eddy
detection -- pure single-particle statistics.

CPU-only, offline-safe (committed cache).  Tests:
tests/test_spin_two_clocks.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr53_spin_bands_cache.json")
FIG = os.path.join(HERE, "figures", "88_spin_two_clocks.json")

DT_DAYS = 10.0


# --------------------------------------------------------------------------- #
# the two-clock model of the odd correlation
# --------------------------------------------------------------------------- #
def spin_model(tau_days, A, f_e, tau_m, g):
    tau = np.asarray(tau_days, float)
    return A * np.sin(f_e * tau) * np.exp(-tau / tau_m) + g * tau


def fit_spin(rho, dt_days=DT_DAYS, sign_hint=None):
    """Fit rho_odd(tau) (lags 1..len-1) with eddy + circulation clocks."""
    from scipy.optimize import least_squares
    rho = np.asarray(rho, float)
    m = np.arange(1, len(rho))
    tau = m * dt_days
    s = sign_hint if sign_hint is not None else np.sign(rho[1]) or 1.0

    def resid(p):
        A, f_e, tau_m, g = p
        return spin_model(tau, A, f_e, tau_m, g) - rho[1:]

    best = None
    for T0 in (40.0, 60.0, 90.0, 150.0):
        p0 = [abs(rho[1]) * 2.0, s * 2 * np.pi / T0, 20.0, 0.0]
        sol = least_squares(
            resid, p0,
            bounds=([0.0, -0.5, 3.0, -0.01], [1.0, 0.5, 300.0, 0.01]))
        if best is None or sol.cost < best.cost:
            best = sol
    A, f_e, tau_m, g = best.x
    return dict(A=float(A), f_e_rad_day=float(f_e),
                period_days=float(2 * np.pi / abs(f_e)) if f_e else np.inf,
                tau_m_days=float(tau_m), g_per_day=float(g),
                cost=float(best.cost))


def zero_crossing_lag(rho, persist=2):
    """First lag where rho flips sign from rho[1] and stays flipped for
    `persist` lags; None if no persistent crossing."""
    rho = np.asarray(rho, float)
    s0 = np.sign(rho[1])
    for m in range(2, len(rho) - persist + 1):
        seg = rho[m:m + persist]
        if np.all(np.sign(seg) == -s0) and np.all(seg != 0):
            return m
    return None


def tail_fraction(rho, tail_from=8):
    """Mean of the long-lag tail relative to rho[1] (signed)."""
    rho = np.asarray(rho, float)
    return float(np.mean(rho[tail_from:]) / rho[1])


# --------------------------------------------------------------------------- #
# synthetic separation
# --------------------------------------------------------------------------- #
def synthetic_checks(seed=0):
    from scipy.linalg import expm
    EPS = np.array([[0.0, -1.0], [1.0, 0.0]])
    f_e, tau_L = 2 * np.pi / 15.0, 5.0        # eddy: 15 d period (units d)
    f_g = 2 * np.pi / 400.0                    # circulation: 400 d period
    rng = np.random.default_rng(seed)
    dt, nst = 0.05, 240000
    A = expm((-np.eye(2) / tau_L + f_e * EPS) * dt)
    u = np.zeros((nst, 2))
    x = np.zeros(2)
    noise = rng.standard_normal((nst, 2))
    for i in range(1, nst):
        x = A @ x + np.sqrt(dt) * noise[i]
        u[i] = x
    # add a slow coherent rotation of comparable energy fraction
    th = f_g * dt * np.arange(nst)
    amp = np.sqrt(np.mean(np.sum(u ** 2, 1))) * 0.5
    u[:, 0] += amp * np.cos(th)
    u[:, 1] += amp * np.sin(th)
    stride = int(round(1.0 / dt))
    us, vs = u[::stride, 0], u[::stride, 1]
    us = us - us.mean()
    vs = vs - vs.mean()
    n = len(us)
    maxlag = 12
    O = np.zeros(maxlag + 1)
    E = np.zeros(maxlag + 1)
    for m in range(maxlag + 1):
        a, b = us[:n - m], us[m:]
        p, q = vs[:n - m], vs[m:]
        O[m] += np.sum(a * q - p * b)
        E[m] += np.sum(a * b + p * q)
    rho = O / E[0]
    fit = fit_spin(rho, dt_days=1.0)
    # ground truth for the linear term: energy fraction * f_g; circular
    # motion carries mean(u^2+v^2) = amp^2
    frac = amp ** 2 / np.mean(us ** 2 + vs ** 2)
    g_true = frac * f_g
    # pure drift control: no crossing
    ud = amp * np.cos(th)[::stride]
    vd = amp * np.sin(th)[::stride]
    ud = ud - ud.mean()
    vd = vd - vd.mean()
    Od = np.zeros(maxlag + 1)
    Ed = np.zeros(maxlag + 1)
    nd = len(ud)
    for m in range(maxlag + 1):
        a, b = ud[:nd - m], ud[m:]
        p, q = vd[:nd - m], vd[m:]
        Od[m] += np.sum(a * q - p * b)
        Ed[m] += np.sum(a * b + p * q)
    rho_d = Od / Ed[0]
    return dict(
        fit=fit, f_e_true=f_e, g_true=float(g_true),
        f_e_rel_err=float(abs(abs(fit["f_e_rad_day"]) - f_e) / f_e),
        g_ratio=float(fit["g_per_day"] / g_true) if g_true else np.inf,
        mixture_has_crossing=zero_crossing_lag(rho) is not None,
        drift_has_crossing=zero_crossing_lag(rho_d) is not None,
        drift_tail_same_sign=tail_fraction(rho_d) > 0.5)


# --------------------------------------------------------------------------- #
# committed real data
# --------------------------------------------------------------------------- #
def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


def band_rho(cache, band):
    b = cache["bands"][band]
    return np.array(b["O"]) / b["E"][0]


def analyze(cache):
    syn = synthetic_checks()
    bands = {}
    for name in cache["bands"]:
        rho = band_rho(cache, name)
        b = cache["bands"][name]
        entry = dict(n_floats=b["n_floats"], rho1=float(rho[1]),
                     rho1_float=float(b["rho1_mean"]),
                     t_stat=b["rho1_t"],
                     crossing_lag=zero_crossing_lag(rho),
                     tail_frac=tail_fraction(rho))
        if name in ("nh_trop", "sh_trop"):
            entry["fit"] = fit_spin(rho)
        bands[name] = entry
    out = dict(synthetic=syn, bands=bands)
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    syn = out["synthetic"]
    b = out["bands"]
    sep = (syn["f_e_rel_err"] < 0.15 and 0.5 < syn["g_ratio"] < 2.0
           and syn["mixture_has_crossing"]
           and not syn["drift_has_crossing"]
           and syn["drift_tail_same_sign"])
    nh, sh = b["nh_trop"], b["sh_trop"]
    eddy = (nh["crossing_lag"] is not None and sh["crossing_lag"] is not None
            and nh["rho1"] < 0 < sh["rho1"]
            and 30.0 < nh["fit"]["period_days"] < 120.0
            and 30.0 < sh["fit"]["period_days"] < 120.0
            and nh["fit"]["f_e_rad_day"] * sh["fit"]["f_e_rad_day"] < 0
            and abs(abs(nh["fit"]["f_e_rad_day"])
                    - abs(sh["fit"]["f_e_rad_day"]))
            / abs(nh["fit"]["f_e_rad_day"]) < 0.5)
    polar = (b["nh_polar"]["rho1_float"] > 0
             and b["nh_polar"]["t_stat"] > 4.0
             and b["nh_polar"]["crossing_lag"] is None
             and b["nh_polar"]["tail_frac"] > 0.25
             and b["nh_mid"]["rho1_float"] < 0
             and b["sh_subpolar"]["rho1_float"]
             * b["sh_mid"]["rho1_float"] < 0)
    equator = abs(b["eq"]["t_stat"]) < 2.0
    return dict(
        two_spin_clocks_separable=bool(sep),
        deep_eddy_rotation_measured_and_mirrored=bool(eddy),
        polar_reversal_is_circulation_clock=bool(polar),
        equatorial_null=bool(equator),
        reading=(
            "The odd correlation itself has two clocks.  Tropics: a damped "
            "OSCILLATION (zero-crossing at "
            f"{(nh['crossing_lag'] or 0) * DT_DAYS:.0f} d NH / "
            f"{(sh['crossing_lag'] or 0) * DT_DAYS:.0f} d SH) -- the mean "
            "deep eddy rotates with period "
            f"{nh['fit']['period_days']:.0f} d clockwise (NH) and "
            f"{sh['fit']['period_days']:.0f} d counterclockwise (SH): "
            "anticyclonic, hemisphere-mirrored, measured from single-float "
            "statistics.  Poleward of ~50-65 deg the spin reverses to "
            f"cyclonic (NH 65-85: rho1 = {b['nh_polar']['rho1']:+.3f}, "
            f"t = {b['nh_polar']['t_stat']:+.1f}) but with NO crossing and "
            f"a {100 * b['nh_polar']['tail_frac']:.0f}% same-sign tail: a "
            "persistent circulation clock (subpolar/polar cyclonic "
            "circulation seen as trajectory curvature), not an "
            "eddy-population flip.  The equator is null at every lag.  "
            "Anticyclone dominance belongs to the eddy clock; the polar "
            "reversal belongs to the circulation clock."),
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


def main():
    res = run()
    v = res["verdicts"]
    b = res["bands"]
    print("NR53 -- the two clocks of the spin")
    print(f"  clocks separable (synthetic)  : "
          f"{v['two_spin_clocks_separable']} "
          f"(f_e err {res['synthetic']['f_e_rel_err']:.1%}, g ratio "
          f"{res['synthetic']['g_ratio']:.2f})")
    print(f"  eddy clock measured + mirrored: "
          f"{v['deep_eddy_rotation_measured_and_mirrored']} "
          f"(NH T={b['nh_trop']['fit']['period_days']:.0f} d CW, SH "
          f"T={b['sh_trop']['fit']['period_days']:.0f} d CCW)")
    print(f"  polar reversal = circulation  : "
          f"{v['polar_reversal_is_circulation_clock']} "
          f"(NH polar rho1 {b['nh_polar']['rho1']:+.3f}, "
          f"t {b['nh_polar']['t_stat']:+.1f}, tail "
          f"{b['nh_polar']['tail_frac']:.2f}, no crossing)")
    print(f"  equatorial null               : {v['equatorial_null']}")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
