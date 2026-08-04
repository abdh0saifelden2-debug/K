r"""NR62 (theory + real data) -- the reversible reference phase is set by
time-parity: 0 deg for equal-parity channels, +-90 deg for opposite
parity.  The entropy production is the DEPARTURE from it.  This corrects
NR61 and resolves the Sun's I-V phase into a reversible conservative
oscillation plus an irreversible non-adiabatic departure.

NR60/NR61 read irreversibility off the imaginary part of a cross-spectrum
(the quad-spectrum).  That is only correct when the two channels have the
SAME sign under time reversal.  Real observables carry a definite
time-parity eps_i = +-1 (positions, intensities, temperatures: even;
velocities, currents, fluxes: odd), and the correct time-reversal of a
Gaussian process maps its spectral matrix S(w) -> E S(w)^T E with
E = diag(eps) (Maes; Risken; Weiss 1975 with signature).  So

    reversible  <=>  S_ij(w) = eps_i eps_j S_ij(w)*   for all i,j,w,

and for two channels the irreversibility carrier flips with the parity
product:

    eps1 eps2 = +1  (equal parity):  carrier = Im S12   (NR60/61)
    eps1 eps2 = -1  (opposite):      carrier = Re S12   (this NR)

    d sigma/dw = (1/pi) (carrier-coherency)^2 / (1 - |Coh|^2).

Consequences
============
1. **The equilibrium oscillator is reversible -- but only with parity.**
   A thermally-driven damped oscillator (x even, v odd) has a PURELY
   IMAGINARY cross-spectrum S_xv(w) = i w S_xx(w): the naive (equal-parity)
   reading calls the 90-deg x-v phase "maximally irreversible", while the
   parity-correct reading gives Re S_xv = 0 -> sigma = 0, exactly as
   detailed balance requires.  Quadrature between an even and an odd
   variable is the REVERSIBLE conservative oscillation, not dissipation.
2. **The reference phase is +-90 deg for opposite parity.**  For an even-
   odd pair the reversible relationship is quadrature (the derivative
   coupling); irreversibility is the departure from +-90 deg, i.e. the
   in-phase (cospectrum) component -- the mirror of the equal-parity case.
3. **Genuine irreversibility survives the parity correction.**  A two-
   temperature coupled system (two even positions, T1 != T2) has Im S12 !=
   0 and sigma > 0 = (heat current)^2-like; a non-reciprocally coupled
   oscillator has Re S_xv != 0 and sigma > 0.  The correction removes the
   reversible reference, never a real current.

Real solar (committed NR46 cache: 26 y SOHO GOLF x VIRGO-green, I-V,
395 p-mode bins + background + 13 epochs; BiSON x GOLF V-V control)
==================================================================
The intensity I is even, the Doppler velocity V is odd -> the I-V pair is
OPPOSITE parity, reversible reference = the adiabatic -90 deg.

* **The p-modes are nearly reversible; the irreversibility is the non-
  adiabatic departure.**  Mode-comb median phase -104 deg, i.e. only
  15.8 deg from the adiabatic -90-deg reference: the conservative
  oscillation sits at the reversible reference.  The parity-correct EPR
  density (median 0.19) is the small non-adiabatic (radiative/convective)
  departure; the NAIVE equal-parity reading (median 1.62) OVER-counts the
  solar irreversibility 8.5x by mistaking the conservative -90-deg
  oscillation for dissipation.
* **The non-adiabatic irreversibility is stable over two solar cycles.**
  Across 13 annual epochs the mode phase holds to std 2.2 deg (NR46) and
  the irreversibility proxy cos^2(phase) to 0.09 +- 0.022 -- the Sun's
  non-adiabaticity is a fixed material property, not activity-driven.
* **The convective background is the broadband entropy source.**  The
  background cross-phase sits far from +-90 deg (111, 139, ...), i.e.
  large in-phase (irreversible) component, exactly where NR46 found the
  channel-local convective slow clock -- convection, not the modes, is
  where the Sun pays.
* **Parity control (BiSON x GOLF, V-V, EQUAL parity).**  Here the
  reference is 0 deg and the irreversibility carrier is the QUADRATURE:
  the top-quintile pair has high coherence (0.91) with a small -15.7-deg
  quadrature = NR48's -10.3 s inter-instrument all-pass delay -- an
  instrumental, not solar, irreversibility, correctly flagged only by the
  equal-parity criterion.

So NR46's -118.8-deg I-V phase, NR48's -10.3 s V-V delay, and NR60/61's
quad-spectrum EPR are one picture once parity is included: the reversible
reference is a parity choice, and the two solar pairs measure two
different irreversibilities (solar non-adiabaticity vs instrument delay)
because they have two different parity products.

Mainstream anchors: Maes / Seifert (even-odd variables in stochastic
thermodynamics); Risken (Fokker-Planck detailed balance with parity);
Weiss 1975; Jimenez et al. 1999 / NR46 (solar I-V phase).  CPU-only,
offline-safe (committed NR46 cache).  Figures: figures/97_parity_epr.json.
Tests: tests/test_parity_epr.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLAR_CACHE = os.path.join(HERE, "data", "nr46_solar_cache.json")
FIG = os.path.join(HERE, "figures", "97_parity_epr.json")


# --------------------------------------------------------------------------- #
# parity-aware entropy-production integrand
# --------------------------------------------------------------------------- #
def parity_reversal(S, eps):
    """Time reversal of a Gaussian spectral matrix with parities eps."""
    E = np.diag(eps).astype(float)
    return E @ S.T @ E


def epr_integrand_parity(S, eps):
    """tr[(E S^T E)^{-1} S - I], the KL-rate integrand vs parity-reversal."""
    Sr = parity_reversal(S, eps)
    return float(np.real(np.trace(np.linalg.solve(Sr, S))) - S.shape[0])


def epr_integrand_2x2_parity(S, eps):
    """Closed form: 4 (carrier)^2/det S, carrier = Im S12 if eps1 eps2=+1
    else Re S12."""
    det = float(np.real(np.linalg.det(S)))
    if eps[0] * eps[1] > 0:
        carrier = float(np.imag(S[0, 1]))
    else:
        carrier = float(np.real(S[0, 1]))
    return 4.0 * carrier ** 2 / det


def coherency(S):
    denom = np.sqrt(np.real(S[0, 0]) * np.real(S[1, 1]))
    c = S[0, 1] / denom
    return float(np.real(c)), float(np.imag(c)), float(abs(c))


def density_from_coh(re, im, cabs, parity_product):
    carrier = im if parity_product > 0 else re
    return carrier ** 2 / max(1.0 - cabs ** 2, 1e-12)


# --------------------------------------------------------------------------- #
# linear-SDE spectra:  dX = -A X dt + F dW,  S(w)=(A+iwI)^{-1} 2D (A^T-iwI)^{-1}
# --------------------------------------------------------------------------- #
def sde_psd(A, D, w):
    d = A.shape[0]
    M = A + 1j * w * np.eye(d)
    return np.linalg.solve(M, 2.0 * D) @ np.linalg.inv(A.T - 1j * w * np.eye(d))


def sde_epr(A, D, eps, n=30001):
    w0 = 4.0 * max(1.0, float(np.max(np.abs(np.linalg.eigvals(A)))))
    us = np.linspace(-np.pi / 2 + 1e-7, np.pi / 2 - 1e-7, n)
    ws = w0 * np.tan(us)
    jac = w0 / np.cos(us) ** 2
    vals = np.array([epr_integrand_parity(sde_psd(A, D, w), eps)
                     for w in ws]) * jac
    return float(np.trapezoid(vals, us) / (4.0 * np.pi))


def damped_oscillator_xspectrum(n=4000, seed=0):
    """Exact even-odd quadrature: v = dx/dt => S_xv(w) = i w S_xx(w),
    purely imaginary, for ANY real x-spectrum.  No matrix inverse needed.
    Returns (max |Re coherency|, min |Im coherency| over the band)."""
    rng = np.random.default_rng(seed)
    # arbitrary real, even auto-spectrum (AR(2)-like), sampled on a grid
    ws = np.linspace(0.05, 6.0, n)
    Sxx = 1.0 / ((ws ** 2 - 1.3 ** 2) ** 2 + (0.5 * ws) ** 2) + \
        0.2 / (1.0 + ws ** 2)
    re_max = 0.0
    im_min = np.inf
    for w, sx in zip(ws, Sxx):
        sv = w ** 2 * sx                       # S_vv = w^2 S_xx (v=dx/dt)
        S = np.array([[sx, 1j * w * sx], [-1j * w * sx, sv]])
        # regularize the exact rank-1 degeneracy with independent sensor
        # noise on each channel (does not change the phase of S12)
        S = S + np.diag([0.02 * sx, 0.02 * sv])
        re, im, _ = coherency(S)
        re_max = max(re_max, abs(re))
        im_min = min(im_min, abs(im))
    return re_max, im_min


def two_temperature(kc=1.0, g=1.0, T1=1.0, T2=3.0):
    """Two positions (both even) coupled, at different temperatures:
    genuinely irreversible (heat current) -- equal-parity carrier=Im."""
    A = np.array([[g + kc, -kc], [-kc, g + kc]])
    D = np.array([[g * T1, 0.0], [0.0, g * T2]])
    return A, D, np.array([+1, +1])


def nonreciprocal_osc(k=2.0, gamma=0.5, a=0.9, T=1.0, q=0.3):
    """Oscillator with a non-reciprocal extra coupling and FULL-RANK noise:
    genuinely irreversible with opposite parity (Re S_xv != 0)."""
    A = np.array([[0.0, -1.0 + a], [k, gamma]])     # a breaks the -1 symmetry
    D = np.array([[q, 0.0], [0.0, gamma * T]])
    return A, D, np.array([+1, -1])


# --------------------------------------------------------------------------- #
# real solar
# --------------------------------------------------------------------------- #
def load_solar(path=SOLAR_CACHE):
    with open(path) as fh:
        return json.load(fh)


def solar_analysis(cache):
    g = cache["golf_x_green"]
    rows = g["mode_rows"]
    g2 = np.array([r["g2"] for r in rows], float)
    ph = np.deg2rad(np.array([r["ph"] for r in rows], float))
    # I even, V odd => opposite parity => reversible ref = adiabatic -90 deg;
    # carrier = in-phase (cos) component
    epr_opp = g2 * np.cos(ph) ** 2 / np.maximum(1 - g2, 1e-6)
    epr_naive = g2 * np.sin(ph) ** 2 / np.maximum(1 - g2, 1e-6)
    dep = np.abs(np.rad2deg(ph) + 90.0)             # departure from -90
    ep = g["epochs"]
    eph = np.array([e["ph"] for e in ep], float)
    ecos2 = np.cos(np.deg2rad(eph)) ** 2
    bc = g["back_curve"]
    back_dep = np.abs((np.array([b["ph"] for b in bc]) + 90.0 + 180) %
                      360 - 180)
    bg = cache["bison_x_golf"]
    return dict(
        n_mode_bins=len(rows),
        mode_phase_med_deg=float(np.rad2deg(np.median(ph))),
        mode_g2_med=float(np.median(g2)),
        nonadiab_departure_med_deg=float(np.median(dep)),
        epr_parity_med=float(np.median(epr_opp)),
        epr_naive_med=float(np.median(epr_naive)),
        naive_overcount=float(np.median(epr_naive) /
                              max(np.median(epr_opp), 1e-9)),
        n_epochs=len(ep),
        epoch_phase_std_deg=float(np.std(eph)),
        epoch_cos2_mean=float(np.mean(ecos2)),
        epoch_cos2_std=float(np.std(ecos2)),
        back_departure_from_ref_deg=[float(x) for x in back_dep],
        back_median_departure_deg=float(np.median(back_dep)),
        vv_top_phase_deg=float(bg["top_phase"]),
        vv_top_g2=float(bg["g2_top_quintile"]),
        vv_quad_carrier=float(np.sqrt(bg["g2_top_quintile"]) *
                              abs(np.sin(np.deg2rad(bg["top_phase"])))),
    )


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def synthetic_checks(seed=0):
    rng = np.random.default_rng(seed)
    # closed form vs general, both parities
    cf = 0.0
    for _ in range(300):
        X = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        S = X @ X.conj().T + 0.05 * np.eye(2)
        for eps in (np.array([1, 1]), np.array([1, -1])):
            cf = max(cf, abs(epr_integrand_parity(S, eps) -
                             epr_integrand_2x2_parity(S, eps)))
    # equal-parity vs opposite-parity carrier on the even-odd oscillator:
    # v=dx/dt => S_xv purely imaginary => opposite-parity (Re) carrier ~0,
    # naive equal-parity (Im) carrier is O(1): the naive reading wrongly
    # flags the reversible conservative oscillation as irreversible
    osc_re_max, osc_im_min = damped_oscillator_xspectrum()
    # two-temperature: irreversible (equal parity)
    At, Dt, epst = two_temperature()
    tt_epr = sde_epr(At, Dt, epst)
    tt_eq = two_temperature(T1=2.0, T2=2.0)
    tt_epr_eq = sde_epr(tt_eq[0], tt_eq[1], tt_eq[2])
    # non-reciprocal oscillator: irreversibility INCREASES with the
    # non-reciprocal coupling a (both full-rank -> both irreversible; the
    # clean reversible opposite-parity reference is the rank-1 oscillator
    # above, Re-carrier == 0)
    An, Dn, epsn = nonreciprocal_osc()
    nr_epr = sde_epr(An, Dn, epsn)
    nr_epr_a0 = sde_epr(
        *nonreciprocal_osc(a=0.0)[:2], np.array([+1, -1]))
    # positivity: parity EPR >= 0 on random systems
    pos_min = np.inf
    for _ in range(24):
        d = 2
        M = rng.standard_normal((d, d))
        A = M @ M.T + d * np.eye(d) + 0.7 * (rng.standard_normal((d, d)) -
                                             rng.standard_normal((d, d)).T)
        Dm = np.diag(rng.uniform(0.4, 2.0, d))
        eps = rng.choice([-1, 1], d)
        pos_min = min(pos_min, sde_epr(A, Dm, eps))
    return dict(
        closed_form_max_err=cf,
        osc_re_carrier_max=osc_re_max, osc_im_carrier_min=osc_im_min,
        two_temp_epr=tt_epr, two_temp_equal_epr=tt_epr_eq,
        nonrecip_epr=nr_epr, nonrecip_epr_a0=nr_epr_a0,
        positivity_min=float(pos_min),
    )


def analyze(write=True):
    syn = synthetic_checks()
    solar = solar_analysis(load_solar())
    out = dict(description=__doc__.split("\n")[0], synthetic=syn, solar=solar)
    out["verdicts"] = verdicts(out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    return out


def verdicts(out):
    s, so = out["synthetic"], out["solar"]
    return dict(
        closed_form_exact=bool(s["closed_form_max_err"] < 1e-10),
        equilibrium_oscillator_reversible_with_parity=bool(
            s["osc_re_carrier_max"] < 0.05 and s["osc_im_carrier_min"] > 0.3),
        two_temperature_irreversible=bool(
            s["two_temp_epr"] > 1e-3 and abs(s["two_temp_equal_epr"]) < 1e-6),
        nonreciprocal_increases_epr=bool(
            s["nonrecip_epr"] > s["nonrecip_epr_a0"] > 1e-3),
        parity_epr_nonnegative=bool(s["positivity_min"] > -1e-9),
        solar_pmodes_near_reference=bool(
            so["nonadiab_departure_med_deg"] < 30.0),
        naive_overcounts_solar=bool(so["naive_overcount"] > 3.0),
        solar_irreversibility_epoch_stable=bool(
            so["epoch_phase_std_deg"] < 5.0 and
            so["epoch_cos2_std"] < 0.05),
        solar_background_is_irreversible=bool(
            so["back_median_departure_deg"] > 40.0),
        vv_control_flags_instrument_delay=bool(
            so["vv_top_g2"] > 0.8 and 0.0 < so["vv_quad_carrier"] < 0.5),
    )


def main():
    out = analyze(write=True)
    s, so = out["synthetic"], out["solar"]
    print("closed-form err:", s["closed_form_max_err"])
    print("oscillator even-odd: Re-carrier max", s["osc_re_carrier_max"],
          " Im-carrier min", s["osc_im_carrier_min"])
    print("two-temp EPR", s["two_temp_epr"], " (equal T)",
          s["two_temp_equal_epr"])
    print("non-reciprocal EPR", s["nonrecip_epr"], " (a=0)",
          s["nonrecip_epr_a0"], " positivity min",
          s["positivity_min"])
    print("\nSOLAR:", json.dumps(so, indent=1))
    print("\nverdicts:", json.dumps(out["verdicts"], indent=1))


if __name__ == "__main__":
    main()
