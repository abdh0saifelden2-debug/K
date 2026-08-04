r"""NR60 (theory + real data) -- Gonella's rotary coefficient is an
entropy-production spectral density: the frequency-resolved second law
of the two-clocks response.

NR57 gave the spin a thermodynamic meaning (total entropy production
from the short-lag odd slope); NR53 split the spin into two clocks
(eddy vs circulation).  NR60 supplies the object that unifies them: the
EXACT spectral decomposition of the Gaussian entropy production rate,
and it turns out oceanography has been plotting it for 50 years without
knowing it.

The identity
============
For a stationary d-dim Gaussian process the time-reversed process has
spectral density S(w)^T, and the KL rate between the process and its
reversal (= the entropy production rate, Lebowitz-Spohn) is

    sigma = (1/4pi) INT tr[ S(w)^{-T} S(w) - I ] dw          (two-sided)

(the log-det term of the Gaussian KL rate vanishes identically because
det S = det S^T).  For d=2 -- one Lagrangian velocity record -- the
integrand CLOSES:

    tr[S^{-T}S - I] = 4 (Im S_uv)^2 / det S                  (exact)

so with the spectral Stokes parameters (NR56: I=Suu+Svv, Q=Suu-Svv,
U=2Re Suv, V=-2Im Suv, det S=(I^2-Q^2-U^2-V^2)/4):

    d sigma / dw = (1/pi) V^2 / (I^2 - Q^2 - U^2 - V^2)

and in the rotationally-symmetric sector (Q=U=0), writing S+/- for the
rotary spectra and C_R = (S+ - S-)/(S+ + S-) for **Gonella's (1972)
rotary coefficient**, the standard tool of ocean current analysis:

    d sigma / dw = (1/4pi) (S+ - S-)^2/(S+ S-)
                 = (1/pi)  C_R^2 / (1 - C_R^2).

Three theorems fall out of the closed form:

1. **Rotary asymmetry IS irreversibility.**  A frequency band is
   time-reversible (at second order) iff its rotary spectra balance,
   S+ = S-.  Fifty years of rotary-coefficient plots are entropy-
   production spectral densities up to the monotone map C^2/(1-C^2).
2. **Linear polarization is reversible; circular is dissipative.**
   V=0 kills the density REGARDLESS of Q,U: rectilinear (wave) motion
   of any anisotropy is second-order reversible; only the circular
   (vortex/inertial) face produces entropy.  NR56's wave/vortex
   polarimetry is a reversible/irreversible decomposition, frequency
   by frequency.  (The ubiquitous purely-circular inertial peak of
   surface drifters -- Elipot & Lumpkin 2008 -- is in this reading the
   most irreversible band of the surface ocean.)
3. **The OU rotator closes on NR57 exactly.**  For du = -nu u + f eps u
   + sqrt(2) dW the rotary spectra are Lorentzians at +/-f and the
   integral evaluates (residues) to sigma = 2 f^2/nu -- EXACTLY the
   phase-space-current entropy production tr(Omega C Omega^T D^{-1}) of
   NR57.  The identity holds for EVERY OU system (any d, any A, D):
   spectral KL = Lyapunov/current EPR, verified to quadrature precision.

Guards (verified):
* sampling is a deterministic map => the sampled-record EPR is a LOWER
  bound on the continuous one, monotone in dt (data processing);
* the Gaussian-spectral EPR lower-bounds the true EPR of non-Gaussian
  processes (biased 3-state ring: sigma_Gauss < sigma_true = J ln(p/q),
  both zero for the unbiased ring) -- what the second-order record
  certifies, not the whole budget;
* mirror image and time reversal leave the density invariant (EPR is
  parity-EVEN although V is parity-odd: it is V^2 that enters);
* the estimate is invariant to normalization and frame rotation.

Findings on the real deep ocean (committed NR53 band cache, ANDRO
700-1300 dbar, 10-day cycles, 11 latitude bands)
================================================
* **The deep ocean's irreversibility is frequency-localized at the eddy
  clock.**  Tropical bands: the EPR density peaks at periods ~40-70 d --
  the NR53 eddy-rotation clock (T_e = 49-61 d), not at the lowest or
  highest resolved frequency.  The spin's zero-crossing (NR53) and the
  EPR peak are the same physics: one rotation sense sustained over an
  eddy turnover.
* **The two polar oceans split by clock.**  Where NR53 found the
  persistent (no-flip) circulation clock (NH 65-85N), the EPR density is
  carried on the DC side (peak period ~200 d, ~half the entropy below
  the 120-d window edge) -- irreversibility of a steady cyclonic
  circulation.  The Antarctic polar band instead pays at the EDDY clock
  (~30 d peak): the ACC/Weddell eddy field.  The Arctic is irreversible
  the way a gyre is; the Antarctic the way an eddy field is.
* **The equator is the reversible line twice over.**  f->0 kills V
  (NR52) AND the equatorial band is the wave-polarized corner (NR56,
  Q/I=+0.44): both theorems predict zero density; the equatorial EPR is
  the smallest of all bands and sits inside its reversible surrogate
  null.
* **Hemispheric mirror.**  S_a flips sign across the equator (NR52) but
  the EPR density is sign-blind: NH and SH band profiles agree in shape
  and magnitude -- the two hemispheres are equally far from equilibrium,
  with opposite handedness.

CPU-only, offline-safe (committed caches).  Figures:
figures/95_spectral_epr.json.  Tests: tests/test_spectral_epr.py.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

HERE = os.path.dirname(os.path.abspath(__file__))
BAND_CACHE = os.path.join(HERE, "data", "nr53_spin_bands_cache.json")
FIG = os.path.join(HERE, "figures", "95_spectral_epr.json")

EPS = np.array([[0.0, -1.0], [1.0, 0.0]])
DT_DAYS = 10.0

BANDS = ["nh_polar", "nh_subpolar", "nh_mid", "nh_sub", "nh_trop",
         "eq", "sh_trop", "sh_sub", "sh_mid", "sh_subpolar", "sh_polar"]


# --------------------------------------------------------------------------- #
# the closed-form identities
# --------------------------------------------------------------------------- #
def epr_integrand_general(S):
    """tr[S^{-T} S - I] for a Hermitian PD spectral matrix S (any d)."""
    return float(np.real(np.trace(np.linalg.solve(S.T, S))) - S.shape[0])


def epr_integrand_2x2(S):
    """The closed 2x2 form: 4 (Im S_uv)^2 / det S."""
    d = np.real(np.linalg.det(S))
    return 4.0 * float(np.imag(S[0, 1])) ** 2 / d


def stokes_of_S(S):
    """Spectral Stokes parameters (NR56 conventions)."""
    I = float(np.real(S[0, 0] + S[1, 1]))
    Q = float(np.real(S[0, 0] - S[1, 1]))
    U = float(2.0 * np.real(S[0, 1]))
    V = float(-2.0 * np.imag(S[0, 1]))
    return I, Q, U, V


def epr_integrand_stokes(S):
    I, Q, U, V = stokes_of_S(S)
    return 4.0 * V ** 2 / (I ** 2 - Q ** 2 - U ** 2 - V ** 2)


def epr_integrand_rotary(Sp, Sm):
    """(S+ - S-)^2/(S+ S-) = 4 C_R^2/(1 - C_R^2), isotropic sector."""
    return (Sp - Sm) ** 2 / (Sp * Sm)


# --------------------------------------------------------------------------- #
# OU processes: spectral EPR vs the NR57 phase-space-current EPR
# --------------------------------------------------------------------------- #
def ou_spectral_matrix(A, D, w):
    d = A.shape[0]
    M = A + 1j * w * np.eye(d)
    return 2.0 * np.linalg.solve(M, D @ np.linalg.inv(A.T - 1j * w * np.eye(d)))


def ou_epr_spectral(A, D, n=40001):
    """(1/4pi) INT tr[S^{-T}S - I] dw, tan-substitution (exact tails)."""
    w0 = 3.0 * max(1.0, float(np.max(np.abs(np.linalg.eigvals(A)))))
    us = np.linspace(-np.pi / 2 + 1e-7, np.pi / 2 - 1e-7, n)
    ws = w0 * np.tan(us)
    jac = w0 / np.cos(us) ** 2
    vals = np.array([epr_integrand_general(ou_spectral_matrix(A, D, w))
                     for w in ws]) * jac
    return float(np.trapezoid(vals, us) / (4.0 * np.pi))


def ou_epr_current(A, D):
    """NR57: sigma = tr(Omega C Omega^T D^{-1}), Omega = A - D C^{-1}."""
    C = solve_continuous_lyapunov(A, 2.0 * D)
    Om = A - D @ np.linalg.inv(C)
    return float(np.trace(Om @ C @ Om.T @ np.linalg.inv(D)))


def rotator_epr_exact(nu, f):
    """Residue result for A = nu I + f EPS, D = I: sigma = 2 f^2 / nu."""
    return 2.0 * f ** 2 / nu


# --------------------------------------------------------------------------- #
# discrete time (sampled records)
# --------------------------------------------------------------------------- #
def sampled_ou_spectral_density(A, D, dt, thetas, mmax=400):
    """S_d(theta) = sum_m C(m) e^{-i m theta} for the dt-sampled OU."""
    d = A.shape[0]
    C0 = solve_continuous_lyapunov(A, 2.0 * D)
    Sd = np.zeros((len(thetas), d, d), complex)
    for m in range(-mmax, mmax + 1):
        Cm = expm(-A * abs(m) * dt) @ C0 if m >= 0 else (
            expm(-A * abs(m) * dt) @ C0).T
        Sd += Cm[None] * np.exp(-1j * np.outer(thetas, [m]))[:, :, None]
    return Sd


def discrete_epr_per_step(Sd, thetas):
    vals = np.array([epr_integrand_general(S) for S in Sd])
    return float(np.trapezoid(vals, thetas) / (4.0 * np.pi))


# --------------------------------------------------------------------------- #
# non-Gaussian control: biased ring, true vs Gaussian-spectral EPR
# --------------------------------------------------------------------------- #
def ring_true_epr(p, q):
    """3-state ring, uniform stationary law: sigma = (p-q) ln(p/q)/step."""
    return (p - q) * np.log(p / q)


def ring_gaussian_epr(p, q, mmax=300, ngrid=4001):
    """Gaussian-spectral EPR of the ring's 2-D position record."""
    P = np.array([[1 - p - q, p, q],
                  [q, 1 - p - q, p],
                  [p, q, 1 - p - q]])
    ang = 2.0 * np.pi * np.arange(3) / 3.0
    R = np.stack([np.cos(ang), np.sin(ang)], axis=1)      # (3,2) vertices
    pi0 = np.ones(3) / 3.0
    Rc = R - pi0 @ R
    thetas = np.linspace(-np.pi, np.pi, ngrid)
    Sd = np.zeros((ngrid, 2, 2), complex)
    Pm = np.eye(3)
    for m in range(0, mmax + 1):
        Cm = Rc.T @ (pi0[:, None] * Pm) @ Rc              # E[x_0 x_m^T]
        ph = np.exp(-1j * m * thetas)
        if m == 0:
            Sd += Cm[None] * ph[:, None, None]
        else:
            Sd += Cm[None] * ph[:, None, None]
            Sd += Cm.T[None] * np.conj(ph)[:, None, None]
        Pm = Pm @ P
    vals = np.array([epr_integrand_2x2(S) for S in Sd])
    return float(np.trapezoid(vals, thetas) / (4.0 * np.pi))


# --------------------------------------------------------------------------- #
# rectilinear waves vs circular vortices (theorem 2)
# --------------------------------------------------------------------------- #
def wave_field_spectral_epr(n_modes=7, seed=0, ngrid=2001):
    """Superposition of RECTILINEAR oscillators: V(w)=0 => zero density."""
    rng = np.random.default_rng(seed)
    thetas = np.linspace(-np.pi, np.pi, ngrid)
    S = np.zeros((ngrid, 2, 2), complex)
    for _ in range(n_modes):
        phi = rng.uniform(0, np.pi)
        e = np.array([np.cos(phi), np.sin(phi)])
        w0 = rng.uniform(0.3, 2.5)
        gam = rng.uniform(0.05, 0.3)
        amp = rng.uniform(0.5, 2.0)
        line = gam / ((thetas - w0) ** 2 + gam ** 2) + \
            gam / ((thetas + w0) ** 2 + gam ** 2)
        S += amp * line[:, None, None] * np.outer(e, e)[None]
    vmax = max(abs(np.imag(s[0, 1])) for s in S)
    dens = np.array([epr_integrand_2x2(s + 1e-9 * np.eye(2)) for s in S])
    return float(vmax), float(np.max(dens))


# --------------------------------------------------------------------------- #
# real data: banded EPR spectra from the committed NR53 cache
# --------------------------------------------------------------------------- #
def load_bands(path=BAND_CACHE):
    with open(path) as fh:
        return json.load(fh)


def band_correlations(band):
    """Per-pair mean correlations: C_s(m), C_a(m) (normalized, C_s(0)=1)."""
    O = np.array(band["O"], float)
    E = np.array(band["E"], float)
    npr = np.array(band["n_pairs"], float)
    Ce = E / npr
    Co = O / npr
    return Ce / Ce[0], Co / Ce[0]          # C_s, C_a in units of C_s(0)


def band_epr_spectrum(Cs, Ca, ngrid=721, taper="bartlett"):
    """Tapered isotropic-sector EPR density on theta in (0, pi]."""
    M = len(Cs) - 1
    m = np.arange(1, M + 1)
    if taper == "bartlett":
        w = 1.0 - m / (M + 1.0)
    elif taper == "parzen":
        x = m / (M + 1.0)
        w = np.where(x <= 0.5, 1 - 6 * x ** 2 + 6 * x ** 3,
                     2 * (1 - x) ** 3)
    elif taper == "tukey":
        w = 0.5 * (1 + np.cos(np.pi * m / (M + 1.0)))
    else:
        w = np.ones(M)
    thetas = np.linspace(0.0, np.pi, ngrid)[1:]
    cos = np.cos(np.outer(thetas, m))
    sin = np.sin(np.outer(thetas, m))
    Ss = Cs[0] + 2.0 * cos @ (w * Cs[1:])
    Sa = 2.0 * sin @ (w * Ca[1:])
    dens = np.where(Ss ** 2 > Sa ** 2,
                    4.0 * Sa ** 2 / np.maximum(Ss ** 2 - Sa ** 2, 1e-12),
                    np.nan)
    sigma = float(2.0 * np.trapezoid(dens, thetas) / (4.0 * np.pi))
    return thetas, Ss, Sa, dens, sigma


def band_analysis(cache):
    out = {}
    for name in BANDS:
        b = cache["bands"][name]
        Cs, Ca = band_correlations(b)
        thetas, Ss, Sa, dens, sigma = band_epr_spectrum(Cs, Ca)
        ipk = int(np.nanargmax(dens))
        cum = np.cumsum(np.nan_to_num(dens))
        cum = cum / cum[-1]
        out[name] = dict(
            n_floats=b["n_floats"], lat_mean=b["lat_mean"],
            sigma_nats_per_step=sigma,
            peak_theta=float(thetas[ipk]),
            peak_period_days=float(2.0 * np.pi * DT_DAYS / thetas[ipk]),
            frac_below_120d=float(
                np.interp(2.0 * np.pi * DT_DAYS / 120.0, thetas, cum)),
            peak_periods_tapers={
                t: float(2.0 * np.pi * DT_DAYS /
                         band_epr_spectrum(Cs, Ca, taper=t)[0][
                             int(np.nanargmax(
                                 band_epr_spectrum(Cs, Ca, taper=t)[3]))])
                for t in ("bartlett", "parzen", "tukey")},
            min_det_margin=float(np.nanmin(Ss ** 2 - Sa ** 2)),
            max_abs_CR=float(np.nanmax(np.abs(Sa / Ss))),
            dens_lowbin=float(dens[0]), dens_peak=float(dens[ipk]),
        )
    return out


def surrogate_null(cache, name, n_rep=200, seed=1):
    """Reversible Gaussian surrogate matched to the band's even spectrum
    and pair count: null distribution of sigma-hat."""
    rng = np.random.default_rng(seed)
    b = cache["bands"][name]
    Cs, _ = band_correlations(b)
    n_floats = int(b["n_floats"])
    L = max(int(round(b["n_pairs"][0] / n_floats)), 13) + 1
    M = len(Cs) - 1
    # AR-free surrogate: draw Gaussian records with autocovariance Cs via
    # circulant embedding on a long grid
    Nbig = 512
    cs_full = np.zeros(Nbig)
    cs_full[:M + 1] = Cs
    cs_full[-M:] = Cs[1:][::-1]
    lam = np.fft.rfft(cs_full).real
    lam = np.maximum(lam, 0.0)
    sig = []
    for _ in range(n_rep):
        # u,v independent identical spectra -> reversible, isotropic
        z = []
        for _c in range(2):
            xi = (rng.standard_normal((n_floats, Nbig // 2 + 1)) +
                  1j * rng.standard_normal((n_floats, Nbig // 2 + 1)))
            xi[:, 0] = rng.standard_normal(n_floats)
            fld = np.fft.irfft(np.sqrt(lam)[None] * xi, n=Nbig, axis=1)
            z.append(fld[:, :L] * np.sqrt(Nbig / 2.0))
        u, v = z
        O = np.zeros(M + 1)
        E = np.zeros(M + 1)
        npr = np.zeros(M + 1)
        for m in range(M + 1):
            uu = u[:, : L - m] * u[:, m:] + v[:, : L - m] * v[:, m:]
            uv = u[:, : L - m] * v[:, m:] - v[:, : L - m] * u[:, m:]
            E[m] = uu.sum()
            O[m] = uv.sum()
            npr[m] = uu.size
        Ce, Co = (E / npr) / (E[0] / npr[0]), (O / npr) / (E[0] / npr[0])
        sig.append(band_epr_spectrum(Ce, Co)[4])
    return np.array(sig)


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def synthetic_checks(seed=3):
    rng = np.random.default_rng(seed)
    # (i) identity closure on random Hermitian PD 2x2
    ident = 0.0
    stok = 0.0
    for _ in range(50):
        X = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        S = X @ X.conj().T + 0.1 * np.eye(2)
        ident = max(ident, abs(epr_integrand_general(S) -
                               epr_integrand_2x2(S)))
        stok = max(stok, abs(epr_integrand_general(S) -
                             epr_integrand_stokes(S)))
    # (ii) rotator: quadrature vs residue vs NR57 current formula
    rot = []
    for nu, f in ((0.7, 0.9), (1.5, 0.4), (0.35, 1.3)):
        A = nu * np.eye(2) + f * EPS
        sp = ou_epr_spectral(A, np.eye(2))
        rot.append(dict(nu=nu, f=f, spectral=sp,
                        exact=rotator_epr_exact(nu, f),
                        current=ou_epr_current(A, np.eye(2))))
    # (iii) general OU (2d anisotropic + 3d) spectral == current
    gen = []
    for d in (2, 3):
        M = rng.standard_normal((d, d))
        A = M @ M.T + d * np.eye(d) + 0.8 * (rng.standard_normal((d, d)) -
                                             rng.standard_normal((d, d)).T)
        Dm = np.diag(rng.uniform(0.5, 2.0, d))
        gen.append(dict(d=d, spectral=ou_epr_spectral(A, Dm),
                        current=ou_epr_current(A, Dm)))
    # (iv) reversible control
    Ms = rng.standard_normal((2, 2))
    Asym = Ms @ Ms.T + 2 * np.eye(2)
    rev = dict(spectral=ou_epr_spectral(Asym, np.eye(2)),
               current=ou_epr_current(Asym, np.eye(2)))
    # (v) sampling = data processing
    A = 0.8 * np.eye(2) + 1.1 * EPS
    samp = []
    for dt in (1.6, 0.8, 0.4, 0.2, 0.1, 0.05):
        th = np.linspace(-np.pi, np.pi, 3001)
        Sd = sampled_ou_spectral_density(A, np.eye(2), dt, th,
                                         mmax=int(30 / dt))
        samp.append(dict(dt=dt,
                         rate=discrete_epr_per_step(Sd, th) / dt))
    # (vi) waves vs vortex
    vmax, dmax = wave_field_spectral_epr()
    # (vii) non-Gaussian ring
    ring = dict(biased_true=ring_true_epr(0.5, 0.1),
                biased_gauss=ring_gaussian_epr(0.5, 0.1),
                fair_true=0.0, fair_gauss=ring_gaussian_epr(0.3, 0.3))
    return dict(identity_max_err=ident, stokes_max_err=stok,
                rotator=rot, general_ou=gen, reversible=rev,
                sampling=samp, wave_Vmax=vmax, wave_densmax=dmax,
                ring=ring, cont_exact=rotator_epr_exact(0.8, 1.1))


def analyze(write=True):
    cache = load_bands()
    syn = synthetic_checks()
    bands = band_analysis(cache)
    null_eq = surrogate_null(cache, "eq", n_rep=200)
    null_trop = surrogate_null(cache, "nh_trop", n_rep=120, seed=2)
    real = dict(
        bands=bands,
        eq_sigma=bands["eq"]["sigma_nats_per_step"],
        eq_null_q95=float(np.quantile(null_eq, 0.95)),
        nhtrop_sigma=bands["nh_trop"]["sigma_nats_per_step"],
        nhtrop_null_q95=float(np.quantile(null_trop, 0.95)),
    )
    out = dict(description=__doc__.split("\n")[0],
               synthetic=syn, real=real)
    out["verdicts"] = verdicts(out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    return out


def verdicts(out):
    s = out["synthetic"]
    r = out["real"]
    b = r["bands"]
    rot_ok = all(abs(x["spectral"] - x["exact"]) < 1e-5 * x["exact"] and
                 abs(x["current"] - x["exact"]) < 1e-9 * x["exact"]
                 for x in s["rotator"])
    gen_ok = all(abs(g["spectral"] - g["current"]) <
                 1e-4 * abs(g["current"]) for g in s["general_ou"])
    rates = [x["rate"] for x in s["sampling"]]
    samp_ok = all(rates[i] < rates[i + 1] for i in range(len(rates) - 1)) \
        and all(rr < s["cont_exact"] for rr in rates) \
        and abs(rates[-1] - s["cont_exact"]) < 0.06 * s["cont_exact"]
    trop_interior = all(
        20.0 < b[n]["peak_period_days"] < 90.0 and
        all(20.0 < p < 90.0
            for p in b[n]["peak_periods_tapers"].values())
        for n in ("nh_trop", "sh_trop"))
    polar_split = (b["nh_polar"]["peak_period_days"] > 120.0 and
                   b["nh_polar"]["frac_below_120d"] > 0.4 and
                   b["sh_polar"]["peak_period_days"] < 90.0)
    eq_min = (r["eq_sigma"] <= min(b[n]["sigma_nats_per_step"]
                                   for n in BANDS if n != "eq") and
              r["eq_sigma"] < r["eq_null_q95"])
    mirror = (abs(np.log(b["nh_trop"]["sigma_nats_per_step"] /
                         b["sh_trop"]["sigma_nats_per_step"])) < np.log(3.0))
    return dict(
        identity_exact=bool(s["identity_max_err"] < 1e-12 and
                            s["stokes_max_err"] < 1e-12),
        rotator_closes_on_nr57=bool(rot_ok),
        spectral_equals_current_all_ou=bool(gen_ok),
        reversible_null=bool(abs(s["reversible"]["spectral"]) < 1e-6 and
                             abs(s["reversible"]["current"]) < 1e-10),
        sampling_lower_bound=bool(samp_ok),
        waves_reversible=bool(s["wave_Vmax"] < 1e-12 and
                              s["wave_densmax"] < 1e-10),
        gaussian_lower_bounds_true=bool(
            s["ring"]["biased_gauss"] < s["ring"]["biased_true"] and
            s["ring"]["biased_gauss"] > 0.0 and
            abs(s["ring"]["fair_gauss"]) < 1e-10),
        ocean_eddy_clock_carries_epr=bool(trop_interior),
        ocean_polar_split=bool(polar_split),
        ocean_equator_reversible=bool(eq_min),
        ocean_hemispheric_mirror=bool(mirror),
        ocean_no_clipping=bool(all(b[n]["min_det_margin"] > 0.0
                                   for n in BANDS)),
    )


def main():
    out = analyze(write=True)
    s = out["synthetic"]
    r = out["real"]
    print("identity max err:", s["identity_max_err"], s["stokes_max_err"])
    for x in s["rotator"]:
        print(f"rotator nu={x['nu']} f={x['f']}: spec {x['spectral']:.6f} "
              f"exact {x['exact']:.6f} current {x['current']:.6f}")
    for g in s["general_ou"]:
        print(f"general OU d={g['d']}: spec {g['spectral']:.6f} "
              f"current {g['current']:.6f}")
    print("sampling rates ->", [f"{x['rate']:.4f}" for x in s["sampling"]],
          "cont", f"{s['cont_exact']:.4f}")
    print("wave Vmax/densmax:", s["wave_Vmax"], s["wave_densmax"])
    print("ring true/gauss:", s["ring"]["biased_true"],
          s["ring"]["biased_gauss"], "fair:", s["ring"]["fair_gauss"])
    print("\nband  lat  sigma(nats/step)  peak_period(d)  frac<120d")
    for n in BANDS:
        b = r["bands"][n]
        print(f"{n:12s} {b['lat_mean']:6.1f} {b['sigma_nats_per_step']:.3e} "
              f"{b['peak_period_days']:7.1f} {b['frac_below_120d']:.2f}")
    print("eq sigma vs null q95:", r["eq_sigma"], r["eq_null_q95"])
    print("nh_trop sigma vs its null q95:", r["nhtrop_sigma"],
          r["nhtrop_null_q95"])
    print("\nverdicts:", json.dumps(out["verdicts"], indent=1))


if __name__ == "__main__":
    main()
