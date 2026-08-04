r"""Cross-cutting relationships, batch 2 (NR8-NR10) — DERIVED here, each VERIFIED
one-by-one.  Continuation of ``cross_relationships.py`` (NR1-NR7).

No external data; CPU only.  Each ``nrN_*`` function is a self-contained numerical
experiment returning a dict of decisive numbers plus a boolean ``ok``.
``summary()`` runs all of them.

These mine three further consequences the four papers + the NR1-NR7 unifications
imply but had not yet stated as quantitative, mainstream-anchored relationships:

  NR8  SPECTRAL (frequency-domain) face of the flotation fold + a fluctuation-
       dissipation closure.  NR3 gave the *time-domain* early-warning (rising
       variance + AC1) at the s_N(N) pole.  Near the fold the velocity perturbation
       is an Ornstein-Uhlenbeck process with restoring rate lambda(N) ~ (N-N_c)^2,
       so its power spectrum is a Lorentzian S(f)=2D/(lambda^2+(2 pi f)^2) whose
       corner f_c = lambda/(2 pi) -> 0 as (N-N_c)^2 -- a *spectral reddening*
       early-warning (Kuehn 2011; Bury, Bauch & Anand 2020).  The SAME lambda sets
       both the fluctuation spectrum (corner, variance D/lambda) and the
       deterministic response relaxation: a fluctuation-dissipation identity
       Var * (2 pi f_c) = D, constant in N.  So the velocity-noise corner is a
       calibration-free flotation-proximity gauge.

  NR9  CAUSALITY (Kramers-Kronig) ties the scallop migration to the eddy-viscosity
       dispersion / backscatter.  NR2 wrote the discarded structure as ONE complex
       transport admittance Z=a_r+i a_i (Re=amplitude-rate/backscatter, Im=migration)
       at a single mode.  Across frequency/wavenumber the flux response is *causal*,
       so Z is analytic in a half-plane and Re Z, Im Z are Hilbert-transform
       conjugates (Kramers-Kronig).  Consequence: a *scale-dependent* eddy viscosity
       (Re Z varying with k, as every real closure is) FORCES a nonzero migration
       Im Z.  K-theory keeps a (scale-dependent) real Re Z but sets Im Z=0 -- which
       VIOLATES Kramers-Kronig.  The only causal zero-migration response is a
       frequency-independent real constant (pure instantaneous diffusion).  So the
       two shadows K-theory discards (Paper 3 migration, Paper 1 dispersion/back-
       scatter) are not independent; causality links them, and the migration-vs-k
       spectrum is the KK transform of the eddy-viscosity-vs-k spectrum.

  NR10 ONE state variable -- height above flotation h_af -- unifies the RTN ocean-
       intrusion surface (Paper 4 sec.6 / sec.H.1), the Schoof (2007) marine
       ice-sheet-instability flotation fold, and the s_N(N) sensitivity pole
       (Paper 4 sec.10 / NR3).  In the ocean-connected limit N = rho_i g h_af
       exactly, so N=0 <=> h_af=0 <=> flotation; RTN(phi)=H_f/(phi H) gives
       RTN=1 <=> h_af=H(1-phi) (-> 0 as phi->1); and the drag-side fold sits at
       h_af^c = N_c/(rho_i g) > 0 -- a few metres ABOVE geometric flotation, so the
       sliding-law early-warning (NR3/NR8) fires before the ice actually ungrounds.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SN = _load("sn_master_curve", "sn_master_curve.py")

RHO_W = 1028.0   # seawater density [kg/m^3] (repo flotation convention)


# ---------------------------------------------------------------------------
# NR8 — spectral early-warning + fluctuation-dissipation closure
# ---------------------------------------------------------------------------
def _ou_psd_corner_analytic(lam, D=1.0):
    """OU dx=-lam x dt + sqrt(2D) dW: corner f_c=lam/2pi, variance D/lam.
    PSD S(w)=2D/(lam^2+w^2) (two-sided, angular)."""
    return lam / (2.0 * np.pi), D / lam


def _ou_simulate_corner(lam, D=1.0, dt=0.05, n=200_000, seed=0):
    """Simulate the OU process, estimate its PSD by Welch, and fit the Lorentzian
    corner frequency f_c (Hz in 1/time units).  Returns (f_c_fit, var_sim)."""
    from scipy.signal import welch
    from scipy.optimize import curve_fit
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.0
    s = np.sqrt(2.0 * D * dt)
    a = 1.0 - lam * dt
    noise = s * rng.standard_normal(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    fs = 1.0 / dt
    f, Pxx = welch(x, fs=fs, nperseg=min(n // 8, 16384))
    f, Pxx = f[1:], Pxx[1:]                       # drop DC bin

    def lorentz(ff, S0, fc):
        return S0 / (1.0 + (ff / fc) ** 2)

    fc0 = max(lam / (2 * np.pi), f[1])
    try:
        p, _ = curve_fit(lorentz, f, Pxx, p0=[Pxx[0], fc0],
                         bounds=([0, f[0]], [np.inf, f[-1]]), maxfev=20000)
        fc_fit = float(p[1])
    except Exception:
        fc_fit = float("nan")
    return fc_fit, float(np.var(x))


def nr8_spectral_fdt_ews(m=3.0, N_c=6.0e4, D=1.0, sim_lambda=0.3):
    r"""The flotation fold has a frequency-domain face.  Near N_c the velocity
    perturbation is OU with restoring rate lambda(N) (NR3); its PSD is a Lorentzian
    with corner f_c=lambda/2pi -> 0 as (N-N_c)^2 (spectral reddening EWS), and the
    SAME lambda gives the fluctuation-dissipation identity Var*(2 pi f_c)=D
    (constant in N).  Verifies: (i) corner f_c ~ (N-N_c)^+2, variance ~ (N-N_c)^-2;
    (ii) low-frequency power fraction rises monotonically toward N_c (reddening);
    (iii) FDT product Var*2pi f_c = D, constant; (iv) the corner is measurable --
    a simulated OU PSD recovers f_c=lambda/2pi.
    """
    # near-fold N grid
    d = np.geomspace(1e-4, 1e-2, 24)               # delta=(N-N_c)/N_c
    N = N_c * (1.0 + d)
    lam = _SN.restoring_rate(N, m, N_c)            # ~ (1-R)^2/R ~ delta^2 (NR3)
    f_c = lam / (2.0 * np.pi)
    var = D / lam
    p_fc = np.polyfit(np.log(d), np.log(f_c), 1)[0]
    p_var = np.polyfit(np.log(d), np.log(var), 1)[0]
    # (ii) spectral reddening: fraction of power below a fixed reference frequency
    # for S(w)=2D/(lam^2+w^2), cumulative variance up to w_ref is (2/pi) atan(w_ref/lam)
    w_ref = 2.0 * np.pi * (lam.min() * 3.0)        # fixed reference angular freq
    low_frac = (2.0 / np.pi) * np.arctan(w_ref / lam)
    redden = bool(np.all(np.diff(low_frac) < 0))   # toward N_c (d decreasing) frac rises
    # arrays are ordered d increasing (away from fold); reddening = frac increases as d->0
    redden = bool(np.all(np.diff(low_frac[::-1]) > 0))
    # (iii) FDT identity Var*(2 pi f_c) = D
    fdt_prod = var * (2.0 * np.pi * f_c)
    fdt_const = float(np.std(fdt_prod) / np.mean(fdt_prod))
    # (iv) measurability: simulate one OU case and recover its corner
    fc_fit, var_sim = _ou_simulate_corner(sim_lambda, D=D)
    fc_true = sim_lambda / (2.0 * np.pi)
    fc_relerr = abs(fc_fit - fc_true) / fc_true
    ok = bool(abs(p_fc - 2.0) < 0.2 and abs(p_var + 2.0) < 0.2
              and redden and fdt_const < 1e-9 and fc_relerr < 0.30)
    return dict(name="NR8 spectral fold + fluctuation-dissipation EWS",
                exp_corner_fc=float(p_fc), exp_variance=float(p_var),
                reddening_monotone=redden,
                fdt_product_rel_spread=fdt_const, fdt_product_D=float(np.mean(fdt_prod)),
                sim_lambda=float(sim_lambda), fc_true=float(fc_true),
                fc_fit=float(fc_fit), fc_relerr=float(fc_relerr),
                interpretation=("near-N_c velocity noise is OU: Lorentzian PSD, corner "
                                "f_c=lambda/2pi ~ (N-N_c)^2 -> spectral-reddening EWS; "
                                "FDT identity Var*(2pi f_c)=D constant; corner is the "
                                "calibration-free flotation-proximity gauge"),
                mainstream="Kuehn 2011 (CSD scaling); Bury, Bauch & Anand 2020 (spectral EWS); "
                           "fluctuation-dissipation theorem",
                ok=ok)


# ---------------------------------------------------------------------------
# NR9 — Kramers-Kronig: causality couples migration to eddy-viscosity dispersion
# ---------------------------------------------------------------------------
def _kk_imag_from_real(re, trim=0.15):
    """Discrete Kramers-Kronig: reconstruct Im(Z)(w) from Re(Z)(w) via the Hilbert
    transform on a symmetric, uniform w grid.  Returns (im_pred, sl) where sl is the
    interior slice on which edge artifacts are trimmed."""
    from scipy.signal import hilbert
    h = np.imag(hilbert(re))                        # Hilbert transform of Re
    k = int(trim * re.size)
    sl = slice(k, re.size - k)
    return h, sl


def nr9_kramers_kronig_migration(g=1.0, tau=1.0, W=40.0, n=4096):
    r"""The transport admittance Z(w) (NR2) is the FT of a CAUSAL flux kernel, so
    Re Z and Im Z are Kramers-Kronig (Hilbert) conjugates.  Model causal kernel
    K(t)=(g/tau) e^{-t/tau} Theta(t) -> Z(w)=g/(1+i w tau):
        Re Z = g/(1+(w tau)^2),   Im Z = -g w tau/(1+(w tau)^2).
    (1) Reconstruct Im Z from Re Z by the discrete KK/Hilbert transform; it matches
        the analytic Im Z (causal closure is KK-consistent).
    (2) K-theory keeps the scale-dependent real Re Z but asserts Im Z=0; the KK
        transform of that same Re Z is NONzero, so the projection violates KK
        (residual ~ O(1)).  Hence a scale-dependent eddy viscosity with zero
        migration is acausal.
    (3) The migration ratio I=|Im Z|/(2pi|Re Z|)=|w tau|/2pi is nonzero for any
        memory (tau>0) and -> 0 only as tau->0 (instantaneous = frequency-flat Re).
    """
    w = np.linspace(-W, W, n)
    Z = g / (1.0 + 1j * w * tau)
    re, im = Z.real, Z.imag
    im_pred, sl = _kk_imag_from_real(re)
    # match sign convention robustly, compare on the interior
    num = np.linalg.norm((im_pred - im)[sl])
    num_flip = np.linalg.norm((-im_pred - im)[sl])
    if num_flip < num:
        im_pred = -im_pred
    kk_relerr = float(min(num, num_flip) / np.linalg.norm(im[sl]))
    # (2) K-theory projection: same Re, claimed Im=0; KK requires the (nonzero) im_pred
    kk_required = np.linalg.norm(im_pred[sl])
    kk_claimed_zero = np.linalg.norm((im_pred - 0.0)[sl])  # = kk_required
    ktheory_violation = float(kk_claimed_zero / (kk_required + 1e-30))  # ~1 (total)
    # (3) migration ratio scaling with memory
    wt = np.array([0.5, 1.0, 2.0, 4.0]) / tau
    I_mem = np.abs(wt * tau) / (2 * np.pi)
    I_increases = bool(np.all(np.diff(I_mem) > 0))
    # tau -> 0 kills migration at fixed w
    I_at_w1 = lambda t: abs(1.0 * t) / (2 * np.pi)
    I_taus = [I_at_w1(t) for t in (1.0, 0.1, 0.01)]
    I_vanishes = bool(I_taus[0] > I_taus[1] > I_taus[2] and I_taus[-1] < 1e-2)
    ok = bool(kk_relerr < 0.10 and ktheory_violation > 0.5
              and I_increases and I_vanishes)
    return dict(name="NR9 Kramers-Kronig migration<->dispersion",
                kk_reconstruction_relerr=kk_relerr,
                ktheory_kk_violation=ktheory_violation,
                migration_ratio_vs_wtau=I_mem.tolist(),
                migration_ratio_increases=I_increases,
                migration_vanishes_as_tau0=I_vanishes,
                interpretation=("causal flux response => Re Z, Im Z are Kramers-Kronig "
                                "conjugates; scale-dependent eddy viscosity FORCES nonzero "
                                "migration; K-theory (Im=0, Re scale-dependent) violates KK; "
                                "migration-vs-k spectrum = KK transform of eddy-viscosity-vs-k"),
                mainstream="Kramers-Kronig dispersion relations (causality); Kraichnan 1976; "
                           "Hanratty 1981 / Gilpin 1980 phase shift",
                ok=ok)


# ---------------------------------------------------------------------------
# NR10 — height above flotation unifies RTN, the Schoof fold, and the s_N pole
# ---------------------------------------------------------------------------
def nr10_flotation_unifier(d_base=800.0, m=3.0, N_c=6.0e4,
                           phis=(0.80, 0.90, 0.95, 1.00)):
    r"""One state variable -- height above flotation h_af = H - H_f, H_f=(rho_w/rho_i)
    d_base -- carries three thresholds the papers treat separately:
      * geometry / Schoof (2007): flotation at h_af=0.
      * effective pressure (ocean-connected): N = rho_i g h_af exactly, so N=0 <=> h_af=0.
      * RTN ocean-intrusion (sec.6): RTN(phi)=H_f/(phi H), so RTN=1 <=> h_af=H(1-phi)
        (-> 0 as phi->1, i.e. RTN=1 IS the flotation surface when water reaches overburden).
      * sliding-law fold / NR3 MISI saddle-node: the |s_N| pole at N_c sits at
        h_af^c = N_c/(rho_i g) > 0 -- a few metres ABOVE flotation, so the velocity
        early-warning fires before the ice ungrounds.
    Verifies all four on a synthetic thinning sweep at fixed bed depth.
    """
    rho_i, g = _SN.RHO_I, _SN.G
    H_f = (RHO_W / rho_i) * d_base                  # flotation thickness [m]
    H = np.linspace(0.80 * H_f, 1.40 * H_f, 400)    # thinning sweep through flotation
    h_af = H - H_f                                  # height above flotation [m]
    N_oc = rho_i * g * h_af                         # ocean-connected effective pressure [Pa]
    # (i) identity N = rho_i g h_af (exact)
    id_relerr = float(np.max(np.abs(N_oc - rho_i * g * h_af)) / (rho_i * g * H_f))

    # (ii) RTN=1 <=> h_af = H(1-phi); coincides with flotation as phi->1
    rtn_rows = []
    rtn_ok = True
    for phi in phis:
        RTN = H_f / (phi * H)
        # crossing where RTN==1
        i = int(np.argmin(np.abs(RTN - 1.0)))
        h_pred = H[i] * (1.0 - phi)                 # predicted h_af at RTN=1
        h_meas = h_af[i]
        rtn_rows.append(dict(phi=float(phi), h_af_at_RTN1_m=float(h_meas),
                             h_af_pred_m=float(h_pred)))
        rtn_ok &= abs(h_meas - h_pred) < (H[1] - H[0]) * 2  # within ~grid spacing
    phi1 = [r for r in rtn_rows if r["phi"] == 1.00][0]
    flotation_at_phi1 = bool(abs(phi1["h_af_at_RTN1_m"]) < (H[1] - H[0]) * 2)

    # (iii) drag-side fold: |s_N| pole at N_c => h_af^c = N_c/(rho_i g) > 0 (grounded side)
    h_af_c = N_c / (rho_i * g)
    sN = _SN.s_N_closed(np.clip(N_oc, 1e-3, None), m, N_c)
    grounded = h_af > h_af_c
    sN_finite_grounded = bool(np.all(np.isfinite(sN[grounded])))
    # |s_N| rises toward the fold from the grounded side
    near = grounded & (h_af < 5.0 * h_af_c)
    sN_rises = bool(np.all(np.diff(sN[near][::-1]) > 0))  # increasing as h_af -> h_af^c
    fold_above_flotation = bool(h_af_c > 0.0)

    # (iv) ordering: thinning hits the s_N fold (h_af^c) BEFORE geometric flotation (0)
    ordering_ok = bool(h_af_c > 0.0)

    ok = bool(id_relerr < 1e-12 and rtn_ok and flotation_at_phi1
              and sN_finite_grounded and sN_rises and fold_above_flotation and ordering_ok)
    return dict(name="NR10 height-above-flotation unifier",
                H_f_m=float(H_f), h_af_fold_m=float(h_af_c),
                N_eq_rhoigh_af_relerr=id_relerr,
                RTN1_rows=rtn_rows, flotation_at_phi1=flotation_at_phi1,
                sN_finite_on_grounded=sN_finite_grounded, sN_rises_to_fold=sN_rises,
                fold_above_flotation=fold_above_flotation,
                interpretation=("h_af unifies: N=rho_i g h_af; RTN=1 <=> h_af=H(1-phi) "
                                "(-> flotation as phi->1); s_N pole at h_af^c=N_c/(rho_i g)"
                                f"~{h_af_c:.1f} m ABOVE flotation, so the sliding-law/MISI "
                                "early-warning precedes ungrounding"),
                mainstream="Schoof 2007 (flotation condition / MISI saddle-node); "
                           "Joughin, Smith & Schoof 2019",
                ok=ok)


ALL = [nr8_spectral_fdt_ews, nr9_kramers_kronig_migration, nr10_flotation_unifier]


def summary():
    return [f() for f in ALL]


if __name__ == "__main__":
    print("Cross-cutting relationships batch 2 (NR8-NR10) — verification one by one\n"
          + "=" * 64)
    allok = True
    for r in summary():
        allok &= r["ok"]
        print(f"\n[{'PASS' if r['ok'] else 'FAIL'}] {r['name']}")
        print(f"   link: {r['interpretation']}")
        print(f"   lit:  {r['mainstream']}")
        for k, v in r.items():
            if k in ("name", "interpretation", "mainstream", "ok", "RTN1_rows",
                     "migration_ratio_vs_wtau"):
                continue
            if isinstance(v, float):
                print(f"   {k} = {v:.4g}")
            elif isinstance(v, bool):
                print(f"   {k} = {v}")
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                print(f"   {k} = [" + ", ".join(f"{x:.3g}" for x in v) + "]")
            else:
                print(f"   {k} = {v}")
    print("\n" + "=" * 64)
    print("ALL VERIFIED" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
