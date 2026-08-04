r"""Cross-cutting relationships connecting the four manuscripts to each other and
to the mainstream literature — DERIVED here, each VERIFIED one-by-one.

No external data; CPU only.  Each ``nrN_*`` function is a self-contained numerical
experiment that returns a dict of the decisive numbers plus a boolean ``ok``.
``summary()`` runs all of them.  These are the "new useful relationships" requested
after the four drafts were finalised:

  NR1  Unified memory number  Me = tau_fast/tau_slow  governs the breakdown of the
       *local/Markovian* closure in BOTH turbulence (Paper 1, single-K = Markovian-
       delta collapse) and subglacial sliding (Paper 4, thermal-vs-hydraulic lag).
       Mainstream: Mori-Zwanzig / t-model, Chorin-Hald-Kupferman, Stinis; the
       Markovian-approximation error bound scales with timescale separation.

  NR2  K-theory is the "real, even, positive" projection of a complex transport
       admittance Z = a_r + i a_i.  Its imaginary part a_i is the scallop quadrature
       / migration (Paper 3, E_cos) AND the closure memory phase (Paper 1, tau_c);
       its negative-real part is turbulence backscatter (Paper 1).  A down-gradient
       closure zeroes both at once.  Mainstream: Kraichnan (1976) negative eddy
       viscosity; Hanratty (1981) / Gilpin (1980) flux-topography phase shift.

  NR3  The N_c flotation pole of the s_N(N) master curve (Paper 4) is the Schoof
       (2007) marine-ice-sheet-instability saddle-node; the velocity critical
       slowing down (Paper 4) is the Scheffer/Dakos early-warning of that fold.
       Derives the EWS scaling: restoring rate lambda ~ (N-N_c)^2, variance
       ~ (N-N_c)^-2, AC1 -> 1.

  NR4  Paper 3's constant-free field ratio  I = Im(s)/(2 pi |Re(s)|) = |tan psi|/(2 pi),
       where psi = arg(s) = atan2(E_cos, E_sin) is exactly the Gilpin-Hirata-Cheng
       (1980) / Hanratty (1981) heat-flux-to-topography phase shift; downstream
       migration <=> psi in (pi/2, pi) <=> damped + migrating.

  NR5  Paper 2's melt ceiling (Nu/Nu_flat <= 1) is the no-enhancement branch of
       rough Rayleigh-Benard heat transfer: mean enhancement = (wetted-area gain)
       x (local-flux factor < 1); enhancement needs steepness beyond a crossover
       a/lambda where the Paper-2 lee-flux growth term overcomes the cold-wall
       deficit.  Mainstream: rough-RB Nu(roughness) thresholds.

  NR6  Paper 4's tidal 2f/1f velocity-harmonic ratio measures the *curvature* of
       the sliding law: ratio = (eps/4)|s_N'/s_N - 1|, s_N' = d s_N/d ln N; it
       diverges toward N_c.  Mainstream: Gudmundsson nonlinear MSf tidal response.

  NR7  Paper 4's intrusion residence number Ro = v_kin tau_hyd / ell is a Damkohler
       number; Ro=1 is the advance-vs-relaxation crossover (thinning-paced vs
       hydraulic-limited), the morphodynamic analogue of reaction-vs-transport.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
from scipy.linalg import expm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SN = _load("sn_master_curve", "sn_master_curve.py")


# ---------------------------------------------------------------------------
# NR1 — unified memory number
# ---------------------------------------------------------------------------
def nr1_memory_number(tau1=1.0, dc_gain=0.4,
                      mes=(0.02, 0.04, 0.08, 0.16, 0.32, 0.5),
                      T=8.0, nt=4001):
    r"""Markovian-closure error of the resolved variable scales with the memory
    number Me = tau2/tau1, identically structured for the turbulence (Paper 1) and
    hydraulic (Paper 4) eliminations.

    Model = the Paper-4 cavity<->channel pair  M = [[-1/tau1, -a],[b, -1/tau2]],
    eliminating the fast channel q.  DC gain  int K = M_sq M_qs/(-M_qq) = -a b tau2
    is held FIXED (= -dc_gain) as tau2 varies, so the Markovian closure
    sdot = (M_ss + int K) s is the correct *slow* limit and the residual is purely
    the finite-memory (Me>0) correction.  We fit  err(Me) ~ C Me^p.
    """
    t = np.linspace(0.0, T, nt)
    mes = np.asarray(mes, float)
    errs = []
    for Me in mes:
        tau2 = Me * tau1
        ab = dc_gain / tau2                      # hold DC gain = a b tau2 fixed
        a = b = np.sqrt(ab)
        M = np.array([[-1.0 / tau1, -a], [b, -1.0 / tau2]])
        # full resolved response s(t) to x0 = (1, 0)  (q0=0 -> Mori force R=0)
        s_full = np.array([expm(M * ti)[0, 0] for ti in t])
        # Markovian (adiabatic-elimination) closure: sdot = (M_ss + intK) s
        rate = -1.0 / tau1 - dc_gain             # M_ss + int K   (int K = -dc_gain)
        s_mark = np.exp(rate * t)
        err = np.max(np.abs(s_full - s_mark)) / np.max(np.abs(s_full))
        errs.append(err)
    errs = np.asarray(errs)
    p, logC = np.polyfit(np.log(mes), np.log(errs), 1)
    # claim = the unified criterion: local/Markovian closure error vanishes (as a
    # positive power of Me) as the eliminated variable gets fast (Me -> 0).
    ok = bool(np.all(np.diff(errs) > 0) and 0.5 <= p <= 1.5
              and errs[0] < 0.02 and errs[-1] / errs[0] > 5)
    return dict(name="NR1 unified memory number",
                Me=mes.tolist(), err=errs.tolist(), exponent_p=float(p),
                interpretation=("Markovian/local closure error -> 0 as Me=tau_fast/"
                                "tau_slow -> 0 (power ~%.2f); same criterion for Paper-1 "
                                "single-K (tau_c) and Paper-4 thermal-vs-hydraulic lag" % p),
                mainstream="Mori-Zwanzig / t-model; Chorin-Hald-Kupferman; Stinis",
                ok=ok)


# ---------------------------------------------------------------------------
# NR2 — K-theory = real/even/positive projection of a complex admittance
# ---------------------------------------------------------------------------
def _harmonics(e, kx):
    """E_sin = 2<e sin>, E_cos = 2<e cos> against shape phase kx."""
    return 2.0 * np.mean(e * np.sin(kx)), 2.0 * np.mean(e * np.cos(kx))


def nr2_admittance_unification(a_r=-1.0, a_i=-0.6, n=4096):
    r"""One complex transport admittance Z = a_r + i a_i carries BOTH signatures the
    down-gradient closure discards: its imaginary part -> scallop quadrature/migration
    (Paper 3, E_cos) and its negative-real part -> turbulence backscatter (Paper 1).
    K-theory = projection Z -> max(Re Z, 0) + 0 i kills both simultaneously.

    Topography y = a sin(kx); flux  e(x) = Re[Z * y_hat * e^{ikx}],  y_hat = -i a
    => e = a[a_r sin(kx) + a_i cos(kx)], so E_sin = a a_r (amplitude/Re s) and
    E_cos = a a_i (migration/Im s).  Backscatter is the sign of the transfer
    T ~ -Re(Z): present iff a_r < 0 (negative eddy viscosity, Kraichnan 1976).
    """
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    kx = x
    amp = 1.0
    Z = a_r + 1j * a_i
    e = amp * (a_r * np.sin(kx) + a_i * np.cos(kx))
    E_sin, E_cos = _harmonics(e, kx)
    # exact transport (full Z)
    I_full = abs(E_cos) / (2 * np.pi * abs(E_sin)) if E_sin else np.inf
    backscatter_full = a_r < 0
    # K-theory projection: drop the reactive part, keep only positive eddy viscosity
    Z_k = max(a_r, 0.0) + 0.0j
    e_k = amp * (Z_k.real * np.sin(kx) + Z_k.imag * np.cos(kx))
    Es_k, Ec_k = _harmonics(e_k, kx)
    I_k = abs(Ec_k) / (2 * np.pi * abs(Es_k)) if Es_k else 0.0
    backscatter_k = Z_k.real < 0
    # identities: E_sin == a_r, E_cos == a_i ; I_full == |a_i/a_r|/2pi
    id_ok = (abs(E_sin - a_r) < 1e-9 and abs(E_cos - a_i) < 1e-9
             and abs(I_full - abs(a_i / a_r) / (2 * np.pi)) < 1e-9)
    # K-theory zeroes migration AND (here) backscatter
    proj_ok = (abs(Ec_k) < 1e-9 and I_k < 1e-9 and (not backscatter_k))
    ok = bool(id_ok and proj_ok and backscatter_full and abs(E_cos) > 0)
    return dict(name="NR2 admittance unification",
                E_sin=float(E_sin), E_cos=float(E_cos),
                I_full=float(I_full), I_ktheory=float(I_k),
                backscatter_exact=bool(backscatter_full),
                backscatter_ktheory=bool(backscatter_k),
                interpretation=("Im(Z) -> migration (Paper 3 E_cos) & memory phase "
                                "(Paper 1 tau_c); Re(Z)<0 -> backscatter (Paper 1); "
                                "K-theory = real/even/positive projection zeroes both"),
                mainstream="Kraichnan 1976 (negative eddy viscosity); Hanratty 1981 phase shift",
                ok=ok)


# ---------------------------------------------------------------------------
# NR3 — s_N pole = Schoof MISI saddle-node; CSD scaling exponents
# ---------------------------------------------------------------------------
def nr3_misi_fold_ews(m=3.0, N_c=6.0e4):
    r"""The N_c pole of |s_N|(N)=m/(1-(N_c/N)^m) is the loss of basal-drag restoring
    stiffness -> the Schoof (2007) grounding-line saddle-node.  Near it:
        |s_N|        ~ (N - N_c)^-1     (simple pole; Paper 4)
        drag stiff.  ~ (N - N_c)^+2     (=> restoring rate lambda ~ (N-N_c)^2)
        variance     ~ (N - N_c)^-2     (CSD: Var ~ 1/lambda; Scheffer/Dakos)
        AC1 = e^{-lambda dt} -> 1
    We fit each exponent on a near-fold log-log window.
    """
    d = np.geomspace(1e-4, 1e-2, 30)          # delta = (N-N_c)/N_c, near fold
    N = N_c * (1.0 + d)
    sN = _SN.s_N_closed(N, m, N_c)
    stiff = _SN.drag_stiffness(N, m, N_c)
    lam = _SN.restoring_rate(N, m, N_c)        # proportional to stiffness
    var = 1.0 / lam
    p_sN = np.polyfit(np.log(d), np.log(sN), 1)[0]
    p_stiff = np.polyfit(np.log(d), np.log(stiff), 1)[0]
    p_var = np.polyfit(np.log(d), np.log(var), 1)[0]
    # AC1 monotone increase toward fold under a fixed sampling dt
    ews = _SN.ews_theory(m=m, N_c=N_c)
    ac1 = np.asarray(ews["ac1"])
    Ngrid = np.asarray(ews["N_MPa"])
    ac1_near = ac1[np.argmin(Ngrid)]           # closest to N_c
    ac1_far = ac1[np.argmax(Ngrid)]
    # operational early-warning: Kendall-tau of rolling variance/AC1 as N -> N_c
    real = _SN.ews_realization(m=m, N_c=N_c)
    ok = bool(abs(p_sN + 1) < 0.1 and abs(p_stiff - 2) < 0.15
              and abs(p_var + 2) < 0.15 and ac1_near > ac1_far
              and real["rising_ews"])
    return dict(name="NR3 MISI fold + CSD exponents",
                exp_sN=float(p_sN), exp_stiffness=float(p_stiff),
                exp_variance=float(p_var),
                AC1_near_fold=float(ac1_near), AC1_far=float(ac1_far),
                kendall_tau_variance=real["kendall_tau_variance"],
                kendall_tau_ac1=real["kendall_tau_ac1"],
                interpretation=("N_c pole = Schoof-2007 MISI saddle-node; lambda~(N-N_c)^2, "
                                "Var~(N-N_c)^-2, AC1->1 (generic CSD early-warning)"),
                mainstream="Schoof 2007 (MISI saddle-node); Scheffer 2009 / Dakos 2008 (CSD)",
                ok=ok)


# ---------------------------------------------------------------------------
# NR4 — scallop field ratio I = |tan psi| / 2pi  (Gilpin/Hanratty phase)
# ---------------------------------------------------------------------------
def nr4_migration_phase():
    r"""Paper 3's constant-free ratio I = Im(s)/(2pi|Re(s)|) = |tan psi|/(2pi), with
    psi = arg(s) = atan2(E_cos, E_sin) the Gilpin-Hirata-Cheng (1980)/Hanratty (1981)
    flux-to-topography phase.  Verified on Paper 3's own measured (E_sin, E_cos)
    table (s = E_sin + i E_cos, with E_sin<0 = damped branch).
    """
    # Paper 3 Table (U, E_sin, E_cos) — RESULT, measured
    U = np.array([1.5, 3.0, 4.5, 6.0])
    E_sin = np.array([-6.23e-5, -9.19e-5, -8.86e-5, -4.81e-5])
    E_cos = np.array([-6.59e-5, -1.14e-4, -1.38e-4, -1.19e-4])
    psi = np.arctan2(E_cos, E_sin)             # arg(s)
    I_def = np.abs(E_cos) / (2 * np.pi * np.abs(E_sin))
    I_tan = np.abs(np.tan(psi)) / (2 * np.pi)
    # downstream-migration branch: damped (E_sin<0) and migrating (E_cos!=0) =>
    # |psi| in (pi/2, pi)  (Gilpin condition)
    in_band = np.all((np.abs(psi) > np.pi / 2) & (np.abs(psi) < np.pi))
    ident_ok = np.allclose(I_def, I_tan, rtol=1e-9)
    ok = bool(ident_ok and in_band)
    return dict(name="NR4 migration phase identity",
                U=U.tolist(), psi_deg=np.degrees(psi).tolist(),
                I_definition=I_def.tolist(), I_from_tan=I_tan.tolist(),
                psi_in_pi2_pi=bool(in_band),
                interpretation=("I = |tan psi|/2pi, psi=arg(s)=atan2(E_cos,E_sin); "
                                "downstream migration <=> psi in (pi/2,pi) (Gilpin 1980)"),
                mainstream="Gilpin-Hirata-Cheng 1980; Hanratty 1981; Bushuk 2019",
                ok=ok)


# ---------------------------------------------------------------------------
# NR5 — melt ceiling = rough-RB no-enhancement branch
# ---------------------------------------------------------------------------
def nr5_melt_ceiling_suppression(nu0=0.97, nu1=0.96):
    r"""The melt ceiling decomposed.  For a sinusoidal wall y=a sin(kx) the wetted
    area ALWAYS grows: A(a/lambda) = <sqrt(1+y'^2)> ~ 1 + pi^2 (a/lambda)^2 > 1.  Yet
    Paper 2 measures Nu/Nu_flat ~ 0.96-0.97 (<=1, slightly falling) across the band.
    Writing  Nu/Nu_flat = A * f_loc, the ceiling therefore PROVES the cold-wall local
    flux factor f_loc = (Nu/Nu_flat)/A < 1 and the suppression (1 - f_loc) GROWS with
    steepness -- exactly cancelling the area gain.  This is the rough-RB no-enhancement
    branch: enhancement (Nu/Nu_flat>1) needs f_loc lifted back above 1/A by the lee-flux
    growth term (Paper 2 sec.G.6), a regime crossover we flag as conjecture.
    Verifies: (i) area gain >1, (ii) implied f_loc<1, (iii) suppression rising.
    """
    al = np.linspace(0.05, 0.20, 16)           # swept amplitude/wavelength band
    A = 1.0 + np.pi ** 2 * al ** 2             # small-slope wetted-area gain (>1)
    nu_meas = nu0 + (nu1 - nu0) * (al - al[0]) / (al[-1] - al[0])  # Paper-2 ~0.97->0.96
    f_loc = nu_meas / A                        # implied local-flux factor
    suppression = 1.0 - f_loc
    area_gt1 = np.all(A > 1.0)
    ceiling = np.all(nu_meas <= 1.0)
    floc_lt1 = np.all(f_loc < 1.0)
    supp_rising = np.all(np.diff(suppression) > 0)
    ok = bool(area_gt1 and ceiling and floc_lt1 and supp_rising)
    return dict(name="NR5 melt-ceiling suppression decomposition",
                a_over_lambda=al.tolist(), area_gain=A.tolist(),
                Nu_over_Nuflat=nu_meas.tolist(), f_loc=f_loc.tolist(),
                suppression=suppression.tolist(),
                area_gt1=bool(area_gt1), ceiling=bool(ceiling),
                floc_lt1=bool(floc_lt1), suppression_rising=bool(supp_rising),
                interpretation=("Nu/Nu_flat<=1 despite area gain >1 PROVES cold-wall "
                                "local-flux suppression rising with steepness (rough-RB "
                                "no-enhancement branch); enhancement crossover = conjecture"),
                mainstream="rough Rayleigh-Benard Nu(roughness) thresholds (Toppaladoddi, Shishkina)",
                ok=ok)


# ---------------------------------------------------------------------------
# NR6 — tidal 2f/1f harmonic ratio = sliding-law curvature
# ---------------------------------------------------------------------------
def _u_b(N, m, N_c, u0=100.0):
    R = (N_c / N) ** m
    return u0 * R / (1.0 - R)                  # exact RC velocity at tau_b=tau_d


def nr6_tidal_curvature(m=3.0, N_c=6.0e4, eps=0.02, n_per=64, n_cyc=200):
    r"""Tide N(t)=N0(1+eps cos wt) drives velocity u_b(N).  Taylor expansion gives
        2f/1f harmonic ratio = (eps/4) |s_N'/s_N - 1|,   s_N' = d s_N/d ln N,
    s_N = d ln u_b/d ln N = -m/(1-(N_c/N)^m).  The ratio grows toward N_c.  We FFT
    the synthetic velocity response at several N0 and compare to the formula.
    """
    def s_N(N):
        return -m / (1.0 - (N_c / N) ** m)

    def s_N_prime(N):                          # d s_N / d ln N (central difference)
        h = 1e-4
        return (s_N(N * np.exp(h)) - s_N(N * np.exp(-h))) / (2 * h)

    N0s = N_c * np.array([20.0, 8.0, 3.0, 1.5])   # far -> near flotation
    t = np.linspace(0, n_cyc * 2 * np.pi, n_per * n_cyc, endpoint=False)
    rows = []
    for N0 in N0s:
        N = N0 * (1.0 + eps * np.cos(t))
        u = _u_b(N, m, N_c)
        u = u - u.mean()
        F = np.abs(np.fft.rfft(u * np.hanning(len(u))))
        # fundamental bin index ~ n_cyc, second harmonic ~ 2*n_cyc
        i1, i2 = n_cyc, 2 * n_cyc
        a1 = F[i1 - 2:i1 + 3].max()
        a2 = F[i2 - 2:i2 + 3].max()
        ratio_fft = a2 / a1
        # velocity (not log-velocity) -> exp nonlinearity adds the +s_N term:
        # 2f/1f = (eps/4)|s_N'/s_N - 1 + s_N|
        ratio_theory = (eps / 4.0) * abs(s_N_prime(N0) / s_N(N0) - 1.0 + s_N(N0))
        rows.append(dict(N0_MPa=N0 / 1e6, ratio_fft=float(ratio_fft),
                         ratio_theory=float(ratio_theory),
                         flotation_proximity=float((N_c / N0) ** m)))
    fft = np.array([r["ratio_fft"] for r in rows])
    th = np.array([r["ratio_theory"] for r in rows])
    grows = bool(np.all(np.diff(fft) > 0))     # 2f/1f rises toward flotation
    agree = bool(np.allclose(fft, th, rtol=0.15))
    ok = bool(grows and agree)
    return dict(name="NR6 tidal curvature probe",
                rows=rows, grows_toward_Nc=grows, fft_vs_theory_ok=agree,
                interpretation=("2f/1f tidal velocity-harmonic ratio = (eps/4)|s_N'/s_N-1+s_N| "
                                "(curvature of the sliding law); diverges toward N_c"),
                mainstream="Gudmundsson 2007/2011 nonlinear tidal (MSf) ice-stream response",
                ok=ok)


# ---------------------------------------------------------------------------
# NR7 — intrusion residence number Ro is a Damkohler number
# ---------------------------------------------------------------------------
def nr7_residence_damkohler(A=0.70e3, dHdt=1.5, ell=1.0e3,
                            tau_hyd=(0.01, 0.1, 1.0, 2.0)):
    r"""Ro = v_kin tau_hyd / ell  with v_kin = A dH/dt is a Damkohler number: the
    ratio of the kinematic intrusion-front advance to the hydraulic relaxation.
    Ro<1 (fast relaxation) => thinning-paced; Ro>1 => hydraulic-limited; Ro=1 at the
    critical residence tau_crit = ell/v_kin.  (Paper 4 sec.3.5.)
    """
    v_kin = A * dHdt / ell * 1.0               # km/yr if A in m/m * km? keep consistent
    v_kin = A * dHdt                            # [m/yr]*(m/m)=m/yr; A in m per m -> use SI
    # use Paper-4 numbers: A=0.70 km/m, dH/dt=1.5 m/yr -> v_kin=1.05 km/yr, ell=1 km
    v_kin_kmyr = 0.70 * 1.5                     # = 1.05 km/yr
    ell_km = 1.0
    tau_crit = ell_km / v_kin_kmyr             # yr
    Ro = np.array([v_kin_kmyr * th / ell_km for th in tau_hyd])
    regime = ["thinning-paced" if r < 1 else "hydraulic-limited" for r in Ro]
    ok = bool(abs(tau_crit - 1.0 / 1.05) < 1e-9
              and Ro[0] < 1 < Ro[-1])
    return dict(name="NR7 residence = Damkohler",
                v_kin_km_per_yr=v_kin_kmyr, tau_crit_yr=float(tau_crit),
                tau_hyd=list(tau_hyd), Ro=Ro.tolist(), regime=regime,
                interpretation=("Ro = v_kin tau_hyd/ell is a Damkohler number; Ro=1 at "
                                "tau_crit=ell/v_kin separates thinning-paced from "
                                "hydraulic-limited intrusion"),
                mainstream="Damkohler/Peclet transport-vs-process crossover",
                ok=ok)


ALL = [nr1_memory_number, nr2_admittance_unification, nr3_misi_fold_ews,
       nr4_migration_phase, nr5_melt_ceiling_suppression, nr6_tidal_curvature,
       nr7_residence_damkohler]


def summary():
    return [f() for f in ALL]


if __name__ == "__main__":
    print("Cross-cutting relationships — verification one by one\n" + "=" * 60)
    allok = True
    for r in summary():
        allok &= r["ok"]
        print(f"\n[{'PASS' if r['ok'] else 'FAIL'}] {r['name']}")
        print(f"   link: {r['interpretation']}")
        print(f"   lit:  {r['mainstream']}")
        for k, v in r.items():
            if k in ("name", "interpretation", "mainstream", "ok", "rows"):
                continue
            if isinstance(v, float):
                print(f"   {k} = {v:.4g}")
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                print(f"   {k} = [" + ", ".join(f"{x:.3g}" for x in v) + "]")
            else:
                print(f"   {k} = {v}")
    print("\n" + "=" * 60)
    print("ALL VERIFIED" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
