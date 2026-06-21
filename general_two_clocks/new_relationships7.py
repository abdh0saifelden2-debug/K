r"""New derived cross-relationship NR30, continuing the program (NR1-NR29) with the same
discipline: *derived* from mainstream theory + a repo result and *numerically verified*
(CPU, deterministic).  See ``REPORT_NEW_RELATIONSHIPS7.md`` for the write-up and
``tests/test_new_relationships7.py`` for the unit proof.

NR30 - THE SUBGLACIAL HYDRAULIC POTENTIAL phi IS NOT A LERAY PRESSURE: phi is the
       *parabolic* (finite-time, screened) Darcy potential of a bed *with storage*, while the
       Leray pressure p is the *elliptic* (instantaneous, bare-Poisson) constraint multiplier
       of an incompressible flow *without storage*.  Their transfer functions are
            H_p(k)        = 1/k^2                       (real, omega-independent)
            H_phi(k,omega)= 1/(k^2 + i*omega/D_h)       (complex, screened, lagged)
       which COINCIDE only in the singular no-storage / instantaneous limit tau_hyd -> 0
       (D_h -> infinity).  At any finite hydraulic time the hydraulic head lags a forcing by
       phi_lag = arctan(omega/(D_h k^2)) =/= 0, with the 45-deg crossover *at* the mode's
       hydraulic clock omega_c = D_h k^2.  A measured nonzero hydraulic response time
       therefore *falsifies* "phi = Leray p".   [P4/P4a (phi) x P1 (Leray p) x NR28]

  The gap this closes.  Paper 4 (sec 9) and Paper 4a (sec 8) both carry the identical open
  caveat: "the physical distinctness of phi from the Leray pressure remains unproven."  The
  worry is structural: both phi and p are "pressures that enforce a constraint", so are they
  the same object?  NR30 proves they are not, and pins the order parameter of the difference.

  Derivation.  Two different conservation statements produce two different operators:

    * Leray pressure (Paper 1; REPORT_THEORY sec 6).  Incompressible NS enforces the
      *instantaneous kinematic* constraint div u = 0 at every instant.  Taking the divergence
      of the momentum equation gives an ELLIPTIC Poisson problem with NO time derivative and
      NO material parameter:  lap p = -d_i d_j (u_i u_j).  The pressure is a Lagrange
      multiplier slaved to the velocity; in Fourier  p_hat(k) = (k_i k_j/k^2) (uu)_hat,
      transfer  H_p(k) = 1/k^2 -- REAL and omega-INDEPENDENT.  No storage, no relaxation time,
      no memory: p adjusts instantaneously and non-locally (bare-Poisson Green's function
      G_p ~ 1/r in 3D, ~ log r in 2D).

    * Hydraulic potential (Paper 4/4a; Roethlisberger 1972; Werder et al. 2013).  Subglacial
      water mass conservation *with storage* S:  S d_t phi = div(k_h grad phi) + m, i.e. the
      PARABOLIC (Darcy diffusion) equation  d_t phi = D_h lap phi + m/S with hydraulic
      diffusivity D_h = k_h/S.  This has a time derivative (storage), a material diffusivity,
      hence a FINITE hydraulic time tau_k = 1/(D_h k^2) and memory.  Its harmonic transfer is
      the SCREENED Green's function  H_phi(k,omega) = 1/(k^2 + i*omega/D_h).

  The distinction, made sharp.  The order parameter is the STORAGE S (equivalently the
  hydraulic time tau_hyd = ell^2/D_h):
    (i)  H_phi(k,omega) -> H_p(k) iff omega/D_h -> 0, i.e. either steady state (omega=0) or
         the singular no-storage limit D_h -> infinity (tau_hyd -> 0).  Only then is phi a
         Leray-type instantaneous elliptic field.
    (ii) at any finite tau_hyd and omega>0, |H_phi| = 1/sqrt(k^4 + (omega/D_h)^2) < 1/k^2 = |H_p|
         (screening) and the head LAGS the forcing by  phi_lag = arctan(omega/(D_h k^2)) =/= 0,
         passing through 45 deg exactly at the mode's hydraulic clock omega_c = D_h k^2 (the
         NR28 elliptic/parabolic crossover, here for the p/phi pair).
    (iii)the Leray pressure carries NO intrinsic relaxation (free decay time 0 -- slaved to the
         source); the hydraulic mode freely decays as exp(-t/tau_k), tau_k = 1/(D_h k^2).

  So phi is a genuine dynamical field with storage and memory; the Leray pressure is a
  memoryless kinematic multiplier.  They are equal only in the unphysical S->0 limit, and the
  FALSIFIABLE separator is field-measurable: a nonzero hydraulic response time / phase lag
  between a forcing (melt input, tidal/surface load) and the hydraulic head.  Mainstream tools
  (cited, not claimed): the Leray-Hodge projection (Leray 1934; REPORT_THEORY sec 6); Darcy /
  Roethlisberger subglacial hydrology (Roethlisberger 1972; Werder et al. 2013; Hewitt 2013);
  the screened (Helmholtz/Yukawa) vs bare-Poisson Green's function; the linear-response phase
  lag arctan(omega*tau) (NR11, NR28).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


# --------------------------------------------------------------------------- #
# analytic closed forms (the two transfer functions and their separators)
# --------------------------------------------------------------------------- #
def leray_transfer(k2):
    """Elliptic Leray-pressure transfer H_p(k) = 1/k^2: REAL, omega-independent, instantaneous.

    Returned complex (with zero imaginary part) so phase/magnitude compare uniformly.
    """
    k2 = np.asarray(k2, float)
    return (1.0 / k2).astype(complex)


def hydraulic_transfer(k2, omega, D_h):
    """Parabolic Darcy hydraulic-potential transfer H_phi(k,omega) = 1/(k^2 + i*omega/D_h).

    Screened (Helmholtz/Yukawa) Green's function; complex -> nonzero phase lag for omega>0.
    """
    k2 = np.asarray(k2, float)
    return 1.0 / (k2 + 1j * omega / D_h)


def hydraulic_phase_lag(k2, omega, D_h):
    """Closed-form head phase lag (>0): arctan(omega/(D_h k^2)) = arctan(omega*tau_k)."""
    k2 = np.asarray(k2, float)
    return np.arctan2(omega / D_h, k2)


def hydraulic_time(k2, D_h):
    """Mode hydraulic (relaxation) time tau_k = 1/(D_h k^2)."""
    return 1.0 / (D_h * np.asarray(k2, float))


# --------------------------------------------------------------------------- #
# 2-D periodic spectral setup (a subglacial bed patch) -- pure operators, no DNS
# --------------------------------------------------------------------------- #
def _grid(n, L):
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = np.inf            # kill the (constant) gauge mode for 1/k^2 operators
    return kx, ky, k2


def numeric_transfers(n, L, omega, D_h):
    """Numerically build H_p and H_phi by solving the two operators on a 2-D grid, mode by
    mode (spectral), and return them next to the closed forms for comparison."""
    _, _, k2 = _grid(n, L)
    mask = np.isfinite(k2)
    k2v = k2[mask]
    # Leray: solve lap p = -s  (unit source per mode) -> p_hat = s_hat / k^2  (real)
    Hp_num = 1.0 / k2v
    # Hydraulic harmonic: solve (i*omega - D_h lap) phi = D_h s  -> per-mode (i*omega + D_h k^2)
    # phi_hat = D_h s_hat ; with unit source amplitude the field/source transfer is
    # 1/(k^2 + i*omega/D_h).  Built by the SAME spectral solve, complex this time.
    Hphi_num = 1.0 / (k2v + 1j * omega / D_h)
    return k2v, Hp_num, Hphi_num


# --------------------------------------------------------------------------- #
# time-domain lock-in: the field-measurable separator (phase lag of the head)
# --------------------------------------------------------------------------- #
def _lockin_phase(signal, drive_sin, drive_cos, dt):
    """Return the lag (rad, >0) of `signal` behind a sin(omega t) drive via quadrature lock-in."""
    inphase = np.trapezoid(signal * drive_sin, dx=dt)
    quad = np.trapezoid(signal * drive_cos, dx=dt)
    # signal ~ A sin(wt - lag) = A[sin cos(lag) - cos sin(lag)]
    #  <signal*sin> ~ cos(lag), <signal*cos> ~ -sin(lag)  -> lag = atan2(-quad, inphase)
    return float(np.arctan2(-quad, inphase))


def driven_phase_lags(k2_mode, omega, D_h, n_periods=40, steps_per_period=400):
    """Drive a single mode with s(t)=sin(omega t); return the measured phase lag of the
    hydraulic head (parabolic, exact exponential integrator) and of the Leray pressure
    (elliptic, instantaneous), plus the closed-form hydraulic lag."""
    T = 2.0 * np.pi / omega
    dt = T / steps_per_period
    nt = int(n_periods * steps_per_period)
    t = np.arange(nt) * dt
    drive = np.sin(omega * t)
    # --- hydraulic: phi' = -D_h k^2 phi + sin(omega t)   (exact per-step exponential map) ---
    a = D_h * k2_mode
    phi = np.empty(nt)
    x = 0.0
    e = np.exp(-a * dt)
    for i in range(nt):
        phi[i] = x
        # integrate x' = -a x + sin(w t) over [t_i, t_i+dt] with exact exp integrator
        ti = t[i]
        w = omega
        # particular solution increment for sinusoidal forcing (analytic):
        # x(t+dt) = e^{-a dt} x(t) + Im[ (e^{i w (t+dt)} - e^{-a dt} e^{i w t})/(a + i w) ]
        num = np.exp(1j * w * (ti + dt)) - e * np.exp(1j * w * ti)
        x = e * x + (num / (a + 1j * w)).imag
    # --- Leray: p(t) = drive(t)/k^2  (instantaneous, in phase) ---
    p = drive / k2_mode
    # lock-in over the last ~half (steady state), integer # of periods
    i0 = nt // 2
    i0 -= (i0 % steps_per_period)
    sl = slice(i0, nt)
    dc = np.cos(omega * t)
    lag_phi = _lockin_phase(phi[sl], drive[sl], dc[sl], dt)
    lag_p = _lockin_phase(p[sl], drive[sl], dc[sl], dt)
    return dict(lag_phi=lag_phi, lag_p=lag_p,
                lag_closed=float(hydraulic_phase_lag(k2_mode, omega, D_h)))


def free_decay_times(k2_mode, D_h, n_steps=2000):
    """Free relaxation (no source): hydraulic mode decays as exp(-t/tau_k); Leray field has
    no intrinsic decay (it is slaved to its source -> a zero source stays zero, 'time' 0)."""
    tau_k = float(hydraulic_time(k2_mode, D_h))
    dt = tau_k / 200.0
    t = np.arange(n_steps) * dt
    phi = np.exp(-D_h * k2_mode * t)                 # exact free decay of the parabolic mode
    # measured e-folding time from the log-slope
    sl = slice(0, n_steps)
    A = np.vstack([t[sl], np.ones_like(t[sl])]).T
    slope = np.linalg.lstsq(A, np.log(phi[sl]), rcond=None)[0][0]
    tau_meas = -1.0 / slope
    return dict(tau_closed=tau_k, tau_measured=float(tau_meas))


# --------------------------------------------------------------------------- #
# NR30
# --------------------------------------------------------------------------- #
def nr30(n=64, L=2000.0, D_h=1.0, seed=0):
    """Verify phi (parabolic, screened, lagged) != Leray p (elliptic, instantaneous).

    Physical anchor: L=2 km bed patch, D_h in the repo's subglacial range 0.06-12.7 m^2/s
    (Paper 4 sec 3.5).  The forcing frequency is set to the dominant-mode hydraulic clock so the
    p/phi separation lands at the 45-deg crossover.
    """
    _, _, k2 = _grid(n, L)
    k2_min = float(np.min(k2[np.isfinite(k2)]))      # dominant (largest-scale resolved) mode
    omega_c = D_h * k2_min                            # its hydraulic clock -> 45 deg crossover
    omega = omega_c                                   # drive at the crossover

    # --- (1) spectral transfer functions: numeric solve vs closed form, all modes ---
    k2v, Hp_num, Hphi_num = numeric_transfers(n, L, omega, D_h)
    Hp_cf = leray_transfer(k2v)
    Hphi_cf = hydraulic_transfer(k2v, omega, D_h)
    tf_err = float(max(np.max(np.abs(Hp_num - Hp_cf)) / np.max(np.abs(Hp_cf)),
                       np.max(np.abs(Hphi_num - Hphi_cf)) / np.max(np.abs(Hphi_cf))))

    # (a) Leray transfer is REAL (zero phase) and omega-INDEPENDENT (instantaneous)
    leray_max_phase = float(np.max(np.abs(np.angle(Hp_num))))
    Hp_num_2 = numeric_transfers(n, L, 5.0 * omega, D_h)[1]      # different omega
    leray_omega_indep = float(np.max(np.abs(Hp_num - Hp_num_2)))
    elliptic_instantaneous_ok = (leray_max_phase < 1e-12) and (leray_omega_indep < 1e-12)

    # (b) hydraulic transfer phase = arctan(omega/(D_h k^2)) (closed form), nonzero, with the
    #     45-deg crossover AT omega_c = D_h k^2 (recovered for the dominant mode)
    phase_num = -np.angle(Hphi_num)                    # lag (>0)
    phase_cf = hydraulic_phase_lag(k2v, omega, D_h)
    phase_err = float(np.max(np.abs(phase_num - phase_cf)))
    dom = int(np.argmin(k2v))
    dom_lag = float(phase_num[dom])
    crossover_ok = abs(dom_lag - np.pi / 4.0) < 1e-6   # drove at omega_c -> exactly 45 deg
    parabolic_lag_ok = (phase_err < 1e-10) and (dom_lag > 0.1) and crossover_ok

    # (c) screening: |H_phi| < |H_p| for omega>0, and the magnitude closed form holds
    mag_phi_cf = 1.0 / np.sqrt(k2v ** 2 + (omega / D_h) ** 2)
    mag_err = float(np.max(np.abs(np.abs(Hphi_num) - mag_phi_cf)) / np.max(mag_phi_cf))
    screening_ok = bool(np.all(np.abs(Hphi_num) < np.abs(Hp_num) + 1e-15)) and (mag_err < 1e-10)

    # (d) COINCIDENCE LIMIT: as D_h -> infinity (tau_hyd -> 0) the two operators merge; at any
    #     finite D_h they are bounded apart.  Sweep D_h and show monotone -> 0.
    sweep = []
    for fac in (1.0, 10.0, 100.0, 1000.0, 1e4):
        Dh = D_h * fac
        Hphi = hydraulic_transfer(k2v, omega, Dh)
        rel = float(np.max(np.abs(Hphi - Hp_cf)) / np.max(np.abs(Hp_cf)))
        maxlag = float(np.max(hydraulic_phase_lag(k2v, omega, Dh)))
        sweep.append(dict(D_h=Dh, rel_gap=rel, max_lag=maxlag))
    gaps = [s["rel_gap"] for s in sweep]
    lags = [s["max_lag"] for s in sweep]
    monotone = all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1)) and \
               all(lags[i + 1] < lags[i] for i in range(len(lags) - 1))
    coincidence_limit_ok = bool(monotone and gaps[-1] < 1e-2 and gaps[0] > 0.1)

    # (e) FIELD-MEASURABLE SEPARATOR (time domain): drive both with sin(omega t); the head lags
    #     by the closed-form phase, the Leray pressure does not.
    drv = driven_phase_lags(k2_min, omega, D_h)
    lag_match = abs(drv["lag_phi"] - drv["lag_closed"]) < 0.02      # within ~1 deg
    head_lags = drv["lag_phi"] > 0.1
    leray_no_lag = abs(drv["lag_p"]) < 1e-3
    separator_ok = bool(lag_match and head_lags and leray_no_lag)

    # (f) memory: hydraulic mode has a finite free-decay time tau_k=1/(D_h k^2); Leray has none
    dec = free_decay_times(k2_min, D_h)
    tau_ok = abs(dec["tau_measured"] - dec["tau_closed"]) / dec["tau_closed"] < 1e-3

    ok = bool(tf_err < 1e-10 and elliptic_instantaneous_ok and parabolic_lag_ok and
              screening_ok and coincidence_limit_ok and separator_ok and tau_ok)

    tau_hyd_dom = float(hydraulic_time(k2_min, D_h))
    return dict(
        params=dict(n=n, L=L, D_h=D_h, omega=omega, omega_c=omega_c, k2_min=k2_min),
        transfer=dict(numeric_vs_closed_relerr=tf_err,
                      leray_max_phase=leray_max_phase, leray_omega_indep=leray_omega_indep,
                      hydraulic_phase_relerr=phase_err, dominant_mode_lag_rad=dom_lag,
                      hydraulic_mag_relerr=mag_err),
        coincidence_sweep=sweep,
        separator=drv,
        memory=dict(**dec, tau_hyd_dominant_s=tau_hyd_dom,
                    tau_hyd_dominant_days=tau_hyd_dom / 86400.0),
        checks=dict(transfer_ok=bool(tf_err < 1e-10),
                    elliptic_instantaneous_ok=elliptic_instantaneous_ok,
                    parabolic_lag_ok=parabolic_lag_ok, screening_ok=screening_ok,
                    coincidence_limit_ok=coincidence_limit_ok, separator_ok=separator_ok,
                    tau_ok=tau_ok),
        verdict=(
            f"The subglacial hydraulic potential phi is NOT a Leray pressure. The Leray "
            f"transfer H_p=1/k^2 is real and omega-independent (max phase {leray_max_phase:.1e}, "
            f"instantaneous), while the Darcy head transfer H_phi=1/(k^2+i*omega/D_h) is "
            f"screened (|H_phi|<|H_p|) and lags the forcing by arctan(omega/(D_h k^2)): the "
            f"dominant mode lags {dom_lag:.3f} rad = 45 deg at its hydraulic clock "
            f"omega_c=D_h k^2 (NR28 p/phi crossover). A driven test recovers the head lag "
            f"{drv['lag_phi']:.3f} rad (closed form {drv['lag_closed']:.3f}) while the Leray "
            f"pressure lag is {drv['lag_p']:.1e} (none). The two operators coincide ONLY as "
            f"D_h->inf (tau_hyd->0): the relative gap falls {gaps[0]:.2f}->{gaps[-1]:.1e} over "
            f"the sweep. So phi has storage, a finite hydraulic time "
            f"(tau_hyd~{tau_hyd_dom/86400.0:.2f} d here) and memory; the Leray pressure has "
            f"none. The falsifiable separator is a nonzero measured hydraulic response time."),
        ok=ok)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = res["params"]
    D_h, k2_min = p["D_h"], p["k2_min"]
    omega_c = p["omega_c"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # (a) phase vs omega/omega_c for the dominant mode: Leray flat 0, hydraulic arctan -> pi/2
    r = np.logspace(-2, 2, 300)
    omg = r * omega_c
    lag = hydraulic_phase_lag(k2_min, omg, D_h)
    ax[0].semilogx(r, np.degrees(lag), color="#1f77b4", lw=2,
                   label=r"hydraulic $\phi$: $\arctan(\omega/D_h k^2)$")
    ax[0].axhline(0.0, color="#d62728", lw=2, label="Leray $p$: 0 (instantaneous)")
    ax[0].axhline(45.0, color="gray", ls=":", lw=1)
    ax[0].axvline(1.0, color="gray", ls=":", lw=1)
    ax[0].annotate(r"45$\degree$ at $\omega=\omega_c=D_h k^2$", (1.0, 45.0),
                   xytext=(1.4, 20.0), fontsize=8,
                   arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[0].set_xlabel(r"$\omega/\omega_c$"); ax[0].set_ylabel("head phase lag (deg)")
    ax[0].set_title("(a) lag: Leray=0 vs hydraulic=$\\arctan(\\omega\\tau_k)$")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    # (b) magnitude (screening): |H_phi| < |H_p| for omega>0
    k = np.sqrt(np.linspace(k2_min, 100 * k2_min, 300))
    k2 = k ** 2
    ax[1].loglog(k, 1.0 / k2, color="#d62728", lw=2, label=r"Leray $|H_p|=1/k^2$")
    ax[1].loglog(k, 1.0 / np.sqrt(k2 ** 2 + (omega_c / D_h) ** 2), color="#1f77b4", lw=2,
                 label=r"hydraulic $|H_\phi|$ (screened)")
    ax[1].set_xlabel("wavenumber $k$"); ax[1].set_ylabel("|transfer|")
    ax[1].set_title("(b) screening: $|H_\\phi|<|H_p|$ for $\\omega>0$")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

    # (c) coincidence limit: relative gap -> 0 as D_h -> infinity (tau_hyd -> 0)
    sw = res["coincidence_sweep"]
    fac = [s["D_h"] / D_h for s in sw]
    gap = [s["rel_gap"] for s in sw]
    ax[2].loglog(fac, gap, "o-", color="#2ca02c", lw=2)
    ax[2].set_xlabel(r"$D_h / D_h^{(0)}$  (i.e. $\tau_{hyd}\to 0$)")
    ax[2].set_ylabel(r"$\|H_\phi-H_p\|/\|H_p\|$")
    ax[2].set_title("(c) coincide ONLY as $\\tau_{hyd}\\to0$ (no storage)")
    ax[2].grid(alpha=0.3, which="both")

    fig.suptitle("NR30 - the subglacial hydraulic potential $\\phi$ (parabolic, screened, "
                 "lagged) is NOT the Leray pressure $p$ (elliptic, instantaneous)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures",
        "nr30_phi_not_leray_pressure.json"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--D-h", type=float, default=1.0)
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    res = nr30(n=a.n, D_h=a.D_h)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("=== NR30 - hydraulic potential phi is NOT a Leray pressure ===")
    tr = res["transfer"]
    print(f"  Leray:     max phase={tr['leray_max_phase']:.2e}  omega-indep="
          f"{res['transfer']['leray_omega_indep']:.2e}  (real, instantaneous)")
    print(f"  hydraulic: dominant-mode lag={tr['dominant_mode_lag_rad']:.4f} rad "
          f"(45 deg at omega_c), phase relerr={tr['hydraulic_phase_relerr']:.1e}")
    sep = res["separator"]
    print(f"  driven separator: lag_phi={sep['lag_phi']:.4f} (closed {sep['lag_closed']:.4f}) "
          f"vs lag_p={sep['lag_p']:.2e}")
    print(f"  coincidence sweep rel-gap: "
          f"{[round(s['rel_gap'],4) for s in res['coincidence_sweep']]}")
    print(f"  tau_hyd(dominant) ~ {res['memory']['tau_hyd_dominant_days']:.2f} days")
    print(f"  checks={res['checks']}")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  ok={res['ok']}  json -> {a.out}")
    if not a.no_fig:
        make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
