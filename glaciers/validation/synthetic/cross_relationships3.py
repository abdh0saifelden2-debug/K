r"""Cross-cutting relationships, batch 3 (NR11-NR12) — DERIVED here, each VERIFIED
one-by-one.  Continuation of ``cross_relationships.py`` (NR1-NR7) and
``cross_relationships2.py`` (NR8-NR10).

No external data; CPU only.  Each ``nrN_*`` function is a self-contained numerical
experiment returning a dict of decisive numbers plus a boolean ``ok``.

  NR11 The tidal velocity PHASE LAG measures the hydraulic residence time.  NR6 read
       the sliding-law *curvature* from the tidal 2f/1f amplitude ratio; the phase is
       the complementary, untapped channel.  If the subglacial hydrology low-passes
       the tidal/ocean head with a lumped impedance (the sec.7.3 / sec.K kernel
       K_hydraulic(t)=(1/RC)e^{-t/RC}, an analogy not a derivation), then the effective
       pressure -- and hence the surface velocity, u' ∝ s_N N' (phase-preserving) --
       lags the tide by phi(omega)=arctan(omega RC), with gain 1/sqrt(1+(omega RC)^2).
       So a tidal velocity phase-lag spectrum MEASURES RC (the hydraulic residence
       time, sec.9 gate #2) from surface velocity alone.  Gain and phase are a causal
       (minimum-phase) pair -- the Bode/Kramers-Kronig relation of NR9.

  NR12 ONE ice clock tau_d = kappa/Vbar^2 governs BOTH the ice memory-kernel cutoff
       (sec.B.2 G(t), short-time t^{-1/2} tail, exp cutoff at tau_d) AND the interface-
       coupling rolloff (sec.A.1 Lambda(omega), half-power at omega ~ 1/tau_d).  The
       coupling shape Lambda(omega)/Lambda(0) depends only on the product omega*tau_d
       (a universal curve), so the rolloff half-power frequency obeys omega_half*tau_d =
       const independent of Vbar.  Therefore measuring the rolloff (or kernel cutoff)
       INVERTS for the basal ablation velocity Vbar = sqrt(kappa/tau_d) -- a new
       method of measurement, the ice-side analogue of NR11's hydraulic RC.
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


_IC = _load("interface_coupling_number", "interface_coupling_number.py")
_IK = _load("ice_kernel_synthetic", "ice_kernel_synthetic.py")


# ---------------------------------------------------------------------------
# NR11 — tidal velocity phase lag measures the hydraulic residence time RC
# ---------------------------------------------------------------------------
def _rc_phase_measured(omega, RC, n_cyc=60, n_per=240):
    """Integrate the lumped hydraulic low-pass  RC x' = -x + cos(omega t)  (RK4) and
    measure the steady-state phase lag of x relative to the cos forcing."""
    steps = n_cyc * n_per
    dt = (2.0 * np.pi / omega) / n_per
    t = np.arange(steps) * dt

    def f(x, tt):
        return (-x + np.cos(omega * tt)) / RC

    x = 0.0
    xs = np.empty(steps)
    for i in range(steps):
        xs[i] = x                                    # state at t[i] (no off-by-dt)
        tt = t[i]
        k1 = f(x, tt)
        k2 = f(x + 0.5 * dt * k1, tt + 0.5 * dt)
        k3 = f(x + 0.5 * dt * k2, tt + 0.5 * dt)
        k4 = f(x + dt * k3, tt + dt)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    # use the last 50% (steady) over whole cycles; project onto cos/sin
    i0 = steps // 2
    i0 -= (steps - i0) % n_per                      # align to whole cycles
    seg = xs[i0:]
    tt = t[i0:]
    c = np.mean(seg * np.cos(omega * tt))
    s = np.mean(seg * np.sin(omega * tt))
    phi = np.arctan2(s, c)                           # x = R cos(omega t - phi)
    gain = 2.0 * np.hypot(c, s)
    return float(phi), float(gain)


def nr11_tidal_phase_hydraulic_rc(RC_true=0.30, s_N=-2.0,
                                  omegas=(0.5, 1.0, 2.0, 4.0, 8.0)):
    r"""Plant a hydraulic residence time RC; drive the lumped low-pass with tides at
    several omega; measure the velocity phase lag; recover RC = tan(phi)/omega.  Also
    checks the causal gain-phase (Bode) consistency.  s_N only rescales amplitude
    (phase-preserving), so the recovered RC is independent of it.
    """
    omegas = np.asarray(omegas, float)
    phis, gains, RC_rec = [], [], []
    for w in omegas:
        phi, gain = _rc_phase_measured(w, RC_true)
        phis.append(phi)
        gains.append(gain * abs(s_N))               # velocity gain (phase unchanged)
        RC_rec.append(np.tan(phi) / w)              # invert phi=arctan(omega RC)
    phis = np.asarray(phis); gains = np.asarray(gains); RC_rec = np.asarray(RC_rec)
    # theory
    phi_th = np.arctan(omegas * RC_true)
    gain_th = abs(s_N) / np.sqrt(1.0 + (omegas * RC_true) ** 2)
    phase_increases = bool(np.all(np.diff(phis) > 0))
    rc_relerr = float(np.max(np.abs(RC_rec - RC_true) / RC_true))
    phase_ok = bool(np.allclose(phis, phi_th, atol=0.02))
    gain_ok = bool(np.allclose(gains, gain_th, rtol=0.05))
    ok = bool(phase_increases and rc_relerr < 0.05 and phase_ok and gain_ok)
    return dict(name="NR11 tidal phase -> hydraulic residence RC",
                RC_true=float(RC_true), RC_recovered=RC_rec.tolist(),
                RC_max_relerr=rc_relerr,
                omega=omegas.tolist(), phase_lag_rad=phis.tolist(),
                phase_lag_theory=phi_th.tolist(),
                gain=gains.tolist(), gain_theory=gain_th.tolist(),
                phase_increases=phase_increases,
                interpretation=("tidal velocity phase lag phi(omega)=arctan(omega RC) "
                                "measures the hydraulic residence time RC from surface "
                                "velocity alone; gain & phase are a causal (Bode/KK) pair; "
                                "complements NR6 (amplitude curvature)"),
                mainstream="Gudmundsson 2007/2011 tidal ice-stream response; "
                           "Bode gain-phase / Kramers-Kronig; Roethlisberger/GlaDS hydrology",
                ok=ok)


# ---------------------------------------------------------------------------
# NR12 — one ice clock tau_d sets kernel cutoff + coupling rolloff; invert for Vbar
# ---------------------------------------------------------------------------
def _omega_half(theta_far, Vbar):
    """Half-power angular frequency where Lambda(omega) = Lambda(0)/2."""
    tau_d = _IC.KAPPA / Vbar ** 2
    w = np.geomspace(1e-4 / tau_d, 1e4 / tau_d, 4000)
    lam = _IC.coupling_number(w, theta_far, Vbar)
    lam0 = float(_IC.coupling_number(1e-12 / tau_d, theta_far, Vbar))
    target = lam0 / 2.0
    # Lambda decreases with omega; find first crossing by interpolation in log-omega
    below = np.where(lam <= target)[0]
    if below.size == 0:
        return float("nan"), tau_d
    i = below[0]
    if i == 0:
        return float(w[0]), tau_d
    lw = np.log(w[i - 1:i + 1]); ll = lam[i - 1:i + 1]
    w_half = float(np.exp(np.interp(target, ll[::-1], lw[::-1])))
    return w_half, tau_d


def nr12_ice_clock_inversion(theta_far=-1.0,
                             Vbars_m_per_yr=(0.03, 0.1, 0.3, 1.0)):
    r"""The ice clock tau_d=kappa/Vbar^2 governs both the sec.B.2 memory kernel (short
    t^{-1/2} tail, cutoff at tau_d) and the sec.A.1 coupling rolloff Lambda(omega).
    Because Lambda(omega)/Lambda(0) is a universal function of omega*tau_d, the
    half-power frequency satisfies omega_half*tau_d = const (Vbar-independent), so
    measuring the rolloff INVERTS for Vbar = sqrt(kappa/tau_d).  Verifies: (i)
    universality of omega_half*tau_d, (ii) Vbar recovery, (iii) kernel short-time
    slope ~ -1/2 (the diffusive t^{-1/2} tail).
    """
    Vb = np.array(Vbars_m_per_yr) / _IC.SEC_PER_YR     # m/s
    x_half, w_half, tau_d = [], [], []
    for V in Vb:
        wh, td = _omega_half(theta_far, V)
        w_half.append(wh); tau_d.append(td); x_half.append(wh * td)
    x_half = np.asarray(x_half); w_half = np.asarray(w_half); tau_d = np.asarray(tau_d)
    # (i) universality: omega_half * tau_d constant across Vbar
    x_rel_spread = float(np.std(x_half) / np.mean(x_half))
    # (ii) inversion using the calibrated universal constant c = <omega_half tau_d>
    c = float(np.median(x_half))
    tau_d_rec = c / w_half
    Vbar_rec = np.sqrt(_IC.KAPPA / tau_d_rec)
    vbar_relerr = float(np.max(np.abs(Vbar_rec - Vb) / Vb))
    # (iii) kernel short-time slope ~ -1/2
    V0 = Vb[1]
    td0 = _IC.KAPPA / V0 ** 2
    t = np.geomspace(1e-5 * td0, 1e-2 * td0, 40)
    G = np.abs(_IK.kernel_G(t, kappa=_IC.KAPPA, Vbar=V0, theta_far=theta_far,
                            rho_c=_IC.RHO_C))
    slope = float(np.polyfit(np.log(t), np.log(G), 1)[0])
    ok = bool(x_rel_spread < 1e-3 and vbar_relerr < 1e-3 and abs(slope + 0.5) < 0.05)
    return dict(name="NR12 ice clock tau_d: kernel cutoff = coupling rolloff -> Vbar",
                omega_half_times_tau_d=x_half.tolist(),
                universal_const_rel_spread=x_rel_spread,
                tau_d_yr=[float(td / _IC.SEC_PER_YR) for td in tau_d],
                Vbar_true_m_per_yr=list(Vbars_m_per_yr),
                Vbar_recovered_m_per_yr=(Vbar_rec * _IC.SEC_PER_YR).tolist(),
                Vbar_max_relerr=vbar_relerr,
                kernel_shorttime_slope=slope,
                interpretation=("one ice clock tau_d=kappa/Vbar^2 sets both the B.2 kernel "
                                "cutoff and the A.1 coupling half-power omega_half~1/tau_d; "
                                "omega_half*tau_d is universal, so the rolloff inverts for "
                                "Vbar=sqrt(kappa/tau_d) -- ice-side analogue of NR11's RC"),
                mainstream="moving-boundary Stefan diffusion; Carslaw & Jaeger; "
                           "Cuffey & Paterson 2010 ice properties",
                ok=ok)


ALL = [nr11_tidal_phase_hydraulic_rc, nr12_ice_clock_inversion]


def summary():
    return [f() for f in ALL]


if __name__ == "__main__":
    print("Cross-cutting relationships batch 3 (NR11-NR12) — verification one by one\n"
          + "=" * 64)
    allok = True
    for r in summary():
        allok &= r["ok"]
        print(f"\n[{'PASS' if r['ok'] else 'FAIL'}] {r['name']}")
        print(f"   link: {r['interpretation']}")
        print(f"   lit:  {r['mainstream']}")
        for k, v in r.items():
            if k in ("name", "interpretation", "mainstream", "ok"):
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
