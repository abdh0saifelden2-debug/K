"""Tidal melt phase-lag probe v3 (scratch, not committed).
 
Structural discriminator (NOT a tidal-cavity model). Drives an oscillatory tide
U_target(t)=U0+dU sin(wt) + spatial standing-wave force in a periodic penalized
cavity; measures the phase of the turbulent vertical heat flux Fturb=<v'theta'>
(near the ice base) relative to the forcing, for four closures:
  none / smagorinsky / white-FDT (bs_tau=0) / colored-FDT (bs_tau=tau_mem),
ENSEMBLE-AVERAGED over seeds so a systematic memory lag can be told apart from
chaotic realization scatter.
 
usage: tidal_probe.py [n] [T_tide] [ncyc] [tau_mem] [nu] [f_amp] [nseed]
"""
import sys, time
import numpy as np
import os
# repo reorg: make sibling domain folders importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("general_two_clocks", "atmosphere", "glaciers", "ocean"):
    _p = os.path.join(_REPO_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _d, _p

from subglacial.flow import SubglacialConfig, SubglacialFlow
 
OBS = ["ustar", "Fturb", "Fadv"]
 
 
def near_ice_band(f):
    return f.fluid & (f.Y > f.cfg.ice_base - 0.6) & (f.Y < f.cfg.ice_base)
 
 
def observables(f, band):
    u, v, th = f.u, f.v, f.theta
    ub, vb, thb = u[band], v[band], th[band]
    um, vm, thm = ub.mean(), vb.mean(), thb.mean()
    ustar = float(np.sqrt(np.mean((ub - um) ** 2 + (vb - vm) ** 2)))
    Fadv = float(np.mean(vb * thb))
    Fturb = float(np.mean((vb - vm) * (thb - thm)))
    return dict(ustar=ustar, Fadv=Fadv, Fturb=Fturb)
 
 
def fourier_ab(t, m, w):
    t = np.asarray(t); m = np.asarray(m, float); m = m - m.mean()
    A = np.column_stack([np.sin(w*t), np.cos(w*t), np.sin(2*w*t), np.cos(2*w*t),
                         np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, m, rcond=None)
    a, b = coef[0], coef[1]
    rms = float(np.sqrt(np.mean(m**2)) + 1e-30)
    return float(a), float(b), float(np.hypot(a, b) / rms)
 
 
def run_one(sgs, bs_tau, w, dU, A_t, k_tid, n, nu, f_amp, spinup, nsteps, rec_every, seed):
    cfg = SubglacialConfig(n=n, nu=nu, kappa=nu, sgs=sgs, bs_tau=bs_tau,
                           backscatter=0.6, f_amp=f_amp, k_f=12.0, U0=1.0, seed=seed)
    f = SubglacialFlow(cfg)
    f.run(spinup, ramp=max(1, spinup // 3))
    t0 = f.t
    band = near_ice_band(f)
    sinX = np.sin(k_tid * f.X) * f.fluid
    rec = {k: [] for k in OBS}; ts = []
    eps_acc = []; ke_acc = []
    for s in range(nsteps):
        ph = w * (f.t - t0)
        fx = A_t * sinX * np.sin(ph)
        eps = f.step(Utarget=cfg.U0 + dU * np.sin(ph), fext=(fx, np.zeros_like(fx)))
        if s % rec_every == 0:
            ts.append(f.t - t0); o = observables(f, band)
            for k in OBS: rec[k].append(o[k])
            eps_acc.append(eps); ke_acc.append(f.kinetic_energy())
    out = dict(eps=float(np.mean(eps_acc)), ke=float(np.mean(ke_acc)))
    for k in OBS:
        a, b, coh = fourier_ab(ts, rec[k], w)
        out[k] = (a, b, coh)
    return out
 
 
def ensemble(sgs, bs_tau, seeds, **kw):
    """Average the complex Fourier coefficient across seeds; phase of the mean is
    the systematic phase, and the per-seed phase scatter is the noise floor."""
    res = {k: [] for k in OBS}; eps = []; ke = []
    for sd in seeds:
        r = run_one(sgs, bs_tau, seed=sd, **kw)
        eps.append(r["eps"]); ke.append(r["ke"])
        for k in OBS:
            res[k].append(r[k])
    summary = dict(eps=np.mean(eps), ke=np.mean(ke))
    for k in OBS:
        ab = np.array([(a, b) for a, b, _ in res[k]])
        coh = np.mean([c for _, _, c in res[k]])
        a_m, b_m = ab[:, 0].mean(), ab[:, 1].mean()
        delta_ens = -np.arctan2(b_m, a_m)
        per = np.array([-np.arctan2(b, a) for a, b in ab])
        # wrapped scatter about the ensemble mean
        d = np.angle(np.exp(1j * (per - delta_ens)))
        summary[k] = dict(delta=float(delta_ens), deg=float(np.degrees(delta_ens)),
                          scatter_deg=float(np.degrees(np.std(d))), coh=float(coh),
                          amp=float(np.hypot(a_m, b_m)))
    return summary
 
 
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    T_tide = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    ncyc = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    tau_mem = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    nu = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5e-4
    f_amp = float(sys.argv[6]) if len(sys.argv) > 6 else 12.0
    nseed = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    dt = 3e-4; w = 2*np.pi / T_tide; dU = 0.6; A_t = 3.0; k_tid = 1.0
    spinup = 3000; nsteps = int(ncyc * T_tide / dt); rec_every = max(1, nsteps // 500)
    seeds = list(range(1, nseed + 1))
    kw = dict(w=w, dU=dU, A_t=A_t, k_tid=k_tid, n=n, nu=nu, f_amp=f_amp,
              spinup=spinup, nsteps=nsteps, rec_every=rec_every)
    print(f"[probe v3] n={n} nu={nu:.1e} f_amp={f_amp} T_tide={T_tide} ncyc={ncyc} "
          f"nsteps={nsteps} tau_mem={tau_mem}(={tau_mem/T_tide:.2f}T) seeds={seeds}")
    variants = [("none", 0.0, "DNS(none)"), ("smagorinsky", 0.0, "Smagorinsky"),
                ("backscatter", 0.0, "white-FDT"),
                ("backscatter", tau_mem, f"colored-FDT(t={tau_mem})")]
    summ = {}
    for sgs, bst, lab in variants:
        t0 = time.time()
        s = ensemble(sgs, bst, seeds, **kw); summ[lab] = s
        print(f"\n  === {lab}  [{time.time()-t0:.0f}s]  <eps>={s['eps']:.3e} <KE>={s['ke']:.3e}")
        for k in OBS:
            o = s[k]
            print(f"    {k:6s} phase={o['deg']:+7.1f}deg  seed-scatter=+/-{o['scatter_deg']:4.1f}deg"
                  f"  coh={o['coh']:.2f} amp={o['amp']:.3e}")
    print("\n  DISCRIMINATOR -- Fturb phase (deg), memory effect = colored - white:")
    labs = [v[2] for v in variants]
    for k in OBS:
        wf = summ["white-FDT"][k]["deg"]; cf = summ[labs[3]][k]["deg"]
        sc = max(summ["white-FDT"][k]["scatter_deg"], summ[labs[3]][k]["scatter_deg"])
        dd = np.degrees(np.angle(np.exp(1j*np.radians(cf-wf))))
        verdict = "SIGNAL" if abs(dd) > 2*sc else "within noise"
        print(f"    {k:6s}: Smag={summ['Smagorinsky'][k]['deg']:+6.1f}  white={wf:+6.1f}  "
              f"colored={cf:+6.1f}  -> colored-white={dd:+5.1f}deg (2*scatter={2*sc:.1f}) [{verdict}]")
 
 
if __name__ == "__main__":
    main()
