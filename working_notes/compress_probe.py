"""Direction A: tidal phase-lag probe in the FINITE-c compressible solver (scratch).
 
Tests whether colored-FDT (OU backscatter, memory time tau_mem) produces a phase
lag in the turbulent heat flux Fturb=<v'theta'> that white-FDT (memoryless) cannot
-- specifically in the regime tau_mem ~ tau_adj = L/c, where the pressure has a
FINITE adjustment clock for the memory clock to beat (unlike the incompressible
solver, where instantaneous Leray projection gives tau_adj~0 and the test was null).
 
Closures: DNS(none) / Smagorinsky / white-FDT(bs_tau=0) / colored-FDT(bs_tau=tau_mem).
Sustains turbulence with a fixed solenoidal forcing; drives the tide via an
oscillating mean acceleration + a spatial standing wave; ensemble-averages the
complex Fourier coefficient over seeds so a systematic memory lag is separable
from stochastic-realization scatter.
 
usage: compress_probe.py [n] [c] [T_tide] [ncyc] [tau_mem] [mu] [nseed]
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

from compressible.ns import IsothermalCompressibleNS, NSState
 
OBS = ["ustar", "Fturb", "Fadv"]
 
 
def solenoidal_forcing(sp, kmax, amp, seed):
    """Fixed random divergence-free low-k acceleration field (sustains turbulence)."""
    rng = np.random.default_rng(seed)
    n = sp.n
    fx = np.zeros((n, n)); fy = np.zeros((n, n))
    x, y = sp.grid()
    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            kk = kx * kx + ky * ky
            if kk == 0 or kk > kmax * kmax:
                continue
            ph = rng.uniform(0, 2 * np.pi)
            a = rng.standard_normal()
            fx += a * np.cos(kx * x + ky * y + ph) * (-ky)   # curl of a scalar -> solenoidal
            fy += a * np.cos(kx * x + ky * y + ph) * (kx)
    nrm = np.sqrt(np.mean(fx**2 + fy**2)) + 1e-30
    return amp * fx / nrm, amp * fy / nrm
 
 
def observables(rho, mx, my, theta):
    u = mx / rho; v = my / rho
    um, vm, thm = u.mean(), v.mean(), theta.mean()
    ustar = float(np.sqrt(np.mean((u - um)**2 + (v - vm)**2)))
    Fadv = float(np.mean(v * theta))
    Fturb = float(np.mean((v - vm) * (theta - thm)))
    return dict(ustar=ustar, Fadv=Fadv, Fturb=Fturb)
 
 
def fourier_ab(t, m, w):
    t = np.asarray(t); m = np.asarray(m, float); m = m - m.mean()
    A = np.column_stack([np.sin(w*t), np.cos(w*t), np.sin(2*w*t), np.cos(2*w*t),
                         np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, m, rcond=None)
    a, b = coef[0], coef[1]
    rms = float(np.sqrt(np.mean(m**2)) + 1e-30)
    return float(a), float(b), float(np.hypot(a, b) / rms)
 
 
def run_one(cs, backscatter, bs_tau, *, n, c, mu, kappa, G, w, a0, A_t, k_tid,
            fkeep, t_spin, t_run, rec_dt, seed):
    s = IsothermalCompressibleNS(n, c, mu, 1.0, cfl=0.3, cs=cs,
                                 backscatter=backscatter, bs_tau=bs_tau,
                                 kappa=kappa, scalar_grad=G, seed=seed)
    sp = s.sp
    x, y = sp.grid()
    rho = np.ones((n, n))
    mx = np.zeros((n, n)); my = np.zeros((n, n))
    th = np.zeros((n, n))
    st = NSState(rho, mx, my, 0.0, theta=th)
    sinX = np.sin(k_tid * x)
    # spin up turbulence with the steady forcing only (no tide)
    t0 = 0.0
    while st.t < t_spin:
        dt = s.dt_cfl(st)
        s.step_closure(st, dt, fext=(fkeep[0], fkeep[1]))
        if not np.isfinite(st.rho).all():
            raise RuntimeError("diverged in spinup")
    t0 = st.t
    ts = []; rec = {k: [] for k in OBS}; eps_acc = []
    next_rec = st.t
    while st.t - t0 < t_run:
        dt = s.dt_cfl(st)
        ph = w * (st.t - t0)
        tide = a0 * np.sin(ph) + A_t * sinX * np.sin(ph)
        fx = fkeep[0] + tide
        fy = fkeep[1]
        s.step_closure(st, dt, fext=(fx, fy))
        if st.t >= next_rec:
            ts.append(st.t - t0)
            o = observables(st.rho, st.mx, st.my, st.theta)
            for k in OBS: rec[k].append(o[k])
            eps_acc.append(s._last_eps)
            next_rec += rec_dt
        if not np.isfinite(st.rho).all():
            raise RuntimeError("diverged in run")
    out = dict(eps=float(np.mean(eps_acc)) if eps_acc else 0.0,
               mach=float(np.max(np.sqrt((st.mx/st.rho)**2+(st.my/st.rho)**2))/c))
    for k in OBS:
        out[k] = fourier_ab(ts, rec[k], w)
    return out
 
 
def ensemble(cs, backscatter, bs_tau, seeds, **kw):
    res = {k: [] for k in OBS}; eps = []; mach = []
    for sd in seeds:
        ts = time.time()
        r = run_one(cs, backscatter, bs_tau, seed=sd, **kw)
        print(f"      seed {sd} done [{time.time()-ts:.0f}s] Fturb_coh={r['Fturb'][2]:.2f}", flush=True)
        eps.append(r["eps"]); mach.append(r["mach"])
        for k in OBS:
            res[k].append(r[k])
    summary = dict(eps=np.mean(eps), mach=np.mean(mach))
    for k in OBS:
        ab = np.array([(a, b) for a, b, _ in res[k]])
        coh = np.mean([cc for _, _, cc in res[k]])
        a_m, b_m = ab[:, 0].mean(), ab[:, 1].mean()
        delta_ens = -np.arctan2(b_m, a_m)
        per = np.array([-np.arctan2(b, a) for a, b in ab])
        d = np.angle(np.exp(1j * (per - delta_ens)))
        summary[k] = dict(deg=float(np.degrees(delta_ens)),
                          scatter_deg=float(np.degrees(np.std(d))), coh=float(coh),
                          amp=float(np.hypot(a_m, b_m)))
    return summary
 
 
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    c = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    T_tide = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    ncyc = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    tau_mem = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
    mu = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01
    nseed = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    tau_adj = (2.0 * np.pi) / c           # L/c
    if tau_mem is None:
        tau_mem = tau_adj                 # default: put memory clock AT the adjustment clock
    w = 2 * np.pi / T_tide
    kappa = mu; G = 1.0; cs = 0.16
    a0 = 0.25; A_t = 0.5; k_tid = 1.0
    t_spin = 4.0; t_run = ncyc * T_tide; rec_dt = T_tide / 120.0
    seeds = list(range(1, nseed + 1))
    sp_dummy = IsothermalCompressibleNS(n, c, mu).sp
    fkeep = solenoidal_forcing(sp_dummy, kmax=3, amp=0.15, seed=0)
    print(f"[compress probe] n={n} c={c} M~{1.0/c:.2f} mu={mu} T_tide={T_tide} ncyc={ncyc}", flush=True)
    print(f"  tau_adj=L/c={tau_adj:.3f}  t90~2.28L/c={2.28*tau_adj:.3f}  tau_mem={tau_mem:.3f}"
          f" (={tau_mem/tau_adj:.2f} tau_adj, ={tau_mem/T_tide:.2f}T)  seeds={seeds}", flush=True)
    kw = dict(n=n, c=c, mu=mu, kappa=kappa, G=G, w=w, a0=a0, A_t=A_t, k_tid=k_tid,
              fkeep=fkeep, t_spin=t_spin, t_run=t_run, rec_dt=rec_dt)
    variants = [(0.0, 0.0, 0.0, "DNS(none)"), (cs, 0.0, 0.0, "Smagorinsky"),
                (cs, 0.5, 0.0, "white-FDT"),
                (cs, 0.5, tau_mem, f"colored-FDT(t={tau_mem:.2f})")]
    summ = {}
    for cc, bs, bst, lab in variants:
        t0 = time.time()
        print(f"\n  === {lab} ...", flush=True)
        s = ensemble(cc, bs, bst, seeds, **kw); summ[lab] = s
        print(f"  === {lab}  [{time.time()-t0:.0f}s]  <eps>={s['eps']:.3e} Mmax~{s['mach']:.2f}", flush=True)
        for k in OBS:
            o = s[k]
            print(f"    {k:6s} phase={o['deg']:+7.1f}deg  seed-scatter=+/-{o['scatter_deg']:4.1f}deg"
                  f"  coh={o['coh']:.2f} amp={o['amp']:.3e}", flush=True)
    labs = [v[3] for v in variants]
    print("\n  DISCRIMINATOR -- Fturb phase, memory effect = colored - white:")
    for k in OBS:
        wf = summ["white-FDT"][k]["deg"]; cf = summ[labs[3]][k]["deg"]
        sc = max(summ["white-FDT"][k]["scatter_deg"], summ[labs[3]][k]["scatter_deg"])
        dd = np.degrees(np.angle(np.exp(1j*np.radians(cf-wf))))
        verdict = "SIGNAL" if abs(dd) > 2*sc else "within noise"
        print(f"    {k:6s}: Smag={summ['Smagorinsky'][k]['deg']:+6.1f}  white={wf:+6.1f}  "
              f"colored={cf:+6.1f}  -> colored-white={dd:+5.1f}deg (2*scatter={2*sc:.1f}) [{verdict}]", flush=True)
 
 
if __name__ == "__main__":
    main()
