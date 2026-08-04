"""P1 a-posteriori closure test (paper1 sec 8.1, the named next step).

The committed benchmark (run_closure.py) is *a-priori*: it scores closures on a
single frozen filtered-DNS snapshot.  The implemented projected_fdt_force MEASURES
its eddy viscosity from the true transfer, so it cannot run a-posteriori.  This
harness instead runs a genuinely *predictive*, self-contained coarse LES forward in
time under each closure and compares the long-time statistics to the
spectrally-filtered DNS truth -- the test paper1 defers to sec 8.1.

Closures (all predictive: use only resolved data, never the DNS truth):
  none   : molecular viscosity + drag only (under-resolved control)
  smag   : Smagorinsky (closure/sgs.smagorinsky_force), curl -> vorticity tendency
  specEV : positive spectral eddy viscosity nu_t(k) ~ sqrt(E(kc)/kc)*(plateau+cusp)
           (Kraichnan 1976 / Chollet-Lesieur 1981 structure; paper1 Eq. 8)
  specEV+bs : specEV dissipation + an FDT-tied, divergence-free stochastic backscatter
           injecting a fraction of the near-cutoff dissipated energy (Leith 1990;
           Mason & Thomson 1992) -- the predictive analogue of the projected-FDT model.

Diagnostics vs filtered DNS (time-averaged over snapshots):
  - resolved energy spectrum E(k), 1..kc  (overall + low-k inverse-cascade band)
  - total resolved kinetic energy and enstrophy
  - stability (blow-up guard)

CPU only, no downloaded data.
"""
from __future__ import annotations

import argparse
import numpy as np

from compressible.ns import Spectral2D
from closure.dns2d import Vorticity2D
from closure.sgs import smagorinsky_force


# ---------------------------------------------------------------------------
# spectral helpers (shell binning, energy spectrum from vorticity)
# ---------------------------------------------------------------------------
def shell_idx(sp):
    return np.round(np.sqrt(sp.k2)).astype(int)


def vel_from_w(sp, w_h):
    psi_h = w_h * sp.k2_inv
    u = sp.ifft(1j * sp.ky * psi_h)
    v = sp.ifft(-1j * sp.kx * psi_h)
    return u, v


def energy_spectrum_from_w(sp, w_h, kc):
    """Shell-summed resolved kinetic-energy spectrum E(k) for k=0..kc."""
    psi_h = w_h * sp.k2_inv
    uh = 1j * sp.ky * psi_h
    vh = -1j * sp.kx * psi_h
    P = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2) / (sp.n ** 4)
    idx = shell_idx(sp)
    E = np.zeros(int(kc) + 1)
    for k in range(int(kc) + 1):
        E[k] = P[idx == k].sum()
    return E


def ke_ens_from_w(sp, w_h, kc):
    """Resolved KE and enstrophy from the |k|<=kc band (real-space, filtered)."""
    mask = (np.sqrt(sp.k2) <= kc)
    wf = w_h * mask
    u, v = vel_from_w(sp, wf)
    ke = 0.5 * float(np.mean(u * u + v * v))
    w = sp.ifft(wf)
    ens = 0.5 * float(np.mean(w * w))
    return ke, ens


# ---------------------------------------------------------------------------
# forcing (same construction/statistics as Vorticity2D, replicated per grid)
# ---------------------------------------------------------------------------
class Forcing:
    def __init__(self, sp, k_f=24.0, f_amp=2.5, seed=0):
        kr = np.sqrt(sp.k2)
        self.ring = ((kr >= k_f - 1.0) & (kr <= k_f + 1.0)).astype(float)
        self.f_amp = f_amp
        self.n = sp.n
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        white = self.rng.standard_normal((self.n, self.n))
        f_h = np.fft.fft2(white) * self.ring
        f_phys = np.real(np.fft.ifft2(f_h))
        rms = float(np.sqrt(np.mean(f_phys ** 2))) + 1e-30
        return f_h * (self.f_amp / rms)


# ---------------------------------------------------------------------------
# predictive spectral eddy viscosity (paper1 Eq. 8 structure, positive form)
# ---------------------------------------------------------------------------
def spectral_eddy_visc(sp, w_h, kc, c_ev=0.30, cusp=8.0, p=4, plateau=1.0):
    """nu_t(k) = c_ev*sqrt(E(kc_shell)/kc)*(plateau + cusp*(k/kc)^p), per-shell -> 2D map.
    Predictive: E(kc_shell) read from the current resolved field only.
    plateau=0 -> pure cusp (dissipation concentrated at the cutoff, resolved scales
    left untouched -- the physically-correct 2-D limit)."""
    E = energy_spectrum_from_w(sp, w_h, kc)
    kc_i = int(kc)
    base = c_ev * np.sqrt(max(E[kc_i], 1e-30) / kc)
    kk = np.arange(kc_i + 1)
    nu_shell = base * (plateau + cusp * (kk / kc) ** p)
    idx = shell_idx(sp)
    nu_map = np.zeros_like(sp.k2)
    for k in range(kc_i + 1):
        nu_map[idx == k] = nu_shell[k]
    return nu_map, nu_shell, E


def backscatter_force_w(sp, w_h, kc, nu_map, frac, rng, dt=2.0e-3):
    """FDT-tied divergence-free stochastic backscatter as a vorticity tendency.
    Inject, per shell near the cutoff, a random (Hermitian) vorticity forcing whose
    energy-injection RATE is `frac` x the local eddy-viscous dissipation rate.
    For white-in-time forcing the velocity-space energy target is rate/dt (an
    Euler-Maruyama injection ~ 0.5|f|^2 dt = rate*dt => |f|^2 energy = rate/dt).
    Concentrated in the upper half-band [kc/2, kc] (the backscatter range)."""
    idx = shell_idx(sp)
    kc_i = int(kc)
    # eddy-viscous energy dissipation per shell:  2 nu_t(k) k^2 E(k)
    E = energy_spectrum_from_w(sp, w_h, kc)
    white = rng.standard_normal((sp.n, sp.n))
    wf = np.fft.fft2(white)
    band = ((np.sqrt(sp.k2) >= 0.5 * kc) & (np.sqrt(sp.k2) <= kc)).astype(float)
    wf *= band
    # normalize per shell to inject frac * dissipated energy
    out = np.zeros_like(wf)
    psi_pow = sp.k2_inv  # |u|^2 per |w|^2 = 1/k^2
    for k in range(kc_i + 1):
        m = (idx == k) & (band > 0)
        if not m.any():
            continue
        nu_k = nu_map[m].mean()
        diss = 2.0 * nu_k * (k ** 2) * E[k]
        inj = frac * max(diss, 0.0) / dt     # rate -> velocity-energy target (white-in-time)
        # energy of unit-variance vorticity noise in shell, in velocity units:
        cur = (np.abs(wf[m]) ** 2 * psi_pow[m]).sum() / (sp.n ** 4) * 0.5
        if cur <= 1e-30 or inj <= 0.0:
            continue
        out[m] = wf[m] * np.sqrt(inj / cur)
    return out


# ---------------------------------------------------------------------------
# coarse LES integrator (vorticity, integrating-factor RK2 = Heun)
# ---------------------------------------------------------------------------
def jacobian_w(sp, w_h):
    u, v = vel_from_w(sp, w_h)
    wx = sp.ifft(1j * sp.kx * w_h)
    wy = sp.ifft(1j * sp.ky * w_h)
    adv = -(u * wx + v * wy)
    return sp.fft(adv) * sp.dealias


def run_les(n, closure, kc, nu=2.0e-4, mu=0.02, k_f=24.0, f_amp=2.5,
            steps=8000, dt=2.0e-3, seed=1, n_snap=40, c_ev=0.30, bs_frac=0.5,
            warmup_frac=0.5):
    sp = Spectral2D(n)
    force = Forcing(sp, k_f=k_f, f_amp=f_amp, seed=seed)
    rng = np.random.default_rng(seed + 777)
    k2 = sp.k2
    Llin = -nu * k2 - mu
    Llin[0, 0] = 0.0
    E1 = np.exp(Llin * dt)
    # seed
    w_h = np.fft.fft2(0.1 * rng.standard_normal((n, n)))
    w_h[0, 0] = 0.0

    def closure_tend(wh):
        """extra spectral vorticity tendency from the closure (explicit part)."""
        if closure == "none":
            return 0.0
        if closure == "smag":
            u, v = vel_from_w(sp, wh)
            mx, my = smagorinsky_force(sp, u, v, kc)
            curl = sp.ddx(my) - sp.ddy(mx)
            return sp.fft(curl) * sp.dealias
        if closure in ("specEV", "specEV_bs", "cuspEV", "cuspEV_bs"):
            plateau = 0.0 if closure.startswith("cusp") else 1.0
            cuspamp = 30.0 if closure.startswith("cusp") else 8.0
            pexp = 8 if closure.startswith("cusp") else 4
            nu_map, _, _ = spectral_eddy_visc(sp, wh, kc, c_ev=c_ev, cusp=cuspamp,
                                              p=pexp, plateau=plateau)
            tend = -nu_map * k2 * wh
            if closure.endswith("_bs"):
                tend = tend + backscatter_force_w(sp, wh, kc, nu_map, bs_frac, rng, dt=dt)
            return tend * sp.dealias
        raise ValueError(closure)

    snaps = []
    warm = int(steps * warmup_frac)
    snap_every = max(1, (steps - warm) // n_snap)
    blew = False
    for s in range(steps):
        a = jacobian_w(sp, w_h) + force() + closure_tend(w_h)
        w1 = E1 * w_h + dt * E1 * a
        b = jacobian_w(sp, w1) + force() + closure_tend(w1)
        w_h = E1 * w_h + 0.5 * dt * (E1 * a + b)
        w_h[0, 0] = 0.0
        if not np.all(np.isfinite(w_h)) or np.max(np.abs(w_h)) > 1e12:
            blew = True
            break
        if s >= warm and (s - warm) % snap_every == 0:
            snaps.append(w_h.copy())
    if blew or not snaps:
        return {"blew": True, "sp": sp}
    Es = np.array([energy_spectrum_from_w(sp, wh, kc) for wh in snaps])
    kes, enss = zip(*[ke_ens_from_w(sp, wh, kc) for wh in snaps])
    return {"blew": False, "sp": sp, "E": Es.mean(0), "E_std": Es.std(0),
            "ke": float(np.mean(kes)), "ens": float(np.mean(enss)), "nsnap": len(snaps)}


# ---------------------------------------------------------------------------
# DNS truth reference (filtered to |k|<=kc), time-averaged statistics
# ---------------------------------------------------------------------------
def dns_reference(n=256, kc=32, nu=2.0e-4, mu=0.02, k_f=24.0, f_amp=2.5,
                  spinup=6000, dt=2.5e-3, n_snap=40, snap_every=120, seed=0):
    dns = Vorticity2D(n=n, nu=nu, mu=mu, k_f=k_f, f_amp=f_amp, seed=seed)
    sp = dns.sp
    w_h = dns.run(spinup, dt=dt)
    E1 = np.exp(dns.L * dt)
    Es, kes, enss = [], [], []
    for s in range(n_snap * snap_every):
        a = dns._jacobian(w_h) + dns._forcing()
        w1 = E1 * w_h + dt * E1 * a
        b = dns._jacobian(w1) + dns._forcing()
        w_h = E1 * w_h + 0.5 * dt * (E1 * a + b)
        w_h[0, 0] = 0.0
        if s % snap_every == 0:
            Es.append(energy_spectrum_from_w(sp, w_h, kc))
            ke, ens = ke_ens_from_w(sp, w_h, kc)
            kes.append(ke); enss.append(ens)
    Es = np.array(Es)
    return {"sp": sp, "E": Es.mean(0), "E_std": Es.std(0),
            "ke": float(np.mean(kes)), "ens": float(np.mean(enss)), "nsnap": len(Es)}


def spec_err(E_ref, E_les, klo, khi):
    """Relative L2 spectrum error over band [klo,khi]."""
    a = E_ref[klo:khi + 1]; b = E_les[klo:khi + 1]
    return float(np.sqrt(np.sum((b - a) ** 2)) / (np.sqrt(np.sum(a ** 2)) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kc", type=int, default=32)
    ap.add_argument("--n-les", type=int, default=96)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--dns-spinup", type=int, default=6000)
    ap.add_argument("--c-ev", type=float, default=0.30)
    ap.add_argument("--bs-frac", type=float, default=0.5)
    ap.add_argument("--dns-n", type=int, default=256)
    ap.add_argument("--dns-snaps", type=int, default=40)
    ap.add_argument("--dns-snap-every", type=int, default=120)
    ap.add_argument("--k-f", type=float, default=24.0)
    args = ap.parse_args()
    kc = args.kc

    print(f"[DNS] {args.dns_n}^2 truth, filter kc={kc}, k_f={args.k_f} ...", flush=True)
    ref = dns_reference(n=args.dns_n, kc=kc, spinup=args.dns_spinup, k_f=args.k_f,
                        n_snap=args.dns_snaps, snap_every=args.dns_snap_every)
    print(f"[DNS] KE={ref['ke']:.4e}  ens={ref['ens']:.4e}  ({ref['nsnap']} snaps)", flush=True)

    klo_lo, khi_lo = 1, max(2, int(0.6 * args.k_f))      # low-k inverse-cascade band (below forcing)
    results = {}
    clos_list = ["none", "smag", "specEV", "specEV_bs", "cuspEV", "cuspEV_bs"]
    for clos in clos_list:
        print(f"[LES] {clos} (n={args.n_les}) ...", flush=True)
        r = run_les(args.n_les, clos, kc, steps=args.steps, c_ev=args.c_ev,
                    bs_frac=args.bs_frac, k_f=args.k_f)
        results[clos] = r
        if r["blew"]:
            print(f"   -> BLEW UP / unstable", flush=True)
            continue
        e_all = spec_err(ref["E"], r["E"], 1, kc)
        e_low = spec_err(ref["E"], r["E"], klo_lo, khi_lo)
        print(f"   KE={r['ke']:.4e} (DNS {ref['ke']:.4e})  ens={r['ens']:.4e} (DNS {ref['ens']:.4e})"
              f"  specErr[all]={e_all:.3f}  specErr[low-k]={e_low:.3f}", flush=True)

    # summary table
    print("\n=== A-POSTERIORI SUMMARY (time-averaged, vs filtered 256^2 DNS) ===")
    print(f"{'closure':12s} {'stable':7s} {'KE/KE_dns':10s} {'ens/ens_dns':12s} {'specErr_all':12s} {'specErr_lowk':12s}")
    for clos in clos_list:
        r = results[clos]
        if r["blew"]:
            print(f"{clos:12s} {'NO':7s} {'-':10s} {'-':12s} {'-':12s} {'-':12s}")
            continue
        e_all = spec_err(ref["E"], r["E"], 1, kc)
        e_low = spec_err(ref["E"], r["E"], klo_lo, khi_lo)
        print(f"{clos:12s} {'yes':7s} {r['ke']/ref['ke']:<10.3f} {r['ens']/ref['ens']:<12.3f} "
              f"{e_all:<12.3f} {e_low:<12.3f}")
    np.savez("/tmp/aposteriori_result.npz",
             k=np.arange(kc + 1), E_dns=ref["E"],
             **{f"E_{c}": (results[c]["E"] if not results[c]["blew"] else np.zeros(kc + 1))
                for c in results})
    print("\n[saved] /tmp/aposteriori_result.npz")


if __name__ == "__main__":
    main()
