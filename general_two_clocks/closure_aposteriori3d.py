r"""P1 a-posteriori closure test in 3-D (paper1 sec 8.1 / the decisive NR31 test).

The committed 3-D benchmark ``run_closure3d.py`` is *a-priori* (frozen-field): it
scores closures on a single filtered DNS snapshot.  The 2-D a-posteriori test
``closure_aposteriori.py`` established the honest **2-D ceiling**: no eddy-viscosity
closure beats no-model on the resolved spectrum, because in 2-D the net
resolved->subgrid energy flux is ~0 (energy goes UP-scale).

NR31 (general_two_clocks/new_relationships4.py) derives *why* and predicts the
**opposite ordering in 3-D**: the variational optimum is

    nu_opt = <Pi> / (2 <|S|^2>),     Pi = -tau^d_ij S_ij  (resolved->subgrid flux),

which is ~0 in 2-D (<Pi>~0) but strictly POSITIVE in 3-D (forward cascade,
<Pi> > 0, the n=128-192 a-priori runs measure backscatter fraction ~0.48 yet a net
forward mean flux).  So in 3-D a structured spectral eddy viscosity -- and, even
better, one carrying an FDT-tied backscatter -- *should* beat no-model a-posteriori.
This harness runs that genuinely predictive, self-contained coarse LES forward in
time in 3-D and compares its long-time resolved statistics to the
spectrally-filtered 3-D DNS truth.  It is the test paper1 sec 8.1 defers to and the
falsification test of NR31's 2-D-vs-3-D prediction.

Closures (all predictive: use only resolved data, never the DNS truth):
  none      molecular viscosity + forcing only (under-resolved control)
  smag      Smagorinsky (sgs3d.smagorinsky_force3d), positive-definite eddy visc.
  specEV    positive spectral eddy viscosity nu_t(k) = c_ev sqrt(E(kc)/kc)(plateau+cusp)
  specEV_bs specEV + FDT-tied, Leray-divergence-free stochastic backscatter
  cuspEV    cusp-only spectral eddy viscosity (dissipation concentrated at the cutoff)
  cuspEV_bs cusp-only + FDT backscatter

The spectral eddy viscosity is integrated *semi-implicitly* (folded into the exact
viscous integrating factor each step) so the stiff near-cutoff dissipation is
unconditionally stable; the nonlinear term, forcing and backscatter are explicit
(integrating-factor Heun / RK2), the same scheme as ``closure/dns3d.py``.

Backend-agnostic: ``--gpu`` switches the array module to CuPy (Kaggle Tesla P100)
and bumps the resolution.  CPU/NumPy defaults are a fast smoke test.

Usage:
    python closure_aposteriori3d.py                       # CPU smoke
    python closure_aposteriori3d.py --gpu --dns-n 192 --n-les 72 --kc 24
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from closure.spectral3d import Spectral3D, project3d_spectral, shell_index, to_host
from closure.dns3d import ForcedNS3D, DNS3DConfig
from closure.sgs3d import smagorinsky_force3d


# ---------------------------------------------------------------------------
# resolved energy spectrum E(k), k = 0..kc  (matches energy_spectrum3d norm)
# ---------------------------------------------------------------------------
def resolved_espec(sp, uh, vh, wh, idx_flat, kc_i):
    """Shell-summed resolved kinetic-energy spectrum E(k), k=0..kc (vectorized via
    bincount so it is one reduction, not a per-shell Python loop -- matters on GPU)."""
    xp = sp.xp
    n = sp.n
    P = 0.5 * (xp.abs(uh) ** 2 + xp.abs(vh) ** 2 + xp.abs(wh) ** 2) / (n ** 6)
    E = xp.bincount(idx_flat, weights=P.ravel(), minlength=kc_i + 1)[:kc_i + 1]
    return E


def ke_ens_resolved(sp, uh, vh, wh, kc):
    """Resolved KE and enstrophy from the |k|<=kc band."""
    xp = sp.xp
    mask = (sp.kmag <= kc)
    uf, vf, wf = uh * mask, vh * mask, wh * mask
    u, v, w = sp.ifft(uf), sp.ifft(vf), sp.ifft(wf)
    ke = 0.5 * float(xp.mean(u * u + v * v + w * w))
    # enstrophy = 0.5 <|omega|^2>
    ox = sp.ifft(1j * (sp.ky * wf - sp.kz * vf))
    oy = sp.ifft(1j * (sp.kz * uf - sp.kx * wf))
    oz = sp.ifft(1j * (sp.kx * vf - sp.ky * uf))
    ens = 0.5 * float(xp.mean(ox * ox + oy * oy + oz * oz))
    return ke, ens


# ---------------------------------------------------------------------------
# predictive spectral eddy viscosity (paper1 Eq. 8 structure), 3-D shell map
# ---------------------------------------------------------------------------
def nu_t_shell_3d(E, kc_i, kc, c_ev, cusp, p, plateau):
    base = c_ev * (float(E[kc_i]) / kc) ** 0.5 if float(E[kc_i]) > 0 else 0.0
    kk = np.arange(kc_i + 1)
    nu_shell = base * (plateau + cusp * (kk / kc) ** p)
    return nu_shell  # numpy array length kc_i+1


def build_nu_map(sp, nu_shell, idx, kc_i):
    """Map per-shell nu_t(k) onto the 3-D wavenumber grid by a single gather
    nu_shell[idx] (vectorized; modes above kc -- dealiased to zero anyway -- get 0)."""
    xp = sp.xp
    nu_shell_dev = xp.asarray(nu_shell)
    idxc = xp.minimum(idx, kc_i)
    nu_map = nu_shell_dev[idxc]
    return xp.where(idx <= kc_i, nu_map, xp.asarray(0.0))


def backscatter_force(sp, E, nu_shell, idx, kc_i, kc, frac, rng, dt):
    """FDT-tied, Leray-divergence-free stochastic backscatter (velocity form).

    Inject, per shell in the upper band [kc/2, kc], a random *solenoidal* velocity
    force whose energy-injection RATE equals ``frac`` x the local eddy-viscous
    dissipation rate ``2 nu_t(k) k^2 E(k)``.  White-in-time: the force-field
    velocity-energy target per shell is rate/dt, so the energy added over a step
    (~ 0.5 |dt f|^2 / dt) integrates to ``frac`` x dissipation per unit time.
    Returns spectral (solenoidal) force components (xp arrays)."""
    xp = sp.xp
    shape = (sp.n,) * 3
    # random solenoidal field
    rxh = sp.fft(xp.asarray(rng.standard_normal(shape)))
    ryh = sp.fft(xp.asarray(rng.standard_normal(shape)))
    rzh = sp.fft(xp.asarray(rng.standard_normal(shape)))
    rxh, ryh, rzh = project3d_spectral(sp, rxh, ryh, rzh)
    band = (sp.kmag >= 0.5 * kc) & (sp.kmag <= kc)
    rxh, ryh, rzh = rxh * band, ryh * band, rzh * band
    fx = xp.zeros_like(rxh); fy = xp.zeros_like(ryh); fz = xp.zeros_like(rzh)
    n6 = sp.n ** 6
    klo = max(1, int(0.5 * kc))
    for k in range(klo, kc_i + 1):
        diss = 2.0 * float(nu_shell[k]) * (k ** 2) * float(E[k])
        if diss <= 0.0:
            continue
        target = frac * diss / dt                      # force-field velocity energy
        m = idx == k
        cur = 0.5 * float((xp.abs(rxh[m]) ** 2 + xp.abs(ryh[m]) ** 2
                           + xp.abs(rzh[m]) ** 2).sum()) / n6
        if cur <= 1e-30:
            continue
        s = (target / cur) ** 0.5
        fx[m] = rxh[m] * s; fy[m] = ryh[m] * s; fz[m] = rzh[m] * s
    return fx, fy, fz


# ---------------------------------------------------------------------------
# coarse 3-D LES integrator (velocity form, integrating-factor Heun)
# ---------------------------------------------------------------------------
class CoarseLES3D:
    def __init__(self, n, kc, closure, nu, k_f, f_band, f_amp, dt, seed,
                 c_ev=0.30, bs_frac=0.5, xp=np):
        self.sp = Spectral3D(n, xp)
        self.xp = xp
        self.n = n
        self.kc = kc
        self.kc_i = int(kc)
        self.closure = closure
        self.nu = nu
        self.dt = dt
        self.c_ev = c_ev
        self.bs_frac = bs_frac
        sp = self.sp
        self.idx = shell_index(sp)
        self.idx_flat = self.idx.ravel()
        self.k_f, self.f_band, self.f_amp = k_f, f_band, f_amp
        self.ring = ((sp.kmag >= k_f - f_band) & (sp.kmag <= k_f + f_band))
        self.E1_visc = xp.exp(-nu * sp.k2 * dt)        # molecular only (none/smag)
        try:
            self.rng = xp.random.default_rng(seed)
        except AttributeError:
            self.rng = np.random.default_rng(seed)
        self.rng_bs = np.random.default_rng(seed + 777)
        u = 0.05 * xp.asarray(np.random.default_rng(seed + 1).standard_normal((n,) * 3))
        v = 0.05 * xp.asarray(np.random.default_rng(seed + 2).standard_normal((n,) * 3))
        w = 0.05 * xp.asarray(np.random.default_rng(seed + 3).standard_normal((n,) * 3))
        uh, vh, wh = project3d_spectral(sp, sp.fft(u), sp.fft(v), sp.fft(w))
        self.uh, self.vh, self.wh = uh, vh, wh
        self.cusp = closure.startswith("cusp")
        self.has_ev = closure in ("specEV", "specEV_bs", "cuspEV", "cuspEV_bs")
        self.has_bs = closure.endswith("_bs")

    def _nonlinear(self, uh, vh, wh):
        sp = self.sp
        u, v, w = sp.ifft(uh), sp.ifft(vh), sp.ifft(wh)
        ux, uy, uz = sp.ifft(1j * sp.kx * uh), sp.ifft(1j * sp.ky * uh), sp.ifft(1j * sp.kz * uh)
        vx, vy, vz = sp.ifft(1j * sp.kx * vh), sp.ifft(1j * sp.ky * vh), sp.ifft(1j * sp.kz * vh)
        wx, wy, wz = sp.ifft(1j * sp.kx * wh), sp.ifft(1j * sp.ky * wh), sp.ifft(1j * sp.kz * wh)
        ax = u * ux + v * uy + w * uz
        ay = u * vx + v * vy + w * vz
        az = u * wx + v * wy + w * wz
        axh = sp.fft(ax) * sp.dealias
        ayh = sp.fft(ay) * sp.dealias
        azh = sp.fft(az) * sp.dealias
        axh, ayh, azh = project3d_spectral(sp, axh, ayh, azh)
        return -axh, -ayh, -azh

    def _forcing(self):
        sp, xp = self.sp, self.xp
        shape = (self.n,) * 3
        fxh = sp.fft(xp.asarray(self.rng.standard_normal(shape))) * self.ring
        fyh = sp.fft(xp.asarray(self.rng.standard_normal(shape))) * self.ring
        fzh = sp.fft(xp.asarray(self.rng.standard_normal(shape))) * self.ring
        fxh, fyh, fzh = project3d_spectral(sp, fxh, fyh, fzh)
        fx, fy, fz = sp.ifft(fxh), sp.ifft(fyh), sp.ifft(fzh)
        rms = float(xp.sqrt(xp.mean(fx ** 2 + fy ** 2 + fz ** 2))) + 1e-30
        s = self.f_amp / rms
        return fxh * s, fyh * s, fzh * s

    def _closure_state(self):
        """Per-step closure: returns (E1_eff, bs_force or None, smag_force or None).
        E1_eff folds molecular + (semi-implicit) eddy viscosity into the integrating
        factor; backscatter and Smagorinsky are returned as explicit spectral forces."""
        sp, xp = self.sp, self.xp
        if self.closure == "none":
            return self.E1_visc, None, None
        if self.closure == "smag":
            u, v, w = sp.ifft(self.uh), sp.ifft(self.vh), sp.ifft(self.wh)
            mx, my, mz = smagorinsky_force3d(sp, u, v, w, self.kc)
            return self.E1_visc, None, (sp.fft(mx) * sp.dealias,
                                        sp.fft(my) * sp.dealias, sp.fft(mz) * sp.dealias)
        # spectral eddy viscosity (+/- backscatter)
        E = resolved_espec(sp, self.uh, self.vh, self.wh, self.idx_flat, self.kc_i)
        plateau = 0.0 if self.cusp else 1.0
        cuspamp = 30.0 if self.cusp else 8.0
        pexp = 8 if self.cusp else 4
        nu_shell = nu_t_shell_3d(E, self.kc_i, self.kc, self.c_ev, cuspamp, pexp, plateau)
        nu_map = build_nu_map(sp, nu_shell, self.idx, self.kc_i)
        E1_eff = xp.exp(-(self.nu * sp.k2 + nu_map * sp.k2) * self.dt)
        bs = None
        if self.has_bs:
            bs = backscatter_force(sp, E, nu_shell, self.idx, self.kc_i, self.kc,
                                   self.bs_frac, self.rng_bs, self.dt)
        return E1_eff, bs, None

    def step(self):
        sp, dt = self.sp, self.dt
        uh, vh, wh = self.uh, self.vh, self.wh
        E1, bs, smag = self._closure_state()
        fxh, fyh, fzh = self._forcing()
        ax, ay, az = self._nonlinear(uh, vh, wh)
        ax, ay, az = ax + fxh, ay + fyh, az + fzh
        if smag is not None:
            ax, ay, az = ax + smag[0], ay + smag[1], az + smag[2]
        if bs is not None:
            ax, ay, az = ax + bs[0], ay + bs[1], az + bs[2]
        u1 = E1 * uh + dt * E1 * ax
        v1 = E1 * vh + dt * E1 * ay
        w1 = E1 * wh + dt * E1 * az
        bx, by, bz = self._nonlinear(u1, v1, w1)
        bx, by, bz = bx + fxh, by + fyh, bz + fzh
        if smag is not None:
            bx, by, bz = bx + smag[0], by + smag[1], bz + smag[2]
        if bs is not None:
            bx, by, bz = bx + bs[0], by + bs[1], bz + bs[2]
        uh = E1 * uh + 0.5 * dt * (E1 * ax + bx)
        vh = E1 * vh + 0.5 * dt * (E1 * ay + by)
        wh = E1 * wh + 0.5 * dt * (E1 * az + bz)
        self.uh, self.vh, self.wh = project3d_spectral(sp, uh, vh, wh)

    def stable(self):
        xp = self.xp
        return bool(xp.all(xp.isfinite(self.uh)) and
                    float(xp.max(xp.abs(self.uh))) < 1e12)

    def run_collect(self, steps, warmup_frac, n_snap):
        sp = self.sp
        warm = int(steps * warmup_frac)
        snap_every = max(1, (steps - warm) // n_snap)
        Es, kes, enss = [], [], []
        for s in range(steps):
            self.step()
            if not self.stable():
                return {"blew": True}
            if s >= warm and (s - warm) % snap_every == 0:
                E = resolved_espec(sp, self.uh, self.vh, self.wh, self.idx_flat, self.kc_i)
                ke, ens = ke_ens_resolved(sp, self.uh, self.vh, self.wh, self.kc)
                Es.append(to_host(E)); kes.append(ke); enss.append(ens)
        if not Es:
            return {"blew": True}
        Es = np.array(Es)
        return {"blew": False, "E": Es.mean(0), "E_std": Es.std(0),
                "ke": float(np.mean(kes)), "ens": float(np.mean(enss)), "nsnap": len(Es)}


# ---------------------------------------------------------------------------
# DNS truth reference (filtered to |k|<=kc), time-averaged statistics
# ---------------------------------------------------------------------------
def dns_reference_3d(n, kc, nu, k_f, f_band, f_amp, dt, spinup, n_snap, snap_every,
                     seed, xp=np, report_every=0):
    cfg = DNS3DConfig(n=n, nu=nu, k_f=k_f, f_band=f_band, f_amp=f_amp, dt=dt, seed=seed)
    dns = ForcedNS3D(cfg, xp=xp)
    dns.run(spinup, report_every=report_every)
    sp = dns.sp
    idx = shell_index(sp)
    idx_flat = idx.ravel()
    kc_i = int(kc)
    Es, kes, enss = [], [], []
    for s in range(n_snap * snap_every):
        dns.step()
        if s % snap_every == 0:
            uh, vh, wh = sp.fft(dns.u), sp.fft(dns.v), sp.fft(dns.w)
            E = resolved_espec(sp, uh, vh, wh, idx_flat, kc_i)
            ke, ens = ke_ens_resolved(sp, uh, vh, wh, kc)
            Es.append(to_host(E)); kes.append(ke); enss.append(ens)
    Es = np.array(Es)
    return {"E": Es.mean(0), "E_std": Es.std(0),
            "ke": float(np.mean(kes)), "ens": float(np.mean(enss)), "nsnap": len(Es)}


def spec_err(E_ref, E_les, klo, khi):
    a = np.asarray(E_ref)[klo:khi + 1]; b = np.asarray(E_les)[klo:khi + 1]
    return float(np.sqrt(np.sum((b - a) ** 2)) / (np.sqrt(np.sum(a ** 2)) + 1e-30))


def make_figure(out_png, k, E_dns, results, kc, k_f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    m = (k >= 1) & (k <= kc)
    ax.loglog(k[m], np.asarray(E_dns)[m], "o-", color="black", lw=2, ms=4, label="filtered DNS (truth)")
    styles = {"none": ("#777777", "s--"), "smag": ("#d62728", "^-"),
              "specEV": ("#1f77b4", "v-"), "specEV_bs": ("#2ca02c", "D-"),
              "cuspEV": ("#9467bd", "v:"), "cuspEV_bs": ("#17becf", "D:")}
    for c, r in results.items():
        if r.get("blew"):
            continue
        col, st = styles.get(c, ("#333", "-"))
        ax.loglog(k[m], np.asarray(r["E"])[m], st, color=col, ms=3, lw=1.3, label=c)
    ax.axvline(kc, color="k", ls=":", lw=0.8)
    ax.set_xlabel("|k|"); ax.set_ylabel("E(k)")
    ax.set_title("3-D a-posteriori resolved spectrum vs filtered DNS")
    ax.legend(fontsize=7)
    ax = axes[1]
    names = [c for c in results if not results[c].get("blew")]
    eall = [spec_err(E_dns, results[c]["E"], 1, kc) for c in names]
    cols = [styles.get(c, ("#333", ""))[0] for c in names]
    bars = ax.bar(names, eall, color=cols, edgecolor="k", lw=0.6)
    none_err = spec_err(E_dns, results["none"]["E"], 1, kc) if not results["none"].get("blew") else None
    if none_err is not None:
        ax.axhline(none_err, color="#777777", ls="--", lw=1.2, label=f"no-model = {none_err:.3f}")
        ax.legend(fontsize=8)
    for b, v in zip(bars, eall):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("resolved-spectrum L2 error vs DNS  [1,kc]")
    ax.set_title("Does a closure beat no-model in 3-D? (NR31)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[fig] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="use CuPy (Kaggle P100); bumps defaults")
    ap.add_argument("--dns-n", type=int, default=None)
    ap.add_argument("--n-les", type=int, default=None)
    ap.add_argument("--kc", type=int, default=None)
    ap.add_argument("--nu", type=float, default=None)
    ap.add_argument("--k-f", type=float, default=2.5)
    ap.add_argument("--f-band", type=float, default=1.5)
    ap.add_argument("--f-amp", type=float, default=1.2)
    ap.add_argument("--dt", type=float, default=5.0e-3)
    ap.add_argument("--dns-spinup", type=int, default=None)
    ap.add_argument("--les-steps", type=int, default=None)
    ap.add_argument("--n-snap", type=int, default=30)
    ap.add_argument("--dns-snap-every", type=int, default=None)
    ap.add_argument("--c-ev", type=float, default=0.30)
    ap.add_argument("--bs-frac", type=float, default=0.5)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--json", default="figures/77_closure_aposteriori3d.json")
    args = ap.parse_args()

    if args.gpu:
        import cupy as xp
        dns_n = args.dns_n or 192
        n_les = args.n_les or 72
        kc = args.kc or 24
        nu = args.nu if args.nu is not None else 3.5e-3
        spinup = args.dns_spinup or 4000
        les_steps = args.les_steps or 6000
        dns_snap_every = args.dns_snap_every or 120
    else:
        xp = np
        dns_n = args.dns_n or 48
        n_les = args.n_les or 24
        kc = args.kc or 8
        nu = args.nu if args.nu is not None else 6.0e-3
        spinup = args.dns_spinup or 800
        les_steps = args.les_steps or 1200
        dns_snap_every = args.dns_snap_every or 40
    backend = "cupy" if args.gpu else "numpy"
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[cfg] backend={backend} dns_n={dns_n} n_les={n_les} kc={kc} nu={nu} "
          f"spinup={spinup} les_steps={les_steps}", flush=True)
    assert n_les // 3 >= kc, f"LES grid {n_les} (kmax={n_les//3}) must resolve kc={kc}"
    assert dns_n // 3 > kc, f"DNS grid {dns_n} (kmax={dns_n//3}) must exceed kc={kc}"

    print("[DNS] spinning up 3-D truth ...", flush=True)
    ref = dns_reference_3d(dns_n, kc, nu, args.k_f, args.f_band, args.f_amp, args.dt,
                           spinup, args.n_snap, dns_snap_every, seed=0, xp=xp,
                           report_every=max(1, spinup // 6))
    print(f"[DNS] resolved KE={ref['ke']:.4e} ens={ref['ens']:.4e} ({ref['nsnap']} snaps)", flush=True)

    klo_lo, khi_lo = 1, max(2, int(2 * args.k_f))      # low-k band (around forcing)
    clos_list = ["none", "smag", "specEV", "specEV_bs", "cuspEV", "cuspEV_bs"]
    results = {}
    for clos in clos_list:
        print(f"[LES] {clos} (n={n_les}) ...", flush=True)
        les = CoarseLES3D(n_les, kc, clos, nu, args.k_f, args.f_band, args.f_amp,
                          args.dt, seed=1, c_ev=args.c_ev, bs_frac=args.bs_frac, xp=xp)
        r = les.run_collect(les_steps, warmup_frac=0.5, n_snap=args.n_snap)
        results[clos] = r
        if r["blew"]:
            print("   -> BLEW UP / unstable", flush=True)
            continue
        e_all = spec_err(ref["E"], r["E"], 1, kc)
        e_low = spec_err(ref["E"], r["E"], klo_lo, khi_lo)
        print(f"   KE={r['ke']:.4e} (DNS {ref['ke']:.4e})  ens={r['ens']:.4e} (DNS {ref['ens']:.4e})"
              f"  specErr[all]={e_all:.3f}  specErr[low-k]={e_low:.3f}", flush=True)

    # summary
    none_err = spec_err(ref["E"], results["none"]["E"], 1, kc) if not results["none"]["blew"] else None
    print("\n=== 3-D A-POSTERIORI SUMMARY (time-averaged, vs filtered DNS) ===")
    print(f"{'closure':11s} {'stable':7s} {'KE/KEdns':9s} {'ens/ensdns':11s} {'specErr_all':12s} {'beats_none':10s}")
    summary = {}
    for clos in clos_list:
        r = results[clos]
        if r["blew"]:
            print(f"{clos:11s} {'NO':7s} {'-':9s} {'-':11s} {'-':12s} {'-':10s}")
            summary[clos] = {"stable": False}
            continue
        e_all = spec_err(ref["E"], r["E"], 1, kc)
        beats = (none_err is not None and e_all < none_err and clos != "none")
        print(f"{clos:11s} {'yes':7s} {r['ke']/ref['ke']:<9.3f} {r['ens']/ref['ens']:<11.3f} "
              f"{e_all:<12.3f} {('YES' if beats else ('-' if clos=='none' else 'no')):10s}")
        summary[clos] = {"stable": True, "ke_ratio": r["ke"] / ref["ke"],
                         "ens_ratio": r["ens"] / ref["ens"], "spec_err_all": e_all,
                         "beats_none": bool(beats)}

    best = min((c for c in clos_list if c != "none" and not results[c]["blew"]),
               key=lambda c: spec_err(ref["E"], results[c]["E"], 1, kc), default=None)
    verdict = {
        "backend": backend, "dns_n": dns_n, "n_les": n_les, "kc": kc, "nu": nu,
        "les_steps": les_steps, "dns_spinup": spinup,
        "none_spec_err": none_err,
        "best_closure": best,
        "best_spec_err": (spec_err(ref["E"], results[best]["E"], 1, kc) if best else None),
        "any_closure_beats_none": any(summary[c].get("beats_none") for c in clos_list),
        "dns_ke": ref["ke"], "dns_ens": ref["ens"],
        "summary": summary,
    }
    print("\n[verdict] best closure:", best,
          "| any closure beats no-model:", verdict["any_closure_beats_none"])

    k = np.arange(kc + 1)
    np.savez(os.path.join(args.out_dir, "77_closure_aposteriori3d.npz"),
             k=k, E_dns=ref["E"],
             **{f"E_{c}": (results[c]["E"] if not results[c]["blew"] else np.zeros(kc + 1))
                for c in results})
    with open(args.json, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"[json] {args.json}")
    make_figure(os.path.join(args.out_dir, "77_closure_aposteriori3d.png"),
                k, ref["E"], results, kc, args.k_f)


if __name__ == "__main__":
    main()
