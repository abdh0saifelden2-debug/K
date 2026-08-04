#!/usr/bin/env python
"""Option-3 minimal Stefan prototype -- test matrix A-D.

Runs the four prototype experiments described in the Option-3 spec and writes a
JSON of the diagnostics plus a summary plot.  CPU-only; the defaults finish in a
few minutes at ``n=128``.

  A  flat base, no forcing      -> Stefan sqrt(t) melt recovery (validation)
  B  flat base, steady forcing  -> flow's effect on mean melt
  C  wavy base, steady forcing  -> differential melt, roughness evolution
  D  wavy base, tidal forcing   -> roughness oscillation + melt/forcing phase lag

What the prototype demonstrates (honest scope -- see THEORY_CAVITY.md S11):

  * The moving Brinkman boundary works: H_c(x,t) recedes by melting and the
    no-flow limit recovers the Neumann sqrt(t) similarity law (Test A).
  * Melt -> geometry feedback is real: over a wavy base the melt rate is
    spatially differential and tracks the geometry, reshaping sigma_h (Tests C/D).
  * The flow -> melt leg is NOT activated by gentle body-force-driven *horizontal*
    flow at accessible Re/Pe: melt is vertical-conduction-limited, so Test B shows
    ~no enhancement.  Activating it needs vertical transport -- buoyant convection
    (diffusive timescale tau ~ L^2/kappa, ~1e5 steps) or genuine turbulence.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from subglacial.stefan_prototype import StefanPrototype, StefanConfig
from subglacial.moving_boundary import neumann_lambda


def _series(cfg: StefanConfig, steps: int, nblocks: int, warm=True, seed_noise=0.0):
    s = StefanPrototype(cfg)
    if warm:
        s.init_warm_cavity()
    if seed_noise:
        rng = np.random.default_rng(cfg.seed)
        s.u += seed_noise * rng.standard_normal(s.u.shape) * s.fluid
        s.v += seed_noise * rng.standard_normal(s.v.shape) * s.fluid
    rec = {k: [] for k in ("t", "Hbar", "sigma_h", "KE", "Nu", "melt_dist", "fbody", "mmean")}
    block = max((steps + nblocks - 1) // nblocks, 1)  # ceiling: run >= steps total
    for _ in range(nblocks):
        s.run(block)
        omega = 2.0 * np.pi / cfg.T_tide
        rec["t"].append(s.t)
        rec["Hbar"].append(s.mean_height())
        rec["sigma_h"].append(s.roughness())
        rec["KE"].append(s.kinetic_energy())
        rec["Nu"].append(s.nusselt())
        rec["melt_dist"].append(s.melt_distance())
        rec["fbody"].append(cfg.f0 + cfg.df * np.sin(omega * s.t))
        rec["mmean"].append(float(s.melt_rate().mean()))
    return s, {k: np.array(v) for k, v in rec.items()}


def test_A(n, steps):
    # Test A is the *validation* case -> use the validated n=96 diffuse-interface
    # config (the diffuse-interface accuracy is resolution dependent; n=96 is the
    # setup checked in tests/test_stefan.py).  B/C/D below honour --n.
    if n != 96:
        print(f"[A] note: overriding --n {n} -> 96 (validated Stefan config)")
    n = 96
    St, kappa, ybed = 0.5, 3.0e-3, 0.30
    cfg = StefanConfig(n=n, y_bed=ybed, H0=ybed + 0.18, eps=0.0, f0=0.0, df=0.0,
                       beta=0.0, St=St, kappa=kappa, nu=3.0e-3, dt=5.0e-4,
                       n_mask=5, interface=2.0)
    s, r = _series(cfg, steps, 12)
    sfront = r["Hbar"] - ybed
    A, B = np.polyfit(r["t"], sfront ** 2, 1)
    lam = neumann_lambda(1.0 / St)
    ratio = A / ((2.0 * lam) ** 2 * kappa)
    print(f"[A] flat, no forcing : s={sfront[0]:.3f}->{sfront[-1]:.3f}  "
          f"sqrt(t)-slope ratio vs Neumann = {ratio:.3f}  (1.0 = exact)")
    r["stefan_slope_ratio"] = float(ratio)
    return r


def test_B(n, steps):
    base = dict(n=n, y_bed=0.30, H0=0.55, eps=0.0, df=0.0, beta=0.0, St=1.0,
                kappa=3.0e-3, nu=3.0e-3, dt=5.0e-4, n_mask=5, interface=2.0)
    _, r0 = _series(StefanConfig(f0=0.0, **base), steps, 8)
    sf, rf = _series(StefanConfig(f0=0.15, **base), steps, 8)
    enh = rf["melt_dist"][-1] / max(r0["melt_dist"][-1], 1e-30)
    print(f"[B] flat, forced     : melt(forced)/melt(no-flow) = {enh:.3f}  "
          f"umax={np.abs(sf.u).max():.3f}  -> horizontal flow does not enhance "
          f"conduction-limited melt")
    rf["melt_enhancement"] = float(enh)
    return rf


def test_C(n, steps):
    cfg = StefanConfig(n=n, y_bed=0.30, H0=0.55, eps=0.05, k0=2, f0=0.10, df=0.0,
                       beta=0.0, St=1.0, kappa=3.0e-3, nu=3.0e-3, dt=5.0e-4,
                       n_mask=5, interface=2.0)
    s, r = _series(cfg, steps, 12)
    m, x = s.melt_rate(), s.xcol
    corr = float(np.corrcoef(m, np.sin(cfg.k0 * x))[0, 1])
    print(f"[C] wavy, forced     : sigma_h {r['sigma_h'][0]:.4f}->{r['sigma_h'][-1]:.4f}  "
          f"corr(m, geometry) = {corr:+.2f}  -> differential melt tracks geometry")
    r["melt_geometry_corr"] = corr
    return r


def test_D(n, steps):
    cfg = StefanConfig(n=n, y_bed=0.30, H0=0.55, eps=0.05, k0=2, f0=0.10, df=0.05,
                       T_tide=1.0, beta=0.0, St=1.0, kappa=3.0e-3, nu=3.0e-3,
                       dt=5.0e-4, n_mask=5, interface=2.0)
    s, r = _series(cfg, steps, 24)
    # phase lag between mean melt and the body force (cross-correlation peak)
    a = r["mmean"] - r["mmean"].mean()
    b = r["fbody"] - r["fbody"].mean()
    if a.std() > 0 and b.std() > 0:
        xc = np.correlate(a / a.std(), b / b.std(), mode="full") / len(a)
        lag = int(np.argmax(xc) - (len(a) - 1))
    else:
        lag = 0
    print(f"[D] wavy, tidal      : sigma_h range "
          f"[{r['sigma_h'].min():.4f}, {r['sigma_h'].max():.4f}]  "
          f"melt/forcing lag = {lag} blocks")
    r["melt_forcing_lag_blocks"] = int(lag)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="stefan_sweep.json")
    ap.add_argument("--plot", default="stefan_sweep.png")
    args = ap.parse_args()

    print(f"Stefan prototype sweep  (n={args.n}, steps={args.steps}/test)\n" + "-" * 64)
    res = {"A": test_A(args.n, args.steps), "B": test_B(args.n, args.steps),
           "C": test_C(args.n, args.steps), "D": test_D(args.n, args.steps)}

    dump = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()} for k, v in res.items()}
    with open(args.out, "w") as fh:
        json.dump(dump, fh, indent=2)
    print("-" * 64 + f"\nwrote {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        ax[0, 0].plot(res["A"]["t"], (res["A"]["Hbar"] - 0.30) ** 2, "o-")
        ax[0, 0].set(title="A: flat, no forcing -- s^2 vs t (sqrt(t) Stefan)",
                     xlabel="t", ylabel=r"$s^2$")
        ax[0, 1].plot(res["B"]["t"], res["B"]["Nu"], "o-")
        ax[0, 1].set(title="B: flat, forced -- Nusselt(t)", xlabel="t", ylabel="Nu")
        ax[1, 0].plot(res["C"]["t"], res["C"]["sigma_h"], "o-")
        ax[1, 0].set(title="C: wavy, forced -- roughness sigma_h(t)",
                     xlabel="t", ylabel=r"$\sigma_h$")
        ax[1, 1].plot(res["D"]["t"], res["D"]["fbody"], label="f_body")
        ax[1, 1].plot(res["D"]["t"], res["D"]["mmean"] / max(np.abs(res["D"]["mmean"]).max(), 1e-30)
                      * np.abs(res["D"]["fbody"]).max(), label="melt (scaled)")
        ax[1, 1].set(title="D: wavy, tidal -- melt vs forcing", xlabel="t")
        ax[1, 1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")
    except Exception as exc:  # plotting is optional
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
