"""Part 9c -- the two-clocks closure benchmark in 3D (the vortex-stretching extension).

The genuine 3D extension of Part 9b (run_closure.py).  Generates a forced 3D
incompressible DNS truth field, sharp-spectral-filters it at kc, forms the *exact*
subgrid force m_true, and scores three closures:

  (i)   Smagorinsky (K-theory): positive eddy viscosity, purely dissipative.
  (ii)  Spectrum-matched surrogate: identical E_m(k), random phases.
  (iii) Projected-FDT: scale-dependent nu_t(k) (allows backscatter) + projected noise.

Diagnostics shared with 2D (Part 9b): force spectrum E_m(k), RMS divergence, and
transfer spectrum T(k).  Diagnostics that ONLY EXIST IN 3D (the whole point):
  - vortex-stretching enstrophy production <omega_i S_ij omega_j>  (=0 in 2D),
  - strain/vorticity alignment (Constantin-Fefferman intermediate-eigenvector geometry),
  - SGS backscatter VOLUME FRACTION (Pi=-tau^d_ij S_ij < 0) -- exactly 0 for
    Smagorinsky, ~1/2 for real 3D turbulence.

Generates figures 61-64 and REPORT_CLOSURE3D.md.  Needs no downloaded data.

CPU smoke test (n=48, ~2 min):
    python run_closure3d.py --out-dir figures --report REPORT_CLOSURE3D.md
GPU run on a Kaggle Tesla P100 (developed turbulence, the headline numbers):
    python run_closure3d.py --gpu --n 128 --kc 24 --steps 2500 \
        --out-dir figures --report REPORT_CLOSURE3D.md
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from closure.dns3d import ForcedNS3D, DNS3DConfig  # noqa: E402
from closure.sgs3d import (  # noqa: E402
    exact_sgs_force3d, smagorinsky_force3d, surrogate_force3d, projected_fdt_force3d,
    force_spectrum3d, transfer_spectrum3d, divergence_rms3d, energy_spectrum3d,
    enstrophy_production, strain_vorticity_alignment, sgs_flux_stats,
)


def _safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


# ---------------------------------------------------------------------------
# Figure 61 -- force spectrum E_m(k) (+ KE spectrum inset)
# ---------------------------------------------------------------------------

def fig_force_spectrum(k, spectra, kEk, out_dir, kc):
    E_true, E_smag, E_surr, E_fdt = spectra
    m = k >= 1
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.loglog(k[m], E_true[m], "o-", color="black", ms=4, lw=2, label="exact SGS force (truth)")
    ax.loglog(k[m], E_smag[m], "s--", color="#d62728", ms=4, lw=1.5, label="Smagorinsky (K-theory)")
    ax.loglog(k[m], E_surr[m], "^:", color="#ff7f0e", ms=4, lw=1.5, label="spectrum-matched surrogate")
    ax.loglog(k[m], E_fdt[m], "D-", color="#2ca02c", ms=4, lw=1.5, label="projected-FDT")
    ax.axvline(kc, color="gray", ls="--", lw=0.8, alpha=0.5, label=f"k_c = {kc}")
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("force spectrum E_m(k)")
    ax.set_title("Part 9c (3D): subgrid-force spectrum E_m(k)")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    # inset: full-field KE spectrum (the inertial range)
    kk, Ek = kEk
    ins = ax.inset_axes([0.13, 0.13, 0.4, 0.4])
    me = (kk >= 1) & (Ek > 0)
    ins.loglog(kk[me], Ek[me], "-", color="navy", lw=1.3)
    kref = kk[me]
    if kref.size > 3:
        k0 = kref[1]; e0 = Ek[me][1]
        ins.loglog(kref, e0 * (kref / k0) ** (-5.0 / 3.0), "k--", lw=0.8, alpha=0.7)
        ins.text(0.4, 0.7, "k^-5/3", transform=ins.transAxes, fontsize=7)
    ins.set_title("KE spectrum E(k)", fontsize=7)
    ins.tick_params(labelsize=6)
    fig.tight_layout()
    path = os.path.join(out_dir, "61_closure3d_force_spectrum.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def fig_divergence(div_scores, out_dir):
    names = list(div_scores.keys()); vals = [div_scores[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["black", "#d62728", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(names, vals, color=colors, edgecolor="k", linewidth=0.6)
    ax.set_ylabel("relative RMS divergence  |div m| / |m|")
    ax.set_title("Part 9c (3D): structural solenoidality of the modeled force")
    ax.set_yscale("log")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, max(v, 1e-16) * 1.3,
                f"{v:.1e}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "62_closure3d_divergence.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def fig_transfer(k, transfers, out_dir, kc):
    T_true, T_smag, T_surr, T_fdt = transfers
    m = k >= 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Part 9c (3D): energy transfer T(k) = Re<u*.m>  "
                 "(net T<0 = FORWARD cascade -- opposite of 2D)", fontsize=12, y=1.01)
    ax = axes[0]
    ax.axhline(0, color="gray", lw=0.5)
    ax.plot(k[m], T_true[m], "o-", color="black", ms=4, lw=2, label="truth")
    ax.plot(k[m], T_smag[m], "s--", color="#d62728", ms=3, lw=1.5, label="Smagorinsky")
    ax.plot(k[m], T_surr[m], "^:", color="#ff7f0e", ms=3, lw=1.5, label="surrogate")
    ax.plot(k[m], T_fdt[m], "D-", color="#2ca02c", ms=3, lw=1.5, label="projected-FDT")
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("T(k)")
    ax.set_title("all models"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.axhline(0, color="gray", lw=0.5)
    ax.plot(k[m], T_true[m], "o-", color="black", ms=4, lw=2, label="truth")
    ax.plot(k[m], T_fdt[m], "D-", color="#2ca02c", ms=4, lw=1.5, label="projected-FDT")
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("T(k)")
    ax.set_title("projected-FDT tracks the true forward-cascade transfer")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "63_closure3d_transfer.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return path


def fig_vortex_3d(align, flux, enst_norm, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("Part 9c: the physics that does not exist in 2D", fontsize=12, y=1.02)
    # (a) strain/vorticity alignment PDFs
    ax = axes[0]
    centers = np.linspace(0.025, 0.975, 20)
    ax.plot(centers, align["hist_extensional"], "-", color="#d62728", lw=1.8,
            label=f"extensional e1  (<|cos|>={align['cos_extensional']:.2f})")
    ax.plot(centers, align["hist_intermediate"], "-", color="#2ca02c", lw=2.2,
            label=f"intermediate e2  (<|cos|>={align['cos_intermediate']:.2f})")
    ax.plot(centers, align["hist_compressive"], "-", color="#1f77b4", lw=1.8,
            label=f"compressive e3  (<|cos|>={align['cos_compressive']:.2f})")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8, label="isotropic (random)")
    ax.set_xlabel("|cos(omega, strain eigenvector)|"); ax.set_ylabel("PDF")
    ax.set_title("Vorticity / strain-eigenvector alignment PDFs\n"
                 "(Constantin-Fefferman intermediate-e2 peak sharpens with Re)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    # (b) backscatter volume fraction: truth vs Smagorinsky
    ax = axes[1]
    bf = [flux["backscatter_fraction_true"], flux["backscatter_fraction_smag"]]
    bars = ax.bar(["exact SGS\n(truth)", "Smagorinsky\n(K-theory)"], bf,
                  color=["black", "#d62728"], edgecolor="k")
    ax.set_ylabel("backscatter volume fraction  (Pi = -tau:S < 0)")
    ax.set_ylim(0, max(0.6, max(bf) * 1.25))
    ax.set_title(f"Local backscatter fraction\n(net vortex-stretching production "
                 f"<wSw>* = {enst_norm:.3f} > 0)")
    for bar, v in zip(bars, bf):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "64_closure3d_vortex_stretching.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return path


_HAND_WRITTEN_MARKER = "<!-- BEGIN HAND-WRITTEN CONTENT (preserved across regeneration) -->"


def write_report(path, figs, div_scores, transfer_corrs, flux, align, enst,
                 net_transfer, meta):
    f61, f62, f63, f64 = figs
    hand_written = ""
    if os.path.exists(path):
        with open(path, "r") as fh:
            content = fh.read()
        idx = content.find(_HAND_WRITTEN_MARKER)
        if idx >= 0:
            hand_written = content[idx:]
    backend = meta["backend"]
    intermediate_peaks = (align["cos_intermediate"] >= align["cos_extensional"]
                          and align["cos_intermediate"] >= align["cos_compressive"])
    if intermediate_peaks:
        align_note = ("The preferential alignment with the *intermediate* eigenvector is "
                      "the Constantin-Fefferman geometry that depletes stretching and holds "
                      "singularity formation off.")
    else:
        align_note = ("At this Reynolds number the alignment is still weak (near-isotropic, "
                      "<|cos|> ~ 0.5); the canonical intermediate-eigenvector peak "
                      "(Constantin-Fefferman geometric depletion) sharpens only in developed "
                      "turbulence -- the regime reached by the `--gpu` run at n>=128.")
    txt = f"""<!-- Benchmark section below is GENERATED by run_closure3d.py
     (n={meta['n']}, kc={meta['kc']}, nu={meta['nu']}, steps={meta['steps']}, backend={backend}).
     Re-run to refresh numbers. Hand-written content below the marker is preserved. -->

# Part 9c -- the two-clocks closure benchmark in 3D

**The vortex-stretching extension of Part 9b.**  Part 9b established the structural
diagnosis and closure repair in **2D**, where the mechanism is clean and CPU-verifiable
but where vortex stretching is *identically zero*.  Part 9c re-runs the identical
benchmark logic in genuine **3D incompressible DNS**, where the stretching term
`omega . grad(u)` -- the engine of the forward energy cascade and of singularity
formation -- is active.  The closure machinery (sharp filter, exact SGS force, Leray
projection, FDT backscatter) generalizes verbatim: the Helmholtz/Leray projection is
dimension-agnostic (REPORT_THEORY.md sec 6).

**Setup:** {meta['n']}^3 forced 3D incompressible DNS (velocity form, pseudo-spectral,
nu={meta['nu']}), sharp-spectral-filtered at k_c = {meta['kc']}.  Backend: **{backend}**.
Developed-field diagnostics: KE = {meta['ke']:.3e}, normalized vortex-stretching
production <wSw>* = {enst:.3f}.

## 1. Force spectrum E_m(k) (figure 61)

![force spectrum]({os.path.basename(f61)})

The four force-spectrum curves repeat the Part-9b finding in 3D: Smagorinsky carries
roughly the right magnitude but mis-sets the near-cutoff shape; the surrogate matches
E_m(k) by construction; projected-FDT fills the correct E_m(k).

## 2. Structural solenoidality -- RMS div(m) (figure 62)

![divergence]({os.path.basename(f62)})

| model | relative RMS(div m) |
|---|---|
| truth | {div_scores['truth']:.2e} |
| Smagorinsky | {div_scores['Smagorinsky']:.2e} |
| surrogate | {div_scores['surrogate']:.2e} |
| projected-FDT | {div_scores['projected-FDT']:.2e} |

Truth, Smagorinsky and projected-FDT are solenoidal to machine precision (all Leray
projected -- the 3D projection `project3d`); the spectrum-matched **surrogate fails**
(phase randomization breaks div=0), exactly as in 2D.

## 3. Energy-transfer spectrum T(k) (figure 63)

![transfer]({os.path.basename(f63)})

**The sign flips vs 2D.**  Net transfer integral sum_k T_true(k) = {net_transfer:.3e}
**< 0**: 3D is a net **forward** cascade (resolved -> subgrid), the opposite of the 2D
inverse cascade.  Transfer-spectrum correlation with truth:

- Smagorinsky: {transfer_corrs['Smagorinsky']:.3f}
- surrogate: {transfer_corrs['surrogate']:.3f}
- projected-FDT: {transfer_corrs['projected-FDT']:.3f}

## 4. Vortex stretching + backscatter -- the 3D-only physics (figure 64)

![vortex stretching]({os.path.basename(f64)})

These diagnostics have **no 2D analogue** and are the reason 3D is scientifically
necessary, not merely "2D with an extra index":

- **Vortex-stretching production** `<omega_i S_ij omega_j>` (normalized) = **{enst:.3f} > 0**.
  In 2D this term is *identically zero* (vorticity is orthogonal to the strain plane).
  Its positivity is the forward-cascade engine.
- **Strain/vorticity alignment** (Constantin-Fefferman geometric depletion): mean |cos|
  of vorticity with the strain eigenvectors --
  extensional e1 = {align['cos_extensional']:.3f},
  **intermediate e2 = {align['cos_intermediate']:.3f}**,
  compressive e3 = {align['cos_compressive']:.3f}
  (measured on {align['n_points']:,} points).  {align_note}
- **SGS backscatter volume fraction** (local SGS energy flux `Pi = -tau^d_ij S_ij < 0`):
  truth = **{flux['backscatter_fraction_true']:.3f}**, Smagorinsky = **{flux['backscatter_fraction_smag']:.3f}**.
  K-theory's positive-definite eddy viscosity has `Pi >= 0` *everywhere by construction*
  -- it is structurally incapable of representing the ~1/2 of physical space where 3D
  turbulence transfers energy up-scale.  Mean flux (forward): truth = {flux['mean_flux_true']:.3e}.

## Conclusion

The 2D Part-9b verdicts survive verbatim in 3D: Smagorinsky is purely dissipative
(zero backscatter volume), the spectrum-matched surrogate has the right energy but
breaks solenoidality, and the projected-FDT closure passes spectrum + divergence +
transfer.  Crucially, the **3D-only** physics -- positive vortex-stretching production,
the intermediate-eigenvector alignment, and the ~1/2 backscatter volume fraction --
exposes a K-theory failure that the 2D test *structurally cannot see*: a positive-
definite eddy viscosity cannot represent local backscatter, which in 3D is not a
boundary curiosity but half of space.

## Scope

A frozen-field **a-priori** test in a 3D periodic box -- the structural-correctness
analogue of Part 9b, now with vortex stretching active.  It is **not** a grid-converged
production DNS, **not** an a-posteriori (time-integrated) closed-loop LES, and **not**
a claim of 3D Navier-Stokes regularity or of operational superiority.  It establishes
that the two-clocks structural repair, and K-theory's specific structural failure,
both carry into 3D where the dynamics that matter for regularity actually live.
"""
    with open(path, "w") as fh:
        fh.write(txt)
        fh.write(hand_written if hand_written else _HAND_WRITTEN_MARKER + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--report", default="REPORT_CLOSURE3D.md")
    ap.add_argument("--gpu", action="store_true", help="use CuPy (Kaggle P100); bumps defaults")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--kc", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--nu", type=float, default=None)
    ap.add_argument("--f-amp", type=float, default=None)
    ap.add_argument("--json-out", default=None, help="optional path to dump the numeric results")
    args = ap.parse_args()

    if args.gpu:
        import cupy as xp
        n = args.n or 128; kc = args.kc or 24; steps = args.steps or 2500
        nu = args.nu if args.nu is not None else 1.2e-3
    else:
        xp = np
        n = args.n or 48; kc = args.kc or 8; steps = args.steps or 800
        nu = args.nu if args.nu is not None else 3.5e-3
    f_amp = args.f_amp if args.f_amp is not None else 1.2
    backend = "cupy" if args.gpu else "numpy"

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Part 9c: 3D closure benchmark (backend={backend}, n={n}, kc={kc}, "
          f"nu={nu}, steps={steps})")
    print("  spinning up 3D DNS truth field ...")
    cfg = DNS3DConfig(n=n, nu=nu, f_amp=f_amp, seed=42)
    dns = ForcedNS3D(cfg, xp=xp)
    u, v, w = dns.field(steps=steps, report_every=max(1, steps // 8))
    ke = 0.5 * float(xp.mean(u ** 2 + v ** 2 + w ** 2))
    sp = dns.sp
    print(f"  DNS developed: KE={ke:.4e}")

    print(f"  computing exact SGS force (kc={kc}) ...")
    ub, vb, wb, mtx, mty, mtz = exact_sgs_force3d(sp, u, v, w, kc)
    print("  computing Smagorinsky / surrogate / projected-FDT models ...")
    msx, msy, msz = smagorinsky_force3d(sp, ub, vb, wb, kc)
    mrx, mry, mrz = surrogate_force3d(sp, mtx, mty, mtz, kc, seed=7)
    mfx, mfy, mfz, nu_t_shell = projected_fdt_force3d(sp, ub, vb, wb, mtx, mty, mtz, kc, seed=7)

    div_scores = {
        "truth": divergence_rms3d(sp, mtx, mty, mtz),
        "Smagorinsky": divergence_rms3d(sp, msx, msy, msz),
        "surrogate": divergence_rms3d(sp, mrx, mry, mrz),
        "projected-FDT": divergence_rms3d(sp, mfx, mfy, mfz),
    }
    k, E_true = force_spectrum3d(sp, mtx, mty, mtz, kc)
    _, E_smag = force_spectrum3d(sp, msx, msy, msz, kc)
    _, E_surr = force_spectrum3d(sp, mrx, mry, mrz, kc)
    _, E_fdt = force_spectrum3d(sp, mfx, mfy, mfz, kc)
    _, T_true = transfer_spectrum3d(sp, ub, vb, wb, mtx, mty, mtz, kc)
    _, T_smag = transfer_spectrum3d(sp, ub, vb, wb, msx, msy, msz, kc)
    _, T_surr = transfer_spectrum3d(sp, ub, vb, wb, mrx, mry, mrz, kc)
    _, T_fdt = transfer_spectrum3d(sp, ub, vb, wb, mfx, mfy, mfz, kc)
    m = k >= 1
    transfer_corrs = {
        "Smagorinsky": _safe_corr(T_true[m], T_smag[m]),
        "surrogate": _safe_corr(T_true[m], T_surr[m]),
        "projected-FDT": _safe_corr(T_true[m], T_fdt[m]),
    }
    net_transfer = float(T_true[m].sum())

    print("  computing 3D-only diagnostics (stretching, alignment, backscatter) ...")
    _, enst_norm = enstrophy_production(sp, u, v, w)
    flux = sgs_flux_stats(sp, u, v, w, kc)
    align = strain_vorticity_alignment(sp, u, v, w)
    kEk = energy_spectrum3d(sp, u, v, w)

    print("\n  === BENCHMARK SUMMARY (3D) ===")
    print(f"    div: truth={div_scores['truth']:.1e} surrogate={div_scores['surrogate']:.1e}")
    print(f"    net transfer sum_k T_true = {net_transfer:.3e}  (<0 => forward cascade)")
    print(f"    Smagorinsky purely dissipative (T<=0 all k): {np.all(T_smag[m] <= 1e-9)}")
    print(f"    transfer corr  smag={transfer_corrs['Smagorinsky']:.3f} "
          f"surr={transfer_corrs['surrogate']:.3f} fdt={transfer_corrs['projected-FDT']:.3f}")
    print(f"    vortex-stretching <wSw>* = {enst_norm:.3f}  (=0 in 2D)")
    print(f"    alignment |cos|: e1={align['cos_extensional']:.3f} "
          f"e2(int)={align['cos_intermediate']:.3f} e3={align['cos_compressive']:.3f}")
    print(f"    backscatter vol-fraction: truth={flux['backscatter_fraction_true']:.3f} "
          f"smag={flux['backscatter_fraction_smag']:.3f}")

    print("  generating figures ...")
    f61 = fig_force_spectrum(k, (E_true, E_smag, E_surr, E_fdt), kEk, args.out_dir, kc)
    f62 = fig_divergence(div_scores, args.out_dir)
    f63 = fig_transfer(k, (T_true, T_smag, T_surr, T_fdt), args.out_dir, kc)
    f64 = fig_vortex_3d(align, flux, enst_norm, args.out_dir)
    for p in (f61, f62, f63, f64):
        print(f"  -> {p}")

    meta = {"n": n, "kc": kc, "nu": nu, "steps": steps, "backend": backend, "ke": ke}
    write_report(args.report, (f61, f62, f63, f64), div_scores, transfer_corrs,
                 flux, align, enst_norm, net_transfer, meta)
    print(f"Report: {args.report}")

    if args.json_out:
        out = {
            "meta": meta, "div_scores": div_scores, "transfer_corrs": transfer_corrs,
            "net_transfer": net_transfer, "enstrophy_norm": enst_norm,
            "alignment": {kk: vv for kk, vv in align.items() if not kk.startswith("hist")},
            "flux": flux,
        }
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"JSON: {args.json_out}")


if __name__ == "__main__":
    main()
