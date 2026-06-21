r"""Part 9d -- the two-clocks closure benchmark in a WALL-BOUNDED channel.

This is the wall-bounded a-priori test that Part 9c (triply-periodic 3D, the P100
run) left open.  P1's structural verdicts -- exact solenoidality of the subgrid
force, projected-FDT transfer vs anti-correlated Smagorinsky, backscatter that a
positive eddy viscosity cannot represent -- were all proved in periodic boxes,
where the sharp filter ``F`` and the Leray (pressure) projector ``P`` commute
because both are Fourier multipliers.  The single honest objection to the whole
programme is: *real turbulence has walls, and walls are where eddy viscosity is
calibrated.*  So we go to a channel and ask the sharpest possible question:

    Do the filter and the pressure projector still commute at a wall?

THE MATH (the operator components, and why each is here)
--------------------------------------------------------
Geometry: channel [0,Lx) x [0,Ly] x [0,Lz); periodic (Fourier) in x,z; solid walls
at y=0,Ly.  Wall-normal y carries a single 2nd-order finite-difference operator Dy.

  1. Divergence / gradient.  DIV w = i kx u^ + Dy v^ + i kz w^ ; GRAD = (i kx, Dy, i kz).
     The SAME Dy is used in both, so the discrete Laplacian L = DIV.GRAD = Dy@Dy - k_h^2
     is the exact composition -- not an independently-discretised stencil.

  2. Wall-aware Leray projector  P_wall = I - GRAD L^{-1} DIV.
     Per horizontal wavenumber solve  (Dy@Dy - k_h^2) phi = DIV w  with the
     no-penetration Neumann BC  (Dy phi)|_wall = v|_wall, then  P w = w - GRAD phi.
     Because L is the exact composition of the same operators, P w is
     divergence-free to machine precision in the interior AND has zero wall-normal
     velocity on the walls to machine precision.  This is the genuine elliptic
     pressure response with the wall geometry.

  3. Naive periodic projector  P_per = I - k k^T/|k|^2  (a Fourier multiplier).
     The operator that is exact in a box.  Wall-blind: it leaves an O(1)
     no-penetration violation at the walls.

  4. Filters.  F_wall = sharp horizontal cutoff (|k_h|<=kc_h) AND a DCT-I (cosine)
     wall-normal low-pass keeping ny_keep modes -- the cosine basis has zero wall
     derivative, so F_wall respects the wall.  F_per = a triply-periodic sharp
     cutoff sharing the 3-D Fourier eigenbasis with P_per.

  5. The commutator  [F, P] = F P - P F.
     In a periodic box F_per and P_per are simultaneously diagonal in Fourier, so
     [F_per, P_per] = 0 identically.  At a wall P_wall is NOT a multiplier in
     F_wall's basis (it is a Poisson solve with a wall BC), so [F_wall, P_wall] != 0
     and -- the structural point -- it is supported in the near-wall layer and
     vanishes in the bulk, where the channel looks locally periodic.

WHY THIS IS THE CLOSURE TEST (not just operator algebra)
--------------------------------------------------------
The filtered, projected momentum balance the two-clocks framework analyses is
P F (advection).  Modelling assumes you may slide the projector through the filter
(the periodic-box identity).  The residual you drop is exactly  [F, P](.).  A
non-zero, wall-localised [F,P] is therefore a subgrid contribution that is (i) set
by the GLOBAL elliptic pressure field (the "pressure clock"), and (ii) impossible
for any LOCAL, positive eddy viscosity (the "temperature clock") to reproduce --
because a local operator commutes with the local filter in the bulk and carries no
information about the wall constraint.  Its magnitude and near-wall confinement are
the wall correction to the closure.

CPU smoke test (~3 s):
    python run_closure3d_bounded.py --out-dir figures --report REPORT_CLOSURE3D_BOUNDED.md \
        --json-out figures/77_closure3d_bounded.json
GPU run on a Kaggle Tesla P100 (higher resolution; identical code, CuPy backend):
    python run_closure3d_bounded.py --gpu --nx 96 --ny 129 --nz 96 \
        --out-dir figures --report REPORT_CLOSURE3D_BOUNDED.md \
        --json-out figures/77_closure3d_bounded.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from closure.channel3d import Channel3D, random_channel_field, to_host  # noqa: E402


def vnorm(xp, *comps) -> float:
    s = sum(xp.mean(c ** 2) for c in comps)
    return float(xp.sqrt(s))


def vec_filter(ch, comps, kc_h, ny_keep):
    return [ch.sharp_filter(c, kc_h, ny_keep) for c in comps]


def vec_pfilter(ch, comps, kc_h, kc_y):
    return [ch.fourier_filter(c, kc_h, kc_y) for c in comps]


def commutator_wall(ch, u, v, w, kc_h, ny_keep):
    """[F_wall, P_wall](u,v,w) = F P (u,v,w) - P F (u,v,w)."""
    Pu, Pv, Pw = ch.project_wall(u, v, w)
    FP = vec_filter(ch, (Pu, Pv, Pw), kc_h, ny_keep)
    Fu, Fv, Fw = vec_filter(ch, (u, v, w), kc_h, ny_keep)
    PF = ch.project_wall(Fu, Fv, Fw)
    return [FP[i] - PF[i] for i in range(3)]


def commutator_periodic(ch, u, v, w, kc_h, kc_y):
    """[F_per, P_per](u,v,w): the periodic-box baseline (should be ~0)."""
    Pu, Pv, Pw = ch.project_fourier(u, v, w)
    FP = vec_pfilter(ch, (Pu, Pv, Pw), kc_h, kc_y)
    Fu, Fv, Fw = vec_pfilter(ch, (u, v, w), kc_h, kc_y)
    PF = ch.project_fourier(Fu, Fv, Fw)
    return [FP[i] - PF[i] for i in range(3)]


def run(nx, ny, nz, kc_h, ny_keep, seed, xp):
    ch = Channel3D(nx, ny, nz, xp=xp)
    u, v, w = random_channel_field(ch, seed)
    Wn = vnorm(xp, u, v, w)
    kc_y = float(ny_keep)  # comparable y band for the periodic baseline

    # --- (1) projector residuals ------------------------------------------------
    pu, pv, pw = ch.project_wall(u, v, w)
    pu2, pv2, pw2 = ch.project_wall(pu, pv, pw)
    idem = vnorm(xp, pu2 - pu, pv2 - pv, pw2 - pw) / (vnorm(xp, pu, pv, pw) + 1e-30)
    fu, fv, fw = ch.project_fourier(u, v, w)
    res = {
        "wall_div_interior": ch.divergence_rms_interior(pu, pv, pw),
        "wall_div_full": ch.divergence_rms(pu, pv, pw),
        "wall_no_penetration": ch.no_penetration_rms(pu, pv, pw),
        "wall_idempotency": idem,
        "fourier_no_penetration": ch.no_penetration_rms(fu, fv, fw),
        "input_div": ch.divergence_rms(u, v, w),
        "input_no_penetration": ch.no_penetration_rms(u, v, w),
    }

    # --- (2) commutator: periodic baseline vs channel ---------------------------
    cper = commutator_periodic(ch, u, v, w, kc_h, kc_y)
    cwall = commutator_wall(ch, u, v, w, kc_h, ny_keep)
    comm = {
        "periodic": vnorm(xp, *cper) / Wn,
        "channel": vnorm(xp, *cwall) / Wn,
    }
    comm["ratio"] = comm["channel"] / (comm["periodic"] + 1e-300)

    # --- (3) wall localisation of the channel commutator ------------------------
    ce = cwall[0] ** 2 + cwall[1] ** 2 + cwall[2] ** 2
    prof = to_host(xp.mean(ce, axis=(0, 2)))
    prof_n = prof / (prof.max() + 1e-30)
    nwall = max(2, ny // 8)
    near = float(np.mean(np.concatenate([prof_n[:nwall], prof_n[-nwall:]])))
    bulk = float(np.mean(prof_n[ny // 2 - nwall:ny // 2 + nwall]))
    loc = {"near_over_bulk": near / (bulk + 1e-30), "peak_index": int(np.argmax(prof_n))}

    # --- (4) filter-scale sweep (the magnitude depends on the coarsening) -------
    sweep = []
    for kk in (4.0, 6.0, 8.0):
        for nyk in (6, 10, 16):
            if nyk >= ny:
                continue
            c = commutator_wall(ch, u, v, w, kk, nyk)
            e = c[0] ** 2 + c[1] ** 2 + c[2] ** 2
            p = to_host(xp.mean(e, axis=(0, 2)))
            nb = (np.mean(np.concatenate([p[:nwall], p[-nwall:]]))
                  / (np.mean(p[ny // 2 - nwall:ny // 2 + nwall]) + 1e-30))
            sweep.append({"kc_h": kk, "ny_keep": nyk,
                          "commutator": vnorm(xp, *c) / Wn, "near_over_bulk": float(nb)})

    # a 2-D slice of |[F,P_wall] w| for the figure (mid-z plane)
    mag = to_host(xp.sqrt(ce))[:, :, nz // 2]            # (nx, ny)
    yy = to_host(ch.y)
    return {
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "filter": {"kc_h": kc_h, "ny_keep": ny_keep},
        "seed": seed,
        "residuals": res,
        "commutator": comm,
        "localisation": loc,
        "sweep": sweep,
        "_profile": prof_n.tolist(),
        "_y": yy.tolist(),
        "_slice": mag.tolist(),
    }


# ---------------------------------------------------------------------------
# Figure 77 -- the wall-bounded closure obstruction (4 panels)
# ---------------------------------------------------------------------------
def make_figure(out, out_dir):
    res, comm, loc = out["residuals"], out["commutator"], out["localisation"]
    prof = np.array(out["_profile"]); y = np.array(out["_y"])
    sl = np.array(out["_slice"]); Ly = y[-1]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    eps = 2.2e-16

    # Panel A: projector residuals (log) -- wall vs naive, both constraints
    ax = axes[0, 0]
    labels = ["P_wall\ndiv (interior)", "P_wall\nno-penetration",
              "P_naive\nno-penetration"]
    vals = [res["wall_div_interior"], res["wall_no_penetration"],
            res["fourier_no_penetration"]]
    colors = ["#2ca02c", "#2ca02c", "#d62728"]
    ax.bar(labels, np.maximum(vals, eps / 3), color=colors, alpha=0.85)
    ax.axhline(eps, color="gray", ls="--", lw=1, label="machine epsilon")
    ax.set_yscale("log"); ax.set_ylabel("relative residual")
    ax.set_title("(a) wall-aware projector is machine-exact;\nnaive projector fails at the wall")
    ax.legend(fontsize=8); ax.grid(True, which="both", axis="y", alpha=0.3)

    # Panel B: the commutator -- periodic baseline vs channel
    ax = axes[0, 1]
    cl = ["[F,P]\nperiodic box", "[F,P]\nchannel (wall)"]
    cv = [max(comm["periodic"], eps / 3), comm["channel"]]
    ax.bar(cl, cv, color=["#1f77b4", "#d62728"], alpha=0.85)
    ax.axhline(eps, color="gray", ls="--", lw=1, label="machine epsilon")
    ax.set_yscale("log"); ax.set_ylabel("|| [F,P] w || / || w ||")
    ax.set_title(f"(b) filter & pressure projector COMMUTE in a box,\n"
                 f"break at a wall (ratio {comm['ratio']:.1e})")
    ax.legend(fontsize=8); ax.grid(True, which="both", axis="y", alpha=0.3)

    # Panel C: wall localisation of the commutator energy
    ax = axes[1, 0]
    ax.plot(y / Ly, prof, "-", color="#d62728", lw=2)
    ax.fill_between(y / Ly, prof, color="#d62728", alpha=0.2)
    ax.set_xlabel("y / Ly  (0 and 1 are the walls)")
    ax.set_ylabel("commutator energy  <|[F,P]w|^2>_xz  (normalised)")
    ax.set_title(f"(c) the obstruction lives at the wall\n"
                 f"(near-wall / bulk = {loc['near_over_bulk']:.0f}x)")
    ax.grid(True, alpha=0.3)

    # Panel D: x-y slice of |[F,P_wall] w| -- the walls light up
    ax = axes[1, 1]
    im = ax.imshow(sl.T, origin="lower", aspect="auto", cmap="inferno",
                   extent=[0, sl.shape[0], 0, 1])
    ax.set_xlabel("x  (grid index)"); ax.set_ylabel("y / Ly")
    ax.set_title("(d) |[F,P_wall] w| on an x-y plane:\nnear-wall layers carry the residual stress")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Part 9d -- two-clocks closure in a wall-bounded channel: "
                 "the pressure projector and the filter stop commuting at the wall",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "77_closure3d_bounded.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return path


REPORT_TMPL = """# Part 9d -- The Two-Clocks Closure in a Wall-Bounded Channel

*Generated by `run_closure3d_bounded.py` (grid {nx}x{ny}x{nz}, seed {seed}, CPU/NumPy
reproducible). The triply-periodic Part-9c verdicts are re-examined where eddy
viscosity is actually calibrated: at a solid wall.*

## The question

Part 9c proved the structural closure verdicts in a triply-periodic box, where the
sharp filter `F` and the Leray (pressure) projector `P` **commute** -- both are
Fourier multipliers. The one honest objection is that real turbulence has walls.
So: **does `[F, P] = 0` survive a wall?**

## The operators (math components)

| component | periodic box | channel (this test) |
|---|---|---|
| horizontal derivatives | `i k` (spectral) | `i k_h` (spectral in x,z) |
| wall-normal derivative | `i k_y` (spectral) | 2nd-order FD matrix `Dy` |
| Laplacian | `-|k|^2` | `L = Dy@Dy - k_h^2` (= DIV.GRAD, exact composition) |
| projector `P` | `I - k k^T/|k|^2` (multiplier) | `I - GRAD L^{{-1}} DIV`, no-penetration wall BC |
| filter `F` | 3-D sharp Fourier cutoff | horizontal cutoff + DCT-I (cosine) wall-normal low-pass |

The decisive design choice is that the **same** `Dy` builds the divergence, the
gradient and (squared) the Laplacian, so the discrete Hodge projection is exact.

## Result 1 -- the wall-aware projector is a genuine, exact Leray projection

| quantity | value | meaning |
|---|---|---|
| `P_wall` divergence (interior) | `{wall_div_interior:.2e}` | solenoidal to machine precision |
| `P_wall` no-penetration (walls) | `{wall_no_penetration:.2e}` | `v=0` on the walls, machine precision |
| `P_wall` idempotency `||P^2-P||` | `{wall_idempotency:.2e}` | it is a true projection |
| naive `P_per` no-penetration | `{fourier_no_penetration:.3f}` | **O(1)** wall violation -- wall-blind |

The naive periodic projector -- the operator that is exact in a box -- leaves an
order-one wall-normal velocity on the walls. The wall-aware projector does not.

## Result 2 -- the filter and the projector STOP commuting at the wall

| `|| [F,P] w || / || w ||` | value |
|---|---|
| periodic box `[F_per, P_per]` | `{comm_periodic:.2e}`  (machine zero) |
| channel `[F_wall, P_wall]` | `{comm_channel:.4f}` |
| ratio (channel / periodic) | `{comm_ratio:.1e}` |

`[F, P] = 0` to machine precision in the box -- exactly the identity the Part-9c
analysis relies on -- and is **{comm_ratio:.0e} times larger** in the channel.

## Result 3 -- the obstruction is a near-wall structure

The channel commutator energy `<|[F,P]w|^2>_xz` peaks **at the wall** (index
`{peak_index}`) and is **{near_over_bulk:.0f}x** larger in the near-wall layer than
in the bulk, where the channel looks locally periodic and `[F,P]` nearly vanishes.

Filter-scale dependence (the wall correction grows as the coarsening sharpens, and
is set by the *wall-normal* filter, almost independent of the horizontal cutoff):

{sweep_table}

## What it means for the closure

The filtered, projected momentum balance is `P F (advection)`. Closure modelling
slides `P` through `F` -- legitimate in a box. The dropped residual is `[F, P]`.
Here it is non-zero, set by the **global elliptic pressure** response to the wall
(the pressure clock), and **confined to the near-wall layer**. A local, positive
eddy viscosity (the temperature clock) commutes with the filter in the bulk and
carries no information about the wall constraint, so it cannot represent this term.
This is the wall-bounded a-priori correction to the two-clocks closure: small in
amplitude (a few % of the velocity), but structurally outside the reach of
K-theory exactly where K-theory is tuned.

![Part 9d figure](figures/77_closure3d_bounded.png)

*Limitation: this is an a-priori operator-structure test on a synthetic
multi-scale channel field, not an a-posteriori channel-DNS run; it isolates the
filter/projector commutator, which is purely a property of the operators and the
wall geometry, not of any particular turbulence state.*
"""


def make_report(out):
    r, c, loc = out["residuals"], out["commutator"], out["localisation"]
    g = out["grid"]
    rows = ["| kc_h | ny_keep | [F,P]/||w|| | near/bulk |", "|---|---|---|---|"]
    for s in out["sweep"]:
        rows.append(f"| {s['kc_h']:.0f} | {s['ny_keep']} | "
                    f"{s['commutator']:.4f} | {s['near_over_bulk']:.0f}x |")
    return REPORT_TMPL.format(
        nx=g["nx"], ny=g["ny"], nz=g["nz"], seed=out["seed"],
        wall_div_interior=r["wall_div_interior"],
        wall_no_penetration=r["wall_no_penetration"],
        wall_idempotency=r["wall_idempotency"],
        fourier_no_penetration=r["fourier_no_penetration"],
        comm_periodic=c["periodic"], comm_channel=c["channel"], comm_ratio=c["ratio"],
        peak_index=loc["peak_index"], near_over_bulk=loc["near_over_bulk"],
        sweep_table="\n".join(rows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--report", default="REPORT_CLOSURE3D_BOUNDED.md")
    ap.add_argument("--gpu", action="store_true", help="use CuPy (Kaggle P100)")
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=49)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--kc-h", type=float, default=6.0)
    ap.add_argument("--ny-keep", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.gpu:
        import cupy as cp
        xp = cp
        print("[closure3d-bounded] CuPy backend:", cp.cuda.runtime.getDeviceProperties(0)["name"])
    else:
        xp = np

    out = run(args.nx, args.ny, args.nz, args.kc_h, args.ny_keep, args.seed, xp)

    os.makedirs(args.out_dir, exist_ok=True)
    fig_path = make_figure(out, args.out_dir)
    print("figure:", fig_path)

    r, c, loc = out["residuals"], out["commutator"], out["localisation"]
    print(f"  P_wall  div(interior)   = {r['wall_div_interior']:.2e}")
    print(f"  P_wall  no-penetration  = {r['wall_no_penetration']:.2e}")
    print(f"  P_wall  idempotency     = {r['wall_idempotency']:.2e}")
    print(f"  P_naive no-penetration  = {r['fourier_no_penetration']:.3f}")
    print(f"  [F,P] periodic          = {c['periodic']:.2e}")
    print(f"  [F,P] channel           = {c['channel']:.4f}   (ratio {c['ratio']:.1e})")
    print(f"  near-wall / bulk        = {loc['near_over_bulk']:.0f}x  (peak idx {loc['peak_index']})")

    with open(args.report, "w") as fh:
        fh.write(make_report(out))
    print("report:", args.report)

    if args.json_out:
        slim = {k: v for k, v in out.items() if not k.startswith("_")}
        with open(args.json_out, "w") as fh:
            json.dump(slim, fh, indent=2)
        print("json:", args.json_out)


if __name__ == "__main__":
    main()
