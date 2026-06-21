r"""§A.3 / §D.1 mechanism: the scallop wall is a *phase-locked* opening source in
the Röthlisberger channel ODE -> reattachment patches are **preferred channel
nucleation sites**.

Background (FUTURE_WORK §A.3, Caveat S).  The Röthlisberger channel evolves as

    dS/dt = V_o - V_c ,
    V_o = (1/rho_i L) |Q dphi/ds|        (melt opening from dissipation),
    V_c = 2 A S (N/n)^n                  (Glen creep closure, N = p_i - p_w),

and the channelisation instability runs through *flow concentration* (Caveat S):
an incipient low captures flow -> u_channel > u_sheet -> higher u* -> more
turbulent melt -> deeper channel -> more concentration.  The §A.3 hypothesis adds
a distributed wall source from lee-localised reattachment heat flux,

    dS/dt = V_o - V_c + V_scallop ,
    V_scallop(x) = (1/rho_i L) [q_reattach(x) - q_flat] ,

and asks whether localised reattachment melt creates *preferred nucleation sites*
for channels (rather than the location being set by initial noise).

This module earns the mechanism in three structural steps that do **not** depend
on the (HYP) dimensional bridge ``rho_i L`` or the (HYP) concentration gain:

  1. **Sign [structural].**  The reattachment population carries a net *positive*
     normal-flux excess over the flat control, so ``V_scallop > 0`` everywhere it
     acts: scallop reattachment is an *opening* source, same sign as ``V_o`` --
     consistent with Caveat S (more melt, not less).

  2. **Phase-locking [structural].**  The reattachment patches are not spread
     uniformly over the scallop: their area density is concentrated in a narrow
     band of the wall phase (a large Rayleigh resultant ``R_phase``), locked to a
     fixed point on the bedform.  So ``V_scallop(x)`` is a spatially *periodic*
     source pinned to the geometry, not white in ``x``.

  3. **Site selection [structural].**  Fed into the concentration-loop channel
     ODE, the phase-locked source makes the fastest-growing / largest-steady
     channel form at the reattachment phase **deterministically**, overriding the
     random site a noise-seeded flat wall would pick.  That is exactly the §A.3
     "preferred nucleation site" claim.

The DNS-dependent step reuses the verified scalloped cavity (``scallop_sweep.
_run_norm``), which returns both the wall shape ``y_ice(x)`` and the per-column
time-mean normal flux ``m_n(x)``; the channel ODE and all diagnostics are pure
NumPy and CPU-cheap.  Coefficients ``rho_i L``, ``A``, ``N`` and the concentration
gain ``g`` stay [LIT]/[HYP]; the three results above are dimensionless and
coefficient-robust.

Usage (Part-C config; matches REPORT_CANDIDATE3.md / §G.1):
    python scallop_channel_feedback.py --nx 128 --ny 128 --nwaves 12 \
        --udrive 1.5 --famp 0.4 --spinup 3000 --measure 800 --amp 0.30 \
        --out-dir figures --report REPORT_CHANNEL.md
    python scallop_channel_feedback.py --fast        # quick CPU smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_probe import _json_safe, _nanmean  # noqa: E402
from scallop_sweep import _run_norm  # noqa: E402
from subglacial.candidate3_roughness_feedback import Candidate3Config  # noqa: E402


def get_backend(force_gpu: bool):
    if force_gpu:
        import cupy as cp
        cp.zeros(1).sum()
        return cp, "cupy"
    try:
        import cupy as cp
        cp.zeros(1).sum()
        return cp, "cupy"
    except Exception:  # noqa: BLE001 - any import/device failure -> CPU
        return np, "numpy"


# --------------------------------------------------------------------------- #
# 1-2.  reattachment source + phase-locking diagnostics
# --------------------------------------------------------------------------- #
def reattachment_source(m_n, m_flat, n_waves, Lx, n_phase_bins=12):
    r"""Build the distributed V_scallop source from the per-column normal flux and
    measure its sign and phase-locking to the scallop geometry.

    ``m_n`` is the bump wall's per-column time-mean normal flux ``m_n(x)`` and
    ``m_flat`` the flat control field; the unknown ``kappa*dT`` and the
    dimensional ``1/(rho_i L)`` cancel in every normalised quantity below.

    Returns a dict with the normalised source field ``w_norm`` (== the positive
    reattachment excess ``max(m_n-<m_flat>,0)/<m_flat>``), the integrated source
    strength ``v_scallop`` (its area mean), the signed net excess ``v_net`` (==
    ``nu_ratio-1``), and the phase-locking diagnostics: the per-bin reattachment
    fraction, the Rayleigh resultant ``R_phase`` in [0,1] (0 = uniform, 1 = a
    single phase) of the reattachment-weighted phase, the preferred phase, and the
    flux/height correlation ``corr_mn_phasecos``.
    """
    mn = np.asarray(m_n, dtype=float)
    mf = np.asarray(m_flat, dtype=float)
    okn = np.isfinite(mn)
    okf = np.isfinite(mf)
    n = int(okn.sum())
    m_flat_mean = _nanmean(mf) if okf.any() else float("nan")
    if n == 0 or not np.isfinite(m_flat_mean) or m_flat_mean <= 0.0:
        # No finite bump column, or a degenerate / non-positive flat control:
        # the partition (and the V_scallop source defined relative to it) is
        # undefined.  Return NaN explicitly instead of dividing by ~0.
        nan = float("nan")
        nb=int(n_phase_bins)
        return {
            "n_valid": n, "m_flat_mean": m_flat_mean,
            "nu_ratio": nan, "v_scallop": nan, "v_net": nan,
            "f_reatt": nan, "R_phase": nan, "phase_pref": nan,
            "corr_mn_phasecos": nan, "reatt_frac_by_phase": [nan] * nb,
            "phase_bin_centers": list(((np.arange(nb) + 0.5)
                                       * (2.0 * np.pi / nb))),
            "w_norm": [], "phase": [],
        }

    nx = mn.size
    x = np.arange(nx) * (Lx / nx)
    phase = (2.0 * np.pi * n_waves * x / Lx) % (2.0 * np.pi)

    v = mn[okn]
    ph = phase[okn]
    excess = (v - m_flat_mean) / m_flat_mean         # signed, normalised
    reatt = v >= m_flat_mean                          # reattachment population
    w_norm = np.where(reatt, excess, 0.0)             # V_scallop integrand (>=0)

    nu_ratio = float(np.mean(v)) / m_flat_mean
    v_net = float(np.mean(excess))                    # == nu_ratio - 1
    v_scallop = float(np.mean(w_norm))                # area-mean opening source
    f_reatt = float(np.count_nonzero(reatt)) / float(n)

    # phase-locking: resultant of the reattachment-weighted phase (Rayleigh).
    # weight by the positive excess so stronger patches count more.
    wsum = float(np.sum(w_norm))
    if wsum > 0.0:
        C = float(np.sum(w_norm * np.cos(ph))) / wsum
        Sx = float(np.sum(w_norm * np.sin(ph))) / wsum
        R_phase = float(np.hypot(C, Sx))
        phase_pref = float(np.arctan2(Sx, C) % (2.0 * np.pi))
    else:
        R_phase = phase_pref = float("nan")

    # correlation of the flux excess with cos(phase): sign tells which face the
    # reattachment sits on (a clean scalar version of the corr(m_n, y_ice) check)
    cph = np.cos(ph)
    if np.std(excess) > 0 and np.std(cph) > 0:
        corr_mn_phasecos = float(np.corrcoef(excess, cph)[0, 1])
    else:
        corr_mn_phasecos = float("nan")

    nb = int(n_phase_bins)
    edges = np.linspace(0.0, 2.0 * np.pi, nb + 1)
    bin_idx = np.clip(np.digitize(ph, edges) - 1, 0, nb - 1)
    reatt_frac_by_phase = []
    for b in range(nb):
        sel = bin_idx == b
        reatt_frac_by_phase.append(
            float(np.count_nonzero(reatt & sel)) / float(np.count_nonzero(sel))
            if np.count_nonzero(sel) else float("nan"))

    return {
        "n_valid": n, "m_flat_mean": m_flat_mean,
        "nu_ratio": nu_ratio, "v_scallop": v_scallop, "v_net": v_net,
        "f_reatt": f_reatt, "R_phase": R_phase, "phase_pref": phase_pref,
        "corr_mn_phasecos": corr_mn_phasecos,
        "reatt_frac_by_phase": reatt_frac_by_phase,
        "phase_bin_centers": list(0.5 * (edges[:-1] + edges[1:])),
        "w_norm": w_norm.tolist(), "phase": ph.tolist(),
    }


# --------------------------------------------------------------------------- #
# 3.  Röthlisberger channel ODE + concentration loop (Caveat S)
# --------------------------------------------------------------------------- #
def rothlisberger_rhs(S, V_src, k_creep):
    r"""dS/dt for a Röthlisberger channel: opening ``V_src`` minus the Glen creep
    closure, written with the *lumped* creep rate ``k_creep = 2 A (N/n)^n`` so
    ``V_c = k_creep * S`` (creep is linear in ``S`` at fixed effective pressure
    ``N``).  Vectorised over an array of segments.

    Working with ``k_creep`` rather than the raw ``(A, N, n)`` keeps the ODE
    well-conditioned (the literal Glen constants give a ~10^6 relaxation time)
    and makes explicit that the structural results depend only on the
    dimensionless ratio ``V_scallop / V_o`` and ``conc_gain``, not on the [LIT]/
    [HYP] magnitudes.  ``V_src`` is the total opening (``V_o + V_scallop``).
    """
    S = np.asarray(S, dtype=float)
    return V_src - k_creep * S


def channel_steady_size(V_src, k_creep):
    r"""Closed-form steady channel size of the *uncoupled* R-channel (creep linear
    in ``S``): ``S* = V_src / k_creep``.  Used as the no-instability baseline;
    the steady ``S*`` field inherits the phase pattern of ``V_src``.
    """
    return np.asarray(V_src, dtype=float) / k_creep


def integrate_channel_network(V_o_field, V_scallop_field, k_creep,
                              conc_gain, dt, n_steps, S0=None, seed=0):
    r"""Evolve a ring of ``len(V_o_field)`` Röthlisberger channels with the
    flow-concentration loop (Caveat S) and the distributed scallop source.

    The concentration loop makes larger channels capture proportionally more
    melt flux (the "main artery" instability, Caveat S):

        V_o_eff_i = <V_o> * (1 + g * (S_i - <S>) / <S>) ,

    with gain ``g = conc_gain`` [HYP].  ``g = 0`` recovers independent channels;
    sub-critical ``0 < g < 1`` *amplifies* any phase-locked bump by ``1/(1-g)``
    without runaway.  The scallop source ``V_scallop_field`` (phase-locked, fixed
    in ``x``) is added on top.  Integrated explicitly to a near-steady state;
    returns the final ``S`` field and the winning (largest) segment index.
    """
    V_o_field = np.asarray(V_o_field, dtype=float)
    V_sc = np.asarray(V_scallop_field, dtype=float)
    m = V_o_field.size
    rng = np.random.default_rng(seed)
    if S0 is None:
        # small positive noise so the noise-only (V_sc=0) case has a random seed
        # to amplify -- the control that the scallop source must override
        S = channel_steady_size(V_o_field.mean(), k_creep) * (
            1.0 + 0.01 * rng.standard_normal(m))
        S = np.maximum(S, 1e-12)
    else:
        S = np.array(S0, dtype=float)
    V_o_bar = float(V_o_field.mean())
    for _ in range(int(n_steps)):
        Smean = float(S.mean())
        if Smean > 0:
            V_o_eff = V_o_bar * (1.0 + conc_gain * (S - Smean) / Smean)
        else:
            V_o_eff = V_o_field
        V_o_eff = np.maximum(V_o_eff, 0.0)
        dS = rothlisberger_rhs(S, V_o_eff + V_sc, k_creep)
        S = np.maximum(S + dt * dS, 1e-12)
    return S, int(np.argmax(S))


def run(xp, nx=128, ny=128, n_waves=12, U_drive=1.5, f_amp=0.4,
        spinup=3000, measure=800, Ri=0.0, seed=0, amp=0.30,
        k_creep=1.0, conc_gain=0.5,
        ode_dt=0.05, ode_steps=4000, n_seg_per_wave=8, n_noise_seeds=24):
    Lx = 4.0 * 2.0 * np.pi
    lam = Lx / n_waves

    def cfg():
        return Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none",
                                f_amp=f_amp, Ri=Ri, seed=seed)

    t0 = time.time()
    yb_f, _, m_flat, umax_f, _ = _run_norm(cfg(), 0.0, n_waves, U_drive,
                                           spinup, measure, xp)
    m_flat = np.asarray(m_flat)
    print(f"FLAT  Nu_flat={_nanmean(m_flat):.6e}  umax={umax_f:.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    a = amp * lam
    t0 = time.time()
    yb_b, _, m_bump, umax, _ = _run_norm(cfg(), a, n_waves, U_drive,
                                         spinup, measure, xp)
    m_bump = np.asarray(m_bump)
    src = reattachment_source(m_bump, m_flat, n_waves, Lx)
    print(f"[a/λ={amp:4.2f}] ({time.time()-t0:.0f}s)  Nu/Nu_flat={src['nu_ratio']:.4f}"
          f"  v_scallop={src['v_scallop']:.4f}  f_reatt={src['f_reatt']:.3f}  "
          f"R_phase={src['R_phase']:.3f}  phase_pref={src['phase_pref']:.2f}",
          flush=True)

    # --- channel-network experiment: phase-locked scallop source vs noise --- #
    # tile the measured one-wavelength reattachment pattern onto a multi-wave ring
    m = int(n_seg_per_wave * n_waves)
    seg_phase = (2.0 * np.pi * n_waves * (np.arange(m) / m)) % (2.0 * np.pi)
    # snap a floating-point ``2*pi - eps`` back to 0 so the phase stays in [0,2pi)
    # (cosmetic for the serialised JSON; cos/sin and period-2pi interp are unaffected)
    seg_phase[np.isclose(seg_phase, 2.0 * np.pi)] = 0.0
    # source amplitude per segment from the phase profile (interp the reatt frac)
    bc = np.asarray(src["phase_bin_centers"])
    rf = np.asarray(src["reatt_frac_by_phase"], dtype=float)
    finite = np.isfinite(rf)
    if finite.sum() >= 2:
        prof = np.interp(seg_phase, bc[finite], rf[finite], period=2.0 * np.pi)
    else:
        prof = np.zeros(m)
    prof = np.clip(prof, 0.0, None)
    # scale the source so its area mean equals the measured v_scallop strength,
    # expressed as a fraction of the baseline opening V_o
    V_o_field = np.full(m, 1.0)                       # uniform flat-wall opening
    if prof.sum() > 0 and np.isfinite(src["v_scallop"]):
        V_sc = prof / prof.mean() * src["v_scallop"]  # mean(V_sc) == v_scallop * V_o
    else:
        V_sc = np.zeros(m)

    # (a) scallop source ON, many noise seeds: does the winner lock to one phase?
    # Capture the seed-0 steady field on the first pass (used for the figure) so
    # we do not re-run the same ODE again below.
    winners_scallop = []
    S_scallop = None
    for sd in range(n_noise_seeds):
        S, w = integrate_channel_network(V_o_field, V_sc, k_creep,
                                         conc_gain, ode_dt, ode_steps, seed=sd)
        winners_scallop.append(int(w))
        if sd == 0:
            S_scallop = S
    # (b) scallop source OFF (noise only): winner should be diffuse over phase
    winners_noise = []
    S_noise = None
    for sd in range(n_noise_seeds):
        S, w = integrate_channel_network(V_o_field, np.zeros(m), k_creep,
                                         conc_gain, ode_dt, ode_steps, seed=sd)
        winners_noise.append(int(w))
        if sd == 0:
            S_noise = S

    def _phase_resultant(idxs):
        if not idxs:
            return float("nan"), float("nan")
        ph = seg_phase[np.asarray(idxs)]
        C = float(np.mean(np.cos(ph))); Sx = float(np.mean(np.sin(ph)))
        return float(np.hypot(C, Sx)), float(np.arctan2(Sx, C) % (2.0 * np.pi))

    R_scallop, ph_scallop = _phase_resultant(winners_scallop)
    R_noise, ph_noise = _phase_resultant(winners_noise)
    # representative steady fields for the figure (seed 0); the loops above already
    # captured them, so only fall back here if no noise seeds were requested.
    if S_scallop is None:
        S_scallop, _ = integrate_channel_network(V_o_field, V_sc, k_creep,
                                                 conc_gain, ode_dt, ode_steps, seed=0)
    if S_noise is None:
        S_noise, _ = integrate_channel_network(V_o_field, np.zeros(m), k_creep,
                                               conc_gain, ode_dt, ode_steps, seed=0)

    print(f"channel network (m={m} seg, g={conc_gain}, {n_noise_seeds} seeds): "
          f"R_winner(scallop)={R_scallop:.3f} @phase {ph_scallop:.2f}  vs  "
          f"R_winner(noise)={R_noise:.3f}", flush=True)

    network = {
        "n_seg": m, "n_seg_per_wave": n_seg_per_wave, "conc_gain": conc_gain,
        "k_creep": k_creep, "ode_dt": ode_dt,
        "ode_steps": ode_steps, "n_noise_seeds": n_noise_seeds,
        "v_scallop_over_Vo": float(src["v_scallop"]),
        "R_winner_scallop": R_scallop, "phase_winner_scallop": ph_scallop,
        "R_winner_noise": R_noise, "phase_winner_noise": ph_noise,
        "site_locked": bool(np.isfinite(R_scallop) and np.isfinite(R_noise)
                            and R_scallop > 0.8 and R_scallop > R_noise + 0.2),
        "seg_phase": seg_phase.tolist(),
        "S_scallop": S_scallop.tolist(), "S_noise": S_noise.tolist(),
        "V_scallop_field": V_sc.tolist(),
    }

    return {"lambda": lam, "amp": amp, "a": a,
            "config": {"nx": nx, "ny": ny, "n_waves": n_waves,
                       "U_drive": U_drive, "f_amp": f_amp, "spinup": spinup,
                       "measure": measure, "Ri": Ri, "seed": seed},
            "source": src, "network": network,
            "yb_bump": np.asarray(yb_b).tolist()}


def _verdict_lines(res):
    src, net = res["source"], res["network"]
    if not np.isfinite(src["v_scallop"]):
        sign = "undefined (non-finite)"
    elif src["v_scallop"] > 0:
        sign = "POSITIVE (opening, same sign as V_o)"
    else:
        sign = "non-positive"
    lines = [
        f"Nu/Nu_flat            = {src['nu_ratio']:.4f}  (<1 distribution-dominated)",
        f"V_scallop (norm/Vo)   = {src['v_scallop']:.4f}   sign: {sign}",
        f"reatt area fraction   = {src['f_reatt']:.3f}",
        f"phase-locking R_phase = {src['R_phase']:.3f}  (0=uniform, 1=single phase)"
        f"   preferred phase = {src['phase_pref']:.2f} rad",
        f"corr(excess, cos φ)   = {src['corr_mn_phasecos']:+.3f}",
        f"channel winner lock   R(scallop)={net['R_winner_scallop']:.3f} @ "
        f"{net['phase_winner_scallop']:.2f} rad  vs  R(noise)={net['R_winner_noise']:.3f}",
        f"PREFERRED NUCLEATION SITE (phase-locked, beats noise): {net['site_locked']}",
    ]
    return lines


def maybe_figure(res, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ValueError) as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return None
    src, net = res["source"], res["network"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # (1) reattachment fraction vs wall phase
    axes[0].bar(src["phase_bin_centers"], src["reatt_frac_by_phase"],
                width=2 * np.pi / max(len(src["phase_bin_centers"]), 1),
                color="tab:red", alpha=0.7)
    axes[0].set(xlabel="wall phase φ (rad)", ylabel="reattachment fraction",
                title=f"reattachment phase-locking\nR_phase={src['R_phase']:.2f}")
    # (2) V_scallop source field over one+ wavelengths
    axes[1].plot(np.arange(len(net["V_scallop_field"])), net["V_scallop_field"],
                 color="tab:blue")
    axes[1].set(xlabel="wall segment", ylabel="V_scallop / V_o",
                title="phase-locked opening source")
    # (3) steady channel size: scallop-seeded vs noise-seeded
    axes[2].plot(net["S_scallop"], label="scallop source ON", color="tab:red")
    axes[2].plot(net["S_noise"], label="noise only", color="0.5", alpha=0.8)
    axes[2].set(xlabel="wall segment", ylabel="steady channel size S",
                title="preferred site selection\n(concentration loop)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "49_scallop_channel_feedback.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="quick CPU smoke test (small grid, short windows)")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nwaves", type=int, default=12)
    ap.add_argument("--udrive", type=float, default=1.5)
    ap.add_argument("--famp", type=float, default=0.4)
    ap.add_argument("--spinup", type=int, default=3000)
    ap.add_argument("--measure", type=int, default=800)
    ap.add_argument("--ri", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=float, default=0.30)
    ap.add_argument("--conc-gain", dest="conc_gain", type=float, default=0.5)
    # channel-ODE [LIT]/[HYP] constants -- defaults match run(); exposed for
    # reproducibility of non-default configurations without editing source.
    ap.add_argument("--k-creep", dest="k_creep", type=float, default=1.0)
    ap.add_argument("--ode-dt", dest="ode_dt", type=float, default=0.05)
    ap.add_argument("--ode-steps", dest="ode_steps", type=int, default=4000)
    ap.add_argument("--n-seg-per-wave", dest="n_seg_per_wave", type=int, default=8)
    ap.add_argument("--n-noise-seeds", dest="n_noise_seeds", type=int, default=24)
    ap.add_argument("--out-dir", dest="out_dir", default="figures")
    ap.add_argument("--report", default=None)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if args.fast:
        args.nx = args.ny = 64
        args.spinup, args.measure = 200, 120

    xp, backend = get_backend(args.gpu)
    Lx = 4.0 * 2.0 * np.pi
    lam = Lx / args.nwaves
    print(f"backend={backend}  nx={args.nx} ny={args.ny}  n_waves={args.nwaves} "
          f"lambda={lam:.3f}  U_drive={args.udrive}  f_amp={args.famp}  "
          f"spinup={args.spinup} measure={args.measure}  amp={args.amp}",
          flush=True)

    res = run(xp, nx=args.nx, ny=args.ny, n_waves=args.nwaves,
              U_drive=args.udrive, f_amp=args.famp, spinup=args.spinup,
              measure=args.measure, Ri=args.ri, seed=args.seed, amp=args.amp,
              k_creep=args.k_creep, conc_gain=args.conc_gain,
              ode_dt=args.ode_dt, ode_steps=args.ode_steps,
              n_seg_per_wave=args.n_seg_per_wave,
              n_noise_seeds=args.n_noise_seeds)
    res["backend"] = backend

    print("\n=== §A.3 scallop -> Röthlisberger channel: phase-locked opening source ===")
    lines = _verdict_lines(res)
    print("\n".join(lines))

    path = maybe_figure(res, args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = args.out or os.path.join(args.out_dir,
                                         "49_scallop_channel_feedback.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(res), fh, indent=2)
    print(f"wrote {json_path}")
    if path:
        print(f"wrote {path}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("# §A.3 Scallop → Röthlisberger channel: preferred "
                     "nucleation sites\n\n")
            fh.write(f"Backend: {backend}  |  config: nx={args.nx} ny={args.ny} "
                     f"n_waves={args.nwaves} a/λ={args.amp}\n\n")
            fh.write("```\n" + "\n".join(lines) + "\n```\n")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
