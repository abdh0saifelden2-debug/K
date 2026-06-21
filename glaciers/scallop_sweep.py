r"""Full Λ(λ) / a_sat / Nu sweep for the corrected Candidate 3 (scallop /
melting instability), gated by the GO result of ``scallop_probe.py``.

Three measurements, per the co-thinker's spec:

  1. Λ(λ)  -- frozen-boundary enhancement vs wavelength.  Scan ``n_waves``;
     report R_mean(λ) = ⟨m_flow/m_cond⟩ (vs the column's own conduction) and the
     true-Nu ratio Nu(λ)/Nu_flat = ⟨m_n,bump⟩/⟨m_n,flat⟩ with the flow ON, using
     a *local-normal* interfacial gradient (fixes the dT/dy proxy of the probe).
     The wavelength that maximises the enhancement is the scallop-selected mode.

  2. a_sat(λ) -- at the selected λ, release the Stefan moving boundary
     (``update_boundary``) and track the amplitude of the seeded mode.  Growth
     then plateau => self-limiting finite-amplitude scallop; decay => the
     conduction self-smoothing wins.  Two seed amplitudes bracket the fixed
     point.

  3. Seed ensemble -- ≥4 turbulence seeds at the selected (λ, U_drive) to pin the
     Nu magnitude and its uncertainty (the flow is turbulent => realisation
     scatter, as seen CPU↔GPU in the probe).

Closure is fixed to 'none' (the probe showed the effect is resolved-scale and
closure-independent to ≲0.5%); a spot 3-closure check at the optimum is included.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scallop_probe import ProbeFlow, _enh_stats, _json_safe, _nanmean, _run  # noqa: E402
from subglacial.candidate3_roughness_feedback import Candidate3Config  # noqa: E402


# --------------------------------------------------------------------------- #
# local-normal interfacial gradient  m_n = -kappa dT/dn,  n = (-s,1)/sqrt(1+s^2)
# --------------------------------------------------------------------------- #
def melt_normal(s, top_skip=3, span=5):
    """Per-column melt using the true surface-normal gradient (slope-corrected),
    same penalty-clear window as ``melt_field`` but combining dT/dx and dT/dy."""
    cfg = s.cfg
    theta = s._to_host(s.theta)
    fl = s._to_host(s.fluid).astype(bool)
    nx, ny = cfg.nx, cfg.ny
    idx = np.arange(ny)
    jtop = np.where(fl, idx[None, :], -1).max(axis=1)
    cols = np.arange(nx)
    jhi = jtop - top_skip
    jlo = jhi - span
    ok = (jtop >= 0) & (jlo >= 0)
    out = np.full(nx, np.nan)
    jhic = np.clip(jhi, 0, ny - 1)
    Ty = (theta[cols, jhic] - theta[cols, np.clip(jlo, 0, ny - 1)]) / (span * s.sp.dy)
    # interface-row temperature, masked to NaN on invalid columns (where jhi/jlo
    # were clipped to the bed row). The horizontal gradient below uses only valid
    # neighbours, so a valid column adjacent to an invalid one is never
    # contaminated by the clipped bed temperature (one-sided fallback there).
    thj = np.where(ok, theta[cols, jhic], np.nan)
    fwd, bwd = np.roll(thj, -1), np.roll(thj, 1)
    both = np.isfinite(fwd) & np.isfinite(bwd)
    fwd_only = np.isfinite(fwd) & ~np.isfinite(bwd)
    bwd_only = np.isfinite(bwd) & ~np.isfinite(fwd)
    Tx = np.zeros(nx)
    Tx[both] = (fwd[both] - bwd[both]) / (2.0 * s.sp.dx)
    Tx[fwd_only] = (fwd[fwd_only] - thj[fwd_only]) / s.sp.dx
    Tx[bwd_only] = (thj[bwd_only] - bwd[bwd_only]) / s.sp.dx
    yb = s._to_host(s.y_ice_x).astype(float)
    slope = np.gradient(yb, s.sp.dx)
    nrm = np.sqrt(1.0 + slope ** 2)
    dTdn = (-slope * Tx + Ty) / nrm
    out[ok] = (-cfg.kappa * dTdn)[ok]
    return out


def _run_norm(cfg, a, n_waves, U_drive, spinup, measure, xp=np):
    """Like scallop_probe._run but time-averages the *normal* melt too."""
    s = ProbeFlow(cfg, U_drive=U_drive, xp=xp)
    yb = s.set_single_mode(a, n_waves)
    for _ in range(spinup):
        s.step()
    accv = np.zeros(cfg.nx); cntv = np.zeros(cfg.nx)
    accn = np.zeros(cfg.nx); cntn = np.zeros(cfg.nx)
    for _ in range(max(measure, 1)):
        s.step()
        mv = s.melt_field(); mn = melt_normal(s)
        okv = np.isfinite(mv); accv[okv] += mv[okv]; cntv[okv] += 1.0
        okn = np.isfinite(mn); accn[okn] += mn[okn]; cntn[okn] += 1.0
    m_v = np.where(cntv > 0, accv / np.maximum(cntv, 1.0), np.nan)
    m_n = np.where(cntn > 0, accn / np.maximum(cntn, 1.0), np.nan)
    umax = float(s.xp.abs(s.u).max())
    umean = float((s.u * s.fluid).sum() / s.fluid.sum())
    return yb, m_v, m_n, umax, umean


def mode_amp(yb, n_waves, Lx, dx):
    x = np.arange(yb.size) * dx
    ybm = yb - yb.mean()
    a_sin = 2.0 * np.mean(ybm * np.sin(2.0 * np.pi * n_waves * x / Lx))
    a_cos = 2.0 * np.mean(ybm * np.cos(2.0 * np.pi * n_waves * x / Lx))
    return float(np.hypot(a_sin, a_cos))


# --------------------------------------------------------------------------- #
# 1. Lambda(lambda): frozen-boundary enhancement vs wavelength
# --------------------------------------------------------------------------- #
def lambda_scan(nx, ny, n_waves_list, U_drive, spinup, measure, Ri, seed, xp):
    Lx = 4.0 * 2.0 * np.pi
    dx = Lx / nx

    # flat-wall control with the flow ON -> Nu_flat (normal gradient)
    cfgf = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.4, Ri=Ri, seed=seed)
    _, _, mnf, umax_f, _ = _run_norm(cfgf, 0.0, max(n_waves_list), U_drive, spinup, measure, xp)
    Nu_flat = _nanmean(mnf)
    print(f"FLAT Nu_flat(normal)={Nu_flat:.6e} umax={umax_f:.3f}", flush=True)

    rows = []
    for nw in n_waves_list:
        lam = Lx / nw
        a = 0.1 * lam
        t0 = time.time()
        # conduction baseline (flow OFF) on the same bumpy geometry -> m_cond
        cfg0 = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.0, Ri=Ri, seed=seed)
        yb, m_cond, _, _ = _run(cfg0, a, nw, U_drive=0.0, spinup=spinup, xp=xp)
        # flow ON
        cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.4, Ri=Ri, seed=seed)
        yb, m_v, m_n, umax, umean = _run_norm(cfg, a, nw, U_drive, spinup, measure, xp)
        st = _enh_stats(yb, m_v, m_cond, dx)
        Nu_bump = _nanmean(m_n)
        st.update({"n_waves": nw, "lambda": lam, "lambda_over_dx": lam / dx,
                   "a": a, "Nu_bump": Nu_bump, "Nu_flat": Nu_flat,
                   "Nu_ratio": Nu_bump / Nu_flat, "umax": umax, "umean": umean})
        rows.append(st)
        print(f"[nw={nw:2d} λ={lam:.3f} λ/dx={lam/dx:4.1f}] ({time.time()-t0:.0f}s) "
              f"R_mean={st['R_mean']:.4f} R_max={st['R_max']:.3f} "
              f"Nu/Nu_flat={st['Nu_ratio']:.4f} corr={st['corr_excess_slope']:+.3f}",
              flush=True)
    return {"Nu_flat": Nu_flat, "rows": rows}


# --------------------------------------------------------------------------- #
# 2. a_sat(lambda): release the Stefan boundary, track the mode amplitude
# --------------------------------------------------------------------------- #
def a_sat_run(nx, ny, n_waves, U_drive, a0, spinup, n_update_steps, Ri, seed, xp,
              St=None, ice_side=False, ice_kappa_ratio=8.0, T_ice_cold=-1.0,
              n_ice=16):
    Lx = 4.0 * 2.0 * np.pi
    cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.4, Ri=Ri, seed=seed)
    if St is not None:
        cfg.St = St
    if ice_side:
        cfg.ice_side = True
        cfg.ice_kappa_ratio = ice_kappa_ratio
        cfg.T_ice_cold = T_ice_cold
        cfg.n_ice = n_ice
    s = ProbeFlow(cfg, U_drive=U_drive, xp=xp)
    s.set_single_mode(a0, n_waves)
    for _ in range(spinup):
        s.step()
    traj = []; mice_t = []
    acc = np.zeros(nx); cnt = np.zeros(nx)
    for k in range(n_update_steps):
        s.step()
        mi = s.melt_field()
        ok = np.isfinite(mi); acc[ok] += mi[ok]; cnt[ok] += 1.0
        if (k + 1) % cfg.N_update == 0:
            m_mean = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
            s.update_boundary(m=m_mean)
            acc[:] = 0.0; cnt[:] = 0.0
            yb = s._to_host(s.y_ice_x).astype(float)
            traj.append((k + 1, mode_amp(yb, n_waves, Lx, s.sp.dx)))
            if ice_side:
                mice_t.append((k + 1, _nanmean(s.m_ice)))
    return {"a0": a0, "n_waves": n_waves, "St": cfg.St, "ice_side": bool(ice_side),
            "ice_kappa_ratio": ice_kappa_ratio if ice_side else None,
            "T_ice_cold": T_ice_cold if ice_side else None,
            "amp_t": traj, "amp_final": traj[-1][1] if traj else a0,
            "m_ice_t": mice_t}


# --------------------------------------------------------------------------- #

# 2c. amplitude scan at fixed (optimal) lambda: R_mean/R_max/Nu(a/lambda).
#     Closes the z_0(a) gap -- frozen boundary, one wavelength, varying a/lambda.
# --------------------------------------------------------------------------- #
def amp_scan(nx, ny, n_waves, U_drive, spinup, measure, Ri, seed, a_over_lam, xp):
    Lx = 4.0 * 2.0 * np.pi
    dx = Lx / nx
    lam = Lx / n_waves
    # flat-wall control, flow ON -> Nu_flat (normal gradient)
    cfgf = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.4, Ri=Ri, seed=seed)
    _, _, mnf, umax_f, _ = _run_norm(cfgf, 0.0, n_waves, U_drive, spinup, measure, xp)
    Nu_flat = _nanmean(mnf)
    print(f"FLAT Nu_flat(normal)={Nu_flat:.6e} umax={umax_f:.3f} "
          f"(n_waves={n_waves} lam={lam:.3f} lam/dx={lam/dx:.1f})", flush=True)
    rows = []
    for r in a_over_lam:
        a = r * lam
        t0 = time.time()
        cfg0 = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.0, Ri=Ri, seed=seed)
        yb, m_cond, _, _ = _run(cfg0, a, n_waves, U_drive=0.0, spinup=spinup, xp=xp)
        cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.4, Ri=Ri, seed=seed)
        yb, m_v, m_n, umax, umean = _run_norm(cfg, a, n_waves, U_drive, spinup, measure, xp)
        st = _enh_stats(yb, m_v, m_cond, dx)
        Nu_bump = _nanmean(m_n)
        st.update({"a_over_lam": r, "a": a, "n_waves": n_waves, "lambda": lam,
                   "Nu_bump": Nu_bump, "Nu_flat": Nu_flat,
                   "Nu_ratio": Nu_bump / Nu_flat, "umax": umax, "umean": umean})
        rows.append(st)
        print(f"[a/λ={r:.2f} a={a:.3f}] ({time.time()-t0:.0f}s) "
              f"R_mean={st['R_mean']:.4f} R_max={st['R_max']:.3f} "
              f"Nu/Nu_flat={st['Nu_ratio']:.4f} corr={st['corr_excess_slope']:+.3f} "
              f"umax={umax:.3f}", flush=True)
    return {"n_waves": n_waves, "lambda": lam, "Nu_flat": Nu_flat, "rows": rows}
 
 
# --------------------------------------------------------------------------- #
# 3. seed ensemble at the selected lambda
# --------------------------------------------------------------------------- #
def seed_ensemble(nx, ny, n_waves, U_drive, seeds, spinup, measure, Ri, xp):
    Lx = 4.0 * 2.0 * np.pi
    dx = Lx / nx
    lam = Lx / n_waves
    a = 0.1 * lam
    out = []
    for sd in seeds:
        cfg0 = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.0, Ri=Ri, seed=sd)
        yb, m_cond, _, _ = _run(cfg0, a, n_waves, U_drive=0.0, spinup=spinup, xp=xp)
        cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.4, Ri=Ri, seed=sd)
        yb, m_v, m_n, umax, umean = _run_norm(cfg, a, n_waves, U_drive, spinup, measure, xp)
        st = _enh_stats(yb, m_v, m_cond, dx)
        st["Nu_bump"] = _nanmean(m_n); st["seed"] = sd
        out.append(st)
        print(f"  seed={sd} R_mean={st['R_mean']:.4f} R_max={st['R_max']:.3f} "
              f"Nu_bump={st['Nu_bump']:.4e}", flush=True)
    Rm = np.array([r["R_mean"] for r in out])
    return {"n_waves": n_waves, "lambda": lam, "per_seed": out,
            "R_mean_mean": float(Rm.mean()), "R_mean_std": float(Rm.std())}


def main(xp=np, nx=128, ny=128, spinup=3000, measure=800, Ri=0.0,
         n_waves_list=(4, 6, 8, 10, 12, 16, 20, 24), U_drive=1.5,
         a_sat_steps=6000, seeds=(0, 1, 2, 3)):
    out = {"config": {"nx": nx, "ny": ny, "spinup": spinup, "measure": measure,
                      "Ri": Ri, "U_drive": U_drive,
                      "n_waves_list": list(n_waves_list)}}
    print("=== 1. Lambda(lambda) frozen-boundary scan ===", flush=True)
    out["lambda_scan"] = lambda_scan(nx, ny, list(n_waves_list), U_drive,
                                     spinup, measure, Ri, seed=0, xp=xp)
    rows = out["lambda_scan"]["rows"]
    best = max(rows, key=lambda r: r["Nu_ratio"])
    nw_opt = best["n_waves"]
    print(f"--> optimal n_waves={nw_opt} (λ={best['lambda']:.3f}, "
          f"Nu/Nu_flat={best['Nu_ratio']:.4f})", flush=True)

    print("=== 2. a_sat at optimal lambda (Stefan boundary released) ===", flush=True)
    lam = best["lambda"]
    out["a_sat"] = [
        a_sat_run(nx, ny, nw_opt, U_drive, a0=0.10 * lam, spinup=2000,
                  n_update_steps=a_sat_steps, Ri=Ri, seed=0, xp=xp),
        a_sat_run(nx, ny, nw_opt, U_drive, a0=0.20 * lam, spinup=2000,
                  n_update_steps=a_sat_steps, Ri=Ri, seed=0, xp=xp),
    ]
    for r in out["a_sat"]:
        print(f"  a0={r['a0']:.4f} -> amp_final={r['amp_final']:.4f} "
              f"({len(r['amp_t'])} updates)", flush=True)

    print("=== 2b. a_sat WITH ice-side conduction (q_water - q_ice) ===", flush=True)
    out["a_sat_ice"] = [
        a_sat_run(nx, ny, nw_opt, U_drive, a0=0.10 * lam, spinup=2000,
                  n_update_steps=a_sat_steps, Ri=Ri, seed=0, xp=xp, ice_side=True),
        a_sat_run(nx, ny, nw_opt, U_drive, a0=0.20 * lam, spinup=2000,
                  n_update_steps=a_sat_steps, Ri=Ri, seed=0, xp=xp, ice_side=True),
    ]
    for r0, ri in zip(out["a_sat"], out["a_sat_ice"]):
        mice = ri["m_ice_t"][-1][1] if ri["m_ice_t"] else float("nan")
        print(f"  a0={ri['a0']:.4f} -> water-only={r0['amp_final']:.4f} "
              f"ice-side={ri['amp_final']:.4f} (q_ice~{mice:.3e})", flush=True)

    print("=== 3. seed ensemble at optimal lambda ===", flush=True)
    out["seed_ensemble"] = seed_ensemble(nx, ny, nw_opt, U_drive, list(seeds),
                                         spinup, measure, Ri, xp)
    se = out["seed_ensemble"]
    print(f"  Nu/R_mean ensemble: {se['R_mean_mean']:.4f} ± {se['R_mean_std']:.4f}",
          flush=True)

    print("=== 4. closure spot-check at optimal lambda ===", flush=True)
    a = 0.1 * lam
    cfg0 = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs="none", f_amp=0.0, Ri=Ri, seed=0)
    yb, m_cond, _, _ = _run(cfg0, a, nw_opt, U_drive=0.0, spinup=spinup, xp=xp)
    cc = {}
    dx = (4.0 * 2.0 * np.pi) / nx
    for closure in ("none", "smagorinsky", "backscatter"):
        cfg = Candidate3Config(nx=nx, ny=ny, A=4.0, sgs=closure, f_amp=0.4, Ri=Ri, seed=0)
        yb, m_v, m_n, _, _ = _run_norm(cfg, a, nw_opt, U_drive, spinup, measure, xp)
        st = _enh_stats(yb, m_v, m_cond, dx)
        st["Nu_bump"] = _nanmean(m_n)
        cc[closure] = st
        print(f"  {closure:12s} R_mean={st['R_mean']:.4f} Nu_bump={st['Nu_bump']:.4e}",
              flush=True)
    out["closure_check"] = cc
    return out

def amp_main(xp, nx=128, ny=128, n_waves=12, U_drive=1.5, spinup=3000,
             measure=800, Ri=0.0, seed=0,
             a_over_lam=(0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
             long_a_over_lam=(0.20,), long_updates=40000, fast=False):
    """Amplitude scan at fixed lambda (closes z_0(a)) + a long-time ice-memory
    a_sat at the top amplitude(s) to test saturate-vs-oscillate."""
    if fast:
        nx = ny = 64; spinup = 200; measure = 80
        a_over_lam = (0.05, 0.10, 0.20); long_updates = 1200
    out = {"config": {"nx": nx, "ny": ny, "n_waves": n_waves, "U_drive": U_drive,
                      "spinup": spinup, "measure": measure, "Ri": Ri, "seed": seed,
                      "a_over_lam": list(a_over_lam)}}
    print("=== amplitude scan at fixed lambda (closes z_0(a)) ===", flush=True)
    out["amp_scan"] = amp_scan(nx, ny, n_waves, U_drive, spinup, measure, Ri,
                               seed, list(a_over_lam), xp)
    rows = out["amp_scan"]["rows"]
    best = max(rows, key=lambda r: r["R_mean"])
    print(f"--> R_mean peaks at a/λ={best['a_over_lam']:.2f} "
          f"(R_mean={best['R_mean']:.4f}); monotone? "
          f"{[round(r['R_mean'],4) for r in rows]}", flush=True)
 
    print("=== long-time ice-memory a_sat (saturate vs oscillate) ===", flush=True)
    lam = out["amp_scan"]["lambda"]
    out["long_a_sat"] = []
    for r in long_a_over_lam:
        res = a_sat_run(nx, ny, n_waves, U_drive, a0=r * lam, spinup=2000,
                        n_update_steps=long_updates, Ri=Ri, seed=seed, xp=xp,
                        ice_side=True)
        amp_t = res["amp_t"]
        tail = [a for _, a in amp_t[-10:]]
        out["long_a_sat"].append({"a0_over_lam": r, "a0": r * lam,
                                  "amp_final": res["amp_final"],
                                  "n_updates": len(amp_t),
                                  "tail_mean": float(np.mean(tail)) if tail else None,
                                  "tail_std": float(np.std(tail)) if tail else None,
                                  "amp_t": amp_t})
        print(f"  a0/λ={r:.2f} a0={r*lam:.4f} -> amp_final={res['amp_final']:.4f} "
              f"({len(amp_t)} updates) tail={np.mean(tail):.4f}±{np.std(tail):.4f}",
              flush=True)
    return out
 

if __name__ == "__main__":
    use_gpu = "--gpu" in sys.argv
    if use_gpu:
        import cupy as cp
        xp = cp; tag = "gpu"
    else:
        xp = np; tag = "cpu"
    if "--amp" in sys.argv:                 # amplitude scan + long ice-memory
        out = amp_main(xp, fast=("--fast" in sys.argv))
        path = os.path.join(os.getcwd(), f"scallop_ampscan_{tag}.json")
    else:
        kw = {}
        if "--fast" in sys.argv:           # tiny CPU smoke
            kw = dict(nx=64, ny=64, spinup=200, measure=80,
                      n_waves_list=(6, 12, 20), a_sat_steps=400, seeds=(0, 1))
        out = main(xp=xp, **kw)
        path = os.path.join(os.getcwd(), f"scallop_sweep_{tag}.json")
    out = _json_safe(out)
    # Only OSError (permissions, disk full) is recoverable by echoing to stdout:
    # _json_safe already guarantees finite-only data so allow_nan=False never
    # raises ValueError, and catching it would be a trap -- the fallback
    # json.dumps reuses the same data and allow_nan=False, re-raising it uncaught.
    try:
        with open(path, "w") as f:
            json.dump(out, f, indent=2, allow_nan=False)
        print("WROTE " + path, flush=True)
    except OSError:
        print("SWEEP_JSON " + json.dumps(out, allow_nan=False), flush=True)
