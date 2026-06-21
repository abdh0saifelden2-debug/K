r"""§D.2 scallop -> double-diffusion: does a scalloped wall create *both* DD
regimes in one geometry?

The double-diffusion null (Candidate 2, ``subglacial/candidate2_doublediff.py``)
was measured on a **smooth** wall: with a single global density ratio
``R_rho = alpha_S dS / (alpha_T dT)`` the cavity sits in one regime and the
thermal Nusselt ``Nu_T`` falls monotonically with ``R_rho`` (no finger hump).
The §D.2 hypothesis is that a *scalloped* wall breaks that homogeneity: the lee
of each bump traps warm/salty water (locally finger-favourable) while the stoss
face thins the cold/fresh meltwater layer (locally diffusive-convective) -- so a
single nominal ``R_rho`` should produce **spatially heterogeneous** ``Nu_T(x)``,
``Nu_S(x)`` that are *phase-locked to the wall*, i.e. geometry-induced regime
mixing rather than a subgrid effect.

This probe reuses the Candidate-2 solver with its new backward-compatible
``wall_amp`` / ``wall_nwaves`` knobs (``wall_amp=0`` reproduces the smooth wall
bit-for-bit) and adds **per-column** diagnostics:

  * ``F_T(x) = <v'theta'>_y(x)``, ``F_S(x) = <v'S'>_y(x)`` -- the interior
    turbulent vertical fluxes resolved by column (global fluid means removed, so
    the fluid-weighted column mean reproduces the domain-mean flux Candidate 2
    reports).  ``Nu_T(x) = 1 + F_T(x)/(kappa_T grad)``, likewise ``Nu_S(x)``.
  * ``gamma(x) = F_T(x)/(R_rho F_S(x))`` -- the *local* Turner flux ratio.

Three dimensionless, geometry-driven claims are earned by comparing the
scalloped run to the smooth run at identical physics:

  1. **heterogeneity** -- the across-column spread ``std_x[Nu_T(x)]`` is much
     larger for the scalloped wall than the (near-homogeneous) smooth wall;
  2. **phase-locking** -- the variation of ``Nu_T(x)`` is explained by the wall
     *geometry* (total wall-coherent fraction ``eta^2`` near 1, vs ~0 for the
     smooth wall whose anomalies are incoherent turbulence).  The coherent
     response is resolved into wall harmonics: the **fundamental** (``k=n_waves``)
     carries the *antisymmetric* lee/stoss directional signal, while the
     **second harmonic** (``k=2 n_waves``) carries the *symmetric* constriction
     response that peaks at every crest *and* trough -- so a fundamental-only
     index (``f_lock``) undercounts a corrugation whose response is symmetric;
  3. **regime mixing** -- the local Turner ratio ``gamma(x)`` *brackets* the
     smooth-wall value, i.e. some columns are pushed finger-enhanced and others
     suppressed within one geometry (both regimes coexist), whereas the smooth
     wall sits at a single value.

The diagnostic helpers (:func:`column_turb_fluxes`, :func:`phase_lock_fraction`,
:func:`wall_coherent_fraction`, :func:`harmonic_fractions`, :func:`regime_split`)
are pure array reductions and are unit-tested on synthetic fields in
``tests/test_doublediff_scallop.py`` (no DNS required).

Usage:
    python scallop_doublediff.py --out dd.json
    python scallop_doublediff.py --gpu --out dd.json
    python scallop_doublediff.py --fast            # tiny CPU smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# repo reorg: make sibling domain folders importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("general_two_clocks", "atmosphere", "glaciers", "ocean"):
    _p = os.path.join(_REPO_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _d, _p

from scallop_probe import _json_safe  # noqa: E402
from subglacial.candidate2_doublediff import DoubleDiffConfig, DoubleDiffFlow  # noqa: E402


def get_backend(force_cpu=False):
    if force_cpu:
        return np, "numpy(CPU)"
    try:
        import cupy as cp
        cp.zeros(1).sum()
        return cp, "cupy(GPU)"
    except (ImportError, RuntimeError, MemoryError):
        return np, "numpy(CPU)"


def _host(a):
    """Return a host NumPy array whether ``a`` is NumPy or CuPy."""
    return np.asarray(a.get() if hasattr(a, "get") else a)


# --------------------------------------------------------------------------- #
# diagnostics (pure NumPy reductions -- unit-tested without any DNS)
# --------------------------------------------------------------------------- #
def column_turb_fluxes(v, theta, S, fluid):
    """Per-column interior turbulent vertical fluxes ``<v'theta'>(x)`` and
    ``<v'S'>(x)``.

    The fluctuation means (``vbar``, ``cbar``) are the **global** fluid means,
    so the fluid-weighted average of the returned per-column flux reproduces the
    domain-mean turbulent flux used by Candidate 2's :meth:`nusselt`.  Columns
    with no fluid cells return ``0`` (no information), not ``NaN``.

    Parameters are 2-D ``(nx, ny)`` host arrays; ``fluid`` is a boolean mask.
    Returns ``(F_T, F_S)`` each shape ``(nx,)``.
    """
    v = np.asarray(v, float)
    theta = np.asarray(theta, float)
    S = np.asarray(S, float)
    f = np.asarray(fluid, float)
    fvol = f.sum() + 1e-30
    vbar = (v * f).sum() / fvol
    tbar = (theta * f).sum() / fvol
    sbar = (S * f).sum() / fvol
    vp = v - vbar
    n_col = f.sum(axis=1)
    denom = np.where(n_col > 0, n_col, 1.0)
    F_T = (vp * (theta - tbar) * f).sum(axis=1) / denom
    F_S = (vp * (S - sbar) * f).sum(axis=1) / denom
    return F_T, F_S


def local_nusselt(F_T, F_S, kappa_T, kappa_S, grad):
    """Per-column Nusselt numbers from the turbulent fluxes and the imposed bulk
    gradient ``grad = 1/H_cav`` (matches Candidate 2's normalisation)."""
    Nu_T = 1.0 + np.asarray(F_T, float) / (kappa_T * grad)
    Nu_S = 1.0 + np.asarray(F_S, float) / (kappa_S * grad)
    return Nu_T, Nu_S


def wall_phase(nx, n_waves):
    """Wall phase ``phi(x) in [0,2pi)`` of ``sin(2*pi*n_waves*x/Lx)`` sampled at
    the ``nx`` column centres (the cell grid ``x_i = i*dx``)."""
    phi = (2.0 * np.pi * n_waves * (np.arange(nx) / nx)) % (2.0 * np.pi)
    phi[np.isclose(phi, 2.0 * np.pi)] = 0.0
    return phi


def phase_lock_fraction(a, phase):
    """Fraction of the across-column variance of ``a(x)`` explained by the wall
    fundamental at ``phase(x)`` (a phase-locking index in ``[0,1]``).

    Projects the anomaly ``a - mean(a)`` onto ``{cos phi, sin phi}``; for a pure
    first-harmonic field the fraction is 1, for an ``x``-uniform field it is 0,
    and for incoherent noise it is ``~2/N``.  Returns ``(f_lock, pref_phase)``.
    """
    a = np.asarray(a, float)
    ap = a - a.mean()
    n = a.size
    tot = float((ap * ap).sum())
    if n == 0 or tot <= 1e-18 * (float((a * a).sum()) + 1e-300):
        return 0.0, 0.0
    c = float((ap * np.cos(phase)).sum())
    s = float((ap * np.sin(phase)).sum())
    frac = (2.0 * (c * c + s * s) / n) / tot
    pref = float(np.arctan2(s, c) % (2.0 * np.pi))
    return float(min(frac, 1.0)), pref


def harmonic_fractions(a, phase, n_harm=4):
    """Fractions of the across-column variance of ``a(x)`` carried by the first
    ``n_harm`` harmonics of the wall (``k = 1 .. n_harm`` times the wall
    wavenumber).  Returns a list of floats.

    ``frac[0]`` is the **fundamental** (``k=1``) -- the *antisymmetric* lee/stoss
    directional response; ``frac[1]`` is the **second harmonic** (``k=2``) -- the
    *symmetric* constriction response that peaks at every crest *and* trough of
    the corrugation.  Each entry uses the same single-harmonic projection as
    :func:`phase_lock_fraction`, so ``frac[0]`` equals that function's ``f_lock``.
    """
    a = np.asarray(a, float)
    ap = a - a.mean()
    n = a.size
    tot = float((ap * ap).sum())
    if n == 0 or tot <= 1e-18 * (float((a * a).sum()) + 1e-300):
        return [0.0] * n_harm
    phi = np.asarray(phase, float)
    out = []
    for k in range(1, n_harm + 1):
        c = float((ap * np.cos(k * phi)).sum())
        s = float((ap * np.sin(k * phi)).sum())
        out.append(float(min(2.0 * (c * c + s * s) / n / tot, 1.0)))
    return out


def wall_coherent_fraction(a, phase, nbins=12):
    """Total fraction of the across-column variance of ``a(x)`` that is coherent
    with the wall -- the correlation ratio ``eta^2`` of binning ``a`` by wall
    phase, in ``[0,1]``.

    Unlike :func:`phase_lock_fraction` (which projects onto the wall *fundamental*
    only), binning by phase groups every column sitting at the same point on the
    corrugation, so the binned profile retains the response at *all* harmonics
    resolvable by the bins -- in particular the symmetric ``2x``-wavenumber lock
    of a corrugation.  An ``x``-uniform field gives 0, a perfectly
    wall-determined field gives 1, and incoherent noise gives only the small
    bin-count floor ``~(nbins-1)/N``.
    """
    a = np.asarray(a, float)
    phi = np.asarray(phase, float)
    n = a.size
    if n == 0:
        return 0.0
    grand = float(a.mean())
    ap = a - grand
    tot = float((ap * ap).sum())
    # relative guard: a field that is constant to rounding has no variance to
    # apportion (avoids dividing residual float dust).
    if tot <= 1e-18 * (float((a * a).sum()) + 1e-300):
        return 0.0
    edges = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
    idx = np.clip(np.digitize(phi, edges) - 1, 0, nbins - 1)
    ss_between = 0.0
    for b in range(nbins):
        sel = idx == b
        nb = int(sel.sum())
        if nb > 0:
            ss_between += nb * (float(a[sel].mean()) - grand) ** 2
    return float(min(ss_between / tot, 1.0))


def phase_profile(values, phase, nbins=12):
    """Bin ``values(x)`` into ``nbins`` equal wall-phase bins and return the
    per-bin mean profile and its peak-to-trough amplitude.

    Binning by the wall phase averages together all columns (and, when fed
    time-accumulated fields, all sampled times) that sit at the same point on
    the corrugation, so incoherent turbulence cancels and only the part of
    ``values(x)`` that is *coherent with the wall* survives.  A smooth wall has
    no real corrugation, so its binned profile is flat to the noise floor; a
    scalloped wall shows a coherent peak-to-trough.  Empty bins are dropped.
    """
    v = np.asarray(values, float)
    phi = np.asarray(phase, float)
    edges = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
    idx = np.clip(np.digitize(phi, edges) - 1, 0, nbins - 1)
    prof = np.full(nbins, np.nan)
    for b in range(nbins):
        sel = idx == b
        if sel.any():
            prof[b] = float(v[sel].mean())
    finite = prof[np.isfinite(prof)]
    p2p = float(finite.max() - finite.min()) if finite.size else 0.0
    return prof, p2p


def regime_split(gamma_x, gamma_ref, tol=0.0):
    """Classify columns by their local Turner ratio ``gamma(x)`` relative to the
    homogeneous (smooth-wall) reference ``gamma_ref``.

    Returns a dict with the fraction of columns *enhanced* (``gamma`` further
    from the conductive ``gamma=0`` axis, i.e. ``|gamma| > |gamma_ref|+tol``) and
    *suppressed* (``|gamma| < |gamma_ref|-tol``), plus the across-column range.
    ``both_regimes`` is True when neither subpopulation is empty -- the geometry
    has pushed columns to *both* sides of the homogeneous value.
    """
    g = np.asarray(gamma_x, float)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return dict(frac_enhanced=0.0, frac_suppressed=0.0,
                    gamma_range=0.0, both_regimes=False)
    ref = abs(float(gamma_ref))
    enh = float(np.mean(np.abs(g) > ref + tol))
    sup = float(np.mean(np.abs(g) < ref - tol))
    return dict(frac_enhanced=enh, frac_suppressed=sup,
                gamma_range=float(g.max() - g.min()),
                both_regimes=bool(enh > 0.0 and sup > 0.0))


# --------------------------------------------------------------------------- #
# DNS driver
# --------------------------------------------------------------------------- #
def _time_avg_columns(flow, measure, sample_every, xp):
    """Run ``measure`` steps, accumulating the per-column turbulent fluxes and
    the global Nusselt every ``sample_every`` steps; return time-means."""
    nblocks = max(measure // sample_every, 1)
    FT = None
    FS = None
    NuT_g, NuS_g, gam_g, ke = [], [], [], []
    for _ in range(nblocks):
        flow.run(sample_every)
        v = _host(flow.v)
        th = _host(flow.theta); S = _host(flow.S)
        fluid = _host(flow.fluid).astype(bool)
        ft, fs = column_turb_fluxes(v, th, S, fluid)
        FT = ft if FT is None else FT + ft
        FS = fs if FS is None else FS + fs
        d = flow.nusselt()
        NuT_g.append(d["Nu_T"]); NuS_g.append(d["Nu_S"]); gam_g.append(d["gamma"])
        ke.append(flow.kinetic_energy())
    FT /= nblocks
    FS /= nblocks
    return dict(F_T=FT, F_S=FS,
                Nu_T_global=float(np.mean(NuT_g)),
                Nu_S_global=float(np.mean(NuS_g)),
                gamma_global=float(np.mean(gam_g)),
                KE_mean=float(np.mean(ke)))


def _one_run(xp, *, wall_amp, wall_nwaves, spinup, measure, sample_every, cfg_kw):
    cfg = DoubleDiffConfig(wall_amp=wall_amp, wall_nwaves=wall_nwaves, **cfg_kw)
    flow = DoubleDiffFlow(cfg, xp=xp)
    flow.run(spinup)
    out = _time_avg_columns(flow, measure, sample_every, xp)
    grad = 1.0 / flow.Hcav
    Nu_T_x, Nu_S_x = local_nusselt(out["F_T"], out["F_S"],
                                   cfg.kappa_T, cfg.kappa_S, grad)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_x = out["F_T"] / (cfg.R_rho * out["F_S"])
    out.update(Nu_T_x=Nu_T_x, Nu_S_x=Nu_S_x, gamma_x=gamma_x,
               y_ice_x=_host(flow.y_ice_x))
    return cfg, out


def run(xp, *, nx=256, ny=96, A=4.0, R_rho=1.5, Ri_T=2.0, Le=100.0,
        f_amp=0.3, amp=0.30, n_waves=12, spinup=4000, measure=2000,
        sample_every=20, seed=1):
    """Smooth vs scalloped double-diffusion at one nominal ``R_rho``; returns the
    per-column diagnostics and the three §D.2 structural verdicts."""
    cfg_kw = dict(nx=nx, ny=ny, A=A, R_rho=R_rho, Ri_T=Ri_T, Le=Le,
                  f_amp=f_amp, seed=seed)

    cfg_s, smooth = _one_run(xp, wall_amp=0.0, wall_nwaves=0,
                             spinup=spinup, measure=measure,
                             sample_every=sample_every, cfg_kw=cfg_kw)
    cfg_w, scal = _one_run(xp, wall_amp=amp, wall_nwaves=n_waves,
                           spinup=spinup, measure=measure,
                           sample_every=sample_every, cfg_kw=cfg_kw)

    phi = wall_phase(nx, n_waves)

    def _het(d):
        m = float(np.mean(d["Nu_T_x"]))
        sd = float(np.std(d["Nu_T_x"]))
        return sd, (sd / abs(m) if m != 0 else float("nan"))

    het_s_sd, het_s_cv = _het(smooth)
    het_w_sd, het_w_cv = _het(scal)
    flock_w, pref_w = phase_lock_fraction(scal["Nu_T_x"], phi)
    flock_s, pref_s = phase_lock_fraction(smooth["Nu_T_x"], phi)
    # total wall-coherent fraction (all harmonics) -- the phase-locking gate;
    # the fundamental-only f_lock above is kept as a sub-diagnostic.  A
    # corrugation locks symmetrically at 2x the wall wavenumber, which a
    # fundamental-only index misses.
    coh_w = wall_coherent_fraction(scal["Nu_T_x"], phi)
    coh_s = wall_coherent_fraction(smooth["Nu_T_x"], phi)
    harm_w = harmonic_fractions(scal["Nu_T_x"], phi)
    harm_s = harmonic_fractions(smooth["Nu_T_x"], phi)
    split = regime_split(scal["gamma_x"], smooth["gamma_global"])

    # coherent (phase-binned) thermal-Nusselt profile: turbulence cancels, only
    # the wall-locked part survives -> robust geometric signal vs the smooth
    # wall's flat (noise-floor) profile.
    prof_w, p2p_w = phase_profile(scal["Nu_T_x"], phi)
    prof_s, p2p_s = phase_profile(smooth["Nu_T_x"], phi)

    heterogeneity_amplified = bool(p2p_w > 3.0 * max(p2p_s, 1e-12))
    phase_locked = bool(coh_w > 0.5 and coh_w > 3.0 * max(coh_s, 1e-9))

    res = dict(
        params=dict(nx=nx, ny=ny, A=A, R_rho=R_rho, Ri_T=Ri_T, Le=Le,
                    f_amp=f_amp, amp=amp, n_waves=n_waves, spinup=spinup,
                    measure=measure, sample_every=sample_every, seed=seed),
        smooth=dict(Nu_T_global=smooth["Nu_T_global"],
                    Nu_S_global=smooth["Nu_S_global"],
                    gamma_global=smooth["gamma_global"],
                    KE_mean=smooth["KE_mean"],
                    Nu_T_x_std=het_s_sd, Nu_T_x_cv=het_s_cv,
                    Nu_T_p2p=p2p_s, f_lock=flock_s,
                    wall_coherent=coh_s, harmonic_fracs=harm_s),
        scallop=dict(Nu_T_global=scal["Nu_T_global"],
                     Nu_S_global=scal["Nu_S_global"],
                     gamma_global=scal["gamma_global"],
                     KE_mean=scal["KE_mean"],
                     Nu_T_x_std=het_w_sd, Nu_T_x_cv=het_w_cv,
                     Nu_T_p2p=p2p_w, f_lock=flock_w, pref_phase=pref_w,
                     wall_coherent=coh_w, harmonic_fracs=harm_w),
        regime=split,
        verdict=dict(heterogeneity_amplified=heterogeneity_amplified,
                     phase_locked=phase_locked,
                     both_regimes=split["both_regimes"]),
        fields=dict(phi=phi,
                    smooth_Nu_T_x=smooth["Nu_T_x"], smooth_Nu_S_x=smooth["Nu_S_x"],
                    scallop_Nu_T_x=scal["Nu_T_x"], scallop_Nu_S_x=scal["Nu_S_x"],
                    scallop_gamma_x=scal["gamma_x"],
                    scallop_y_ice_x=scal["y_ice_x"],
                    smooth_Nu_T_profile=prof_s, scallop_Nu_T_profile=prof_w),
    )
    return res


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _verdict_lines(res):
    s, w, rg, v = res["smooth"], res["scallop"], res["regime"], res["verdict"]
    return [
        f"R_rho (nominal)        = {res['params']['R_rho']:.2f}   "
        f"amp(a/lam)={res['params']['amp']:.2f}  n_waves={res['params']['n_waves']}",
        f"Nu_T global  smooth={s['Nu_T_global']:+.3f}  scallop={w['Nu_T_global']:+.3f}",
        f"Nu_S global  smooth={s['Nu_S_global']:+.2f}  scallop={w['Nu_S_global']:+.2f}",
        f"phase-binned Nu_T peak-to-trough  smooth={s['Nu_T_p2p']:.4f}  "
        f"scallop={w['Nu_T_p2p']:.4f}"
        f"   -> heterogeneity amplified: {v['heterogeneity_amplified']}",
        f"phase-lock wall-coherent eta^2  smooth={s['wall_coherent']:.3f}  "
        f"scallop={w['wall_coherent']:.3f}"
        f"   -> phase-locked: {v['phase_locked']}",
        f"   scallop lock split: fundamental(lee/stoss)={w['harmonic_fracs'][0]:.3f}"
        f"  2nd-harm(symmetric)={w['harmonic_fracs'][1]:.3f}"
        f"  (f_lock fund-only={w['f_lock']:.3f} @ {w.get('pref_phase', float('nan')):.2f} rad)",
        f"local Turner gamma(x): range={rg['gamma_range']:.3f}  "
        f"frac_enhanced={rg['frac_enhanced']:.2f}  frac_suppressed={rg['frac_suppressed']:.2f}"
        f"   -> BOTH regimes in one geometry: {rg['both_regimes']}",
    ]


def maybe_figure(res, out_dir, fname="50_scallop_doublediff.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fld = res["fields"]
    phi = np.asarray(fld["phi"])
    x = np.arange(phi.size)
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.4))

    axes[0].plot(x, fld["smooth_Nu_T_x"], color="0.6", lw=1.0, label="smooth")
    axes[0].plot(x, fld["scallop_Nu_T_x"], color="tab:red", lw=1.2, label="scallop")
    axes[0].axhline(1.0, color="k", lw=0.6, ls=":")
    axes[0].set_ylabel(r"$Nu_T(x)$")
    axes[0].set_title("§D.2 per-column thermal Nusselt: smooth vs scalloped wall")
    axes[0].legend(loc="best", fontsize=8)

    yi = np.asarray(fld["scallop_y_ice_x"])
    ax1b = axes[1].twinx()
    ax1b.plot(x, yi, color="0.7", lw=1.0, ls="--", label="ice wall $y_{ice}(x)$")
    ax1b.set_ylabel(r"$y_{ice}(x)$", color="0.5")
    axes[1].plot(x, fld["scallop_gamma_x"], color="tab:blue", lw=1.2)
    axes[1].axhline(res["smooth"]["gamma_global"], color="tab:green", lw=1.0,
                    ls="-", label="smooth $\\gamma$ (homogeneous)")
    axes[1].set_ylabel(r"local Turner $\gamma(x)$")
    axes[1].legend(loc="best", fontsize=8)

    prof_w = np.asarray(fld["scallop_Nu_T_profile"])
    prof_s = np.asarray(fld["smooth_Nu_T_profile"])
    nb = prof_w.size
    bins = (np.arange(nb) + 0.5) * (2.0 * np.pi / nb)
    axes[2].plot(bins, prof_s, "o-", color="0.6", lw=1.2, label="smooth (flat → noise floor)")
    axes[2].plot(bins, prof_w, "o-", color="tab:red", lw=1.4, label="scallop (wall-locked)")
    axes[2].set_xlabel(r"wall phase $\phi$ (rad)")
    axes[2].set_ylabel(r"$\langle Nu_T\rangle$ per phase bin")
    axes[2].set_title("phase-binned thermal Nusselt (turbulence averaged out)")
    axes[2].legend(loc="best", fontsize=8)
    w = res["scallop"]
    hf = w.get("harmonic_fracs", [0.0, 0.0])
    axes[2].text(
        0.02, 0.04,
        "wall-coherent $\\eta^2$=%.2f  (fund/lee-stoss %.2f + 2nd-harm/sym %.2f)"
        % (w.get("wall_coherent", float("nan")), hf[0], hf[1]),
        transform=axes[2].transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _write_report(res, path):
    p = res["params"]
    s, w, rg, v = res["smooth"], res["scallop"], res["regime"], res["verdict"]
    lines = [
        "# §D.2 Scallop → double-diffusion: regime mixing in one geometry",
        "",
        "A scalloped ice wall (`wall_amp`, `wall_nwaves` on the Candidate-2 "
        "double-diffusion solver) is compared to the smooth wall at identical "
        "physics. Per-column turbulent fluxes give `Nu_T(x)`, `Nu_S(x)` and the "
        "local Turner ratio `gamma(x) = F_T(x)/(R_rho F_S(x))`.",
        "",
        f"- Grid `nx={p['nx']}`, `ny={p['ny']}`, aspect `A={p['A']}`; "
        f"`R_rho={p['R_rho']}`, `Ri_T={p['Ri_T']}`, `Le={p['Le']}`, "
        f"`f_amp={p['f_amp']}`; wall `a/λ={p['amp']}`, `n_waves={p['n_waves']}`; "
        f"spinup `{p['spinup']}`, measure `{p['measure']}`.",
        "",
        "## Result",
        "",
        "- **Heterogeneity** — phase-binned `Nu_T` peak-to-trough (turbulence "
        "averaged out) smooth `{:.4f}` → scallop `{:.4f}` (**amplified: "
        "{}**).".format(s["Nu_T_p2p"], w["Nu_T_p2p"],
                        v["heterogeneity_amplified"]),
        f"- **Phase-locking** — total wall-coherent variance fraction "
        f"`η²` (all harmonics, correlation ratio) smooth `{s['wall_coherent']:.3f}` "
        f"→ scallop `{w['wall_coherent']:.3f}` (**locked: {v['phase_locked']}**). "
        f"The coherent response splits into a *fundamental* (lee/stoss directional) "
        f"fraction `{w['harmonic_fracs'][0]:.3f}` and a dominant *2nd-harmonic* "
        f"(symmetric constriction, peaks at every crest **and** trough) fraction "
        f"`{w['harmonic_fracs'][1]:.3f}`; the fundamental-only index "
        f"`f_lock={w['f_lock']:.3f}` @ `{w.get('pref_phase', float('nan')):.2f}` rad "
        f"undercounts the lock because the corrugation response is symmetric.",
        f"- **Regime mixing** — local `gamma(x)` range `{rg['gamma_range']:.3f}`, "
        f"enhanced frac `{rg['frac_enhanced']:.2f}`, suppressed frac "
        f"`{rg['frac_suppressed']:.2f}` about the smooth value "
        f"`{s['gamma_global']:+.3f}` (**both regimes coexist: "
        f"{rg['both_regimes']}**).",
        f"- Global means (sanity vs Candidate 2): `Nu_T` smooth "
        f"`{s['Nu_T_global']:+.3f}` / scallop `{w['Nu_T_global']:+.3f}`; "
        f"`Nu_S` smooth `{s['Nu_S_global']:+.2f}` / scallop "
        f"`{w['Nu_S_global']:+.2f}`.",
        "",
        f"- Backend: `{res.get('backend', 'n/a')}`.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="§D.2 scallop -> double-diffusion")
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--ny", type=int, default=96)
    ap.add_argument("--A", dest="A", type=float, default=4.0)
    ap.add_argument("--rrho", dest="R_rho", type=float, default=1.5)
    ap.add_argument("--ri-t", dest="Ri_T", type=float, default=2.0)
    ap.add_argument("--le", dest="Le", type=float, default=100.0)
    ap.add_argument("--f-amp", dest="f_amp", type=float, default=0.3)
    ap.add_argument("--amp", type=float, default=0.30, help="wall a/lambda")
    ap.add_argument("--nwaves", dest="n_waves", type=int, default=12)
    ap.add_argument("--spinup", type=int, default=4000)
    ap.add_argument("--measure", type=int, default=2000)
    ap.add_argument("--sample-every", dest="sample_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu", action="store_true", help="force GPU backend")
    ap.add_argument("--cpu", action="store_true", help="force CPU backend")
    ap.add_argument("--fast", action="store_true", help="tiny CPU smoke run")
    ap.add_argument("--out-dir", dest="out_dir", default="figures")
    ap.add_argument("--report", default=None)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if args.fast:
        args.nx, args.ny = 96, 64
        args.spinup, args.measure, args.sample_every = 400, 300, 20
        args.cpu = True

    xp, backend = get_backend(force_cpu=args.cpu and not args.gpu)
    print(f"Backend: {backend}", flush=True)
    t0 = time.time()
    res = run(xp, nx=args.nx, ny=args.ny, A=args.A, R_rho=args.R_rho,
              Ri_T=args.Ri_T, Le=args.Le, f_amp=args.f_amp, amp=args.amp,
              n_waves=args.n_waves, spinup=args.spinup, measure=args.measure,
              sample_every=args.sample_every, seed=args.seed)
    res["backend"] = backend
    res["walltime_s"] = round(time.time() - t0, 2)

    print("\n=== §D.2 scallop -> double-diffusion: regime mixing ===")
    for ln in _verdict_lines(res):
        print(ln)

    fig = maybe_figure(res, args.out_dir)
    if fig:
        print("wrote", fig)

    out = args.out or os.path.join(args.out_dir, "50_scallop_doublediff.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(_json_safe(res), f, indent=2, allow_nan=False)
    print("wrote", out)

    if args.report:
        _write_report(res, args.report)
        print("wrote", args.report)


if __name__ == "__main__":
    main()
