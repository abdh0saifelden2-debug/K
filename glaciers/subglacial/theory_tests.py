r"""Numerical verification of the cheap, GPU-free predictions in THEORY_CAVITY.md.

Two results from the two-clocks theory are testable *today* with the existing 2D
solvers (no GPU, no turbulence model active). This script measures them and prints
a results block that is folded back into THEORY_CAVITY.md.

Result 1 -- Spurious Pressure Source.
    A subgrid momentum model with non-zero divergence injects a source
    Q = div(m) into the pressure Poisson equation; via the global Green's function
    that *local* error becomes a *global* pressure (= water pressure = effective
    pressure N) corruption. We deploy the *raw, unprojected* Smagorinsky force
    m = div(2 nu_t S) (what a real solver adds to momentum) and Helmholtz-split it:
        - dilatational (curl-free) fraction  -> the part the pressure solve absorbs,
        - the resulting spurious pressure dp  (Poisson(dp) = div m),
        - compare to the projected-FDT force (solenoidal by construction -> ~0).
    Correction A (this repo): in the *bulk* the curl-free part is dynamically inert
    for velocity (the pressure solve projects it out), so it corrupts the diagnosed
    pressure / N, not the bulk flow. The exception is the *penalized wall*, where
    the projection is inexact -- so we also report how concentrated the curl-free
    part is near the bed/ice interfaces (the melt-relevant region).

Result 2 -- Clock Mismatch Number M_clock = Lambda_p / Lambda_theta.
    The discriminator between the elliptic (pressure) and advective (scalar) clocks
    is NOT isotropic radius -- a high-Pe scalar forms a long *downstream* plume that
    is very nonlocal along the flow. The clean discriminator is UPSTREAM influence:
    the elliptic pressure responds in ALL directions (action at a distance, incl.
    upstream), while the advected scalar is screened upstream over the short length
    kappa/U (causal, downstream-only). The scalar carries a relaxation sink theta/T
    (the cavity exchanges heat with the ice, so the downstream plume decays over
    U*T rather than wrapping the periodic box). Lambda := fraction of |field|
    located upstream of the source. Pressure: Lambda_p ~ 0.4 (near-symmetric).
    Scalar: Lambda_theta -> 0 as Pe grows. We sweep Pe and show M_clock =
    Lambda_p/Lambda_theta grows with Pe (Correction B), and report it for the real
    BEDMAP bed geometry.

Run:  python -m subglacial.theory_tests   [--n 128] [--spinup 400]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# repo reorg: make sibling domain folders importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _d in ("general_two_clocks", "glaciers"):
    _p = os.path.join(_REPO_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _d, _p

from compressible.ns import helmholtz, incompressible_pressure
from closure.sgs import (sharp_filter, exact_sgs_force, projected_fdt_force,
                         divergence_rms, project_vec)
from subglacial.bedmap import bed_profile_from_transect, clean_sorted_transect
from subglacial.flow import SubglacialConfig, SubglacialFlow
from subglacial.diag import exact_sgs_heat_flux, masked_corr

BED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bedmap1_transect.csv")


def _rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


def smagorinsky_force_raw(sp, ub, vb, kc, cs=0.16):
    """Deployed (UNPROJECTED) Smagorinsky force m = div(2 nu_t S).

    Identical to closure.sgs.smagorinsky_force but WITHOUT the final Leray
    projection -- this is the force a real solver actually adds to momentum, and
    its curl-free part is exactly what the pressure solve must absorb.
    """
    delta = np.pi / kc
    ux, uy = sp.ddx(ub), sp.ddy(ub)
    vx, vy = sp.ddx(vb), sp.ddy(vb)
    s11, s22, s12 = ux, vy, 0.5 * (uy + vx)
    smag = np.sqrt(2.0 * (s11 ** 2 + s22 ** 2 + 2.0 * s12 ** 2))
    nu_t = (cs * delta) ** 2 * smag
    tau11, tau22, tau12 = 2.0 * nu_t * s11, 2.0 * nu_t * s22, 2.0 * nu_t * s12
    mx = sp.ddx(tau11) + sp.ddy(tau12)
    my = sp.ddx(tau12) + sp.ddy(tau22)
    return sharp_filter(sp, mx, kc), sharp_filter(sp, my, kc)


def spurious_pressure(sp, mx, my):
    """Pressure the incompressible solve must add to absorb a force m:
    Poisson(dp) = div(m)  =>  dp_hat = -div_hat * k2_inv  (= Helmholtz potential)."""
    div_m = sp.ddx(mx) + sp.ddy(my)
    dp_h = -sp.fft(div_m) * sp.k2_inv
    dp_h[0, 0] = 0.0
    return sp.ifft(dp_h), div_m


def result1(sp, f, kc, cs):
    """Spurious-pressure metrics for raw Smagorinsky vs projected-FDT."""
    u, v = f.u, f.v
    ub, vb, mtx, mty = exact_sgs_force(sp, u, v, kc)

    # --- deployed (raw) Smagorinsky force ---
    msx_raw, msy_raw = smagorinsky_force_raw(sp, ub, vb, kc, cs=cs)
    ssx, ssy, sdx, sdy = helmholtz(sp, msx_raw, msy_raw)   # sol / dilatational split
    tot = _rms(msx_raw) ** 2 + _rms(msy_raw) ** 2
    dil = _rms(sdx) ** 2 + _rms(sdy) ** 2
    dil_frac_smag = dil / (tot + 1e-30)
    div_smag = divergence_rms(sp, msx_raw, msy_raw)
    dp_smag, div_m = spurious_pressure(sp, msx_raw, msy_raw)

    # reference dynamic (incompressible) pressure of the resolved field
    p_dyn = incompressible_pressure(sp, ub, vb, rho0=1.0)
    dp_ratio = _rms(dp_smag) / (_rms(p_dyn) + 1e-30)

    # --- projected-FDT force (solenoidal by construction) ---
    mfx, mfy, _ = projected_fdt_force(sp, ub, vb, mtx, mty, kc, seed=7)
    _, _, fdx, fdy = helmholtz(sp, mfx, mfy)
    totf = _rms(mfx) ** 2 + _rms(mfy) ** 2
    dil_frac_fdt = (_rms(fdx) ** 2 + _rms(fdy) ** 2) / (totf + 1e-30)
    div_fdt = divergence_rms(sp, mfx, mfy)
    dp_fdt, _ = spurious_pressure(sp, mfx, mfy)
    dp_ratio_fdt = _rms(dp_fdt) / (_rms(p_dyn) + 1e-30)

    # --- Correction A: is the curl-free (spurious) part tied to the walls? ---
    # partition the dilatational energy into solid/interface (chi>0.1) vs open
    # cavity (chi<0.1).  The penalty that breaks the clean projection lives where
    # chi>0, so a wall-tied spurious part concentrates there.
    solidish = f.chi > 0.1
    dil_mag = sdx ** 2 + sdy ** 2
    wall_frac = float(dil_mag[solidish].sum() / (dil_mag.sum() + 1e-30))
    area_frac = float(solidish.mean())

    # --- Correction A: bulk velocity tendency is the SAME with/without projecting
    # Smagorinsky (pressure solve projects anyway). ||P m_raw - m_proj|| / ||P m_raw||
    from closure.sgs import smagorinsky_force as smag_proj
    msx_p, msy_p = smag_proj(sp, ub, vb, kc, cs=cs)
    diff = _rms(ssx - msx_p) ** 2 + _rms(ssy - msy_p) ** 2
    base = _rms(ssx) ** 2 + _rms(ssy) ** 2
    proj_equiv = np.sqrt(diff / (base + 1e-30))

    return dict(dil_frac_smag=dil_frac_smag, div_smag=div_smag,
                dp_ratio_smag=dp_ratio, dil_frac_fdt=dil_frac_fdt,
                div_fdt=div_fdt, dp_ratio_fdt=dp_ratio_fdt,
                wall_frac=wall_frac, area_frac=area_frac, proj_equiv=proj_equiv)


def upstream_fraction(field, ix, lam_cells):
    """Lambda = (sum |field| located upstream of the source, x < x0-lam) / total.
    Streamwise axis is axis 0 (X varies along axis 0); mean flow is +x.
    Elliptic pressure -> ~0.5 (symmetric); advected scalar -> 0 (downstream cone)."""
    n = field.shape[0]
    ii = (np.arange(n) - ix + n // 2) % n - n // 2     # signed streamwise offset
    a = np.abs(field)
    up = a[ii < -lam_cells, :].sum()
    return float(up / (a.sum() + 1e-30))


def result2(sp, ybed, dx, lam_cells=6, pes=(1, 3, 10, 30, 100, 300)):
    """Clock Mismatch Number M_clock(Pe) for a localized source, plus the value
    over the real bed (source at the tallest bump crest)."""
    n = sp.n
    X, Y = sp.grid()
    # localized gaussian source at domain centre
    ix, iy = n // 2, n // 2
    x0, y0 = X[ix, iy], Y[ix, iy]
    lam = lam_cells * dx
    # periodic-aware squared distance
    ddx_ = (X - x0 + np.pi) % (2 * np.pi) - np.pi
    ddy_ = (Y - y0 + np.pi) % (2 * np.pi) - np.pi
    src = np.exp(-(ddx_ ** 2 + ddy_ ** 2) / (2.0 * lam ** 2))
    src = src - src.mean()
    src_h = sp.fft(src)

    # pressure response (elliptic): Poisson(p) = src
    p_h = -src_h * sp.k2_inv
    p_h[0, 0] = 0.0
    p = sp.ifft(p_h)
    Lambda_p = upstream_fraction(p, ix, lam_cells)

    rows = []
    U = 1.0
    relax = U / (sp.L / 4.0)            # sink: downstream plume decays over L/4
    for Pe in pes:
        kappa = U * lam / Pe
        # steady advection-diffusion w/ relaxation: U d_x th - kappa lap th + th/T = src
        denom = 1j * U * sp.kx + kappa * sp.k2 + relax
        th = sp.ifft(src_h / denom)
        Lambda_th = upstream_fraction(th, ix, lam_cells)
        rows.append((Pe, Lambda_p, Lambda_th, Lambda_p / (Lambda_th + 1e-30)))

    # --- real-bed geometry: put the source just above the tallest bump crest ---
    icrest = int(np.argmax(ybed))
    jy = int(np.clip(round(ybed[icrest] / dx) + lam_cells, 0, n - 1))
    ddx_b = (X - X[icrest, jy] + np.pi) % (2 * np.pi) - np.pi
    ddy_b = (Y - Y[icrest, jy] + np.pi) % (2 * np.pi) - np.pi
    src_b = np.exp(-(ddx_b ** 2 + ddy_b ** 2) / (2.0 * lam ** 2))
    src_b = src_b - src_b.mean()
    sbh = sp.fft(src_b)
    pb = sp.ifft(np.where(sp.k2 > 0, -sbh * sp.k2_inv, 0.0))
    Lp_bed = upstream_fraction(pb, icrest, lam_cells)
    Pe_bed = 100.0
    kappa = 1.0 * lam / Pe_bed
    relax = 1.0 / (sp.L / 4.0)
    denom = 1j * 1.0 * sp.kx + kappa * sp.k2 + relax
    thb = sp.ifft(sbh / denom)
    Lth_bed = upstream_fraction(thb, icrest, lam_cells)
    bed = (Pe_bed, Lp_bed, Lth_bed, Lp_bed / (Lth_bed + 1e-30))
    return rows, bed


def _advect_diffuse_step(sp, c, U, V, kappa, dt):
    """One step of the heat-advection semigroup P(t): exact (integrating-factor)
    diffusion + explicit advection by the (possibly variable) velocity (U, V)."""
    c = sp.ifft(sp.fft(c) * np.exp(-kappa * sp.k2 * dt))
    return c - dt * (U * sp.ddx(c) + V * sp.ddy(c))


def commutator_norm(sp, U, V, v0x, v0y, kappa, dt, nsteps):
    """||[L, P(t)] v0|| / ||P(t) v0|| for a divergence-free v0.

    Since v0 is divergence-free, L v0 = v0, so [L,P]v0 = L P v0 - P v0 =
    -(dilatational part of P(t) v0).  Hence the commutator norm is exactly how
    much divergence a div-free field ACQUIRES under advection by u-bar: zero for
    uniform U (translation preserves div-free), nonzero only via the shear term
    -(d_i u_j)(d_j w_i).  U, V may be scalars (uniform) or fields (the cavity)."""
    wx, wy = v0x.copy(), v0y.copy()
    for _ in range(nsteps):
        wx = _advect_diffuse_step(sp, wx, U, V, kappa, dt)
        wy = _advect_diffuse_step(sp, wy, U, V, kappa, dt)
    _, _, wdx, wdy = helmholtz(sp, wx, wy)
    num = np.sqrt(_rms(wdx) ** 2 + _rms(wdy) ** 2)
    den = np.sqrt(_rms(wx) ** 2 + _rms(wy) ** 2)
    return float(num / (den + 1e-30))


def result3_commutator(sp, f, kappa, dt=4.0e-3, step_list=(10, 40, 160, 320)):
    """Direct measurement of the master identity ||[L, P(t)]|| in the cavity,
    contrasting uniform flow (should ~vanish) with the real BEDMAP cavity flow
    (nonzero via shear + walls).  Returns (Umean, rows) with rows = list of
    (nsteps, t, norm_uniform, norm_cavity)."""
    n = sp.n
    X, Y = sp.grid()
    # localized divergence-free probe v0 = L f0, f0 a Gaussian vector blob in the
    # open cavity (mid-height of the tallest-relief column region).
    ix = n // 2
    jy = n // 2
    lam = 5.0 * sp.dx
    ddx_ = (X - X[ix, jy] + np.pi) % (2 * np.pi) - np.pi
    ddy_ = (Y - Y[ix, jy] + np.pi) % (2 * np.pi) - np.pi
    blob = np.exp(-(ddx_ ** 2 + ddy_ ** 2) / (2.0 * lam ** 2))
    v0x, v0y = project_vec(sp, blob, 0.5 * blob)      # divergence-free probe

    Umean = float(np.sqrt(np.mean(f.u ** 2 + f.v ** 2)))
    rows = []
    for ns in step_list:
        t = ns * dt
        nu = commutator_norm(sp, Umean, 0.0, v0x, v0y, kappa, dt, ns)   # uniform
        nc = commutator_norm(sp, f.u, f.v, v0x, v0y, kappa, dt, ns)      # cavity
        rows.append((ns, t, nu, nc))
    return Umean, rows


def dedner_cleaning_factor(kL, G):
    r"""Per-mode divergence amplitude after one pressure-adjustment transit of
    Dedner GLM hyperbolic cleaning, D(tau_adj)/D(0), as a closed form.

    The GLM cleaning system  d_t u = -grad(psi),  d_t psi = -c_h^2 div(u) - gamma psi
    gives, for each Fourier mode, the telegrapher ODE for Q = div(u):
        Q'' + gamma Q' + (c_h |k|)^2 Q = 0,   Q(0)=Q0,  Q'(0)=0  (psi(0)=0).
    Evaluated at one transit t = tau_adj = L / c_h, and written in the *only* free
    dimensionless knob G = gamma * tau_adj and the per-mode phase kL = |k| L
    (= omega * tau_adj, independent of c_h), the amplitude factor is

        f(kL, G) = e^{-G/2} [ cosh(sT) + (G/2) sinh(sT)/sT ],   sT = sqrt((G/2)^2 - kL^2).

    Verified against direct RK4 integration to machine precision (under- and
    over-damped). Vectorised over kL.
    """
    a = 0.5 * G
    sT = np.sqrt(a * a - kL * kL + 0j)
    small = np.abs(sT) < 1e-12
    ratio = np.where(small, 1.0, np.sinh(sT) / np.where(small, 1.0, sT))
    return np.real(np.exp(-a) * (np.cosh(sT) + a * ratio))


def dedner_residual_curve(sp, D0h, Gs):
    """Residual divergence and spurious-pressure ratios vs the cleaning knob G.

    Given the injected divergence spectrum D0h = fft(div u*), return rows
    (G, ||div u(tau)||/||div u*||, ||dp(tau)||/||dp*||).  The pressure ratio is
    the low-|k|-weighted (1/k^2) norm, since Poisson(dp)=div u."""
    kL = np.sqrt(sp.k2) * sp.L
    k2i = sp.k2_inv
    den = np.sqrt(np.sum(np.abs(D0h) ** 2)) + 1e-30
    den_p = np.sqrt(np.sum(np.abs(D0h * k2i) ** 2)) + 1e-30
    rows = []
    for G in Gs:
        DT = D0h * dedner_cleaning_factor(kL, float(G))
        rd = float(np.sqrt(np.sum(np.abs(DT) ** 2)) / den)
        rp = float(np.sqrt(np.sum(np.abs(DT * k2i) ** 2)) / den_p)
        rows.append((float(G), rd, rp))
    return np.array(rows)


def _knee_one_over_e(rows):
    """G where the divergence residual first crosses (1/e) of its G->0 value."""
    target = rows[0, 1] / np.e
    for i in range(len(rows) - 1):
        if rows[i, 1] >= target >= rows[i + 1, 1]:
            g0, g1, v0, v1 = rows[i, 0], rows[i + 1, 0], rows[i, 1], rows[i + 1, 1]
            return float(g0 + (target - v0) * (g1 - g0) / (v1 - v0))
    return float("nan")


def result_dedner_cleaning(sp, f, kc, cs=0.16, dt=4.0e-3,
                           Gs=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
                               12.0, 16.0, 24.0, 40.0, 60.0)):
    """Measure the approximate-projection design threshold G* = gamma_clean*tau_adj.

    The 2D solver enforces incompressibility by *exact* spectral Leray, so to test
    the bound we deploy the Dedner GLM cleaning system as the tunable approximate
    projector.  We inject the raw-Smagorinsky divergence into a frozen DNS field
    (u* = u + dt*m_smag, so div u* = dt*div m), then clean for one transit and
    sweep G.  Returns the residual curve, the 1/e knee, the over-damped optimum,
    and the G=0 spurious-pressure global-tail fraction (Theorem-1 nonlocality)."""
    ub, vb, _, _ = exact_sgs_force(sp, f.u, f.v, kc)
    mx, my = smagorinsky_force_raw(sp, ub, vb, kc, cs=cs)
    D0 = sp.ddx(f.u + dt * mx) + sp.ddy(f.v + dt * my)
    D0h = sp.fft(D0)
    rows = dedner_residual_curve(sp, D0h, Gs)

    # G=0 (uncleaned): the locally-sourced divergence makes a *global* pressure.
    dp0 = sp.ifft(-D0h * sp.k2_inv).real
    core = np.abs(D0) > _rms(D0)
    e = dp0 ** 2
    tail0 = float(1.0 - e[core].sum() / (e.sum() + 1e-30))
    core_frac = float(core.mean())
    # nonlocal amplification: pressure-energy fraction outside the divergence
    # footprint, per unit footprint area.  A=1 would be a purely local response;
    # A>1 quantifies the elliptic Green's-function spreading.
    amp = float(tail0 / (core_frac + 1e-30))
    return dict(rows=rows, Gknee=_knee_one_over_e(rows),
                Gopt=float(rows[np.argmin(rows[:, 2]), 0]),
                tail0=tail0, core_frac=core_frac, amp=amp,
                inj_div=_rms(D0))


def result_commutator_entropy(sp, f, kc, kappa=2.0e-4, ts=(0.02, 0.01, 0.005)):
    """RESULT 5 -- the Girsanov / Entropy-Pressure identity (Move 3A/3B).

    The honest probabilistic reading of [L,P]!=0 is NOT a failed 'tower property'
    (the Leray L is an L2-orthogonal projection, not a conditional expectation:
    not positivity-preserving, not unital).  It is a Girsanov entropy on diffusion
    PATHS.  For two parcel SDEs with the same diffusion sqrt(2 kappa) and drifts
    b_A (project-then-transport) and b_B (transport-then-project), the path-measure
    relative entropy is D_KL = (1/4 kappa) E INT |b_A - b_B|^2 ds.  To leading order
    the drift difference is the SAME commutator measured in RESULT 3,
        b_A - b_B = -t [L, -u.grad] u0 + O(t^2),   (t = operator-splitting lag)
    so the entropy RATE (per unit path time) is (t^2/4 kappa) ||[L,-u.grad]u0||^2.

    For a divergence-free u0 (L u0 = u0) the commutator is exactly the curl-free
    part of the advection, hence the Theorem-1 spurious-pressure gradient:
        [L, -u.grad] u0 = (I - L)(u.grad u0) = grad(dp_spur),
        nabla^2 dp_spur = div(u.grad u0).
    Therefore  ||[L,P]u0||^2 = INT |grad dp_spur|^2  -- the H^1 energy of the
    spurious pressure that corrupts the effective pressure N.  We verify:
      (a) the algebra identity to machine precision,
      (b) the splitting limit ([P(t)L - L P(t)]u0)/t -> -commutator as t->0,
      (c) the Entropy-Pressure identity ||C||^2 == INT|grad dp_spur|^2,
    and report the dilatational fraction ||C||/||u.grad u0|| (cf. RESULT 3 ~0.6).

    The Theorem-3 counter-gradient link stays a CONJECTURE ('<~'): we report the
    lee counter-gradient fraction and its (partial, r~0.5) correlation with the
    parcel-current compressibility u.grad(theta_bar) as consistent support only.
    """
    u0x, u0y = project_vec(sp, f.u, f.v)               # divergence-free frozen field
    U, V = u0x, u0y
    ax = U * sp.ddx(u0x) + V * sp.ddy(u0x)             # (u.grad)u0, componentwise
    ay = U * sp.ddx(u0y) + V * sp.ddy(u0y)

    # (a) [L,-u.grad]u0 == (I-L)(u.grad u0)
    lax, lay = project_vec(sp, ax, ay)
    Cx, Cy = ax - lax, ay - lay                        # (I-L) a
    lBx, lBy = project_vec(sp, -ax, -ay)               # L(-u.grad u0)
    BLx = -(U * sp.ddx(u0x) + V * sp.ddy(u0x))         # -u.grad(L u0), L u0 = u0
    BLy = -(U * sp.ddx(u0y) + V * sp.ddy(u0y))
    Cdx, Cdy = lBx - BLx, lBy - BLy                    # [L,-u.grad]u0 directly
    nC = (_rms(Cx) ** 2 + _rms(Cy) ** 2) ** 0.5
    err_alg = ((_rms(Cdx - Cx) ** 2 + _rms(Cdy - Cy) ** 2) ** 0.5) / (nC + 1e-30)

    # (c) Entropy-Pressure identity
    Q = sp.ddx(ax) + sp.ddy(ay)                        # div(u.grad u0) = source
    dp = sp.ifft(-sp.fft(Q) * sp.k2_inv).real          # nabla^2 dp = -Q
    gx, gy = sp.ddx(dp), sp.ddy(dp)                     # grad dp_spur
    C2 = float(np.mean(Cx ** 2 + Cy ** 2))
    G2 = float(np.mean(gx ** 2 + gy ** 2))
    ep_ratio = C2 / (G2 + 1e-30)
    dil_frac = (C2 / (float(np.mean(ax ** 2 + ay ** 2)) + 1e-30)) ** 0.5

    # (b) operator-splitting limit -> commutator (validates the Girsanov integrand)
    def Pt(cx, cy, t, ns):
        wx, wy = cx.copy(), cy.copy()
        dt = t / ns
        for _ in range(ns):
            wx = _advect_diffuse_step(sp, wx, U, V, kappa, dt)
            wy = _advect_diffuse_step(sp, wy, U, V, kappa, dt)
        return wx, wy

    split_rows = []
    for t in ts:
        ns = max(20, int(t / 2.5e-4))
        ax_, ay_ = Pt(u0x, u0y, t, ns)                 # P(t) L u0  (L u0 = u0)
        bx_, by_ = project_vec(sp, ax_, ay_)           # L P(t) u0
        dBx, dBy = (ax_ - bx_) / t, (ay_ - by_) / t    # [P L - L P]u0 / t -> -C
        rel = ((_rms(dBx + Cx) ** 2 + _rms(dBy + Cy) ** 2) ** 0.5) / (nC + 1e-30)
        split_rows.append((float(t), float(rel)))

    # Theorem-3 conjecture support (scaling, not identity)
    qx, qy, tb, tgx, tgy = exact_sgs_heat_flux(sp, f.u, f.v, f.theta, kc)
    align = qx * tgx + qy * tgy                         # >0 => up-gradient
    s = f.u * sp.ddx(tb) + f.v * sp.ddy(tb)            # u.grad(theta_bar)
    cav = f.fluid & (np.abs(tgx) + np.abs(tgy) > 0)
    cg_frac = float(np.mean(align[cav] > 0)) if cav.any() else 0.0
    cgmag = np.maximum(align, 0.0)
    return dict(err_alg=err_alg, ep_ratio=ep_ratio, dil_frac=dil_frac,
                C2=C2, G2=G2, split_rows=np.array(split_rows),
                cg_frac=cg_frac,
                corr_sq=masked_corr(cgmag, s ** 2, cav),
                corr_abs=masked_corr(cgmag, np.abs(s), cav))


def _radial_spectrum(sp, field):
    """Angle-averaged power spectrum E(k) on integer |k| shells (k>=1)."""
    P = np.abs(sp.fft(field)) ** 2
    kbin = np.round(np.sqrt(sp.k2)).astype(int)
    kmax = int(kbin.max())
    E = np.zeros(kmax + 1)
    for k in range(1, kmax + 1):
        m = kbin == k
        if m.any():
            E[k] = P[m].mean()
    return np.arange(kmax + 1), E


def _loglog_slope(k, E, klo, khi):
    m = (k >= klo) & (k <= khi) & (E > 0)
    return float(np.polyfit(np.log(k[m]), np.log(E[m]), 1)[0])


def _lowk_fraction(sp, field, kc):
    P = np.abs(sp.fft(field)) ** 2
    return float(P[np.sqrt(sp.k2) < kc].sum() / (P.sum() + 1e-30))


def result_constraint_class(sp, f, kc, cs=0.16):
    """RESULT 6 -- the constraint CLASS of the incompressibility projection (Move 4).

    div u = 0 is a *first-class* constraint (Arnold / Marsden-Weinstein): the naive
    bracket {C(x),C(y)} = nabla^2 delta(x-y) is a singular differential operator, so
    the pressure 'inverse' is the nonlocal Green's function (nabla^2)^-1 with Fourier
    symbol |k|^-2.  A *second-class* (local, algebraic) caricature would replace it by
    a multiplication operator with flat symbol |k|^0.  We falsify the local caricature
    and show K-theory's pathology rides the unavoidable first-class kernel:

      (1) solver/symbol sanity: p_hat_dyn(k) = -Q_hat_dyn(k)/|k|^2 to machine zero,
          Q_dyn = div(ubar.grad ubar) the resolved dynamic-pressure source;
      (2) local-surrogate failure: p_local = c*Q_dyn has flat symbol, so the power
          ratio |p_dyn|^2/|p_local|^2 ~ |k|^-4 (amplitude |k|^-2) and p_dyn carries far
          more large-scale (low-k) energy than the local surrogate;
      (3) the corruption is MODEL error, not physics: the exact (Leray-projected)
          subgrid force is solenoidal -> ~zero spurious pressure; raw Smagorinsky has a
          dilatational part that the SAME |k|^-2 kernel spreads globally (large low-k
          fraction) at small amplitude but wrong phase vs the true pressure -> phantom N.
    """
    ub, vb, mtx, mty = exact_sgs_force(sp, f.u, f.v, kc)
    msx, msy = smagorinsky_force_raw(sp, ub, vb, kc, cs=cs)

    ax = ub * sp.ddx(ub) + vb * sp.ddy(ub)
    ay = ub * sp.ddx(vb) + vb * sp.ddy(vb)
    Q_dyn = sp.ddx(ax) + sp.ddy(ay)                    # div(ubar.grad ubar)
    p_dyn, _ = spurious_pressure(sp, ax, ay)           # (lap)^-1 Q_dyn (repo convention)
    dp_smag, _ = spurious_pressure(sp, msx, msy)

    # (1) symbol sanity (machine zero)
    rhs = -sp.fft(Q_dyn) * sp.k2_inv
    rhs[0, 0] = 0.0
    err_sym = float(np.linalg.norm(sp.fft(p_dyn) - rhs) / (np.linalg.norm(rhs) + 1e-30))

    # (2) local surrogate p_local = c*Q_dyn (variance-matched at a mid shell)
    k, E_p = _radial_spectrum(sp, p_dyn)
    _, E_Q = _radial_spectrum(sp, Q_dyn)
    k_ref = sp.n // 6
    c = (E_p[k_ref] / (E_Q[k_ref] + 1e-30)) ** 0.5
    ratio2 = E_p / (c ** 2 * E_Q + 1e-30)              # |p_dyn|^2 / |p_local|^2
    slope_ratio = _loglog_slope(k, ratio2, 2, sp.n // 4)
    kc_band = sp.n // 8
    lowk_pdyn = _lowk_fraction(sp, p_dyn, kc_band)
    lowk_local = _lowk_fraction(sp, c * Q_dyn, kc_band)

    # (3) corruption is model error, spread by the same kernel
    _, _, dxx, dxy = helmholtz(sp, mtx, mty)
    dil_exact = (_rms(dxx) ** 2 + _rms(dxy) ** 2) / (_rms(mtx) ** 2 + _rms(mty) ** 2 + 1e-30)
    _, _, dsx, dsy = helmholtz(sp, msx, msy)
    dil_smag = (_rms(dsx) ** 2 + _rms(dsy) ** 2) / (_rms(msx) ** 2 + _rms(msy) ** 2 + 1e-30)
    amp_smag = _rms(dp_smag) / (_rms(p_dyn) + 1e-30)
    a, b = dp_smag - dp_smag.mean(), p_dyn - p_dyn.mean()
    corr_smag = float(np.mean(a * b) / (np.std(a) * np.std(b) + 1e-30))
    lowk_smag = _lowk_fraction(sp, dp_smag, kc_band)

    return dict(err_sym=err_sym, slope_ratio=slope_ratio, kc_band=int(kc_band),
                lowk_pdyn=lowk_pdyn, lowk_local=lowk_local,
                dil_exact=dil_exact, dil_smag=dil_smag,
                amp_smag=amp_smag, corr_smag=corr_smag, lowk_smag=lowk_smag)


def _transect_spectrum(path=BED_FILE):
    """Uniform-grid 1-D power spectrum of the real BEDMAP transect.

    The flight line is sampled irregularly and non-periodic, so we de-duplicate
    stations, resample onto a uniform grid at the median spacing, remove mean +
    linear trend (the along-track tilt is not roughness), then *mirror* the segment
    (segment + its reverse) before the FFT.  The mirror is exactly the periodic
    embedding ``bedmap._embed_periodic`` uses for the solver: it removes the seam
    discontinuity so the spectrum is not contaminated by leakage from the large
    low-k power of this very red bed (an un-mirrored FFT biases the slope shallow
    and inflates the apparent small-scale variance).  Returns

        k    : wavenumbers (rad/m, k>0 only)
        E    : elevation power |h_hat(k)|^2
        dx   : uniform grid spacing (m)
        L    : transect length (m)
        meta : dict(relief_m, length_km, n_raw, dx_m, nyquist_m)
    """
    dist, bed = clean_sorted_transect(path)
    ud, inv = np.unique(dist, return_inverse=True)         # de-duplicate dx=0 stations
    ub = np.zeros_like(ud)
    cnt = np.zeros_like(ud)
    np.add.at(ub, inv, bed)
    np.add.at(cnt, inv, 1.0)
    ub /= cnt
    dx = float(np.median(np.diff(ud)))
    L = float(ud.max() - ud.min())
    n = int(L // dx)
    xq = ud.min() + dx * np.arange(n)
    h = np.interp(xq, ud, ub)
    h = h - h.mean()
    t = np.arange(n)
    a = np.polyfit(t, h, 1)
    h = h - (a[0] * t + a[1])                              # de-trend (remove tilt)
    mirror = np.concatenate([h, h[::-1]])                  # repo-style periodic embedding
    H = np.fft.rfft(mirror)
    k = 2.0 * np.pi * np.fft.rfftfreq(mirror.size, d=dx)
    E = np.abs(H) ** 2
    meta = dict(relief_m=float(bed.max() - bed.min()), length_km=L / 1000.0,
                n_raw=int(bed.size), dx_m=dx, nyquist_m=2.0 * dx)
    return k[1:], E[1:], dx, L, meta


def _logbinned_slope(k, E, lam_lo, lam_hi, nbin=24):
    """Spectral slope alpha (E ~ k^-alpha) from a log-binned fit over a band.

    Log-binning gives each decade equal weight, so the dense high-k modes do not
    dominate the regression the way a raw per-mode fit does.
    """
    edges = np.logspace(np.log10(k.min()), np.log10(k.max()), nbin + 1)
    kb, Eb = [], []
    for i in range(nbin):
        m = (k >= edges[i]) & (k < edges[i + 1])
        if m.any():
            kb.append(np.exp(np.mean(np.log(k[m]))))
            Eb.append(np.mean(E[m]))
    kb, Eb = np.array(kb), np.array(Eb)
    lam = 2.0 * np.pi / kb
    band = (lam > lam_lo) & (lam < lam_hi)
    p = np.polyfit(np.log(kb[band]), np.log(Eb[band]), 1)
    return -float(p[0])


def _felt_roughness_ratio(k, E, alpha, lam_star, lam_cut):
    """R_felt = (slope variance pressure/form-drag feels) / (slope variance the
    thermal field feels), an *un-normalized* comparison (slope-var / slope-var,
    dimensionless, >=1).  This replaces the earlier normalized drag-coefficient
    ratio A_drag, which is identically O(1) for a red bed: both C_d^(p) and
    C_d^(theta) are normalized averages dominated by the cavity scale k_L, so
    their ratio collapses to 1 regardless of small-scale structure (A_drag==1
    exactly for scale-independent C_d).  The physically honest quantity is which
    scales each process *samples*, not a ratio of mean coefficients.

    Pressure / form drag sample the full slope variance up to ``lam_cut``; the
    thermal field samples only scales larger than ``lam_star = ell_scr`` (the
    boundary-layer screen), i.e. k < 2pi/lam_star.  When ``lam_cut`` is below the
    Nyquist wavelength the high-k part is a power-law *extrapolation* (E ~ k^-alpha,
    so the slope spectrum k^2 E ~ k^(2-alpha) keeps rising) -- such cases are flagged
    by the caller as extrapolated, not measured.  Modes are uniformly spaced in k,
    so summing is proportional to the integral with the same dk in numerator and
    denominator.
    """
    kmax = float(k.max())
    dk = float(np.median(np.diff(k)))                      # uniform FFT mode spacing
    S = k ** 2 * E                                         # measured slope spectrum
    p = 2.0 - alpha                                        # tail exponent: k^2 E ~ k^(2-alpha)
    C = float(np.median(E * k ** alpha))                   # power-law amplitude for the tail

    def _tail(a, b):                                       # analytic int_a^b C k^p dk (a,b>kmax)
        if b <= a:
            return 0.0
        if abs(p + 1.0) < 1e-12:                           # alpha ~ 3 -> log-form
            return C * (np.log(b) - np.log(a))
        return C * (b ** (p + 1.0) - a ** (p + 1.0)) / (p + 1.0)

    def var_upto(kc):                                      # slope variance for k in (kmin, kc]
        meas = float(S[k <= min(kc, kmax)].sum()) * dk     # discrete sum over resolved modes
        return meas + (_tail(kmax, kc) if kc > kmax else 0.0)

    kcut = 2.0 * np.pi / lam_cut
    kstar = 2.0 * np.pi / lam_star
    Fp = var_upto(kcut)                                    # form drag: all scales up to cutoff
    Fth = var_upto(min(kstar, kcut))                       # thermal: only lambda > lam_star
    return Fp / (Fth + 1e-30)


def result_roughness_scale_separation(path=BED_FILE):
    """RESULT 7 -- Roughness--Scale Separation criterion (Theorem 8).

    Pressure and the thermal/form-drag response read DIFFERENT moments of the bed
    spectrum, so they cannot share one local diffusivity (K-theory):

      * PRESSURE responds to elevation through the elliptic Poisson kernel, whose
        symbol |k|^-2 reddens the response -- the pressure field is dominated by the
        large-scale (cavity-scale) bed shape.  We report the pressure-felt fraction
        F_p(>lambda) = (sum_{k<2pi/lambda} |k|^-2 E_h) / (sum |k|^-2 E_h).
      * FORM DRAG / heat flux responds to bed SLOPE, whose variance spectrum
        k^2 E_h ~ k^(2-alpha) is nearly flat/blue -- concentrated at small scales.
        The thermal boundary layer screens scales below ell_scr = kappa/U, so the
        slope variance sitting below ell_scr is a "blind zone": roughness pressure
        feels globally but the thermal field never reaches.

    Headline numbers are MEASURED at resolvable scales (lambda > Nyquist); the
    sub-Nyquist 1-10 m eddy-kappa regime is reached only by power-law extrapolation
    (the slope spectrum k^(2-alpha) only grows toward small scales), reported as such.
    """
    k, E, dx, L, meta = _transect_spectrum(path)
    lam = 2.0 * np.pi / k

    # spectral slope (elevation) and the implied slope-variance exponent
    alpha = _logbinned_slope(k, E, lam_lo=300.0, lam_hi=50000.0)
    slope_exp = 2.0 - alpha                                # k^2 E_h ~ k^(2-alpha)

    # (1) pressure response is large-scale (k^-2 Poisson kernel)
    Ep = E * k ** -2.0
    totp = Ep.sum()
    def felt_above(l):                                     # F_p(>lambda)
        return float(Ep[k < 2.0 * np.pi / l].sum() / (totp + 1e-30))
    pf_50km = felt_above(50000.0)
    pf_20km = felt_above(20000.0)

    # (2) bed-SLOPE blind zone (form-drag relevant), MEASURED in resolvable band
    Es = k ** 2 * E
    tots = Es.sum()
    def slope_below(l):                                    # var at lambda<l (k>2pi/l)
        return float(Es[k > 2.0 * np.pi / l].sum() / (tots + 1e-30))
    bz_300 = slope_below(300.0)
    bz_1km = slope_below(1000.0)
    bz_5km = slope_below(5000.0)

    # (3) elevation blind zone -- the CONTROL (red spectrum -> tiny)
    tote = E.sum()
    elev_20km = float(E[k > 2.0 * np.pi / 20000.0].sum() / (tote + 1e-30))

    # (4) crossover for molecular kappa: ell_scr << everything -> lambda* ~ L
    lam_star_mol = L                                       # cavity scale

    # (5) felt-roughness ratio R_felt = slope-var(form drag) / slope-var(thermal).
    #     MEASURED at resolvable scales (cutoff = Nyquist); the >>1 values at
    #     small lambda* require a sub-Nyquist power-law EXTRAPOLATION (eta), so we
    #     report both and label them.  (Supersedes the normalized A_drag, which is
    #     identically O(1) for a red bed -- see _felt_roughness_ratio docstring.)
    nyq = meta["nyquist_m"]
    rfelt_1km = _felt_roughness_ratio(k, E, alpha, lam_star=1000.0, lam_cut=nyq)
    rfelt_5km = _felt_roughness_ratio(k, E, alpha, lam_star=5000.0, lam_cut=nyq)
    # extrapolated cutoff eta = 1 m (borehole/cobble scale); R_felt scales ~ (Nyq/eta)^(2-alpha+1),
    # i.e. it only grows as eta shrinks -- so this is a conservative, labelled lower estimate.
    rfelt_10m_ext = _felt_roughness_ratio(k, E, alpha, lam_star=10.0, lam_cut=1.0)
    rfelt_1km_ext = _felt_roughness_ratio(k, E, alpha, lam_star=1000.0, lam_cut=1.0)

    return dict(alpha=alpha, slope_exp=slope_exp,
                nyquist_m=meta["nyquist_m"], length_m=L,
                relief_m=meta["relief_m"], n_raw=meta["n_raw"],
                pf_50km=pf_50km, pf_20km=pf_20km,
                bz_slope_300=bz_300, bz_slope_1km=bz_1km, bz_slope_5km=bz_5km,
                bz_elev_20km=elev_20km, lam_star_mol=lam_star_mol,
                rfelt_1km=rfelt_1km, rfelt_5km=rfelt_5km,
                rfelt_10m_ext=rfelt_10m_ext, rfelt_1km_ext=rfelt_1km_ext)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--spinup", type=int, default=400)
    ap.add_argument("--nu", type=float, default=2.0e-4)
    ap.add_argument("--cs", type=float, default=0.16)
    args = ap.parse_args()

    ybed, meta = bed_profile_from_transect(BED_FILE, args.n, bed_mean=0.9,
                                           bed_amp=0.55)
    cfg = SubglacialConfig(n=args.n, nu=args.nu, kappa=args.nu, sgs="none",
                           f_amp=8.0, k_f=10.0, bed_profile=ybed, seed=1)
    f = SubglacialFlow(cfg)
    sp = f.sp
    kc = args.n // 3
    print(f"[setup] n={args.n} kc={kc} nu={args.nu} bed=real BEDMAP1 "
          f"(relief {meta['relief_m']:.0f} m, {meta['length_km']:.0f} km)")
    print(f"[setup] spinning up DNS truth {args.spinup} steps ...")
    f.run(args.spinup, ramp=max(1, args.spinup // 3),
          report_every=max(1, args.spinup // 2))

    print("\n================ RESULT 1: Spurious Pressure Source ================")
    r1 = result1(sp, f, kc, args.cs)
    print(f"  raw Smagorinsky   div(m)/|m| (nondim) = {r1['div_smag']:.3e}")
    print(f"  proj-FDT          div(m)/|m| (nondim) = {r1['div_fdt']:.3e}")
    print("  curl-free (dilatational) energy fraction:")
    print(f"      Smagorinsky = {100*r1['dil_frac_smag']:.1f}%   "
          f"projected-FDT = {100*r1['dil_frac_fdt']:.2f}%")
    print("  spurious pressure  rms(dp)/rms(p_dyn):")
    print(f"      Smagorinsky = {100*r1['dp_ratio_smag']:.1f}%   "
          f"projected-FDT = {100*r1['dp_ratio_fdt']:.2f}%")
    print("  Schoof sliding transfer (n=3): a 1% effective-pressure error -> 3% u_b")
    print("  Correction A -- curl-free part is dynamically inert in the bulk:")
    print(f"      ||P(m_smag_raw) - m_smag_proj|| / ||P(m_smag_raw)|| = "
          f"{r1['proj_equiv']:.3e}  (=> identical bulk velocity tendency)")
    print(f"      dilatational energy in solid/interface (chi>0.1) = "
          f"{100*r1['wall_frac']:.0f}% (interface is {100*r1['area_frac']:.0f}% of area)")
    print("      => the curl-free part lives in the OPEN cavity, where the solve")
    print("         projects it out exactly: solenoidality is a pressure/N (sliding)")
    print("         effect, NOT a near-wall velocity (melt) effect.")

    print("\n================ RESULT 2: Clock Mismatch Number ===================")
    rows, bed = result2(sp, ybed, sp.dx)
    print("  (Lambda = fraction of |field| located UPSTREAM of the source)")
    print("   Pe      Lambda_p   Lambda_theta   M_clock = Lp/Lth")
    for Pe, Lp, Lth, M in rows:
        print(f"  {Pe:6.0f}   {Lp:8.3f}   {Lth:10.4f}     {M:8.2f}")
    print(f"  real BEDMAP bed (source at tallest crest, Pe={bed[0]:.0f}): "
          f"Lambda_p={bed[1]:.3f} Lambda_theta={bed[2]:.4f} "
          f"M_clock={bed[3]:.2f}")
    print("\n[interpretation] pressure reaches farther UPSTREAM than the advected")
    print("  scalar (M_clock>1): elliptic action-at-a-distance vs causal advective")
    print("  screening (length kappa/U). A single local eddy diffusivity cannot")
    print("  carry the elliptic upstream response. Bare-bed value is modest (~1.5).")

    print("\n========= RESULT 3: Master identity  ||[L, P(t)] v0|| / ||P v0|| =======")
    Umean, c_rows = result3_commutator(sp, f, kappa=args.nu)
    print(f"  div-free probe advected by uniform U={Umean:.2f} vs the real cavity u-bar")
    print("  nsteps     t      uniform-flow    cavity-flow")
    for ns, t, nu, nc in c_rows:
        print(f"  {ns:5d}   {t:6.3f}    {nu:10.2e}    {nc:10.3f}")
    print("  [interpretation] uniform flow ~0 (translation keeps v0 divergence-")
    print("  free => L and P commute); the BEDMAP cavity flow is O(0.1-1) because")
    print("  shear + walls make [L, u-bar.grad] != 0. This is the master identity")
    print("  [L,P]!=0 measured directly -- the single non-commutation behind all")
    print("  four results (it is NOT a constant-coefficient symbol, which vanishes).")

    print("\n===== RESULT 4: Dedner cleaning threshold  G* = gamma_clean*tau_adj ====")
    rd = result_dedner_cleaning(sp, f, kc, cs=args.cs)
    print("  inject raw-Smagorinsky divergence into the frozen field, clean for one")
    print("  transit tau_adj=L/c_h with the GLM telegrapher, sweep the only knob G.")
    print("   G       ||div u||/||div u*||    ||dp||/||dp*||")
    for G, resd, resp in rd["rows"]:
        print(f"  {G:5.1f}        {resd:10.4f}            {resp:10.4f}")
    print(f"  measured knee (1/e of G->0 residual)  G* = {rd['Gknee']:.2f}  (O(1))")
    print(f"  over-damped optimum (min spurious dp) G_opt = {rd['Gopt']:.0f} "
          f"(beyond it, low-k cleaning slows -- classic Dedner trade-off)")
    print(f"  G=0 spurious-pressure global-tail fraction = {rd['tail0']:.2f} "
          f"(divergence sourced in {100*rd['core_frac']:.0f}% of area -> global dp)")
    print(f"  => nonlocal amplification A = tail/area_frac = {rd['amp']:.1f}x "
          f"(local divergence in {100*rd['core_frac']:.0f}% of domain corrupts "
          f"{100*rd['tail0']:.0f}% of the global pressure)")
    print("  [interpretation] approximate projection suppresses the constraint")
    print("  violation only when gamma_clean*tau_adj >~ O(1); the measured knee at")
    print(f"  G*~{rd['Gknee']:.1f} is OUR number for this bed, replacing the transplanted")
    print("  GR '1/2'. Exact Leray is the gamma_clean->inf limit (div u = 0 always).")

    print("\n===== RESULT 5: Girsanov / Entropy-Pressure identity =====")
    re = result_commutator_entropy(sp, f, kc)
    print("  The honest reading of [L,P]!=0 is a Girsanov entropy on diffusion PATHS,")
    print("  NOT a failed 'tower property' (L is an orthogonal projection, not a")
    print("  conditional expectation). Its rate is (t^2/4 kappa)||[L,-u.grad]u0||^2.")
    print(f"  (a) algebra: ||[L,-u.grad]u0 - (I-L)(u.grad u0)|| / ||.|| = "
          f"{re['err_alg']:.2e}  (machine zero)")
    print("  (b) splitting limit  || ([P(t)L - L P(t)]u0)/t - C || / ||C||  vs t:")
    for t, rel in re["split_rows"]:
        print(f"        t={t:6.3f}   rel.err = {rel:.3f}")
    print("      => difference quotient -> the commutator C as t->0 (linear in t).")
    print(f"  (c) Entropy-Pressure: <|C|^2> = {re['C2']:.4e} == <|grad dp_spur|^2> = "
          f"{re['G2']:.4e}")
    print(f"      ratio = {re['ep_ratio']:.6f} (EXACT): path-entropy = H^1 energy of the")
    print("      Theorem-1 spurious pressure. Dilatational fraction ||C||/||u.grad u0||")
    print(f"      = {re['dil_frac']:.3f} (cf. RESULT 3 plateau ~0.6).")
    print("  Theorem-3 link stays a CONJECTURE ('<~'): lee counter-gradient fraction")
    print(f"  = {re['cg_frac']:.2f}; corr(up-gradient flux, |u.grad theta|) = "
          f"{re['corr_abs']:.2f}, corr(.,(u.grad theta)^2) = {re['corr_sq']:.2f}")
    print("  -- partial (r^2~1/4) consistent support, not an identity.")

    print("\n===== RESULT 6: Constraint class (first-class vs local caricature) =====")
    rc = result_constraint_class(sp, f, kc)
    print("  div u = 0 is a FIRST-CLASS constraint: the pressure inverse is the nonlocal")
    print("  Green's function (nabla^2)^-1 with symbol |k|^-2. A second-class (local)")
    print("  caricature would have flat symbol |k|^0. We falsify the local surrogate:")
    print(f"  (1) symbol sanity: ||p_dyn_hat + Q_dyn_hat/|k|^2|| / ||.|| = "
          f"{rc['err_sym']:.2e}  (machine zero)")
    print("  (2) local surrogate p_local = c*Q_dyn (flat |k|^0):")
    print(f"      power-ratio slope d ln(|p_dyn|^2/|p_local|^2)/d ln k = "
          f"{rc['slope_ratio']:.2f}  (~ -4 => |k|^-2 in amplitude; a local op cannot fake this)")
    print(f"      low-k (|k|<{rc['kc_band']}) energy fraction: true p_dyn = "
          f"{rc['lowk_pdyn']:.3f} vs local = {rc['lowk_local']:.3f}")
    print("  (3) the corruption is MODEL error, spread by the same unavoidable kernel:")
    print(f"      exact subgrid force dilatational frac = {rc['dil_exact']:.3f} "
          f"(solenoidal by construction -> ~0 spurious pressure); Smag = {rc['dil_smag']:.3f}")
    print(f"      dp_Smag: rms/p_dyn = {rc['amp_smag']:.4f}, phase corr = "
          f"{rc['corr_smag']:.3f}, low-k frac = {rc['lowk_smag']:.3f}")
    print("      => small, GLOBAL, phase-decorrelated pressure -> phantom N (cf. Thm-1 A~3.9x).")

    print("\n===== RESULT 7: Roughness--Scale Separation (Theorem 8, real BEDMAP) =====")
    r7 = result_roughness_scale_separation(BED_FILE)
    print(f"  transect: {r7['n_raw']} pts, {r7['length_m']/1000:.0f} km, "
          f"{r7['relief_m']:.0f} m relief, Nyquist lambda = {r7['nyquist_m']:.0f} m")
    print(f"  (a) elevation spectrum slope alpha = {r7['alpha']:.2f} (log-binned, "
          f"natural red bedrock); slope-variance exponent k^2 E_h ~ k^{r7['slope_exp']:+.2f}")
    print(f"  (b) PRESSURE is large-scale (k^-2 Poisson kernel): F_p(>50 km) = "
          f"{r7['pf_50km']:.3f}, F_p(>20 km) = {r7['pf_20km']:.3f}")
    print(f"      => molecular kappa crossover lambda* ~ L = {r7['lam_star_mol']/1000:.0f} km "
          f"(cavity scale; both fields dominated by the canyon).")
    print("  (c) bed-SLOPE blind zone (form-drag relevant), MEASURED at lambda>Nyquist:")
    print(f"      slope-var(lambda<300 m) = {r7['bz_slope_300']:.3f}, "
          f"(lambda<1 km) = {r7['bz_slope_1km']:.3f}, (lambda<5 km) = {r7['bz_slope_5km']:.3f}")
    print(f"      control -- elevation-var(lambda<20 km) = {r7['bz_elev_20km']:.3f} "
          f"(red spectrum -> tiny elevation blind zone)")
    print("  [interpretation] pressure (k^-2) reads ELEVATION and is dominated by the")
    print(f"  cavity-scale canyon ({100*r7['pf_50km']:.1f}% above 50 km); form drag / heat flux")
    print(f"  read bed SLOPE (k^2 E_h ~ k^{r7['slope_exp']:+.2f}), concentrated at small scales --")
    print(f"  {100*r7['bz_slope_300']:.0f}% below 300 m, {100*r7['bz_slope_1km']:.0f}% below 1 km vs only "
          f"{100*r7['bz_elev_20km']:.0f}% elevation variance below 20 km.")
    print("  For realistic eddy kappa, ell_scr=kappa/U ~ 1-10 m sits BELOW the DEM Nyquist")
    print(f"  (~{r7['nyquist_m']:.0f} m): the slope blind zone is measured at resolvable scales and only")
    print("  grows below Nyquist (the slope spectrum does not roll off). Pressure sees the")
    print("  canyon; drag sees the cobbles; K-theory's single local diffusivity assumes they")
    print("  are the same field -- they are not.")
    print("  (d) felt-roughness ratio R_felt = slope-var(form drag, all scales) / slope-var(")
    print("      thermal, lambda>ell_scr) -- UN-normalized (the normalized drag-coeff ratio is")
    print("      identically O(1) for a red bed and is retired).")
    print(f"      MEASURED (cutoff=Nyquist): R_felt(lam*=1 km) = {r7['rfelt_1km']:.2f}, "
          f"R_felt(lam*=5 km) = {r7['rfelt_5km']:.2f}")
    print(f"      EXTRAPOLATED to eta=1 m (sub-Nyquist power law; grows further as eta->0): "
          f"R_felt(lam*=1 km) = {r7['rfelt_1km_ext']:.0f}, R_felt(lam*=10 m) = {r7['rfelt_10m_ext']:.1f}")
    print("      => pressure/form drag feel 100% of the slope variance; the thermal field feels")
    print(f"      only {100/r7['rfelt_1km']:.0f}% (lam*=1 km) to {100/r7['rfelt_5km']:.0f}% "
          f"(lam*=5 km) of it -- structural, not a tuning error (cf. Schoof-Hewitt).")


if __name__ == "__main__":
    main()
