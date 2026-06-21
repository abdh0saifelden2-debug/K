r"""§I.6 — SPATIAL early-warning of ungrounding: variance and along-flow correlation
length of surface speed rise toward the grounding line as N -> N_c.

Why this exists (vs §I.3 temporal CSD and §I.4 tidal EWS)
--------------------------------------------------------
§I.3 derived a *temporal* critical-slowing-down precursor (rising variance + lag-1
autocorrelation in time) from the regularized-Coulomb fold at N_c, and §I.4 made it
operational with tides. Both need a *time series*. This module gives the **spatial**
counterpart, which needs only a **single velocity snapshot** (e.g. one ITS_LIVE
mosaic): along a flowline N decreases toward the grounding line, so the basal
restoring stiffness ``lambda(N(x)) ∝ (1-R)^2/R`` (``R=(N_c/N)^m``) vanishes there.
A spatially-coupled (longitudinal-stress) velocity field then has, at stationarity,

    Var(x)  ∝ 1 / sqrt(D * lambda(x))         (rising toward the GL)
    xi(x)   ∝ sqrt(D / lambda(x))             (correlation length, rising toward GL)

i.e. **rising spatial variance and along-flow correlation length toward the
grounding line** — a spatial early-warning signal (Dakos et al. 2010), here for
marine ungrounding, derived from the sliding law.

Method: the linear stochastic field ``du/dt = (D d2/dx2 - lambda(x)) u + xi`` has a
stationary covariance ``Sigma`` solving the Lyapunov equation
``A Sigma + Sigma A^T = -2 D_noise I`` with ``A = D*Laplacian - diag(lambda)`` — solved
exactly (no time-stepping, no CFL limit). No GPU, no data download.

Honest scope: a reduced 1-D longitudinal-coupling model (D lumps the membrane-stress
coupling length); the correlation length saturates near the GL at the domain/coupling
scale (shown). The field test computes spatial variance + along-flow correlation of an
ITS_LIVE speed field vs distance-to-GL — single-snapshot, no time series needed.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORTS = os.path.normpath(os.path.join(_HERE, "..", "reports"))

TAU_D, C_FRIC, U0, M_EXP = 3.0e4, 0.5, 100.0, 3.0
N_C = TAU_D / C_FRIC


def drag_stiffness(N, m=M_EXP, N_c=N_C):
    R = (N_c / N) ** m
    return (1.0 - R) ** 2 / R


def lambda_profile(N, N_ref=6.0e5, m=M_EXP, N_c=N_C):
    """Restoring rate normalised to 1 at N_ref; -> 0 as N -> N_c."""
    return drag_stiffness(N, m, N_c) / drag_stiffness(np.array(N_ref), m, N_c)


def stationary_covariance(N_x, D, dx, D_noise=1.0):
    """Exact stationary covariance of du/dt=(D d2/dx2 - diag(lambda(N_x)))u + xi via
    the continuous Lyapunov equation A Sigma + Sigma A^T = -2 D_noise I."""
    from scipy.linalg import solve_continuous_lyapunov
    nx = N_x.size
    lam = lambda_profile(N_x)
    Lap = np.zeros((nx, nx))
    for i in range(1, nx - 1):
        Lap[i, i - 1] = Lap[i, i + 1] = 1.0 / dx ** 2
        Lap[i, i] = -2.0 / dx ** 2
    Lap[0, 0] = -1.0 / dx ** 2; Lap[0, 1] = 1.0 / dx ** 2          # Neumann ends
    Lap[-1, -1] = -1.0 / dx ** 2; Lap[-1, -2] = 1.0 / dx ** 2
    A = D * Lap - np.diag(lam)
    Sigma = solve_continuous_lyapunov(A, -2.0 * D_noise * np.eye(nx))
    return Sigma, lam


def _corr_length(Sigma, i, dx):
    row = Sigma[i] / Sigma[i, i]
    below = np.where(row < 1.0 / np.e)[0]
    if below.size == 0:
        return np.nan
    j = below[np.argmin(np.abs(below - i))]
    return abs(j - i) * dx


def run(nx=160, D=2.0e-3, N_far=6.0e5, N_gl=1.25 * N_C):
    x = np.linspace(0.0, 1.0, nx)
    dx = float(x[1] - x[0])
    N_x = np.linspace(N_far, N_gl, nx)               # N declines toward GL at x=1
    Sigma, lam = stationary_covariance(N_x, D, dx)
    var = np.diag(Sigma)
    xi = np.array([_corr_length(Sigma, i, dx) for i in range(nx)])
    xi_pred = np.sqrt(D / lam)
    var_pred = 1.0 / np.sqrt(D * lam)                 # ∝ 1/sqrt(D lambda)
    from scipy.stats import kendalltau
    tau_var = float(kendalltau(x, var)[0])
    ok = np.isfinite(xi)
    tau_xi = float(kendalltau(x[ok], xi[ok])[0])
    # interior plateau check: var*sqrt(lambda) ~ const away from the boundaries
    interior = slice(nx // 8, nx - nx // 8)
    plateau = var[interior] * np.sqrt(lam[interior])
    plateau_cv = float(np.std(plateau) / np.mean(plateau))
    dec = max(1, nx // 50)
    idx = list(range(0, nx, dec))
    return dict(
        what="spatial early-warning of ungrounding: variance + along-flow correlation "
             "length of surface speed rise toward the grounding line as N->N_c",
        sliding_law="regularized-Coulomb; N_c=%.3f MPa, m=%.0f" % (N_C / 1e6, M_EXP),
        scaling="Var ∝ 1/sqrt(D*lambda(x)); xi ∝ sqrt(D/lambda(x)); lambda ∝ (1-R)^2/R -> 0 at N_c",
        D_coupling=D, nx=nx,
        kendall_tau_variance_toward_GL=tau_var,
        kendall_tau_corrlength_toward_GL=tau_xi,
        var_ratio_GL_over_interior=float(var[-nx // 16:].mean() / var[nx // 16:nx // 8].mean()),
        corrlen_ratio_GL_over_interior=float(np.nanmean(xi[-nx // 16:]) /
                                             np.nanmean(xi[nx // 16:nx // 8])),
        interior_var_sqrtlambda_cv=plateau_cv,
        rising_spatial_ews=bool(tau_var > 0.7 and tau_xi > 0.5),
        single_snapshot_advantage=("computed from one velocity field (no time series): "
                                   "spatial variance + along-flow correlation vs distance-to-GL"),
        field_test=("ITS_LIVE speed along flowlines, binned by distance-to-grounding-line; "
                    "expect variance + along-flow correlation length to rise toward the GL."),
        references="Dakos et al. 2010 (spatial EWS); Scheffer et al. 2009; Schoof 2007 (MISI)",
        series=dict(x=[x[i] for i in idx], N_MPa=[N_x[i] / 1e6 for i in idx],
                    lambda_=[lam[i] for i in idx], var=[var[i] for i in idx],
                    var_pred=[var_pred[i] for i in idx],
                    corr_length=[None if not np.isfinite(xi[i]) else xi[i] for i in idx],
                    corr_length_pred=[xi_pred[i] for i in idx]),
        verdict=(
            f"Exact stationary covariance gives spatial variance rising x"
            f"{var[-nx//16:].mean()/var[nx//16:nx//8].mean():.0f} and along-flow correlation "
            f"length rising x{np.nanmean(xi[-nx//16:])/np.nanmean(xi[nx//16:nx//8]):.0f} toward "
            f"the grounding line (Kendall tau var={tau_var:.2f}, xi={tau_xi:.2f}); interior "
            f"Var*sqrt(lambda) constant to CV={plateau_cv:.2f} confirms the 1/sqrt(D*lambda) law. "
            "A single-snapshot spatial early-warning for ungrounding."),
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = res["series"]
    x = np.array(s["x"]); N = np.array(s["N_MPa"]); var = np.array(s["var"])
    vpred = np.array(s["var_pred"]); xi = np.array([np.nan if v is None else v for v in s["corr_length"]])
    xipred = np.array(s["corr_length_pred"])
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax[0].plot(x, var / var[:len(var)//8].mean(), "o-", color="#d95f02", ms=3,
               label="variance (sim, norm)")
    ax[0].plot(x, vpred / vpred[:len(vpred)//8].mean(), "k--", lw=1,
               label=r"$\propto 1/\sqrt{D\lambda(x)}$")
    ax[0].set_xlabel("along-flow x  (GL at x=1, N -> N_c)")
    ax[0].set_ylabel("spatial variance (norm)")
    ax[0].set_title(f"(a) variance rises toward GL  (tau={res['kendall_tau_variance_toward_GL']:.2f})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(x, xi, "o-", color="#1b9e77", ms=3, label="correlation length (sim)")
    ax[1].plot(x, xipred, "k--", lw=1, label=r"$\propto \sqrt{D/\lambda(x)}$")
    ax[1].set_xlabel("along-flow x  (GL at x=1)")
    ax[1].set_ylabel("along-flow correlation length")
    ax[1].set_title(f"(b) correlation length rises toward GL  (tau={res['kendall_tau_corrlength_toward_GL']:.2f})")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("Spatial early-warning of ungrounding (single-snapshot): variance + "
                 "correlation length rise toward the grounding line", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_REPORTS, "spatial_ews.json"))
    a = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("=== spatial early-warning of ungrounding ===")
    print(f"  var rise toward GL x{res['var_ratio_GL_over_interior']:.0f} "
          f"(Kendall tau={res['kendall_tau_variance_toward_GL']:.2f})")
    print(f"  corr-length rise toward GL x{res['corrlen_ratio_GL_over_interior']:.0f} "
          f"(Kendall tau={res['kendall_tau_corrlength_toward_GL']:.2f})")
    print(f"  interior Var*sqrt(lambda) CV={res['interior_var_sqrtlambda_cv']:.2f} (1/sqrt(D lambda) law)")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
