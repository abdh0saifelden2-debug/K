r"""NR34 -- the ice kernel IS a (tempered) half-order fractional derivative:
Warburg interface, Mittag-Leffler relaxation, and the Ste^2 visibility window.

Ledger item E2 (papers/RESEARCH_RECAP_AND_HORIZON.md).  Everything here is
about the exact ice-side memory kernel derived and PDE-validated in
``glaciers/validation/synthetic/ice_kernel_synthetic.py`` (P4a SSB.2 / SSV.4):

    H(s) = q_ice'(s)/v'(s) = A (1 - sqrt(1 + 4 tau_d s)) / s,
    A = k_th theta_far Vbar^2 / (2 kappa^2) (< 0),   tau_d = kappa / Vbar^2.

Claim 1 (EXACT operator identity -- tempered half-derivative).
    sqrt(1 + 4 tau_d s) = 2 sqrt(tau_d) sqrt(s + lam),  lam = 1/(4 tau_d)
exactly, so with the Warburg coefficient W = -2 A sqrt(tau_d) (> 0):

    H(s) = A/s + W sqrt(s + lam) / s        -- EXACT, all s.

``sqrt(s + lam)`` is the Laplace symbol of the *exponentially tempered*
half-derivative (Meerschaert & Sabzikar 2014); lam = 1/(4 tau_d) is the
tempering rate, i.e. the kernel's exp(-t/4 tau_d) cutoff is the tempering
factor.  The B.2 kernel is not *like* a fractional operator -- it *is* one.

Claim 2 (Warburg / Caputo-1/2 limit).  For |s| tau_d >> 1,

    H(s) -> W s^{-1/2}   (relative error -> 0 as (s tau_d)^{-1/2}),

i.e. q_ice = W I^{1/2} v: the flux is the half-order fractional INTEGRAL of
the interface velocity (equivalently the half-DERIVATIVE of interface
displacement -- the classic Caputo flux law of a half-space).  NOTE: the
horizon ledger wrote "q_ice ~ d^{1/2} v/dt^{1/2}"; the correct reading is
I^{1/2} on velocity / D^{1/2} on displacement (fixed here).  On the imaginary
axis the phase of H tends to exactly -45 deg -- a Warburg element, the E3
(EIS/Randles) contact point.  The coefficient closes a circle: algebraically,

    W = (k_th/sqrt(kappa)) * |thetabar'(0)|,   thetabar'(0) = theta_far Vbar/kappa,

the textbook half-space Caputo coefficient evaluated on the advected base
gradient (unit-proved as an identity).

Claim 3 (coupled interface = fractional relaxation, in closed form).  Close
the loop with the Stefan condition rho_i L_f v = q_w - (G * v).  In the
Warburg window the interface obeys the fractional (Langevin-type) equation

    rho_i L_f v(t) + W I^{1/2} v(t) = q_w(t),

whose step response is EXACTLY Mittag-Leffler of order 1/2:

    v(t) = v_0 E_{1/2}(-sqrt(t/tau_f)) = v_0 e^{t/tau_f} erfc(sqrt(t/tau_f)),
    v_0 = q_0/(rho_i L_f),   tau_f = (rho_i L_f / W)^2,

computable to machine precision as ``scipy.special.erfcx``.  Asymptotics:
1 - 2 sqrt(t/(pi tau_f)) at short time; the heavy ALGEBRAIC tail
sqrt(tau_f/(pi t)) at long time -- no exponential stage: the classic
fractional-relaxation signature (Mainardi; Metzler & Klafter 2000).

Claim 4 (the visibility window is the square of the Stefan number).  The
fractional stage is only visible while the Warburg limit holds (t << tau_d),
and the relaxation lives on tau_f; their ratio collapses to a pure number:

    tau_f / tau_d = Ste^{-2},      Ste = c |theta_far| / L_f.

Ste >> 1 (strong subcooling: lab dissolution/ablation analogues) => the full
Mittag-Leffler stage including the t^{-1/2} tail plays out inside the window;
Ste << 1 (glacial: Ste ~ 6e-3 per kelvin of subcooling) => tau_f >> tau_d and
the kernel's tempering cuts the fractional stage off first -- the coupled
response collapses to the quasi-steady limit v_inf = v_0/(1 + Ste) with only
a short sqrt(t) transient.  One dimensionless group decides whether an
interface is "fractional" or "classical" -- a falsifiable design rule for
where half-order interface dynamics can actually be observed.

Verification (this module + 13 unit proofs):
  * tempered identity: machine-precision over a complex-s grid;
  * Warburg limit + exact -45 deg asymptotic phase;
  * W-coefficient algebraic identity vs the classic half-space law;
  * E_{1/2} solution solves the fractional integral equation (quadrature);
  * product-integration Volterra solver for the FULL kernel closed loop,
    validated against the exact Mittag-Leffler answer on the pure-Warburg
    kernel, then run at Ste = 5 (window open: matches ML inside t << tau_d)
    and Ste = 6.3e-3 (window shut: quasi-steady limit, ML stage absent);
  * long-time limits v_inf = v_0/(1+Ste) from H(0) = -rho c theta_far.

Artifacts -> figures/nr34_fractional_ice_kernel.{json,png}
Run:  python general_two_clocks/new_relationships11.py
Test: pytest general_two_clocks/tests/test_new_relationships11.py -v
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.special import erfc, erfcx

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
_GLACIER_SYN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "glaciers", "validation", "synthetic")
sys.path.insert(0, _GLACIER_SYN)
from ice_kernel_synthetic import kernel_G  # noqa: E402  (PDE-validated, SSV.4)


# ------------------------------------------------------------ exact symbols --
def coeff_A(kappa=1.0, Vbar=1.0, theta_far=-1.0, rho_c=1.0):
    """A = k_th theta_far Vbar^2 / (2 kappa^2)  [W/m^3 per (m/s)]."""
    return rho_c * kappa * theta_far * Vbar**2 / (2.0 * kappa**2)


def tau_diff(kappa=1.0, Vbar=1.0):
    """Advective thermal time tau_d = kappa / Vbar^2."""
    return kappa / Vbar**2


def coeff_W(kappa=1.0, Vbar=1.0, theta_far=-1.0, rho_c=1.0):
    """Warburg coefficient W = -2 A sqrt(tau_d)  (> 0 for theta_far < 0)."""
    return -2.0 * coeff_A(kappa, Vbar, theta_far, rho_c) \
        * np.sqrt(tau_diff(kappa, Vbar))


def coeff_W_classic(kappa=1.0, Vbar=1.0, theta_far=-1.0, rho_c=1.0):
    """The textbook half-space Caputo coefficient (k_th/sqrt(kappa)) |thetabar'(0)|.

    thetabar'(0) = theta_far Vbar / kappa is the advected base gradient; the
    claim (unit-proved) is that this EQUALS coeff_W identically.
    """
    k_th = rho_c * kappa
    return k_th / np.sqrt(kappa) * (-theta_far) * Vbar / kappa


def H_exact(s, kappa=1.0, Vbar=1.0, theta_far=-1.0, rho_c=1.0):
    """Exact transfer function A(1 - sqrt(1 + 4 tau_d s))/s (complex-safe)."""
    s = np.asarray(s, dtype=complex)
    A = coeff_A(kappa, Vbar, theta_far, rho_c)
    td = tau_diff(kappa, Vbar)
    return A * (1.0 - np.sqrt(1.0 + 4.0 * td * s)) / s


def H_tempered(s, kappa=1.0, Vbar=1.0, theta_far=-1.0, rho_c=1.0):
    """A/s + W sqrt(s + lam)/s with lam = 1/(4 tau_d) -- claimed EXACT rewrite."""
    s = np.asarray(s, dtype=complex)
    A = coeff_A(kappa, Vbar, theta_far, rho_c)
    W = coeff_W(kappa, Vbar, theta_far, rho_c)
    lam = 1.0 / (4.0 * tau_diff(kappa, Vbar))
    return A / s + W * np.sqrt(s + lam) / s


def H_warburg(s, kappa=1.0, Vbar=1.0, theta_far=-1.0, rho_c=1.0):
    """Pure Warburg limit W s^{-1/2} (valid |s| tau_d >> 1)."""
    s = np.asarray(s, dtype=complex)
    return coeff_W(kappa, Vbar, theta_far, rho_c) / np.sqrt(s)


# --------------------------------------------------- fractional closed loop --
def stefan_number(theta_far=-1.0, rho_c=1.0, rhoL=1.0):
    """Ste = c |theta_far| / L_f = rho_c |theta_far| / (rho L_f)."""
    return rho_c * abs(theta_far) / rhoL


def tau_frac(kappa=1.0, Vbar=1.0, theta_far=-1.0, rho_c=1.0, rhoL=1.0):
    """Fractional relaxation time tau_f = (rho L_f / W)^2."""
    return (rhoL / coeff_W(kappa, Vbar, theta_far, rho_c)) ** 2


def ml_step_response(t, kappa=1.0, Vbar=1.0, theta_far=-1.0, rho_c=1.0,
                     rhoL=1.0, q0=1.0):
    """Exact Warburg-window step response v(t) = v0 E_{1/2}(-sqrt(t/tau_f)).

    E_{1/2}(-x) = e^{x^2} erfc(x) = scipy.special.erfcx(x) -- machine precision,
    no Mittag-Leffler numerics needed.
    """
    t = np.asarray(t, dtype=float)
    tf = tau_frac(kappa, Vbar, theta_far, rho_c, rhoL)
    return (q0 / rhoL) * erfcx(np.sqrt(np.maximum(t, 0.0) / tf))


def _cum_S_on_grid(tgrid, kappa, Vbar, theta_far, rho_c, M=200001):
    """S(t) = integral_0^t G on the requested grid via the u = sqrt(t)
    substitution (kills the integrable t^{-1/2} singularity), then interp."""
    t_end = float(tgrid[-1])
    u = np.linspace(0.0, np.sqrt(t_end), M)
    t = u**2
    Gv = kernel_G(t, kappa, Vbar, theta_far, rho_c)
    integrand = 2.0 * u * Gv
    integrand[0] = 0.0
    S = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(u))])
    return np.interp(tgrid, t, S)


def _cum_S_warburg(tgrid, kappa, Vbar, theta_far, rho_c):
    """S(t) for the untempered Warburg kernel G_W = W t^{-1/2}/sqrt(pi):
    S_W(t) = 2 W sqrt(t/pi) (exact)."""
    W = coeff_W(kappa, Vbar, theta_far, rho_c)
    return 2.0 * W * np.sqrt(np.asarray(tgrid, float) / np.pi)


def volterra_step(t_end, n=2000, kappa=1.0, Vbar=1.0, theta_far=-1.0,
                  rho_c=1.0, rhoL=1.0, q0=1.0, kernel="full"):
    """Closed-loop step response by product-integration of the Volterra
    equation  rhoL v(t) = q0 - integral_0^t G(t-t') v(t') dt'.

    The weakly singular kernel is handled exactly through its running
    integral S: each sub-interval contributes (v_j + v_{j+1})/2 *
    [S(t_n - t_j) - S(t_n - t_{j+1})]; the last sub-interval makes the
    scheme implicit in v_n (linear solve per step).
    """
    tg = np.linspace(0.0, t_end, n + 1)
    S = (_cum_S_on_grid(tg, kappa, Vbar, theta_far, rho_c) if kernel == "full"
         else _cum_S_warburg(tg, kappa, Vbar, theta_far, rho_c))
    # dS[m] = S(t_{m+1}) - S(t_m) = integral of G over [t_m, t_{m+1}]
    dS = np.diff(S)
    v = np.empty(n + 1)
    v[0] = q0 / rhoL                      # instantaneous response (S(0)=0)
    for m in range(1, n + 1):
        # sum_{j=0}^{m-1} (v_j + v_{j+1})/2 * dS[m-1-j]; split off j = m-1
        w = dS[m - 1 - np.arange(m - 1)]  # weights for j = 0..m-2
        known = 0.5 * np.dot(v[:m - 1] + v[1:m], w) if m > 1 else 0.0
        known += 0.5 * v[m - 1] * dS[0]
        v[m] = (q0 - known) / (rhoL + 0.5 * dS[0])
    return tg, v


def half_integral(f_vals, tgrid):
    """Riemann-Liouville I^{1/2} f on a uniform grid by product integration
    of (t - tau)^{-1/2}/Gamma(1/2) with f piecewise-linear (midpoint value)."""
    tg = np.asarray(tgrid, float)
    n = tg.size
    out = np.zeros(n)
    for m in range(1, n):
        seg = 2.0 * (np.sqrt(tg[m] - tg[:m]) - np.sqrt(tg[m] - tg[1:m + 1]))
        out[m] = np.dot(0.5 * (f_vals[:m] + f_vals[1:m + 1]), seg) / np.sqrt(np.pi)
    return out


# --------------------------------------------------------------- diagnostics --
def tempered_identity_error(ngrid=41):
    """max relative |H_exact - H_tempered| over a log complex-s grid."""
    re = np.logspace(-3, 3, ngrid)
    im = np.logspace(-3, 3, ngrid)
    S = (re[:, None] + 1j * im[None, :]).ravel()
    e = np.abs(H_exact(S) - H_tempered(S)) / np.abs(H_exact(S))
    return float(np.max(e))


def warburg_phase_deg(omega):
    """Phase of H(i omega) in degrees (should -> -45 as omega tau_d -> inf)."""
    return float(np.degrees(np.angle(H_exact(1j * float(omega)))))


def run():
    td = 1.0                                    # kappa = Vbar = rho_c = 1
    out = {
        "identity": {
            "tempered_max_rel_err": tempered_identity_error(),
            "W_equals_classic_max_rel_err": float(max(
                abs(coeff_W(k, V, th, rc) - coeff_W_classic(k, V, th, rc))
                / coeff_W(k, V, th, rc)
                for k, V, th, rc in [(1, 1, -1, 1), (2.3, 0.7, -4.2, 3.1),
                                     (0.11, 5.0, -0.03, 917 * 2100.0)])),
            "statement": "H(s) = A/s + W sqrt(s + 1/(4 tau_d))/s exactly: the "
                         "B.2 kernel is a tempered half-derivative "
                         "(tempering rate = 1/(4 tau_d)); W matches the "
                         "textbook half-space Caputo coefficient on the "
                         "advected base gradient",
        },
        "warburg": {
            "phase_deg_at_wtd": {str(w): warburg_phase_deg(w)
                                 for w in (1.0, 10.0, 100.0, 1000.0)},
            "rel_err_at_std": {str(x): float(abs(
                (H_exact(complex(x)) - H_warburg(complex(x)))
                / H_exact(complex(x))))
                for x in (1.0, 10.0, 100.0, 1000.0)},
        },
        "window_law": {
            "statement": "tau_f/tau_d = Ste^{-2}",
            "examples": {},
        },
    }
    for ste in (5.0, 1.0, 0.1, 6.3e-3):
        rhoL = 1.0 / ste                        # nondim: Ste = |theta_far|/rhoL
        out["window_law"]["examples"][f"Ste={ste:g}"] = {
            "tau_f_over_tau_d": tau_frac(rhoL=rhoL) / td,
            "vinf_over_v0": 1.0 / (1.0 + ste),
        }

    # closed loop: Ste = 5 (window open) --------------------------------------
    ste = 5.0
    rhoL = 1.0 / ste
    tf = tau_frac(rhoL=rhoL)
    t_end = 0.25 * td                            # >= 6 tau_f, << tau_d region
    tg, v_full = volterra_step(t_end, n=4000, rhoL=rhoL)
    _, v_wb = volterra_step(t_end, n=4000, rhoL=rhoL, kernel="warburg")
    v_ml = ml_step_response(tg, rhoL=rhoL)
    sel = tg <= 5.0 * tf
    # convergence to the fractional limit: the Warburg corrections enter at
    # s tau_d ~ Ste^2, i.e. relative size (s tau_d)^{-1/2} ~ 1/Ste, so the
    # full-kernel deviation from E_{1/2} inside the window must fall ~ 1/Ste
    conv = {}
    for ste_k in (5.0, 10.0, 20.0):
        rl = 1.0 / ste_k
        tfk = tau_frac(rhoL=rl)
        tgk, vfk = volterra_step(5.0 * tfk, n=3000, rhoL=rl)
        vmk = ml_step_response(tgk, rhoL=rl)
        conv[f"Ste={ste_k:g}"] = float(np.max(np.abs(vfk - vmk) / vmk))
    out["closed_loop_Ste5"] = {
        "tau_f_over_tau_d": tf / td, "t_end_over_tau_d": t_end / td,
        "solver_vs_ML_on_warburg_kernel_max_rel_err": float(np.max(
            np.abs(v_wb - v_ml) / v_ml)),
        "full_kernel_vs_ML_max_rel_err_within_5tauf": float(np.max(
            np.abs(v_full[sel] - v_ml[sel]) / v_ml[sel])),
        "convergence_to_fractional_limit": conv,
        "note": "solver validated against the exact E_{1/2} answer on the "
                "pure-Warburg kernel; the full (tempered) kernel deviates "
                "from the ML stage by O(1/Ste) inside the window and "
                "converges onto it as Ste grows (the window-law rate)",
    }

    # closed loop: glacial Ste (window shut) ----------------------------------
    ste_g = 6.3e-3
    rhoL_g = 1.0 / ste_g
    tgg, v_g = volterra_step(30.0 * td, n=3000, rhoL=rhoL_g)
    vinf = (1.0 / rhoL_g) / (1.0 + ste_g)
    out["closed_loop_glacial"] = {
        "Ste": ste_g,
        "tau_f_over_tau_d": tau_frac(rhoL=rhoL_g) / td,
        "endpoint_over_quasisteady": float(v_g[-1] / vinf),
        "ML_stage_absent_check": float(np.max(np.abs(
            v_g / (1.0 / rhoL_g) - 1.0))),   # stays within Ste of v0: no decay stage
        "note": "tau_f ~ 2.5e4 tau_d: tempering cuts the fractional stage off; "
                "response moves only O(Ste) from v0 toward the quasi-steady "
                "limit -- the fractional phase is invisible at glacial Ste",
    }
    return out, (tg, v_full, v_ml, tf, rhoL, tgg, v_g, rhoL_g, vinf)


def make_figure(out, aux, path_png):                       # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tg, v_full, v_ml, tf, rhoL, tgg, v_g, rhoL_g, vinf = aux

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    w = np.logspace(-2, 4, 400)
    Hj = H_exact(1j * w)
    ax.loglog(w, np.abs(Hj), lw=2, label="|H| exact")
    ax.loglog(w, np.abs(H_warburg(1j * w)), "--", label="Warburg W/sqrt(s)")
    ax2 = ax.twinx()
    ax2.semilogx(w, np.degrees(np.angle(Hj)), color="tab:red", lw=1,
                 label="phase")
    ax2.axhline(-45, color="tab:red", ls=":", lw=1)
    ax2.set_ylabel("phase [deg]", color="tab:red")
    ax.set_xlabel(r"$\omega\,\tau_d$"); ax.set_ylabel("|H|")
    ax.set_title("tempered Warburg: -45 deg plateau above 1/tau_d")
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[1]
    ax.plot(tg / tf, v_full * rhoL, lw=2, label="full kernel (Volterra)")
    ax.plot(tg / tf, v_ml * rhoL, "--", lw=2,
            label=r"$E_{1/2}$ = erfcx (exact, Warburg)")
    ax.set_xlabel(r"$t/\tau_f$"); ax.set_ylabel(r"$v/v_0$")
    ax.set_title("Ste = 5: Mittag-Leffler relaxation, window open")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.semilogx(tgg, v_g * rhoL_g, lw=2, label="full kernel")
    ax.axhline(vinf * rhoL_g, color="k", ls="--", lw=1,
               label=r"quasi-steady $1/(1+\mathrm{Ste})$")
    ax.set_ylim(0.99 * vinf * rhoL_g, 1.001)
    ax.set_xlabel(r"$t/\tau_d$"); ax.set_ylabel(r"$v/v_0$")
    ax.set_title("glacial Ste = 6.3e-3: fractional stage invisible")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path_png, dpi=140)
    plt.close(fig)


def main():                                                # pragma: no cover
    out, aux = run()
    os.makedirs(FIGDIR, exist_ok=True)
    jpath = os.path.join(FIGDIR, "nr34_fractional_ice_kernel.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, aux, os.path.join(FIGDIR, "nr34_fractional_ice_kernel.png"))
    print(json.dumps(out, indent=1))
    print(f"wrote {jpath} and the png")


if __name__ == "__main__":                                 # pragma: no cover
    main()
