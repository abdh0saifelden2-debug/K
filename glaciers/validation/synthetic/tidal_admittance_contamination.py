"""Contamination robustness for the tidal-admittance probe (P4b-R2/R3).

Referee objection (paper 4b): velocity harmonics at tidal frequencies are
contaminated by ice-shelf flexure leakage and by tidal grounding-line
migration (Gudmundsson 2007, 2011; Rosier & Gudmundsson 2020; Minchew et al.
2017), and real records carry red (autocorrelated) noise from concurrent
forcing - so the 2f/1f ratio that reads flotation proximity R=(N_c/N)^m could
be corrupted, or even faked, by non-sliding mechanisms.

This module quantifies the inversion bias from each mechanism and
demonstrates the built-in discriminant:

  A memoryless sliding law u = u_b(N(t)) driven by a cosine tide produces
  harmonics that are ALL pure cosines (a Chebyshev expansion), i.e. every
  harmonic phase sits in {0, pi} exactly - "phase closure". Lagged
  grounding-line-migration rectification injects a 2f component with an
  arbitrary phase, breaking closure. So the same harmonic fit that does the
  (m, R) inversion also flags the dangerous contaminant, and flexure leakage
  at the fundamental is bounded by upstream station placement.

Artifacts: validation/reports/tidal_admittance_contamination.{json,png}
Test:      tests/test_tidal_admittance_contamination.py
Compute:   analytic forward maps only; no GPU, no download.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tidal_admittance_probe import (  # noqa: E402
    M_EXP, N_C, U0, u_b, s_N_signed, tides_only_invert,
)

REPORTS = os.path.normpath(os.path.join(HERE, "..", "reports"))

# reference operating point: mid proximity to flotation
N0_REF = 1.0e5                      # Pa    -> R_true = (N_C/N0)^m = 0.216
EPS_REF = 0.05                      # tidal effective-pressure modulation


# --------------------------------------------------------------------------- #
# harmonic fit with phases (cosine-forcing convention)
# --------------------------------------------------------------------------- #
def harmonic_fit_phase(t, y, w, nharm=3):
    """LSQ harmonics of y: amp_k, phi_k with y ~ sum_k amp_k*cos(k*w*t - phi_k).

    With cosine forcing, a memoryless response has phi_k in {0, pi} for all k.
    """
    cols = [np.ones_like(t)]
    for k in range(1, nharm + 1):
        cols += [np.cos(k * w * t), np.sin(k * w * t)]
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    out = []
    for k in range(1, nharm + 1):
        cc, cs = float(c[1 + 2 * (k - 1)]), float(c[2 + 2 * (k - 1)])
        out.append((float(np.hypot(cc, cs)), float(np.arctan2(cs, cc))))
    return out


def closure_residual(phi):
    """Distance (rad) of a harmonic phase from the memoryless set {0, +-pi}."""
    return float(min(abs(phi), abs(np.pi - abs(phi))))


# --------------------------------------------------------------------------- #
# contaminated forward model
# --------------------------------------------------------------------------- #
def contaminated_response(N0=N0_REF, eps=EPS_REF, m=M_EXP, N_c=N_C, u0=U0,
                          flex_amp=0.0, flex_phase=np.pi / 3.0,
                          gl_amp=0.0, gl_lag_cycles=0.15,
                          noise_sigma=0.0, noise_tau_cycles=1.0, seed=0,
                          ncyc=40.0, npts=40000):
    """Tidal response with flexure leakage, lagged GL-migration rectification,
    and red observational/forcing noise; returns the tides-only inversion and
    the phase-closure diagnostic.

    - flexure: multiplicative at the tidal frequency, u *= 1 + a*cos(wt+phi_f)
      (vertical flexure projecting into the velocity record near the GL);
    - GL migration: lagged half-wave rectification, u *= 1 + g*max(cos(w(t-lag)),0)
      (the stream sees a shorter grounded reach at high tide, with a migration
      lag) - injects even harmonics with a NON-closed phase;
    - red noise: AR(1) in ln u with std sigma and correlation time tau cycles.
    """
    w = 2.0 * np.pi
    t = np.linspace(0.0, ncyc, npts)
    N = N0 * (1.0 + eps * np.cos(w * t))
    u = u_b(N, m, N_c, u0)
    if flex_amp:
        u = u * (1.0 + flex_amp * np.cos(w * t + flex_phase))
    if gl_amp:
        u = u * (1.0 + gl_amp * np.maximum(np.cos(w * (t - gl_lag_cycles)), 0.0))
    lu = np.log(u)
    if noise_sigma:
        rng = np.random.default_rng(seed)
        dt = t[1] - t[0]
        rho = float(np.exp(-dt / noise_tau_cycles))
        x = np.empty(npts)
        x[0] = rng.standard_normal()
        innov = rng.standard_normal(npts) * np.sqrt(1.0 - rho * rho)
        for i in range(1, npts):
            x[i] = rho * x[i - 1] + innov[i]
        lu = lu + noise_sigma * x
    lu = lu - lu.mean()
    lN = np.log(N) - np.log(N).mean()
    Hu = harmonic_fit_phase(t, lu, w)
    HN = harmonic_fit_phase(t, lN, w)
    A1 = Hu[0][0] / HN[0][0]
    r21 = Hu[1][0] / Hu[0][0]
    inv = tides_only_invert(A1, r21, eps)
    return dict(A1=float(A1), ratio_2f_1f=float(r21),
                phi1=Hu[0][1], phi2=Hu[1][1],
                closure_residual_1f=closure_residual(Hu[0][1]),
                closure_residual_2f=closure_residual(Hu[1][1]),
                R_rec=inv["R_recovered"], m_rec=inv["m_recovered"])


def _truth(N0=N0_REF, m=M_EXP, N_c=N_C):
    return dict(R_true=float((N_c / N0) ** m), m_true=float(m),
                s_N_true=float(abs(s_N_signed(np.array(N0), m, N_c))))


# --------------------------------------------------------------------------- #
# studies
# --------------------------------------------------------------------------- #
def flexure_sweep(amps=(0.0, 0.005, 0.01, 0.02)):
    """Flexure leakage at the fundamental: biases A1 (and, via the cross-term,
    weakly 2f) but keeps phase closure - indistinguishable in phase, bounded
    by placement upstream of the flexure zone."""
    tr = _truth()
    rows = []
    for a in amps:
        r = contaminated_response(flex_amp=a)
        rows.append(dict(flex_amp=a, **{k: r[k] for k in
                                        ("A1", "ratio_2f_1f", "R_rec", "m_rec",
                                         "closure_residual_1f",
                                         "closure_residual_2f")},
                         R_bias_rel=abs(r["R_rec"] - tr["R_true"]) / tr["R_true"],
                         m_bias_rel=abs(r["m_rec"] - tr["m_true"]) / tr["m_true"]))
    return dict(truth=tr, rows=rows)


def gl_migration_sweep(amps=(0.0, 0.005, 0.01, 0.02), lag_cycles=0.15):
    """Lagged GL-migration rectification: injects 2f directly (fakes proximity)
    BUT breaks phase closure - the discriminant the paper states."""
    tr = _truth()
    rows = []
    for g in amps:
        r = contaminated_response(gl_amp=g, gl_lag_cycles=lag_cycles)
        rows.append(dict(gl_amp=g, lag_cycles=lag_cycles,
                         **{k: r[k] for k in
                            ("A1", "ratio_2f_1f", "R_rec", "m_rec",
                             "closure_residual_1f", "closure_residual_2f")},
                         R_bias_rel=abs(r["R_rec"] - tr["R_true"]) / tr["R_true"],
                         m_bias_rel=abs(r["m_rec"] - tr["m_true"]) / tr["m_true"]))
    return dict(truth=tr, rows=rows)


def red_noise_mc(sigma=0.01, tau_cycles=1.0, n_mc=200):
    """Red-noise (concurrent forcing / observational) scatter of the inversion
    over a finite 40-cycle record."""
    tr = _truth()
    R, m = [], []
    for k in range(n_mc):
        r = contaminated_response(noise_sigma=sigma, noise_tau_cycles=tau_cycles,
                                  seed=k)
        R.append(r["R_rec"]); m.append(r["m_rec"])
    R, m = np.asarray(R), np.asarray(m)
    return dict(truth=tr, sigma=sigma, tau_cycles=tau_cycles, n_mc=n_mc,
                R_median=float(np.median(R)),
                R_ci68=(float(np.percentile(R, 16)), float(np.percentile(R, 84))),
                m_median=float(np.median(m)),
                m_ci68=(float(np.percentile(m, 16)), float(np.percentile(m, 84))))


def run():
    clean = contaminated_response()
    tr = _truth()
    flex = flexure_sweep()
    gl = gl_migration_sweep()
    mc1 = red_noise_mc(sigma=0.01)
    mc3 = red_noise_mc(sigma=0.03)
    gl_mid = gl["rows"][2]           # g = 0.01
    out = dict(
        what=("P4b-R2/R3: tidal-admittance inversion robustness to flexure "
              "leakage, lagged GL-migration rectification, and red noise; "
              "phase-closure discriminant for the dangerous (2f-faking) "
              "contaminant"),
        operating_point=dict(N0_Pa=N0_REF, eps=EPS_REF, **tr),
        phase_closure_principle=(
            "memoryless u=u_b(N(t)) under cosine forcing is a Chebyshev "
            "expansion: every harmonic is a pure cosine, phase in {0, pi}; "
            "clean closure residual = %.1e rad" % clean["closure_residual_2f"]),
        clean=clean,
        flexure=flex,
        gl_migration=gl,
        red_noise=dict(sigma_1pct=mc1, sigma_3pct=mc3),
        verdict=dict(
            clean_phase_closed=bool(clean["closure_residual_2f"] < 0.02
                                    and clean["closure_residual_1f"] < 0.02),
            flexure_2f_phase_closed=bool(all(r["closure_residual_2f"] < 0.05
                                             for r in flex["rows"])),
            flexure_flagged_at_1f_when_lagged=bool(
                flex["rows"][-1]["closure_residual_1f"] > 0.05),
            flexure_R_bias_at_2pct=flex["rows"][-1]["R_bias_rel"],
            gl_2f_fakes_proximity=bool(gl["rows"][-1]["R_bias_rel"] > 0.10),
            gl_flagged_by_phase=bool(gl_mid["closure_residual_2f"] > 0.20),
            red_noise_R_ci68_halfwidth_1pct=float(
                0.5 * (mc1["R_ci68"][1] - mc1["R_ci68"][0])),
            mitigation=("flexure: lagged (viscoelastic) flexure is flagged at "
                        "the fundamental (|phi_1 - {0,pi}| > 0.05 rad); exactly "
                        "in-phase flexure is phase-invisible and must be capped "
                        "by station placement upstream of the flexure zone; "
                        "GL migration: breaks 2f phase closure -> screen "
                        "records with |phi_2 - {0,pi}| > 0.2 rad before "
                        "inverting; red noise: 68% scatter quantified, shrink "
                        "with record length"),
        ),
    )
    return out


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    fx = res["flexure"]["rows"]; gm = res["gl_migration"]["rows"]
    tr = res["operating_point"]

    axs[0].plot([r["flex_amp"] for r in fx], [r["R_rec"] for r in fx], "o-",
                label="flexure (phase-closed)")
    axs[0].plot([r["gl_amp"] for r in gm], [r["R_rec"] for r in gm], "s-",
                label="GL migration (lagged)")
    axs[0].axhline(tr["R_true"], color="k", ls="--", lw=0.8, label="R true")
    axs[0].set_xlabel("contamination amplitude")
    axs[0].set_ylabel("recovered R")
    axs[0].legend(fontsize=7)
    axs[0].set_title("inversion bias")

    axs[1].plot([r["flex_amp"] for r in fx],
                [r["closure_residual_2f"] for r in fx], "o-", label="flexure (2f)")
    axs[1].plot([r["flex_amp"] for r in fx],
                [r["closure_residual_1f"] for r in fx], "o--", label="flexure (1f)")
    axs[1].plot([r["gl_amp"] for r in gm],
                [r["closure_residual_2f"] for r in gm], "s-", label="GL migration (2f)")
    axs[1].axhline(0.2, color="r", ls=":", lw=0.8, label="screen threshold")
    axs[1].set_xlabel("contamination amplitude")
    axs[1].set_ylabel(r"$|\phi_2 - \{0,\pi\}|$ (rad)")
    axs[1].legend(fontsize=7)
    axs[1].set_title("phase-closure discriminant")

    mc = res["red_noise"]["sigma_1pct"]
    axs[2].errorbar([1.0], [mc["R_median"]],
                    yerr=[[mc["R_median"] - mc["R_ci68"][0]],
                          [mc["R_ci68"][1] - mc["R_median"]]], fmt="ko")
    mc3 = res["red_noise"]["sigma_3pct"]
    axs[2].errorbar([3.0], [mc3["R_median"]],
                    yerr=[[mc3["R_median"] - mc3["R_ci68"][0]],
                          [mc3["R_ci68"][1] - mc3["R_median"]]], fmt="ko")
    axs[2].axhline(tr["R_true"], color="k", ls="--", lw=0.8)
    axs[2].set_xlabel("red-noise sigma (% of ln u)")
    axs[2].set_ylabel("recovered R (68%)")
    axs[2].set_title("red-noise scatter (40 cycles)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    res = run()
    os.makedirs(REPORTS, exist_ok=True)
    jpath = os.path.join(REPORTS, "tidal_admittance_contamination.json")
    with open(jpath, "w") as fh:
        json.dump(res, fh, indent=2)
    make_figure(res, os.path.join(REPORTS, "tidal_admittance_contamination.png"))
    v = res["verdict"]
    print("tidal admittance contamination study (P4b-R2/R3)")
    print(f"  clean phase closure: residual={res['clean']['closure_residual_2f']:.2e} rad "
          f"(closed={v['clean_phase_closed']})")
    print(f"  flexure at 2%: R bias {100*v['flexure_R_bias_at_2pct']:.1f}%, "
          f"2f closed={v['flexure_2f_phase_closed']}, "
          f"lagged flexure flagged at 1f={v['flexure_flagged_at_1f_when_lagged']}")
    print(f"  GL migration: fakes proximity (R bias at 2% = "
          f"{100*res['gl_migration']['rows'][-1]['R_bias_rel']:.0f}%) but "
          f"flagged by phase={v['gl_flagged_by_phase']}")
    print(f"  red noise 1%: R 68% halfwidth = "
          f"{v['red_noise_R_ci68_halfwidth_1pct']:.3f} "
          f"(R true {res['operating_point']['R_true']:.3f})")
    print(f"  -> {v['mitigation']}")
    return res


if __name__ == "__main__":
    main()
