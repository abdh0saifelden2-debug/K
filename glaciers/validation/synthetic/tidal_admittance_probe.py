r"""§I.5 — Tidal velocity admittance as a THIRD field probe of the s_N(N) sliding law,
its harmonic-generation fingerprint, and a tides-only inversion for proximity to
ungrounding.

Why this exists (new vs sn_master_curve.py / the two existing probes)
--------------------------------------------------------------------
The repo measures the regularized-Coulomb (RC) sliding sensitivity
``s_N = d ln u_b/d ln N`` with two field probes: lake-drainage steps
(``lake_lag_trunk.py``) and ocean thermal-forcing gating
(``efp_gate_direct_n.py``). Ocean **tides** give a *third*, independent, and
**continuous high-cadence** probe: a tide raises/lowers the sea level at the
grounding zone, modulating the basal effective pressure ``N`` and therefore the
basal speed through the *same* ``s_N(N)``.

Mainstream context. Gudmundsson (2006, Nature) found Rutford Ice Stream's surface
speed varies by ~20 % at the fortnightly (MSf, 14.76 d) period; Gudmundsson
(2007, 2011) showed this requires a *non-linear* sliding law (the M2-S2 tidal beat
rectified into MSf), with a Weertman exponent close to 3, and that a *linear* law
produces no MSf signal. Minchew et al. (2017) found the MSf flow amplitude
*increases toward the grounding line*. (Other mechanisms — subglacial hydrology,
grounding-line migration, margin widening — can also produce MSf; see Rosier et al.
2014/2015, Robel et al. 2017. This module is the **sliding-law reading**.)

What this module adds
---------------------
Casting the tidal response through the RC ``s_N(N)`` (not a bare Weertman law)
yields three concrete, falsifiable results:

1. **Tidal admittance = s_N (third probe).** The fundamental velocity admittance
   ``A1 = |d ln u_b| / |d ln N|`` at the tidal frequency *is* ``|s_N(N)|`` — so it
   steepens toward flotation exactly like the drainage and ocean probes.

2. **Harmonic-generation fingerprint.** Because ``s_N(N)`` is *curved*, a sinusoidal
   tidal ``N`` forcing produces velocity **harmonics**. The 2f/1f ratio is, to
   leading order, ``(eps/4)|s_N'/s_N - 1|`` with ``s_N'/s_N = -mR/(1-R)``
   (``R=(N_c/N)^m``), which **diverges as N -> N_c**. So harmonic strength is a
   curvature meter that grows toward ungrounding — a sliding-law explanation for the
   observed toward-GL increase of the (nonlinear) MSf signal.

3. **Tides-only inversion (a new method of measurement).** The fundamental
   admittance ``A1=|s_N|`` and the 2f/1f ratio, with the *known* tidal amplitude
   ``eps``, over-determine ``(m, R)``: one recovers the Weertman exponent **and the
   dimensionless proximity to flotation** ``R=(N_c/N)^m in [0,1]`` (``R->1`` at the
   ungrounding fold) **from surface velocity alone** — no basal-pressure measurement.
   This directly answers Joughin et al. (2019)'s "no reliable knowledge of basal
   water pressure": the tide self-calibrates the bed's position on the s_N curve.

Together with ``sn_master_curve.py`` this turns the ungrounding early-warning into an
*operational, continuous* signal: track ``R(t) -> 1`` (rising admittance + rising
harmonics) at any tidally-forced ice stream.

Honest scope: quasi-static tidal limit (tidal period << secular N-decline), single
sliding mechanism (the elastic-stress-transmission + RC-sliding reading of
Gudmundsson 2007/2011), and the field test needs high-cadence GPS/InSAR velocity
decomposed by ``N`` (the project's standing co-located-GPS gap). No GPU, no download.
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


def u_b(N, m=M_EXP, N_c=N_C, u0=U0):
    N = np.asarray(N, float)
    R = (N_c / N) ** m
    return u0 * R / (1.0 - R)


def s_N_signed(N, m=M_EXP, N_c=N_C):
    R = (N_c / N) ** m
    return -m / (1.0 - R)


def s_N_curvature_term(N, m=M_EXP, N_c=N_C):
    """s_N'/s_N = -mR/(1-R) (d/dlnN of ln s_N), which sets the harmonic ratio."""
    R = (N_c / N) ** m
    return -m * R / (1.0 - R)


def analytic_2f_1f(N, eps, m=M_EXP, N_c=N_C):
    """Leading-order second-harmonic ratio (eps/4)|s_N'/s_N - 1|."""
    return eps / 4.0 * np.abs(s_N_curvature_term(N, m, N_c) - 1.0)


# --------------------------------------------------------------------------- #
# forward: harmonic content of the tidal velocity response
# --------------------------------------------------------------------------- #
def _harmonic_fit(t, y, w, nharm=3):
    cols = [np.ones_like(t)]
    for k in range(1, nharm + 1):
        cols += [np.sin(k * w * t), np.cos(k * w * t)]
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return [float(np.hypot(c[1 + 2 * (k - 1)], c[2 + 2 * (k - 1)])) for k in range(1, nharm + 1)]


def harmonic_response(N0, eps, m=M_EXP, N_c=N_C, u0=U0, ncyc=40.0, npts=40000):
    """Drive N(t)=N0(1+eps sin wt) through u_b(N); return the fundamental admittance
    A1=|d ln u/d ln N| and the harmonic ratios A2/A1, A3/A1 (least-squares fit)."""
    w = 2.0 * np.pi
    t = np.linspace(0.0, ncyc, npts)
    N = N0 * (1.0 + eps * np.sin(w * t))
    lu = np.log(u_b(N, m, N_c, u0)); lu = lu - lu.mean()
    lN = np.log(N); lN = lN - lN.mean()
    Au = _harmonic_fit(t, lu, w)
    AN = _harmonic_fit(t, lN, w)
    A1 = Au[0] / AN[0]
    return dict(N0_MPa=N0 / 1e6, eps=eps, admittance_fundamental=A1,
                ratio_2f_1f=Au[1] / Au[0], ratio_3f_1f=Au[2] / Au[0],
                analytic_2f_1f=float(analytic_2f_1f(np.array(N0), eps, m, N_c)),
                s_N_true=float(abs(s_N_signed(np.array(N0), m, N_c))))


def verify_probe(eps=0.05):
    """Sweep N: fundamental admittance -> |s_N| (small-eps limit), measured 2f/1f ==
    analytic curvature law, and both grow toward flotation."""
    Ns = np.array([2.0e6, 5.0e5, 2.0e5, 1.2e5, 9.0e4, 7.5e4, 6.8e4])
    rows = [harmonic_response(N0, eps) for N0 in Ns]
    # admittance -> |s_N| is exact only as eps->0; check the small-amplitude limit
    lin = [harmonic_response(N0, 0.01) for N0 in Ns]
    adm_err = max(abs(r["admittance_fundamental"] - r["s_N_true"]) / r["s_N_true"] for r in lin)
    h_err = max(abs(r["ratio_2f_1f"] - r["analytic_2f_1f"]) /
                r["analytic_2f_1f"] for r in rows)
    return dict(max_admittance_vs_sN_relerr_smalleps=adm_err,
                max_harmonic_vs_analytic_relerr=h_err,
                ratio_2f_1f_wellgrounded=rows[0]["ratio_2f_1f"],
                ratio_2f_1f_near_flotation=rows[-1]["ratio_2f_1f"],
                grows_toward_flotation=bool(rows[-1]["ratio_2f_1f"] > 3 * rows[0]["ratio_2f_1f"]),
                sweep=rows)


# --------------------------------------------------------------------------- #
# inverse: tides-only recovery of (m, R = (N_c/N)^m)  [proximity to ungrounding]
# --------------------------------------------------------------------------- #
def tides_only_invert(admittance, ratio_2f_1f, eps):
    """From the fundamental admittance |s_N|, the 2f/1f ratio, and known tidal eps,
    recover R=(N_c/N)^m (flotation proximity, ->1 at ungrounding) and the exponent m."""
    sN = abs(admittance)
    R = (4.0 * ratio_2f_1f / eps - 1.0) / sN
    m = sN * (1.0 - R)
    return dict(R_recovered=float(R), m_recovered=float(m),
                flotation_proximity=float(np.clip(R, 0.0, 1.0)))


def inversion_test(eps=0.05, m_true=M_EXP, N_c=N_C):
    out = []
    for N0 in (3.0e5, 1.5e5, 1.0e5, 8.0e4, 7.0e4):
        r = harmonic_response(N0, eps, m_true, N_c)
        inv = tides_only_invert(r["admittance_fundamental"], r["ratio_2f_1f"], eps)
        R_true = float((N_c / N0) ** m_true)
        out.append(dict(N0_MPa=N0 / 1e6, R_true=R_true, R_rec=inv["R_recovered"],
                        m_rec=inv["m_recovered"],
                        R_relerr=abs(inv["R_recovered"] - R_true) / R_true,
                        m_relerr=abs(inv["m_recovered"] - m_true) / m_true))
    return dict(per_point=out,
                max_R_relerr=max(o["R_relerr"] for o in out),
                max_m_relerr=max(o["m_relerr"] for o in out),
                note=("R=(N_c/N)^m is a dimensionless distance-to-ungrounding (R->1 at "
                      "the fold) recovered from velocity alone given the known tidal eps; "
                      "no basal-pressure measurement is used."))


# --------------------------------------------------------------------------- #
# operational early-warning: track R(t) -> 1 as N declines (quasi-static tide)
# --------------------------------------------------------------------------- #
def tidal_ews(eps=0.05, N_start=4.0e5, N_stop=1.15 * N_C, n=40):
    Ns = np.linspace(N_start, N_stop, n)
    adm, h2, Rrec = [], [], []
    for N0 in Ns:
        r = harmonic_response(N0, eps)
        adm.append(r["admittance_fundamental"]); h2.append(r["ratio_2f_1f"])
        Rrec.append(tides_only_invert(r["admittance_fundamental"], r["ratio_2f_1f"], eps)["R_recovered"])
    return dict(N_MPa=(Ns / 1e6).tolist(), admittance=adm, ratio_2f_1f=h2,
                R_recovered=Rrec,
                admittance_rise=float(adm[-1] / adm[0]),
                harmonic_rise=float(h2[-1] / h2[0]),
                note="continuous tidal monitoring tracks R(t)->1 (rising admittance + harmonics) toward ungrounding")


def field_connection():
    return dict(
        rutford_MSf_amplitude_pct=20,                       # Gudmundsson 2006
        rutford_exponent_from_tides=3,                      # Gudmundsson 2007 (~m=3 here)
        observed_MSf_increases_toward_GL=True,              # Minchew et al. 2017
        consistency=("Rutford is a low-N, near-flotation site (it is also this repo's "
                     "largest drainage surge, du/u=21.7%, lake_lag_trunk Rutford_1); the "
                     "observed strong, nonlinear, toward-GL MSf response matches s_N(N) "
                     "curvature steepening toward flotation. Exponent ~3 from tides = m here."),
        competing_mechanisms=("subglacial hydrology (Thompson 2014; Rosier 2015), GL "
                              "migration (Rosier 2014; Robel 2017), margin widening "
                              "(Minchew 2016) can also produce MSf; the sliding-law reading "
                              "is falsifiable by decomposing GPS admittance + harmonics vs N."),
        references="Gudmundsson 2006 Nature; 2007 JGR; 2011 TC; Minchew et al. 2017 ESSD; Rosier et al. 2014/2015")


def run():
    verify = verify_probe()
    inv = inversion_test()
    ews = tidal_ews()
    return dict(
        what="tidal velocity admittance as a third probe of s_N(N): admittance=|s_N|, "
             "harmonic ratio = s_N curvature (diverges at N_c), tides-only inversion for (m, R)",
        sliding_law="regularized-Coulomb; N_c=tau_d/C=%.3f MPa, m=%.0f" % (N_C / 1e6, M_EXP),
        probe_fundamental="A1 = |d ln u_b/d ln N| at the tidal frequency = |s_N(N)|",
        harmonic_fingerprint="A2/A1 ~ (eps/4)|s_N'/s_N - 1|, s_N'/s_N=-mR/(1-R) -> diverges at N_c",
        verification=verify,
        inversion=inv,
        early_warning=ews,
        field_connection=field_connection(),
        verdict=(
            f"Tidal admittance reproduces |s_N| to {verify['max_admittance_vs_sN_relerr_smalleps']:.1e} (small-eps) and the "
            f"2f/1f harmonic ratio matches the analytic curvature law to "
            f"{verify['max_harmonic_vs_analytic_relerr']:.1e}, rising from "
            f"{verify['ratio_2f_1f_wellgrounded']:.3f} (well-grounded) to "
            f"{verify['ratio_2f_1f_near_flotation']:.3f} near flotation. Tides-only inversion "
            f"recovers the flotation-proximity R to {100*inv['max_R_relerr']:.1f}% and m to "
            f"{100*inv['max_m_relerr']:.1f}% from velocity alone. So ocean tides are a continuous, "
            "self-calibrating third probe of the same sliding law and an operational ungrounding "
            "early-warning (R(t)->1)."),
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))
    sw = res["verification"]["sweep"]
    N = np.array([r["N0_MPa"] for r in sw])
    adm = np.array([r["admittance_fundamental"] for r in sw])
    sN = np.array([r["s_N_true"] for r in sw])
    h2 = np.array([r["ratio_2f_1f"] for r in sw])
    h3 = np.array([r["ratio_3f_1f"] for r in sw])
    # (a) admittance == |s_N|
    ax[0, 0].plot(N, sN, "k-", lw=2, label=r"$|s_N(N)|$ (closed form)")
    ax[0, 0].scatter(N, adm, c="#1f77b4", s=45, zorder=5, label="tidal fundamental admittance")
    ax[0, 0].axvline(N_C / 1e6, color="k", ls="--", label=f"$N_c$={N_C/1e6:.3f} MPa")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log"); ax[0, 0].invert_xaxis()
    ax[0, 0].set_xlabel("N [MPa] (-> flotation)"); ax[0, 0].set_ylabel(r"$|s_N|$")
    ax[0, 0].set_title("(a) tidal admittance IS the sliding sensitivity"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3, which="both")
    # (b) harmonic ratios rise toward flotation
    ax[0, 1].plot(N, 100 * h2, "o-", color="#d62728", label="2f/1f")
    ax[0, 1].plot(N, 100 * h3, "s--", color="#7570b3", label="3f/1f")
    ax[0, 1].axvline(N_C / 1e6, color="k", ls="--")
    ax[0, 1].set_xscale("log"); ax[0, 1].invert_xaxis()
    ax[0, 1].set_xlabel("N [MPa] (-> flotation)"); ax[0, 1].set_ylabel("harmonic ratio [%]")
    ax[0, 1].set_title("(b) harmonic generation: curvature meter diverges at N_c")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)
    # (c) tides-only inversion
    ip = res["inversion"]["per_point"]
    Rt = [p["R_true"] for p in ip]; Rr = [p["R_rec"] for p in ip]
    ax[1, 0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[1, 0].scatter(Rt, Rr, c="#2ca02c", s=55, zorder=5)
    for p in ip:
        ax[1, 0].annotate(f"{p['N0_MPa']:.2f}", (p["R_true"], p["R_rec"]),
                          fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1, 0].set_xlabel("R true = (N_c/N)^m"); ax[1, 0].set_ylabel("R recovered (tides only)")
    ax[1, 0].set_title(f"(c) tides-only proximity-to-ungrounding\n"
                       f"max R err {100*res['inversion']['max_R_relerr']:.1f}%, "
                       f"m err {100*res['inversion']['max_m_relerr']:.1f}%")
    ax[1, 0].grid(alpha=0.3)
    # (d) operational EWS
    ew = res["early_warning"]
    Nm = np.array(ew["N_MPa"])
    ax[1, 1].plot(Nm, np.array(ew["admittance"]) / ew["admittance"][0], color="#1f77b4", label="admittance (norm)")
    ax[1, 1].plot(Nm, np.array(ew["ratio_2f_1f"]) / ew["ratio_2f_1f"][0], color="#d62728", label="2f/1f (norm)")
    ax[1, 1].plot(Nm, ew["R_recovered"], color="#2ca02c", ls="--", label="R (proximity)")
    ax[1, 1].axvline(N_C / 1e6, color="k", ls="--")
    ax[1, 1].invert_xaxis()
    ax[1, 1].set_xlabel("N [MPa] (declining toward flotation)"); ax[1, 1].set_ylabel("normalized / R")
    ax[1, 1].set_title("(d) operational early-warning: R(t) -> 1"); ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)
    fig.suptitle("Tidal admittance: a continuous third probe of s_N(N) and an ungrounding early-warning",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_REPORTS, "tidal_admittance_probe.json"))
    a = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    v = res["verification"]; inv = res["inversion"]
    print("=== tidal admittance: third probe of s_N(N) ===")
    print(f"  admittance->|s_N| (small-eps) to {v['max_admittance_vs_sN_relerr_smalleps']:.1e}; "
          f"2f/1f==analytic to {v['max_harmonic_vs_analytic_relerr']:.1e}")
    print(f"  2f/1f: {v['ratio_2f_1f_wellgrounded']:.3f} (well-grounded) -> "
          f"{v['ratio_2f_1f_near_flotation']:.3f} (near flotation)")
    print(f"  tides-only inversion: R err {100*inv['max_R_relerr']:.1f}%, m err {100*inv['max_m_relerr']:.1f}%")
    print(f"  EWS: admittance x{res['early_warning']['admittance_rise']:.1f}, "
          f"harmonic x{res['early_warning']['harmonic_rise']:.1f} toward flotation")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
