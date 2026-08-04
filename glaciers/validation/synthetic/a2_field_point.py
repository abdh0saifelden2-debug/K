r"""§A.2 — the field point, closed from published drag measurements.

The one residual `[HYP]` in the scallop -> roughness -> hydraulic-law chain
(`FUTURE_WORK.md` §A.2; `a2_z0_roughness.py`) was the roughness-length
prefactor: ``z_0 = c_z * a`` with ``c_z = alpha_s/30`` and ``alpha_s in
[0.3, 3]`` leaning on a Nikuradse sand-grain analogy, "until a real scallop
train pins it" — a field point requiring BOTH geometry (lambda, a) AND an
independent drag measurement on the same train.

That field point exists in the literature and has for fifty years:

  **Blumberg & Curl (1974, JFM 65:735-751, pp. 742-744)** measured the friction
  factor of stable, fluid-selected dissolution scallop/flute trains in a flume
  (gypsum; the same experiments that anchor ``Re* = u* L32 / nu ~ 2200``, the
  Curl criterion this repo already uses for wavelength selection) and expressed
  it as a rough-wall log-law additive constant

      u+ = (1/kappa) ln(y/L32) + B_L,   B_L = 9.4,

  with the scallop **Sauter-mean length** ``L32`` as the roughness scale.  The
  constant is standard in speleological paleo-velocity work (Curl 1974 NSS
  Bull. 36(2):1-5; Woodward & Sasowsky 2009 ScallopEx; Springer & Hall 2020,
  Int. J. Speleol. 49:2292, who stress the Sauter-mean convention).

Conversion to the repo's ``z_0`` (this module, closed form):

      (1/kappa) ln(y/z_0) = (1/kappa) ln(y/L32) + B_L
        =>  z_0 = L32 * exp(-kappa * B_L)                      [exact]
        =>  z_0/L32 = e^{-0.4*9.4} = 0.0233   (kappa = 0.40)
                     = e^{-0.41*9.4} = 0.0212  (kappa = 0.41)

  Sand-grain equivalent (Nikuradse fully-rough B_s = 8.5, z_0 = k_s/30):
      k_s = 30 z_0 = L32 * 30 e^{-kappa B_L} ~ 0.70 L32  (kappa 0.4)
  — a scalloped wall drags like sand grains ~70% of the scallop length.

  In amplitude terms (scallop depth-to-length a/L ~ 1/12 - 1/8; Curl 1966,
  Blumberg & Curl 1974 stable profile):
      c_z = z_0/a ~ 0.19 - 0.28   i.e.  alpha_s = 30 c_z ~ 5.6 - 8.4,

  **2-3x ABOVE the previously assumed alpha_s in [0.3, 3] band**: scalloped
  surfaces are substantially rougher per unit amplitude than the sand-grain
  analogy suggested.  The §A.2 log-law buffering still compresses the impact:
  at the §A.3 anchor (H = 2.4 m, lambda = 7.9 cm) the drag coefficient moves
  from the assumed band C_d in [1.6e-3, 2.6e-3] to 3.3e-3 — a <= 2x shift for
  a ~20x prefactor correction, exactly the buffering §A.2 derived.

Honest scope / caveats carried:
- ``B_L`` is back-calculated with the **Sauter mean** L32 (Springer & Hall
  2020): using an arithmetic-mean length would change the constant.  The repo's
  fluid-selected lambda is the stable-train spacing — the L32 of a developed
  train — so the convention matches; flagged, not hidden.
- Goodchild & Ford (1971) report different constants for natural cave scallops
  (their data mixes developing/relict trains); Blumberg & Curl's
  flume-controlled, velocity-measured value is adopted as primary, and the
  spread is carried as the systematic (B_L 9.4 +/- ~1 moves z_0/L by e^{+/-0.4}
  ~ 1.5x — still inside the buffered C_d shift above).
- Gypsum flume at Re* ~ 2200: the same Reynolds anchor the repo's wavelength
  law already leans on (consistent regime, no extrapolation).
- This closes the *calibration*; it is still not an in-situ subglacial
  measurement (none exists) — tag moves [HYP] -> [LIT-calibrated].

Run:
    python glaciers/validation/synthetic/a2_field_point.py
      -> validation/reports/a2_field_point.{json,png}
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

KAPPA = 0.40                    # von Karman (Curl's 2.5 = 1/0.40 convention)
B_L_SCALLOP = 9.4               # Blumberg & Curl 1974, pp. 742-744 (Sauter L32)
B_S_SAND = 8.5                  # Nikuradse fully-rough additive constant
RE_STAR_CURL = 2200.0           # Curl criterion (already used in-repo)


def z0_over_L(B_L=B_L_SCALLOP, kappa=KAPPA):
    r"""Exact log-law identity: ``z_0 = L e^{-kappa B_L}``."""
    return float(np.exp(-kappa * B_L))


def ks_over_L(B_L=B_L_SCALLOP, kappa=KAPPA):
    r"""Sand-grain equivalent ``k_s = 30 z_0`` (Nikuradse z_0 = k_s/30)."""
    return float(30.0 * np.exp(-kappa * B_L))


def c_z(a_over_L, B_L=B_L_SCALLOP, kappa=KAPPA):
    r"""The §A.2 prefactor ``c_z = z_0/a`` for a given depth-to-length ratio."""
    return float(z0_over_L(B_L, kappa) / a_over_L)


def drag_coefficient(H, z0, kappa=KAPPA):
    r"""Log-law drag coefficient ``C_d = [kappa/ln(H/z_0)]^2`` (§A.2)."""
    return float((kappa / np.log(H / z0)) ** 2)


def run(H_anchor=2.4, lam_anchor=0.079, a_over_L=(1.0 / 12.0, 0.10, 1.0 / 8.0),
        assumed_alpha_band=(0.3, 3.0)):
    z0L_40 = z0_over_L(kappa=0.40)
    z0L_41 = z0_over_L(kappa=0.41)
    a_mid = 0.10
    out = {
        "field_point": {
            "source": "Blumberg & Curl (1974) JFM 65:735-751, friction factor of "
                       "stable dissolution scallop trains, pp. 742-744",
            "constant": {"B_L": B_L_SCALLOP, "roughness_scale": "Sauter mean L32",
                         "Re_star": RE_STAR_CURL},
            "corroborating_usage": [
                "Curl 1974, NSS Bulletin 36(2):1-5 (paleo-velocity law)",
                "Woodward & Sasowsky 2009, Acta Carsologica 38 (ScallopEx)",
                "Springer & Hall 2020, Int. J. Speleology 49 (Sauter-mean caveat)",
                "Roberts 1983, OUCC Proc. 11 (B_L=9.4 'for scalloped surfaces')",
            ],
            "dissent": "Goodchild & Ford 1971 (natural-cave constants differ; "
                       "flume-controlled B&C adopted as primary)",
        },
        "conversion": {
            "z0_over_L32_kappa0.40": z0L_40,
            "z0_over_L32_kappa0.41": z0L_41,
            "ks_over_L32": ks_over_L(),
            "identity": "z_0 = L32 * exp(-kappa*B_L); k_s = 30*z_0",
        },
        "amplitude_form": {
            "a_over_L_range": list(a_over_L),
            "c_z_range": [c_z(r) for r in a_over_L],
            "alpha_s_range": [30.0 * c_z(r) for r in a_over_L],
            "assumed_alpha_band": list(assumed_alpha_band),
            "exceeds_assumed_band": bool(
                min(30.0 * c_z(r) for r in a_over_L) > assumed_alpha_band[1]),
        },
        "anchor_update": {},
        "b_l_systematic": {
            "B_L_pm1_z0_factor": float(np.exp(KAPPA * 1.0)),
            "note": "B_L 9.4 +/- 1 moves z_0 by e^{kappa} ~ 1.5x",
        },
    }
    # §A.3 anchor: lambda = 7.9 cm scallops in an H = 2.4 m conduit
    z0_new = z0L_40 * lam_anchor
    a_anchor = a_mid * lam_anchor
    z0_band_old = [assumed_alpha_band[0] / 30.0 * a_anchor,
                   assumed_alpha_band[1] / 30.0 * a_anchor]
    cd_old = [drag_coefficient(H_anchor, z) for z in z0_band_old]
    cd_new = drag_coefficient(H_anchor, z0_new)
    out["anchor_update"] = {
        "H_m": H_anchor, "lambda_m": lam_anchor, "a_m": a_anchor,
        "z0_old_band_m": z0_band_old, "z0_new_m": z0_new,
        "z0_shift_factor_vs_band_top": float(z0_new / z0_band_old[1]),
        "C_d_old_band": cd_old, "C_d_new": cd_new,
        "C_d_shift_factor_vs_band": [float(cd_new / c) for c in cd_old],
        "buffering_note": "~20x prefactor correction -> <=2x C_d shift "
                          "(the §A.2 log-law compression, now used in anger)",
    }
    return out


def make_figure(out, path_png):                              # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    a = ax[0]
    r = np.linspace(1 / 15, 1 / 6, 100)
    a.plot(r, [30 * c_z(x) for x in r], label="alpha_s = 30 z0/a (B_L=9.4)")
    band = out["amplitude_form"]["assumed_alpha_band"]
    a.axhspan(band[0], band[1], alpha=0.2, color="C1",
              label="previously assumed band")
    for x in out["amplitude_form"]["a_over_L_range"]:
        a.axvline(x, color="k", lw=0.5, ls=":")
    a.set_xlabel("scallop depth-to-length a/L")
    a.set_ylabel("alpha_s")
    a.set_title("§A.2 field point: measured drag exceeds the\nassumed sand-grain band 2-3x")
    a.legend(fontsize=8)
    b = ax[1]
    au = out["anchor_update"]
    b.bar([0, 1, 2], [au["C_d_old_band"][0], au["C_d_old_band"][1], au["C_d_new"]],
          tick_label=["old band lo", "old band hi", "B&C 1974"])
    b.set_ylabel("C_d at the §A.3 anchor")
    b.set_title("log-law buffering: 20x prefactor -> <=2x C_d")
    fig.tight_layout()
    fig.savefig(path_png, dpi=140)
    plt.close(fig)


def main():                                                  # pragma: no cover
    here = os.path.dirname(os.path.abspath(__file__))
    rep = os.path.join(here, "..", "reports")
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=os.path.join(rep, "a2_field_point.json"))
    ap.add_argument("--png-out", default=os.path.join(rep, "a2_field_point.png"))
    a = ap.parse_args()
    out = run()
    with open(a.json_out, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, a.png_out)
    print(json.dumps(out["conversion"], indent=1))
    print(json.dumps(out["amplitude_form"], indent=1))
    print(json.dumps(out["anchor_update"], indent=1))
    print("wrote", a.json_out, "and", a.png_out)


if __name__ == "__main__":
    main()
