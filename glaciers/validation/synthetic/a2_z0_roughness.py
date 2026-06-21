r"""§A.2 — the scallop roughness law z_0(λ,a), MAX TRACTABLE THEORY + the one field
point: how much does the un-calibrated Nikuradse prefactor actually matter?

What this sharpens (the §A.2 [HYP, leans on LIT — Caveat D])
-----------------------------------------------------------
§A.2 closes the loop dφ/ds → Q → u* → λ (Curl) → z_0(λ,a) → C_d → dφ/ds. Two of the
three legs are settled:
  * **wavelength selection** is [VERIFIED]/[LIT]: λ = Re*·ν/u* with Re*≈2200 (Curl
    1966; Blumberg & Curl 1974), matching the solver's Re_L band;
  * the **amplitude** direction of the roughness is [MEASURED] (§G.6/figures/59): the
    mean wall-flux deficit is amplitude-INDEPENDENT over a/λ∈[0.05,0.30], so z_0 is set
    by the wavelength/separation geometry, not by how steep the bumps are.
The one genuine residual is the **wavelength (field) prefactor of z_0**, which must
lean on a Nikuradse sand-grain closure (`k_s≈α_s·a`, `z_0=k_s/30=c_z·a`, `c_z=α_s/30`)
until pinned by a real scallop train. `c_z` is uncertain by ~10× (`α_s∈[0.3,3]` for
2D wavy/bar roughness ⇒ `c_z∈[0.01,0.1]`).

The tractable question this answers
-----------------------------------
Rather than pretend a field point exists, this module asks the answerable theory
question: **given the log-law drag `C_d=[κ/ln(H/z_0)]²`, how much does the 10×
prefactor uncertainty actually propagate into `C_d` and the hydraulic gradient
`dφ/ds=ρ_w C_d u²/H`?** Because `C_d` depends only *logarithmically* on `z_0`, a 10×
roughness uncertainty should compress to a much smaller drag uncertainty — quantifying
that tells us how badly the missing field point really hurts.

Result
------
At the Curl anchor (`u*=0.05 m/s` ⇒ `λ≈7.9 cm`, `a=0.1λ≈7.9 mm`) with channel depth
`H≈2.4 m` (§A.3), `z_0∈[7.9e-5, 7.9e-4] m` across the full `c_z` band, but
`C_d∈[1.6e-3, 2.6e-3]` — a **10× roughness uncertainty buffers to ~1.7× in drag** (and
the same factor in `dφ/ds`). So the un-calibrated Nikuradse prefactor is the framework's
*least* damaging open closure: the log law structurally absorbs it, valid wherever
`H/z_0≫1` (metre channels, sub-mm roughness: `H/z_0~10³–10⁴`). The **one field point**
that would still pin `c_z` is specified below; none exists in-repo.

The one field point (flagged, not faked)
----------------------------------------
Calibrating `c_z` needs a single scallop train with BOTH its geometry `(λ, a)` AND an
independent drag measure on the SAME train — either a near-wall velocity profile giving
`z_0` directly, or a measured `u*` together with `dφ/ds`. Candidate analog datasets:
Curl (1966) / Blumberg & Curl (1974) limestone & ice scallops; cave-scallop
morphometry; subglacial-conduit roughness from dye-trace + pressure. The amplitude leg
being already settled means the field point only has to pin the *geometry* prefactor,
not an amplitude law. No GPU, no download.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

KAPPA = 0.41                  # von Karman
RE_STAR_CURL = 2200.0        # Curl (1966) scallop-selection Reynolds number
NU_WATER = 1.8e-6            # m^2/s near 0 C
RHO_W = 1000.0               # kg/m^3
G = 9.81                     # m/s^2
CZ_BAND = (0.01, 0.10)      # z_0 = c_z * a ; c_z = alpha_s/30, alpha_s in [0.3,3]
# anchor (Curl) + geometry/channel context
USTAR_ANCHOR = 0.05          # m/s (repo §G.2 Curl anchor)
A_OVER_LAM = 0.10            # representative scallop steepness
H_CHANNEL_M = 2.4            # channel/flow depth ~ §A.3 central R*
U_FLOW = 1.0                 # m/s representative conduit velocity


def lam_from_ustar(ustar, re_star=RE_STAR_CURL, nu=NU_WATER):
    """Curl-selected scallop wavelength λ = Re*·ν/u* [m]."""
    return re_star * nu / ustar


def z0_from_geom(a, c_z):
    """Roughness length z_0 = c_z·a (Nikuradse-type, geometry-set)."""
    return c_z * np.asarray(a, float)


def drag_Cd(z0, H=H_CHANNEL_M, kappa=KAPPA):
    """Log-law drag coefficient C_d = [κ/ln(H/z_0)]²."""
    return (kappa / np.log(np.asarray(H, float) / np.asarray(z0, float))) ** 2


def dphi_ds(Cd, u=U_FLOW, H=H_CHANNEL_M, rho_w=RHO_W):
    """Hydraulic-potential gradient dφ/ds = ρ_w C_d u²/H [Pa/m]."""
    return rho_w * np.asarray(Cd, float) * u ** 2 / H


def run():
    lam = lam_from_ustar(USTAR_ANCHOR)
    a = A_OVER_LAM * lam
    cz = np.array(CZ_BAND)
    z0 = z0_from_geom(a, cz)
    Cd = drag_Cd(z0)
    grad = dphi_ds(Cd)
    # buffering factor: ratio of output spread to input spread
    cz_ratio = CZ_BAND[1] / CZ_BAND[0]
    Cd_ratio = float(Cd.max() / Cd.min())
    grad_ratio = float(grad.max() / grad.min())
    H_over_z0 = (H_CHANNEL_M / z0).tolist()
    # central (geometric-mean prefactor)
    cz_c = float(np.sqrt(CZ_BAND[0] * CZ_BAND[1]))
    z0_c = float(z0_from_geom(a, cz_c)); Cd_c = float(drag_Cd(z0_c))
    return dict(
        what="max-tractable z_0(λ,a) roughness closure + log-law sensitivity + one-field-point flag",
        anchor=dict(ustar_m_s=USTAR_ANCHOR, lam_m=lam, a_m=a, a_over_lam=A_OVER_LAM,
                    H_channel_m=H_CHANNEL_M, u_flow_m_s=U_FLOW),
        cz_band=list(CZ_BAND), alpha_s_band=[CZ_BAND[0] * 30, CZ_BAND[1] * 30],
        z0_band_m=[float(z0.min()), float(z0.max())],
        H_over_z0_band=[float(min(H_over_z0)), float(max(H_over_z0))],
        Cd_band=[float(Cd.min()), float(Cd.max())], Cd_central=Cd_c,
        dphi_ds_band_Pa_m=[float(grad.min()), float(grad.max())],
        central=dict(c_z=cz_c, z0_m=z0_c, Cd=Cd_c),
        buffering=dict(input_cz_ratio=float(cz_ratio), output_Cd_ratio=Cd_ratio,
                       output_dphi_ratio=grad_ratio,
                       log_law_compression=float(cz_ratio / Cd_ratio)),
        settled_legs=dict(
            wavelength="VERIFIED/LIT: λ=Re*ν/u*, Re*≈2200 (Curl 1966)",
            amplitude="MEASURED amplitude-independent (§G.6/figures/59) -> z_0 is "
                      "geometry-set, not steepness-set"),
        one_field_point=dict(
            needed="a single scallop train with BOTH geometry (λ,a) AND an independent "
                   "drag measure (near-wall velocity profile -> z_0, or measured u* + dφ/ds) "
                   "on the same train",
            pins="the Nikuradse geometry prefactor c_z=α_s/30 (amplitude leg already settled)",
            candidates="Curl 1966 / Blumberg & Curl 1974 (limestone & ice scallops); "
                       "cave-scallop morphometry; subglacial-conduit dye-trace + pressure",
            in_repo=False),
        verdict=(
            f"MAX-TRACTABLE THEORY: with wavelength (Curl) and amplitude (§G.6 measured) "
            f"legs settled, only the Nikuradse geometry prefactor c_z∈[{CZ_BAND[0]},{CZ_BAND[1]}] "
            f"(~10× uncertain) is open. At the Curl anchor (u*=0.05 -> λ≈{100*lam:.1f} cm, "
            f"a≈{1000*a:.1f} mm, H={H_CHANNEL_M} m), z_0∈[{z0.min():.1e},{z0.max():.1e}] m gives "
            f"C_d∈[{Cd.min():.1e},{Cd.max():.1e}] — the 10× roughness uncertainty BUFFERS to "
            f"only ~{Cd_ratio:.1f}× in C_d (and dφ/ds), because the log law depends only on "
            f"ln(H/z_0) with H/z_0~10³-10⁴. So the missing field point is the framework's least "
            f"damaging open closure; the single (geometry+drag) scallop-train point that would "
            f"pin c_z is specified, and does not exist in-repo (no faked field verification)."),
        references="this repo §A.2/§G.6 (figures/59); Curl 1966; Blumberg & Curl 1974; "
                   "Nikuradse 1933; Röthlisberger 1972; Werder et al. 2013",
    )


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = res["anchor"]["a_m"]; H = res["anchor"]["H_channel_m"]
    cz = np.geomspace(*res["cz_band"], 100)
    z0 = z0_from_geom(a, cz); Cd = drag_Cd(z0, H)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    # (a) C_d vs z_0 : shallow log dependence with the c_z band shaded
    ax[0].plot(z0 * 1e3, Cd * 1e3, "-", color="#1f77b4", lw=2)
    ax[0].axvspan(res["z0_band_m"][0] * 1e3, res["z0_band_m"][1] * 1e3,
                  alpha=0.18, color="#ff7f0e", label="c_z band (10× roughness)")
    ax[0].axhspan(res["Cd_band"][0] * 1e3, res["Cd_band"][1] * 1e3,
                  alpha=0.18, color="#2ca02c", label=f"C_d band (~{res['buffering']['output_Cd_ratio']:.1f}×)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"roughness length $z_0$ [mm]")
    ax[0].set_ylabel(r"drag coefficient $C_d \times 10^{3}$")
    ax[0].set_title("(a) log-law buffers a 10× z_0 uncertainty into ~1.7× C_d")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    # (b) closed loop schematic of settled vs open legs
    ax[1].axis("off")
    txt = ("§A.2 closed loop  dφ/ds → Q → u* → λ → z_0 → C_d → dφ/ds\n\n"
           "u* → λ      : VERIFIED/LIT  (Curl Re*≈2200)\n"
           "a-direction : MEASURED      (§G.6 amplitude-independent)\n"
           "λ → z_0     : OPEN prefactor c_z=α_s/30 ∈ [0.01,0.10]\n"
           "z_0 → C_d   : LIT log law  (buffers c_z: 10× → "
           f"{res['buffering']['output_Cd_ratio']:.1f}×)\n\n"
           f"central: λ≈{100*res['anchor']['lam_m']:.1f} cm, a≈{1000*res['anchor']['a_m']:.1f} mm,\n"
           f"z_0≈{1e3*res['central']['z0_m']:.2f} mm, C_d≈{1e3*res['central']['Cd']:.2f}e-3\n\n"
           "ONE FIELD POINT (none in-repo): a scallop train with\n"
           "(λ,a) + drag (velocity profile z_0, or u*+dφ/ds)\n"
           "→ pins c_z. Candidates: Curl 1966; Blumberg & Curl 1974.")
    ax[1].text(0.02, 0.98, txt, va="top", ha="left", fontsize=9, family="monospace")
    ax[1].set_title("(b) settled vs open legs + the flagged field point")
    fig.suptitle("§A.2 scallop roughness z_0(λ,a): max tractable theory + one field point", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "reports", "a2_z0_roughness.json")))
    a = ap.parse_args()
    res = run()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    b = res["buffering"]
    print("=== §A.2 z_0(λ,a) roughness: tractable theory + one field point ===")
    print(f"  anchor: λ={100*res['anchor']['lam_m']:.1f} cm, a={1000*res['anchor']['a_m']:.1f} mm, "
          f"H={res['anchor']['H_channel_m']} m")
    print(f"  z_0 band: {res['z0_band_m'][0]:.1e}-{res['z0_band_m'][1]:.1e} m "
          f"(H/z_0 {res['H_over_z0_band'][0]:.0f}-{res['H_over_z0_band'][1]:.0f})")
    print(f"  C_d band: {res['Cd_band'][0]:.2e}-{res['Cd_band'][1]:.2e}")
    print(f"  BUFFERING: input c_z {b['input_cz_ratio']:.0f}× -> output C_d {b['output_Cd_ratio']:.2f}× "
          f"(log-law compression {b['log_law_compression']:.1f}×)")
    print(f"  one field point in-repo: {res['one_field_point']['in_repo']}")
    print(f"  json -> {a.out}")
    make_figure(res, os.path.splitext(a.out)[0] + ".png")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
