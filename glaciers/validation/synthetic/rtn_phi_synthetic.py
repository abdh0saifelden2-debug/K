r"""§V.3 synthetic calibration of the §H.1.1 φ-area inversion (§G.3 gauge RTN).

No external data; CPU only.  §H.1.1 claims the *intruded area* — the fraction of
grounded ice with ``RTN > 1`` — is a **calibrated, monotone inverse** for the
continent-effective basal-water fraction ``φ`` (the gauge RTN collapses to the
Atwood ratio ``RTN = (ρ_w/φρ_i)·(d_base/H)``, so ``RTN>1 ⇔ d_base/H > φ·ρ_i/ρ_w``
with `g` and the atmosphere cancelled).  The real driver
``validation/external/rtn_phi_calibration.py`` reports that inverse on Bedmap2 as
a steep, monotone table — but the *inversion itself* (uniqueness + finite-sample
unbiasedness) was asserted from the deterministic table, never calibrated.

This is the plant-and-recover that closes that gap.  We reuse the **real**
``build_rtn`` / ``classify`` math (so the equation under test is identical to the
field driver), plant a population of grounded cells with a ratio
``r = d_base/H`` whose tail spans the same area range as Antarctica
(``≈3.85 %`` at φ=0.70 → ``≈0.11 %`` at φ=0.98), and check:

  1. **Monotone & unique.** ``A(φ)`` is strictly decreasing over the φ-grid, so
     its inverse is single-valued (a mapped area picks out exactly one φ).
  2. **Critical-thickness identity is exact.** ``RTN>1`` reproduces
     ``H < H* = (ρ_w/φρ_i)·d_base`` cell-for-cell (zero mismatch) — the
     φ-parameterised threshold the MISI-margin claim (§H.1.1 #2) rests on.
  3. **Population inverse is faithful.** Inverting the monotone ``A(φ)`` curve at
     a planted ``φ_true`` recovers it to interpolation precision.
  4. **Finite-sample inverse is unbiased.** With only ``N`` sampled cells the area
     estimate has sampling scatter; over many seeds the recovered ``φ̂`` is
     unbiased (mean ≈ φ_true) and its spread shrinks like ``1/√N`` — so a mapped
     finite survey inverts for φ without bias.
  5. **Sensitivity has the right sign.** ``dA/dφ < 0`` everywhere (a wetter bed
     ⇒ more intrusion), matching the driver's reported ``dfrac_dφ``.

This validates the *inversion* (the estimator), not the physics: it does not
assert any particular φ for Antarctica, only that the area↦φ map is a unique,
unbiased, shape-faithful readout — the §G.3 analogue of ``rtn_synthetic`` (the
classifier), ``glmig_synthetic`` (RESULT 16) and the §G.4 lag estimator
(RESULT 17).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external.run_rtn_bedmap2 import build_rtn, classify, RHO_W, RHO_I  # noqa: E402

# ratio r = d_base/H lognormal params chosen so the RTN>1 area spans the real
# Bedmap2 table (≈3.85 % at φ=0.70 → ≈0.11 % at φ=0.98); see module docstring.
_R_MEDIAN = 0.394
_R_SIGMA = 0.26
_H_MEDIAN = 1500.0
_H_SIGMA = 0.5


def make_population(n_cells=200_000, seed=0):
    """Plant a grounded-cell population: lognormal thickness ``H`` and a ratio
    ``r = d_base/H`` whose tail crosses the ``φ·ρ_i/ρ_w`` threshold band.

    Returns ``(H, bed)`` with ``bed < 0`` (so ``d_base = -bed = r·H``), matching
    the real driver's ``d_base = max(0, -bed)`` convention.
    """
    rng = np.random.default_rng(seed)
    H = np.exp(rng.normal(np.log(_H_MEDIAN), _H_SIGMA, n_cells))
    r = np.exp(rng.normal(np.log(_R_MEDIAN), _R_SIGMA, n_cells))
    bed = -r * H
    return H, bed


def area_fraction(H, bed, phi):
    """Fraction of cells with ``RTN > 1`` at ``phi`` (uses the real driver math)."""
    return float(classify(build_rtn(H, bed, phi)).mean())


def area_curve(H, bed, phis):
    """``A(φ)`` over a φ-grid."""
    return np.array([area_fraction(H, bed, phi) for phi in phis], dtype=float)


def invert_area(phis, fr, a_obs):
    """Invert the monotone-decreasing ``A(φ)`` curve for the φ matching ``a_obs``.

    ``fr`` is strictly decreasing, so we interpolate on the reversed (increasing)
    axis.  Returns ``np.nan`` if ``a_obs`` is outside the curve's range.
    """
    lo, hi = float(fr.min()), float(fr.max())
    if a_obs < lo or a_obs > hi:
        return float("nan")
    order = np.argsort(fr)  # ascending area
    return float(np.interp(a_obs, fr[order], np.asarray(phis, float)[order]))


def critical_thickness_mismatch(H, bed, phi):
    """Cells where ``RTN>1`` disagrees with ``H < H* = (ρ_w/φρ_i)·d_base``.

    Only finite-bed cells are counted: where ``bed`` is NaN, ``build_rtn`` maps
    RTN→+inf (classify=True) while ``H*`` is NaN (``H < H*`` is False), which would
    otherwise register a spurious mismatch.  The §H.1.1 identity is only defined on
    finite beds, so NaN-bed cells are excluded rather than miscounted.
    """
    H = np.asarray(H, dtype=float)
    bed = np.asarray(bed, dtype=float)
    finite = np.isfinite(bed)
    d_base = np.maximum(0.0, -bed[finite])
    h_star = (RHO_W / (phi * RHO_I)) * d_base
    pred = classify(build_rtn(H, bed, phi))[finite]
    return int(np.sum(pred != (H[finite] < h_star)))


def flotation_thickness(bed):
    """Classical flotation thickness ``H_flot = (ρ_w/ρ_i)·d_base`` (φ=1)."""
    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    return (RHO_W / RHO_I) * d_base


def critical_thickness(bed, phi):
    """RTN=1 critical thickness ``H* = (ρ_w/φρ_i)·d_base = H_flot/φ``."""
    return flotation_thickness(bed) / phi


def misi_band_mask(H, bed, phi):
    """§H.1.1 #2 grounded-but-intrudable band: ``H_flot < H < H* = H_flot/φ``."""
    h_flot = flotation_thickness(bed)
    return (H > h_flot) & (H < h_flot / phi)


def run_misi_band(n_cells=400_000, seed=0):
    r"""§H.1.1 #2 — the hydrology-corrected MISI margin band.

    ``RTN=1 ⇔ H < H* = H_flot/φ``, so for φ<1 the RTN=1 line sits *inland* of the
    classical flotation line by a grounded-but-intrudable band
    ``H_flot < H < H*`` of relative width ``(H*−H_flot)/H_flot = (1−φ)/φ`` — i.e.
    width ∝(1−φ).  Verifies, on the planted population:

      * the per-cell relative band width equals ``(1−φ)/φ`` exactly (the "∝(1−φ)"
        claim made precise; independent of the cell);
      * the band equals ``grounded(H>H_flot) ∧ (RTN>1 at φ)`` cell-for-cell;
      * the RTN=1 line is *inland* of flotation (``H* > H_flot`` ⇒ band non-empty
        where ``d_base>0``);
      * the band-area fraction → 0 monotonically as φ→1, and is **linear in
        (1−φ)** to leading order (the ratio ``A_band/(1−φ)`` converges as φ→1).
    """
    H, bed = make_population(n_cells=n_cells, seed=seed)
    h_flot = flotation_thickness(bed)
    has_base = h_flot > 0.0

    # exact per-cell relative band width = (1-phi)/phi, independent of cell
    width_err = 0.0
    for phi in (0.80, 0.90, 0.96):
        h_star = h_flot / phi
        rel = (h_star[has_base] - h_flot[has_base]) / h_flot[has_base]
        width_err = max(width_err, float(np.max(np.abs(rel - (1.0 - phi) / phi))))

    # band == grounded-above-flotation AND RTN>1 (cell-for-cell), and inland
    mask_mismatch, inland_ok = 0, True
    for phi in (0.80, 0.90, 0.96):
        band = misi_band_mask(H, bed, phi)
        rtn1 = classify(build_rtn(H, bed, phi))
        mask_mismatch = max(mask_mismatch, int(np.sum(band != ((H > h_flot) & rtn1))))
        inland_ok = inland_ok and bool(np.all((h_flot / phi)[has_base] > h_flot[has_base]))

    # area scaling: vanish monotonically as phi->1; linear in (1-phi) at the limit
    phis = np.array([0.99, 0.98, 0.97, 0.96, 0.94, 0.90, 0.85, 0.80, 0.70])
    band_frac = np.array([misi_band_mask(H, bed, p).mean() for p in phis])
    # phis listed descending => (1-phi) ascending => band_frac must be ascending
    vanishes_monotone = bool(np.all(np.diff(band_frac) > 0.0))
    ratio = band_frac / (1.0 - phis)              # -> constant as phi->1
    limit_linear = abs(ratio[0] - ratio[1]) / ratio[1]  # |r(.99)-r(.98)|/r(.98)

    ok = bool(
        width_err <= 1e-9
        and mask_mismatch == 0
        and inland_ok
        and vanishes_monotone
        and limit_linear <= 0.25
    )
    return {
        "rel_band_width_max_abs_err": float(width_err),
        "band_vs_rtn_mismatch_cells": int(mask_mismatch),
        "rtn1_inland_of_flotation": inland_ok,
        "phis": [float(p) for p in phis],
        "band_frac": [float(x) for x in band_frac],
        "band_pct": [float(100 * x) for x in band_frac],
        "band_frac_vanishes_monotone": vanishes_monotone,
        "ratio_band_over_1mphi": [float(x) for x in ratio],
        "limit_linear_rel_gap": float(limit_linear),
        "pass": ok,
    }


def run(phi_true=0.90, n_cells=200_000, n_finite=4000, n_seeds=200, seed=0):
    """Assemble the φ-inversion calibration.  Returns a dict with a top ``pass``."""
    phis = np.round(np.arange(0.70, 0.981, 0.02), 2)

    # (1) population curve + monotonicity / uniqueness
    H, bed = make_population(n_cells=n_cells, seed=seed)
    fr = area_curve(H, bed, phis)
    diffs = np.diff(fr)
    monotone = bool(np.all(diffs < 0.0))

    # (5) sensitivity sign (per +0.01 in φ), central difference
    dfdphi = np.gradient(fr, phis) * 0.01
    sensitivity_negative = bool(np.all(dfdphi < 0.0))

    # (2) critical-thickness identity is exact at several φ
    id_mismatch = max(critical_thickness_mismatch(H, bed, p)
                      for p in (0.70, 0.80, 0.90, 0.94, 0.98))

    # (3) population inverse is faithful at several planted φ
    pop_inv_err = 0.0
    for p in (0.74, 0.84, 0.90, 0.96):
        a = area_fraction(H, bed, p)
        phat = invert_area(phis, fr, a)
        pop_inv_err = max(pop_inv_err, abs(phat - p))

    # (4) finite-sample unbiasedness + 1/sqrt(N) shrink
    def finite_spread(nn, base_seed):
        rng = np.random.default_rng(base_seed)
        ests = []
        for _ in range(n_seeds):
            idx = rng.integers(0, n_cells, size=nn)
            a = float(classify(build_rtn(H[idx], bed[idx], phi_true)).mean())
            phat = invert_area(phis, fr, a)
            if np.isfinite(phat):
                ests.append(phat)
        ests = np.array(ests)
        return float(ests.mean()), float(ests.std()), len(ests)

    mean_n, std_n, k_n = finite_spread(n_finite, seed + 1)
    mean_2n, std_2n, _ = finite_spread(2 * n_finite, seed + 2)
    bias = abs(mean_n - phi_true)
    # std should fall toward 1/sqrt(2) ~ 0.707 of std_n when N doubles
    shrink_ratio = std_2n / std_n if std_n > 0 else float("inf")

    ok = bool(
        monotone
        and sensitivity_negative
        and id_mismatch == 0
        and pop_inv_err <= 0.01
        and bias <= 0.01
        and shrink_ratio < 0.85
    )

    return {
        "phi_true": float(phi_true),
        "phis": [float(x) for x in phis],
        "area_frac": [float(x) for x in fr],
        "area_pct": [float(100 * x) for x in fr],
        "dfrac_dphi_per_0p01": [float(x) for x in dfdphi],
        "monotone_decreasing": monotone,
        "sensitivity_negative": sensitivity_negative,
        "crit_thickness_mismatch_cells": id_mismatch,
        "population_inverse_max_abs_err": float(pop_inv_err),
        "finite_sample": {
            "n_finite": n_finite,
            "n_seeds": n_seeds,
            "mean_phi_hat": mean_n,
            "std_phi_hat": std_n,
            "bias": float(bias),
            "std_phi_hat_2n": std_2n,
            "shrink_ratio_2n_over_n": float(shrink_ratio),
            "n_valid": k_n,
        },
        "pass": ok,
    }


def main():
    out = run()
    phis, pct = out["phis"], out["area_pct"]
    print("§H.1.1 φ-area inversion — synthetic calibration (real build_rtn math)")
    print("  φ      RTN>1 area %   dA/dφ[/0.01]")
    for p, a, d in zip(phis, pct, out["dfrac_dphi_per_0p01"]):
        print(f"  {p:.2f}   {a:8.3f}      {100*d:+.4f}")
    fs = out["finite_sample"]
    print(f"monotone={out['monotone_decreasing']} sens<0={out['sensitivity_negative']} "
          f"id_mismatch={out['crit_thickness_mismatch_cells']} "
          f"pop_inv_err={out['population_inverse_max_abs_err']:.2e}")
    print(f"finite N={fs['n_finite']}: mean φ̂={fs['mean_phi_hat']:.4f} "
          f"(bias {fs['bias']:.2e}) std={fs['std_phi_hat']:.4f} -> "
          f"std(2N)={fs['std_phi_hat_2n']:.4f} ratio={fs['shrink_ratio_2n_over_n']:.3f}")
    print(f"PASS={out['pass']}")

    print("\n§H.1.1 #2 hydrology-corrected MISI margin band (width ∝(1−φ))")
    b = run_misi_band()
    print("  φ      band %    band%/(1−φ)")
    for p, pc, r in zip(b["phis"], b["band_pct"], b["ratio_band_over_1mphi"]):
        print(f"  {p:.2f}   {pc:7.4f}    {100*r:7.4f}")
    print(f"rel_width_err={b['rel_band_width_max_abs_err']:.2e} "
          f"band==RTN1_inland_mismatch={b['band_vs_rtn_mismatch_cells']} "
          f"inland={b['rtn1_inland_of_flotation']} "
          f"vanish_monotone={b['band_frac_vanishes_monotone']} "
          f"limit_linear_gap={b['limit_linear_rel_gap']:.3f}")
    print(f"PASS={b['pass']}")


if __name__ == "__main__":
    main()
