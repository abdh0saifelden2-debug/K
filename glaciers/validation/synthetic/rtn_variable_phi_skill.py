r"""§V.4 — does a *spatially-varying* basal connectivity fraction phi(x,y) let the
Regime Transition Number add skill over thickness-above-flotation?  (plant-and-recover)

Why this exists
---------------
The committed baseline-skill test (``rtn_baseline_skill.py``) proved, on 478,767
real Bedmap2 cells, that with a **constant** phi the RTN is an exact monotone
function of the flotation fraction ``f = 1 - H_af/H``:

    RTN = (1 - f)/phi      =>  Spearman(RTN, f) = -1 ,  {RTN>1} == {f < 1-phi}.

So constant-phi RTN adds **zero** skill -- it *is* thickness-above-flotation,
nondimensionalised by phi.  But that test only falsifies the *constant-phi* case.
RTN's only possible source of independent information was always the
**connectivity fraction phi itself**: if phi = phi(x,y) is a data-derived basal
water-system field (radar bed specularity; hydraulic-potential routing) rather
than a single continent number, then ``RTN = (rho_w d_base)/(phi(x,y) rho_i H)``
is no longer a monotone function of ``f`` alone -- the Spearman degeneracy breaks
and RTN *can* reorder intrusion-favourability beyond the flotation threshold.

Physical sign convention (matches ``rtn_validator``): ``p_w = phi*p_i`` so
``RTN = p_ocean/(phi*p_i)``.  Efficient, *well-connected* channelized drainage
holds a **low** steady water-pressure fraction (small phi, large effective
pressure ``N=(1-phi)p_i``) -- so the ocean head exceeds it more easily and RTN is
**higher** there.  Hence connectivity ``c`` enters as a *decreasing* map
``phi = phi_hi - (phi_hi-phi_lo) c``: well-connected (c->1) => low phi => high RTN.

This module is the plant-and-recover that makes the rescue claim precise and
falsifiable:

  0. **Anchor.** With phi constant it reproduces the committed result exactly --
     Spearman(RTN, f) = -1 and {RTN>1} == {f<1-phi}, zero disagreeing cells.
  1. **Degeneracy break.** With a planted variable phi(x,y), |Spearman(RTN_phi, f)|
     < 1 and {RTN_phi>1} disagrees with the constant-phi flotation threshold on a
     non-empty set of cells.
  2. **Skill gain is real.** Against a planted ground truth whose intrusion depends
     on *both* flotation proximity and connectivity (S_true = f/phi_true > 1), the
     phi-aware screen ``{RTN_{phi_obs}>1}`` beats the flotation-only baseline
     (higher ROC-AUC and F1) using only a *noisy observable* phi_obs.
  3. **Null control.** If phi_obs is replaced by a connectivity-blind random field
     of the same marginal distribution, the skill gain vanishes (AUC ~ baseline) --
     so the gain comes from genuine connectivity information, not variance per se.
  4. **Dose-response.** The skill gain rises monotonically with the informativeness
     corr(phi_obs, phi_true) and -> 0 as phi_obs -> const or -> random.

It validates the *method* (variable-phi is the only thing that can rescue RTN, and
it works iff phi carries real connectivity information), not any phi for Antarctica.
The real-geometry run is ``validation/external/rtn_variable_phi_real.py``.  No GPU,
no download.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from external.run_rtn_bedmap2 import build_rtn, classify, RHO_W, RHO_I  # noqa: E402

# connectivity -> phi map (decreasing): well-connected (c=1) => low phi => high RTN
PHI_LO = 0.80   # well-connected / channelized: low water-pressure fraction
PHI_HI = 0.97   # poorly connected / distributed: high water-pressure fraction

# planted-population params (reuse the realistic Bedmap2 tail from rtn_phi_synthetic)
_R_MEDIAN = 0.62    # ratio r = d_base/H; tuned so a few % of cells cross flotation
_R_SIGMA = 0.30
_H_MEDIAN = 1500.0
_H_SIGMA = 0.45


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def flot_fraction(H, bed):
    r"""Flotation fraction ``f = (rho_w/rho_i) d_base / H = 1 - H_af/H`` (in [0,1+])."""
    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    return (RHO_W / RHO_I) * d_base / H


def phi_of_c(c):
    """Decreasing connectivity->phi map: phi = PHI_HI - (PHI_HI-PHI_LO) c."""
    return PHI_HI - (PHI_HI - PHI_LO) * clip01(c)


def make_population(n_cells=300_000, seed=0):
    """Plant grounded cells: thickness ``H``, bed (so ``d_base=-bed=r*H``), and a
    latent connectivity field ``c`` in [0,1] independent of the flotation fraction.

    Returns ``(H, bed, c, f)`` with ``f`` the flotation fraction.
    """
    rng = np.random.default_rng(seed)
    H = np.exp(rng.normal(np.log(_H_MEDIAN), _H_SIGMA, n_cells))
    r = np.exp(rng.normal(np.log(_R_MEDIAN), _R_SIGMA, n_cells))
    bed = -r * H
    # connectivity ~ Beta(2,2): smooth, bounded, independent of (H,r) by construction
    c = rng.beta(2.0, 2.0, n_cells)
    f = flot_fraction(H, bed)
    return H, bed, c, f


def roc_auc(score, y):
    """ROC-AUC of a continuous ``score`` (higher => more positive) vs boolean ``y``.

    Rank-sum (Mann-Whitney) identity with average-rank tie handling.
    """
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    m = np.isfinite(score)
    score, y = np.asarray(score, float)[m], y[m]
    n_pos = int(y.sum()); n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def f1_score(pred, y):
    pred = np.asarray(pred, bool); y = np.asarray(y, bool)
    tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    if not np.isfinite(prec) or not np.isfinite(rec) or (prec + rec) == 0:
        return float("nan"), prec, rec
    return 2 * prec * rec / (prec + rec), prec, rec


def spearman(a, b, rng, cap=400_000):
    from scipy import stats
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size > cap:
        idx = rng.choice(a.size, size=cap, replace=False)
        a, b = a[idx], b[idx]
    return float(stats.spearmanr(a, b).statistic)


def run(n_cells=300_000, sigma_obs=0.12, seed=0):
    """Assemble the variable-phi skill calibration.  Returns a dict with top ``pass``."""
    rng = np.random.default_rng(seed)
    H, bed, c, f = make_population(n_cells=n_cells, seed=seed)
    finite = np.isfinite(f) & (f > 0)

    phi_true = phi_of_c(c)
    phi_bar = float(np.mean(phi_true))            # fairest constant baseline = mean(phi_true)

    # ----- ground truth: intrusion favoured iff RTN_true>1 i.e. f > phi_true -----
    rtn_true = build_rtn(H, bed, phi_true)
    intrude = classify(rtn_true) & finite          # {f > phi_true}
    n_pos = int(intrude.sum()); base_rate = n_pos / int(finite.sum())

    # ----- (0) ANCHOR: constant phi reproduces the committed baseline result -----
    # RTN = f/phi with f = (rho_w/rho_i) d_base/H = 1 - H_af/H, so {RTN>1} == {f>phi},
    # equivalently {H_af/H < 1-phi} -- the committed thickness-above-flotation threshold.
    h_af_frac = 1.0 - f                            # = H_af/H (committed 'flotation fraction')
    rtn_const = build_rtn(H, bed, phi_bar)
    pred_const = classify(rtn_const) & finite
    flot_thresh = (f > phi_bar) & finite           # {RTN_const>1} == {H_af/H < 1-phi}
    disagree_const = int(np.sum((pred_const ^ flot_thresh) & finite))
    sp_const = spearman(np.where(finite, rtn_const, np.nan),
                        np.where(finite, f, np.nan), rng)            # +1 (RTN increases in f)
    sp_const_haf = spearman(np.where(finite, rtn_const, np.nan),
                            np.where(finite, h_af_frac, np.nan), rng)  # -1 (committed convention)

    # ----- (1) DEGENERACY BREAK with a noisy *observable* of connectivity -----
    c_obs = clip01(c + rng.normal(0.0, sigma_obs, n_cells))
    phi_obs = phi_of_c(c_obs)
    rtn_obs = build_rtn(H, bed, phi_obs)
    pred_obs = classify(rtn_obs) & finite
    sp_var = spearman(np.where(finite, rtn_obs, np.nan),
                      np.where(finite, f, np.nan), rng)
    disagree_var = int(np.sum((pred_obs ^ flot_thresh) & finite))

    # ----- (2) SKILL: phi-aware screen vs flotation-only baseline -----
    ff = f[finite]; yy = intrude[finite]
    score_base = ff                                  # f/phi_bar  (monotone in f)
    score_obs = ff / phi_obs[finite]                 # monotone in RTN_obs
    auc_base = roc_auc(score_base, yy)
    auc_obs = roc_auc(score_obs, yy)
    f1_base, p_base, r_base = f1_score(flot_thresh[finite], yy)
    f1_obs, p_obs, r_obs = f1_score(pred_obs[finite], yy)

    # ----- (3) NULL control: connectivity-blind random phi of same marginal -----
    c_rand = rng.beta(2.0, 2.0, n_cells)             # independent of truth
    phi_rand = phi_of_c(c_rand)
    pred_null = classify(build_rtn(H, bed, phi_rand)) & finite
    score_null = ff / phi_rand[finite]
    auc_null = roc_auc(score_null, yy)
    f1_null, _, _ = f1_score(pred_null[finite], yy)

    # ----- (2b) where the gain lives: the near-flotation *contested band* -----
    # baseline (single cutoff phi_bar) and the per-cell truth (cutoff phi_true) can
    # only disagree where f is in the phi range [PHI_LO, PHI_HI]; restrict there to
    # localize the skill gain to the grounding-zone band the screen actually targets.
    band = (ff > PHI_LO) & (ff < PHI_HI)
    if int(band.sum()) > 0:
        f1_base_band, _, _ = f1_score(flot_thresh[finite][band], yy[band])
        f1_obs_band, _, _ = f1_score(pred_obs[finite][band], yy[band])
        n_band = int(band.sum())
    else:
        f1_base_band = f1_obs_band = float("nan"); n_band = 0

    # ----- (4) DOSE-RESPONSE: skill gain vs informativeness corr(phi_obs,phi_true) -----
    dose = []
    for s in (0.0, 0.05, 0.12, 0.25, 0.5, 1.0, 3.0):
        co = clip01(c + rng.normal(0.0, s, n_cells))
        po = phi_of_c(co)
        info = float(np.corrcoef(po[finite], phi_true[finite])[0, 1])
        auc_s = roc_auc(ff / po[finite], yy)
        dose.append(dict(sigma=s, corr_phi_obs_true=info, auc=auc_s,
                         gain=auc_s - auc_base))
    gains = np.array([d["gain"] for d in dose])
    infos = np.array([d["corr_phi_obs_true"] for d in dose])
    # gain should be monotone non-increasing as noise grows (allow tiny sampling wiggle)
    dose_monotone = bool(np.all(np.diff(gains) <= 1e-3))
    # and positively related to informativeness
    dose_corr = float(np.corrcoef(infos, gains)[0, 1])

    # anchor: constant phi => RTN is a *perfect monotone* function of the flotation
    # fraction (Spearman -1 vs H_af/H, matching the committed result) and {RTN>1} is
    # *identical* to the thickness-above-flotation threshold (0 disagreeing cells).
    anchor_ok = bool(abs(sp_const) > 0.999 and sp_const_haf < -0.999 and disagree_const == 0)
    degeneracy_ok = bool(abs(sp_var) < 0.999 and disagree_var > 0)
    # the operationally relevant gain is F1 at the natural operating point {RTN>1};
    # AUC supplements it (threshold-free). The gain must be (a) positive and (b)
    # demonstrably connectivity-sourced: it must beat the random-phi null, which
    # carries the same variance but no information.
    skill_ok = bool(
        f1_obs > f1_base + 0.01 and auc_obs > auc_base + 0.001
        and f1_obs > f1_null and auc_obs > auc_null)
    null_ok = bool(auc_null <= auc_base + 0.005 and f1_null <= f1_base + 0.02)
    dose_ok = bool(dose_monotone and dose_corr > 0.9 and gains[-1] <= 0.01)

    ok = anchor_ok and degeneracy_ok and skill_ok and null_ok and dose_ok

    return {
        "what": ("variable-phi(x,y) is the only way RTN can add skill over "
                 "thickness-above-flotation; it works iff phi carries real "
                 "connectivity information"),
        "phi_map": {"PHI_LO": PHI_LO, "PHI_HI": PHI_HI,
                    "phi=PHI_HI-(PHI_HI-PHI_LO)*c": "well-connected c->1 => low phi => high RTN"},
        "population": {"n_cells": int(n_cells), "n_finite": int(finite.sum()),
                       "phi_bar_baseline": phi_bar, "intrusion_base_rate": base_rate,
                       "n_intrude_true": n_pos},
        "anchor_constant_phi": {
            "spearman_rtn_oceanratio_f": sp_const,
            "spearman_rtn_flotfrac_Haf": sp_const_haf,
            "disagree_vs_flotthresh_cells": disagree_const,
            "note": ("reproduces committed baseline: constant-phi RTN is an exact monotone "
                     "function of the flotation fraction (Spearman -1 vs H_af/H) and {RTN>1} "
                     "is identical to thickness-above-flotation -- zero skill"),
            "pass": anchor_ok},
        "degeneracy_break_variable_phi": {
            "sigma_obs": sigma_obs, "spearman_rtn_f": sp_var,
            "disagree_vs_flotthresh_cells": disagree_var, "pass": degeneracy_ok},
        "skill": {
            "auc_baseline_flotation": auc_base, "auc_phi_aware": auc_obs,
            "auc_gain": auc_obs - auc_base,
            "f1_baseline": f1_base, "f1_phi_aware": f1_obs,
            "precision_baseline": p_base, "recall_baseline": r_base,
            "precision_phi_aware": p_obs, "recall_phi_aware": r_obs,
            "contested_band": {"PHI_LO<f<PHI_HI": [PHI_LO, PHI_HI], "n_cells": n_band,
                               "f1_baseline": f1_base_band, "f1_phi_aware": f1_obs_band,
                               "note": "where baseline and per-cell truth can disagree"},
            "pass": skill_ok},
        "null_random_phi": {"auc_null": auc_null, "auc_gain_vs_baseline": auc_null - auc_base,
                            "f1_null": f1_null, "f1_baseline": f1_base, "pass": null_ok},
        "dose_response": {"rows": dose, "monotone_nonincreasing": dose_monotone,
                          "corr_info_vs_gain": dose_corr, "pass": dose_ok},
        "verdict": (
            "constant-phi RTN == thickness-above-flotation (Spearman %.4f vs H_af/H, 0 "
            "disagreeing cells); a data-derived variable phi(x,y) breaks the degeneracy "
            "(Spearman %.3f, %d disagreeing cells) and adds genuine operating-point skill "
            "(F1 %.3f -> %.3f; AUC %.3f -> %.3f) -- but ONLY when phi carries connectivity "
            "information (random-phi F1 %.3f, AUC gain %.3f). RTN is rescuable iff a real "
            "connectivity field (specularity / hydraulic routing) is supplied; the "
            "real-geometry test is rtn_variable_phi_real.py."
            % (sp_const_haf, sp_var, disagree_var, f1_base, f1_obs, auc_base, auc_obs,
               f1_null, auc_null - auc_base)),
        "pass": ok,
    }


def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dose = res["dose_response"]["rows"]
    info = [d["corr_phi_obs_true"] for d in dose]
    gain = [d["gain"] for d in dose]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    s = res["skill"]
    ax[0].bar(["baseline\n(flotation)", "phi-aware\nRTN", "random-phi\n(null)"],
              [s["auc_baseline_flotation"], s["auc_phi_aware"], res["null_random_phi"]["auc_null"]],
              color=["#888", "#1f77b4", "#d62728"])
    ax[0].set_ylim(0.5, 1.0); ax[0].set_ylabel("ROC-AUC vs planted intrusion truth")
    ax[0].set_title("(a) variable-phi adds skill; random-phi does not")
    ax[0].axhline(s["auc_baseline_flotation"], color="k", ls="--", lw=0.8)
    ax[1].plot(info, gain, "o-", color="#1f77b4")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel(r"informativeness  corr($\phi_{obs}$, $\phi_{true}$)")
    ax[1].set_ylabel("AUC gain over flotation baseline")
    ax[1].set_title("(b) dose-response: skill gain vs phi informativeness")
    ax[1].grid(alpha=0.3)
    fig.suptitle("§V.4 RTN variable-phi skill (plant-and-recover)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "reports", "rtn_variable_phi_skill.json")))
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    res = run(n_cells=a.n)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    s = res["skill"]
    print("=== §V.4 RTN variable-phi skill (plant-and-recover) ===")
    ac = res['anchor_constant_phi']
    print(f"  anchor (constant phi): Spearman(RTN, H_af/H)={ac['spearman_rtn_flotfrac_Haf']:+.4f}"
          f"  disagree={ac['disagree_vs_flotthresh_cells']}  PASS={ac['pass']}")
    db = res["degeneracy_break_variable_phi"]
    print(f"  degeneracy break (var phi): Spearman(RTN,f)={db['spearman_rtn_f']:+.4f}  "
          f"disagree={db['disagree_vs_flotthresh_cells']}  PASS={db['pass']}")
    print(f"  skill: AUC {s['auc_baseline_flotation']:.3f} (baseline) -> {s['auc_phi_aware']:.3f} "
          f"(phi-aware), gain +{s['auc_gain']:.3f}; F1 {s['f1_baseline']:.3f}->{s['f1_phi_aware']:.3f}  "
          f"PASS={s['pass']}")
    print(f"  null (random phi): AUC={res['null_random_phi']['auc_null']:.3f} "
          f"(gain {res['null_random_phi']['auc_gain_vs_baseline']:+.3f})  PASS={res['null_random_phi']['pass']}")
    dr = res["dose_response"]
    print(f"  dose-response: corr(info,gain)={dr['corr_info_vs_gain']:.3f} monotone={dr['monotone_nonincreasing']}  "
          f"PASS={dr['pass']}")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  PASS={res['pass']}")
    print(f"  json -> {a.out}")
    if not a.no_fig:
        make_figure(res, os.path.splitext(a.out)[0] + ".png")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
