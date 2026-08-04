r"""§V.3 synthetic unit test for the RTN validator (§G.3).

No external data.  We *plant* a known intrusion geometry and check that the
validator's math recovers it:

  1. Build a synthetic ice-thickness map H(x,y) (thin at the margin, thick
     inland) with fixed ocean pressure and effective pressure.
  2. The analytic prediction is ``RTN > 1`` <=> ``H < H*`` with
     ``H* = (p_ocean - p_atm + N) / (rho_i g)``.
  3. Check ``classify(rtn(...))`` reproduces ``H < H*`` exactly (cell for cell).
  4. Scoring pipeline:
       (a) complete survey (obs == truth) -> F1 == 1.
       (b) sparse survey (obs = random subset of truth) -> **recall == 1**
           (every observed site is predicted) while "precision" collapses to the
           survey-sparsity fraction -- the honest "point data, not systematic
           surveys" caveat: precision is not interpretable for sparse catalogues,
           recall is.
       (c) false sites (intrusion planted where RTN < 1) -> recall drops below 1,
           quantifying validator sensitivity.

This validates the *equation and scoring pipeline* before any real BedMachine /
tide data are involved.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from validators.rtn_validator import (  # noqa: E402
    rtn, classify, thickness_threshold, precision_recall,
)


def make_field(n=64, H_min=150.0, H_max=1600.0, seed=0):
    """Margin-to-interior thickness ramp with mild 2-D structure + noise."""
    rng = np.random.default_rng(seed)
    yy = np.linspace(0.0, 1.0, n)[:, None] * np.ones((1, n))
    H = H_min + (H_max - H_min) * yy            # thin (y=0) -> thick (y=1)
    H = H + 60.0 * np.sin(4 * np.pi * np.linspace(0, 1, n))[None, :]
    H = H + rng.normal(0.0, 15.0, size=(n, n))
    return np.clip(H, H_min, H_max)


def run(n=64, p_ocean=8.0e6, N_eff=2.0e5, seed=0, subsample=0.5, tol_cells=0):
    H = make_field(n=n, seed=seed)
    Hstar = float(thickness_threshold(p_ocean, N_eff))
    rtn_map = rtn(H, p_ocean=p_ocean, N_eff=N_eff)
    pred = classify(rtn_map, threshold=1.0)
    truth = H < Hstar                            # analytic intrusion-favourable

    # (3) cell-for-cell agreement between RTN>1 and the analytic threshold
    mismatch = int(np.sum(pred != truth))
    exact = mismatch <= tol_cells

    # (4a) perfect (complete) survey -> F1 == 1
    s_full = precision_recall(pred, truth)

    # (4b) sparse survey: obs = random subset of the true region.
    # recall must be 1 (all observed sites are predicted); "precision" collapses
    # to the survey-sparsity fraction and is *not* interpretable.
    rng = np.random.default_rng(seed + 1)
    obs_sub = truth & (rng.random(truth.shape) < subsample)
    s_sub = precision_recall(pred, obs_sub)

    # (4c) false sites: intrusion planted where RTN < 1 (in the thick-ice region)
    # -> they become misses, so recall < 1.
    not_truth_idx = np.argwhere(~truth)
    pick = rng.choice(len(not_truth_idx), size=max(1, len(not_truth_idx) // 20),
                      replace=False)
    obs_false = obs_sub.copy()
    for i in pick:
        r_, c_ = not_truth_idx[i]
        obs_false[r_, c_] = True
    s_false = precision_recall(pred, obs_false)

    ok = bool(
        exact
        and abs(s_full.f1 - 1.0) < 1e-9
        and abs(s_sub.recall - 1.0) < 1e-9       # sparse obs fully recovered
        and s_false.recall < 1.0 - 1e-9          # false sites are missed
    )
    return {
        "Hstar_m": Hstar,
        "pred_frac": float(pred.mean()),
        "truth_frac": float(truth.mean()),
        "mismatch_cells": mismatch,
        "f1_full": s_full.f1,
        "recall_sparse": s_sub.recall,
        "precision_sparse": s_sub.precision,
        "recall_with_false_sites": s_false.recall,
        "pass": ok,
    }


def main():
    r = run()
    print("=== §V.3 RTN synthetic unit test ===")
    print(f"  planted H* (RTN=1)         : {r['Hstar_m']:.1f} m")
    print(f"  RTN>1 fraction / truth      : {r['pred_frac']:.3f} / {r['truth_frac']:.3f}")
    print(f"  RTN>1 vs analytic mismatch  : {r['mismatch_cells']} cells")
    print(f"  F1 (complete survey)        : {r['f1_full']:.4f}")
    print(f"  sparse-survey recall        : {r['recall_sparse']:.3f}  "
          f"(precision {r['precision_sparse']:.3f} = survey sparsity, not interpretable)")
    print(f"  recall with false sites     : {r['recall_with_false_sites']:.3f}"
          f"  (< 1 => misses planted false sites)")
    print(f"  PASS                        : {r['pass']}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
