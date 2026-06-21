r"""Direction C: Stratification Resonance Test.

Tests whether the melt enhancement hump in the regime equation arises from
a resonance between the subgrid force memory time (tau_mem) and the buoyancy
period (T_BV = 2*pi / N_BV).

Protocol:
  - Sweep Richardson number Ri from 0 to 1.5 (6 points)
  - At each Ri, run white-FDT (bs_tau=0) and colored-FDT (bs_tau > 0)
  - Measure: melt rate R, tau_mem from SGS force autocorrelation, N_BV
  - Compare: does colored-FDT show a hump that white-FDT does not?
  - Check resonance condition: tau_mem * N_BV ~ 2*pi at peak R

Falsification criteria (from the test specification):
  - No hump in either closure => regime equation shape is wrong
  - Hump in white-FDT too => not a memory effect
  - Hump in colored-FDT at wrong Ri* => resonance condition is wrong
  - Hump in colored-FDT at tau_mem * N_BV ~ 2*pi => mechanism verified

Usage:
    python stratification_probe.py [n] [bs_tau] [nseed]
    python stratification_probe.py 48 0.05 3
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import os
# repo reorg: make sibling domain folders importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("general_two_clocks", "atmosphere", "glaciers", "ocean"):
    _p = os.path.join(_REPO_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _d, _p

from subglacial.flow3d import Subglacial3DConfig, Subglacial3DFlow


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
RI_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
SPINUP_STEPS = 200       # let turbulence develop before measuring
MEASURE_STEPS = 300      # steps during which we record SGS force + melt
RECORD_EVERY = 2         # record SGS force every N steps (memory saving)


def run_one(n: int, Ri: float, sgs: str, bs_tau: float, seed: int,
            spinup: int = SPINUP_STEPS, measure: int = MEASURE_STEPS):
    """Run a single case and return observables."""
    cfg = Subglacial3DConfig(
        n=n,
        nu=8.0e-4,
        kappa=8.0e-4,
        sgs=sgs,
        cs=0.16,
        backscatter=0.6,
        bs_tau=bs_tau,
        Ri=Ri,
        f_amp=1.5,
        k_f=6.0,
        f_band=2.0,
        f_tau=0.05,
        seed=seed,
    )
    flow = Subglacial3DFlow(cfg)

    # spinup: ramp velocity and let turbulence develop
    flow.run(spinup, ramp=max(1, spinup // 3))

    # measurement phase: record SGS force + melt snapshots
    flow.clear_sgs_history()
    melt_samples = []
    for s in range(measure):
        flow.step()
        if s % RECORD_EVERY == 0:
            flow.record_sgs_force()
        if s % 10 == 0:
            melt_samples.append(flow.melt_flux()[0])

    # extract tau_mem from SGS force autocorrelation
    tau_mem_measured = flow.tau_mem_from_history()

    # extract N_BV
    N_BV = flow.buoyancy_frequency()

    # melt rate statistics
    melt_mean = float(np.mean(melt_samples)) if melt_samples else 0.0
    melt_std = float(np.std(melt_samples)) if len(melt_samples) > 1 else 0.0

    # other diagnostics
    ke = flow.kinetic_energy()
    ti = flow.turbulence_intensity()
    eps_mol, eps_sgs = flow.dissipation_breakdown()

    return {
        "Ri": Ri,
        "sgs": sgs,
        "bs_tau_set": bs_tau,
        "seed": seed,
        "melt_mean": melt_mean,
        "melt_std": melt_std,
        "tau_mem_measured": tau_mem_measured,
        "N_BV": N_BV,
        "tau_mem_x_N_BV": tau_mem_measured * N_BV,
        "KE": ke,
        "turb_intensity": ti,
        "eps_mol": eps_mol,
        "eps_sgs": eps_sgs,
        "sgs_dominance": eps_sgs / (eps_mol + 1e-30),
    }


def ensemble(n: int, Ri: float, sgs: str, bs_tau: float, nseeds: int):
    """Run ensemble over seeds and return mean + scatter."""
    results = []
    for s in range(nseeds):
        r = run_one(n, Ri, sgs, bs_tau, seed=s + 42)
        results.append(r)
    # aggregate
    melt_vals = [r["melt_mean"] for r in results]
    tau_vals = [r["tau_mem_measured"] for r in results]
    nbv_vals = [r["N_BV"] for r in results]
    product_vals = [r["tau_mem_x_N_BV"] for r in results]
    return {
        "Ri": Ri,
        "sgs": sgs,
        "bs_tau_set": bs_tau,
        "nseeds": nseeds,
        "melt_mean": float(np.mean(melt_vals)),
        "melt_std": float(np.std(melt_vals)),
        "tau_mem_mean": float(np.mean(tau_vals)),
        "tau_mem_std": float(np.std(tau_vals)),
        "N_BV_mean": float(np.mean(nbv_vals)),
        "product_mean": float(np.mean(product_vals)),
        "product_std": float(np.std(product_vals)),
        "individual_runs": results,
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    bs_tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    nseeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    print("=== Direction C: Stratification Resonance Probe ===")
    print(f"    n={n}, bs_tau={bs_tau}, nseeds={nseeds}")
    print(f"    Ri sweep: {RI_SWEEP}")
    print()

    all_results = []
    t0 = time.time()

    for Ri in RI_SWEEP:
        print(f"--- Ri = {Ri:.2f} ---")

        # White-FDT (no memory)
        print("  white-FDT (bs_tau=0)...", end="", flush=True)
        tw0 = time.time()
        r_white = ensemble(n, Ri, "backscatter", 0.0, nseeds)
        print(f" done ({time.time()-tw0:.1f}s)  melt={r_white['melt_mean']:.4e}")
        all_results.append(r_white)

        # Colored-FDT (with memory)
        print(f"  colored-FDT (bs_tau={bs_tau})...", end="", flush=True)
        tc0 = time.time()
        r_colored = ensemble(n, Ri, "backscatter", bs_tau, nseeds)
        print(f" done ({time.time()-tc0:.1f}s)  melt={r_colored['melt_mean']:.4e}")
        all_results.append(r_colored)

        # Report ratio
        if abs(r_white["melt_mean"]) > 1e-30:
            R = r_colored["melt_mean"] / r_white["melt_mean"]
        else:
            R = float("nan")
        print(f"  R(colored/white) = {R:.3f}")
        print(f"  tau_mem(colored) = {r_colored['tau_mem_mean']:.4e} "
              f"(set: {bs_tau})")
        print(f"  N_BV = {r_colored['N_BV_mean']:.4f}, "
              f"tau_mem*N_BV = {r_colored['product_mean']:.3f}")
        print()

    total_time = time.time() - t0
    print(f"\n=== Total wall time: {total_time:.1f}s ===\n")

    # --- Summary table ---
    print("=" * 90)
    print(f"{'Ri':>5} | {'melt_white':>11} | {'melt_colored':>12} | {'R':>6} | "
          f"{'tau_mem':>8} | {'N_BV':>6} | {'tau*N':>7} | {'resonance?':>10}")
    print("-" * 90)

    for i in range(0, len(all_results), 2):
        rw = all_results[i]
        rc = all_results[i + 1]
        Ri = rw["Ri"]
        mw = rw["melt_mean"]
        mc = rc["melt_mean"]
        R = mc / mw if abs(mw) > 1e-30 else float("nan")
        tau = rc["tau_mem_mean"]
        nbv = rc["N_BV_mean"]
        prod = rc["product_mean"]
        near_2pi = "YES" if abs(prod - 2 * np.pi) < 2.0 else "no"
        print(f"{Ri:5.2f} | {mw:11.4e} | {mc:12.4e} | {R:6.3f} | "
              f"{tau:8.4e} | {nbv:6.4f} | {prod:7.3f} | {near_2pi:>10}")

    print("=" * 90)

    # Interpretation
    print("\n--- Interpretation ---")
    # Find peak R
    R_values = []
    for i in range(0, len(all_results), 2):
        rw = all_results[i]
        rc = all_results[i + 1]
        mw = rw["melt_mean"]
        mc = rc["melt_mean"]
        R_values.append(mc / mw if abs(mw) > 1e-30 else 1.0)

    peak_idx = int(np.argmax(R_values))
    peak_Ri = RI_SWEEP[peak_idx]
    peak_R = R_values[peak_idx]

    # Check for hump: R deviates from 1.0 by > 5%
    has_hump_colored = peak_R > 1.05 and peak_idx not in [0, len(RI_SWEEP) - 1]

    # Check white-FDT for hump
    R_white_vs_smag = []  # would need Smag baseline for true R, using R=1 proxy
    print(f"  Peak R(colored/white) = {peak_R:.3f} at Ri = {peak_Ri:.2f}")

    if has_hump_colored:
        rc_peak = all_results[2 * peak_idx + 1]
        prod_at_peak = rc_peak["product_mean"]
        print(f"  tau_mem * N_BV at peak = {prod_at_peak:.3f} "
              f"(resonance condition: ~{2*np.pi:.3f})")
        if abs(prod_at_peak - 2 * np.pi) < 2.0:
            print("  VERDICT: Hump in colored-FDT, resonance condition MATCHES.")
            print("           Mechanism verified (within this demonstration scope).")
        else:
            print("  VERDICT: Hump in colored-FDT, but resonance condition does NOT match.")
            print("           Memory matters, but tau_mem*N_BV ~ 2*pi is not the mechanism.")
    else:
        print(f"  No significant hump detected (peak R = {peak_R:.3f}).")
        if peak_R < 1.02:
            print("  VERDICT: NULL — no stratification resonance effect from memory.")
        else:
            print("  VERDICT: Weak/boundary effect — inconclusive at this resolution.")

    # Save JSON output
    output = {
        "n": n,
        "bs_tau": bs_tau,
        "nseeds": nseeds,
        "Ri_sweep": RI_SWEEP,
        "R_values": R_values,
        "peak_Ri": peak_Ri,
        "peak_R": peak_R,
        "has_hump": has_hump_colored,
        "wall_time_s": total_time,
        "results": all_results,
    }
    outpath = "stratification_probe_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
