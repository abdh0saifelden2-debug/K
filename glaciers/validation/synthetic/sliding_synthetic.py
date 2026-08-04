r"""§V.3 synthetic unit test for the non-local sliding-law validator (§G.4).

No external data.  We *plant* a known forcing->response lag and check that the
lag-detection methodology recovers it:

  1. Generate a subglacial-water forcing ``q(t)`` (random lake-drainage pulses).
  2. Produce a synthetic basal-velocity response ``u_b(t) = base + (K * q)(t)``
     using a causal kernel whose mode sits at a *planted* lag ``tau_true``,
     then add observational noise.
  3. Check that ``estimate_lag`` (cross-correlation peak) recovers ``tau_true``
     within tolerance, and that the event-based ``validate_lags`` recovers it
     from the drainage-event catalogue.
  4. Control: a *memoryless* response (``tau_true = 0``) must return ~zero lag,
     so the test can distinguish the §G.4 prediction (delayed surge) from the
     mainstream Schoof law (instantaneous).

This validates the *lag-recovery pipeline* before touching GPS / InSAR data.  It
does **not** assert the §G.4 kernel is physically correct -- that kernel is
[HYP, not dimensionally closed]; only the empirical lag is interpreted.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from validators.sliding_validator import (  # noqa: E402
    gamma_kernel, lagged_response, estimate_lag, validate_lags,
)


def make_forcing(spacing, n_events=6, width=4, amp=1.0, seed=0):
    """Well-separated positive drainage pulses on a quiet baseline.

    Events are spaced ``spacing`` samples apart so each memory response decays
    before the next event -- this isolates the lag (no response overlap).
    """
    rng = np.random.default_rng(seed)
    n = int(spacing * (n_events + 1))
    q = np.zeros(n)
    ev = (np.arange(1, n_events + 1) * spacing).astype(int)
    for t0 in ev:
        q[t0:t0 + width] += amp * (1.0 + 0.3 * rng.standard_normal())
    return q, ev, n


def run(tau_true=40.0, noise=0.03, seed=0):
    # space events well beyond the kernel extent (~6*tau) so responses isolate
    spacing = int(max(300.0, 8.0 * tau_true))
    window = int(0.8 * spacing)
    q, ev, n = make_forcing(spacing, seed=seed)
    rng = np.random.default_rng(seed + 7)

    # planted memory response (kernel mode == tau_true)
    K = gamma_kernel(n=max(400, 6 * int(tau_true)), tau_lag=tau_true, dt=1.0, shape=2.0)
    resp = lagged_response(q, K)
    u_b = 100.0 + resp / resp.max() * 5.0
    u_b = u_b + rng.normal(0.0, noise * 5.0, size=n)

    lag_hat, lags, cc = estimate_lag(q, u_b, dt=1.0, max_lag=window)
    ev_scores = validate_lags(ev, u_b, np.arange(n), max_lag_samples=window)

    # memoryless control (tau ~ 0): response is instantaneous
    u_b0 = 100.0 + q / q.max() * 5.0 + rng.normal(0.0, noise * 5.0, size=n)
    lag0, _, _ = estimate_lag(q, u_b0, dt=1.0, max_lag=window)

    tol = 0.20 * tau_true + 3.0     # samples
    ok = bool(
        abs(lag_hat - tau_true) <= tol
        and abs(ev_scores.mean_lag - tau_true) <= 2 * tol
        and lag0 <= 0.25 * tau_true            # control stays near zero
    )
    return {
        "tau_true": tau_true,
        "lag_xcorr": float(lag_hat),
        "lag_events_mean": ev_scores.mean_lag,
        "lag_events_std": ev_scores.std_lag,
        "n_events": ev_scores.n_events,
        "lag_memoryless_control": float(lag0),
        "pass": ok,
    }


# --- kernel-shape-genericity (§G.4 "kernel not closed" caveat) ---------------
# The §G.4 kernel *shape* is [HYP]; only its lag (mode) is physically interpreted
# (validators/sliding_validator.py). The lag estimator must therefore recover the
# planted lag *whatever the kernel shape* -- otherwise a recovered lag would be an
# artefact of the assumed gamma form. These builders all place their mode at the
# same target lag but differ markedly in shape (skew, symmetry, tail).
def _lognormal_kernel(n, tau_lag, dt=1.0, sigma=0.5):
    s = np.arange(n, dtype=float)
    mu = np.log(tau_lag / dt) + sigma ** 2          # mode at exp(mu - sigma^2)
    K = np.zeros(n)
    m = s > 0
    K[m] = (1.0 / s[m]) * np.exp(-(np.log(s[m]) - mu) ** 2 / (2 * sigma ** 2))
    tot = K.sum()
    return K / tot if tot > 0 else K


def _biexp_kernel(n, tau_lag, dt=1.0, ratio=3.0):
    """Difference-of-exponentials (rise then decay); mode placed at tau_lag.
    Requires ``ratio > 1`` (the two time-scales must differ; ratio<=1 gives a
    zero/negative log).  Called only with the hardcoded ``ratio=3.0``."""
    t = tau_lag / dt
    a = t / ((ratio * np.log(ratio)) / (ratio - 1.0))   # so mode s* = t
    b = ratio * a
    s = np.arange(n, dtype=float)
    K = np.exp(-s / b) - np.exp(-s / a)
    K[K < 0] = 0.0
    tot = K.sum()
    return K / tot if tot > 0 else K


def _raised_cosine_kernel(n, tau_lag, dt=1.0):
    """Symmetric raised-cosine bump on [0, 2*tau] (peak at tau) -- zero skew."""
    t = tau_lag / dt
    s = np.arange(n, dtype=float)
    K = np.zeros(n)
    m = (s >= 0) & (s <= 2 * t)
    K[m] = 0.5 * (1.0 - np.cos(np.pi * s[m] / t))
    tot = K.sum()
    return K / tot if tot > 0 else K


def kernel_families(N, tau_true, dt=1.0):
    return {
        "gamma_k2": gamma_kernel(N, tau_true, dt=dt, shape=2.0),
        "gamma_k4": gamma_kernel(N, tau_true, dt=dt, shape=4.0),
        "lognormal": _lognormal_kernel(N, tau_true, dt=dt, sigma=0.5),
        "biexp": _biexp_kernel(N, tau_true, dt=dt, ratio=3.0),
        "raised_cosine": _raised_cosine_kernel(N, tau_true, dt=dt),
    }


def run_kernel_shapes(tau_true=40.0, noise=0.03, seed=0):
    """Recover the planted lag across five distinct kernel shapes, confirming the
    cross-correlation lag estimator is kernel-shape-generic (§G.4 caveat)."""
    spacing = int(max(300.0, 8.0 * tau_true))
    window = int(0.8 * spacing)
    q, ev, n = make_forcing(spacing, seed=seed)
    rng = np.random.default_rng(seed + 11)
    N = max(400, 6 * int(tau_true))

    tol = 0.12 * tau_true + 1.0      # samples: recover lag to ~10%
    families = {}
    worst_mode = 0.0
    worst_err = 0.0
    for name, K in kernel_families(N, tau_true).items():
        mode = float(np.argmax(K))                   # kernel's own peak (mode)
        resp = lagged_response(q, K)
        u_b = 100.0 + resp / resp.max() * 5.0 + rng.normal(0.0, noise * 5.0, size=n)
        est, _, _ = estimate_lag(q, u_b, dt=1.0, max_lag=window)
        families[name] = {
            "mode": mode, "lag_xcorr": float(est),
            "abs_err_vs_mode": float(abs(est - mode)),
            "abs_err_vs_tau": float(abs(est - tau_true)),
        }
        worst_mode = max(worst_mode, abs(mode - tau_true))
        worst_err = max(worst_err, abs(est - tau_true))

    # memoryless control (delta kernel) must still return ~0 lag.  This control
    # is noise-free (unlike run()'s noisy memoryless case); the noisy-memoryless
    # robustness check lives in run(), so this only verifies the shape itself.
    Kd = np.zeros(N); Kd[0] = 1.0
    u0 = 100.0 + lagged_response(q, Kd) / max(q.max(), 1e-9) * 5.0
    lag0, _, _ = estimate_lag(q, u0, dt=1.0, max_lag=window)

    ok = bool(
        worst_mode <= 2.0                       # all shapes peak at the target
        and all(f["abs_err_vs_tau"] <= tol for f in families.values())
        and lag0 <= 0.25 * tau_true
    )
    return {
        "tau_true": tau_true,
        "families": families,
        "worst_mode_offset": float(worst_mode),
        "worst_lag_err": float(worst_err),
        "tol_samples": float(tol),
        "memoryless_control": float(lag0),
        "pass": ok,
    }


def main():
    r = run()
    rk = run_kernel_shapes()
    print("=== §V.3 non-local sliding-law synthetic unit test ===")
    print(f"  planted lag tau_true        : {r['tau_true']:.1f} samples")
    print(f"  recovered (cross-corr)      : {r['lag_xcorr']:.1f}")
    print(f"  recovered (event catalogue) : {r['lag_events_mean']:.1f}"
          f" +- {r['lag_events_std']:.1f}  (n={r['n_events']})")
    print(f"  memoryless control lag      : {r['lag_memoryless_control']:.1f}"
          f"  (~0 => instantaneous Schoof)")
    print(f"  PASS                        : {r['pass']}")
    print("  -- kernel-shape genericity (§G.4 kernel [HYP]) --")
    for name, f in rk["families"].items():
        print(f"     {name:14s} mode={f['mode']:.0f}  lag_hat={f['lag_xcorr']:.0f}"
              f"  err_vs_tau={f['abs_err_vs_tau']:.0f}")
    print(f"     worst lag err {rk['worst_lag_err']:.1f} <= tol {rk['tol_samples']:.1f} samples"
          f"; control {rk['memoryless_control']:.0f}  PASS={rk['pass']}")
    return 0 if (r["pass"] and rk["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
