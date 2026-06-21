r"""§V.2 -- Non-local (memory) sliding-law validator for §G.4.

The §G.4 claim (FUTURE_WORK.md) is a memory-modified Schoof/Weertman law

    tau_b(t) = C [ N(t) + dp_thermal(t)/g ] u_b(t)^{1/m}
               + \int_0^t K_ice(t-s) tau_b(s) ds

whose **falsifiable signature** is a *lag*: after a step in subglacial-water
forcing ``q_water`` (e.g. a lake drainage), basal sliding ``u_b`` should surge
with a delay ``tau_lag``, whereas the mainstream (memoryless) Schoof law
predicts an *instantaneous* response.

This validator implements the part that is testable against observations -- the
**lag-detection methodology** -- so it can be exercised on synthetic data now
(``validation/synthetic/sliding_synthetic.py``) and on GPS / InSAR + lake-drainage
catalogues later, without committing to the un-closed kernel.

IMPORTANT (carried verbatim from §G.4 / §B.2): the literal ``K_ice(tau) =
(kappa/(pi tau))^{1/2} exp(-H^2/4 kappa tau)`` form is **not dimensionally
closed** ((kappa/tau)^{1/2} is a velocity, not a rate) and the naive
``H_ice^2/kappa_ice`` lag is ~10^4 yr for H~1 km -- far too long for the
"surges lag by years" claim.  The physical lag must use the **thermal
boundary-layer depth**, not the full thickness.  We therefore treat ``tau_lag``
as an *empirical* quantity to be measured, and keep the kernel shape generic and
explicitly labelled **[HYP, kernel not closed]**.

Keeping the kernel shape generic is only sound if ``estimate_lag`` recovers the
lag *independent of that shape*.  This is verified synthetically in
``validation/synthetic/sliding_synthetic.py::run_kernel_shapes`` (RESULT 17): the
planted lag is recovered to <=10% across five distinct causal kernels (Gamma
k=2/4, log-normal, bi-exponential, symmetric raised-cosine), with a delta-kernel
control returning zero.  So a recovered lag is a property of the data, not of the
assumed Gamma form -- the *estimator* genericity is [VERIFIED]; the *physical*
kernel remains [HYP].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gamma_kernel(n, tau_lag, dt=1.0, shape=2.0):
    r"""Causal, normalised memory kernel peaked at ``tau_lag``.

    Uses a Gamma-shaped kernel ``K(s) \propto s^{k-1} e^{-s/theta}`` with mode at
    ``(k-1) theta = tau_lag`` (``k = shape``).  Generic on purpose -- this is the
    *response-shape* placeholder for the un-closed §G.4 kernel; only its lag
    (mode) is physically interpreted.
    """
    k = float(shape)
    if k <= 1.0:
        raise ValueError("shape must be > 1 so the kernel has an interior mode")
    theta = float(tau_lag) / (k - 1.0) / dt  # in samples
    s = np.arange(n, dtype=float)
    K = np.power(s, k - 1.0) * np.exp(-s / max(theta, 1e-9))
    tot = K.sum()
    return K / tot if tot > 0 else K


def lagged_response(q, kernel):
    """Causal convolution ``(K * q)`` truncated to ``len(q)`` (one-sided)."""
    q = np.asarray(q, float)
    full = np.convolve(q, np.asarray(kernel, float), mode="full")
    return full[: len(q)]


def estimate_lag(q, u_b, dt=1.0, max_lag=None, detrend=True):
    r"""Estimate the forcing->response lag by cross-correlation peak.

    Returns the lag (in the same time units as ``dt``) that maximises the
    cross-correlation of the (optionally detrended) response ``u_b`` with the
    forcing ``q``.  Positive lag => response *follows* forcing (the §G.4
    prediction); ~zero lag => memoryless (Schoof).
    """
    q = np.asarray(q, float).copy()
    u = np.asarray(u_b, float).copy()
    if detrend:
        q -= q.mean()
        u -= u.mean()
    n = len(q)
    if max_lag is None:
        max_lag = n - 1
    max_lag = int(min(max_lag, n - 1))
    lags = np.arange(0, max_lag + 1)
    cc = np.empty(len(lags), float)
    for i, L in enumerate(lags):
        a = u[L:]
        b = q[: n - L]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        cc[i] = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    best = int(np.argmax(cc))
    return lags[best] * dt, lags * dt, cc


@dataclass
class LagScores:
    mean_lag: float
    std_lag: float
    n_events: int
    lags: list


def validate_lags(event_times, u_b, t, max_lag_samples, dt=1.0):
    """For each forcing-event time, find the subsequent ``u_b`` peak and record
    the lag.  Methodology mirrors the §G.4 retrospective test on lake-drainage
    catalogues (e.g. Whillans 2012)."""
    t = np.asarray(t, float)
    u = np.asarray(u_b, float)
    lags = []
    for te in event_times:
        i0 = int(np.searchsorted(t, te))
        i1 = int(min(i0 + max_lag_samples, len(u) - 1))
        if i1 <= i0:
            continue
        ipk = i0 + int(np.argmax(u[i0:i1 + 1]))
        lags.append(float(t[ipk] - te))
    lags = np.asarray(lags, float)
    if len(lags) == 0:
        return LagScores(float("nan"), float("nan"), 0, [])
    return LagScores(float(lags.mean()), float(lags.std()), len(lags), lags.tolist())
