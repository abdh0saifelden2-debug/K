r"""§V.1 -- Regime Transition Number (RTN) validator for §G.3.

Implements the repo's RTN definition (FUTURE_WORK.md §G.3) and a precision /
recall harness for comparing the predicted intrusion-favourable region
(``RTN > 1``) against observed ocean-intrusion sites.

    RTN = p_ocean_gauge / p_w ,   p_w = p_i - N ,   p_i = rho_i g H_ice

``N`` is the subglacial effective pressure (``N_R`` in §G.3).  ``RTN > 1`` means
the ocean head exceeds the subglacial water pressure -> seawater intrusion is
favoured.  Thinner ice (smaller ``p_i``), larger channels (larger ``N``) and
higher ocean pressure all *raise* RTN, matching the 2024 intrusion literature
[LIT].

**Gauge convention (resolved §G.3 caveat #1).** RTN is a ratio of two pressures
measured against the *same* reference, so the atmosphere must cancel.  Physically,
at the grounding-line bed seawater intrudes when the absolute ocean pressure
``p_atm + rho_w g d_base`` exceeds the absolute subglacial water pressure
``p_atm + p_w``; the ``p_atm`` cancels and the criterion is the gauge ratio
``rho_w g d_base / p_w > 1`` with **no** ``p_atm`` term.  Accordingly ``p_ocean``
and ``N_eff`` are **gauge** pressures (relative to atmosphere) and the default
``p_atm = 0``.  The ``p_atm`` argument is retained only to convert an *absolute*
``p_ocean`` input to gauge (pass the local atmosphere); it must **not** be supplied
for an already-gauge ocean head.  (The earlier code subtracted ``p_atm`` from a
gauge ``p_ocean`` -- a spurious offset whose relative size ``p_atm/(rho_w g d_base)``
blows up as ``d_base -> 0``, i.e. worst exactly in the shallow grounding zone the
prediction targets.)

This module is **pure equation + scoring**; it has no external-data dependency,
so it is exercised directly by ``validation/synthetic/rtn_synthetic.py`` and is
ready to run on real BedMachine / tide fields once those are provided (see
``validation/external/`` and ``validation/README.md``).

Caveats carried from §G.3 (do not over-claim): the gauge mismatch (numerator
subtracting ``p_atm`` while the denominator did not) is now **resolved** -- RTN is
the gauge ratio in which the atmosphere cancels exactly (see the gauge note above).
The one remaining caveat is that the conduction-limited discharge the channel term
relies on rests on the §G.1 *empirical* result, not a bound, so the *threshold
magnitude* is still **[HYP]**; the *direction* (RTN>1 concentrates near grounding
lines) is [VERIFIED] on real Bedmap2 (§H.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# physical constants (SI)
RHO_I = 917.0      # ice density           [kg m^-3]
G = 9.81           # gravity               [m s^-2]
P_ATM = 1.01325e5  # standard atmosphere   [Pa]

# Floor on the subglacial water pressure ``p_w`` [Pa] used by ``rtn`` to avoid
# divide-by-zero / sign flips at/over flotation.  It is the single source of
# truth for this floor: ``rtn`` uses it as the default ``eps`` and downstream
# code paths that replicate the RTN equation (the GPU-generic ``build_rtn`` and
# the synthetic tie-in's ``above_floor`` filter) import it rather than
# re-hardcoding ``1.0``, so changing it here propagates everywhere.
RTN_PW_EPS_PA = 1.0


def ice_overburden(H_ice):
    """p_i = rho_i g H_ice  [Pa].  ``H_ice`` in metres."""
    return RHO_I * G * np.asarray(H_ice, dtype=float)


def rtn(H_ice, p_ocean, N_eff, p_atm=0.0, eps=RTN_PW_EPS_PA):
    r"""Regime Transition Number, elementwise.

    Parameters
    ----------
    H_ice : array_like   ice thickness [m]
    p_ocean : array_like **gauge** ocean head at the grounding line [Pa]
        (e.g. ``rho_w g d_base``).  If an *absolute* ocean pressure is supplied
        instead, pass ``p_atm`` to convert it to gauge.
    N_eff : array_like   subglacial effective pressure N = p_i - p_w [Pa] (gauge)
    p_atm : float        atmosphere to subtract iff ``p_ocean`` is absolute;
        default ``0`` (inputs already gauge -- the atmosphere cancels in the ratio)
    eps : float          floor on p_w to avoid divide-by-zero / sign flips [Pa]

    Returns
    -------
    RTN : ndarray
        ``(p_ocean - p_atm) / p_w`` with ``p_w = max(p_i - N_eff, eps)``.
        With the default ``p_atm = 0`` this is the gauge ratio
        ``p_ocean / p_w`` (atmosphere cancels; see module gauge note).
        Where ``p_i - N_eff <= 0`` (unphysical: water pressure exceeds
        overburden) RTN is set to ``+inf`` (already floated / flotation).
    """
    p_i = ice_overburden(H_ice)
    p_w_raw = p_i - np.asarray(N_eff, dtype=float)
    p_w = np.where(p_w_raw > eps, p_w_raw, np.nan)
    out = (np.asarray(p_ocean, dtype=float) - p_atm) / p_w
    # p_w <= eps  =>  at/over flotation  =>  intrusion trivially favoured
    out = np.where(np.isnan(out), np.inf, out)
    return out


def thickness_threshold(p_ocean, N_eff, p_atm=0.0):
    """Ice thickness ``H*`` at which ``RTN == 1`` for given ``p_ocean``, ``N``.

    ``RTN = 1`` <=> p_ocean - p_atm = p_i - N = rho_i g H* - N
                <=> H* = (p_ocean - p_atm + N) / (rho_i g).
    With the default ``p_atm = 0`` (gauge inputs), ``H* = (p_ocean + N)/(rho_i g)``.
    Ice thinner than ``H*`` is intrusion-favourable (``RTN > 1``).
    """
    return (np.asarray(p_ocean, float) - p_atm + np.asarray(N_eff, float)) / (RHO_I * G)


def classify(rtn_map, threshold=1.0):
    """Boolean intrusion-favourable mask ``RTN > threshold`` (NaN -> False).

    ``rtn`` sets ``RTN = +inf`` for cells at/over flotation (``p_w <= eps``),
    where intrusion is trivially favoured, so ``+inf`` must classify as ``True``.
    ``r > threshold`` already yields ``True`` for ``+inf`` and ``False`` for
    ``NaN``; we keep the explicit ``NaN -> False`` guard for clarity.
    """
    r = np.asarray(rtn_map, float)
    return np.where(np.isnan(r), False, r > threshold)


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    n_pred: int
    n_obs: int


def precision_recall(pred_mask, obs_mask):
    """Score a predicted intrusion mask against an observed-site mask.

    Both inputs are boolean arrays of the same shape.  ``obs_mask`` is the set
    of pixels/locations where intrusion was actually observed.
    """
    pred = np.asarray(pred_mask, bool)
    obs = np.asarray(obs_mask, bool)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs obs {obs.shape}")
    tp = int(np.sum(pred & obs))
    fp = int(np.sum(pred & ~obs))
    fn = int(np.sum(~pred & obs))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return Scores(precision, recall, f1, tp, fp, fn,
                  int(pred.sum()), int(obs.sum()))
