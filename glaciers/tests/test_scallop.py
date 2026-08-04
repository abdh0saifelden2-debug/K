"""Tests for the corrected Candidate 3 scallop / melting-instability tooling:
:mod:`scallop_probe`, :mod:`scallop_sweep` and :mod:`scallop_battery`.

CPU-only behaviour tests covering

  * ``ProbeFlow.set_single_mode`` writes the requested single-mode ice base and
    ``_forcing`` adds the steady mean drive on top of the (unchanged) ring
    turbulence,
  * ``mode_amp`` recovers the amplitude of a known sinusoid,
  * ``_enh_stats`` computes the enhancement ratio statistics and ignores
    invalid (non-finite / non-positive conduction) columns,
  * ``melt_normal`` returns the slope-corrected normal-gradient melt and -- the
    regression for the ``np.roll`` bug -- never lets an *invalid* column's
    clipped (bed-row) temperature leak into a valid neighbour's horizontal
    gradient, while staying identical to the plain centred difference when all
    columns are valid,
  * ``_json_safe`` sanitises NaN/Inf so a degenerate run still serialises to
    strict JSON instead of crashing / emitting unparseable tokens.
"""

import json
import math
import warnings
from types import SimpleNamespace

import numpy as np

from scallop_g1_populations import population_decomposition
from scallop_probe import ProbeFlow, _enh_stats, _json_safe
from scallop_sublayer_probe import _sublayer_stats
from scallop_sweep import melt_normal, mode_amp
from subglacial.candidate3_roughness_feedback import Candidate3Config, _thomas_solve


# --------------------------------------------------------------------------- #
# a light stub exposing only what ``melt_normal`` touches, so the gradient
# logic can be exercised deterministically without running the solver.
# --------------------------------------------------------------------------- #
class _StubFlow:
    def __init__(self, theta, fluid, y_ice_x, kappa=1.0, dx=1.0, dy=1.0):
        nx, ny = theta.shape
        self.cfg = SimpleNamespace(nx=nx, ny=ny, kappa=kappa)
        self.theta = np.asarray(theta, dtype=float)
        self.fluid = np.asarray(fluid)
        self.y_ice_x = np.asarray(y_ice_x, dtype=float)
        self.sp = SimpleNamespace(dx=dx, dy=dy)

    def _to_host(self, arr):
        return np.asarray(arr)


def _tiny_cfg(**kw):
    base = dict(nx=32, ny=32, A=4.0, f_amp=0.4, seed=1)
    base.update(kw)
    return Candidate3Config(**base)


# --------------------------------------------------------------------------- #
# ProbeFlow
# --------------------------------------------------------------------------- #
def test_set_single_mode_writes_requested_base():
    s = ProbeFlow(_tiny_cfg(), U_drive=0.0, xp=np)
    a, n_waves = 0.25, 5
    y = s.set_single_mode(a, n_waves)
    # mean is preserved and the recovered amplitude matches the seeded one
    assert np.isclose(y.mean(), s.cfg.y_ice_mean, atol=1e-6)
    rec = mode_amp(y, n_waves, s.Lx, s.sp.dx)
    assert np.isclose(rec, a, rtol=1e-6)
    assert np.allclose(np.asarray(s.y_ice_x), y)


def test_forcing_adds_steady_drive_only_in_x():
    cfg = _tiny_cfg()
    s0 = ProbeFlow(cfg, U_drive=0.0, xp=np)
    s1 = ProbeFlow(cfg, U_drive=0.7, xp=np)
    # identical seed + construction => identical ring forcing on the first call,
    # so the only difference is the steady body force U_drive * fluid in x.
    fx0, fy0 = s0._forcing()
    fx1, fy1 = s1._forcing()
    assert np.allclose(fx1 - fx0, 0.7 * np.asarray(s1.fluid))
    assert np.allclose(fy1, fy0)


# --------------------------------------------------------------------------- #
# mode_amp
# --------------------------------------------------------------------------- #
def test_mode_amp_recovers_amplitude():
    Lx = 4.0 * 2.0 * np.pi
    nx = 128
    dx = Lx / nx
    x = np.arange(nx) * dx
    a, n_waves = 0.37, 8
    yb = 4.8 + a * np.sin(2.0 * np.pi * n_waves * x / Lx)
    assert np.isclose(mode_amp(yb, n_waves, Lx, dx), a, rtol=1e-6)
    # an absent mode reads as (essentially) zero amplitude
    assert mode_amp(yb, n_waves + 3, Lx, dx) < 1e-9


# --------------------------------------------------------------------------- #
# _enh_stats
# --------------------------------------------------------------------------- #
def test_enh_stats_basic_ratio_and_masking():
    n = 64
    yb = np.zeros(n)
    m_cond = np.ones(n)
    m_flow = 1.1 * np.ones(n)
    # poison a few columns: NaN flow, NaN cond, non-positive cond -> all dropped
    m_flow[0] = np.nan
    m_cond[1] = np.nan
    m_cond[2] = 0.0
    st = _enh_stats(yb, m_flow, m_cond, dx=1.0)
    assert np.isclose(st["R_mean"], 1.1)
    assert np.isclose(st["R_max"], 1.1)
    assert np.isclose(st["R_min"], 1.1)
    # flat excess => no slope correlation
    assert st["corr_excess_slope"] == 0.0


def test_enh_stats_all_invalid_columns_returns_nan_without_crashing():
    """Zero valid columns (e.g. the fluid domain entirely penalised) must yield
    NaN ratio stats instead of crashing ``np.max`` on an empty array."""
    n = 16
    yb = np.zeros(n)
    m_flow = np.full(n, np.nan)        # every column invalid
    m_cond = np.full(n, np.nan)        # all-NaN conduction too => degenerate
    with warnings.catch_warnings():
        # the all-NaN nanmean of m_flow/m_cond must not emit the noisy
        # "Mean of empty slice" RuntimeWarning (regression for the suppressed
        # warning); promote it to an error so a regression fails the test.
        warnings.simplefilter("error", RuntimeWarning)
        st = _enh_stats(yb, m_flow, m_cond, dx=1.0)
    for key in ("R_mean", "R_max", "R_min", "R_std", "m_cond_mean", "m_flow_mean"):
        assert math.isnan(st[key])
    # the result still serialises to strict JSON once sanitised
    json.dumps(_json_safe(st), allow_nan=False)


# --------------------------------------------------------------------------- #
# melt_normal -- the np.roll contamination regression
# --------------------------------------------------------------------------- #
def _contamination_grid():
    nx, ny = 6, 16
    rng = np.random.default_rng(0)
    theta = rng.standard_normal((nx, ny))
    fluid = np.zeros((nx, ny), dtype=bool)
    fluid[:5, 0:13] = True          # cols 0-4 valid (jtop=12 => jlo=4 >= 0)
    fluid[5, 0:3] = True            # col 5 invalid (jtop=2 => jhi<0)
    y_ice_x = 8.0 + 0.1 * rng.standard_normal(nx)
    return nx, ny, theta, fluid, y_ice_x


def test_melt_normal_marks_invalid_columns_nan():
    nx, ny, theta, fluid, y_ice_x = _contamination_grid()
    m = melt_normal(_StubFlow(theta, fluid, y_ice_x))
    assert np.isfinite(m[:5]).all()     # valid columns resolved
    assert np.isnan(m[5])               # invalid column is NaN


def test_melt_normal_valid_columns_immune_to_invalid_neighbour():
    """Regression: a valid column adjacent to an invalid one must not pick up
    the invalid column's clipped bed-row temperature through ``np.roll``.
    Changing only the invalid column's field must leave every valid column's
    melt untouched."""
    nx, ny, theta, fluid, y_ice_x = _contamination_grid()
    m_a = melt_normal(_StubFlow(theta, fluid, y_ice_x))
    theta_b = theta.copy()
    theta_b[5, :] = np.random.default_rng(99).standard_normal(ny)  # perturb invalid col only
    m_b = melt_normal(_StubFlow(theta_b, fluid, y_ice_x))
    assert np.allclose(m_a[:5], m_b[:5], equal_nan=True)


def test_melt_normal_matches_centred_difference_when_all_valid():
    """When every column is valid the fix must reduce to the original centred
    np.roll horizontal difference (no behavioural change in the common case)."""
    nx, ny = 8, 20
    rng = np.random.default_rng(3)
    theta = rng.standard_normal((nx, ny))
    fluid = np.zeros((nx, ny), dtype=bool)
    fluid[:, 0:17] = True           # all columns valid
    y_ice_x = 8.0 + 0.2 * rng.standard_normal(nx)
    kappa, dx, dy, top_skip, span = 0.7, 1.3, 0.9, 3, 5
    s = _StubFlow(theta, fluid, y_ice_x, kappa=kappa, dx=dx, dy=dy)
    got = melt_normal(s, top_skip=top_skip, span=span)

    cols = np.arange(nx)
    jtop = np.where(fluid, np.arange(ny)[None, :], -1).max(axis=1)
    jhi = jtop - top_skip
    jlo = jhi - span
    Ty = (theta[cols, jhi] - theta[cols, jlo]) / (span * dy)
    thj = theta[cols, jhi]
    Tx = (np.roll(thj, -1) - np.roll(thj, 1)) / (2.0 * dx)
    slope = np.gradient(y_ice_x, dx)
    nrm = np.sqrt(1.0 + slope ** 2)
    expect = -kappa * (-slope * Tx + Ty) / nrm
    assert np.allclose(got, expect)


# --------------------------------------------------------------------------- #
# _json_safe
# --------------------------------------------------------------------------- #
def test_json_safe_sanitises_non_finite():
    out = {"a": 1.0, "b": float("nan"), "c": float("inf"),
           "nested": {"d": -float("inf"), "ok": 2}, "lst": [1.0, float("nan")]}
    safe = _json_safe(out)
    assert safe["a"] == 1.0
    assert safe["b"] is None
    assert safe["c"] is None
    assert safe["nested"]["d"] is None
    assert safe["nested"]["ok"] == 2
    assert safe["lst"] == [1.0, None]
    # strict JSON now succeeds where the raw dict would raise / emit NaN tokens
    json.dumps(safe, allow_nan=False)


def test_json_safe_coerces_numpy_scalar_and_array_types():
    """On NumPy 2.x ``np.float32`` / ``np.int64`` are not ``float`` / ``int``
    subclasses, so they must be coerced (and non-finite numpy floats nulled)
    rather than passed through to crash ``json.dump`` with a ``TypeError``."""
    out = {
        "f32": np.float32(1.5),
        "f32_nan": np.float32("nan"),
        "f64_inf": np.float64("inf"),
        "i64": np.int64(7),
        "b": np.bool_(True),
        "arr": np.array([1.0, np.nan, 3.0]),
        "iarr": np.array([4, 6, 8]),       # the n_waves_list reuse case
    }
    safe = _json_safe(out)
    assert isinstance(safe["f32"], float) and math.isclose(safe["f32"], 1.5, rel_tol=1e-6)
    assert safe["f32_nan"] is None
    assert safe["f64_inf"] is None
    assert isinstance(safe["i64"], int) and safe["i64"] == 7
    assert isinstance(safe["b"], bool) and safe["b"] is True
    assert safe["arr"] == [1.0, None, 3.0]
    assert safe["iarr"] == [4, 6, 8] and all(isinstance(v, int) for v in safe["iarr"])
    # serialises strictly, which the raw numpy-typed dict could not
    json.dumps(safe, allow_nan=False)


def test_json_safe_coerces_numpy_dict_keys():
    """A numpy-typed dict key (e.g. ``np.int64``) must be coerced to a native
    Python key, otherwise ``json.dumps`` raises ``TypeError: keys must be str,
    int, ...`` on NumPy 2.x where ``np.int64`` is not an ``int`` subclass."""
    out = {np.int64(3): "a", np.float64(1.5): "b", "plain": np.int64(9)}
    safe = _json_safe(out)
    assert set(type(k) for k in safe) <= {int, float, str}
    assert safe[3] == "a" and safe[1.5] == "b" and safe["plain"] == 9
    # strict serialisation now succeeds where the raw numpy-keyed dict raised
    json.dumps(safe, allow_nan=False)


def test_json_safe_non_finite_float_keys_not_collapsed():
    """Non-finite float keys (nan / +-inf) have no strict-JSON form. They must
    be stringified to distinct, lossless keys rather than each coerced to a
    single ``None`` key (which would silently collapse them and drop data)."""
    out = {float("nan"): "a", float("inf"): "b", float("-inf"): "c", 1.5: "d"}
    safe = _json_safe(out)
    # all four entries survive -- no collapse onto a shared ``None`` key
    assert len(safe) == 4
    assert safe[repr(float("nan"))] == "a"
    assert safe[repr(float("inf"))] == "b"
    assert safe[repr(float("-inf"))] == "c"
    assert safe[1.5] == "d"
    # a genuine ``None`` key is preserved as ``None`` (json renders it "null")
    assert _json_safe({None: 1})[None] == 1
    json.dumps(safe, allow_nan=False)


# --------------------------------------------------------------------------- #
# ice-side conduction in the Stefan balance (opt-in)
# --------------------------------------------------------------------------- #
def test_ice_side_off_by_default():
    """The ice-side conduction layer must be absent unless explicitly enabled,
    so existing runs are bit-for-bit unchanged."""
    s = ProbeFlow(_tiny_cfg(), U_drive=0.0, xp=np)
    assert s.cfg.ice_side is False
    assert s.T_ice is None


def test_ice_layer_initial_flux_matches_analytic_conduction():
    """The freshly-initialised linear profile must give the analytic steady
    conduction flux q_ice = kappa_ice * (0 - T_cold) / H_layer, uniform across
    columns (H_layer = (n_ice-1) * dxi)."""
    ratio, Tc, n_ice = 8.0, -1.0, 16
    s = ProbeFlow(_tiny_cfg(ice_side=True, ice_kappa_ratio=ratio,
                            T_ice_cold=Tc, n_ice=n_ice), U_drive=0.0, xp=np)
    kappa_ice = ratio * s.cfg.kappa
    H_layer = (n_ice - 1) * s.ice_dxi
    expect = kappa_ice * (0.0 - Tc) / H_layer
    q_ice = s._ice_loss()
    assert q_ice.shape == (s.cfg.nx,)
    assert np.allclose(q_ice, expect)
    assert (q_ice > 0).all()            # cold ice => heat sink => positive loss


def test_ice_side_zero_contrast_matches_water_only():
    """With no ice/interface temperature contrast (T_cold=0) the ice loss is
    identically zero, so the moving-boundary update reduces exactly to the
    water-only Stefan update."""
    cfg_w = _tiny_cfg()
    cfg_i = _tiny_cfg(ice_side=True, ice_kappa_ratio=8.0, T_ice_cold=0.0)
    sw = ProbeFlow(cfg_w, U_drive=0.0, xp=np)
    si = ProbeFlow(cfg_i, U_drive=0.0, xp=np)
    m = 1.0e-3 * np.ones(cfg_w.nx)
    sw.update_boundary(m=m.copy())
    si.update_boundary(m=m.copy())
    assert np.allclose(np.asarray(si.m_ice), 0.0)
    assert np.allclose(np.asarray(sw.y_ice_x), np.asarray(si.y_ice_x))


def test_ice_side_reduces_net_melt():
    """A cold ice layer (q_ice > 0) must subtract from the water-side melt, so
    the interface advances less than the water-only update for the same
    q_water."""
    cfg_w = _tiny_cfg()
    cfg_i = _tiny_cfg(ice_side=True, ice_kappa_ratio=8.0, T_ice_cold=-1.0)
    sw = ProbeFlow(cfg_w, U_drive=0.0, xp=np)
    si = ProbeFlow(cfg_i, U_drive=0.0, xp=np)
    y0 = np.asarray(sw.y_ice_x).copy()
    m = 1.0e-3 * np.ones(cfg_w.nx)
    sw.update_boundary(m=m.copy())
    si.update_boundary(m=m.copy())
    disp_w = np.asarray(sw.y_ice_x) - y0
    disp_i = np.asarray(si.y_ice_x) - y0
    assert (np.asarray(si.m_ice) > 0).all()
    assert (disp_i < disp_w).all()
    # BCs preserved after the in-frame diffusion/advection update
    assert np.allclose(si.T_ice[:, 0], 0.0)
    assert np.allclose(si.T_ice[:, -1], cfg_i.T_ice_cold)
    assert np.isfinite(si.T_ice).all()


# --------------------------------------------------------------------------- #
# implicit ice-layer integrator (backward-Euler tridiagonal Thomas solve)
# --------------------------------------------------------------------------- #
def test_thomas_solve_matches_dense_tridiagonal():
    """The vectorised Thomas solver must reproduce a dense ``np.linalg.solve``
    on each row's tridiagonal system."""
    rng = np.random.default_rng(0)
    nsys, m = 5, 7
    a = rng.standard_normal((nsys, m))      # sub-diagonal (a[:,0] unused)
    c = rng.standard_normal((nsys, m))      # super-diagonal (c[:,-1] unused)
    b = 4.0 + rng.random((nsys, m))         # dominant main diagonal
    d = rng.standard_normal((nsys, m))
    x = _thomas_solve(a, b, c, d)
    for i in range(nsys):
        M = np.diag(b[i]) + np.diag(a[i, 1:], -1) + np.diag(c[i, :-1], 1)
        assert np.allclose(x[i], np.linalg.solve(M, d[i]))


def test_ice_layer_zero_velocity_keeps_linear_profile_steady():
    """With v_int = 0 the seeded linear conduction profile is the exact steady
    state, so an implicit step must leave it (and hence q_ice) unchanged."""
    s = ProbeFlow(_tiny_cfg(ice_side=True, ice_kappa_ratio=8.0, T_ice_cold=-1.0),
                  U_drive=0.0, xp=np)
    s.v_int = np.zeros(s.cfg.nx)
    T0 = s.T_ice.copy()
    s._evolve_ice_layer(dt_eff=s.cfg.dt * s.cfg.N_update)
    assert np.allclose(s.T_ice, T0)


def test_ice_layer_implicit_stable_under_large_interface_velocity():
    """Regression for the explicit-scheme NaN overflow: a large interface
    velocity (v_int = m/St with small St) used to violate the advection CFL and
    blow up.  The implicit solve must stay finite and bounded within the
    [T_cold, 0] envelope set by the two Dirichlet boundaries, for either sign of
    v_int, even over a large effective time step."""
    cfg = _tiny_cfg(ice_side=True, ice_kappa_ratio=8.0, T_ice_cold=-1.0,
                    n_ice=16)
    Tc = cfg.T_ice_cold
    for sign in (+1.0, -1.0):
        s = ProbeFlow(cfg, U_drive=0.0, xp=np)
        # m ~ 1e-2 with St=2e-4 => |v_int| ~ 50, far past the old explicit limit
        s.v_int = sign * 1.0e-2 / cfg.St * np.ones(cfg.nx)
        for _ in range(50):                 # many updates, no divergence allowed
            s._evolve_ice_layer(dt_eff=cfg.dt * cfg.N_update * 10.0)
        assert np.isfinite(s.T_ice).all()
        assert (s.T_ice <= 1e-9).all() and (s.T_ice >= Tc - 1e-9).all()
        assert np.allclose(s.T_ice[:, 0], 0.0)
        assert np.allclose(s.T_ice[:, -1], Tc)
        q_ice = s._ice_loss()
        assert np.isfinite(q_ice).all()


# --------------------------------------------------------------------------- #
# scallop_sweep Nu means route through the warning-suppressing _nanmean helper
# --------------------------------------------------------------------------- #
def test_scallop_sweep_uses_nanmean_helper_not_bare_nanmean():
    """The Nu(λ)/Nu_flat means in ``scallop_sweep`` must go through the shared
    ``_nanmean`` helper (which suppresses the noisy 'Mean of empty slice'
    RuntimeWarning) rather than a bare ``float(np.nanmean(...))``. Guards against
    a degenerate all-NaN sweep column re-emitting the warning ``_nanmean`` was
    built to silence, and against reintroducing the bare pattern."""
    from pathlib import Path

    import scallop_probe
    import scallop_sweep

    # the helper is the same object, so suppression behaviour is shared
    assert scallop_sweep._nanmean is scallop_probe._nanmean
    src = Path(scallop_sweep.__file__).read_text()
    assert "np.nanmean(" not in src, "bare np.nanmean reintroduced in scallop_sweep"

    # and it genuinely suppresses the warning on an all-NaN slice
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert math.isnan(scallop_sweep._nanmean(np.full(8, np.nan)))


# --------------------------------------------------------------------------- #
# _sublayer_stats -- §1 thermal-sublayer decomposition (scallop_sublayer_probe)
# --------------------------------------------------------------------------- #
def test_sublayer_stats_all_positive_identities():
    """With every column positive, the three Nu ratios coincide and the exact
    harmonic-mean identity ``harm_ratio == nu_ratio_pos == nu_ratio`` holds."""
    m_flat = np.full(8, 2.0)                 # <m_n,flat> = 2.0
    m_bump = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 1.2, 1.8, 2.2])
    st = _sublayer_stats(m_bump, m_flat)

    assert st["n_pos"] == st["n_tot"] == 8
    assert np.isclose(st["nu_flat"], 2.0)
    assert np.isclose(st["nu_bump"], float(np.mean(m_bump)))
    # no reversed columns -> the positive-subset ratio equals the full ratio
    assert np.isclose(st["nu_ratio_pos"], st["nu_ratio"])
    # harmonic-mean route is the exact ratio over the (here: all) positive cols
    assert np.isclose(st["harm_ratio"], st["nu_ratio_pos"], rtol=1e-12)
    # delta_flat / delta_mean is the thickening factor's reciprocal
    assert np.isclose(st["thicken"], st["delta_mean"] / st["delta_flat"])
    assert np.isclose(st["convex"], 1.0 + st["cv2"])


def test_sublayer_stats_nonfinite_columns_dropped_from_population():
    """Non-finite columns (NaN/Inf) leave the finite population entirely: they
    count toward neither ``n_tot`` nor ``n_pos``, and ``nu_bump`` averages only
    the finite columns."""
    m_flat = np.ones(6)
    m_bump = np.array([1.0, 2.0, 3.0, -1.0, np.nan, np.inf])
    st = _sublayer_stats(m_bump, m_flat)

    assert st["n_tot"] == 4                  # NaN and Inf are not finite -> dropped
    assert st["n_pos"] == 3                  # the -1.0 column is finite but excluded
    assert np.isclose(st["nu_bump"], (1.0 + 2.0 + 3.0 - 1.0) / 4.0)
    # subset ratio still diverges from the full ratio (reversed column kept in full)
    assert st["nu_ratio_pos"] != st["nu_ratio"]
    assert np.isclose(st["harm_ratio"], st["nu_ratio_pos"], rtol=1e-12)


def test_sublayer_stats_full_ratio_includes_reversed_columns():
    """``nu_ratio`` (full) and ``nu_ratio_pos`` (positive-only) diverge by
    exactly the reversed-flux contribution; harm_ratio matches the latter."""
    m_flat = np.ones(4)
    m_bump = np.array([1.0, 2.0, 3.0, -1.0])     # one reversed lee column
    st = _sublayer_stats(m_bump, m_flat)

    assert st["n_pos"] == 3 and st["n_tot"] == 4
    assert np.isclose(st["nu_ratio"], (1.0 + 2.0 + 3.0 - 1.0) / 4.0)   # 1.25
    assert np.isclose(st["nu_ratio_pos"], (1.0 + 2.0 + 3.0) / 3.0)     # 2.00
    assert np.isclose(st["harm_ratio"], st["nu_ratio_pos"], rtol=1e-12)


def test_sublayer_stats_no_positive_columns_returns_nan_guard():
    """With every finite column reversed (``m_n <= 0``) no physical sublayer
    exists, so the ``n_pos == 0`` guard returns NaN for every delta_T-derived
    quantity while keeping the flux means well-defined -- and the guard never
    emits an empty-slice RuntimeWarning."""
    m_flat = np.ones(3)                          # <m_n,flat> = 1.0
    m_bump = np.array([-1.0, -2.0, -3.0])        # all reversed -> no sublayer

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        st = _sublayer_stats(m_bump, m_flat)

    assert st["n_pos"] == 0 and st["n_tot"] == 3
    # flux means are still defined; delta_flat = 1/<m_n,flat> stays finite
    assert np.isclose(st["nu_flat"], 1.0)
    assert np.isclose(st["nu_bump"], -2.0)
    assert np.isclose(st["nu_ratio"], -2.0)
    assert np.isclose(st["delta_flat"], 1.0)
    assert st["nu_lt_1"] is True                 # nu_ratio = -2.0 < 1
    assert st["thicken_beats_convex"] is False
    # every delta_T-derived quantity is NaN
    for k in ("nu_ratio_pos", "delta_mean", "cv2", "thicken", "convex",
              "pred_ratio", "harm_ratio"):
        assert math.isnan(st[k]), k

    # downstream JSON serialisation: _json_safe + allow_nan=False yields strict
    # RFC 8259 JSON (NaN -> null) instead of crashing or emitting `NaN` tokens
    dumped = json.dumps(_json_safe(st), allow_nan=False)
    reloaded = json.loads(dumped)
    assert reloaded["nu_ratio_pos"] is None
    assert reloaded["thicken"] is None
    assert reloaded["n_pos"] == 0


# --------------------------------------------------------------------------- #
# population_decomposition -- §G.1 exact area-partition mechanism
# (reattachment / thickened / reversed). Earns "why Nu/Nu_flat < 1" without the
# falsified (1+CV^2) moment truncation: the populations tile the interface, so
# their area-weighted flux shares sum *exactly* to the ratio.
# --------------------------------------------------------------------------- #
def test_population_decomposition_exact_reconstruction_nu_lt_1():
    """One reattachment patch, a thickened majority and one reversed lee column.
    The three flux contributions reconstruct ``Nu/Nu_flat`` exactly and the
    signed excesses sum to ``nu_ratio - 1``; here the thickened+reversed deficit
    outweighs the reattachment surplus, so ``Nu < 1``."""
    m_flat = np.ones(10)                                   # <m_n,flat> = 1.0
    m_bump = np.array([2.0,                                # reattachment (>=1)
                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # thickened (0<m<1)
                       -0.5])                              # reversed (<=0)
    d = population_decomposition(m_bump, m_flat)

    assert d["n_valid"] == 10
    assert np.isclose(d["nu_ratio"], 0.55)                 # (2 + 8*0.5 - 0.5)/10
    # area fractions tile the interface
    assert np.isclose(d["f_reatt"], 0.1)
    assert np.isclose(d["f_thick"], 0.8)
    assert np.isclose(d["f_rev"], 0.1)
    assert np.isclose(d["f_reatt"] + d["f_thick"] + d["f_rev"], 1.0)
    # exactness: contributions sum to the ratio to machine precision (no truncation)
    assert np.isclose(d["C_sum"], d["nu_ratio"], rtol=0, atol=1e-12)
    assert abs(d["C_sum"] - d["nu_ratio"]) < 2e-16
    # signed excesses sum to nu_ratio - 1
    assert np.isclose(d["e_reatt"] + d["e_thick"] + d["e_rev"], d["nu_ratio"] - 1.0)
    # surplus - deficit is exactly nu_ratio - 1
    assert np.isclose(d["surplus"] - d["deficit"], d["nu_ratio"] - 1.0)
    assert np.isclose(d["surplus"], 0.1)                   # C_reatt - f_reatt
    assert np.isclose(d["deficit"], 0.55)
    # mechanism verdict: deficit beats surplus <=> Nu < 1
    assert d["deficit_beats_surplus"] is True
    assert d["nu_lt_1"] is True


def test_population_decomposition_surplus_wins_gives_nu_gt_1():
    """The same machinery is sign-honest: when the reattachment surplus dominates
    (no reversed columns, strong thin-sublayer patches) it reconstructs ``Nu>1``
    and reports ``deficit_beats_surplus == False``."""
    m_flat = np.ones(10)
    m_bump = np.array([3.0, 3.0, 3.0,                      # reattachment
                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # thickened
    d = population_decomposition(m_bump, m_flat)

    assert np.isclose(d["nu_ratio"], 1.25)                 # (9 + 3.5)/10
    assert np.isclose(d["f_rev"], 0.0)
    assert np.isclose(d["C_sum"], d["nu_ratio"], atol=1e-12)
    assert np.isclose(d["surplus"], 0.6) and np.isclose(d["deficit"], 0.35)
    assert np.isclose(d["surplus"] - d["deficit"], d["nu_ratio"] - 1.0)
    assert d["deficit_beats_surplus"] is False
    assert d["nu_lt_1"] is False


def test_population_decomposition_nonfinite_dropped_and_exact():
    """Non-finite columns leave the population entirely (neither counted nor
    averaged); the exact reconstruction still holds on the finite remainder."""
    m_flat = np.ones(5)
    m_bump = np.array([2.0, 0.5, -0.5, np.nan, np.inf])
    d = population_decomposition(m_bump, m_flat)

    assert d["n_valid"] == 3                               # NaN / Inf dropped
    assert np.isclose(d["nu_ratio"], (2.0 + 0.5 - 0.5) / 3.0)
    assert np.isclose(d["C_sum"], d["nu_ratio"], atol=1e-12)
    assert np.isclose(d["f_reatt"] + d["f_thick"] + d["f_rev"], 1.0)


def test_population_decomposition_all_nonfinite_bump_returns_nan_guard():
    """With every bump column non-finite no column survives (``n == 0``), so the
    area partition is undefined.  The ``n == 0`` guard returns NaN for every
    quantity -- mirroring ``_sublayer_stats``' ``n_pos == 0`` guard -- instead of
    raising ZeroDivisionError in ``frac()`` (``0.0 / 0.0`` on Python floats) or
    emitting an empty-slice RuntimeWarning, and the result stays JSON-safe."""
    m_flat = np.ones(4)                                    # <m_n,flat> = 1.0
    m_bump = np.array([np.nan, np.inf, -np.inf, np.nan])   # no finite column

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        d = population_decomposition(m_bump, m_flat)

    assert d["n_valid"] == 0
    assert np.isclose(d["m_flat_mean"], 1.0)               # flat control intact
    assert d["nu_lt_1"] is False
    assert d["deficit_beats_surplus"] is False
    for k in ("nu_ratio", "f_reatt", "f_thick", "f_rev",
              "C_reatt", "C_thick", "C_rev", "C_sum",
              "e_reatt", "e_thick", "e_rev", "surplus", "deficit",
              "cv2", "cond_skew", "top_decile_cond_share",
              "pred_1pcv2", "nu_ratio_pos"):
        assert math.isnan(d[k]), k

    # downstream JSON serialisation: _json_safe + allow_nan=False yields strict
    # RFC 8259 JSON (NaN -> null) instead of crashing or emitting `NaN` tokens
    reloaded = json.loads(json.dumps(_json_safe(d), allow_nan=False))
    assert reloaded["nu_ratio"] is None
    assert reloaded["C_sum"] is None
    assert reloaded["n_valid"] == 0


def test_population_decomposition_all_nonfinite_flat_no_warning():
    """With every *flat* control column non-finite the normalisation
    ``<m_n,flat>`` is undefined (NaN).  The degenerate guard returns the all-NaN
    result *before* the ``v >= m_flat_mean`` comparison, so (a) no
    ``RuntimeWarning`` is emitted -- neither the empty-slice mean nor, on older
    NumPy, the NaN comparison -- and (b) the area fractions ``f_*`` are NaN
    rather than the physically meaningless ``f_reatt = 0`` / ``f_thick ~ 1`` a
    NaN reference would otherwise produce.  ``n_valid`` still reports the finite
    bump count, so this stays distinct from the all-non-finite-*bump* ``n == 0``
    case above."""
    m_bump = np.ones(4)                                    # finite bump columns
    m_flat = np.array([np.nan, np.inf, -np.inf, np.nan])   # no finite control

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)     # the fix: no warning
        d = population_decomposition(m_bump, m_flat)

    assert d["n_valid"] == 4                               # bump columns intact
    assert math.isnan(d["m_flat_mean"])                    # control undefined
    assert d["nu_lt_1"] is False
    assert d["deficit_beats_surplus"] is False
    # every quantity is undefined without a finite reference -- including the
    # area fractions, which a NaN comparison would otherwise misreport.
    for k in ("nu_ratio", "f_reatt", "f_thick", "f_rev",
              "C_reatt", "C_thick", "C_rev", "C_sum",
              "e_reatt", "e_thick", "e_rev", "surplus", "deficit",
              "cv2", "cond_skew", "top_decile_cond_share",
              "pred_1pcv2", "nu_ratio_pos"):
        assert math.isnan(d[k]), k

    reloaded = json.loads(json.dumps(_json_safe(d), allow_nan=False))
    assert reloaded["m_flat_mean"] is None
    assert reloaded["nu_ratio"] is None
    assert reloaded["f_thick"] is None                     # was ~1.0 before fix
    assert reloaded["n_valid"] == 4
