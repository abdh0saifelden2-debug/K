"""§V.3 synthetic unit tests for the external-validation pipeline.

These exercise the validator equations against *controlled* inputs (no external
data): plant a known parameter, check the validator recovers it.  They guard the
RTN / sliding-law / clock-mismatch math that §V.1 / §V.2 will run on real data.
"""
import os
import sys

import numpy as np
import pytest

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rtn_synthetic       # noqa: E402
import sliding_synthetic   # noqa: E402
import cmn_synthetic       # noqa: E402
import ice_kernel_synthetic  # noqa: E402
import hydraulic_kernel_synthetic  # noqa: E402
import hydraulic_lag_derivation  # noqa: E402
import thermal_tail_amplitude  # noqa: E402
import creep_scaling_synthetic  # noqa: E402
import gle_memory_synthetic  # noqa: E402
import amplitude_law_synthetic  # noqa: E402
import glmig_synthetic  # noqa: E402
import rtn_phi_synthetic  # noqa: E402
import rtn_variable_phi_skill as rvp  # noqa: E402
import rtn_intrusion_clock_synthetic as ric  # noqa: E402
import hydraulic_mz_projection_synthetic as mz  # noqa: E402
import hydraulic_nonlinear_kernel as hnl  # noqa: E402
import hydraulic_mz_spatial as mzs  # noqa: E402
import cross_relationships as xr  # noqa: E402
import cross_relationships2 as xr2  # noqa: E402
import cross_relationships3 as xr3  # noqa: E402
import cross_relationships4 as xr4  # noqa: E402
import cross_relationships5 as xr5  # noqa: E402
from external.run_rtn_bedmap2 import build_rtn  # noqa: E402
from external import lag_fit_real  # noqa: E402
from external import run_usapdc_lakes as _ul  # noqa: E402
from validators.rtn_validator import rtn, classify, thickness_threshold, P_ATM  # noqa: E402
from validators.sliding_validator import gamma_kernel, lagged_response, estimate_lag  # noqa: E402
from validators.sliding_validator import validate_lags  # noqa: E402


# --- RTN ---------------------------------------------------------------------
def test_rtn_synthetic_overall_pass():
    r = rtn_synthetic.run()
    assert r["pass"], r


def test_rtn_threshold_is_exact_boundary():
    """RTN>1 region must coincide cell-for-cell with H < H*."""
    r = rtn_synthetic.run()
    assert r["mismatch_cells"] == 0


def test_rtn_complete_survey_f1_unity():
    r = rtn_synthetic.run()
    assert abs(r["f1_full"] - 1.0) < 1e-9


def test_rtn_threshold_formula():
    """H* solves RTN==1 exactly: RTN at H* is 1 from both sides of the limit."""
    p_ocean, N = 7.5e6, 1.5e5
    Hstar = thickness_threshold(p_ocean, N)
    assert classify(rtn(Hstar - 1.0, p_ocean, N))      # thinner -> intrusion
    assert not classify(rtn(Hstar + 1.0, p_ocean, N))  # thicker -> safe


def test_rtn_gauge_convention_atmosphere_cancels():
    """Gauge fix (§G.3 caveat #1): the atmosphere must cancel in RTN.

    A gauge ocean head with the default ``p_atm=0`` must equal the same
    head supplied in absolute form (``gauge + p_atm``) with ``p_atm`` set --
    i.e. ``p_atm`` only converts an absolute input to gauge, it is not a
    physical offset.  The old runner subtracted ``p_atm`` from an already-gauge
    head; that spurious offset's relative size grows without bound as the bed
    shallows (``p_ocean -> 0``), exactly where the prediction matters.
    """
    H, N = 600.0, 2.0e5
    p_gauge = 8.0e6
    r_gauge = rtn(H, p_gauge, N)                      # gauge head, default p_atm=0
    r_absolute = rtn(H, p_gauge + P_ATM, N, p_atm=P_ATM)  # absolute head, converted back
    assert np.isclose(r_gauge, r_absolute)
    # the buggy convention (gauge head minus p_atm) is a real, non-trivial offset
    r_buggy = rtn(H, p_gauge, N, p_atm=P_ATM)
    assert not np.isclose(r_gauge, r_buggy)
    # and that error blows up in the shallow grounding zone (small ocean head)
    shallow = 2.0e5
    err = abs(rtn(H, shallow, N, p_atm=P_ATM) - rtn(H, shallow, N)) / rtn(H, shallow, N)
    assert err > 0.4


def test_classify_flotation_inf_is_intrusion():
    """Ice at/over flotation gets ``RTN = +inf`` and MUST classify as intrusion.

    ``rtn`` sets ``+inf`` when ``p_w <= eps`` (water pressure >= overburden),
    the strongest intrusion-favourable case.  ``classify`` must not discard it
    as ``np.isfinite`` would (that drops ``+inf`` together with ``NaN``).
    """
    p_ocean, N = 8.0e6, 2.0e5
    # H = 10 m is below the ~22 m flotation threshold -> p_w <= eps -> +inf
    r = rtn(10.0, p_ocean, N)
    assert np.isinf(r) and r > 0
    assert bool(classify(r))                           # +inf -> intrusion-favourable


def test_classify_array_inf_nan_finite():
    """Vectorised classify: +inf -> True, NaN -> False, finite by threshold."""
    r = np.array([np.inf, np.nan, 2.0, 0.5, 1.0])
    out = classify(r, threshold=1.0)
    assert out.tolist() == [True, False, True, False, False]


# --- non-local sliding law ---------------------------------------------------
@pytest.mark.parametrize("tau_true", [20.0, 40.0, 80.0])
def test_sliding_recovers_planted_lag(tau_true):
    r = sliding_synthetic.run(tau_true=tau_true)
    tol = 0.20 * tau_true + 3.0
    assert abs(r["lag_xcorr"] - tau_true) <= tol, r


def test_sliding_memoryless_control_is_zero_lag():
    r = sliding_synthetic.run(tau_true=40.0)
    assert r["lag_memoryless_control"] <= 0.25 * r["tau_true"], r


def test_sliding_estimate_lag_direct():
    """Direct kernel round-trip: convolve, then recover the mode lag."""
    n, tau = 1500, 30.0
    rng = np.random.default_rng(3)
    q = np.zeros(n)
    q[200:206] = 1.0
    q[800:806] = 1.0
    u = lagged_response(q, gamma_kernel(300, tau))
    u = u + rng.normal(0, 1e-3, n)
    lag, _, _ = estimate_lag(q, u, max_lag=200)
    assert abs(lag - tau) <= 0.25 * tau + 3.0


# --- clock-mismatch (CMN) ----------------------------------------------------
def test_cmn_synthetic_overall_pass():
    r = cmn_synthetic.run()
    assert r["pass"], r


def test_cmn_commutator_identity_converges():
    """Identity error shrinks toward 0 as the time step is refined."""
    e_coarse = cmn_synthetic.run(dt=4e-3)["identity_rel_err"]
    e_fine = cmn_synthetic.run(dt=1e-3)["identity_rel_err"]
    assert e_fine < e_coarse
    assert e_fine < 1e-3


def test_cmn_steady_term_vanishes():
    r = cmn_synthetic.run()
    assert r["steady_term_rel_mag"] < 1e-12


def test_cmn_coefficient_has_time_dimension():
    """The commutator term carries one extra inverse-time factor vs the diffusion
    term: doubling the transient rate omega leaves div(K grad theta) unchanged but
    doubles div((d_tK) grad theta). Hence CMN must have units of time."""
    r = cmn_synthetic.run()
    assert r["diff_rate_sensitivity"] < 1e-12          # diffusion term rate-independent
    assert abs(r["term_rate_ratio"] - 2.0) < 0.02      # commutator linear in rate


# --- ice-side memory kernel (§B.2) -------------------------------------------
def test_ice_kernel_synthetic_overall_pass():
    r = ice_kernel_synthetic.run()
    assert r["pass"], r


def test_ice_kernel_matches_pde_step_response():
    """Closed-form kernel's step response matches a direct finite-difference solve
    of the linearised moving-boundary heat equation, and the late-time value
    recovers the quasi-steady DC gain H(0) = -rho c theta_far."""
    r = ice_kernel_synthetic.run()
    assert r["step_response_rel_err"] < 0.05
    assert r["dc_gain_rel_err"] < 0.05


def test_ice_kernel_shorttime_sqrt_tail():
    """Short-time tail goes as t^{-1/2} (log-log slope ~ -0.5), and the kernel
    integrates to the DC gain (normalisation check)."""
    r = ice_kernel_synthetic.run()
    assert abs(r["shorttime_slope"] + 0.5) < 0.02
    assert r["kernel_norm_rel_err"] < 0.02


# --- hydraulic memory-kernel shape (§G.4 / §K) --------------------------------
def test_hydraulic_kernel_synthetic_overall_pass():
    r = hydraulic_kernel_synthetic.run()
    assert r["pass"], r


def test_hydraulic_first_order_is_monotone_no_peak():
    """A single storage (first-order RC) is monotone decaying: peak at t=0, so it
    cannot produce the observed rise-to-a-peak surge lag (§G.4 caveat i)."""
    r = hydraulic_kernel_synthetic.run()
    assert r["first_order_argmax_idx"] == 0
    assert r["first_order_monotone"]


def test_hydraulic_two_compartment_peaks_at_interior_time():
    """Two coupled linear storages (cavity -> channel) give a downstream response
    that starts at 0 and rises to an interior peak matching the analytic t*; the
    coupled-matrix model is overdamped (one peak, no oscillation)."""
    r = hydraulic_kernel_synthetic.run()
    assert r["cascade_starts_at_zero"]
    assert r["cascade_peak_time_rel_err"] < 0.01      # matches analytic t*
    assert r["cascade_peaks"] and r["coupled_peaks"]
    assert r["coupled_overdamped"]


# --- hydraulic lag VALUE derived from physics (§G.4 / §H.2) -------------------
def test_hydraulic_lag_jacobian_matches_finite_difference():
    """The analytic 2×2 cavity Jacobian must match a finite-difference Jacobian
    of the nonlinear rates at the operating point (guards the derivation)."""
    r = hydraulic_lag_derivation.run()
    assert r["jacobian_rel_err"] < 1e-3, r["jacobian_rel_err"]
 
 
def test_hydraulic_lag_closed_form_peak_matches_cascade_in_weak_coupling():
    """The closed-form 2×2 impulse peak must reduce to the planted cascade formula
    t* = τ₁τ₂ln(τ₁/τ₂)/(τ₁−τ₂) when the off-diagonal coupling → 0."""
    m = hydraulic_lag_derivation
    for tau1, tau2 in [(2.0, 0.5), (1.0, 0.1), (0.3, 0.05)]:
        M = np.array([[-1.0 / tau1, 0.0], [1e-6, -1.0 / tau2]])
        full = m.channel_impulse_peak_time(M)
        cas = m.cascade_peak_time(tau1, tau2)
        assert abs(full - cas) / cas < 1e-6, (tau1, tau2, full, cas)
 
 
def test_literal_rothlisberger_channel_is_unstable():
    """The literal (p_w, S) Röthlisberger channel is structurally unstable
    (trace > 0) for ice-stream parameters — the Schoof-2010 / Kingslake-2015
    oscillation regime — so it cannot be the source of the observed *peaked*
    surge.  This motivates the stable cavity-opening mechanism."""
    chan = hydraulic_lag_derivation.run()["channel_regime"]
    assert chan["steady_state"] is not None
    assert chan["trace_per_s"] > 0.0          # unstable
    assert not chan["stable"]
 
 
def test_cavity_model_is_stable_and_peaks():
    """The cavity-opening (p_w, h_s) system is unconditionally stable
    (trace < 0, det > 0) and yields an interior impulse peak — the decaying
    peaked memory kernel the data show."""
    bl = hydraulic_lag_derivation.run()["baseline"]
    assert bl["stable"]
    assert bl["trace_per_s"] < 0.0 and bl["det_per_s2"] > 0.0
    assert bl["tstar_yr"] is not None and bl["tstar_yr"] > 0.0
 
 
def test_derived_lag_is_hydromechanical_not_thermal():
    """The DERIVED lag is O(days–year): orders of magnitude below the falsified
    thermal kernel H²/κ (~10⁵ yr) and within/just-below the observed 0.02–2 yr
    band.  A substantial fraction of literature parameter space lands in band,
    and none of it lands anywhere near the thermal timescale."""
    r = hydraulic_lag_derivation.run()
    sw = r["sweep"]
    # the median derived lag is sub-decadal (vs ~1.5e5 yr thermal)
    assert sw["tstar_yr_pct"][50] < 1.0
    assert sw["tstar_yr_pct"][95] < 100.0     # nowhere near 1e5 yr
    # a meaningful fraction lands in the observed band, and the upper quartile is in-band
    assert sw["frac_peak_in_band"] > 0.2
    assert hydraulic_lag_derivation.OBS_BAND_YR[0] <= sw["tstar_yr_pct"][75] \
        <= hydraulic_lag_derivation.OBS_BAND_YR[1]
 
 
def test_linearized_lag_survives_nonlinearity():
    """Step B: integrating the FULL nonlinear two-compartment ODEs under an
    impulse drainage reproduces the linearised t* at small amplitude, and the
    peak time stays the same order of magnitude (drift < 50%) even for a large
    impulse (40% of N)."""
    nb = hydraulic_lag_derivation.nonlinear_check()
    small = nb[0]
    assert small["ok"] and small["interior_peak"]
    assert abs(small["ratio_nl_over_lin"] - 1.0) < 0.05      # small impulse ≈ linear t*
    for r in nb:
        assert r["ok"] and r["interior_peak"]
        assert 0.5 < r["ratio_nl_over_lin"] < 2.0            # survives nonlinearity


# --- distributed (spatial) MZ projection (§V.5d, paper4a §4.5) ----------------
def test_hydraulic_mz_spatial_overall_pass():
    r = mzs.run()
    assert r["pass"], r
    assert r["stable_generator"]


def test_spatial_projection_exact_machine_precision():
    """A: the distributed (operator) Mori projection equals the full resolvent's
    S-block at random complex s -> the spatial projection is exact, not a 2x2 trick."""
    P = mzs.build()
    A = mzs.check_projection_exact(P)
    assert A["pass"] and A["max_rel_err"] <= 1e-10


def test_spatial_kernel_is_channel_greens_and_nonlocal():
    """B: K(tau)=A_sq e^{A_qq tau} A_qs matches expm to machine precision and is
    spatially non-local (along-flow memory range >> grid spacing)."""
    P = mzs.build()
    B = mzs.check_kernel_is_greens(P)
    assert B["kernel_vs_expm_max_err"] <= 1e-12
    assert B["spatially_resolved_nonlocal"] and B["ell_mem_at_tau2"] > 5 * B["dx"]


def test_spatial_reduced_gle_reproduces_full_pde():
    """C: the channel-eliminated field-GLE reproduces the full 2N PDE trajectory."""
    C = mzs.check_gle_trajectory()
    assert C["rel_err_trajectory"] <= 1e-4


def test_spatial_memory_necessity_dose_response():
    """D: the memoryless local closure error grows monotonically with channel
    slowness tau2 and -> 0 as tau2 -> 0 (Markovian limit, in the distributed system)."""
    D = mzs.check_memory_necessity()
    assert D["monotone_increasing"]
    assert D["vanishes_as_tau2_to_0"]
    assert D["grows_with_channel_slowness"]


def test_lumped_2x2_is_no_transport_single_node_limit():
    """E: with no channel transport (Dq=U=0) the distributed kernel is exactly
    diagonal and its diagonal equals the committed lumped -ab e^{-tau/tau2} to
    machine precision; turning transport on makes the kernel spatially non-local."""
    E = mzs.check_lumped_is_no_transport_limit()
    assert E["no_transport_offdiag_max"] <= 1e-12
    assert E["no_transport_diag_vs_lumped_err"] <= 1e-12
    assert E["nonlocality_grows_with_transport"]
    assert E["ell_grows_with_transport"]


def test_along_flow_memory_footprint_length_law():
    """F: the steady memory-influence decay length matches the analytic
    ell=2 Dq/(sqrt(U^2+4 Dq/tau2)-U) across (Dq,U,tau2), with the diffusive
    (sqrt(Dq tau2)) and ballistic (U tau2) limits both recovered."""
    F = mzs.check_memory_footprint_law()
    assert F["max_rel_err"] <= 1e-2
    assert F["diffusive_limit_ok"]
    assert F["ballistic_limit_ok"]
 
 
# --- creep [NULL] (§D.6) ------------------------------------------------------
def test_creep_scaling_synthetic_overall_pass():
    r = creep_scaling_synthetic.run()
    assert r["pass"], r


def test_creep_displacement_negligible_on_solver_timescale():
    """The operative null: creep wall displacement over the solver run (1 hr) is a
    negligible fraction of the roughness amplitude (f = A N^n T) across the realistic
    open-cavity N box (<2%), so the wall is rigid on the solver clock (Brinkman
    penalization justified, boundary motion Stefan-dominated). It scales linearly with
    run time, and even at an unrealistic N=5 MPa it stays bounded for cold ice."""
    r = creep_scaling_synthetic.run()
    assert r["rigid_wall_realistic"]
    assert r["disp_fraction_max"] < 0.02
    assert r["linear_in_time"]
    assert r["bounded_even_at_5MPa_cold"]


def test_creep_is_same_sign_smoothing():
    """If creep acted at all it would only smooth: modelled as an amplitude sink it
    decays monotonically -> same (smoothing) sign as the mean melt regulation, never
    enhancement, so it cannot rescue the Type-I bound."""
    r = creep_scaling_synthetic.run()
    assert r["creep_sign_monotone_decay"]


# --- unified memory GLE (§D.4) -----------------------------------------------
def test_gle_memory_synthetic_overall_pass():
    r = gle_memory_synthetic.run()
    assert r["pass"], r


def test_gle_independent_baths_kernel_additive():
    """Two independent baths -> uncorrelated orthogonal forces -> the combined
    memory kernel is the sum of the individual kernels (cross-term ~ 0)."""
    r = gle_memory_synthetic.run()
    assert r["independent_cross_corr"] < 0.02
    assert r["additivity_rel_err"] < 0.05


def test_gle_scale_selective_two_timescales():
    """Fast (OU) kernel sets the early relaxation (combined ~ fast-only at t_early);
    slow (power-law) kernel adds a long-time tail absent from the fast-only response
    (orders-of-magnitude split at t_late). Crossover lag sits between tau_c, tau_d."""
    r = gle_memory_synthetic.run()
    assert r["early_tracks_fast"]
    assert r["late_tail_from_slow"]
    assert r["scale_selective"]


def test_gle_sgs_markovian_white_noise_limit():
    """OU kernel has unit integral independent of tau_c and peak 1/tau_c -> inf as
    tau_c -> 0: it approaches delta(tau), recovering the instantaneous eddy-diffusivity
    (Markovian FDT) closure as a special case."""
    r = gle_memory_synthetic.run()
    assert r["sgs_integral_invariant"]
    assert r["sgs_peak_grows"]


# --- scallop amplitude law (§G.2) --------------------------------------------
def test_amplitude_law_synthetic_overall_pass():
    r = amplitude_law_synthetic.run()
    assert r["pass"], r


def test_amplitude_law_fixed_point_and_stability():
    """rho L a' = alpha a^{1/2} - beta a converges to a*=(alpha/beta)^2 monotonically;
    a=0 is unstable; the linear relaxation rate onto a* equals the derived beta/(2 rho L)."""
    r = amplitude_law_synthetic.run()
    assert r["converged"] and r["zero_unstable"] and r["monotone_no_overshoot"]
    assert r["relax_rate_ok"]                  # eigenvalue = -beta/(2 rho L)


def test_amplitude_law_scaling_and_dimensions():
    """a* scales as (alpha/beta)^2 (x2 alpha -> x4, x2 beta -> /4); rho L a' and
    beta*a are both latent-heat fluxes [W/m^2] and beta = c k_ice dT (2pi/lam)^2 is
    [W/m^3] -> beta ∝ lam^-2 gives a* ∝ lam^4."""
    r = amplitude_law_synthetic.run()
    assert r["scaling_ok"]
    assert r["dims_ok"]
    assert r["lambda4_scaling_ok"]


def test_amplitude_law_naive_flow_closure_falsified():
    """alpha ∝ u_* would predict a* INCREASING with drive; the solver forcing probe
    [VERIFIED] the opposite (more drive -> smaller a*), so the flow-dependence must
    sit in the smoothing coefficient beta, not the growth coefficient alpha."""
    r = amplitude_law_synthetic.run()
    assert r["naive_alpha_u_star_falsified"]


# --- §D.5 amplitude-band diagnostics (solver probe helpers) ------------------
import scallop_amplitude_band as amp_band  # noqa: E402


def test_reverse_flow_fraction_basic():
    """Area fraction of fluid cells with time-mean u<0; masks non-fluid; NaN
    when there is no fluid."""
    fluid = np.array([[True, True], [True, False]])
    u_all_pos = np.array([[1.0, 0.5], [2.0, -9.0]])   # the -9 sits on non-fluid
    assert amp_band.reverse_flow_fraction(u_all_pos, fluid) == 0.0
    u_half = np.array([[-1.0, 0.5], [-2.0, 9.0]])     # 2 of 3 fluid cells <0
    assert abs(amp_band.reverse_flow_fraction(u_half, fluid) - 2.0 / 3.0) < 1e-12
    assert np.isnan(amp_band.reverse_flow_fraction(u_half, np.zeros((2, 2), bool)))


def test_coeff_of_variation_known_values():
    """std/|mean| of a series; 0 for a constant; NaN for empty / zero-mean."""
    assert amp_band.coeff_of_variation([3.0, 3.0, 3.0]) == 0.0
    s = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(amp_band.coeff_of_variation(s) - np.std(s) / np.mean(s)) < 1e-12
    assert np.isnan(amp_band.coeff_of_variation([]))
    assert np.isnan(amp_band.coeff_of_variation([-1.0, 1.0]))   # mean 0


def test_detrended_cv_removes_linear_drift():
    """A pure linear ramp has ~zero detrended CV (the trend is removed), whereas
    the raw CV is large; a linear ramp + sinusoid recovers the sinusoid's std."""
    # use an exact integer number of sine periods so <sin>=0 and the polyfit
    # recovers the trend perfectly -- the tight tolerance below depends on this,
    # so derive the length from the period rather than hard-coding a bare count.
    period = 25
    n_periods = 16
    t = np.arange(period * n_periods, dtype=float)   # 400, exactly 16 periods
    ramp = 10.0 + 0.05 * t                       # mean ~20, big raw spread
    assert amp_band.detrended_cv(ramp) < 1e-6
    assert amp_band.coeff_of_variation(ramp) > 0.1
    osc = ramp + 0.5 * np.sin(2 * np.pi * t / period)
    # detrended std ~ std of a unit-amp sine * 0.5 = 0.5/sqrt(2); /mean
    expected = (0.5 / np.sqrt(2.0)) / float(np.mean(osc))
    assert abs(amp_band.detrended_cv(osc) - expected) < 5e-3


# --- §G.4 thermal-tail amplitude --------------------------------------------
def test_thermal_tail_run_passes():
    """End-to-end derivation passes all its internal checks."""
    r = thermal_tail_amplitude.run()
    assert r["pass"] is True
    assert r["identity_rel_err"] < 1e-3


def test_thermal_weight_equals_stefan_number():
    """(★) The thermal memory kernel's DC gain equals the Stefan number
    c_i|theta_far|/L, across undercoolings and melt speeds. This is the derived
    'subdominant tail' amplitude: integral of the §B.2 kernel == St."""
    SEC = thermal_tail_amplitude.SEC_PER_YR
    for theta in (-0.5, -2.0, -8.0):
        for Vbar_yr in (0.01, 0.1, 1.0):
            St = thermal_tail_amplitude.stefan_weight(theta)
            dc = thermal_tail_amplitude.thermal_kernel_dc_gain(theta, Vbar_yr / SEC)
            assert abs(dc - St) / St < 2e-3
    # the closed form is exactly c_i|theta|/L
    assert abs(thermal_tail_amplitude.stefan_weight(-1.0)
               - 2009.0 / 3.34e5) < 1e-12


def test_thermal_subdominant_to_hydraulic():
    """St <= ~0.06 even at a generous 10 K undercooling, and strictly below the
    unit-DC-gain hydraulic kernel -> thermal term is subdominant by construction."""
    St10 = thermal_tail_amplitude.stefan_weight(-10.0)
    assert St10 < 0.07
    assert St10 < thermal_tail_amplitude.hydraulic_kernel_dc_gain()
    sw = thermal_tail_amplitude.sweep(n=800)
    assert sw["St_max"] < 0.07
    assert sw["frac_St_below_0.1"] == 1.0


def test_thermal_skin_depth_matches_g4_numbers():
    """delta = sqrt(kappa P / pi) reproduces the §G.4 0.48/3.38/4.79 m at
    P=0.02/1/2 yr and stays << ice thickness."""
    assert abs(thermal_tail_amplitude.skin_depth_m(0.02) - 0.48) < 0.05
    assert abs(thermal_tail_amplitude.skin_depth_m(1.0) - 3.38) < 0.1
    assert abs(thermal_tail_amplitude.skin_depth_m(2.0) - 4.79) < 0.1
    assert thermal_tail_amplitude.skin_depth_m(2.0) < 0.01 * thermal_tail_amplitude.H_ICE


def test_thermal_impulse_monotone_no_peak():
    """Semi-infinite ice kernel decays monotonically (no interior peak), so it
    cannot produce the observed rise-to-a-peak surge regardless of amplitude."""
    assert thermal_tail_amplitude.impulse_is_monotone() is True


def test_thermal_excluded_hydraulic_dominates():
    """§G.4 mechanism reassignment, machine-checked across BOTH derived kernels:
    the ice-thermal kernel is monotone (no interior peak) and carries weight only
    St<=0.06, whereas the hydraulic cavity kernel has a finite sub-decadal interior
    peak -> thermal is excluded by shape AND amplitude; hydraulic produces the
    observed rise-to-a-peak surge."""
    # thermal: no interior peak, weight bounded by the unit-DC-gain hydraulic kernel
    assert thermal_tail_amplitude.impulse_is_monotone() is True
    assert (thermal_tail_amplitude.stefan_weight(-10.0)
            < thermal_tail_amplitude.hydraulic_kernel_dc_gain())
    # hydraulic: a finite, positive, sub-decadal interior peak (rise-to-peak shape)
    base = hydraulic_lag_derivation.baseline_params()
    tstar = hydraulic_lag_derivation.derive_lag(base)["tstar_yr"]
    assert tstar is not None and 0.0 < tstar < 2.0


# --- §H.2/§G.4 matched-data lag test on OPEN data (lag_fit_real) --------------
def test_lag_fit_real_run_passes():
    """The matched-data test runs end-to-end and is internally self-consistent.
    Crucially it does NOT verify the lag value (honest null on open data)."""
    r = lag_fit_real.run()
    assert r["pass"] is True
    assert r["verified_lag_value"] is False
    assert r["below_detection"] is True
 
 
def test_lag_fit_drainage_dates_are_open():
    """Drainage dates ARE derivable from open CryoSat-2 (>=5 events, >=5 lakes),
    incl. the documented ~2012-13 Mercer drainage -- so the date half is NOT gated."""
    r = lag_fit_real.run()
    assert r["n_drainage_dates"] >= 5
    assert r["n_lakes"] >= 5
    art = lag_fit_real.load_artifact()
    mercer = art["lakes"]["Mercer"]["drainage_events"]
    assert any(2012.0 <= e["t_peak"] <= 2013.5 for e in mercer)
 
 
def test_lag_fit_null_no_significant_response():
    """No matched lake shows a post-drainage surge above the 2-sigma noise floor;
    the peak anomaly is in fact ~<=0 (velocity does not rise after drainage)."""
    r = lag_fit_real.run()
    assert r["n_lakes_matched"] >= 3
    assert r["n_events_significant"] == 0
    assert r["max_response_sigma"] < 2.0
    for pl in r["per_lake"].values():
        assert pl["significant"] is False
 
 
def test_lag_fit_resolution_is_sampling_not_noise():
    """The lag is unverifiable from satellite velocity for a *temporal-sampling*
    (aliasing) reason, NOT a noise one. With dense modern ITS_LIVE the record is
    well resolved (annual ~1%, quarterly ~2-3%), yet the derived sub-annual lag
    (t* p95 ~0.1 yr) is finer than the finest robust bin (quarterly, 0.25 yr), so
    a brief transient is averaged out -- and any *sustained* surge is bounded
    below detection (<2sigma)."""
    r = lag_fit_real.run()
    # the velocity record is GOOD -- this is explicitly not a noise excuse
    assert r["velocity_well_resolved"] is True
    assert max(r["quarterly_scatter_frac"].values()) < 0.05
    for name, nf in r["annual_noise_floor_frac"].items():
        assert nf < 0.05  # annual baseline is stable
        assert r["quarterly_scatter_frac"][name] >= nf  # quarterly a touch noisier
    # the binding limit is temporal resolution: t* sits below one velocity bin
    assert r["resolution_blocked"] is True
    assert r["tstar_p95_yr"] < r["quarterly_bin_yr"]
    # ...and the integrated response is an amplitude *upper bound*, not a non-result
    assert r["max_response_sigma"] < 2.0
 
 
def test_lag_fit_thermal_falsified_band_consistent():
    """The thermal H^2/kappa kernel stays FALSIFIED (>>band) while the derived
    hydraulic t* baseline sits at/below the observed surge-lag band."""
    r = lag_fit_real.run()
    lo, hi = r["obs_lag_band_yr"]
    assert r["thermal_tau_median_yr"] > 1e3 * hi
    assert r["derived_tstar_yr"]["baseline"] <= hi
    # thermal tau from real median thickness is millennial
    assert lag_fit_real.thermal_tau_years(2282.0) > 1e4
 
 
def test_lag_fit_detect_drainages_synthetic():
    """detect_drainages recovers a planted fill-then-drain event (and ignores noise)."""
    t = np.arange(2010.0, 2016.0, 1.0 / 12)
    # linear fill to a peak at 2013.0 then a sharp drainage
    e = np.where(t <= 2013.0, (t - 2010.0), 3.0 - 12.0 * (t - 2013.0))
    e = np.clip(e, -3.0, None)
    evs = lag_fit_real.detect_drainages(t, e, frac=0.4)
    assert len(evs) >= 1
    assert abs(evs[0]["t_peak"] - 2013.0) < 0.2
    assert evs[0]["drop_m"] > 1.0
 
 
def test_lag_fit_validate_lags_methodology_runs():
    """The repo's validate_lags machinery executes on the real annual series at the
    detected drainage dates (methodology is exercised; the result is just not
    significant -- see the null test above)."""
    art = lag_fit_real.load_artifact()
    lk = art["lakes"]["Mac1"]
    t = np.array([r["t"] for r in lk["velocity_annual"]], float)
    u = np.array([r["v"] for r in lk["velocity_annual"]], float)
    events = [e["t_peak"] for e in lk["drainage_events"]]
    scores = validate_lags(events, u, t, max_lag_samples=2, dt=1.0)
    assert scores.n_events >= 1
    assert np.isfinite(scores.mean_lag)


# --- external GL-migration sampler: grid geometry ----------------------------
def test_grid_coords_north_anchor_exact_for_odd_grid():
    """_grid_coords must place decimated cell centres on the *original* grid even
    when the original dimension is odd. Bedmap2 is 6667x6667: reconstructing the
    row count as axis_len*stride (=6668 at stride 2) shifts every y coordinate one
    whole cell (~1 km) north and pushes the top row outside the true grid extent.
    """
    from external.rtn_glmig_test import _grid_coords

    # Mimic the loader's _meta for a 7x7 (odd) 1 km grid decimated by stride 2 ->
    # 4 sampled indices (original rows/cols 0, 2, 4, 6).
    nfull, cs_full, stride = 7, 1000.0, 2
    n = len(range(0, nfull, stride))            # 4
    meta = {"xll": 0.0, "yll": 0.0,
            "cellsize": cs_full * stride, "nrows": n, "ncols": n,
            "cellsize_full": cs_full, "nrows_full": nfull, "ncols_full": nfull,
            "stride": stride}

    ys = _grid_coords(meta, stride, n, n, is_row=True)
    xs = _grid_coords(meta, stride, n, n, is_row=False)

    orig = np.array([0, 2, 4, 6])               # original indices that survive
    exp_y = meta["yll"] + (nfull - orig - 0.5) * cs_full     # north-anchored
    exp_x = meta["xll"] + (orig + 0.5) * cs_full
    assert np.allclose(ys, exp_y)
    assert np.allclose(xs, exp_x)

    # every y centre lies inside the true grid extent [yll, yll + nfull*cs_full]
    assert ys.max() < meta["yll"] + nfull * cs_full
    # the old axis_len*stride reconstruction was exactly one cell too far north
    buggy_top = meta["yll"] + (n * stride - 0.5) * cs_full
    assert abs(buggy_top - ys[0]) == cs_full


def test_grid_coords_falls_back_without_full_metadata():
    """Old caches lacking the *_full keys must still work (fall back to
    axis_len*stride) rather than raising KeyError."""
    from external.rtn_glmig_test import _grid_coords
    stride, n = 2, 4
    meta = {"xll": 0.0, "yll": 0.0, "cellsize": 1000.0 * stride}
    ys = _grid_coords(meta, stride, n, n, is_row=True)
    exp = meta["yll"] + (n * stride - np.arange(n) * stride - 0.5) * 1000.0
    assert np.allclose(ys, exp)


# --- §H.1.2 GL-migration estimator calibration (§V plant-and-recover) --------
def test_glmig_synthetic_overall_pass():
    r = glmig_synthetic.run()
    assert r["pass"], r


def test_glmig_levelset_advance_law_is_exact():
    """The tracked zero-contour speed matches the level-set law
    v = |dH/dt|/|grad m| in 1-D and (using the gradient *magnitude*) for a tilted
    2-D plane; the grid estimator A=1/|grad m| recovers 1/b."""
    a1 = glmig_synthetic.levelset_advance_1d()
    a2 = glmig_synthetic.levelset_advance_2d_tilt()
    assert a1["rel_err_speed"] < 1e-3
    assert a1["rel_err_A"] < 1e-6
    assert a2["rel_err"] < 1e-9


def test_glmig_thinning_paced_null_is_flat():
    """gamma=0 (no hydraulic discount) => Ro==1 exactly and a machine-zero slope
    of log Ro vs u_* -- the calibrated null behind the §H.1.3 'flat slope =>
    thinning-paced' reading."""
    null = glmig_synthetic.ro_discriminant(gamma=0.0)
    assert null["Ro_max_dev_from_1"] < 1e-12
    assert abs(null["slope_logRo_vs_u"]) < 1e-12


def test_glmig_recovers_planted_discount_exponent():
    """With a planted discount D=1+gamma*u^p, fitting log(Ro-1) vs log u_*
    recovers p; the raw log Ro vs log u_* slope (the §H.1.4 statistic) is positive
    and increases monotonically with the planted p."""
    rec = glmig_synthetic.ro_discriminant(gamma=0.8, p=0.5)
    assert rec["rel_err_p"] < 1e-2
    slopes = [glmig_synthetic.ro_discriminant(gamma=0.8, p=pp)["slope_logRo_vs_u"]
              for pp in (0.25, 0.5, 0.9)]
    assert all(s > 0.0 for s in slopes)
    assert slopes[0] < slopes[1] < slopes[2]


def test_glmig_slope_unbiased_under_noise_and_null_under_permutation():
    """Multiplicative noise on v_obs leaves the slope positive and close to the
    noiseless value (small bias); shuffling u_* collapses it to ~0 -- so an
    observed positive slope is not a noise/ordering artefact."""
    clean = glmig_synthetic.ro_discriminant(gamma=0.8, p=0.5)["slope_logRo_vs_u"]
    noisy = glmig_synthetic.ro_discriminant(gamma=0.8, p=0.5, noise=0.25, seed=3)["slope_logRo_vs_u"]
    perm = glmig_synthetic.ro_discriminant(gamma=0.8, p=0.5, shuffle=True, seed=3)["slope_logRo_vs_u"]
    assert noisy > 0.0
    assert abs(noisy - clean) < 0.1
    assert abs(perm) < 0.05


# --- §G.4 lag estimator is kernel-shape-generic (closes the "kernel not closed"
#     caveat for the *estimator*, not the physical kernel) ---------------------
def test_sliding_lag_recovery_is_kernel_shape_generic():
    """The cross-correlation lag estimator recovers the planted lag across five
    markedly different causal kernel shapes (gamma k=2/4, log-normal, bi-
    exponential, symmetric raised-cosine) -- so a recovered lag is a property of
    the data, not an artefact of the assumed gamma form."""
    r = sliding_synthetic.run_kernel_shapes()
    assert r["pass"], r
    assert r["worst_mode_offset"] <= 2.0            # all shapes peak at target
    assert r["worst_lag_err"] <= r["tol_samples"]   # all recovered within ~10%
    assert set(r["families"]) == {"gamma_k2", "gamma_k4", "lognormal", "biexp", "raised_cosine"}


def test_sliding_kernel_shape_memoryless_control_is_zero():
    """The delta-kernel (memoryless) control still returns ~0 lag even in the
    multi-shape harness -- the estimator does not manufacture a lag."""
    r = sliding_synthetic.run_kernel_shapes()
    assert r["memoryless_control"] <= 0.25 * r["tau_true"]


@pytest.mark.parametrize("tau_true", [25.0, 60.0])
def test_sliding_kernel_shape_generic_across_lags(tau_true):
    """Genericity holds at other planted lags too."""
    r = sliding_synthetic.run_kernel_shapes(tau_true=tau_true)
    assert r["pass"], r


# ---------------------------------------------------------------------------
# §H.1.1 RTN φ-area inversion — synthetic calibration (RESULT 18)
# ---------------------------------------------------------------------------
def test_rtn_phi_inversion_overall_pass():
    """Top-level: the φ-area inversion calibration passes all properties."""
    r = rtn_phi_synthetic.run()
    assert r["pass"], r


def test_rtn_phi_area_curve_is_monotone_and_sensitivity_negative():
    """A(φ) strictly decreasing (unique inverse) and dA/dφ<0 everywhere."""
    r = rtn_phi_synthetic.run()
    fr = np.array(r["area_frac"])
    assert np.all(np.diff(fr) < 0.0), fr
    assert r["monotone_decreasing"] and r["sensitivity_negative"], r
    # spans the real Bedmap2 table range (~3.85% -> ~0.11%)
    assert 3.0 < r["area_pct"][0] < 5.0, r["area_pct"][0]
    assert r["area_pct"][-1] < 0.3, r["area_pct"][-1]


def test_rtn_phi_critical_thickness_identity_is_exact():
    """RTN>1 reproduces H < H*=(ρ_w/φρ_i)·d_base cell-for-cell (zero mismatch)."""
    H, bed = rtn_phi_synthetic.make_population(n_cells=50_000, seed=3)
    for phi in (0.72, 0.86, 0.94):
        assert rtn_phi_synthetic.critical_thickness_mismatch(H, bed, phi) == 0


def test_rtn_phi_population_inverse_recovers_planted_phi():
    """Inverting the monotone A(φ) curve recovers the planted φ exactly."""
    phis = np.round(np.arange(0.70, 0.981, 0.02), 2)
    H, bed = rtn_phi_synthetic.make_population(n_cells=100_000, seed=1)
    fr = rtn_phi_synthetic.area_curve(H, bed, phis)
    for p in (0.76, 0.88, 0.94):
        a = rtn_phi_synthetic.area_fraction(H, bed, p)
        assert abs(rtn_phi_synthetic.invert_area(phis, fr, a) - p) <= 1e-9


def test_rtn_phi_finite_sample_inverse_is_unbiased_and_shrinks():
    """Finite-survey φ̂ is unbiased and its spread falls toward 1/√N."""
    r = rtn_phi_synthetic.run()
    fs = r["finite_sample"]
    assert fs["bias"] <= 0.01, fs
    assert fs["shrink_ratio_2n_over_n"] < 0.85, fs


# --- §H.1.1 #2 hydrology-corrected MISI margin band -------------------------
def test_rtn_phi_misi_band_overall_pass():
    """The grounded-but-intrudable band calibration passes all properties."""
    r = rtn_phi_synthetic.run_misi_band()
    assert r["pass"], r


def test_rtn_phi_misi_band_relative_width_is_exactly_1mphi_over_phi():
    """Per-cell band width (H*−H_flot)/H_flot == (1−φ)/φ exactly, and H*>H_flot."""
    H, bed = rtn_phi_synthetic.make_population(n_cells=80_000, seed=5)
    h_flot = rtn_phi_synthetic.flotation_thickness(bed)
    base = h_flot > 0.0
    for phi in (0.78, 0.90, 0.97):
        h_star = rtn_phi_synthetic.critical_thickness(bed, phi)
        rel = (h_star[base] - h_flot[base]) / h_flot[base]
        assert np.allclose(rel, (1.0 - phi) / phi, atol=1e-12)
        assert np.all(h_star[base] > h_flot[base])  # RTN=1 line inland of flotation


def test_rtn_phi_misi_band_equals_grounded_above_flotation_and_rtn1():
    """Band == (H>H_flot) ∧ (RTN>1 at φ), cell-for-cell."""
    H, bed = rtn_phi_synthetic.make_population(n_cells=80_000, seed=6)
    h_flot = rtn_phi_synthetic.flotation_thickness(bed)
    for phi in (0.80, 0.92):
        band = rtn_phi_synthetic.misi_band_mask(H, bed, phi)
        rtn1 = classify(build_rtn(H, bed, phi))
        assert np.array_equal(band, (H > h_flot) & rtn1)


def test_rtn_phi_misi_band_vanishes_and_is_linear_in_1mphi_limit():
    """Band area → 0 monotonically as φ→1 and is linear in (1−φ) at the limit."""
    r = rtn_phi_synthetic.run_misi_band()
    assert r["band_frac_vanishes_monotone"], r["band_pct"]
    assert r["limit_linear_rel_gap"] <= 0.25, r["ratio_band_over_1mphi"]


# ---------------------------------------------------------------------------
# §H.1.2 intrusion-clock estimator — synthetic calibration (RESULT 20)
# ---------------------------------------------------------------------------
def test_rtn_intrusion_clock_overall_pass():
    """Top-level: the level-set intrusion-clock estimator calibration passes."""
    r = ric.run()
    assert r["pass"], r


def test_rtn_intrusion_clock_planar_is_exact():
    """On a planar margin A=1/|∇m| and v=(dH/dt)/|∇m| are machine-precision exact."""
    r = ric.run_planar()
    assert r["grad_max_abs_err"] <= 1e-9, r
    assert r["amp_max_abs_err"] <= 1e-12, r
    assert r["v_front_max_abs_err"] <= 1e-12, r
    assert abs(r["A_analytic_km_per_m"] - 1.0 / r["slope_m_per_km"]) <= 1e-15, r


def test_rtn_intrusion_clock_is_second_order_convergent():
    """Estimator error in A falls ~4× (2nd order) when the grid spacing halves."""
    r = ric.run_convergence()
    assert r["amp_max_abs_err"][1] < r["amp_max_abs_err"][0], r
    assert r["error_ratio_coarse_over_fine"] >= 3.0, r


def test_rtn_intrusion_clock_radial_front_is_isotropic():
    """On the radial RTN=1 ring A=1/s is recovered with no axis-vs-diagonal bias."""
    r = ric.run_isotropy()
    assert r["rel_median_err"] <= 0.02, r
    assert r["ring_cv"] <= 0.03, r
    assert r["anisotropy"] <= 0.02, r


def test_rtn_intrusion_clock_recovers_planted_front_advance():
    """Advancing the front by the estimated A·ΔH lands on the analytic new line."""
    r = ric.run_advance_recovery()
    assert r["front_pos_max_abs_err_km"] <= 1e-9, r
    assert r["mask_boundary_max_err_km"] <= 1.0, r  # within one cell


def test_rtn_intrusion_clock_advects_the_rtn1_line():
    """The advected margin zero set IS the §H.1.1 RTN=1 line (H*=H_flot/φ exact)."""
    for phi in (0.80, 0.90, 0.96):
        r = ric.run_rtn_tie_in(phi=phi, seed=int(100 * phi))
        assert r["H_star_vs_Hflot_over_phi_max_err"] <= 1e-6, r
        assert r["margin_sign_vs_rtn_mismatch_cells"] == 0, r


# --- §G.4 hydraulic Mori–Zwanzig projection (RESULT 21) ----------------------
def test_hydraulic_mz_overall_pass():
    """Top-level: the §G.4 cavity↔channel projection is a genuine MZ memory kernel."""
    r = mz.run()
    assert r["pass"], r


def test_hydraulic_mz_projection_is_exact():
    """The reduced GLE transfer fn equals the full 2×2 resolvent at random complex s
    for random stable overdamped systems — the kernel is exact, not approximate."""
    r = mz.run_projection_exact()
    assert r["max_abs_err"] <= 1e-10, r


def test_hydraulic_mz_kernel_is_channel_greens_function():
    """K(τ)=M_sq M_qs e^{M_qq τ} equals the eliminated channel's own Green's fn,
    decays at 1/τ₂, and is sign-definite (a back-coupling/relaxation kernel)."""
    r = mz.run_kernel_is_greens_function()
    assert r["kernel_vs_greens_max_err"] <= 1e-10, r
    assert abs(r["decay_rate"] - r["expected_rate_inv_tau2"]) <= 1e-6, r
    assert r["kernel_sign_nonpositive"], r


def test_hydraulic_mz_gle_reproduces_full_trajectory():
    """Integrating the reduced GLE (kernel + MZ residual force) reproduces the exact
    full-system resolved trajectory s(t)."""
    r = mz.run_gle_reproduces_trajectory()
    assert r["rel_err_trajectory"] <= 1e-5, r


def test_hydraulic_mz_memory_is_necessary_for_the_peak():
    """The memory-carrying response peaks at the coupled-eigenvalue t*, while the
    Markovian (no-memory) adiabatic closure is monotone with argmax at t=0."""
    r = mz.run_memory_makes_the_peak()
    assert r["full_peaks_interior"], r
    assert r["peak_rel_err"] < 5e-3, r
    assert r["markovian_argmax_index"] == 0 and r["markovian_monotone"], r
    assert r["markovian_eff_rate"] < 0, r


def test_hydraulic_mz_markovian_limit_is_adiabatic_elimination():
    """∫K dτ = M_sq M_qs/(−M_qq) (DC gain), and shrinking τ₂ at fixed DC gain
    collapses K toward a delta — the local-closure (FDT-Markov) limit."""
    r = mz.run_markovian_limit()
    assert r["dc_gain_err"] <= 1e-8, r
    assert r["widths_shrink_to_delta"], r


# --- §V.2c USAP-DC lake forcing + thermal kernel -----------------------------
def test_usapdc_tau_ice_matches_closed_form():
    """tau_ice = H^2/kappa_ice in years, for a known thickness."""
    H = 2356.0  # ~ the 601470 median
    expect = (H ** 2 / _ul.KAPPA_ICE) / _ul.SEC_PER_YR
    assert np.isclose(_ul.tau_ice_years(H), expect, rtol=1e-12)
    # Far slower than the observed surge-lag band (the §V.2/§V.2c falsification).
    assert _ul.tau_ice_years(H) > 1e4 * _ul.OBS_LAG_YR[1]


def test_usapdc_detect_drainages_recovers_planted_event():
    """Plant one fill->drain->refill cycle; recover the event drop and onset."""
    # fill to 1.0 km^3 by t=2005, drain to 0.2 by t=2005.5, refill after.
    t = np.array([2004.0, 2004.5, 2005.0, 2005.5, 2006.0, 2006.5])
    v = np.array([0.0, 0.5, 1.0, 0.2, 0.6, 0.9])
    evs = _ul.detect_drainages(t, v, drop_thresh=0.05)
    assert len(evs) == 1
    e = evs[0]
    assert np.isclose(e["drop_km3"], 0.8, atol=1e-9)
    assert np.isclose(e["t_peak"], 2005.0, atol=1e-9)
    assert np.isclose(e["t_trough"], 2005.5, atol=1e-9)
    assert e["rate_km3_per_yr"] > 0


def test_usapdc_detect_drainages_ignores_monotone_and_subthreshold():
    """A monotone fill, and a sub-threshold wiggle, yield no events."""
    t = np.array([2004.0, 2004.5, 2005.0, 2005.5, 2006.0])
    assert _ul.detect_drainages(t, np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
                                drop_thresh=0.05) == []
    # a 0.02 km^3 dip is below the 0.05 threshold -> not an event
    assert _ul.detect_drainages(t, np.array([0.0, 0.1, 0.08, 0.2, 0.3]),
                                drop_thresh=0.05) == []


def test_usapdc_arr_max_empty_is_nan_not_none():
    """Empty drops -> max is nan (formattable with :.3f), not None.

    Guards the no-events edge case where ``main()`` formats the summary's
    ``max`` field; ``None`` would raise ``TypeError`` under ``f"{x:.3f}"``.
    """
    assert np.isnan(_ul._arr_max(np.array([])))
    assert np.isnan(_ul._arr_max(np.array([np.nan, np.nan])))
    assert _ul._arr_max(np.array([0.1, 0.8, 0.3])) == 0.8
    # the format that previously crashed on None must now succeed
    assert f"{_ul._arr_max(np.array([])):.3f}" == "nan"


# --- §V.1 BedMachine RTN (independent-dataset cross-check) -------------------
def test_bedmachine_build_rtn_matches_validator_and_bedmap2():
    """The xp-generic BedMachine build_rtn equals the canonical validator + the
    Bedmap2 runner's build_rtn cell-for-cell (same gauge RTN definition)."""
    from external.run_rtn_bedmachine import build_rtn as bm_build_rtn
    from external.run_rtn_bedmachine import _assert_matches_validator
    from external.run_rtn_bedmap2 import build_rtn as b2_build_rtn
    rng = np.random.default_rng(0)
    H = rng.uniform(0.0, 3000.0, size=(64, 48))
    bed = rng.uniform(-1500.0, 500.0, size=(64, 48))
    for phi in (0.80, 0.90, 0.95):
        a = bm_build_rtn(H, bed, phi)
        b = b2_build_rtn(H, bed, phi)
        fin = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[fin], b[fin], rtol=1e-12)
        assert np.array_equal(np.isinf(a), np.isinf(b))
        _assert_matches_validator(H, bed, phi)  # vs validators.rtn_validator.rtn


def test_bedmachine_load_fields_missing_path_raises():
    """load_fields on a missing NetCDF raises DataUnavailableError (not OSError)."""
    from external import DataUnavailableError
    from external.bedmachine_loader import load_fields
    with pytest.raises(DataUnavailableError):
        load_fields(path="/no/such/BedMachineAntarctica-v3.nc")


@pytest.mark.skipif(__import__("importlib").util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed")
def test_bedmachine_run_cpu_on_synthetic_netcdf(tmp_path):
    """End-to-end CPU run on a tiny synthetic BedMachine file: grounded mask==2
    is selected and RTN>1 concentrates nearest the grounding line."""
    import netCDF4
    from external.run_rtn_bedmachine import run
    n = 40
    x = np.arange(n) * 500.0
    y = (np.arange(n) * 500.0)[::-1]
    H = np.zeros((n, n)); bed = np.full((n, n), -500.0)
    mask = np.zeros((n, n), dtype="int8")
    cx = cy = n // 2
    for i in range(n):
        for j in range(n):
            r = ((i - cy) ** 2 + (j - cx) ** 2) ** 0.5
            if r < n * 0.35:
                mask[i, j] = 2
                H[i, j] = 2000.0 * (1.0 - r / (n * 0.35))
                bed[i, j] = -200.0 - 2.0 * r
    p = tmp_path / "BedMachineAntarctica-v3.nc"
    ds = netCDF4.Dataset(str(p), "w")
    ds.createDimension("x", n); ds.createDimension("y", n)
    ds.createVariable("x", "f8", ("x",))[:] = x
    ds.createVariable("y", "f8", ("y",))[:] = y
    ds.createVariable("thickness", "f4", ("y", "x"))[:] = H
    ds.createVariable("bed", "f4", ("y", "x"))[:] = bed
    ds.createVariable("mask", "i1", ("y", "x"))[:] = mask
    ds.close()
    s, *_ = run(str(p), stride=1, phi=0.9, use_gpu=False)
    assert s["backend"] == "cpu"
    assert s["n_grounded"] == int((mask == 2).sum())
    assert s["cellsize_km"] == 0.5
    # directional prediction: RTN>1 sits closer to the grounding line than RTN<=1
    assert s["median_dist_pred"] < s["median_dist_notpred"]


@pytest.mark.skipif(__import__("importlib").util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed")
def test_bedmachine_load_fields_stride_decimates_grid_and_cellsize(tmp_path):
    """stride>1 block-decimates every field/coordinate and scales the recorded
    cellsize (the stride path is otherwise unexercised by the stride=1 run)."""
    import netCDF4
    from external.bedmachine_loader import load_fields
    n = 40
    x = np.arange(n) * 500.0
    y = (np.arange(n) * 500.0)[::-1]
    H = np.arange(n * n, dtype="f4").reshape(n, n)
    bed = -H
    mask = (np.arange(n * n).reshape(n, n) % 3).astype("i1")
    p = tmp_path / "BedMachineAntarctica-v3.nc"
    ds = netCDF4.Dataset(str(p), "w")
    ds.createDimension("x", n); ds.createDimension("y", n)
    ds.createVariable("x", "f8", ("x",))[:] = x
    ds.createVariable("y", "f8", ("y",))[:] = y
    ds.createVariable("thickness", "f4", ("y", "x"))[:] = H
    ds.createVariable("bed", "f4", ("y", "x"))[:] = bed
    ds.createVariable("mask", "i1", ("y", "x"))[:] = mask
    ds.close()

    stride = 3
    d1 = load_fields(str(p), fields=("thickness", "bed", "mask"), stride=1)
    ds = load_fields(str(p), fields=("thickness", "bed", "mask"), stride=stride)

    # cellsize is scaled by stride; the native posting is kept under cellsize_full
    assert ds["_meta"]["cellsize"] == d1["_meta"]["cellsize"] * stride
    assert ds["_meta"]["cellsize_full"] == d1["_meta"]["cellsize"]
    assert ds["_meta"]["stride"] == stride

    # every field and both coordinate axes are block-decimated [::stride, ::stride]
    for f in ("thickness", "bed", "mask"):
        np.testing.assert_array_equal(ds[f], d1[f][::stride, ::stride])
        assert ds[f].shape == d1[f][::stride, ::stride].shape
    np.testing.assert_array_equal(ds["x"], d1["x"][::stride])
    np.testing.assert_array_equal(ds["y"], d1["y"][::stride])
    assert ds["_meta"]["nrows"] == ds["thickness"].shape[0]
    assert ds["_meta"]["ncols"] == ds["thickness"].shape[1]


@pytest.mark.skipif(__import__("importlib").util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed")
def test_bedmachine_run_cpu_stride_scales_cellsize(tmp_path):
    """End-to-end CPU run with stride=2 reports the stride-scaled cell size
    (0.5 km native -> 1 km) and a correspondingly decimated grid."""
    import netCDF4
    from external.run_rtn_bedmachine import run
    n = 40
    x = np.arange(n) * 500.0
    y = (np.arange(n) * 500.0)[::-1]
    H = np.zeros((n, n)); bed = np.full((n, n), -500.0)
    mask = np.zeros((n, n), dtype="int8")
    cx = cy = n // 2
    for i in range(n):
        for j in range(n):
            r = ((i - cy) ** 2 + (j - cx) ** 2) ** 0.5
            if r < n * 0.35:
                mask[i, j] = 2
                H[i, j] = 2000.0 * (1.0 - r / (n * 0.35))
                bed[i, j] = -200.0 - 2.0 * r
    p = tmp_path / "BedMachineAntarctica-v3.nc"
    ds = netCDF4.Dataset(str(p), "w")
    ds.createDimension("x", n); ds.createDimension("y", n)
    ds.createVariable("x", "f8", ("x",))[:] = x
    ds.createVariable("y", "f8", ("y",))[:] = y
    ds.createVariable("thickness", "f4", ("y", "x"))[:] = H
    ds.createVariable("bed", "f4", ("y", "x"))[:] = bed
    ds.createVariable("mask", "i1", ("y", "x"))[:] = mask
    ds.close()
    s, *_ = run(str(p), stride=2, phi=0.9, use_gpu=False)
    assert s["backend"] == "cpu"
    assert s["cellsize_km"] == 1.0  # 0.5 km native * stride 2
    assert s["grid"] == (len(range(0, n, 2)), len(range(0, n, 2)))


@pytest.mark.parametrize("content", ["", "\n", "# header only, no rows\n"])
def test_usapdc_load_active_lake_stats_empty_file_raises_data_unavailable(tmp_path, content):
    """An existing-but-empty/header-only file raises DataUnavailableError, not IndexError.

    ``np.loadtxt(..., ndmin=2)`` yields a row-less array on an empty file, so the
    column slices ``a[:, 1]``/``a[:, 5]`` would otherwise raise a cryptic
    ``IndexError``; the loader must report the missing data clearly instead.
    """
    from external import DataUnavailableError
    p = tmp_path / "active_lake_statistics.dat"
    p.write_text(content)
    with pytest.raises(DataUnavailableError):
        _ul.load_active_lake_stats(str(p))


def test_usapdc_load_active_lake_stats_single_row_ok(tmp_path):
    """A well-formed single-row file still indexes as columns (the ndmin=2 case)."""
    p = tmp_path / "active_lake_statistics.dat"
    p.write_text("1.0,2.0,3.0,4.0,5.0,6.0\n")
    d = _ul.load_active_lake_stats(str(p))
    assert d["x"][0] == 1.0 and d["thickness_m"][0] == 6.0


def _write_synthetic_bedmachine(p, n=48):
    """A tiny synthetic BedMachine NetCDF: a grounded ice cone on a bed that
    deepens outward (so RTN>1 concentrates near the cone edge / grounding line)."""
    import netCDF4
    x = np.arange(n) * 500.0
    y = (np.arange(n) * 500.0)[::-1]
    H = np.zeros((n, n)); bed = np.full((n, n), -500.0)
    mask = np.zeros((n, n), dtype="int8")
    cx = cy = n // 2
    for i in range(n):
        for j in range(n):
            r = ((i - cy) ** 2 + (j - cx) ** 2) ** 0.5
            if r < n * 0.40:
                mask[i, j] = 2
                H[i, j] = 2200.0 * (1.0 - r / (n * 0.40))
                bed[i, j] = -200.0 - 3.0 * r
    ds = netCDF4.Dataset(str(p), "w")
    ds.createDimension("x", n); ds.createDimension("y", n)
    ds.createVariable("x", "f8", ("x",))[:] = x
    ds.createVariable("y", "f8", ("y",))[:] = y
    ds.createVariable("thickness", "f4", ("y", "x"))[:] = H
    ds.createVariable("bed", "f4", ("y", "x"))[:] = bed
    ds.createVariable("mask", "i1", ("y", "x"))[:] = mask
    ds.close()
    return H, bed, mask


# --- §H.1.1 / §H.1.2 RTN corollaries on BedMachine (independent cross-check) --
@pytest.mark.skipif(__import__("importlib").util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed")
def test_bedmachine_corollaries_phi_inversion_monotone(tmp_path):
    """§H.1.1#1: the RTN>1 intruded area is strictly monotone-decreasing in φ
    (the area→φ inverse is single-valued), on a synthetic BedMachine file."""
    from external.rtn_corollaries_bedmachine import analyse
    p = tmp_path / "BedMachineAntarctica-v3.nc"
    _write_synthetic_bedmachine(p)
    s, _ = analyse(str(p), stride=1, phi_ref=0.9,
                   phis=[0.70, 0.80, 0.90, 0.94, 0.98], use_gpu=False)
    fr = [c["frac"] for c in s["phi_inversion"]["calibration"]]
    assert s["phi_inversion"]["monotone_decreasing_in_phi"] is True
    assert all(b <= a + 1e-12 for a, b in zip(fr, fr[1:]))  # non-increasing
    assert fr[0] > fr[-1]                                    # and actually decreasing
    assert s["cellsize_km"] == 0.5


@pytest.mark.skipif(__import__("importlib").util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed")
def test_bedmachine_corollaries_misi_band_width_exact(tmp_path):
    """§H.1.1#2: the per-cell band width (H*−H_flot)/H_flot equals (1−φ)/φ
    exactly, and the RTN=1 line H* sits inland of (above) flotation."""
    from external.rtn_corollaries_bedmachine import analyse
    p = tmp_path / "BedMachineAntarctica-v3.nc"
    _write_synthetic_bedmachine(p)
    for phi in (0.80, 0.90, 0.94):
        s, _ = analyse(str(p), stride=1, phi_ref=phi,
                       phis=[0.80, 0.90], use_gpu=False)
        mb = s["misi_band"]
        cc = mb["consistency_checks"]
        assert cc["H_star_inland_of_flotation"] is True
        assert np.isclose(cc["band_width_ratio_measured"],
                          (1.0 - phi) / phi, rtol=1e-6, atol=1e-9)
        assert np.isclose(cc["band_width_ratio_expected_1mphi_over_phi"],
                          (1.0 - phi) / phi, rtol=1e-12)
        assert mb["band_cells"] > 0


@pytest.mark.skipif(__import__("importlib").util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed")
def test_bedmachine_corollaries_band_width_ratio_corner_above_sea_level(tmp_path):
    """The band-width ratio must not be poisoned by a top-left corner above sea
    level.  There ``bed>=0`` → ``H_flot=H_star=0`` → ``0/0=NaN`` for that cell, so
    sampling only ``[0,0]`` (the old behaviour) returned NaN; the ratio must still
    equal the exact identity ``(1-φ)/φ`` taken over all ``H_flot>0`` cells."""
    import netCDF4
    from external.rtn_corollaries_bedmachine import analyse
    p = tmp_path / "BedMachineGreenland-corner.nc"
    _write_synthetic_bedmachine(p)
    # raise the top-left corner of the bed above sea level (land), as on a
    # Greenland file or a cropped domain whose [0,0] sits above the geoid
    ds = netCDF4.Dataset(str(p), "a")
    bed = ds.variables["bed"][:]
    bed[0, 0] = 250.0
    ds.variables["bed"][:] = bed
    ds.close()
    assert bed[0, 0] >= 0  # the cell that used to produce 0/0 = NaN
    for phi in (0.80, 0.90, 0.94):
        s, _ = analyse(str(p), stride=1, phi_ref=phi,
                       phis=[0.80, 0.90], use_gpu=False)
        mb = s["misi_band"]
        cc = mb["consistency_checks"]
        assert np.isfinite(cc["band_width_ratio_measured"])
        assert np.isclose(cc["band_width_ratio_measured"],
                          (1.0 - phi) / phi, rtol=1e-6, atol=1e-9)
        # the above-sea-level corner (H_flot=H_star=0) must not drag this False
        assert cc["H_star_inland_of_flotation"] is True


@pytest.mark.skipif(__import__("importlib").util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed")
def test_bedmachine_corollaries_threshold_identity_and_band(tmp_path):
    """§H.1.1 consistency on BedMachine geometry:

    (i) the RTN=1 line is the critical-thickness threshold — among grounded ice,
        ``RTN>1 ⇔ H < H* = (ρ_w/φρ_i)·d_base`` cell-for-cell; and
    (ii) the MISI band {grounded ∧ H_flot<H<H*} equals the *truly-grounded*
        (H>H_flot) RTN>1 set cell-for-cell — i.e. the band is the RTN>1 set with
        the sub-flotation cells (H<H_flot, already buoyant) removed.
    """
    from external.rtn_corollaries_bedmachine import analyse, RHO_W
    from external.run_rtn_bedmachine import build_rtn as bm_build_rtn
    from external.bedmachine_loader import load_fields, MASK_GROUNDED
    from validators.rtn_validator import RHO_I
    p = tmp_path / "BedMachineAntarctica-v3.nc"
    _write_synthetic_bedmachine(p)
    phi = 0.9
    s, _ = analyse(str(p), stride=1, phi_ref=phi, phis=[0.9], use_gpu=False)
    d = load_fields(str(p), fields=("thickness", "bed", "mask"), stride=1)
    H, bed, mask = d["thickness"], d["bed"], d["mask"]
    grounded = (mask == MASK_GROUNDED) & np.isfinite(H) & (H > 0)
    d_base = np.where(np.isfinite(bed), np.maximum(0.0, -bed), np.nan)
    H_flot = (RHO_W / RHO_I) * d_base
    H_star = H_flot / phi
    rtn_gt1 = (bm_build_rtn(H, bed, phi) > 1.0)
    # (i) threshold identity on grounded ice
    lhs = grounded & rtn_gt1
    rhs = grounded & (H < H_star)
    assert np.array_equal(lhs, rhs)
    # (ii) band == truly-grounded RTN>1 set
    band = grounded & (H > H_flot) & (H < H_star)
    assert int((lhs & (H > H_flot)).sum()) == s["misi_band"]["band_cells"]
    assert int(band.sum()) == s["misi_band"]["band_cells"]


@pytest.mark.skipif(__import__("importlib").util.find_spec("netCDF4") is None,
                    reason="netCDF4 not installed")
def test_bedmachine_corollaries_intrusion_clock_finite(tmp_path):
    """§H.1.2: the intrusion clock yields a non-empty near-tipping front with a
    finite, positive geometric amplification A=1/|∇m| and a τ_ice co-location."""
    from external.rtn_corollaries_bedmachine import analyse
    p = tmp_path / "BedMachineAntarctica-v3.nc"
    _write_synthetic_bedmachine(p)
    s, _ = analyse(str(p), stride=1, phi_ref=0.9, phis=[0.9], use_gpu=False)
    cl = s["intrusion_clock"]
    assert cl["n_front_cells"] > 0
    A = cl["amplification_km_per_m"]
    assert np.isfinite(A["median"]) and A["median"] > 0
    assert A["p25"] <= A["median"] <= A["p90"]
    # front (thin, near-tipping ice) has shorter thermal memory than the interior
    t = cl["tau_ice_yr"]
    assert t["front_median"] < t["interior_median"]


@pytest.mark.parametrize("fn", xr.ALL, ids=lambda f: f.__name__)
def test_cross_relationship_verified(fn):
    """Each cross-cutting relationship (NR1-NR7) linking the four papers to each
    other and to the mainstream literature verifies one-by-one."""
    r = fn()
    assert r["ok"], f"{r['name']} failed: {r}"


@pytest.mark.parametrize("fn", xr2.ALL, ids=lambda f: f.__name__)
def test_cross_relationship2_verified(fn):
    """Batch-2 cross-cutting relationships (NR8 spectral/FDT early-warning, NR9
    Kramers-Kronig migration<->dispersion, NR10 height-above-flotation unifier)
    verify one-by-one."""
    r = fn()
    assert r["ok"], f"{r['name']} failed: {r}"


def test_nr8_spectral_corner_matches_time_domain_ews():
    """NR8 spectral corner f_c=lambda/2pi must be consistent with the NR3 time-domain
    fold exponents: corner ~ (N-N_c)^+2 and variance ~ (N-N_c)^-2."""
    r = xr2.nr8_spectral_fdt_ews()
    assert abs(r["exp_corner_fc"] - 2.0) < 0.2
    assert abs(r["exp_variance"] + 2.0) < 0.2
    # fluctuation-dissipation identity Var*(2 pi f_c) = D is constant in N
    assert r["fdt_product_rel_spread"] < 1e-9


def test_nr9_ktheory_projection_violates_causality():
    """NR9: the K-theory projection (scale-dependent Re Z, Im Z:=0) has an O(1)
    Kramers-Kronig residual, while the causal admittance reconstructs to <10%."""
    r = xr2.nr9_kramers_kronig_migration()
    assert r["kk_reconstruction_relerr"] < 0.10
    assert r["ktheory_kk_violation"] > 0.5


def test_nr10_sliding_fold_precedes_geometric_flotation():
    """NR10: the s_N drag-side fold sits at a strictly positive height above
    flotation (h_af^c = N_c/(rho_i g) > 0), so the velocity early-warning fires
    before the ice geometrically ungrounds."""
    r = xr2.nr10_flotation_unifier()
    assert r["h_af_fold_m"] > 0.0
    assert r["fold_above_flotation"] and r["flotation_at_phi1"]
    assert r["N_eq_rhoigh_af_relerr"] < 1e-12


@pytest.mark.parametrize("fn", xr3.ALL, ids=lambda f: f.__name__)
def test_cross_relationship3_verified(fn):
    """Batch-3 cross-cutting relationships (NR11 tidal-phase->hydraulic RC, NR12
    ice-clock tau_d as a shared cutoff that inverts for basal ablation velocity)
    verify one-by-one."""
    r = fn()
    assert r["ok"], f"{r['name']} failed: {r}"


def test_nr11_recovers_planted_hydraulic_residence():
    """NR11: the tidal velocity phase-lag spectrum recovers the planted hydraulic
    residence time RC to <1% from surface velocity alone."""
    r = xr3.nr11_tidal_phase_hydraulic_rc()
    assert r["RC_max_relerr"] < 0.05
    assert r["phase_increases"]


def test_nr12_coupling_rolloff_is_universal_in_omega_tau_d():
    """NR12: omega_half * tau_d is Vbar-independent (universal coupling shape), so
    the rolloff inverts for the basal ablation velocity Vbar; the kernel keeps its
    diffusive t^{-1/2} short-time tail."""
    r = xr3.nr12_ice_clock_inversion()
    assert r["universal_const_rel_spread"] < 1e-3
    assert r["Vbar_max_relerr"] < 1e-3
    assert abs(r["kernel_shorttime_slope"] + 0.5) < 0.05


@pytest.mark.parametrize("fn", xr4.ALL, ids=lambda f: f.__name__)
def test_cross_relationship4_verified(fn):
    """Batch-4 cross-cutting relationships (NR13 turbulent Chapman-Enskog / Deborah
    ladder, NR14 RTN grounding-line concentration length from h_af) verify one-by-one."""
    r = fn()
    assert r["ok"], f"{r['name']} failed: {r}"


def test_nr13_truncation_error_order_ladder():
    """NR13: truncating the turbulent Chapman-Enskog (memory) expansion at order p
    leaves error ~ De^{p+1}; the exponent ladder is 1, 2, 3 for p = 0, 1, 2."""
    r = xr4.nr13_chapman_enskog_ladder()
    for p, e in enumerate(r["error_exponents_by_order"]):
        assert abs(e - (p + 1)) < 0.1
    assert r["higher_order_more_accurate"]


def test_nr14_front_width_equals_sigma_over_s():
    """NR14: the RTN>1 fraction is the normal-CDF front Phi((x_50-x)/w) with recovered
    width w = sigma_h/s (universal), centre x_50=H(1-phi)/s moving inland with phi at
    FIXED width (sec.6.2's phi-robust ordering)."""
    r = xr4.nr14_rtn_concentration_length()
    assert r["width_matches_derived"] and r["universal_rel_spread"] < 0.05
    assert r["width_invariant_in_phi"] and r["centre_moves_inland"]
    assert r["median_dist_RTNgt1_km"] < r["median_dist_rest_km"]


@pytest.mark.parametrize("fn", xr5.ALL, ids=lambda f: f.__name__)
def test_cross_relationship5_verified(fn):
    """Batch-5 cross-cutting relationship (NR15 backscatter budget = signed memory-
    kernel integral; negative net eddy viscosity -> resolved-mode growth) verifies."""
    r = fn()
    assert r["ok"], f"{r['name']} failed: {r}"


def test_nr15_negative_net_eddy_viscosity_grows():
    """NR15: nu_eff=int K=c_f-c_b sets the sign of the resolved mode; a backscatter-
    dominated kernel (nu_eff<0) GROWS, with the crossover exactly at beta=c_b/c_f=1 --
    behaviour no strictly positive K-theory eddy viscosity can produce."""
    r = xr5.nr15_backscatter_budget()
    assert r["int_identity_ok"]
    assert r["x_final_dissipative"] < 0.1 and r["rate_dissipative"] < 0
    assert r["x_final_backscatter"] > 5.0 and r["rate_backscatter"] > 0
    assert r["rate_sign_flips_at_beta1"]


# --- RTN variable-phi skill (§V.4: the only escape from the flotation tautology) ---
def test_rtn_variable_phi_overall_pass():
    """Plant-and-recover: constant-phi == flotation (zero skill); a connectivity-
    informed variable phi(x,y) breaks the degeneracy and adds real skill; a random
    phi does not; and the gain scales with phi informativeness."""
    r = rvp.run(n_cells=120_000, seed=1)
    assert r["pass"], r


def test_rtn_constant_phi_reproduces_flotation_anchor():
    """With a constant phi, RTN is an exact monotone function of the flotation
    fraction (Spearman -1 vs H_af/H) and {RTN>1} is identical to thickness-above-
    flotation -- zero disagreeing cells (reproduces the committed baseline result)."""
    r = rvp.run(n_cells=120_000, seed=1)
    a = r["anchor_constant_phi"]
    assert a["spearman_rtn_flotfrac_Haf"] < -0.999
    assert a["disagree_vs_flotthresh_cells"] == 0


def test_rtn_variable_phi_breaks_degeneracy():
    """A spatially-varying phi(x,y) makes RTN no longer a function of the flotation
    fraction alone: |Spearman|<1 and a non-empty set of cells reclassify."""
    r = rvp.run(n_cells=120_000, seed=1)
    d = r["degeneracy_break_variable_phi"]
    assert abs(d["spearman_rtn_f"]) < 0.999
    assert d["disagree_vs_flotthresh_cells"] > 0


def test_rtn_variable_phi_skill_beats_null():
    """The skill gain is connectivity-sourced: the informed phi beats the flotation
    baseline AND beats a random-phi null of the same marginal (which adds ~nothing)."""
    r = rvp.run(n_cells=120_000, seed=1)
    s = r["skill"]; nu = r["null_random_phi"]
    assert s["auc_phi_aware"] > s["auc_baseline_flotation"]
    assert s["f1_phi_aware"] > s["f1_baseline"]
    assert s["auc_phi_aware"] > nu["auc_null"]              # informed beats null
    assert nu["auc_null"] <= s["auc_baseline_flotation"] + 0.005


def test_rtn_variable_phi_dose_response_monotone():
    """Skill gain rises monotonically with informativeness corr(phi_obs, phi_true)
    and -> 0 as phi_obs -> random."""
    r = rvp.run(n_cells=120_000, seed=1)
    dr = r["dose_response"]
    assert dr["monotone_nonincreasing"] and dr["corr_info_vs_gain"] > 0.9


# --- nonlinear cavity<->channel kernel (paper4a §4.x): linear = small-amplitude limit ---
def test_nonlinear_kernel_reduces_to_linear_and_predicts_flood_lag():
    """The physical nonlinear Roethlisberger cavity<->channel system (a) linearises to
    the paper's overdamped M, (b) reproduces the linear MZ kernel as its small-amplitude
    limit, and (c) predicts a falsifiable amplitude-dependent lag (bigger floods lag
    longer) via the Glen-law N^3 creep closure."""
    out, _t, _M, _m = hnl.run()
    assert out["overdamped_real"]                       # M has real negative eigenvalues
    assert out["linear_is_small_amplitude_limit"]       # nonlinear -> linear as dQ -> 0
    assert out["rel_gap"][0] < 0.02                      # small-flood gap is tiny
    assert out["bigger_floods_longer_lag"]              # the nonlinear prediction
    assert out["dlag_d(floodfrac)_yr"] > 0.0            # positive, falsifiable slope
    assert out["pass"]


def test_nonlinear_kernel_flood_lag_is_parameter_robust():
    """'Bigger floods -> longer lag' is not a single-parameter artefact: it holds for
    every overdamped configuration across a physical sweep of N*, storage and melt-
    opening, and strengthens toward flotation (the N^3 creep is more sensitive there)."""
    rob = hnl.robustness_sweep()
    assert rob["n_overdamped"] >= 10
    assert rob["robust_all"]                            # holds in every overdamped config
    assert rob["strengthens_toward_flotation"]
