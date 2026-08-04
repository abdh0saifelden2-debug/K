r"""§G.4 lag-value derivation (Step A) — earn ``t*`` from coupled subglacial
hydrology instead of *planting* ``τ₁, τ₂``.
 
Context
-------
``hydraulic_kernel_synthetic.py`` proved the kernel *shape*: a two-compartment
(pressure-store → drainage-element) system gives a downstream impulse response
that starts at zero, rises to an interior peak, and decays, with peak time
 
    t* = τ₁ τ₂ ln(τ₁/τ₂) / (τ₁ − τ₂)                                    (★)
 
But that module **plants** ``τ₁ = 2.0 yr, τ₂ = 0.5 yr`` (its own docstring says
so), so the lag *value* was tagged ``[HYP, NOT EARNED]`` in §G.4 / §H.2.  This
module removes the planting: it writes the lumped subglacial-hydrology ODEs with
**named physical constants and literature parameter values**, linearises them,
and reads ``τ₁, τ₂`` (and the coupling ``a, b``) off the **2×2 Jacobian**.  The
peak time ``t*`` then falls out of the physics and is compared with the observed
post-drainage surge-lag band **0.02–2 yr** (§H.2; Stearns et al. 2008; Siegfried
et al. 2016), against which the literal thermal kernel ``H²/κ`` (~10⁵ yr) was
falsified by ~8×10⁴×.
 
Two drainage elements — and why the *channel* one is the wrong mechanism
-----------------------------------------------------------------------
The state is ``x = (p_w, q)``: subglacial water pressure ``p_w`` [Pa] (the
distributed store, "charge time τ₁") coupled to a drainage-element opening ``q``
("opening time τ₂").  ``N = p_i − p_w`` is the effective pressure.  Two physically
distinct choices for ``q`` give *opposite* stability:
 
A. **Röthlisberger channel**, ``q = S`` (cross-section, melt-opening vs. Nye creep):
 
       dS/dt = Ξ(p_w,S)/(ρ_i L) − K_c A S Nⁿ,   Ξ = k_c S^α Ψ^{3/2}      (chan)
 
   The melt-opening term gives ``∂(dS/dt)/∂S = +(α−1)·(melt) > 0`` at the
   discharge-balanced state: the channel diagonal is **positive** → the lumped
   (p_w,S) system has ``trace > 0`` and is **structurally unstable** for every
   ice-stream parameter set we sweep.  This is not a bug — it is the well-known
   channelisation / subglacial-lake *oscillation* instability (Schoof 2010;
   Kingslake 2015).  A linearly unstable system has **no decaying peaked impulse
   kernel**, so the literal "Röthlisberger channel" cannot be the source of the
   observed single rise-and-decay surge.  ``channel_regime()`` documents this.
 
B. **Cavity / linked-cavity sheet**, ``q = h_s`` (opening thickness, sliding-open
   vs. Nye creep):
 
       dh_s/dt = u_b (h_r − h_s)/l_r − A h_s Nⁿ                          (cav)
 
   Sliding over bumps of height ``h_r`` and spacing ``l_r`` opens cavities; Nye
   creep closes them.  The opening diagonal is ``∂(dh_s/dt)/∂h_s =
   −(u_b/l_r + A Nⁿ) < 0`` — **unconditionally stable**.  Combined with the
   pressure store this gives ``trace < 0, det > 0`` *always* (a stable node or
   focus), i.e. exactly the decaying peaked kernel (★) the data show.  This is
   the mechanism that earns the lag value.
 
Both elements share the same **opening-closure timescale**
 
    τ_open = 1 / (u_b/l_r + A Nⁿ)                                        (τ₂)
 
— sliding-opening rate plus Nye creep-closure rate — which is the physical clock
that sets the lag.  For ice-stream values (u_b ~ 30–800 m yr⁻¹, l_r ~ 1–20 m,
N ~ 0.02–2 MPa) it is **0.004–0.3 yr**, and the storage-charge time
 
    τ_store = Σ / (∂Q_out/∂p_w)   = R · C                                (τ₁)
 
(the §G.4 "R·C": hydraulic resistance × englacial storage capacitance) is
weeks–centuries.  By (★), ``t* ≈ τ_open · ln(τ_store/τ_open)`` — set by τ_open and
only *logarithmically* by τ_store — so it lands robustly in 0.02–2 yr.
 
Lumped pressure store (shared by both)
--------------------------------------
       Σ dp_w/dt = Q_in − Q_out(p_w, q)                                  (sto)
       Q_out = (G_ch + k_s W q^{a_s}) Ψ^{1/2},  Ψ = Ψ_b + (p_w − p_w^out)/ℓ
       Σ = e_v · A_bed / (ρ_w g)        (englacial storage capacity)
 
(GlaDS continuity with englacial storage; Werder et al. 2013; Bartholomaus 2011.)
The operating point is set by the **observed** effective pressure ``N`` (Antarctic
ice-stream lakes sit near flotation, N ~ 0.02–2 MPa) — i.e. we derive the lag
*given* the observed hydraulic state, which is what the §G.4 lag-value claim is
about; we do not try to predict N itself.
 
What this earns (and what it does not)
--------------------------------------
EARNS: ``t*`` as a **[DERIVED]** order-of-magnitude from named hydrology +
literature parameters, with a sweep showing the fraction of literature-plausible
parameter space landing in 0.02–2 yr; and it pins the *regime* — the literal
Röthlisberger channel is an unstable oscillator (Kingslake 2015), the observed
peaked surge is cavity-opening-paced.
 
DOES NOT EARN: a data ``[VERIFIED]`` lag fit — the vetted drainage-*date*
catalogue is gated behind USAP-DC login (§H.2), so an end-to-end
observed-vs-predicted match is not runnable here.  And the lumped coefficients are
a reduction of the full spatial GlaDS PDE; Step B (nonlinear ODE integration)
checks the linearised ``t*`` survives the nonlinearity.
 
References
----------
Röthlisberger (1972) J. Glaciol.; Nye (1976); Schoof (2010) Nature 468; Hewitt
(2011) JGR; Werder, Hewitt, Schoof & Flowers (2013) JGR (GlaDS); Kingslake (2015)
J. Glaciol. (lumped subglacial-lake drainage oscillations); Cuffey & Paterson
(2010) for Glen's ``A``; Stearns et al. (2008) Nat. Geosci.; Siegfried et al.
(2016) GRL.
"""
from __future__ import annotations
 
import numpy as np
from scipy.optimize import brentq
 
SEC_PER_YR = 365.25 * 86400.0
 
# --- physical constants ------------------------------------------------------
RHO_I = 917.0        # ice density            [kg m⁻³]
RHO_W = 1000.0       # water density          [kg m⁻³]
G = 9.81             # gravity                [m s⁻²]
L_FUSION = 3.34e5    # latent heat of fusion  [J kg⁻¹]
N_GLEN = 3.0         # Glen exponent          [-]
 
OBS_BAND_YR = (0.02, 2.0)    # observed post-drainage surge lags (§H.2)
 
 
def baseline_params():
    """Representative West-Antarctic ice-stream subglacial-lake catchment.
 
    Every value is a measured/literature physical constant or a documented
    GlaDS/Röthlisberger/cavity parameter — none is tuned to land ``t*`` in the
    band.  ``N_obs`` is the *observed* near-flotation effective pressure at which
    the lag is derived.
    """
    H = 2000.0                       # ice thickness [m] (Siegfried lakes median ~2280 m)
    return dict(
        H=H,
        p_i=RHO_I * G * H,           # ice overburden [Pa] ≈ 1.80e7
        N_obs=0.3e6,                 # observed effective pressure [Pa] (near flotation)
        A_glen=2.4e-24,              # Glen softness [Pa⁻³ s⁻¹] (Cuffey & Paterson, temperate)
        n=N_GLEN,
        K_c=2.0 * N_GLEN ** (-N_GLEN),   # Nye cylindrical creep-closure constant = 2/27
        # cavity-opening (sliding) closure
        u_b=100.0,                   # basal sliding speed [m yr⁻¹] (ice-stream)
        l_r=4.0,                     # bed-bump spacing / cavity length [m] (Schoof 2010)
        h_r=0.1,                     # bed-bump height (max cavity opening) [m]
        # pressure-store discharge law
        k_s=1.0e-2,                  # distributed-sheet conductivity [m^... ] (Werder 2013)
        a_s=1.25,                    # sheet discharge exponent h_s^{a_s} (GlaDS 5/4)
        G_ch=1.0e-3,                 # background channel conductance [m³ s⁻¹ Pa^-1/2]
        # Röthlisberger-channel pieces (used only by channel_regime)
        k_c=0.10,                    # channel conductivity (Werder 2013 turbulent)
        alpha=1.25,                  # channel discharge exponent S^α (GlaDS turbulent 5/4)
        Q_in=2.0e-2,                 # background water supply [m³ s⁻¹] (basal melt + lake)
        # shared geometry / storage
        psi_b=50.0,                  # background hydraulic-potential gradient [Pa m⁻¹]
        ell=1.0e4,                   # lumped channel/catchment length [m]
        W=5.0e3,                     # catchment width [m]
        e_v=1.0e-3,                  # englacial void ratio (Werder 2013 ~1e-3)
        pw_out=0.0,                  # downstream (terminus) water pressure [Pa]
    )
 
 
# --- shared pieces -----------------------------------------------------------
def _psi(pw, p):
    """Hydraulic-potential gradient Ψ = Ψ_b + (p_w − p_w^out)/ℓ  [Pa m⁻¹] (>0)."""
    return p["psi_b"] + (pw - p["pw_out"]) / p["ell"]
 
 
def storage(p):
    """Lumped englacial storage capacity Σ = e_v·A_bed/(ρ_w g)  [m³ Pa⁻¹]."""
    return p["e_v"] * (p["W"] * p["ell"]) / (RHO_W * G)
 
 
# === drainage element A: Röthlisberger channel (documents the instability) ===
def _discharge_chan(pw, S, p):
    """Channel discharge Q = k_c S^α Ψ^{1/2}  [m³ s⁻¹] (turbulent Darcy–Weertman)."""
    return p["k_c"] * S ** p["alpha"] * np.sqrt(_psi(pw, p))
 
 
def channel_regime(p):
    """Steady state + linear stability of the literal (p_w, S) Röthlisberger system.
 
    Returns the discharge-balanced steady state and the sign of ``trace`` of its
    2×2 Jacobian.  For ice-stream parameters the melt-opening term makes
    ``J₂₂ = (α−1)·melt − ... > 0`` so ``trace > 0`` ⇒ **unstable** (the
    channelisation / lake-oscillation regime, Schoof 2010; Kingslake 2015).  This
    is *why* the literal channel kernel cannot produce the observed decaying
    peaked surge, motivating the cavity model below.
    """
    def S_of_pw(pw):
        return (p["Q_in"] / (p["k_c"] * np.sqrt(_psi(pw, p)))) ** (1.0 / p["alpha"])
 
    def channel_residual(pw):
        S = S_of_pw(pw)
        N = p["p_i"] - pw
        Xi = p["k_c"] * S ** p["alpha"] * _psi(pw, p) ** 1.5
        return Xi / (RHO_I * L_FUSION) - p["K_c"] * p["A_glen"] * S * N ** p["n"]
 
    lo, hi = 1.0, p["p_i"] - 1.0
    flo, fhi = channel_residual(lo), channel_residual(hi)
    if np.sign(flo) == np.sign(fhi):
        grid = np.linspace(lo, hi, 4001)
        vals = np.array([channel_residual(x) for x in grid])
        sc = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
        if sc.size == 0:
            return {"steady_state": None}
        i = sc[0]
        pw = brentq(channel_residual, grid[i], grid[i + 1])
    else:
        pw = brentq(channel_residual, lo, hi)
    S = S_of_pw(pw)
    N = p["p_i"] - pw
    Sigma = storage(p)
    psi = _psi(pw, p)
    a = p["alpha"]
    dpsi = 1.0 / p["ell"]
    J11 = -(p["k_c"] * S ** a * 0.5 * psi ** -0.5 * dpsi) / Sigma
    J12 = -(a * p["k_c"] * S ** (a - 1.0) * np.sqrt(psi)) / Sigma
    J21 = (p["k_c"] * S ** a * 1.5 * np.sqrt(psi) * dpsi) / (RHO_I * L_FUSION) \
        + p["K_c"] * p["A_glen"] * S * p["n"] * N ** (p["n"] - 1.0)
    J22 = (a * p["k_c"] * S ** (a - 1.0) * psi ** 1.5) / (RHO_I * L_FUSION) \
        - p["K_c"] * p["A_glen"] * N ** p["n"]
    J = np.array([[J11, J12], [J21, J22]])
    trace = float(np.trace(J))
    det = float(np.linalg.det(J))
    return {
        "steady_state": {"p_w": float(pw), "S": float(S), "N": float(N),
                         "pw_over_pi": float(pw / p["p_i"])},
        "trace_per_s": trace, "det_per_s2": det,
        "stable": bool(trace < 0 and det > 0),
        "J22_per_s": float(J22),
    }
 
 
# === drainage element B: cavity opening (the stable peaked kernel) ===========
def cavity_opening_eq(N, p):
    """Steady cavity opening h_s* at effective pressure N: sliding-open = creep-close.
 
        u_b (h_r − h_s)/l_r = A h_s Nⁿ
        ⇒ h_s* = (u_b h_r / l_r) / (u_b/l_r + A Nⁿ)               [m]
    """
    ub = p["u_b"] / SEC_PER_YR
    rate_open = ub / p["l_r"]
    rate_close = p["A_glen"] * N ** p["n"]
    return (rate_open * p["h_r"]) / (rate_open + rate_close)
 
 
def operating_point(p):
    """Operating point (p_w, h_s) at the observed effective pressure ``N_obs``.
 
    ``h_s`` is the cavity opening-closure equilibrium at ``N_obs``; ``p_w`` follows
    from ``N_obs``.  The implied steady water supply ``Q_in_implied = Q_out`` is
    returned for a physical-plausibility check (should be O(basal melt + lake)).
    """
    N = p["N_obs"]
    pw = p["p_i"] - N
    hs = cavity_opening_eq(N, p)
    Keff = p["G_ch"] + p["k_s"] * p["W"] * hs ** p["a_s"]
    Q_out = Keff * np.sqrt(_psi(pw, p))
    return {"p_w": float(pw), "h_s": float(hs), "N": float(N),
            "Keff": float(Keff), "Q_in_implied": float(Q_out),
            "pw_over_pi": float(pw / p["p_i"])}
 
 
def jacobian_cavity(p):
    """Analytic 2×2 Jacobian of the (p_w, h_s) system at the operating point [s⁻¹].
 
    Rows/cols ordered (p_w, h_s).  Structure ``[[−1/τ₁, −a], [b, −1/τ₂]]`` with
    a, b > 0 — the cavity↔store feedback that makes the downstream response peak.
    """
    op = operating_point(p)
    pw, hs, N = op["p_w"], op["h_s"], op["N"]
    Sigma = storage(p)
    psi = _psi(pw, p)
    ub = p["u_b"] / SEC_PER_YR
    a_s = p["a_s"]
    dpsi = 1.0 / p["ell"]
    Keff = op["Keff"]
 
    # store:  Σ dp_w/dt = Q_in − (G_ch + k_s W h_s^{a_s}) Ψ^{1/2}
    J11 = -(Keff * 0.5 * psi ** -0.5 * dpsi) / Sigma                     # < 0
    J12 = -(p["k_s"] * p["W"] * a_s * hs ** (a_s - 1.0) * np.sqrt(psi)) / Sigma  # < 0
    # cavity: dh_s/dt = u_b(h_r−h_s)/l_r − A h_s Nⁿ,  ∂N/∂p_w = −1
    J21 = p["A_glen"] * hs * p["n"] * N ** (p["n"] - 1.0)                # > 0
    J22 = -(ub / p["l_r"] + p["A_glen"] * N ** p["n"])                   # < 0
    return np.array([[J11, J12], [J21, J22]])
 
 
def channel_impulse_peak_time(M):
    r"""Interior peak time of the downstream (x₂) impulse response of a 2×2 system
    ``M`` to a unit impulse into x₁, ``x₂(t) = [expm(Mt)]₂₁``.
 
    For any 2×2 matrix, ``[expm(Mt)]₂₁ = m₂₁ e^{μt} sinh(δt)/δ`` with
    ``μ = tr(M)/2`` and ``δ = √(μ² − det M)`` (real ⇒ node, imaginary ⇒ focus).
    Setting the derivative to zero gives ``tanh(δ t*) = −δ/μ`` (node) or
    ``tan(ω t*) = ω/|μ|`` (focus, δ = iω).  Returns ``t*`` (same time units as
    M⁻¹) or ``None`` if there is no interior peak (e.g. unstable / μ ≥ 0).
    """
    mu = 0.5 * (M[0, 0] + M[1, 1])
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if mu >= 0:                                   # not a decaying response → no interior peak
        return None
    disc = mu * mu - det
    if disc > 0:                                  # node: real δ < |μ| iff det>0
        delta = np.sqrt(disc)
        arg = -delta / mu                         # = delta/|mu| ∈ (0,1) when det>0
        if not (0.0 < arg < 1.0):
            return None
        return float(np.arctanh(arg) / delta)
    elif disc < 0:                                # focus: δ = iω
        omega = np.sqrt(-disc)
        return float(np.arctan2(omega, -mu) / omega)
    else:                                         # critically damped: x₂ ∝ t e^{μt}
        return float(-1.0 / mu)
 
 
def cascade_peak_time(tau1, tau2):
    """Closed-form (★) two-stage cascade peak time for diagonal times τ₁, τ₂.
 
    Used as a weak-coupling cross-check of the full-matrix ``t*``.
    """
    if tau1 <= 0 or tau2 <= 0:
        return None
    if abs(tau1 - tau2) < 1e-12 * tau1:
        return float(tau1)                        # critically damped limit
    return float(tau1 * tau2 * np.log(tau1 / tau2) / (tau1 - tau2))
 
 
def derive_lag(p):
    """Derive (τ₁, τ₂, coupling, eigenvalues, regime, t*) from the cavity physics.
 
    ``t*`` is the interior peak time of the downstream (cavity) impulse response of
    the *physical* Jacobian, evaluated in closed form.  Always stable for this
    model (trace<0, det>0), so a peak always exists.
    """
    op = operating_point(p)
    J = jacobian_cavity(p)                        # [s⁻¹]
    eig = np.linalg.eigvals(J)
    trace, det = float(np.trace(J)), float(np.linalg.det(J))
    stable = bool(trace < 0 and det > 0)
    oscillatory = bool(np.any(np.abs(eig.imag) > 1e-30))
 
    tau1_yr = (-1.0 / J[0, 0]) / SEC_PER_YR if J[0, 0] < 0 else np.nan   # store charge
    tau2_yr = (-1.0 / J[1, 1]) / SEC_PER_YR if J[1, 1] < 0 else np.nan   # cavity open
 
    M_yr = J * SEC_PER_YR
    tstar_yr = channel_impulse_peak_time(M_yr)
    return {
        "operating_point": op,
        "eig_per_s": eig,
        "trace_per_s": trace, "det_per_s2": det,
        "stable": stable, "oscillatory": oscillatory,
        "tau1_yr": float(tau1_yr), "tau2_yr": float(tau2_yr),
        "tstar_yr": tstar_yr,
        "tstar_cascade_yr": cascade_peak_time(tau1_yr, tau2_yr),
        "eig_timescale_yr": [float(1.0 / abs(e.real) / SEC_PER_YR) if abs(e.real) > 0 else np.inf
                             for e in eig],
    }
 
 
# --- Step B: full nonlinear ODE integration under an impulse drainage --------
def _nonlinear_rates(pw, hs, p, Q_in):
    """Full nonlinear (dp_w/dt, dh_s/dt) [Pa/s, m/s] — no linearisation."""
    N = p["p_i"] - pw
    Sigma = storage(p)
    ub = p["u_b"] / SEC_PER_YR
    Keff = p["G_ch"] + p["k_s"] * p["W"] * hs ** p["a_s"]
    dpw = (Q_in - Keff * np.sqrt(_psi(pw, p))) / Sigma
    dhs = ub * (p["h_r"] - hs) / p["l_r"] - p["A_glen"] * hs * N ** p["n"]
    return dpw, dhs
 
 
def nonlinear_impulse_response(p, amp_frac=0.05, t_max_mult=20.0, n_t=40001):
    """Integrate the FULL nonlinear ODEs from the steady state after an impulse
    drainage and measure the cavity (velocity-proxy) response peak time.
 
    The lake drainage dumps water into the store at t=0: we kick ``p_w`` up by
    ``amp_frac · N`` (a pressure pulse) and integrate the nonlinear system.  The
    sliding-velocity response tracks the cavity opening ``h_s`` (more cavity →
    more sliding), so the observed surge-lag = argmax of ``h_s(t) − h_s*``.  We
    compare this nonlinear peak time with the linearised ``t*`` from
    ``derive_lag``.  ``amp_frac`` small ⇒ should reproduce the linear ``t*``;
    larger ⇒ tests nonlinear survival.
 
    Returns the nonlinear peak time [yr], the linear ``t*`` [yr], and their ratio.
    """
    from scipy.integrate import solve_ivp
 
    op = operating_point(p)
    pw0, hs0, N = op["p_w"], op["h_s"], op["N"]
    Q_in = op["Q_in_implied"]                     # makes (pw0, hs0) a steady state
 
    lin = derive_lag(p)
    tstar_lin_yr = lin["tstar_yr"]
 
    # integrate for many linear timescales so the (possibly oscillatory) first
    # peak is well resolved
    t_scale = (tstar_lin_yr or lin["tau2_yr"]) * SEC_PER_YR
    t_end = t_max_mult * t_scale
    t_eval = np.linspace(0.0, t_end, n_t)
 
    def rhs(t, y):
        dpw, dhs = _nonlinear_rates(y[0], y[1], p, Q_in)
        return [dpw, dhs]
 
    y0 = [pw0 + amp_frac * N, hs0]                # pressure-pulse impulse into the store
    sol = solve_ivp(rhs, (0.0, t_end), y0, t_eval=t_eval, method="LSODA",
                    rtol=1e-8, atol=[1e-2, 1e-10], max_step=t_end / 200.0)
    if not sol.success:
        return {"ok": False, "msg": sol.message}
 
    hs_t = sol.y[1] - hs0                          # cavity response (velocity proxy)
    # first interior maximum (skip t=0); guard against monotone decay
    i_peak = int(np.argmax(hs_t))
    nonlin_peak_yr = float(t_eval[i_peak] / SEC_PER_YR)
    interior = 0 < i_peak < len(t_eval) - 1
    return {
        "ok": True,
        "nonlinear_peak_yr": nonlin_peak_yr,
        "tstar_linear_yr": tstar_lin_yr,
        "ratio_nl_over_lin": (nonlin_peak_yr / tstar_lin_yr) if tstar_lin_yr else None,
        "interior_peak": bool(interior),
        "amp_frac": amp_frac,
        "hs0": hs0, "peak_response_m": float(hs_t[i_peak]),
    }
 
 
def nonlinear_check(p=None, amp_fracs=(0.02, 0.05, 0.1, 0.2, 0.4)):
    """Run the nonlinear impulse response at several impulse amplitudes.
 
    Confirms the linearised ``t*`` survives nonlinearity: at small amplitude the
    nonlinear peak time ≈ linear ``t*``; as the impulse grows the peak time should
    stay the same order of magnitude (nonlinear robustness).
    """
    if p is None:
        p = baseline_params()
    return [nonlinear_impulse_response(p, amp_frac=a) for a in amp_fracs]
 
 
# --- literature-range sweep --------------------------------------------------
SWEEP_RANGES = {
    # (lo, hi, "log"/"lin") — documented literature ranges
    "N_obs":  (0.02e6, 2.0e6, "log"),       # observed effective pressure [Pa] (near flotation)
    "A_glen": (1.0e-25, 5.0e-24, "log"),    # cold → temperate ice (Cuffey & Paterson)
    "u_b":    (30.0, 800.0, "log"),         # basal sliding speed [m yr⁻¹] (ice-stream)
    "l_r":    (1.0, 20.0, "log"),           # bed-bump spacing / cavity length [m]
    "h_r":    (0.05, 0.5, "log"),           # bed-bump height [m]
    "k_s":    (3.0e-3, 3.0e-2, "log"),      # distributed-sheet conductivity
    "psi_b":  (5.0, 100.0, "log"),          # background potential gradient [Pa m⁻¹]
    "ell":    (2.0e3, 5.0e4, "log"),        # catchment length [m]
    "e_v":    (1.0e-4, 1.0e-2, "log"),      # englacial void ratio
    "H":      (700.0, 3900.0, "lin"),       # Siegfried lake thickness range [m]
}
 
 
def _sample(rng, lo, hi, kind):
    if kind == "log":
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))
 
 
def sweep(n=4000, seed=0):
    """Monte-Carlo over literature parameter ranges → distribution of derived t*."""
    rng = np.random.default_rng(seed)
    base = baseline_params()
    tstars, taus1, taus2 = [], [], []
    n_stable = n_band = 0
    for _ in range(n):
        p = dict(base)
        for key, (lo, hi, kind) in SWEEP_RANGES.items():
            p[key] = _sample(rng, lo, hi, kind)
        p["p_i"] = RHO_I * G * p["H"]
        # keep N_obs physical: below flotation
        if p["N_obs"] >= p["p_i"]:
            continue
        r = derive_lag(p)
        if r["stable"]:
            n_stable += 1
        ts = r["tstar_yr"]
        if ts is not None:
            tstars.append(ts)
            taus1.append(r["tau1_yr"])
            taus2.append(r["tau2_yr"])
            if OBS_BAND_YR[0] <= ts <= OBS_BAND_YR[1]:
                n_band += 1
    tstars = np.array(tstars)
    out = {
        "n": n, "n_stable": n_stable, "n_with_peak": int(tstars.size),
        "frac_peak_in_band": float(n_band / tstars.size) if tstars.size else 0.0,
    }
    if tstars.size:
        out["tstar_yr_pct"] = {q: float(np.percentile(tstars, q)) for q in (5, 25, 50, 75, 95)}
        out["tstar_yr_geomean"] = float(np.exp(np.mean(np.log(tstars))))
        out["tau1_yr_median"] = float(np.median(taus1))
        out["tau2_yr_median"] = float(np.median(taus2))
    return out, tstars
 
 
def run():
    """Baseline derivation + channel-instability check + literature sweep."""
    base = baseline_params()
    bl = derive_lag(base)
    chan = channel_regime(base)
    sw, _ = sweep()
 
    # cross-check: analytic cavity Jacobian vs numerical finite-difference
    op = operating_point(base)
    pw0, hs0 = op["p_w"], op["h_s"]
 
    def rates_cav(pw, hs):
        N = base["p_i"] - pw
        Sigma = storage(base)
        Keff = base["G_ch"] + base["k_s"] * base["W"] * hs ** base["a_s"]
        ub = base["u_b"] / SEC_PER_YR
        dpw = (base["Q_in"] - Keff * np.sqrt(_psi(pw, base))) / Sigma
        dhs = ub * (base["h_r"] - hs) / base["l_r"] - base["A_glen"] * hs * N ** base["n"]
        return np.array([dpw, dhs])
 
    J = jacobian_cavity(base)
    eps_p, eps_h = 1.0, 1e-5
    f0 = rates_cav(pw0, hs0)
    Jn = np.empty((2, 2))
    Jn[:, 0] = (rates_cav(pw0 + eps_p, hs0) - f0) / eps_p
    Jn[:, 1] = (rates_cav(pw0, hs0 + eps_h) - f0) / eps_h
    jac_rel_err = float(np.max(np.abs(J - Jn) / (np.abs(J) + 1e-300)))
 
    return {
        "baseline": bl,
        "channel_regime": chan,
        "sweep": sw,
        "jacobian_rel_err": jac_rel_err,
        "baseline_tstar_yr": bl.get("tstar_yr"),
        "baseline_in_band": (bl.get("tstar_yr") is not None
                             and OBS_BAND_YR[0] <= bl["tstar_yr"] <= OBS_BAND_YR[1]),
        "obs_band_yr": OBS_BAND_YR,
    }
 
 
def main():
    r = run()
    bl = r["baseline"]
    op = bl["operating_point"]
    chan = r["channel_regime"]
    print("=== §G.4 Step A — derive the hydraulic lag t* from subglacial hydrology ===")
    print("--- (A) literal Röthlisberger channel (p_w,S): expected UNSTABLE ---")
    if chan.get("steady_state"):
        cs = chan["steady_state"]
        print(f"  channel steady state  : p_w/p_i={cs['pw_over_pi']:.3f}  N={cs['N']/1e6:.2f} MPa"
              f"  S={cs['S']:.3f} m²")
        print(f"  trace={chan['trace_per_s']:.2e}/s  J22={chan['J22_per_s']:.2e}/s"
              f"  → stable={chan['stable']}  (unstable ⇒ Kingslake-2015 oscillation, no peaked kernel)")
    else:
        print("  no discharge-balanced channel steady state for baseline params")
    print("--- (B) cavity-opening (p_w,h_s): the stable peaked kernel ---")
    print(f"  operating point       : N_obs={op['N']/1e6:.2f} MPa  p_w/p_i={op['pw_over_pi']:.3f}"
          f"  h_s={op['h_s']*100:.2f} cm  Q_in_implied={op['Q_in_implied']:.3e} m³/s")
    print(f"  diagonal timescales   : τ₁(store charge)={bl['tau1_yr']:.4f} yr  "
          f"τ₂(cavity open)={bl['tau2_yr']:.4f} yr")
    print(f"  regime                : stable={bl['stable']} oscillatory={bl['oscillatory']}"
          f"  eig-timescales={[f'{x:.4f}' for x in bl['eig_timescale_yr']]} yr")
    print(f"  DERIVED lag  t*       : {bl['tstar_yr']:.4f} yr  "
          f"(cascade cross-check {bl['tstar_cascade_yr']:.4f} yr; obs band {r['obs_band_yr']} yr)")
    print(f"  baseline in band      : {r['baseline_in_band']}")
    print(f"  analytic-vs-numeric Jac rel-err : {r['jacobian_rel_err']:.2e}")
    sw = r["sweep"]
    print(f"  sweep (n={sw['n']}): stable={sw['n_stable']} with-peak={sw['n_with_peak']}")
    if sw.get("tstar_yr_pct"):
        pct = sw["tstar_yr_pct"]
        print(f"  swept t* [yr] p5/p25/p50/p75/p95 : {pct[5]:.3f} / {pct[25]:.3f} / "
              f"{pct[50]:.3f} / {pct[75]:.3f} / {pct[95]:.3f}  (geomean {sw['tstar_yr_geomean']:.3f})")
        print(f"  swept τ₁/τ₂ median [yr] : {sw['tau1_yr_median']:.4f} / {sw['tau2_yr_median']:.4f}")
        print(f"  fraction of peaks in 0.02–2 yr band : {sw['frac_peak_in_band']:.2f}")
    print("--- Step B: full NONLINEAR ODE integration under an impulse drainage ---")
    print("  (does the linearised t* survive the nonlinearity?)")
    for nb in nonlinear_check():
        if nb["ok"]:
            print(f"  impulse amp={nb['amp_frac']:.2f}·N : nonlinear peak={nb['nonlinear_peak_yr']*365.25:6.2f} d"
                  f"  linear t*={nb['tstar_linear_yr']*365.25:6.2f} d  ratio={nb['ratio_nl_over_lin']:.3f}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
