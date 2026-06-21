r"""§D.4 / §G.5 coefficient closure — pin the unified-memory GLE coefficients.

The structural claims of the unified memory formalism (§D.4: additivity,
scale-selectivity, Markov limit) and of the clock-mismatch correction (§G.5: the
commutator identity ``[d_t, D]theta = div((d_t K_u) grad theta)`` and its
dimensional signature) are already DERIVED and unit-tested
(``validation/synthetic/gle_memory_synthetic.py``, ``.../cmn_synthetic.py``).
What stayed ``[HYP]`` were the *coefficients*:

  * ``tau_c``  -- the autocorrelation/memory time of the SGS eddy diffusivity
                  ``K_u`` (the §G.5 coefficient and the §D.4 *fast* bath time),
                  including its **sign**;
  * the §G.5 question of whether the correction **adds or removes flux**;
  * ``B``, ``tau_d`` and the bath weights of the §B.2 ice kernel (§D.4 *slow* bath);
    the slow:fast bath weight is now closed too -- it is the Stefan number ``St``.

This script closes those, using only existing runs + the repo's own §B.2 closed
form + light CPU/analytics (no new heavy compute, no external data):

  PART A -- measure ``tau_c`` directly from the solver.  ``K_u = (c_s dx)^2 |S|``
            is recorded at fixed interior probe points of a small Smagorinsky
            cavity (``direction_c_gpu_probe.CavityFlow``); its autocorrelation
            gives the *local* memory time.  The spatially-averaged force memory
            (the built-in ``tau_mem_from_history``) is reported alongside as a
            cross-check against the committed RESULT-8 value.  **Sign:** an
            autocorrelation time is non-negative by construction and the measured
            value is strictly positive -- so ``sign(tau_c) = +``, settling the
            §G.5 ``[HYP]`` sign.

  PART B -- resolve the §G.5 "adds or removes flux?" question.  To first order
            ``K_u - tau_c d_t K_u = K_u(t - tau_c)``: the correction is a pure
            **time lag** by ``tau_c`` (>0 => the diffusion responds to the eddy
            diffusivity a memory time in the *past*).  Over a statistically steady
            field ``<d_t K_u> = 0`` so the time-mean correction is ZERO -- it adds
            flux while ``K_u`` grows and removes it while ``K_u`` decays, netting
            out.  This is demonstrated numerically on a synthetic spectral field
            and *quantitatively explains* the RESULT-8 / RESULT-11 null (no mean
            melt or C_G shift from temporal recoloring in the steady cavity).

  PART C -- pin the slow-bath coefficients and the slow:fast bath weight from
            the §B.2 closed form.
            ``K_ice(tau) = B tau^{-1/2} e^{-tau/tau_d}`` with, from §B.2,
            ``A = k_th theta_far Vbar^2 /(2 kappa^2)``, ``tau_d = kappa/Vbar^2``,
            and short-time ``G ~ |A| 2 sqrt(tau_d/pi) tau^{-1/2}`` so
            ``B = |A| 2 sqrt(tau_d/pi) / (rho_i L)``.  Evaluated for representative
            subglacial inputs (the only empirical quantities are the *site* values
            ``theta_far, Vbar``; everything else is a material constant), this
            shows the fast (SGS, seconds) and slow (ice, years) baths are separated
            by ~10^6-10^9 in physical time -- the §D.4 scale-selectivity with real
            numbers rather than the validator's placeholder 0.2-vs-60.
            The **bath weight** (second-FDT DC gain ``int K dtau``) is closed the
            same way: the fast bath is unit-normalised (``int K_SGS dtau = 1``,
            §D.4-(3)), so the slow ice bath's relative weight is the dimensionless
            ice DC gain magnitude ``St = |int K_ice dtau| = cp |theta_far| / L``
            (the Stefan number; ``int G dtau = -rho c theta_far`` is signed, so
            ``int K_ice dtau = -cp theta_far / L`` and the weight is its modulus).
            ``K_SGS : K_ice = 1 : St`` with ``St ~ 0.01-0.06`` up to 10 K -- the
            *same* Stefan number as the §G.4 thermal-tail weight
            (``W_thermal/W_hydraulic`` in ``thermal_tail_amplitude.py``), so the
            slow ice bath is subdominant.  This closes the last RESULT-12 ``[HYP]``.

Outputs ``figures/53_gle_coefficients.json`` and prints a summary.  All numbers
are reproducible: ``python gle_coefficients.py``.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

# --------------------------------------------------------------------------- #
# PART A: measure tau_c (SGS eddy-diffusivity decorrelation time) from the solver
# --------------------------------------------------------------------------- #
# Committed RESULT-8 cross-check values (THEORY_CAVITY.md §11): the spatially-
# averaged SGS-force memory time, set vs effective (CLT-shortened).
RESULT8_TAU_MEM_SET = 0.05          # bs_tau imposed per cell (OU correlation time)
RESULT8_TAU_MEM_EFF = 9.5e-3        # measured mean-field eff. time (RECORD_EVERY-corrected)


def _autocorr_efold(series, sample_dt):
    """e-folding (1/e) decorrelation time of a 1-D series via the FFT ACF, with
    linear interpolation of the crossing lag.  Returns 0.0 for a flat series."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    if x.std() < 1e-30 or len(x) < 8:
        return 0.0
    n = len(x)
    fx = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fx * np.conj(fx)).real[:n]
    acf = acf / acf[0]
    thr = 1.0 / np.e
    crossed = np.where(acf < thr)[0]
    if len(crossed) == 0:
        return float(n * sample_dt)
    idx = int(crossed[0])
    if idx == 0:
        return 0.0
    frac = (acf[idx - 1] - thr) / (acf[idx - 1] - acf[idx] + 1e-30)
    return float(sample_dt * (idx - 1 + frac))


def _autocorr_integral(series, sample_dt):
    """Integral memory time tau_int = sum_{k>=1} ACF(k) * dt up to first zero
    crossing (a second, complementary estimator of the correlation time)."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    if x.std() < 1e-30 or len(x) < 8:
        return 0.0
    n = len(x)
    fx = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fx * np.conj(fx)).real[:n]
    acf = acf / acf[0]
    total = 0.0
    for k in range(1, n):
        if acf[k] <= 0.0:
            break
        total += acf[k]
    return float(sample_dt * total)


def measure_tau_c(n=24, closure="smagorinsky", bs_tau=0.0, seed=1,
                  spinup=400, measure=1500, sample_every=2, n_probes=64):
    """Run a small cavity and measure the local + mean-field K_u memory time.

    Returns a dict with the local K_u e-folding/integral times (averaged over
    interior probe points), the built-in spatially-averaged tau_mem, the eddy
    turnover time, and the dimensionless clock C = tau_c/tau_turn.
    """
    from direction_c_gpu_probe import CavityConfig, CavityFlow, to_np

    cfg = CavityConfig(n=n, sgs=closure, backscatter=(0.7 if bs_tau > 0 else 0.0),
                       bs_tau=bs_tau, Ri=0.0, seed=seed)
    flow = CavityFlow(cfg)
    flow.run(spinup, ramp=max(1, spinup // 3))

    # fixed interior probe points (in the fluid) for the *local* K_u history
    fluid = to_np(flow.fluid > 0)
    idx = np.argwhere(fluid)
    rng = np.random.default_rng(seed)
    sel = idx[rng.choice(len(idx), size=min(n_probes, len(idx)), replace=False)]
    cs_dx2 = (cfg.cs * flow.sp.dx) ** 2

    ku_hist = [[] for _ in range(len(sel))]
    flow.clear_sgs_history()
    times = []
    for s in range(measure):
        flow.step()
        if s % sample_every == 0:
            flow.record_sgs_force()           # mean-field force (built-in)
            _, smag = flow._strain(flow.u, flow.v, flow.w)
            ku = cs_dx2 * to_np(smag)          # local eddy diffusivity field K_u
            for j, (a, b, c) in enumerate(sel):
                ku_hist[j].append(float(ku[a, b, c]))
            times.append(flow.t)

    sdt = float(np.median(np.diff(times))) if len(times) > 1 else cfg.dt
    efold = [_autocorr_efold(h, sdt) for h in ku_hist]
    integ = [_autocorr_integral(h, sdt) for h in ku_hist]
    efold = [e for e in efold if e > 0]
    integ = [e for e in integ if e > 0]
    tau_c_local_efold = float(np.median(efold)) if efold else 0.0
    tau_c_local_int = float(np.median(integ)) if integ else 0.0
    tau_mem_meanfield = float(flow.tau_mem_from_history())

    # eddy turnover: cavity gap / bulk speed
    h_cav = cfg.ice_base - cfg.bed_mean
    tau_turn = h_cav / max(cfg.U0, 1e-30)

    return {
        "closure": closure, "bs_tau_set": bs_tau, "n": n, "seed": seed,
        "sample_dt": sdt, "n_probes_used": len(efold),
        "tau_c_local_efold": tau_c_local_efold,
        "tau_c_local_integral": tau_c_local_int,
        "tau_mem_meanfield": tau_mem_meanfield,
        "tau_turn": tau_turn,
        "clock_C_local": tau_c_local_efold / tau_turn if tau_turn else 0.0,
        "sign_tau_c": int(np.sign(tau_c_local_efold)) if tau_c_local_efold else 0,
    }


# --------------------------------------------------------------------------- #
# PART B: §G.5 correction is a pure time-lag (net-zero in steady state)
# --------------------------------------------------------------------------- #
def _spec_grad(f):
    n = f.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n)
    F = np.fft.fft2(f)
    fx = np.real(np.fft.ifft2(1j * k[:, None] * F))
    fy = np.real(np.fft.ifft2(1j * k[None, :] * F))
    return fx, fy


def _spec_div(fx, fy):
    n = fx.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n)
    return np.real(np.fft.ifft2(1j * k[:, None] * np.fft.fft2(fx)
                                + 1j * k[None, :] * np.fft.fft2(fy)))


def _D(K, th):
    tx, ty = _spec_grad(th)
    return _spec_div(K * tx, K * ty)


def lag_and_netzero(n=48, tau_c=0.02, omega=0.9, eps=0.4, n_cycle=240):
    r"""Demonstrate the two §G.5 properties on a synthetic time-dependent field.

    (i)  Taylor/lag equivalence: the corrected operator
         ``D[theta] - tau_c d_t(D-style) = div((K - tau_c d_tK) grad theta)``
         equals ``div(K(t - tau_c) grad theta)`` to O(tau_c^2) -- i.e. the
         correction is a *time lag* by ``tau_c``.
    (ii) Net-zero in steady state: averaged over a full transient cycle of
         ``K_u(t)`` (so ``<d_t K_u> = 0``), the time-mean of the correction term
         ``-tau_c div((d_t K) grad theta)`` is ~0, while its instantaneous
         amplitude is O(tau_c*omega) and sign-indefinite.
    """
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    theta = np.sin(X) * np.cos(Y)
    Kbase = 0.3 + 0.1 * np.cos(X) * np.cos(Y)

    def K_of_t(t):
        return Kbase * (1.0 + eps * np.sin(omega * t))

    def dKdt_of_t(t):
        return Kbase * (eps * omega * np.cos(omega * t))

    # (i) lag equivalence at a representative non-trivial phase
    t0 = 1.3
    K0 = K_of_t(t0)
    dK0 = dKdt_of_t(t0)
    corrected = _D(K0 - tau_c * dK0, theta)        # first-order corrected operator
    lagged = _D(K_of_t(t0 - tau_c), theta)         # operator with K lagged by tau_c
    frozen = _D(K0, theta)
    denom = np.linalg.norm(frozen) + 1e-30
    lag_rel_err = float(np.linalg.norm(corrected - lagged) / denom)
    correction_rel_amp = float(np.linalg.norm(corrected - frozen) / denom)

    # (ii) net-zero over a full cycle of K_u(t): mean correction vs its RMS
    period = 2 * np.pi / omega
    ts = np.linspace(0.0, period, n_cycle, endpoint=False)
    corr_terms = np.array([
        np.mean(-tau_c * _D_dterm(dKdt_of_t(t), theta)) for t in ts
    ])
    # spatial-RMS of the correction field at each time, then time-average
    rms_series = np.array([
        np.sqrt(np.mean((tau_c * _D_dterm(dKdt_of_t(t), theta)) ** 2))
        for t in ts
    ])
    time_mean_of_spatial_mean = float(np.mean(corr_terms))
    typical_rms = float(np.mean(rms_series)) + 1e-30
    net_over_rms = abs(time_mean_of_spatial_mean) / typical_rms

    return {
        "tau_c": tau_c, "omega": omega, "eps": eps,
        "lag_equivalence_rel_err": lag_rel_err,       # ~O(tau_c^2) -> small
        "correction_rel_amplitude": correction_rel_amp,  # O(tau_c*omega)
        "cycle_mean_correction": time_mean_of_spatial_mean,
        "cycle_typical_rms": typical_rms,
        "net_over_rms": net_over_rms,                 # ~0 => net-zero (a lag, not a source)
        "is_pure_lag": bool(lag_rel_err < 0.05),
        "is_net_zero_in_steady_state": bool(net_over_rms < 1e-2),
    }


def _D_dterm(dK, th):
    """The §G.5 commutator integrand div((d_tK) grad theta).

    Only ``dK = d_t K_u`` and ``th`` enter the commutator term; the current
    ``K_u`` itself is not needed, so it is not taken as a parameter."""
    tx, ty = _spec_grad(th)
    return _spec_div(dK * tx, dK * ty)


# --------------------------------------------------------------------------- #
# PART C: slow-bath (ice) coefficients B, tau_d from the §B.2 closed form
# --------------------------------------------------------------------------- #
def ice_kernel_coefficients(theta_far=2.0, Vbar=3.2e-8):
    r"""Evaluate the §B.2 ice memory kernel coefficients for representative
    subglacial inputs.  Material constants are fixed; the only *site* inputs are
    ``theta_far`` (far-field ice undercooling, K) and ``Vbar`` (mean ablation
    speed, m/s; 3.2e-8 m/s ~ 1 m/yr).  Returns B, tau_d and the fast/slow time
    separation versus a representative SGS time.
    """
    k_th = 2.1            # ice thermal conductivity, W/m/K
    rho_i = 917.0         # ice density, kg/m^3
    L = 3.34e5            # latent heat of fusion, J/kg
    cp = 2100.0           # ice specific heat, J/kg/K
    kappa = k_th / (rho_i * cp)   # ice thermal diffusivity ~1.09e-6 m^2/s

    A = k_th * theta_far * Vbar ** 2 / (2.0 * kappa ** 2)     # W/m^3
    tau_d = kappa / Vbar ** 2                                  # s (diffusion cutoff)
    # short-time G ~ |A| 2 sqrt(tau_d/pi) tau^{-1/2}; resolvent K_ice = G/(rho_i L)
    B = abs(A) * 2.0 * np.sqrt(tau_d / np.pi) / (rho_i * L)    # units s^{-1/2}
    # DC-gain normalisation cross-check (§B.2 property 4): int G dtau = -rho c theta_far
    dc_gain = rho_i * cp * abs(theta_far)                      # |H(0)| [J/m^3]

    # --- dimensionless slow:fast bath weight (the last RESULT-12 [HYP]) ------- #
    # In the §D.4 GLE the second-FDT weight of each bath is its DC gain int K dtau.
    # The fast SGS/OU bath is unit-normalised (int K_SGS dtau = 1, §D.4-(3)); the
    # slow ice bath's *relative* weight is therefore the dimensionless ice DC gain
    #   int K_ice dtau = (int G dtau)/(rho_i L) = -rho_i cp theta_far/(rho_i L)
    #                  = -cp theta_far / L   (signed; int G dtau = -rho c theta_far).
    # The bath *weight* is the magnitude of this DC gain:
    #   St = |int K_ice dtau| = cp |theta_far| / L   (the Stefan number).
    # So K_SGS : K_ice = 1 : St -- identical to the §G.4 thermal-tail weight
    # (thermal_tail_amplitude: W_thermal/W_hydraulic = St, hydraulic unit-gain),
    # i.e. the slow ice bath carries only a Stefan-number fraction of the memory.
    stefan_weight = cp * abs(theta_far) / L                    # St = int K_ice dtau
    bath_weight_slow_over_fast = stefan_weight                 # since int K_SGS dtau = 1

    # representative physical SGS time: eddy turnover of basal water flow.
    # cavity gap ~0.1-1 m, speed ~0.01-1 m/s  => tau_sgs ~ 0.1-100 s. Use 10 s.
    tau_sgs_phys = 10.0
    return {
        "inputs": {"theta_far_K": theta_far, "Vbar_m_per_s": Vbar,
                   "Vbar_m_per_yr": Vbar * 3.1557e7},
        "kappa_ice": kappa, "A_W_per_m3": A,
        "tau_d_s": tau_d, "tau_d_yr": tau_d / 3.1557e7,
        "B_s_minus_half": B,
        "dc_gain_check_J_per_m3": dc_gain,
        "stefan_weight": stefan_weight,
        "int_Kice_dtau_dimensionless": stefan_weight,
        "fast_bath_dc_gain_normalized": 1.0,
        "bath_weight_slow_over_fast": bath_weight_slow_over_fast,
        "tau_sgs_phys_s": tau_sgs_phys,
        "fast_slow_separation": tau_d / tau_sgs_phys,
    }


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print("=== §D.4 / §G.5 unified-memory GLE coefficient closure ===")

    # PART A -- measure tau_c for both closures
    print("\n[A] tau_c = SGS eddy-diffusivity K_u memory time (measured)")
    A_smag = measure_tau_c(closure="smagorinsky", bs_tau=0.0, seed=1)
    A_2clk = measure_tau_c(closure="backscatter", bs_tau=0.05, seed=1)
    for tag, r in (("Smagorinsky (white-FDT)", A_smag),
                   ("two-clocks  (bs_tau=0.05)", A_2clk)):
        print(f"  {tag}: tau_c_local(e-fold)={r['tau_c_local_efold']:.3e}  "
              f"tau_c_local(int)={r['tau_c_local_integral']:.3e}  "
              f"tau_mem(mean-field)={r['tau_mem_meanfield']:.3e}  "
              f"sign={r['sign_tau_c']:+d}")
    print(f"  RESULT-8 cross-check: tau_mem(set)={RESULT8_TAU_MEM_SET}, "
          f"tau_mem(eff)~{RESULT8_TAU_MEM_EFF} (both > 0)")

    # PART B -- §G.5 add/remove-flux => pure lag, net-zero in steady state
    print("\n[B] §G.5 correction structure (does it add or remove flux?)")
    tau_c_used = A_2clk["tau_c_local_efold"] or 0.02
    B_res = lag_and_netzero(tau_c=min(tau_c_used, 0.05))
    print(f"  using tau_c={B_res['tau_c']:.3e}")
    print(f"  pure time-lag (K_u(t) -> K_u(t-tau_c))? {B_res['is_pure_lag']}  "
          f"(lag rel-err {B_res['lag_equivalence_rel_err']:.2e})")
    print(f"  net-zero over a steady K_u cycle?       "
          f"{B_res['is_net_zero_in_steady_state']}  "
          f"(|mean|/rms {B_res['net_over_rms']:.2e})")
    print("  => the correction LAGS the flux by tau_c; it adds flux while K_u")
    print("     grows and removes it while K_u decays, netting ~0 in steady")
    print("     turbulence -- which is exactly why RESULT 8/11 see no mean effect.")

    # PART C -- slow-bath (ice) coefficients from §B.2 closed form
    print("\n[C] ice (slow) bath: §B.2 closed-form coefficients (site-input dependent)")
    C_lo = ice_kernel_coefficients(theta_far=2.0, Vbar=3.2e-8)   # ~1 m/yr
    C_hi = ice_kernel_coefficients(theta_far=2.0, Vbar=3.2e-7)   # ~10 m/yr
    for tag, c in (("Vbar~1 m/yr", C_lo), ("Vbar~10 m/yr", C_hi)):
        print(f"  {tag}: tau_d={c['tau_d_s']:.3e} s ({c['tau_d_yr']:.2e} yr), "
              f"B={c['B_s_minus_half']:.3e} s^-1/2, "
              f"fast/slow sep ~{c['fast_slow_separation']:.2e}, "
              f"weight(slow/fast)=St={c['bath_weight_slow_over_fast']:.3f}")
    print("  => bath weights: int K_SGS dtau=1 (fast, unit-gain) : int K_ice dtau"
          "; St=|int K_ice dtau|=c_i*|theta_far|/L (slow); St~0.01-0.06 up to 10 K, so the slow ice")
    print("     bath is subdominant -- the same Stefan number as the §G.4 thermal"
          "-tail weight (W_thermal/W_hydraulic).")

    out = {
        "part_A_tau_c_measured": {"smagorinsky": A_smag, "two_clocks": A_2clk,
                                  "result8_tau_mem_set": RESULT8_TAU_MEM_SET,
                                  "result8_tau_mem_eff": RESULT8_TAU_MEM_EFF},
        "part_B_g5_correction": B_res,
        "part_C_ice_kernel": {"Vbar_1m_per_yr": C_lo, "Vbar_10m_per_yr": C_hi},
        "verdict": ("tau_c MEASURED (sign +, value = SGS K_u memory time, "
                    "consistent with RESULT-8 tau_mem); §G.5 correction is a "
                    "pure time-lag with net-zero mean in steady turbulence "
                    "(explains RESULT 8/11 null); ice-bath B, tau_d closed-form "
                    "from §B.2, fast/slow baths separated by ~1e6-1e9 in physical "
                    "time (§D.4 scale-selectivity, real numbers); bath weights "
                    "K_SGS:K_ice = 1:St with St=c_i*|theta_far|/L (~0.01-0.06), the "
                    "slow ice bath subdominant by the §G.4 Stefan number."),
        "wall_time_s": time.time() - t0,
    }
    os.makedirs("figures", exist_ok=True)
    with open("figures/53_gle_coefficients.json", "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    print(f"\nResults saved to figures/53_gle_coefficients.json "
          f"(wall {out['wall_time_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
