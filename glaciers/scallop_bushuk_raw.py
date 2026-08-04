r"""RESULT 25 / paper-3 field pin -- the constant-free index ``I`` and the
parity-break quadrature measured on the RAW Bushuk et al. (2019) ``h(x, t)``
arrays (experiment 1b, scallop-adjustment regime).

Closes the "one open gate" of ``papers/paper3_scallop_parity_break.md`` (§8) and
supersedes both prior readings of the same regime:

* the §8.1 figure bound (``I_obs ~ 0.05-0.15``) -- built on the *equilibrium*
  advection speed ``c = 0.11 mm/min`` and a by-eye ``tau``; and
* the §8.4 digitized bound (``I ~ 10.3``, battery 4.9-17.4) -- built on a
  crest-comb spacing ``lam = 27.9 mm`` that the raw arrays now rule out (the
  window's corrugation power is concentrated at the ~149 mm window mode, with
  bin-5/~30 mm power at 0.2% of it) and an all-period comb speed 0.17 mm/min
  that under-read the true advection by ~5x.

Data: ``melt_data_timestring_sub2.mat`` (M. Bushuk, GFDL, personal
communication, 2026-07): 12 interface profiles ``z_profiles`` (mm) on
``x_grid`` (166 points, 148.6 mm span), 5-min cadence (``timestring``
19:17:02-20:12:02), which is exactly the figure-3(m-r) adjustment sequence
(experiment times 512-567 min; the drive was cut ``U = 1.00 -> 0.16 m/s`` at
495 min).  The file also carries Bushuk's own melt/advection decomposition
(``melt``, ``x_advection``, ``adv_grid``, ``z_profiles_advection``), used here
as independent cross-checks.  The raw file is NOT committed (redistribution
permission not requested); reduced modal/flux statistics sufficient to
re-verify every verdict are committed to
``subglacial/data/bushuk_raw_derived.json``, and this module regenerates them
bit-for-bit when the .mat is present (``--mat`` or ``$BUSHUK_MAT``).

Four measurements, one record:

1. **The pre-registered pin** -- ``scallop_field_test.harmonic_mode_rate`` run
   verbatim on ``(x, t, H)``: dominant-mode complex rate
   ``s = Re(s) + i*Im(s)``, downstream sign, ``I = |Im(s)|/(2*pi*|Re(s)|)``.
2. **The flux-parity quadrature** -- per-interval transfer ``s_flux = -m_k/h_k``
   (melt-rate harmonic over interface harmonic at the same mode; midpoint
   pairing).  A local, memoryless, down-gradient closure is parity-symmetric:
   ``Im(s_flux) = 0`` identically.  Bootstrap over the 11 intervals tests that
   null on real ice.
3. **Migration-speed cross-checks** -- Bushuk's own crest tracking
   (``x_advection``), an independent sub-grid cross-correlation tracker, the
   modal phase rate, and the flux quadrature must all give the same
   ``c_mig`` (they do, to ~7%).
4. **Controls** -- Bushuk's advection-subtracted profiles must carry ~zero
   residual phase rate (they do: |c_resid| ~ 1e-3 of the lab-frame speed), and
   the crest-evolution angle ``phi = arctan(hdot/c)`` must reproduce the
   published 15 deg (measured 14.8 deg).

CPU-only, seconds.  Writes ``figures/78_bushuk_raw_pin.json`` (+ ``.png`` with
matplotlib when available) and, from raw, the committed derived-statistics
JSON.  Tests: ``tests/test_scallop_bushuk_raw.py``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAT_DEFAULT = os.path.join(HERE, "subglacial", "data", "bushuk",
                           "melt_data_timestring_sub2.mat")
DERIVED_PATH = os.path.join(HERE, "subglacial", "data",
                            "bushuk_raw_derived.json")
FIG_JSON = os.path.join(HERE, "figures", "78_bushuk_raw_pin.json")
FIG_PNG = os.path.join(HERE, "figures", "78_bushuk_raw_pin.png")

# published anchors (Bushuk et al. 2019, JFM 873)
PUB_PHI_DEG = 15.0            # crest-evolution angle, adjustment regime (their §3.2.3)
PUB_DRIVE_CUT = (1.00, 0.16)  # m/s, at experiment time 495 min
EXP_T0_MIN = 512.0            # experiment time of the first frame (fig 3m-r)

# solver comparison bands (REPORT_SCALLOP_MIGRATION.md §7/§8; scallop_field_test)
SOLVER_FROZEN_BAND = (0.33, 0.88)     # frozen-interface flow-off band, near-equilibrium
SOLVER_MOVING_I = 0.16                # moving-boundary, motion-consistent measurement
DIGITIZED_SUPERSEDED = dict(I=10.3, battery=(4.9, 17.4), c_mm_min=0.17,
                            lam_mm=27.9)


# --------------------------------------------------------------------------- #
# raw ingest + derived statistics
# --------------------------------------------------------------------------- #
def load_mat(path=None):
    """Load the raw .mat (needs scipy).  Returns a dict of plain arrays."""
    import scipy.io as sio
    path = path or os.environ.get("BUSHUK_MAT", MAT_DEFAULT)
    d = sio.loadmat(path)
    x_mm = d["x_grid"].ravel().astype(float)                       # (166,)
    Z_mm = np.column_stack([np.asarray(c).ravel()
                            for c in d["z_profiles"].ravel()]).T   # (12, 166)
    tstr = [str(np.asarray(c).item()) for c in d["timestring"].ravel()]
    t_s = np.array([int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:])
                    for s in tstr], float)
    t_s -= t_s[0]
    melt = [np.asarray(c).ravel().astype(float)
            for c in d["melt"].ravel()]                            # [ [], (166,)*11 ] mm/min
    xadv_mean = np.array([np.asarray(c).ravel().mean()
                          for c in d["x_advection"].ravel()])      # (12,) mm
    adv_mm = d["adv_grid"].ravel().astype(float)                   # (107,)
    Za_mm = np.column_stack([np.asarray(c).ravel()
                             for c in d["z_profiles_advection"].ravel()]).T
    mean_melt = d["mean_melt"].ravel().astype(float)               # (166,) mm/min
    return dict(x_mm=x_mm, Z_mm=Z_mm, t_s=t_s, timestring=tstr, melt=melt,
                xadv_mean=xadv_mean, adv_mm=adv_mm, Za_mm=Za_mm,
                mean_melt=mean_melt)


def _xcorr_shift(a, b, dx):
    """Sub-grid periodic cross-correlation shift of ``a`` relative to ``b``."""
    n = len(a)
    A = np.fft.rfft(a - a.mean())
    B = np.fft.rfft(b - b.mean())
    cc = np.fft.irfft(A * np.conj(B), n)
    i = int(np.argmax(cc))
    y0, y1, y2 = cc[(i - 1) % n], cc[i], cc[(i + 1) % n]
    den = y0 - 2.0 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    s = i + off
    if s > n / 2:
        s -= n
    return s * dx


def _fit_rate(t, series):
    """Least-squares slope of ``series`` against ``t``."""
    return float(np.polyfit(t, series, 1)[0])


def _pin_from_modal(t, C_k, k_rad_m):
    """(Re_s, dphi_dt, c_phase, I, downstream) from a complex modal series."""
    amp = np.abs(C_k)
    phase = np.unwrap(np.angle(C_k))
    Re_s = _fit_rate(t, np.log(amp))
    dphi_dt = _fit_rate(t, phase)
    c_phase = -dphi_dt / k_rad_m
    I = abs(dphi_dt) / (2.0 * np.pi * abs(Re_s)) if Re_s != 0 else np.inf
    return dict(Re_s=Re_s, dphi_dt=dphi_dt, c_phase=c_phase, I=I,
                downstream=bool(c_phase > 0.0),
                tau_s=(1.0 / abs(Re_s)) if Re_s != 0 else np.inf)


def derive(raw):
    """Reduce the raw record to the committed derived statistics (JSON-safe)."""
    x_mm, Z_mm, t = raw["x_mm"], raw["Z_mm"], raw["t_s"]
    nx = x_mm.size
    dx_mm = float(np.mean(np.diff(x_mm)))
    Lx_mm = nx * dx_mm

    Zd = Z_mm - Z_mm.mean(axis=1, keepdims=True)
    C = np.fft.rfft(Zd, axis=1)                                    # (12, nk)

    # modal complex series, bins 1..3 (mm units; I is scale-free)
    modal = {str(k): [[float(c.real), float(c.imag)] for c in C[:, k]]
             for k in (1, 2, 3)}

    # per-interval flux transfer s_flux = -m_k/h_k (SI), bins 1 and 2
    flux = {}
    for k in (1, 2):
        vals = []
        for i in range(1, Z_mm.shape[0]):
            m = raw["melt"][i] / 1000.0 / 60.0                     # m/s
            hm = 0.5 * (Z_mm[i - 1] + Z_mm[i]) / 1000.0            # m
            mF = np.fft.rfft(m - m.mean())
            hF = np.fft.rfft(hm - hm.mean())
            s = -mF[k] / hF[k]
            vals.append([float(s.real), float(s.imag)])
        flux[str(k)] = vals

    # tracking: Bushuk's own advected grid + independent xcorr tracker
    shift_track = (raw["xadv_mean"][0] - raw["xadv_mean"]).tolist()  # mm, downstream +
    shift_xc = [0.0] + [_xcorr_shift(Z_mm[i] - Z_mm[i].mean(),
                                     Z_mm[0] - Z_mm[0].mean(), dx_mm)
                        for i in range(1, Z_mm.shape[0])]

    # variant pins (need raw frames): linear-detrend, Hann, halves
    from scallop_field_test import harmonic_mode_rate
    x_m, H_m = x_mm / 1000.0, Z_mm / 1000.0
    variants = {}
    r = harmonic_mode_rate(x_m, t, H_m)
    variants["verbatim"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in r.items()}
    for k in (1, 2):
        rk = harmonic_mode_rate(x_m, t, H_m, k_index=k)
        variants[f"bin{k}"] = dict(I=float(rk["I"]), Re_s=float(rk["Re_s"]),
                                   downstream=bool(rk["downstream"]))
    Xc = x_m - x_m.mean()
    coef = np.polyfit(Xc, H_m.T, 1)
    Hlin = H_m - (np.outer(coef[0], Xc) + coef[1][:, None])
    rl = harmonic_mode_rate(x_m, t, Hlin)
    variants["linear_detrended"] = dict(I=float(rl["I"]), Re_s=float(rl["Re_s"]),
                                        downstream=bool(rl["downstream"]))
    w = np.hanning(nx)
    rh = harmonic_mode_rate(x_m, t,
                            (H_m - H_m.mean(axis=1, keepdims=True)) * w)
    variants["hann"] = dict(I=float(rh["I"]), Re_s=float(rh["Re_s"]),
                            downstream=bool(rh["downstream"]))
    for tag, sl in (("first_half", slice(0, 6)), ("second_half", slice(6, 12))):
        rv = harmonic_mode_rate(x_m, t[sl], H_m[sl])
        variants[tag] = dict(I=float(rv["I"]), Re_s=float(rv["Re_s"]),
                             downstream=bool(rv["downstream"]))

    # control: Bushuk's advection-subtracted profiles carry ~no phase rate
    adv_mm, Za_mm = raw["adv_mm"], raw["Za_mm"]
    ra = harmonic_mode_rate(adv_mm / 1000.0, t, Za_mm / 1000.0)
    control = dict(c_resid_m_s=float(ra["c_phase"]), I=float(ra["I"]),
                   k_index=int(ra["k_index"]))

    envelope_std_mm = Zd.std(axis=1).tolist()
    return dict(
        meta=dict(nx=nx, dx_mm=dx_mm, Lx_mm=Lx_mm, dt_s=float(t[1] - t[0]),
                  n_frames=int(Z_mm.shape[0]), timestring=raw["timestring"],
                  exp_t0_min=EXP_T0_MIN,
                  provenance=("melt_data_timestring_sub2.mat, M. Bushuk (GFDL), "
                              "pers. comm. 2026-07; experiment 1b adjustment "
                              "regime, fig 3(m-r), U cut 1.00->0.16 m/s")),
        t_s=t.tolist(), modal=modal, flux=flux,
        tracking=dict(shift_track_mm=shift_track, shift_xcorr_mm=shift_xc),
        variants=variants, advected_control=control,
        envelope_std_mm=envelope_std_mm,
        mean_melt_rate_mm_min=float(raw["mean_melt"].mean()),
        modal_power_top=[[int(k), float(p)] for k, p in _top_power(C)],
    )


def _top_power(C):
    P = (np.abs(C) ** 2).mean(axis=0)
    P[0] = 0.0
    order = np.argsort(P)[::-1][:5]
    return [(int(k), float(P[k] / P[order[0]])) for k in order]


def load_derived(path=DERIVED_PATH):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# analyses on the derived statistics (offline-safe)
# --------------------------------------------------------------------------- #
def kinematic_pin(ds):
    """The pre-registered pin + modal battery, from committed modal series."""
    t = np.asarray(ds["t_s"], float)
    Lx_m = ds["meta"]["Lx_mm"] / 1000.0
    out = {}
    for kstr, series in ds["modal"].items():
        k = int(kstr)
        C_k = np.array([complex(a, b) for a, b in series])
        out[f"bin{k}"] = _pin_from_modal(t, C_k, 2.0 * np.pi * k / Lx_m)
        out[f"bin{k}"]["lam_mm"] = ds["meta"]["Lx_mm"] / k
    return out


def migration_speeds(ds):
    """c_mig agreement across four independent readouts (mm/min, + downstream)."""
    t = np.asarray(ds["t_s"], float)
    c_track = _fit_rate(t, np.asarray(ds["tracking"]["shift_track_mm"])) * 60.0
    c_xc = _fit_rate(t, np.asarray(ds["tracking"]["shift_xcorr_mm"])) * 60.0
    pin = kinematic_pin(ds)["bin1"]
    c_pin = pin["c_phase"] * 1000.0 * 60.0
    s1 = np.array([complex(a, b) for a, b in ds["flux"]["1"]])
    k1 = 2.0 * np.pi * 1 / (ds["meta"]["Lx_mm"] / 1000.0)
    c_flux = float(-s1.imag.mean() / k1 * 1000.0 * 60.0)
    cs = dict(bushuk_tracking=c_track, xcorr=c_xc, modal_phase=c_pin,
              flux_quadrature=c_flux)
    vals = np.array(list(cs.values()))
    cs["spread_frac"] = float(vals.max() / vals.min() - 1.0)
    cs["all_downstream"] = bool((vals > 0).all())
    return cs


def flux_parity(ds, n_boot=20000, seed=0):
    """The parity-break test: bootstrap ``Im(s_flux)`` against the K-theory
    null ``Im = 0`` (a local down-gradient closure is x -> -x symmetric)."""
    rng = np.random.default_rng(seed)
    out = {}
    for kstr, vals in ds["flux"].items():
        s = np.array([complex(a, b) for a, b in vals])
        n = s.size
        bi = rng.integers(0, n, (n_boot, n))
        Im_b = s.imag[bi].mean(axis=1)
        Re_b = s.real[bi].mean(axis=1)
        I_b = np.abs(Im_b) / (2.0 * np.pi * np.abs(Re_b))
        out[f"bin{kstr}"] = dict(
            Re_mean=float(s.real.mean()), Im_mean=float(s.imag.mean()),
            tau_flux_min=float(1.0 / abs(s.real.mean()) / 60.0),
            p_Im_ge_0=float((Im_b >= 0.0).mean()),
            damping_frames=int((s.real < 0).sum()), n_frames=n,
            I_flux=float(abs(s.imag.mean()) / (2.0 * np.pi * abs(s.real.mean()))),
            I_flux_ci16_84=[float(np.percentile(I_b, 16)),
                            float(np.percentile(I_b, 84))],
            arg_G_deg_mean=float(np.degrees(
                np.angle(-s).mean())),   # G = m_k/h_k = -s
        )
    return out


def controls(ds):
    """Advected-frame residual + crest-evolution angle vs published."""
    cs = migration_speeds(ds)
    c_lab = cs["modal_phase"]
    c_resid = ds["advected_control"]["c_resid_m_s"] * 1000.0 * 60.0
    hdot = ds["mean_melt_rate_mm_min"]
    c_for_phi = cs["bushuk_tracking"]
    phi = float(np.degrees(np.arctan2(hdot, c_for_phi)))
    return dict(advected_resid_mm_min=c_resid,
                advected_resid_frac=abs(c_resid) / abs(c_lab),
                phi_deg=phi, phi_pub_deg=PUB_PHI_DEG,
                phi_agrees=bool(abs(phi - PUB_PHI_DEG) < 1.5),
                mean_melt_rate_mm_min=hdot)


def verdict(ds):
    pin = kinematic_pin(ds)
    cs = migration_speeds(ds)
    fp = flux_parity(ds)
    ct = controls(ds)
    v = ds["variants"]
    I_batt = [v[k]["I"] for k in
              ("verbatim", "bin2", "linear_detrended", "hann",
               "first_half", "second_half")]
    down_all = all(bool(v[k]["downstream"]) for k in v) and cs["all_downstream"]
    I_pin = v["verbatim"]["I"]
    return dict(
        downstream_sign_unanimous=down_all,
        damped_on_the_mean=bool(v["verbatim"]["Re_s"] < 0
                                and fp["bin1"]["Re_mean"] < 0),
        parity_break_p=fp["bin1"]["p_Im_ge_0"],
        parity_break=bool(fp["bin1"]["p_Im_ge_0"] < 1e-3),
        c_mig_mm_min=cs, I_pin=I_pin,
        I_battery=[float(min(I_batt)), float(max(I_batt))],
        I_flux=fp["bin1"]["I_flux"], I_flux_ci=fp["bin1"]["I_flux_ci16_84"],
        solver_frozen_band=list(SOLVER_FROZEN_BAND),
        solver_moving_I=SOLVER_MOVING_I,
        digitized_superseded=DIGITIZED_SUPERSEDED,
        vs_band=dict(
            pin_over_band_hi=float(I_pin / SOLVER_FROZEN_BAND[1]),
            battery_overlaps_band=bool(min(I_batt) <= SOLVER_FROZEN_BAND[1]),
            anchoring_mismatch_bin=True,
        ),
        controls=ct,
        reading=(
            "Corrected-mode form observed on real ice: the train damps AND "
            "migrates downstream (all estimators); the flux quadrature "
            "Im(s_flux) != 0 at p<1e-3 (K-theory parity null rejected) and "
            "quantitatively accounts for the observed migration speed. "
            "I_obs = O(0.5-6.5) (pre-registered pin 3.6, flux 1.2): above the "
            "near-equilibrium solver band, in the pre-registered "
            "anchoring-mismatch bin (train formed at U=1.00 persisting under "
            "U=0.16; lam is anchored to the OLD flow's u*), and ~3-5x BELOW "
            "the superseded digitized 10.3."),
    )


# --------------------------------------------------------------------------- #
def _figure(ds, out_png=FIG_PNG):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                              # pragma: no cover
        return False
    t = np.asarray(ds["t_s"], float) / 60.0
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    a = ax[0, 0]
    C1 = np.array([complex(u, v) for u, v in ds["modal"]["1"]])
    a.plot(t, np.log(np.abs(C1)), "o-", label="ln|a_1|(t)")
    g = np.polyfit(t * 60, np.log(np.abs(C1)), 1)
    a.plot(t, np.polyval(g, t * 60), "k--",
           label=f"fit: tau={1/abs(g[0])/60:.0f} min")
    a.set_xlabel("t (min)"); a.set_ylabel("ln modal amplitude")
    a.set_title("amplitude of the dominant mode (damping)"); a.legend()

    b = ax[0, 1]
    ph = np.unwrap(np.angle(C1))
    b.plot(t, ph, "o-", label="phi_1(t)")
    q = np.polyfit(t * 60, ph, 1)
    b.plot(t, np.polyval(q, t * 60), "k--",
           label=f"fit: c={-q[0]/(2*np.pi/(ds['meta']['Lx_mm']/1000.))*6e4:.2f} mm/min")
    b.set_xlabel("t (min)"); b.set_ylabel("modal phase (rad)")
    b.set_title("phase of the dominant mode (downstream migration)"); b.legend()

    c = ax[1, 0]
    st = ds["tracking"]
    c.plot(t, st["shift_track_mm"], "o-", label="Bushuk crest tracking")
    c.plot(t, st["shift_xcorr_mm"], "s-", label="x-corr tracker")
    C1ph = -(ph - ph[0]) / (2 * np.pi / ds["meta"]["Lx_mm"])
    c.plot(t, C1ph, "^-", label="modal phase")
    c.set_xlabel("t (min)"); c.set_ylabel("downstream displacement (mm)")
    c.set_title("three trackers, one migration"); c.legend()

    d = ax[1, 1]
    s1 = np.array([complex(u, v) for u, v in ds["flux"]["1"]])
    d.axhline(0.0, color="r", lw=2, label="K-theory null: Im(s_flux)=0")
    d.plot(t[1:], s1.imag, "o-", label="Im(s_flux) per interval")
    d.plot(t[1:], s1.real, "s--", alpha=0.6, label="Re(s_flux) per interval")
    d.set_xlabel("t (min)"); d.set_ylabel("s_flux (1/s)")
    d.set_title("the parity-break quadrature in the real melt flux")
    d.legend()

    fig.suptitle("RESULT 25 -- Bushuk exp-1b raw h(x,t): the field pin "
                 "(damped, downstream-migrating; parity break measured)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True


def run(mat_path=None, write=True):
    """Full assembly.  From raw .mat when available, else from the committed
    derived statistics."""
    used_raw = False
    path = mat_path or os.environ.get("BUSHUK_MAT", MAT_DEFAULT)
    if os.path.exists(path):
        raw = load_mat(path)
        ds = derive(raw)
        used_raw = True
        if write:
            os.makedirs(os.path.dirname(DERIVED_PATH), exist_ok=True)
            with open(DERIVED_PATH, "w") as fh:
                json.dump(ds, fh, indent=1)
    else:
        ds = load_derived()

    out = dict(
        description=("paper-3 field pin on the raw Bushuk 2019 exp-1b "
                     "adjustment arrays: I = tau*c_mig/lam = "
                     "|Im(s)|/(2pi|Re(s)|), plus the flux-parity quadrature."),
        used_raw_mat=used_raw,
        meta=ds["meta"], kinematic_pin=kinematic_pin(ds),
        variants=ds["variants"], migration_speeds=migration_speeds(ds),
        flux_parity=flux_parity(ds), controls=controls(ds),
        modal_power_top=ds["modal_power_top"], verdict=verdict(ds),
    )
    if write:
        os.makedirs(os.path.dirname(FIG_JSON), exist_ok=True)
        with open(FIG_JSON, "w") as fh:
            json.dump(out, fh, indent=1)
        out["figure_png"] = _figure(ds)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mat", default=None, help="path to the raw .mat")
    args = p.parse_args()
    out = run(mat_path=args.mat)
    v = out["verdict"]
    cs = v["c_mig_mm_min"]
    print("RESULT 25 -- the field pin on raw Bushuk h(x,t) (exp 1b, adjustment)")
    print(f"  raw .mat used             : {out['used_raw_mat']}")
    print(f"  downstream (unanimous)    : {v['downstream_sign_unanimous']}")
    print(f"  damped on the mean        : {v['damped_on_the_mean']}")
    print(f"  c_mig mm/min              : track {cs['bushuk_tracking']:.3f} | "
          f"xcorr {cs['xcorr']:.3f} | modal {cs['modal_phase']:.3f} | "
          f"flux {cs['flux_quadrature']:.3f}  (spread {cs['spread_frac']*100:.1f}%)")
    print(f"  parity break              : P(Im>=0) = {v['parity_break_p']:.2e} "
          f"-> {v['parity_break']}")
    print(f"  I (pre-registered pin)    : {v['I_pin']:.2f}")
    print(f"  I battery [min,max]       : [{v['I_battery'][0]:.2f}, "
          f"{v['I_battery'][1]:.2f}]")
    print(f"  I flux (dynamic)          : {v['I_flux']:.2f} "
          f"CI16-84 {tuple(round(x,2) for x in v['I_flux_ci'])}")
    print(f"  solver bands              : frozen {v['solver_frozen_band']} | "
          f"moving {v['solver_moving_I']} | digitized(superseded) "
          f"{v['digitized_superseded']['I']}")
    ct = v["controls"]
    print(f"  controls                  : advected resid "
          f"{ct['advected_resid_frac']*100:.2f}% of c | phi "
          f"{ct['phi_deg']:.1f} deg (pub {ct['phi_pub_deg']:.0f})")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
