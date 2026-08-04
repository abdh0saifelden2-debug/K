r"""NR49 (theory) -- the frame-pair transfer of an evolving interface
factorizes into gain (growth clock, Galilean-invariant) x all-pass
(transport clock); a Galilean boost is EXACTLY the spatial all-pass factor,
the k-domain twin of NR48's delay.

Follows NR48 (minimum-phase/all-pass factorisation in omega: relaxational
memory is amplitude-determined, a transport delay is the unique all-pass
factor e^{-i omega tau_d}).  NR49 transfers the theorem to the conjugate
variable: for a field h(x,t) observed as snapshots, the two-snapshot
transfer in wavenumber k factorizes into gain x all-pass, and the all-pass
factor IS kinematic transport (Galilean boost / migration), while the gain
IS dynamics (growth/damping) that no boost can create or remove.

The claim
=========
1. **Evolution operator.**  Any linear interface evolution ``d h_hat/dt =
   lambda(k) h_hat`` with ``lambda = sigma - i k c`` gives the frame-pair
   transfer ``T(k) = h_hat(t+Dt)/h_hat(t) = e^{lambda(k) Dt}``:

       ``|T| = e^{sigma Dt}``  (gain -> growth clock),
       ``arg T = -k c Dt``    (phase -> transport clock).

2. **The boost identity (exact).**  Under a Galilean boost ``x -> x - V t``
   the spectrum maps ``h_hat -> h_hat e^{i k V t}``, i.e.

       ``lambda(k) -> lambda(k) + i k V``.

   The growth spectrum ``sigma(k) = Re lambda`` is boost-INVARIANT; the
   celerity ``c(k) = -Im lambda / k`` shifts by ``-V``... a boost is a pure
   all-pass factor ``e^{i k V Dt}`` (|T| = 1, phase linear in k).  This is
   NR48's delay ``e^{-i omega tau_d}`` with ``omega tau_d <-> -k V Dt``:
   delay in time <-> displacement in space.  Conversely: gain structure
   (damping/growth) can NEVER produce a k-linear phase, and no boost can
   change |T| -- growth and transport separate exactly, per snapshot pair.
   (Aliasing bound: the phase is principal-valued, so the celerity is
   readable only where ``|k V Dt| < pi`` -- displacement per frame under
   half a wavelength, the same Nyquist condition every phase tracker has.)

3. **Measurement method.**  ``lambda_hat(k) = ln T(k) / Dt`` from ANY two
   snapshots; robust (power-weighted median) over successive pairs gives
   the full complex dispersion relation of the interface: sigma(k) = the
   linear-stability fingerprint, c(k) = the migration dispersion --
   measured model-free from data that only reports interface heights.

Findings (figures/84_dispersion_boost.json), real Bushuk h(x,t) (12 frames,
Dt = 300 s, committed .mat, glaciers/subglacial/data/bushuk/):

* **Migration is one all-pass factor.**  Lab-frame pattern-band (top modal
  wavelengths >= 45 mm) phase celerity = 48.3 mm/hr, agreeing with the two
  committed trackers (55.8 xcorr / 57.2 Bushuk advection) to ~14 % (Hann
  window + pattern evolution bias the spectral estimate low).  In Bushuk's
  co-moving (advected) frame the same estimator gives 3.4 mm/hr -- the
  boost annihilates 93 % of the phase clock, as an all-pass factor must.
* **The gain is Galilean-invariant.**  Pattern growth sigma = -0.06 /hr
  (lab) vs -0.45 /hr (advected); roughness-band (15-45 mm) damping
  -4.7 /hr (lab) vs -5.9 /hr (advected): same sign, same scale, in both
  frames -- boost moves the phase, never the gain.
* **Marginal pattern, damped roughness (NR44's attractor, spectrally).**
  |sigma_pattern| < 0.5 /hr while sigma_roughness ~ -5 /hr (lifetimes of
  minutes): the developed scallop field is a saturated, rigidly migrating
  state -- the pattern band sits at the marginal-stability fixed point
  while sub-pattern roughness is strongly relaxational.

Consequence for the program: NR48's factorisation is not just spectroscopy
-- it is the general kinematics/dynamics splitter for ANY evolving field.
The transport clock (all-pass; NR2/NR4's migration, NR48's delay, this
boost) and the memory/growth clock (gain; NR45's modulus, NR47's memory
phase) are the same two clocks in every conjugate pair (omega <-> k).

CPU-only, offline-safe (the .mat and the tracker cache are committed).
Tests: tests/test_dispersion_boost.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MAT = os.path.join(REPO, "glaciers", "subglacial", "data", "bushuk",
                   "melt_data_timestring_sub2.mat")
DERIVED = os.path.join(REPO, "glaciers", "subglacial", "data",
                       "bushuk_raw_derived.json")
FIG = os.path.join(HERE, "figures", "84_dispersion_boost.json")

DT_S = 300.0
PATTERN_MM = (45.0, 200.0)      # top modal band (committed modal_power_top)
ROUGH_MM = (15.0, 45.0)         # sub-pattern roughness


# --------------------------------------------------------------------------- #
# estimator: complex dispersion from snapshot pairs
# --------------------------------------------------------------------------- #
def frame_spectra(frames, dx_mm):
    """Hann-windowed spectra of mean-and-trend-removed snapshots."""
    F = np.asarray(frames, float)
    n = F.shape[1]
    W = np.hanning(n)
    i = np.arange(n)
    H = []
    for h in F:
        h = h - h.mean()
        cc = np.polyfit(i, h, 1)
        H.append(np.fft.rfft((h - cc[0] * i - cc[1]) * W))
    return np.array(H), 2.0 * np.pi * np.fft.rfftfreq(n, dx_mm)


def pair_dispersion(H, k, dt_s=DT_S):
    """lambda_hat(k) per successive pair: sigma (1/hr), celerity (mm/hr),
    and the |h|^2 weights of the source frame."""
    T = H[1:] / H[:-1]
    sig = np.log(np.abs(T)) / dt_s * 3600.0
    cel = -np.angle(T) / (np.maximum(k, 1e-12)[None, :] * dt_s) * 3600.0
    return sig, cel, np.abs(H[:-1]) ** 2


def weighted_median(a, w):
    a = np.asarray(a, float).ravel()
    w = np.asarray(w, float).ravel()
    o = np.argsort(a)
    a, w = a[o], w[o]
    cw = np.cumsum(w)
    return float(a[np.searchsorted(cw, 0.5 * cw[-1])])


def snr_gate(H, k, hi_k=2.0, factor=30.0):
    P = np.mean(np.abs(H) ** 2, 0)
    return P > factor * np.median(P[k > hi_k])


def band_indices(k, gate, lo_mm, hi_mm):
    lam = np.where(k > 0, 2.0 * np.pi / np.maximum(k, 1e-12), np.inf)
    return np.where(gate & (lam >= lo_mm) & (lam <= hi_mm))[0]


def band_stats(sig, cel, w, k, idx):
    return dict(
        sigma_hr=weighted_median(sig[:, idx], w[:, idx]),
        celerity_mm_hr=weighted_median(cel[:, idx],
                                       w[:, idx] * k[idx][None, :] ** 2),
        n_bins=int(len(idx)))


# --------------------------------------------------------------------------- #
# synthetic exactness checks
# --------------------------------------------------------------------------- #
def synthetic_checks(seed=0):
    """Exact boost identity on a periodic synthetic field: sigma invariant,
    c = V recovered to machine precision; pure damping has zero fitted
    celerity; a pure boost has |T| = 1."""
    rng = np.random.default_rng(seed)
    n, dx, dt = 256, 1.0, 1.0
    k = 2.0 * np.pi * np.fft.rfftfreq(n, dx)
    H0 = np.fft.rfft(rng.standard_normal(n))
    nu, V = 0.02, 3.0
    lam_true = -nu * k ** 2 - 1j * k * V
    nfr = 6
    H = np.array([H0 * np.exp(lam_true * dt * j) for j in range(nfr)])
    sig, cel, w = pair_dispersion(H, k, dt_s=dt)
    m = (k > 0.1) & (k < 2.0)
    mc = (k > 0.1) & (k * V * dt < 0.9 * np.pi)   # phase-Nyquist bound
    sig_err = float(np.max(np.abs(
        np.median(sig, 0)[m] / 3600.0 - (-nu * k[m] ** 2))))
    cel_err = float(np.max(np.abs(np.median(cel, 0)[mc] / 3600.0 - V)))
    # boosted frame: remove the boost exactly -> celerity 0, sigma unchanged
    Hb = H * np.exp(1j * k[None, :] * V * dt * np.arange(nfr)[:, None])
    sb, cb, wb = pair_dispersion(Hb, k, dt_s=dt)
    sig_inv = float(np.max(np.abs((np.median(sb, 0) - np.median(sig, 0))[m])))
    cel_resid = float(np.max(np.abs(np.median(cb, 0)[m]))) / 3600.0
    # pure damping: no phase; pure boost: no gain
    Hd = np.array([H0 * np.exp(-nu * k ** 2 * dt * j) for j in range(nfr)])
    _, cd, _ = pair_dispersion(Hd, k, dt_s=dt)
    damp_cel = float(np.max(np.abs(np.median(cd, 0)[m]))) / 3600.0
    Hp = np.array([H0 * np.exp(-1j * k * V * dt * j) for j in range(nfr)])
    sp, _, _ = pair_dispersion(Hp, k, dt_s=dt)
    boost_gain = float(np.max(np.abs(np.median(sp, 0)[m]))) / 3600.0
    return dict(sigma_max_err=sig_err, celerity_max_err=cel_err,
                boost_sigma_invariance=sig_inv,
                boost_residual_celerity=cel_resid,
                damping_fitted_celerity=damp_cel,
                pure_boost_fitted_gain=boost_gain, V_true=V)


# --------------------------------------------------------------------------- #
# real data (committed .mat + committed trackers)
# --------------------------------------------------------------------------- #
def load_bushuk(path=MAT):
    import scipy.io as sio
    d = sio.loadmat(path)
    x_mm = d["x_grid"].ravel().astype(float)
    Z = np.column_stack([np.asarray(c).ravel()
                         for c in d["z_profiles"].ravel()]).T
    Za = np.column_stack([np.asarray(c).ravel()
                          for c in d["z_profiles_advection"].ravel()]).T
    tstr = [str(np.asarray(c).item()) for c in d["timestring"].ravel()]
    t_s = np.array([int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:])
                    for s in tstr], float)
    t_s -= t_s[0]
    xadv = np.array([np.asarray(c).ravel().mean()
                     for c in d["x_advection"].ravel()])
    return dict(x_mm=x_mm, Z_lab=Z, Z_adv=Za, t_s=t_s, xadv_mm=xadv)


def committed_trackers(path=DERIVED):
    with open(path) as fh:
        d = json.load(fh)
    t_h = np.array(d["t_s"]) / 3600.0
    return dict(
        v_xcorr_mm_hr=float(np.polyfit(t_h, d["tracking"]["shift_xcorr_mm"],
                                       1)[0]),
        v_track_mm_hr=float(np.polyfit(t_h, d["tracking"]["shift_track_mm"],
                                       1)[0]))


def analyze():
    syn = synthetic_checks()
    raw = load_bushuk()
    trk = committed_trackers()
    dx = float(np.mean(np.diff(raw["x_mm"])))
    boost = float(np.polyfit(raw["t_s"] / 3600.0, raw["xadv_mm"], 1)[0])

    frames = {}
    for name, Z in (("lab", raw["Z_lab"]), ("adv", raw["Z_adv"])):
        H, k = frame_spectra(Z, dx)
        sig, cel, w = pair_dispersion(H, k)
        g = snr_gate(H, k)
        frames[name] = dict(
            pattern=band_stats(sig, cel, w, k,
                               band_indices(k, g, *PATTERN_MM)),
            rough=band_stats(sig, cel, w, k, band_indices(k, g, *ROUGH_MM)),
            n_gated=int(g.sum()))

    c_lab = frames["lab"]["pattern"]["celerity_mm_hr"]
    c_adv = frames["adv"]["pattern"]["celerity_mm_hr"]
    out = dict(
        synthetic=syn, frames=frames,
        boost_speed_mm_hr=boost, trackers=trk,
        celerity_ratio_lab_vs_xcorr=c_lab / trk["v_xcorr_mm_hr"],
        adv_residual_fraction=abs(c_adv) / abs(c_lab))
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    syn = out["synthetic"]
    fr = out["frames"]
    boost_exact = (syn["sigma_max_err"] < 1e-9
                   and syn["celerity_max_err"] < 1e-9
                   and syn["boost_sigma_invariance"] < 1e-9
                   and syn["boost_residual_celerity"] < 1e-9
                   and syn["damping_fitted_celerity"] < 1e-9
                   and syn["pure_boost_fitted_gain"] < 1e-9)
    allpass = (0.7 < out["celerity_ratio_lab_vs_xcorr"] < 1.3
               and out["adv_residual_fraction"] < 0.2)
    s_pat_l = fr["lab"]["pattern"]["sigma_hr"]
    s_pat_a = fr["adv"]["pattern"]["sigma_hr"]
    s_ro_l = fr["lab"]["rough"]["sigma_hr"]
    s_ro_a = fr["adv"]["rough"]["sigma_hr"]
    gain_inv = (abs(s_pat_l - s_pat_a) < 1.5
                and abs(s_ro_l - s_ro_a) < 3.0
                and s_ro_l < -3.0 and s_ro_a < -3.0)
    attractor = (abs(s_pat_l) < 1.0 and abs(s_pat_a) < 1.0
                 and s_ro_l < -3.0 and s_ro_a < -3.0)
    return dict(
        boost_identity_exact=bool(boost_exact),
        migration_is_one_allpass_factor=bool(allpass),
        gain_is_boost_invariant=bool(gain_inv),
        marginal_pattern_damped_roughness=bool(attractor),
        reading=(
            "The two-snapshot transfer factorizes exactly into gain x "
            "all-pass, and the all-pass factor is kinematics: on the real "
            "scallop field the pattern-band phase celerity "
            f"({fr['lab']['pattern']['celerity_mm_hr']:.1f} mm/hr) matches "
            f"the committed trackers ({out['trackers']['v_xcorr_mm_hr']:.1f}"
            " xcorr), and Bushuk's co-moving frame annihilates "
            f"{100 * (1 - out['adv_residual_fraction']):.0f}% of it, while "
            "the growth spectrum stays put (pattern "
            f"{s_pat_l:+.2f} vs {s_pat_a:+.2f} /hr; roughness "
            f"{s_ro_l:+.1f} vs {s_ro_a:+.1f} /hr) -- boosts move phase, "
            "never gain.  The developed scallop state is a marginal pattern "
            "riding on minutes-lifetime damped roughness: NR44's attractor, "
            "seen spectrally.  NR48's delay and this boost are the same "
            "all-pass clock in conjugate variables (omega <-> k)."),
    )


def run(write=True):
    out = analyze()
    res = dict(description=__doc__.splitlines()[0],
               source=dict(mat=os.path.relpath(MAT, REPO),
                           derived=os.path.relpath(DERIVED, REPO),
                           dt_s=DT_S, pattern_mm=PATTERN_MM,
                           rough_mm=ROUGH_MM), **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    fr = res["frames"]
    print("NR49 -- dispersion factorisation: growth clock x boost (all-pass)")
    print(f"  boost identity exact (synth)  : {v['boost_identity_exact']}")
    print(f"  migration = one all-pass      : "
          f"{v['migration_is_one_allpass_factor']} "
          f"(lab {fr['lab']['pattern']['celerity_mm_hr']:.1f} vs xcorr "
          f"{res['trackers']['v_xcorr_mm_hr']:.1f} mm/hr; co-moving residual "
          f"{100 * res['adv_residual_fraction']:.0f}%)")
    print(f"  gain is boost-invariant       : {v['gain_is_boost_invariant']} "
          f"(pattern {fr['lab']['pattern']['sigma_hr']:+.2f}/"
          f"{fr['adv']['pattern']['sigma_hr']:+.2f}, roughness "
          f"{fr['lab']['rough']['sigma_hr']:+.1f}/"
          f"{fr['adv']['rough']['sigma_hr']:+.1f} /hr)")
    print(f"  marginal pattern + damped k>  : "
          f"{v['marginal_pattern_damped_roughness']}")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
