r"""NR51 (theory) -- closed spectral polygons are shape, open ones are
motion: triad correlators are exactly boost-blind, Im(bispectrum) is the
parity-odd shape coordinate, and the real scallops break parity in the
flux clock while their shape stays near-symmetric.

Follows NR49 (pair transfer = gain x all-pass; the boost lives entirely in
the open two-point phase) and Paper 3 (the melt-flux quadrature E_sin is
the parity-break detector).  NR51 adds the three-point face and the
theorem that separates them.

The claim
=========
1. **Closed polygons are boost-blind (exact).**  Under ``x -> x - Vt``
   every Fourier coefficient picks up ``e^{ikVt}``, so an n-point spectral
   correlator ``<h_hat(k_1)...h_hat(k_n)>`` acquires
   ``e^{i(k_1+...+k_n)Vt}`` -- equal to 1 IFF the wavevector polygon
   closes.  The pair transfer (open, k vs -k separated in time) carries
   the whole boost (NR49); the equal-time bispectrum
   ``B(k_1,k_2) = <h_hat(k_1) h_hat(k_2) h_hat*(k_1+k_2)>`` carries NONE.
   Triad phases are pure *shape*; pair phases are pure *motion*.

2. **Im B is the parity-odd shape coordinate.**  Under the mirror
   ``x -> -x``: ``h_hat(k) -> h_hat*(k)``, so ``B -> B*``: Re B (skewness
   family) is parity-EVEN, Im B (asymmetry family, Elgar-Guza) is
   parity-ODD, and the biphase flips sign exactly.  Exact moment sum
   rules (mod-N closed triads, machine precision):

       ``mean(h^3)      = (1/N^3) SUM_closed  h1 h2 h3``      (Re face)
       ``mean(H[h]^3)   = (1/N^3) SUM_closed  s1 s2 s3 * i * h1 h2 h3``
                          with s = sign of the frequency        (Im face)

   -- bulk skewness/asymmetry are triad sums; a parity-symmetric shape
   has ALL biphases at 0 or 180 deg.

3. **Migration does not imply asymmetric shape.**  Because the boost is
   invisible to every closed polygon, a rigidly migrating pattern can be
   perfectly parity-symmetric: linear (pair-level) parity breaking --
   migration, flux quadrature -- leaves NO triad imprint.  Shape
   asymmetry is a strictly NONLINEAR (asymmetric triad coupling) effect.
   Dune-style lee-stoss asymmetry is therefore a *nonlinearity* meter,
   not a migration meter; the flux quadrature (Paper 3) reads parity at
   the linear level, where the scallop parity breaking actually lives.

Findings (figures/86_triad_parity.json), real Bushuk h(x,t):

* **Exactness on data.**  Random spectral shifts of every frame change
  the dominant biphase by < 1e-6 deg (boost-blind); mirroring every frame
  flips it exactly; the moment sum rules hold to machine precision on
  each frame.
* **Positive control.**  A Stokes-like profile ``cos(th) +
  0.3 cos(2 th - 90 deg)`` (maximally leaning) gives asymmetry ~ +-0.9
  with biphase +-90 deg; the phase-aligned control (phi = 0) is skewed
  but has ZERO asymmetry -- the detector separates the two parities.
* **The scallops: flux breaks parity, shape barely does.**  Committed
  flux quadrature (Paper 3): E_sin t-stat = -4.9 over 11 pairs
  (one-sided bootstrap p ~ 3e-4) -- decisive.  Shape: per-frame
  asymmetry 0.17 +- 0.24 (t ~ 2.4 with correlated frames), dominant-triad
  biphase 30 deg with 47 deg frame-to-frame circular std -- weak and
  unstable, while the same 12 frames give a decisive migration
  (NR49: 48 mm/hr, co-moving residual 7 %).  The parity breaking is in
  the transport clock, not the stored shape.

Consequence for the program: the two-point phase carries kinematics
(NR49), the three-point phase carries shape parity -- and the scallop
instability breaks parity dynamically (flux quadrature, migration) while
storing almost none of it statically.  Parity-break detectors must
therefore read the FLUX/pair level (Paper 3's E_sin), not the shape/triad
level; conversely a strong Im B (dunes) certifies asymmetric NONLINEAR
coupling, not just migration.

CPU-only, offline-safe (committed .mat + committed flux cache).
Tests: tests/test_triad_parity.py.
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
FIG = os.path.join(HERE, "figures", "86_triad_parity.json")


# --------------------------------------------------------------------------- #
# spectra, bispectrum, moments
# --------------------------------------------------------------------------- #
def detrend(h):
    h = np.asarray(h, float)
    i = np.arange(len(h))
    h = h - h.mean()
    cc = np.polyfit(i, h, 1)
    return h - cc[0] * i - cc[1]


def clean_bins(h):
    """Zero the mean and Nyquist bins so the DFT-Hilbert sign convention is
    unambiguous (exactness of the sum rules)."""
    F = np.fft.fft(np.asarray(h, float))
    F[0] = 0.0
    n = len(F)
    if n % 2 == 0:
        F[n // 2] = 0.0
    return np.real(np.fft.ifft(F))


def dft_hilbert(h):
    F = np.fft.fft(np.asarray(h, float))
    n = len(F)
    freq = np.fft.fftfreq(n)
    F = -1j * np.sign(freq) * F
    return np.real(np.fft.ifft(F))


def triad_moment_sums(h):
    """Exact mod-N closed-triad sums: returns (sum_re, sum_im) such that
    mean(h^3) = sum_re and mean(H[h]^3) = sum_im (machine precision after
    clean_bins)."""
    F = np.fft.fft(np.asarray(h, float))
    n = len(F)
    s = np.sign(np.fft.fftfreq(n))
    m1 = np.arange(n)[:, None]
    m2 = np.arange(n)[None, :]
    m3 = (-(m1 + m2)) % n
    T = F[m1] * F[m2] * F[m3]
    sum_re = float(np.real(np.sum(T)) / n ** 3)
    S3 = s[m1] * s[m2] * s[m3]
    sum_im = float(np.real(np.sum(1j * S3 * T)) / n ** 3)
    return sum_re, sum_im


def rfft_frames(frames, window=True, detrend_frames=True):
    F = np.asarray(frames, float)
    n = F.shape[1]
    W = np.hanning(n) if window else np.ones(n)
    prep = detrend if detrend_frames else (lambda h: h - h.mean())
    return np.array([np.fft.rfft(prep(h) * W) for h in F])


def bispectrum(H, k1, k2):
    return complex(np.mean(H[:, k1] * H[:, k2] * np.conj(H[:, k1 + k2])))


def biphase_per_frame_deg(H, k1, k2):
    return np.degrees(np.angle(H[:, k1] * H[:, k2]
                               * np.conj(H[:, k1 + k2])))


def circular_stats_deg(angles_deg):
    z = np.exp(1j * np.radians(np.asarray(angles_deg, float)))
    R = abs(z.mean())
    mu = float(np.degrees(np.angle(z.mean())))
    std = float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))
    return mu, std


# --------------------------------------------------------------------------- #
# synthetic controls
# --------------------------------------------------------------------------- #
def synthetic_checks():
    th = 2.0 * np.pi * np.arange(512) / 512.0
    lean = np.cos(th) + 0.3 * np.cos(2 * th - np.pi / 2)   # maximally leaning
    skew = np.cos(th) + 0.3 * np.cos(2 * th)               # skewed, symmetric
    out = {}
    for name, h in (("leaning", lean), ("skewed_symmetric", skew)):
        hc = clean_bins(h)
        sr, si = triad_moment_sums(hc)
        sig = hc.std()
        out[name] = dict(
            skewness=float(np.mean(hc ** 3) / sig ** 3),
            asymmetry=float(np.mean(dft_hilbert(hc) ** 3) / sig ** 3),
            sum_rule_re_err=abs(np.mean(hc ** 3) - sr),
            sum_rule_im_err=abs(np.mean(dft_hilbert(hc) ** 3) - si))
    # boost-blindness + parity, exact (raw circular shifts; a linear
    # detrend is NOT boost-covariant on a discrete grid -- the ramp has
    # projections on every Fourier mode -- so the exact statement uses the
    # field as-is; mean removal only touches k=0, outside every triad)
    H = rfft_frames([lean, np.roll(lean, 37), np.roll(lean, 191)],
                    window=False, detrend_frames=False)
    bp = biphase_per_frame_deg(H, 1, 1)
    out["boost_biphase_spread_deg"] = float(np.ptp(bp))
    Hm = rfft_frames([lean[::-1]], window=False, detrend_frames=False)
    out["mirror_flip_err_deg"] = float(abs(
        biphase_per_frame_deg(Hm, 1, 1)[0] + bp[0]))
    return out


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #
def load_frames(path=MAT):
    import scipy.io as sio
    d = sio.loadmat(path)
    Z = np.column_stack([np.asarray(c).ravel()
                         for c in d["z_profiles"].ravel()]).T
    return Z


def analyze():
    syn = synthetic_checks()
    Z = load_frames()
    n = Z.shape[1]

    # exactness on data: sum rules per frame (cleaned bins)
    errs_re, errs_im, asym, skew = [], [], [], []
    for h in Z:
        hc = clean_bins(detrend(h))
        sr, si = triad_moment_sums(hc)
        errs_re.append(abs(np.mean(hc ** 3) - sr))
        errs_im.append(abs(np.mean(dft_hilbert(hc) ** 3) - si))
        sig = hc.std()
        skew.append(np.mean(hc ** 3) / sig ** 3)
        asym.append(np.mean(dft_hilbert(hc) ** 3) / sig ** 3)
    asym = np.array(asym)
    skew = np.array(skew)

    # dominant-triad biphase: value, stability, boost-blindness, parity
    H = rfft_frames(Z)
    bp = biphase_per_frame_deg(H, 1, 1)
    bp_mu, bp_std = circular_stats_deg(bp)
    b_weighted = float(np.degrees(np.angle(bispectrum(H, 1, 1))))
    k = 2.0 * np.pi * np.fft.rfftfreq(n, 1.0)
    rng = np.random.default_rng(0)
    Hs = H * np.exp(-1j * k[None, :]
                    * rng.uniform(-50, 50, len(Z))[:, None])
    boost_err = abs(np.degrees(np.angle(bispectrum(Hs, 1, 1)))
                    - b_weighted)
    Hm = rfft_frames(Z[:, ::-1])
    mirror_err = abs(np.degrees(np.angle(bispectrum(Hm, 1, 1)))
                     + b_weighted)

    # committed flux parity (Paper 3)
    with open(DERIVED) as fh:
        d = json.load(fh)
    Es = np.array(d["flux"]["1"])[:, 1]
    t_flux = float(Es.mean() / (Es.std(ddof=1) / np.sqrt(len(Es))))
    t_shape = float(asym.mean() / (asym.std(ddof=1) / np.sqrt(len(asym))))

    out = dict(
        synthetic=syn,
        data_sum_rule_re_max_err=float(max(errs_re)),
        data_sum_rule_im_max_err=float(max(errs_im)),
        shape_skewness_mean=float(skew.mean()),
        shape_asymmetry_mean=float(asym.mean()),
        shape_asymmetry_std=float(asym.std(ddof=1)),
        t_shape_asymmetry=t_shape,
        biphase_weighted_deg=b_weighted,
        biphase_frame_circ_std_deg=bp_std,
        boost_blind_err_deg=float(boost_err),
        mirror_flip_err_deg=float(mirror_err),
        t_flux_quadrature=t_flux,
        n_frames=int(len(Z)))
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    syn = out["synthetic"]
    exact = (syn["boost_biphase_spread_deg"] < 1e-6
             and syn["mirror_flip_err_deg"] < 1e-6
             and max(syn["leaning"]["sum_rule_re_err"],
                     syn["leaning"]["sum_rule_im_err"],
                     syn["skewed_symmetric"]["sum_rule_re_err"],
                     syn["skewed_symmetric"]["sum_rule_im_err"]) < 1e-12
             and out["data_sum_rule_re_max_err"] < 1e-9
             and out["data_sum_rule_im_max_err"] < 1e-9
             and out["boost_blind_err_deg"] < 1e-6
             and out["mirror_flip_err_deg"] < 1e-6)
    detector = (abs(syn["leaning"]["asymmetry"]) > 0.5
                and abs(syn["skewed_symmetric"]["asymmetry"]) < 1e-12
                and abs(syn["skewed_symmetric"]["skewness"]) > 0.5)
    flux_not_shape = (abs(out["t_flux_quadrature"]) > 4.0
                      and abs(out["t_shape_asymmetry"])
                      < 0.6 * abs(out["t_flux_quadrature"])
                      and out["biphase_frame_circ_std_deg"] > 30.0)
    return dict(
        closed_polygons_boost_blind_and_parity_odd=bool(exact),
        asymmetry_detector_separates_parities=bool(detector),
        scallops_break_parity_in_flux_not_shape=bool(flux_not_shape),
        reading=(
            "Closed spectral polygons are shape, open ones are motion: "
            "random boosts move the dominant scallop biphase by "
            f"{out['boost_blind_err_deg']:.1e} deg while the mirror flips "
            "it exactly, and the bulk skewness/asymmetry are closed-triad "
            "sums to machine precision.  On the real scallops the parity "
            "breaking is dynamic, not stored: the flux quadrature is "
            f"decisive (t = {out['t_flux_quadrature']:.1f}) while the "
            f"shape asymmetry is weak ({out['shape_asymmetry_mean']:.2f} "
            f"+- {out['shape_asymmetry_std']:.2f}, t = "
            f"{out['t_shape_asymmetry']:.1f}) with an unstable biphase "
            f"({out['biphase_weighted_deg']:.0f} deg, frame circ-std "
            f"{out['biphase_frame_circ_std_deg']:.0f} deg) -- the same "
            "frames that migrate decisively (NR49).  Migration and flux "
            "parity live at the pair level, invisible to every closed "
            "polygon; strong Im B (dunes) would certify asymmetric "
            "nonlinear coupling, which these scallops barely have."),
    )


def run(write=True):
    out = analyze()
    res = dict(description=__doc__.splitlines()[0],
               source=dict(mat=os.path.relpath(MAT, REPO),
                           derived=os.path.relpath(DERIVED, REPO)), **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    print("NR51 -- triad parity: closed polygons are shape, open are motion")
    print(f"  boost-blind + parity-odd exact: "
          f"{v['closed_polygons_boost_blind_and_parity_odd']} "
          f"(boost {res['boost_blind_err_deg']:.1e} deg, sum rules "
          f"{res['data_sum_rule_im_max_err']:.1e})")
    print(f"  detector separates parities   : "
          f"{v['asymmetry_detector_separates_parities']} "
          f"(leaning A={res['synthetic']['leaning']['asymmetry']:+.2f}, "
          f"symmetric A="
          f"{res['synthetic']['skewed_symmetric']['asymmetry']:+.1e})")
    print(f"  flux breaks parity, not shape : "
          f"{v['scallops_break_parity_in_flux_not_shape']} "
          f"(flux t={res['t_flux_quadrature']:.1f} vs shape "
          f"t={res['t_shape_asymmetry']:.1f}, biphase circ-std "
          f"{res['biphase_frame_circ_std_deg']:.0f} deg)")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
