r"""NR46 (ledger E12) -- the Sun's two clocks in one cross-spectrum: NR28's
coherence-gated phase protocol at stellar scale on 26 years of SOHO
(GOLF velocity x VIRGO/SPM irradiance) with BiSON as the ground control.

Ledger E12 conjectured a pure PROTOCOL transfer ("no new equation"): NR28's
two-clocks cross-spectral fingerprint -- coherence gating, one-sided bounded
phase in the coupled band, coherence drop as the decoupling meter -- applied
to helioseismic p-mode + granulation series.

Data (none committed; set env paths and run ``build_cache()``):

* ``$GOLF_FITS``   -- GOLF 26-yr calibrated velocity, 20 s cadence, TAI
  (SOHO mission-long bundle GOLF_26y_MEAN.fits, Appourchaux et al. 2018
  calibration; 1996-04-11..2022-02-28, m/s).
* ``$VIRGO_GREEN_FITS``, ``$VIRGO_BLUE_FITS`` -- VIRGO/SPM level-2
  mission-long irradiance (ppm), 60 s cadence, TAI, 1996-01-23..2023-04-30.
* ``$BISON_FITS``  -- BiSON network velocity residuals, 40 s cadence, JD(UT),
  1976-2025 (Davies et al. 2014 / Hale et al. 2016; fill-optimised set).

Method: gap-gated accumulating Welch (16384-min Hann windows, >=99 % joint
validity, linear detrend, 50 % overlap; df = 1.02 uHz resolves modes from
background) -> magnitude-squared coherence gamma^2(nu) and cross phase.
GOLF is linearly interpolated (zero-phase) onto the VIRGO TAI grid.  The
relative timing of the two mission-long products is self-calibrated with the
phase-slope method of Jimenez et al. 1999 (S 2.4: a timing error tilts the
I-V phase diagram): dt = +55 s -- consistent with the documented VIRGO 30-s
daily-pulse convention (their S 2.1.1) plus sample-centering conventions --
anchored to ONE literature number (raw <all> I-V = -121.3 deg, their Fig. 2).
All coherence magnitudes and the mode-vs-background phase separation are
dt-invariant; only absolute phase levels/slopes use the calibration.

Findings (figures/81_solar_two_clocks.json; V = GOLF, I = SPM green,
I-V phase convention: V downward-positive, adiabatic = -90 deg):

1. **Cross-channel coherence is a resonance detector.**  Only the global
   standing p-modes (the fast clock) are coherent ACROSS observables:
   V x I mode bins (gamma^2 >= 0.4 gate, 2400-4300 uHz): n = 401, median
   0.65, peaks 0.96; BiSON x GOLF (two independent velocity instruments,
   2003-2006): top-quintile 0.91.
2. **The convective slow clock is channel-LOCAL.**  In the granulation band
   (300-1200 uHz) the cross-channel pairs decohere -- V x I 0.067,
   V x V 0.035 -- while the same-signal control I(green) x I(blue) holds
   gamma^2 = 0.988.  The slow clock is one physical signal per channel, not
   one global signal: decoherence across channels, coherence within.
3. **Above the acoustic cutoff the cross-channel field collapses**
   (V x I 0.014, V x V 0.032) while I x I keeps 0.85 -- the collapse is a
   property of the cross-channel pair (traveling/incoherent field), not of
   the band.
4. **Inside the coupled band the phase is one-sided, bounded, and
   nonadiabatic on the convection side.**  Calibrated I-V mode phase
   -119 deg at 2.9-3.3 mHz (Jimenez et al. 1999: -121.3 +/- 1.2 raw <all>;
   IPHIR: -119 +/- 3), i.e. ~29 deg BELOW the adiabatic -90.  The departure
   is on the side pure Newtonian radiative cooling cannot produce
   (Marmolino & Severino 1991 model curves lie above -90): the phase meter
   lands on the convection-coupled models (Houdek et al. 1995), exactly
   Jimenez's model-3 conclusion.  The frequency shape reproduces theirs:
   central dip (-118 at 2.9-3.5 mHz) rising to -84 by 4.3 mHz.
5. **The background is a second, distinct phase branch in the SAME band**
   (inter-mode bins: +111 deg at 2.6 mHz, 141 deg from the mode branch)
   which CONVERGES monotonically into the mode branch (141 -> 104 -> 56 ->
   32 -> 14 deg by 4.2 mHz) as mode linewidths blend into the background --
   both the branch and its convergence are the Severino/Jimenez background-
   interference structure; coherence gating separates the two clocks
   bin-by-bin.
6. **Protocol stability at stellar scale**: 14 consecutive 2-yr epochs give
   mode-phase spread <= 8.4 deg (circular std 2.4 deg) across two solar
   activity cycles -- the fingerprint is a property of the star, not the
   epoch (extends Jimenez 2002).
7. Controls: I x I zero-phase (-0.6 deg; published green-blue +1.1) pins the
   pipeline; V x V +15.7 deg is timing-convention-limited (~14 s at 3 mHz),
   consistent with 0.

What transfers from NR28 verbatim: coherence gating for phase validity; the
one-sided bounded phase inside the coupled band; coherence structure as the
clock separator.  What does NOT: the arctan(omega/omega_c) first-order-lag
form -- the solar mode-phase departure has the OPPOSITE sign (convective,
not relaxational), and the decoupling axis is cross-CHANNEL at low
frequency rather than high-frequency small-scale: the protocol is portable,
the constitutive lag law is not.  That boundary is the E12 result.

CPU-only.  Tests: tests/test_solar_two_clocks.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr46_solar_cache.json")
FIG = os.path.join(HERE, "figures", "81_solar_two_clocks.json")

L_MIN = 16384          # 60-s samples per window (16384 min = 11.38 d)
DT_CAL = 55.0          # fitted GOLF->VIRGO relative timing (s); see docstring
JD_GOLF0 = 2450184.5   # 1996-04-11 00:00 as JD
GRAN = (300.0, 1200.0)
PBAND = (2400.0, 4300.0)
CUT = (5800.0, 8000.0)
MODE_G2 = 0.4


# --------------------------------------------------------------------------- #
# accumulating gap-gated Welch machinery
# --------------------------------------------------------------------------- #
def welch_pair(x, y, ok, L, min_valid=0.99, groups=None):
    """Accumulate Hann cross/auto spectra over >=min_valid windows.

    Returns dict with Sxx, Syy, Sxy, nwin (+ per-group accumulators)."""
    H = np.hanning(L)
    step = L // 2
    n2 = L // 2 + 1
    tot = dict(Sxx=np.zeros(n2), Syy=np.zeros(n2),
               Sxy=np.zeros(n2, complex), nwin=0)
    gacc = {}
    ii = np.arange(L)
    for start in range(0, len(x) - L, step):
        m = ok[start:start + L]
        if m.mean() < min_valid:
            continue
        xx = x[start:start + L].astype(float).copy()
        yy = y[start:start + L].astype(float).copy()
        if (~m).any():
            xx[~m] = np.interp(ii[~m], ii[m], xx[m])
            yy[~m] = np.interp(ii[~m], ii[m], yy[m])
        for arr in (xx, yy):
            c = np.polyfit(ii, arr, 1)
            arr -= c[0] * ii + c[1]
        X = np.fft.rfft(xx * H)
        Y = np.fft.rfft(yy * H)
        sxx = (X * np.conj(X)).real
        syy = (Y * np.conj(Y)).real
        sxy = X * np.conj(Y)
        tot["Sxx"] += sxx
        tot["Syy"] += syy
        tot["Sxy"] += sxy
        tot["nwin"] += 1
        if groups is not None:
            g = groups(start)
            b = gacc.setdefault(g, dict(Sxx=np.zeros(n2), Syy=np.zeros(n2),
                                        Sxy=np.zeros(n2, complex), nwin=0))
            b["Sxx"] += sxx
            b["Syy"] += syy
            b["Sxy"] += sxy
            b["nwin"] += 1
    tot["groups"] = gacc
    return tot


def coh_phase(acc):
    g2 = np.abs(acc["Sxy"]) ** 2 / np.maximum(acc["Sxx"] * acc["Syy"], 1e-300)
    ph = np.degrees(np.angle(np.conj(acc["Sxy"])))   # phase of Y relative to X
    return g2, ph


def circmean_deg(ph):
    z = np.exp(1j * np.radians(np.asarray(ph)))
    return float(np.degrees(np.angle(np.mean(z))))


def circdist_deg(a, b):
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


# --------------------------------------------------------------------------- #
# raw ingest (only when the FITS archives are present)
# --------------------------------------------------------------------------- #
def _load_golf():
    from astropy.io import fits
    g = fits.open(os.environ.get("GOLF_FITS", "/home/golf_mean.fits"))[0]
    d = g.data.astype(np.float64)
    return np.where(d == 0, np.nan, d)


def _load_spm(env, default):
    from astropy.io import fits
    v = fits.open(os.environ.get(env, default))[0].data
    return v[:, 0], v[:, 1]


def _golf_on(times_sec):
    """GOLF linearly interpolated (zero-phase) onto given VIRGO-frame times."""
    gv = _load_golf()
    off = 79 * 86400.0 - 4.46 + DT_CAL   # GOLF t=0 in VIRGO SPM time frame
    pos = (times_sec - off) / 20.0
    idx = np.clip(np.ceil(pos).astype(int), 1, len(gv) - 1)
    w1 = pos - (idx - 1)
    out = gv[idx - 1] * (1 - w1) + gv[idx] * w1
    out[(pos < 0) | (pos > len(gv) - 1)] = np.nan
    return out


def band_median(g2, uhz, band):
    m = (uhz >= band[0]) & (uhz <= band[1])
    return float(np.median(g2[m]))


def build_cache(write=True):
    tg, sg = _load_spm("VIRGO_GREEN_FITS", "/home/virgo_spm_green.fits")
    fr = np.fft.rfftfreq(L_MIN, 60.0)
    uhz = fr * 1e6

    # -- pair 1: GOLF velocity x SPM green irradiance (the physics pair)
    gi = _golf_on(tg)
    ok = np.isfinite(gi) & np.isfinite(sg)
    yr0 = 1996.06                        # SPM start epoch (1996-01-23)
    sec_yr = 365.25 * 86400.0

    def epoch_of(start):
        return int((yr0 + tg[start] / sec_yr - 1996.0) // 2)

    vxi = welch_pair(np.nan_to_num(gi), np.nan_to_num(sg), ok, L_MIN,
                     groups=epoch_of)
    g2, ph = coh_phase(vxi)
    pb = (uhz >= PBAND[0]) & (uhz <= PBAND[1])
    mode = pb & (g2 >= MODE_G2)
    back = pb & (g2 < 0.15)
    mode_rows = [dict(uhz=float(uhz[i]), g2=float(g2[i]), ph=float(ph[i]))
                 for i in np.where(mode)[0]]
    back_curve = []
    for lo in range(2000, 4400, 400):
        bb = back & (uhz >= lo) & (uhz < lo + 400)
        if bb.sum():
            back_curve.append(dict(uhz_mid=lo + 200, n=int(bb.sum()),
                                   ph=circmean_deg(ph[bb])))
    mode_curve = []
    for lo in range(2400, 4400, 200):
        mm = mode & (uhz >= lo) & (uhz < lo + 200)
        if mm.sum():
            mode_curve.append(dict(uhz_mid=lo + 100, n=int(mm.sum()),
                                   ph=circmean_deg(ph[mm]),
                                   g2_med=float(np.median(g2[mm]))))
    lvl = (uhz >= 2900) & (uhz <= 3300) & (g2 >= MODE_G2)
    epochs = []
    for ep in sorted(vxi["groups"]):
        b = vxi["groups"][ep]
        if b["nwin"] < 6:
            continue
        g2e, phe = coh_phase(b)
        me = pb & (g2e >= MODE_G2)
        epochs.append(dict(label=f"{1996 + 2 * ep}-{1997 + 2 * ep}",
                           nwin=int(b["nwin"]), n_mode_bins=int(me.sum()),
                           ph=circmean_deg(phe[me])))
    pair_vxi = dict(
        nwin=int(vxi["nwin"]), n_mode_bins=int(mode.sum()),
        n_back_bins=int(back.sum()),
        g2_mode_median=float(np.median(g2[mode])),
        g2_pband_max=float(g2[pb].max()),
        g2_gran=band_median(g2, uhz, GRAN),
        g2_cutoff=band_median(g2, uhz, CUT),
        phase_level_2900_3300=circmean_deg(ph[lvl]),
        gran_phase=circmean_deg(
            ph[(uhz >= GRAN[0]) & (uhz <= GRAN[1]) & (g2 > 0.02)]),
        mode_rows=mode_rows, mode_curve=mode_curve, back_curve=back_curve,
        epochs=epochs)

    # -- pair 2: SPM green x SPM blue (same-signal, zero-phase control)
    tb, sb = _load_spm("VIRGO_BLUE_FITS", "/home/virgo_spm_blue.fits")
    okb = np.isfinite(sg) & np.isfinite(sb)
    ixi = welch_pair(np.nan_to_num(sg), np.nan_to_num(sb), okb, L_MIN)
    g2b, phb = coh_phase(ixi)
    modeb = pb & (g2b >= MODE_G2)
    pair_ixi = dict(
        nwin=int(ixi["nwin"]),
        g2_gran=band_median(g2b, uhz, GRAN),
        g2_pband_mode=float(np.median(g2b[modeb])),
        g2_cutoff=band_median(g2b, uhz, CUT),
        mode_phase=circmean_deg(phb[modeb]),
        gran_phase=circmean_deg(phb[(uhz >= GRAN[0]) & (uhz <= GRAN[1])]))

    # -- pair 3: BiSON x GOLF velocity (independent instruments, 2003-2006)
    from astropy.io import fits as _f
    bi = _f.open(os.environ.get("BISON_FITS", "/home/bison_fill.fits"))[0].data
    tjd, vb = bi[:, 0], bi[:, 1]
    sel = (tjd >= 2452640.5) & (tjd <= 2454101.5)
    tjd, vb = tjd[sel], np.where(vb[sel] == 0, np.nan, vb[sel])
    gv = _load_golf()
    tsec = (tjd - JD_GOLF0) * 86400.0 + 32.0     # UT->TAI (approx, 2003-2006)
    pos = tsec / 20.0
    idx = np.clip(np.ceil(pos).astype(int), 1, len(gv) - 1)
    w1 = pos - (idx - 1)
    gi2 = gv[idx - 1] * (1 - w1) + gv[idx] * w1
    okv = np.isfinite(gi2) & np.isfinite(vb)
    # contiguity mask: windows must span exactly L*40 s
    L2 = 2048
    fr2 = np.fft.rfftfreq(L2, 40.0)
    u2 = fr2 * 1e6
    dt_ok = np.ones(len(tjd), bool)
    dj = np.diff(tjd) * 86400.0
    dt_ok[1:] = np.abs(dj - 40.0) < 1.0
    vxv = welch_pair(np.nan_to_num(gi2), np.nan_to_num(vb), okv & dt_ok, L2,
                     min_valid=0.98)
    g2v, phv = coh_phase(vxv)
    pb2 = (u2 >= PBAND[0]) & (u2 <= PBAND[1])
    top = pb2 & (g2v >= np.quantile(g2v[pb2], 0.8))
    pair_vxv = dict(
        nwin=int(vxv["nwin"]),
        g2_pband_median=float(np.median(g2v[pb2])),
        g2_top_quintile=float(np.median(g2v[top])),
        g2_pband_max=float(g2v[pb2].max()),
        g2_gran=band_median(g2v, u2, GRAN),
        g2_cutoff=band_median(g2v, u2, CUT),
        top_phase=circmean_deg(phv[top]))

    cache = dict(
        meta=dict(
            window_min=L_MIN, df_uhz=float(uhz[1]),
            dt_cal_s=DT_CAL,
            dt_cal_method="phase-slope timing self-calibration (Jimenez et "
                          "al. 1999 S2.4) anchored to their Fig.2 raw <all> "
                          "I-V = -121.3 deg at 2.9-3.3 mHz; coherences and "
                          "phase separations are dt-invariant",
            bands=dict(gran=GRAN, pband=PBAND, cutoff=CUT),
            mode_gate_g2=MODE_G2,
            phase_convention="I-V, V downward positive (adiabatic -90 deg); "
                             "computed as angle(conj(Sxy)) with X=V, Y=I",
            provenance="SOHO mission-long bundles: GOLF_26y_MEAN.fits "
                       "(Appourchaux et al. 2018), VIRGO-SPM-GREEN/BLUE-L2-"
                       "MISSIONLONG.fits; BiSON open data portal "
                       "allsites-alldata-waverage-fill.fits (Davies et al. "
                       "2014, Hale et al. 2016)"),
        golf_x_green=pair_vxi, green_x_blue=pair_ixi, bison_x_golf=pair_vxv)
    if write:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w") as fh:
            json.dump(cache, fh)
    return cache


def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #
def analyze(cache):
    vxi = cache["golf_x_green"]
    ixi = cache["green_x_blue"]
    vxv = cache["bison_x_golf"]
    fast = (vxi["n_mode_bins"] >= 300 and vxi["g2_mode_median"] >= 0.55
            and vxi["g2_pband_max"] >= 0.90
            and vxv["g2_top_quintile"] >= 0.70)
    slow_local = (vxi["g2_gran"] <= 0.10 and vxv["g2_gran"] <= 0.06
                  and ixi["g2_gran"] >= 0.90)
    cut_collapse = (vxi["g2_cutoff"] <= 0.03 and vxv["g2_cutoff"] <= 0.05
                    and ixi["g2_cutoff"] >= 0.70)
    lvl = vxi["phase_level_2900_3300"]
    nonadiabatic = (-135.0 <= lvl <= -105.0)      # Jimenez -121.3, IPHIR -119
    low = [r for r in vxi["mode_curve"] if 2400 <= r["uhz_mid"] <= 3400]
    one_sided = all(r["ph"] < -90.0 for r in low) and all(
        -180.0 < r["ph"] < -60.0 for r in vxi["mode_curve"])
    hi = [r["ph"] for r in vxi["mode_curve"] if r["uhz_mid"] >= 4000]
    ctr = [r["ph"] for r in vxi["mode_curve"] if 2900 <= r["uhz_mid"] <= 3500]
    shape = (len(hi) > 0 and len(ctr) > 0
             and (np.mean(hi) - np.mean(ctr)) >= 20.0)
    sep_by_mid = {}
    for m in vxi["mode_curve"]:
        for b in vxi["back_curve"]:
            if abs(m["uhz_mid"] - b["uhz_mid"]) <= 100:
                sep_by_mid.setdefault(b["uhz_mid"], []).append(
                    circdist_deg(m["ph"], b["ph"]))
    sep_pairs = sorted((mid, float(np.mean(v)))
                       for mid, v in sep_by_mid.items())
    seps = [s for _, s in sep_pairs]
    # distinct second branch at the band core, converging into the mode
    # branch as linewidths blend toward the cutoff (Jimenez et al. 1999 S6)
    two_branches = (len(seps) >= 4 and min(seps[:2]) >= 80.0
                    and all(a >= b for a, b in zip(seps, seps[1:]))
                    and seps[-1] <= 30.0)
    eph = [e["ph"] for e in vxi["epochs"]]
    z = np.exp(1j * np.radians(np.array(eph)))
    circ_std = float(np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean(z))))))
    stable = len(eph) >= 12 and circ_std <= 5.0
    controls = (abs(ixi["mode_phase"]) <= 1.5
                and circdist_deg(vxv["top_phase"], 0.0) <= 20.0)
    verdicts = dict(
        fast_clock_cross_channel=bool(fast),
        slow_clock_channel_local=bool(slow_local),
        cutoff_cross_channel_collapse=bool(cut_collapse),
        mode_phase_nonadiabatic_convection_side=bool(nonadiabatic
                                                     and one_sided),
        phase_shape_matches_literature=bool(shape),
        background_second_branch=bool(two_branches),
        branch_separation_deg=[list(p) for p in sep_pairs],
        epoch_stable_across_cycles=bool(stable),
        instrument_controls_pass=bool(controls),
        epoch_circ_std_deg=circ_std,
        reading=(
            "NR28's protocol at stellar scale: cross-channel coherence is a "
            "resonance detector -- only the global p-modes (fast clock) are "
            "coherent across V and I (and across two independent velocity "
            "instruments), while the convective slow clock is channel-local "
            "(V-I and V-V decohere in the granulation band where I-I holds "
            "0.99) and the above-cutoff field collapses cross-channel only. "
            "Inside the coupled band the coherence-gated I-V phase is "
            "one-sided, bounded, 29 deg below adiabatic -90 -- the "
            "convection-coupled nonadiabatic side, reproducing Jimenez et "
            "al. 1999 across 26 years with 2.4-deg epoch stability -- and "
            "the inter-mode background forms a second, distinct phase "
            "branch in the same band.  The protocol transfers; the "
            "arctan first-order-lag form does not (opposite departure "
            "sign): that boundary is the E12 result."),
    )
    return verdicts


def run(write=True):
    cache = load_cache()
    verdicts = analyze(cache)
    res = dict(description=__doc__.splitlines()[0], meta=cache["meta"],
               golf_x_green={k: v for k, v in cache["golf_x_green"].items()
                             if k != "mode_rows"},
               green_x_blue=cache["green_x_blue"],
               bison_x_golf=cache["bison_x_golf"], verdicts=verdicts)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    x = res["golf_x_green"]
    print("NR46 (E12) -- solar two clocks: GOLF x VIRGO x BiSON")
    print(f"  V x I mode bins           : {x['n_mode_bins']} "
          f"(g2 med {x['g2_mode_median']:.2f}, max {x['g2_pband_max']:.2f}; "
          f"{x['nwin']} windows)")
    print(f"  slow clock channel-local  : "
          f"{v['slow_clock_channel_local']} "
          f"(gran g2: VxI {x['g2_gran']:.3f}, VxV "
          f"{res['bison_x_golf']['g2_gran']:.3f}, IxI "
          f"{res['green_x_blue']['g2_gran']:.3f})")
    print(f"  cutoff collapse (x-chan)  : "
          f"{v['cutoff_cross_channel_collapse']} "
          f"(VxI {x['g2_cutoff']:.3f}, VxV "
          f"{res['bison_x_golf']['g2_cutoff']:.3f}, IxI "
          f"{res['green_x_blue']['g2_cutoff']:.3f})")
    print(f"  I-V phase @2.9-3.3 mHz    : {x['phase_level_2900_3300']:.1f} "
          f"deg (Jimenez -121.3; nonadiabatic="
          f"{v['mode_phase_nonadiabatic_convection_side']})")
    print(f"  shape / branches / epochs : {v['phase_shape_matches_literature']}"
          f" / {v['background_second_branch']} / "
          f"{v['epoch_stable_across_cycles']} "
          f"(circ std {v['epoch_circ_std_deg']:.1f} deg, "
          f"{len(x['epochs'])} epochs)")
    print(f"  controls (IxI, VxV)       : {v['instrument_controls_pass']} "
          f"(IxI {res['green_x_blue']['mode_phase']:.2f} deg, VxV "
          f"{res['bison_x_golf']['top_phase']:.1f} deg)")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
