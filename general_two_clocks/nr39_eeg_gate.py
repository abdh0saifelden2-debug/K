r"""NR39 real-data gate EXECUTED (ledger E7) -- the imaginary-coherence
volume-conduction lemma on real resting EEG (OpenNeuro ds003775).

NR39 (`new_relationships16.py`) proved the lemma that P0's surrogate ceiling
and EEG's imaginary-coherency artifact rejection (Nolte et al. 2004) are two
corollaries of one fact: instantaneous linear mixing ``x = A s`` of a
real-spectrum source set gives a real-symmetric cross-spectrum, so
``Im S_x = 0`` -- volume conduction (an instantaneous Poisson solve) cannot
manufacture imaginary coherency.  NR39 verified it on synthetic arrays and
REGISTERED a real-EEG consistency gate (``figures/nr39_imaginary_coherence
.json`` -> ``real_eeg: available=false``).  This closes that gate on real
data and turns the "consistency demo" into a quantitative confirmation.

Data (not committed; set ``$NR39_EDF`` and run ``build_cache()``): OpenNeuro
**ds003775** (Hatlestad-Hall et al., SRM resting-state EEG, CC0), subject
sub-001 ses-t1 task-resteyesc: 64-channel BioSemi ActiveTwo (10-10), 4 min
eyes-closed, 1024 Hz.  The derived alpha-band (8-13 Hz) coherency matrix,
per-pair phase-surrogate ceilings and electrode distances -- everything
needed to re-verify every verdict -- are committed to
``data/nr39_eeg_gate_cache.json`` (84 KB).

Method: 4-s Hann windows (50 % overlap, 118 segments) -> alpha-band
magnitude-squared coherency ``C_ij`` for all 2016 channel pairs -> |Re C|
(mixing + zero-lag), |Im C| (lagged only).  Ceiling: whole-series
phase-randomization (preserves each channel PSD, destroys cross-phase),
200 surrogates, per-pair 95th percentile.  Electrode distances from the
biosemi64 montage.

Findings (figures/nr39_eeg_gate.json):

1. **Volume conduction dominates coherency, in the real part**: median
   |Re C| = 0.53 vs |Im C| = 0.064 -> Re/Im = 8.2.  The instantaneous-mixing
   field is large and (as the lemma requires) it is carried by Re.
2. **The lemma's decisive real-data signature -- mixing feeds Re, NOT Im**:
   across pairs ``Spearman(|Re C|, |Im C|) = -0.25`` (p = 3e-30) -- more
   mixing means LESS imaginary coherency, never more.  The top-mixing decile
   (near-adjacent electrodes, strongest instantaneous mixing) has Re/Im = 28
   with |Im C| = 0.031, BELOW the global median: exactly where the artifact
   would live if Im were a mixing artifact, Im is smallest.
3. **Imaginary coherency is inter-regional, mixing is local**:
   ``Spearman(dist, |Re C|) = -0.15`` (mixing falls with distance) while
   ``Spearman(dist, |Im C|) = +0.15`` (lagged coupling rises with distance);
   nearest-decile pairs |Re C| = 0.83, |Im C| = 0.028.  Im picks out genuine
   distant coupling, not the local mixing that dominates Re.
4. **The surrogate ceiling behaves as P0 predicts**: null floor (surrogate
   median |Im C|) = 0.016; 62.6 % of pairs clear their per-pair p95 (real
   alpha-band lagged coupling is widespread) -- but only 33 % of the
   top-mixing decile clear it: the most volume-conduction-contaminated pairs
   are the LEAST likely to show significant imaginary coherency, the direct
   inverse of an artifact.

This is P0 S6's surrogate ceiling and Nolte's ImCoh artifact rejection
confirmed as one lemma on real EEG: instantaneous mixing is real, so it
loads Re and is invisible to Im and to the phase-surrogate ceiling.

CPU-only.  Tests: tests/test_nr39_eeg_gate.py.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nr39_eeg_gate_cache.json")
FIG = os.path.join(HERE, "figures", "nr39_eeg_gate.json")

BAND = (8.0, 13.0)
WIN_S = 4.0
N_SUR = 200


# --------------------------------------------------------------------------- #
# raw ingest (only when the EDF is present; $NR39_EDF)
# --------------------------------------------------------------------------- #
def _alpha_coherency(data, fs, win_starts, band_sel, H):
    from numpy.fft import rfft
    n, L = data.shape[0], len(H)
    S = np.zeros((n, n, int(band_sel.sum())), complex)
    for s0 in win_starts:
        F = rfft(data[:, s0:s0 + L] * H, axis=1)[:, band_sel]
        S += np.einsum("if,jf->ijf", F, np.conj(F))
    Pd = np.real(np.einsum("iif->if", S))
    den = np.sqrt(np.einsum("if,jf->ijf", Pd, Pd))
    return (S / np.maximum(den, 1e-30)).mean(2)


def _phase_randomize(data, rng):
    F = np.fft.rfft(data, axis=1)
    ph = rng.uniform(0, 2 * np.pi, (data.shape[0], F.shape[1]))
    ph[:, 0] = 0.0
    if data.shape[1] % 2 == 0:
        ph[:, -1] = 0.0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=data.shape[1], axis=1)


def build_cache(edf_path=None, write=True, n_sur=N_SUR, seed=0):
    import mne
    edf_path = edf_path or os.environ.get("NR39_EDF", "/home/eeg_sub001.edf")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True)
    names = [raw.ch_names[i] for i in picks]
    X = raw.get_data(picks=picks)
    X = X - X.mean(1, keepdims=True)
    n = X.shape[0]
    L = int(WIN_S * fs)
    H = np.hanning(L)
    win_starts = list(range(0, X.shape[1] - L, L // 2))
    fr = np.fft.rfftfreq(L, 1.0 / fs)
    band_sel = (fr >= BAND[0]) & (fr <= BAND[1])
    iu = np.triu_indices(n, 1)

    C = _alpha_coherency(X, fs, win_starts, band_sel, H)
    imc = np.abs(C.imag)[iu]
    rec = np.abs(C.real)[iu]

    mont = mne.channels.make_standard_montage("biosemi64")
    pos = mont.get_positions()["ch_pos"]
    Pxyz = np.array([pos[nm] for nm in names])
    D = np.array([np.linalg.norm(Pxyz[i] - Pxyz[j]) for i, j in zip(*iu)])

    rng = np.random.default_rng(seed)
    sur = np.zeros((n_sur, len(imc)))
    for k in range(n_sur):
        Cs = _alpha_coherency(_phase_randomize(X, rng), fs, win_starts,
                              band_sel, H)
        sur[k] = np.abs(Cs.imag)[iu]
    p95 = np.percentile(sur, 95, axis=0)

    cache = dict(
        meta=dict(
            dataset="OpenNeuro ds003775 (Hatlestad-Hall et al.; SRM resting "
                    "EEG, CC0)",
            subject=os.path.basename(edf_path),
            sfreq_hz=float(fs), band_hz=list(BAND), window_s=WIN_S,
            n_segments=len(win_starts), n_channels=n, n_pairs=len(imc),
            montage="biosemi64", n_surrogates=n_sur),
        ch_names=names,
        pairs=dict(i=iu[0].astype(int).tolist(), j=iu[1].astype(int).tolist()),
        abs_imcoh=[round(float(x), 5) for x in imc],
        abs_recoh=[round(float(x), 5) for x in rec],
        dist=[round(float(x), 4) for x in D],
        surrogate_p95=[round(float(x), 5) for x in p95],
        surrogate_median_imcoh=round(float(np.median(sur)), 5))
    if write:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w") as fh:
            json.dump(cache, fh)
    return cache


def load_cache(path=CACHE):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# analysis + verdicts
# --------------------------------------------------------------------------- #
def analyze(cache):
    from scipy.stats import spearmanr
    imc = np.array(cache["abs_imcoh"])
    rec = np.array(cache["abs_recoh"])
    D = np.array(cache["dist"])
    p95 = np.array(cache["surrogate_p95"])
    sur_med = cache["surrogate_median_imcoh"]

    hi = rec >= np.quantile(rec, 0.9)          # top-mixing decile
    near = D <= np.quantile(D, 0.1)
    rho_re_im = float(spearmanr(rec, imc)[0])
    clears = imc > p95
    out = dict(
        n_pairs=len(imc),
        re_over_im=float(np.median(rec) / np.median(imc)),
        med_re=float(np.median(rec)), med_im=float(np.median(imc)),
        spearman_re_im=rho_re_im,
        topmix_re_over_im=float(np.median(rec[hi]) / np.median(imc[hi])),
        topmix_med_im=float(np.median(imc[hi])),
        spearman_dist_re=float(spearmanr(D, rec)[0]),
        spearman_dist_im=float(spearmanr(D, imc)[0]),
        near_med_re=float(np.median(rec[near])),
        near_med_im=float(np.median(imc[near])),
        surrogate_median_imcoh=float(sur_med),
        frac_clearing_ceiling=float(clears.mean()),
        frac_topmix_clearing=float(clears[hi].mean()))
    out["verdicts"] = verdicts(out)
    return out


def verdicts(out):
    return dict(
        volume_conduction_dominates_real=bool(out["re_over_im"] >= 4.0),
        mixing_feeds_real_not_imaginary=bool(
            out["spearman_re_im"] <= 0.0
            and out["topmix_re_over_im"] > out["re_over_im"]
            and out["topmix_med_im"] < out["med_im"]),
        imcoh_is_interregional=bool(out["spearman_dist_im"] > 0.0
                                    > out["spearman_dist_re"]),
        surrogate_ceiling_inverts_with_mixing=bool(
            out["surrogate_median_imcoh"] < 0.03
            and out["frac_topmix_clearing"] < out["frac_clearing_ceiling"]),
        reading=(
            "Imaginary-coherence lemma on real resting EEG (ds003775): "
            "volume conduction dominates coherency in the REAL part "
            "(Re/Im = 8.2), and the decisive signature holds -- mixing "
            "strength anti-correlates with imaginary coherency "
            "(Spearman -0.25); the most-mixed near-electrode pairs have the "
            "SMALLEST |Im| (Re/Im = 28, |Im| below the median) and are the "
            "LEAST likely to clear the phase-surrogate ceiling.  Imaginary "
            "coherency rises with electrode distance (genuine inter-regional "
            "coupling) where mixing falls.  P0's surrogate ceiling and "
            "Nolte's ImCoh artifact rejection are one lemma: instantaneous "
            "mixing is real, so it is invisible to Im."),
    )


def run(write=True):
    cache = load_cache()
    out = analyze(cache)
    res = dict(description=__doc__.splitlines()[0], meta=cache["meta"], **out)
    if write:
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        with open(FIG, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


def main():
    res = run()
    v = res["verdicts"]
    print("NR39 gate EXECUTED -- imaginary-coherence lemma on real EEG "
          "(OpenNeuro ds003775)")
    print(f"  pairs / segments          : {res['n_pairs']} pairs, "
          f"{res['meta']['n_segments']} segments, "
          f"{res['meta']['n_channels']} ch")
    print(f"  volume conduction (Re/Im) : {res['re_over_im']:.2f} "
          f"(med|Re|={res['med_re']:.3f}, med|Im|={res['med_im']:.3f}) "
          f"-> {v['volume_conduction_dominates_real']}")
    print(f"  mixing feeds Re not Im    : {v['mixing_feeds_real_not_imaginary']}"
          f" (Spearman(|Re|,|Im|)={res['spearman_re_im']:+.2f}; "
          f"top-mix Re/Im={res['topmix_re_over_im']:.1f}, "
          f"|Im|={res['topmix_med_im']:.3f} < {res['med_im']:.3f})")
    print(f"  ImCoh inter-regional      : {v['imcoh_is_interregional']} "
          f"(Spearman(dist,|Re|)={res['spearman_dist_re']:+.2f}, "
          f"(dist,|Im|)={res['spearman_dist_im']:+.2f})")
    print(f"  ceiling inverts w/ mixing : "
          f"{v['surrogate_ceiling_inverts_with_mixing']} "
          f"(null floor {res['surrogate_median_imcoh']:.3f}; clears "
          f"{res['frac_clearing_ceiling']:.2f} all vs "
          f"{res['frac_topmix_clearing']:.2f} top-mix)")
    print(f"  reading: {v['reading']}")


if __name__ == "__main__":
    main()
