r"""NR36 -- the well-barometric response IS a two-clocks kernel: USGS wells
select the operator composition, with the NR11/NR30/NR32 dictionary.

Ledger item E6 (papers/RESEARCH_RECAP_AND_HORIZON.md).  Rojstaczer (1988 WRR)
and Hsieh et al. (1987) describe how an open well responds to barometric
loading in three frequency bands.  The claim executed here: that response is
the two-clocks program's kernel family with renamed constants, and open USGS
NWIS records suffice to *identify which composition of the family a given
well obeys* -- the identification machinery of P4a (fit + information
criterion + honest nulls) transplanted onto a fully instrumented mainstream
system with public data.

The kernel family (all elements already in the repo)
----------------------------------------------------
With ``b`` the barometric head [m H2O] and ``d`` the depth-to-water [m]
(positive down, so a *confined* static response has ``+BE`` gain):

  * static plateau       ``BE``                    -- barometric efficiency;
  * wellbore/aquifer     ``L(w) = 1/(1 + i w tau_w)``   -- NR11's RC element:
    casing storage C charged through aquifer transmissivity R
    (``tau_w = RC``; phase ``-arctan(w tau_w)``, the NR11 form exactly);
  * water-table drainage ``D(w) = i w tau_l/(1 + i w tau_l)`` -- the
    complementary first-order element (NR32's two-compartment face): at DC
    the aquifer equilibrates with the water table and the confined response
    is *screened out* -- the operator-level analogue of NR30's screened
    Poisson (finite penetration replaces DC transmission);
  * vadose diffusion     ``V(w) = exp(-(1+i) sqrt(w tau_v / 2))`` -- the
    B.2/G.4 diffusive half-space transfer (t^{-1/2} family) for the air
    column above the water table.

Candidate compositions fit to each well (complex, coherence-weighted):

    M0:  W = BE                       (static only -- the null)
    M1:  W = BE * L                   (wellbore RC only)
    M2:  W = BE * D * L               (drainage x wellbore; Rojstaczer's
                                       confined/water-table band-pass)
    M3:  W = BE * V * L               (vadose x wellbore)

AICc on the same weighted complex residuals selects the composition; the
parameter dictionary (tau_w <-> R*C, tau_l <-> drainage time, tau_v <->
L_v^2/D_air) is emitted with the fit.  This is deliberately the same
grammar as P4a SS4/SS6: identify the kernel, state what its constants mean,
and let an information criterion referee.

Data
----
USGS NWIS instantaneous values, discovered programmatically: active
groundwater sites carrying BOTH 72019 (depth to water) and 00025 (on-site
barometric pressure) as unit values with multi-year overlap.  The two
longest-overlap co-located pairs in the 6-state scan (CA/NE/NV/TX/OK/KS):

    415546121205401, 415104121232901   (Modoc Plateau, CA; 2-hourly, 3 yr
                                        slices 2021-2024 used here)

Provision (no auth):
    curl -L "https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites=<SITE>
    &parameterCd=72019,00025&startDT=YYYY-MM-DD&endDT=YYYY-MM-DD"
    -> /home/data_nwis/iv_<SITE>.rdb   (chunks may be concatenated)

Method honesty
--------------
Earth-tide lines contaminate the well but not the barometer, so O1/P1/K1/
N2/M2/S2 bands (+-0.035 cpd) are excluded from all fits; the S1/S2
radiational lines are excluded conservatively even though partly true baro
forcing.  Welch cross-spectra (60-day Hann segments, 50% overlap); weights
``coh^2/(1-coh^2)`` (the Bendat-Piersol variance of a transfer-function
estimate); segment-level moving-block bootstrap for parameter CIs.  The
high-frequency tail carries instrument quantisation (0.01 ft steps), which
biases |W| down and phase toward 0 there -- stated, and down-weighted
automatically by coherence.

Artifacts -> figures/nr36_well_barometric_kernel.{json,png}
Run:  python general_two_clocks/new_relationships13.py
Test: pytest general_two_clocks/tests/test_new_relationships13.py -v
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import signal
from scipy.optimize import least_squares

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
DATA_DIR = "/home/data_nwis"
SITES = ("415546121205401", "415104121232901")

FT = 0.3048                      # feet -> m
MMHG = 0.0135951                 # mm Hg -> m H2O
FS_CPD = 12.0                    # 2-hourly sampling
TIDE_LINES_CPD = (0.9295, 0.9973, 1.0027, 1.8960, 1.9324, 2.0000)
TIDE_HALFWIDTH = 0.035
FBAND = (0.05, 4.0)


# ------------------------------------------------------------------ models --
def elem_wellbore(w, tau_w):
    """NR11 RC low-pass 1/(1+i w tau) -- phase -arctan(w tau)."""
    return 1.0 / (1.0 + 1j * w * tau_w)


def elem_drainage(w, tau_l):
    """Water-table drainage high-pass i w tau/(1+i w tau) (NR32 face):
    screens the DC response out -- the NR30 operator analogue."""
    return 1j * w * tau_l / (1.0 + 1j * w * tau_l)


def elem_vadose(w, tau_v):
    """Diffusive half-space transfer exp(-(1+i) sqrt(w tau_v/2))."""
    return np.exp(-(1.0 + 1j) * np.sqrt(w * tau_v / 2.0))


def model_W(name, w, params):
    if name == "M0_static":
        (be,) = params
        return be * np.ones_like(w, dtype=complex)
    if name == "M1_wellbore":
        be, tw = params
        return be * elem_wellbore(w, tw)
    if name == "M2_drainage_wellbore":
        be, tl, tw = params
        return be * elem_drainage(w, tl) * elem_wellbore(w, tw)
    if name == "M3_vadose_wellbore":
        be, tv, tw = params
        return be * elem_vadose(w, tv) * elem_wellbore(w, tw)
    if name == "M4_vadose":
        be, tv = params
        return be * elem_vadose(w, tv)
    if name == "M5_twopath":
        a1, a2, tv = params
        return a1 - a2 * elem_vadose(w, tv)
    if name == "M6_twopath_wellbore":
        a1, a2, tv, tw = params
        return (a1 - a2 * elem_vadose(w, tv)) * elem_wellbore(w, tw)
    if name == "M7_drainage":
        be, tl = params
        return be * elem_drainage(w, tl)
    raise KeyError(name)


MODELS = {
    "M0_static": (1, [0.5], [(0.0, 1.5)]),
    "M1_wellbore": (2, [0.5, 0.05], [(0.0, 1.5), (1e-4, 30.0)]),
    "M2_drainage_wellbore": (3, [0.6, 3.0, 0.05],
                             [(0.0, 1.5), (1e-3, 300.0), (1e-4, 30.0)]),
    "M3_vadose_wellbore": (3, [0.6, 0.2, 0.05],
                           [(0.0, 1.5), (1e-4, 300.0), (1e-4, 30.0)]),
    "M4_vadose": (2, [0.6, 0.2], [(0.0, 1.5), (1e-4, 300.0)]),
    "M5_twopath": (3, [0.8, 0.5, 0.2],
                   [(0.0, 1.5), (0.0, 1.5), (1e-4, 300.0)]),
    "M6_twopath_wellbore": (4, [0.8, 0.5, 0.2, 0.05],
                            [(0.0, 1.5), (0.0, 1.5), (1e-4, 300.0),
                             (1e-4, 30.0)]),
    "M7_drainage": (2, [0.6, 0.5], [(0.0, 1.5), (1e-3, 300.0)]),
}


# ------------------------------------------------------------------ parser --
def parse_nwis_rdb(fn):
    """Concatenated-chunk-safe NWIS rdb parser -> (datetime64[], depth_m,
    baro_mH2O).  Column ids differ per chunk; identified from each header."""
    recs = {}
    c72 = c25 = None
    for line in open(fn, errors="ignore"):
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if p[0] == "agency_cd":
            c72 = [i for i, c in enumerate(p) if c.endswith("_72019")]
            c25 = [i for i, c in enumerate(p) if c.endswith("_00025")]
            continue
        if p[0] != "USGS" or not c72 or not c25:
            continue
        try:
            t = np.datetime64(p[2].replace(" ", "T"))
            d = float(p[c72[0]]) * FT
            b = float(p[c25[0]]) * MMHG
        except (ValueError, IndexError):
            continue
        recs[t] = (d, b)
    ts = sorted(recs)
    if not ts:
        raise RuntimeError(f"no joint records parsed from {fn}")
    t = np.array(ts)
    d = np.array([recs[k][0] for k in ts])
    b = np.array([recs[k][1] for k in ts])
    return t, d, b


def regularise(t, d, b, step_h=2.0, max_gap_h=12.0):
    """Uniform grid by linear interpolation; returns (grid_h, d, b,
    coverage_frac).  Long gaps are interpolated too but reported."""
    ih = ((t - t[0]) / np.timedelta64(1, "h")).astype(float)
    grid = np.arange(0.0, ih[-1] + step_h / 2, step_h)
    di = np.interp(grid, ih, d)
    bi = np.interp(grid, ih, b)
    gaps = np.diff(ih)
    frac_long_gap = float(np.sum(gaps[gaps > max_gap_h]) / ih[-1])
    return grid, di, bi, 1.0 - frac_long_gap


# --------------------------------------------------------------- estimator --
def brf_welch(d, b, fs_cpd=FS_CPD, seg_days=60.0):
    """Welch transfer estimate W(f) = S_db/S_bb + magnitude-squared
    coherence, Hann, 50% overlap.

    CONVENTION GUARD: the transfer of d RESPONDING to b is
    H = <D conj(B)>/<|B|^2>.  scipy.signal.csd(x, y) returns
    <conj(X) Y>, so csd(b, d) (input FIRST) gives <conj(B) D> = the
    correct numerator; csd(d, b) would return conj(H) and silently flip
    every phase (unit-proved against a synthetic pure delay).
    """
    nper = int(seg_days * fs_cpd)
    f, Pbb = signal.welch(b - b.mean(), fs=fs_cpd, nperseg=nper)
    _, Pdd = signal.welch(d - d.mean(), fs=fs_cpd, nperseg=nper)
    _, Pbd = signal.csd(b - b.mean(), d - d.mean(), fs=fs_cpd, nperseg=nper)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.abs(Pbd) ** 2 / (Pbb * Pdd)
        W = Pbd / Pbb
    return f, W, coh


def analysis_mask(f, fband=FBAND, lines=TIDE_LINES_CPD, hw=TIDE_HALFWIDTH):
    m = (f >= fband[0]) & (f <= fband[1])
    for c in lines:
        m &= np.abs(f - c) > hw
    return m


def fit_models(f, W, coh, mask):
    """Coherence-weighted complex least squares for each candidate; AICc."""
    w = 2.0 * np.pi * f[mask]
    Wm = W[mask]
    g2 = np.clip(coh[mask], 1e-3, 0.999)
    wt = np.sqrt(g2 / (1.0 - g2))          # ~ 1/sigma of the W estimate
    n_res = 2 * Wm.size                    # real+imag residuals
    out = {}
    for name, (k, x0, bounds) in MODELS.items():
        lo = [bb[0] for bb in bounds]
        hi = [bb[1] for bb in bounds]

        def resid(x, name=name):
            Wf = model_W(name, w, x)
            r = (Wf - Wm) * wt
            return np.concatenate([r.real, r.imag])

        sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=20000)
        rss = float(np.sum(sol.fun ** 2))
        aic = n_res * np.log(rss / n_res) + 2 * k
        aicc = aic + 2 * k * (k + 1) / max(n_res - k - 1, 1)
        at_bound = [bool(abs(v - l) < 1e-12 or abs(v - h) < 1e-12)
                    for v, l, h in zip(sol.x, lo, hi)]
        out[name] = {"params": [float(v) for v in sol.x], "rss": rss,
                     "aicc": float(aicc), "k": k, "at_bound": at_bound}
    best = min(out, key=lambda n: out[n]["aicc"])
    for n in out:
        out[n]["delta_aicc"] = out[n]["aicc"] - out[best]["aicc"]
    return out, best


def _segment_spectra(d, b, seg_days=60.0, fs_cpd=FS_CPD):
    """Per-segment Hann periodograms (auto bb, dd; cross db) on 50%%-overlap
    segments -- the raw material Welch averages, kept per segment so the
    bootstrap can resample SEGMENTS in the spectral domain (no concatenation
    seams)."""
    nper = int(seg_days * fs_cpd)
    step = nper // 2
    win = np.hanning(nper)
    norm = fs_cpd * (win ** 2).sum()
    dd = d - d.mean()
    bb = b - b.mean()
    f = np.fft.rfftfreq(nper, d=1.0 / fs_cpd)
    Pbb, Pdd, Pdb = [], [], []
    for s0 in range(0, d.size - nper + 1, step):
        segd = dd[s0:s0 + nper]
        segb = bb[s0:s0 + nper]
        # per-segment constant detrend, matching scipy.signal.welch/csd
        # defaults (without it, slow water-table drift leaks into low f and
        # biases every bootstrap replicate)
        D = np.fft.rfft(win * (segd - segd.mean()))
        B = np.fft.rfft(win * (segb - segb.mean()))
        Pbb.append((np.abs(B) ** 2) / norm)
        Pdd.append((np.abs(D) ** 2) / norm)
        Pdb.append(D * np.conj(B) / norm)
    return f, np.array(Pbb), np.array(Pdd), np.array(Pdb)


def bootstrap_ci(d, b, best_name, n_boot=60, seg_days=60.0, seed=0):
    """Segment-spectra bootstrap: resample per-segment periodograms with
    replacement, average, re-estimate W and coherence, refit the winning
    model.  No time-domain concatenation, hence no seam artifacts."""
    rng = np.random.default_rng(seed)
    f, Pbb, Pdd, Pdb = _segment_spectra(d, b, seg_days=seg_days)
    nseg = Pbb.shape[0]
    m = analysis_mask(f)
    w = 2.0 * np.pi * f[m]
    k, x0, bounds = MODELS[best_name]
    lo = [bb_[0] for bb_ in bounds]
    hi = [bb_[1] for bb_ in bounds]
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, nseg, size=nseg)
        Sbb = Pbb[pick].mean(axis=0)
        Sdd = Pdd[pick].mean(axis=0)
        Sdb = Pdb[pick].mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            W = (Sdb / Sbb)[m]
            g2 = np.clip((np.abs(Sdb) ** 2 / (Sbb * Sdd))[m], 1e-3, 0.999)
        wt = np.sqrt(g2 / (1.0 - g2))

        def resid(x):
            r = (model_W(best_name, w, x) - W) * wt
            return np.concatenate([r.real, r.imag])

        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=20000)
            samples.append(sol.x)
        except Exception:                                # pragma: no cover
            continue
    s = np.array(samples)
    return {"p16": [float(v) for v in np.percentile(s, 16, axis=0)],
            "p84": [float(v) for v in np.percentile(s, 84, axis=0)],
            "n_boot": int(s.shape[0])}


# --------------------------------------------------------------------- run --
def run(data_dir=DATA_DIR, sites=SITES, n_boot=60):
    out = {"dictionary": {
        "BE": "barometric efficiency (static plateau)",
        "tau_w": "wellbore RC = casing storage x aquifer resistance "
                 "(NR11 element; phase -arctan(w tau_w))",
        "tau_l": "water-table drainage time (NR32 two-compartment face; "
                 "DC screening = NR30 operator analogue)",
        "tau_v": "vadose diffusion time L_v^2/D_air (B.2/G.4 t^{-1/2} "
                 "family)",
    }, "wells": {}}
    for s in sites:
        fn = os.path.join(data_dir, f"iv_{s}.rdb")
        if not os.path.exists(fn):
            out["wells"][s] = {"available": False, "hint": __doc__.split(
                "Provision")[1][:200]}
            continue
        t, d, b = parse_nwis_rdb(fn)
        grid, di, bi, coverage = regularise(t, d, b)
        f, W, coh = brf_welch(di, bi)
        m = analysis_mask(f)
        fits, best = fit_models(f, W, coh, m)
        ci = bootstrap_ci(di, bi, best, n_boot=n_boot)
        plateau = (f >= 0.3) & (f <= 1.5) & m
        out["wells"][s] = {
            "available": True,
            "n_samples": int(di.size), "coverage_frac": coverage,
            "span": [str(t[0]), str(t[-1])],
            "coh_mid_band_mean": float(coh[plateau].mean()),
            "absW_mid_band_mean": float(np.abs(W[plateau]).mean()),
            "fits": fits, "best_model": best, "best_ci_68": ci,
            "n_freq_bins_used": int(m.sum()),
        }
    return out, locals()


# ------------------------------------------------------------------ figure --
def make_figure(out, data_dir=DATA_DIR, sites=SITES, path_png=None):  # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(sites), figsize=(6.5 * len(sites), 7),
                             sharex=True)
    if len(sites) == 1:
        axes = axes.reshape(2, 1)
    for col, s in enumerate(sites):
        wi = out["wells"].get(s, {})
        if not wi.get("available"):
            continue
        t, d, b = parse_nwis_rdb(os.path.join(data_dir, f"iv_{s}.rdb"))
        _, di, bi, _ = regularise(t, d, b)
        f, W, coh = brf_welch(di, bi)
        m = analysis_mask(f)
        wgrid = 2 * np.pi * f[m]
        ax = axes[0, col]
        ax.semilogx(f[m], np.abs(W[m]), ".", ms=3, alpha=0.5, label="data")
        for name, st in wi["fits"].items():
            ax.semilogx(f[m], np.abs(model_W(name, wgrid, st["params"])),
                        lw=1.5 if name == wi["best_model"] else 0.8,
                        label=f"{name} (dAICc={st['delta_aicc']:.0f})")
        ax.set_ylabel("|W|  [m / m H2O]")
        ax.set_title(f"{s}\nbest: {wi['best_model']}")
        ax.legend(fontsize=6)
        ax2 = axes[1, col]
        ax2.semilogx(f[m], np.degrees(np.angle(W[m])), ".", ms=3, alpha=0.5)
        for name, st in wi["fits"].items():
            ax2.semilogx(f[m], np.degrees(np.angle(
                model_W(name, wgrid, st["params"]))),
                lw=1.5 if name == wi["best_model"] else 0.8)
        ax2t = ax2.twinx()
        ax2t.semilogx(f[m], coh[m], color="0.7", lw=0.7)
        ax2t.set_ylabel("coherence", color="0.5")
        ax2.set_xlabel("frequency [cpd]")
        ax2.set_ylabel("phase [deg]")
    fig.tight_layout()
    fig.savefig(path_png, dpi=140)
    plt.close(fig)


def main():                                                # pragma: no cover
    out, _ = run()
    os.makedirs(FIGDIR, exist_ok=True)
    jpath = os.path.join(FIGDIR, "nr36_well_barometric_kernel.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, path_png=os.path.join(
        FIGDIR, "nr36_well_barometric_kernel.png"))
    print(json.dumps(out, indent=1)[:3500])
    print(f"wrote {jpath} and the png")


if __name__ == "__main__":                                 # pragma: no cover
    main()
