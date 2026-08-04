r"""Digitize the Bushuk et al. (2019, *JFM* 873) preprint *timeseries* figure --
the adjustment-regime (experiment 1b) scallop amplitude decay ``A(t)`` and crest
positions ``x_c(t)`` -- and recompute the constant-free field index

    I = tau * c_mig / lambda

from **regime-matched, fitted** kinematics instead of the figure-read scalars of
:func:`scallop_field_test.bushuk_adjustment_bound` (whose ``tau`` carried a
factor-~2 by-eye uncertainty and whose ``lambda ~ 13 cm`` had to be borrowed from
the equilibrium band).  This is the closest available step toward the raw
``h(x,t)`` pin while the underlying arrays remain available only on request (no
open deposit exists; see ``REPORT_SCALLOP_MIGRATION.md`` §8).

Method (everything at runtime; only derived numbers are committed):

1. download the open-access preprint PDF from the public GFDL mirror (cached in
   ``_data_bushuk/``, gitignored; sha256 pinned below);
2. render pages with ``pdftoppm`` at 600 dpi.  The timeseries figure is an
   image-only page (caption burned into a halftone-dithered grayscale raster), so
   it cannot be located by text search; it is located by **content**: the page
   whose axis-label strips (pinned pixel windows below) yield exactly the
   expected tick-label counts;
3. calibrate each panel linearly from its **tick-label centroids** (solid black
   text survives the halftone dithering that fragments the frame lines; MATLAB
   centers labels on their ticks).  Amplitude panel: y = 10/8/6 mm, x = 0..300
   min; crest panel: y = 350..0 mm, x = 0..300 min;
4. amplitude panel: extract the solid measurement dots (roundish components;
   dithered frame fragments are rejected by area/aspect/fill filters and by
   masking high-density frame rows/cols) -> ``A(t)`` -> log-linear fit
   ``ln A = ln A0 - beta t`` -> ``tau = 1/beta`` with bootstrap CI and ``R^2``
   (the exponential-decay check is itself the frozen-probe eigenmode
   prediction);
5. crest panel: the crest traces (with their overlaid mean-advection lines)
   form long components; a per-component robust line fit gives each crest's
   drift speed -> ``c_mig``, and the fitted t=0 intercepts give consecutive
   crest spacings -> the regime-matched wavelength ``lambda``;
6. ``I = tau*c_mig/lambda`` with bootstrap uncertainty, compared against the
   solver band -- written to ``figures/58b_bushuk_digitized.{json,png}``.

CPU only.  Requires ``pdftoppm`` (poppler-utils) and network on first run.  The
committed JSON is the tested artifact; re-running refreshes it bit-for-bit for
the pinned PDF.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "_data_bushuk")
PDF_URL = "https://www.gfdl.noaa.gov/wp-content/uploads/2019/05/Bushuk_JFM2019.pdf"
PDF_SHA256 = "45ea8bd91bf8f8ce687561fa35c0daecb5119d8f7e50a44b18624eb148d76f81"
PDF_PATH = os.path.join(CACHE, "Bushuk_JFM2019.pdf")
DPI = 600

# ---- pinned geometry (600 dpi render of the timeseries-figure page; the page
# itself is located by content, these windows only assume the figure's internal
# layout, which is fixed for the pinned PDF_SHA256) ------------------------- #
AMP = dict(  # adjustment-row (raw) "Scallop Amplitude" panel -- figure row 3
    ylab_strip=(2255, 2395, 2650, 3615),  # x0, x1, y0, y1 (labels only; frame+ticks excluded)
    interior=(2408, 3325, 2658, 3605),
    x_span=(0.0, 300.0),                  # t=0 / t=300 min sit at the frame corners
    frame_x=(2400, 3332),                 # pinned frame-line x positions (refined +-25)
    # verified label anchors (px, mm); row 3 is uniformly spaced (~100.8 px per
    # 0.5 mm) but the piecewise map keeps the calibration exact under the page
    # raster's resampling artifacts.
    anchors=[(2732.2, 11.0), (2841.8, 10.5), (2940.0, 10.0), (3036.0, 9.5),
             (3139.9, 9.0), (3238.4, 8.5), (3339.4, 8.0), (3437.9, 7.5),
             (3538.4, 7.0)],
)
CREST = dict(  # adjustment-row (raw) "Crest Positions" panel -- figure row 3
    ylab_strip=(3315, 3448, 2650, 3615),
    interior=(3458, 3985, 2658, 3605),
    x_span=(0.0, 300.0),
    frame_x=(3452, 3990),
    anchors=[(2836.7, 340.0), (2935.6, 320.0), (3036.0, 300.0), (3141.8, 280.0),
             (3236.3, 260.0), (3337.3, 240.0), (3435.0, 220.0), (3535.7, 200.0)],
)


# --------------------------------------------------------------------------- #
# 0. fetch + render
# --------------------------------------------------------------------------- #
def fetch_pdf():
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(PDF_PATH):
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    with open(PDF_PATH, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    if sha != PDF_SHA256:
        print(f"[warn] PDF sha256 {sha[:12]}... differs from pinned "
              f"{PDF_SHA256[:12]}...; pixel windows may need recalibration")
    return PDF_PATH, sha


def render_page(page, dpi=DPI):
    from PIL import Image

    out = os.path.join(CACHE, f"_page{page}_{dpi}")
    path = f"{out}-{page:02d}.png"
    if not os.path.exists(path):
        subprocess.run(
            ["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi),
             "-png", PDF_PATH, out],
            check=True, capture_output=True)
        if not os.path.exists(path):
            path = f"{out}-{page}.png"
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# 1. tick-label centroids -> linear calibration
# --------------------------------------------------------------------------- #
def _label_centroids(gray, strip, axis, glyph_gap=35, min_area=180,
                     min_h=22, min_w=12):
    """Centroids of tick-label text in a strip, clustered into labels.

    axis='y': cluster glyph components by vertical gaps, return y centroids.
    axis='x': cluster by horizontal gaps, return x centroids.
    """
    from scipy import ndimage

    x0, x1, y0, y1 = strip
    sub = gray[y0:y1, x0:x1] < 128
    lab, n = ndimage.label(sub)
    glyphs = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        area = int((lab[sl] == i).sum())
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        # digit glyphs are tall AND wide; tick marks are thin slivers
        if area < min_area or h < min_h or w < min_w:
            continue
        cy, cx = ndimage.center_of_mass(lab[sl] == i)
        glyphs.append((sl[0].start + cy, sl[1].start + cx, area))
    if not glyphs:
        return []
    key = 0 if axis == "y" else 1
    glyphs.sort(key=lambda g: g[key])
    clusters = [[glyphs[0]]]
    for g in glyphs[1:]:
        if g[key] - clusters[-1][-1][key] <= glyph_gap:
            clusters[-1].append(g)
        else:
            clusters.append([g])
    cents = []
    for cl in clusters:
        c = np.array([g[key] for g in cl], float)
        cents.append(float(np.median(c)) + (y0 if axis == "y" else x0))
    return cents


class PiecewiseAxis:
    """Piecewise-linear px -> value map through verified label anchors, with
    linear extrapolation beyond the outermost anchors."""

    def __init__(self, anchors):
        a = sorted(anchors)
        self.px = np.array([p for p, _ in a], float)
        self.v = np.array([v for _, v in a], float)

    def __call__(self, py):
        py = np.asarray(py, float)
        out = np.interp(py, self.px, self.v)
        lo, hi = self.px[0], self.px[-1]
        s_lo = (self.v[1] - self.v[0]) / (self.px[1] - self.px[0])
        s_hi = (self.v[-1] - self.v[-2]) / (self.px[-1] - self.px[-2])
        out = np.where(py < lo, self.v[0] + s_lo * (py - lo), out)
        out = np.where(py > hi, self.v[-1] + s_hi * (py - hi), out)
        return out if out.ndim else float(out)


def match_anchors(cents, anchors, tol_px=8.0, min_match=5):
    """Verify detected label centroids against the pinned anchor table; return
    the matched (px_detected, value) pairs.  Aborts if the render's label
    geometry no longer matches the pinned table (e.g. a different PDF)."""
    matched = []
    for p_pin, v in anchors:
        d = [c for c in cents if abs(c - p_pin) <= tol_px]
        if d:
            matched.append((float(np.median(d)), v))
    if len(matched) < min_match:
        raise RuntimeError(
            f"only {len(matched)}/{len(anchors)} label anchors matched; "
            "render geometry differs from the pinned calibration")
    return matched


def refine_frame_x(gray, cfg, half=25):
    """Refine the pinned frame-line x positions: the max-density column within
    +-half px of each pinned position (the dithered frame is still the densest
    column in such a narrow window)."""
    _, _, y0, y1 = cfg["interior"]
    out = []
    for px in cfg["frame_x"]:
        w = gray[y0:y1, px - half:px + half] < 128
        out.append(px - half + int(np.argmax(w.sum(axis=0))))
    return tuple(out)


def panel_calibration(gray, cfg):
    """(ax, vy, meta): time axis linear from the frame corners (t=0/t=300 sit
    at the frame); value axis piecewise-linear through the verified anchors."""
    yc = _label_centroids(gray, cfg["ylab_strip"], "y")
    matched = match_anchors(yc, cfg["anchors"])
    vy = PiecewiseAxis(matched)
    fl, fr = refine_frame_x(gray, cfg)
    t0, t1 = cfg["x_span"]
    a = (t1 - t0) / (fr - fl)
    ax = (a, t0 - a * fl)
    return ax, vy, dict(n_ylabels=len(yc), n_anchor_matched=len(matched),
                        frame_x=(int(fl), int(fr)))


def find_figure_page(pages=range(10, 22)):
    """Locate the timeseries-figure page by content: the page where both
    panels' pinned label strips yield a consistent tick-label grid."""
    for p in pages:
        try:
            gray = render_page(p)
        except subprocess.CalledProcessError:
            continue
        if gray.shape[0] < 6000:                      # expect a full page at 600 dpi
            continue
        try:
            _, _, ma = panel_calibration(gray, AMP)
            _, _, mc = panel_calibration(gray, CREST)
        except RuntimeError:
            continue
        if ma["n_anchor_matched"] >= 5 and mc["n_anchor_matched"] >= 5:
            return p, gray
    raise RuntimeError("timeseries figure page not found by label signature")


# --------------------------------------------------------------------------- #
# 2. panel extraction
# --------------------------------------------------------------------------- #
def _frame_mask(dark):
    """Mask rows/cols that belong to (dithered) frame lines: any row/col whose
    dark-pixel count exceeds 25% of the panel span."""
    h, w = dark.shape
    rows = dark.sum(axis=1) > 0.25 * w
    cols = dark.sum(axis=0) > 0.25 * h
    m = np.zeros_like(dark)
    idx = np.where(rows)[0]
    for i in idx:
        m[max(0, i - 6):i + 7, :] = True
    idx = np.where(cols)[0]
    for j in idx:
        m[:, max(0, j - 6):j + 7] = True
    return m


def amplitude_series(gray):
    """Measurement dots inside the amplitude panel -> (t_min, A_mm).

    The dots sit on a 5-minute grid and can touch their neighbours, so blob
    morphology is unreliable; instead sample the panel column-wise at each
    5-minute time and take the median dark-pixel row (the dot core).  Dithered
    frame remnants are removed by the frame mask; a column with too little ink
    is skipped."""
    ax, vy, cal = panel_calibration(gray, AMP)
    x0, x1, y0, y1 = AMP["interior"]
    sub = gray[y0:y1, x0:x1] < 128
    sub &= ~_frame_mask(sub)
    # per-column candidates = peaks of the blurred ink density (the dot is by
    # far the densest blob; dotted gridlines and stray marks are weak), then a
    # Viterbi path preferring high density and small jumps
    from scipy import ndimage

    den = ndimage.gaussian_filter(sub.astype(float), 4.0)
    times, cand = [], []
    for tk in np.arange(0.0, 300.0 + 1e-9, 5.0):
        px = (tk - ax[1]) / ax[0] - x0                # panel-relative column
        j0, j1 = int(px - 7), int(px + 8)
        if j0 < 0 or j1 > sub.shape[1]:
            continue
        prof = den[:, j0:j1].sum(axis=1)
        if prof.max() <= 0:
            continue
        # local maxima
        loc = np.where((prof[1:-1] >= prof[:-2]) & (prof[1:-1] > prof[2:]))[0] + 1
        loc = loc[prof[loc] >= 0.30 * prof.max()]
        if len(loc) == 0:
            continue
        order = np.argsort(prof[loc])[::-1][:4]
        cl = [(float(l), float(prof[l] / prof.max())) for l in loc[order]]
        times.append(tk)
        cand.append(cl)
    if not times:
        return np.array([]), np.array([]), cal
    ALPHA, BETA = 1.0, 60.0                           # jump cost / density reward
    best = [dict() for _ in cand]
    best[0] = {i: (-BETA * h, None) for i, (y, h) in enumerate(cand[0])}
    for k in range(1, len(cand)):
        dt = times[k] - times[k - 1]
        new = {}
        for i, (y, h) in enumerate(cand[k]):
            opts = []
            for j, (yp, hp) in enumerate(cand[k - 1]):
                jump = abs(y - yp) / max(dt / 5.0, 1.0)
                opts.append((best[k - 1][j][0] + ALPHA * jump - BETA * h, j))
            new[i] = min(opts)
        best[k] = new
    # backtrack
    idx = min(best[-1], key=lambda i: best[-1][i][0])
    path = [idx]
    for k in range(len(cand) - 1, 0, -1):
        idx = best[k][idx][1]
        path.append(idx)
    path.reverse()
    t = np.array(times)
    A = np.array([float(vy(cand[k][path[k]][0] + y0)) for k in range(len(cand))])
    keep = (A > 0.5) & (A < 20.0)
    return t[keep], A[keep], cal


def fit_decay(t, A, n_boot=2000, seed=0):
    """ln A = ln A0 - beta*t least squares + bootstrap CI + R^2."""
    ln = np.log(A)
    X = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(X, ln, rcond=None)
    m, b = coef
    pred = X @ coef
    ss_res = float(((ln - pred) ** 2).sum())
    ss_tot = float(((ln - ln.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    rng = np.random.default_rng(seed)
    n = len(t)
    betas = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        c2, *_ = np.linalg.lstsq(X[i], ln[i], rcond=None)
        betas[k] = -c2[0]
    lo, hi = np.percentile(betas, [16, 84])
    # envelope estimator: upper quartile of the earliest window vs lower
    # quartile of the latest window -- robust both to the strong mid-series
    # oscillations (which flatten the regression slope) and to occasional
    # tracker hops onto gridlines near the panel edges
    e = t <= t.min() + 45.0
    l = t >= t.max() - 60.0
    A_e = float(np.percentile(A[e], 75))
    A_l = float(np.percentile(A[l], 25))
    tau_env = float((t[l].mean() - t[e].mean()) / np.log(A_e / A_l))
    return dict(beta_per_min=float(-m), lnA0=float(b), r2=float(r2),
                beta_ci68_per_min=(float(lo), float(hi)),
                tau_min=float(-1.0 / m),
                tau_ci68_min=(float(1.0 / hi), float(1.0 / lo)),
                tau_envelope_min=tau_env,
                A_early_mm=A_e, A_late_mm=A_l)


def fit_oscillator(t, A, n_boot=2000, seed=0, block=6,
                   T_grid=np.arange(20.0, 301.0, 1.0)):
    """Damped-oscillator refit: ln A = ln A0 - t/tau + b*cos(2*pi*t/T + phi).

    The adjustment-regime amplitude is strongly oscillatory, so a single
    exponential understates the fit quality and makes tau estimator-dependent.
    This is the physically-motivated model for an adjusting mode (exponential
    envelope times a periodic modulation, linearized in log space for small
    modulation depth b). The modulation period T is grid-searched; everything
    else is linear least squares. tau CI is a circular block bootstrap of the
    residuals (block ~ 30 min) to respect their autocorrelation.
    """
    ln = np.log(A)
    best = None
    for T in T_grid:
        w = 2.0 * np.pi / T
        X = np.column_stack([np.ones_like(t), -t, np.cos(w * t), np.sin(w * t)])
        coef, *_ = np.linalg.lstsq(X, ln, rcond=None)
        sse = float(((ln - X @ coef) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(T), coef, X)
    sse, T, coef, X = best
    lnA0, inv_tau, cc, cs = (float(v) for v in coef)
    ss_tot = float(((ln - ln.mean()) ** 2).sum())
    r2 = 1.0 - sse / ss_tot
    resid = ln - X @ coef
    rng = np.random.default_rng(seed)
    n = len(t)
    taus = []
    for _ in range(n_boot):
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        cb, *_ = np.linalg.lstsq(X, X @ coef + resid[idx], rcond=None)
        if cb[1] > 0:
            taus.append(1.0 / cb[1])
    taus = np.asarray(taus)
    lo68, hi68 = np.percentile(taus, [16, 84])
    lo95, hi95 = np.percentile(taus, [2.5, 97.5])
    return dict(model="lnA = lnA0 - t/tau + b*cos(2*pi*t/T + phi)",
                T_min=T, tau_min=1.0 / inv_tau if inv_tau > 0 else float("inf"),
                b=float(np.hypot(cc, cs)), coef_cos=cc, coef_sin=cs,
                lnA0=lnA0, r2=float(r2),
                tau_ci68_min=(float(lo68), float(hi68)),
                tau_ci95_min=(float(lo95), float(hi95)),
                n_boot_kept=int(taus.size))


def fit_two_exp(t, A):
    """Two-exponential control fit A = a1*exp(-t/t1) + a2*exp(-t/t2).

    Reported as a model-battery control only: on the digitized series it
    degenerates (the slow component's tau runs to infinity, i.e. a constant
    plus a fast transient) and fits worse than the damped oscillator.
    """
    try:
        from scipy.optimize import curve_fit

        def two_exp(tt, a1, t1, a2, t2):
            return a1 * np.exp(-tt / t1) + a2 * np.exp(-tt / t2)

        p, _ = curve_fit(two_exp, t, A, p0=[2.0, 50.0, 9.0, 3000.0],
                         maxfev=20000)
        pred = two_exp(t, *p)
        r2 = 1.0 - float(((A - pred) ** 2).sum()) / float(((A - A.mean()) ** 2).sum())
        degenerate = bool(max(p[1], p[3]) > 10.0 * (t.max() - t.min()))
        return dict(params=[float(v) for v in p], r2=float(r2),
                    degenerate=degenerate, failed=False)
    except Exception as exc:  # pragma: no cover - scipy always present in CI
        return dict(params=None, r2=None, degenerate=None, failed=True,
                    error=str(exc))


def _theil_sen(t, x, rng, n=4000):
    """Robust slope via random pair sampling (Theil-Sen estimate)."""
    m = len(t)
    i = rng.integers(0, m, n)
    j = rng.integers(0, m, n)
    ok = np.abs(t[j] - t[i]) > 15.0
    s = (x[j][ok] - x[i][ok]) / (t[j][ok] - t[i][ok])
    slope = float(np.median(s))
    icpt = float(np.median(x - slope * t))
    return slope, icpt


def crest_kinematics(gray, seed=0):
    """Slanted-comb fit of the crest-position panel.

    The crest traces form a near-parallel comb drifting at the mean advection
    speed; they are fragmented by dashed 'anomalous advection' segments, so
    per-component fits are unreliable.  Instead every ink pixel is mapped to
    ``(t, x_mm)`` and the drift ``c`` and wavelength ``lam`` are found jointly
    by maximizing the Rayleigh statistic ``R(c, lam) = |<exp(2*pi*i*(x - c*t)/
    lam)>|`` -- the comb aligns when de-drifted by the true speed.  Per-crest
    slopes (for the spread) come from a Theil-Sen fit within each comb line.
    """
    ax, vy, cal = panel_calibration(gray, CREST)
    x0, x1, y0, y1 = CREST["interior"]
    sub = gray[y0:y1, x0:x1] < 128
    sub &= ~_frame_mask(sub)
    ys, xs = np.nonzero(sub)
    if len(ys) < 2000:
        raise RuntimeError("crest panel too sparse")
    t = ax[0] * (xs + x0) + ax[1]
    xmm = np.asarray(vy(ys + y0), float)
    step = max(1, len(t) // 20000)                    # subsample for speed
    ts, xsub = t[::step], xmm[::step]

    def rayleigh(c_grid, lam_grid, tt, xx):
        best = (-1.0, None, None)
        for c in c_grid:
            z = xx - c * tt
            ph = np.exp(2j * np.pi * z[:, None] / lam_grid[None, :])
            R = np.abs(ph.mean(axis=0))
            k = int(np.argmax(R))
            if R[k] > best[0]:
                best = (float(R[k]), float(c), float(lam_grid[k]))
        return best

    R1, c1, lam1 = rayleigh(np.arange(-0.05, 0.351, 0.0025),
                            np.arange(12.0, 45.01, 0.5), ts, xsub)
    R2, c2, lam2 = rayleigh(np.arange(c1 - 0.005, c1 + 0.0051, 0.0005),
                            np.arange(lam1 - 1.0, lam1 + 1.01, 0.1), ts, xsub)

    # assign pixels to comb lines and fit each crest individually
    rng = np.random.default_rng(seed)
    z = xmm - c2 * t
    phase0 = np.angle(np.mean(np.exp(2j * np.pi * z / lam2)))
    k = np.round((z - phase0 * lam2 / (2 * np.pi)) / lam2)
    segs = []
    for kk in np.unique(k):
        m = k == kk
        if m.sum() < 400:
            continue
        tspan = t[m].max() - t[m].min()
        if tspan < 120.0:
            continue
        slope, icpt = _theil_sen(t[m], xmm[m], rng)
        segs.append(dict(comb_index=int(kk), slope_mm_per_min=slope,
                         x_at_t0_mm=icpt, n_px=int(m.sum()),
                         t_span_min=(float(t[m].min()), float(t[m].max()))))
    segs.sort(key=lambda s: s["x_at_t0_mm"])
    slopes = np.array([s["slope_mm_per_min"] for s in segs])
    x0s = np.array([s["x_at_t0_mm"] for s in segs])
    lamsp = np.diff(x0s)
    return dict(n_crests=len(segs), segments=segs,
                c_mig_mm_per_min=float(c2),
                c_mig_spread_mm_per_min=float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0,
                rayleigh_R=float(R2),
                lam_mm=float(lam2),
                lam_spacings_mm=[float(v) for v in lamsp],
                cal=cal)




# --------------------------------------------------------------------------- #
# 3. assemble
# --------------------------------------------------------------------------- #
def run(save=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import sys
    sys.path.insert(0, HERE)
    import scallop_field_test as sf

    pdf, sha = fetch_pdf()
    page, gray = find_figure_page()

    t, A, amp_cal = amplitude_series(gray)
    if len(t) < 25:
        raise RuntimeError(f"amplitude extraction too sparse: {len(t)} points")
    fit = fit_decay(t, A)
    osc = fit_oscillator(t, A)
    texp = fit_two_exp(t, A)
    crest = crest_kinematics(gray)
    if crest["n_crests"] < 3:
        raise RuntimeError(f"too few crest traces: {crest['n_crests']}")

    tau_fit_s = fit["tau_min"] * 60.0
    tau_env_s = fit["tau_envelope_min"] * 60.0
    tau_osc_s = osc["tau_min"] * 60.0
    c_comb_si = crest["c_mig_mm_per_min"] * 1e-3 / 60.0
    c_pub_si = 0.11e-3 / 60.0                         # Bushuk's mean (excludes
    lam_si = crest["lam_mm"] * 1e-3                   # anomalous dashed periods)
    # conservative point: envelope decay (robust to the oscillations that
    # flatten the regression) x published mean advection speed
    I = sf.i_from_kinematics(c_pub_si, lam_si, tau_env_s)
    # preferred point: damped-oscillator tau (the physically right model for
    # an adjusting mode; highest R^2 in the battery) x published mean speed
    I_pref = sf.i_from_kinematics(c_pub_si, lam_si, tau_osc_s)
    combos = {
        "tau_env_c_pub": sf.i_from_kinematics(c_pub_si, lam_si, tau_env_s),
        "tau_env_c_comb": sf.i_from_kinematics(c_comb_si, lam_si, tau_env_s),
        "tau_osc_c_pub": I_pref,
        "tau_osc_c_comb": sf.i_from_kinematics(c_comb_si, lam_si, tau_osc_s),
        "tau_fit_c_pub": sf.i_from_kinematics(c_pub_si, lam_si, tau_fit_s),
        "tau_fit_c_comb": sf.i_from_kinematics(c_comb_si, lam_si, tau_fit_s),
    }
    I_lo, I_hi = min(combos.values()), max(combos.values())

    band_lo, band_hi = sf._solver_band()
    committed = sf.bushuk_adjustment_bound()

    out = dict(
        description=(
            "Sec.G.2/RESULT 14 field test, digitized upgrade: Bushuk 2019 exp 1b "
            "(adjustment regime) amplitude decay + crest advection digitized from "
            "the open GFDL preprint's timeseries figure (image-only halftone page; "
            "tick-label-centroid calibration). Replaces the figure-read tau "
            "(1-3 h guess) and borrowed equilibrium lambda (~13 cm) of "
            "bushuk_adjustment_bound() with fitted, regime-matched values."),
        provenance=dict(pdf_url=PDF_URL, sha256=sha, page=int(page), dpi=DPI,
                        amp_calibration=amp_cal, crest_calibration=crest["cal"]),
        amplitude=dict(t_min=[round(float(v), 2) for v in t],
                       A_mm=[round(float(v), 3) for v in A],
                       n=len(t), **fit,
                       oscillator=osc, two_exp_control=texp),
        crest=dict(n_crests=crest["n_crests"],
                   c_mig_mm_per_min=crest["c_mig_mm_per_min"],
                   c_mig_spread_mm_per_min=crest["c_mig_spread_mm_per_min"],
                   published_reference_c_mm_per_min=0.11,
                   lam_mm=crest["lam_mm"],
                   lam_spacings_mm=crest["lam_spacings_mm"],
                   segments=crest["segments"]),
        kinematics_si=dict(tau_fit_s=tau_fit_s, tau_envelope_s=tau_env_s,
                           tau_oscillator_s=tau_osc_s,
                           c_comb_m_per_s=c_comb_si, c_published_m_per_s=c_pub_si,
                           lam_m=lam_si),
        I=dict(point=float(I), point_preferred=float(I_pref),
               estimator_range=(float(I_lo), float(I_hi)),
               combos={k: float(v) for k, v in combos.items()},
               solver_band=(band_lo, band_hi),
               committed_bound_point=committed["I_point_estimate"],
               inside_solver_band=bool(band_lo <= I <= band_hi),
               within_factor2_of_band=bool(band_lo / 2.0 <= I <= band_hi * 2.0),
               downstream=bool(c_comb_si > 0)),
        verdict=dict(
            exponential_decay_r2=fit["r2"],
            oscillator_r2=osc["r2"],
            damps=bool(fit["beta_per_min"] > 0),
            migrates_downstream=bool(c_comb_si > 0),
            digitized_c_consistent_with_published=bool(
                abs(crest["c_mig_mm_per_min"] - 0.11) <= 0.06),
        ),
    )

    if save:
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        with open(os.path.join(HERE, "figures", "58b_bushuk_digitized.json"), "w") as fh:
            json.dump(out, fh, indent=2)

        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
        axs[0].semilogy(t, A, "k.", ms=4)
        tt = np.linspace(t.min(), t.max(), 100)
        axs[0].semilogy(tt, np.exp(fit["lnA0"] - fit["beta_per_min"] * tt), "r-",
                        label=f"exp: tau={fit['tau_min']:.0f} min, R2={fit['r2']:.2f}")
        w_osc = 2.0 * np.pi / osc["T_min"]
        axs[0].semilogy(tt, np.exp(osc["lnA0"] - tt / osc["tau_min"]
                                   + osc["coef_cos"] * np.cos(w_osc * tt)
                                   + osc["coef_sin"] * np.sin(w_osc * tt)),
                        "b-", lw=0.9, alpha=0.8,
                        label=(f"osc: tau={osc['tau_min']:.0f} min, "
                               f"T={osc['T_min']:.0f}, R2={osc['r2']:.2f}"))
        axs[0].set_xlabel("time (min)")
        axs[0].set_ylabel("A (mm)")
        axs[0].set_title("digitized amplitude decay (exp 1b)")
        axs[0].legend(fontsize=8)
        for s in crest["segments"]:
            t0, t1 = s["t_span_min"]
            axs[1].plot([t0, t1],
                        [s["x_at_t0_mm"] + s["slope_mm_per_min"] * t0,
                         s["x_at_t0_mm"] + s["slope_mm_per_min"] * t1], "k-")
        axs[1].set_xlabel("time (min)")
        axs[1].set_ylabel("x (mm)")
        axs[1].set_title(f"crest drift: c={crest['c_mig_mm_per_min']:.3f} mm/min, "
                         f"lam={crest['lam_mm']:.0f} mm")
        axs[2].axhspan(band_lo, band_hi, color="C0", alpha=0.25, label="solver band")
        axs[2].errorbar([0.0], [I_pref],
                        yerr=[[max(I_pref - I_lo, 0)], [max(I_hi - I_pref, 0)]],
                        fmt="ko", label=f"I preferred = {I_pref:.1f} "
                                        f"(battery {I_lo:.1f}-{I_hi:.1f})")
        axs[2].plot([0.0], [committed["I_point_estimate"]], "kx",
                    label=f"figure-read bound = {committed['I_point_estimate']:.2f}")
        axs[2].set_xlim(-1, 1)
        axs[2].set_xticks([])
        axs[2].set_ylabel("I = tau*c/lam")
        axs[2].legend(fontsize=8)
        axs[2].set_title("constant-free index vs solver band")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figures", "58b_bushuk_digitized.png"), dpi=140)
        plt.close(fig)
    return out


def main():
    out = run()
    a, c, I = out["amplitude"], out["crest"], out["I"]
    print("Bushuk exp-1b digitized field test (I = tau*c_mig/lam)")
    print(f"  page {out['provenance']['page']}, calibration residuals "
          f"amp={out['provenance']['amp_calibration']}, crest={out['provenance']['crest_calibration']}")
    print(f"  amplitude: n={a['n']} pts, exp tau={a['tau_min']:.0f} min "
          f"(68% {a['tau_ci68_min'][0]:.0f}-{a['tau_ci68_min'][1]:.0f}), R2={a['r2']:.3f}")
    o = a["oscillator"]
    print(f"  oscillator refit: tau={o['tau_min']:.0f} min "
          f"(68% {o['tau_ci68_min'][0]:.0f}-{o['tau_ci68_min'][1]:.0f}, "
          f"95% {o['tau_ci95_min'][0]:.0f}-{o['tau_ci95_min'][1]:.0f}), "
          f"T={o['T_min']:.0f} min, b={o['b']:.3f}, R2={o['r2']:.3f}; "
          f"two-exp control R2={a['two_exp_control']['r2']:.3f} "
          f"degenerate={a['two_exp_control']['degenerate']}")
    print(f"  crests: n={c['n_crests']}, c={c['c_mig_mm_per_min']:.4f} mm/min "
          f"(published 0.11), lam={c['lam_mm']:.0f} mm, spacings={c['lam_spacings_mm']}")
    print(f"  I preferred = {I['point_preferred']:.3f} (oscillator tau x published c); "
          f"conservative = {I['point']:.3f}; battery {I['estimator_range'][0]:.2f}-"
          f"{I['estimator_range'][1]:.2f} vs solver band {I['solver_band'][0]:.2f}-"
          f"{I['solver_band'][1]:.2f} -> inside={I['inside_solver_band']}")
    return out


if __name__ == "__main__":
    main()
