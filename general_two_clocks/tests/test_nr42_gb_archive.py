"""Unit proofs for the NR42 real-data gate execution
(`general_two_clocks/nr42_gb_archive.py`): GB 1-s frequency archives vs
measured Outturn Inertia.  Offline-safe -- all numbers come from the committed
aggregate cache ``data/nr42_gb_archive_cache.json`` (derived from NESO Data
Portal frequency/inertia/demand datasets; regeneration via ``build_cache()``
with ``$NR42_GB_DIR`` set).

Covered: the identifiable SFR transmission shape (limits + parameter
recovery); the octile-ordering machinery on synthetic monotone/peaked data;
cache schema; the registered dichotomy on the real archives (Markovian null
rejected in every stratum, window-side ordering confirmed in the controlled
strata); the honest non-resolution of the interior peak (range under-spans
the band); band specificity (out-of-band inversion); the DC-era flattening;
and the ambient-knee readout confound.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from nr42_gb_archive import (  # noqa: E402
    BANDS, FIG, N_OCT, _octile_block, analyze, fit_sfr, load_cache,
    sfr_psd_shape,
)


@pytest.fixture(scope="module")
def cache():
    return load_cache()


@pytest.fixture(scope="module")
def out(cache):
    return analyze(cache)


# --------------------------------------------------------------------------- #
# SFR shape closed forms (synthetic)
# --------------------------------------------------------------------------- #
def test_sfr_shape_limits():
    # low-f plateau -> 1 (beta-normalized); high-f -> 1/(w tau)^2
    w = np.array([1e-6, 1e3])
    s = sfr_psd_shape(w, tau=10.0, k=0.7, tg=8.0)
    assert s[0] == pytest.approx(1.0, rel=1e-3)
    assert s[1] == pytest.approx(1.0 / (w[1] * 10.0) ** 2, rel=1e-3)


def test_sfr_fit_recovers_parameters():
    f = np.geomspace(1e-3, 0.45, 300)
    w = 2 * np.pi * f
    true = dict(tau=12.0, k=0.8, tg=6.0)
    P = 3.3e-5 * sfr_psd_shape(w, **true)
    fit = fit_sfr(f, P)
    assert fit["tau_sys"] == pytest.approx(true["tau"], rel=0.05)
    assert fit["amp"] == pytest.approx(3.3e-5, rel=0.1)
    assert fit["cost"] < 1e-6


# --------------------------------------------------------------------------- #
# ordering machinery (synthetic)
# --------------------------------------------------------------------------- #
def test_octile_block_monotone():
    rng = np.random.default_rng(0)
    H = rng.uniform(100, 400, 4000)
    v = 1.0 / H + 0.0003 * rng.standard_normal(4000)
    b = _octile_block(H, v)
    assert b["argmax_bin"] == 0
    assert b["rho"] < -0.5
    assert len(b["H_med"]) == N_OCT
    assert sum(b["n"]) == 4000


def test_octile_block_interior_peak():
    rng = np.random.default_rng(1)
    H = rng.uniform(100, 400, 6000)
    v = np.exp(-((H - 250.0) / 60.0) ** 2) + 0.05 * rng.standard_normal(6000)
    b = _octile_block(H, v)
    assert 1 <= b["argmax_bin"] <= N_OCT - 2
    assert b["p_interior"] >= 0.9


# --------------------------------------------------------------------------- #
# committed cache schema
# --------------------------------------------------------------------------- #
def test_cache_schema(cache):
    assert set(cache["meta"]["eras"].values()) == {"preDC", "DC"}
    assert len(cache["meta"]["months"]) == 7
    assert cache["meta"]["n_halfhours"]["preDC"] > 3000
    assert cache["meta"]["n_halfhours"]["DC"] > 4000
    for era in ("preDC", "DC"):
        for band in BANDS:
            for st in ("all", "day", "night", "demandnorm"):
                r = cache["ordering"][f"{era}|{band}|{st}"]
                assert len(r["H_med"]) == N_OCT == len(r["var_med"])
                assert r["n_halfhours"] >= 300
        hm = [cache["spectra"][f"{era}_T{j}"]["H_med"] for j in range(3)]
        assert hm[0] < hm[1] < hm[2]


# --------------------------------------------------------------------------- #
# the registered dichotomy on the real archives
# --------------------------------------------------------------------------- #
def test_markovian_null_rejected_everywhere(cache, out):
    assert out["verdicts"]["markovian_null_rejected"]
    for era in ("preDC", "DC"):
        for st in ("all", "day", "night", "demandnorm"):
            r = cache["ordering"][f"{era}|b2_30|{st}"]
            assert r["argmax_bin"] != N_OCT - 1     # never the top-H bin
            assert r["rho"] <= 0.05                 # never the Markovian sign


def test_window_side_ordering_confirmed(cache, out):
    assert out["verdicts"]["window_side_ordering_confirmed"]
    pooled = cache["ordering"]["preDC|b2_30|all"]
    assert pooled["rho"] < 0 and pooled["p"] < 1e-6
    assert cache["ordering"]["preDC|b2_30|day"]["rho"] <= -0.5
    assert cache["ordering"]["preDC|b2_30|demandnorm"]["rho"] <= -0.5


def test_interior_peak_honestly_unresolved(out):
    # the criterion's own crossing requirement: the observed inertia span is
    # far below the band width, so the turning point cannot be resolved
    assert not out["verdicts"]["interior_peak_resolved_in_range"]
    assert out["verdicts"]["inertia_range_under_spans_band"]
    for era in ("preDC", "DC"):
        assert out["range_vs_band"][era]["frac_of_band"] <= 0.5
    # and the c*~0.30 readout puts the peak at/below the observed low edge
    lo = out["range_vs_band"]["preDC"]["H_lo"]
    assert out["predicted_peak_H"]["400"] <= lo * 1.05


def test_band_specificity_out_of_band_inverts(cache, out):
    assert out["verdicts"]["band_specificity"]
    assert cache["ordering"]["preDC|b60_300|all"]["rho"] > 0
    assert cache["ordering"]["DC|b60_300|all"]["rho"] > 0
    assert cache["ordering"]["DC|b60_300|all"]["argmax_bin"] == N_OCT - 1


def test_dc_era_flattening(cache, out):
    assert out["verdicts"]["dc_era_flattening"]
    assert abs(cache["ordering"]["DC|b2_30|all"]["rho"]) < 0.10


def test_ambient_knee_readout_confounded(out):
    assert out["verdicts"]["ambient_knee_not_inertia_readout"]
    taus = out["sfr_fit"]["preDC"]["tau_sys"]
    hs = out["sfr_fit"]["preDC"]["H_med"]
    assert hs[0] < hs[1] < hs[2]                    # H rises across terciles
    assert not (taus[0] < taus[1] < taus[2])        # tau_sys does not
    for era in ("preDC", "DC"):
        for t in out["sfr_fit"][era]["tau_sys"]:
            assert 1.0 <= t <= 60.0                 # physical knee range


def test_committed_figure_artifact(out):
    with open(FIG) as fh:
        fig = json.load(fh)
    assert fig["verdicts"] == out["verdicts"]
    assert fig["range_vs_band"]["band_decades"] == pytest.approx(
        np.log10(15.0), rel=1e-9)
