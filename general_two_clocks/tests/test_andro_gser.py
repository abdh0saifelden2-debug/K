"""Unit proofs for NR45 (`general_two_clocks/new_relationships22.py`): ocean
microrheology -- the generalized Stokes-Einstein relation (GSER) on ANDRO deep
displacements (ledger E11).  Offline-safe: all numbers come from the committed
cache ``data/nr45_andro_gser_cache.json`` (derived from the ANDRO deposit,
SEANOE doi:10.17882/47077; regeneration via ``build_cache()`` with
``$ANDRO_DAT``/``$ANDRO_NC`` set).

Covered: the Mason-form GSER closed forms on synthetic power-law MSDs (exact
exponent recovery, the pure-viscous limit, the normalization-free elastic
fraction and the |G*| scaling); cache schema and float counts; the terminal
viscous window with a plausible eddy diffusivity; the VACF decorrelation knee;
the looper/rest subpopulation split carrying the elasticity; the float-mean
(record-mean removal) bias control; the flagged global pooling artefact; and
the committed figure artifact's verdicts.
"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships22 import (  # noqa: E402
    DT_DAYS, FIG, LOOPER_Q, MAXLAG, analyze, gser, load_cache,
)


@pytest.fixture(scope="module")
def cache():
    return load_cache()


@pytest.fixture(scope="module")
def out(cache):
    return analyze(cache)


# --------------------------------------------------------------------------- #
# GSER closed forms (synthetic, no data needed)
# --------------------------------------------------------------------------- #
def test_gser_recovers_power_law_exponent():
    t = np.arange(0, MAXLAG + 1) * DT_DAYS
    for a in (0.5, 2.0 / 3.0, 1.0):
        g = gser(np.where(t > 0, t ** a, 0.0))
        assert np.allclose(g["alpha"], a, atol=1e-10)


def test_gser_pure_viscous_limit():
    t = np.arange(0, MAXLAG + 1) * DT_DAYS
    g = gser(np.where(t > 0, t, 0.0))
    assert np.allclose(g["elastic_frac"], 0.0, atol=1e-8)   # G' = 0
    assert np.allclose(g["delta_rad"], math.pi / 2.0, atol=1e-8)


def test_gser_elastic_fraction_closed_form():
    # G'/G'' = 1/tan(pi*alpha/2) = tan((1-alpha)*pi/2); alpha=2/3 -> 1/tan(pi/3)
    t = np.arange(0, MAXLAG + 1) * DT_DAYS
    g = gser(np.where(t > 0, t ** (2.0 / 3.0), 0.0))
    assert np.allclose(g["elastic_frac"], 1.0 / math.tan(math.pi / 3.0),
                       atol=1e-10)


def test_gser_elastic_fraction_is_normalization_free():
    # no ocean-"kT", no tracer radius: alpha, delta, G'/G'' are scale-invariant
    t = np.arange(0, MAXLAG + 1) * DT_DAYS
    m = np.where(t > 0, 7.7 * t ** 0.8, 0.0)
    a, b = gser(m), gser(1e6 * m)
    assert np.allclose(a["alpha"], b["alpha"])
    assert np.allclose(a["elastic_frac"], b["elastic_frac"])
    assert np.allclose(a["delta_rad"], b["delta_rad"])
    # only |G*| carries the normalization, inversely
    assert np.allclose(np.array(a["Gmag_arb"]) / np.array(b["Gmag_arb"]), 1e6)


# --------------------------------------------------------------------------- #
# committed cache schema
# --------------------------------------------------------------------------- #
def test_cache_schema_and_counts(cache):
    assert cache["meta"]["n_cycles_valid"] == 1383958
    assert cache["meta"]["n_floats"] == 9379
    assert cache["meta"]["dt_days"] == DT_DAYS
    assert cache["meta"]["maxlag"] == MAXLAG
    assert len(cache["na_atlas"]) == 301
    assert len(cache["na_floatmean"]) == 301
    assert cache["global_atlas_ens"]["n_floats"] == 6936
    r = cache["na_atlas"][0]
    for k in ("wmo", "spin", "msd_s", "msd_c", "vac_n", "vac_c"):
        assert k in r
        assert len(r["msd_s"]) == MAXLAG + 1
    # the bias control is computed on the SAME float population
    assert ({int(r["wmo"]) for r in cache["na_atlas"]}
            == {int(r["wmo"]) for r in cache["na_floatmean"]})


# --------------------------------------------------------------------------- #
# verdicts on the real data
# --------------------------------------------------------------------------- #
def test_terminal_viscous_window(out):
    assert out["verdicts"]["terminal_viscous_window"]
    assert abs(out["ens"]["alpha_90_140"] - 1.0) <= 0.05


def test_eddy_diffusivity_plausible(out):
    assert out["verdicts"]["K_plausible"]
    assert 1500.0 <= out["K_m2_s_100d"] <= 3000.0        # measured ~2.2e3 m2/s


def test_vacf_decorrelation_knee(out):
    v = np.array(out["ens"]["vacf"])
    assert v[0] == pytest.approx(1.0)
    assert np.all(np.diff(v[:4]) < 0)                    # initial monotone decay
    assert 30.0 <= out["vacf_zero_cross_days"] <= 60.0


def test_looper_split_carries_elasticity(out):
    assert out["verdicts"]["elasticity_subpopulation_carried"]
    assert out["n_loopers"] == 76 and out["n_rest"] == 225
    assert out["loop"]["alpha_100_190"] <= 0.95          # measured 0.91
    assert out["rest"]["alpha_90_140"] >= 0.97           # measured 1.03
    lo, hi = out["looper_elastic_frac_100_190"]
    assert 0.0 <= lo <= hi
    assert hi >= 0.10                                    # measured up to 0.22
    assert out["verdicts"]["looper_elastic_detected"]


def test_looper_threshold_is_top_quartile(cache, out):
    spins = np.array([abs(r["spin"]) for r in cache["na_atlas"]])
    assert out["looper_threshold"] == pytest.approx(
        float(np.quantile(spins, LOOPER_Q)))
    assert out["n_loopers"] == int(np.sum(spins >= out["looper_threshold"]))
    assert out["n_loopers"] + out["n_rest"] == out["n_floats"]


def test_floatmean_bias_control(out):
    # record-mean removal fakes subdiffusion at record-length lags; the
    # independent atlas mean is the unbiased control
    assert out["verdicts"]["floatmean_bias_quantified"]
    assert (out["ens_floatmean"]["alpha_150_190"]
            < out["ens"]["alpha_150_190"] - 0.04)        # 0.85 vs 0.93


def test_global_pooling_artefact_flagged_not_claimed(out):
    # pooled global ensemble is superdiffusive from cross-float K
    # heterogeneity -- documented as an artefact, never as physics
    assert 1.05 <= out["global"]["alpha_90_140"] <= 1.5  # measured 1.18
    assert out["global"]["n_floats"] == 6936


def test_committed_figure_artifact():
    with open(FIG) as fh:
        fig = json.load(fh)
    assert fig["n_floats"] == 301
    v = fig["verdicts"]
    assert v["terminal_viscous_window"]
    assert v["elasticity_subpopulation_carried"]
    assert v["floatmean_bias_quantified"]
    assert v["looper_elastic_detected"]
    assert v["K_plausible"]
    assert "alpha_150_190" in fig["curves"]["ens"]
