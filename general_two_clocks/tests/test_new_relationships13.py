"""Unit proofs for NR36 (`general_two_clocks/new_relationships13.py`):
the well-barometric response as a two-clocks kernel -- element identities,
the cross-spectral CONVENTION GUARD, synthetic end-to-end recovery, tide-line
exclusion, and (data-gated) the real-well identification regression.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from new_relationships13 import (  # noqa: E402
    DATA_DIR, FS_CPD, MODELS, TIDE_LINES_CPD, analysis_mask, brf_welch,
    elem_drainage, elem_vadose, elem_wellbore, fit_models, model_W,
    parse_nwis_rdb, regularise)


# ------------------------------------------------------------ conventions --
def test_convention_guard_pure_delay_lags():
    """d(t) = b(t - tau) MUST estimate a NEGATIVE phase ~ -w tau.  This is
    the guard against the scipy.signal.csd argument-order trap (csd(x,y)
    returns <conj(X) Y>): with the arguments flipped the estimated phase
    would come out POSITIVE and silently select the wrong kernel family."""
    rng = np.random.default_rng(0)
    n = 20000
    b = np.cumsum(rng.normal(size=n))          # red noise input
    lag = 3                                    # samples: 6 h at 2-h cadence
    d = np.roll(b, lag)
    d[:lag] = d[lag]
    f, W, coh = brf_welch(d, b, seg_days=60.0)
    m = (f > 0.1) & (f < 1.0)
    ph = np.angle(W[m])
    expected = -2 * np.pi * f[m] * (lag / FS_CPD)
    assert np.all(ph[f[m] < 0.7] < 0)          # lags, never leads
    assert np.allclose(ph, expected, atol=0.15)


def test_element_phase_identities():
    """drainage LEADS (arg = pi/2 - arctan(w tau) > 0); wellbore LAGS by
    exactly -arctan(w tau) (the NR11 form); vadose lags with the diffusive
    sqrt: arg = -sqrt(w tau/2), |V| = exp(-sqrt(w tau/2))."""
    w = np.array([0.1, 1.0, 10.0])
    tau = 0.7
    assert np.allclose(np.angle(elem_wellbore(w, tau)),
                       -np.arctan(w * tau), rtol=1e-12)
    assert np.allclose(np.angle(elem_drainage(w, tau)),
                       np.pi / 2 - np.arctan(w * tau), rtol=1e-12)
    assert np.all(np.angle(elem_drainage(w, tau)) > 0)
    assert np.allclose(np.angle(elem_vadose(w, tau)),
                       -np.sqrt(w * tau / 2), rtol=1e-12)
    assert np.allclose(np.abs(elem_vadose(w, tau)),
                       np.exp(-np.sqrt(w * tau / 2)), rtol=1e-12)
    # drainage screens DC, passes high f (the NR30 operator analogue)
    assert abs(elem_drainage(1e-9, tau)) < 1e-8
    assert abs(elem_drainage(1e9, tau) - 1.0) < 1e-8


def test_analysis_mask_excludes_tide_lines():
    f = np.linspace(0.01, 5, 5000)
    m = analysis_mask(f)
    for c in TIDE_LINES_CPD:
        assert not np.any(m & (np.abs(f - c) < 0.03))
    assert not np.any(m & (f < 0.05)) and not np.any(m & (f > 4.0))
    assert m.sum() > 1000


# ------------------------------------------------------- synthetic pipeline --
def _synthetic_pair(be=0.5, tau_l=0.6, n=13140, seed=1, tide_amp=0.02):
    """Baro red-noise + S2 line -> through BE*drainage -> well depth series
    with EARTH-TIDE lines (in d only) + white noise.  2-h cadence."""
    rng = np.random.default_rng(seed)
    t_days = np.arange(n) / FS_CPD
    # red-noise baro (m H2O) + atmospheric S2
    white = rng.normal(size=n)
    b = np.convolve(white, np.exp(-np.arange(200) / 30.0), mode="same")
    b = 0.05 * b / b.std() + 0.005 * np.cos(2 * np.pi * 2.0 * t_days)
    # exact filtering in the frequency domain (circular; fine for testing)
    B = np.fft.rfft(b - b.mean())
    fcpd = np.fft.rfftfreq(n, d=1.0 / FS_CPD)
    Wtrue = model_W("M7_drainage", 2 * np.pi * fcpd, [be, tau_l])
    d = np.fft.irfft(Wtrue * B, n)
    # earth tides in the WELL only (M2, O1) + instrument noise
    d = d + tide_amp * np.cos(2 * np.pi * 1.9324 * t_days + 0.3) \
          + tide_amp * np.cos(2 * np.pi * 0.9295 * t_days + 1.1) \
          + 0.002 * rng.normal(size=n)
    return d, b


def test_end_to_end_recovery_selects_drainage():
    """Full pipeline on synthetic truth BE*drainage: the winner is in the
    lead family with (BE, tau_l) within 10%, and the lag families are
    rejected decisively."""
    be, tau_l = 0.5, 0.6
    d, b = _synthetic_pair(be, tau_l)
    f, W, coh = brf_welch(d, b)
    m = analysis_mask(f)
    fits, best = fit_models(f, W, coh, m)
    assert best in ("M7_drainage", "M2_drainage_wellbore")
    p = fits["M7_drainage"]["params"]
    assert abs(p[0] - be) / be < 0.10
    assert abs(p[1] - tau_l) / tau_l < 0.10
    assert fits["M0_static"]["delta_aicc"] > 30
    assert fits["M1_wellbore"]["delta_aicc"] > 30
    assert fits["M4_vadose"]["delta_aicc"] > 30


def test_tide_line_exclusion_is_load_bearing():
    """Lines present only in the WELL are already killed by coherence
    weighting (their b-d coherence ~ 0).  The load-bearing case is S2: the
    barometer carries an atmospheric S2 while the well carries an EARTH-tide
    S2 at unrelated phase -- a coherent contamination that weighting cannot
    remove.  The mask must protect the fit there."""
    be, tau_l = 0.5, 0.6
    d, b = _synthetic_pair(be, tau_l, tide_amp=0.02, seed=3)
    n = d.size
    t_days = np.arange(n) / FS_CPD
    # strong atmospheric S2 in b (passes through the true BRF into d via the
    # synthetic construction only weakly) + an INDEPENDENT-phase S2 earth
    # tide directly in the well
    b = b + 0.02 * np.cos(2 * np.pi * 2.0 * t_days + 0.2)
    d = d + 0.03 * np.cos(2 * np.pi * 2.0 * t_days + 2.4)
    f, W, coh = brf_welch(d, b)
    m_ok = analysis_mask(f)
    m_raw = (f >= 0.05) & (f <= 4.0)
    fits_ok, _ = fit_models(f, W, coh, m_ok)
    fits_raw, _ = fit_models(f, W, coh, m_raw)
    err_ok = abs(fits_ok["M7_drainage"]["params"][1] - tau_l)
    err_raw = abs(fits_raw["M7_drainage"]["params"][1] - tau_l)
    assert err_raw > 2.0 * err_ok


def test_aicc_parsimony_on_nested_truth():
    """On M7 truth, the nested M2 (extra wellbore param) must not beat M7
    by more than the AICc parsimony margin."""
    d, b = _synthetic_pair(0.5, 0.6, seed=5)
    f, W, coh = brf_welch(d, b)
    fits, best = fit_models(f, W, coh, analysis_mask(f))
    assert fits["M2_drainage_wellbore"]["aicc"] >= fits["M7_drainage"]["aicc"] - 0.5


def test_regularise_and_coverage():
    t0 = np.datetime64("2021-01-01T00:00")
    t = t0 + np.arange(0, 200, 2).astype("timedelta64[h]")
    t = np.delete(t, slice(30, 45))            # 30-h hole
    d = np.linspace(0, 1, t.size)
    b = np.linspace(1, 2, t.size)
    grid, di, bi, coverage = regularise(t, d, b)
    assert grid[1] - grid[0] == 2.0
    assert 0.8 < coverage < 1.0                # the hole is accounted


def test_parser_handles_concatenated_chunks(tmp_path):
    """Two rdb chunks with DIFFERENT column orders in one file (the way the
    chunked download concatenates) parse into one coherent series."""
    chunk1 = ("# c1\nagency_cd\tsite_no\tdatetime\ttz_cd\t100_00025\t"
              "100_00025_cd\t200_72019\t200_72019_cd\n5s\t15s\t20d\t6s\t14n\t"
              "10s\t14n\t10s\nUSGS\tX\t2021-01-01 00:00\tPST\t660\tA\t7.0\tA\n"
              "USGS\tX\t2021-01-01 02:00\tPST\t661\tA\t7.1\tA\n")
    chunk2 = ("# c2\nagency_cd\tsite_no\tdatetime\ttz_cd\t200_72019\t"
              "200_72019_cd\t100_00025\t100_00025_cd\n5s\t15s\t20d\t6s\t14n\t"
              "10s\t14n\t10s\nUSGS\tX\t2021-01-01 04:00\tPST\t7.2\tA\t662\tA\n")
    fn = tmp_path / "iv_X.rdb"
    fn.write_text(chunk1 + chunk2)
    t, d, b = parse_nwis_rdb(str(fn))
    assert t.size == 3
    assert np.allclose(d / 0.3048, [7.0, 7.1, 7.2])
    assert np.allclose(b / 0.0135951, [660, 661, 662])


# ------------------------------------------------------------ real data ----
_HAVE = all(os.path.exists(os.path.join(DATA_DIR, f"iv_{s}.rdb"))
            for s in ("415546121205401", "415104121232901"))


@pytest.mark.skipif(not _HAVE, reason="local NWIS downloads absent")
def test_real_well_identifies_drainage_family():
    """The responsive Modoc well: winner in the drainage/two-path (lead)
    family; static and both LAG families (RC, vadose) rejected by
    dAICc > 100; parameters in the physical range."""
    t, d, b = parse_nwis_rdb(os.path.join(DATA_DIR, "iv_415546121205401.rdb"))
    _, di, bi, cov = regularise(t, d, b)
    assert cov > 0.95
    f, W, coh = brf_welch(di, bi)
    fits, best = fit_models(f, W, coh, analysis_mask(f))
    assert best in ("M7_drainage", "M2_drainage_wellbore", "M5_twopath",
                    "M6_twopath_wellbore")
    assert fits["M0_static"]["delta_aicc"] > 100
    assert fits["M1_wellbore"]["delta_aicc"] > 100
    assert fits["M4_vadose"]["delta_aicc"] > 100
    be, tl = fits["M7_drainage"]["params"]
    assert 0.3 < be < 0.8 and 0.2 < tl < 2.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
