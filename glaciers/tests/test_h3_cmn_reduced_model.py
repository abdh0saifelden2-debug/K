"""Unit tests for §H.3 CMN 2-D reduced-model proxy
(validation/synthetic/h3_cmn_reduced_model.py)."""
import os
import sys

import numpy as np

_VALIDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "validation")
for _p in (os.path.join(_VALIDATION, "synthetic"), _VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h3_cmn_reduced_model as H3  # noqa: E402

RES = H3.run()   # computed once (~9s)


def test_K_field_plume_and_steady():
    # outside the event window K is the uniform background
    K_pre = H3.K_field(0.0)
    assert np.allclose(K_pre, H3.K0)
    assert np.allclose(H3.dKdt_fd(0.0), 0.0)
    # mid-event the plume raises K and ∂_tK is non-zero somewhere
    t_mid = 0.5 * (H3.EVENT[0] + H3.EVENT[1])
    assert H3.K_field(t_mid).max() > H3.K0 + 1e-6
    assert np.abs(H3.dKdt_fd(t_mid)).max() > 0


def test_operators_constant_field():
    c = np.full((H3.NY, H3.NX), 3.0)
    # diffusion and advection of a constant field vanish (periodic)
    assert np.allclose(H3._diffusion(c, np.full_like(c, H3.K0)), 0.0, atol=1e-10)
    assert np.allclose(H3._advection(c, H3.U_ADV), 0.0, atol=1e-10)


def test_correction_cuts_transient_error():
    h = RES["headline"]
    # corrected error is much smaller than naive
    assert h["max_err_corrected"] < h["max_err_naive"]
    assert h["cut_factor"] > 3.0


def test_order_of_accuracy():
    s = RES["scaling"]
    # naive is first-order in τ_c, corrected is second-order
    assert 0.8 < s["slope_naive"] < 1.2
    assert 1.8 < s["slope_corrected"] < 2.2


def test_steady_null_is_exact():
    n = RES["steady_null"]
    # no event -> all clocks coincide -> machine-zero error
    assert n["max_err_naive"] < 1e-12
    assert n["max_err_corrected"] < 1e-12


def test_plus_tau_c_is_unique_corrector():
    st = RES["sign_test"]
    # +τ_c reduces below naive; −τ_c is worse than naive
    assert st["err_plus_tau_c"] < st["err_naive"] < st["err_minus_tau_c"]
    assert st["plus_is_unique_corrector"] is True


def test_scope_is_honest_proxy():
    # the module must not claim to be the real ISSM/GlaDS test
    assert "proxy" in RES["scope_note"].lower()
    assert "not" in RES["what"].lower() or "NOT" in RES["what"]
