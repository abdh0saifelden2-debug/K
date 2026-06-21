r"""Unit tests for the Mach -> 0 regularity bridge (mach_regularity_bridge.py).

Fast: two analytic helper tests (no DNS) plus one small compressible-DNS sweep
(n=48, 3 Mach numbers, short integration) that verifies the M -> 0 convergence of
the pressure Hessian to the incompressible (elliptic) limit.
"""
from __future__ import annotations

import numpy as np
import pytest

import mach_regularity_bridge as mrb
from compressible.ns import Spectral2D


def test_spectral_hessian_matches_analytic():
    """spectral_hessian recovers the exact Hessian of a known periodic field."""
    n = 32
    sp = Spectral2D(n)
    x, y = sp.grid()
    kx, ky = 3, 2
    p = np.cos(kx * x) * np.cos(ky * y)
    pxx, pxy, pyy = mrb.spectral_hessian(sp, p)
    exact_xx = -(kx ** 2) * np.cos(kx * x) * np.cos(ky * y)
    exact_yy = -(ky ** 2) * np.cos(kx * x) * np.cos(ky * y)
    exact_xy = (kx * ky) * np.sin(kx * x) * np.sin(ky * y)
    assert np.allclose(pxx, exact_xx, atol=1e-10)
    assert np.allclose(pyy, exact_yy, atol=1e-10)
    assert np.allclose(pxy, exact_xy, atol=1e-10)


def test_anisotropy_fraction_limits():
    """Pure-isotropic Hessian -> fraction ~0; pure-deviatoric Hessian -> fraction ~1."""
    n = 16
    rng = np.random.default_rng(0)
    field = rng.standard_normal((n, n))
    # isotropic: pxx = pyy = f, pxy = 0  ->  no anisotropy
    frac_iso, _ = mrb.anisotropy_fraction(field, np.zeros((n, n)), field)
    assert frac_iso < 1e-10
    # deviatoric/traceless: pxx = -pyy = f, pxy = 0  ->  fully anisotropic
    frac_dev, _ = mrb.anisotropy_fraction(field, np.zeros((n, n)), -field)
    assert abs(frac_dev - 1.0) < 1e-10
    # any field: fraction is in [0, 1]
    frac_any, _ = mrb.anisotropy_fraction(field, 0.5 * field, -0.3 * field)
    assert 0.0 <= frac_any <= 1.0


@pytest.mark.parametrize("machs", [(0.3, 0.1, 0.05)])
def test_mach_to_zero_installs_nonlocal_hessian(machs):
    """As M -> 0 the compressible (local EOS) pressure Hessian converges to the
    incompressible (nonlocal elliptic) one: field, Hessian and Poisson-constraint
    residuals all decrease, anisotropy converges to the elliptic value, and the
    Hessian L2 residual equals the Poisson-constraint residual (Parseval)."""
    s = mrb.mach_sweep(n=48, machs=machs, t_end_factor=0.6, cfl=0.3)
    assert len(s["rows"]) == len(machs)

    lo, hi = s["rows"][-1], s["rows"][0]            # lowest / highest Mach
    # residuals shrink toward the incompressible limit
    assert lo["field_resid"] < hi["field_resid"]
    assert lo["hess_resid"] < hi["hess_resid"]
    # anisotropy converges to the incompressible (elliptic) value
    assert lo["aniso_gap"] < hi["aniso_gap"]
    # the incompressible reference satisfies the Poisson constraint (pipeline check)
    assert max(r["inc_constraint_check"] for r in s["rows"]) < 1e-6
    # Parseval: ||P_comp - P_inc||_2 == Poisson-constraint residual, exactly
    for r in s["rows"]:
        assert abs(r["hess_resid"] - r["constraint_resid"]) <= 1e-6 * (
            r["constraint_resid"] + 1e-30)
    # clean low-Mach convergence: residual ~ M^p with p > 1 (expect ~2)
    assert s["p_hess"] > 1.0
    assert s["p_field"] > 1.0
    # the elliptic-limit Hessian is substantially anisotropic (the part restricted
    # Euler discards -> blowup); 2D analog of the 3D 82% in REPORT_REGULARITY
    assert s["aniso_inc"] > 0.3
    assert s["ok"]
