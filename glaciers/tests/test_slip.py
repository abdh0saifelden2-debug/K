"""Regression tests for the bed Navier-slip BC in ``subglacial.flow3d``.

The slip knob ``bed_slip`` must:
  * reduce *exactly* to the no-slip baseline at s=1 (and on the ice/fluid, which
    carry no rock penalty) -- this is the property that keeps every prior no-slip
    result valid as the s=1 limit;
  * actually energize the near-bed tangential velocity as s -> 0 (free slip),
    otherwise the gate would be testing nothing;
  * never let fluid penetrate the bed (the bed-normal stays impermeable).
"""
import numpy as np
import pytest

from subglacial.flow3d import Subglacial3DConfig, Subglacial3DFlow


def _ridge_bed(n, bed_mean=0.9, a=0.30, kz=4.0):
    z = np.arange(n) * (2.0 * np.pi / n)
    _, Z = np.meshgrid(np.arange(n) * (2.0 * np.pi / n), z, indexing="ij")
    return bed_mean + a * np.cos(kz * Z)


def _build(n, bed_slip, steps=60, seed=0):
    cfg = Subglacial3DConfig(
        n=n, sgs="smagorinsky", bed_field=_ridge_bed(n), nu=5e-5, kappa=5e-5,
        eta=5e-5, U0=1.0, f_amp=2.0, dt=4e-4, bed_mean=0.9, ice_base=2.4,
        seed=seed, bed_slip=bed_slip)
    flow = Subglacial3DFlow(cfg, xp=np)
    flow.run(steps)
    return flow


def test_bed_slip_one_is_bit_for_bit_noslip():
    """s=1 must reproduce the no-slip (bed_slip=None) trajectory exactly."""
    base = _build(32, None)
    s1 = _build(32, 1.0)
    assert np.abs(s1.u - base.u).max() == 0.0
    assert np.abs(s1.v - base.v).max() == 0.0
    assert np.abs(s1.w - base.w).max() == 0.0
    assert np.abs(s1.theta - base.theta).max() == 0.0


def test_free_slip_energizes_bed_tangential_velocity():
    """s->0 must raise the near-bed tangential speed well above no-slip."""
    base = _build(32, None)
    s0 = _build(32, 0.0)

    def u_tang(f, ref):
        band = (ref.chi_rock > 0.3) & (ref.chi_rock < 0.7)
        nx, ny, nz = ref._bn_x, ref._bn_y, ref._bn_z
        un = f.u * nx + f.v * ny + f.w * nz
        ut = np.sqrt((f.u - un * nx) ** 2 + (f.v - un * ny) ** 2
                     + (f.w - un * nz) ** 2)
        return float(ut[band].mean())

    # bed_slip=None has no precomputed normal; reuse the s=0 geometry (identical
    # bed) to define the band/normal for both fields.
    assert u_tang(s0, s0) > 2.0 * u_tang(base, s0)


def test_bed_normal_stays_impermeable_under_free_slip():
    """Even at free slip the bed-normal velocity in the bed stays tiny: no
    fluid is injected through the bed (only tangential sliding is allowed)."""
    s0 = _build(32, 0.0)
    nx, ny, nz = s0._bn_x, s0._bn_y, s0._bn_z
    un = s0.u * nx + s0.v * ny + s0.w * nz
    ut = np.sqrt((s0.u - un * nx) ** 2 + (s0.v - un * ny) ** 2
                 + (s0.w - un * nz) ** 2)
    deep = (s0.chi_rock > 0.5)
    band = (s0.chi_rock > 0.3) & (s0.chi_rock < 0.7)
    # bed-normal speed (penetration) stays small vs the tangential slip speed
    assert float(np.abs(un[deep]).mean()) < 0.5 * float(ut[band].mean())


def test_bed_slip_rejects_out_of_range():
    with pytest.raises(ValueError):
        _build(16, 1.5, steps=1)
    with pytest.raises(ValueError):
        _build(16, -0.1, steps=1)
