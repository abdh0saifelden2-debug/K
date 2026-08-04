"""Keystone positive control for RESULT 14 (scallop_ktheory_control.py):

Implication #1 says the migration term ``Im(s) = omega_mig != 0`` is a
parity-symmetry break that **no K-theory (down-gradient eddy-diffusivity)
closure can produce**.  A short CPU run must show that, on the *same* frozen
corrugated interface and the *same* mean drive:

  * the resolved advective flux produces a non-zero quadrature gain ``E_cos``
    (migration); but
  * a uniform down-gradient eddy-diffusivity closure (advection of theta
    removed) produces ``E_cos`` at *machine precision* -- exactly zero, because
    a symmetric diffusion operator on a sinusoid carries no flow direction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import scallop_ktheory_control as kt
from subglacial.candidate3_roughness_feedback import Candidate3Config


def test_unknown_eddy_closure_raises():
    """A mistyped eddy closure (e.g. ``"smogorinsky"``) must fail loudly rather
    than silently falling through to the Smagorinsky path."""
    cfg = Candidate3Config(nx=32, ny=32, A=4.0, sgs="none", f_amp=0.0, Ri=0.0,
                           seed=0)
    flow = kt.KTheoryProbeFlow(cfg, U_drive=0.0, eddy="smogorinsky")
    with pytest.raises(ValueError):
        flow._eddy_diffusivity(flow.u, flow.v)

    # the supported closures still resolve without error
    for eddy in ("uniform", "smagorinsky", "mixing_length",
                 "mixing_length_capped"):
        ok = kt.KTheoryProbeFlow(cfg, U_drive=0.0, eddy=eddy)
        assert np.all(np.isfinite(ok._eddy_diffusivity(ok.u, ok.v)))


def test_ktheory_closure_produces_no_migration():
    r = kt.ktheory_control(nw=12, U=3.0, seeds=(0,), nx=48, ny=48, spinup=300,
                           measure=100, afrac=0.20, eddies=("uniform",))
    m = r["models"]

    # the resolved advective flux DOES migrate: a clearly non-zero quadrature gain
    assert abs(m["resolved"]["Ecos_mean"]) > 1e-7

    # the down-gradient K-theory closure does NOT: E_cos is at machine precision
    # (parity-symmetric -> identically zero quadrature), and utterly negligible
    # relative to the resolved migration.
    assert abs(m["uniform"]["Ecos_mean"]) < 1e-12
    assert m["uniform"]["Ecos_ratio_to_resolved"] < 1e-6


def test_closure_battery_artifact():
    """The committed battery (figures/59b_ktheory_closure_battery.json): the
    direction-blind member is machine zero; flow-aware members produce at most
    a fraction of the resolved quadrature and with the WRONG (upstream) sign
    (P3-R2)."""
    import json, os
    art = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures", "59b_ktheory_closure_battery.json")
    with open(art) as fh:
        m = json.load(fh)["models"]
    assert m["resolved"]["Ecos_mean"] < -1e-5              # downstream
    assert abs(m["uniform"]["Ecos_mean"]) < 1e-12          # exact parity
    for k in ("smagorinsky", "mixing_length", "mixing_length_capped"):
        assert m[k]["Ecos_mean"] > 0.0                     # wrong sign
        assert m[k]["Ecos_ratio_to_resolved"] < 0.30       # and smaller
    # the l-cap does not matter (wall region dominates)
    a, b = m["mixing_length"]["Ecos_mean"], m["mixing_length_capped"]["Ecos_mean"]
    assert abs(a - b) / abs(a) < 0.01
