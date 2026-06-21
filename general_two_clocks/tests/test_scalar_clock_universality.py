"""Regression test for RESULT 13 (scalar_clock_universality.py): the turbulent
flux memory time tau_c is a property of the *turbulence*, not of the scalar.
 
A short CPU double-diffusion run (heat vs salt, Le=100, same velocity field) must
return matching flux-memory times for the two scalars despite the 100x molecular
diffusivity contrast, and strongly co-varying flux series.
"""
import os
import sys
 
import numpy as np
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import scalar_clock_universality as scu
 
 
def test_heat_and_salt_share_the_same_flux_memory_time():
    """tau_c(salt)/tau_c(heat) ~ 1 even though kappa_S = kappa_T/100."""
    r = scu.measure_scalar_clocks(nx=96, ny=48, spinup=600, measure=1500,
                                  sample_every=2, seed=1)
    # both memory times are positive and finite
    assert r["tau_c_heat_efold"] > 0 and np.isfinite(r["tau_c_heat_efold"])
    assert r["tau_c_salt_efold"] > 0 and np.isfinite(r["tau_c_salt_efold"])
    # and equal to within 10% despite the 100x molecular-diffusivity contrast
    assert abs(r["salt_over_heat_efold"] - 1.0) < 0.10
    # the two flux series are driven by the same velocity field -> co-vary
    assert r["flux_xcorr"] > 0.8
 
 
def test_molecular_contrast_is_real_but_does_not_set_the_clock():
    """Sanity: the Lewis contrast genuinely separates the *transport efficiency*
    (Nu_S vs Nu_T differ by a large factor) while leaving the *memory time*
    unchanged -- i.e. the equality above is not because the two scalars are
    trivially identical in every respect."""
    r = scu.measure_scalar_clocks(nx=96, ny=48, spinup=600, measure=1500,
                                  sample_every=2, seed=1)
    # Nu_S and Nu_T differ substantially (the kappa normalisation makes the
    # haline Nusselt ~Le times the thermal one); guard only on a clear gap.
    assert abs(r["Nu_S"]) > 3.0 * abs(r["Nu_T"])
    # yet the memory ratio stayed ~1
    assert abs(r["salt_over_heat_efold"] - 1.0) < 0.10
 
 
def test_flux_memory_consistent_with_Ku_memory():
    """The shared scalar-flux clock is the same order as the eddy-diffusivity
    K_u clock measured in the same run (both are the velocity's clock)."""
    r = scu.measure_scalar_clocks(nx=96, ny=48, spinup=600, measure=1500,
                                  sample_every=2, seed=1)
    ratio = r["tau_c_heat_efold"] / (r["tau_c_Ku_efold"] + 1e-30)
    assert 0.3 < ratio < 3.0


def test_le_sweep_separates_clock_from_amplitude():
    """RESULT 13b: across the molecular-diffusivity contrast the *clock* must
    stay pinned while the *amplitude* moves.  The falsifier would be a
    tau_c(salt)/tau_c(heat) that trends with Le.  Here:

    * salt/heat tau_c stays ~1 at *every* Le (clock = shared velocity field), and
    * Nu_S/Nu_T grows with Le (the contrast is real, in the kappa normalisation).
    """
    out = scu.sweep_le(les=(1.0, 100.0), seeds=(1,),
                       nx=96, ny=48, spinup=300, measure=800, sample_every=2)
    by_le = {s["Le"]: s for s in out["summary"]}

    # clock: the memory ratio is ~1 regardless of Le (within 10%)
    for s in out["summary"]:
        assert abs(s["salt_over_heat_mean"] - 1.0) < 0.10
        assert s["flux_xcorr_mean"] > 0.8

    # control: Le=1 is heat==salt, so Nu_S/Nu_T ~ 1
    assert abs(by_le[1.0]["NuS_over_NuT_mean"] - 1.0) < 0.20
    # amplitude: the 100x contrast shows up as a large Nu_S/Nu_T separation that
    # the clock ratio does NOT mirror (clock decoupled from amplitude).
    assert by_le[100.0]["NuS_over_NuT_mean"] > 5.0 * by_le[1.0]["NuS_over_NuT_mean"]
