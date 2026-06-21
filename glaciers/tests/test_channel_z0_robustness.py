"""Tests for the §D.1 closure-robustness sweep
(:mod:`scallop_channel_z0_robustness`).

CPU-only, DNS-free: a synthetic phase-locked scallop source is fed through the
deterministic channel ODE across a small closure box, checking the directional
verdict (scallop locks the winner, beats noise, stays pinned to the source
phase) is invariant to the ``g``/``k_creep``/strength closure -- the property
the full 405-point sweep establishes on the committed DNS field.
"""

import numpy as np

from scallop_channel_z0_robustness import (
    _circ_dist,
    _phase_resultant,
    _strength_threshold,
    run_point,
    write_report,
)


def _report_kw(**over):
    """Minimal valid kwargs for :func:`write_report`."""
    kw = dict(
        g_grid=np.array([0.1, 0.5, 0.9]), kc_grid=np.array([0.5, 1.0, 2.0]),
        scale_grid=np.array([0.001, 0.1, 1.0, 2.0]), v_scallop=0.33,
        phase_pref=4.55, m=96, seeds=24, ode_steps=4000,
        R_sc_meas=(1.0, 1.0, 1.0), R_no_meas=(0.12, 0.48, 0.67),
        margin_meas=(0.33, 0.52), dphi_meas=0.166, invariant_closure=True,
        worst_margin_meas=(0.33, "g=0.40,kc=0.50"),
        worst_phase_meas=(0.166, "g=0.10,kc=0.25"),
        scale_lock=[(0.001, 1.0), (0.1, 1.0), (1.0, 1.0), (2.0, 1.0)],
        source_json="figures/49_scallop_channel_feedback.json")
    kw.update(over)
    return kw


def _synthetic_source(m=48, n_waves=6, strength=0.3, center=np.pi, width=0.5):
    seg_phase = (2.0 * np.pi * n_waves * np.arange(m) / m) % (2.0 * np.pi)
    d = (seg_phase - center + np.pi) % (2.0 * np.pi) - np.pi
    V_sc = strength * np.exp(-(d / width) ** 2)
    return seg_phase, V_sc


def test_write_report_no_break_emits_magnitude_independent(tmp_path):
    """--report must actually write a file (the regression) reporting the
    magnitude-independent verdict when no strength break is found."""
    path = tmp_path / "rep.md"
    write_report(str(path), **_report_kw())
    txt = path.read_text()
    assert "magnitude-independent" in txt.lower()
    assert "405" not in txt  # 3x3x4 == 36 points for this synthetic grid
    assert "36-point" in txt
    assert "`R_winner(scallop)`" in txt and "1.000" in txt


def test_write_report_with_break_reports_threshold(tmp_path):
    """When a strength break exists the report states the survival threshold
    rather than the magnitude-independent wording."""
    path = tmp_path / "rep.md"
    write_report(str(path), **_report_kw(
        scale_lock=[(0.001, 0.0), (0.1, 1.0), (1.0, 1.0), (2.0, 1.0)]))
    txt = path.read_text()
    assert "survives down to" in txt
    assert "0.1" in txt
    assert "magnitude-independent" not in txt.lower()


def test_strength_threshold_monotone_and_all_locked():
    """All-locked -> no break and the smallest scale is a genuine floor;
    a clean monotone break -> threshold above the largest failing scale."""
    thr, last = _strength_threshold(
        [(0.001, 1.0), (0.1, 1.0), (1.0, 1.0), (2.0, 1.0)])
    assert last is None and thr == 0.001
    thr, last = _strength_threshold(
        [(0.001, 0.0), (0.1, 1.0), (1.0, 1.0), (2.0, 1.0)])
    assert last == 0.001 and thr == 0.1


def test_strength_threshold_non_monotone_has_no_floor():
    """A break sitting ABOVE a locked scale must not be dropped, and yields no
    single survival threshold (regression for the silent-ignore bug)."""
    thr, last = _strength_threshold(
        [(0.001, 1.0), (0.1, 0.0), (1.0, 1.0), (2.0, 1.0)])
    assert last == 0.1        # the broken scale above the locked 0.001 is seen
    assert thr is None        # no monotone floor -> no "survives down to" claim


def test_strength_threshold_all_broken_has_no_threshold():
    """All scales broken -> a real largest-failing scale and no threshold (the
    threshold must NOT default to the smallest scale)."""
    thr, last = _strength_threshold(
        [(0.001, 0.0), (0.1, 0.0), (1.0, 0.0), (2.0, 0.0)])
    assert last == 2.0 and thr is None


def test_write_report_non_monotone_does_not_claim_invariance(tmp_path):
    """A non-monotone break (broken scale above a locked one) must not be
    reported as 'magnitude-independent' nor as 'survives down to ...'."""
    path = tmp_path / "rep.md"
    write_report(str(path), **_report_kw(
        scale_lock=[(0.001, 1.0), (0.1, 0.0), (1.0, 1.0), (2.0, 1.0)]))
    txt = path.read_text().lower()
    assert "is magnitude-independent" not in txt  # affirmative claim absent
    assert "survives down to" not in txt
    assert "non-monotone" in txt


def test_write_report_all_broken_reports_failure(tmp_path):
    """When nothing locks anywhere, the report must say the verdict does not
    hold -- not 'survives down to below the smallest scale tested'."""
    path = tmp_path / "rep.md"
    write_report(str(path), **_report_kw(
        scale_lock=[(0.001, 0.0), (0.1, 0.0), (1.0, 0.0), (2.0, 0.0)]))
    txt = path.read_text().lower()
    assert "is magnitude-independent" not in txt
    assert "survives down to" not in txt
    assert "does **not**" in txt or "no strength locks" in txt


def test_circ_dist_wraps_around_2pi():
    assert _circ_dist(0.1, 2.0 * np.pi - 0.1) < 0.25
    assert np.isclose(_circ_dist(0.0, np.pi), np.pi)


def test_phase_resultant_unanimous_is_one():
    seg_phase = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    R, ph = _phase_resultant(seg_phase, [7, 7, 7, 7])
    assert np.isclose(R, 1.0)
    assert np.isclose(ph, seg_phase[7])
    # a diffuse set of winners gives a small resultant
    R_diff, _ = _phase_resultant(seg_phase, list(range(24)))
    assert R_diff < 0.2


def test_site_lock_invariant_across_closure_box():
    """Scallop source locks the winner (R=1), beats noise, and pins to the
    source phase for every point in a small g x k_creep x strength box."""
    m = 48
    center = np.pi
    seg_phase, V_sc_base = _synthetic_source(m=m, center=center)
    V_o = np.full(m, 1.0)
    locked_everywhere = True
    for scale in (0.5, 1.0, 2.0):
        for kc in (0.5, 1.0, 2.0):
            for g in (0.2, 0.5, 0.8):
                R_sc, ph_sc, R_no, ph_no = run_point(
                    V_o, V_sc_base * scale, seg_phase,
                    k_creep=kc, conc_gain=g, ode_dt=0.05,
                    ode_steps=2000, n_noise_seeds=12)
                # deterministic lock, clear of the noise control
                if not (R_sc > 0.8 and (R_sc - R_no) > 0.2):
                    locked_everywhere = False
                # winner pinned near the source maximum (phase == center)
                assert _circ_dist(ph_sc, center) < 0.3
    assert locked_everywhere
