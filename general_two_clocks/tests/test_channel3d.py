"""Unit tests for the wall-bounded (channel) closure module (Part 9d).

These exercise the *one* property the periodic-box Part-9c proof cannot: the
behaviour of the sharp filter and the Leray pressure projector at a solid wall.
The decisive checks are (i) the wall-aware projector is an exact Leray projection
(machine-zero interior divergence, machine-zero wall-normal velocity, idempotent),
(ii) the naive periodic projector is wall-blind, and (iii) filter and projector
COMMUTE to machine precision in a periodic box but NOT in the channel, where the
commutator is concentrated at the wall.  CPU only (NumPy backend).
"""
from __future__ import annotations

import numpy as np
import pytest

from closure.channel3d import Channel3D, random_channel_field
from run_closure3d_bounded import (
    vnorm, commutator_wall, commutator_periodic,
)

KC_H = 6.0
NY_KEEP = 10


@pytest.fixture(scope="module")
def chan():
    """Small channel + a deterministic divergence-laden field."""
    ch = Channel3D(16, 33, 16, xp=np)
    u, v, w = random_channel_field(ch, seed=1)
    return ch, u, v, w


# --- the wall-aware projector is a genuine, exact Leray projection ----------

def test_project_wall_is_divergence_free_interior(chan):
    ch, u, v, w = chan
    pu, pv, pw = ch.project_wall(u, v, w)
    # consistent discrete Hodge projection -> machine zero in the interior
    assert ch.divergence_rms_interior(pu, pv, pw) < 1e-9


def test_project_wall_enforces_no_penetration(chan):
    ch, u, v, w = chan
    pu, pv, pw = ch.project_wall(u, v, w)
    # wall-normal velocity vanishes ON the walls to machine precision
    assert ch.no_penetration_rms(pu, pv, pw) < 1e-9


def test_project_wall_is_idempotent(chan):
    ch, u, v, w = chan
    pu, pv, pw = ch.project_wall(u, v, w)
    qu, qv, qw = ch.project_wall(pu, pv, pw)
    rel = vnorm(np, qu - pu, qv - pv, qw - pw) / (vnorm(np, pu, pv, pw) + 1e-30)
    assert rel < 1e-9  # P^2 = P : it really is a projection


def test_project_wall_actually_changes_a_divergent_field(chan):
    ch, u, v, w = chan
    # guard against a trivial (identity) projector passing the above by accident
    assert ch.divergence_rms(u, v, w) > 0.1
    assert ch.no_penetration_rms(u, v, w) > 0.1


# --- the naive periodic projector is wall-blind -----------------------------

def test_naive_projector_violates_no_penetration(chan):
    ch, u, v, w = chan
    pu, pv, pw = ch.project_wall(u, v, w)
    fu, fv, fw = ch.project_fourier(u, v, w)
    np_wall = ch.no_penetration_rms(pu, pv, pw)
    np_naive = ch.no_penetration_rms(fu, fv, fw)
    assert np_naive > 0.05                      # O(1) wall violation
    assert np_naive > 1e6 * np_wall             # ... and vastly worse than wall-aware


# --- filter / projector commutation: box vs wall ----------------------------

def test_periodic_commutator_vanishes(chan):
    ch, u, v, w = chan
    c = commutator_periodic(ch, u, v, w, KC_H, float(NY_KEEP))
    rel = vnorm(np, *c) / vnorm(np, u, v, w)
    assert rel < 1e-9   # [F_per, P_per] = 0 : both are Fourier multipliers


def test_wall_breaks_filter_projector_commutation(chan):
    ch, u, v, w = chan
    cper = commutator_periodic(ch, u, v, w, KC_H, float(NY_KEEP))
    cwall = commutator_wall(ch, u, v, w, KC_H, NY_KEEP)
    rper = vnorm(np, *cper) / vnorm(np, u, v, w)
    rwall = vnorm(np, *cwall) / vnorm(np, u, v, w)
    assert rwall > 1e-3                 # the wall breaks commutation
    assert rwall > 1e6 * rper           # ... by many orders of magnitude


def test_commutator_is_wall_localized(chan):
    ch, u, v, w = chan
    cwall = commutator_wall(ch, u, v, w, KC_H, NY_KEEP)
    ce = cwall[0] ** 2 + cwall[1] ** 2 + cwall[2] ** 2
    prof = np.mean(ce, axis=(0, 2))
    ny = ch.ny
    nwall = max(2, ny // 8)
    near = np.mean(np.concatenate([prof[:nwall], prof[-nwall:]]))
    bulk = np.mean(prof[ny // 2 - nwall:ny // 2 + nwall])
    assert near > 3.0 * bulk            # the obstruction lives at the wall
    assert int(np.argmax(prof)) in (0, ny - 1)  # peaks ON a wall


# --- the wall-respecting filter genuinely band-limits -----------------------

def test_sharp_filter_band_limits_horizontally(chan):
    ch, u, v, w = chan
    f = ch.sharp_filter(u, KC_H, NY_KEEP)
    F = ch.fft_h(f)
    mask = (ch.K2h[:, None, :] > KC_H ** 2 + 1e-9)   # (nx,1,nz), broadcasts over y
    assert np.max(np.abs(F) * mask) < 1e-10   # no energy above the horizontal cutoff


# --- determinism / reproducibility ------------------------------------------

def test_reproducible(chan):
    ch, _, _, _ = chan
    a = random_channel_field(ch, seed=1)
    b = random_channel_field(ch, seed=1)
    for ca, cb in zip(a, b):
        assert np.array_equal(ca, cb)
    # and a downstream diagnostic is bit-identical
    c1 = commutator_wall(ch, *a, KC_H, NY_KEEP)
    c2 = commutator_wall(ch, *b, KC_H, NY_KEEP)
    assert np.array_equal(c1[0], c2[0])
