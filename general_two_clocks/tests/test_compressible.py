"""Unit tests for the linear-acoustics solver and Poisson reference."""

import numpy as np

from compressible.solver import (
    LinearAcoustics2D,
    gaussian_bump,
    gradient_forcing,
    poisson_fft,
)


def test_poisson_fft_recovers_known_field():
    # If p = sin(2*pi*x) sin(2*pi*y), then laplacian(p) = -2*(2*pi)^2 p.
    n, L = 64, 1.0
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    xx, yy = np.meshgrid(x, x, indexing="ij")
    p_true = np.sin(2 * np.pi * xx) * np.sin(2 * np.pi * yy)
    rhs = -2.0 * (2 * np.pi) ** 2 * p_true
    p = poisson_fft(rhs, dx)
    p -= p.mean()
    assert np.linalg.norm(p - p_true) / np.linalg.norm(p_true) < 1e-2


def test_acoustic_steady_state_matches_elliptic():
    # Under a fixed gradient forcing with damping, the acoustic pressure must
    # relax to the elliptic Poisson solution (the c -> inf limit).
    n, L = 64, 1.0
    dx = L / n
    psi = gaussian_bump(n, L, L / 2, L / 2, sigma=0.12, amp=1.0)
    fx, fy = gradient_forcing(psi, dx)
    div_f = (np.roll(fx, -1, 0) - np.roll(fx, 1, 0)) / (2 * dx) \
        + (np.roll(fy, -1, 1) - np.roll(fy, 1, 1)) / (2 * dx)
    p_ell = poisson_fft(div_f, dx)
    p_ell -= p_ell.mean()

    c = 16.0
    sim = LinearAcoustics2D(n, L, c, gamma=1.5 * c / L, cfl=0.4)
    sim.run(t_end=8.0 * L / c, fx=fx, fy=fy)
    p = sim.state.p - sim.state.p.mean()
    assert np.linalg.norm(p - p_ell) / np.linalg.norm(p_ell) < 0.05


def test_higher_c_relaxes_faster():
    # Time to reach the elliptic field should shrink as c grows (~ L/c).
    n, L = 48, 1.0
    dx = L / n
    psi = gaussian_bump(n, L, L / 2, L / 2, sigma=0.12, amp=1.0)
    fx, fy = gradient_forcing(psi, dx)
    div_f = (np.roll(fx, -1, 0) - np.roll(fx, 1, 0)) / (2 * dx) \
        + (np.roll(fy, -1, 1) - np.roll(fy, 1, 1)) / (2 * dx)
    p_ell = poisson_fft(div_f, dx)
    p_ell -= p_ell.mean()
    norm = np.linalg.norm(p_ell)

    def t90(c):
        sim = LinearAcoustics2D(n, L, c, gamma=1.5 * c / L, cfl=0.4)
        t_end = 10.0 * L / c
        while sim.state.t < t_end:
            err = np.linalg.norm((sim.state.p - sim.state.p.mean()) - p_ell) / norm
            if err < 0.10:
                return sim.state.t
            sim.step(fx=fx, fy=fy)
        return t_end

    assert t90(16.0) < t90(4.0)
