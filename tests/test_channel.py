"""Unit tests for MU-MISO geometry, path-loss, and Rician channel generation."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SystemConfig
from src.channel import (
    compute_path_gain,
    generate_rician_channel,
    sample_user_positions,
    ula_steering_matrix,
)


def test_user_positions_within_annular_sector() -> None:
    rng = np.random.default_rng(42)
    distances, angles, xy = sample_user_positions(
        num_users=500,
        r_min_m=20.0,
        r_max_m=100.0,
        sector_angle_deg=120.0,
        rng=rng,
    )
    assert distances.shape == (500,)
    assert angles.shape == (500,)
    assert xy.shape == (500, 2)
    assert np.all(distances >= 20.0 - 1e-12)
    assert np.all(distances <= 100.0 + 1e-12)
    half_rad = np.deg2rad(60.0)
    assert np.all(angles >= -half_rad - 1e-12)
    assert np.all(angles <= half_rad + 1e-12)
    np.testing.assert_allclose(np.linalg.norm(xy, axis=1), distances, rtol=1e-12)


def test_path_gain_monotonic_decreasing() -> None:
    d = np.array([10.0, 25.0, 50.0, 100.0])
    beta = compute_path_gain(d, beta_0=1e-4, d0_m=1.0, path_loss_exponent=3.0)
    assert np.all(beta > 0.0)
    assert np.all(np.diff(beta) < 0.0)
    np.testing.assert_allclose(beta[0] / beta[3], (100.0 / 10.0) ** 3.0, rtol=1e-12)


def test_ula_steering_matrix_norm() -> None:
    M = 12
    angles = np.linspace(-np.pi / 3, np.pi / 3, 7)
    A = ula_steering_matrix(M, angles)
    assert A.shape == (M, 7)
    col_norm_sq = np.sum(np.abs(A) ** 2, axis=0)
    np.testing.assert_allclose(col_norm_sq, M, rtol=1e-12)


def test_rician_channel_expected_power_invariant() -> None:
    """Verify E[||h_k||_2^2] ~= M * beta_k across many independent draws."""
    sys_cfg = SystemConfig(r_min_m=49.999, r_max_m=50.001, rician_k_db=6.0)
    rng = np.random.default_rng(12345)
    M, K = 8, 4
    n_samples = 2000

    ratios = []
    for _ in range(n_samples):
        ch = generate_rician_channel(M, K, sys_cfg, rng)
        ch_norm_sq = np.sum(np.abs(ch.H) ** 2, axis=0)
        ratios.append(ch_norm_sq / (M * ch.beta))

    mean_ratio = float(np.mean(ratios))
    assert abs(mean_ratio - 1.0) < 0.02, f"Expected E[||h_k||^2]/(M*beta_k) ~= 1, got {mean_ratio}"


def test_channel_deterministic_with_seed() -> None:
    sys_cfg = SystemConfig()
    ch1 = generate_rician_channel(8, 4, sys_cfg, np.random.default_rng(999))
    ch2 = generate_rician_channel(8, 4, sys_cfg, np.random.default_rng(999))
    np.testing.assert_array_equal(ch1.H, ch2.H)
    np.testing.assert_array_equal(ch1.positions_xy, ch2.positions_xy)


def test_invalid_geometry_raises() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        sample_user_positions(0, 10.0, 50.0, 120.0, rng)
    with pytest.raises(ValueError):
        sample_user_positions(4, 100.0, 20.0, 120.0, rng)
