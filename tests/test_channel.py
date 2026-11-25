"""Unit tests for polar-circle UE placement, log-distance path loss, and direct-sum Rician channel."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SystemConfig
from src.channel import (
    compute_path_loss,
    generate_rician_channel,
    sample_user_positions_polar,
    ula_steering_matrix,
)
from src.utils import db_to_linear


def test_polar_positions_within_circle() -> None:
    rng = np.random.default_rng(42)
    R, d0 = 100.0, 1.0
    radii, dists, phis, xy = sample_user_positions_polar(
        num_users=1000,
        cell_radius_m=R,
        d0_m=d0,
        rng=rng,
    )
    assert radii.shape == (1000,)
    assert dists.shape == (1000,)
    assert phis.shape == (1000,)
    assert xy.shape == (1000, 2)
    assert np.all((radii >= 0.0) & (radii <= R))
    assert np.all(dists >= d0)
    assert np.all((phis >= 0.0) & (phis < 2.0 * np.pi))
    np.testing.assert_allclose(xy[:, 0], radii * np.cos(phis), rtol=1e-12)
    np.testing.assert_allclose(xy[:, 1], radii * np.sin(phis), rtol=1e-12)


def test_log_distance_path_loss_formula() -> None:
    """Verify PL_k = PL_0 + 10*n*log10(d/d_0) with PL_0=30 dB, d_0=1 m, n=3.0."""
    d = np.array([1.0, 10.0, 100.0])
    pl_db, pl_lin = compute_path_loss(d, pl0_db=30.0, d0_m=1.0, path_loss_exponent=3.0)
    np.testing.assert_allclose(pl_db, [30.0, 60.0, 90.0], rtol=1e-12)
    np.testing.assert_allclose(pl_lin, [1e-3, 1e-6, 1e-9], rtol=1e-12)


def test_ula_steering_matrix_norm() -> None:
    Nt = 12
    angles = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    A = ula_steering_matrix(Nt, angles)
    assert A.shape == (Nt, 9)
    col_norm_sq = np.sum(np.abs(A) ** 2, axis=0)
    np.testing.assert_allclose(col_norm_sq, Nt, rtol=1e-12)


def test_direct_sum_rician_structure_and_power_invariant() -> None:
    """Verify h_k = sqrt(PL_k) * (h_{k,NLoS} + h_{k,LoS}), NLoS ~ N(0,1/2)+jN(0,1/2),
    and E[||h_k||_2^2] ~= PL_k_lin * N_t * (1 + kappa)."""
    sys_cfg = SystemConfig(cell_radius_m=100.0, pl0_db=30.0, d0_m=1.0, path_loss_exponent=3.0, rician_k_db=0.0)
    rng = np.random.default_rng(12345)
    Nt, K = 8, 4
    kappa = float(db_to_linear(sys_cfg.rician_k_db))

    ratios = []
    nlos_re_vars = []
    nlos_im_vars = []
    for _ in range(1500):
        ch = generate_rician_channel(Nt, K, sys_cfg, rng)
        assert ch.H.shape == (Nt, K)
        # Exact direct-sum algebraic check
        np.testing.assert_allclose(ch.H_hat, ch.H_nlos + ch.H_los, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(ch.H, np.sqrt(ch.pl_linear)[None, :] * ch.H_hat, rtol=1e-14, atol=1e-14)

        ch_norm_sq = np.sum(np.abs(ch.H) ** 2, axis=0)
        ratios.append(ch_norm_sq / (ch.pl_linear * Nt * (1.0 + kappa)))
        nlos_re_vars.append(np.mean(ch.H_nlos.real**2))
        nlos_im_vars.append(np.mean(ch.H_nlos.imag**2))

    assert abs(float(np.mean(nlos_re_vars)) - 0.5) < 0.02
    assert abs(float(np.mean(nlos_im_vars)) - 0.5) < 0.02
    mean_ratio = float(np.mean(ratios))
    assert abs(mean_ratio - 1.0) < 0.02, f"Expected E[||h_k||^2]/(PL*Nt*(1+kappa)) ~= 1, got {mean_ratio}"


def test_channel_deterministic_with_seed() -> None:
    sys_cfg = SystemConfig()
    ch1 = generate_rician_channel(8, 4, sys_cfg, np.random.default_rng(999))
    ch2 = generate_rician_channel(8, 4, sys_cfg, np.random.default_rng(999))
    np.testing.assert_array_equal(ch1.H, ch2.H)
    np.testing.assert_array_equal(ch1.positions_xy, ch2.positions_xy)


def test_invalid_geometry_raises() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        sample_user_positions_polar(0, 100.0, 1.0, rng)
    with pytest.raises(ValueError):
        sample_user_positions_polar(4, 1.0, 5.0, rng)
