"""MU-MISO downlink geometry, large-scale path loss, and Rician fading channel model.

Conventions:
- Base Station (BS) sits at the origin (0, 0) with an M-element Uniform Linear
  Array (ULA) of half-wavelength spacing (d_a = lambda / 2).
- Users (UEs) are uniformly distributed by area in an annular sector
  r in [R_min, R_max], theta in [-sector/2, +sector/2].
- Channel matrix H has shape (M, K) with dtype complex128, where column k is
  h_k (M, 1) and downlink received signal is y = H^H W s + n.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config import SystemConfig
from src.utils import db_to_linear


@dataclass(frozen=True)
class ChannelRealization:
    """Container for one independent MU-MISO channel realization."""

    H: np.ndarray              # shape (M, K), complex128: composite Rician channels
    H_los: np.ndarray          # shape (M, K), complex128: unit-magnitude LoS steering matrix
    H_nlos: np.ndarray         # shape (M, K), complex128: CN(0, 1) NLoS scattering matrix
    beta: np.ndarray           # shape (K,), float64: linear large-scale path gain beta_k
    distances_m: np.ndarray    # shape (K,), float64: BS-UE distances d_k (m)
    angles_rad: np.ndarray     # shape (K,), float64: UE azimuth angles theta_k (rad)
    positions_xy: np.ndarray   # shape (K, 2), float64: UE Cartesian coordinates (x_k, y_k) in m


def sample_user_positions(
    num_users: int,
    r_min_m: float,
    r_max_m: float,
    sector_angle_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample UE positions uniformly in area inside an annular sector.

    Args:
        num_users: Number of single-antenna users K.
        r_min_m: Minimum distance from BS (m).
        r_max_m: Maximum distance from BS (m).
        sector_angle_deg: Sector angular span in degrees (e.g., 120 or 360).
        rng: Deterministic NumPy random generator.

    Returns:
        distances_m: shape (K,), distances d_k in [r_min_m, r_max_m].
        angles_rad: shape (K,), azimuth angles theta_k in [-half_span, +half_span].
        positions_xy: shape (K, 2), Cartesian coordinates (x, y) in meters.
    """
    if num_users <= 0:
        raise ValueError("num_users must be positive.")
    if not (0.0 < r_min_m < r_max_m):
        raise ValueError("Require 0 < r_min_m < r_max_m.")

    u = rng.uniform(0.0, 1.0, size=num_users)
    distances_m = np.sqrt(r_min_m**2 + u * (r_max_m**2 - r_min_m**2))

    half_span_rad = np.deg2rad(sector_angle_deg) / 2.0
    angles_rad = rng.uniform(-half_span_rad, half_span_rad, size=num_users)

    x = distances_m * np.cos(angles_rad)
    y = distances_m * np.sin(angles_rad)
    positions_xy = np.column_stack((x, y))
    return distances_m, angles_rad, positions_xy


def compute_path_gain(
    distances_m: np.ndarray,
    beta_0: float,
    d0_m: float,
    path_loss_exponent: float,
) -> np.ndarray:
    """Compute linear large-scale path gain beta_k = beta_0 * (d_k / d_0)^(-alpha).

    Args:
        distances_m: shape (K,), BS-UE distances in meters.
        beta_0: Reference linear path gain at distance d0_m.
        d0_m: Reference distance d_0 in meters.
        path_loss_exponent: Propagation exponent alpha.

    Returns:
        beta: shape (K,), positive linear path gains.
    """
    d = np.asarray(distances_m, dtype=np.float64)
    if np.any(d <= 0.0):
        raise ValueError("All distances must be strictly positive.")
    return beta_0 * (d / d0_m) ** (-path_loss_exponent)


def ula_steering_matrix(num_antennas: int, angles_rad: np.ndarray) -> np.ndarray:
    """Compute half-wavelength ULA array response matrix A(theta).

    Each column k is:
        a(theta_k) = [1, exp(j*pi*sin(theta_k)), ..., exp(j*(M-1)*pi*sin(theta_k))]^T

    Args:
        num_antennas: Number of BS antennas M.
        angles_rad: shape (K,), azimuth angles in radians.

    Returns:
        A: shape (M, K), complex128 steering matrix with ||a_k||_2^2 = M.
    """
    m_idx = np.arange(num_antennas, dtype=np.float64)[:, None]     # (M, 1)
    sin_theta = np.sin(np.asarray(angles_rad, dtype=np.float64))[None, :]  # (1, K)
    return np.exp(1j * np.pi * m_idx * sin_theta)


def generate_rician_channel(
    num_antennas: int,
    num_users: int,
    sys_cfg: SystemConfig,
    rng: np.random.Generator,
    rician_k_db: float | None = None,
) -> ChannelRealization:
    """Generate one independent MU-MISO downlink Rician fading channel realization.

    Model:
        h_k = sqrt(beta_k) * (sqrt(kappa/(kappa+1)) * h_k^LoS
                              + sqrt(1/(kappa+1))   * h_k^NLoS)
    where E[||h_k||_2^2] = M * beta_k.

    Args:
        num_antennas: Number of BS antennas M.
        num_users: Number of single-antenna users K.
        sys_cfg: Physical system configuration.
        rng: Deterministic NumPy random generator.
        rician_k_db: Optional override for Rician K-factor in dB.

    Returns:
        ChannelRealization containing H (M, K), geometry, path gains, and components.
    """
    distances_m, angles_rad, positions_xy = sample_user_positions(
        num_users=num_users,
        r_min_m=sys_cfg.r_min_m,
        r_max_m=sys_cfg.r_max_m,
        sector_angle_deg=sys_cfg.sector_angle_deg,
        rng=rng,
    )
    beta = compute_path_gain(
        distances_m=distances_m,
        beta_0=sys_cfg.ref_path_gain_linear,
        d0_m=sys_cfg.d0_m,
        path_loss_exponent=sys_cfg.path_loss_exponent,
    )

    k_db = sys_cfg.rician_k_db if rician_k_db is None else float(rician_k_db)
    kappa = float(db_to_linear(k_db))

    h_los = ula_steering_matrix(num_antennas, angles_rad)
    h_nlos = (
        rng.standard_normal((num_antennas, num_users))
        + 1j * rng.standard_normal((num_antennas, num_users))
    ) / np.sqrt(2.0)

    los_weight = np.sqrt(kappa / (kappa + 1.0))
    nlos_weight = np.sqrt(1.0 / (kappa + 1.0))

    H = (los_weight * h_los + nlos_weight * h_nlos) * np.sqrt(beta)[None, :]

    return ChannelRealization(
        H=H,
        H_los=h_los,
        H_nlos=h_nlos,
        beta=beta,
        distances_m=distances_m,
        angles_rad=angles_rad,
        positions_xy=positions_xy,
    )
