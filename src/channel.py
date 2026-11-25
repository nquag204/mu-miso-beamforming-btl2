"""MU-MISO downlink polar-coordinate geometry, log-distance path loss, and direct-sum Rician channel.

Model specification:
- Base Station (BS) with N_t ULA antennas (d_a = lambda/2) at circle center O(0, 0)
  of radius R.
- User k placed in polar coordinates (r_k, Phi_k):
      r_k = R * random(),    Phi_k = 2*pi * random()
  with Cartesian position (x_k, y_k) = (r_k cos(Phi_k), r_k sin(Phi_k)) and
  effective propagation distance d_k = max(r_k, d_0).
- Log-distance path loss (dB & linear power gain):
      PL_k_dB  = PL_0 + 10 * n * log10(d_k / d_0)
      PL_k_lin = 10 ** (-PL_k_dB / 10)
- Direct-sum small-scale Rician fading vector h_hat_k (N_t x 1):
      h_{k,NLoS} ~ N(0, 1/2) + j * N(0, 1/2)    (i.i.d. CN(0, 1))
      h_{k,LoS}  = sqrt(kappa) * [1, exp(j*pi*sin(Phi_k)), ..., exp(j*(N_t-1)*pi*sin(Phi_k))]^T
      h_hat_k    = h_{k,NLoS} + h_{k,LoS}
- Composite channel vector for user k (N_t x 1):
      h_k = sqrt(PL_k_lin) * h_hat_k
  stacked into channel matrix H = [h_1, ..., h_K] of shape (N_t, K) complex128.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config import SystemConfig
from src.utils import db_to_linear


@dataclass(frozen=True)
class ChannelRealization:
    """Container for one independent MU-MISO downlink channel realization."""

    H: np.ndarray              # shape (N_t, K), complex128: composite Rician channel matrix
    H_hat: np.ndarray          # shape (N_t, K), complex128: small-scale h_hat = h_NLoS + h_LoS
    H_los: np.ndarray          # shape (N_t, K), complex128: LoS component h_{k,LoS}
    H_nlos: np.ndarray         # shape (N_t, K), complex128: NLoS component N(0,1/2) + jN(0,1/2)
    pl_db: np.ndarray          # shape (K,), float64: path loss PL_k in dB
    pl_linear: np.ndarray      # shape (K,), float64: linear large-scale power gain 10^(-PL_k/10)
    radii_m: np.ndarray        # shape (K,), float64: raw polar radius r_k = R * random() (m)
    distances_m: np.ndarray    # shape (K,), float64: effective distance d_k = max(r_k, d_0) (m)
    angles_rad: np.ndarray     # shape (K,), float64: polar angle Phi_k = 2*pi * random() (rad)
    positions_xy: np.ndarray   # shape (K, 2), float64: Cartesian coordinates (x_k, y_k) (m)


def sample_user_positions_polar(
    num_users: int,
    cell_radius_m: float,
    d0_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample UE positions in polar coordinates inside circle O(0, 0) of radius R.

    Follows:
        r_k   = R * random()        in [0, R)
        Phi_k = 2 * pi * random()   in [0, 2*pi)
    Effective distance d_k = max(r_k, d_0) prevents log10(0) singularity.

    Returns:
        radii_m: shape (K,), raw sampled polar radii r_k (m).
        distances_m: shape (K,), effective propagation distances d_k >= d_0 (m).
        angles_rad: shape (K,), polar angles Phi_k in [0, 2*pi) (rad).
        positions_xy: shape (K, 2), Cartesian coordinates (x_k, y_k) (m).
    """
    if num_users <= 0:
        raise ValueError("num_users must be positive.")
    if cell_radius_m <= 0.0 or d0_m <= 0.0 or d0_m >= cell_radius_m:
        raise ValueError("Require 0 < d0_m < cell_radius_m.")

    radii_m = cell_radius_m * rng.random(num_users)
    distances_m = np.maximum(radii_m, d0_m)
    angles_rad = 2.0 * np.pi * rng.random(num_users)

    x = radii_m * np.cos(angles_rad)
    y = radii_m * np.sin(angles_rad)
    positions_xy = np.column_stack((x, y))
    return radii_m, distances_m, angles_rad, positions_xy


def compute_path_loss(
    distances_m: np.ndarray,
    pl0_db: float,
    d0_m: float,
    path_loss_exponent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log-distance path loss PL_k [dB] and linear power gain PL_k_lin.

    PL_k [dB] = PL_0 + 10 * n * log10(d_k / d_0)
    PL_k_lin  = 10 ** (-PL_k_dB / 10)
    """
    d = np.asarray(distances_m, dtype=np.float64)
    if np.any(d <= 0.0):
        raise ValueError("All distances must be strictly positive.")
    pl_db = pl0_db + 10.0 * path_loss_exponent * np.log10(d / d0_m)
    pl_linear = 10.0 ** (-pl_db / 10.0)
    return pl_db, pl_linear


def ula_steering_matrix(num_antennas: int, angles_rad: np.ndarray) -> np.ndarray:
    """Compute half-wavelength ULA array steering matrix A(Phi) of shape (N_t, K).

    Column k:
        a(Phi_k) = [1, exp(j*pi*sin(Phi_k)), ..., exp(j*(N_t-1)*pi*sin(Phi_k))]^T
    with ||a(Phi_k)||_2^2 = N_t.
    """
    m_idx = np.arange(num_antennas, dtype=np.float64)[:, None]                # (N_t, 1)
    sin_phi = np.sin(np.asarray(angles_rad, dtype=np.float64))[None, :]       # (1, K)
    return np.exp(1j * np.pi * m_idx * sin_phi)


def generate_rician_channel(
    num_antennas: int,
    num_users: int,
    sys_cfg: SystemConfig,
    rng: np.random.Generator,
    rician_k_db: float | None = None,
) -> ChannelRealization:
    """Generate one independent MU-MISO direct-sum Rician channel realization.

    Direct-sum model:
        h_{k,NLoS} ~ N(0, 1/2) + j * N(0, 1/2)
        h_{k,LoS}  = sqrt(kappa) * a(Phi_k)
        h_hat_k    = h_{k,NLoS} + h_{k,LoS}
        h_k        = sqrt(PL_k_lin) * h_hat_k   (N_t x 1)

    Statistical invariant:
        E[||h_k||_2^2] = PL_k_lin * N_t * (1 + kappa)
    where kappa = 10^(K_Rician_dB / 10) (default K_Rician = 0 dB => kappa = 1).
    """
    radii_m, distances_m, angles_rad, positions_xy = sample_user_positions_polar(
        num_users=num_users,
        cell_radius_m=sys_cfg.cell_radius_m,
        d0_m=sys_cfg.d0_m,
        rng=rng,
    )
    pl_db, pl_linear = compute_path_loss(
        distances_m=distances_m,
        pl0_db=sys_cfg.pl0_db,
        d0_m=sys_cfg.d0_m,
        path_loss_exponent=sys_cfg.path_loss_exponent,
    )

    k_db = sys_cfg.rician_k_db if rician_k_db is None else float(rician_k_db)
    kappa = float(db_to_linear(k_db))

    # NLoS component: N(0, 1/2) + j * N(0, 1/2)
    h_nlos = (
        rng.normal(0.0, np.sqrt(0.5), size=(num_antennas, num_users))
        + 1j * rng.normal(0.0, np.sqrt(0.5), size=(num_antennas, num_users))
    )
    # LoS component: sqrt(kappa) * a(Phi_k) (unit amplitude when kappa = 1)
    h_los = np.sqrt(kappa) * ula_steering_matrix(num_antennas, angles_rad)

    # Direct sum: h_hat_k = h_{k,NLoS} + h_{k,LoS}
    h_hat = h_nlos + h_los

    # Composite channel: [h_{1,k}, ..., h_{Nt,k}]^T = sqrt(PL_k_lin) * h_hat_k
    H = np.sqrt(pl_linear)[None, :] * h_hat

    return ChannelRealization(
        H=H,
        H_hat=h_hat,
        H_los=h_los,
        H_nlos=h_nlos,
        pl_db=pl_db,
        pl_linear=pl_linear,
        radii_m=radii_m,
        distances_m=distances_m,
        angles_rad=angles_rad,
        positions_xy=positions_xy,
    )
