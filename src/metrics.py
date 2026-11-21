"""Vectorized power normalization, SINR, sum-rate, and ULA beampattern metrics.

Supports both single beamforming matrices W of shape (M, K) and batched
populations W_batch of shape (pop_size, M, K) using NumPy broadcasting / einsum.
"""

from __future__ import annotations

import numpy as np

from src.channel import ula_steering_matrix


EPS: float = 1e-15


def normalize_power(W: np.ndarray, p_max_watt: float) -> np.ndarray:
    """Project beamforming matrix (or population batch) onto ||W||_F^2 = P_max.

    For sum-rate maximization under a total BS power budget, optimal solutions
    always saturate the boundary ||W||_F^2 = P_max.

    Args:
        W: Beamforming matrix of shape (M, K) or batch (pop_size, M, K), complex128.
        p_max_watt: Maximum total transmit power P_max in Watts (> 0).

    Returns:
        W_norm: Same shape as W, with Frobenius norm squared equal to p_max_watt.
    """
    if p_max_watt <= 0.0:
        raise ValueError("p_max_watt must be strictly positive.")

    W_arr = np.asarray(W, dtype=np.complex128)
    target_norm = np.sqrt(p_max_watt)

    if W_arr.ndim == 2:
        frob = float(np.linalg.norm(W_arr, ord="fro"))
        if frob < EPS:
            # Deterministic non-zero fallback avoiding division by zero
            fallback = np.ones_like(W_arr, dtype=np.complex128)
            return fallback * (target_norm / np.linalg.norm(fallback, ord="fro"))
        return W_arr * (target_norm / frob)

    if W_arr.ndim == 3:
        # Norms per candidate over axes (1, 2): shape (pop_size, 1, 1)
        frob = np.sqrt(np.sum(W_arr.real**2 + W_arr.imag**2, axis=(1, 2), keepdims=True))
        zero_mask = frob < EPS
        if np.any(zero_mask):
            W_arr = W_arr.copy()
            W_arr[zero_mask[:, 0, 0]] = 1.0 + 0.0j
            frob = np.sqrt(np.sum(W_arr.real**2 + W_arr.imag**2, axis=(1, 2), keepdims=True))
        return W_arr * (target_norm / frob)

    raise ValueError(f"Expected 2D or 3D array for W, got ndim={W_arr.ndim}.")


def compute_sinr(H: np.ndarray, W: np.ndarray, noise_power_watt: float) -> np.ndarray:
    """Compute per-user SINR for a single precoder (M, K) or batch (pop_size, M, K).

    Effective channel matrix is G = H^H W, where G[k, j] = h_k^H w_j.
    SINR_k = |G[k, k]|^2 / (sum_{j != k} |G[k, j]|^2 + sigma^2).

    Args:
        H: Channel matrix of shape (M, K), complex128.
        W: Beamforming matrix (M, K) or batch (pop_size, M, K), complex128.
        noise_power_watt: Receiver AWGN variance sigma^2 in Watts (> 0).

    Returns:
        sinr: Linear SINR array of shape (K,) or (pop_size, K).
    """
    if noise_power_watt <= 0.0:
        raise ValueError("noise_power_watt must be strictly positive.")

    H_conj = np.conj(np.asarray(H, dtype=np.complex128))
    W_arr = np.asarray(W, dtype=np.complex128)

    if W_arr.ndim == 2:
        G = H_conj.T @ W_arr                         # (K, K), G[k, j] = h_k^H w_j
        power_mat = G.real**2 + G.imag**2            # |G|^2, shape (K, K)
        signal = np.diag(power_mat)                  # (K,)
        total_rx = np.sum(power_mat, axis=1)         # (K,)
        interference = np.maximum(total_rx - signal, 0.0)
        return signal / (interference + noise_power_watt)

    if W_arr.ndim == 3:
        # G[p, k, j] = sum_m conj(H[m, k]) * W[p, m, j]
        G = np.einsum("mk,pmj->pkj", H_conj, W_arr, optimize=True)
        power_mat = G.real**2 + G.imag**2            # (pop_size, K, K)
        signal = np.diagonal(power_mat, axis1=1, axis2=2)  # (pop_size, K)
        total_rx = np.sum(power_mat, axis=2)         # (pop_size, K)
        interference = np.maximum(total_rx - signal, 0.0)
        return signal / (interference + noise_power_watt)

    raise ValueError(f"Expected 2D or 3D array for W, got ndim={W_arr.ndim}.")


def compute_sum_rate(H: np.ndarray, W: np.ndarray, noise_power_watt: float) -> float | np.ndarray:
    """Compute Shannon sum-rate sum_{k=1}^K log2(1 + SINR_k) in bps/Hz.

    Args:
        H: Channel matrix of shape (M, K), complex128.
        W: Beamforming matrix (M, K) or batch (pop_size, M, K), complex128.
        noise_power_watt: Receiver AWGN variance sigma^2 in Watts (> 0).

    Returns:
        Sum-rate in bps/Hz: float for 2D W, or float64 array (pop_size,) for 3D W.
    """
    sinr = compute_sinr(H, W, noise_power_watt)
    rates = np.log2(1.0 + sinr)
    if sinr.ndim == 1:
        return float(np.sum(rates))
    return np.sum(rates, axis=1)


def compute_beampattern(
    W: np.ndarray,
    angles_rad: np.ndarray,
) -> np.ndarray:
    """Compute angular radiation power pattern |a(theta)^H w_k|^2 for each beam k.

    Args:
        W: Beamforming matrix of shape (M, K), complex128.
        angles_rad: Azimuth scan grid of shape (N_angles,), radians.

    Returns:
        pattern_watt: shape (K, N_angles), directional array power per user beam.
    """
    M = W.shape[0]
    A = ula_steering_matrix(M, angles_rad)           # (M, N_angles)
    resp = np.conj(A).T @ W                          # (N_angles, K)
    return (resp.real**2 + resp.imag**2).T           # (K, N_angles)
