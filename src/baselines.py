"""Reference downlink MU-MISO linear and iterative beamforming baselines.

Implements:
- MRT (Maximum Ratio Transmission)
- ZF (Zero-Forcing, with optional water-filling power allocation)
- RZF / MMSE (Regularized Zero-Forcing, Peel et al., IEEE TCOM 2005)
- WMMSE (Iteratively Weighted MMSE, Shi et al., IEEE TSP 2011)

All beamformers take H of shape (M, K) complex128 and return W of shape (M, K)
complex128 satisfying ||W||_F^2 <= P_max.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from src.metrics import EPS, compute_sum_rate, normalize_power


@dataclass(frozen=True)
class BaselineResult:
    """Output container for a baseline beamforming algorithm."""

    W: np.ndarray            # shape (M, K), complex128
    sum_rate: float          # achieved sum-rate in bps/Hz
    runtime_sec: float       # execution wall-clock time in seconds
    history: np.ndarray      # shape (n_steps,), sum-rate history (len 1 for closed-form)
    method: str


def _waterfilling_power(gains: np.ndarray, p_max_watt: float) -> np.ndarray:
    """Solve exact water-filling: max sum log2(1 + p_k * g_k) s.t. sum p_k = P_max, p_k >= 0."""
    g = np.asarray(gains, dtype=np.float64)
    k_users = g.size
    order = np.argsort(g)[::-1]
    g_sorted = g[order]

    powers_sorted = np.zeros(k_users, dtype=np.float64)
    for active in range(k_users, 0, -1):
        g_act = g_sorted[:active]
        if np.any(g_act <= EPS):
            continue
        mu = (p_max_watt + np.sum(1.0 / g_act)) / active
        p_try = mu - 1.0 / g_act
        if np.all(p_try >= -1e-14):
            powers_sorted[:active] = np.maximum(p_try, 0.0)
            break

    total = float(np.sum(powers_sorted))
    if total > EPS:
        powers_sorted *= p_max_watt / total
    else:
        powers_sorted[:] = p_max_watt / k_users

    powers = np.zeros(k_users, dtype=np.float64)
    powers[order] = powers_sorted
    return powers


def mrt_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
) -> BaselineResult:
    """Maximum Ratio Transmission (MRT) with equal per-user power allocation.

    w_k = sqrt(P_max / K) * h_k / ||h_k||_2
    """
    t0 = time.perf_counter()
    H_arr = np.asarray(H, dtype=np.complex128)
    _, K = H_arr.shape

    col_norms = np.linalg.norm(H_arr, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, EPS)
    W = (H_arr / col_norms) * np.sqrt(p_max_watt / K)
    W = normalize_power(W, p_max_watt)

    rate = float(compute_sum_rate(H_arr, W, noise_power_watt))
    elapsed = time.perf_counter() - t0
    return BaselineResult(
        W=W,
        sum_rate=rate,
        runtime_sec=elapsed,
        history=np.array([rate], dtype=np.float64),
        method="MRT",
    )


def zf_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    waterfilling: bool = False,
) -> BaselineResult:
    """Zero-Forcing (ZF) precoder via Moore-Penrose pseudoinverse of H^H.

    When M >= K and H has full column rank, H^H W is strictly diagonal so
    multi-user interference is completely nulled. When M < K, gracefully
    falls back to the minimum-norm pseudoinverse normalized to P_max.
    """
    t0 = time.perf_counter()
    H_arr = np.asarray(H, dtype=np.complex128)
    M, K = H_arr.shape

    # Right pseudoinverse of H^H (K, M) has shape (M, K): H^H @ V = I_K when M >= K
    V = np.linalg.pinv(H_arr.conj().T)
    col_norms = np.linalg.norm(V, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, EPS)
    V_hat = V / col_norms

    if waterfilling and M >= K:
        eff_diag = np.abs(np.sum(H_arr.conj() * V_hat, axis=0)) ** 2
        gains = eff_diag / noise_power_watt
        powers = _waterfilling_power(gains, p_max_watt)
        W = V_hat * np.sqrt(powers)[None, :]
    else:
        W = V_hat * np.sqrt(p_max_watt / K)

    W = normalize_power(W, p_max_watt)
    rate = float(compute_sum_rate(H_arr, W, noise_power_watt))
    elapsed = time.perf_counter() - t0
    return BaselineResult(
        W=W,
        sum_rate=rate,
        runtime_sec=elapsed,
        history=np.array([rate], dtype=np.float64),
        method="ZF-WF" if waterfilling else "ZF",
    )


def rzf_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
) -> BaselineResult:
    """Regularized Zero-Forcing (RZF / MMSE) precoder (Peel et al., 2005).

    V = H (H^H H + (K * sigma^2 / P_max) * I_K)^{-1} with unit-norm columns and
    equal per-user power allocation P_max / K (consistent with MRT and ZF under
    heterogeneous path losses). Smoothly interpolates between MRT (low SNR) and
    ZF (high SNR).
    """
    t0 = time.perf_counter()
    H_arr = np.asarray(H, dtype=np.complex128)
    _, K = H_arr.shape

    # Normalize H by sqrt(sigma^2) for numerical conditioning across wide path-loss ranges
    H_tilde = H_arr / np.sqrt(noise_power_watt)
    gram = H_tilde.conj().T @ H_tilde
    reg = K / p_max_watt
    reg_mat = gram + reg * np.eye(K, dtype=np.complex128)
    # Solve (gram + reg*I) X = H_tilde^H => V = X^H = H_tilde (gram + reg*I)^{-1}
    V = np.linalg.solve(reg_mat, H_tilde.conj().T).conj().T
    col_norms = np.linalg.norm(V, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, EPS)
    W = (V / col_norms) * np.sqrt(p_max_watt / K)
    W = normalize_power(W, p_max_watt)

    rate = float(compute_sum_rate(H_arr, W, noise_power_watt))
    elapsed = time.perf_counter() - t0
    return BaselineResult(
        W=W,
        sum_rate=rate,
        runtime_sec=elapsed,
        history=np.array([rate], dtype=np.float64),
        method="RZF",
    )


def wmmse_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> BaselineResult:
    """Iteratively Weighted MMSE (WMMSE) precoder (Shi et al., IEEE TSP 2011).

    Solves sum-rate maximization to a KKT stationary point by alternating
    closed-form updates for receive scalars u_k, MSE weights rho_k, and
    Lagrange-constrained transmit beamformers W(mu) = (A + mu*I)^{-1} B.
    Initialized at the best of {MRT, ZF, RZF} so monotonic ascent strictly
    dominates all linear baselines even under strong near-far path-loss spread.
    """
    t0 = time.perf_counter()
    H_arr = np.asarray(H, dtype=np.complex128)
    M, K = H_arr.shape

    # Scale channel by 1/sqrt(sigma^2) so effective noise variance is 1.0
    H_tilde = H_arr / np.sqrt(noise_power_watt)

    # Warm-start at the strongest feasible linear baseline
    init_candidates = (
        mrt_beamformer(H_arr, p_max_watt, noise_power_watt),
        zf_beamformer(H_arr, p_max_watt, noise_power_watt),
        rzf_beamformer(H_arr, p_max_watt, noise_power_watt),
    )
    W = max(init_candidates, key=lambda res: res.sum_rate).W
    prev_rate = float(compute_sum_rate(H_tilde, W, 1.0))
    history: list[float] = [prev_rate]

    for _ in range(max_iter):
        # 1. Update MMSE scalar receivers u_k (K,)
        G = H_tilde.conj().T @ W                          # (K, K), G[k, j] = h_k^H w_j
        diag_g = np.diag(G)                               # (K,)
        rx_power = np.sum(G.real**2 + G.imag**2, axis=1) + 1.0  # (K,)
        u = diag_g / rx_power                             # (K,)

        # 2. Update MSE e_k and weights rho_k = 1 / e_k (K,)
        mse = np.maximum(1.0 - np.real(np.conj(u) * diag_g), EPS)
        rho = 1.0 / mse                                   # (K,)

        # 3. Update transmit beamformers W(mu) = (A + mu*I)^{-1} B
        # A = sum_k rho_k |u_k|^2 h_k h_k^H  (M, M) Hermitian PSD
        # B = [rho_1 u_1 h_1, ..., rho_K u_K h_K]  (M, K)
        weights_a = rho * (u.real**2 + u.imag**2)         # (K,)
        A = (H_tilde * weights_a[None, :]) @ H_tilde.conj().T
        A = 0.5 * (A + A.conj().T)                        # enforce exact Hermiticity
        B = H_tilde * (rho * u)[None, :]

        # Eigendecompose A = Q diag(d) Q^H to solve ||(A + mu*I)^{-1} B||_F^2 = P_max fast
        eigvals, Q = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 0.0)
        B_rot = Q.conj().T @ B                            # (M, K)
        c_vec = np.sum(B_rot.real**2 + B_rot.imag**2, axis=1)  # (M,)
        c_sum = float(np.sum(c_vec))

        if c_sum <= EPS:
            break

        # Check unconstrained mu = 0 first (only possible if A is strictly PD)
        use_mu_zero = False
        if eigvals[0] > 1e-12:
            power_at_zero = float(np.sum(c_vec / (eigvals**2)))
            if power_at_zero <= p_max_watt:
                use_mu_zero = True
                mu = 0.0

        if not use_mu_zero:
            # Since eigvals >= 0, g(mu) <= c_sum / mu^2 => mu_hi guarantees g(mu_hi) <= P_max
            mu_lo = 0.0
            mu_hi = float(np.sqrt(c_sum / p_max_watt)) * 1.05
            # Ensure bracket g(mu_hi) <= p_max_watt
            while float(np.sum(c_vec / ((eigvals + mu_hi) ** 2))) > p_max_watt:
                mu_hi *= 2.0

            # 60 bisection steps achieve ~1e-18 relative precision
            for _ in range(60):
                mu_mid = 0.5 * (mu_lo + mu_hi)
                val = float(np.sum(c_vec / ((eigvals + mu_mid) ** 2)))
                if val > p_max_watt:
                    mu_lo = mu_mid
                else:
                    mu_hi = mu_mid
            mu = 0.5 * (mu_lo + mu_hi)

        inv_diag = 1.0 / np.maximum(eigvals + mu, EPS)
        W = Q @ (inv_diag[:, None] * B_rot)

        # Safeguard exact power feasibility
        if float(np.linalg.norm(W, ord="fro") ** 2) > p_max_watt * (1.0 + 1e-9):
            W = normalize_power(W, p_max_watt)

        cur_rate = float(compute_sum_rate(H_tilde, W, 1.0))
        history.append(cur_rate)
        if abs(cur_rate - prev_rate) < tol:
            break
        prev_rate = cur_rate

    W = normalize_power(W, p_max_watt)
    final_rate = float(compute_sum_rate(H_arr, W, noise_power_watt))
    elapsed = time.perf_counter() - t0
    return BaselineResult(
        W=W,
        sum_rate=final_rate,
        runtime_sec=elapsed,
        history=np.asarray(history, dtype=np.float64),
        method="WMMSE",
    )
