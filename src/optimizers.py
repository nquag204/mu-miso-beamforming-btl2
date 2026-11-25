"""Vectorized Real-Coded Genetic Algorithm (GA) optimizers for downlink MU-MISO beamforming.

Implements three Genetic Algorithm variants sharing the same real-coded evolutionary
core (Tournament Selection + BLX-alpha Blend Crossover + Adaptive Gaussian Mutation
+ Elitism):
1. Direct-GA (dim = 2 * N_t * K, pure random initialization):
   Encodes Re(W) and Im(W) directly in [-1, 1]^{2 N_t K}, enforcing the BS total
   transmit power constraint via exact Frobenius projection ||W||_F^2 = P_max.

2. Hybrid-GA (dim = 2 * N_t * K, warm-started initialization):
   Uses the same direct 2*N_t*K chromosome representation, but seeds 3 initial
   chromosomes with MRT, ZF, and RZF solutions alongside (N_p - 3) random
   individuals, then evolves purely via GA operators.

3. Struct-GA (dim = 2 * K, structure-guided dual encoding):
   Exploits the optimal multiuser transmit beamforming structure via
   uplink-downlink duality (Bjornson, Bengtsson, Ottersten, IEEE SPM 2014):
       w_k(p, lambda) = sqrt(p_k) * u_k(lambda) / ||u_k(lambda)||_2
       u_k(lambda) = (I_{N_t} + sum_j (lambda_j / sigma^2) h_j h_j^H)^{-1} h_k
   Using the Woodbury/push-through identity with the K x K Gram matrix H^H H,
   batch fitness evaluation runs in O(pop_size * K^3) independent of N_t.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from config import OptimizerConfig
from src.baselines import mrt_beamformer, rzf_beamformer, zf_beamformer
from src.metrics import EPS, compute_sum_rate, normalize_power


@dataclass(frozen=True)
class OptimizationResult:
    """Output container for Genetic Algorithm beamforming optimizers."""

    W: np.ndarray            # shape (N_t, K), complex128, guaranteed ||W||_F^2 = P_max
    best_sum_rate: float     # achieved sum-rate in bps/Hz
    history: np.ndarray      # shape (n_iters + 1,), monotonic non-decreasing best sum-rate
    runtime_sec: float       # wall-clock runtime in seconds
    n_evals: int             # total objective evaluations
    method: str


# =============================================================================
# Chromosome Encoders / Decoders & Vectorized Fitness Evaluators
# =============================================================================

def encode_direct_single(W: np.ndarray) -> np.ndarray:
    """Encode complex matrix W (N_t, K) into real chromosome x in [-1, 1]^{2*N_t*K}."""
    W_arr = np.asarray(W, dtype=np.complex128)
    raw = np.concatenate((W_arr.real.ravel(), W_arr.imag.ravel()))
    max_abs = float(np.max(np.abs(raw)))
    if max_abs < EPS:
        return np.zeros_like(raw, dtype=np.float64)
    return raw / max_abs


def decode_direct_batch(
    X: np.ndarray,
    num_antennas: int,
    num_users: int,
    p_max_watt: float,
) -> np.ndarray:
    """Decode real population X of shape (pop, 2*N_t*K) into power-normalized W (pop, N_t, K)."""
    pop = X.shape[0]
    half = num_antennas * num_users
    re = X[:, :half].reshape(pop, num_antennas, num_users)
    im = X[:, half:].reshape(pop, num_antennas, num_users)
    return normalize_power(re + 1j * im, p_max_watt)


def decode_direct_single(
    x: np.ndarray,
    num_antennas: int,
    num_users: int,
    p_max_watt: float,
) -> np.ndarray:
    """Decode single real chromosome x of shape (2*N_t*K,) into W of shape (N_t, K)."""
    half = num_antennas * num_users
    W_raw = x[:half].reshape(num_antennas, num_users) + 1j * x[half:].reshape(num_antennas, num_users)
    return normalize_power(W_raw, p_max_watt)


def _extract_simplex_weights(raw: np.ndarray, total_budget: float) -> np.ndarray:
    """Map raw genes (pop, K) -> non-negative simplex weights summing to total_budget.

    Negative gene values clip to 0 so GA can explicitly turn off weak or
    heavily interfering users (essential in overloaded K > N_t or low-SNR regimes).
    """
    clipped = np.maximum(raw, 0.0)
    row_sums = np.sum(clipped, axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] < EPS
    if np.any(zero_rows):
        clipped = clipped.copy()
        clipped[zero_rows] = 1.0
        row_sums = np.sum(clipped, axis=1, keepdims=True)
    return clipped * (total_budget / row_sums)


class StructureEvaluator:
    """Fast O(K^3) vectorized evaluator for structure-guided GA chromosomes in R^{2K}.

    Precomputes H_tilde = H / sqrt(sigma^2) and Gram matrix G_H = H_tilde^H H_tilde
    once per channel realization.
    """

    def __init__(self, H: np.ndarray, p_max_watt: float, noise_power_watt: float) -> None:
        self.H = np.asarray(H, dtype=np.complex128)
        self.Nt, self.K = self.H.shape
        self.p_max_watt = float(p_max_watt)
        self.noise_power_watt = float(noise_power_watt)

        self.H_tilde = self.H / np.sqrt(self.noise_power_watt)
        self.gram = self.H_tilde.conj().T @ self.H_tilde     # (K, K) Hermitian PSD
        self.eye_k = np.eye(self.K, dtype=np.complex128)

    def evaluate_batch(self, Z: np.ndarray) -> np.ndarray:
        """Evaluate sum-rate (bps/Hz) for population batch Z of shape (pop, 2*K)."""
        pop = Z.shape[0]
        K = self.K

        p_vec = _extract_simplex_weights(Z[:, :K], self.p_max_watt)       # (pop, K)
        lam_vec = _extract_simplex_weights(Z[:, K:], self.p_max_watt)     # (pop, K)

        # Push-through identity:
        # (I_{N_t} + H_tilde diag(lam) H_tilde^H)^{-1} H_tilde = H_tilde (I_K + diag(lam) G_H)^{-1}
        mat = self.eye_k[None, :, :] + lam_vec[:, :, None] * self.gram[None, :, :]
        rhs = np.broadcast_to(self.eye_k, (pop, K, K))
        U = np.linalg.solve(mat, rhs)                                     # (pop, K, K)

        # Effective channel before column normalization: Q = G_H U
        Q = np.einsum("ij,pjk->pik", self.gram, U, optimize=True)         # (pop, K, K)

        # Column squared norms ||v_k||^2 = u_k^H G_H u_k
        col_norm_sq = np.real(np.sum(np.conj(U) * Q, axis=1))             # (pop, K)
        col_norm = np.sqrt(np.maximum(col_norm_sq, EPS))                  # (pop, K)

        scale = np.sqrt(p_vec) / col_norm                                 # (pop, K)
        G_eff = Q * scale[:, None, :]                                     # (pop, K, K)

        power_mat = G_eff.real**2 + G_eff.imag**2
        signal = np.diagonal(power_mat, axis1=1, axis2=2)                 # (pop, K)
        total_rx = np.sum(power_mat, axis=2)                              # (pop, K)
        interference = np.maximum(total_rx - signal, 0.0)
        sinr = signal / (interference + 1.0)
        return np.sum(np.log2(1.0 + sinr), axis=1)

    def decode_single(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct exact (N_t, K) beamforming matrix W from chromosome z of shape (2*K,)."""
        z2d = np.asarray(z, dtype=np.float64).reshape(1, 2 * self.K)
        p_vec = _extract_simplex_weights(z2d[:, :self.K], self.p_max_watt)[0]
        lam_vec = _extract_simplex_weights(z2d[:, self.K:], self.p_max_watt)[0]

        mat = self.eye_k + lam_vec[:, None] * self.gram
        U = np.linalg.solve(mat, self.eye_k)
        V = self.H_tilde @ U                                              # (N_t, K)

        col_norms = np.linalg.norm(V, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, EPS)
        W = (V / col_norms) * np.sqrt(p_vec)[None, :]
        return normalize_power(W, self.p_max_watt)


# =============================================================================
# Core Vectorized Real-Coded Genetic Algorithm Engine
# =============================================================================

def _run_ga_core(
    eval_fn,
    dim: int,
    low: float,
    high: float,
    pop_size: int,
    max_iter: int,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
    seed_chromosomes: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Vectorized Real-Coded GA (Tournament + BLX-alpha + Adaptive Gaussian Mutation + Elitism)."""
    pop = rng.uniform(low, high, size=(pop_size, dim))
    if seed_chromosomes is not None and len(seed_chromosomes) > 0:
        n_seed = min(len(seed_chromosomes), pop_size)
        pop[:n_seed] = np.clip(seed_chromosomes[:n_seed], low, high)

    fitness = eval_fn(pop)
    n_evals = pop_size

    best_idx = int(np.argmax(fitness))
    best_x = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])

    history = np.empty(max_iter + 1, dtype=np.float64)
    history[0] = best_fit

    n_elite = min(opt_cfg.ga_elitism_count, pop_size - 2)
    n_offspring = pop_size - n_elite
    n_pairs = (n_offspring + 1) // 2
    t_size = min(opt_cfg.ga_tournament_size, pop_size)
    span = high - low
    stagnation = 0

    for it in range(1, max_iter + 1):
        # 1. Elitism: preserve top-k chromosomes unchanged
        elite_idx = np.argpartition(fitness, -n_elite)[-n_elite:]
        elites = pop[elite_idx].copy()

        # 2. Vectorized Tournament Selection for 2 * n_pairs parents
        cand_idx = rng.integers(0, pop_size, size=(2 * n_pairs, t_size))
        cand_fit = fitness[cand_idx]
        winners = cand_idx[np.arange(2 * n_pairs), np.argmax(cand_fit, axis=1)]
        p1 = pop[winners[:n_pairs]]
        p2 = pop[winners[n_pairs:]]

        # 3. BLX-alpha Blend Crossover (Eshelman & Schaffer, 1993)
        do_cross = rng.random((n_pairs, 1)) < opt_cfg.ga_crossover_rate
        lo_p = np.minimum(p1, p2)
        hi_p = np.maximum(p1, p2)
        diff = hi_p - lo_p
        ext_lo = lo_p - opt_cfg.ga_blx_alpha * diff
        ext_hi = hi_p + opt_cfg.ga_blx_alpha * diff

        c1 = np.where(do_cross, rng.uniform(ext_lo, ext_hi), p1)
        c2 = np.where(do_cross, rng.uniform(ext_lo, ext_hi), p2)
        children = np.vstack((c1, c2))[:n_offspring]

        # 4. Linearly decaying Gaussian mutation + box boundary clipping
        frac = it / max(max_iter, 1)
        sigma = (
            opt_cfg.ga_mutation_sigma_start
            - frac * (opt_cfg.ga_mutation_sigma_start - opt_cfg.ga_mutation_sigma_end)
        ) * span
        mut_mask = rng.random(children.shape) < opt_cfg.ga_mutation_rate
        children = children + mut_mask * rng.normal(0.0, sigma, size=children.shape)
        children = np.clip(children, low, high)

        # 5. Form next generation & evaluate population batch
        pop = np.vstack((elites, children))
        fitness = eval_fn(pop)
        n_evals += pop_size

        cur_best_idx = int(np.argmax(fitness))
        cur_best_fit = float(fitness[cur_best_idx])
        if cur_best_fit > best_fit:
            if cur_best_fit - best_fit < opt_cfg.early_stop_tol:
                stagnation += 1
            else:
                stagnation = 0
            best_fit = cur_best_fit
            best_x = pop[cur_best_idx].copy()
        else:
            stagnation += 1

        history[it] = best_fit

        if opt_cfg.early_stop_patience > 0 and stagnation >= opt_cfg.early_stop_patience:
            history[it + 1 :] = best_fit
            break

    return best_x, best_fit, history, n_evals


# =============================================================================
# Public Genetic Algorithm Beamformer APIs
# =============================================================================

def direct_ga_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
    pop_size: int | None = None,
    max_iter: int | None = None,
) -> OptimizationResult:
    """Direct Real-Coded GA in R^{2*N_t*K} with random initialization & Frobenius projection."""
    t0 = time.perf_counter()
    H_arr = np.asarray(H, dtype=np.complex128)
    Nt, K = H_arr.shape
    dim = 2 * Nt * K

    def eval_fn(X: np.ndarray) -> np.ndarray:
        W_batch = decode_direct_batch(X, Nt, K, p_max_watt)
        return np.asarray(compute_sum_rate(H_arr, W_batch, noise_power_watt), dtype=np.float64)

    best_x, _, history, n_evals = _run_ga_core(
        eval_fn=eval_fn,
        dim=dim,
        low=-1.0,
        high=1.0,
        pop_size=opt_cfg.direct_pop_size if pop_size is None else pop_size,
        max_iter=opt_cfg.direct_max_iter if max_iter is None else max_iter,
        opt_cfg=opt_cfg,
        rng=rng,
    )
    W_best = decode_direct_single(best_x, Nt, K, p_max_watt)
    best_rate = float(compute_sum_rate(H_arr, W_best, noise_power_watt))
    return OptimizationResult(
        W=W_best,
        best_sum_rate=best_rate,
        history=history,
        runtime_sec=time.perf_counter() - t0,
        n_evals=n_evals,
        method="Direct-GA",
    )


def hybrid_ga_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
    pop_size: int | None = None,
    max_iter: int | None = None,
) -> OptimizationResult:
    """Hybrid Warm-Start Real-Coded GA in R^{2*N_t*K} seeded with MRT, ZF, and RZF."""
    t0 = time.perf_counter()
    H_arr = np.asarray(H, dtype=np.complex128)
    Nt, K = H_arr.shape
    dim = 2 * Nt * K

    seeds = np.vstack(
        (
            encode_direct_single(mrt_beamformer(H_arr, p_max_watt, noise_power_watt).W),
            encode_direct_single(zf_beamformer(H_arr, p_max_watt, noise_power_watt).W),
            encode_direct_single(rzf_beamformer(H_arr, p_max_watt, noise_power_watt).W),
        )
    )

    def eval_fn(X: np.ndarray) -> np.ndarray:
        W_batch = decode_direct_batch(X, Nt, K, p_max_watt)
        return np.asarray(compute_sum_rate(H_arr, W_batch, noise_power_watt), dtype=np.float64)

    best_x, _, history, n_evals = _run_ga_core(
        eval_fn=eval_fn,
        dim=dim,
        low=-1.0,
        high=1.0,
        pop_size=opt_cfg.direct_pop_size if pop_size is None else pop_size,
        max_iter=opt_cfg.direct_max_iter if max_iter is None else max_iter,
        opt_cfg=opt_cfg,
        rng=rng,
        seed_chromosomes=seeds,
    )
    W_best = decode_direct_single(best_x, Nt, K, p_max_watt)
    best_rate = float(compute_sum_rate(H_arr, W_best, noise_power_watt))
    return OptimizationResult(
        W=W_best,
        best_sum_rate=best_rate,
        history=history,
        runtime_sec=time.perf_counter() - t0,
        n_evals=n_evals,
        method="Hybrid-GA",
    )


def struct_ga_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
    pop_size: int | None = None,
    max_iter: int | None = None,
) -> OptimizationResult:
    """Structure-Guided Real-Coded GA (Struct-GA) in R^{2K} via uplink-downlink duality."""
    t0 = time.perf_counter()
    evaluator = StructureEvaluator(H, p_max_watt, noise_power_watt)
    dim = 2 * evaluator.K

    best_z, _, history, n_evals = _run_ga_core(
        eval_fn=evaluator.evaluate_batch,
        dim=dim,
        low=-0.05,
        high=1.0,
        pop_size=opt_cfg.struct_pop_size if pop_size is None else pop_size,
        max_iter=opt_cfg.struct_max_iter if max_iter is None else max_iter,
        opt_cfg=opt_cfg,
        rng=rng,
    )
    W_best = evaluator.decode_single(best_z)
    best_rate = float(compute_sum_rate(evaluator.H, W_best, noise_power_watt))
    return OptimizationResult(
        W=W_best,
        best_sum_rate=best_rate,
        history=history,
        runtime_sec=time.perf_counter() - t0,
        n_evals=n_evals,
        method="Struct-GA",
    )
