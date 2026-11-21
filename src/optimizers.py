"""Vectorized heuristic & evolutionary optimizers for downlink MU-MISO beamforming.

Implements two complementary solution representations:
1. Direct Complex Weight Encoding (dim = 2*M*K):
   Optimizes real & imaginary parts of W in [-1, 1]^{2MK} directly, enforcing
   the total BS power constraint via exact Frobenius projection ||W||_F^2 = P_max.
   Algorithms: Direct-DE, Direct-PSO, Direct-GA.

2. Structure-Guided Optimal Beamforming Encoding (dim = 2*K):
   Exploits the optimal multiuser transmit beamforming structure via
   uplink-downlink duality (Bjornson, Bengtsson, Ottersten, IEEE SPM 2014):
       w_k(p, lambda) = sqrt(p_k) * u_k(lambda) / ||u_k(lambda)||_2
       u_k(lambda) = (I_M + sum_j (lambda_j / sigma^2) h_j h_j^H)^{-1} h_k
   Using the Woodbury/push-through identity with the K x K Gram matrix H^H H,
   batch fitness evaluation runs in O(pop_size * K^3) independent of M.
   Algorithms: Struct-DE, Struct-PSO, plus an open-source SciPy DE cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import differential_evolution as scipy_de

from config import OptimizerConfig
from src.metrics import EPS, compute_sum_rate, normalize_power


@dataclass(frozen=True)
class OptimizationResult:
    """Output container for evolutionary / metaheuristic beamforming optimizers."""

    W: np.ndarray            # shape (M, K), complex128, guaranteed ||W||_F^2 = P_max
    best_sum_rate: float     # achieved sum-rate in bps/Hz
    history: np.ndarray      # shape (n_iters + 1,), monotonic non-decreasing best sum-rate
    runtime_sec: float       # wall-clock runtime in seconds
    n_evals: int             # total objective evaluations
    method: str


# =============================================================================
# Solution Encoders / Vectorized Evaluators
# =============================================================================

def decode_direct_batch(
    X: np.ndarray,
    num_antennas: int,
    num_users: int,
    p_max_watt: float,
) -> np.ndarray:
    """Decode real population X of shape (pop, 2*M*K) into power-normalized W (pop, M, K)."""
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
    """Decode single real vector x of shape (2*M*K,) into W of shape (M, K)."""
    half = num_antennas * num_users
    W_raw = x[:half].reshape(num_antennas, num_users) + 1j * x[half:].reshape(num_antennas, num_users)
    return normalize_power(W_raw, p_max_watt)


def _extract_simplex_weights(raw: np.ndarray, total_budget: float) -> np.ndarray:
    """Map raw genes (pop, K) -> non-negative simplex weights summing to total_budget.

    Negative values clip to 0 so the optimizer can explicitly switch off weak or
    excess users (essential in overloaded K > M or low-SNR regimes).
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
    """Fast O(K^3) vectorized evaluator for structure-guided beamforming in R^{2K}.

    Precomputes H_tilde = H / sqrt(sigma^2) and Gram matrix G_H = H_tilde^H H_tilde
    once per channel realization.
    """

    def __init__(self, H: np.ndarray, p_max_watt: float, noise_power_watt: float) -> None:
        self.H = np.asarray(H, dtype=np.complex128)
        self.M, self.K = self.H.shape
        self.p_max_watt = float(p_max_watt)
        self.noise_power_watt = float(noise_power_watt)

        self.H_tilde = self.H / np.sqrt(self.noise_power_watt)
        self.gram = self.H_tilde.conj().T @ self.H_tilde     # (K, K) Hermitian PSD
        self.eye_k = np.eye(self.K, dtype=np.complex128)

    def evaluate_batch(self, Z: np.ndarray) -> np.ndarray:
        """Evaluate sum-rate (bps/Hz) for batch Z of shape (pop, 2*K)."""
        pop = Z.shape[0]
        K = self.K

        p_vec = _extract_simplex_weights(Z[:, :K], self.p_max_watt)       # (pop, K)
        lam_vec = _extract_simplex_weights(Z[:, K:], self.p_max_watt)     # (pop, K)

        # Push-through identity:
        # (I_M + H_tilde diag(lam) H_tilde^H)^{-1} H_tilde = H_tilde (I_K + diag(lam) G_H)^{-1}
        # Solve (I_K + diag(lam) G_H) U = I_K for batch of (pop, K, K)
        mat = self.eye_k[None, :, :] + lam_vec[:, :, None] * self.gram[None, :, :]
        rhs = np.broadcast_to(self.eye_k, (pop, K, K))
        U = np.linalg.solve(mat, rhs)                                     # (pop, K, K)

        # Effective channel before column normalization: Q = H_tilde^H (H_tilde U) = G_H U
        Q = np.einsum("ij,pjk->pik", self.gram, U, optimize=True)         # (pop, K, K)

        # Column squared norms ||v_k||^2 = u_k^H G_H u_k = Re(sum_r conj(U[r,k]) * Q[r,k])
        col_norm_sq = np.real(np.sum(np.conj(U) * Q, axis=1))             # (pop, K)
        col_norm = np.sqrt(np.maximum(col_norm_sq, EPS))                  # (pop, K)

        # Effective channel with normalized columns & allocated powers: G_eff[p, k, j]
        scale = np.sqrt(p_vec) / col_norm                                 # (pop, K)
        G_eff = Q * scale[:, None, :]                                     # (pop, K, K)

        power_mat = G_eff.real**2 + G_eff.imag**2
        signal = np.diagonal(power_mat, axis1=1, axis2=2)                 # (pop, K)
        total_rx = np.sum(power_mat, axis=2)                              # (pop, K)
        interference = np.maximum(total_rx - signal, 0.0)
        sinr = signal / (interference + 1.0)                              # effective noise = 1.0
        return np.sum(np.log2(1.0 + sinr), axis=1)

    def decode_single(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct exact (M, K) beamforming matrix W from best vector z of shape (2*K,)."""
        z2d = np.asarray(z, dtype=np.float64).reshape(1, 2 * self.K)
        p_vec = _extract_simplex_weights(z2d[:, :self.K], self.p_max_watt)[0]
        lam_vec = _extract_simplex_weights(z2d[:, self.K:], self.p_max_watt)[0]

        mat = self.eye_k + lam_vec[:, None] * self.gram
        U = np.linalg.solve(mat, self.eye_k)
        V = self.H_tilde @ U                                              # (M, K)

        col_norms = np.linalg.norm(V, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, EPS)
        W = (V / col_norms) * np.sqrt(p_vec)[None, :]
        return normalize_power(W, self.p_max_watt)


# =============================================================================
# Core Vectorized Metaheuristic Engines
# =============================================================================

def _run_de_core(
    eval_fn,
    dim: int,
    low: float,
    high: float,
    pop_size: int,
    max_iter: int,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Vectorized DE/current-to-best/1/bin with dithered F and greedy selection."""
    pop = rng.uniform(low, high, size=(pop_size, dim))
    fitness = eval_fn(pop)
    n_evals = pop_size

    best_idx = int(np.argmax(fitness))
    best_x = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])
    history = np.empty(max_iter + 1, dtype=np.float64)
    history[0] = best_fit

    idx_range = np.arange(pop_size)
    stagnation = 0

    for it in range(1, max_iter + 1):
        # Sample two distinct random partner indices r1, r2 != i per individual
        r1 = rng.integers(0, pop_size - 1, size=pop_size)
        r1 = np.where(r1 >= idx_range, r1 + 1, r1)
        r2 = rng.integers(0, pop_size - 2, size=pop_size)
        # Shift past min(i, r1) then max(i, r1) so r2 is distinct from both i and r1
        low_i = np.minimum(idx_range, r1)
        high_i = np.maximum(idx_range, r1)
        r2 = np.where(r2 >= low_i, r2 + 1, r2)
        r2 = np.where(r2 >= high_i, r2 + 1, r2)

        # Dithered mutation factor F per individual for robust exploration/exploitation
        f_scale = rng.uniform(opt_cfg.de_f_min, opt_cfg.de_f_max, size=(pop_size, 1))
        mutant = pop + f_scale * (best_x[None, :] - pop) + f_scale * (pop[r1] - pop[r2])
        mutant = np.clip(mutant, low, high)

        # Binomial crossover guaranteeing at least one mutated coordinate j_rand
        cross_mask = rng.random((pop_size, dim)) < opt_cfg.de_cr
        j_rand = rng.integers(0, dim, size=pop_size)
        cross_mask[idx_range, j_rand] = True
        trial = np.where(cross_mask, mutant, pop)

        trial_fit = eval_fn(trial)
        n_evals += pop_size

        improved = trial_fit >= fitness
        pop = np.where(improved[:, None], trial, pop)
        fitness = np.where(improved, trial_fit, fitness)

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


def _run_pso_core(
    eval_fn,
    dim: int,
    low: float,
    high: float,
    pop_size: int,
    max_iter: int,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Vectorized Global-Best PSO with linear inertia decay and velocity clamping."""
    span = high - low
    v_max = opt_cfg.pso_v_max_ratio * span

    pos = rng.uniform(low, high, size=(pop_size, dim))
    vel = rng.uniform(-v_max, v_max, size=(pop_size, dim))

    fitness = eval_fn(pos)
    n_evals = pop_size

    pbest_pos = pos.copy()
    pbest_fit = fitness.copy()

    gbest_idx = int(np.argmax(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    history = np.empty(max_iter + 1, dtype=np.float64)
    history[0] = gbest_fit
    stagnation = 0

    for it in range(1, max_iter + 1):
        frac = it / max(max_iter, 1)
        w_inertia = opt_cfg.pso_w_start - frac * (opt_cfg.pso_w_start - opt_cfg.pso_w_end)

        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        vel = (
            w_inertia * vel
            + opt_cfg.pso_c1 * r1 * (pbest_pos - pos)
            + opt_cfg.pso_c2 * r2 * (gbest_pos[None, :] - pos)
        )
        vel = np.clip(vel, -v_max, v_max)

        pos = pos + vel
        # Reflecting / damping boundary handling keeps particles active inside box
        below = pos < low
        above = pos > high
        pos = np.clip(pos, low, high)
        vel = np.where(below | above, -0.5 * vel, vel)

        fitness = eval_fn(pos)
        n_evals += pop_size

        improved = fitness > pbest_fit
        pbest_pos = np.where(improved[:, None], pos, pbest_pos)
        pbest_fit = np.where(improved, fitness, pbest_fit)

        cur_best_idx = int(np.argmax(pbest_fit))
        cur_best_fit = float(pbest_fit[cur_best_idx])
        if cur_best_fit > gbest_fit:
            if cur_best_fit - gbest_fit < opt_cfg.early_stop_tol:
                stagnation += 1
            else:
                stagnation = 0
            gbest_fit = cur_best_fit
            gbest_pos = pbest_pos[cur_best_idx].copy()
        else:
            stagnation += 1

        history[it] = gbest_fit

        if opt_cfg.early_stop_patience > 0 and stagnation >= opt_cfg.early_stop_patience:
            history[it + 1 :] = gbest_fit
            break

    return gbest_pos, gbest_fit, history, n_evals


def _run_ga_core(
    eval_fn,
    dim: int,
    low: float,
    high: float,
    pop_size: int,
    max_iter: int,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Vectorized Real-Coded GA (Tournament + BLX-alpha + Adaptive Gaussian Mutation + Elitism)."""
    pop = rng.uniform(low, high, size=(pop_size, dim))
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

    for it in range(1, max_iter + 1):
        # Preserve top-k elites
        elite_idx = np.argpartition(fitness, -n_elite)[-n_elite:]
        elites = pop[elite_idx].copy()

        # Vectorized tournament selection for 2 * n_pairs parents
        cand_idx = rng.integers(0, pop_size, size=(2 * n_pairs, t_size))
        cand_fit = fitness[cand_idx]
        winners = cand_idx[np.arange(2 * n_pairs), np.argmax(cand_fit, axis=1)]
        p1 = pop[winners[:n_pairs]]
        p2 = pop[winners[n_pairs:]]

        # BLX-alpha blend crossover
        do_cross = rng.random((n_pairs, 1)) < opt_cfg.ga_crossover_rate
        lo_p = np.minimum(p1, p2)
        hi_p = np.maximum(p1, p2)
        diff = hi_p - lo_p
        ext_lo = lo_p - opt_cfg.ga_blx_alpha * diff
        ext_hi = hi_p + opt_cfg.ga_blx_alpha * diff

        c1 = np.where(do_cross, rng.uniform(ext_lo, ext_hi), p1)
        c2 = np.where(do_cross, rng.uniform(ext_lo, ext_hi), p2)
        children = np.vstack((c1, c2))[:n_offspring]

        # Linearly decaying Gaussian mutation
        frac = it / max(max_iter, 1)
        sigma = (
            opt_cfg.ga_mutation_sigma_start
            - frac * (opt_cfg.ga_mutation_sigma_start - opt_cfg.ga_mutation_sigma_end)
        ) * span
        mut_mask = rng.random(children.shape) < opt_cfg.ga_mutation_rate
        children = children + mut_mask * rng.normal(0.0, sigma, size=children.shape)
        children = np.clip(children, low, high)

        pop = np.vstack((elites, children))
        fitness = eval_fn(pop)
        n_evals += pop_size

        cur_best_idx = int(np.argmax(fitness))
        cur_best_fit = float(fitness[cur_best_idx])
        if cur_best_fit > best_fit:
            best_fit = cur_best_fit
            best_x = pop[cur_best_idx].copy()

        history[it] = best_fit

    return best_x, best_fit, history, n_evals


# =============================================================================
# Public Optimizer APIs (Direct Encoding 2*M*K)
# =============================================================================

def direct_de_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
    pop_size: int | None = None,
    max_iter: int | None = None,
) -> OptimizationResult:
    """Direct Differential Evolution (DE) in R^{2MK} with Frobenius power projection."""
    t0 = time.perf_counter()
    H_arr = np.asarray(H, dtype=np.complex128)
    M, K = H_arr.shape
    dim = 2 * M * K

    def eval_fn(X: np.ndarray) -> np.ndarray:
        W_batch = decode_direct_batch(X, M, K, p_max_watt)
        return np.asarray(compute_sum_rate(H_arr, W_batch, noise_power_watt), dtype=np.float64)

    best_x, _, history, n_evals = _run_de_core(
        eval_fn=eval_fn,
        dim=dim,
        low=-1.0,
        high=1.0,
        pop_size=opt_cfg.direct_pop_size if pop_size is None else pop_size,
        max_iter=opt_cfg.direct_max_iter if max_iter is None else max_iter,
        opt_cfg=opt_cfg,
        rng=rng,
    )
    W_best = decode_direct_single(best_x, M, K, p_max_watt)
    best_rate = float(compute_sum_rate(H_arr, W_best, noise_power_watt))
    return OptimizationResult(
        W=W_best,
        best_sum_rate=best_rate,
        history=history,
        runtime_sec=time.perf_counter() - t0,
        n_evals=n_evals,
        method="Direct-DE",
    )


def direct_pso_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
    pop_size: int | None = None,
    max_iter: int | None = None,
) -> OptimizationResult:
    """Direct Particle Swarm Optimization (PSO) in R^{2MK} with power projection."""
    t0 = time.perf_counter()
    H_arr = np.asarray(H, dtype=np.complex128)
    M, K = H_arr.shape
    dim = 2 * M * K

    def eval_fn(X: np.ndarray) -> np.ndarray:
        W_batch = decode_direct_batch(X, M, K, p_max_watt)
        return np.asarray(compute_sum_rate(H_arr, W_batch, noise_power_watt), dtype=np.float64)

    best_x, _, history, n_evals = _run_pso_core(
        eval_fn=eval_fn,
        dim=dim,
        low=-1.0,
        high=1.0,
        pop_size=opt_cfg.direct_pop_size if pop_size is None else pop_size,
        max_iter=opt_cfg.direct_max_iter if max_iter is None else max_iter,
        opt_cfg=opt_cfg,
        rng=rng,
    )
    W_best = decode_direct_single(best_x, M, K, p_max_watt)
    best_rate = float(compute_sum_rate(H_arr, W_best, noise_power_watt))
    return OptimizationResult(
        W=W_best,
        best_sum_rate=best_rate,
        history=history,
        runtime_sec=time.perf_counter() - t0,
        n_evals=n_evals,
        method="Direct-PSO",
    )


def direct_ga_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
    pop_size: int | None = None,
    max_iter: int | None = None,
) -> OptimizationResult:
    """Direct Real-Coded Genetic Algorithm (GA) in R^{2MK} with power projection."""
    t0 = time.perf_counter()
    H_arr = np.asarray(H, dtype=np.complex128)
    M, K = H_arr.shape
    dim = 2 * M * K

    def eval_fn(X: np.ndarray) -> np.ndarray:
        W_batch = decode_direct_batch(X, M, K, p_max_watt)
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
    W_best = decode_direct_single(best_x, M, K, p_max_watt)
    best_rate = float(compute_sum_rate(H_arr, W_best, noise_power_watt))
    return OptimizationResult(
        W=W_best,
        best_sum_rate=best_rate,
        history=history,
        runtime_sec=time.perf_counter() - t0,
        n_evals=n_evals,
        method="Direct-GA",
    )


# =============================================================================
# Public Optimizer APIs (Structure-Guided Encoding 2*K via Uplink-Downlink Duality)
# =============================================================================

def struct_de_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
    pop_size: int | None = None,
    max_iter: int | None = None,
) -> OptimizationResult:
    """Structure-Guided Differential Evolution (Struct-DE) in R^{2K}."""
    t0 = time.perf_counter()
    evaluator = StructureEvaluator(H, p_max_watt, noise_power_watt)
    dim = 2 * evaluator.K

    best_z, _, history, n_evals = _run_de_core(
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
        method="Struct-DE",
    )


def struct_pso_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    opt_cfg: OptimizerConfig,
    rng: np.random.Generator,
    pop_size: int | None = None,
    max_iter: int | None = None,
) -> OptimizationResult:
    """Structure-Guided Particle Swarm Optimization (Struct-PSO) in R^{2K}."""
    t0 = time.perf_counter()
    evaluator = StructureEvaluator(H, p_max_watt, noise_power_watt)
    dim = 2 * evaluator.K

    best_z, _, history, n_evals = _run_pso_core(
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
        method="Struct-PSO",
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
    """Structure-Guided Genetic Algorithm (Struct-GA) in R^{2K}."""
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


def scipy_de_beamformer(
    H: np.ndarray,
    p_max_watt: float,
    noise_power_watt: float,
    seed: int = 2026,
    encoding: str = "struct",
    maxiter: int = 80,
    popsize: int = 10,
) -> OptimizationResult:
    """Open-source benchmark wrapper around scipy.optimize.differential_evolution.

    Used to cross-validate custom vectorized NumPy optimizers against SciPy's
    reference implementation.
    """
    t0 = time.perf_counter()
    H_arr = np.asarray(H, dtype=np.complex128)
    M, K = H_arr.shape

    if encoding == "struct":
        evaluator = StructureEvaluator(H_arr, p_max_watt, noise_power_watt)
        bounds = [(-0.05, 1.0)] * (2 * K)

        def neg_obj_vec(X_T: np.ndarray) -> np.ndarray:
            # SciPy vectorized=True passes array of shape (dim, S)
            return -evaluator.evaluate_batch(X_T.T)

        res = scipy_de(
            neg_obj_vec,
            bounds=bounds,
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=False,
            updating="deferred",
            vectorized=True,
            seed=seed,
        )
        W_best = evaluator.decode_single(res.x)
    elif encoding == "direct":
        bounds = [(-1.0, 1.0)] * (2 * M * K)

        def neg_obj_direct(X_T: np.ndarray) -> np.ndarray:
            W_batch = decode_direct_batch(X_T.T, M, K, p_max_watt)
            return -np.asarray(compute_sum_rate(H_arr, W_batch, noise_power_watt), dtype=np.float64)

        res = scipy_de(
            neg_obj_direct,
            bounds=bounds,
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=False,
            updating="deferred",
            vectorized=True,
            seed=seed,
        )
        W_best = decode_direct_single(res.x, M, K, p_max_watt)
    else:
        raise ValueError(f"Unsupported encoding: {encoding!r}")

    best_rate = float(compute_sum_rate(H_arr, W_best, noise_power_watt))
    return OptimizationResult(
        W=W_best,
        best_sum_rate=best_rate,
        history=np.array([best_rate], dtype=np.float64),
        runtime_sec=time.perf_counter() - t0,
        n_evals=int(res.nfev),
        method=f"SciPy-DE-{encoding}",
    )
