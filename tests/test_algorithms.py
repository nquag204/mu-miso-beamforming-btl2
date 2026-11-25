"""Verification tests for baselines (MRT, ZF, RZF, WMMSE) and GA optimizers (Direct/Hybrid/Struct)."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OptimizerConfig, SystemConfig
from src.baselines import mrt_beamformer, rzf_beamformer, wmmse_beamformer, zf_beamformer
from src.channel import generate_rician_channel
from src.metrics import compute_sum_rate
from src.optimizers import (
    StructureEvaluator,
    direct_ga_beamformer,
    hybrid_ga_beamformer,
    struct_ga_beamformer,
)
from src.utils import dbm_to_watt


def _sample_channel(Nt: int = 8, K: int = 4, seed: int = 2026) -> tuple[np.ndarray, float, float]:
    sys_cfg = SystemConfig()
    ch = generate_rician_channel(Nt, K, sys_cfg, np.random.default_rng(seed))
    p_max = float(dbm_to_watt(sys_cfg.default_p_max_dbm))
    noise = sys_cfg.noise_power_watt
    return ch.H, p_max, noise


def test_zf_eliminates_inter_user_interference() -> None:
    """When N_t >= K, |h_k^H w_j| must be ~0 for all j != k."""
    H, p_max, noise = _sample_channel(Nt=8, K=4, seed=101)
    res = zf_beamformer(H, p_max, noise)
    assert np.linalg.norm(res.W, ord="fro") ** 2 <= p_max + 1e-9

    G = H.conj().T @ res.W
    diag_mag = np.abs(np.diag(G))
    off_diag = G - np.diag(np.diag(G))
    assert np.max(np.abs(off_diag)) < 1e-10 * np.min(diag_mag)


def test_wmmse_monotonic_and_beats_linear_baselines() -> None:
    H, p_max, noise = _sample_channel(Nt=8, K=4, seed=202)
    mrt = mrt_beamformer(H, p_max, noise)
    zf = zf_beamformer(H, p_max, noise)
    rzf = rzf_beamformer(H, p_max, noise)
    wmmse = wmmse_beamformer(H, p_max, noise)

    assert np.linalg.norm(wmmse.W, ord="fro") ** 2 <= p_max + 1e-9
    assert np.all(np.diff(wmmse.history) >= -1e-7)
    assert wmmse.sum_rate >= rzf.sum_rate - 1e-6
    assert wmmse.sum_rate >= mrt.sum_rate - 1e-6
    assert wmmse.sum_rate >= zf.sum_rate - 1e-6


def test_structure_evaluator_matches_exact_sum_rate() -> None:
    """Verify O(K^3) Gram-matrix evaluator matches explicit W reconstruction to <1e-10."""
    H, p_max, noise = _sample_channel(Nt=12, K=5, seed=303)
    evaluator = StructureEvaluator(H, p_max, noise)
    rng = np.random.default_rng(77)
    Z = rng.uniform(-0.05, 1.0, size=(20, 10))
    fast_rates = evaluator.evaluate_batch(Z)

    for i in range(Z.shape[0]):
        W = evaluator.decode_single(Z[i])
        assert np.linalg.norm(W, ord="fro") ** 2 <= p_max + 1e-9
        exact_rate = float(compute_sum_rate(H, W, noise))
        np.testing.assert_allclose(fast_rates[i], exact_rate, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "solver_fn",
    [
        direct_ga_beamformer,
        hybrid_ga_beamformer,
        struct_ga_beamformer,
    ],
)
def test_ga_variants_feasibility_and_monotonic_convergence(solver_fn) -> None:
    H, p_max, noise = _sample_channel(Nt=6, K=3, seed=404)
    opt_cfg = OptimizerConfig(direct_pop_size=24, direct_max_iter=40, struct_pop_size=24, struct_max_iter=35)
    res = solver_fn(H, p_max, noise, opt_cfg, np.random.default_rng(55))

    assert res.W.shape == (6, 3)
    assert np.linalg.norm(res.W, ord="fro") ** 2 <= p_max + 1e-9
    assert np.isfinite(res.best_sum_rate) and res.best_sum_rate > 0.0
    assert np.all(np.diff(res.history) >= -1e-12), f"{res.method} history decreased!"
    assert np.isclose(res.best_sum_rate, float(compute_sum_rate(H, res.W, noise)), rtol=1e-10)


def test_hybrid_ga_dominates_linear_baselines() -> None:
    """Elitism + MRT/ZF/RZF seeding guarantees Hybrid-GA >= max(MRT, ZF, RZF)."""
    H, p_max, noise = _sample_channel(Nt=8, K=4, seed=455)
    opt_cfg = OptimizerConfig(direct_pop_size=30, direct_max_iter=50)
    mrt = mrt_beamformer(H, p_max, noise)
    zf = zf_beamformer(H, p_max, noise)
    rzf = rzf_beamformer(H, p_max, noise)
    hga = hybrid_ga_beamformer(H, p_max, noise, opt_cfg, np.random.default_rng(9))

    assert hga.best_sum_rate >= max(mrt.sum_rate, zf.sum_rate, rzf.sum_rate) - 1e-9


def test_overloaded_regime_nt_less_than_k() -> None:
    """Verify all beamformers remain numerically stable when K > N_t (overloaded)."""
    H, p_max, noise = _sample_channel(Nt=4, K=6, seed=505)
    opt_cfg = OptimizerConfig(direct_pop_size=24, direct_max_iter=30, struct_pop_size=24, struct_max_iter=40)

    for fn in (mrt_beamformer, zf_beamformer, rzf_beamformer, wmmse_beamformer):
        r = fn(H, p_max, noise)
        assert np.isfinite(r.sum_rate) and r.sum_rate > 0.0
        assert np.linalg.norm(r.W, ord="fro") ** 2 <= p_max + 1e-9

    for ga_fn in (direct_ga_beamformer, hybrid_ga_beamformer, struct_ga_beamformer):
        g = ga_fn(H, p_max, noise, opt_cfg, np.random.default_rng(8))
        assert np.isfinite(g.best_sum_rate) and g.best_sum_rate > 0.0
        assert np.linalg.norm(g.W, ord="fro") ** 2 <= p_max + 1e-9


def test_struct_ga_outperforms_mrt_and_matches_wmmse() -> None:
    H, p_max, noise = _sample_channel(Nt=8, K=4, seed=606)
    opt_cfg = OptimizerConfig(struct_pop_size=40, struct_max_iter=100)

    mrt = mrt_beamformer(H, p_max, noise)
    wmmse = wmmse_beamformer(H, p_max, noise)
    sga = struct_ga_beamformer(H, p_max, noise, opt_cfg, np.random.default_rng(123))

    assert sga.best_sum_rate > mrt.sum_rate
    assert sga.best_sum_rate >= 0.98 * wmmse.sum_rate
