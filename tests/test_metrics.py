"""Unit and Hypothesis property-based tests for power normalization & sum-rate metrics."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from hypothesis import given, settings, strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_beampattern, compute_sinr, compute_sum_rate, normalize_power


def test_normalize_power_single_and_batch() -> None:
    rng = np.random.default_rng(7)
    p_max = 0.25

    W = rng.standard_normal((8, 4)) + 1j * rng.standard_normal((8, 4))
    W_norm = normalize_power(W, p_max)
    assert np.isclose(np.linalg.norm(W_norm, ord="fro") ** 2, p_max, rtol=1e-12)

    W_batch = rng.standard_normal((20, 8, 4)) + 1j * rng.standard_normal((20, 8, 4))
    W_batch[3] = 0.0  # degenerate zero candidate must still be normalized safely
    W_batch_norm = normalize_power(W_batch, p_max)
    norms_sq = np.sum(np.abs(W_batch_norm) ** 2, axis=(1, 2))
    np.testing.assert_allclose(norms_sq, p_max, rtol=1e-12)


def test_batch_matches_single_sum_rate() -> None:
    rng = np.random.default_rng(11)
    M, K, pop = 8, 4, 15
    H = (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))) * 1e-4
    W_batch = normalize_power(
        rng.standard_normal((pop, M, K)) + 1j * rng.standard_normal((pop, M, K)),
        p_max_watt=0.1,
    )
    noise = 1e-12

    batch_sinr = compute_sinr(H, W_batch, noise)
    batch_rates = compute_sum_rate(H, W_batch, noise)
    assert batch_sinr.shape == (pop, K)
    assert batch_rates.shape == (pop,)

    for i in range(pop):
        s_sinr = compute_sinr(H, W_batch[i], noise)
        s_rate = compute_sum_rate(H, W_batch[i], noise)
        np.testing.assert_allclose(batch_sinr[i], s_sinr, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(batch_rates[i], s_rate, rtol=1e-12, atol=1e-15)


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    M=st.integers(min_value=2, max_value=12),
    K=st.integers(min_value=1, max_value=6),
    p_max=st.floats(min_value=1e-4, max_value=10.0),
)
def test_phase_rotation_and_scale_invariance(seed: int, M: int, K: int, p_max: float) -> None:
    """Property test: Sum-rate is invariant to per-beam phase shifts e^{j phi_k}
    and simultaneous (H -> c*H, sigma^2 -> c^2*sigma^2) scaling."""
    rng = np.random.default_rng(seed)
    H = (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))) * 1e-3
    W = normalize_power(rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K)), p_max)
    noise = 1e-9

    assert np.linalg.norm(W, ord="fro") ** 2 <= p_max + 1e-9

    base_rate = float(compute_sum_rate(H, W, noise))
    assert np.isfinite(base_rate) and base_rate >= 0.0

    # Per-column phase rotation invariance
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(1, K)))
    rotated_rate = float(compute_sum_rate(H, W * phases, noise))
    assert np.isclose(base_rate, rotated_rate, rtol=1e-11, atol=1e-12)

    # Joint channel/noise scaling invariance
    scale = 3.7
    scaled_rate = float(compute_sum_rate(H * scale, W, noise * (scale**2)))
    assert np.isclose(base_rate, scaled_rate, rtol=1e-11, atol=1e-12)


def test_beampattern_shape_and_nonnegativity() -> None:
    rng = np.random.default_rng(3)
    W = normalize_power(rng.standard_normal((8, 4)) + 1j * rng.standard_normal((8, 4)), 0.1)
    grid = np.linspace(-np.pi / 2, np.pi / 2, 181)
    pat = compute_beampattern(W, grid)
    assert pat.shape == (4, 181)
    assert np.all(pat >= 0.0)
