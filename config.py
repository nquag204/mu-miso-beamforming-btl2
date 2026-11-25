"""Centralized configuration for BTL2 Topic 1: MU-MISO Beamforming via Genetic Algorithm (GA).

System & channel model (polar coordinates):
- BS with N_t ULA antennas at origin O(0, 0) of a circular cell of radius R.
- K single-antenna UEs randomly placed at polar coordinates (r_k, Phi_k) with:
      r_k = R * random(),   Phi_k = 2*pi * random()
- Path loss: PL_k(dB) = PL_0 + 10 * n * log10(d_k / d_0), PL_0 = 30 dB, d_0 = 1 m.
- Direct-sum Rician channel:
      h_hat_k = h_{k,NLoS} + h_{k,LoS},   h_{k,NLoS} ~ N(0, 1/2) + j*N(0, 1/2)
      h_k = sqrt(PL_k_linear) * h_hat_k   (shape N_t x 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Canonical project paths resolved relative to this file (CWD-independent)
BTL2_ROOT: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = BTL2_ROOT / "results"
DATA_DIR: Path = RESULTS_DIR / "data"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
REPORT_DIR: Path = BTL2_ROOT / "report"


@dataclass(frozen=True)
class SystemConfig:
    """Physical geometry, log-distance path loss, and direct-sum Rician parameters."""

    # Circular cell geometry centered at BS O(0, 0)
    cell_radius_m: float = 100.0          # Coverage circle radius R (m)
    d0_m: float = 1.0                     # Reference distance d_0 (m)

    # Log-distance path loss: PL_k [dB] = PL_0 + 10 * n * log10(d_k / d_0)
    pl0_db: float = 30.0                  # Path loss at reference distance d_0 = 1 m (dB)
    path_loss_exponent: float = 3.0       # Path-loss exponent n (urban cellular environment)

    # Direct-sum Rician channel: h_hat_k = h_{k,NLoS} + sqrt(kappa) * a(Phi_k)
    # Default K_Rician = 0 dB (kappa = 1.0) => unit-amplitude LoS vector a(Phi_k)
    # added directly to h_{k,NLoS} ~ N(0, 1/2) + j*N(0, 1/2)
    rician_k_db: float = 0.0

    # Receiver AWGN power sigma^2 (dBm)
    noise_power_dbm: float = -84.0        # Total AWGN variance (-84 dBm ~= 3.98e-12 W)

    # Default BS antennas (N_t), single-antenna users (K), and BS transmit power
    default_nt: int = 8                   # Number of BS transmit antennas N_t
    default_k: int = 4                    # Number of single-antenna users K
    default_p_max_dbm: float = 20.0       # Default BS transmit power P_max (dBm)

    @property
    def noise_power_watt(self) -> float:
        """Total receiver AWGN power sigma^2 in linear Watts."""
        return 10.0 ** ((self.noise_power_dbm - 30.0) / 10.0)


@dataclass(frozen=True)
class OptimizerConfig:
    """Hyperparameters for WMMSE baseline and Real-Coded Genetic Algorithm (GA)."""

    # WMMSE baseline stopping rules
    wmmse_max_iter: int = 100
    wmmse_tol: float = 1e-6

    # Direct-GA & Hybrid-GA encoding (2 * N_t * K real genes)
    direct_pop_size: int = 60
    direct_max_iter: int = 250

    # Struct-GA encoding (2 * K real genes via uplink-downlink duality)
    struct_pop_size: int = 40
    struct_max_iter: int = 100

    # Real-coded GA operators (Tournament + BLX-alpha + Adaptive Gaussian Mutation + Elitism)
    ga_tournament_size: int = 3
    ga_crossover_rate: float = 0.90
    ga_blx_alpha: float = 0.30
    ga_mutation_rate: float = 0.12
    ga_mutation_sigma_start: float = 0.25
    ga_mutation_sigma_end: float = 0.03
    ga_elitism_count: int = 2

    # Optional early stopping on best sum-rate stagnation (0 disables early stop)
    early_stop_patience: int = 0
    early_stop_tol: float = 1e-6


@dataclass(frozen=True)
class ExperimentConfig:
    """Monte-Carlo sweep grids and deterministic reproducibility settings."""

    seed: int = 2026
    num_trials: int = 100                 # Independent channel realizations per sweep point
    stability_runs: int = 30              # Independent GA runs on one fixed channel

    # Sweep grids
    p_max_dbm_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    nt_grid: tuple[int, ...] = (4, 8, 12, 16, 24, 32)
    k_grid: tuple[int, ...] = (2, 4, 6, 8, 10)
    rician_k_db_grid: tuple[float, ...] = (-20.0, -6.0, 0.0, 6.0, 10.0, 15.0)

    # Plotting settings
    figure_dpi: int = 300
    save_pdf: bool = True

    system: SystemConfig = field(default_factory=SystemConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


DEFAULT_CONFIG = ExperimentConfig()
