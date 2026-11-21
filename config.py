"""Centralized configuration for BTL2 Topic 1: MU-MISO Beamforming Optimization.

All physical, channel, baseline, metaheuristic, and Monte-Carlo experiment
parameters live in this module so no magic numbers are hardcoded across scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Canonical workspace paths resolved relative to this file (CWD-independent)
BTL2_ROOT: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = BTL2_ROOT / "results"
DATA_DIR: Path = RESULTS_DIR / "data"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
REPORT_DIR: Path = BTL2_ROOT / "report"


@dataclass(frozen=True)
class SystemConfig:
    """Physical and radio-propagation parameters for downlink MU-MISO."""

    # Radio frequency & propagation (5G NR n78 urban microcell)
    fc_hz: float = 3.5e9                  # Carrier frequency: 3.5 GHz
    c_mps: float = 3.0e8                  # Speed of light (m/s)
    d0_m: float = 1.0                     # Reference distance d0 (m)
    path_loss_exponent: float = 3.0       # Log-distance path-loss exponent alpha
    rician_k_db: float = 6.0              # Default Rician K-factor (dB)

    # Cell geometry (120-degree sector annulus centered at BS (0, 0))
    r_min_m: float = 20.0                 # Minimum BS-UE distance (m)
    r_max_m: float = 100.0                # Maximum BS-UE distance (m)
    sector_angle_deg: float = 120.0       # BS ULA coverage sector width (degrees)

    # Receiver noise budget: sigma^2 = N0 + 10*log10(B) + NF = -94 dBm
    bandwidth_hz: float = 10.0e6          # Channel bandwidth: 10 MHz
    noise_psd_dbm_hz: float = -174.0      # Thermal noise PSD (dBm/Hz)
    noise_figure_db: float = 10.0         # Receiver noise figure (dB)

    # Default antenna / user / transmit power setup
    default_m: int = 8                    # Number of BS ULA antennas (M)
    default_k: int = 4                    # Number of single-antenna UEs (K)
    default_p_max_dbm: float = 20.0       # Default BS transmit power P_max (dBm)

    @property
    def wavelength_m(self) -> float:
        """Carrier wavelength lambda = c / f_c in meters."""
        return self.c_mps / self.fc_hz

    @property
    def ref_path_gain_linear(self) -> float:
        """Free-space reference path gain beta_0 = (lambda / (4*pi*d0))^2."""
        import math

        return (self.wavelength_m / (4.0 * math.pi * self.d0_m)) ** 2

    @property
    def noise_power_dbm(self) -> float:
        """Total receiver AWGN power sigma^2 in dBm."""
        import math

        return self.noise_psd_dbm_hz + 10.0 * math.log10(self.bandwidth_hz) + self.noise_figure_db

    @property
    def noise_power_watt(self) -> float:
        """Total receiver AWGN power sigma^2 in linear Watts."""
        return 10.0 ** ((self.noise_power_dbm - 30.0) / 10.0)


@dataclass(frozen=True)
class OptimizerConfig:
    """Hyperparameters for WMMSE baseline and evolutionary/heuristic optimizers."""

    # WMMSE baseline stopping rules
    wmmse_max_iter: int = 100
    wmmse_tol: float = 1e-6

    # Direct encoding (2*M*K real decision variables)
    direct_pop_size: int = 50
    direct_max_iter: int = 250

    # Structure-guided encoding (2*K real variables via uplink-downlink duality)
    struct_pop_size: int = 36
    struct_max_iter: int = 100

    # Differential Evolution (DE/current-to-best/1/bin)
    de_f_min: float = 0.4                 # Dithered mutation scale lower bound
    de_f_max: float = 0.8                 # Dithered mutation scale upper bound
    de_cr: float = 0.85                   # Binomial crossover probability

    # Particle Swarm Optimization (Linearly decreasing inertia Global-Best PSO)
    pso_w_start: float = 0.9              # Initial inertia weight
    pso_w_end: float = 0.4                # Final inertia weight
    pso_c1: float = 1.5                   # Cognitive acceleration coefficient
    pso_c2: float = 1.8                   # Social acceleration coefficient
    pso_v_max_ratio: float = 0.25         # Velocity clamp fraction of search span

    # Real-coded Genetic Algorithm (Tournament + BLX-alpha + Gaussian mutation + Elitism)
    ga_tournament_size: int = 3
    ga_crossover_rate: float = 0.9
    ga_blx_alpha: float = 0.3
    ga_mutation_rate: float = 0.12
    ga_mutation_sigma_start: float = 0.25
    ga_mutation_sigma_end: float = 0.03
    ga_elitism_count: int = 2

    # Early stopping tolerance on best sum-rate improvement (set 0 to disable)
    early_stop_patience: int = 0
    early_stop_tol: float = 1e-6


@dataclass(frozen=True)
class ExperimentConfig:
    """Monte-Carlo sweep ranges and reproducibility settings."""

    seed: int = 2026
    num_trials: int = 100                 # Independent channel realizations per sweep point
    stability_runs: int = 30              # Independent optimizer runs on one fixed channel

    # Sweep grids
    p_max_dbm_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    m_grid: tuple[int, ...] = (4, 8, 12, 16, 24, 32)
    k_grid: tuple[int, ...] = (2, 4, 6, 8, 10)
    rician_k_db_grid: tuple[float, ...] = (-20.0, 0.0, 3.0, 6.0, 10.0, 15.0)

    # Plotting settings
    figure_dpi: int = 300
    save_pdf: bool = True

    system: SystemConfig = field(default_factory=SystemConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


DEFAULT_CONFIG = ExperimentConfig()
