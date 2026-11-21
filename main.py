"""Single entry point to reproduce all BTL2 Topic 1 simulations and report figures.

Usage (Windows PowerShell):
    $env:MPLBACKEND = "Agg"; python "btl2\\main.py"
    $env:MPLBACKEND = "Agg"; python "btl2\\main.py" --quick   # Fast smoke run
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, DEFAULT_CONFIG, FIGURES_DIR
from plot_results import plot_all_figures
from src.experiments import run_all_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTL2 Topic 1: MU-MISO Beamforming Optimization using Evolutionary/Heuristic Algorithms."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced-trial smoke simulation (15 Monte-Carlo trials, 10 stability runs).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override number of independent Monte-Carlo channel realizations per sweep point.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip simulation and regenerate plots from cached results/data files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DEFAULT_CONFIG

    if args.quick:
        cfg = replace(cfg, num_trials=15, stability_runs=10)
    if args.trials is not None:
        cfg = replace(cfg, num_trials=int(args.trials))

    if not args.plot_only:
        run_all_experiments(cfg=cfg, data_dir=DATA_DIR)

    plot_all_figures(data_dir=DATA_DIR, fig_dir=FIGURES_DIR, cfg=cfg)


if __name__ == "__main__":
    main()
