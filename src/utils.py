"""Unit-conversion and filesystem helper utilities.

All RF powers inside numerical algorithms are in linear Watts (W) and channel
gains are linear ratios; conversions to/from dB and dBm must go through these
explicit helpers.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def dbm_to_watt(power_dbm: float | np.ndarray) -> float | np.ndarray:
    """Convert power from dBm to linear Watts."""
    return 10.0 ** ((np.asarray(power_dbm, dtype=np.float64) - 30.0) / 10.0) if isinstance(power_dbm, np.ndarray) else 10.0 ** ((float(power_dbm) - 30.0) / 10.0)


def watt_to_dbm(power_watt: float | np.ndarray) -> float | np.ndarray:
    """Convert positive power from linear Watts to dBm."""
    if isinstance(power_watt, np.ndarray):
        return 10.0 * np.log10(np.maximum(power_watt, 1e-30)) + 30.0
    return 10.0 * float(np.log10(max(float(power_watt), 1e-30))) + 30.0


def db_to_linear(val_db: float | np.ndarray) -> float | np.ndarray:
    """Convert logarithmic ratio in dB to linear scale."""
    if isinstance(val_db, np.ndarray):
        return 10.0 ** (val_db / 10.0)
    return 10.0 ** (float(val_db) / 10.0)


def linear_to_db(val_linear: float | np.ndarray) -> float | np.ndarray:
    """Convert positive linear ratio to dB."""
    if isinstance(val_linear, np.ndarray):
        return 10.0 * np.log10(np.maximum(val_linear, 1e-30))
    return 10.0 * float(np.log10(max(float(val_linear), 1e-30)))


def ensure_dirs(*paths: Path) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
