"""Prediction and operational metrics.

Kept in one place so every phase reports the same numbers the same way, and so the
columns in results/runs.csv have a single definition.
"""

from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def max_abs_error(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(pred) - np.asarray(true))))


def p95_abs_error(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.percentile(np.abs(np.asarray(pred) - np.asarray(true)), 95))


def hotspot_rmse(pred: np.ndarray, true: np.ndarray, quantile: float = 0.9) -> float:
    """RMSE restricted to the hottest samples.

    Mean error over a whole hall is dominated by racks that never get near a limit.
    What a controller needs is accuracy where the temperature is high, so this is
    reported alongside plain RMSE rather than instead of it.
    """
    true = np.asarray(true)
    cut = np.quantile(true, quantile)
    m = true >= cut
    return rmse(np.asarray(pred)[m], true[m]) if m.any() else float("nan")


def violation_kelvin_seconds(inlet_temp_c: np.ndarray, limit_c: float,
                             dt_s: float) -> float:
    """Integrated thermal violation, in kelvin-seconds, over an (T, N) trajectory.

    The operational cost of running hot. Reported with cooling energy and job
    completion time, never alone.
    """
    excess = np.clip(np.asarray(inlet_temp_c) - limit_c, 0.0, None)
    return float(excess.sum() * dt_s)


def breach_fraction(inlet_temp_c: np.ndarray, limit_c: float) -> float:
    """Fraction of timesteps where any rack exceeds the limit.

    This is the quantity the Phase 1 generation gate is stated in: the brief asks for
    at least 2% of timesteps breaching before the dataset is usable.
    """
    arr = np.asarray(inlet_temp_c)
    return float(np.mean(np.any(arr > limit_c, axis=1)))


def rack_breach_fraction(inlet_temp_c: np.ndarray, limit_c: float) -> float:
    """Fraction of (timestep, rack) pairs above the limit."""
    return float(np.mean(np.asarray(inlet_temp_c) > limit_c))
