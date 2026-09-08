"""Persistence: temperature at t+H equals temperature at t.

From the brief, this is here to calibrate the reader. In SustainDC it was exact,
because the inlet model had no power term and nothing moved unless the setpoint did.
Against the replacement twin it is a real baseline with a real error, and its error
growing with horizon is the first evidence that the horizons measure different things.

In delta form the prediction is simply zero, so there is nothing to fit.
"""

from __future__ import annotations

import numpy as np


class Persistence:
    name = "persistence"

    def fit(self, *args, **kwargs) -> "Persistence":
        return self

    def predict_delta(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features)
        return np.zeros(f.shape[0] if f.ndim > 1 else 1, dtype=np.float64)
