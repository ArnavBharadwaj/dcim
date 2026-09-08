"""Gradient boosting on per-rack features with k-nearest-neighbour context.

The real competitor. From the brief: build it before the graph model, because if
boosting is already close to the ceiling the paper has to be reshaped around transfer
rather than accuracy.

One pooled model over all rack-timesteps rather than 200 separate per-rack models. The
features are per-rack and rack-relative -- own load and temperature history, the load
and temperature of the k nearest racks, the plant setpoints -- so a pooled model sees
every rack's situation and carries no rack identity. Fitting 200 separate models would
give each one 1/200th of the data, make the k-neighbour features far weaker, and
prevent the model from transferring to a hall with a different rack count at all.
"""

from __future__ import annotations

import numpy as np


class LightGBMRegressor:
    name = "lightgbm"

    # Single-threaded on purpose. LightGBM and torch each ship an OpenMP runtime, and
    # on macOS arm64 they cannot both use a thread pool in one process. Measured here:
    #
    #   import torch, then lightgbm, n_jobs=-1  -> segfault, exit 139, no traceback
    #   import torch, then lightgbm, n_jobs=1   -> fine
    #   import lightgbm, then torch, any n_jobs -> lightgbm fine, torch_geometric
    #                                              then segfaults instead
    #
    # Under a shell pipeline the crash is invisible: the pipe's status is reported, so
    # the run looks successful and silently produces nothing. Importing torch first and
    # keeping LightGBM to one thread is the only combination where both libraries work,
    # and a slower fit is a fair price for not having to trust a crashing process.
    N_JOBS = 1

    def __init__(self, k_neighbours: int, seed: int = 0, n_estimators: int = 400,
                 learning_rate: float = 0.05, num_leaves: int = 63,
                 min_child_samples: int = 40, subsample: float = 0.8,
                 colsample_bytree: float = 0.8):
        self.k_neighbours = k_neighbours
        self.params = dict(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples,
            subsample=subsample, subsample_freq=1, colsample_bytree=colsample_bytree,
            random_state=seed, n_jobs=self.N_JOBS, verbose=-1)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: tuple[np.ndarray, np.ndarray] | None = None,
            early_stopping_rounds: int = 40) -> "LightGBMRegressor":
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(**self.params)
        kwargs = {}
        if eval_set is not None:
            kwargs["eval_set"] = [eval_set]
            kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds,
                                                      verbose=False)]
        self.model.fit(X, y, **kwargs)
        return self

    def predict_delta(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() before predict_delta()")
        return self.model.predict(X)

    @property
    def n_parameters(self) -> int:
        """Number of leaves across the ensemble -- the honest size comparison."""
        if self.model is None:
            return 0
        return int(self.model.booster_.trees_to_dataframe().shape[0])
