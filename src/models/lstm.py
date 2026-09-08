"""Per-rack LSTM with no spatial input. The weak baseline.

From the brief: include it because prior work uses it. Its job is to show what a
sequence model can do from a rack's own history alone, with no knowledge that other
racks exist. The gap between it and the neighbour-aware baselines is the measurement
of how much spatial information matters at all.

"Per-rack" means the model sees one rack's sequence at a time and never sees another
rack's state. One shared set of weights is used across racks rather than 200 separate
models: separate models would each get 1/200th of the data and the comparison would be
about data volume rather than about spatial information. The model still carries no
rack identity, so nothing about which rack it is enters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PerRackLSTM(nn.Module):
    name = "lstm"

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L, F) one rack's history. Returns (B,) predicted delta."""
        out, _ = self.lstm(seq)
        return self.head(out[:, -1]).squeeze(-1)


def sequence_features(data: dict, t_idx: np.ndarray, lags: int,
                      horizon_steps: int | None = None) -> np.ndarray:
    """(S, N, L, F) per-rack sequences, oldest step first.

    Strictly local: utilisation, inlet temperature and power for the rack itself, plus
    the plant setpoints. No neighbour information of any kind -- that is the point of
    this baseline.

    The control plan over the horizon is appended as constant channels so this
    baseline sees exactly the same plant information as every other model. Giving it
    less would make the comparison a statement about the inputs rather than about
    spatial structure.
    """
    from ..data.dataset import control_plan

    util, inlet, power = data["util"], data["inlet"], data["power"]
    supply, fan = data["supply_temp"], data["fan_frac"]
    n = util.shape[1]
    plan = (control_plan(data, t_idx, horizon_steps)
            if horizon_steps is not None else None)
    steps = []
    for i in range(lags - 1, -1, -1):          # oldest first
        idx = t_idx - i
        chans = [util[idx], inlet[idx], power[idx] / 1e3,
                 np.repeat(supply[idx][:, None], n, axis=1),
                 np.repeat(fan[idx][:, None], n, axis=1)]
        if plan is not None:
            chans += [np.repeat(plan[:, j][:, None], n, axis=1)
                      for j in range(plan.shape[1])]
        steps.append(np.stack(chans, axis=2))
    return np.stack(steps, axis=2).astype(np.float32)
