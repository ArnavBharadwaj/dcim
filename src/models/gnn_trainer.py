"""Training and evaluation for the graph model.

Factored out of the Phase 3 script because the Phase 6 ablations run exactly the same
loop with one thing changed -- layer count, aggregation, edge features, edge shuffling
-- and a second copy of the loop would let the ablations drift away from the headline
number they are supposed to be compared against.

The normalisation contract is enforced here rather than trusted: `fit_normalizer` is
called once on the source hall's training fold, and `evaluate` refuses to run with a
normaliser fitted on a different hall than the one it was trained on. Recomputing
statistics on hall_b or hall_c is leakage and it inflates transfer numbers.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import Normalizer, WindowSpec, feature_names, node_features, targets
from ..graph.build import HallGraph
from .gnn import ThermalGNN


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    hidden: int = 96
    layers: int = 3
    heads: int = 4
    aggregation: str = "attention"
    use_edge_features: bool = True
    dropout: float = 0.05
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 6


def crac_features(data: dict, t_idx: np.ndarray, graph: HallGraph,
                  flow_share: np.ndarray) -> np.ndarray:
    """(S, C, 3) features for the CRAC nodes: setpoint, fan speed, duty share.

    No identity and no position, same rule as the rack nodes. Flow share is a property
    of the unit's duty in the hall, which an operator knows and which differs between
    halls, so it transfers as a number rather than as a label.
    """
    s = t_idx.size
    sup = np.repeat(data["supply_temp"][t_idx][:, None], graph.n_cracs, axis=1)
    fan = np.repeat(data["fan_frac"][t_idx][:, None], graph.n_cracs, axis=1)
    share = np.repeat(flow_share[None, :], s, axis=0)
    return np.stack([sup, fan, share], axis=2).astype(np.float32)


def fit_normalizer(data: dict, idx: np.ndarray, spec: WindowSpec,
                   horizon_steps: int, hall: str) -> Normalizer:
    """Fit on one hall's training fold. Call once, then carry it everywhere."""
    f = node_features(data, idx, spec, horizon_steps=horizon_steps)
    f = f.reshape(-1, len(feature_names(spec.lags)))
    y = targets(data, idx, horizon_steps).reshape(-1)
    return Normalizer.fit(f, y, hall, feature_names(spec.lags))


def _batch(data, idx, spec, horizon_steps, graph, flow_share, nz, device):
    rx = node_features(data, idx, spec, horizon_steps=horizon_steps)
    rx = nz.transform(rx.reshape(-1, rx.shape[-1])).reshape(rx.shape).astype(np.float32)
    cx = crac_features(data, idx, graph, flow_share)
    y = targets(data, idx, horizon_steps)
    return (torch.from_numpy(rx).to(device),
            torch.from_numpy(cx).to(device),
            torch.from_numpy(nz.transform_target(y).astype(np.float32)).to(device))


def train(data: dict, train_idx: np.ndarray, val_idx: np.ndarray, graph: HallGraph,
          flow_share: np.ndarray, spec: WindowSpec, horizon_steps: int,
          nz: Normalizer, cfg: TrainConfig, seed: int, device: str,
          verbose: bool = True) -> tuple[ThermalGNN, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = ThermalGNN(
        rack_features=len(feature_names(spec.lags)), crac_features=3,
        edge_dim=graph.edge_dim, hidden=cfg.hidden, layers=cfg.layers,
        heads=cfg.heads, aggregation=cfg.aggregation,
        use_edge_features=cfg.use_edge_features, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    ei = torch.from_numpy(graph.edge_index).to(device)
    ea = torch.from_numpy(graph.edge_attr).to(device)

    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(train_idx.size)
        total = 0.0
        for b in range(0, order.size - cfg.batch_size + 1, cfg.batch_size):
            sel = train_idx[order[b:b + cfg.batch_size]]
            rx, cx, y = _batch(data, sel, spec, horizon_steps, graph, flow_share,
                               nz, device)
            opt.zero_grad()
            loss = F.smooth_l1_loss(model(rx, cx, ei, ea), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss)
        sched.step()

        val = _val_loss(model, data, val_idx, spec, horizon_steps, graph,
                        flow_share, nz, cfg, device, ei, ea)
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            print(f"    epoch {epoch:3d}  train {total / max(1, order.size // cfg.batch_size):.5f}"
                  f"  val {val:.5f}{'  *' if bad == 0 else ''}", flush=True)
        if bad >= cfg.patience:
            if verbose:
                print(f"    early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best, "train_seconds": time.time() - t0,
                   "n_parameters": model.n_parameters}


@torch.no_grad()
def _val_loss(model, data, idx, spec, horizon_steps, graph, flow_share, nz, cfg,
              device, ei, ea) -> float:
    model.eval()
    total, n = 0.0, 0
    for b in range(0, idx.size - cfg.batch_size + 1, cfg.batch_size):
        sel = idx[b:b + cfg.batch_size]
        rx, cx, y = _batch(data, sel, spec, horizon_steps, graph, flow_share, nz, device)
        total += float(F.smooth_l1_loss(model(rx, cx, ei, ea), y)) * sel.size
        n += sel.size
    return total / max(1, n)


@torch.no_grad()
def predict(model: ThermalGNN, data: dict, idx: np.ndarray, graph: HallGraph,
            flow_share: np.ndarray, spec: WindowSpec, horizon_steps: int,
            nz: Normalizer, device: str, batch_size: int = 32) -> np.ndarray:
    """Predicted temperature delta, in kelvin, shape (S, N).

    The normaliser is the one fitted on the source hall. When this is called on
    hall_b or hall_c its statistics are *not* recomputed -- that is the whole point,
    and `evaluate` asserts it.
    """
    model.eval()
    ei = torch.from_numpy(graph.edge_index).to(device)
    ea = torch.from_numpy(graph.edge_attr).to(device)
    out = []
    for b in range(0, idx.size, batch_size):
        sel = idx[b:b + batch_size]
        rx, cx, _ = _batch(data, sel, spec, horizon_steps, graph, flow_share, nz, device)
        out.append(model(rx, cx, ei, ea).cpu().numpy())
    return nz.inverse_target(np.concatenate(out))


def assert_no_renormalisation(nz: Normalizer, source_hall: str) -> None:
    """Guard the transfer contract. Cheap, and it has to hold on every path."""
    if nz.fitted_on != source_hall:
        raise RuntimeError(
            f"normaliser was fitted on {nz.fitted_on!r} but the model was trained on "
            f"{source_hall!r}. Recomputing statistics on the target hall is leakage "
            "and inflates transfer numbers.")
