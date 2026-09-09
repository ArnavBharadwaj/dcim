"""Phase 3: train the graph model on hall_a, evaluate it there and on hall_b/hall_c.

Encode, process, decode. Racks and CRAC units as nodes, directed edges following the
air path, three message-passing layers with attention over incoming edges, and a
temperature *delta* target so the model cannot score well by memorising a per-hall
offset.

Transfer is zero-shot: the same weights and the same hall_a normaliser are applied to
hall_b (scale transfer) and hall_c (layout transfer). Nothing is refitted.

    uv run python scripts/phase3_train_gnn.py
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import sys

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# torch must be imported before lightgbm on macOS arm64, and LightGBM must stay
# single-threaded. See the note in src/models/lgbm.py.
import torch  # noqa: E402,F401

from src.data.dataset import WindowSpec, targets                       # noqa: E402
from src.data.generate import TRAINING_ARRAYS, load                    # noqa: E402
from src.data.split import sample_indices, time_split                  # noqa: E402
from src.eval import metrics                                           # noqa: E402
from src.eval.results import log_run                                   # noqa: E402
from src.graph.build import GraphSpec, build_graph                     # noqa: E402
from src.models.gnn_trainer import (TrainConfig, assert_no_renormalisation,  # noqa: E402
                                    fit_normalizer, predict, train)
from src.twin.config import build                                      # noqa: E402


def load_hall(name: str, data_dir: pathlib.Path, graph_spec: GraphSpec):
    """Load one hall, reading only the arrays the models consume.

    Memory is the binding constraint here, not compute. This machine has 8.6 GB of
    RAM; each hall's per-rack arrays are about 145 MB, and an earlier version of this
    script held all three resident for the whole sweep. That pushed training into swap
    and made an epoch take 139 s against the 34 s the same work takes on resident
    tensors -- a four-fold slowdown that looked like a slow model. Transfer halls are
    now loaded one at a time, after training, and freed.
    """
    data, meta = load(data_dir, name, arrays=TRAINING_ARRAYS)
    twin, geom, cfg = build(REPO / "configs" / "hall" / f"{name}.yaml",
                            REPO / "configs" / "twin" / "default.yaml")
    return data, meta, geom, build_graph(geom, graph_spec)


def pick_device(requested: str) -> str:
    """Choose a torch device.

    Defaults to CPU, deliberately, and that is not a fallback. Benchmarked on this
    machine with hall_a's trajectories resident:

        MPS   7697 ms/step  ->  1010 s/epoch
        CPU    110 ms/step  ->    14.5 s/epoch

    a 70x difference. An earlier micro-benchmark with no data loaded put MPS at
    186 ms/step, which is why this ran on MPS at first and an epoch appeared to take
    ten minutes. The machine has 8.6 GB of unified memory with several GB already
    swapped; once a few hundred megabytes of trajectories are resident, MPS buffer
    allocation thrashes. The model is small enough that CPU is simply the right
    device here.
    """
    if requested != "auto":
        return requested
    return "cpu"


def _evaluate(model, data, idx, graph, geom, spec, hs, nz, device, hall, seed,
              cfg, horizon_s, info, tag, src) -> float:
    """Predict, score, log one row, return the RMSE."""
    pred = predict(model, data, idx, graph, geom.crac_flow_share, spec, hs, nz, device)
    true = targets(data, idx, hs)
    last = data["inlet"][idx]
    m = {
        "rmse_k": metrics.rmse(pred, true),
        "mae_k": metrics.mae(pred, true),
        "p95_abs_err_k": metrics.p95_abs_error(pred, true),
        "max_abs_err_k": metrics.max_abs_error(pred, true),
        "hotspot_rmse_k": metrics.hotspot_rmse(last + pred, last + true, 0.9),
        "persistence_rmse_k": metrics.rmse(np.zeros_like(true), true),
    }
    log_run(phase="3", experiment=f"gnn_{tag}", model="gnn", hall=hall, seed=seed,
            config={**cfg, "horizon_s": horizon_s},
            metrics={**m, **info, "horizon_s": horizon_s, "source_hall": src,
                     "n_eval_snapshots": int(idx.size),
                     "normalizer_fitted_on": nz.fitted_on},
            notes=f"zero-shot from {src}" if tag == "transfer" else "")
    return m["rmse_k"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/phase3.yaml")
    ap.add_argument("--data", default="data/trajectories")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--horizons", default=None)
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--device", default="auto",
                    help="auto (=cpu, see pick_device), cpu, or mps")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text())
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.horizons:
        cfg["horizons_s"] = [int(h) for h in args.horizons.split(",")]

    device = pick_device(args.device)
    gspec = GraphSpec(**cfg["graph"])
    data_dir = REPO / args.data
    src = cfg["source_hall"]

    s_data, s_meta, s_geom, s_graph = load_hall(src, data_dir, gspec)
    spec = WindowSpec(lags=cfg["lags"], dt_s=s_meta["dt_s"],
                      horizons_s=tuple(cfg["horizons_s"]))
    n_ep = int(s_data["episode_id"].max()) + 1
    split = time_split(n_ep, cfg["split"]["train_frac"], cfg["split"]["val_frac"])
    tcfg = TrainConfig(**cfg["model"], **cfg["train"])

    print(f"device={device}  source={src} ({s_geom.n_racks} racks, "
          f"{s_graph.n_edges} edges)")
    print(f"split train {split.train} val {split.val} test {split.test}\n")

    results: dict[tuple, list[float]] = {}
    trained: dict[tuple, tuple] = {}      # (horizon, seed) -> (model, normaliser)

    # ---- train on the source hall, evaluating there as we go -------------------
    for horizon_s in cfg["horizons_s"]:
        hs = spec.horizon_steps(horizon_s)
        idx_tr = sample_indices(s_data["episode_id"], split, "train", cfg["lags"], hs,
                                stride=cfg["sample_stride"])
        idx_va = sample_indices(s_data["episode_id"], split, "val", cfg["lags"], hs,
                                stride=cfg["sample_stride"])
        idx_te = sample_indices(s_data["episode_id"], split, "test", cfg["lags"], hs,
                                stride=cfg["sample_stride"])

        # Fitted once, on the source hall's training fold only.
        nz = fit_normalizer(s_data, idx_tr, spec, hs, src)
        assert_no_renormalisation(nz, src)

        print(f"--- horizon {horizon_s}s: {idx_tr.size} train / {idx_va.size} val / "
              f"{idx_te.size} test snapshots")

        for seed in cfg["seeds"]:
            model, info = train(s_data, idx_tr, idx_va, s_graph,
                                s_geom.crac_flow_share, spec, hs, nz, tcfg, seed,
                                device, verbose=(seed == cfg["seeds"][0]))
            trained[(horizon_s, seed)] = (model, nz, info)
            r = _evaluate(model, s_data, idx_te, s_graph, s_geom, spec, hs, nz,
                          device, src, seed, cfg, horizon_s, info, "test", src)
            results.setdefault((horizon_s, src), []).append(r)
            print(f"  seed {seed}: {src} {r:.4f}", flush=True)

    # ---- zero-shot transfer, one hall resident at a time -----------------------
    if not args.no_transfer:
        for hall in cfg["transfer_halls"]:
            h_data, h_meta, h_geom, h_graph = load_hall(hall, data_dir, gspec)
            n = int(h_data["episode_id"].max()) + 1
            hsplit = time_split(n, cfg["split"]["train_frac"], cfg["split"]["val_frac"])
            print(f"--- transfer to {hall} ({h_geom.n_racks} racks, "
                  f"{h_graph.n_edges} edges), zero-shot from {src}")
            for horizon_s in cfg["horizons_s"]:
                hs = spec.horizon_steps(horizon_s)
                idx = sample_indices(h_data["episode_id"], hsplit, "test",
                                     cfg["lags"], hs, stride=cfg["sample_stride"])
                for seed in cfg["seeds"]:
                    model, nz, info = trained[(horizon_s, seed)]
                    assert_no_renormalisation(nz, src)
                    r = _evaluate(model, h_data, idx, h_graph, h_geom, spec, hs, nz,
                                  device, hall, seed, cfg, horizon_s, info,
                                  "transfer", src)
                    results.setdefault((horizon_s, hall), []).append(r)
                print(f"  {horizon_s}s: "
                      + " ".join(f"{v:.4f}" for v in results[(horizon_s, hall)]),
                      flush=True)
            del h_data
            gc.collect()

    halls = [src] + ([] if args.no_transfer else list(cfg["transfer_halls"]))
    print("\n" + "=" * 76)
    print(f"PHASE 3 GNN: test RMSE of the temperature delta (K), mean +/- sd over "
          f"{len(cfg['seeds'])} seeds")
    print("trained on " + src + " only; other halls are zero-shot, no refitting")
    print("=" * 76)
    print(f"{'hall':<10}" + "".join(f"{h:>8}s{'':<8}" for h in cfg["horizons_s"]))
    for hall in halls:
        row = f"{hall:<10}"
        for h in cfg["horizons_s"]:
            v = results.get((h, hall), [])
            row += f"{np.mean(v):>9.4f}+/-{np.std(v):<6.4f}" if v else " " * 17
        print(row)


if __name__ == "__main__":
    main()
