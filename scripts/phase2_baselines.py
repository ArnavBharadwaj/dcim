"""Phase 2: the four baselines, before the graph model.

From the brief, in this order and for this reason: if gradient boosting is already
close to the ceiling, that has to be known now, so the paper can be reshaped around
transfer rather than accuracy.

    1. LightGBM on per-rack features with k-nearest-neighbour context (the competitor)
    2. Fitted RC network, one conductance per graph edge, one capacitance per rack
    3. Per-rack LSTM with no spatial input (the weak baseline)
    4. Persistence (calibrates the reader)

Reported at 30 s, 60 s and 300 s, five seeds each, mean and standard deviation --
never a single best run. Every fit appends a row to results/runs.csv.

    uv run python scripts/phase2_baselines.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# torch must be imported before lightgbm on macOS arm64, and LightGBM must stay
# single-threaded. See the note in src/models/lgbm.py.
import torch  # noqa: E402,F401

from src.data.dataset import (Normalizer, WindowSpec, feature_names, flatten,  # noqa: E402
                              neighbour_index, node_features, targets)
from src.data.generate import load                                       # noqa: E402
from src.data.split import sample_indices, time_split                    # noqa: E402
from src.eval import metrics                                             # noqa: E402
from src.eval.results import log_run                                     # noqa: E402
from src.graph.build import build_graph                                  # noqa: E402
from src.models.lgbm import LightGBMRegressor                            # noqa: E402
from src.models.persistence import Persistence                           # noqa: E402
from src.models.rc import RCNetwork                                      # noqa: E402
from src.twin.config import build                                        # noqa: E402


def subsample(idx: np.ndarray, seed: int, max_rows: int, n_racks: int) -> np.ndarray:
    """Pick training timesteps for one seed.

    The seed changes which timesteps a model trains on as well as its initialisation,
    so the spread across seeds reflects real sensitivity to the training sample rather
    than only to weight initialisation. Deterministic models such as RC and
    persistence would otherwise show exactly zero variance for uninteresting reasons.
    """
    rng = np.random.default_rng(seed)
    budget = max(1, max_rows // n_racks)
    if idx.size <= budget:
        return idx
    return np.sort(rng.choice(idx, size=budget, replace=False))


def evaluate(pred_delta: np.ndarray, true_delta: np.ndarray,
             last_inlet: np.ndarray) -> dict[str, float]:
    """Metrics on the delta and on the reconstructed absolute temperature."""
    abs_pred = last_inlet + pred_delta
    abs_true = last_inlet + true_delta
    return {
        "rmse_k": metrics.rmse(pred_delta, true_delta),
        "mae_k": metrics.mae(pred_delta, true_delta),
        "p95_abs_err_k": metrics.p95_abs_error(pred_delta, true_delta),
        "max_abs_err_k": metrics.max_abs_error(pred_delta, true_delta),
        "hotspot_rmse_k": metrics.hotspot_rmse(abs_pred, abs_true, 0.9),
    }


def run_lstm(train_data, test_data, cfg, seed, horizon_steps, lags, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.lstm import PerRackLSTM, sequence_features

    torch.manual_seed(seed)
    (d_tr, idx_tr), (d_te, idx_te) = train_data, test_data
    xs = sequence_features(d_tr, idx_tr, lags, horizon_steps)  # (S, N, L, F)
    ys = targets(d_tr, idx_tr, horizon_steps)
    s, n, L, f = xs.shape
    xs = torch.from_numpy(xs.reshape(s * n, L, f))
    ys = torch.from_numpy(ys.reshape(s * n).astype(np.float32))

    mu, sd = xs.mean((0, 1)), xs.std((0, 1)).clamp_min(1e-6)
    y_mu, y_sd = ys.mean(), ys.std().clamp_min(1e-6)
    xs = (xs - mu) / sd

    model = PerRackLSTM(f, hidden=cfg["hidden"], layers=cfg["layers"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    loader = DataLoader(TensorDataset(xs, (ys - y_mu) / y_sd),
                        batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    model.train()
    for _ in range(cfg["epochs"]):
        for xb, yb in loader:
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()

    model.eval()
    xt = torch.from_numpy(
        sequence_features(d_te, idx_te, lags, horizon_steps).reshape(-1, L, f))
    xt = (xt - mu) / sd
    preds = []
    with torch.no_grad():
        for i in range(0, xt.shape[0], 8192):
            preds.append(model(xt[i:i + 8192].to(device)).cpu())
    pred = (torch.cat(preds) * y_sd + y_mu).numpy()
    return pred, sum(p.numel() for p in model.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/phase2.yaml")
    ap.add_argument("--data", default="data/trajectories")
    ap.add_argument("--models", default="",
                    help="comma-separated subset, e.g. persistence,rc")
    ap.add_argument("--hall", default=None, help="override the hall in the config")
    ap.add_argument("--seeds", default=None, help="comma-separated seed override")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text())
    only = set(args.models.split(",")) if args.models else None
    if args.hall:
        cfg["hall"] = args.hall
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    hall = cfg["hall"]
    lags = cfg["lags"]

    data, meta = load(REPO / args.data, hall)
    twin, geom, twin_cfg = build(REPO / "configs" / "hall" / f"{hall}.yaml",
                                 REPO / "configs" / "twin" / "default.yaml")
    graph = build_graph(geom)
    spec = WindowSpec(lags=lags, dt_s=meta["dt_s"],
                      horizons_s=tuple(cfg["horizons_s"]))
    n_ep = int(data["episode_id"].max()) + 1
    split = time_split(n_ep, cfg["split"]["train_frac"], cfg["split"]["val_frac"])

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"hall={hall}  {data['inlet'].shape[0]:,} timesteps x {geom.n_racks} racks")
    print(f"split: train {split.train}  val {split.val}  test {split.test} (episodes)")
    print(f"device={device}\n")

    results: dict[tuple, list[dict]] = {}

    for horizon_s in cfg["horizons_s"]:
        hs = spec.horizon_steps(horizon_s)
        idx_tr_full = sample_indices(data["episode_id"], split, "train", lags, hs,
                                     stride=cfg["sample_stride"])
        idx_te = sample_indices(data["episode_id"], split, "test", lags, hs,
                                stride=cfg["sample_stride"])
        y_te = targets(data, idx_te, hs)
        last_te = data["inlet"][idx_te]
        # The plant's planned supply over the horizon. A twin is asked "if I set the
        # plant to this, what happens", so the RC baseline gets the horizon-mean
        # setpoint rather than the value at t, matching what the feature-based models
        # receive through the control-plan block.
        offs = np.arange(1, hs + 1)
        sup_te = data["supply_temp"][idx_te[:, None] + offs].mean(axis=1)
        fan_te = data["fan_frac"][idx_te[:, None] + offs].mean(axis=1)
        print(f"--- horizon {horizon_s}s ({hs} steps): "
              f"{idx_tr_full.size:,} train / {idx_te.size:,} test timesteps  "
              f"| target delta std {y_te.std():.4f} K")

        for seed in cfg["seeds"]:
            idx_tr = subsample(idx_tr_full, seed, cfg["max_train_rows"], geom.n_racks)
            y_tr = targets(data, idx_tr, hs)
            sup_tr = data["supply_temp"][idx_tr[:, None] + offs].mean(axis=1)
            fan_tr = data["fan_frac"][idx_tr[:, None] + offs].mean(axis=1)

            def record(model_name, pred, extra=None, elapsed=0.0):
                m = evaluate(pred, y_te, last_te)
                m["fit_seconds"] = elapsed
                m.update(extra or {})
                results.setdefault((horizon_s, model_name), []).append(m)
                log_run(phase="2", experiment="baseline", model=model_name,
                        hall=hall, seed=seed,
                        config={**cfg, "horizon_s": horizon_s, "model": model_name},
                        metrics={**m, "horizon_s": horizon_s,
                                 "n_train_timesteps": int(idx_tr.size),
                                 "n_test_timesteps": int(idx_te.size)},
                        notes=f"lags={lags} stride={cfg['sample_stride']}")

            # ---- persistence -------------------------------------------------
            if only is None or "persistence" in only:
                record("persistence", np.zeros_like(y_te))

            # ---- RC network --------------------------------------------------
            if only is None or "rc" in only:
                t0 = time.time()
                rc = RCNetwork(graph, ridge=cfg["rc"]["ridge"]).fit(
                    data["inlet"][idx_tr], data["power"][idx_tr] / 1e3,
                    sup_tr, fan_tr, y_tr)
                pred = rc.predict_delta(data["inlet"][idx_te],
                                        data["power"][idx_te] / 1e3, sup_te, fan_te)
                record("rc", pred, {"n_parameters": rc.n_parameters},
                       time.time() - t0)

            # ---- LightGBM ----------------------------------------------------
            if only is None or "lightgbm" in only:
                for k in cfg["lightgbm"]["k_neighbours"]:
                    t0 = time.time()
                    nb = neighbour_index(geom, k)
                    Xtr, ytr = flatten(
                        node_features(data, idx_tr, spec, nb, horizon_steps=hs), y_tr)
                    Xte, _ = flatten(
                        node_features(data, idx_te, spec, nb, horizon_steps=hs), y_te)
                    nz = Normalizer.fit(Xtr, ytr, hall, feature_names(lags, k))
                    model = LightGBMRegressor(
                        k, seed=seed,
                        n_estimators=cfg["lightgbm"]["n_estimators"],
                        learning_rate=cfg["lightgbm"]["learning_rate"],
                        num_leaves=cfg["lightgbm"]["num_leaves"])
                    model.fit(nz.transform(Xtr), ytr)
                    pred = model.predict_delta(nz.transform(Xte)).reshape(y_te.shape)
                    record(f"lightgbm_k{k}", pred,
                           {"n_parameters": model.n_parameters,
                            "k_neighbours": k}, time.time() - t0)

            # ---- per-rack LSTM -----------------------------------------------
            if only is None or "lstm" in only:
                t0 = time.time()
                pred, nparam = run_lstm((data, idx_tr), (data, idx_te), cfg["lstm"],
                                        seed, hs, lags, device)
                record("lstm", pred.reshape(y_te.shape), {"n_parameters": nparam},
                       time.time() - t0)

            done = ", ".join(sorted({k[1] for k in results if k[0] == horizon_s}))
            print(f"  seed {seed}: {done}", flush=True)

    # ------------------------------- table -------------------------------
    print("\n" + "=" * 78)
    print(f"PHASE 2 BASELINES on {hall}: test RMSE of the temperature delta (K), "
          f"mean +/- sd over {len(cfg['seeds'])} seeds")
    print("=" * 78)
    models = sorted({k[1] for k in results})
    header = f"{'model':<16}" + "".join(f"{h:>7}s{'':<8}" for h in cfg["horizons_s"])
    print(header)
    for m in models:
        row = f"{m:<16}"
        for h in cfg["horizons_s"]:
            vals = [r["rmse_k"] for r in results.get((h, m), [])]
            row += f"{np.mean(vals):>8.4f}+/-{np.std(vals):<6.4f}" if vals else " " * 16
        print(row)
    print("\nhot-spot RMSE (K), hottest 10% of test samples:")
    print(header)
    for m in models:
        row = f"{m:<16}"
        for h in cfg["horizons_s"]:
            vals = [r["hotspot_rmse_k"] for r in results.get((h, m), [])]
            row += f"{np.mean(vals):>8.4f}+/-{np.std(vals):<6.4f}" if vals else " " * 16
        print(row)
    print(f"\nrows appended to results/runs.csv")


if __name__ == "__main__":
    main()
