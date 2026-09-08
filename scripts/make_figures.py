"""Build the results figures from results/runs.csv.

Like `make_tables.py`, this only reads the CSV -- no metric is recomputed here, so a
figure can never disagree with the table beside it.

    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.make_tables import latest_runs   # noqa: E402


def agg(df, metric, keys=("model", "horizon_s")):
    g = df.groupby(list(keys), dropna=False)[metric].agg(["mean", "std", "count"])
    return g.reset_index().fillna({"std": 0.0})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = latest_runs(pd.read_csv(REPO / args.runs))
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Phase 2/3 accuracy comparison ----------------
    acc = df[df.phase.astype(str).isin(["2", "3"])]
    acc = acc[acc.experiment.isin(["baseline", "gnn_test"])]
    if not acc.empty:
        horizons = sorted(acc.horizon_s.dropna().unique())
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, metric, label in (
                (axes[0], "rmse_k", "test RMSE of the temperature delta (K)"),
                (axes[1], "hotspot_rmse_k", "hot-spot RMSE (K), hottest 10%")):
            a = agg(acc, metric)
            models = sorted(a.model.unique())
            width = 0.8 / len(models)
            x = np.arange(len(horizons))
            for i, m in enumerate(models):
                sub = a[a.model == m].set_index("horizon_s").reindex(horizons)
                ax.bar(x + i * width - 0.4 + width / 2, sub["mean"], width,
                       yerr=sub["std"], capsize=2.5, label=m,
                       color="#e45756" if m == "gnn" else None)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(h)} s" for h in horizons])
            ax.set_ylabel(label)
            ax.set_xlabel("prediction horizon")
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_title("Accuracy on hall_a")
        axes[1].set_title("Accuracy where it matters")
        axes[1].legend(fontsize=8, ncol=2)
        fig.suptitle("Phases 2-3: baselines and the graph model, "
                     "mean +/- sd over seeds", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = outdir / "phase23_accuracy.png"
        fig.savefig(p, dpi=150)
        print(f"wrote {p.relative_to(REPO)}")

    # ---------------- Phase 3 transfer ----------------
    tr = df[(df.phase.astype(str) == "3")]
    if not tr.empty and tr.hall.nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        a = agg(tr, "rmse_k", keys=("hall", "horizon_s"))
        horizons = sorted(a.horizon_s.dropna().unique())
        halls = sorted(a.hall.unique())
        width = 0.8 / len(halls)
        x = np.arange(len(horizons))
        for i, h in enumerate(halls):
            sub = a[a.hall == h].set_index("horizon_s").reindex(horizons)
            ax.bar(x + i * width - 0.4 + width / 2, sub["mean"], width,
                   yerr=sub["std"], capsize=3, label=h)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(h)} s" for h in horizons])
        ax.set_ylabel("test RMSE of the temperature delta (K)")
        ax.set_xlabel("prediction horizon")
        ax.set_title("Phase 3 zero-shot transfer: trained on hall_a only,\n"
                     "same weights and same normaliser applied to hall_b and hall_c")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        p = outdir / "phase3_transfer.png"
        fig.savefig(p, dpi=150)
        print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
