"""Build the paper's results tables from results/runs.csv.

From the brief: tables come from the CSV, not from console output. Nothing here
recomputes a metric -- it only groups, aggregates and formats what the runs recorded.

runs.csv is append-only and accumulates debugging runs alongside real ones, so by
default only the most recent config_hash per (phase, experiment, model, hall, horizon)
is used, and any group with fewer seeds than requested is flagged rather than quietly
averaged over whatever happens to be there.

    uv run python scripts/make_tables.py
    uv run python scripts/make_tables.py --markdown > results/tables.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

KEY = ["phase", "experiment", "model", "hall", "horizon_s"]


def latest_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows from the most recent config_hash for each group."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    keep = []
    for _, g in df.groupby(KEY, dropna=False):
        newest = g.sort_values("timestamp").iloc[-1]["config_hash"]
        keep.append(g[g["config_hash"] == newest])
    return pd.concat(keep) if keep else df


def agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = (df.groupby(KEY, dropna=False)[metric]
             .agg(["mean", "std", "count"]).reset_index())
    out["std"] = out["std"].fillna(0.0)
    return out


def fmt(row, digits: int = 4) -> str:
    return f"{row['mean']:.{digits}f} +/- {row['std']:.{digits}f}"


def table(df: pd.DataFrame, metric: str, expected_seeds: int,
          markdown: bool) -> str:
    a = agg(df, metric)
    if a.empty:
        return "(no rows)\n"
    horizons = sorted(a["horizon_s"].dropna().unique())
    rows, warn = [], []
    for (model, hall), g in a.groupby(["model", "hall"], dropna=False):
        cells = []
        for h in horizons:
            m = g[g["horizon_s"] == h]
            if m.empty:
                cells.append("--")
                continue
            r = m.iloc[0]
            cells.append(fmt(r))
            if r["count"] != expected_seeds:
                warn.append(f"{model}/{hall}/{int(h)}s: {int(r['count'])} seeds")
        rows.append((model, hall, cells))
    rows.sort()

    lines = []
    if markdown:
        lines.append("| model | hall | " +
                     " | ".join(f"{int(h)} s" for h in horizons) + " |")
        lines.append("|---|---|" + "---|" * len(horizons))
        for model, hall, cells in rows:
            lines.append(f"| {model} | {hall} | " + " | ".join(cells) + " |")
    else:
        head = f"{'model':<16}{'hall':<9}" + "".join(
            f"{str(int(h)) + 's':>20}" for h in horizons)
        lines.append(head)
        lines.append("-" * len(head))
        for model, hall, cells in rows:
            lines.append(f"{model:<16}{hall:<9}" +
                         "".join(f"{c:>20}" for c in cells))
    if warn:
        lines.append("")
        lines.append("incomplete seed counts: " + "; ".join(sorted(set(warn))))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--seeds", type=int, default=5,
                    help="expected seeds per cell; groups with fewer are flagged")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--all-runs", action="store_true",
                    help="include superseded config hashes")
    args = ap.parse_args()

    path = REPO / args.runs
    if not path.exists():
        raise SystemExit(f"{path} not found; run a phase script first")
    df = pd.read_csv(path)
    if not args.all_runs:
        df = latest_runs(df)

    sections = [
        ("Phase 1: data generation", df[df.phase.astype(str) == "1"], None),
        ("Phase 2: baselines, test RMSE of the temperature delta (K)",
         df[(df.phase.astype(str) == "2")], "rmse_k"),
        ("Phase 2: hot-spot RMSE (K), hottest 10% of test samples",
         df[(df.phase.astype(str) == "2")], "hotspot_rmse_k"),
        ("Phase 3: graph model, test RMSE of the temperature delta (K)",
         df[(df.phase.astype(str) == "3")], "rmse_k"),
        ("Phase 3: hot-spot RMSE (K)",
         df[(df.phase.astype(str) == "3")], "hotspot_rmse_k"),
    ]

    for title, sub, metric in sections:
        if sub.empty:
            continue
        print(f"\n## {title}\n" if args.markdown else f"\n{title}\n{'=' * len(title)}")
        if metric is None:
            cols = ["hall", "seed", "n_timesteps", "n_racks", "util_mean_pct",
                    "inlet_mean_c", "inlet_max_c", "breach_frac_recommended",
                    "breach_frac_allowable", "gate_pass"]
            g = sub[[c for c in cols if c in sub.columns]].sort_values("hall")
            print(g.to_markdown(index=False, floatfmt=".3f") if args.markdown
                  else g.to_string(index=False))
        else:
            print(table(sub, metric, args.seeds, args.markdown))

    n_sha = df["git_sha"].nunique()
    dirty = int(df["git_dirty"].sum())
    print(f"\n{len(df)} runs, {n_sha} distinct git SHA(s), {dirty} recorded with a "
          f"dirty working tree.")


if __name__ == "__main__":
    main()
