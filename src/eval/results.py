"""One CSV, one row per run.

From the brief: every run is logged to a single CSV with the config hash, seed and git
SHA alongside every metric, and tables are built from that file rather than from
console output. Rows are appended, never rewritten, and the header grows if a later
phase reports a metric earlier phases did not.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
from typing import Any

RUNS_CSV = "results/runs.csv"

BASE_COLUMNS = [
    "run_id", "timestamp", "phase", "experiment", "model", "hall", "seed",
    "config_hash", "git_sha", "git_dirty", "notes",
]


def config_hash(config: Any) -> str:
    """Stable 12-character hash of a config, for grouping runs that share settings."""
    blob = json.dumps(_plain(config), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return obj


def git_state(repo_root: pathlib.Path | None = None) -> tuple[str, bool]:
    """(sha, dirty). Returns ('unknown', True) outside a git checkout, rather than
    raising -- a missing SHA should not stop a run, only be visible in the CSV."""
    root = repo_root or pathlib.Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        return sha, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def log_run(phase: str, experiment: str, metrics: dict[str, Any], *,
            model: str = "", hall: str = "", seed: int | None = None,
            config: Any = None, notes: str = "",
            path: str | pathlib.Path = RUNS_CSV,
            repo_root: pathlib.Path | None = None) -> str:
    """Append one row and return its run_id.

    If `metrics` introduces columns the file does not have, the file is rewritten with
    the widened header and existing rows are preserved with blanks in the new columns.
    """
    root = repo_root or pathlib.Path(__file__).resolve().parents[2]
    csv_path = pathlib.Path(path)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    sha, dirty = git_state(root)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    chash = config_hash(config) if config is not None else ""
    run_id = hashlib.sha256(
        f"{now}{phase}{experiment}{model}{hall}{seed}{chash}".encode()).hexdigest()[:10]

    row: dict[str, Any] = {
        "run_id": run_id, "timestamp": now, "phase": phase,
        "experiment": experiment, "model": model, "hall": hall,
        "seed": "" if seed is None else seed,
        "config_hash": chash, "git_sha": sha, "git_dirty": int(dirty),
        "notes": notes,
    }
    overlap = set(metrics) & set(BASE_COLUMNS)
    if overlap:
        raise ValueError(f"metric name(s) {sorted(overlap)} collide with run metadata")
    row.update(metrics)

    existing_rows: list[dict[str, Any]] = []
    columns = list(BASE_COLUMNS)
    if csv_path.exists():
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            columns = list(reader.fieldnames or BASE_COLUMNS)
            existing_rows = list(reader)

    new_cols = [c for c in row if c not in columns]
    if new_cols and existing_rows:
        columns = columns + new_cols
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns, restval="")
            w.writeheader()
            w.writerows(existing_rows)
            w.writerow(row)
        return run_id

    columns = columns + new_cols
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)
    return run_id
