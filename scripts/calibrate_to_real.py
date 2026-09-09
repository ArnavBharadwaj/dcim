"""Calibrate the recirculation kernel against measured spatial statistics.

Targets, none of which came from us:

  mean pairwise rack-rack temperature correlation   0.58    NYCU, across racks
  fraction of anti-correlated pairs                 7.7%    NYCU
  rack spread at identical load                     4.02 K  ThermoFOAM CFD, C00

The previous calibration used published *ranges* for each parameter independently and
never checked the joint behaviour. It produced 0.93 / 0.0% / 2.47 K.

    uv run python scripts/calibrate_to_real.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.twin.geometry import build_hall            # noqa: E402
from src.twin.recirculation import RecircParams     # noqa: E402
from src.twin.thermal import ThermalTwin            # noqa: E402

TARGET = {"mean_r": 0.58, "frac_neg": 0.077, "uniform_spread_K": 4.02}


def statistics(geom, params, n_points=36, seed=0):
    """Correlation structure and spread over a spread of operating points."""
    rng = np.random.default_rng(seed)
    n = geom.n_racks
    twin = ThermalTwin(geom, recirc=params)
    util = np.clip(rng.normal(62, 30, (n_points, n)), 0, 100)
    # Concentrate load in a random block for a third of the points, as the placement
    # policies do; correlation measured only over uniform load is meaningless.
    for k in range(0, n_points, 3):
        s = rng.integers(0, n - 40)
        util[k] = rng.uniform(0, 25, n)
        util[k, s:s + 40] = rng.uniform(70, 100, 40)
    sup = rng.uniform(16, 21, n_points)
    fan = rng.uniform(0.55, 1.0, n_points)

    T = []
    for k in range(n_points):
        try:
            T.append(twin.steady_state(util[k], float(sup[k]), float(fan[k])))
        except RuntimeError:
            return None
    T = np.asarray(T)
    C = np.corrcoef(T.T)
    r = C[np.triu_indices(n, 1)]
    r = r[np.isfinite(r)]
    uni = twin.steady_state(np.full(n, 100.0), 20.0, 1.0)
    return {
        "mean_r": float(np.mean(r)),
        "frac_neg": float((r < 0).mean()),
        "uniform_spread_K": float(uni.max() - uni.min()),
        "loaded_spread_K": float(np.mean(T.max(1) - T.min(1))),
        "mean_temp_C": float(T.mean()),
    }


def score(s):
    if s is None:
        return float("inf")
    return (abs(s["mean_r"] - TARGET["mean_r"]) / 0.15
            + abs(s["frac_neg"] - TARGET["frac_neg"]) / 0.04
            + abs(s["uniform_spread_K"] - TARGET["uniform_spread_K"]) / 1.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", default="configs/hall/hall_a.yaml")
    ap.add_argument("--out", default="results/calibration_to_real.json")
    args = ap.parse_args()
    geom = build_hall(yaml.safe_load((REPO / args.hall).read_text()))

    grid = {
        "decay_length_m": (1.2, 2.0, 3.0),
        "escape_base": (0.30, 0.40),
        "competition_length_m": (1.5, 3.0, 5.0),
        "decay_flow_exponent": (0.6, 1.4, 2.4),
        "asymmetry_flow_gain": (0.6, 1.6),
    }
    keys = list(grid)
    print(f"{'decay':>6}{'esc':>6}{'comp':>6}{'dfe':>6}{'afg':>6}"
          f"{'mean r':>9}{'neg%':>7}{'unif K':>8}{'load K':>8}{'score':>8}")
    results, best = [], (float("inf"), None, None)
    for combo in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, combo))
        try:
            p = RecircParams(**kw)
        except ValueError:
            continue
        s = statistics(geom, p)
        sc = score(s)
        results.append({**kw, **(s or {}), "score": sc})
        if s is None:
            print(f"{combo[0]:6.1f}{combo[1]:6.2f}{combo[2]:6.1f}{combo[3]:6.1f}"
                  f"{combo[4]:6.1f}   runaway")
            continue
        print(f"{combo[0]:6.1f}{combo[1]:6.2f}{combo[2]:6.1f}{combo[3]:6.1f}"
              f"{combo[4]:6.1f}{s['mean_r']:9.3f}{100*s['frac_neg']:7.1f}"
              f"{s['uniform_spread_K']:8.2f}{s['loaded_spread_K']:8.2f}{sc:8.2f}",
              flush=True)
        if sc < best[0]:
            best = (sc, kw, s)

    print("\nTARGET                    "
          f"{TARGET['mean_r']:9.3f}{100*TARGET['frac_neg']:7.1f}"
          f"{TARGET['uniform_spread_K']:8.2f}      (NYCU / ThermoFOAM)")
    if best[1]:
        print(f"\nbest: {best[1]}")
        print(f"      {best[2]}")
    (REPO / args.out).write_text(json.dumps(
        {"target": TARGET, "best": {"params": best[1], "stats": best[2]},
         "grid": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
