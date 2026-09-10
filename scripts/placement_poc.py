"""Proof of concept: what thermal-aware placement buys, measured in the twin.

The same total load is placed five ways on a hall and each result is solved to steady
state:

  round_robin    spread evenly across every rack
  random         each chunk to a random rack with room
  packed         fill racks in index order, as consolidation / best-fit does
  heat_map       each chunk to the rack that is coolest right now -- a heat map with no
                 model of where the added heat will go
  model_guided   each chunk to the rack whose extra load raises the hottest inlet least,
                 predicted with the twin's influence matrix. This is the job the graph
                 model is meant to do; here the twin plays it, so this is the upper
                 bound on what a perfect model could get.

Each placement is scored three ways: peak rack inlet at fixed cooling, the warmest
supply air that keeps every rack at or below the 27 C limit, and the lowest CRAC fan
speed that does, with fan power by the cube law.

    uv run python scripts/placement_poc.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.twin.config import build        # noqa: E402
from src.twin.power import rack_power    # noqa: E402

LIMIT_C = 27.0                    # ASHRAE A1 recommended maximum
REF_SUPPLY_C, REF_FAN = 17.0, 0.9
CHUNK = 12.5                      # % of one rack per placement decision: 8 of 64 GPU slots
REFRESH = 20                      # re-solve the operating point every this many chunks
POLICIES = ("round_robin", "random", "packed", "heat_map", "model_guided")


def solve(twin, util, supply=REF_SUPPLY_C, fan=REF_FAN):
    return twin.steady_state(util, supply, fan, tol=1e-6)


def place(twin, n, n_chunks, policy, rng):
    util = np.zeros(n)
    if policy == "round_robin":
        for c in range(n_chunks):
            util[c % n] += CHUNK
        return util
    if policy == "packed":
        for _ in range(n_chunks):
            util[np.flatnonzero(util < 100)[0]] += CHUNK
        return util
    if policy == "random":
        for _ in range(n_chunks):
            util[rng.choice(np.flatnonzero(util < 100))] += CHUNK
        return util

    # heat_map and model_guided both track temperatures as load lands. They differ only
    # in whether they look at where the added heat will go.
    for c in range(n_chunks):
        if c % REFRESH == 0:
            T = solve(twin, util)
            D = twin.influence_matrix(util, T, REF_FAN)       # T = T_supply + D @ P
            p0 = rack_power(util, T, twin.power).total_power_w
            dP = rack_power(np.minimum(util + CHUNK, 100), T, twin.power).total_power_w - p0
        cand = np.flatnonzero(util < 100)
        if policy == "heat_map":
            i = cand[np.argmin(T[cand])]
        else:
            peak = (T[:, None] + D[:, cand] * dP[cand][None, :]).max(axis=0)
            i = cand[np.argmin(peak)]
        util[i] += CHUNK
        T = T + D[:, i] * dP[i]
    return util


def _ok(twin, util, supply, fan):
    try:
        return solve(twin, util, supply, fan).max() <= LIMIT_C
    except RuntimeError:          # thermal runaway counts as a violation
        return False


def score(twin, util):
    T = solve(twin, util)
    out = {"peak_c": float(T.max()), "mean_c": float(T.mean()),
           "spread_k": float(T.max() - T.min())}
    lo, hi = 10.0, 30.0           # warmest supply air at the reference fan speed
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _ok(twin, util, mid, REF_FAN) else (lo, mid)
    out["max_supply_c"] = lo
    if not _ok(twin, util, REF_SUPPLY_C, 1.0):
        out["min_fan"] = out["fan_power_rel"] = float("nan")
        return out
    lo, hi = 0.3, 1.0             # slowest fans at the reference supply temperature
    for _ in range(10):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if _ok(twin, util, REF_SUPPLY_C, mid) else (mid, hi)
    out["min_fan"] = hi
    out["fan_power_rel"] = hi ** 3    # fan affinity law, relative to full speed
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halls", nargs="+", default=["hall_a", "hall_c"])
    ap.add_argument("--util", type=float, nargs="+", default=[70.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/placement_poc.json")
    args = ap.parse_args()

    rows = []
    for hall in args.halls:
        twin, geom, _ = build(REPO / "configs" / "hall" / f"{hall}.yaml",
                              REPO / "configs" / "twin" / "default.yaml")
        n = geom.n_racks
        for u in args.util:
            n_chunks = int(round(n * u / CHUNK))
            print(f"\n{hall}: {n} racks, mean utilisation {u:.0f}%, "
                  f"reference cooling {REF_SUPPLY_C} C / fan {REF_FAN}, limit {LIMIT_C} C")
            print(f"  {'policy':14s}{'peak C':>8}{'spread K':>10}{'max supply C':>14}"
                  f"{'min fan':>9}{'fan power':>11}{'secs':>7}", flush=True)
            for pol in POLICIES:
                t0 = time.time()
                util = place(twin, n, n_chunks, pol, np.random.default_rng(args.seed))
                s = score(twin, util)
                rows.append({"hall": hall, "mean_util": u, "policy": pol, **s,
                             "util": util.tolist()})
                print(f"  {pol:14s}{s['peak_c']:8.2f}{s['spread_k']:10.2f}"
                      f"{s['max_supply_c']:14.2f}{s['min_fan']:9.2f}"
                      f"{s['fan_power_rel']:11.2f}{time.time() - t0:7.1f}", flush=True)
            by = {r["policy"]: r for r in rows if r["hall"] == hall and r["mean_util"] == u}
            rr, mg = by["round_robin"], by["model_guided"]
            print(f"  model_guided vs round_robin: peak {mg['peak_c'] - rr['peak_c']:+.2f} K, "
                  f"supply air can run {mg['max_supply_c'] - rr['max_supply_c']:+.2f} C warmer, "
                  f"fan power {100 * (mg['fan_power_rel'] / rr['fan_power_rel'] - 1):+.0f}%")

    out = REPO / args.out
    out.write_text(json.dumps({"limit_c": LIMIT_C, "ref_supply_c": REF_SUPPLY_C,
                               "ref_fan": REF_FAN, "chunk_pct": CHUNK, "rows": rows},
                              indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
