"""Phase 1: generate trajectories for one hall and report the thermal gate.

The gate, from the brief: plot the distribution of rack inlet temperature, and if
fewer than 2% of timesteps breach the thermal limit, increase the load density and
regenerate. This script reports the breach rate and says plainly whether the gate
passed; it does not adjust anything by itself.

    uv run python scripts/phase1_generate.py --hall hall_a --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.generate import GenConfig, generate_hall, save          # noqa: E402
from src.data.trace import load_trace                                 # noqa: E402
from src.eval import metrics                                          # noqa: E402
from src.eval.results import config_hash, log_run                     # noqa: E402
from src.twin.config import build                                     # noqa: E402


def load_gen_config(path: pathlib.Path, overrides: dict) -> GenConfig:
    raw = yaml.safe_load(path.read_text())
    known = {f.name for f in dataclasses.fields(GenConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}")
    raw["policies"] = tuple(raw.get("policies", GenConfig.policies))
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return GenConfig(**raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", default="hall_a")
    ap.add_argument("--twin", default="configs/twin/default.yaml")
    ap.add_argument("--gen", default="configs/data/gen_default.yaml")
    ap.add_argument("--trace", default="data/raw/pai_job_duration_estimate_100K.csv")
    ap.add_argument("--out", default="data/trajectories")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-hours", type=float, default=None,
                    help="override total_hours, for quick checks")
    ap.add_argument("--target-util-min", type=float, default=None)
    ap.add_argument("--target-util-max", type=float, default=None)
    args = ap.parse_args()

    gen = load_gen_config(REPO / args.gen, {
        "total_hours": args.total_hours,
        "target_util_min": args.target_util_min,
        "target_util_max": args.target_util_max,
    })
    hall_path = REPO / "configs" / "hall" / f"{args.hall}.yaml"
    twin, geom, cfg = build(hall_path, REPO / args.twin)
    trace_df = load_trace(REPO / args.trace)

    print(f"hall={geom.name} racks={geom.n_racks} cracs={geom.n_cracs} "
          f"pattern={geom.orientation_pattern}")
    print(f"{gen.n_episodes} episodes x {gen.episode_hours} h at {gen.dt_s:.0f} s "
          f"= {gen.total_hours:.0f} h, seed {args.seed}")

    t0 = time.time()
    data = generate_hall(twin, geom, cfg, gen, trace_df, seed=args.seed)
    elapsed = time.time() - t0

    inlet = data["inlet"]
    lim = cfg.limits
    rec = metrics.breach_fraction(inlet, lim.inlet_recommended_max_c)
    allow = metrics.breach_fraction(inlet, lim.inlet_allowable_max_c)
    rack_rec = metrics.rack_breach_fraction(inlet, lim.inlet_recommended_max_c)

    print(f"\ngenerated in {elapsed / 60:.1f} min: {inlet.shape[0]:,} timesteps "
          f"x {inlet.shape[1]} racks")
    print(f"  utilisation   mean {data['util'].mean():5.1f}%  "
          f"p95 {np.percentile(data['util'], 95):5.1f}%")
    print(f"  inlet temp    min {inlet.min():5.2f}  mean {inlet.mean():5.2f}  "
          f"p99 {np.percentile(inlet, 99):5.2f}  max {inlet.max():5.2f} C")
    print(f"  supply temp   {data['supply_temp'].min():.1f} to "
          f"{data['supply_temp'].max():.1f} C")
    print(f"  CRAC fan      {data['fan_frac'].min():.2f} to "
          f"{data['fan_frac'].max():.2f}")
    print(f"  provisioning  {data['provisioning'].min():.2f} to "
          f"{data['provisioning'].max():.2f}")

    print(f"\nTHERMAL GATE (limit {lim.inlet_recommended_max_c:.0f} C, ASHRAE A1 "
          "recommended):")
    print(f"  timesteps with any rack breaching : {100 * rec:6.2f}%   "
          f"{'PASS' if rec >= 0.02 else 'FAIL, need >= 2.00%'}")
    print(f"  rack-timesteps breaching          : {100 * rack_rec:6.2f}%")
    print(f"  timesteps breaching allowable "
          f"({lim.inlet_allowable_max_c:.0f} C): {100 * allow:6.2f}%")
    if rec < 0.02:
        print("\n  Gate not met. Raise target_util_min/target_util_max in "
              f"{args.gen} and regenerate.")

    meta = {
        "hall": geom.name, "seed": args.seed, "n_racks": geom.n_racks,
        "dt_s": gen.dt_s, "gen_config": dataclasses.asdict(gen),
        "config_hash": config_hash({"gen": dataclasses.asdict(gen),
                                    "hall": args.hall, "twin": args.twin}),
        "breach_fraction_recommended": rec,
        "breach_fraction_allowable": allow,
        "policies": list(gen.policies),
    }
    npz = save(REPO / args.out, geom.name, data, meta)
    print(f"\nwrote {npz.relative_to(REPO)} "
          f"({npz.stat().st_size / 1e6:.0f} MB)")

    log_run(
        phase="1", experiment="generate", hall=geom.name, seed=args.seed,
        config={"gen": dataclasses.asdict(gen), "hall": args.hall},
        metrics={
            "n_timesteps": int(inlet.shape[0]), "n_racks": int(inlet.shape[1]),
            "util_mean_pct": float(data["util"].mean()),
            "inlet_mean_c": float(inlet.mean()), "inlet_max_c": float(inlet.max()),
            "inlet_p99_c": float(np.percentile(inlet, 99)),
            "breach_frac_recommended": rec, "breach_frac_allowable": allow,
            "rack_breach_frac_recommended": rack_rec,
            "gate_pass": int(rec >= 0.02),
            "generation_minutes": elapsed / 60.0,
        },
        notes=f"trace={pathlib.Path(args.trace).name}")


if __name__ == "__main__":
    main()
