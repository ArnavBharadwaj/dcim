"""
Phase 0 probe: measure the spatial coupling of SustainDC's rack thermal model.

Question this script answers empirically, without reading the source:
  If I change the power drawn by ONE rack, how does the inlet temperature of
  EVERY OTHER rack respond, as a function of distance?

The script calls SustainDC's thermal kernel directly
(envs.datacenter.DataCenter_ITModel.compute_datacenter_IT_load_outlet_temp)
rather than going through dc_gymenv.step(), because dc_gymenv.step() hardcodes a
single scalar utilisation for every rack:

    ITE_load_pct_list = [self.cpu_load_frac * 100 for i in range(NUM_RACKS)]

so the gym wrapper cannot express a per-rack perturbation at all.

Outputs
  results/phase0_coupling.csv          one row per (source rack, target rack)
  results/figures/phase0_coupling.png  response vs. grid distance
Nothing here is a modelling assumption of ours: every number is read back out of
the simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
SUSTAINDC = REPO / "external" / "dc-rl"
if not SUSTAINDC.exists():
    raise SystemExit(f"SustainDC not found at {SUSTAINDC}; run the Phase 0 clone step first.")
sys.path.insert(0, str(SUSTAINDC))

import envs.datacenter as dc  # noqa: E402
from utils.dc_config_reader import DC_Config  # noqa: E402


def build_model(config_file: str):
    cfg = DC_Config(dc_config_file=config_file)
    model = dc.DataCenter_ITModel(
        num_racks=cfg.NUM_RACKS,
        rack_supply_approach_temp_list=cfg.RACK_SUPPLY_APPROACH_TEMP_LIST,
        rack_CPU_config=cfg.RACK_CPU_CONFIG,
        max_W_per_rack=cfg.MAX_W_PER_RACK,
        DC_ITModel_config=cfg,
    )
    # The constructor zips racks against rack_supply_approach_temp_list, so the
    # number of racks the model actually simulates can be smaller than
    # cfg.NUM_RACKS if the config's approach-temperature list is short.
    n_effective = min(len(model.racks_list), len(cfg.RACK_SUPPLY_APPROACH_TEMP_LIST))
    return cfg, model, n_effective


def evaluate(model, load_pct, crac_setpoint):
    """Run the simulator's thermal kernel once. Returns (inlet, outlet, power) arrays."""
    cpu_pwr, itfan_pwr, outlet = model.compute_datacenter_IT_load_outlet_temp(
        ITE_load_pct_list=list(load_pct), CRAC_setpoint=float(crac_setpoint)
    )
    inlet = np.asarray(model.rackwise_inlet_temp, dtype=np.float64)
    power = np.asarray(cpu_pwr, dtype=np.float64) + np.asarray(itfan_pwr, dtype=np.float64)
    return inlet, np.asarray(outlet, dtype=np.float64), power


def grid_positions(n_racks, racks_per_row):
    """Rack index -> (row, col). SustainDC indexes racks row-major over the hall."""
    return np.array([(i // racks_per_row, i % racks_per_row) for i in range(n_racks)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dc_config.json",
                    help="config file inside dc-rl/utils/")
    ap.add_argument("--baseline-load", type=float, default=50.0,
                    help="utilisation %% applied to every rack in the unperturbed state")
    ap.add_argument("--perturbed-load", type=float, default=100.0,
                    help="utilisation %% applied to the single source rack")
    ap.add_argument("--crac-setpoint", type=float, default=20.0)
    ap.add_argument("--out-prefix", default="phase0")
    args = ap.parse_args()

    cfg, model, n = build_model(args.config)
    racks_per_row = cfg.NUM_RACKS_PER_ROW
    pos = grid_positions(n, racks_per_row)

    print(f"config={args.config}  NUM_RACKS(declared)={cfg.NUM_RACKS}  "
          f"racks simulated={n}  rows={cfg.NUM_ROWS} x {racks_per_row}")
    print(f"supply approach temps (unique): "
          f"{sorted(set(cfg.RACK_SUPPLY_APPROACH_TEMP_LIST[:n]))}")

    base_load = np.full(n, args.baseline_load)
    base_inlet, base_outlet, base_power = evaluate(model, base_load, args.crac_setpoint)

    # ---- Experiment A/B: one-rack power perturbation, response everywhere ----
    rows = []
    d_inlet = np.zeros((n, n))
    d_outlet = np.zeros((n, n))
    for src in range(n):
        load = base_load.copy()
        load[src] = args.perturbed_load
        inlet, outlet, power = evaluate(model, load, args.crac_setpoint)
        d_inlet[src] = inlet - base_inlet
        d_outlet[src] = outlet - base_outlet
        dP_src = power[src] - base_power[src]
        for tgt in range(n):
            manhattan = int(abs(pos[src, 0] - pos[tgt, 0]) + abs(pos[src, 1] - pos[tgt, 1]))
            rows.append({
                "source_rack": src,
                "target_rack": tgt,
                "source_row": int(pos[src, 0]), "source_col": int(pos[src, 1]),
                "target_row": int(pos[tgt, 0]), "target_col": int(pos[tgt, 1]),
                "manhattan_distance": manhattan,
                "same_row": int(pos[src, 0] == pos[tgt, 0]),
                "source_delta_power_W": dP_src,
                "delta_inlet_K": d_inlet[src, tgt],
                "delta_outlet_K": d_outlet[src, tgt],
            })

    off_diag = ~np.eye(n, dtype=bool)
    max_cross_inlet = float(np.abs(d_inlet[off_diag]).max())
    max_self_inlet = float(np.abs(np.diag(d_inlet)).max())
    max_cross_outlet = float(np.abs(d_outlet[off_diag]).max())
    max_self_outlet = float(np.abs(np.diag(d_outlet)).max())
    mean_dP = float(np.mean([r["source_delta_power_W"] for r in rows]))

    # ---- Experiment C: sensitivity of inlet temperature to the CRAC setpoint ----
    sp_lo, sp_hi = 16.0, 24.0
    inlet_lo, _, _ = evaluate(model, base_load, sp_lo)
    inlet_hi, _, _ = evaluate(model, base_load, sp_hi)
    d_inlet_d_setpoint = (inlet_hi - inlet_lo) / (sp_hi - sp_lo)

    # ---- Experiment D: is own inlet temperature a function of own load at all? ----
    own_load_sweep = []
    for pct in [0.0, 25.0, 50.0, 75.0, 100.0]:
        load = base_load.copy()
        load[0] = pct
        inlet, outlet, power = evaluate(model, load, args.crac_setpoint)
        own_load_sweep.append((pct, float(inlet[0]), float(outlet[0]), float(power[0])))

    # ---- Experiment E: global (non-spatial) coupling through CRAC return air ----
    ret_base = dc.calculate_avg_CRAC_return_temp(
        rack_return_approach_temp_list=cfg.RACK_RETURN_APPROACH_TEMP_LIST[:n],
        rackwise_outlet_temp=list(base_outlet))
    load = base_load.copy()
    load[0] = args.perturbed_load
    _, outlet_p, _ = evaluate(model, load, args.crac_setpoint)
    ret_pert = dc.calculate_avg_CRAC_return_temp(
        rack_return_approach_temp_list=cfg.RACK_RETURN_APPROACH_TEMP_LIST[:n],
        rackwise_outlet_temp=list(outlet_p))

    # ---------------------------- report ----------------------------
    print()
    print("A/B  one rack driven from "
          f"{args.baseline_load:.0f}% to {args.perturbed_load:.0f}% utilisation")
    print(f"     mean power change at the perturbed rack : {mean_dP:,.1f} W")
    print(f"     max |d inlet| at the PERTURBED rack     : {max_self_inlet:.3e} K")
    print(f"     max |d inlet| at ANY OTHER rack         : {max_cross_inlet:.3e} K")
    print(f"     max |d outlet| at the PERTURBED rack    : {max_self_outlet:.3e} K")
    print(f"     max |d outlet| at ANY OTHER rack        : {max_cross_outlet:.3e} K")
    print(f"     float64 eps at these magnitudes         : {np.spacing(float(base_inlet.max())):.3e} K")
    print()
    print("C    d(inlet)/d(CRAC setpoint), per rack:")
    print(f"     min {d_inlet_d_setpoint.min():.6f}   max {d_inlet_d_setpoint.max():.6f}")
    print()
    print("D    rack 0 own-load sweep (others held at baseline):")
    print("     load%    inlet_C     outlet_C     rack_power_W")
    for pct, it, ot, pw in own_load_sweep:
        print(f"     {pct:5.0f}   {it:8.4f}   {ot:10.4f}   {pw:14,.1f}")
    print()
    print("E    avg CRAC return temperature (the one global coupling path):")
    print(f"     baseline {ret_base:.5f} C -> one rack perturbed {ret_pert:.5f} C "
          f"(delta {ret_pert - ret_base:+.5f} K)")

    # ---------------------------- artefacts ----------------------------
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "figures").mkdir(parents=True, exist_ok=True)
    csv_path = REPO / "results" / f"{args.out_prefix}_coupling.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "config": args.config,
        "racks_simulated": n,
        "baseline_load_pct": args.baseline_load,
        "perturbed_load_pct": args.perturbed_load,
        "crac_setpoint_C": args.crac_setpoint,
        "mean_source_delta_power_W": mean_dP,
        "max_abs_delta_inlet_self_K": max_self_inlet,
        "max_abs_delta_inlet_cross_K": max_cross_inlet,
        "max_abs_delta_outlet_self_K": max_self_outlet,
        "max_abs_delta_outlet_cross_K": max_cross_outlet,
        "d_inlet_d_setpoint_min": float(d_inlet_d_setpoint.min()),
        "d_inlet_d_setpoint_max": float(d_inlet_d_setpoint.max()),
        "avg_crac_return_temp_baseline_C": float(ret_base),
        "avg_crac_return_temp_perturbed_C": float(ret_pert),
        "own_load_sweep_rack0": [
            {"load_pct": p, "inlet_C": i, "outlet_C": o, "power_W": w_}
            for p, i, o, w_ in own_load_sweep
        ],
    }
    summary_path = REPO / "results" / f"{args.out_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plot(rows, d_inlet_d_setpoint, own_load_sweep, args,
         REPO / "results" / "figures" / f"{args.out_prefix}_coupling.png",
         max_cross_inlet)

    print()
    print(f"wrote {csv_path.relative_to(REPO)}")
    print(f"wrote {summary_path.relative_to(REPO)}")


def plot(rows, d_inlet_d_setpoint, own_load_sweep, args, out_png, max_cross_inlet):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dists = np.array([r["manhattan_distance"] for r in rows])
    din = np.array([r["delta_inlet_K"] for r in rows])
    dout = np.array([r["delta_outlet_K"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ax.scatter(dists + np.random.default_rng(0).uniform(-0.12, 0.12, dists.size), din,
               s=14, alpha=0.5, color="#1f77b4")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Manhattan grid distance from perturbed rack")
    ax.set_ylabel(r"$\Delta$ inlet temperature (K)")
    ax.set_title("Inlet temperature response")
    span = max(max_cross_inlet, 1e-3) * 1.6
    ax.set_ylim(-span, span)
    ax.text(0.5, 0.55,
            f"identically zero at every distance\n(max |$\\Delta$| off-diagonal = {max_cross_inlet:.1e} K)",
            transform=ax.transAxes, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="#d39e00"))

    ax = axes[1]
    ax.scatter(dists + np.random.default_rng(1).uniform(-0.12, 0.12, dists.size), dout,
               s=14, alpha=0.5, color="#d62728")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Manhattan grid distance from perturbed rack")
    ax.set_ylabel(r"$\Delta$ outlet temperature (K)")
    ax.set_title("Outlet temperature response")
    ax.text(0.5, 0.5, "non-zero only at distance 0\n(self term only)",
            transform=ax.transAxes, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="#d39e00"))

    ax = axes[2]
    pcts = [p for p, _, _, _ in own_load_sweep]
    inl = [i for _, i, _, _ in own_load_sweep]
    outl = [o for _, _, o, _ in own_load_sweep]
    ax.plot(pcts, inl, "o-", label="own inlet", color="#1f77b4")
    ax.plot(pcts, outl, "s-", label="own outlet", color="#d62728")
    ax.set_xlabel("own utilisation (%)")
    ax.set_ylabel("temperature (C)")
    ax.set_title("Rack 0: response to its own load")
    ax.legend()

    fig.suptitle(
        f"SustainDC thermal coupling probe  |  {args.config}  |  "
        f"CRAC setpoint {args.crac_setpoint:.0f} C, "
        f"{args.baseline_load:.0f}% -> {args.perturbed_load:.0f}% on one rack",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=160)
    print(f"wrote {out_png.relative_to(REPO)}")


if __name__ == "__main__":
    main()
