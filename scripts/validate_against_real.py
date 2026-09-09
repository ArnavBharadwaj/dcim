"""Validate the air twin against real data-centre measurements.

Four downloaded datasets, none of which was used to build the twin:

  nycu_rack_temp/   135-sensor rack telemetry, CC Wang Lab, NYCU
  thermofoam_cfd/   3D OpenFOAM CFD benchmark, 6 racks, 10 parameter cases
  deepcool/         24 h closed-loop trace: ambient, IT load, setpoint, rack inlet
  chilled_water_dc/ plant-side chilled water telemetry

The twin was calibrated against published *ranges*, never against measurements. This
script checks it against the measurements, and the result is not favourable.

    uv run python scripts/validate_against_real.py --data ~/Downloads/real
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.generate import TRAINING_ARRAYS, load          # noqa: E402


def nycu(root: pathlib.Path) -> dict:
    inl = pd.read_csv(root / "nycu_rack_temp" / "all_inlets.csv")
    out = pd.read_csv(root / "nycu_rack_temp" / "all_outlets.csv")
    sens = [c for c in inl.columns if c.isdigit()]
    X = inl[sens].to_numpy(float)

    identical = bool(np.allclose(X, out[sens].to_numpy(float)))
    C = np.corrcoef(X.T)
    iu = np.triu_indices(len(sens), 1)
    r = C[iu]
    # Sensors appear to run 9 per rack (vertical positions), 15 racks.
    grp = np.arange(len(sens)) // 9
    same = (grp[:, None] == grp[None, :])[iu]
    return {
        "n_sensors": len(sens), "n_rows": int(len(inl)),
        "inlets_identical_to_outlets": identical,
        "mean_pairwise_r": float(np.nanmean(r)),
        "median_pairwise_r": float(np.nanmedian(r)),
        "frac_negative_r": float((r < 0).mean()),
        "frac_r_above_099": float((r > 0.99).mean()),
        "mean_within_rack_r": float(np.nanmean(r[same])),
        "mean_across_rack_r": float(np.nanmean(r[~same])),
        "mean_spread_across_sensors_K": float(np.mean(X.max(1) - X.min(1))),
        "max_spread_across_sensors_K": float(np.max(X.max(1) - X.min(1))),
    }


def thermofoam(root: pathlib.Path) -> dict:
    pr = pd.read_csv(root / "thermofoam_cfd" / "v02_per_rack_summary.csv")
    pr["T_C"] = pr.TavgRackInlet_mean - 273.15
    base = pr[pr.caseID == "C00"].set_index("rackID").T_C

    spreads = {}
    for cid, g in pr.groupby("caseID"):
        if g.rackPowerW.nunique() == 1:
            spreads[f"{cid}_{g.caseName.iloc[0]}"] = float(g.T_C.max() - g.T_C.min())

    # Overloading one rack: what happens to the others?
    overload = {}
    for cid in ("C05", "C08", "C09"):
        g = pr[pr.caseID == cid].set_index("rackID")
        delta = (g.T_C - base).to_dict()
        overload[f"{cid}_{g.caseName.iloc[0]}"] = {
            "per_rack_delta_K": {k: float(v) for k, v in delta.items()},
            "delta_at_overloaded_rack_K": float(delta["L2"]),
            "max_delta_elsewhere_K": float(max(v for k, v in delta.items() if k != "L2")),
            "min_delta_elsewhere_K": float(min(v for k, v in delta.items() if k != "L2")),
        }
    return {
        "baseline_spread_at_identical_load_K": float(base.max() - base.min()),
        "baseline_left_aisle_mean_C": float(base[["L1", "L2", "L3"]].mean()),
        "baseline_right_aisle_mean_C": float(base[["R1", "R2", "R3"]].mean()),
        "spread_by_case_K": spreads,
        "overloaded_rack_response": overload,
    }


def deepcool(root: pathlib.Path) -> dict:
    dc = pd.read_csv(root / "deepcool" / "run1.csv")
    X = np.column_stack([np.ones(len(dc)), dc.cooling_setpoint_C, dc.it_load,
                         dc.ambient_C])
    y = dc.rack_inlet_C.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    steps = {f"abs_delta_inlet_{h}min_mean_K": float(dc.rack_inlet_C.diff(h).abs().mean())
             for h in (1, 5, 10)}
    return {
        "n_rows": int(len(dc)),
        "d_inlet_d_setpoint": float(beta[1]),
        "d_inlet_d_itload_K": float(beta[2]),
        "d_inlet_d_ambient": float(beta[3]),
        "r2": float(r2),
        **steps,
    }


def twin_comparison() -> dict:
    """The same statistics, computed on our generated hall_a."""
    d, _ = load(REPO / "data" / "trajectories", "hall_a", arrays=TRAINING_ARRAYS)
    X = d["inlet"]
    C = np.corrcoef(X.T)
    r = C[np.triu_indices(X.shape[1], 1)]
    return {
        "mean_pairwise_r": float(np.nanmean(r)),
        "median_pairwise_r": float(np.nanmedian(r)),
        "frac_negative_r": float((r < 0).mean()),
        "frac_r_above_099": float((r > 0.99).mean()),
        "mean_spread_across_racks_K": float(np.mean(X.max(1) - X.min(1))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(pathlib.Path.home() / "Downloads" / "real"))
    ap.add_argument("--out", default="results/real_data_validation.json")
    args = ap.parse_args()
    root = pathlib.Path(args.data).expanduser()
    if not root.exists():
        raise SystemExit(f"{root} not found")

    n, t, dcl, mine = nycu(root), thermofoam(root), deepcool(root), twin_comparison()

    print("=" * 74)
    print("SPATIAL STRUCTURE  —  pairwise correlation between rack temperatures")
    print("=" * 74)
    print(f"{'':<26}{'real (NYCU)':>16}{'our twin':>14}")
    print(f"{'mean pairwise r':<26}{n['mean_pairwise_r']:>16.4f}"
          f"{mine['mean_pairwise_r']:>14.4f}")
    print(f"{'median pairwise r':<26}{n['median_pairwise_r']:>16.4f}"
          f"{mine['median_pairwise_r']:>14.4f}")
    print(f"{'fraction negative':<26}{n['frac_negative_r']:>15.1%}"
          f"{mine['frac_negative_r']:>14.1%}")
    print(f"{'fraction above 0.99':<26}{n['frac_r_above_099']:>15.1%}"
          f"{mine['frac_r_above_099']:>14.1%}")
    print(f"{'mean spread across (K)':<26}"
          f"{n['mean_spread_across_sensors_K']:>16.2f}"
          f"{mine['mean_spread_across_racks_K']:>14.2f}")
    print()
    print("  Our Phase 2 conclusion -- that neighbour features add almost nothing --")
    print("  rested on a correlation of 0.995. Real sensors show 0.59, with 7.7% of")
    print("  pairs ANTI-correlated. Our kernel produces none at any setting.")

    print()
    print("=" * 74)
    print("CFD CROSS-CHECK  —  ThermoFOAM, six racks, identical 3 kW load")
    print("=" * 74)
    print(f"  spread across racks at identical load : "
          f"{t['baseline_spread_at_identical_load_K']:.2f} K")
    print(f"  left aisle {t['baseline_left_aisle_mean_C']:.2f} C vs right "
          f"{t['baseline_right_aisle_mean_C']:.2f} C")
    print("\n  overloading one rack (L2, 3 -> 8 kW) changes the OTHERS by:")
    for k, v in t["overloaded_rack_response"].items():
        print(f"    {k:<34} at L2 {v['delta_at_overloaded_rack_K']:+6.2f} K   "
              f"elsewhere {v['min_delta_elsewhere_K']:+6.2f} to "
              f"{v['max_delta_elsewhere_K']:+6.2f} K")
    print("\n  Note C05: the largest rise is NOT at the overloaded rack.")
    print("  Note C09: at high flow, loading a rack makes every rack COLDER.")
    print("  Our D is non-negative by construction and its diagonal dominates;")
    print("  neither behaviour is representable.")

    print()
    print("=" * 74)
    print("CLOSED LOOP  —  DeepCool, 24 h at 1 min")
    print("=" * 74)
    print(f"  d(inlet)/d(setpoint) = {dcl['d_inlet_d_setpoint']:+.3f}   "
          f"(R2 {dcl['r2']:.3f}, observational, so confounded)")
    print(f"    our twin 0.878-0.923   SustainDC exactly 1.000")
    print(f"  d(inlet)/d(IT load)  = {dcl['d_inlet_d_itload_K']:+.3f} K per unit load")
    print(f"  |d inlet| over 1 / 5 / 10 min: "
          f"{dcl['abs_delta_inlet_1min_mean_K']:.3f} / "
          f"{dcl['abs_delta_inlet_5min_mean_K']:.3f} / "
          f"{dcl['abs_delta_inlet_10min_mean_K']:.3f} K")

    if n["inlets_identical_to_outlets"]:
        print()
        print("!! DATA ISSUE: all_inlets.csv and all_outlets.csv are numerically")
        print("   identical in every sensor column. One of the two is mislabelled or")
        print("   the export duplicated a file; the outlet data is effectively absent.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"nycu": n, "thermofoam": t, "deepcool": dcl, "our_twin": mine}, indent=2))
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
