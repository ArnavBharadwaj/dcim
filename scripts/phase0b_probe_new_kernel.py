"""Phase 0b: re-run the Phase 0 coupling probe against the replacement kernel.

Same experiment, same hall indexing, same distance metric as
`scripts/phase0_probe_thermal_coupling.py`, so the two are directly comparable. The
question is unchanged: perturb one rack's load, and see how every other rack's inlet
temperature responds as a function of distance.

Phase 0's answer for SustainDC was 0.000e+00 K at every distance including zero.
This script measures what the replacement model does instead, and reports:

  * the response-vs-distance profile
  * how much of the influence is carried at each hop order
  * how directed the coupling is
  * d(inlet)/d(supply setpoint), which SustainDC pinned at exactly 1.0 everywhere

Outputs
  results/phase0b_coupling.csv
  results/phase0b_summary.json
  results/figures/phase0b_new_kernel.png
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.twin.config import build                                    # noqa: E402
from src.twin.power import rack_power                                # noqa: E402
from src.twin.recirculation import (cross_interference,              # noqa: E402
                                    escape_fractions, geometric_kernel,
                                    hop_decomposition)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", default="configs/hall/hall_a.yaml")
    ap.add_argument("--twin", default="configs/twin/default.yaml")
    ap.add_argument("--baseline-load", type=float, default=50.0)
    ap.add_argument("--perturbed-load", type=float, default=100.0)
    ap.add_argument("--supply-temp", type=float, default=20.0)
    ap.add_argument("--crac-fan", type=float, default=1.0)
    ap.add_argument("--out-prefix", default="phase0b")
    args = ap.parse_args()

    twin, geom, cfg = build(REPO / args.hall, REPO / args.twin)
    n = geom.n_racks
    print(f"hall={geom.name}  racks={n}  cracs={geom.n_cracs}  "
          f"pattern={geom.orientation_pattern}")

    base_load = np.full(n, args.baseline_load)
    base_inlet = twin.steady_state(base_load, args.supply_temp, args.crac_fan)
    base_state = rack_power(base_load, base_inlet, cfg.power)

    # ---- one-rack perturbation, response everywhere ----
    t0 = time.time()
    d_inlet = np.zeros((n, n))
    d_power = np.zeros(n)
    for src in range(n):
        load = base_load.copy()
        load[src] = args.perturbed_load
        inlet = twin.steady_state(load, args.supply_temp, args.crac_fan)
        d_inlet[src] = inlet - base_inlet
        st = rack_power(load, inlet, cfg.power)
        d_power[src] = st.total_power_w[src] - base_state.total_power_w[src]
    print(f"{n} perturbation solves in {time.time() - t0:.1f}s")

    manhattan = (np.abs(geom.row[:, None] - geom.row[None, :])
                 + np.abs(geom.col[:, None] - geom.col[None, :])).astype(int)

    rows = []
    for src in range(n):
        for tgt in range(n):
            rows.append({
                "source_rack": src, "target_rack": tgt,
                "source_row": int(geom.row[src]), "source_col": int(geom.col[src]),
                "target_row": int(geom.row[tgt]), "target_col": int(geom.col[tgt]),
                "manhattan_distance": int(manhattan[src, tgt]),
                "euclidean_m": float(np.linalg.norm(
                    geom.inlet_xy[tgt] - geom.exhaust_xy[src])),
                "source_delta_power_W": float(d_power[src]),
                "delta_inlet_K": float(d_inlet[src, tgt]),
                "delta_inlet_K_per_kW": float(
                    d_inlet[src, tgt] / (d_power[src] / 1e3)),
            })

    off = ~np.eye(n, dtype=bool)
    summary = {
        "hall": geom.name,
        "racks": n,
        "baseline_load_pct": args.baseline_load,
        "perturbed_load_pct": args.perturbed_load,
        "supply_temp_c": args.supply_temp,
        "crac_fan_frac": args.crac_fan,
        "mean_source_delta_power_W": float(d_power.mean()),
        "max_abs_delta_inlet_self_K": float(np.abs(np.diag(d_inlet)).max()),
        "max_abs_delta_inlet_cross_K": float(np.abs(d_inlet[off]).max()),
        "mean_abs_delta_inlet_cross_K": float(np.abs(d_inlet[off]).mean()),
        "frac_pairs_above_1mK": float((np.abs(d_inlet[off]) > 1e-3).mean()),
    }

    # ---- how far does influence actually reach? ----
    by_dist = {}
    for dd in sorted(set(manhattan.ravel().tolist())):
        mask = (manhattan == dd) & off
        if mask.sum():
            by_dist[int(dd)] = float(np.abs(d_inlet[mask]).mean())
    summary["mean_abs_response_by_manhattan_distance_K"] = by_dist

    # ---- hop decomposition of the influence matrix ----
    phi = twin._provisioning(args.crac_fan, base_state.airflow_m3s, None)
    A = cross_interference(geometric_kernel(geom, cfg.recirculation),
                           escape_fractions(geom, cfg.recirculation, phi),
                           base_state.k_w_per_k)
    D = twin.influence_matrix(base_load, base_inlet, args.crac_fan)
    terms = hop_decomposition(A, base_state.k_w_per_k, max_order=8)
    hop_share = [float(t.sum() / D.sum()) for t in terms]
    summary["hop_order_share"] = hop_share
    summary["provisioning_ratio"] = float(phi)
    summary["D_density"] = float((D > 1e-12).mean())
    summary["D_asymmetry"] = float(np.abs(D - D.T).sum() / np.abs(D).sum())
    summary["A_asymmetry"] = float(np.abs(A - A.T).sum() / np.abs(A).sum())

    # ---- setpoint gain, the quantity SustainDC pinned at exactly 1.0 ----
    lo = twin.steady_state(base_load, args.supply_temp - 2.0, args.crac_fan)
    hi = twin.steady_state(base_load, args.supply_temp + 2.0, args.crac_fan)
    gain = (hi - lo) / 4.0
    summary["d_inlet_d_setpoint_min"] = float(gain.min())
    summary["d_inlet_d_setpoint_max"] = float(gain.max())
    summary["d_inlet_d_setpoint_mean"] = float(gain.mean())

    # ---------------------------- report ----------------------------
    print()
    print(f"A/B  one rack driven {args.baseline_load:.0f}% -> {args.perturbed_load:.0f}%")
    print(f"     mean power change at the perturbed rack : "
          f"{summary['mean_source_delta_power_W']:,.1f} W")
    print(f"     max |d inlet| at the PERTURBED rack     : "
          f"{summary['max_abs_delta_inlet_self_K']:.4f} K")
    print(f"     max |d inlet| at ANY OTHER rack         : "
          f"{summary['max_abs_delta_inlet_cross_K']:.4f} K")
    print(f"     mean |d inlet| over all other racks     : "
          f"{summary['mean_abs_delta_inlet_cross_K']:.4f} K")
    print(f"     rack pairs coupled above 1 mK           : "
          f"{100 * summary['frac_pairs_above_1mK']:.1f}%")
    print()
    print("     mean |response| by Manhattan distance (K):")
    for dd, v in list(by_dist.items())[:10]:
        print(f"       d={dd:2d}  {v:.5f}")
    print()
    print("hop-order share of total influence:")
    for m, s in enumerate(hop_share, 1):
        if s > 5e-4:
            print(f"     {m} hop: {100 * s:5.1f}%")
    print(f"     D density {summary['D_density']:.3f}   "
          f"D asymmetry {summary['D_asymmetry']:.3f}")
    print()
    print(f"C    d(inlet)/d(supply setpoint): min {gain.min():.6f}  "
          f"max {gain.max():.6f}  (SustainDC: exactly 1.000000 everywhere)")

    # ---------------------------- artefacts ----------------------------
    (REPO / "results" / "figures").mkdir(parents=True, exist_ok=True)
    csv_path = REPO / "results" / f"{args.out_prefix}_coupling.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (REPO / "results" / f"{args.out_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2))

    plot(geom, d_inlet, manhattan, hop_share, gain, args, summary,
         REPO / "results" / "figures" / f"{args.out_prefix}_new_kernel.png")
    print(f"\nwrote {csv_path.relative_to(REPO)}")


def plot(geom, d_inlet, manhattan, hop_share, gain, args, summary, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    off = ~np.eye(d_inlet.shape[0], dtype=bool)

    # (a) response vs distance, against SustainDC's flat zero
    ax = axes[0, 0]
    dists = manhattan[off]
    resp = np.abs(d_inlet[off])
    uniq = np.array(sorted(set(dists.tolist())))
    med = np.array([np.median(resp[dists == d]) for d in uniq])
    p90 = np.array([np.percentile(resp[dists == d], 90) for d in uniq])
    ax.fill_between(uniq, 0, p90, alpha=0.25, color="#1f77b4", label="90th percentile")
    ax.plot(uniq, med, "o-", color="#1f77b4", label="median")
    ax.axhline(0, color="#d62728", lw=2.0, ls="--",
               label="SustainDC (exactly 0 at every distance)")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("Manhattan grid distance from perturbed rack")
    ax.set_ylabel(r"$|\Delta$ inlet temperature$|$ (K)")
    ax.set_title("(a) Influence decays with distance instead of vanishing")
    ax.legend(fontsize=8, loc="upper right")

    # (b) hop decomposition
    ax = axes[0, 1]
    orders = np.arange(1, len(hop_share) + 1)
    shares = 100 * np.array(hop_share)
    ax.bar(orders, shares, color="#2ca02c")
    for o, s in zip(orders, shares):
        if s > 0.4:
            ax.text(o, s + 1.0, f"{s:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("hops through intermediate racks")
    ax.set_ylabel("share of total influence (%)")
    ax.set_title(f"(b) Multi-hop structure: {100 - shares[0]:.0f}% beyond one hop\n"
                 "(SustainDC: zero hops, D is the zero matrix)")
    ax.set_ylim(0, max(shares) * 1.2)

    # (c) spatial response map for one central source rack
    ax = axes[1, 0]
    src = int(np.argmin(np.linalg.norm(
        geom.centre - geom.centre.mean(axis=0), axis=1)))
    field = d_inlet[src]
    grid = np.full((geom.n_rows, geom.racks_per_row), np.nan)
    grid[geom.row, geom.col] = field
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="magma")
    ax.scatter([geom.col[src]], [geom.row[src]], marker="*", s=220,
               edgecolor="white", facecolor="none", linewidth=1.6,
               label="perturbed rack")
    ax.set_xlabel("rack column")
    ax.set_ylabel("rack row")
    ax.set_title("(c) Where one rack's heat lands\n"
                 "(anisotropic: air is carried toward the returns)")
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, label=r"$\Delta$ inlet (K)")

    # (d) setpoint gain
    ax = axes[1, 1]
    ax.hist(gain, bins=30, color="#9467bd", edgecolor="white")
    ax.axvline(1.0, color="#d62728", lw=2.0, ls="--",
               label="SustainDC: exactly 1.0 at every rack")
    ax.set_xlabel(r"$d(T_{\mathrm{inlet}})\,/\,d(T_{\mathrm{supply}})$")
    ax.set_ylabel("racks")
    ax.set_title("(d) The setpoint reshapes the field,\nrather than translating it")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Replacement thermal kernel  |  {geom.name}, {geom.n_racks} racks  |  "
        f"one rack {args.baseline_load:.0f}% -> {args.perturbed_load:.0f}%, "
        f"supply {args.supply_temp:.0f} C, CRAC fan {args.crac_fan:.0%}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=155)
    print(f"wrote {out_png.relative_to(REPO)}")


if __name__ == "__main__":
    main()
