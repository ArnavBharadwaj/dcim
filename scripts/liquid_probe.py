"""Probe the liquid-cooled twin: does thermal coupling follow the plumbing or the floor?

This is the liquid analogue of scripts/phase0b_probe_new_kernel.py, and it asks the
question that decides whether a graph model is worth building for this hall.

In the air-cooled twin, coupling was diffuse and followed geometry, so neighbouring
racks correlated at 0.995 and neighbour features added almost nothing (Phase 2). In a
direct-to-chip hall the coupling should instead follow the hydraulics: racks sharing a
branch manifold and a CDU are coupled; racks on different loops are not, however close
they stand.

If that holds, a Euclidean k-nearest-neighbour feature set is not merely weak here --
it is choosing the wrong neighbours.

    uv run python scripts/liquid_probe.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.graph.build import build_liquid_graph                       # noqa: E402
from src.twin.config import build_liquid                             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", default="hall_l1")
    ap.add_argument("--twin", default="configs/twin/liquid.yaml")
    ap.add_argument("--util", type=float, default=85.0)
    ap.add_argument("--water", type=float, default=34.0)
    ap.add_argument("--pump", type=float, default=0.85)
    ap.add_argument("--out-prefix", default="liquid")
    args = ap.parse_args()

    twin, geom, topo, cfg = build_liquid(
        REPO / "configs" / "hall" / f"{args.hall}.yaml", REPO / args.twin)
    graph = build_liquid_graph(geom, topo)
    n = geom.n_racks
    util = np.full(n, args.util)

    print(f"hall={geom.name}  racks={n}  cdus={topo.n_cdus}  branches={topo.n_branches}")
    print(f"graph: {graph.n_nodes} nodes, {graph.n_edges} edges\n")

    # Finite-difference influence of each rack's power on every rack's coolant supply.
    D = twin.influence_matrix(util, args.water, args.pump)
    off = ~np.eye(n, dtype=bool)

    same_branch = (topo.branch_of_rack[:, None] == topo.branch_of_rack[None, :]) & off
    same_cdu = (topo.cdu_of_rack[:, None] == topo.cdu_of_rack[None, :]) & off
    same_cdu_other_branch = same_cdu & ~same_branch
    other_cdu = (~same_cdu) & off

    euclid = np.linalg.norm(geom.centre[:, None, :] - geom.centre[None, :, :], axis=2)

    def stat(mask, label):
        v = np.abs(D[mask])
        d = euclid[mask]
        print(f"  {label:<34} n={mask.sum():6d}  mean |dT| {v.mean():.5f} K   "
              f"mean distance {d.mean():5.2f} m")
        return v.mean()

    print("Coupling by hydraulic relationship:")
    a = stat(same_branch, "same branch (shared manifold)")
    b = stat(same_cdu_other_branch, "same CDU, different branch")
    c = stat(other_cdu, "different CDU")

    print("\nCoupling by physical proximity:")
    # 2.6 m clears one row pitch, so adjacent-row pairs are included -- those are
    # exactly the physically-close pairs that can sit on different CDUs.
    near = off & (euclid < 2.6)
    far = off & (euclid > 8.0)
    stat(near, "physically within 2.6 m")
    stat(far, "physically beyond 8 m")

    # The decisive comparison: among physically close pairs, does sharing a loop matter?
    near_same = near & same_cdu
    near_diff = near & ~same_cdu
    print("\nAmong pairs that are physically close (< 2.6 m, one row pitch):")
    ns = np.abs(D[near_same]).mean() if near_same.any() else float("nan")
    nd = np.abs(D[near_diff]).mean() if near_diff.any() else float("nan")
    print(f"  on the same CDU      n={int(near_same.sum()):6d}  mean |dT| {ns:.5f} K")
    print(f"  on different CDUs    n={int(near_diff.sum()):6d}  mean |dT| {nd:.5f} K")
    ratio = ns / nd if nd and np.isfinite(nd) and nd > 0 else float("inf")
    print(f"  ratio                {ratio:.1f}x")

    # How much does Euclidean distance explain, against hydraulic relationship?
    flat = np.abs(D[off])
    ed = euclid[off]
    r_dist = float(np.corrcoef(flat, ed)[0, 1])
    grp = np.where(same_branch[off], 2, np.where(same_cdu[off], 1, 0))
    means = np.array([flat[grp == g].mean() if (grp == g).any() else 0.0
                      for g in range(3)])
    ss_tot = float(((flat - flat.mean()) ** 2).sum())
    ss_res = float(((flat - means[grp]) ** 2).sum())
    r2_topo = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\ncorrelation of |dT| with Euclidean distance : {r_dist:+.3f}")
    print(f"variance of |dT| explained by loop topology  : {r2_topo:.3f}")
    print("\nIf the second number is large and the first is near zero, a k-nearest-"
          "\nneighbour feature set built on distance is selecting the wrong racks.")

    summary = {
        "hall": geom.name, "racks": n, "cdus": int(topo.n_cdus),
        "branches": int(topo.n_branches),
        "mean_abs_dT_same_branch_K": float(a),
        "mean_abs_dT_same_cdu_other_branch_K": float(b),
        "mean_abs_dT_other_cdu_K": float(c),
        "near_same_cdu_K": float(ns), "near_different_cdu_K": float(nd),
        "near_ratio": float(ratio),
        "corr_abs_dT_with_euclidean_distance": r_dist,
        "variance_explained_by_topology": float(r2_topo),
        "operating_point": {"util_pct": args.util, "facility_water_c": args.water,
                            "pump_frac": args.pump},
    }
    p = REPO / "results" / f"{args.out_prefix}_coupling_summary.json"
    p.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {p.relative_to(REPO)}")

    plot(D, euclid, same_branch, same_cdu, off, geom, topo,
         REPO / "results" / "figures" / f"{args.out_prefix}_coupling.png")


def plot(D, euclid, same_branch, same_cdu, off, geom, topo, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    COOL, WARM, MID = "#1F6F8B", "#B23A2E", "#5A9E6F"
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))

    a = ax[0]
    v = np.abs(D[off]); d = euclid[off]
    sb = same_branch[off]; sc = same_cdu[off] & ~sb
    rng = np.random.default_rng(0)
    k = rng.choice(v.size, min(9000, v.size), replace=False)
    for m, col, lab in ((~(sb | sc))[k], "#B9C6CC", "different CDU"), \
                       ((sc)[k], MID, "same CDU, other branch"), \
                       ((sb)[k], COOL, "same branch"):
        a.scatter(d[k][m], np.maximum(v[k][m], 1e-7), s=2.2, alpha=0.35,
                  color=col, label=lab, edgecolors="none")
    a.set_yscale("log")
    a.set_xlabel("Euclidean distance between racks (m)")
    a.set_ylabel("$|\\Delta T_{coolant}|$ from a 5 kW step (K)")
    a.set_title("Coupling does not follow the floor plan")
    leg = a.legend(frameon=False, markerscale=4, fontsize=7.5, loc="lower left")

    b = ax[1]
    groups = [np.abs(D[same_branch]), np.abs(D[same_cdu & ~same_branch]),
              np.abs(D[off & ~same_cdu])]
    labels = ["same\nbranch", "same CDU,\nother branch", "different\nCDU"]
    bp = b.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True,
                   widths=0.55, medianprops=dict(color="black", lw=1.1))
    for patch, col in zip(bp["boxes"], (COOL, MID, "#B9C6CC")):
        patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor("black")
    b.set_yscale("log")
    b.set_ylabel("$|\\Delta T_{coolant}|$  (K)")
    b.set_title("Coupling follows the hydraulics")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
