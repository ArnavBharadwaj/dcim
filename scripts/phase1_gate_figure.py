"""Phase 1 gate figure: the distribution of rack inlet temperature.

The brief asks for this plot explicitly, and states the gate in terms of it: if fewer
than 2% of timesteps breach the thermal limit, the load density goes up and the dataset
is regenerated.

    uv run python scripts/phase1_gate_figure.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.generate import load                    # noqa: E402
from src.eval import metrics                          # noqa: E402
from src.twin.config import build                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halls", default="hall_a,hall_b,hall_c")
    ap.add_argument("--data", default="data/trajectories")
    ap.add_argument("--out", default="results/figures/phase1_thermal_gate.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    halls = [h for h in args.halls.split(",") if h]
    fig, axes = plt.subplots(2, len(halls), figsize=(5.0 * len(halls), 8.0),
                             squeeze=False)

    print(f"{'hall':<9} {'racks':>6} {'steps':>8} {'mean':>7} {'p99':>7} {'max':>7} "
          f"{'>27C steps':>11} {'>27C racks':>11} {'>32C':>7}")
    for col, hall in enumerate(halls):
        data, meta = load(REPO / args.data, hall)
        _, geom, cfg = build(REPO / "configs" / "hall" / f"{hall}.yaml",
                             REPO / "configs" / "twin" / "default.yaml")
        inlet = data["inlet"]
        rec = cfg.limits.inlet_recommended_max_c
        allow = cfg.limits.inlet_allowable_max_c
        f_rec = metrics.breach_fraction(inlet, rec)
        f_rack = metrics.rack_breach_fraction(inlet, rec)
        f_allow = metrics.breach_fraction(inlet, allow)
        print(f"{hall:<9} {geom.n_racks:>6} {inlet.shape[0]:>8,} "
              f"{inlet.mean():>7.2f} {np.percentile(inlet, 99):>7.2f} "
              f"{inlet.max():>7.2f} {100 * f_rec:>10.2f}% {100 * f_rack:>10.2f}% "
              f"{100 * f_allow:>6.2f}%")

        ax = axes[0][col]
        ax.hist(inlet.ravel(), bins=120, color="#4c78a8", edgecolor="none")
        ax.axvline(rec, color="#e45756", ls="--", lw=1.8,
                   label=f"ASHRAE recommended {rec:.0f} C")
        ax.axvline(allow, color="#772d2d", ls=":", lw=1.8,
                   label=f"allowable {allow:.0f} C")
        ax.set_yscale("log")
        ax.set_xlabel("rack inlet temperature (C)")
        ax.set_ylabel("rack-timesteps")
        ax.set_title(f"{hall}: {geom.n_racks} racks, "
                     f"{inlet.shape[0] * meta['dt_s'] / 3600:.0f} h\n"
                     f"{100 * f_rec:.1f}% of timesteps breach "
                     f"({'PASS' if f_rec >= 0.02 else 'FAIL'}, gate is 2%)")
        ax.legend(fontsize=8)

        # Per-policy distribution: the deliberately bad policy has to be visibly
        # hotter, or sampling it during generation bought nothing.
        ax = axes[1][col]
        names = meta["gen_config"]["policies"]
        for pid, name in enumerate(names):
            m = data["policy_id"] == pid
            if not m.any():
                continue
            ax.hist(inlet[m].ravel(), bins=90, histtype="step", lw=1.6, label=name,
                    density=True)
        ax.axvline(rec, color="#e45756", ls="--", lw=1.4)
        ax.set_xlabel("rack inlet temperature (C)")
        ax.set_ylabel("density")
        ax.set_title("by placement policy")
        ax.legend(fontsize=8)

    fig.suptitle("Phase 1 thermal gate: rack inlet temperature distribution",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
