"""Figures for the PDF status report.

Colour lives here and nowhere else in the report: the document body is set in black
on white, so the figures are the only place a reader's eye is drawn by hue. Each one
is drawn from a file already in the repository -- the Phase 0 and 0b probe CSVs, the
generated trajectories, or results/runs.csv -- and nothing is recomputed from scratch.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

COOL, WARM, MID, DARK = "#1F6F8B", "#B23A2E", "#5A9E6F", "#2B2B2B"


def fig_coupling():
    """The Phase 0 finding and its repair, side by side."""
    old = pd.read_csv(REPO / "results" / "phase0_coupling.csv")
    new = pd.read_csv(REPO / "results" / "phase0b_coupling.csv")
    fig, ax = plt.subplots(figsize=(5.4, 2.9))

    for df, lab, col, mk in ((new, "Replacement kernel", COOL, "o"),
                             (old, "SustainDC", WARM, "s")):
        d = df[df.source_rack != df.target_rack]
        g = d.groupby("manhattan_distance")["delta_inlet_K"].apply(
            lambda s: np.abs(s).mean())
        g = g[g.index <= 20]
        ax.plot(g.index, np.maximum(g.values, 1e-8), mk + "-", color=col,
                label=lab, ms=3.2, lw=1.2)

    ax.set_yscale("log")
    ax.set_ylim(3e-9, 3e-1)
    ax.set_xlabel("Manhattan grid distance from the perturbed rack")
    ax.set_ylabel("mean $|\\Delta T_{inlet}|$  (K)")
    # SustainDC's values are exactly zero, which a log axis cannot place. They are
    # drawn on the floor of the axis and labelled, rather than silently dropped.
    ax.text(7.6, 1.3e-8, "exactly 0.000e+00 K, plotted on the axis floor",
            color=WARM, fontsize=7.2, ha="left", va="center")
    ax.text(20.0, 4.5e-2, "one hop", color=COOL, fontsize=7.2, ha="right")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.12))
    fig.tight_layout()
    p = OUT / "fig1_coupling.png"
    fig.savefig(p); plt.close(fig)
    return p


def fig_hops():
    """How much influence arrives after m racks."""
    import json
    s = json.loads((REPO / "results" / "phase0b_summary.json").read_text())
    share = np.array(s["hop_order_share"][:6]) * 100
    fig, ax = plt.subplots(figsize=(2.6, 2.5))
    ax.bar(np.arange(1, len(share) + 1), share, color=COOL, width=0.66)
    for i, v in enumerate(share, 1):
        if v > 0.4:
            ax.text(i, v + 1.6, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xlabel("hops through intermediate racks")
    ax.set_ylabel("share of total influence (%)")
    ax.set_ylim(0, 82)
    fig.tight_layout()
    p = OUT / "fig2_hops.png"
    fig.savefig(p); plt.close(fig)
    return p


def fig_redundancy():
    """Why neighbour features add nothing: the racks already agree."""
    from src.data.generate import TRAINING_ARRAYS, load
    from src.data.dataset import neighbour_index
    from src.twin.config import build
    d, _ = load(REPO / "data" / "trajectories", "hall_a", arrays=TRAINING_ARRAYS)
    _, geom, _ = build(REPO / "configs" / "hall" / "hall_a.yaml",
                       REPO / "configs" / "twin" / "default.yaml")
    nb = neighbour_index(geom, 6)
    rng = np.random.default_rng(0)
    t = rng.choice(d["inlet"].shape[0], 900, replace=False)
    own = d["inlet"][t]
    nbr = d["inlet"][t][:, nb].mean(axis=2)
    own, nbr = own.ravel(), nbr.ravel()
    k = rng.choice(own.size, 12000, replace=False)
    r = np.corrcoef(own, nbr)[0, 1]

    fig, ax = plt.subplots(figsize=(2.6, 2.5))
    ax.scatter(own[k], nbr[k], s=1.1, alpha=0.10, color=COOL, edgecolors="none")
    lo, hi = 17.0, 33.5
    ax.plot([lo, hi], [lo, hi], color=DARK, lw=0.7, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("own inlet temperature (°C)")
    ax.set_ylabel("mean of 6 nearest racks (°C)")
    ax.text(0.05, 0.93, f"r = {r:.4f}", transform=ax.transAxes, fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    p = OUT / "fig3_redundancy.png"
    fig.savefig(p); plt.close(fig)
    return p, r


def fig_gate():
    """Where each hall sits against the ASHRAE limits."""
    from src.data.generate import load
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    for hall, col in (("hall_a", COOL), ("hall_b", MID), ("hall_c", WARM)):
        d, _ = load(REPO / "data" / "trajectories", hall, arrays=("inlet",))
        ax.hist(d["inlet"].ravel(), bins=200, histtype="step", lw=1.0,
                color=col, label=hall, density=True)
        del d
    ax.axvline(27, color=DARK, lw=0.9, ls="--")
    ax.axvline(32, color=DARK, lw=0.9, ls=":")
    ax.text(27.15, ax.get_ylim()[1] * 0.86, "27 °C recommended", fontsize=7.5)
    ax.text(32.15, ax.get_ylim()[1] * 0.70, "32 °C allowable", fontsize=7.5)
    ax.set_xlabel("rack inlet temperature (°C)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    p = OUT / "fig4_gate.png"
    fig.savefig(p); plt.close(fig)
    return p


def _latest(df, keys):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df.timestamp, format="mixed", utc=True)
    out = []
    for _, g in df.groupby(keys):
        out.append(g[g.config_hash == g.sort_values("timestamp").iloc[-1].config_hash])
    return pd.concat(out) if out else df


def fig_accuracy():
    df = pd.read_csv(REPO / "results" / "runs.csv")
    b = _latest(df[(df.phase.astype(str) == "2") & (df.hall == "hall_a")],
                ["model", "horizon_s"])
    g = _latest(df[(df.phase.astype(str) == "3") & (df.experiment == "gnn_test")],
                ["model", "horizon_s"])
    a = pd.concat([b, g])
    a = a[a.model.isin(["persistence", "rc", "lstm", "lightgbm_k4", "gnn"])]
    order = ["persistence", "rc", "lstm", "lightgbm_k4", "gnn"]
    names = {"persistence": "persistence", "rc": "RC network", "lstm": "LSTM",
             "lightgbm_k4": "LightGBM $k$=4", "gnn": "graph model"}
    hz = [30, 60, 300]
    m = a.groupby(["model", "horizon_s"]).rmse_k.agg(["mean", "std"]).fillna(0)

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    w = 0.16
    x = np.arange(len(hz))
    greys = ["#BFBFBF", "#9E9E9E", "#7A7A7A", "#565656"]
    for i, mod in enumerate(order):
        vals = [m.loc[(mod, h), "mean"] if (mod, h) in m.index else np.nan for h in hz]
        errs = [m.loc[(mod, h), "std"] if (mod, h) in m.index else 0 for h in hz]
        col = COOL if mod == "gnn" else greys[i]
        ax.bar(x + i * w - 2 * w, vals, w, yerr=errs, capsize=1.8,
               color=col, label=names[mod],
               edgecolor=DARK if mod == "gnn" else "none", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{h} s" for h in hz])
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("test RMSE of $\\Delta T$  (K)")
    ax.legend(frameon=False, ncol=2, fontsize=7.5)
    fig.tight_layout()
    p = OUT / "fig5_accuracy.png"
    fig.savefig(p); plt.close(fig)
    return p


def fig_transfer():
    df = pd.read_csv(REPO / "results" / "runs.csv")
    d = _latest(df[df.phase.astype(str) == "3"], ["hall", "horizon_s"])
    d = d.assign(skill=1 - d.rmse_k / d.persistence_rmse_k)
    m = d.groupby(["hall", "horizon_s"]).skill.agg(["mean", "std"]).fillna(0)
    hz = [30, 60, 300]
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    w = 0.26
    x = np.arange(len(hz))
    for i, (hall, col, lab) in enumerate((
            ("hall_a", COOL, "hall_a  (in-domain)"),
            ("hall_b", MID, "hall_b  (140 racks, zero-shot)"),
            ("hall_c", WARM, "hall_c  (uncontained, zero-shot)"))):
        vals = [100 * m.loc[(hall, h), "mean"] for h in hz]
        errs = [100 * m.loc[(hall, h), "std"] for h in hz]
        ax.bar(x + i * w - w, vals, w, yerr=errs, capsize=2, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"{h} s" for h in hz])
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("skill vs persistence (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    p = OUT / "fig6_transfer.png"
    fig.savefig(p); plt.close(fig)
    return p


if __name__ == "__main__":
    print(fig_coupling())
    print(fig_hops())
    p, r = fig_redundancy(); print(p, f"r={r:.4f}")
    print(fig_gate())
    print(fig_accuracy())
    print(fig_transfer())
