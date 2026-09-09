"""Figures for the liquid-cooling PDF report.

Colour appears only here; the report body is black on white. Every figure is produced
from the twin in this repository at a stated operating point.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "figures" / "liquid_report"
OUT.mkdir(parents=True, exist_ok=True)

from src.twin.config import build_liquid                     # noqa: E402
from src.twin.liquid import solve_coolant                    # noqa: E402
from src.twin.power import rack_power                        # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300,
})
COOL, WARM, MID, GOLD, DARK = "#1F6F8B", "#B23A2E", "#5A9E6F", "#C08A2E", "#2B2B2B"
CDU_COLS = [COOL, WARM, MID, GOLD]

TWIN, GEOM, TOPO, CFG = build_liquid(REPO / "configs" / "hall" / "hall_l1.yaml",
                                     REPO / "configs" / "twin" / "liquid.yaml")
N = GEOM.n_racks


def fig_topology():
    """The floor plan, coloured by coolant loop. Adjacency is not coupling."""
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for c in range(TOPO.n_cdus):
        m = TOPO.cdu_of_rack == c
        ax.scatter(GEOM.centre[m, 0], GEOM.centre[m, 1], s=26, marker="s",
                   color=CDU_COLS[c % len(CDU_COLS)], label=f"CDU {c}",
                   edgecolors="white", linewidths=0.4)
    # Mark a pair that stands side by side but shares no coolant loop.
    d = np.linalg.norm(GEOM.centre[:, None, :] - GEOM.centre[None, :, :], axis=2)
    diff = TOPO.cdu_of_rack[:, None] != TOPO.cdu_of_rack[None, :]
    np.fill_diagonal(diff, False)
    dd = np.where(diff, d, np.inf)
    i, j = np.unravel_index(np.argmin(dd), dd.shape)
    ax.plot(GEOM.centre[[i, j], 0], GEOM.centre[[i, j], 1], "-", color=DARK, lw=1.1)
    ax.annotate(f"{d[i, j]:.1f} m apart,\ndifferent CDUs,\nzero coupling",
                xy=(GEOM.centre[[i, j], 0].mean(), GEOM.centre[[i, j], 1].mean()),
                xytext=(GEOM.centre[[i, j], 0].mean() + 2.2,
                        GEOM.centre[[i, j], 1].mean() - 4.6),
                fontsize=7.4, color=DARK,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=DARK))
    ax.set_xlabel("hall x (m)"); ax.set_ylabel("hall y (m)")
    ax.set_aspect("equal")
    ax.legend(frameon=False, ncol=4, fontsize=7.4, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    p = OUT / "L1_topology.png"; fig.savefig(p); plt.close(fig); return p


def _influence(util=85.0, water=34.0, pump=0.85):
    return TWIN.influence_matrix(np.full(N, util), water, pump)


def fig_coupling():
    """Coupling against distance, and against hydraulic relationship."""
    D = _influence()
    off = ~np.eye(N, dtype=bool)
    same_b = (TOPO.branch_of_rack[:, None] == TOPO.branch_of_rack[None, :]) & off
    same_c = (TOPO.cdu_of_rack[:, None] == TOPO.cdu_of_rack[None, :]) & off
    euclid = np.linalg.norm(GEOM.centre[:, None, :] - GEOM.centre[None, :, :], axis=2)

    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.6),
                           gridspec_kw={"width_ratios": [1.45, 1]})
    a = ax[0]
    v, dd = np.abs(D[off]), euclid[off]
    sb, sc = same_b[off], same_c[off] & ~same_b[off]
    rng = np.random.default_rng(0)
    k = rng.choice(v.size, min(7000, v.size), replace=False)
    for mask, col, lab in (((~(sb | sc))[k], "#C3CDD2", "different CDU"),
                           ((sc)[k], MID, "same CDU, other branch"),
                           ((sb)[k], COOL, "same branch")):
        a.scatter(dd[k][mask], np.maximum(v[k][mask], 3e-7), s=1.8, alpha=0.4,
                  color=col, label=lab, edgecolors="none")
    a.set_yscale("log"); a.set_ylim(1.5e-7, 4e-1)
    a.set_xlabel("Euclidean distance (m)")
    a.set_ylabel("$|\\Delta T_{coolant}|$ (K)")
    a.set_title("not the floor plan", fontsize=9)
    a.legend(frameon=False, markerscale=4.5, fontsize=6.8, loc="lower left")
    a.text(0.97, 0.06, "zeros drawn\non the floor", transform=a.transAxes,
           fontsize=6.6, color="#7C8A90", ha="right")

    b = ax[1]
    groups = [np.abs(D[same_b]), np.abs(D[same_c & ~same_b])]
    bp = b.boxplot(groups, tick_labels=["same\nbranch", "same CDU,\nother branch"],
                   showfliers=False, patch_artist=True, widths=0.5,
                   medianprops=dict(color="black", lw=1.0))
    for patch, col in zip(bp["boxes"], (COOL, MID)):
        patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor("black")
    b.axhline(0.0, color=DARK, lw=0)
    b.set_ylabel("$|\\Delta T_{coolant}|$ (K)")
    b.set_title("the hydraulics", fontsize=9)
    b.text(0.5, -0.30, "different CDU: exactly 0", transform=b.transAxes,
           ha="center", fontsize=7.2, color="#7C8A90")
    fig.tight_layout()
    p = OUT / "L2_coupling.png"; fig.savefig(p); plt.close(fig); return p


def fig_matrix():
    """The influence matrix with racks ordered by loop, then manifold position."""
    D = _influence()
    order = np.lexsort((TOPO.manifold_dist_m, TOPO.branch_of_rack, TOPO.cdu_of_rack))
    M = np.abs(D[np.ix_(order, order)])
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(np.log10(np.maximum(M, 1e-7)), cmap="magma", aspect="equal",
                   vmin=-6, vmax=np.log10(max(M.max(), 1e-6)))
    # Loop boundaries.
    cdu_sorted = TOPO.cdu_of_rack[order]
    for edge in np.flatnonzero(np.diff(cdu_sorted)) + 1:
        ax.axhline(edge - 0.5, color="white", lw=0.7)
        ax.axvline(edge - 0.5, color="white", lw=0.7)
    ax.set_xlabel("source rack (ordered by loop)")
    ax.set_ylabel("affected rack")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("$\\log_{10}|\\Delta T_{coolant}|$ (K)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    p = OUT / "L3_matrix.png"; fig.savefig(p); plt.close(fig); return p


def fig_envelope():
    """Case temperature across the ASHRAE warm-water band, by pump speed."""
    waters = np.linspace(26, 42, 9)
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    for pump, col, ls in ((1.0, COOL, "-"), (0.8, MID, "--"), (0.55, WARM, ":")):
        peak = []
        for w in waters:
            s = TWIN.reset(np.full(N, 100.0), facility_water_c=float(w),
                           air_supply_c=22.0, pump_frac=pump)
            peak.append(s.case_temp_c.max())
        ax.plot(waters, peak, ls, color=col, lw=1.4, marker="o", ms=3,
                label=f"pumps at {pump:.0%}")
    ax.axhline(CFG.limits.case_temp_max_c, color=DARK, lw=1.0, ls="--")
    ax.text(26.3, CFG.limits.case_temp_max_c + 1.0, "85 °C case limit", fontsize=7.5)
    ax.set_xlabel("facility water temperature (°C)   —   ASHRAE W32 to W45")
    ax.set_ylabel("peak chip case temperature (°C)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    p = OUT / "L4_envelope.png"; fig.savefig(p); plt.close(fig); return p


def fig_branch():
    """Along a single branch manifold: what the cross-talk term actually does."""
    util = np.full(N, 90.0)
    rs = rack_power(util, np.full(N, 55.0), CFG.power)
    s = solve_coolant(TOPO, CFG.liquid, rs.total_power_w, 34.0, 0.85)
    order = TOPO.racks_on_branch(0)
    pos = TOPO.manifold_dist_m[order]

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.plot(pos, s.supply_temp_c[order], "o-", color=COOL, ms=3.4, lw=1.3,
            label="coolant supply")
    ax.plot(pos, s.return_temp_c[order], "s--", color=WARM, ms=3.4, lw=1.3,
            label="coolant return")
    ax.set_xlabel("distance along the branch manifold from the feed (m)")
    ax.set_ylabel("temperature (°C)")
    ax2 = ax.twinx()
    ax2.plot(pos, s.flow_kgs[order], "^-", color=MID, ms=3.4, lw=1.1,
             label="coolant flow")
    ax2.set_ylabel("flow (kg/s)", color=MID)
    ax2.tick_params(axis="y", labelcolor=MID)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center left", fontsize=7.6)
    rise = s.supply_temp_c[order][-1] - s.supply_temp_c[order][0]
    ax.set_title(f"supply warms {rise:.2f} K along the branch; flow falls "
                 f"{100*(1-s.flow_kgs[order][-1]/s.flow_kgs[order][0]):.0f}%",
                 fontsize=8.6)
    fig.tight_layout()
    p = OUT / "L5_branch.png"; fig.savefig(p); plt.close(fig); return p


if __name__ == "__main__":
    for f in (fig_topology, fig_coupling, fig_matrix, fig_envelope, fig_branch):
        print(f())
