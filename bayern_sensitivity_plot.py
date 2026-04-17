"""
Bayern Munich weight-sensitivity figure.

Reads bayern_sweep.json and produces a multi-panel figure:
  (A) Objective components (KPI, NET, COH, Total) along the gamma-sweep.
  (B) Lineup overlap (out of 11) with the talent-heavy and chemistry-heavy
      reference XIs, as a function of gamma.
  (C, D) Side-by-side pitch diagrams of talent-heavy vs. chemistry-heavy XIs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Circle, Rectangle

REPO_ROOT = Path(__file__).resolve().parent
IN_PATH = REPO_ROOT / "bayern_sweep.json"
OUT_PATH = REPO_ROOT / "paper" / "figs" / "bayern_sensitivity.pdf"

PITCH_GREEN = "#e8efe3"
LINE_WHITE  = "#ffffff"

COL_KPI = "#2E86AB"   # talent
COL_NET = "#A23B72"   # network
COL_COH = "#F18F01"   # cohesion
COL_TOT = "#3a3a3a"


def draw_pitch_lineup(ax, lineup: dict, title: str, accent: str) -> None:
    ax.set_facecolor(PITCH_GREEN)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=LINE_WHITE, linewidth=2))
    ax.plot([0, 1], [0.5, 0.5], color=LINE_WHITE, linewidth=1.2)
    ax.add_patch(Circle((0.5, 0.5), 0.08, fill=False, edgecolor=LINE_WHITE, linewidth=1))
    ax.add_patch(Rectangle((0.30, 0.00), 0.40, 0.12, fill=False, edgecolor=LINE_WHITE, linewidth=1))
    ax.add_patch(Rectangle((0.30, 0.88), 0.40, 0.12, fill=False, edgecolor=LINE_WHITE, linewidth=1))

    coords = {
        "GK":  (0.50, 0.08),
        "LB":  (0.12, 0.24), "CB1": (0.36, 0.18), "CB2": (0.64, 0.18), "RB": (0.88, 0.24),
        "DM":  (0.50, 0.36),
        "CM":  (0.30, 0.52), "AM":  (0.70, 0.52),
        "LW":  (0.15, 0.76), "ST":  (0.50, 0.88), "RW":  (0.85, 0.76),
    }
    for slot, (x, y) in coords.items():
        name = lineup.get(slot, "")
        ax.add_patch(Circle((x, y), 0.045, facecolor=accent, edgecolor="white", lw=1.5, zorder=3))
        ax.text(x, y, slot, ha="center", va="center",
                color="white", fontsize=7.0, fontweight="bold", zorder=4)
        ax.text(x, y - 0.075, name, ha="center", va="top",
                fontsize=7.5, zorder=4, color="#111")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=6)


def main():
    with open(IN_PATH) as f:
        data = json.load(f)

    sweep = data["sweep"]
    gammas = np.array([s["gamma"] for s in sweep])
    alphas = np.array([s["alpha"] for s in sweep])
    kpi   = np.array([s["kpi_norm"] for s in sweep])
    net   = np.array([s["net_norm"] for s in sweep])
    coh   = np.array([s["coh_norm"] for s in sweep])
    total = np.array([s["total"]    for s in sweep])

    talent_lineup = data["corners"]["talent_heavy"]["lineup"]
    chem_lineup   = data["corners"]["chemistry_heavy"]["lineup"]
    default_lineup = data["corners"]["default"]["lineup"]

    talent_players = set(talent_lineup.values())
    chem_players   = set(chem_lineup.values())

    overlap_talent = np.array([len(set(s["lineup"].values()) & talent_players) for s in sweep])
    overlap_chem   = np.array([len(set(s["lineup"].values()) & chem_players)   for s in sweep])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(11.5, 10.5))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.05, 0.95, 1.35],
                           hspace=0.55, wspace=0.18)

    # ---- Panel A: trajectories ---------------------------------
    axA = fig.add_subplot(gs[0, :])
    axA.plot(gammas, kpi,   "o-",  color=COL_KPI, label="KPI  (talent aggregate)", linewidth=2, markersize=6)
    axA.plot(gammas, net,   "s-",  color=COL_NET, label="NET  (network centrality)", linewidth=2, markersize=6)
    axA.plot(gammas, coh,   "^-",  color=COL_COH, label="COH  (passing cohesion)",  linewidth=2, markersize=6)
    axA.plot(gammas, total, "D--", color=COL_TOT, label="Total objective",          linewidth=1.5, markersize=5, alpha=0.7)

    axA.axvline(0.30, color="gray", linestyle=":", alpha=0.7, linewidth=1.2)
    axA.annotate("default\n(α=0.60, γ=0.30)",
                 xy=(0.30, 0.74), xytext=(0.40, 0.80),
                 fontsize=8.5, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    axA.set_xlabel(r"cohesion weight $\gamma$   (with $\alpha = 0.90 - \gamma$, $\beta = 0.10$)",
                   fontsize=10)
    axA.set_ylabel("normalized component value", fontsize=10)
    axA.set_title("(A)  Objective-component trajectories under the $\\gamma$-sweep",
                  fontsize=11, loc="left", fontweight="bold", pad=6)
    axA.set_ylim(0.0, 0.92)
    axA.legend(loc="center right", frameon=False, fontsize=9, ncol=1,
               bbox_to_anchor=(1.0, 0.50))
    axA.grid(alpha=0.25)

    # ---- Panel B: XI overlap -----------------------------------
    axB = fig.add_subplot(gs[1, :])
    xs = np.arange(len(gammas))
    width = 0.38
    axB.bar(xs - width/2, overlap_talent, width, color=COL_KPI,
            label="overlap with Talent-heavy XI", edgecolor="white", linewidth=0.5)
    axB.bar(xs + width/2, overlap_chem,   width, color=COL_COH,
            label="overlap with Chemistry-heavy XI", edgecolor="white", linewidth=0.5)
    axB.axhline(11, color="#bbb", linewidth=0.6, linestyle="--")
    for i, (t, c) in enumerate(zip(overlap_talent, overlap_chem)):
        axB.text(i - width/2, t + 0.15, str(t), ha="center", va="bottom", fontsize=8, color=COL_KPI)
        axB.text(i + width/2, c + 0.15, str(c), ha="center", va="bottom", fontsize=8, color=COL_COH)
    axB.set_xticks(xs)
    axB.set_xticklabels([f"{g:.2f}" for g in gammas])
    axB.set_xlabel(r"cohesion weight $\gamma$", fontsize=10)
    axB.set_ylabel("# shared players  (out of 11)", fontsize=10)
    axB.set_ylim(0, 12.5)
    axB.set_title(r"(B)  Lineup phase transition: XI overlap with the two reference configurations",
                  fontsize=11, loc="left", fontweight="bold", pad=6)
    axB.legend(loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    axB.grid(axis="y", alpha=0.25)

    # ---- Panel C / D: pitch XIs --------------------------------
    axC = fig.add_subplot(gs[2, 0])
    draw_pitch_lineup(axC, talent_lineup,
                      "(C)  Talent-heavy XI   (α=0.90, β=0.05, γ=0.05)",
                      accent=COL_KPI)
    axD = fig.add_subplot(gs[2, 1])
    draw_pitch_lineup(axD, chem_lineup,
                      "(D)  Chemistry-heavy XI   (α=0.05, β=0.05, γ=0.90)",
                      accent=COL_COH)

    fig.suptitle("FC Bayern Munich — Sensitivity of the Optimal XI to Objective-Weight Choice",
                 fontsize=13, fontweight="bold", y=0.995)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=220)
    fig.savefig(OUT_PATH.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print(f"Wrote: {OUT_PATH}")
    print(f"Wrote: {OUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
