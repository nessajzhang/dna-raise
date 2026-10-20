"""Render the V8.8.9 Figure 3 from frozen V8.7 aggregate Source Data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
OUT = ROOT / "presentation_figures" / "nc_v8_8_9_target_use.pdf"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


target = read_csv("Figure_3_target_use_source_data.csv")
burden = read_csv("Figure_3a_risk_burden_source_data.csv")
by_panel = {panel: [row for row in target if row["panel"] == panel] for panel in "abcd"}
assert len(by_panel["b"]) == 3 and len(by_panel["c"]) == 2 and len(by_panel["d"]) == 1
assert len(burden) == 4

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

blue = "#23658a"
orange = "#bd6c25"
green = "#28775c"
gray = "#626a70"
red = "#a93936"

fig = plt.figure(figsize=(6.8, 5.0))
grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.51, wspace=0.44)
ax_a = fig.add_subplot(grid[0, :])
ax_b = fig.add_subplot(grid[1, 0])
ax_c = fig.add_subplot(grid[1, 1])
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.95)

# The periodic comparator has no deferral authorization by design. Its
# 0/20,000 test bound is an uncertainty bound, not the value of its risk point.
styles = {
    "Five-year periodic": (gray, "s", "Five-year periodic\n(structural zero)"),
    "Posterior mean": (orange, "D", "Posterior mean"),
    "Posterior Q95": (green, "^", "Posterior Q95"),
    "Selected screen": (blue, "o", "Selected screen"),
}
for row in burden:
    name = row["policy"]
    color, marker, label = styles[name]
    x = float(row["mean_refreshes"])
    y = 0.0 if name == "Five-year periodic" else float(row["max_hazard_specific_upper_percent"])
    ax_a.scatter(x, y, s=55, color=color, marker=marker, zorder=4)
    offset = (-7, 8) if name == "Five-year periodic" else (7, 7)
    ax_a.annotate(
        label, (x, y), xytext=offset, textcoords="offset points",
        ha="right" if name == "Five-year periodic" else "left",
        va="bottom", color=color, fontsize=8,
    )
ax_a.axhline(5, color=red, linestyle="--", linewidth=0.9)
ax_a.text(11.45, 5.8, "5% gate", color=red, ha="right", fontsize=8)
ax_a.set_yscale("symlog", linthresh=0.04)
ax_a.set_xlim(4.6, 11.6)
ax_a.set_ylim(-0.013, 90)
ax_a.set_yticks([0, 0.1, 1, 5, 10, 50])
ax_a.set_yticklabels(["0", "0.1", "1", "5", "10", "50"])
ax_a.set_xlabel("Mean refreshes per 60-year mission")
ax_a.set_ylabel("Maximum hazard-specific\nupper bound (%)")
ax_a.set_title("a  Risk versus refresh burden", loc="left", fontweight="bold")
ax_a.grid(axis="y", color="#d8dddf", linewidth=0.6)

risk_rows = by_panel["b"]
positions = np.arange(3)
for pos, row in zip(positions, risk_rows):
    observed = float(row["point_estimate"])
    upper = float(row["upper_bound"])
    count = int(row["numerator"])
    ax_b.plot([pos, pos], [observed, upper], color=blue, linewidth=1.8, zorder=2)
    ax_b.scatter([pos], [observed], s=28, color=blue, zorder=3)
    ax_b.scatter([pos], [upper], marker="_", s=140, color="black", linewidths=1.6, zorder=4)
    ax_b.annotate(f"{count}/20,000", (pos, upper), xytext=(0, 7),
                  textcoords="offset points", ha="center", fontsize=8)
ax_b.axhline(5, color=red, linestyle="--", linewidth=0.9)
ax_b.set_yscale("symlog", linthresh=0.035)
ax_b.set_ylim(-0.01, 11)
ax_b.set_yticks([0, 0.1, 1, 5, 10])
ax_b.set_yticklabels(["0", "0.1", "1", "5", "10"])
ax_b.set_xlim(-0.38, 2.38)
ax_b.set_xticks(positions, [f"{float(row['x']):.3f}" for row in risk_rows])
ax_b.set_xlabel("Annual hazard")
ax_b.set_ylabel("Error rate / upper bound (%)")
ax_b.set_title("b  Hazard-specific error", loc="left", fontweight="bold")
ax_b.grid(axis="y", color="#d8dddf", linewidth=0.6)

support = by_panel["c"]
labels = ["Eligible passage", "Final authorization"]
for pos, row in enumerate(support):
    point = float(row["point_estimate"])
    lower = float(row["lower_bound"])
    floor = float(row["threshold"])
    color = blue if pos == 0 else green
    ax_c.plot([lower, point], [pos, pos], color=color, linewidth=3, zorder=2)
    ax_c.scatter([point], [pos], color=color, s=35, zorder=3)
    ax_c.scatter([lower], [pos], color="black", marker="|", s=125, linewidths=1.6, zorder=4)
    ax_c.scatter([floor], [pos], color=red, marker="D", s=28, zorder=4)
    unit = "lambda = 0.002 only" if pos == 0 else "all hazard strata"
    count = f"{int(row['numerator']):,}/{int(row['denominator']):,}"
    ax_c.text(0, pos - 0.32, f"{labels[pos]} ({unit})", fontsize=7.8)
    ax_c.text(0, pos + 0.22,
              f"{count}; {point:.2f}% (LB {lower:.2f}%)", fontsize=7.5)
ax_c.set_xlim(0, 100)
ax_c.set_ylim(1.48, -0.55)
ax_c.set_yticks([])
ax_c.set_xlabel("Candidates or decisions (%)")
ax_c.set_title("c  Screen use", loc="left", fontweight="bold")
ax_c.grid(axis="x", color="#d8dddf", linewidth=0.6)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print(OUT)
