#!/usr/bin/env python3
"""Rebuild the review Figure 1 and selected Figure 4 panels from stated models/CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data/inherited"
FIGURES = ROOT / "figures"
PRESENTATION = ROOT / "presentation_figures"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                     "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def stage(ax, x, y, w, h, title, expression, face, edge):
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes,
                           facecolor=face, edgecolor=edge, linewidth=1))
    ax.text(x+w/2, y+h*.70, title, ha="center", va="center", transform=ax.transAxes,
            fontsize=9, fontweight="bold")
    ax.text(x+w/2, y+h*.32, expression, ha="center", va="center", transform=ax.transAxes,
            fontsize=9)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
                 arrowstyle="-|>", mutation_scale=11, linewidth=1.0,
                 linestyle="--" if dashed else "-", color="#59636d"))


def framework():
    fig, ax = plt.subplots(figsize=(7.15, 3.25))
    ax.axis("off")
    w, h = .273, .22
    xs, ys = (.035, .365, .695), (.70, .28)
    stages = [
        ("Inspection count", r"$Y_j$ of $n_j$ model trials", "#e8f3f8", "#2878a6"),
        ("Observation bound", r"$U_j(Y_j;\alpha_j)$", "#e8f3f8", "#2878a6"),
        ("Current-state bound", r"$\bar{q}_j=\phi_j(U_j)$", "#e7f4ef", "#278066"),
        ("Next-state bound", r"$\widehat{q}^{U}_{j+1}=T_j(\bar{q}_j)$", "#e7f4ef", "#278066"),
        ("Service bound", r"$L_j$ evaluated at $\widehat{q}^{U}_{j+1}$", "#e7f4ef", "#278066"),
        ("Screen passage", r"$P_j:\ L_j\geq r_{\star,j}$", "#fff3df", "#ad7327"),
    ]
    positions = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    for (row, col), (title, expression, face, edge) in zip(positions, stages):
        stage(ax, xs[col], ys[row], w, h, title, expression, face, edge)
    for col in (0, 1):
        arrow(ax, (xs[col]+w+.005, ys[0]+h/2), (xs[col+1]-.01, ys[0]+h/2))
        arrow(ax, (xs[col+1]-.01, ys[1]+h/2), (xs[col]+w+.005, ys[1]+h/2))
    arrow(ax, (xs[2]+w/2, ys[0]-.01), (xs[2]+w/2, ys[1]+h+.01))
    ax.text(.5, .115, r"Final authorization: $N_j\subseteq P_j$." "\n"
            "The base policy and other prespecified gates may withhold deferral.",
            ha="center", va="center", transform=ax.transAxes, fontsize=8.5, linespacing=1.6)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "nc_v8_framework.pdf", bbox_inches="tight", pad_inches=.08)
    fig.savefig(FIGURES / "nc_v8_framework.svg", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def records(name):
    with (DATA / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def selected_assurance():
    source = records("paper4_policy_sensitivity_summary.csv")
    design = [r for r in source if r["sweep"] == "design" and float(r["true_hazard"]) == .012]
    mismatch = [r for r in source if r["sweep"] == "misspecification"]
    priors = ["broad", "high_biased", "low_biased"]
    intervals = [2, 5, 10]
    sentinels = [100, 300, 1000]
    left_rows = [(prior, gap) for prior in priors for gap in intervals]
    left = np.array([[100*float(next(r for r in design if r["prior"] == prior and int(r["inspection_interval"]) == gap and int(r["sentinel_size"]) == n)["episode_failure_rate"])
                      for n in sentinels] for prior, gap in left_rows])
    right_rows = [(h, policy, profile) for h in (.006, .012) for policy in ("posterior_mean", "posterior_conservative") for profile in ("constant", "accelerating")]
    factors = [.8, 1., 1.2]
    right = np.array([[100*float(next(r for r in mismatch if float(r["true_hazard"]) == h and r["policy"] == policy and r["hazard_profile"] == profile and float(r["observation_factor"]) == factor)["episode_failure_rate"])
                       for factor in factors] for h, policy, profile in right_rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), constrained_layout=True)
    for ax, matrix, labels, columns, title, xlabel in [
        (axes[0], left, [f"{p.replace('_biased','-biased')} | {g} y" for p, g in left_rows], [str(v) for v in sentinels], "a  Design sensitivity at hazard 0.012", "Sentinel size"),
        (axes[1], right, [f"{h:.3f} | {'PM' if p == 'posterior_mean' else 'Q95'} | {'const.' if prof == 'constant' else 'accel.'}" for h, p, prof in right_rows], [f"{f:.1f}" for f in factors], "b  Observation-scale sensitivity", "Observation-scale factor"),
    ]:
        image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_title(title, loc="left")
        ax.set_xticks(range(3), columns)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel(xlabel)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if val > 55 else "#26323a")
    bar = fig.colorbar(image, ax=axes, location="bottom", shrink=.55, pad=.12)
    bar.set_label("Missions with a target violation (%)")
    PRESENTATION.mkdir(parents=True, exist_ok=True)
    fig.savefig(PRESENTATION / "policy_assurance_limits.pdf", bbox_inches="tight", pad_inches=.08)
    fig.savefig(PRESENTATION / "policy_assurance_limits.svg", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def policy_comparison():
    rows = records("paper4_seed_uncertainty_summary.csv")
    order = ["fixed_schedule", "oracle_true_hazard", "posterior_mean", "posterior_conservative"]
    labels = ["Fixed schedule", "Full-information oracle", "Posterior mean", "Conservative posterior"]
    colors = ["#626a70", "#23658a", "#bd6c25", "#28775c"]
    markers = ["s", "D", "o", "^"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0))
    for policy, label, color, marker in zip(order, labels, colors, markers):
        group = sorted((r for r in rows if r["policy"] == policy), key=lambda r: float(r["true_hazard"]))
        x = [float(r["true_hazard"]) for r in group]
        for ax, metric, lo, hi, factor in [
            (axes[0], "mean_refreshes", "mean_refreshes_ci95_low", "mean_refreshes_ci95_high", 1),
            (axes[1], "episode_failure_rate", "episode_failure_rate_ci95_low", "episode_failure_rate_ci95_high", 100),
        ]:
            y = np.array([factor*float(r[metric]) for r in group])
            lower = np.array([factor*float(r[lo]) for r in group])
            upper = np.array([factor*float(r[hi]) for r in group])
            ax.errorbar(x, y, yerr=np.vstack((y-lower, upper-y)), marker=marker,
                        color=color, linewidth=1.2, markersize=4, capsize=2,
                        label=label)
    axes[0].set_title("a  Refresh burden", loc="left")
    axes[0].set_ylabel("Mean refreshes / 60-year mission")
    axes[0].set_ylim(-.3, 12.8)
    axes[1].set_title("b  Modeled service-target crossings", loc="left")
    axes[1].set_ylabel("Missions with any crossing (%)")
    axes[1].set_yscale("symlog", linthresh=.1)
    axes[1].set_ylim(-.025, 125)
    axes[1].set_yticks([0, .1, 1, 10, 100], ["0", "0.1", "1", "10", "100"])
    for ax in axes:
        ax.set_xticks([.002, .006, .012], ["0.002", "0.006", "0.012"])
        ax.set_xlabel("Annual synthetic hazard")
        ax.grid(axis="y", linewidth=.5, color="#d9dfe1")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(.5, .98), fontsize=8, frameon=False)
    fig.subplots_adjust(left=.11, right=.98, top=.70, bottom=.20, wspace=.33)
    fig.savefig(FIGURES / "policy_assurance_policy_comparison.pdf", bbox_inches="tight", pad_inches=.08)
    fig.savefig(FIGURES / "policy_assurance_policy_comparison.svg", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["framework", "selected", "comparison", "all"], default="all")
    args = parser.parse_args()
    for name, function in [("framework", framework), ("selected", selected_assurance), ("comparison", policy_comparison)]:
        if args.only in (name, "all"):
            function()
            print(f"Rebuilt {name} from equations/frozen CSV")
