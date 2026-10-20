#!/usr/bin/env python3
"""Readable SI sensitivity and public-endpoint figures; frozen CSVs are read-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

import inherited_submission_figures as inherited

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data/inherited"
AUDIT = ROOT.parents[1] / "outputs/round2/figures"
KEY = ["sweep", "policy", "true_hazard", "prior", "sentinel_size",
       "inspection_interval", "hazard_profile", "observation_factor"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.labelsize": 8.5, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, relative):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix("." + extension), dpi=180,
                    facecolor="white")
    plt.close(fig)


def full_grid():
    with (DATA / "paper4_policy_sensitivity_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    row_ids = [tuple(row[key] for key in KEY) for row in rows]
    assert len(rows) == 78 and len(set(row_ids)) == 78
    assert sum(row["sweep"] == "design" for row in rows) == 54
    assert all(int(row["episodes"]) == 300 for row in rows)
    plotted = []
    fig = plt.figure(figsize=(7.15, 6.55))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1],
                           left=.225, right=.945, bottom=.17, top=.94,
                           wspace=.88, hspace=.62)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    cmap = LinearSegmentedColormap.from_list("target_crossing", ["#fffdfa", "#f5c99a", "#ba4839", "#852c34"])
    priors = ["broad", "high_biased", "low_biased"]
    combinations = [(prior, gap) for prior in priors for gap in (2, 5, 10)]
    for index, hazard in enumerate((.006, .012)):
        matrix = np.zeros((9, 3))
        for i, (prior, gap) in enumerate(combinations):
            for j, size in enumerate((100, 300, 1000)):
                matches = [row for row in rows if row["sweep"] == "design"
                           and float(row["true_hazard"]) == hazard and row["prior"] == prior
                           and int(row["inspection_interval"]) == gap and int(row["sentinel_size"]) == size]
                assert len(matches) == 1
                row = matches[0]
                matrix[i, j] = 100 * float(row["episode_failure_rate"])
                plotted.append(dict(row, panel="ab"[index], plot_row=i, plot_column=j,
                                    plotted_percentage=matrix[i, j]))
        ax = axes[index]
        image = heatmap(ax, matrix, cmap)
        ax.set_yticks(range(9), [f"{p.replace('_biased', '-biased')} | {g} y" for p, g in combinations])
        ax.set_xticks(range(3), ["100", "300", "1,000"])
        ax.set_title(f"{'ab'[index]}  Annual hazard {hazard:.3f}", loc="left", pad=10)
        ax.set_xlabel("Sentinel trials")
    combinations = [(h, policy, profile) for h in (.006, .012)
                    for policy in ("posterior_mean", "posterior_conservative")
                    for profile in ("constant", "accelerating")]
    matrix = np.zeros((8, 3))
    for i, (hazard, policy, profile) in enumerate(combinations):
        for j, factor in enumerate((.8, 1, 1.2)):
            matches = [row for row in rows if row["sweep"] == "misspecification"
                       and float(row["true_hazard"]) == hazard and row["policy"] == policy
                       and row["hazard_profile"] == profile and float(row["observation_factor"]) == factor]
            assert len(matches) == 1
            row = matches[0]
            matrix[i, j] = 100 * float(row["episode_failure_rate"])
            plotted.append(dict(row, panel="c", plot_row=i, plot_column=j,
                                plotted_percentage=matrix[i, j]))
    ax = axes[2]
    heatmap(ax, matrix, cmap)
    ax.set_yticks(range(8), [f"{h:.3f} | {'PM' if p == 'posterior_mean' else 'Q95'} | {'const.' if prof == 'constant' else 'accel.'}"
                           for h, p, prof in combinations])
    ax.set_xticks(range(3), ["0.8", "1.0", "1.2"])
    ax.set_title("c  Observation-scale sensitivity", loc="left", pad=10)
    ax.set_xlabel("Observation-scale factor")
    cax = fig.add_axes([.225, .058, .72, .018])
    fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    cax.set_xlabel("Missions with any modeled target violation (%)", labelpad=5)
    assert sorted(tuple(row[key] for key in KEY) for row in plotted) == sorted(row_ids)
    AUDIT.mkdir(parents=True, exist_ok=True)
    with (AUDIT / "full_grid_plotted_rows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    (AUDIT / "full_grid_receipt.json").write_text(json.dumps({
        "status": "PASS_ALL_78_UNIQUE_ROWS", "design": 54, "misspecification": 24,
        "missions_per_cell": 300, "source_sha256": hashlib.sha256(
            (DATA / "paper4_policy_sensitivity_summary.csv").read_bytes()).hexdigest(),
        "source_values_changed": False}, indent=2) + "\n", encoding="utf-8")
    save(fig, "presentation_figures/policy_sensitivity_full_grid.pdf")


def heatmap(ax, matrix, cmap):
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if value > 60 else "#24313b")
    ax.tick_params(length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def public_endpoints():
    inherited.DATA = DATA
    qpcr = pd.read_csv(DATA / "paper4_dt4dds_qpcr_decay_timepoints.csv")
    summary = pd.read_csv(DATA / "paper4_dt4dds_qpcr_decay_summary.csv")
    loo = pd.read_csv(DATA / "paper4_dt4dds_qpcr_leave_one_pool_out.csv")
    detection = inherited.build_detection_source()
    fig = plt.figure(figsize=(7.15, 6.85))
    ax = fig.add_axes([.06, .82, .91, .14]); ax.axis("off")
    ax.set_title("a  Measurement endpoints", loc="left", pad=8)
    for x, title, description, color in [
        (0, "qPCR", "Amplifiable concentration\n(public measurements)", "#e8f3f8"),
        (.34, "Reference detection", "Post-workflow missing sequences\n(public measurements)", "#e7f4ef"),
        (.68, "Service model", "Erasure-budget success\n(synthetic archive)", "#fff3df")]:
        ax.add_patch(Rectangle((x, .04), .31, .77, transform=ax.transAxes,
                               facecolor=color, edgecolor="#b9c5ca", linewidth=.7))
        ax.text(x+.155, .61, title, transform=ax.transAxes, ha="center", weight="bold", fontsize=9)
        ax.text(x+.155, .25, description, transform=ax.transAxes, ha="center", fontsize=8, linespacing=1.5)
    axb = fig.add_axes([.105, .415, .435, .29])
    axc = fig.add_axes([.67, .415, .29, .29])
    xfit = np.linspace(0, 7.05, 150)
    keys = ["Genscript_GCall", "Genscript_GCfix", "Twist_GCall", "Twist_GCfix"]
    colors = ["#23658a", "#85afc5", "#28775c", "#9276ad"]
    for key, color in zip(keys, colors):
        group = qpcr[qpcr.pool_key == key].sort_values("time_days")
        mean = group.relative_concentration_mean.to_numpy(float)
        sd = group.relative_concentration_sd.to_numpy(float)
        lower = sd.copy()
        axb.errorbar(group.time_days, mean, yerr=np.vstack([lower, sd]),
                     color=color, marker="o", markersize=3, linestyle="none", capsize=2, linewidth=.8)
        slope = float(summary[summary.pool_key == key].log10_decay_slope_per_day.iloc[0])
        label = key.replace("_", " ").replace("Genscript", "Gen.")
        axb.plot(xfit, 10**(slope*xfit), color=color,
                 linestyle="--" if key.endswith("GCfix") else "-", linewidth=1.2, label=label)
        row = loo[loo.holdout_pool_key == key].iloc[0]
        axc.scatter(row.holdout_final_observed_relative_concentration,
                    row.holdout_final_predicted_relative_concentration, s=30, color=color,
                    marker="s" if key.endswith("GCfix") else "o", zorder=3)
    axb.set_yscale("log"); axb.set_xlim(-.15, 7.2); axb.set_ylim(.0018, 1.7)
    axb.set_title("b  Accelerated-aging qPCR", loc="left", pad=9)
    axb.set_xlabel("Aging time (days)"); axb.set_ylabel("Relative concentration")
    axb.legend(fontsize=8, ncol=2, frameon=False, loc="lower left", columnspacing=.8)
    axc.set_xscale("log"); axc.set_yscale("log")
    vals = np.r_[loo.holdout_final_observed_relative_concentration,
                 loo.holdout_final_predicted_relative_concentration]
    lo = 10**np.floor(np.log10(vals.min())-.05)
    hi = 10**np.ceil(np.log10(vals.max())+.05)
    axc.plot([lo, hi], [lo, hi], linestyle="--", color="#aab4bc", linewidth=.8)
    axc.set_xlim(lo, hi); axc.set_ylim(lo, hi)
    axc.set_title("c  Held-out-pool prediction", loc="left", pad=9)
    axc.set_xlabel("Observed concentration"); axc.set_ylabel("LOO prediction")
    max_factor = np.maximum(loo.holdout_final_observed_relative_concentration /
                            loo.holdout_final_predicted_relative_concentration,
                            loo.holdout_final_predicted_relative_concentration /
                            loo.holdout_final_observed_relative_concentration).max()
    axc.text(.04, .96, f"Final-point error\nup to {max_factor:.2f}×", transform=axc.transAxes,
             va="top", fontsize=8, linespacing=1.4)
    fig.text(.105, .298, "d  Post-workflow missing-reference fractions", weight="bold", fontsize=9)
    for i, (study, title, color, labels) in enumerate([
        ("Genscript_GCall", "Genscript", colors[0], ["0", "2", "4", "7"]),
        ("Twist_GCall", "Twist", colors[2], ["0", "2", "4", "7"]),
        ("Meiser", "Meiser", colors[3], ["U", "A", "R"]),
        ("Song", "Song", "#bd6c25", ["0", "28", "56", "70"]) ]):
        mini = fig.add_axes([.105+i*.225, .10, .16, .155])
        group = detection[detection.study == study].sort_values("condition_order")
        mini.plot(range(len(group)), 100*group.missing_fraction, color=color, marker="o", markersize=4, linewidth=1.2)
        mini.set_yscale("log"); mini.set_ylim(.02, 1.8)
        mini.set_yticks([.03, .1, .3, 1], ["0.03", "0.1", "0.3", "1"] if i == 0 else [])
        mini.set_xticks(range(len(group)), labels)
        mini.set_title(title, loc="left", fontsize=8.5, pad=6)
        mini.set_xlabel("Condition", fontsize=8)
        if i == 0: mini.set_ylabel("Missing references (%)")
        style_axis(mini)
    for item in (axb, axc):
        style_axis(item)
        item.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axc.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    save(fig, "figures/public_endpoint_mismatch_audit.pdf")


def style_axis(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", labelsize=8)
    ax.grid(axis="y", color="#e1e6e8", linewidth=.5)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["grid", "public", "all"], default="all")
    args = parser.parse_args()
    if args.only in ("grid", "all"): full_grid()
    if args.only in ("public", "all"): public_endpoints()
