#!/usr/bin/env python3
"""Build the Paper 4 service-model and operational-limit figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "navy": "#234E70",
    "blue": "#3B82A0",
    "teal": "#2A9D8F",
    "orange": "#E07A35",
    "red": "#B64B4B",
    "purple": "#7663A5",
    "gray": "#667085",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(data_dir: Path, figure_dir: Path) -> None:
    service = read_csv(data_dir / "paper4_service_model_sensitivity.csv")
    policy = read_csv(data_dir / "paper4_extended_policy_summary.csv")

    service_selected = [
        row for row in service if float(row["recoverability_target"]) == 0.99
    ]
    dependence = [row for row in policy if row["family"] == "dependence"]
    nonideal = [row for row in policy if row["family"] == "nonideal_refresh"]
    common_design = [
        row
        for row in policy
        if row["family"] == "common_design_policy"
        and abs(float(row["true_hazard_per_year"]) - 0.009) < 1e-12
    ]
    write_csv(data_dir / "source_data_figure4_service.csv", service_selected)
    write_csv(data_dir / "source_data_figure4_dependence.csv", dependence)
    write_csv(data_dir / "source_data_figure4_nonideal_refresh.csv", nonideal)
    write_csv(
        data_dir / "source_data_figure4_common_design_policies.csv",
        common_design,
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.1,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 6.40), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # a: service threshold contraction under a modest declared dependence.
    ns = [1_000, 10_000, 100_000]
    taus = [0.05, 0.10, 0.15, 0.20]
    ratio = np.full((len(ns), len(taus)), np.nan)
    for row in service_selected:
        i = ns.index(int(row["archive_oligos"]))
        j = taus.index(float(row["erasure_tolerance"]))
        ratio[i, j] = 100.0 * float(row["robust_to_independent_threshold_ratio"])
    image_a = ax_a.imshow(
        ratio,
        cmap="Blues",
        vmin=min(90.0, float(np.nanmin(ratio))),
        vmax=100.0,
        aspect="auto",
    )
    for i in range(len(ns)):
        for j in range(len(taus)):
            ax_a.text(j, i, f"{ratio[i, j]:.1f}", ha="center", va="center", fontsize=7.0)
    ax_a.set_xticks(range(len(taus)), [f"{value:.2f}" for value in taus])
    ax_a.set_yticks(range(len(ns)), [f"{value:,}" for value in ns])
    ax_a.set_xlabel("Erasure tolerance")
    ax_a.set_ylabel("Modeled oligos")
    ax_a.set_title("Threshold retained under the beta-binomial payload model (%)", pad=5)
    cbar_a = fig.colorbar(image_a, ax=ax_a, fraction=0.046, pad=0.025)
    cbar_a.set_label("Beta-binomial / binomial threshold (%)")

    # b: turn the dependence stress into a usable service certificate.
    policies = ["certificate_independent", "certificate_robust"]
    labels = ["Independent-binomial certificate", "Fixed-ICC beta-binomial certificate"]
    markers = ["o", "s"]
    colors = [COLORS["orange"], COLORS["navy"]]
    for name, label, marker, color in zip(policies, labels, markers, colors):
        rows = sorted(
            [row for row in dependence if row["policy"] == name],
            key=lambda row: float(row["actual_rho"]),
        )
        xs = [max(float(row["actual_rho"]), 2e-6) for row in rows]
        ys = [float(row["mean_refresh_attempts"]) for row in rows]
        ax_b.plot(xs, ys, marker=marker, color=color, lw=1.4, ms=4.2, label=label)
    ax_b.set_xscale("log")
    ax_b.set_xticks([2e-6, 1e-5, 1e-4, 1e-3], ["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$"])
    ax_b.set_xlabel("Payload-erasure intraclass correlation")
    ax_b.set_ylabel("Mean refresh attempts / 60 y")
    ax_b.set_title("Dependence changes the actionable threshold", pad=5)
    ax_b.legend(frameon=False, loc="best")
    ax_b.grid(axis="y", color="#D0D5DD", lw=0.55, alpha=0.7)

    # c: violation duration induced by nonideal refresh.
    q0s = [0.0, 0.005, 0.01, 0.02]
    eps = [0.0, 1e-4, 1e-3, 1e-2]
    violation = np.full((len(q0s), len(eps)), np.nan)
    for row in nonideal:
        i = q0s.index(float(row["post_refresh_residual_dropout"]))
        j = eps.index(float(row["refresh_failure_probability"]))
        violation[i, j] = 100.0 * float(row["mean_continuous_time_violation_fraction"])
    image_c = ax_c.imshow(violation, cmap="OrRd", vmin=0.0, aspect="auto")
    for i in range(len(q0s)):
        for j in range(len(eps)):
            ax_c.text(j, i, f"{violation[i, j]:.3f}", ha="center", va="center", fontsize=7.0)
    ax_c.set_xticks(range(len(eps)), ["0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
    ax_c.set_yticks(range(len(q0s)), [f"{value:.3f}" for value in q0s])
    ax_c.set_xlabel("Unobserved refresh-failure probability")
    ax_c.set_ylabel("Post-refresh residual dropout")
    ax_c.set_title("Refresh imperfections escape decision-error control", pad=5)
    cbar_c = fig.colorbar(image_c, ax=ax_c, fraction=0.046, pad=0.025)
    cbar_c.set_label("Target-violation time (%)")

    # d: common-design independent cohorts at the informative intermediate hazard.
    # Category is encoded primarily by color; marker shape is secondary. Marker
    # area is fixed and no longer encodes inspection burden.
    styles = {
        "fixed_5y": (COLORS["gray"], "o", COLORS["gray"], 38),
        "fixed_10y": (COLORS["gray"], "o", COLORS["gray"], 38),
        "oracle_service_model": ("#344054", "o", "white", 42),
        "naive_plugin": (COLORS["orange"], "^", COLORS["orange"], 42),
        "beta_sentinel_q95": (COLORS["orange"], "^", "white", 42),
        "posterior_hazard_q95": (COLORS["orange"], "o", COLORS["orange"], 46),
        "certificate_independent": (COLORS["navy"], "s", "white", 48),
        "certificate_robust": (COLORS["navy"], "s", COLORS["navy"], 60),
    }
    plot_order = {
        "fixed_5y": 0,
        "fixed_10y": 1,
        "oracle_service_model": 2,
        "naive_plugin": 3,
        "beta_sentinel_q95": 4,
        "posterior_hazard_q95": 5,
        "certificate_independent": 6,
        "certificate_robust": 7,
    }
    points: dict[str, tuple[float, float]] = {}
    for row in sorted(common_design, key=lambda item: plot_order[item["policy"]]):
        name = row["policy"]
        x = float(row["mean_refresh_attempts"])
        y = 100.0 * float(row["episode_unsafe_deferral_rate"])
        points[name] = (x, y)
        edgecolor, marker, facecolor, size = styles[name]
        ax_d.scatter(
            x,
            y,
            marker=marker,
            s=size,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.15,
            zorder=3 + plot_order[name],
        )

    label_box = {"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.86}
    annotations = {
        "fixed_10y": ("Fixed 10 y", (8, -8), COLORS["gray"], "left"),
        "naive_plugin": ("Plug-in", (5, 7), COLORS["orange"], "left"),
        "beta_sentinel_q95": ("Beta sentinel 95%", (-4, 9), COLORS["orange"], "right"),
        "posterior_hazard_q95": ("Posterior 95%", (-22, 16), COLORS["orange"], "right"),
        "certificate_robust": (
            "Zero observed in certificates,\nfixed 5 y and oracle",
            (-22, -30),
            COLORS["navy"],
            "right",
        ),
    }
    for name, (label, offset, color, alignment) in annotations.items():
        ax_d.annotate(
            label,
            points[name],
            xytext=offset,
            textcoords="offset points",
            fontsize=8.2,
            color=color,
            ha=alignment,
            va="center",
            bbox=label_box,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55},
        )
    ax_d.set_xlabel("Mean refresh attempts / 60 y")
    ax_d.set_ylabel("under-target authorization episodes (%)")
    ax_d.set_title("Risk, refresh, and inspection-panel trade-offs", pad=5)
    ax_d.set_yscale("symlog", linthresh=0.1, linscale=0.8)
    ax_d.set_ylim(-0.30, 150.0)
    ax_d.set_yticks([0.0, 0.1, 1.0, 5.0, 10.0, 100.0])
    ax_d.set_yticklabels(["0", "0.1", "1", "5", "10", "100"])
    ax_d.grid(color="#D0D5DD", lw=0.55, alpha=0.7)

    for label, ax in zip("abcd", axes.flat):
        ax.text(
            -0.22,
            1.10,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    figure_dir.mkdir(parents=True, exist_ok=True)
    base = figure_dir / "service_model_operational_limits"
    # Preserve the shared >=5-pt final-text gate after SVG/PDF conversion.
    for artist in fig.findobj():
        if hasattr(artist, "get_fontsize") and hasattr(artist, "set_fontsize"):
            if artist.get_fontsize() < 6.6:
                artist.set_fontsize(6.6)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print("PASS_PAPER4_EXTENDED_FIGURE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "papers/paper4_adaptive_archive_maintenance/figures",
    )
    args = parser.parse_args()
    build(args.data_dir, args.figure_dir)


if __name__ == "__main__":
    main()
