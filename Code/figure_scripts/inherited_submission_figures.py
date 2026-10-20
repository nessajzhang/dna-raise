#!/usr/bin/env python3
"""Build the five V6 submission figures from audited CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from make_submission_figures import (
    LABEL_BBOX,
    PALETTE,
    add_panel_label,
    setup_style,
    style_axis,
    subtitle,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "papers" / "paper4_adaptive_archive_maintenance" / "figures"

POLICY_ORDER = [
    "fixed_schedule",
    "oracle_true_hazard",
    "posterior_mean",
    "posterior_conservative",
]
POLICY_LABELS = {
    "fixed_schedule": "Fixed schedule",
    "oracle_true_hazard": "Full-information oracle",
    "posterior_mean": "Posterior mean",
    "posterior_conservative": "Conservative posterior",
}
POLICY_COLORS = {
    "fixed_schedule": PALETTE["neutral_2"],
    "oracle_true_hazard": PALETTE["blue"],
    "posterior_mean": PALETTE["gold"],
    "posterior_conservative": PALETTE["teal"],
}
POLICY_MARKERS = {
    "fixed_schedule": "s",
    "oracle_true_hazard": "o",
    "posterior_mean": "^",
    "posterior_conservative": "D",
}

POOL_COLORS = {
    "Genscript_GCall": PALETTE["blue"],
    "Genscript_GCfix": PALETTE["blue_mid"],
    "Twist_GCall": PALETTE["teal"],
    "Twist_GCfix": PALETTE["violet"],
}
POOL_LABELS = {
    "Genscript_GCall": "Genscript GCall",
    "Genscript_GCfix": "Genscript GCfix",
    "Twist_GCall": "Twist GCall",
    "Twist_GCfix": "Twist GCfix",
}

FAILURE_CMAP = LinearSegmentedColormap.from_list(
    "assurance_failure",
    ["#FFFDFC", PALETTE["gold_soft"], PALETTE["salmon"], PALETTE["red"]],
)
EPSILON = 1e-12  # strictly positive guard for logarithmic display transforms


def save_paper4_figure(fig: mpl.figure.Figure, path: Path) -> None:
    """Export a fixed 183-mm canvas so panel labels cannot widen the PDF."""
    stem = path.with_suffix("")
    fig.patch.set_facecolor("white")
    # Keep the smallest final vector text above the 5-pt publication gate.
    # Matplotlib's SVG conversion is slightly smaller than the point value;
    # 6.6 pt is the conservative lower source size for this export route.
    for text in fig.findobj(mpl.text.Text):
        if text.get_fontsize() < 6.6:
            text.set_fontsize(6.6)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=450)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)


def plain_log_tick(value: float, _position: int) -> str:
    """Use editable plain labels instead of sub-5-pt mathtext exponents."""
    return f"{value:g}"


def export_source_data(
    qpcr: pd.DataFrame,
    loo: pd.DataFrame,
    detection: pd.DataFrame,
    seed: pd.DataFrame,
    design: pd.DataFrame,
    misspecification: pd.DataFrame,
) -> None:
    """Export panel-level tables without changing source values."""
    qpcr.to_csv(DATA / "paper4_figure1_qpcr_source.csv", index=False)
    loo.to_csv(DATA / "paper4_figure1_transfer_source.csv", index=False)
    detection.to_csv(DATA / "paper4_figure1_detection_source.csv", index=False)
    seed.to_csv(DATA / "paper4_figure2_seed_source.csv", index=False)
    design.to_csv(DATA / "paper4_figure2_design_source.csv", index=False)
    misspecification.to_csv(
        DATA / "paper4_figure2_misspecification_source.csv", index=False
    )


def paper4_guarantee_map() -> None:
    """Build the main guarantee/estimand map without adding data claims."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.55), squeeze=False)
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.07, top=0.91,
                        wspace=0.055, hspace=0.15)
    panels = [
        ("a", "CONTROLLED", PALETTE["blue_soft"], PALETTE["blue"]),
        ("b", "STRUCTURALLY BOUNDED", PALETTE["gold_soft"], PALETTE["gold"]),
        ("c", "DESCRIPTIVE", PALETTE["teal_soft"], PALETTE["teal"]),
        ("d", "UNQUALIFIED", PALETTE["red_soft"], PALETTE["red"]),
    ]
    for ax, (label, title, face, accent) in zip(axes.flat, panels):
        ax.set_axis_off()
        ax.set_facecolor("white")
        ax.add_patch(
            FancyBboxPatch(
                (0.01, 0.025), 0.98, 0.94,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                transform=ax.transAxes, linewidth=0.7,
                edgecolor=PALETTE["neutral_3"], facecolor=face, alpha=0.42,
            )
        )
        ax.text(0.045, 0.91, label, transform=ax.transAxes, fontsize=8.3,
                fontweight="bold", color=PALETTE["neutral_0"], va="top")
        ax.text(0.13, 0.91, title, transform=ax.transAxes, fontsize=7.4,
                fontweight="bold", color=accent, va="top")

    ax = axes[0, 0]
    ax.text(0.06, 0.74, "conditioned inspection bound", transform=ax.transAxes,
            fontsize=6.6, color=PALETTE["neutral_0"])
    ax.text(0.06, 0.60, "inspection bound \u2192 observation envelope\n\u2192 transition envelope",
            transform=ax.transAxes, fontsize=7.2, linespacing=1.25, color=accent)
    ax.text(0.06, 0.41, r"look-local event: $F_{\mathrm{loc}}$ over valid looks",
            transform=ax.transAxes,
            fontsize=7.6, color=PALETTE["neutral_0"])
    ax.text(0.06, 0.20, r"$\Pr(F_{\mathrm{loc}})\leq\alpha$",
            transform=ax.transAxes, fontsize=8.4, fontweight="bold", color=accent)

    ax = axes[0, 1]
    ax.text(0.06, 0.73, "admitted actions + qualified ledger", transform=ax.transAxes,
            fontsize=6.6, color=PALETTE["neutral_0"])
    ax.text(0.06, 0.56, r"all-path event: $F_{\mathrm{all}}$",
            transform=ax.transAxes, fontsize=8.4, fontweight="bold",
            color=PALETTE["neutral_0"])
    ax.text(0.06, 0.40, r"$\Pr(F_{\mathrm{all}})\leq\alpha+$ action-envelope risk",
            transform=ax.transAxes, fontsize=7.4, color=accent)
    ax.text(0.06, 0.21, "mission allocation: alpha + beta + envelope terms",
            transform=ax.transAxes, fontsize=7.2, color=PALETTE["neutral_0"])

    ax = axes[1, 0]
    ax.text(0.06, 0.77, "mission-clustered estimands", transform=ax.transAxes,
            fontsize=6.6, color=PALETTE["neutral_0"])
    ax.text(0.06, 0.60, "admitted looks: R = count of no-refresh decisions", transform=ax.transAxes,
            fontsize=7.6, color=accent)
    ax.text(0.06, 0.45, "local and all-path error counts",
            transform=ax.transAxes, fontsize=7.6, color=accent)
    ax.text(0.06, 0.27, "mission-selective risk; pooled count ratio",
            transform=ax.transAxes, fontsize=6.9, color=PALETTE["neutral_0"])
    ax.text(0.06, 0.12, "rates are not a substitute for mission risk",
            transform=ax.transAxes, fontsize=6.2, color=PALETTE["neutral_1"])

    ax = axes[1, 1]
    ax.text(0.06, 0.77, "requires separate qualification/design", transform=ax.transAxes,
            fontsize=6.6, color=PALETTE["neutral_0"])
    ax.text(0.06, 0.59, "permissiveness / authorization rate",
            transform=ax.transAxes, fontsize=7.1, color=accent)
    ax.text(0.06, 0.43, "inspection + refresh burden",
            transform=ax.transAxes, fontsize=7.1, color=accent)
    ax.text(0.06, 0.25, "not certified here:", transform=ax.transAxes,
            fontsize=6.3, color=PALETTE["neutral_0"])
    ax.text(0.06, 0.13, "physical envelope, returned file, mission floor",
            transform=ax.transAxes, fontsize=6.4, color=PALETTE["red"])

    fig.suptitle("What the finite-horizon design controls—and what it does not",
                 x=0.045, y=0.975, ha="left", fontsize=9.2,
                 fontweight="bold", color=PALETTE["neutral_0"])
    save_paper4_figure(fig, FIGURES / "guarantee_map.pdf")
    plt.close(fig)


def build_detection_source() -> pd.DataFrame:
    dt4dds = pd.read_csv(DATA / "paper4_dt4dds_coverage_detection_timepoints.csv")
    dt4dds = (
        dt4dds.groupby(["pool", "day"], as_index=False)
        .agg(
            missing_fraction=("missing_fraction", "mean"),
            total_assigned_reads=("total_assigned_reads", "mean"),
            design_count=("reference_count", "first"),
        )
        .rename(columns={"pool": "study", "day": "condition_order"})
    )
    dt4dds["condition"] = dt4dds["condition_order"].map(lambda value: f"{int(value)}d")
    dt4dds["source_layer"] = "post-workflow reference detection"

    independent = pd.read_csv(DATA / "paper4_independent_aging_detection_summary.csv")
    order = {
        ("Meiser", "unaged"): 0,
        ("Meiser", "aged"): 1,
        ("Meiser", "repaired"): 2,
        ("Song", "0d"): 0,
        ("Song", "28d"): 1,
        ("Song", "56d"): 2,
        ("Song", "70d"): 3,
    }
    independent["condition_order"] = [
        order[(study, condition)]
        for study, condition in zip(independent["study"], independent["condition"])
    ]
    independent["source_layer"] = "post-workflow reference detection"
    independent = independent.rename(
        columns={
            "design_count": "design_count",
            "total_assigned_reads": "total_assigned_reads",
        }
    )
    keep = [
        "study",
        "condition",
        "condition_order",
        "missing_fraction",
        "total_assigned_reads",
        "design_count",
        "source_layer",
    ]
    return pd.concat([dt4dds[keep], independent[keep]], ignore_index=True)


def schematic_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    detail: str,
    facecolor: str,
    side_label: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        linewidth=0.7,
        edgecolor=PALETTE["neutral_3"],
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.04,
        y + height - 0.065,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
        fontweight="bold",
        color=PALETTE["neutral_0"],
    )
    ax.text(
        x + 0.04,
        y + 0.055,
        detail,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=PALETTE["neutral_1"],
        linespacing=1.25,
    )
    ax.text(
        x + width - 0.025,
        y + height - 0.05,
        side_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.3,
        color=PALETTE["neutral_2"],
        fontstyle="italic",
    )


def paper4_measurement_layers() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    qpcr = pd.read_csv(DATA / "paper4_dt4dds_qpcr_decay_timepoints.csv")
    summary = pd.read_csv(DATA / "paper4_dt4dds_qpcr_decay_summary.csv")
    qpcr = qpcr.merge(
        summary[
            [
                "pool_key",
                "log10_decay_slope_per_day",
                "half_life_days",
                "half_life_technical_bootstrap_ci95_low_days",
                "half_life_technical_bootstrap_ci95_high_days",
            ]
        ],
        on="pool_key",
        how="left",
        validate="many_to_one",
    )
    loo = pd.read_csv(DATA / "paper4_dt4dds_qpcr_leave_one_pool_out.csv")
    detection = build_detection_source()

    fig = plt.figure(figsize=(7.2, 5.45))
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=[1.0, 1.0, 1.1, 1.1, 1.0, 1.0],
        height_ratios=[1.05, 0.95],
        left=0.055,
        right=0.985,
        bottom=0.065,
        top=0.91,
        hspace=0.52,
        wspace=0.42,
    )

    ax = fig.add_subplot(grid[:, :2])
    ax.set_axis_off()
    ax.set_title("Endpoints are not interchangeable", loc="left")
    subtitle(ax, "material, sequence and file-recovery layers")
    add_panel_label(ax, "a", x=-0.08, y=1.08)

    schematic_box(
        ax,
        (0.05, 0.69),
        0.89,
        0.20,
        "Material state",
        "qPCR amplifiable\nconcentration",
        PALETTE["blue_soft"],
        "measured",
    )
    schematic_box(
        ax,
        (0.05, 0.39),
        0.89,
        0.20,
        "Sequence detection",
        "library preparation, PCR,\nsequencing and alignment",
        PALETTE["teal_soft"],
        "post-workflow",
    )
    schematic_box(
        ax,
        (0.05, 0.09),
        0.89,
        0.20,
        "Modeled erasure-budget success",
        "i.i.d. exact-binomial policy model\nand ideal erasure tolerance",
        PALETTE["gold_soft"],
        "synthetic",
    )
    for y0, y1 in [(0.69, 0.59), (0.39, 0.29)]:
        ax.annotate(
            "",
            xy=(0.50, y1),
            xytext=(0.50, y0),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                color=PALETTE["red"],
                lw=1.0,
                linestyle=(0, (3, 2)),
            ),
        )
        ax.text(
            0.54,
            (y0 + y1) / 2,
            "not calibrated",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.7,
            color=PALETTE["red"],
        )
    ax.text(
        0.50,
        0.015,
        "No returned-file endpoint in the reused data",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=PALETTE["neutral_2"],
    )

    ax = fig.add_subplot(grid[0, 2:4])
    xfit = np.linspace(0.0, 7.05, 150)
    for pool_key in [
        "Genscript_GCall",
        "Genscript_GCfix",
        "Twist_GCall",
        "Twist_GCfix",
    ]:
        group = qpcr[qpcr["pool_key"] == pool_key].sort_values("time_days")
        color = POOL_COLORS[pool_key]
        linestyle = "--" if pool_key.endswith("GCfix") else "-"
        mean = group["relative_concentration_mean"].to_numpy(float)
        sd = group["relative_concentration_sd"].to_numpy(float)
        lower = sd.copy()
        ax.errorbar(
            group["time_days"],
            mean,
            yerr=np.vstack([lower, sd]),
            marker="o",
            markersize=3.0,
            linewidth=0,
            elinewidth=0.7,
            capsize=1.5,
            color=color,
            alpha=0.92,
            zorder=3,
        )
        slope = float(group["log10_decay_slope_per_day"].iloc[0])
        ax.plot(
            xfit,
            10 ** (slope * xfit),
            color=color,
            linestyle=linestyle,
            linewidth=1.25,
            label=POOL_LABELS[pool_key],
        )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(plain_log_tick))
    ax.set_xlim(-0.15, 7.2)
    ax.set_ylim(0.0018, 1.7)
    ax.set_xlabel("accelerated-aging time (days)")
    ax.set_ylabel("relative concentration")
    ax.set_title("qPCR reports rapid concentration loss", loc="left")
    subtitle(ax, "mean +/- s.d.; full technical-repeat SD")
    ax.legend(
        loc="lower left",
        ncol=2,
        fontsize=5.2,
        handlelength=1.8,
        columnspacing=0.8,
        borderaxespad=0.25,
    )
    add_panel_label(ax, "b", x=-0.14, y=1.12)
    style_axis(ax)

    ax = fig.add_subplot(grid[0, 4:6])
    label_specs = {
        "Genscript_GCall": ((4, 9), "left"),
        "Genscript_GCfix": ((4, -12), "left"),
        "Twist_GCall": ((7, 2), "left"),
        "Twist_GCfix": ((-4, 8), "right"),
    }
    for _, row in loo.iterrows():
        key = row["holdout_pool_key"]
        ax.scatter(
            row["holdout_final_observed_relative_concentration"],
            row["holdout_final_predicted_relative_concentration"],
            s=31,
            marker="o",
            color=POOL_COLORS[key],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        ax.annotate(
            POOL_LABELS[key].replace("Genscript ", "Gen. ").replace("Twist ", "Tw. "),
            (
                row["holdout_final_observed_relative_concentration"],
                row["holdout_final_predicted_relative_concentration"],
            ),
            xytext=label_specs[key][0],
            textcoords="offset points",
            ha=label_specs[key][1],
            fontsize=5.2,
            color=POOL_COLORS[key],
        )
    values = np.concatenate(
        [
            loo["holdout_final_observed_relative_concentration"].to_numpy(float),
            loo["holdout_final_predicted_relative_concentration"].to_numpy(float),
        ]
    )
    lo = 10 ** np.floor(np.log10(max(float(values.min()), EPSILON)) - 0.05)
    hi = 10 ** np.ceil(np.log10(max(float(values.max()), EPSILON)) + 0.05)
    ax.plot([lo, hi], [lo, hi], color=PALETTE["neutral_3"], lw=0.9, ls="--")
    max_factor = np.maximum(
        loo["holdout_final_observed_relative_concentration"]
        / loo["holdout_final_predicted_relative_concentration"],
        loo["holdout_final_predicted_relative_concentration"]
        / loo["holdout_final_observed_relative_concentration"],
    ).max()
    ax.text(
        0.04,
        0.95,
        f"maximum final-point error: {max_factor:.2f}x",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=PALETTE["neutral_1"],
        bbox=LABEL_BBOX,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(plain_log_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_log_tick))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("observed final concentration")
    ax.set_ylabel("LOO prediction")
    ax.set_title("A common slope transfers poorly", loc="left")
    subtitle(ax, "leave-one-pool-out prediction at the final time point")
    add_panel_label(ax, "c", x=-0.14, y=1.12)
    style_axis(ax)

    parent = fig.add_subplot(grid[1, 2:6])
    parent.set_axis_off()
    parent.set_title(
        "Post-workflow missing-reference fractions do not track aging",
        loc="left",
        pad=11,
    )
    parent.text(
        0.0,
        0.995,
        "condition order is study-specific; common log scale",
        transform=parent.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=PALETTE["neutral_2"],
    )
    add_panel_label(parent, "d", x=-0.065, y=1.10)

    study_specs = [
        ("Genscript_GCall", "Genscript", PALETTE["blue"], ["0", "2", "4", "7"]),
        ("Twist_GCall", "Twist", PALETTE["teal"], ["0", "2", "4", "7"]),
        ("Meiser", "Meiser", PALETTE["violet"], ["U", "A", "R"]),
        ("Song", "Song", PALETTE["gold"], ["0", "28", "56", "70"]),
    ]
    for index, (study, title, color, tick_labels) in enumerate(study_specs):
        mini = parent.inset_axes([index * 0.25 + 0.015, 0.03, 0.215, 0.80])
        group = detection[detection["study"] == study].sort_values("condition_order")
        x = np.arange(len(group))
        mini.plot(
            x,
            group["missing_fraction"].to_numpy(float) * 100.0,
            marker="o",
            markersize=3.5,
            color=color,
            linewidth=1.25,
        )
        mini.set_yscale("log")
        mini.yaxis.set_major_formatter(FuncFormatter(plain_log_tick))
        mini.set_ylim(0.02, 1.8)
        mini.set_yticks([0.03, 0.1, 0.3, 1.0])
        if index == 0:
            mini.set_yticklabels(["0.03", "0.1", "0.3", "1"])
            mini.set_ylabel("missing references (%)", fontsize=5.7)
        else:
            mini.set_yticklabels([])
        mini.set_xticks(x)
        mini.set_xticklabels(tick_labels)
        mini.set_title(title, fontsize=6.2, loc="left", pad=4)
        mini.set_xlabel("condition", fontsize=5.5, labelpad=1.5)
        mini.tick_params(labelsize=5.4, length=2.0, width=0.55, pad=1.5)
        mini.spines["top"].set_visible(False)
        mini.spines["right"].set_visible(False)
        mini.spines["left"].set_linewidth(0.6)
        mini.spines["bottom"].set_linewidth(0.6)
        mini.yaxis.grid(True, color=PALETTE["neutral_4"], linewidth=0.4)
        mini.set_axisbelow(True)

    save_paper4_figure(fig, FIGURES / "public_endpoint_mismatch_audit.pdf")
    plt.close(fig)

    qpcr_source = qpcr.copy()
    qpcr_source.insert(0, "panel", "b")
    loo_source = loo.copy()
    loo_source.insert(0, "panel", "c")
    detection_source = detection.copy()
    detection_source.insert(0, "panel", "d")
    return qpcr_source, loo_source, detection_source


def annotated_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    panel: str,
) -> mpl.image.AxesImage:
    norm = Normalize(vmin=0.0, vmax=100.0)
    image = ax.imshow(matrix, cmap=FAILURE_CMAP, norm=norm, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            text_color = "white" if value >= 45.0 else PALETTE["neutral_0"]
            ax.text(
                col,
                row,
                f"{value:.0f}" if value >= 1.0 else f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=5.2,
                color=text_color,
            )
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0, labelsize=5.6, pad=2)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Sentinel size" if panel == "c" else "Observation-scale factor")
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, panel, x=-0.16, y=1.10)
    return image


def paper4_policy_assurance() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    seed = pd.read_csv(DATA / "paper4_seed_uncertainty_summary.csv")
    sensitivity = pd.read_csv(DATA / "paper4_policy_sensitivity_summary.csv")
    design = sensitivity[sensitivity["sweep"] == "design"].copy()
    misspecification = sensitivity[sensitivity["sweep"] == "misspecification"].copy()

    fig = plt.figure(figsize=(7.2, 5.65))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.86, 1.28],
        width_ratios=[0.92, 1.08],
        left=0.105,
        right=0.985,
        bottom=0.17,
        top=0.84,
        hspace=0.62,
        wspace=0.88,
    )

    ax = fig.add_subplot(grid[0, 0])
    for policy in POLICY_ORDER:
        group = seed[seed["policy"] == policy].sort_values("true_hazard")
        y = group["mean_refreshes"].to_numpy(float)
        yerr = np.vstack(
            [
                y - group["mean_refreshes_ci95_low"].to_numpy(float),
                group["mean_refreshes_ci95_high"].to_numpy(float) - y,
            ]
        )
        ax.errorbar(
            group["true_hazard"],
            y,
            yerr=yerr,
            marker=POLICY_MARKERS[policy],
            markersize=4.1,
            linewidth=1.35,
            elinewidth=0.75,
            capsize=1.6,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
    ax.set_xticks([0.002, 0.006, 0.012])
    ax.set_xticklabels(["0.002", "0.006", "0.012"])
    ax.set_ylim(-0.3, 12.8)
    ax.set_xlabel("true synthetic hazard")
    ax.set_ylabel("mean refreshes over 60 years")
    ax.set_title("Hazard sets the full-information refresh burden", loc="left")
    subtitle(ax, "mean and episode-bootstrap 95% interval; 400 trajectories/cell")
    add_panel_label(ax, "a", x=-0.14, y=1.12)
    style_axis(ax)

    ax = fig.add_subplot(grid[0, 1])
    for policy in POLICY_ORDER:
        group = seed[seed["policy"] == policy].sort_values("true_hazard")
        y = group["episode_failure_rate"].to_numpy(float) * 100.0
        yerr = np.vstack(
            [
                y - group["episode_failure_rate_ci95_low"].to_numpy(float) * 100.0,
                group["episode_failure_rate_ci95_high"].to_numpy(float) * 100.0 - y,
            ]
        )
        ax.errorbar(
            group["true_hazard"],
            y,
            yerr=yerr,
            marker=POLICY_MARKERS[policy],
            markersize=4.1,
            linewidth=1.35,
            elinewidth=0.75,
            capsize=1.6,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
    ax.set_yscale("symlog", linthresh=0.1, linscale=1.0, base=10)
    ax.set_yticks([0.0, 0.1, 1.0, 10.0, 100.0])
    ax.set_yticklabels(["0", "0.1", "1", "10", "100"])
    ax.set_ylim(-0.025, 125)
    ax.set_xticks([0.002, 0.006, 0.012])
    ax.set_xticklabels(["0.002", "0.006", "0.012"])
    ax.set_xlabel("true synthetic hazard")
    ax.set_ylabel("episodes with any violation (%)")
    ax.set_title("Partial observation leaves residual violations", loc="left")
    subtitle(ax, "Wilson 95% intervals; symlog axis preserves observed zeros")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.982),
        ncol=4,
        fontsize=5.7,
        handlelength=1.55,
        columnspacing=0.85,
    )
    add_panel_label(ax, "b", x=-0.13, y=1.12)
    style_axis(ax)

    high_design = design[design["true_hazard"] == 0.012].copy()
    prior_order = ["broad", "high_biased", "low_biased"]
    interval_order = [2, 5, 10]
    sentinel_order = [100, 300, 1000]
    rows = [(prior, interval) for prior in prior_order for interval in interval_order]
    matrix = np.array(
        [
            [
                high_design[
                    (high_design["prior"] == prior)
                    & (high_design["inspection_interval"] == interval)
                    & (high_design["sentinel_size"] == sentinel)
                ]["episode_failure_rate"].iloc[0]
                * 100.0
                for sentinel in sentinel_order
            ]
            for prior, interval in rows
        ]
    )
    row_labels = [
        f"{prior.replace('_biased', '-biased')} | {interval} y"
        for prior, interval in rows
    ]
    ax_c = fig.add_subplot(grid[1, 0])
    image = annotated_heatmap(
        ax_c,
        matrix,
        row_labels,
        [str(value) for value in sentinel_order],
        "Prior support and inspection timing dominate",
        "c",
    )
    ax_c.text(
        0.0,
        1.015,
        "episodes with target violation (%), conservative policy, hazard 0.012",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=PALETTE["neutral_2"],
    )

    row_keys = [
        (hazard, policy, profile)
        for hazard in [0.006, 0.012]
        for policy in ["posterior_mean", "posterior_conservative"]
        for profile in ["constant", "accelerating"]
    ]
    factor_order = [0.8, 1.0, 1.2]
    matrix_misspec = np.array(
        [
            [
                misspecification[
                    (misspecification["true_hazard"] == hazard)
                    & (misspecification["policy"] == policy)
                    & (misspecification["hazard_profile"] == profile)
                    & (misspecification["observation_factor"] == factor)
                ]["episode_failure_rate"].iloc[0]
                * 100.0
                for factor in factor_order
            ]
            for hazard, policy, profile in row_keys
        ]
    )
    row_labels_misspec = [
        (
            f"{hazard:.3f} | "
            f"{'mean' if policy == 'posterior_mean' else 'conservative'} | "
            f"{'const.' if profile == 'constant' else 'accel.'}"
        )
        for hazard, policy, profile in row_keys
    ]
    ax_d = fig.add_subplot(grid[1, 1])
    annotated_heatmap(
        ax_d,
        matrix_misspec,
        row_labels_misspec,
        ["0.8", "1.0", "1.2"],
        "Sentinel bias exposes model fragility",
        "d",
    )
    ax_d.text(
        0.0,
        1.015,
        "Target-violation episodes (%); relative observation scale",
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=PALETTE["neutral_2"],
    )

    colorbar_ax = fig.add_axes([0.28, 0.065, 0.50, 0.018])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label("episodes with target violation (%)", fontsize=6.0)
    colorbar.set_ticks([0, 25, 50, 75, 100])
    colorbar.ax.tick_params(labelsize=5.6, length=2)
    colorbar.outline.set_linewidth(0.55)

    save_paper4_figure(fig, FIGURES / "policy_assurance_limits.pdf")
    plt.close(fig)

    seed_source = seed.copy()
    seed_source.insert(0, "panels", "a;b")
    design_source = high_design.copy()
    design_source.insert(0, "panel", "c")
    misspecification_source = misspecification.copy()
    misspecification_source.insert(0, "panel", "d")
    return seed_source, design_source, misspecification_source


def paper4_finite_horizon_certificate() -> pd.DataFrame:
    """Visualize the exact finite-horizon certificate and its failure modes."""
    summary = pd.read_csv(DATA / "paper4_finite_horizon_certificate_summary.csv")
    summary.insert(0, "panels", "b;c;d")
    summary.to_csv(DATA / "paper4_figure3_certificate_source.csv", index=False)

    valid_color = PALETTE["blue"]
    invalid_color = PALETTE["red"]
    fig = plt.figure(figsize=(7.2, 5.65))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.985,
        bottom=0.12,
        top=0.93,
        hspace=0.57,
        wspace=0.48,
    )

    ax = fig.add_subplot(grid[0, 0])
    ax.set_axis_off()
    ax.set_title("A prespecified error budget protects no-refresh decisions", loc="left")
    subtitle(ax, "finite horizon; local bound; envelope risk charged on admitted paths")
    add_panel_label(ax, "a", x=-0.14, y=1.12)
    boxes = [
        ("Sentinel count", "fresh inspection sample under a binomial model", PALETTE["blue_soft"]),
        ("Exact one-sided bound", "Clopper--Pearson upper inspection limit", PALETTE["teal_soft"]),
        ("Observation + transition envelopes", "declared detection floor and aging increment", PALETTE["gold_soft"]),
        ("Declared lower service envelope", "exact erasure-tail instance at the service target", "#F2F4F7"),
    ]
    y_positions = [0.78, 0.57, 0.36, 0.15]
    for (title, detail, face), y in zip(boxes, y_positions):
        patch = FancyBboxPatch(
            (0.04, y),
            0.91,
            0.15,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            transform=ax.transAxes,
            linewidth=0.65,
            edgecolor=PALETTE["neutral_3"],
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            0.075,
            y + 0.105,
            title,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color=PALETTE["neutral_0"],
        )
        ax.text(
            0.075,
            y + 0.042,
            detail,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.8,
            color=PALETTE["neutral_1"],
        )
    for upper, lower in zip(y_positions[:-1], y_positions[1:]):
        ax.annotate(
            "",
            xy=(0.50, lower + 0.16),
            xytext=(0.50, upper - 0.005),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["neutral_2"], lw=0.8),
        )
    ax.text(
        0.50,
        0.035,
        r"look-local familywise bound: $\leq 0.05$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
        color=PALETTE["blue"],
    )

    display = summary[
        summary["scenario"].isin(
            [
                "exact_constant_h0002",
                "exact_constant_h0006",
                "exact_constant_h0012",
                "naive_acceleration_and_underdetection_high_information",
                "enveloped_acceleration_and_underdetection_high_information",
            ]
        )
    ].copy()
    order = [
        "exact_constant_h0002",
        "exact_constant_h0006",
        "exact_constant_h0012",
        "naive_acceleration_and_underdetection_high_information",
        "enveloped_acceleration_and_underdetection_high_information",
    ]
    display["scenario"] = pd.Categorical(display["scenario"], order, ordered=True)
    display = display.sort_values("scenario")
    labels = ["constant 0.002", "constant 0.006", "constant 0.012", "misspecified high-info", "enveloped high-info"]
    y = np.arange(len(display))
    rates = display["episode_false_safe_rate"].to_numpy(float) * 100.0
    low = display["episode_false_safe_rate_ci95_low"].to_numpy(float) * 100.0
    high = display["episode_false_safe_rate_ci95_high"].to_numpy(float) * 100.0
    colors = [valid_color if bool(value) else invalid_color for value in display["model_envelope_valid"]]
    ax = fig.add_subplot(grid[0, 1])
    for index in range(len(display)):
        ax.errorbar(
            rates[index],
            y[index],
            xerr=[[rates[index] - low[index]], [high[index] - rates[index]]],
            marker="o",
            markersize=4.2,
            linewidth=0,
            elinewidth=1.0,
            capsize=1.8,
            color=colors[index],
        )
    ax.axvline(5.0, color=PALETTE["neutral_2"], linestyle="--", linewidth=0.9)
    ax.set_xscale("symlog", linthresh=0.05, linscale=1.0, base=10)
    ax.set_xlim(-0.01, 55)
    ax.set_xticks([0.0, 0.1, 1.0, 5.0, 10.0, 50.0])
    ax.set_xticklabels(["0", "0.1", "1", "5", "10", "50"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("episodes with any under-target authorization (%)")
    ax.set_title("Finite-sample control survives calibrated stress", loc="left")
    subtitle(ax, "point and Wilson 95% interval; red is outside the envelope")
    add_panel_label(ax, "b", x=-0.14, y=1.12)
    style_axis(ax)

    burden_order = [
        "exact_constant_h0002",
        "exact_constant_h0006",
        "naive_acceleration_and_underdetection_high_information",
        "enveloped_acceleration_and_underdetection_high_information",
        "enveloped_acceleration_and_underdetection_high_information_coarse",
        "envelope_cost_under_constant_hazard_high_information",
        "envelope_cost_under_constant_hazard_high_information_coarse",
    ]
    burden = summary.set_index("scenario").loc[burden_order]
    burden_labels = [
        "0.002",
        "0.006",
        "stress\nnaive",
        "stress\ntight",
        "stress\ncoarse",
        "const.\ntight",
        "const.\ncoarse",
    ]
    x = np.arange(len(burden))
    ax = fig.add_subplot(grid[1, 0])
    bars = ax.bar(
        x,
        burden["mean_refreshes"].to_numpy(float),
        color=[
            valid_color,
            valid_color,
            invalid_color,
            PALETTE["teal"],
            PALETTE["teal_soft"],
            PALETTE["gold"],
            PALETTE["gold_soft"],
        ],
        width=0.70,
        edgecolor="white",
        linewidth=0.5,
    )
    for rect, no_refresh in zip(bars, burden["mean_no_refresh_decisions"]):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.25,
            f"{no_refresh:.2f} NR",
            ha="center",
            va="bottom",
            fontsize=5.2,
            color=PALETTE["neutral_1"],
        )
    ax.set_ylim(0, 12.2)
    ax.set_xticks(x)
    ax.set_xticklabels(burden_labels, fontsize=6.2)
    ax.set_ylabel("mean refreshes over 60 years")
    ax.set_title("The tight propagation bound recovers maintenance margin", loc="left")
    subtitle(ax, "tight vs coarse fixed-cycle bounds; labels show mean no-refresh decisions")
    add_panel_label(ax, "c", x=-0.14, y=1.12)
    style_axis(ax)

    decomposition_order = [
        "exact_constant_h0002",
        "exact_constant_h0006",
        "exact_constant_h0012",
        "naive_acceleration_and_underdetection_high_information",
        "enveloped_acceleration_and_underdetection_high_information",
        "ten_year_schedule_gap",
    ]
    decomposition = summary.set_index("scenario").loc[decomposition_order]
    decomposition_labels = [
        "0.002",
        "0.006",
        "0.012",
        "misspec.",
        "envelope",
        "10-y gap",
    ]
    x = np.arange(len(decomposition))
    width = 0.36
    ax = fig.add_subplot(grid[1, 1])
    ax.bar(
        x - width / 2,
        decomposition["episode_false_safe_rate"].to_numpy(float) * 100.0,
        width,
        label="Under-target deferral authorization",
        color=PALETTE["teal"],
    )
    ax.bar(
        x + width / 2,
        decomposition["episode_violation_rate"].to_numpy(float) * 100.0,
        width,
        label="Any target violation",
        color=PALETTE["neutral_2"],
    )
    ax.set_yscale("symlog", linthresh=0.1, linscale=1.0, base=10)
    ax.set_ylim(-0.025, 140)
    ax.set_yticks([0.0, 0.1, 1.0, 10.0, 100.0])
    ax.set_yticklabels(["0", "0.1", "1", "10", "100"])
    ax.set_xticks(x)
    ax.set_xticklabels(decomposition_labels, fontsize=6.2)
    ax.set_ylabel("episodes (%)")
    ax.set_title("Decision control does not certify cycle-entry gaps", loc="left")
    subtitle(ax, "a 10-year first or post-refresh gap can violate before action")
    ax.legend(loc="upper left", fontsize=5.5, frameon=False)
    add_panel_label(ax, "d", x=-0.14, y=1.12)
    style_axis(ax)

    save_paper4_figure(fig, FIGURES / "finite_horizon_certificate.pdf")
    plt.close(fig)
    return summary


def main() -> None:
    global DATA, FIGURES
    parser = argparse.ArgumentParser(
        description="Build the five legacy Paper 4 figure families from audited CSV data."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "papers" / "paper4_adaptive_archive_maintenance" / "figures",
    )
    args = parser.parse_args()
    DATA = args.data_dir
    FIGURES = args.figure_dir
    setup_style()
    paper4_guarantee_map()
    qpcr, loo, detection = paper4_measurement_layers()
    seed, design, misspecification = paper4_policy_assurance()
    paper4_finite_horizon_certificate()
    export_source_data(
        qpcr,
        loo,
        detection,
        seed,
        design,
        misspecification,
    )
    print("PASS_PAPER4_SUBMISSION_FIGURE_GENERATION")
    print("figure_count=5 source_tables=7 backend=python-matplotlib-agg")


if __name__ == "__main__":
    main()
