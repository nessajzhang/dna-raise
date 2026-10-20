#!/usr/bin/env python3
"""Build the two V8.3 NC lead figures from frozen Paper 4 summaries.

This script does not rerun the simulations. It reads the frozen V7 finite-horizon
summary, writes an auditable V8 source-data table, and renders submission exports
with the project-selected Python/matplotlib backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_FIGURES = ROOT / "papers" / "paper4_adaptive_archive_maintenance" / "figures"

MM = 1 / 25.4
FIG_WIDTH_MM = 183
COLORS = {
    "ink": "#20262E",
    "muted": "#626D78",
    "grid": "#D9DEE3",
    "blue": "#2878B5",
    "teal": "#2A9D8F",
    "amber": "#E69F00",
    "red": "#C84A3F",
    "purple": "#6F5AA8",
    "light_blue": "#EAF3F8",
    "light_teal": "#EAF6F3",
    "light_amber": "#FFF5DD",
    "light_red": "#FCEDEA",
    "light_grey": "#F3F5F6",
}


def set_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_all(fig: mpl.figure.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=450)
    fig.savefig(base.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.045,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
        va="bottom",
        ha="right",
    )


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    face,
    edge,
    title_size=8.0,
    sub_size=8.0,
    sub_color=None,
    sub_y_fraction=0.31,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.035 * w,
        y + 0.66 * h,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + 0.035 * w,
        y + sub_y_fraction * h,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=sub_size,
        color=sub_color or COLORS["muted"],
    )


def arrow(ax, x0, y0, x1, y1, color=None):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=color or COLORS["muted"],
            shrinkA=1,
            shrinkB=1,
        )
    )


def build_framework_figure(figure_dir: Path) -> None:
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM, 118 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], width_ratios=[1.2, 0.8], hspace=0.31, wspace=0.22)

    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_axis_off()
    panel_label(ax_a, "a")
    stages = [
        ("Candidate deferral", "Existing policy", COLORS["light_grey"], COLORS["muted"]),
        ("Inspection bound", r"Count bound $U_j$", COLORS["light_blue"], COLORS["blue"]),
        ("State bound", "Transition envelope", COLORS["light_teal"], COLORS["teal"]),
        (r"Screen passage $P_j$", "Service criterion", COLORS["light_amber"], COLORS["amber"]),
        ("Final authorization", r"$N_j\subseteq P_j$", COLORS["light_red"], COLORS["red"]),
    ]
    x0, y, w, h, gap = 0.01, 0.22, 0.174, 0.56, 0.03
    for idx, (title, subtitle, face, edge) in enumerate(stages):
        x = x0 + idx * (w + gap)
        box(ax_a, x, y, w, h, title, subtitle, face, edge)
        if idx < len(stages) - 1:
            arrow(ax_a, x + w + 0.004, y + h / 2, x + w + gap - 0.004, y + h / 2)

    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_axis_off()
    panel_label(ax_b, "b")
    box(
        ax_b,
        0.02,
        0.52,
        0.45,
        0.36,
        "Error-control gate",
        r"$\sum_j\alpha_j\leq\alpha$",
        COLORS["light_blue"],
        COLORS["blue"],
        title_size=7.8,
    )
    box(
        ax_b,
        0.53,
        0.52,
        0.45,
        0.36,
        "Screen-usefulness check",
        "Exact " + r"$\mathrm{BinCDF}(y_j^*;n_j,p)$"
        + "\n"
        + "at a model-safe history",
        COLORS["light_amber"],
        COLORS["amber"],
        title_size=8.2,
        sub_size=8.0,
        sub_color=COLORS["ink"],
        sub_y_fraction=0.24,
    )
    ax_b.text(0.50, 0.71, "+", transform=ax_b.transAxes, ha="center", va="center", fontsize=13, color=COLORS["ink"], fontweight="bold")
    arrow(ax_b, 0.50, 0.53, 0.50, 0.34, COLORS["ink"])
    box(
        ax_b,
        0.22,
        0.04,
        0.56,
        0.29,
        "Application qualification",
        "History-uniform floor\nif required",
        COLORS["light_teal"],
        COLORS["teal"],
        title_size=8.5,
        sub_size=8.2,
        sub_y_fraction=0.26,
    )

    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_axis_off()
    panel_label(ax_c, "c")
    layers = [
        ("Statistical", "sampling-law calibration", COLORS["blue"]),
        ("Engineering", "observation + transition", COLORS["teal"]),
        ("Service", "no-refresh service envelope", COLORS["amber"]),
    ]
    for idx, (title, subtitle, color) in enumerate(layers):
        yy = 0.78 - idx * 0.25
        ax_c.plot([0.04, 0.12], [yy, yy], transform=ax_c.transAxes, color=color, linewidth=4.2, solid_capstyle="round")
        ax_c.text(0.16, yy + 0.035, title, transform=ax_c.transAxes, fontsize=7.8, fontweight="bold", color=COLORS["ink"], va="center")
        ax_c.text(0.16, yy - 0.055, subtitle, transform=ax_c.transAxes, fontsize=7.3, color=COLORS["muted"], va="center")
    note = FancyBboxPatch(
        (0.04, -0.015),
        0.91,
        0.185,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0,
        edgecolor="none",
        facecolor=COLORS["light_grey"],
        transform=ax_c.transAxes,
        clip_on=False,
    )
    ax_c.add_patch(note)
    ax_c.text(
        0.075,
        0.105,
        "Separate mission terms:",
        transform=ax_c.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )
    ax_c.text(
        0.075,
        0.035,
        "cycle entry + refresh state/service",
        transform=ax_c.transAxes,
        fontsize=7.3,
        color=COLORS["muted"],
        va="center",
    )

    save_all(fig, figure_dir / "nc_v8_framework")


def prepare_tradeoff_data(data_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(data_dir / "paper4_finite_horizon_certificate_summary.csv")
    label_map = {
        "exact_constant_h0002": "Exact h=0.002",
        "exact_constant_h0006": "Exact h=0.006",
        "exact_constant_h0012": "Exact h=0.012",
        "naive_acceleration_and_underdetection_high_information": "Undeclared stress",
        "enveloped_acceleration_and_underdetection_high_information": "Declared stress",
        "enveloped_acceleration_and_underdetection_high_information_coarse": "Declared stress, relaxed map",
        "envelope_cost_under_constant_hazard_high_information": "Envelope cost, tight map",
        "envelope_cost_under_constant_hazard_high_information_coarse": "Envelope cost, relaxed map",
        "ten_year_schedule_gap": "Infeasible 10-y gap",
    }
    main_ids = [
        "exact_constant_h0002",
        "exact_constant_h0006",
        "exact_constant_h0012",
        "naive_acceleration_and_underdetection_high_information",
        "enveloped_acceleration_and_underdetection_high_information",
    ]
    out = source.copy()
    out["display_label"] = out["scenario"].map(label_map)
    if out["display_label"].isna().any():
        missing = out.loc[out["display_label"].isna(), "scenario"].tolist()
        raise ValueError(f"Unmapped scenarios: {missing}")
    out["no_refresh_look_fraction"] = out["no_refresh_decisions"] / out["decisions"]
    out["main_figure"] = out["scenario"].isin(main_ids)
    out["main_figure_reason"] = np.where(
        out["main_figure"],
        "core exact-hazard or matched undeclared-versus-declared stress comparison",
        "secondary propagation, envelope-cost, or cycle-entry sensitivity retained in SI",
    )
    columns = [
        "scenario",
        "display_label",
        "main_figure",
        "main_figure_reason",
        "episodes",
        "decisions",
        "no_refresh_decisions",
        "no_refresh_look_fraction",
        "episode_false_safe_rate",
        "episode_false_safe_rate_ci95_low",
        "episode_false_safe_rate_ci95_high",
        "mean_refreshes",
        "model_envelope_valid",
        "theorem_applicable",
        "guarantee_scope",
    ]
    out = out[columns]
    out.to_csv(data_dir / "source_data_nc_v8_risk_admission_burden.csv", index=False)
    return out


def build_tradeoff_figure(data_dir: Path, figure_dir: Path) -> None:
    all_rows = prepare_tradeoff_data(data_dir)
    df = all_rows.loc[all_rows["main_figure"]].copy()
    order = [
        "Exact h=0.002",
        "Exact h=0.006",
        "Exact h=0.012",
        "Undeclared stress",
        "Declared stress",
    ]
    df["display_label"] = pd.Categorical(df["display_label"], categories=order, ordered=True)
    df = df.sort_values("display_label").reset_index(drop=True)

    y = np.arange(len(df))[::-1]
    point_colors = [COLORS["blue"], COLORS["blue"], COLORS["purple"], COLORS["red"], COLORS["teal"]]
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM, 133 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.42, wspace=0.36)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :])

    labels = df["display_label"].astype(str).tolist()
    rate = 100 * df["episode_false_safe_rate"].to_numpy(float)
    low = 100 * df["episode_false_safe_rate_ci95_low"].to_numpy(float)
    high = 100 * df["episode_false_safe_rate_ci95_high"].to_numpy(float)
    ax_a.errorbar(rate, y, xerr=np.vstack([rate - low, high - rate]), fmt="none", ecolor=COLORS["muted"], elinewidth=0.9, capsize=2)
    ax_a.scatter(rate, y, s=28, c=point_colors, edgecolor="white", linewidth=0.5, zorder=3)
    for xx, yy in zip(rate, y):
        label = "0" if xx == 0 else f"{xx:.2f}"
        ax_a.text(xx + 0.45, yy, label, va="center", fontsize=6.3, color=COLORS["ink"])
    ax_a.set_yticks(y, labels)
    ax_a.set_xlabel("Affected episodes (%)")
    ax_a.set_xlim(-0.5, 26.5)
    ax_a.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    panel_label(ax_a, "a")

    no_refresh = 100 * df["no_refresh_look_fraction"].to_numpy(float)
    ax_b.scatter(no_refresh, y, s=28, c=point_colors, edgecolor="white", linewidth=0.5, zorder=3)
    for xx, yy in zip(no_refresh, y):
        text = f"{xx:.1f}" if xx >= 0.1 else f"{xx:.3f}"
        ax_b.text(xx + 1.2, yy, text, va="center", fontsize=6.3, color=COLORS["ink"])
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([])
    ax_b.set_xlabel("No-refresh looks (%)")
    ax_b.set_xlim(-2, 82)
    ax_b.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    panel_label(ax_b, "b")

    refresh = df["mean_refreshes"].to_numpy(float)
    ax_c.scatter(refresh, y, s=28, c=point_colors, edgecolor="white", linewidth=0.5, zorder=3)
    for xx, yy in zip(refresh, y):
        ax_c.text(xx + 0.22, yy, f"{xx:.2f}", va="center", fontsize=6.3, color=COLORS["ink"])
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([])
    ax_c.set_xlabel("Mean refreshes")
    ax_c.set_xlim(2, 12.2)
    ax_c.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    panel_label(ax_c, "c")

    sizes = 42 + 4.5 * np.sqrt(np.maximum(rate, 0))
    sc = ax_d.scatter(no_refresh, refresh, c=rate, s=sizes, cmap=mpl.colors.LinearSegmentedColormap.from_list("risk", [COLORS["teal"], COLORS["amber"], COLORS["red"]]), vmin=0, vmax=max(rate), edgecolor="white", linewidth=0.7, zorder=3)
    offsets = [(-78, 6), (4, 4), (4, -12), (5, 5), (5, 5)]
    for xx, yy, label, off in zip(no_refresh, refresh, labels, offsets):
        ax_d.annotate(label, (xx, yy), xytext=off, textcoords="offset points", fontsize=6.8, color=COLORS["ink"])
    ax_d.set_xlabel("No-refresh looks (%)")
    ax_d.set_ylabel("Mean refreshes over 60 y")
    ax_d.set_xlim(-3, 82)
    ax_d.set_ylim(2.0, 12.1)
    ax_d.grid(color=COLORS["grid"], linewidth=0.6)
    cbar = fig.colorbar(sc, ax=ax_d, fraction=0.025, pad=0.018)
    cbar.set_label("Affected episodes (%)")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    panel_label(ax_d, "d")

    save_all(fig, figure_dir / "nc_v8_risk_admission_burden")


def build_framework_figure_v86(figure_dir: Path) -> None:
    """Render the V8.6 decision timeline and its two probability statements."""
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM, 88 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 0.95], hspace=0.31)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_axis_off()
    panel_label(ax_a, "a")
    stages = [
        ("Inspect", r"count $Y_j$", COLORS["light_blue"], COLORS["blue"]),
        ("Propagate bound", r"$U_j\;\longrightarrow\;\widehat q^U_{j+1}$", COLORS["light_teal"], COLORS["teal"]),
        ("Base policy", "proposes deferral", COLORS["light_grey"], COLORS["muted"]),
        ("Service screen", r"pass $P_j$ or veto", COLORS["light_amber"], COLORS["amber"]),
        ("Authorize interval", r"$N_j\subseteq P_j$", COLORS["light_blue"], COLORS["blue"]),
    ]
    ax_a.text(
        0.5,
        0.94,
        r"Eligible look $j$; repeated for $j=1,\ldots,K$",
        transform=ax_a.transAxes,
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=COLORS["ink"],
    )
    x0, y, w, h, gap = 0.01, 0.23, 0.177, 0.56, 0.027
    for idx, (title, subtitle, face, edge) in enumerate(stages):
        x = x0 + idx * (w + gap)
        box(
            ax_a,
            x,
            y,
            w,
            h,
            title,
            subtitle,
            face,
            edge,
            title_size=8.2,
            sub_size=8.0,
            sub_color=COLORS["ink"],
            sub_y_fraction=0.30,
        )
        if idx < len(stages) - 1:
            arrow(ax_a, x + w + 0.003, y + h / 2, x + w + gap - 0.003, y + h / 2)
    ax_a.text(
        0.5,
        0.06,
        "Outside the protected sequence: entry and post-refresh intervals",
        transform=ax_a.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["red"],
    )

    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_axis_off()
    panel_label(ax_b, "b")
    box(
        ax_b,
        0.02,
        0.22,
        0.43,
        0.58,
        "LOOK-LOCAL AUTHORIZATION EVENT",
        r"$\Pr(F_{\rm loc})\leq\min\{1,\alpha+\Delta_{\rm samp}\}$",
        COLORS["light_blue"],
        COLORS["blue"],
        title_size=8.2,
        sub_size=10.0,
        sub_color=COLORS["ink"],
        sub_y_fraction=0.31,
    )
    arrow(ax_b, 0.46, 0.51, 0.54, 0.51, COLORS["muted"])
    box(
        ax_b,
        0.55,
        0.22,
        0.43,
        0.58,
        "ALL AUTHORIZED PATHS",
        r"$\Pr(F_{\rm all})\leq\min\{1,\alpha+\Delta_{\rm samp}+\Pr(B_{\rm act})\}$",
        COLORS["light_teal"],
        COLORS["teal"],
        title_size=8.2,
        sub_size=9.2,
        sub_color=COLORS["ink"],
        sub_y_fraction=0.31,
    )
    ax_b.text(
        0.5,
        0.04,
        r"Declared conditional sampling law: $\Delta_{\rm samp}=0$",
        transform=ax_b.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["muted"],
    )
    save_all(fig, figure_dir / "nc_v8_framework")


def build_wrapper_figure(data_dir: Path, figure_dir: Path) -> None:
    """Render the paired base-policy wrapper audit from its frozen summary."""
    summary = pd.read_csv(data_dir / "paper4_policy_wrapper_summary.csv")
    policy_order = ["posterior_mean", "posterior_conservative"]
    policy_short = {"posterior_mean": "PM", "posterior_conservative": "Q95"}
    summary["policy_rank"] = summary["policy"].map(
        {policy: index for index, policy in enumerate(policy_order)}
    )
    summary = summary.sort_values(["true_hazard", "policy_rank"]).reset_index(drop=True)
    if len(summary) != 6 or summary["policy_rank"].isna().any():
        raise ValueError("expected six prespecified wrapper-summary rows")
    summary["display_label"] = [
        f"{hazard:.3f} · {policy_short[policy]}"
        for hazard, policy in zip(summary["true_hazard"], summary["policy"])
    ]
    source_columns = [column for column in summary.columns if column != "policy_rank"]
    summary[source_columns].to_csv(
        data_dir / "source_data_nc_v8_policy_wrapper.csv", index=False
    )

    y = np.arange(len(summary))[::-1]
    total_decisions = summary["episodes"].to_numpy(float) * summary[
        "decisions_per_episode"
    ].to_numpy(float)
    authorized = 100.0 * summary["screened_total_authorizations"].to_numpy(float) / total_decisions
    overrides = 100.0 * summary["screened_total_overrides"].to_numpy(float) / total_decisions
    base_refresh_requests = np.maximum(0.0, 100.0 - authorized - overrides)

    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM, 91 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.05, 0.93], wspace=0.38)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.83)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    ax_a.barh(y, authorized, color=COLORS["blue"], label="Authorized")
    ax_a.barh(y, overrides, left=authorized, color=COLORS["amber"], label="Screen override")
    ax_a.barh(
        y,
        base_refresh_requests,
        left=authorized + overrides,
        color="#D8DDE2",
        label="Base-policy refresh",
    )
    ax_a.set_yticks(y, summary["display_label"])
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("screened-path decisions (%)")
    ax_a.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    panel_label(ax_a, "a")

    base_rate = 100.0 * summary["base_episode_under_target_authorization_rate"].to_numpy(float)
    screen_rate = 100.0 * summary["screened_episode_under_target_authorization_rate"].to_numpy(float)
    base_low = 100.0 * summary["base_episode_under_target_authorization_ci95_low"].to_numpy(float)
    base_high = 100.0 * summary["base_episode_under_target_authorization_ci95_high"].to_numpy(float)
    screen_low = 100.0 * summary["screened_episode_under_target_authorization_ci95_low"].to_numpy(float)
    screen_high = 100.0 * summary["screened_episode_under_target_authorization_ci95_high"].to_numpy(float)
    ax_b.errorbar(
        base_rate,
        y + 0.12,
        xerr=np.vstack([base_rate - base_low, base_high - base_rate]),
        fmt="o",
        color=COLORS["muted"],
        markersize=4.2,
        linewidth=0.9,
        capsize=2,
        label="Base",
    )
    ax_b.errorbar(
        screen_rate,
        y - 0.12,
        xerr=np.vstack([screen_rate - screen_low, screen_high - screen_rate]),
        fmt="o",
        color=COLORS["teal"],
        markersize=4.2,
        linewidth=0.9,
        capsize=2,
        label="Base + screen",
    )
    ax_b.set_xscale("symlog", linthresh=0.05, linscale=1.0, base=10)
    ax_b.set_xlim(-0.012, 55)
    ax_b.set_xticks([0, 0.1, 1, 10, 40])
    ax_b.set_xticklabels(["0", "0.1", "1", "10", "40"])
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([])
    ax_b.set_xlabel("affected missions (%)")
    ax_b.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    panel_label(ax_b, "b")

    difference = summary["paired_mean_refresh_difference"].to_numpy(float)
    difference_low = summary["paired_mean_refresh_difference_ci95_low"].to_numpy(float)
    difference_high = summary["paired_mean_refresh_difference_ci95_high"].to_numpy(float)
    ax_c.errorbar(
        difference,
        y,
        xerr=np.vstack([difference - difference_low, difference_high - difference]),
        fmt="o",
        color=COLORS["blue"],
        markersize=4.6,
        linewidth=1.0,
        capsize=2,
    )
    ax_c.axvline(0, color=COLORS["muted"], linewidth=0.8)
    ax_c.set_xlim(-0.25, 6.75)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([])
    ax_c.set_xlabel("added refreshes per mission")
    ax_c.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    panel_label(ax_c, "c")

    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    fig.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=5,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.55,
    )

    save_all(fig, figure_dir / "nc_v8_policy_wrapper")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument(
        "--only",
        choices=("all", "framework", "wrapper", "tradeoff"),
        default="all",
        help="render both V8 lead figures or one named family",
    )
    args = parser.parse_args()
    set_style()
    if args.only in {"all", "framework"}:
        build_framework_figure_v86(args.figure_dir)
    if args.only in {"all", "wrapper", "tradeoff"}:
        build_wrapper_figure(args.data_dir, args.figure_dir)
    family_count = 2 if args.only == "all" else 1
    source_contract = "source_rows=6 paired_wrapper" if args.only in {"all", "wrapper", "tradeoff"} else "source_data=unchanged"
    print(
        f"NC_V8_3_FIGURES_PASS figures={family_count} formats=4 {source_contract} "
        "backend=python-matplotlib-agg"
    )


if __name__ == "__main__":
    main()
