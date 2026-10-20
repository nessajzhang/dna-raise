#!/usr/bin/env python3
"""Build publication-ready manuscript figures from project CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FIGURES = {
    "paper1": ROOT / "papers" / "paper1_composite_robust_ssa" / "figures",
    "paper2": ROOT / "papers" / "paper2_thermodynamic_risk_coding" / "figures",
    "paper3": ROOT / "papers" / "paper3_nanopore_hairpin_codec" / "figures",
}

PALETTE = {
    "blue": "#285C8A",
    "blue_mid": "#5C8FC0",
    "blue_soft": "#D5E2F0",
    "teal": "#3F8F94",
    "teal_soft": "#D7ECEB",
    "green": "#6F9F79",
    "red": "#AD4B4B",
    "salmon": "#CE817A",
    "red_soft": "#E4B0AB",
    "violet": "#756BB1",
    "violet_soft": "#DCD8EE",
    "gold": "#C38A32",
    "gold_soft": "#F2E3C6",
    "neutral_0": "#242424",
    "neutral_1": "#4C4C4C",
    "neutral_2": "#747474",
    "neutral_3": "#B6B6B6",
    "neutral_4": "#E7E7E7",
    "neutral_5": "#F5F5F5",
    "panel_bg": "#FBFCFD",
}

LABEL_BBOX = {
    "boxstyle": "square,pad=0.13",
    "facecolor": "white",
    "edgecolor": "#D8D8D8",
    "linewidth": 0.25,
    "alpha": 0.78,
}


def save_figure(fig, path: Path) -> None:
    stem = path.with_suffix("")
    fig.patch.set_facecolor("white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.2,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "figure.dpi": 180,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.70,
            "ytick.major.width": 0.70,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.frameon": False,
            "axes.titlepad": 11.5,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
        }
    )
    mpl.rcParams["path.simplify_threshold"] = 0.05


def add_panel_label(ax, label: str, *, x: float = 0.0, y: float = 1.145) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["neutral_0"],
    )


def add_panel_footer(ax, label: str, title: str, *, y: float = -0.25) -> None:
    ax.text(
        0.5,
        y,
        f"({label}) {title}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        fontweight="semibold",
        color=PALETTE["neutral_0"],
        clip_on=False,
    )


def style_axis(ax) -> None:
    ax.set_facecolor(PALETTE["panel_bg"])
    ax.tick_params(
        color=PALETTE["neutral_1"],
        labelcolor=PALETTE["neutral_0"],
        direction="out",
        pad=2.2,
    )
    for spine in ax.spines.values():
        spine.set_color(PALETTE["neutral_1"])
    ax.yaxis.grid(True, color=PALETTE["neutral_4"], linewidth=0.46, alpha=0.54)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def add_row_bands(ax, rows: np.ndarray, color: str = PALETTE["neutral_5"]) -> None:
    for yi in rows[::2]:
        ax.axhspan(yi - 0.36, yi + 0.36, color=color, alpha=0.24, linewidth=0, zorder=-1)


def add_status_section_bands(
    ax,
    sections: list[tuple[int, int, str, str]],
    label_x: float = 0.035,
) -> None:
    for start, end, color, label in sections:
        ax.axhspan(start - 0.47, end + 0.47, color=color, alpha=0.12, linewidth=0, zorder=-2)
        if start > 0:
            ax.axhline(start - 0.50, color=PALETTE["neutral_4"], linewidth=0.55, zorder=-1)
        ax.text(
            label_x,
            start - 0.31,
            label,
            ha="left",
            va="bottom",
            fontsize=5.6,
            fontweight="bold",
            color=PALETTE["neutral_2"],
        )


def direct_line_label(
    ax,
    x: float,
    y: float,
    text: str,
    color: str,
    dx: float = 0.55,
    dy: float = 0.0,
) -> None:
    ax.text(
        x + dx,
        y + dy,
        text,
        color=color,
        ha="left",
        va="center",
        fontsize=6.1,
        clip_on=False,
        bbox=LABEL_BBOX,
    )


def callout(ax, text: str, xy: tuple[float, float], xytext: tuple[float, float], color: str | None = None) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        arrowprops=dict(arrowstyle="-", color=PALETTE["neutral_2"], lw=0.62),
        color=color or PALETTE["neutral_1"],
        fontsize=6.2,
        bbox=LABEL_BBOX,
    )


def style_status_axis(ax) -> None:
    ax.set_facecolor(PALETTE["panel_bg"])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.tick_params(color=PALETTE["neutral_1"], labelcolor=PALETTE["neutral_0"], direction="out", pad=2.0)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["neutral_1"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(0.0, 1.0)


def subtitle(ax, text: str) -> None:
    ax.text(
        0.0,
        1.012,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=PALETTE["neutral_2"],
    )


def add_panel_caption(ax, label: str, title: str, *, y: float = -0.29) -> None:
    """Place the panel letter and concise title below the plotting area."""
    ax.text(
        0.5,
        y,
        f"({label}) {title}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
        color=PALETTE["neutral_0"],
        clip_on=False,
    )


def add_flow_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#B8C2CC",
    fontsize: float = 7.2,
    textcolor: str = PALETTE["neutral_0"],
    linewidth: float = 0.75,
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
    )


def add_flow_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = PALETTE["neutral_2"],
    connectionstyle: str = "arc3",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 0.9,
            "mutation_scale": 8,
            "connectionstyle": connectionstyle,
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
    )


def paper1_robust_framework() -> None:
    out_dir = FIGURES["paper1"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "Avenir", "Helvetica Neue", "Arial", "sans-serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "mathtext.fontset": "stix",
            "mathtext.default": "it",
        }
    ):
        ink = "#1F2529"
        muted = "#657078"
        hairline = "#C9CED2"
        accent = "#2F6988"
        accent_soft = "#EAF1F5"
        teal = "#2F7B78"
        danger = "#B3423F"
        danger_soft = "#F8ECEB"

        def token(ax, x, y, width, height, label, *, mixed=False) -> None:
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    width,
                    height,
                    facecolor=accent_soft if mixed else "white",
                    edgecolor=accent if mixed else hairline,
                    linewidth=0.8,
                )
            )
            ax.text(
                x + width / 2,
                y + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=8.0,
                color=ink,
            )

        def graph_node(ax, x, y, label, *, edgecolor=accent) -> None:
            ax.add_patch(
                patches.Circle(
                    (x, y),
                    radius=0.048,
                    facecolor="white",
                    edgecolor=edgecolor,
                    linewidth=1.0,
                    zorder=3,
                )
            )
            ax.text(x, y, label, ha="center", va="center", fontsize=7.0, color=ink, zorder=4)

        def graph_edge(ax, start, end, *, color=muted, curve="arc3") -> None:
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 0.9,
                    "mutation_scale": 8,
                    "connectionstyle": curve,
                    "shrinkA": 10,
                    "shrinkB": 10,
                },
                zorder=2,
            )

        fig = plt.figure(figsize=(7.18, 3.12))
        grid = fig.add_gridspec(
            1,
            3,
            left=0.025,
            right=0.99,
            top=0.965,
            bottom=0.18,
            wspace=0.20,
            width_ratios=[1.02, 1.12, 1.38],
        )

        ax = fig.add_subplot(grid[0, 0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.text(0.025, 0.865, r"$x=$", fontsize=8.2, color=ink, va="center")
        symbol_x = [0.175, 0.385, 0.595, 0.805]
        symbol_text = ["A/C", "A", "T", "G/T"]
        for index, (x0, text0) in enumerate(zip(symbol_x, symbol_text)):
            token(ax, x0, 0.815, 0.16, 0.10, text0, mixed=index in {0, 3})
        ax.annotate(
            "",
            xy=(0.575, 0.70),
            xytext=(0.575, 0.80),
            arrowprops={"arrowstyle": "-|>", "color": hairline, "lw": 0.8, "mutation_scale": 7},
        )
        realization_rows = [("AA", "TG"), ("AA", "TT"), ("CA", "TT"), ("CA", "TG")]
        row_y = [0.625, 0.525, 0.425, 0.325]
        ax.add_patch(
            patches.Rectangle(
                (0.30, 0.285),
                0.55,
                0.078,
                facecolor=danger_soft,
                edgecolor="none",
                zorder=0,
            )
        )
        for index, ((left_word, right_word), y0) in enumerate(zip(realization_rows, row_y)):
            ax.text(
                0.575,
                y0,
                f"{left_word}  |  {right_word}",
                ha="center",
                va="center",
                fontsize=8.1,
                color=ink,
            )
        ax.text(0.50, 0.195, r"$\exists\,y\in\mathcal{R}(x):\ CA=\mathrm{RC}(TG)$", ha="center", va="center", fontsize=7.0, color=ink)
        ax.text(0.50, 0.085, r"one unsafe realization  $\Longrightarrow$  reject $x$", ha="center", va="center", fontsize=6.8, fontweight="medium", color=danger)
        add_panel_caption(ax, "a", "All-realization rejection", y=-0.045)

        ax = fig.add_subplot(grid[0, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.text(
            0.50,
            0.93,
            r"$U=(\{A,C\},A),\quad V=(T,\{G,T\})$",
            ha="center",
            va="center",
            fontsize=7.7,
            color=ink,
        )
        coordinate_x = [0.31, 0.78]
        upper_sets = [r"$B(U_1)=\{A,C\}$", r"$B(U_2)=\{A\}$"]
        lower_sets = [r"$\overline{B(V_2)}=\{A,C\}$", r"$\overline{B(V_1)}=\{A\}$"]
        for index, (cx, upper_set, lower_set) in enumerate(zip(coordinate_x, upper_sets, lower_sets), start=1):
            ax.text(cx, 0.79, rf"coordinate ${index}$", ha="center", va="center", fontsize=6.6, color=muted)
            ax.text(cx, 0.665, upper_set, ha="center", va="center", fontsize=7.3, color=ink)
            ax.text(cx, 0.535, lower_set, ha="center", va="center", fontsize=7.3, color=ink)
            ax.text(cx - 0.015, 0.415, "intersection", ha="right", va="center", fontsize=6.5, color=muted)
            ax.text(cx + 0.02, 0.415, r"$\ne\varnothing$", ha="left", va="center", fontsize=7.3, color=teal)
        ax.plot([coordinate_x[0], coordinate_x[0], 0.55], [0.37, 0.31, 0.31], color=hairline, linewidth=0.8)
        ax.plot([coordinate_x[1], coordinate_x[1], 0.55], [0.37, 0.31, 0.31], color=hairline, linewidth=0.8)
        ax.plot([0.55, 0.55], [0.31, 0.27], color=hairline, linewidth=0.8)
        ax.text(0.55, 0.235, "both coordinates", ha="center", va="center", fontsize=6.2, fontweight="medium", color=teal, backgroundcolor="white")
        ax.plot([0.55, 0.55], [0.205, 0.19], color=hairline, linewidth=0.8)
        ax.text(
            0.55,
            0.155,
            r"$B(U_i)\cap\overline{B(V_{3-i})}\ne\varnothing\quad\forall i$",
            ha="center",
            va="center",
            fontsize=7.0,
            color=ink,
        )
        ax.text(0.55, 0.065, "robust reverse-complement conflict", ha="center", va="center", fontsize=7.1, fontweight="medium", color=danger)
        add_panel_caption(ax, "b", "Coordinatewise conflict test", y=-0.045)

        ax = fig.add_subplot(grid[0, 2])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.text(0.18, 0.92, r"productive SCC $H$", ha="center", va="center", fontsize=7.2, color=accent)
        ax.text(0.82, 0.92, r"overlap graph for $F_H$", ha="center", va="center", fontsize=7.2, color=teal)
        graph_node(ax, 0.17, 0.73, r"$q_1$")
        graph_node(ax, 0.29, 0.53, r"$q_2$")
        graph_node(ax, 0.05, 0.53, r"$q_3$")
        graph_edge(ax, (0.05, 0.53), (0.17, 0.73), color=accent)
        graph_edge(ax, (0.17, 0.73), (0.29, 0.53), color=accent)
        graph_edge(ax, (0.29, 0.53), (0.05, 0.53), color=accent)

        ax.annotate(
            "",
            xy=(0.62, 0.63),
            xytext=(0.39, 0.63),
            arrowprops={"arrowstyle": "-|>", "color": ink, "lw": 0.9, "mutation_scale": 8},
        )
        ax.text(0.505, 0.69, r"$\pi$", ha="center", va="center", fontsize=7.7, color=ink)
        ax.text(0.505, 0.575, "recurring windows", ha="center", va="center", fontsize=6.2, color=muted)

        graph_node(ax, 0.83, 0.73, r"$w_1$", edgecolor=teal)
        graph_node(ax, 0.95, 0.53, r"$w_2$", edgecolor=teal)
        graph_node(ax, 0.71, 0.53, r"$w_3$", edgecolor=teal)
        graph_edge(ax, (0.71, 0.53), (0.83, 0.73), color=teal)
        graph_edge(ax, (0.83, 0.73), (0.95, 0.53), color=teal)
        graph_edge(ax, (0.95, 0.53), (0.71, 0.53), color=teal)
        ax.text(0.50, 0.405, r"$F_H$ is conflict-free", ha="center", va="center", fontsize=6.8, color=teal)
        ax.text(0.50, 0.305, r"$\rho(H)\leq\rho(M_{F_H})$", ha="center", va="center", fontsize=9.3, color=ink)
        ax.plot([0.25, 0.75], [0.235, 0.235], color=hairline, linewidth=0.7)
        ax.text(
            0.5,
            0.13,
            r"$C_{\mathrm{rob}}=\max_F\log_2^+\rho(M_F)$",
            ha="center",
            va="center",
            fontsize=9.5,
            color=accent,
        )
        add_panel_caption(ax, "c", "Recurrent structure determines capacity", y=-0.045)

        save_figure(fig, out_dir / "robust_capacity_framework.pdf")
        plt.close(fig)


def paper1_capacity_landscape() -> None:
    out_dir = FIGURES["paper1"]
    out_dir.mkdir(parents=True, exist_ok=True)

    m2 = pd.read_csv(DATA / "all_mixture_count_capacity_summary_m2.csv")
    intervals = pd.read_csv(DATA / "path_extension_upper_bounds_m3.csv")
    intervals = intervals.sort_values(["preset", "horizon"]).groupby("preset").tail(1)
    order = ["standard", "at_mix", "ac_mix", "ag_mix", "two_mixes"]
    labels = {
        "standard": "Base",
        "at_mix": "+AT",
        "ac_mix": "+AC",
        "ag_mix": "+AG",
        "two_mixes": "+AC+GT",
    }
    intervals["order"] = intervals["preset"].map({name: idx for idx, name in enumerate(order)})
    intervals = intervals.sort_values("order")

    with mpl.rc_context(
        {
            "font.size": 8.6,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
        }
    ):
        fig = plt.figure(figsize=(7.24, 3.84))
        grid = fig.add_gridspec(
            1,
            3,
            left=0.075,
            right=0.992,
            top=0.94,
            bottom=0.245,
            wspace=0.36,
            width_ratios=[1.0, 1.08, 1.16],
        )

        ax = fig.add_subplot(grid[0, 0])
        ax.axhspan(2.0, 2.07, color=PALETTE["red_soft"], alpha=0.16, zorder=0)
        ax.plot(
            m2["mixture_count"],
            m2["best_capacity"],
            marker="o",
            markersize=4.3,
            color=PALETTE["blue"],
            linewidth=1.65,
        )
        ax.axhline(2.0, color=PALETTE["red"], linestyle=(0, (4, 2)), linewidth=0.9)
        ax.text(5.95, 2.018, "unconstrained quaternary rate", ha="right", va="bottom", fontsize=7.0, color=PALETTE["red"])
        ax.text(3.12, 1.715, "plateau 1.749", ha="left", va="top", fontsize=7.2, color=PALETTE["blue"])
        ax.set_xlabel("added two-base mixtures")
        ax.set_ylabel("capacity (bits/symbol)")
        ax.set_xticks(m2["mixture_count"])
        ax.set_ylim(1.08, 2.07)
        add_panel_caption(ax, "a", "m=2 remains below 2 bits", y=-0.31)
        style_axis(ax)

        ax = fig.add_subplot(grid[0, 1])
        x = np.arange(len(intervals))
        lower = intervals["lower_bound_bits_per_symbol"].astype(float).to_numpy()
        upper = intervals["path_upper_bits_per_symbol"].astype(float).to_numpy()
        is_base = intervals["preset"].eq("standard").to_numpy()
        is_composite = ~is_base
        ax.axhspan(2.0, 2.45, color=PALETTE["red_soft"], alpha=0.11, zorder=0)
        ax.vlines(x[is_composite], lower[is_composite], upper[is_composite], color=PALETTE["neutral_2"], linewidth=1.35, zorder=2)
        ax.scatter(
            x[is_composite],
            lower[is_composite],
            marker="o",
            s=32,
            color=PALETTE["teal"],
            edgecolor="white",
            linewidth=0.55,
            label="composite lower",
            zorder=3,
        )
        ax.scatter(
            x[is_composite],
            upper[is_composite],
            marker="_",
            s=160,
            color=PALETTE["neutral_1"],
            linewidth=1.45,
            label="composite upper",
            zorder=4,
        )
        base_index = int(np.flatnonzero(is_base)[0])
        ax.scatter(
            [x[base_index]],
            [lower[base_index]],
            marker="D",
            s=36,
            color=PALETTE["blue"],
            edgecolor="white",
            linewidth=0.6,
            label="Base exact",
            zorder=5,
        )
        ax.axhline(2.0, color=PALETTE["red"], linestyle=(0, (4, 2)), linewidth=0.9)
        ax.set_xticks(list(x))
        ax.set_xticklabels([labels[p] for p in intervals["preset"]], rotation=22, ha="right")
        ax.set_ylabel("certified rate (bits/symbol)")
        ax.set_ylim(1.44, 2.45)
        ax.legend(loc="upper left", ncols=2, handlelength=1.0, columnspacing=0.55, borderaxespad=0.25, fontsize=7.0)
        add_panel_caption(ax, "b", "Known Base point; composite m=3 intervals", y=-0.31)
        style_axis(ax)

        ax = fig.add_subplot(grid[0, 2])
        alphabet_size = m2["alphabet_size"].astype(float).to_numpy()
        robust_rate = m2["best_capacity"].astype(float).to_numpy()
        nominal_rate = np.log2(alphabet_size)
        ax.fill_between(
            alphabet_size,
            robust_rate,
            nominal_rate,
            color=PALETTE["red_soft"],
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(
            alphabet_size,
            nominal_rate,
            marker="s",
            markersize=3.8,
            linestyle=(0, (3, 2)),
            color=PALETTE["neutral_2"],
            linewidth=1.25,
        )
        ax.plot(
            alphabet_size,
            robust_rate,
            marker="o",
            markersize=4.0,
            color=PALETTE["blue"],
            linewidth=1.55,
        )
        ax.axhline(2.0, color=PALETTE["red"], linestyle=(0, (4, 2)), linewidth=0.85)
        final_gap = nominal_rate[-1] - robust_rate[-1]
        ax.text(9.85, 3.31, "nominal log2 alphabet size", ha="right", va="top", fontsize=7.0, color=PALETTE["neutral_2"])
        ax.text(9.85, 1.79, "strict robust m=2 capacity", ha="right", va="bottom", fontsize=7.0, color=PALETTE["blue"])
        ax.text(8.0, 2.47, f"nominal-to-robust gap\n{final_gap:.3f} bits at size 10", ha="center", va="center", fontsize=7.0, color=PALETTE["neutral_1"])
        ax.set_xticks(alphabet_size)
        ax.set_xlim(3.7, 10.25)
        ax.set_ylim(1.08, 3.40)
        ax.set_xlabel("composite alphabet size")
        ax.set_ylabel("bits/symbol")
        add_panel_caption(ax, "c", "Alphabet size overstates robust rate", y=-0.31)
        style_axis(ax)

        save_figure(fig, out_dir / "capacity_landscape.pdf")
        plt.close(fig)


def paper1_certificate_convergence() -> None:
    out_dir = FIGURES["paper1"]
    out_dir.mkdir(parents=True, exist_ok=True)

    intervals = pd.read_csv(DATA / "path_extension_upper_bounds_m3.csv")
    plot_specs = [
        ("at_mix", "+AT", PALETTE["salmon"]),
        ("ac_mix", "+AC / +AG", PALETTE["teal"]),
        ("two_mixes", "+AC+GT", PALETTE["violet"]),
    ]
    gap_label_offsets = {
        "at_mix": (0.25, 0.015),
        "ac_mix": (0.25, -0.028),
        "two_mixes": (0.42, 0.026),
    }
    upper_label_offsets = {
        "at_mix": (0.25, 0.006),
        "ac_mix": (0.25, -0.004),
        "two_mixes": (0.42, 0.014),
    }

    with mpl.rc_context(
        {
            "font.size": 8.6,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
        }
    ):
        fig = plt.figure(figsize=(7.24, 3.64))
        grid = fig.add_gridspec(
            1,
            2,
            left=0.075,
            right=0.94,
            top=0.95,
            bottom=0.25,
            wspace=0.32,
        )

        ax = fig.add_subplot(grid[0, 0])
        for preset, label, color in plot_specs:
            sub = intervals.loc[intervals["preset"].eq(preset)].sort_values("horizon")
            ax.plot(
                sub["horizon"],
                sub["path_upper_bits_per_symbol"],
                marker="o",
                markersize=3.2,
                linewidth=1.35,
                color=color,
            )
            last = sub.iloc[-1]
            dx, dy = upper_label_offsets[preset]
            ax.text(
                float(last["horizon"]) + dx,
                float(last["path_upper_bits_per_symbol"]) + dy,
                label,
                ha="left",
                va="center",
                fontsize=7.5,
                color=color,
                clip_on=False,
            )
        ax.set_xlim(1.5, 14.4)
        ax.set_ylim(2.02, 2.54)
        ax.set_xticks(np.arange(2, 15, 2))
        ax.set_xlabel("finite-path horizon L")
        ax.set_ylabel("finite-path upper bound (bits/symbol)")
        add_panel_caption(ax, "a", "Upper bounds decrease with horizon", y=-0.28)
        style_axis(ax)

        ax = fig.add_subplot(grid[0, 1])
        for preset, label, color in plot_specs:
            sub = intervals.loc[intervals["preset"].eq(preset)].sort_values("horizon")
            ax.plot(
                sub["horizon"],
                sub["gap_bits"],
                marker="o",
                markersize=3.2,
                linewidth=1.35,
                color=color,
            )
            last = sub.iloc[-1]
            dx, dy = gap_label_offsets[preset]
            ax.text(
                float(last["horizon"]) + dx,
                float(last["gap_bits"]) + dy,
                label,
                ha="left",
                va="center",
                fontsize=7.5,
                color=color,
                clip_on=False,
            )
        ax.set_xlim(1.5, 14.4)
        ax.set_ylim(0.09, 0.54)
        ax.set_xticks(np.arange(2, 15, 2))
        ax.set_xlabel("finite-path horizon L")
        ax.set_ylabel("upper-minus-lower gap (bits/symbol)")
        add_panel_caption(ax, "b", "Certificate slack contracts with horizon", y=-0.28)
        style_axis(ax)

        save_figure(fig, out_dir / "certificate_convergence.pdf")
        plt.close(fig)


def paper2_proxy_thermo() -> None:
    out_dir = FIGURES["paper2"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(DATA / "vienna_thermo_validation_110nt.csv")
    quant = pd.read_csv(DATA / "vienna_thermo_validation_110nt_quantiles.csv")
    triangulation = pd.read_csv(DATA / "paper2_thermodynamic_triangulation_audit.tsv", sep="\t")
    within_policy = pd.read_csv(DATA / "paper2_vienna_within_policy_validation_summary.csv")
    l10_interval = pd.read_csv(DATA / "paper2_completion_safe_interval_codec_short_summary.csv")
    l12_language = pd.read_csv(DATA / "paper2_compressed_suffix_state_full_language_rank_unrank_l12_summary.csv")
    l13_language = pd.read_csv(DATA / "paper2_compressed_suffix_state_full_language_rank_unrank_l13_summary.csv")
    l14_language = pd.read_csv(DATA / "paper2_weighted_pair_completion_rank_unrank_len14_summary.csv")
    l15_language = pd.read_csv(DATA / "paper2_weighted_pair_completion_rank_unrank_summary.csv")
    l16_language = pd.read_csv(DATA / "paper2_weighted_pair_completion_rank_unrank_len16_summary.csv")
    l17_language = pd.read_csv(DATA / "paper2_weighted_pair_completion_rank_unrank_len17_summary.csv")
    methods_route = pd.read_csv(DATA / "paper2_methods_submission_route_audit.tsv", sep="\t")
    full_l12 = pd.read_csv(DATA / "paper2_compressed_suffix_state_full_language_rank_unrank_l12_summary.csv")
    l13_root = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_branch_masses_summary.csv")
    l13_depth1 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth1_summary.csv")
    l13_depth2 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth2_summary.csv")
    l13_depth3 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth3_summary.csv")
    l13_depth4 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth4_summary.csv")
    l13_depth5 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth5_summary.csv")
    l13_depth6 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth6_summary.csv")
    l13_depth7 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth7_summary.csv")
    l13_depth8 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth8_summary.csv")
    l13_depth9 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth9_compact_summary.csv")
    l13_depth10 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth10_compact_summary.csv")
    l13_depth11 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth11_compact_summary.csv")
    l13_depth12 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_prefix_depth12_terminal_compact_summary.csv")
    l13_depth0_12 = pd.read_csv(DATA / "paper2_compressed_suffix_state_l13_depth0_12_branch_layer_ledger_summary.csv")
    l13_rank_unrank = pd.read_csv(DATA / "paper2_compressed_suffix_state_rank_unrank_l13_summary.csv")
    full_l13_rank_unrank = pd.read_csv(
        DATA / "paper2_compressed_suffix_state_full_language_rank_unrank_l13_summary.csv"
    )
    l13_replay_iterator = pd.read_csv(DATA / "paper2_l13_branch_mass_replay_iterator.tsv", sep="\t")
    l14_selected_prefix = pd.read_csv(DATA / "paper2_compressed_suffix_state_l14_selected_prefix_summary.csv")
    material_bundle = pd.read_csv(DATA / "paper2_material_recovery_validation_bundle_audit.tsv", sep="\t")
    material_sendability = pd.read_csv(DATA / "paper2_material_recovery_dispatch_sendability_audit.tsv", sep="\t")
    quant = quant[quant["scheme"].isin(["random", "gc_hp", "risk_screened", "greedy_risk_aware"])].copy()
    scheme_order = ["random", "gc_hp", "risk_screened", "greedy_risk_aware"]
    scheme_labels = {
        "random": "Random",
        "gc_hp": "GC/HP",
        "risk_screened": "Risk screened",
        "greedy_risk_aware": "Greedy risk-aware",
    }
    scheme_tick_labels = {
        "random": "Random",
        "gc_hp": "GC/HP",
        "risk_screened": "Risk-screen",
        "greedy_risk_aware": "Greedy",
    }
    colors = {
        "random": PALETTE["neutral_3"],
        "gc_hp": PALETTE["blue_mid"],
        "risk_screened": PALETTE["salmon"],
        "greedy_risk_aware": PALETTE["teal"],
    }

    fig = plt.figure(figsize=(7.18, 5.72), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.06, wspace=0.055, hspace=0.11)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.04])

    ax = fig.add_subplot(grid[0, 0])
    ax.axvspan(0, 30, color=PALETTE["teal_soft"], alpha=0.28, zorder=0)
    alpha_by_scheme = {
        "random": 0.16,
        "gc_hp": 0.30,
        "risk_screened": 0.32,
        "greedy_risk_aware": 0.58,
    }
    size_by_scheme = {
        "random": 9,
        "gc_hp": 10,
        "risk_screened": 10,
        "greedy_risk_aware": 12,
    }
    for scheme in scheme_order:
        sub = rows[rows["scheme"] == scheme]
        ax.scatter(
            sub["weighted_pairs"],
            sub["vienna_mfe_kcal_mol"],
            s=size_by_scheme[scheme],
            alpha=alpha_by_scheme[scheme],
            color=colors[scheme],
            linewidths=0,
            rasterized=True,
        )
    trend = rows[["weighted_pairs", "vienna_mfe_kcal_mol"]].dropna().copy()
    trend["bin"] = pd.qcut(trend["weighted_pairs"], q=9, duplicates="drop")
    trend_summary = (
        trend.groupby("bin", observed=True)
        .agg(weighted_pairs=("weighted_pairs", "median"), vienna_mfe_kcal_mol=("vienna_mfe_kcal_mol", "median"))
        .sort_values("weighted_pairs")
    )
    ax.plot(
        trend_summary["weighted_pairs"],
        trend_summary["vienna_mfe_kcal_mol"],
        color=PALETTE["neutral_0"],
        linewidth=1.05,
        alpha=0.74,
        zorder=5,
    )
    ax.text(
        198,
        -33.8,
        "binned median trend",
        ha="left",
        va="center",
        fontsize=5.8,
        color=PALETTE["neutral_1"],
        bbox=LABEL_BBOX,
    )
    direct_labels = [
        ("Greedy risk-aware", 9, -9.5, "left", colors["greedy_risk_aware"]),
        ("Risk screened", 48, -12.0, "left", colors["risk_screened"]),
        ("GC/HP", 142, -11.8, "center", colors["gc_hp"]),
        ("Random", 258, -10.9, "center", PALETTE["neutral_2"]),
    ]
    for label, x_text, y_text, ha, color in direct_labels:
        ax.text(
            x_text,
            y_text,
            label,
            ha=ha,
            va="center",
            fontsize=6.4,
            color=color,
            bbox=LABEL_BBOX,
        )
    ax.set_xlabel("weighted reverse-complement proxy")
    ax.set_ylabel("ViennaRNA MFE (kcal/mol)")
    ax.set_xlim(-8, 356)
    ax.set_ylim(-47.5, -7.5)
    style_axis(ax)
    add_panel_footer(ax, "a", "Proxy score stratifies predicted folding", y=-0.24)

    ax = fig.add_subplot(grid[0, 1])
    quant["scheme_order"] = quant["scheme"].map({name: i for i, name in enumerate(scheme_order)})
    quant = quant.sort_values("scheme_order")
    xpos = np.arange(len(quant))
    bottom = quant["bottom_mfe_mean"].astype(float).to_numpy()
    top = quant["top_mfe_mean"].astype(float).to_numpy()
    ax.axhline(0, color=PALETTE["neutral_3"], linewidth=0.75)
    for i, (lo, hi) in enumerate(zip(bottom, top)):
        ax.plot([i, i], [lo, hi], color=PALETTE["neutral_3"], linewidth=1.05, zorder=1)
        ax.annotate(
            "",
            xy=(i, hi),
            xytext=(i, lo),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["neutral_2"], lw=0.9, shrinkA=2, shrinkB=2),
            zorder=2,
        )
    ax.scatter(
        xpos,
        bottom,
        s=36,
        color=PALETTE["blue_mid"],
        edgecolor="white",
        linewidth=0.55,
        label="bottom 20%",
        zorder=3,
    )
    ax.scatter(
        xpos,
        top,
        s=36,
        color=PALETTE["red"],
        edgecolor="white",
        linewidth=0.55,
        label="top 20%",
        zorder=4,
    )
    for i, row in enumerate(quant.itertuples(index=False)):
        delta = float(row.top_minus_bottom_mfe)
        ax.text(
            i + 0.08,
            (float(row.bottom_mfe_mean) + float(row.top_mfe_mean)) / 2,
            f"{delta:.1f}",
            ha="left",
            va="center",
            fontsize=6.0,
            color=PALETTE["neutral_1"],
        )
    ax.set_xticks(list(xpos))
    ax.set_xticklabels([scheme_tick_labels[s] for s in quant["scheme"]], rotation=18, ha="right")
    ax.set_ylabel("mean MFE (kcal/mol)")
    ax.set_ylim(-31.5, -18.5)
    ax.legend(loc="lower right", handlelength=1.2)
    style_axis(ax)
    add_panel_footer(ax, "b", "High-proxy tails shift to lower MFE", y=-0.34)

    ax = fig.add_subplot(grid[1, 0])
    metric_rows = {
        row.metric: float(row.observed)
        for row in triangulation.itertuples(index=False)
        if row.status == "PASS"
    }
    within_policy_rows = {
        row.target: row
        for row in within_policy.itertuples(index=False)
        if row.status == "PASS_VIENNA_WITHIN_POLICY_VALIDATION"
    }
    bridge_rows = [
        ("Vienna DNA MFE", -metric_rows["spearman_weighted_mfe"], PALETTE["blue"], "mfe_risk"),
        (
            "Vienna ensemble FE",
            -metric_rows["spearman_weighted_ensemble_free_energy"],
            PALETTE["blue_mid"],
            "ensemble_free_energy_risk",
        ),
        ("Expected pairs", metric_rows["spearman_weighted_expected_pairs"], PALETTE["teal"], "expected_pair_mass"),
        ("Primer3 hairpin", metric_rows["spearman_weighted_hairpin_neg_min_dg_kcal_mol"], PALETTE["gold"], None),
        ("Primer3 homodimer", metric_rows["spearman_weighted_homodimer_neg_min_dg_kcal_mol"], PALETTE["gold"], None),
    ]
    labels = [row[0] for row in bridge_rows]
    scores = np.array([row[1] for row in bridge_rows])
    y = np.arange(len(bridge_rows))
    pooled_y = y.astype(float)
    pooled_y[:3] -= 0.11
    ax.axvspan(0.0, 0.3, color=PALETTE["neutral_5"], alpha=0.85, zorder=0)
    ax.axvspan(0.55, 0.72, color=PALETTE["teal_soft"], alpha=0.34, zorder=0)
    ax.hlines(pooled_y, 0, scores, color=PALETTE["neutral_3"], linewidth=1.05, zorder=1)
    ax.scatter(
        scores,
        pooled_y,
        s=30,
        color=[row[2] for row in bridge_rows],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for yi, score in zip(pooled_y, scores):
        ax.text(score + 0.016, yi, f"{score:.2f}", va="center", ha="left", fontsize=5.6, color=PALETTE["neutral_1"])
    for yi, (_, _, color, target) in zip(y[:3] + 0.11, bridge_rows[:3]):
        row = within_policy_rows[target]
        score = float(row.stratified_rank_correlation)
        ci_low = float(row.bootstrap_ci95_low)
        ci_high = float(row.bootstrap_ci95_high)
        ax.errorbar(
            score,
            yi,
            xerr=[[score - ci_low], [ci_high - score]],
            fmt="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.85,
            ecolor=color,
            elinewidth=0.85,
            capsize=1.8,
            capthick=0.75,
            zorder=4,
        )
        ax.text(score + 0.016, yi, f"{score:.2f}", va="center", ha="left", fontsize=5.6, color=color)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_ylim(4.62, -0.88)
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("risk-direction Spearman score")
    ax.xaxis.grid(True, color=PALETTE["neutral_4"], linewidth=0.6)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(color=PALETTE["neutral_1"], labelcolor=PALETTE["neutral_0"], direction="out", pad=2.0)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["neutral_1"])
    ax.scatter([0.025], [-0.61], s=25, color=PALETTE["neutral_1"], edgecolor="white", linewidth=0.45, zorder=5)
    ax.text(0.047, -0.61, "pooled", ha="left", va="center", fontsize=5.5, color=PALETTE["neutral_1"])
    ax.errorbar(
        0.215,
        -0.61,
        xerr=0.025,
        fmt="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor=PALETTE["neutral_1"],
        markeredgewidth=0.8,
        ecolor=PALETTE["neutral_1"],
        elinewidth=0.8,
        capsize=1.6,
        zorder=5,
    )
    ax.text(0.252, -0.61, "within-policy, 95% CI", ha="left", va="center", fontsize=5.5, color=PALETTE["neutral_1"])
    add_panel_footer(ax, "c", "Policy-stratified signal persists", y=-0.24)

    ax = fig.add_subplot(grid[1, 1])

    def route_text(gate: str) -> str:
        if gate == "FULL_L12_RANK_UNRANK":
            if full_l12.empty:
                return "missing"
            row = full_l12.iloc[0]
            if (
                str(row.get("status", "")).startswith("PASS")
                and str(row.get("checked_ranks", "")) == str(row.get("total_count", ""))
                and str(row.get("roundtrip_failures", "")) == "0"
            ):
                return "3.56M/3.56M"
            return "partial"
        if gate == "L13_BRANCH_LAYERS":
            if (
                l13_root.empty
                or l13_depth1.empty
                or l13_depth2.empty
                or l13_depth3.empty
                or l13_depth4.empty
                or l13_depth5.empty
                or l13_depth6.empty
                or l13_depth7.empty
                or l13_depth8.empty
                or l13_depth9.empty
                or l13_depth10.empty
                or l13_depth11.empty
                or l13_depth12.empty
                or l13_depth0_12.empty
            ):
                return "missing"
            root = l13_root.iloc[0]
            depth1 = l13_depth1.iloc[0]
            depth2 = l13_depth2.iloc[0]
            depth3 = l13_depth3.iloc[0]
            depth4 = l13_depth4.iloc[0]
            depth5 = l13_depth5.iloc[0]
            depth6 = l13_depth6.iloc[0]
            depth7 = l13_depth7.iloc[0]
            depth8 = l13_depth8.iloc[0]
            depth9 = l13_depth9.iloc[0]
            depth10 = l13_depth10.iloc[0]
            depth11 = l13_depth11.iloc[0]
            depth12 = l13_depth12.iloc[0]
            all_depth = l13_depth0_12.iloc[0]
            checks = [
                str(root.get("status", "")).startswith("PASS"),
                str(depth1.get("status", "")).startswith("PASS"),
                str(depth2.get("status", "")).startswith("PASS"),
                str(depth3.get("status", "")).startswith("PASS"),
                str(depth4.get("status", "")).startswith("PASS"),
                str(depth5.get("status", "")).startswith("PASS"),
                str(depth6.get("status", "")).startswith("PASS"),
                str(depth7.get("status", "")).startswith("PASS"),
                str(depth8.get("status", "")).startswith("PASS"),
                str(depth9.get("status", "")) == "PASS_COMPRESSED_SUFFIX_STATE_L13_PREFIX_DEPTH9_COMPACT_COVERAGE",
                str(depth10.get("status", "")) == "PASS_COMPRESSED_SUFFIX_STATE_L13_PREFIX_DEPTH10_COMPACT_COVERAGE",
                str(depth11.get("status", "")) == "PASS_COMPRESSED_SUFFIX_STATE_L13_PREFIX_DEPTH11_COMPACT_COVERAGE",
                str(depth12.get("status", ""))
                == "PASS_COMPRESSED_SUFFIX_STATE_L13_PREFIX_DEPTH12_TERMINAL_COMPACT_COVERAGE",
                str(all_depth.get("status", "")) == "PASS_COMPRESSED_SUFFIX_STATE_L13_DEPTH0_12_BRANCH_LAYER_LEDGER",
                str(root.get("compressed_total_count", "")) == "26062080",
                str(depth1.get("query_rows", "")) == "16",
                str(depth2.get("query_rows", "")) == "64",
                str(depth3.get("query_rows", "")) == "256",
                str(depth3.get("positive_branch_rows", "")) == "252",
                str(depth3.get("zero_branch_rows", "")) == "4",
                str(depth4.get("query_rows", "")) == "1012",
                str(depth4.get("valid_prefix_count", "")) == "252",
                str(depth4.get("invalid_prefix_rows", "")) == "4",
                str(depth4.get("positive_branch_rows", "")) == "996",
                str(depth4.get("valid_zero_branch_rows", "")) == "12",
                str(depth4.get("global_total_count", "")) == "26062080",
                str(depth5.get("query_rows", "")) == "4012",
                str(depth5.get("valid_prefix_count", "")) == "996",
                str(depth5.get("invalid_prefix_rows", "")) == "28",
                str(depth5.get("positive_branch_rows", "")) == "3936",
                str(depth5.get("valid_zero_branch_rows", "")) == "48",
                str(depth5.get("global_total_count", "")) == "26062080",
                str(depth6.get("query_rows", "")) == "15904",
                str(depth6.get("valid_prefix_count", "")) == "3936",
                str(depth6.get("invalid_prefix_rows", "")) == "160",
                str(depth6.get("positive_branch_rows", "")) == "15552",
                str(depth6.get("valid_zero_branch_rows", "")) == "192",
                str(depth6.get("global_total_count", "")) == "26062080",
                str(depth7.get("query_rows", "")) == "63040",
                str(depth7.get("valid_prefix_count", "")) == "15552",
                str(depth7.get("invalid_prefix_rows", "")) == "832",
                str(depth7.get("positive_branch_rows", "")) == "61128",
                str(depth7.get("valid_zero_branch_rows", "")) == "1080",
                str(depth7.get("global_total_count", "")) == "26062080",
                str(depth8.get("query_rows", "")) == "248920",
                str(depth8.get("valid_prefix_count", "")) == "61128",
                str(depth8.get("invalid_prefix_rows", "")) == "4408",
                str(depth8.get("positive_branch_rows", "")) == "235536",
                str(depth8.get("valid_zero_branch_rows", "")) == "8976",
                str(depth8.get("global_total_count", "")) == "26062080",
                str(depth9.get("prefix_count", "")) == "262144",
                str(depth9.get("valid_prefix_count", "")) == "235536",
                str(depth9.get("invalid_prefix_rows", "")) == "26608",
                str(depth9.get("query_rows_streamed", "")) == "968752",
                str(depth9.get("valid_branch_rows", "")) == "942144",
                str(depth9.get("positive_branch_rows", "")) == "871496",
                str(depth9.get("valid_zero_branch_rows", "")) == "70648",
                str(depth9.get("global_total_count", "")) == "26062080",
                str(depth9.get("depth8_branch_totals_match", "")) == "True",
                str(depth9.get("query_table_retained", "")) == "False",
                str(depth10.get("prefix_count", "")) == "1048576",
                str(depth10.get("valid_prefix_count", "")) == "871496",
                str(depth10.get("invalid_prefix_rows", "")) == "177080",
                str(depth10.get("query_rows_streamed", "")) == "3663064",
                str(depth10.get("valid_branch_rows", "")) == "3485984",
                str(depth10.get("positive_branch_rows", "")) == "3024992",
                str(depth10.get("valid_zero_branch_rows", "")) == "460992",
                str(depth10.get("global_total_count", "")) == "26062080",
                str(depth10.get("depth9_branch_totals_match", "")) == "True",
                str(depth10.get("parent_query_table_retained", "")) == "False",
                str(depth10.get("query_table_retained", "")) == "False",
                str(depth11.get("prefix_count", "")) == "4194304",
                str(depth11.get("valid_prefix_count", "")) == "3024992",
                str(depth11.get("invalid_prefix_rows", "")) == "1169312",
                str(depth11.get("query_rows_streamed", "")) == "13269280",
                str(depth11.get("valid_branch_rows", "")) == "12099968",
                str(depth11.get("positive_branch_rows", "")) == "9562368",
                str(depth11.get("valid_zero_branch_rows", "")) == "2537600",
                str(depth11.get("global_total_count", "")) == "26062080",
                str(depth11.get("depth10_branch_totals_match", "")) == "True",
                str(depth11.get("parent_query_table_retained", "")) == "False",
                str(depth11.get("query_table_retained", "")) == "False",
                str(depth12.get("prefix_count", "")) == "16777216",
                str(depth12.get("valid_prefix_count", "")) == "9562368",
                str(depth12.get("invalid_prefix_rows", "")) == "7214848",
                str(depth12.get("query_rows_streamed", "")) == "45464320",
                str(depth12.get("valid_branch_rows", "")) == "38249472",
                str(depth12.get("positive_branch_rows", "")) == "26062080",
                str(depth12.get("zero_branch_rows", "")) == "12187392",
                str(depth12.get("global_total_count", "")) == "26062080",
                str(depth12.get("terminal_branch_masses_are_binary", "")) == "True",
                str(depth12.get("query_table_retained", "")) == "False",
                str(all_depth.get("covered_depths", "")) == "0;1;2;3;4;5;6;7;8;9;10;11;12",
                str(all_depth.get("terminal_depth_recorded", "")) == "True",
                str(all_depth.get("total_prefix_nodes_including_root", "")) == "22369621",
                str(all_depth.get("compact_query_layers", "")) == "9;10;11;12",
                str(all_depth.get("terminal_layer_binary", "")) == "True",
                str(depth3.get("global_total_count", "")) == "26062080",
            ]
            return "depth 0-12" if all(checks) else "partial"
        if gate == "L13_RANK_UNRANK_SAMPLE":
            if l13_rank_unrank.empty:
                return "missing"
            row = l13_rank_unrank.iloc[0]
            checks = [
                str(row.get("status", "")) == "PASS_COMPRESSED_SUFFIX_STATE_RANK_UNRANK_L13",
                str(row.get("total_count", "")) == "26062080",
                str(row.get("sample_rows", "")) == "4",
                str(row.get("all_roundtrip", "")) == "True",
                str(row.get("all_sequences_valid", "")) == "True",
                str(row.get("known_count_match", "")) == "True",
                str(row.get("row_boundaries_ok", "")) == "True",
                str(row.get("coverage_boundary", "")) == "representative_four_rank_samples_not_exhaustive_all_rank_coverage",
            ]
            return "4/4 reps" if all(checks) else "partial"
        if gate == "FULL_L13_RANK_UNRANK":
            if full_l13_rank_unrank.empty:
                return "missing"
            row = full_l13_rank_unrank.iloc[0]
            checks = [
                str(row.get("status", "")) == "PASS_COMPRESSED_SUFFIX_STATE_FULL_LANGUAGE_RANK_UNRANK_L13",
                str(row.get("total_count", "")) == "26062080",
                str(row.get("checked_ranks", "")) == "26062080",
                str(row.get("roundtrip_failures", "")) == "0",
                str(row.get("invalid_sequences", "")) == "0",
                str(row.get("nonmonotone_sequences", "")) == "0",
            ]
            return "26.1M/26.1M" if all(checks) else "partial"
        if gate == "L13_REPLAY_ITERATOR":
            if l13_replay_iterator.empty:
                return "missing"
            decision = l13_replay_iterator[
                l13_replay_iterator["iterator_id"].eq("P2_L13_REPLAY_ITERATOR_DECISION")
            ]
            retained = l13_replay_iterator[
                l13_replay_iterator["iterator_id"].eq("P2_L13_REPLAY_ITERATOR_RETAINED_LOOKUP")
            ]
            compact = l13_replay_iterator[
                l13_replay_iterator["iterator_id"].eq("P2_L13_REPLAY_ITERATOR_COMPACT_STATUS")
            ]
            negatives = l13_replay_iterator[
                l13_replay_iterator["iterator_id"].eq("P2_L13_REPLAY_ITERATOR_FAIL_CLOSED_NEGATIVES")
            ]
            if decision.empty or retained.empty or compact.empty or negatives.empty:
                return "partial"
            row = decision.iloc[0]
            checks = [
                str(row.get("status", "")) == "PASS_PAPER2_L13_BRANCH_MASS_REPLAY_ITERATOR_BOUNDARY_LOCKED",
                str(row.get("chunks_seen", "")) == "45",
                str(row.get("replayed_chunks", "")) == "45",
                str(row.get("total_source_rows", "")) == "333470",
                str(row.get("total_source_bytes", "")) == "76858619",
                str(row.get("total_replayed_bytes", "")) == "76858619",
                str(retained.iloc[0].get("positive_query_cases", "")) == "9/9",
                str(compact.iloc[0].get("compact_status_cases", "")) == "4/4",
                str(negatives.iloc[0].get("negative_cases", "")) == "7/7",
            ]
            return "45/45 replay" if all(checks) else "partial"
        if gate == "L14_SELECTED_PREFIX_ORACLE":
            if l14_selected_prefix.empty:
                return "missing"
            row = l14_selected_prefix.iloc[0]
            checks = [
                str(row.get("status", "")) == "PASS_COMPRESSED_SUFFIX_STATE_L14_SELECTED_PREFIX_PROBE_BOUNDARY",
                str(row.get("length", "")) == "14",
                str(row.get("global_status", "")) == "timeout_boundary",
                str(row.get("query_rows", "")) == "16",
                str(row.get("observed_prefix_totals", "")) == "13;56;56;13",
                str(row.get("local_branch_partitions_match", "")) == "True",
                str(row.get("boundary_ok", "")) == "True",
            ]
            return "4 prefixes" if all(checks) else "partial"
        if gate == "MATERIAL_RECOVERY_INPUT_BUNDLE":
            hit = material_bundle[
                material_bundle["check_id"].eq("MATERIAL_RECOVERY_VALIDATION_INPUT_BUNDLE_DECISION")
            ]
            if hit.empty:
                return "missing"
            status = str(hit["status"].iloc[0])
            return "160/64/200" if status == "PASS_PAPER2_MATERIAL_RECOVERY_VALIDATION_INPUT_BUNDLE_NO_RESULTS" else "FAIL"
        if gate == "MATERIAL_RECOVERY_SENDABILITY":
            hit = material_sendability[material_sendability["check_id"].eq("OVERALL")]
            if hit.empty:
                return "missing"
            row = hit.iloc[0]
            status = str(row.get("status", ""))
            if (
                status == "PASS_PAPER2_MATERIAL_RECOVERY_DISPATCH_SENDABILITY_NO_RESULTS"
                and str(row.get("dispatch_sendable", "")) == "True"
                and str(row.get("nupack_results_recorded", "")) == "False"
                and str(row.get("empirical_recovery_results_recorded", "")) == "False"
            ):
                return "fixed set"
            return "FAIL"
        hit = methods_route[methods_route["gate"].eq(gate)]
        if hit.empty:
            return "missing"
        status = str(hit["status"].iloc[0])
        return "PASS" if status.startswith("PASS") else "FAIL"

    exact_rows = [l10_interval.iloc[0], l12_language.iloc[0], l13_language.iloc[0]]
    sampled_rows = [l14_language.iloc[0], l15_language.iloc[0], l16_language.iloc[0], l17_language.iloc[0]]
    exact_lengths = np.array([int(row["length"]) for row in exact_rows])
    exact_bits = np.array([float(row["log2_total_count"]) for row in exact_rows])
    exact_counts = np.array([int(row["total_count"]) for row in exact_rows])
    sampled_lengths = np.array([int(row["length"]) for row in sampled_rows])
    sampled_bits = np.array([float(row["log2_total_count"]) for row in sampled_rows])
    sampled_counts = np.array([int(row["total_count"]) for row in sampled_rows])

    all_lengths = np.concatenate([exact_lengths, sampled_lengths])
    all_bits = np.concatenate([exact_bits, sampled_bits])
    order_idx = np.argsort(all_lengths)
    ax.plot(
        all_lengths[order_idx],
        all_bits[order_idx],
        color=PALETTE["neutral_3"],
        linewidth=1.0,
        zorder=1,
    )
    ax.scatter(
        exact_lengths,
        exact_bits,
        s=38,
        color=PALETTE["green"],
        edgecolor="white",
        linewidth=0.55,
        label="all ranks checked",
        zorder=3,
    )
    ax.scatter(
        sampled_lengths,
        sampled_bits,
        s=38,
        facecolor="white",
        edgecolor=PALETTE["teal"],
        linewidth=1.25,
        label="four ranks checked",
        zorder=3,
    )

    def compact_count(value: int) -> str:
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        return f"{value / 1_000:.0f}k"

    for length, bits, count in zip(exact_lengths, exact_bits, exact_counts):
        ax.text(length + 0.12, bits - 0.55, compact_count(count), fontsize=5.8, color=PALETTE["green"])
    for length, bits, count in zip(sampled_lengths, sampled_bits, sampled_counts):
        ax.text(length + 0.12, bits + 0.35, compact_count(count), fontsize=5.8, color=PALETTE["teal"])
    ax.set_xticks(all_lengths)
    ax.set_xlim(9.4, 17.8)
    ax.set_ylim(16.8, 34.0)
    ax.set_xlabel("block length")
    ax.set_ylabel("log$_2$ exact language size")
    ax.legend(loc="upper left", handlelength=1.2)
    style_axis(ax)
    add_panel_footer(ax, "d", "Exact completion mass across short blocks", y=-0.24)

    save_figure(fig, out_dir / "proxy_thermo_signal.pdf")
    plt.close(fig)


def paper3_product_frontier() -> None:
    out_dir = FIGURES["paper3"]
    out_dir.mkdir(parents=True, exist_ok=True)

    current = pd.read_csv(DATA / "nanopore_r10_current_capacity_scan.csv")
    counts = pd.read_csv(DATA / "nanopore_product_counts_delta05_seed3.csv")
    gc_counts = pd.read_csv(DATA / "nanopore_product_counts_delta05_seed3_gc.csv")
    codec = pd.read_csv(DATA / "nanopore_product_codec_smoke_len30.csv")
    samples = pd.read_csv(DATA / "nanopore_product_delta05_ablation_110nt.csv")
    ablation_stats = pd.read_csv(
        DATA / "nanopore_product_delta05_ablation_110nt_bootstrap_summary.csv"
    ).set_index("policy")
    weighted_codec = pd.read_csv(DATA / "nanopore_product_weighted_practical_110_codec.csv")

    fig = plt.figure(figsize=(7.34, 5.05), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.055, h_pad=0.085, wspace=0.055, hspace=0.13)
    grid = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(grid[0, 0])
    ax.axvline(0.5, color=PALETTE["neutral_3"], linestyle=(0, (3, 2)), linewidth=0.9, zorder=0)
    ax.fill_between(
        current["delta"],
        current["capacity_bits_per_base_est"],
        2.0,
        where=current["capacity_bits_per_base_est"].le(2.0),
        color=PALETTE["blue_soft"],
        alpha=0.25,
        linewidth=0,
        zorder=0,
    )
    ax.plot(
        current["delta"],
        current["capacity_bits_per_base_est"],
        marker="o",
        markersize=3.8,
        color=PALETTE["blue"],
        linewidth=1.45,
    )
    ax.axhline(2.0, color=PALETTE["neutral_3"], linestyle=(0, (1.5, 2.5)), linewidth=0.85, zorder=0)
    ax.text(
        0.02,
        2.012,
        "2-bit unconstrained baseline",
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=PALETTE["neutral_2"],
        bbox=LABEL_BBOX,
    )
    op = current[current["delta"].round(6).eq(0.5)]
    if not op.empty:
        op_y = float(op["capacity_bits_per_base_est"].iloc[0])
        ax.scatter(
            [0.5],
            [op_y],
            s=44,
            color=PALETTE["gold"],
            edgecolor=PALETTE["neutral_0"],
            linewidth=0.45,
            zorder=4,
        )
        ax.annotate(
            "δ=0.5\noperating point",
            xy=(0.5, op_y),
            xytext=(0.62, op_y + 0.16),
            arrowprops=dict(arrowstyle="-", color=PALETTE["neutral_2"], lw=0.7),
            color=PALETTE["neutral_1"],
            fontsize=6.3,
            bbox=LABEL_BBOX,
        )
    ax.set_xlabel("current threshold δ")
    ax.set_ylabel("capacity (bits/base)")
    ax.set_title("Current separation imposes a rate cost", loc="left")
    subtitle(ax, "ONT R10.4.1 9-mer threshold graph")
    ax.set_ylim(0.98, 2.05)
    add_panel_label(ax, "a", x=-0.15, y=1.16)
    style_axis(ax)

    ax = fig.add_subplot(grid[0, 1])
    series = [
        ("current_only", "unbounded", "Current", PALETTE["blue"], 0.009),
        ("product_budget", "0", "Budget 0", PALETTE["salmon"], -0.011),
        ("product_budget", "2", "Budget 2", PALETTE["teal"], 0.002),
    ]
    for mode, budget, label, color, label_dy in series:
        sub = counts[(counts["mode"] == mode) & (counts["budget"].astype(str) == budget)]
        ax.plot(
            sub["length"],
            sub["rate_bits_per_base"],
            marker="o",
            markersize=3.8,
            linewidth=1.35,
            color=color,
            label=label,
        )
        if not sub.empty:
            direct_line_label(
                ax,
                float(sub["length"].iloc[-1]),
                float(sub["rate_bits_per_base"].iloc[-1]),
                label,
                color,
                dx=0.95,
                dy=label_dy,
            )
    sub = gc_counts[(gc_counts["mode"] == "product_budget") & (gc_counts["budget"].astype(str) == "2")]
    ax.plot(
        sub["length"],
        sub["rate_bits_per_base"],
        marker="s",
        markersize=3.6,
        linewidth=1.15,
        linestyle="--",
        color=PALETTE["violet"],
        label="Budget 2 + GC",
    )
    if not sub.empty:
        direct_line_label(
            ax,
            float(sub["length"].iloc[-1]),
            float(sub["rate_bits_per_base"].iloc[-1]),
            "Budget 2 + GC",
            PALETTE["violet"],
            dx=0.95,
            dy=-0.007,
        )
    smoke = codec.iloc[0]
    ax.scatter(
        [int(smoke["length"])],
        [float(smoke["rate_bits_per_base"])],
        marker="*",
        s=54,
        color=PALETTE["gold"],
        edgecolor=PALETTE["neutral_0"],
        linewidth=0.4,
        zorder=4,
    )
    ax.annotate(
        "rank/unrank\nsmoke test",
        xy=(int(smoke["length"]), float(smoke["rate_bits_per_base"])),
        xytext=(31.6, float(smoke["rate_bits_per_base"]) + 0.014),
        arrowprops=dict(arrowstyle="-", color=PALETTE["neutral_2"], lw=0.7),
        color=PALETTE["neutral_1"],
        fontsize=6.3,
        bbox=LABEL_BBOX,
    )
    ax.set_xlabel("block length")
    ax.set_ylabel("finite-block rate (bits/base)")
    ax.set_title("Product constraints retain code space", loc="left")
    subtitle(ax, "finite-block counts at δ=0.5 with rank/unrank witness")
    ax.set_xlim(18, 73.5)
    ax.set_ylim(1.575, 1.755)
    add_panel_label(ax, "b", x=-0.15, y=1.16)
    style_axis(ax)

    ax = fig.add_subplot(grid[1, 0])
    policies = ["random_gc_hp", "hairpin_only", "current_only", "product_joint"]
    labels = ["GC/HP\nonly", "Risk\nonly", "Current\nonly", "Joint"]
    colors = [PALETTE["neutral_3"], PALETTE["gold"], PALETTE["blue"], PALETTE["teal"]]
    positions = np.arange(len(policies))
    rng = np.random.default_rng(7)
    values = [samples.loc[samples["policy"].eq(policy), "weighted_pairs"].to_numpy() for policy in policies]
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["neutral_0"], "linewidth": 1.0},
        whiskerprops={"color": PALETTE["neutral_2"], "linewidth": 0.8},
        capprops={"color": PALETTE["neutral_2"], "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.24)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    for xpos, policy, vals, color in zip(positions, policies, values, colors):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            s=8,
            color=color,
            alpha=0.42,
            edgecolor="none",
            rasterized=True,
        )
        p95 = float(np.percentile(vals, 95))
        p95_low = float(ablation_stats.loc[policy, "weighted_p95_ci_low"])
        p95_high = float(ablation_stats.loc[policy, "weighted_p95_ci_high"])
        current_mean = float(samples.loc[samples["policy"].eq(policy), "current_violations"].mean())
        ax.hlines(p95, xpos - 0.28, xpos + 0.28, color=color, linewidth=1.25)
        ax.errorbar(
            xpos + 0.23,
            p95,
            yerr=[[p95 - p95_low], [p95_high - p95]],
            fmt="o",
            markersize=2.4,
            color=color,
            capsize=2.0,
            elinewidth=0.9,
            capthick=0.9,
            zorder=5,
        )
        ax.text(xpos, p95 + 8, f"p95 {p95:.1f}", color=color, ha="center", va="bottom", fontsize=5.8)
        ax.text(
            xpos,
            294,
            f"current violations\nmean {current_mean:.1f}",
            color=PALETTE["red"] if current_mean > 0 else PALETTE["green"],
            ha="center",
            va="top",
            fontsize=5.7,
            bbox=LABEL_BBOX,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-5, 305)
    ax.set_ylabel("weighted-pair proxy score")
    ax.set_title("Only joint policy controls both tested axes", loc="left")
    subtitle(ax, "100 oligos/arm; p95 whiskers show conditional 95% bootstrap CIs")
    add_panel_label(ax, "c", x=-0.15, y=1.16)
    style_axis(ax)

    ax = fig.add_subplot(grid[1, 1])
    ax.axhspan(80, 88, color=PALETTE["red_soft"], alpha=0.16, linewidth=0)
    ax.axhline(80, color=PALETTE["red"], linestyle=(0, (3, 2)), linewidth=0.9)
    ax.scatter(
        weighted_codec["payload_bits_consumed"],
        weighted_codec["weighted_pairs"],
        s=24,
        color=PALETTE["teal"],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.86,
    )
    ax.text(
        164.7,
        81.2,
        "budget 80",
        ha="right",
        va="bottom",
        fontsize=6.0,
        color=PALETTE["red"],
    )
    ax.text(
        142.4,
        10.0,
        "32/32 exact round-trips\nminimum current jump 0.500062",
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=PALETTE["neutral_1"],
        bbox=LABEL_BBOX,
    )
    ax.set_xlim(140.5, 165.5)
    ax.set_ylim(-4, 88)
    ax.set_xlabel("payload bits per 110-nt strand")
    ax.set_ylabel("weighted-pair proxy score")
    ax.set_title("Reversible streams satisfy both budgets", loc="left")
    subtitle(ax, "weighted-pair budget 80; ONT current threshold δ=0.5")
    add_panel_label(ax, "d", x=-0.15, y=1.16)
    style_axis(ax)

    save_figure(fig, out_dir / "product_frontier.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper",
        choices=["all", "paper1", "paper2", "paper3"],
        default="all",
        help="Regenerate one manuscript figure suite or all figure suites.",
    )
    args = parser.parse_args()
    setup_style()
    if args.paper in {"all", "paper1"}:
        paper1_robust_framework()
        paper1_capacity_landscape()
        paper1_certificate_convergence()
    if args.paper in {"all", "paper2"}:
        paper2_proxy_thermo()
    if args.paper in {"all", "paper3"}:
        paper3_product_frontier()


if __name__ == "__main__":
    main()
