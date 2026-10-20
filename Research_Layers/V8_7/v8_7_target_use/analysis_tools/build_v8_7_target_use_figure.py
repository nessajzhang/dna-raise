#!/usr/bin/env python3
"""Build the V8.7 held-out target-use figure from locked result files only."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


BLUE = "#1F5A94"
BLUE_LIGHT = "#8FB7D9"
ORANGE = "#C56A24"
NEUTRAL = "#6F7782"
NEUTRAL_LIGHT = "#C8CDD3"
RED = "#B54A4A"
INK = "#20252B"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )


def load_and_verify(root: Path) -> dict[str, object]:
    test_dir = root / "results" / "test"
    summary_path = test_dir / "adaptive_summary.csv"
    periodic_path = test_dir / "periodic_frontier.csv"
    gates_path = test_dir / "PRIMARY_GATE_TABLE.csv"
    primary_path = test_dir / "PRIMARY_RESULT.json"
    episodes_path = test_dir / "adaptive_episodes.csv.gz"

    summaries = read_csv(summary_path)
    summary_by_policy = {row["policy"]: row for row in summaries}
    require(len(summary_by_policy) == len(summaries), "adaptive summary policy rows are not unique")
    require(
        set(summary_by_policy)
        == {
            "posterior_mean",
            "posterior_mean+screen",
            "posterior_conservative",
            "posterior_conservative+screen",
        },
        "adaptive summary policy set changed",
    )
    selected_rows = [r for r in summaries if r["policy"] == "posterior_mean+screen"]
    require(len(selected_rows) == 1, "selected policy summary row is not unique")
    selected = selected_rows[0]
    require(int(selected["missions"]) == 60000, "unexpected selected-policy mission count")
    require(int(selected["hazards"]) == 3, "unexpected hazard count")

    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    require(primary["all_primary_gates_pass"] is True, "primary gates did not all pass")
    require(primary["selected_adaptive_policy"] == selected["policy"], "selection mismatch")
    require(primary["selected_periodic_risk_only_interval"] == 5, "unexpected comparator")
    require(
        primary["selected_periodic_joint_interval"] == "NO_QUALIFIED_PERIODIC_JOINT",
        "periodic joint-comparator status changed",
    )

    gates = {r["gate"]: r for r in read_csv(gates_path)}
    required_gates = {
        "risk_h0020",
        "risk_h0060",
        "risk_h0120",
        "eligible_passage",
        "overall_authorization",
        "refresh_gain",
    }
    require(set(gates) == required_gates, "primary gate set changed")
    require(all(int(gates[g]["pass"]) == 1 for g in required_gates), "a primary gate failed")

    descriptive_base = [summary_by_policy["posterior_mean"], summary_by_policy["posterior_conservative"]]
    require(all(int(row["missions"]) == 60000 for row in descriptive_base), "base-policy mission count changed")
    require(all(int(row["hazards"]) == 3 for row in descriptive_base), "base-policy hazard count changed")
    require(int(summary_by_policy["posterior_mean"]["risk_pass_all_strata"]) == 0, "posterior-mean risk status changed")
    require(
        int(summary_by_policy["posterior_conservative"]["risk_pass_all_strata"]) == 1,
        "posterior-0.95-quantile risk status changed",
    )

    periodic_rows = read_csv(periodic_path)
    by_interval: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in periodic_rows:
        by_interval[int(row["periodic_interval_years"])].append(row)
    require(set(by_interval) == set(range(5, 61, 5)), "periodic interval grid changed")
    periodic: list[dict[str, object]] = []
    for interval in sorted(by_interval):
        rows = by_interval[interval]
        require(len(rows) == 3, f"periodic interval {interval} lacks three hazard strata")
        refreshes = {float(r["mean_refreshes_equal_weight_domain"]) for r in rows}
        auth = {float(r["authorization_rate_equal_weight_domain"]) for r in rows}
        risk_pass = {int(r["risk_pass_all_strata"]) for r in rows}
        joint_pass = {int(r["joint_nonvacuity_pass"]) for r in rows}
        require(
            len(refreshes) == len(auth) == len(risk_pass) == len(joint_pass) == 1,
            f"periodic aggregate fields disagree across hazard rows at {interval} years",
        )
        periodic.append(
            {
                "interval": interval,
                "mean_refreshes": refreshes.pop(),
                "authorization_rate": auth.pop(),
                "risk_pass": risk_pass.pop(),
                "joint_pass": joint_pass.pop(),
            }
        )
    require(sum(int(r["risk_pass"]) for r in periodic) == 1, "risk-qualified periodic count changed")
    require(all(int(r["joint_pass"]) == 0 for r in periodic), "a joint periodic comparator appeared")

    raw_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    raw_totals = defaultdict(int)
    with gzip.open(episodes_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] != selected["policy"]:
                continue
            hazard = f"{float(row['true_hazard']):.3f}"
            raw_counts[hazard]["missions"] += 1
            raw_counts[hazard]["events"] += int(row["any_authorized_under_target"])
            raw_totals["refreshes"] += int(row["refreshes"])
            raw_totals["authorizations"] += int(row["final_no_refresh"])
            raw_totals["eligible_candidates"] += int(row["eligible_candidates"])
            raw_totals["eligible_passes"] += int(row["eligible_passes"])

    require(set(raw_counts) == {"0.002", "0.006", "0.012"}, "raw selected-policy hazards changed")
    require(all(v["missions"] == 20000 for v in raw_counts.values()), "per-hazard test count changed")
    risk_events = json.loads(selected["risk_events_json"])
    expected_events = {
        "0.002": int(risk_events["h0020"]),
        "0.006": int(risk_events["h0060"]),
        "0.012": int(risk_events["h0120"]),
    }
    require(
        all(raw_counts[h]["events"] == expected_events[h] for h in expected_events),
        "raw risk-event counts disagree with the locked summary",
    )
    require(close(raw_totals["refreshes"] / 60000, float(selected["mean_refreshes"])), "refresh mean mismatch")
    require(
        close(raw_totals["authorizations"] / (60000 * 11), float(selected["overall_authorization_rate"])),
        "authorization-rate mismatch",
    )
    require(raw_totals["eligible_candidates"] == int(selected["eligible_candidates"]), "eligible count mismatch")
    require(raw_totals["eligible_passes"] == int(selected["eligible_passes"]), "eligible pass mismatch")

    return {
        "selected": selected,
        "primary": primary,
        "gates": gates,
        "periodic": periodic,
        "descriptive_base": descriptive_base,
        "raw_counts": raw_counts,
        "raw_totals": raw_totals,
        "input_paths": [summary_path, periodic_path, gates_path, primary_path, episodes_path],
    }


def write_source_data(path: Path, payload: dict[str, object]) -> None:
    selected = payload["selected"]
    primary = payload["primary"]
    gates = payload["gates"]
    periodic = payload["periodic"]
    descriptive_base = payload["descriptive_base"]
    raw_counts = payload["raw_counts"]
    risk_rates = json.loads(selected["risk_rates_json"])
    risk_upper = json.loads(selected["risk_upper_json"])

    fields = [
        "panel",
        "record",
        "x",
        "y",
        "point_estimate",
        "lower_bound",
        "upper_bound",
        "threshold",
        "numerator",
        "denominator",
        "mean_refreshes",
        "comparator_mean_refreshes",
        "risk_qualified",
        "joint_qualified",
        "interval_method",
        "analysis_unit",
        "source_file",
        "note",
    ]
    rows: list[dict[str, object]] = []
    for row in periodic:
        rows.append(
            {
                "panel": "a",
                "record": f"periodic_{row['interval']}_years",
                "x": row["mean_refreshes"],
                "y": 100 * row["authorization_rate"],
                "point_estimate": 100 * row["authorization_rate"],
                "mean_refreshes": row["mean_refreshes"],
                "risk_qualified": row["risk_pass"],
                "joint_qualified": row["joint_pass"],
                "analysis_unit": "deterministic periodic path across three hazards",
                "source_file": "periodic_frontier.csv",
                "note": "three hazard rows share the displayed equal-weight-domain aggregates",
            }
        )
    for row in descriptive_base:
        risk_upper_triple = json.loads(row["risk_upper_json"])
        rows.append(
            {
                "panel": "a",
                "record": row["policy"],
                "x": float(row["mean_refreshes"]),
                "y": 100 * float(row["overall_authorization_rate"]),
                "point_estimate": 100 * float(row["overall_authorization_rate"]),
                "mean_refreshes": float(row["mean_refreshes"]),
                "risk_qualified": int(row["risk_pass_all_strata"]),
                "analysis_unit": "complete 60-year mission",
                "source_file": "adaptive_summary.csv",
                "note": (
                    "prespecified descriptive unwrapped-policy check; risk upper triple="
                    + json.dumps(risk_upper_triple, sort_keys=True)
                    + "; no screen-based finite-horizon qualification"
                ),
            }
        )
    rows.append(
        {
            "panel": "a",
            "record": "posterior_mean_plus_screen",
            "x": float(selected["mean_refreshes"]),
            "y": 100 * float(selected["overall_authorization_rate"]),
            "point_estimate": 100 * float(selected["overall_authorization_rate"]),
            "mean_refreshes": float(selected["mean_refreshes"]),
            "risk_qualified": 1,
            "joint_qualified": 1,
            "analysis_unit": "complete 60-year mission",
            "source_file": "adaptive_summary.csv",
            "note": "tuning-selected policy evaluated once on the held-out test set",
        }
    )

    hazard_keys = [("0.002", "h0020"), ("0.006", "h0060"), ("0.012", "h0120")]
    for hazard, key in hazard_keys:
        rows.append(
            {
                "panel": "b",
                "record": f"risk_hazard_{hazard}",
                "x": hazard,
                "y": 100 * float(risk_upper[key]),
                "point_estimate": 100 * float(risk_rates[key]),
                "upper_bound": 100 * float(risk_upper[key]),
                "threshold": 5.0,
                "numerator": raw_counts[hazard]["events"],
                "denominator": raw_counts[hazard]["missions"],
                "interval_method": "one-sided Clopper-Pearson upper bound; alpha_gate=0.008333333333333333",
                "analysis_unit": "complete 60-year mission",
                "source_file": "adaptive_summary.csv; adaptive_episodes.csv.gz; PRIMARY_GATE_TABLE.csv",
                "note": "zero event counts are represented by their positive exact upper bound",
            }
        )

    rows.extend(
        [
            {
                "panel": "c",
                "record": "eligible_passage",
                "x": "Eligible passage",
                "y": 100 * float(selected["eligible_passage_rate"]),
                "point_estimate": 100 * float(selected["eligible_passage_rate"]),
                "lower_bound": 100 * float(gates["eligible_passage"]["estimate_or_bound"]),
                "threshold": 100 * float(gates["eligible_passage"]["threshold"]),
                "numerator": int(selected["eligible_passes"]),
                "denominator": int(selected["eligible_candidates"]),
                "interval_method": "complete-mission cluster-bootstrap lower bound; B=20000; alpha_gate=0.008333333333333333",
                "analysis_unit": "complete 60-year mission cluster",
                "source_file": "adaptive_summary.csv; PRIMARY_GATE_TABLE.csv",
                "note": "candidate looks are retained within mission clusters",
            },
            {
                "panel": "c",
                "record": "overall_authorization",
                "x": "Authorization",
                "y": 100 * float(selected["overall_authorization_rate"]),
                "point_estimate": 100 * float(selected["overall_authorization_rate"]),
                "lower_bound": 100 * float(gates["overall_authorization"]["estimate_or_bound"]),
                "threshold": 100 * float(gates["overall_authorization"]["threshold"]),
                "numerator": int(payload["raw_totals"]["authorizations"]),
                "denominator": 60000 * 11,
                "interval_method": "complete-mission cluster-bootstrap lower bound; B=20000; alpha_gate=0.008333333333333333",
                "analysis_unit": "complete 60-year mission cluster",
                "source_file": "adaptive_summary.csv; adaptive_episodes.csv.gz; PRIMARY_GATE_TABLE.csv",
                "note": "eleven scheduled decisions per mission",
            },
        ]
    )

    rows.append(
        {
            "panel": "d",
            "record": "screen_minus_risk_only_periodic_refreshes",
            "x": "posterior_mean+screen minus 5-year periodic",
            "y": float(primary["refresh_difference_point_estimate"]),
            "point_estimate": float(primary["refresh_difference_point_estimate"]),
            "upper_bound": float(primary["refresh_difference_upper"]),
            "threshold": -0.5,
            "mean_refreshes": float(selected["mean_refreshes"]),
            "comparator_mean_refreshes": 11.0,
            "interval_method": "paired complete-mission bootstrap upper bound; B=20000; alpha_gate=0.008333333333333333",
            "analysis_unit": "complete 60-year mission",
            "source_file": "PRIMARY_GATE_TABLE.csv; PRIMARY_RESULT.json; adaptive_summary.csv",
            "note": "five-year periodic schedule is the only periodic schedule passing the risk criterion; no periodic schedule is jointly qualified",
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_figure(output_base: Path, payload: dict[str, object]) -> None:
    selected = payload["selected"]
    primary = payload["primary"]
    gates = payload["gates"]
    periodic = payload["periodic"]
    descriptive_base = payload["descriptive_base"]
    raw_counts = payload["raw_counts"]
    risk_upper = json.loads(selected["risk_upper_json"])

    fig_width_mm = 183
    fig_height_mm = 132
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(fig_width_mm / 25.4, fig_height_mm / 25.4),
        gridspec_kw={"width_ratios": [1.15, 1.0], "height_ratios": [1.05, 0.95]},
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # a: comparator frontier
    for row in periodic:
        passed = bool(row["risk_pass"])
        ax_a.scatter(
            row["mean_refreshes"],
            100 * row["authorization_rate"],
            s=28,
            marker="o",
            facecolor=NEUTRAL if passed else "white",
            edgecolor=NEUTRAL,
            linewidth=1.0,
            zorder=2,
        )
    base_labels = {
        "posterior_mean": ("PM base", (-24, 18), "right"),
        "posterior_conservative": ("Q95 base", (10, -12), "left"),
    }
    for row in descriptive_base:
        passed = bool(int(row["risk_pass_all_strata"]))
        x_value = float(row["mean_refreshes"])
        y_value = 100 * float(row["overall_authorization_rate"])
        ax_a.scatter(
            x_value,
            y_value,
            s=38,
            marker="^",
            facecolor=ORANGE if passed else "white",
            edgecolor=ORANGE,
            linewidth=1.0,
            zorder=4,
        )
        label, offset, align = base_labels[row["policy"]]
        ax_a.annotate(
            label,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=6.6,
            color=ORANGE,
            arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.6},
        )
    adaptive_x = float(selected["mean_refreshes"])
    adaptive_y = 100 * float(selected["overall_authorization_rate"])
    ax_a.scatter(
        adaptive_x,
        adaptive_y,
        s=46,
        marker="D",
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    ax_a.axhline(5.0, color=RED, linestyle=(0, (3, 2)), linewidth=1.0, zorder=1)
    ax_a.text(0.2, 7.5, "Authorization floor", color=RED, fontsize=6.6, ha="left", va="bottom")
    grouped_labels = {
        (11.0, 0.0): ("5 y", -15, 6),
        (5.0, 54.54545454545454): ("10 y", 5, -1),
        (3.0, 72.72727272727273): ("15 y", 5, -1),
        (2.0, 81.81818181818181): ("20–25 y", 5, -1),
        (1.0, 90.9090909090909): ("30–55 y", 5, -1),
        (0.0, 100.0): ("60 y", 5, -7),
    }
    for (x, y), (label, dx, dy) in grouped_labels.items():
        ax_a.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.6, color=INK)
    ax_a.annotate(
        "Selected screen",
        (adaptive_x, adaptive_y),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=BLUE,
    )
    ax_a.set_xlim(-0.55, 11.75)
    ax_a.set_ylim(-4, 106)
    ax_a.set_xlabel("Mean refreshes per mission")
    ax_a.set_ylabel("Authorized no-refresh decisions (%)")
    ax_a.set_title("Comparator frontier", loc="left", fontweight="bold")
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="D", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=6, label="Selected screen; joint pass"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor=ORANGE, markeredgecolor=ORANGE, markersize=5.5, label="Unscreened; risk pass"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor=ORANGE, markersize=5.5, label="Unscreened; risk fail"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=NEUTRAL, markeredgecolor=NEUTRAL, markersize=5, label="Periodic; risk pass"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=NEUTRAL, markersize=5, label="Periodic; risk fail"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.99),
        handletextpad=0.5,
        borderaxespad=0,
        fontsize=6.1,
    )
    add_panel_label(ax_a, "a")

    # b: hazard-specific familywise risk bounds
    hazards = [0.002, 0.006, 0.012]
    keys = ["h0020", "h0060", "h0120"]
    upper_pct = [100 * float(risk_upper[k]) for k in keys]
    require(all(value > 0 for value in upper_pct), "log-axis bounds must be strictly positive")
    event_labels = [f"{raw_counts[f'{h:.3f}']['events']:,}/20,000" for h in hazards]
    ax_b.plot(range(3), upper_pct, color=BLUE_LIGHT, marker="o", markerfacecolor=BLUE, markeredgecolor="white", markersize=5.8)
    ax_b.axhline(5.0, color=RED, linestyle=(0, (3, 2)), linewidth=1.0)
    ax_b.text(2.05, 5.15, "5% gate", color=RED, fontsize=6.6, ha="right", va="bottom")
    for idx, (value, label) in enumerate(zip(upper_pct, event_labels)):
        ax_b.annotate(label, (idx, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.6, color=INK)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.01, 10)
    ax_b.set_xticks(range(3), [f"{h:.3f}" for h in hazards])
    ax_b.set_yticks([0.01, 0.1, 1, 10])
    ax_b.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax_b.set_xlabel("Synthetic annualized hazard")
    ax_b.set_ylabel("Exact upper bound (%)")
    ax_b.set_title("Held-out mission-level risk", loc="left", fontweight="bold")
    add_panel_label(ax_b, "b")

    # c: non-vacuity gates
    metrics = [
        (
            "Eligible passage",
            100 * float(selected["eligible_passage_rate"]),
            100 * float(gates["eligible_passage"]["estimate_or_bound"]),
            100 * float(gates["eligible_passage"]["threshold"]),
        ),
        (
            "Authorization",
            100 * float(selected["overall_authorization_rate"]),
            100 * float(gates["overall_authorization"]["estimate_or_bound"]),
            100 * float(gates["overall_authorization"]["threshold"]),
        ),
    ]
    y_positions = [1, 0]
    for y_pos, (label, point, lower, threshold) in zip(y_positions, metrics):
        ax_c.hlines(y_pos, lower, point, color=BLUE_LIGHT, linewidth=2.0, zorder=1)
        ax_c.scatter(point, y_pos, color=BLUE, s=34, marker="o", edgecolor="white", linewidth=0.6, zorder=3)
        ax_c.scatter(lower, y_pos, color=BLUE, s=30, marker="|", linewidth=1.4, zorder=3)
        ax_c.scatter(threshold, y_pos, facecolor=RED, edgecolor="white", s=32, marker="<", linewidth=0.5, zorder=3)
        ax_c.text(point + 2.2, y_pos, f"{point:.2f}%", va="center", ha="left", fontsize=6.8, color=INK)
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(-0.65, 1.65)
    ax_c.set_yticks(y_positions, [m[0] for m in metrics])
    ax_c.set_xlabel("Decision proportion (%)")
    ax_c.set_title("Non-vacuity gates", loc="left", fontweight="bold")
    ax_c.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=5.5, label="Point estimate"),
            Line2D([0], [0], marker="|", color=BLUE, markersize=7, label="Lower bound"),
            Line2D([0], [0], marker="<", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=5.5, label="Prespecified floor"),
        ],
        loc="lower right",
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    add_panel_label(ax_c, "c")

    # d: paired burden contrast
    point = float(primary["refresh_difference_point_estimate"])
    upper = float(primary["refresh_difference_upper"])
    threshold = -0.5
    ax_d.axvline(0, color=NEUTRAL_LIGHT, linewidth=0.9)
    ax_d.axvline(threshold, color=RED, linestyle=(0, (3, 2)), linewidth=1.0)
    ax_d.hlines(0, point, upper, color=BLUE_LIGHT, linewidth=3.0, zorder=1)
    ax_d.scatter(point, 0, color=BLUE, s=40, marker="o", edgecolor="white", linewidth=0.6, zorder=3)
    ax_d.scatter(upper, 0, color=BLUE, s=34, marker="|", linewidth=1.5, zorder=3)
    ax_d.annotate(f"Point {point:.3f}", (point, 0), xytext=(-2, 12), textcoords="offset points", ha="left", fontsize=6.8, color=INK)
    ax_d.annotate(f"Upper {upper:.3f}", (upper, 0), xytext=(3, -13), textcoords="offset points", ha="left", fontsize=6.8, color=INK)
    ax_d.text(threshold - 0.04, 0.45, "Required < −0.5", color=RED, fontsize=6.6, ha="right", va="center")
    ax_d.set_xlim(-3.55, 0.15)
    ax_d.set_ylim(-0.75, 0.75)
    ax_d.set_yticks([])
    ax_d.set_xlabel("Screen − 5-y periodic refreshes per mission")
    ax_d.set_title("Refresh-burden contrast", loc="left", fontweight="bold")
    ax_d.spines["left"].set_visible(False)
    add_panel_label(ax_d, "d")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".svg"))
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    payload = load_and_verify(root)
    write_source_data(args.source_data.resolve(), payload)
    build_figure(args.output_base.resolve(), payload)
    provenance = {
        "figure": "nc_v8_7_target_use",
        "backend": "Python matplotlib",
        "data_generation": "deterministic aggregation of locked V8.7 result files; no simulation",
        "physical_qualification": "NOT_RUN",
        "input_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in payload["input_paths"]
        },
        "source_data_sha256": sha256_file(args.source_data.resolve()),
        "exports_sha256": {
            suffix[1:]: sha256_file(args.output_base.resolve().with_suffix(suffix))
            for suffix in (".pdf", ".svg", ".tiff", ".png")
        },
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS_V8_7_TARGET_USE_FIGURE_BUILD")
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
