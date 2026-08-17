#!/usr/bin/env python3
"""Render the public synthetic-aspect and Netlib-27 solver timing figures.

This script does not run solvers.  It normalizes the frozen raw benchmark
artifacts, writes one compact JSON evidence file, and renders SVG and PNG
figures from that evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTH = (
    ROOT / "records" / "twalker_synth_nm" /
    "ratio_1_to_512_20260816" / "summary.json"
)
DEFAULT_OUT = ROOT / "figures" / "solver_timings_20260816"

NETLIB_SOURCES = {
    "twalker_base": ROOT / "records" / "twalker_cpp" /
        "native_seed_t0_netlib27_full_budgeted_20260816.json",
    "twalker_fit1d": ROOT / "records" / "twalker_cpp" /
        "native_seed_t0_fit1d_long90_20260816.json",
    "twalker_lotfi": ROOT / "records" / "twalker_cpp" /
        "native_seed_t0_fit1d_lotfi_long_20260816.json",
    "newton": ROOT / "records" / "noise_floor" /
        "20260809T202623Z-pid60355" / "runs.jsonl",
    "highs": ROOT / "records" /
        "highs_baselines_20260809" /
        "hnp_baseline_results_20260809.jsonl",
    "wolfe_dual_twalker": ROOT / "records" /
        "wolfe_dual_twalker_netlib27_161.json",
}

SERIES = {
    "twalker": ("t-walker", "#3567a5", "o"),
    "newton": ("Newton (published SOTA)", "#238b8d", "s"),
    "simplex": ("HiGHS simplex", "#68737d", "^"),
    "ipm": ("HiGHS IPM", "#9b6a9c", "v"),
}
SYNTH_WALKER_ROWS = (
    ("twalker_newton", "Pure Mangasarian", "native Newton seed"),
)
FRONTIER_LABEL = "min(t-walker, Newton)"
SYNTH_FRONTIER_LABEL = "min(t-walkers, Newton)"
NETLIB_FRONTIER_LABEL = "min(t-walker, Wolfe-seed t-walker, Newton)"
FRONTIER_COLOR = "#111111"
WOLFE_DUAL_STYLE = ("Wolfe-seeded t-walker (end-to-end)", "#6d94bd", "P")
SYNTH_TRIANGULAR_STYLE = ("triangular-seeded t-walker", "#6d94bd", "P")
SYNTHETIC_MAX_RATIO = 32.0
POST_INITIALIZATION_FLOOR_MS = 0.05
REPRESENTATIVE_LP_RATIO_RANGE = (1.52, 2.87)

plt.rcParams["svg.hashsalt"] = "solver-timings-20260816"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _normalize_svg(path: Path) -> None:
    """Make Matplotlib SVG output deterministic and git-clean."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n",
                    encoding="utf-8")


def _json_lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _time_label(milliseconds, _position=None):
    if milliseconds >= 1000:
        value = milliseconds / 1000
        return f"{value:g} s"
    if milliseconds >= 1:
        return f"{milliseconds:g} ms"
    return f"{milliseconds * 1000:g} us"


def _sample_stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return {"milliseconds": statistics.median(values),
            "minimum_ms": min(values), "maximum_ms": max(values),
            "samples": values}


def _paired_samples(left, right, combine):
    left_by_repeat = {int(sample["repeat"]): sample
                      for sample in left.get("run_samples", [])}
    right_by_repeat = {int(sample["repeat"]): sample
                       for sample in right.get("run_samples", [])}
    values = []
    for repeat in sorted(left_by_repeat.keys() & right_by_repeat.keys()):
        value = combine(left_by_repeat[repeat], right_by_repeat[repeat])
        if value is not None:
            values.append(value)
    return values


def load_synthetic(summary_path: Path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for cell in summary["cells"].values():
        row = {"m": int(cell["m"]), "n": int(cell["n"]),
               "ratio": float(cell["ratio"]), "arms": {}}
        for arm, data in cell["arms"].items():
            seconds = data["seconds"]
            row["arms"][arm] = {
                "milliseconds": (1000 * float(seconds["median"])
                                 if seconds else None),
                "minimum_ms": (1000 * float(seconds["min"])
                               if seconds else None),
                "maximum_ms": (1000 * float(seconds["max"])
                               if seconds else None),
                "statuses": data["statuses"],
                "certified_runs": int(data["certified"]),
                "runs": int(data["runs"]),
                "solve_milliseconds": (
                    1000 * float(data["solve_seconds"]["median"])
                    if data.get("solve_seconds") else None),
                "run_samples": data.get("run_samples", []),
            }
        walker_arms = [key for key in ("twalker_newton", "twalker_triangular",
                                       "twalker_highs")
                       if key in row["arms"]]
        if not walker_arms and "twalker" in row["arms"]:
            walker_arms = ["twalker"]
        row["mangasarian_frontiers"] = {}
        for walker_arm in walker_arms:
            twalker = row["arms"][walker_arm]
            newton = row["arms"]["newton"]
            twalker_ok = (twalker["certified_runs"] == twalker["runs"]
                          and twalker["runs"] > 0)
            frontier_samples = _paired_samples(
                twalker, newton,
                lambda walker, baseline: 1000 * min(
                    float(walker["seconds"]), float(baseline["seconds"]))
                if walker.get("certified") else
                1000 * float(baseline["seconds"]))
            frontier_stats = _sample_stats(frontier_samples)
            source = (walker_arm if twalker_ok and
                      twalker["milliseconds"] < newton["milliseconds"]
                      else "newton")
            row["mangasarian_frontiers"][walker_arm] = {
                "milliseconds": (frontier_stats["milliseconds"]
                                 if frontier_stats else
                                 row["arms"][source]["milliseconds"]),
                "minimum_ms": (frontier_stats["minimum_ms"]
                               if frontier_stats else
                               row["arms"][source]["minimum_ms"]),
                "maximum_ms": (frontier_stats["maximum_ms"]
                               if frontier_stats else
                               row["arms"][source]["maximum_ms"]),
                "source": source,
                "rule": "fastest certified of this t-walker seed mode and shipped Newton",
            }
        complete_candidates = ["newton"] + [
            arm for arm in walker_arms
            if (row["arms"][arm]["certified_runs"] == row["arms"][arm]["runs"]
                and row["arms"][arm]["runs"] > 0)]
        complete_source = min(
            complete_candidates,
            key=lambda arm: row["arms"][arm]["milliseconds"])
        repeat_samples = []
        samples_by_arm = {
            arm: {int(sample["repeat"]): sample
                  for sample in row["arms"][arm].get("run_samples", [])}
            for arm in complete_candidates
        }
        common_repeats = set.intersection(*(
            set(samples) for samples in samples_by_arm.values()))
        for repeat in sorted(common_repeats):
            repeat_samples.append(1000 * min(
                float(samples_by_arm[arm][repeat]["seconds"])
                for arm in complete_candidates))
        complete_stats = _sample_stats(repeat_samples)
        row["mangasarian_frontier"] = {
            "milliseconds": (complete_stats["milliseconds"] if complete_stats
                             else row["arms"][complete_source]["milliseconds"]),
            "minimum_ms": (complete_stats["minimum_ms"] if complete_stats
                            else row["arms"][complete_source]["minimum_ms"]),
            "maximum_ms": (complete_stats["maximum_ms"] if complete_stats
                            else row["arms"][complete_source]["maximum_ms"]),
            "source": complete_source,
            "rule": "fastest certified t-walker seed mode or shipped Newton",
        }
        required = {"newton", "simplex", "simplex_init", "ipm", "ipm_init"}
        walker_pairs = [(arm, arm + "_init") for arm in walker_arms
                        if arm + "_init" in row["arms"]]
        if required.issubset(row["arms"]) and walker_pairs:
            post = {}
            for arm, init_arm in (walker_pairs
                                  + [("simplex", "simplex_init"),
                                     ("ipm", "ipm_init")]):
                paired = _paired_samples(
                    row["arms"][arm], row["arms"][init_arm],
                    lambda full, init: 1000 * (
                        float(full["seconds"]) - float(init["seconds"])))
                stats = _sample_stats([
                    max(POST_INITIALIZATION_FLOOR_MS, value)
                    for value in paired])
                raw = (row["arms"][arm]["milliseconds"]
                       - row["arms"][init_arm]["milliseconds"])
                post[arm] = dict(row["arms"][arm])
                post[arm]["raw_milliseconds"] = raw
                post[arm]["milliseconds"] = (
                    stats["milliseconds"] if stats else
                    max(POST_INITIALIZATION_FLOOR_MS, raw))
                post[arm]["minimum_ms"] = (
                    stats["minimum_ms"] if stats else
                    post[arm]["milliseconds"])
                post[arm]["maximum_ms"] = (
                    stats["maximum_ms"] if stats else
                    post[arm]["milliseconds"])
                post[arm]["post_samples"] = (stats["samples"] if stats else [])
                post[arm]["timing_floor_applied"] = raw <= 0.0
            post["newton"] = dict(row["arms"]["newton"])
            newton_samples = [
                1000 * float(sample["solve_seconds"])
                for sample in row["arms"]["newton"].get("run_samples", [])
                if sample.get("solve_seconds") is not None]
            newton_stats = _sample_stats([
                max(POST_INITIALIZATION_FLOOR_MS, value)
                for value in newton_samples])
            newton_core = row["arms"]["newton"]["solve_milliseconds"]
            post["newton"]["milliseconds"] = (
                newton_stats["milliseconds"] if newton_stats else
                max(POST_INITIALIZATION_FLOOR_MS, newton_core))
            post["newton"]["minimum_ms"] = (
                newton_stats["minimum_ms"] if newton_stats else
                post["newton"]["milliseconds"])
            post["newton"]["maximum_ms"] = (
                newton_stats["maximum_ms"] if newton_stats else
                post["newton"]["milliseconds"])
            post["newton"]["post_samples"] = (
                newton_stats["samples"] if newton_stats else [])
            post["newton"]["raw_milliseconds"] = newton_core
            post["newton"]["timing_floor_applied"] = newton_core <= 0.0
            row["post_initialization_arms"] = post
            post_candidates = ["newton"] + [
                arm for arm, _init in walker_pairs
                if (row["arms"][arm]["certified_runs"]
                    == row["arms"][arm]["runs"]
                    and row["arms"][arm]["runs"] > 0)]
            source = min(post_candidates,
                         key=lambda arm: post[arm]["milliseconds"])
            post_sample_lists = [post[arm].get("post_samples", [])
                                 for arm in post_candidates]
            post_frontier_stats = _sample_stats([
                min(values) for values in zip(*post_sample_lists)
            ])
            row["post_initialization_frontier"] = {
                "milliseconds": (post_frontier_stats["milliseconds"]
                                 if post_frontier_stats else
                                 post[source]["milliseconds"]),
                "minimum_ms": (post_frontier_stats["minimum_ms"]
                               if post_frontier_stats else
                               post[source]["minimum_ms"]),
                "maximum_ms": (post_frontier_stats["maximum_ms"]
                               if post_frontier_stats else
                               post[source]["maximum_ms"]),
                "source": source,
                "rule": "minimum certified post-initialization t-walker seed mode or Newton time",
            }
        rows.append(row)
    return sorted(rows, key=lambda row: (row["m"], row["ratio"])), summary


def load_netlib():
    base_rows = json.loads(NETLIB_SOURCES["twalker_base"].read_text())
    twalker = {row["model"]: row for row in base_rows}
    fit1d = json.loads(NETLIB_SOURCES["twalker_fit1d"].read_text())[0]
    twalker["fit1d"] = fit1d
    lotfi_rows = json.loads(NETLIB_SOURCES["twalker_lotfi"].read_text())
    twalker["lotfi"] = next(row for row in lotfi_rows
                             if row["model"] == "lotfi")

    newton_groups = {}
    for row in _json_lines(NETLIB_SOURCES["newton"]):
        newton_groups.setdefault(row["model"], []).append(
            1000 * float(row["wall_seconds"]))
    newton = {name: statistics.median(values)
              for name, values in newton_groups.items()}

    highs = {row["model"]: row
             for row in _json_lines(NETLIB_SOURCES["highs"])}
    wolfe_dual = {
        row["model"]: row
        for row in json.loads(
            NETLIB_SOURCES["wolfe_dual_twalker"].read_text(encoding="utf-8"))
    }

    rows = []
    for model in sorted(twalker):
        trow = twalker[model]
        walker = trow.get("walker") or {}
        wall_ms = walker.get("wall_ms")
        if wall_ms is None:
            process_seconds = trow.get("process_seconds")
            wall_ms = (1000 * float(process_seconds)
                       if process_seconds is not None else None)
        status = "CERTIFIED" if trow.get("status") == "CERTIFIED" else "DNF"
        hrow = highs[model]
        row = {
            "model": model,
            "n": int(hrow["n"]),
            "m": int(hrow["m"]),
            "ratio": float(hrow["n"]) / float(hrow["m"]),
            "twalker": {"milliseconds": wall_ms, "status": status,
                         "detail": trow.get("status")},
            "newton": {"milliseconds": newton[model],
                       "status": "CERTIFIED"},
            "simplex": {"milliseconds": 1000 * float(
                hrow["presolve_off__simplex"]["wall"]),
                "status": "OPTIMAL"},
            "ipm": {"milliseconds": 1000 * float(
                hrow["presolve_off__ipm"]["wall"]),
                "status": "OPTIMAL"},
        }
        if model in wolfe_dual:
            wolfe = wolfe_dual[model]
            wolfe_walker = wolfe.get("walker") or {}
            wolfe_status = wolfe.get("status", "UNKNOWN")
            wolfe_ms = wolfe_walker.get("wall_ms")
            lower_bound = wolfe_status == "RESOURCE_LIMIT"
            if wolfe_ms is None and wolfe.get("process_seconds") is not None:
                wolfe_ms = 1000 * float(wolfe["process_seconds"])
            if wolfe_ms is None and lower_bound:
                wolfe_ms = 20000.0
            row["wolfe_dual_twalker"] = {
                "milliseconds": wolfe_ms,
                "status": ("CERTIFIED" if wolfe_status == "CERTIFIED"
                           else wolfe_status),
                "detail": "opt-in triangular fixed-t seed; unchanged walker",
                "seed_ms": wolfe_walker.get("seed_ms"),
                "lower_bound": lower_bound,
            }
        else:
            row["wolfe_dual_twalker"] = None
        candidates = [("newton", row["newton"]["milliseconds"])]
        if row["twalker"]["status"] == "CERTIFIED":
            candidates.append(("twalker", row["twalker"]["milliseconds"]))
        if (row["wolfe_dual_twalker"] is not None
                and row["wolfe_dual_twalker"]["status"] == "CERTIFIED"):
            candidates.append(("wolfe_dual_twalker",
                               row["wolfe_dual_twalker"]["milliseconds"]))
        source, _ = min(candidates, key=lambda candidate: candidate[1])
        row["mangasarian_frontier"] = {
            "milliseconds": row[source]["milliseconds"],
            "source": source,
            "status": "CERTIFIED",
            "rule": "fastest certified of default-seed t-walker, Wolfe-seed t-walker, and shipped Newton",
        }
        rows.append(row)
    return rows


def render_synthetic(rows, output: Path, repeat_count: int):
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "semibold"})
    ms = sorted({row["m"] for row in rows})
    walker_arms = [arm for arm in ("twalker_newton", "twalker_triangular")
                   if arm in rows[0]["arms"]]
    if not walker_arms and "twalker" in rows[0]["arms"]:
        walker_arms = ["twalker"]
    available_rows = [("complete", "End to end", "seed + startup included")]
    if all("post_initialization_arms" in row for row in rows):
        available_rows.append(("post", "Post-initialization kernel",
                               "solver-specific startup removed"))
    figure_height = 4.6 if len(available_rows) == 1 else 6.05
    fig, axes = plt.subplots(len(available_rows), len(ms),
                             figsize=(12.6, figure_height), sharex=True,
                             sharey=False,
                             squeeze=False)
    for row_index, (mode, row_title, row_subtitle) in enumerate(available_rows):
        for column, m in enumerate(ms):
            axis = axes[row_index, column]
            certified_y_values = []
            dnf_points = []
            group = [row for row in rows
                     if row["m"] == m and row["ratio"] <= SYNTHETIC_MAX_RATIO]
            x = np.array([row["ratio"] for row in group])
            axis.axvspan(*REPRESENTATIVE_LP_RATIO_RANGE,
                         color="#80909b", alpha=.10, linewidth=0, zorder=0)
            if row_index == 0 and column == 0:
                axis.text(
                    2.09, .055, "real-world\nbenchmark examples",
                    transform=axis.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=6.8, color="#59656d",
                    linespacing=1.05, zorder=1)
            arm_source = ("arms" if mode == "complete"
                          else "post_initialization_arms")
            plotted = []
            if "twalker_newton" in walker_arms:
                plotted.append(("twalker_newton", SERIES["twalker"]))
            elif "twalker" in walker_arms:
                plotted.append(("twalker", SERIES["twalker"]))
            if "twalker_triangular" in walker_arms:
                plotted.append(("twalker_triangular", SYNTH_TRIANGULAR_STYLE))
            plotted.extend((("newton", SERIES["newton"]),
                            ("simplex", SERIES["simplex"]),
                            ("ipm", SERIES["ipm"])))
            for arm, (label, color, marker) in plotted:
                y = np.array([row[arm_source][arm]["milliseconds"]
                              for row in group])
                certified = np.array([
                    row[arm_source][arm]["certified_runs"]
                    == row[arm_source][arm]["runs"]
                    and row[arm_source][arm]["runs"] > 0 for row in group
                ])
                axis.plot(x[certified], y[certified], color=color,
                          marker=marker, linewidth=1.25, markersize=3.8,
                          markerfacecolor=("white" if arm == "twalker_triangular"
                                           else color),
                          alpha=.82, label=label, zorder=2)
                certified_y_values.extend(y[certified])
                if np.any(~certified):
                    dnf_points.extend((xi, yi, color)
                                      for xi, yi in zip(x[~certified],
                                                        y[~certified]))
            frontier_y = np.array([
                (row["mangasarian_frontier"]["milliseconds"]
                 if mode == "complete"
                 else row["post_initialization_frontier"]["milliseconds"])
                for row in group])
            axis.plot(x, frontier_y, color=FRONTIER_COLOR, linestyle="-",
                      marker="D", markerfacecolor="white",
                      markeredgecolor=FRONTIER_COLOR, markeredgewidth=1.2,
                      linewidth=2.65, markersize=5.1,
                      label=SYNTH_FRONTIER_LABEL, zorder=5)
            certified_y_values.extend(frontier_y)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlim(.98, SYNTHETIC_MAX_RATIO * 1.08)
            positive_y = np.asarray(certified_y_values)
            positive_y = positive_y[np.isfinite(positive_y) & (positive_y > 0)]
            if positive_y.size:
                panel_bottom = max(.045, float(np.min(positive_y)) / 1.45)
                panel_top = float(np.max(positive_y)) * 1.45
                axis.set_ylim(panel_bottom, panel_top)
                for xi, yi, color in dnf_points:
                    plotted_y = min(max(yi, panel_bottom * 1.08),
                                    panel_top / 1.08)
                    axis.plot(xi, plotted_y, linestyle="none", marker="x",
                              markersize=4.0, markeredgewidth=.75,
                              color=color, alpha=.32, clip_on=True, zorder=3)
            if row_index == 0:
                axis.set_title(f"{m} variables", pad=8)
            if row_index == len(available_rows) - 1:
                axis.set_xlabel("constraints / variables")
            axis.set_xticks([1, 1.1, 1.5, 2, 4, 8, 16, 32])
            axis.set_xticklabels(["1", "1.1", "1.5", "2", "4", "8", "16", "32×"],
                                 fontsize=8)
            tick_labels = axis.get_xticklabels()
            if len(tick_labels) >= 2:
                tick_labels[0].set_ha("right")
                tick_labels[1].set_ha("left")
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            axis.yaxis.set_major_formatter(FuncFormatter(_time_label))
            axis.yaxis.set_ticks_position("left")
            axis.tick_params(axis="y", right=False, labelright=False,
                             labelleft=True)
            axis.grid(axis="y", color="#d9d9d9", linewidth=.5)
            axis.spines[["top", "right"]].set_visible(False)
        axes[row_index, 0].set_ylabel(
            "complete solve time" if mode == "complete" else "kernel time")
        if len(available_rows) > 1:
            axes[row_index, 0].text(
                .025, .96, row_title + "\n" + row_subtitle,
                transform=axes[row_index, 0].transAxes, ha="left", va="top",
                fontsize=9.2, fontweight="semibold", linespacing=1.25,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88,
                      "pad": 1.5}, zorder=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="x",
                          markersize=4.2, markeredgewidth=.8,
                          color="#777777", alpha=.55))
    labels.append("DNF exit")
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper left",
               bbox_to_anchor=(.08, .84 if len(available_rows) == 1 else .915),
               handlelength=2.5)
    fig.suptitle("Mangasarian Pareto frontier closes Newton’s low-ratio gap",
                 x=.08, y=.99, ha="left", fontsize=15,
                 fontweight="bold")
    fig.text(.08, .90 if len(available_rows) == 1 else .94,
             "Ensemble = min(default-seed t-walker, triangular-seeded t-walker, published Newton SOTA).",
             color="#454545", fontsize=9.6)
    fig.text(.08, .015,
             f"{repeat_count}-run medians. Bottom: startup removed; 0.05 ms floor. "
             "Independent log scales by panel. × = DNF.",
             color="#666666")
    panel_top = .68 if len(available_rows) == 1 else .78
    panel_bottom = .14 if len(available_rows) == 1 else .115
    fig.subplots_adjust(left=.08, right=.985, top=panel_top, bottom=panel_bottom,
                        wspace=.22, hspace=.06)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight",
                metadata={"Date": None})
    _normalize_svg(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight",
                metadata={"Software": "Matplotlib"})
    plt.close(fig)


def render_netlib(rows, output: Path):
    ordered = sorted(rows, key=lambda row: (
        row["mangasarian_frontier"]["milliseconds"] or -1), reverse=True)
    fig, axis = plt.subplots(figsize=(10.6, 8.35))
    y = np.arange(len(ordered))
    for yi, row in zip(y, ordered):
        values = [row[key]["milliseconds"] for key in SERIES
                  if row[key]["milliseconds"] is not None]
        if row["wolfe_dual_twalker"] is not None:
            values.append(row["wolfe_dual_twalker"]["milliseconds"])
        axis.hlines(yi, min(values), max(values), color="#dddddd",
                    linewidth=.65, zorder=0)
    for key, (label, color, marker) in SERIES.items():
        xs, ys = [], []
        for yi, row in zip(y, ordered):
            value = row[key]["milliseconds"]
            if value is None:
                continue
            if key == "twalker" and row[key]["status"] != "CERTIFIED":
                axis.scatter(value, yi, color=color, marker="x", s=24,
                             linewidth=.85, alpha=.38, zorder=3)
            else:
                xs.append(value)
                ys.append(yi)
        axis.scatter(xs, ys, color=color, marker=marker, s=21,
                     label=label, zorder=3)
    wolfe_label, wolfe_color, wolfe_marker = WOLFE_DUAL_STYLE
    wolfe_x, wolfe_y, wolfe_exit_x, wolfe_exit_y = [], [], [], []
    for yi, row in zip(y, ordered):
        value = row["wolfe_dual_twalker"]
        if value is None:
            continue
        if value["status"] == "CERTIFIED":
            wolfe_x.append(value["milliseconds"])
            wolfe_y.append(yi)
        else:
            wolfe_exit_x.append(value["milliseconds"])
            wolfe_exit_y.append(yi)
    axis.scatter(wolfe_x, wolfe_y, facecolors="white",
                 edgecolors=wolfe_color, marker=wolfe_marker, s=35,
                 linewidth=1.15, label=wolfe_label, zorder=4)
    axis.scatter(wolfe_exit_x, wolfe_exit_y, color=wolfe_color, marker="x",
                 s=22, linewidth=.8, alpha=.35, zorder=4)
    frontier_x = [row["mangasarian_frontier"]["milliseconds"]
                  for row in ordered]
    axis.scatter(frontier_x, y, facecolors="white", edgecolors=FRONTIER_COLOR,
                 marker="D", s=39, linewidth=1.2, label=NETLIB_FRONTIER_LABEL,
                 zorder=5)
    axis.set_xscale("log")
    axis.set_yticks(y, [row["model"] for row in ordered])
    axis.invert_yaxis()
    axis.set_xlabel("complete solve time")
    axis.xaxis.set_major_formatter(FuncFormatter(_time_label))
    axis.grid(axis="x", color="#dddddd", linewidth=.55)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="x",
                          markersize=4.2, markeredgewidth=.8,
                          color=SERIES["twalker"][1], alpha=.42))
    labels.append("non-solve exit")
    fig.suptitle("Mangasarian Pareto frontier across Netlib-27",
                 x=.11, y=.99, ha="left", fontsize=15,
                 fontweight="bold")
    fig.text(.11, .95,
             "End-to-end solve time; lower is better. Wolfe seed changes initialization, then uses the same walker.",
             color="#454545", fontsize=9.6)
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper left",
               bbox_to_anchor=(.105, .925), handlelength=2.1)
    fig.text(.11, .018,
             "Cross-session results; compare order of magnitude. × = measured exit, not a solve; Fit1d Wolfe mark is a 20 s lower bound.",
             color="#666666", fontsize=8.4)
    fig.subplots_adjust(left=.11, right=.985, top=.815, bottom=.12)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight",
                metadata={"Date": None})
    _normalize_svg(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight",
                metadata={"Software": "Matplotlib"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-summary", type=Path,
                        default=DEFAULT_SYNTH)
    parser.add_argument("--synthetic-additional-summary", type=Path,
                        action="append", default=[])
    parser.add_argument("--synthetic-m-list", type=int, nargs="+", default=None)
    parser.add_argument(
        "--netlib-only", action="store_true",
        help="render only the Netlib figure and its compact evidence JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.netlib_only:
        netlib = load_netlib()
        ratios = [row["ratio"] for row in netlib]
        source_paths = dict(NETLIB_SOURCES)
        evidence = {
            "schema": "netlib-solver-timing-chart-data-v1",
            "units": "milliseconds",
            "mangasarian_frontier": {
                "members": ["twalker", "wolfe_dual_twalker", "newton"],
                "rule": "per program, fastest certified default-seed t-walker, Wolfe-seed t-walker, or Newton",
                "interpretation": "measured portfolio lower envelope, not an oracle-free dispatch policy",
            },
            "shape": {
                "ratio": "converted constraints / variables",
                "minimum": min(ratios),
                "median": statistics.median(ratios),
                "maximum": max(ratios),
            },
            "protocol": {
                "twalker": "default native t=0 seed; internal wall; one run; long-budget fit1d and lotfi overrides",
                "wolfe_dual_twalker": "opt-in triangular fixed-t seed followed by unchanged t-walker; full 27-program one-thread sweep; 20-second per-program limit",
                "newton": "shipped default staged Newton; per-program median of five runs",
                "highs": "cached 2026-08-09 baseline; one thread; presolve off; model build excluded; solver version was not recorded",
                "comparability": "different sessions; use for order-of-magnitude program-level comparison, not sub-10% claims",
            },
            "sources": {
                name: {"path": _display_path(path), "sha256": _sha256(path)}
                for name, path in source_paths.items()
            },
            "artifact_production": {
                "command": "python3 experiments/render_solver_timing_charts.py --netlib-only --output-dir figures/netlib27_latest_20260816",
                "renderer": str(Path(__file__).relative_to(ROOT)),
                "renderer_sha256": _sha256(Path(__file__)),
                "requirements": "requirements.txt",
                "requirements_sha256": _sha256(ROOT / "requirements.txt"),
            },
            "netlib": netlib,
        }
        data_path = args.output_dir / "netlib27_timing_data.json"
        data_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        render_netlib(netlib, args.output_dir / "netlib27_timings")
        print(json.dumps({
            "data": str(data_path),
            "netlib_svg": str(args.output_dir / "netlib27_timings.svg"),
        }, indent=2))
        return
    synthetic, synthetic_summary = load_synthetic(args.synthetic_summary)
    for extra_path in args.synthetic_additional_summary:
        extra_rows, _extra_summary = load_synthetic(extra_path)
        extra_keys = {(row["m"], row["ratio"]) for row in extra_rows}
        synthetic = [row for row in synthetic
                     if (row["m"], row["ratio"]) not in extra_keys]
        synthetic.extend(extra_rows)
    if args.synthetic_m_list is not None:
        selected_m = set(args.synthetic_m_list)
        synthetic = [row for row in synthetic if row["m"] in selected_m]
    synthetic.sort(key=lambda row: (row["m"], row["ratio"]))
    netlib = load_netlib()
    source_paths = {"synthetic": args.synthetic_summary, **NETLIB_SOURCES}
    for index, path in enumerate(args.synthetic_additional_summary, start=1):
        source_paths[f"synthetic_additional_{index}"] = path
    evidence = {
        "schema": "solver-timing-chart-data-v1",
        "units": "milliseconds",
        "mangasarian_frontier": {
            "members_by_row": {
                "complete": ["twalker_newton", "twalker_triangular", "newton"],
                "post_initialization": ["twalker_newton", "twalker_triangular", "newton"],
            },
            "rule": "per instance and row, fastest certified complete solve; all t-walker times include seed cost",
            "interpretation": "measured portfolio lower envelope, not an oracle-free dispatch policy",
        },
        "synthetic_protocol": synthetic_summary["parameters"],
        "synthetic_render_selection": {
            "m_list": sorted({row["m"] for row in synthetic}),
            "maximum_ratio": SYNTHETIC_MAX_RATIO,
        },
        "synthetic_timed_regions": synthetic_summary["timed_regions"],
        "synthetic_reproduction": synthetic_summary.get("reproduction"),
        "netlib_protocol": {
            "twalker": "default native t=0 seed; internal wall; one run; long-budget fit1d and lotfi overrides",
            "wolfe_dual_twalker": "opt-in triangular fixed-t seed followed by unchanged t-walker; full 27-program one-thread sweep; 20-second per-program limit",
            "newton": "shipped default staged Newton; per-program median of five runs",
            "highs": "cached 2026-08-09 HiGHS baseline; one thread; presolve off; model build excluded; solver version was not recorded",
            "comparability": "different sessions; use for order-of-magnitude program-level comparison, not sub-10% claims",
        },
        "sources": {name: {"path": _display_path(path),
                           "sha256": _sha256(path)}
                    for name, path in source_paths.items()},
        "artifact_production": {
            "renderer": str(Path(__file__).relative_to(ROOT)),
            "renderer_sha256": _sha256(Path(__file__)),
            "requirements": "requirements.txt",
            "requirements_sha256": _sha256(ROOT / "requirements.txt"),
        },
        "synthetic": synthetic,
        "netlib": netlib,
    }
    data_path = args.output_dir / "solver_timing_chart_data.json"
    data_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    render_synthetic(synthetic, args.output_dir / "synthetic_aspect_timings",
                     int(synthetic_summary["parameters"]["repeats"]))
    render_netlib(netlib, args.output_dir / "netlib27_timings")
    print(json.dumps({"data": str(data_path),
                      "synthetic_svg": str(args.output_dir /
                                             "synthetic_aspect_timings.svg"),
                      "netlib_svg": str(args.output_dir /
                                          "netlib27_timings.svg")},
                     indent=2))


if __name__ == "__main__":
    main()
