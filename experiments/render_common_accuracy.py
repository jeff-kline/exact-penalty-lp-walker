#!/usr/bin/env python3
"""Render the common original-data KKT accuracy comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "records/common_accuracy_20260817/accuracy_rows.json"
DEFAULT_OUTPUT = ROOT / "figures/common_accuracy_20260817/common_accuracy"

STYLES = {
    "twalker": ("t-walker", "#3567a5", "o"),
    "twalker_newton": ("t-walker", "#3567a5", "o"),
    "twalker_crossover": ("t-walker + crossover", "#3567a5", "o"),
    "twalker_triangular": ("triangular-seeded t-walker", "#6d94bd", "P"),
    "newton": ("Newton (published SOTA)", "#238b8d", "s"),
    "newton_crossover": ("Newton + crossover", "#238b8d", "s"),
    "simplex": ("HiGHS simplex", "#68737d", "^"),
    "ipm_no_crossover": ("HiGHS IPM, no crossover", "#b18aac", "v"),
    "ipm": ("HiGHS IPM + crossover", "#8d578f", "v"),
}
OPEN_ARMS = {"twalker_crossover", "newton_crossover",
             "twalker_triangular", "ipm_no_crossover"}


def _normalize_svg(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.input.read_text())

    panels = [
        ("netlib27", "Netlib-27", ("twalker", "twalker_crossover",
                                    "twalker_triangular", "newton",
                                    "newton_crossover", "simplex",
                                    "ipm_no_crossover", "ipm")),
        ("synthetic24", "Synthetic panel (24 cells)",
         ("twalker_newton", "twalker_crossover", "twalker_triangular",
          "newton", "newton_crossover", "simplex", "ipm_no_crossover",
          "ipm")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), sharex=True,
                             sharey=True)
    rng = np.random.default_rng(20260817)
    for axis, (suite, title, arms) in zip(axes, panels):
        suite_rows = [row for row in rows if row["suite"] == suite]
        problem_count = len({row["problem"] for row in suite_rows})
        for yi, arm in enumerate(arms):
            label, color, marker = STYLES[arm]
            arm_rows = [row for row in suite_rows if row["arm"] == arm]
            certified = [row for row in arm_rows
                         if row.get("status") == "CERTIFIED"
                         and row.get("certificate")]
            values = np.array([row["certificate"]["max"] for row in certified])
            jitter = rng.uniform(-.13, .13, len(values))
            scatter_style = ({"facecolors": "none", "edgecolors": color}
                             if arm in OPEN_ARMS else {"color": color})
            axis.scatter(values, yi + jitter, marker=marker,
                         s=15, linewidth=.55, alpha=.38, zorder=2,
                         **scatter_style)
            median = statistics.median(values)
            axis.scatter([median], [yi], facecolors="white", edgecolors=color,
                         marker="D", s=48, linewidth=1.6, zorder=4)
            axis.text(1.55e-7, yi,
                      f"{len(certified)}/{problem_count}", ha="left", va="center",
                      fontsize=8, color="#555555", clip_on=False)
        axis.axvline(1e-7, color="#a34c4c", linewidth=1.0, linestyle="--",
                     alpha=.8, zorder=1)
        axis.set_xscale("log")
        axis.set_xlim(3e-17, 1e-7)
        axis.set_ylim(len(arms) - .55, -.55)
        axis.set_yticks(range(len(arms)), [STYLES[arm][0] for arm in arms])
        axis.set_title(title, loc="left", pad=8, fontweight="semibold")
        axis.grid(axis="x", color="#dddddd", linewidth=.5)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_xlabel("maximum normalized KKT error (lower is better)")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].text(.985, .975, "acceptance tolerance", color="#874040",
                 fontsize=8, ha="right", va="top",
                 transform=axes[1].transAxes)
    legend = [
        Line2D([], [], linestyle="none", marker="o", color="#666666",
               alpha=.38, markersize=4, label="one problem"),
        Line2D([], [], linestyle="none", marker="D", markerfacecolor="white",
               markeredgecolor="#333333", markersize=6, label="median"),
    ]
    fig.legend(handles=legend, ncol=2, frameon=False, loc="upper right",
               bbox_to_anchor=(.98, .925))
    fig.suptitle("Original-data accuracy under one certificate",
                 x=.08, y=.98, ha="left", fontsize=15, fontweight="bold")
    fig.text(.08, .91,
             "Same componentwise primal, dual, nonnegativity, and gap scaling for every solver; labels show certified coverage.",
             color="#4d4d4d", fontsize=9.2)
    fig.text(.08, .015,
             "DNF/resource-limit cases are omitted from error points and remain visible in the coverage counts.",
             color="#666666", fontsize=8.3)
    fig.subplots_adjust(left=.22, right=.91, top=.78, bottom=.19, wspace=.22)
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight",
                metadata={"Date": None})
    _normalize_svg(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180, bbox_inches="tight",
                metadata={"Software": "Matplotlib"})
    plt.close(fig)


if __name__ == "__main__":
    main()
