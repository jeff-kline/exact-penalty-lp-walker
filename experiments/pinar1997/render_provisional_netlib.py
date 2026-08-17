#!/usr/bin/env python3
"""Render a provisional Netlib figure with Pinar kept outside the frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
CACHED = (ROOT / "figures" /
          "netlib27_latest_20260816" / "netlib27_timing_data.json")

SERIES = {
    "twalker": ("t-walker", "#3567a5", "o"),
    "newton": ("Newton (published SOTA)", "#238b8d", "s"),
    "simplex": ("HiGHS simplex", "#68737d", "^"),
    "ipm": ("HiGHS IPM", "#9b6a9c", "v"),
}
WOLFE = ("Wolfe-seeded t-walker", "#6d94bd", "P")
PINAR = ("Pinar 1997 reference", "#a33f55", "*")
FRONTIER = ("min(t-walker, Wolfe-seed t-walker, Newton)", "#111111", "D")


def time_label(value, _position=None):
    if value < 1:
        return f"{value * 1000:g} us"
    if value < 1000:
        return f"{value:g} ms"
    if value < 60000:
        return f"{value / 1000:g} s"
    return f"{value / 60000:g} min"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cached = json.loads(CACHED.read_text())
    pinar_payload = json.loads(args.pinar.read_text())
    pinar = {row["model"]: row for row in pinar_payload["models"]}
    rows = sorted(cached["netlib"],
                  key=lambda row: row["mangasarian_frontier"]["milliseconds"],
                  reverse=True)

    fig, axis = plt.subplots(figsize=(11.2, 8.7))
    y = np.arange(len(rows))
    for yi, row in zip(y, rows):
        values = [row[key]["milliseconds"] for key in SERIES]
        if row.get("wolfe_dual_twalker"):
            values.append(row["wolfe_dual_twalker"]["milliseconds"])
        prow = pinar.get(row["model"])
        if prow and prow.get("status") == "CERTIFIED":
            values.append(prow["elapsed_ms"])
        axis.hlines(yi, min(values), max(values), color="#dddddd",
                    linewidth=.65, zorder=0)

    for key, (label, color, marker) in SERIES.items():
        xs, ys, exits_x, exits_y = [], [], [], []
        for yi, row in zip(y, rows):
            value = row[key]
            if value["milliseconds"] is None:
                continue
            if key == "twalker" and value["status"] != "CERTIFIED":
                exits_x.append(value["milliseconds"])
                exits_y.append(yi)
            else:
                xs.append(value["milliseconds"])
                ys.append(yi)
        axis.scatter(xs, ys, color=color, marker=marker, s=22,
                     label=label, zorder=3)
        axis.scatter(exits_x, exits_y, color=color, marker="x", s=17,
                     linewidth=.7, alpha=.25, zorder=2)

    wolfe_label, wolfe_color, wolfe_marker = WOLFE
    wx, wy, wdx, wdy = [], [], [], []
    for yi, row in zip(y, rows):
        value = row.get("wolfe_dual_twalker")
        if not value:
            continue
        target = (wx, wy) if value["status"] == "CERTIFIED" else (wdx, wdy)
        target[0].append(value["milliseconds"])
        target[1].append(yi)
    axis.scatter(wx, wy, facecolors="white", edgecolors=wolfe_color,
                 marker=wolfe_marker, s=36, linewidth=1.1,
                 label=wolfe_label, zorder=4)
    axis.scatter(wdx, wdy, color=wolfe_color, marker="x", s=17,
                 linewidth=.7, alpha=.22, zorder=2)

    pinar_label, pinar_color, pinar_marker = PINAR
    pinar_offset = 0.14
    px, py, pdx, pdy = [], [], [], []
    for yi, row in zip(y, rows):
        value = pinar.get(row["model"])
        if not value or value.get("elapsed_ms") is None:
            continue
        target = (px, py) if value["status"] == "CERTIFIED" else (pdx, pdy)
        target[0].append(value["elapsed_ms"])
        target[1].append(yi + pinar_offset)
    axis.scatter(px, py, color=pinar_color, edgecolors="white",
                 marker=pinar_marker, s=78, linewidth=.55,
                 label=pinar_label, zorder=7)
    axis.scatter(pdx, pdy, color=pinar_color, marker="x", s=19,
                 linewidth=.75, alpha=.32, zorder=3)

    frontier_label, frontier_color, frontier_marker = FRONTIER
    frontier_x = [row["mangasarian_frontier"]["milliseconds"] for row in rows]
    axis.scatter(frontier_x, y, facecolors="white",
                 edgecolors=frontier_color, marker=frontier_marker, s=40,
                 linewidth=1.2, label=frontier_label, zorder=6)

    axis.set_xscale("log")
    axis.set_yticks(y, [row["model"] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("complete solve time")
    axis.xaxis.set_major_formatter(FuncFormatter(time_label))
    axis.grid(axis="x", color="#dddddd", linewidth=.55)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)

    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="x",
                          markersize=4, markeredgewidth=.65,
                          color=pinar_color, alpha=.28))
    labels.append("Pinar non-solve exit")
    fig.suptitle("Netlib-27 program-level solver timing",
                 x=.105, y=.988, ha="left", fontsize=15,
                 fontweight="bold")
    fig.text(.105, .953,
             "Complete solve time; the provisional Pinar 1997 reconstruction is shown separately and excluded from the frontier.",
             color="#454545", fontsize=9.4)
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper left",
               bbox_to_anchor=(.10, .925), handlelength=2.0)
    n_cert = sum(1 for r in pinar_payload["models"]
                 if r.get("status") == "CERTIFIED")
    n_total = len(pinar_payload["models"])
    fig.text(.105, .018,
             f"Pinar: {n_cert}/{n_total} certified; 10 s cap; grow7 not measured. "
             "Python/SciPy cold SVD rebuilds, not the original maintained AAFAC "
             "factors. Faint x = exit, not a solve.",
             color="#666666", fontsize=8.1)
    fig.subplots_adjust(left=.105, right=.985, top=.79, bottom=.10)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    fig.savefig(png_path, dpi=180,
                bbox_inches="tight", metadata={"Software": "Matplotlib"})
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight",
                metadata={"Date": None})
    plt.close(fig)

    # The chart uses a small color vocabulary. A full-resolution indexed PNG
    # preserves its visible content while keeping the public artifact compact.
    with Image.open(png_path) as source:
        compact = source.convert("RGB").quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        compact.save(png_path, optimize=True)


if __name__ == "__main__":
    main()
