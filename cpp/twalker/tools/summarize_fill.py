#!/usr/bin/env python3
"""Summarize direct-C++ SPQR fill and timing rows into Phase 1 evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fixture_io import read_fixture  # noqa: E402


def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def finite_median(rows, field):
    values = [float(row[field]) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return statistics.median(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    result = {"source": str(args.csv), "fill": [], "transitions": [],
              "row_order": []}
    models = sorted({row["model"] for row in rows})
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        for ordering in ("natural", "colamd", "amd", "metis", "default",
                         "best", "fixed_colamd", "fixed_amd",
                         "fixed_default"):
            selected = [row for row in model_rows
                        if row["ordering"] == ordering
                        and row["row_order"] == "original"]
            tri = finite_median(selected, "tri_pair_us")
            total_refactor = (median(selected, "assemble_us")
                              + median(selected, "factor_us"))
            result["fill"].append({
                "model": model,
                "ordering": ordering,
                "faces": len(selected),
                "median_rank": median(selected, "rank"),
                "median_nnz_R": median(selected, "nnz_R"),
                "median_density": median(selected, "density"),
                "median_assemble_us": median(selected, "assemble_us"),
                "median_factor_us": median(selected, "factor_us"),
                "triangular_valid_faces": sum(
                    math.isfinite(float(row["tri_pair_us"]))
                    for row in selected),
                "median_tri_pair_us": tri,
                "refactor_over_tri": (total_refactor / tri
                                       if tri is not None else None),
            })
        for row_order in ("original", "degree", "span"):
            selected = [row for row in model_rows
                        if row["ordering"] == "default"
                        and row["row_order"] == row_order]
            result["row_order"].append({
                "model": model,
                "row_order": row_order,
                "median_nnz_R": median(selected, "nnz_R"),
                "median_factor_us": median(selected, "factor_us"),
            })

    by_path = {path.stem: read_fixture(path) for path in args.fixtures}
    for model in models:
        fixture = by_path[model]
        fixed = {
            int(row["face"]): row for row in rows
            if row["model"] == model and row["ordering"] == "fixed_colamd"
            and row["row_order"] == "original"
        }
        changes = []
        for index in range(1, min(len(fixture["faces"]), len(fixed))):
            before = fixture["faces"][index - 1]["rows"]
            after = fixture["faces"][index]["rows"]
            difference = np.setxor1d(before, after, assume_unique=True)
            if len(difference) != 1:
                continue
            old = int(fixed[index - 1]["nnz_R"])
            new = int(fixed[index]["nnz_R"])
            changes.append({"face": index, "direction": (
                "insert" if len(after) > len(before) else "delete"),
                            "delta_nnz": new - old,
                            "ratio": new / old})
        result["transitions"].append({
            "model": model,
            "ordering": "fixed_colamd",
            "single_row_transitions": len(changes),
            "median_delta_nnz": (statistics.median(
                item["delta_nnz"] for item in changes) if changes else None),
            "max_growth_nnz": (max(
                item["delta_nnz"] for item in changes) if changes else None),
            "median_ratio": (statistics.median(
                item["ratio"] for item in changes) if changes else None),
            "details": changes,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out),
                      "fill_rows": len(result["fill"]),
                      "transition_models": len(result["transitions"])},
                     sort_keys=True))


if __name__ == "__main__":
    main()
