#!/usr/bin/env python3
"""Summarize complete repeats from an interrupted synthetic timing run.

This script never invokes a solver.  It admits only repeats containing every
(m, ratio, arm) cell, which makes exclusion of a partial trailing repeat
mechanical and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def spread(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {}
    return {"n": len(values), "min": min(values),
            "median": statistics.median(values), "max": max(values)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempted-repeats", type=int, default=3)
    parser.add_argument("--minimum-complete-repeats", type=int, default=2)
    parser.add_argument(
        "--require-every-arm", action="store_true",
        help="Treat a repeat as complete only if every arm ran on every cell. "
             "An arm that fails closed on a cell in EVERY repeat (the "
             "triangular seed does this where the face is rank deficient) "
             "then makes every repeat incomplete and rejects the record.")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.runs.read_text().splitlines()
            if line.strip()]
    cells = sorted({(int(row["m"]), float(row["ratio"])) for row in rows})
    arms = sorted({str(row["arm"]) for row in rows})
    by_repeat = defaultdict(list)
    for row in rows:
        by_repeat[int(row["repeat"])].append(row)

    if args.require_every_arm:
        expected = {(m, ratio, arm) for m, ratio in cells for arm in arms}
    else:
        # An arm may legitimately refuse a cell: the triangular seed fails
        # closed where the face lacks full column rank, and does so on the
        # same cells in every repeat.  Demanding the full cartesian product
        # therefore rejects a record that is in fact complete.  Expect only
        # the triples that actually occur somewhere, so a triple missing from
        # ALL repeats is a property of the method, while one missing from
        # SOME repeats still marks those repeats incomplete.
        expected = {(int(row["m"]), float(row["ratio"]), row["arm"])
                    for row in rows}
        refused = ({(m, ratio, arm) for m, ratio in cells for arm in arms}
                   - expected)
        if refused:
            by_arm = Counter(arm for _, _, arm in refused)
            print(json.dumps({
                "note": "arms absent from every repeat; treated as refusals",
                "refused_cells": len(refused),
                "by_arm": dict(by_arm),
            }), file=sys.stderr)

    complete = []
    incomplete = []
    for repeat, group in sorted(by_repeat.items()):
        observed = Counter((int(row["m"]), float(row["ratio"]), row["arm"])
                           for row in group)
        if set(observed) == expected and all(count == 1 for count in observed.values()):
            complete.append(repeat)
        else:
            incomplete.append(repeat)
    if len(complete) < args.minimum_complete_repeats:
        raise SystemExit("fewer than required complete repeats")
    selected = [row for row in rows if int(row["repeat"]) in complete]

    output_cells = {}
    for m, ratio in cells:
        sample = next(row for row in selected
                      if int(row["m"]) == m and float(row["ratio"]) == ratio)
        label = "m%d_r%g" % (m, ratio)
        output_cells[label] = {
            "m": m, "n": int(sample["n"]), "ratio": ratio,
            "nnz": int(sample["nnz"]), "arms": {},
        }
        for arm in arms:
            sub = [row for row in selected if int(row["m"]) == m
                   and float(row["ratio"]) == ratio and row["arm"] == arm]
            output_cells[label]["arms"][arm] = {
                "seconds": spread(row.get("seconds") for row in sub),
                "iterations": spread(row.get("iterations") for row in sub),
                "seed_seconds": spread(row.get("seed_seconds") for row in sub),
                "statuses": sorted({str(row.get("status")) for row in sub}),
                "certified": sum(bool(row.get("certified")) for row in sub),
                "runs": len(sub),
                "certificate_max": spread(row.get("certificate_max") for row in sub),
                "seed_routes": sorted({str(row.get("seed_route")) for row in sub
                                       if row.get("seed_route")}),
                # The renderer's post-initialization panel pairs runs across
                # arms by repeat index, and reads Newton's core solve time
                # separately from its total.  Emit both; without them the
                # post-initialization figure cannot be rebuilt from a summary.
                "solve_seconds": spread(row.get("solve_seconds")
                                        for row in sub),
                "run_samples": [
                    {"repeat": int(row["repeat"]),
                     "seconds": row.get("seconds"),
                     "solve_seconds": row.get("solve_seconds"),
                     "certified": bool(row.get("certified"))}
                    for row in sorted(sub, key=lambda r: int(r["repeat"]))],
            }

    parameters = {
        "m_list": sorted({m for m, _ in cells}),
        "ratios": sorted({ratio for _, ratio in cells}),
        "density": 0.10,
        "seed": 20260810,
        "repeats": len(complete),
        "attempted_repeats": args.attempted_repeats,
        "selected_complete_repeats": complete,
        "excluded_incomplete_repeats": incomplete,
        "seconds": 30.0,
        "walker_pivots": 2000,
        "walker_seeds": [arm.removeprefix("twalker_") for arm in arms
                         if arm.startswith("twalker_")],
        "threads": 1,
    }
    root = Path(__file__).resolve().parents[1]
    harness = root / "experiments" / "bench_twalker_synth_nm.py"
    generator = root / "experiments" / "synth_nm.py"
    verifier = root / "cpp" / "twalker" / "build" / "verify_walker"
    solver_sources = [root / "cpp" / "twalker" / "Makefile",
                      root / "cpp" / "twalker" / "tests" / "verify_walker.cpp"]
    for source_dir in ("include", "src", "revised"):
        solver_sources.extend(sorted(
            path for path in (root / "cpp" / "twalker" / source_dir).glob("*")
            if path.is_file() and path.suffix in (".cpp", ".hpp", ".h")))
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    import highspy
    import numpy
    import scipy

    summary = {
        "experiment": "Newton- and HiGHS-seeded t-walker vs shipped Newton and HiGHS",
        "parameters": parameters,
        "timed_regions": {
            "twalker_newton": "internal C++ wall; includes native Newton t=0 seed",
            "twalker_highs": "internal C++ wall; includes HiGHS projection t=0 seed and fixed-t repair",
            "newton": "shipped wall; includes equilibration",
            "simplex_ipm": "HiGHS run only; presolve off; model build excluded",
        },
        "correctness": "original-data componentwise certificate",
        "selection_rule": "all and only structurally complete repeats; partial trailing repeats excluded",
        "reproduction": {
            "raw_runs": str(args.runs),
            "raw_runs_sha256": sha256(args.runs),
            "summarizer_sha256": sha256(Path(__file__)),
            "benchmark_harness_sha256": sha256(harness),
            "generator_sha256": sha256(generator),
            "verify_walker_sha256": sha256(verifier),
            "solver_source_sha256": {
                str(path.relative_to(root)): sha256(path)
                for path in solver_sources
            },
            "git_head": git_head,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "highspy": highspy.Highs().version(),
            "original_command": [
                "experiments/bench_twalker_synth_nm.py", "--m-list", "50", "200", "400",
                "--ratios", "1", "1.5", "2", "4", "8", "16", "32", "64", "128",
                "--walker-seeds", "newton", "highs", "--density", "0.10",
                "--seed", "20260810", "--repeats", "2",
                "--seconds", "30", "--walker-pivots", "2000",
                "--max-consecutive-twalker-limits", "0",
            ],
            "note": "The user stopped the third repeat to avoid redundant timing churn; repeats 0 and 1 are complete and repeat 2 is mechanically excluded.",
        },
        "warmup_discarded": "performed by benchmark harness; interrupted run did not serialize warmup rows",
        "cells": output_cells,
        "runs": len(selected),
        "raw_runs_including_partial": len(rows),
        "all_certified": all(bool(row.get("certified")) for row in selected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "complete_repeats": complete,
                      "excluded_repeats": incomplete, "runs": len(selected)}))


if __name__ == "__main__":
    main()
