#!/usr/bin/env python3
"""Run the quarantined Pinar 1997 reference on compact Netlib fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOLS = ROOT / "cpp" / "twalker" / "tools"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TOOLS))

from fixture_io import read_fixture  # noqa: E402
from pinar1997 import Options, solve  # noqa: E402


PAPER_COUNTS = {
    "afiro": (24, 9), "sc50b": (17, 4), "sc50a": (25, 8),
    "sc105": (37, 7), "adlittle": (68, 7), "scagr7": (77, 14),
    "stocfor1": (88, 15), "blend": (63, 13), "sc205": (61, 14),
    "share2b": (99, 24),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["afiro", "sc50b", "sc50a"])
    parser.add_argument("--fixtures", type=Path,
                        default=ROOT / "cpp" / "twalker" / "fixtures_panel")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-outer", type=int, default=200)
    parser.add_argument("--max-newton", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = []
    for model in args.models:
        fixture_path = args.fixtures / f"{model}.twfx"
        if not fixture_path.exists():
            row = {
                "model": model,
                "status": "NOT_MEASURED",
                "detail": f"fixture not found: {fixture_path}",
                "algorithm": "Pinar1997-LPPEN",
                "elapsed_ms": None,
                "certificate": None,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            continue
        fixture = read_fixture(fixture_path)
        callback = None
        if args.verbose:
            callback = lambda value, name=model: print(
                json.dumps({"model": name, **value}), flush=True)
        result = solve(
            model, fixture["B"], fixture["b"], fixture["d"],
            Options(timeout_seconds=args.timeout, max_outer=args.max_outer,
                    max_newton_total=args.max_newton), callback)
        row = result.to_dict()
        if model in PAPER_COUNTS:
            row["pinar1997_table1_iterations"] = PAPER_COUNTS[model][0]
            row["pinar1997_table1_reductions"] = PAPER_COUNTS[model][1]
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "schema": "pinar1997-netlib-panel-v1",
        "implementation": "quarantined Python/SciPy reference",
        "solver_fallbacks": "none; HiGHS and t-walker are not called",
        "models": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
