#!/usr/bin/env python3
"""Export post-seed t-walker state and SVD-oracle face solutions.

This file deliberately lives outside ``experiments/``: that tree is the
read-only oracle for the rewrite.  Run it with the repository Python
environment named in the handoff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(HERE))

from exp11_netlib_rank_certificate import to_Bxgeb  # noqa: E402
from exp23_path_primal_dual import piece_y  # noqa: E402
from dualize_panel import dualize  # noqa: E402
import selector_gated_walker as sgw  # noqa: E402
from fixture_io import write_fixture  # noqa: E402


DEFAULT_MODELS = ("sctap1", "brandy", "scorpion", "israel", "boeing2",
                  "scagr7")


def walker_options():
    return dict(
        t=1.0,
        tmax=1e8,
        legacy_repair=False,
        ratio_multiplier="min_norm",
        dual_ascent_repair=True,
        repair_order="adaptive",
        drift_guard=True,
        sparse_b=True,
        sparse_face=True,
    )


def initial_face(B, b, d):
    progress = {}
    sgw.follow_selector_gated(B, b, d, maxpiv=0, progress=progress,
                              **walker_options())
    accepted = progress.get("last_accepted_face")
    if accepted is None or accepted.get("stage") != "initialization":
        raise RuntimeError("initialization did not produce an accepted face")
    return np.asarray(accepted["working_set"], dtype=bool)


def capture_model(name, max_faces, outdir, initial_only=False,
                  dualized=False):
    B, b, d, _ = to_Bxgeb(ROOT / "netlib" / (name + ".mps"))
    original_shape = tuple(np.shape(B))
    if dualized:
        B, b, d = dualize(B, b, d)
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    seed = initial_face(B, b, d)

    captured = []

    def sink(record):
        if record.get("kind") == "breakpoint" and len(captured) < max_faces:
            captured.append({
                "t": float(record["t"]),
                "mask": np.asarray(record["W_before"], dtype=bool).copy(),
            })

    stats = {}
    if initial_only:
        result = {"status": "initial-only", "pivots": 0}
    else:
        result = sgw.follow_selector_gated(
            B, b, d, maxpiv=2000, capture=sink, stats=stats,
            **walker_options())

    # The accepted post-seed face is itself a real face and is essential for
    # the post-seed benchmark.  Keep masks unique without imposing trajectory
    # equality as a correctness requirement.
    candidates = [{"t": 1.0, "mask": seed}, *captured]
    faces = []
    seen = set()
    for item in candidates:
        key = np.packbits(item["mask"]).tobytes()
        if key in seen:
            continue
        seen.add(key)
        rows = np.flatnonzero(item["mask"]).astype(np.uint32)
        truth = piece_y(B, b, d, item["mask"])
        faces.append({"t": item["t"], "rows": rows, "truth": truth})
        if len(faces) >= max_faces:
            break

    metadata = {
        "model": name,
        "formulation": "syntactic-dual" if dualized else "primal",
        "original_shape": [int(v) for v in original_shape],
        "oracle": "experiments.exp23_path_primal_dual.piece_y",
        "oracle_cutoff": "s0 * max(shape) * eps",
        "walk_status": result.get("status"),
        "walk_pivots": int(result.get("pivots", -1)),
        "captured_breakpoints": len(captured),
        "face_solve_count": int(stats.get("face_solves", -1)),
    }
    suffix = "_dual.twfx" if dualized else ".twfx"
    return write_fixture(outdir / (name + suffix), B, b, d, seed, 1.0,
                         faces, metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--faces", type=int, default=60)
    parser.add_argument(
        "--initial-only", action="store_true",
        help="export only the accepted post-seed face for a cheap panel run")
    parser.add_argument(
        "--dualized", action="store_true",
        help="export the syntactic dual as a primal-form walker fixture")
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="record per-model export failures instead of stopping the panel")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "cpp" / "twalker" / "fixtures")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for model in args.models:
        try:
            row = capture_model(model, args.faces, args.outdir,
                                args.initial_only, args.dualized)
            row["export_status"] = "OK"
        except Exception as error:
            if not args.continue_on_error:
                raise
            row = {"model": model, "export_status": "ERROR",
                   "error": f"{type(error).__name__}: {error}"}
        manifest.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
