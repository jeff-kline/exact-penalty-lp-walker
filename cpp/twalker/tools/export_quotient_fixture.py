#!/usr/bin/env python3
"""Freeze an exact equality-quotient fixture for the native C++ walker.

Python is used only for the one-time exact transform and oracle capture.  The
timed walk is performed by the native executable.
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
from exp300_sparse_equality_quotient import sparse_eliminate  # noqa: E402
import selector_gated_walker as sgw  # noqa: E402
from fixture_io import write_fixture  # noqa: E402


def walker_options():
    return dict(
        t=1.0, tmax=1e8, legacy_repair=False,
        ratio_multiplier="min_norm", dual_ascent_repair=True,
        repair_order="adaptive", drift_guard=True, sparse_b=True,
        sparse_face=True,
    )


def accepted_seed(B, b, d):
    progress = {}
    sgw.follow_selector_gated(B, b, d, maxpiv=0, progress=progress,
                              **walker_options())
    accepted = progress.get("last_accepted_face")
    if accepted is None or accepted.get("stage") != "initialization":
        raise RuntimeError("reduced initialization did not accept a face")
    return np.asarray(accepted["working_set"], dtype=bool)


def export_model(name, outdir, max_faces):
    B0, b0, d0, _ = to_Bxgeb(ROOT / "netlib" / (name + ".mps"))
    Br, br, dr, book, info = sparse_eliminate(B0, b0, d0)
    seed = accepted_seed(Br, br, dr)
    captured = []

    def sink(record):
        if record.get("kind") == "breakpoint" and len(captured) < max_faces:
            captured.append((float(record["t"]),
                             np.asarray(record["W_before"], dtype=bool).copy()))

    stats = {}
    result = sgw.follow_selector_gated(
        Br, br, dr, maxpiv=2000, capture=sink, stats=stats,
        **walker_options())
    candidates = [(1.0, seed), *captured]
    faces, seen = [], set()
    for t, mask in candidates:
        key = np.packbits(mask).tobytes()
        if key in seen:
            continue
        seen.add(key)
        rows = np.flatnonzero(mask).astype(np.uint32)
        faces.append({"t": t, "rows": rows,
                      "truth": piece_y(Br, br, dr, mask)})
        if len(faces) >= max_faces:
            break

    outdir.mkdir(parents=True, exist_ok=True)
    fixture_path = outdir / (name + "_quotient.twfx")
    metadata = {
        "model": name + "_quotient",
        "source_model": name,
        "transform": info,
        "oracle": "experiments.exp23_path_primal_dual.piece_y",
        "walk_status": result.get("status"),
        "walk_pivots": int(result.get("pivots", -1)),
        "face_solve_count": int(stats.get("face_solves", -1)),
    }
    manifest = write_fixture(fixture_path, Br, br, dr, seed, 1.0, faces,
                             metadata)
    book_path = outdir / (name + "_quotient_lift.npz")
    np.savez_compressed(
        book_path, pairs=np.asarray(book["pairs"], dtype=np.int64),
        plus=book["plus"], minus=book["minus"], ineq=book["ineq"],
        E=book["E"], Bi=book["Bi"], x0=book["x0"], Z=book["Z"],
        B0=book["B0"], b0=book["b0"], d0=book["d0"])
    manifest["lift_book"] = book_path.name
    print(json.dumps(manifest, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sctap1")
    parser.add_argument("--faces", type=int, default=60)
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "cpp" / "twalker" /
                        "fixtures_quotient")
    args = parser.parse_args()
    export_model(args.model, args.outdir, args.faces)


if __name__ == "__main__":
    main()
