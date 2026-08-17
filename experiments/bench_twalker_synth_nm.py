#!/usr/bin/env python3
"""Interleaved synthetic comparison across t-walker seed modes and baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TOOLS = ROOT / "cpp" / "twalker" / "tools"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TOOLS))

import bench_synth_nm as prior  # noqa: E402
import exp23_path_primal_dual as exp23  # noqa: E402
import synth_nm  # noqa: E402
from fixture_io import write_fixture  # noqa: E402

BASE_ARMS = ("newton", "simplex", "ipm")
VERIFY = ROOT / "cpp" / "twalker" / "build" / "verify_walker"
OUT_ROOT = ROOT / "records" / "twalker_synth_nm"


def _rotate(names, repeat, index):
    names = list(names)
    shift = (repeat + index) % len(names)
    order = names[shift:] + names[:shift]
    return order[::-1] if repeat % 2 else order


def _certificate(B, b, d, x, y):
    ok, detail = exp23.certificate_pair(B, b, d, x, y)
    components = {k: float(detail[k]) for k in
                  ("primal", "dual", "nonnegative", "gap") if k in detail}
    return bool(ok), components, max(components.values(), default=None)


def _write_twalker_fixture(path, inst):
    n = int(inst["B"].shape[0])
    return write_fixture(
        path, inst["B"], inst["b"], inst["d"],
        np.zeros(n, dtype=bool), 0.0, [],
        {"model": path.stem, "source": "experiments/synth_nm.py",
         "instance_digest": synth_nm.digest(inst)})


def _run_twalker(fixture, seconds, max_pivots, seed_mode):
    arm = "twalker_" + seed_mode
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TWALKER_")}
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env["TWALKER_MAX_PIVOTS"] = str(int(max_pivots))
    env["TWALKER_SEED"] = seed_mode
    started = time.perf_counter()
    try:
        proc = subprocess.run([str(VERIFY), str(fixture)], cwd=str(ROOT),
                              env=env, capture_output=True, text=True,
                              timeout=float(seconds), check=False)
    except subprocess.TimeoutExpired:
        return {"arm": arm, "status": "RESOURCE_LIMIT",
                "seconds": float(seconds), "certified": False,
                "iterations": None, "seed_mode": seed_mode,
                "process_seconds": time.perf_counter()-started}
    process_seconds = time.perf_counter() - started
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not lines:
        return {"arm": arm, "status": "NO_JSON",
                "seconds": process_seconds, "certified": False,
                "iterations": None, "seed_mode": seed_mode,
                "stderr": proc.stderr[-1000:]}
    try:
        record = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        return {"arm": arm, "status": "MALFORMED_JSON",
                "seconds": process_seconds, "certified": False,
                "iterations": None, "seed_mode": seed_mode,
                "json_error": str(error), "stdout_tail": proc.stdout[-1000:],
                "stderr": proc.stderr[-1000:]}
    certified = record.get("status") == "CERTIFIED"
    return {
        "arm": arm, "status": record.get("status"),
        "seconds": float(record["wall_ms"]) / 1000.0,
        "process_seconds": process_seconds, "certified": certified,
        "seed_mode": seed_mode,
        "iterations": record.get("pivots"), "seed_seconds":
            float(record.get("seed_ms") or 0.0) / 1000.0,
        "seed_route": record.get("seed_route"),
        "certificate": record.get("certificate"),
        "certificate_max": max((record.get("certificate") or {}).values(),
                               default=None),
        "t": record.get("t"), "face_solves": record.get("face_solves"),
    }


def _run_newton(B, b, d, seconds):
    row = prior._run_native(B, b, d, seconds)
    row["arm"] = "newton"
    return row


def _run_highs(B, b, d, solver, seconds, iteration_limit=None):
    import highspy

    n, m = B.shape
    build_started = time.perf_counter()
    Bs = sp.csr_matrix(B)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("presolve", "off")
    h.setOptionValue("solver", solver)
    h.setOptionValue("time_limit", float(seconds))
    if solver == "simplex":
        h.setOptionValue("simplex_strategy", 1)  # serial dual simplex
        if iteration_limit is not None:
            h.setOptionValue("simplex_iteration_limit", int(iteration_limit))
    else:
        h.setOptionValue("run_crossover", "on")
        if iteration_limit is not None:
            h.setOptionValue("ipm_iteration_limit", int(iteration_limit))
    inf = highspy.kHighsInf
    h.addVars(m, np.full(m, -inf), np.full(m, inf))
    h.changeColsCost(m, np.arange(m, dtype=np.int32),
                     np.ascontiguousarray(np.asarray(d, dtype=float)))
    h.addRows(n, np.asarray(b, dtype=float), np.full(n, inf), Bs.nnz,
              np.asarray(Bs.indptr, dtype=np.int32),
              np.asarray(Bs.indices, dtype=np.int32),
              np.ascontiguousarray(Bs.data))
    build_seconds = time.perf_counter() - build_started
    started = time.perf_counter()
    h.run()
    seconds_run = time.perf_counter() - started
    info = h.getInfo()
    sol = h.getSolution()
    x = np.asarray(sol.col_value, dtype=float)
    y = np.asarray(sol.row_dual, dtype=float)
    ok, cert, cert_max = _certificate(B, b, d, x, y)
    highs_status = h.modelStatusToString(h.getModelStatus())
    if iteration_limit is not None:
        status = "INITIALIZATION_MEASURED"
    elif ok:
        status = "CERTIFIED"
    elif "time limit" in highs_status.lower():
        status = "RESOURCE_LIMIT"
    else:
        status = "GATE_FAILED"
    return {
        "arm": solver, "status": status,
        "seconds": seconds_run, "build_seconds": build_seconds,
        "certified": ok, "certificate": cert,
        "certificate_max": cert_max,
        "iterations": (int(info.simplex_iteration_count) if solver == "simplex"
                       else int(info.ipm_iteration_count)),
        "crossover_iterations": int(info.crossover_iteration_count),
        "highs_status": highs_status,
    }


def _run_arm(arm, inst, fixture, seconds, walker_pivots):
    if arm.startswith("twalker_") and arm.endswith("_init"):
        seed_mode = arm.removeprefix("twalker_").removesuffix("_init")
        row = _run_twalker(fixture, seconds, 0, seed_mode)
        row["arm"] = arm
        row["status"] = "INITIALIZATION_MEASURED"
        row["certified"] = False
        return row
    if arm in ("simplex_init", "ipm_init"):
        solver = arm.removesuffix("_init")
        row = _run_highs(inst["B"], inst["b"], inst["d"], solver,
                         seconds, iteration_limit=1)
        row["arm"] = arm
        return row
    if arm.startswith("twalker_"):
        return _run_twalker(fixture, seconds, walker_pivots,
                            arm.removeprefix("twalker_"))
    if arm == "newton":
        return _run_newton(inst["B"], inst["b"], inst["d"], seconds)
    return _run_highs(inst["B"], inst["b"], inst["d"], arm, seconds)


def _spread(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {}
    return {"n": len(values), "min": min(values),
            "median": statistics.median(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-list", type=int, nargs="+", default=[50, 200, 400])
    parser.add_argument(
        "--ratios", type=float, nargs="+",
        default=[1, 1.5, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument(
        "--walker-seeds", choices=("newton", "highs", "triangular"), nargs="+",
        default=["newton", "highs"],
        help="t-walker seed modes to benchmark as separate interleaved arms")
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=synth_nm.DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--walker-pivots", type=int, default=2000)
    parser.add_argument(
        "--max-consecutive-twalker-limits", type=int, default=3,
        help="abort after this many consecutive t-walker resource limits; "
             "use 0 to complete a censoring-heavy grid")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="append only missing cells to an existing output directory")
    parser.add_argument("--measure-initialization", action="store_true",
                        help="also time zero-pivot walker and one-iteration HiGHS startup runs")
    args = parser.parse_args()
    walker_seeds = list(dict.fromkeys(args.walker_seeds))
    arms = tuple("twalker_" + seed for seed in walker_seeds) + BASE_ARMS
    if args.measure_initialization:
        arms += tuple("twalker_" + seed + "_init" for seed in walker_seeds)
        arms += ("simplex_init", "ipm_init")

    if not VERIFY.exists():
        raise SystemExit("build cpp/twalker/build/verify_walker first")
    required = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    missing = [key for key in required if os.environ.get(key) != "1"]
    if missing:
        raise SystemExit("thread variables must equal 1: " + ", ".join(missing))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.output_dir.resolve() if args.output_dir else
           OUT_ROOT / (stamp + "-pid%d" % os.getpid()))
    out.mkdir(parents=True, exist_ok=args.resume)
    fixtures = out / "fixtures"
    fixtures.mkdir(exist_ok=args.resume)
    runs_path = out / "runs.jsonl"
    if args.resume and not runs_path.exists():
        raise SystemExit("--resume requires an existing runs.jsonl")

    instances = []
    for m in args.m_list:
        for ratio in args.ratios:
            inst = synth_nm.generate(m, ratio=ratio, density=args.density,
                                     seed=args.seed)
            ok, _, _ = _certificate(inst["B"], inst["b"], inst["d"],
                                    inst["x_star"], inst["y_star"])
            if not ok:
                raise SystemExit("planted certificate failed")
            label = "synth_m%d_r%g" % (m, ratio)
            fixture = fixtures / (label + ".twfx")
            if not (args.resume and fixture.exists()):
                _write_twalker_fixture(fixture, inst)
            instances.append((m, ratio, inst, fixture))

    warm = synth_nm.generate(25, ratio=5, density=args.density, seed=args.seed)
    warm_fixture = fixtures / "warmup.twfx"
    if not (args.resume and warm_fixture.exists()):
        _write_twalker_fixture(warm_fixture, warm)
    warmup = []
    for arm in arms:
        row = _run_arm(arm, warm, warm_fixture, args.seconds,
                       args.walker_pivots)
        warmup.append({"arm": arm, "seconds": row.get("seconds"),
                       "status": row.get("status")})

    rows = ([json.loads(line) for line in runs_path.read_text().splitlines()
             if line.strip()] if args.resume else [])
    completed = {(int(row["repeat"]), int(row["m"]), float(row["ratio"]),
                  str(row["arm"])) for row in rows}
    consecutive_twalker_limits = 0
    with runs_path.open("a" if args.resume else "x", encoding="utf-8") as stream:
        for repeat in range(args.repeats):
            for index, (m, ratio, inst, fixture) in enumerate(instances):
                for slot, arm in enumerate(_rotate(arms, repeat, index)):
                    if (repeat, m, float(ratio), arm) in completed:
                        continue
                    row = _run_arm(arm, inst, fixture, args.seconds,
                                   args.walker_pivots)
                    row.update({"repeat": repeat, "slot": slot, "m": m,
                                "n": int(inst["B"].shape[0]),
                                "ratio": ratio,
                                "nnz": int(inst["census"]["nnz"]),
                                "density": inst["census"]["density_achieved"],
                                "instance_digest": synth_nm.digest(inst)[:16]})
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    print(json.dumps({k: row.get(k) for k in
                                     ("repeat", "m", "ratio", "arm", "status",
                                      "seconds", "iterations")}), flush=True)
                    if arm.startswith("twalker_"):
                        if row["status"] == "RESOURCE_LIMIT":
                            consecutive_twalker_limits += 1
                        else:
                            consecutive_twalker_limits = 0
                        if (args.max_consecutive_twalker_limits > 0 and
                                consecutive_twalker_limits >=
                                args.max_consecutive_twalker_limits):
                            raise SystemExit(
                                "%d consecutive t-walker limits" %
                                args.max_consecutive_twalker_limits)
                    if row.get("status") == "GATE_FAILED":
                        raise SystemExit("original-data certificate failed")

    cells = {}
    for m, ratio, inst, _fixture in instances:
        label = "m%d_r%g" % (m, ratio)
        cells[label] = {"m": m, "n": int(inst["B"].shape[0]),
                        "ratio": ratio, "nnz": int(inst["census"]["nnz"]),
                        "arms": {}}
        for arm in arms:
            sub = [row for row in rows if row["m"] == m
                   and row["ratio"] == ratio and row["arm"] == arm]
            sub.sort(key=lambda row: int(row["repeat"]))
            cells[label]["arms"][arm] = {
                "seconds": _spread([row.get("seconds") for row in sub]),
                "iterations": _spread([row.get("iterations") for row in sub]),
                "seed_seconds": _spread([row.get("seed_seconds") for row in sub]),
                "statuses": sorted({str(row.get("status")) for row in sub}),
                "certified": sum(bool(row.get("certified")) for row in sub),
                "runs": len(sub),
                "certificate_max": _spread(
                    [row.get("certificate_max") for row in sub]),
                "seed_routes": sorted({str(row.get("seed_route")) for row in sub
                                       if row.get("seed_route")}),
                "solve_seconds": _spread(
                    [row.get("solve_seconds") for row in sub]),
                "run_samples": [
                    {"repeat": int(row["repeat"]),
                     "seconds": row.get("seconds"),
                     "solve_seconds": row.get("solve_seconds"),
                     "status": row.get("status"),
                     "certified": bool(row.get("certified"))}
                    for row in sub
                ],
            }

    import highspy
    import scipy
    verify_sha256 = hashlib.sha256(VERIFY.read_bytes()).hexdigest()
    solver_source_paths = [ROOT / "cpp" / "twalker" / "Makefile"]
    for source_dir in ("include", "src", "revised"):
        solver_source_paths.extend(sorted(
            path for path in (ROOT / "cpp" / "twalker" / source_dir).glob("*")
            if path.is_file() and path.suffix in (".cpp", ".hpp", ".h")))
    solver_source_sha256 = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in solver_source_paths
    }
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=True,
            capture_output=True, text=True).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain=v1"], cwd=str(ROOT), check=True,
            capture_output=True, text=True).stdout.splitlines()
        git_diff = subprocess.run(
            ["git", "diff", "--binary", "--", "cpp/twalker", "experiments"],
            cwd=str(ROOT), check=True, capture_output=True).stdout
        git_diff_sha256 = hashlib.sha256(git_diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
        git_status = []
        git_diff_sha256 = None

    selected_rows = [
        row for row in rows
        if int(row["m"]) in set(args.m_list)
        and float(row["ratio"]) in set(args.ratios)
        and row["arm"] in set(arms)
    ]
    summary = {
        "experiment": "t-walker seed modes vs shipped Newton and HiGHS",
        "parameters": vars(args) | {"output_dir": str(out)},
        "timed_regions": {
            "twalker_newton": "internal C++ wall; includes native Newton t=0 seed",
            "twalker_highs": "internal C++ wall; includes HiGHS projection t=0 seed and fixed-t repair",
            "twalker_triangular": "internal C++ wall; includes triangular Wolfe projection t=0 seed",
            "newton": "shipped wall; includes equilibration",
            "simplex_ipm": "HiGHS run only; presolve off; build excluded",
        },
        "correctness": "original-data componentwise certificate",
        "reproduction": {
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "highspy": highspy.Highs().version(),
            "verify_walker_sha256": verify_sha256,
            "solver_source_sha256": solver_source_sha256,
            "git_head": git_head,
            "git_status_porcelain": git_status,
            "tracked_solver_experiment_diff_sha256": git_diff_sha256,
        },
        "warmup_discarded": warmup, "cells": cells,
        "runs": len(selected_rows),
        "raw_runs_including_out_of_grid_history": len(rows),
        "all_certified": all(bool(row.get("certified")) for row in selected_rows
                             if not row["arm"].endswith("_init")),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(
            (HERE / "synth_nm.py").read_bytes()).hexdigest(),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True,
                                       default=str) + "\n")
    print(json.dumps({"summary": str(summary_path), "runs": len(rows),
                      "all_certified": summary["all_certified"]}))


if __name__ == "__main__":
    main()
