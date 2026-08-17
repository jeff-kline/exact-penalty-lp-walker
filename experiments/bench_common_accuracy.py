#!/usr/bin/env python3
"""Build an apples-to-apples accuracy panel for Netlib-27 and synthetic LPs.

All reported values are the four components of ``certificate_pair`` on the
original inequality-form program

    min d'x  subject to Bx >= b,
    max b'y  subject to B'y = d, y >= 0.

The native C++ walker records the algebraically identical four-component
certificate.  Cached walker/Newton records are reused; HiGHS is rerun because
the historical Netlib timing artifact used a different residual scaling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exp23_path_primal_dual as exp23  # noqa: E402
import bench_transforms as bt  # noqa: E402
import synth_nm  # noqa: E402
from exp11_netlib_rank_certificate import to_Bxgeb  # noqa: E402


NETLIB_BASE = (ROOT / "records/twalker_cpp/"
               "native_seed_t0_netlib27_full_budgeted_20260816.json")
NETLIB_FIT1D = (ROOT / "records/twalker_cpp/"
                "native_seed_t0_fit1d_long90_20260816.json")
NETLIB_LOTFI = (ROOT / "records/twalker_cpp/"
                "native_seed_t0_fit1d_lotfi_long_20260816.json")
NETLIB_WOLFE = ROOT / "records/wolfe_dual_twalker_netlib27_161.json"
NETLIB_NEWTON = (ROOT / "records/native_presolve/"
                 "20260809T182139Z-pid53229/runs.jsonl")
SYNTHETIC_RUNS = (ROOT / "records/twalker_synth_nm/"
                  "postinit_m25_50_200_r32_20260816/runs.jsonl")
DEFAULT_OUT = ROOT / "records/common_accuracy_20260817"
VERIFY = ROOT / "cpp/twalker/build/verify_walker"
SYNTH_FIXTURES = (ROOT / "records/twalker_synth_nm/"
                  "postinit_m25_50_200_r32_20260816/fixtures")

COMPONENTS = ("primal", "dual", "nonnegative", "gap")
SYNTH_ARMS = ("twalker_newton", "twalker_crossover",
              "twalker_triangular", "newton", "newton_crossover",
              "simplex", "ipm", "ipm_no_crossover")
SYNTH_CACHED_ARMS = ("twalker_newton", "twalker_triangular", "newton",
                     "simplex", "ipm")
NETLIB_ARMS = ("twalker", "twalker_crossover", "twalker_triangular",
               "newton", "newton_crossover", "simplex", "ipm",
               "ipm_no_crossover")


def _json_lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_certificate(detail):
    values = {key: float(detail[key]) for key in COMPONENTS
              if detail.get(key) is not None}
    if len(values) != len(COMPONENTS):
        return None
    values["max"] = max(values.values())
    values["worst_component"] = max(COMPONENTS, key=values.get)
    return values


def _walker_certificate(row):
    return _clean_certificate((row.get("walker") or {}).get("certificate") or {})


def _load_cached_netlib():
    base = {row["model"]: row for row in json.loads(NETLIB_BASE.read_text())}
    base["fit1d"] = json.loads(NETLIB_FIT1D.read_text())[0]
    base["lotfi"] = next(
        row for row in json.loads(NETLIB_LOTFI.read_text())
        if row["model"] == "lotfi")
    wolfe = {row["model"]: row
             for row in json.loads(NETLIB_WOLFE.read_text())}
    newton_rows = [row for row in _json_lines(NETLIB_NEWTON)
                   if row.get("cell") == "equil" and row.get("repeat") == 0]
    newton = {row["model"]: row for row in newton_rows}
    if set(base) != set(wolfe) or set(base) != set(newton):
        raise RuntimeError("Netlib model sets do not agree")
    rows = []
    for model in sorted(base):
        brow, wrow, nrow = base[model], wolfe[model], newton[model]
        rows.extend([
            {"suite": "netlib27", "problem": model, "arm": "twalker",
             "status": brow.get("status"),
             "certificate": _walker_certificate(brow),
             "source": str(NETLIB_BASE.relative_to(ROOT))},
            {"suite": "netlib27", "problem": model,
             "arm": "twalker_triangular", "status": wrow.get("status"),
             "certificate": _walker_certificate(wrow),
             "source": str(NETLIB_WOLFE.relative_to(ROOT))},
            {"suite": "netlib27", "problem": model, "arm": "newton",
             "status": nrow.get("status"),
             "certificate": _clean_certificate(nrow.get("certificate") or {}),
             "source": str(NETLIB_NEWTON.relative_to(ROOT))},
        ])
    return rows, sorted(base)


def _solve_highs(B, b, d, arm, seconds):
    import highspy

    B = sp.csr_matrix(B)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    n, m = B.shape
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("presolve", "off")
    solver = "simplex" if arm == "simplex" else "ipm"
    h.setOptionValue("solver", solver)
    h.setOptionValue("time_limit", float(seconds))
    if solver == "simplex":
        h.setOptionValue("simplex_strategy", 1)
    else:
        h.setOptionValue("run_crossover",
                         "off" if arm == "ipm_no_crossover" else "on")
    inf = highspy.kHighsInf
    h.addVars(m, np.full(m, -inf), np.full(m, inf))
    h.changeColsCost(m, np.arange(m, dtype=np.int32), d)
    h.addRows(n, b, np.full(n, inf), B.nnz,
              np.asarray(B.indptr, dtype=np.int32),
              np.asarray(B.indices, dtype=np.int32),
              np.asarray(B.data, dtype=float))
    h.run()
    solution = h.getSolution()
    x = np.asarray(solution.col_value, dtype=float)
    y = np.asarray(solution.row_dual, dtype=float)
    passed, detail = exp23.certificate_pair(B, b, d, x, y)
    model_status = h.modelStatusToString(h.getModelStatus())
    return {
        "arm": arm, "status": "CERTIFIED" if passed else "GATE_FAILED",
        "certificate": _clean_certificate(detail),
        "highs_model_status": model_status,
        "authority": "exp23.certificate_pair on original converted (B,b,d)",
    }


def _run_highs(model, arm, seconds):
    B, b, d, _ = to_Bxgeb(ROOT / "netlib" / f"{model}.mps")
    row = _solve_highs(B, b, d, arm, seconds)
    row.update({"suite": "netlib27", "problem": model})
    return row


def _crossover(B, b, d, y, seconds):
    """Warm-start simplex from an optimal dual point and recertify.

    This deliberately passes only the point, not a pre-existing basis.  It is
    therefore the portable common crossover applicable to both solver arms;
    a future native walker crossover can additionally pass its retained basis.
    """
    import highspy

    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = B.shape
    matrix = sp.csc_matrix(B.T)
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = -b.copy()
    lp.col_lower_ = np.zeros(n)
    lp.col_upper_ = np.full(n, highspy.kHighsInf)
    lp.row_lower_ = d.copy()
    lp.row_upper_ = d.copy()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = matrix.indptr.astype(np.int32)
    lp.a_matrix_.index_ = matrix.indices.astype(np.int32)
    lp.a_matrix_.value_ = matrix.data.astype(float)

    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("threads", 1)
    solver.setOptionValue("presolve", "off")
    solver.setOptionValue("solver", "simplex")
    solver.setOptionValue("simplex_strategy", 4)
    solver.setOptionValue("time_limit", float(seconds))
    solver.passModel(lp)
    warm_status = solver.setSolution(
        n, np.arange(n, dtype=np.int32), np.ascontiguousarray(y))
    started = time.perf_counter()
    solver.run()
    crossover_seconds = time.perf_counter() - started
    solution = solver.getSolution()
    crossed_y = np.asarray(solution.col_value, dtype=float)
    crossed_x = -np.asarray(solution.row_dual, dtype=float)
    passed, detail = exp23.certificate_pair(B, b, d, crossed_x, crossed_y)
    info = solver.getInfo()
    return {
        "status": "CERTIFIED" if passed else "GATE_FAILED",
        "certificate": _clean_certificate(detail),
        "highs_model_status": solver.modelStatusToString(
            solver.getModelStatus()),
        "crossover_seconds": float(crossover_seconds),
        "crossover_iterations": int(info.simplex_iteration_count),
        "warm_start_status": str(warm_status),
        "support": int(np.count_nonzero(crossed_y > 1e-10)),
        "x": crossed_x,
        "y": crossed_y,
    }


def _run_walker_solution(fixture, seconds, seed="newton"):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("TWALKER_")}
    environment.update({
        "TWALKER_EMIT_SOLUTION": "1",
        "TWALKER_MAX_PIVOTS": "7000",
        "TWALKER_SEED": seed,
    })
    try:
        process = subprocess.run(
            [str(VERIFY), str(fixture)], cwd=str(ROOT), env=environment,
            capture_output=True, text=True, timeout=float(seconds), check=False)
    except subprocess.TimeoutExpired:
        return {"status": "RESOURCE_LIMIT"}
    lines = [line for line in process.stdout.splitlines()
             if line.startswith("{")]
    if not lines:
        return {"status": "NO_JSON"}
    return json.loads(lines[-1])


def _run_newton_solution(model, B, b, d, seconds):
    started = time.perf_counter()
    Bt, bt_, dt, row_scale, col_scale, _ = bt.equilibrate(
        B, b, d, rounds=bt.RUIZ_ROUNDS)
    record, x_transformed, y_transformed = bt._solve_transformed(
        model, Bt, bt_, dt, seconds, 0, 0)
    if x_transformed is None or y_transformed is None:
        return {"status": record.get("status"), "seconds":
                time.perf_counter() - started}
    return {
        "status": record.get("status"),
        "seconds": time.perf_counter() - started,
        "x": col_scale * np.asarray(x_transformed, dtype=float),
        "y": row_scale * np.asarray(y_transformed, dtype=float),
    }


def _compact_crossover(row, suite, problem, arm, source, parent):
    return {
        "suite": suite, "problem": problem, "arm": arm,
        "status": row.get("status"),
        "certificate": row.get("certificate"),
        "source": source, "parent": parent,
        "crossover_seconds": row.get("crossover_seconds"),
        "crossover_iterations": row.get("crossover_iterations"),
        "warm_start_status": row.get("warm_start_status"),
        "support": row.get("support"),
        "authority": "exp23.certificate_pair on original (B,b,d)",
    }


def _load_or_run_crossovers(models, output, seconds, refresh):
    path = output / "solver_crossovers_common_certificate.json"
    if path.exists() and not refresh:
        rows = json.loads(path.read_text())
        if (len(rows) == 2 * (len(models) + 24)
                and {row["arm"] for row in rows}
                == {"twalker_crossover", "newton_crossover"}):
            return rows

    rows = []
    cached_netlib, _ = _load_cached_netlib()
    status = {(row["problem"], row["arm"]): row["status"]
              for row in cached_netlib}
    cached_synthetic = _json_lines(SYNTHETIC_RUNS)
    synthetic_status = {}
    for row in cached_synthetic:
        if (row.get("arm") == "twalker_newton"
                and int(row.get("m", -1)) in (25, 50, 200)
                and float(row.get("ratio", 0.0))
                in (1.1, 1.25, 1.5, 2, 4, 8, 16, 32)):
            key = (int(row["m"]), float(row["ratio"]))
            synthetic_status.setdefault(key, []).append(row.get("status"))
    for model in models:
        B, b, d, _ = to_Bxgeb(ROOT / "netlib" / f"{model}.mps")
        B = np.asarray(B, dtype=float)
        b = np.asarray(b, dtype=float)
        d = np.asarray(d, dtype=float)
        if status[(model, "twalker")] == "CERTIFIED":
            raw = _run_walker_solution(
                ROOT / "cpp/twalker/fixtures_panel" / f"{model}.twfx",
                seconds)
            if raw.get("status") == "CERTIFIED" and raw.get("solution"):
                crossed = _crossover(B, b, d, raw["solution"]["y"], seconds)
            else:
                crossed = {"status": raw.get("status")}
        else:
            crossed = {"status": status[(model, "twalker")]}
        rows.append(_compact_crossover(
            crossed, "netlib27", model, "twalker_crossover",
            str(path.relative_to(ROOT)), "twalker"))

        raw = _run_newton_solution(model, B, b, d, seconds)
        if raw.get("status") == "CERTIFIED" and raw.get("y") is not None:
            crossed = _crossover(B, b, d, raw["y"], seconds)
        else:
            crossed = {"status": raw.get("status")}
        rows.append(_compact_crossover(
            crossed, "netlib27", model, "newton_crossover",
            str(path.relative_to(ROOT)), "newton"))
        print(json.dumps({"suite": "netlib27", "problem": model,
                          "crossovers": [rows[-2]["status"],
                                         rows[-1]["status"]]}), flush=True)

    for m in (25, 50, 200):
        for ratio in (1.1, 1.25, 1.5, 2, 4, 8, 16, 32):
            problem = f"m{m}_r{ratio:g}"
            instance = synth_nm.generate(m, ratio=ratio, density=.10,
                                         seed=20260810)
            B = np.asarray(instance["B"], dtype=float)
            b = np.asarray(instance["b"], dtype=float)
            d = np.asarray(instance["d"], dtype=float)
            parent_statuses = synthetic_status.get((m, float(ratio)), [])
            if parent_statuses and all(value == "CERTIFIED"
                                       for value in parent_statuses):
                raw = _run_walker_solution(
                    SYNTH_FIXTURES / f"synth_m{m}_r{ratio:g}.twfx", seconds)
            else:
                raw = {"status": (parent_statuses[0] if parent_statuses
                                   else "MISSING")}
            if raw.get("status") == "CERTIFIED" and raw.get("solution"):
                crossed = _crossover(B, b, d, raw["solution"]["y"], seconds)
            else:
                crossed = {"status": raw.get("status")}
            row = _compact_crossover(
                crossed, "synthetic24", problem, "twalker_crossover",
                str(path.relative_to(ROOT)), "twalker_newton")
            row.update({"m": m, "ratio": ratio, "runs": 1,
                        "certified_runs": int(row["status"] == "CERTIFIED"),
                        "aggregation": "one deterministic crossover"})
            rows.append(row)

            raw = _run_newton_solution(problem, B, b, d, seconds)
            if raw.get("status") == "CERTIFIED" and raw.get("y") is not None:
                crossed = _crossover(B, b, d, raw["y"], seconds)
            else:
                crossed = {"status": raw.get("status")}
            row = _compact_crossover(
                crossed, "synthetic24", problem, "newton_crossover",
                str(path.relative_to(ROOT)), "newton")
            row.update({"m": m, "ratio": ratio, "runs": 1,
                        "certified_runs": int(row["status"] == "CERTIFIED"),
                        "aggregation": "one deterministic crossover"})
            rows.append(row)
            print(json.dumps({"suite": "synthetic24", "problem": problem,
                              "crossovers": [rows[-2]["status"],
                                             rows[-1]["status"]]}), flush=True)

    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


def _load_or_run_highs(models, output, seconds, refresh):
    path = output / "netlib_highs_common_certificate.json"
    if path.exists() and not refresh:
        rows = json.loads(path.read_text())
        if ({row["problem"] for row in rows} == set(models)
                and {row["arm"] for row in rows}
                == {"simplex", "ipm", "ipm_no_crossover"}):
            return rows
    rows = []
    for model in models:
        for arm in ("simplex", "ipm", "ipm_no_crossover"):
            row = _run_highs(model, arm, seconds)
            rows.append(row)
            print(json.dumps({"problem": model, "arm": arm,
                              "status": row["status"],
                              "max": ((row.get("certificate") or {}).get("max"))}),
                  flush=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


def _load_or_run_synthetic_ipm_no_crossover(output, seconds, refresh):
    path = output / "synthetic_ipm_no_crossover_common_certificate.json"
    if path.exists() and not refresh:
        rows = json.loads(path.read_text())
        if len(rows) == 24:
            return rows
    rows = []
    for m in (25, 50, 200):
        for ratio in (1.1, 1.25, 1.5, 2, 4, 8, 16, 32):
            instance = synth_nm.generate(m, ratio=ratio, density=.10,
                                         seed=20260810)
            row = _solve_highs(instance["B"], instance["b"], instance["d"],
                               "ipm_no_crossover", seconds)
            row.update({
                "suite": "synthetic24", "problem": f"m{m}_r{ratio:g}",
                "m": m, "ratio": ratio, "certified_runs":
                    1 if row["status"] == "CERTIFIED" else 0,
                "runs": 1, "aggregation": "one deterministic solve",
                "source": str(path.relative_to(ROOT)),
            })
            rows.append(row)
            print(json.dumps({"problem": row["problem"],
                              "arm": row["arm"], "status": row["status"],
                              "max": ((row.get("certificate") or {}).get("max"))}),
                  flush=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


def _load_synthetic(no_crossover_rows):
    raw = [row for row in _json_lines(SYNTHETIC_RUNS)
           if int(row.get("m", -1)) in (25, 50, 200)
           and float(row.get("ratio", 0.0)) in (1.1, 1.25, 1.5, 2, 4, 8, 16, 32)
           and row.get("arm") in SYNTH_CACHED_ARMS]
    cells = {}
    for row in raw:
        key = (int(row["m"]), float(row["ratio"]), row["arm"])
        cells.setdefault(key, []).append(row)
    output = []
    for (m, ratio, arm), rows in sorted(cells.items()):
        certified = [row for row in rows if row.get("status") == "CERTIFIED"]
        certificate = None
        if certified:
            components = {}
            for component in COMPONENTS:
                values = [float(row["certificate"][component])
                          for row in certified]
                components[component] = statistics.median(values)
            certificate = _clean_certificate(components)
        output.append({
            "suite": "synthetic24", "problem": f"m{m}_r{ratio:g}",
            "m": m, "ratio": ratio, "arm": arm,
            "status": ("CERTIFIED" if len(certified) == len(rows) else
                       "PARTIAL" if certified else
                       (rows[0].get("status") if rows else "MISSING")),
            "certified_runs": len(certified), "runs": len(rows),
            "certificate": certificate,
            "aggregation": "componentwise median over five deterministic repeats",
            "source": str(SYNTHETIC_RUNS.relative_to(ROOT)),
        })
    output.extend(no_crossover_rows)
    expected = 24 * (len(SYNTH_CACHED_ARMS) + 1)
    if len(output) != expected:
        raise RuntimeError(f"expected {expected} synthetic cells, got {len(output)}")
    return output


def _quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = probability * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _summarize(rows, suite, arms):
    result = {}
    suite_rows = [row for row in rows if row["suite"] == suite]
    problem_count = len({row["problem"] for row in suite_rows})
    for arm in arms:
        arm_rows = [row for row in suite_rows if row["arm"] == arm]
        certified = [row for row in arm_rows
                     if row.get("status") == "CERTIFIED"
                     and row.get("certificate") is not None]
        values = [row["certificate"]["max"] for row in certified]
        worst = max(certified, key=lambda row: row["certificate"]["max"])
        component_worst = {
            component: max((row["certificate"][component] for row in certified),
                           default=None)
            for component in COMPONENTS
        }
        result[arm] = {
            "problems": problem_count,
            "certified": len(certified),
            "median_max_error": statistics.median(values) if values else None,
            "p90_max_error": _quantile(values, .90),
            "maximum_error": max(values) if values else None,
            "worst_problem": worst["problem"] if values else None,
            "worst_component": (worst["certificate"]["worst_component"]
                                if values else None),
            "componentwise_maxima": component_worst,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--refresh-highs", action="store_true")
    parser.add_argument("--refresh-crossovers", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise SystemExit(f"{variable}=1 is required")

    netlib_cached, models = _load_cached_netlib()
    highs = _load_or_run_highs(models, args.output_dir, args.seconds,
                               args.refresh_highs)
    synthetic_no_crossover = _load_or_run_synthetic_ipm_no_crossover(
        args.output_dir, args.seconds, args.refresh_highs)
    synthetic = _load_synthetic(synthetic_no_crossover)
    crossovers = _load_or_run_crossovers(
        models, args.output_dir, args.seconds, args.refresh_crossovers)
    rows = netlib_cached + highs + synthetic + crossovers
    rows.sort(key=lambda row: (row["suite"], row["problem"], row["arm"]))
    (args.output_dir / "accuracy_rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "common-original-data-kkt-accuracy-v1",
        "problem": "min d'x subject to Bx>=b; dual max b'y subject to B'y=d,y>=0",
        "score": "max(primal, dual, nonnegative, gap)",
        "components": {
            "primal": "max_i (b_i-(Bx)_i)_+/(1+|b_i|+(|B||x|)_i)",
            "dual": "max_j |(B'y-d)_j|/(1+|d_j|+(|B|'|y|)_j)",
            "nonnegative": "(-min_i y_i)_+/(1+||y||_inf)",
            "gap": "|d'x-b'y|/(1+|d'x|+|b'y|)",
        },
        "tolerance": float(exp23.TOL_FEAS),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "highs": __import__("highspy").Highs().version(),
            "threads": 1,
            "highs_presolve": "off",
            "highs_ipm_crossover": ["on", "off"],
            "solver_solution_crossover": "HiGHS simplex; warm point; presolve off",
        },
        "netlib27": _summarize(rows, "netlib27", NETLIB_ARMS),
        "synthetic24": _summarize(rows, "synthetic24", SYNTH_ARMS),
        "sources": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (NETLIB_BASE, NETLIB_FIT1D, NETLIB_LOTFI,
                         NETLIB_WOLFE, NETLIB_NEWTON, SYNTHETIC_RUNS,
                         args.output_dir / "netlib_highs_common_certificate.json",
                         args.output_dir /
                            "synthetic_ipm_no_crossover_common_certificate.json",
                         args.output_dir /
                            "solver_crossovers_common_certificate.json",
                         Path(__file__))
        },
        "notes": [
            "Every score is evaluated on the original inequality-form data.",
            "The C++ walker certificate is algebraically identical to the Python evaluator.",
            "DNF/resource-limit cells are coverage failures, not zero-error observations.",
            "Synthetic distributions give each of 24 problem cells equal weight.",
            "HiGHS runs use presolve off and one thread; IPM is reported both with and without crossover.",
            "t-walker and Newton are each reported raw and after the same warm-point HiGHS simplex crossover.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
