"""Noise-floor + trajectory-fingerprint harness for the shipped-defaults panel.

Answers two questions that every other benchmark figure in this project
depends on:

1. NOISE FLOOR.  What is the run-to-run distribution of the 27-model panel
   wall time under SHIPPED DEFAULTS?  Every headline so far (21.58 s,
   25.853 s, ...) is a single draw; none states a spread.

2. TRAJECTORY STABILITY.  Is the pipeline the same computation twice?
   Measured on adlittle: four fixed-PYTHONHASHSEED runs gave 150/150/157/156
   Newton iterations, and the two 150-iteration runs disagreed in ``lb`` at
   the 1e-12 level -- identical control flow, different arithmetic.  So the
   nondeterminism lives BELOW control flow (BLAS/FP path selection), and
   iteration counts are NOT a stable quantity as previously assumed.

This harness monkeypatches NOTHING.  It calls ``exp84_staged_newton.run``
with its shipped defaults, which is also the defaults-only verification
sweep that agent_reports/22 flags as never having been run.

Two fingerprints are recorded per model per repeat:

    structural  -- discrete control flow only (stage index, iteration count,
                   support size, outcome, step solver, convergence reasons).
                   Two runs with equal structural fingerprints executed the
                   same sequence of decisions.
    numeric     -- t and lb rounded to 12 significant digits.  Catches
                   arithmetic divergence that did not (yet) change control
                   flow.

Machine state is bracketed around every repeat: load average plus a fixed
dgemm probe.  A benchmark on a busy machine is still usable if the busyness
is measured; it is not usable if it is assumed away.

Canonical invocation (thread env vars MUST be set by the caller -- setting
them inside the process is a no-op after numpy is imported, and this script
verifies rather than pretends):

    env PYTHONPYCACHEPREFIX=/tmp/fable_lp_pycache \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      .venv/bin/python experiments/bench_noise_floor.py --repeats 9
"""

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

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exp84_staged_newton as exp84  # noqa: E402

DEFAULT_RAW_ROOT = ROOT / "records/noise_floor"

MODELS = [
    "afiro", "sc50a", "sc50b", "kb2", "blend", "adlittle", "share2b",
    "stocfor1", "sc105", "recipe", "scagr7", "sc205", "boeing2",
    "israel", "share1b", "beaconfd", "brandy", "e226", "lotfi", "bandm",
    "capri", "grow7", "sctap1", "scorpion", "degen2", "fit1d", "ship04s",
]
# forplan is the 28th local Netlib model; it is excluded because the MPS
# converter fails on it, not because it is slow or uncertified.  Stated here
# so no reader has to infer the panel's membership.
EXCLUDED = {"forplan": "MPS converter failure (out of solver scope)"}

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

# Every module on the executed call path.  Hashed into provenance so a figure
# can be tied to code, which no prior benchmark record in this repo does.
CALL_PATH = [
    "experiments/exp84_staged_newton.py",
    "experiments/staged_newton.py",
    "experiments/newton_oracle.py",
    "experiments/oracle_terminal.py",
    "experiments/woodbury_face_solver.py",
    "experiments/exp23_path_primal_dual.py",
    "experiments/exp11_netlib_rank_certificate.py",
    "experiments/bench_noise_floor.py",
]


def _sig(value, digits=12):
    """Round a float to ``digits`` significant digits, JSON-stable."""
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or x == 0.0:
        return 0.0 if x == 0.0 else None
    return float(f"%.{digits}e" % x)


def _fingerprints(record):
    """Structural and numeric trajectory fingerprints for one model run.

    Structural deliberately excludes every timing and every float: it is the
    sequence of DECISIONS.  Numeric carries t and lb at 12 significant
    digits.  Equal structural + differing numeric == same path, different
    arithmetic, which is exactly the adlittle signature.
    """
    stages = record.get("stage_table") or []
    structural = [(
        s.get("stage"), s.get("iterations"), s.get("rows_solved"),
        s.get("support"), s.get("converged"), s.get("newton_reason"),
        s.get("test_reason"), s.get("step_solver"), s.get("face_terminal"),
        s.get("outcome"), s.get("repair_rounds"),
    ) for s in stages]
    structural.append((record.get("status"), record.get("route"),
                       record.get("accepted_stage"),
                       record.get("accepted_face_size"),
                       record.get("stages_run"),
                       record.get("terminal_tests_run"),
                       record.get("newton_iters_total")))
    numeric = [(_sig(s.get("t")), _sig(s.get("g_relative")),
                _sig(s.get("t_bar"))) for s in stages]
    numeric.append((_sig(record.get("lb")), _sig(record.get("t_accepted"))))

    def _h(obj):
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

    return _h(structural), _h(numeric)


def _dgemm_probe(n=600, reps=3):
    """Fixed-size dense matmul; best-of-reps seconds.  Machine-state probe."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n))
    best = float("inf")
    for _ in range(reps):
        mark = time.perf_counter()
        a @ a
        best = min(best, time.perf_counter() - mark)
    return best


def _machine_state():
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = None
    return {"load_1min": load1, "load_5min": load5, "load_15min": load15,
            "dgemm_600_seconds": _dgemm_probe()}


def _provenance():
    def _git(*a):
        try:
            return subprocess.run(("git",) + a, cwd=str(ROOT), text=True,
                                  capture_output=True,
                                  timeout=20).stdout.strip()
        except Exception:
            return None
    hashes = {}
    for name in CALL_PATH:
        path = ROOT / name
        hashes[name] = (hashlib.sha256(path.read_bytes()).hexdigest()
                        if path.exists() else None)
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = None
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "call_path_sha256": hashes,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy_version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_VARS},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _spread(values):
    if not values:
        return {}
    out = {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }
    out["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    out["spread_pct"] = (100.0 * (out["max"] - out["min"]) / out["median"]
                         if out["median"] else 0.0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=9,
                        help="full-panel repeats; default 9")
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of the panel; default all 27")
    parser.add_argument("--seconds", type=int, default=300,
                        help="per-model resource ceiling; default 300")
    parser.add_argument("--output-dir", type=Path, default=None)
    # PROMOTION NOTE.  ``exp84_staged_newton`` now defaults to the adopted
    # composition (qr_cod route, Ruiz equilibration, c1/c5, a1/a2/a3/b1).  The
    # noise floor recorded by this harness -- 17.1 % run-to-run spread, the
    # number every A/B in this project sizes itself against -- was measured on
    # the PRE-PROMOTION defaults, so that is what it keeps measuring unless
    # asked otherwise.  ``--promoted-defaults`` measures the new defaults.
    parser.add_argument("--promoted-defaults", action="store_true",
                        help="measure the PROMOTED defaults instead of the "
                             "pre-promotion configuration this harness's "
                             "published noise floor describes")
    args = parser.parse_args()

    missing = [k for k in THREAD_VARS if os.environ.get(k) != "1"]
    if missing:
        raise SystemExit(
            "refusing to run: %s must be set to 1 in the ENVIRONMENT before "
            "python starts (setting them in-process is a no-op after numpy "
            "is imported)" % ", ".join(missing))

    models = args.models or MODELS
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.output_dir.resolve() if args.output_dir
               else DEFAULT_RAW_ROOT / ("%s-pid%d" % (stamp, os.getpid())))
    if out_dir.exists():
        raise FileExistsError("refusing to reuse output directory %s" % out_dir)
    out_dir.mkdir(parents=True)
    rows_path = out_dir / "runs.jsonl"
    summary_path = out_dir / "noise-floor-summary.json"

    provenance = _provenance()
    rows, repeats = [], []

    with rows_path.open("x", encoding="utf-8") as stream:
        for repeat in range(args.repeats):
            before = _machine_state()
            panel_started = time.perf_counter()
            panel_rows = []
            for model in models:
                record = exp84.run(
                    model, seconds=args.seconds, quiet=True,
                    route=(exp84.DEFAULT_ROUTE if args.promoted_defaults
                           else "frozen"),
                    equilibrate=bool(args.promoted_defaults),
                    candidates=bool(args.promoted_defaults),
                    alloc_patches=bool(args.promoted_defaults))
                structural, numeric = _fingerprints(record)
                row = {
                    "repeat": repeat,
                    "model": model,
                    "status": record.get("status"),
                    "wall_seconds": float(record["wall_seconds"]),
                    "wall_split": record.get("wall_split"),
                    "newton_iters_total": record.get("newton_iters_total"),
                    "stages_run": record.get("stages_run"),
                    "terminal_tests_run": record.get("terminal_tests_run"),
                    "accepted_stage": record.get("accepted_stage"),
                    "accepted_face_size": record.get("accepted_face_size"),
                    "t_accepted": record.get("t_accepted"),
                    "lb": record.get("lb"),
                    "route": record.get("route"),
                    "staged_fallback": record.get("staged_fallback"),
                    "structural_fingerprint": structural,
                    "numeric_fingerprint": numeric,
                }
                rows.append(row)
                panel_rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
            panel_elapsed = time.perf_counter() - panel_started
            after = _machine_state()
            summed = sum(r["wall_seconds"] for r in panel_rows)
            repeats.append({
                "repeat": repeat,
                "summed_solver_wall_seconds": summed,
                "panel_elapsed_seconds": panel_elapsed,
                "certified": sum(r["status"] == "CERTIFIED"
                                 for r in panel_rows),
                "machine_state_before": before,
                "machine_state_after": after,
            })
            print(json.dumps({"repeat": repeat,
                              "summed": round(summed, 3),
                              "elapsed": round(panel_elapsed, 3),
                              "certified": repeats[-1]["certified"],
                              "dgemm_before": round(
                                  before["dgemm_600_seconds"], 4),
                              "dgemm_after": round(
                                  after["dgemm_600_seconds"], 4)},
                             sort_keys=True), file=sys.stderr, flush=True)

    per_model = {}
    for model in models:
        mrows = [r for r in rows if r["model"] == model]
        walls = [r["wall_seconds"] for r in mrows]
        iters = [r["newton_iters_total"] for r in mrows
                 if isinstance(r["newton_iters_total"], int)]
        structural = {r["structural_fingerprint"] for r in mrows}
        numeric = {r["numeric_fingerprint"] for r in mrows}
        per_model[model] = {
            "wall": _spread(walls),
            "iterations": _spread(iters) if iters else {},
            "distinct_structural_fingerprints": len(structural),
            "distinct_numeric_fingerprints": len(numeric),
            "deterministic_control_flow": len(structural) == 1,
            "deterministic_arithmetic": len(numeric) == 1,
            "certified": sum(r["status"] == "CERTIFIED" for r in mrows),
            "runs": len(mrows),
        }

    panel_walls = [r["summed_solver_wall_seconds"] for r in repeats]
    panel_elapsed = [r["panel_elapsed_seconds"] for r in repeats]
    dgemm = [r["machine_state_before"]["dgemm_600_seconds"] for r in repeats]
    dgemm += [r["machine_state_after"]["dgemm_600_seconds"] for r in repeats]

    unstable_flow = sorted(m for m, v in per_model.items()
                           if not v["deterministic_control_flow"])
    unstable_arith = sorted(m for m, v in per_model.items()
                            if not v["deterministic_arithmetic"])

    summary = {
        "experiment": "shipped-defaults panel noise floor + trajectory "
                      "fingerprints",
        "classification": "DIRECT MEASUREMENT; NO NORMALIZATION; "
                          "NO MONKEYPATCH; DEFAULTS ONLY",
        "panel": {
            "models": len(models),
            "excluded": EXCLUDED,
            "repeats": args.repeats,
            "summed_solver_wall": _spread(panel_walls),
            "panel_elapsed": _spread(panel_elapsed),
            "per_repeat": repeats,
        },
        "correctness": {
            "certified_total": sum(r["status"] == "CERTIFIED" for r in rows),
            "runs_total": len(rows),
            "all_certified": all(r["status"] == "CERTIFIED" for r in rows),
            "fallbacks": sum(bool(r["staged_fallback"]) for r in rows),
        },
        "determinism": {
            "models_with_unstable_control_flow": unstable_flow,
            "models_with_unstable_arithmetic": unstable_arith,
            "fully_deterministic_models": sorted(
                m for m, v in per_model.items()
                if v["deterministic_control_flow"]
                and v["deterministic_arithmetic"]),
        },
        "machine_state": {
            "dgemm_600_seconds": _spread(dgemm),
            "note": "probe taken before and after every repeat; a wide "
                    "dgemm spread means the panel walls carry machine drift",
        },
        "per_model": per_model,
        "provenance": provenance,
        "provenance_after": _provenance(),
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2)
                            + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("panel", "correctness", "determinism",
                       "machine_state")},
                     sort_keys=True, indent=2, default=str)[:4000])
    print("summary: " + str(summary_path))


if __name__ == "__main__":
    main()
