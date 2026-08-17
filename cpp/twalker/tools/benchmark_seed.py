#!/usr/bin/env python3
"""Bounded linear-algebra seed-crash discriminator for the C++ t-walker.

The two candidate face identifiers are the maintained-QR Lawson--Hanson
penalty crash in ``cpp/nnls`` and the repository's structured semismooth
Newton projection.  Their support is written into a temporary TWFX at a fixed
``t0``; the ordinary C++ walker is then authoritative for fixed-t settle,
forward launch, and the original-data certificate.  Neither candidate makes
an external convex-solver call.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from fixture_io import read_fixture, write_fixture


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CPP_ROOT = ROOT / "cpp"
TWALKER_ROOT = CPP_ROOT / "twalker"
EXPERIMENTS = ROOT / "experiments"
NNLS = CPP_ROOT / "nnls"
VERIFY = TWALKER_ROOT / "build" / "verify_walker"
DEFAULT_MODELS = ("afiro", "blend", "adlittle", "brandy", "capri",
                  "lotfi", "bandm", "fit1d", "ship04s")


def source_fixture(model: str) -> Path:
    normal = TWALKER_ROOT / "fixtures_panel" / f"{model}.twfx"
    if normal.exists():
        return normal
    # Grow7's bounded seed work predates a promoted repository fixture.  Keep
    # this explicit rather than silently substituting another formulation.
    grow7 = Path("/private/tmp/grow7_highs.twfx")
    if model == "grow7" and grow7.exists():
        return grow7
    raise FileNotFoundError(f"no source fixture for {model}")


def write_mwlk(path: Path, fixture: dict) -> None:
    """Write the legacy model-only container consumed by ``cpp/nnls``."""
    matrix = fixture["B"].tocsr(copy=True)
    matrix.sort_indices()
    n, m = matrix.shape
    with path.open("wb") as stream:
        stream.write(b"MWLK")
        stream.write(struct.pack("<4i", 1, n, m, matrix.nnz))
        stream.write(np.asarray(matrix.indptr, dtype="<i4").tobytes())
        stream.write(np.asarray(matrix.indices, dtype="<i4").tobytes())
        stream.write(np.asarray(matrix.data, dtype="<f8").tobytes())
        stream.write(np.asarray(fixture["b"], dtype="<f8").tobytes())
        stream.write(np.asarray(fixture["d"], dtype="<f8").tobytes())
        stream.write(np.zeros(n, dtype=np.uint8).tobytes())
        stream.write(struct.pack("<d", 0.0))


def identify_support_nnls(model: str, t0: float, fixture: dict, timeout: float,
                          temp: Path) -> tuple[np.ndarray | None, dict, str]:
    mwlk = CPP_ROOT / "models" / f"{model}.mwlk"
    if not mwlk.exists():
        mwlk = temp / f"{model}.mwlk"
        write_mwlk(mwlk, fixture)
    mask_path = temp / f"{model}-nnls-t{t0:g}.mask"
    started = time.perf_counter()
    seed = subprocess.run(
        [str(NNLS), str(mwlk), "--t", repr(float(t0)),
         "--mask", str(mask_path)],
        check=False, capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    process_seconds = time.perf_counter() - started
    info = json.loads(seed.stdout.strip().splitlines()[-1])
    info["process_seconds"] = process_seconds
    if seed.returncode != 0 or not info.get("converged") \
            or not mask_path.exists():
        return None, info, seed.stderr.strip()
    mask = np.frombuffer(mask_path.read_bytes(), dtype=np.uint8).astype(bool)
    return mask, info, seed.stderr.strip()


def identify_support_newton(t0: float, fixture: dict,
                            timeout: float) -> tuple[np.ndarray | None, dict, str]:
    # This is a method-selection oracle for the future native port.  The
    # implementation is repository-local linear algebra, not a convex-solver
    # side call.  At t=0 the scaled formulation is undefined, so use the
    # mathematically identical unscaled KKT dual.
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    import newton_oracle as newt

    B = fixture["B"].toarray()
    started = time.perf_counter()
    Bs = newt._as_sparse(B)
    kernel = newt.FaceStepKernel(B, Bs, fixture["b"], fixture["d"],
                                 mode="fast+tik+pchol",
                                 tol=newt.FAST_STEP_TOL)
    result = newt.newton_projection(
        B, fixture["b"], fixture["d"], t0, scaled=(t0 != 0.0),
        maxiter=newt.MAXITER, Bs=Bs, kernel=kernel,
        deadline=time.perf_counter() + timeout)
    result["process_seconds"] = time.perf_counter() - started
    info = {key: value for key, value in result.items()
            if key not in ("x", "x_work", "y", "S")}
    if not result["converged"]:
        return None, info, ""
    return np.asarray(result["S"], dtype=bool), info, ""


def one_cell(model: str, t0: float, method: str, walker_pivots: int,
             timeout: float, temp: Path) -> dict:
    source = source_fixture(model)
    if method == "native":
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TWALKER_MAX_PIVOTS": str(walker_pivots),
        }
        started = time.perf_counter()
        walk = subprocess.run(
            [str(VERIFY), str(source)], check=False, capture_output=True,
            text=True, timeout=timeout, env=environment)
        process_seconds = time.perf_counter() - started
        try:
            result = json.loads(walk.stdout.strip().splitlines()[-1])
        except Exception:
            result = {"status": "UNPARSEABLE", "stdout": walk.stdout}
        reached = result.get("t")
        return {
            "model": model,
            "t0": 0.0,
            "seed_method": method,
            "seed_process_seconds": result.get("seed_ms", 0.0) / 1000.0,
            "process_seconds": process_seconds,
            "seed": {
                "converged": result.get("seed_converged", False),
                "iterations": result.get("seed_iterations", 0),
                "support": result.get("seed_support", 0),
                "dres": result.get("seed_dres"),
                "seconds": result.get("seed_ms", 0.0) / 1000.0,
            },
            "walker": result,
            "walker_stderr": walk.stderr.strip(),
            "initial_face_accepted": result.get("status") not in (
                "native seed failed", "initial face rejected"),
            "strict_forward_progress": (
                isinstance(reached, (int, float)) and reached > 2e-12),
            "status": result.get("status"),
        }
    fixture = read_fixture(source)
    if method == "nnls":
        mask, info, seed_stderr = identify_support_nnls(
            model, t0, fixture, timeout, temp)
    else:
        mask, info, seed_stderr = identify_support_newton(
            t0, fixture, timeout)
    record = {
        "model": model,
        "t0": float(t0),
        "seed_method": method,
        "seed_process_seconds": info.get("process_seconds"),
        "seed": info,
        "seed_stderr": seed_stderr,
    }
    if mask is None:
        record["status"] = "SEED_FAILED"
        return record

    if mask.shape != fixture["post_seed_support"].shape:
        record["status"] = "MASK_SIZE_MISMATCH"
        return record
    trial_fixture = temp / f"{model}-{method}-t{t0:g}.twfx"
    write_fixture(
        trial_fixture, fixture["B"].toarray(), fixture["b"], fixture["d"], mask,
        t0, [], {"model": model, "route": f"{method}-seed-audit"})
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TWALKER_MAX_PIVOTS": str(walker_pivots),
    }
    walk = subprocess.run(
        [str(VERIFY), str(trial_fixture)], check=False, capture_output=True,
        text=True, timeout=timeout, env=environment)
    try:
        result = json.loads(walk.stdout.strip().splitlines()[-1])
    except Exception:
        result = {"status": "UNPARSEABLE", "stdout": walk.stdout}
    record["walker"] = result
    record["walker_stderr"] = walk.stderr.strip()
    record["initial_face_accepted"] = (
        result.get("status") != "initial face rejected")
    reached = result.get("t")
    record["strict_forward_progress"] = (
        isinstance(reached, (int, float))
        and reached > t0 + 1e-12 * max(1.0, abs(t0)) + 1e-12)
    record["status"] = (
        "ACCEPTED" if record["initial_face_accepted"] else "REJECTED")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--t0", nargs="+", type=float, default=(0.0, 256.0))
    parser.add_argument("--method", choices=("nnls", "newton", "native"),
                        default="nnls")
    parser.add_argument("--walker-pivots", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix="twalker-seed-") as directory:
        temp = Path(directory)
        for model in args.models:
            for t0 in args.t0:
                try:
                    row = one_cell(model, t0, args.method, args.walker_pivots,
                                   args.timeout, temp)
                except subprocess.TimeoutExpired:
                    row = {"model": model, "t0": t0,
                           "status": "RESOURCE_LIMIT"}
                except Exception as error:
                    row = {"model": model, "t0": t0, "status": "ERROR",
                           "error": f"{type(error).__name__}: {error}"}
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
