#!/usr/bin/env python3
"""Run the native reduced walk, lift it, and certify on original data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from exp23_path_primal_dual import certificate_pair  # noqa: E402


def lift(book, xr, yr):
    x = book["x0"] + book["Z"] @ xr
    g = book["d0"] - book["Bi"].T @ yr
    lam, *_ = np.linalg.lstsq(book["E"].T, g, rcond=None)
    y = np.zeros(book["B0"].shape[0])
    y[book["ineq"].astype(int)] = yr
    y[book["plus"].astype(int)] = np.maximum(lam, 0.0)
    y[book["minus"].astype(int)] = np.maximum(-lam, 0.0)
    residual = float(np.max(np.abs(book["E"].T @ lam - g))) / max(
        1.0, float(np.max(np.abs(book["d0"]))))
    return x, y, residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("lift_book", type=Path)
    parser.add_argument("--walker", type=Path,
                        default=ROOT / "cpp" / "twalker" / "build" /
                        "verify_walker")
    args = parser.parse_args()
    env = os.environ.copy()
    env["TWALKER_EMIT_SOLUTION"] = "1"
    env["TWALKER_GRAM_MIN_RCOND"] = "1e-6"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    completed = subprocess.run(
        [str(args.walker), str(args.fixture)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)
    if not completed.stdout.strip():
        raise RuntimeError("native walker produced no JSON: " + completed.stderr)
    native = json.loads(completed.stdout.strip().splitlines()[-1])
    with np.load(args.lift_book) as loaded:
        book = {key: loaded[key] for key in loaded.files}
    xr = np.asarray(native["solution"].pop("x"), dtype=float)
    yr = np.asarray(native["solution"].pop("y"), dtype=float)
    x, y, lift_residual = lift(book, xr, yr)
    ok, detail = certificate_pair(book["B0"], book["b0"], book["d0"], x, y)
    native["native_exit_code"] = completed.returncode
    native["original_certificate"] = detail
    native["original_certified"] = bool(ok)
    native["dual_lift_residual"] = lift_residual
    print(json.dumps(native, sort_keys=True))
    return 0 if ok and native.get("status") == "CERTIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
