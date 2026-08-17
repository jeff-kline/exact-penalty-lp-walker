"""Run the C++ t-walker over the Netlib panel, one subprocess per model.

This is the producer for the frozen Netlib walker records in
``records/twalker_cpp/``.  It is deliberately thin: ``verify_walker`` is
authoritative for the walk, the fixed-``t`` settle, and the original-data
certificate.  This script only chooses the fixtures, gives each model its own
process, times that process, and records the result verbatim.

One process per model, so that each row carries its own process timing.  This
was checked rather than assumed: passing all fixtures to a single
``verify_walker`` invocation instead produced structurally identical output on
26 of 26 models (status, pivots, seed iterations, accepted support, face
solves, settle rounds, tied events).  Two independent per-model runs likewise
agreed on all 26.  The walker's path is deterministic on a fixed build; only
wall-clock timing varies, by a few percent except on models finishing under
10 ms, where a 3 ms jitter is a large ratio.

Run:
    .venv/bin/python experiments/bench_twalker_netlib_panel.py \
        --output records/twalker_cpp/native_seed_t0_netlib27_rerun.json
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path


def _child_cpu_seconds() -> float:
    """CPU time consumed by finished child processes.

    ``time.process_time()`` measures only this interpreter and therefore reads
    ~0 for work done in a subprocess; use the children rusage instead.
    """
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "cpp" / "twalker" / "build" / "verify_walker"
FIXTURES = ROOT / "cpp" / "twalker" / "fixtures_panel"

# The canonical 27-model panel.  ``grow7`` has no compact fixture; it is
# reported as NOT_MEASURED rather than silently dropped, so the panel size
# stays 27 in every record.
PANEL = [
    "adlittle", "afiro", "bandm", "beaconfd", "blend", "boeing2", "brandy",
    "capri", "degen2", "e226", "fit1d", "grow7", "israel", "kb2", "lotfi",
    "recipe", "sc105", "sc205", "sc50a", "sc50b", "scagr7", "scorpion",
    "sctap1", "share1b", "share2b", "ship04s", "stocfor1",
]


def run_model(model: str, timeout: float) -> dict:
    fixture = FIXTURES / f"{model}.twfx"
    if not fixture.exists():
        return {"model": model, "status": "NOT_MEASURED",
                "detail": f"fixture not found: {fixture.relative_to(ROOT)}",
                "process_seconds": None, "walker": None}

    start = _child_cpu_seconds()
    wall = time.perf_counter()
    try:
        proc = subprocess.run([str(VERIFY), str(fixture)],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"model": model, "status": "RESOURCE_LIMIT",
                "detail": f"wall-clock budget {timeout}s exhausted",
                "process_seconds": _child_cpu_seconds() - start,
                "wall_seconds": time.perf_counter() - wall, "walker": None}

    process_seconds = _child_cpu_seconds() - start
    wall_seconds = time.perf_counter() - wall

    walker = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                walker = json.loads(line)
            except json.JSONDecodeError:
                pass

    if walker is None:
        return {"model": model, "status": "NO_OUTPUT",
                "detail": f"verify_walker exit {proc.returncode}",
                "process_seconds": process_seconds, "wall_seconds": wall_seconds,
                "walker": None, "walker_stderr": proc.stderr[-2000:]}

    seed_seconds = walker.get("seed_ms")
    return {
        "model": model,
        "status": walker.get("status"),
        "process_seconds": process_seconds,
        "wall_seconds": wall_seconds,
        "seed_process_seconds": (seed_seconds / 1000.0
                                 if seed_seconds is not None else None),
        "seed_method": "native" if walker.get("native_seed") else "external",
        "t0": walker.get("t"),
        "initial_face_accepted": walker.get("seed_converged"),
        "walker": walker,
        "walker_stderr": proc.stderr[-2000:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=PANEL)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-model wall-clock budget in seconds")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if not VERIFY.exists():
        raise SystemExit(
            f"{VERIFY.relative_to(ROOT)} not built. Run:\n"
            f"  cd cpp/twalker && make build/verify_walker "
            f"PYTHON=../../.venv/bin/python")

    rows = [run_model(m, args.timeout) for m in args.models]
    payload = {
        "schema": "twalker-netlib-panel-v1",
        "invocation": "one verify_walker subprocess per model",
        "timeout_seconds": args.timeout,
        "models": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    certified = sum(1 for r in rows if r.get("status") == "CERTIFIED")
    total = sum(r["process_seconds"] for r in rows
                if r.get("process_seconds") is not None)
    print(f"{certified}/{len(rows)} certified; "
          f"{total:.1f} s total process time -> {args.output}")


if __name__ == "__main__":
    main()
