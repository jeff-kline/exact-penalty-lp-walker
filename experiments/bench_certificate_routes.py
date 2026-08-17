"""Quarantined, interleaved A/B of the CERTIFICATE-DETAIL route.

WHY THIS EXISTS
---------------
Under shipped defaults the terminal/certificate bucket -- not Newton -- is the
dominant cost on the panel's most expensive model.  Measured today, single
thread, ``exp84_staged_newton.run``:

    model     wall    newton_solve   terminal_test
    ship04s   4.584   0.911 (19.9%)  3.570 (77.9%)
    sctap1    3.283   1.724 (52.5%)  1.552 (47.3%)
    lotfi     2.308   1.913 (82.9%)  0.392 (17.0%)
    fit1d     2.397   1.840 (76.8%)  0.499 (20.8%)

Two quarantined replacements for ``exp23_path_primal_dual._certificate_detail``
already exist -- ``hybrid_certificate_factor`` (one SVD instead of the
incumbent's ``lstsq`` plus a second, FULL SVD for the null space) and
``terminal_qr_cod`` (a single pivoted QR when the active support is
overcomplete, dispatching to the hybrid otherwise).  Their previous A/B was
recorded but never cited, and the panel-level terminal total it reported
(4.414 s across 27 models) is SMALLER than ship04s alone shows under shipped
defaults (3.570 s).  Those two numbers cannot both describe the same route, so
the substitution's effect has to be measured cleanly.  That is this file.

WHAT IT DOES NOT DO
-------------------
It edits nothing.  It installs a route by assigning ``exp23._certificate_detail``
inside a ``try/finally`` and restores the captured original on every exit path.
It does NOT claim "protected hashes unchanged" -- the previous harness asserted
exactly that while monkeypatching a function defined in one of the very files
it hashed.  Instead every row RECORDS the identity of the function that was
actually installed: module, qualname, defining file and that file's sha256.

The gates stay frozen.  Only ``exp23.certificate_pair`` may declare CERTIFIED,
componentwise on the ORIGINAL ``(B, b, d)`` at ``TOL_FEAS = 1e-7``.  Both
replacement routes return ``True`` only after ``exp23.certificate_pair`` passes,
and this harness ASSERTS that at runtime rather than trusting the source: a
counting spy is installed on ``exp23.certificate_pair`` and any detail route
that returns ``ok=True`` without having called it raises.  The spy is invisible
to ``oracle_terminal._gate``, which bound ``certificate_pair`` by name at import
time, so the gate's own hot path is untouched and unmeasured-by-this.

ACCURACY NUMBER
---------------
The four FROZEN componentwise errors are ``primal``, ``dual``, ``nonnegative``,
``gap`` from ``exp23.certificate_pair`` (see ``certificate_pair``'s ``errors``
dict), surfaced in the record's ``certificate_detail`` block.  Their max is the
accuracy number.  ``gate_full`` (exp84 ~:240-253) is a DIFFERENTLY SCALED
diagnostic -- on fit1d the frozen gate's true worst component is 2.35e-8 while
``gate_full`` reports 1.13e-9 -- so it is recorded for reference and explicitly
labelled, never used as the accuracy number.

One trap is handled rather than hidden.  When ``oracle_terminal._gate`` accepts
the ``x`` built by ``_certificate_detail`` (``gate_x ==
"certificate-detail lstsq/newton4"``), the four components in the record are
those of the FAILED ``certificate_pair`` call on the PREVIOUS candidate, not of
the certified pair -- on afiro that reads 7.56e-3, which is not an error, it is
a different pair.  So the harness stashes the accepted ``(x, y)`` during the
run (three array references, no arithmetic) and re-evaluates the frozen
``certificate_pair`` on it AFTER the timed region.  Every row carries
``components_describe_certified_pair``, the recomputation, and
``certificate_max_authoritative`` with the ``certificate_max_authority`` that
produced it.

METHOD
------
Interleaved and both-orders: with three arms, repeat ``r`` uses cyclic rotation
``r // 2`` reversed on odd ``r``, so six repeats cover all six permutations and
no arm systematically occupies a warmer or cooler slot.  Arms are adjacent
within a model, so the paired delta is taken across the smallest possible time
separation.  Reported as PAIRED per-model deltas with median and min/max across
repeats -- never a single run, never min-of-N.  Every repeat is bracketed by a
machine-state probe (load average plus a fixed 600x600 dgemm, best of 3), on the
model of ``bench_noise_floor.py``.

Canonical invocation (thread env vars MUST be set by the caller -- setting them
in-process is a no-op after numpy is imported, and this script verifies rather
than pretends):

    env PYTHONPYCACHEPREFIX=/tmp/fable_lp_pycache \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      ~/.venvs/claude/bin/python experiments/bench_certificate_routes.py \
      --models ship04s --repeats 6
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import inspect
import itertools
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

# ``hybrid_certificate_factor`` captures ``exp23._certificate_detail`` into its
# own ``_ORIGINAL`` AT IMPORT TIME and calls it for square active systems.  It
# must therefore be imported BEFORE anything is installed, or the "incumbent"
# it falls back to would be a replacement route and the dispatch would recurse.
# The assertion in ``main`` checks that this actually held.
import exp23_path_primal_dual as exp23            # noqa: E402
import hybrid_certificate_factor as hybrid        # noqa: E402
import terminal_qr_cod as qr_cod                  # noqa: E402
import exp84_staged_newton as exp84               # noqa: E402
import quarantined_speedups as qs                 # noqa: E402
import quarantined_pivchol_alloc as qpa           # noqa: E402

DEFAULT_RAW_ROOT = ROOT / "records/certificate_routes"

# The frozen incumbent, captured once, before any install can happen.
FROZEN_CERTIFICATE_DETAIL = exp23._certificate_detail
FROZEN_CERTIFICATE_PAIR = exp23.certificate_pair

ARMS = ("frozen", "hybrid", "qr_cod")

MODELS = [
    "afiro", "sc50a", "sc50b", "kb2", "blend", "adlittle", "share2b",
    "stocfor1", "sc105", "recipe", "scagr7", "sc205", "boeing2",
    "israel", "share1b", "beaconfd", "brandy", "e226", "lotfi", "bandm",
    "capri", "grow7", "sctap1", "scorpion", "degen2", "fit1d", "ship04s",
]
EXCLUDED = {"forplan": "MPS converter failure (out of solver scope)"}

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

CALL_PATH = [
    "experiments/exp84_staged_newton.py",
    "experiments/staged_newton.py",
    "experiments/newton_oracle.py",
    "experiments/oracle_terminal.py",
    "experiments/woodbury_face_solver.py",
    "experiments/exp23_path_primal_dual.py",
    "experiments/exp11_netlib_rank_certificate.py",
    "experiments/hybrid_certificate_factor.py",
    "experiments/terminal_qr_cod.py",
    "experiments/bench_certificate_routes.py",
]

# The four frozen componentwise errors, in ``certificate_pair``'s own key
# names.  ``nonneg`` is accepted as an alias so a future rename does not
# silently drop the component instead of failing loudly.
COMPONENT_KEYS = (("primal", ("primal",)),
                  ("dual", ("dual",)),
                  ("nonnegative", ("nonnegative", "nonneg")),
                  ("gap", ("gap",)))


# --------------------------------------------------------------- route install

_PAIR_CALLS = [0]


def _pair_spy(*args, **kwargs):
    """Counting passthrough for ``exp23.certificate_pair``.

    ``oracle_terminal`` did ``from exp23_path_primal_dual import
    certificate_pair`` at import time, so its ``_gate`` holds a direct
    reference and never sees this wrapper -- the gate's hot path is unchanged.
    What this DOES see is every call made through the ``exp23.`` module
    attribute, which is exactly how both replacement routes and the frozen
    ``_certificate_detail`` reach the gate.  That is the call the assertion
    below is about.
    """
    _PAIR_CALLS[0] += 1
    return FROZEN_CERTIFICATE_PAIR(*args, **kwargs)


def _guarded(fn, arm):
    """Wrap a certificate-detail route so it cannot pass without the gate.

    Fail-closed: if the route returns ``ok=True`` without having called
    ``exp23.certificate_pair`` at least once during that call, raise.  This is
    an assertion about the ROUTE, never a relaxation of any gate.
    """
    @functools.wraps(fn)
    def wrapper(B, b, d, y):
        before = _PAIR_CALLS[0]
        out = fn(B, b, d, y)
        _CALLS[arm]["detail_calls"] += 1
        ok = bool(out[0]) if isinstance(out, tuple) and out else False
        used = _PAIR_CALLS[0] - before
        _CALLS[arm]["pair_calls_inside_detail"] += used
        if ok:
            _CALLS[arm]["detail_true_returns"] += 1
            _ACCEPTED[arm] = (B, b, d, out[1], y)
            if used < 1:
                raise AssertionError(
                    "certificate route %r returned ok=True without calling "
                    "exp23.certificate_pair -- refusing to record a "
                    "certificate that the frozen gate never saw" % arm)
        return out
    return wrapper


_CALLS = {}
# Arguments of the LAST certificate-detail call that returned ``ok=True``.
# ``oracle_terminal._gate`` accepts that call's ``x`` and returns True, which
# ends the run, so this is the pair the solver actually certified.  Stashed
# during the timed region (three array references, no copy, no arithmetic) and
# evaluated AFTER it, so the authoritative accuracy number costs the
# measurement nothing.
_ACCEPTED = {}


def _reset_calls(arm):
    _CALLS[arm] = {"detail_calls": 0, "detail_true_returns": 0,
                   "pair_calls_inside_detail": 0}
    _ACCEPTED.pop(arm, None)
    _PAIR_CALLS[0] = 0


def _route_function(arm):
    if arm == "frozen":
        return FROZEN_CERTIFICATE_DETAIL
    if arm == "hybrid":
        return hybrid.single_factor_certificate_detail
    if arm == "qr_cod":
        return qr_cod.qr_cod_certificate_detail
    raise ValueError("unknown arm %r" % arm)


def _identity(fn):
    """Module + qualname + defining file + that file's sha256.

    Recorded on EVERY row.  A benchmark that substitutes a function must say
    which function it substituted; asserting that a set of file hashes did not
    change says nothing at all about a monkeypatch.
    """
    target = inspect.unwrap(fn)
    try:
        source = inspect.getsourcefile(target)
    except TypeError:                              # pragma: no cover
        source = None
    digest, relative = None, None
    if source:
        path = Path(source).resolve()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:                            # pragma: no cover
            digest = None
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError:                         # pragma: no cover
            relative = str(path)
    return {
        "module": getattr(target, "__module__", None),
        "qualname": getattr(target, "__qualname__", None),
        "source_file": relative,
        "source_sha256": digest,
        "is_frozen_incumbent": target is FROZEN_CERTIFICATE_DETAIL,
    }


# ------------------------------------------------------------- machine state

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
    except OSError:                                # pragma: no cover
        load1 = load5 = load15 = None
    return {"load_1min": load1, "load_5min": load5, "load_15min": load15,
            "dgemm_600_seconds": _dgemm_probe()}


def _provenance():
    def _git(*a):
        try:
            return subprocess.run(("git",) + a, cwd=str(ROOT), text=True,
                                  capture_output=True,
                                  timeout=20).stdout.strip()
        except Exception:                          # pragma: no cover
            return None
    hashes = {}
    for name in CALL_PATH:
        path = ROOT / name
        hashes[name] = (hashlib.sha256(path.read_bytes()).hexdigest()
                        if path.exists() else None)
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:                              # pragma: no cover
        scipy_version = None
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "call_path_sha256": hashes,
        "note": "these hashes describe the FILES on disk; they say nothing "
                "about what was installed at runtime -- see each row's "
                "installed_certificate_detail",
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy_version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_VARS},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


# ------------------------------------------------------------------ summaries

def _spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return {}
    out = {"n": len(values), "min": min(values),
           "median": statistics.median(values), "max": max(values),
           "mean": statistics.fmean(values)}
    out["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return out


def _components(detail):
    """The four frozen componentwise errors, plus their max and the flags.

    Reads ``certificate_pair``'s own key names.  A missing component is
    reported as ``None`` and forces ``max`` to ``None`` rather than silently
    maximising over a shorter list.
    """
    detail = detail or {}
    out, values, complete = {}, [], True
    for name, aliases in COMPONENT_KEYS:
        value = None
        for alias in aliases:
            if alias in detail and detail[alias] is not None:
                value = float(detail[alias])
                break
        out[name] = value
        if value is None:
            complete = False
        else:
            values.append(value)
    out["max"] = max(values) if (values and complete) else None
    out["components_complete"] = complete
    out["reason"] = detail.get("reason")
    out["gate_x"] = detail.get("gate_x")
    out["certificate_detail_skipped"] = detail.get(
        "certificate_detail_skipped")
    out["certificate_detail_ok"] = detail.get("certificate_detail_ok")
    out["certificate_detail_reason"] = detail.get("certificate_detail_reason")
    # When ``_gate`` accepted the x built by ``_certificate_detail``, the four
    # components above belong to the FAILED pair call on the previous
    # candidate, not to the certified pair.  Say so on the row.
    out["components_describe_certified_pair"] = (
        detail.get("gate_x") != "certificate-detail lstsq/newton4")
    return out


def _arm_order(repeat, arms):
    """Cyclic rotation, reversed on odd repeats: all permutations for 3 arms."""
    arms = list(arms)
    rotation = (repeat // 2) % len(arms)
    order = arms[rotation:] + arms[:rotation]
    return order[::-1] if repeat % 2 else order


# ----------------------------------------------------------------------- main

def _run_one(model, arm, seconds, repeat, slot, alloc_targets=()):
    """One PRE-PROMOTION-defaults solve under one installed certificate route.

    PROMOTION NOTE.  ``exp84_staged_newton`` now defaults to the adopted
    composition (``--route qr_cod``, Ruiz equilibration, the c1/c5 candidates
    and the a1/a2/a3/b1 allocation patches).  This function is the hub every
    route/transform/composition harness in the project calls, and each of those
    harnesses defines its OWN control, so it pins the solver back to the
    pre-promotion pipeline and layers only what the caller asked for:

      * the certificate route is the ARM's, installed here as before -- and
        ``exp84.run`` yields to it, because it is no longer the frozen
        incumbent by the time the solve starts;
      * ``equilibrate=False``: the transform harnesses do their own
        equilibration around this call and would otherwise double-transform,
        and the route/composition harnesses never equilibrated at all;
      * the kernel candidates are left exactly as the caller installed them
        (``candidates=None``) when a ``quarantined_speedups.active`` block is
        open, and forced to their PRE-PROMOTION values otherwise;
      * ``alloc_targets`` defaults to EMPTY, i.e. the pre-promotion allocation
        code path.  ``bench_pivchol_alloc`` passes its targets here for the
        treatment cell.

    So every historical arm still means what it meant, and the deltas already
    recorded remain comparable to new ones taken with this file.
    """
    route_fn = _route_function(arm)
    identity = _identity(route_fn)
    hybrid.reset_stats()
    qr_cod.reset_stats()
    _reset_calls(arm)
    exp23.certificate_pair = _pair_spy
    exp23._certificate_detail = _guarded(route_fn, arm)
    try:
        with contextlib.ExitStack() as stack:
            if qs.installed() is None:
                stack.enter_context(qs.active(()))
            if qpa.installed() is None:
                stack.enter_context(qpa.active(alloc_targets))
            elif alloc_targets:
                raise RuntimeError(
                    "alloc_targets=%r requested while a "
                    "quarantined_pivchol_alloc block is already open"
                    % (list(alloc_targets),))
            record = exp84.run(model, seconds=seconds, quiet=True,
                               route=None, equilibrate=False,
                               candidates=None, alloc_patches=None)
    finally:
        exp23._certificate_detail = FROZEN_CERTIFICATE_DETAIL
        exp23.certificate_pair = FROZEN_CERTIFICATE_PAIR
    detail = record.get("certificate_detail")
    components = _components(detail)
    # ---- authoritative accuracy number, computed OUTSIDE the timed region --
    # When ``_gate`` accepted the x built by the certificate-detail route, the
    # four components in ``record["certificate_detail"]`` are those of the
    # FAILED pair on the previous candidate.  Re-run the FROZEN gate on the
    # pair the solver actually certified.  This is measurement, not a gate:
    # the run is already over and its verdict cannot change.
    recomputed = None
    stashed = _ACCEPTED.pop(arm, None)
    if stashed is not None:
        cB, cb, cd, cx, cy = stashed
        try:
            pair_ok, pair_detail = FROZEN_CERTIFICATE_PAIR(cB, cb, cd, cx, cy)
        except Exception as error:                 # pragma: no cover
            recomputed = {"error": type(error).__name__}
        else:
            recomputed = _components(pair_detail)
            recomputed["passed"] = bool(pair_ok)
            recomputed["source"] = ("frozen certificate_pair re-evaluated "
                                    "post-run on the accepted (x, y)")
    if components["components_describe_certified_pair"]:
        authoritative = components["max"]
        authority = "record certificate_detail block"
    elif recomputed is not None and recomputed.get("max") is not None:
        authoritative = recomputed["max"]
        authority = "post-run frozen certificate_pair on the accepted pair"
    else:
        authoritative = None
        authority = "unavailable"
    row = {
        "repeat": repeat,
        "slot": slot,
        "arm": arm,
        "model": model,
        "status": record.get("status"),
        "wall_seconds": float(record["wall_seconds"]),
        "wall_split": record.get("wall_split"),
        "newton_iters_total": record.get("newton_iters_total"),
        "stages_run": record.get("stages_run"),
        "terminal_tests_run": record.get("terminal_tests_run"),
        "terminal_tests_skipped": record.get("terminal_tests_skipped"),
        "accepted_stage": record.get("accepted_stage"),
        "accepted_face_size": record.get("accepted_face_size"),
        "route": record.get("route"),
        "staged_fallback": record.get("staged_fallback"),
        "lb": record.get("lb"),
        "t_accepted": record.get("t_accepted"),
        "certificate": components,
        "certificate_recomputed": recomputed,
        "certificate_max_authoritative": authoritative,
        "certificate_max_authority": authority,
        # Recorded for reference ONLY.  Differently scaled from the frozen
        # componentwise gate; not the accuracy number.
        "gate_full_diagnostic": record.get("gate_full"),
        "gate_full_note": "differently-scaled diagnostic (exp84 ~:240-253); "
                          "NOT the frozen accuracy number",
        "installed_certificate_detail": identity,
        # The promoted kernel flags AS THEY WERE DURING THE SOLVE, read off the
        # modules inside the timed block by ``exp84.run``.  Any state read
        # after the run describes the module defaults, not the arm.
        "solver_configuration": record.get("configuration"),
        "certificate_pair_guard": dict(_CALLS[arm]),
        "route_stats": {"hybrid": hybrid.snapshot(),
                        "qr_cod": qr_cod.snapshot()},
        "terminal_face_stats": record.get("terminal_face_stats"),
    }
    return row


def main():
    parser = argparse.ArgumentParser(
        description="interleaved, both-orders A/B of the certificate-detail "
                    "route under shipped defaults")
    parser.add_argument("--repeats", type=int, default=6,
                        help="repeats; 6 covers all 6 arm orders (default 6)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of the 27-model panel; default all")
    parser.add_argument("--arms", nargs="*", default=list(ARMS),
                        choices=list(ARMS))
    parser.add_argument("--seconds", type=int, default=300,
                        help="per-model resource ceiling; default 300")
    parser.add_argument("--warmup-model", default="afiro",
                        help="model solved once per arm and DISCARDED before "
                             "the repeats begin; 'none' disables.  Measured "
                             "on afiro: the very first solve of a process "
                             "costs 0.029 s against 0.009 s thereafter, so "
                             "without this the arm in repeat 0 slot 0 pays an "
                             "import/allocator cost no other arm pays")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    missing = [k for k in THREAD_VARS if os.environ.get(k) != "1"]
    if missing:
        raise SystemExit(
            "refusing to run: %s must be set to 1 in the ENVIRONMENT before "
            "python starts (setting them in-process is a no-op after numpy "
            "is imported)" % ", ".join(missing))

    # The hybrid route's fallback must be the TRUE incumbent, not a
    # replacement -- otherwise ``frozen`` would not be frozen and the arms
    # would not be independent.
    if hybrid._ORIGINAL is not FROZEN_CERTIFICATE_DETAIL:
        raise SystemExit(
            "refusing to run: hybrid_certificate_factor._ORIGINAL is not the "
            "frozen exp23._certificate_detail (import order violated)")
    if exp23._certificate_detail is not FROZEN_CERTIFICATE_DETAIL:
        raise SystemExit("refusing to run: exp23._certificate_detail was "
                         "already substituted before the harness started")

    models = args.models or MODELS
    arms = list(args.arms)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.output_dir.resolve() if args.output_dir
               else DEFAULT_RAW_ROOT / ("%s-pid%d" % (stamp, os.getpid())))
    if out_dir.exists():
        raise FileExistsError("refusing to reuse output directory %s" % out_dir)
    out_dir.mkdir(parents=True)
    rows_path = out_dir / "runs.jsonl"
    summary_path = out_dir / "certificate-routes-summary.json"

    provenance = _provenance()
    rows, repeat_records = [], []

    warmup = []
    if args.warmup_model and args.warmup_model != "none":
        for arm in arms:
            discarded = _run_one(args.warmup_model, arm, args.seconds, -1, -1)
            warmup.append({"arm": arm, "model": args.warmup_model,
                           "wall_seconds": discarded["wall_seconds"],
                           "status": discarded["status"],
                           "discarded": True})
        print(json.dumps({"warmup": warmup}, sort_keys=True),
              file=sys.stderr, flush=True)

    with rows_path.open("x", encoding="utf-8") as stream:
        for repeat in range(args.repeats):
            order = _arm_order(repeat, arms)
            before = _machine_state()
            started = time.perf_counter()
            for model in models:
                for slot, arm in enumerate(order):
                    row = _run_one(model, arm, args.seconds, repeat, slot)
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True,
                                            default=str) + "\n")
                    stream.flush()
            elapsed = time.perf_counter() - started
            after = _machine_state()
            repeat_records.append({
                "repeat": repeat,
                "arm_order": order,
                "elapsed_seconds": elapsed,
                "machine_state_before": before,
                "machine_state_after": after,
            })
            print(json.dumps({"repeat": repeat, "arm_order": order,
                              "elapsed": round(elapsed, 3),
                              "dgemm_before": round(
                                  before["dgemm_600_seconds"], 4),
                              "dgemm_after": round(
                                  after["dgemm_600_seconds"], 4)},
                             sort_keys=True), file=sys.stderr, flush=True)

    # ---- PAIRED per-model deltas -------------------------------------------
    def _wall(model, arm, repeat):
        for row in rows:
            if (row["model"] == model and row["arm"] == arm
                    and row["repeat"] == repeat):
                return row["wall_seconds"]
        return None

    pairs = [(a, b) for a, b in itertools.combinations(arms, 2)]
    per_model = {}
    for model in models:
        entry = {"arms": {}, "paired_delta_seconds": {},
                 "certificate_max_by_arm": {}, "status_by_arm": {}}
        for arm in arms:
            walls = [r["wall_seconds"] for r in rows
                     if r["model"] == model and r["arm"] == arm]
            entry["arms"][arm] = _spread(walls)
            entry["status_by_arm"][arm] = sorted({
                str(r["status"]) for r in rows
                if r["model"] == model and r["arm"] == arm})
            maxima = [r["certificate_max_authoritative"] for r in rows
                      if r["model"] == model and r["arm"] == arm]
            entry["certificate_max_by_arm"][arm] = _spread(maxima)
        for a, b in pairs:
            deltas = []
            for repeat in range(args.repeats):
                wa, wb = _wall(model, a, repeat), _wall(model, b, repeat)
                if wa is not None and wb is not None:
                    deltas.append(wb - wa)
            entry["paired_delta_seconds"]["%s_minus_%s" % (b, a)] = {
                **_spread(deltas),
                "wins_for_%s" % b: sum(1 for x in deltas if x < 0.0),
                "wins_for_%s" % a: sum(1 for x in deltas if x > 0.0),
            }
        # The four components should be IDENTICAL across arms whenever the
        # gate passed on ``certificate_pair`` itself, because the route only
        # changes the fallback that runs when the pair FAILED.
        signatures = {}
        for arm in arms:
            signatures[arm] = sorted({
                json.dumps([r["certificate"][k] for k, _ in COMPONENT_KEYS]
                           + [(r["certificate_recomputed"] or {}).get(k)
                              for k, _ in COMPONENT_KEYS],
                           sort_keys=True)
                for r in rows if r["model"] == model and r["arm"] == arm})
        entry["certificate_components_identical_across_arms"] = (
            len({json.dumps(v) for v in signatures.values()}) == 1)
        entry["certificate_components_stable_within_arm"] = {
            arm: len(sig) == 1 for arm, sig in signatures.items()}
        per_model[model] = entry

    # ---- panel totals per repeat, then paired deltas of the totals ---------
    panel = {"per_repeat_total_seconds": {}, "paired_delta_seconds": {}}
    totals = {arm: [] for arm in arms}
    for repeat in range(args.repeats):
        for arm in arms:
            walls = [r["wall_seconds"] for r in rows
                     if r["arm"] == arm and r["repeat"] == repeat]
            totals[arm].append(sum(walls) if walls else None)
    for arm in arms:
        panel["per_repeat_total_seconds"][arm] = totals[arm]
    for a, b in pairs:
        deltas = [y - x for x, y in zip(totals[a], totals[b])
                  if x is not None and y is not None]
        panel["paired_delta_seconds"]["%s_minus_%s" % (b, a)] = {
            **_spread(deltas),
            "wins_for_%s" % b: sum(1 for x in deltas if x < 0.0),
            "wins_for_%s" % a: sum(1 for x in deltas if x > 0.0),
        }

    installed = {}
    for arm in arms:
        seen = {json.dumps(r["installed_certificate_detail"], sort_keys=True)
                for r in rows if r["arm"] == arm}
        installed[arm] = [json.loads(s) for s in sorted(seen)]

    # A CERTIFIED row whose AUTHORITATIVE worst component exceeds TOL_FEAS
    # would mean the frozen gate was bypassed.  Judged on the authoritative
    # number, never on the record block alone -- the record block legitimately
    # carries the failed candidate's errors whenever the gate accepted the
    # certificate-detail x.
    guard_violation = any(
        r["certificate_max_authoritative"] is not None
        and r["certificate_max_authoritative"] > exp23.TOL_FEAS
        and r["status"] == "CERTIFIED" for r in rows)
    unavailable = [
        {"model": r["model"], "arm": r["arm"], "repeat": r["repeat"]}
        for r in rows if r["certificate_max_authoritative"] is None]

    summary = {
        "experiment": "certificate-detail route A/B under shipped defaults",
        "classification": "DIRECT MEASUREMENT; INTERLEAVED; BOTH ORDERS; "
                          "PAIRED PER-MODEL DELTAS; NO NORMALIZATION",
        "arms": arms,
        "arm_definition": {
            "frozen": "no substitution (shipped default): "
                      "exp23._certificate_detail",
            "hybrid": "hybrid_certificate_factor."
                      "single_factor_certificate_detail",
            "qr_cod": "terminal_qr_cod.qr_cod_certificate_detail",
        },
        "installed_certificate_detail_by_arm": installed,
        "panel": {"models": len(models), "excluded": EXCLUDED,
                  "repeats": args.repeats, "per_repeat": repeat_records,
                  "warmup_discarded": warmup, **panel},
        "correctness": {
            "runs_total": len(rows),
            "certified_total": sum(r["status"] == "CERTIFIED" for r in rows),
            "all_certified": all(r["status"] == "CERTIFIED" for r in rows),
            "fallbacks": sum(bool(r["staged_fallback"]) for r in rows),
            "certificate_pair_guard_violation": guard_violation,
            "tol_feas": exp23.TOL_FEAS,
            "accuracy_number": "max of the four frozen componentwise errors "
                               "(primal, dual, nonnegative, gap) from "
                               "exp23.certificate_pair, taken on the pair the "
                               "solver actually certified; gate_full is NOT it",
            "worst_certificate_max_by_arm": {
                arm: (max(values) if values else None)
                for arm, values in (
                    (arm, [r["certificate_max_authoritative"] for r in rows
                           if r["arm"] == arm
                           and r["certificate_max_authoritative"] is not None])
                    for arm in arms)},
            "rows_where_record_block_describes_certified_pair": sum(
                bool(r["certificate"]["components_describe_certified_pair"])
                for r in rows),
            "rows_needing_post_run_recomputation": sum(
                r["certificate_max_authority"].startswith("post-run")
                for r in rows),
            "rows_with_no_authoritative_number": unavailable,
        },
        "per_model": per_model,
        "provenance": provenance,
        "provenance_after": _provenance(),
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2,
                                       default=str) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("arms", "installed_certificate_detail_by_arm",
                       "correctness")},
                     sort_keys=True, indent=2, default=str)[:4000])
    print("rows:    " + str(rows_path))
    print("summary: " + str(summary_path))


if __name__ == "__main__":
    main()
