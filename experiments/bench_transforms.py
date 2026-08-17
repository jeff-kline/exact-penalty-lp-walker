"""2x2 transform harness: does PRESOLVE or EQUILIBRATION help the NATIVE lane?

WHAT THIS MEASURES
------------------
The native lane solves ``min d'x s.t. Bx >= b`` (Mangasarian Program (4) via
the Wolfe-dual projection, semismooth Newton on ``theta_t``) with the staged
Newton terminal.  Its current best configuration -- certificate route
``qr_cod`` plus the quarantined kernel candidates ``c1_const_cache`` and
``c5_pchol_screen`` -- runs the 27-model panel in 18.221 s (median of 3,
18.082-18.577), 27/27 CERTIFIED.

HiGHS, on the SAME converted ``(B, b, d)``, measured the same day::

    presolve on,  simplex   0.1620 s      presolve off, simplex   0.4137 s
    presolve on,  ipm       0.3109 s      presolve off, ipm       0.5404 s

Presolve is worth 2.6x to HiGHS simplex.  Nobody has ever measured whether it
is worth anything to US.  Neither has equilibration.  This file is the 2x2 that
answers it, deliberately mirroring the structure of the HiGHS baseline harness
``records/benchmarks_2026-08-06/hnp_baseline.py`` (a 2x2 of presolve
on/off x scale strategy default/off) so the two columns are like-for-like.

    none      current best configuration on the raw (B, b, d)   <- control
    presolve  HiGHS presolve -> solve reduced -> HiGHS postsolve
    equil     Ruiz equilibration -> solve scaled -> unscale
    both      presolve, then equilibration ON THE REDUCED program

Every cell runs the SAME solver configuration.  Only the program handed to the
solver differs.

*** THESE ARE NOT NATIVE-COLUMN RESULTS. ***  The ``presolve`` and ``both``
cells call HiGHS.  The lane is then NOT self-contained, and the standing
project rule is that the native column uses no presolve.  This harness exists
to answer "WOULD presolve help us", not "let us ship presolve".  Any number
from the ``presolve`` or ``both`` cell that is quoted as a native-lane figure
is a misquotation.  The ``equil`` cell is self-contained (pure numpy) and does
not carry that caveat.

WHAT COUNTS AS A RESULT
-----------------------
``exp23_path_primal_dual.certificate_pair(B, b, d, x, y)`` -- FROZEN,
componentwise, ``TOL_FEAS = 1e-7`` -- evaluated on the ORIGINAL raw
``(B, b, d)`` with the pair lifted/unscaled back to the original space is the
ONLY thing that may declare CERTIFIED.  The certificate the solver itself
reached in the reduced or scaled space is recorded (``solve_status``,
``transformed_certificate_max``) as an INTERMEDIATE DIAGNOSTIC and is never
allowed to set the verdict.  This is the discipline ``presolve_walker.py``
documents and implements for the walker lane; this file applies it to the
staged-Newton lane.

THE TIMED REGION
----------------
``wall_seconds = transform_seconds + solve_seconds + inverse_seconds``.

The transform and its inverse are INSIDE it.  HiGHS's presolve-on number
includes HiGHS's own presolve time, so a comparison that timed only our solve
would be rigged in our favour.  ``transform_seconds`` and ``inverse_seconds``
are recorded separately so the result can also be reported solve-only, but the
headline is end-to-end.  For the ``none`` cell both are exactly 0.0 and
``wall_seconds == solve_seconds``, so the control reproduces the existing
18.221 s measurement rather than a re-derivation of it.

``end_to_end_seconds`` additionally brackets ``exp84_staged_newton.run``'s own
record-building overhead; it is recorded but is not the headline, because the
control's published figure does not include it either.

EQUILIBRATION
-------------
Ruiz-style iterative row/column infinity-norm balancing with every factor
ROUNDED TO A POWER OF TWO, so scaling and unscaling are bit-exact in floating
point (a power-of-two multiply changes only the exponent field).  With
``R = diag(r) > 0`` (n x n, rows) and ``C = diag(c) > 0`` (m x m, columns)::

    B' = R B C        b' = R b        d' = C d
    recover  x = C x'          y = R y'

The four identities that make this a legitimate reformulation, each verified
by algebra AND asserted numerically at runtime (``_equil_identities``):

  1. FEASIBILITY.   B' x' = R B C x' = R (B x) and b' = R b, so
     ``B' x' >= b'`` iff ``R(Bx) >= R b`` iff ``Bx >= b``  (R > 0 diagonal).
  2. DUAL EQUALITY. B'' y' = (RBC)' y' = C B' R y' = C (B' y) with y = R y',
     and d' = C d, so ``B''y' = d'`` iff ``C(B'y) = C d`` iff ``B'y = d``
     (C > 0 diagonal, hence invertible).
  3. SIGN.          ``y' >= 0`` iff ``R y' >= 0`` iff ``y >= 0``  (R > 0).
  4. OBJECTIVE.     ``(Cd)' x' = d' C x' = d' (C x') = d' x``  (C diagonal, so
     C' = C).  The dual objective matches too: ``(Rb)' y' = b' (R y') = b' y``,
     so the duality gap is preserved exactly.

Only the MATRIX is equilibrated; ``b`` and ``d`` are then scaled by the factors
the matrix balancing produced.  No separate cost or right-hand-side scaling is
applied -- that would be a second, untested knob.

WHAT PRESOLVE ACTUALLY REMOVES (shape census, no timing)
--------------------------------------------------------
Measured on all 27 panel models before this harness was run.  The reduction is
wildly non-uniform, which is essential context for reading any aggregate::

    reduced to nothing   blend  (200x83 -> 0x0, kReducedToEmpty)
    heavy reduction      beaconfd 0.08 rows, sc50b 0.25, sc50a 0.28,
                         sc105 0.29, sc205 0.34, afiro 0.34, recipe 0.36
    barely touched       fit1d 0.998, grow7 0.995, adlittle 0.976,
                         israel 0.975, share2b 0.936, sctap1 0.927
    (fraction = reduced rows / original rows in the Bx >= b encoding)

So a panel-level speedup could be produced entirely by two or three models
whose programs presolve away, while the models that dominate the panel wall
(ship04s 0.735, fit1d 0.998, sctap1 0.927) keep almost every row.  Every row of
this harness records ``rows_before/after``, ``cols_before/after`` and
``presolve_reduced`` so that reading is available rather than assumed.

``blend`` reduces to a 0x0 program: presolve solves it outright and there is
nothing left to hand the Newton lane.  That branch is handled explicitly
(``solve_status == "EMPTY"``, zero-length vectors straight into postsolve) and
it certifies on the original data, but note what it means -- for that model the
``presolve`` and ``both`` cells measure HiGHS, not us.

METHOD
------
Cells adjacent per model, cell order rotated per (repeat, model) and reversed
on odd repeats, so no cell systematically occupies a warm or cold slot.  One
discarded warm-up per cell before the repeats.  Medians and min/max across
repeats plus PAIRED per-model deltas against ``none``.  Never a single run,
never min-of-N.  Machine state (load average + a fixed 600x600 dgemm, best of
3) bracketed around every model block.  These are ``bench_noise_floor.py`` /
``bench_composition.py`` conventions, reused, not reinvented.

WHAT IS MONKEYPATCHED (nothing is edited)
-----------------------------------------
Three runtime substitutions, all installed inside ``try/finally``, all restored
on every exit path, all applied IDENTICALLY to all four cells (including the
``none`` control) so no cell pays an overhead another cell does not:

  * ``exp23._certificate_detail`` -> guarded ``qr_cod`` route, and
    ``exp23.certificate_pair`` -> counting spy.  Both installed by
    ``bench_certificate_routes._run_one``, which is reused verbatim.
  * ``exp84_staged_newton.to_Bxgeb`` -> returns the already-transformed
    ``(B', b', d')``.  This is how the shipped harness is made to solve the
    transformed program without editing it.  A counter asserts it was called
    exactly once, and the terminal pair's shapes are asserted against
    ``B'.shape``, so a silently unpatched run -- which would return the
    ORIGINAL shapes -- cannot masquerade as a transformed one.
  * ``staged_newton.follow_staged`` -> capturing passthrough, so the terminal
    ``(x', y')`` can be lifted.  ``exp84.run`` keeps its result local and
    returns no pair; one dict assignment per solve is the cheapest honest way
    to get it.

Canonical invocation (thread env vars MUST be set before python starts):

    env PYTHONPYCACHEPREFIX=/tmp/fable_lp_pycache \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      ~/.venvs/claude/bin/python experiments/bench_transforms.py --repeats 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# bench_certificate_routes imports hybrid_certificate_factor BEFORE anything can
# be installed, which is what keeps the hybrid route's fallback pointing at the
# true incumbent.  Importing it first here preserves that ordering.
import bench_certificate_routes as bcr            # noqa: E402
import bench_noise_floor as bnf                   # noqa: E402
import quarantined_speedups as qs                 # noqa: E402
import exp23_path_primal_dual as exp23            # noqa: E402
import exp84_staged_newton as exp84               # noqa: E402
import staged_newton as sn                        # noqa: E402
import presolve_walker as pw                      # noqa: E402
from exp11_netlib_rank_certificate import to_Bxgeb  # noqa: E402

OUT_ROOT = ROOT / "records/transforms"

ROUTE = "qr_cod"
CANDIDATES = ("c1_const_cache", "c5_pchol_screen")

CELLS = ("none", "presolve", "equil", "both")
CELL_USES_PRESOLVE = {"none": False, "presolve": True,
                      "equil": False, "both": True}
CELL_USES_EQUIL = {"none": False, "presolve": False,
                   "equil": True, "both": True}

# The frozen originals, captured before any install can happen.
FROZEN_TO_BXGEB = exp84.to_Bxgeb
FROZEN_FOLLOW_STAGED = sn.follow_staged

# PROMOTION NOTE (see exp84_staged_newton's docstring).  The equilibration
# this harness measured is now the SHIPPED default and lives in
# ``exp84_staged_newton``; it is imported back here so there is exactly ONE
# implementation and every cell of this harness keeps computing what it
# computed.  The names below are the same objects, not copies.
_pow2 = exp84._pow2
_inf_norms = exp84._inf_norms
_ratio = exp84._ratio
equilibrate = exp84.equilibrate
_equil_identities = exp84.equil_identities

RUIZ_ROUNDS = exp84.DEFAULT_RUIZ_ROUNDS


# ------------------------------------------------------------------- the solve

_CAPTURE = {}


def _capturing_follow_staged(*args, **kwargs):
    """Passthrough that stashes the staged result so the pair can be lifted.

    One dict assignment per solve; no copy, no arithmetic.  ``exp84.run``
    returns a record with no ``x``/``y``, and the transformed cells need the
    terminal pair in order to map it back to the original space.
    """
    result = FROZEN_FOLLOW_STAGED(*args, **kwargs)
    _CAPTURE["result"] = result
    return result


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=float))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _solve_transformed(model, Bt, bt, dt, seconds, repeat, slot,
                       alloc_targets=()):
    """One solve of ``(Bt, bt, dt)`` under the current best configuration.

    ``alloc_targets`` names the ``quarantined_pivchol_alloc`` targets to
    install; the default EMPTY tuple is the pre-promotion allocation code path,
    which is what every cell of THIS harness ran when its numbers were taken.
    ``bench_pivchol_alloc`` passes a non-empty set for its treatment cell.

    The shipped harness ``exp84_staged_newton.run`` is reused unchanged; its
    ``to_Bxgeb`` lookup is redirected so that it receives the transformed
    program instead of re-reading the MPS file.  ``bench_certificate_routes.
    _run_one`` supplies the ``qr_cod`` install, the guard that raises if a
    route returns ok without the frozen ``certificate_pair`` having been
    called, and the post-run authoritative accuracy recomputation -- all of
    which apply to the TRANSFORMED program and are therefore intermediate
    diagnostics here, not the verdict.
    """
    calls = {"n": 0}

    def _redirected_to_Bxgeb(_path):
        calls["n"] += 1
        return Bt, bt, dt, None

    _CAPTURE.pop("result", None)
    exp84.to_Bxgeb = _redirected_to_Bxgeb
    sn.follow_staged = _capturing_follow_staged
    try:
        with qs.active(CANDIDATES):
            row = bcr._run_one(model, ROUTE, seconds, repeat, slot,
                               alloc_targets=alloc_targets)
    finally:
        exp84.to_Bxgeb = FROZEN_TO_BXGEB
        sn.follow_staged = FROZEN_FOLLOW_STAGED
    if calls["n"] != 1:
        raise AssertionError(
            "the redirected to_Bxgeb was called %d times, expected exactly 1 "
            "-- the solver may not have seen the transformed program"
            % calls["n"])
    result = _CAPTURE.pop("result", None)
    x = y = None
    if isinstance(result, dict):
        x = result.get("x")
        y = result.get("y")
    if x is not None:
        x = np.asarray(x, dtype=float)
    if y is not None:
        y = np.asarray(y, dtype=float)
    # Proof that the solver really saw the TRANSFORMED program: the terminal
    # pair it returned has the transformed program's shapes.  A silently
    # unpatched run would return the original shapes and be caught here.
    if x is not None and x.shape != (Bt.shape[1],):
        raise AssertionError("solved x has shape %s, transformed program has "
                             "%d columns" % (x.shape, Bt.shape[1]))
    if y is not None and y.shape != (Bt.shape[0],):
        raise AssertionError("solved y has shape %s, transformed program has "
                             "%d rows" % (y.shape, Bt.shape[0]))
    return row, x, y


# ------------------------------------------------------------------- the cells

_MPS_CACHE = {}


def _load(model):
    """Original raw ``(B, b, d)``; a fresh copy per cell, OUTSIDE the timing.

    Parsing is cached, but each cell gets its own arrays so that no in-place
    write by one cell can reach another, and so the copy cost is paid by every
    cell equally and outside every timed region.
    """
    if model not in _MPS_CACHE:
        B, b, d, _names = to_Bxgeb(ROOT / "netlib" / (model + ".mps"))
        _MPS_CACHE[model] = (np.asarray(B, dtype=float),
                             np.asarray(b, dtype=float),
                             np.asarray(d, dtype=float))
    B, b, d = _MPS_CACHE[model]
    return B.copy(), b.copy(), d.copy()


def _components(detail):
    return bcr._components(detail)


def _run_cell(model, cell, seconds, repeat, slot, ruiz_rounds, rng,
              check_identities=False):
    B, b, d = _load(model)                       # UNTIMED, outside everything
    n, m = B.shape
    row = {
        "repeat": repeat, "slot": slot, "cell": cell, "model": model,
        "route": ROUTE, "candidates": list(CANDIDATES),
        "uses_highs_presolve": CELL_USES_PRESOLVE[cell],
        "self_contained": not CELL_USES_PRESOLVE[cell],
        "rows_before": int(n), "cols_before": int(m),
        "rows_after": int(n), "cols_after": int(m),
        "presolve_reduced": False,
        "transform_seconds": 0.0, "inverse_seconds": 0.0,
        "solve_seconds": 0.0,
    }

    handle = book = None
    r = c = None
    end_to_end_start = time.perf_counter()

    # ---------------------------------------------------- forward transform
    transform_start = time.perf_counter()
    Bt, bt, dt = B, b, d
    presolve_failed = None
    if CELL_USES_PRESOLVE[cell]:
        try:
            handle, rlp, pstats = pw.presolve_program(B, b, d)
            Bt, bt, dt, book = pw.reduced_to_Bxgeb(rlp)
            row["presolve"] = pstats
            row["presolve_convert"] = book["stats"]
            row["reduced_offset"] = float(pstats["reduced_offset"])
        except Exception as error:               # fail closed, never silently
            presolve_failed = "%s: %s" % (type(error).__name__, error)
    if presolve_failed is None and CELL_USES_EQUIL[cell]:
        Bt, bt, dt, r, c, einfo = equilibrate(Bt, bt, dt, rounds=ruiz_rounds)
        row["equilibration"] = einfo
    row["transform_seconds"] = time.perf_counter() - transform_start

    if presolve_failed is not None:
        row["status"] = "TRANSFORM-FAILED"
        row["failure"] = presolve_failed
        row["wall_seconds"] = row["transform_seconds"]
        row["end_to_end_seconds"] = time.perf_counter() - end_to_end_start
        return row

    row["rows_after"] = int(Bt.shape[0])
    row["cols_after"] = int(Bt.shape[1])
    row["presolve_reduced"] = bool(CELL_USES_PRESOLVE[cell]
                                   and (Bt.shape[0] != n or Bt.shape[1] != m))
    row["row_reduction"] = (float(Bt.shape[0]) / n) if n else None
    row["col_reduction"] = (float(Bt.shape[1]) / m) if m else None

    # ----------------------------------------------------------- the solve
    empty = (Bt.shape[0] == 0 or Bt.shape[1] == 0)
    if empty:
        # Presolve finished the problem outright; there is nothing to solve.
        # Zero-length vectors go straight into postsolve.
        x_t = np.zeros(int(Bt.shape[1]))
        y_t = np.zeros(int(Bt.shape[0]))
        srow = None
        row["solve_status"] = "EMPTY"
        row["solve_seconds"] = 0.0
    else:
        srow, x_t, y_t = _solve_transformed(model, Bt, bt, dt, seconds,
                                            repeat, slot)
        row["solve_status"] = srow["status"]
        row["solve_seconds"] = float(srow["wall_seconds"])
        row["newton_iters_total"] = srow.get("newton_iters_total")
        row["stages_run"] = srow.get("stages_run")
        row["terminal_tests_run"] = srow.get("terminal_tests_run")
        row["accepted_stage"] = srow.get("accepted_stage")
        row["accepted_face_size"] = srow.get("accepted_face_size")
        row["staged_fallback"] = srow.get("staged_fallback")
        row["solve_route"] = srow.get("route")
        row["wall_split"] = srow.get("wall_split")
        row["installed_certificate_detail"] = srow.get(
            "installed_certificate_detail")
        row["installed_candidates"] = qs.installed_state()
        # What the cell actually ran under, read off the modules INSIDE the
        # timed block (``installed_candidates`` above is read after the block
        # and therefore describes the module defaults).
        row["kernel_flags_during_solve"] = (
            (srow.get("solver_configuration") or {}).get("kernel_flags"))
        # INTERMEDIATE DIAGNOSTIC ONLY -- the certificate the solver reached in
        # the TRANSFORMED space.  Never the verdict.
        row["transformed_certificate_max"] = srow.get(
            "certificate_max_authoritative")
        row["transformed_certificate_authority"] = srow.get(
            "certificate_max_authority")
        if x_t is None or y_t is None:
            row["status"] = "SOLVE-RETURNED-NO-PAIR"
            row["wall_seconds"] = (row["transform_seconds"]
                                   + row["solve_seconds"])
            row["end_to_end_seconds"] = time.perf_counter() - end_to_end_start
            return row

    # --------------------------------------------------- inverse transform
    inverse_start = time.perf_counter()
    lift_failed = None
    x0, y0 = x_t, y_t
    if CELL_USES_EQUIL[cell]:
        # Undo the INNER transform first: for `both`, equilibration was applied
        # to the reduced program, so its inverse precedes postsolve.
        x0 = c * x0
        y0 = r * y0
    if CELL_USES_PRESOLVE[cell]:
        try:
            hx, hz, hrv, hlam = pw.reduced_pair_to_highs(book, x0, y0)
            x0, y0, linfo = pw.postsolve_pair(handle, hx, hz, hrv, hlam)
            row["postsolve"] = linfo
        except Exception as error:
            lift_failed = "%s: %s" % (type(error).__name__, error)
    row["inverse_seconds"] = time.perf_counter() - inverse_start

    row["wall_seconds"] = (row["transform_seconds"] + row["solve_seconds"]
                           + row["inverse_seconds"])
    row["end_to_end_seconds"] = time.perf_counter() - end_to_end_start

    if lift_failed is not None:
        row["status"] = "LIFT-FAILED"
        row["failure"] = lift_failed
        return row

    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if x0.shape != (m,) or y0.shape != (n,):
        row["status"] = "LIFT-SHAPE-FAILED"
        row["failure"] = ("lifted pair shape %s/%s vs original %s/%s"
                          % (x0.shape, y0.shape, (m,), (n,)))
        return row

    # ------------------------------------ THE VERDICT: frozen gate, ORIGINAL
    # Fingerprint the lifted pair, then certify, then fingerprint again: the
    # pair that is certified must be BYTE-IDENTICAL to the one the inverse
    # transform produced.  Nothing may intervene between lift and gate.
    before = _digest(x0, y0)
    certify_start = time.perf_counter()
    ok, detail = bcr.FROZEN_CERTIFICATE_PAIR(B, b, d, x0, y0)
    row["certify_seconds"] = time.perf_counter() - certify_start
    after = _digest(x0, y0)
    row["lifted_pair_digest"] = before
    row["lifted_pair_unmodified_by_gate"] = bool(before == after)
    components = _components(detail)
    row["certificate"] = components
    row["certificate_max"] = components["max"]
    row["certificate_authority"] = ("frozen exp23.certificate_pair on the "
                                    "ORIGINAL (B, b, d)")
    row["status"] = "CERTIFIED" if ok else "GATE-FAILED"
    row["objective"] = float(d @ x0)
    row["lb"] = float(b @ y0)
    row["min_y"] = float(y0.min()) if y0.size else 0.0

    # The four identities on the ACTUAL program that was scaled.  Only the
    # `equil` cell scales the original data; for `both` the scaled program is
    # the reduced one, and the identity audit for that pair is meaningful only
    # against the reduced data, which is recorded here rather than re-derived.
    if check_identities and CELL_USES_EQUIL[cell] and not empty:
        if cell == "equil":
            row["equil_identities"] = _equil_identities(
                B, b, d, Bt, bt, dt, r, c, rng)
        else:
            row["equil_identities"] = {
                "skipped": "scaled program is the presolve-reduced one; see "
                           "the standalone identity audit"}
    return row


# --------------------------------------------------------------------- driver

def _rotate(names, repeat, model_index):
    """Rotate cell order per (repeat, model); reverse on odd repeats."""
    names = list(names)
    shift = (repeat + model_index) % len(names)
    ordered = names[shift:] + names[:shift]
    return ordered[::-1] if repeat % 2 else ordered


def _wall(rows, model, cell, repeat):
    for r in rows:
        if (r["model"] == model and r["cell"] == cell
                and r["repeat"] == repeat):
            return r.get("wall_seconds")
    return None


def _identity_audit(models, ruiz_rounds, seed=20260809):
    """Verify the four equilibration identities on every panel model.

    Done once, outside every timed region.  A transform whose identities do not
    hold on the actual data is not a reformulation and the campaign must not
    proceed on it.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for model in models:
        B, b, d = _load(model)
        Bs, bs, ds, r, c, info = equilibrate(B, b, d, rounds=ruiz_rounds)
        checks = _equil_identities(B, b, d, Bs, bs, ds, r, c, rng)
        out[model] = {"info": info, "identities": checks}
    return out


def main():
    parser = argparse.ArgumentParser(
        description="2x2 presolve x equilibration harness for the NATIVE lane; "
                    "the presolve cells call HiGHS and are NOT native-column "
                    "results")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--cells", nargs="*", default=list(CELLS),
                        choices=list(CELLS))
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--ruiz-rounds", type=int, default=RUIZ_ROUNDS)
    parser.add_argument("--warmup-model", default="afiro")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-identity-audit", action="store_true",
                        help="skip the pre-run check that the four "
                             "equilibration identities hold on every model")
    parser.add_argument("--per-run-identities", action="store_true",
                        help="additionally re-check the identities on every "
                             "equil row (outside the timed region)")
    args = parser.parse_args()

    missing = [k for k in bnf.THREAD_VARS if os.environ.get(k) != "1"]
    if missing:
        raise SystemExit(
            "refusing to run: %s must be set to 1 in the ENVIRONMENT before "
            "python starts" % ", ".join(missing))
    if "none" not in args.cells:
        raise SystemExit("refusing to run without the 'none' control cell")
    if bcr.hybrid._ORIGINAL is not bcr.FROZEN_CERTIFICATE_DETAIL:
        raise SystemExit("refusing to run: hybrid._ORIGINAL is not frozen")
    if exp23._certificate_detail is not bcr.FROZEN_CERTIFICATE_DETAIL:
        raise SystemExit("refusing to run: _certificate_detail pre-substituted")
    if exp84.to_Bxgeb is not FROZEN_TO_BXGEB:
        raise SystemExit("refusing to run: exp84.to_Bxgeb pre-substituted")
    if sn.follow_staged is not FROZEN_FOLLOW_STAGED:
        raise SystemExit("refusing to run: sn.follow_staged pre-substituted")

    models = args.models or list(bnf.MODELS)
    cells = list(args.cells)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.output_dir.resolve() if args.output_dir
               else OUT_ROOT / ("%s-pid%d" % (stamp, os.getpid())))
    if out_dir.exists():
        raise FileExistsError("refusing to reuse %s" % out_dir)
    out_dir.mkdir(parents=True)
    rows_path = out_dir / "runs.jsonl"
    summary_path = out_dir / "transforms-summary.json"

    rng = np.random.default_rng(20260809)

    identity_audit = None
    if not args.skip_identity_audit:
        identity_audit = _identity_audit(models, args.ruiz_rounds)
        bad = [k for k, v in identity_audit.items()
               if not v["info"]["roundtrip_B_exact"]
               or not v["info"]["roundtrip_b_exact"]
               or not v["info"]["roundtrip_d_exact"]
               or not v["info"]["factors_are_powers_of_two"]
               or not v["info"]["all_finite"]]
        if bad:
            raise SystemExit("equilibration is not exactly invertible on: %s"
                             % ", ".join(sorted(bad)))
        print(json.dumps({"identity_audit_models": len(identity_audit),
                          "all_roundtrips_bit_exact": True}), file=sys.stderr,
              flush=True)

    # Discarded warm-up, one per cell: the first solve in a process pays an
    # import/allocator cost no later solve pays.
    warmup = []
    if args.warmup_model and args.warmup_model != "none":
        for cell in cells:
            discarded = _run_cell(args.warmup_model, cell, args.seconds, -1, -1,
                                  args.ruiz_rounds, rng)
            warmup.append({"cell": cell, "model": args.warmup_model,
                           "wall_seconds": discarded.get("wall_seconds"),
                           "status": discarded.get("status"),
                           "discarded": True})
        print(json.dumps({"warmup": warmup}, default=str), file=sys.stderr,
              flush=True)

    rows, blocks = [], []
    with rows_path.open("x", encoding="utf-8") as stream:
        for repeat in range(args.repeats):
            for index, model in enumerate(models):
                before = bnf._machine_state()
                started = time.perf_counter()
                for slot, cell in enumerate(_rotate(cells, repeat, index)):
                    row = _run_cell(model, cell, args.seconds, repeat, slot,
                                    args.ruiz_rounds, rng,
                                    check_identities=args.per_run_identities)
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True,
                                            default=str) + "\n")
                    stream.flush()
                after = bnf._machine_state()
                blocks.append({"repeat": repeat, "model": model,
                               "elapsed": time.perf_counter() - started,
                               "machine_before": before,
                               "machine_after": after})
                # Every substitution must be gone between models.
                if exp23._certificate_detail is not \
                        bcr.FROZEN_CERTIFICATE_DETAIL:
                    raise SystemExit("route not restored after %s" % model)
                if exp23.certificate_pair is not bcr.FROZEN_CERTIFICATE_PAIR:
                    raise SystemExit("pair spy not restored after %s" % model)
                if exp84.to_Bxgeb is not FROZEN_TO_BXGEB:
                    raise SystemExit("to_Bxgeb not restored after %s" % model)
                if sn.follow_staged is not FROZEN_FOLLOW_STAGED:
                    raise SystemExit("follow_staged not restored after %s"
                                     % model)
                if qs.installed_state().get("active"):
                    raise SystemExit("candidates not restored after %s" % model)
            per = {cell: sum(r["wall_seconds"] for r in rows
                             if r["repeat"] == repeat and r["cell"] == cell
                             and r.get("wall_seconds") is not None)
                   for cell in cells}
            print(json.dumps({"repeat": repeat,
                              "panel": {k: round(v, 3)
                                        for k, v in per.items()}},
                             sort_keys=True), file=sys.stderr, flush=True)

    # ---------------------------------------------------------- the summary
    reps = list(range(args.repeats))

    def panel(cell, repeat):
        walls = [r["wall_seconds"] for r in rows
                 if r["cell"] == cell and r["repeat"] == repeat
                 and r.get("wall_seconds") is not None]
        return sum(walls)

    per_cell, deltas = {}, {}
    for cell in cells:
        per_cell[cell] = {
            "panel_wall": bnf._spread([panel(cell, r) for r in reps]),
            "transform_seconds": bnf._spread(
                [r["transform_seconds"] for r in rows if r["cell"] == cell]),
            "inverse_seconds": bnf._spread(
                [r["inverse_seconds"] for r in rows if r["cell"] == cell]),
            "solve_only_panel": bnf._spread(
                [sum(r["solve_seconds"] for r in rows
                     if r["cell"] == cell and r["repeat"] == rep)
                 for rep in reps]),
            # Panel TOTAL cost of the transform and its inverse, per repeat --
            # what presolve/equilibration charges the lane across 27 models.
            "transform_panel": bnf._spread(
                [sum(r["transform_seconds"] for r in rows
                     if r["cell"] == cell and r["repeat"] == rep)
                 for rep in reps]),
            "inverse_panel": bnf._spread(
                [sum(r["inverse_seconds"] for r in rows
                     if r["cell"] == cell and r["repeat"] == rep)
                 for rep in reps]),
            "certified": sum(r["cell"] == cell and r.get("status") == "CERTIFIED"
                             for r in rows),
            "runs": sum(r["cell"] == cell for r in rows),
            "statuses": sorted({str(r.get("status")) for r in rows
                                if r["cell"] == cell}),
        }
        if cell != "none":
            diffs = [panel(cell, r) - panel("none", r) for r in reps]
            wins = sum(1 for mo in models for rp in reps
                       if _wall(rows, mo, cell, rp) is not None
                       and _wall(rows, mo, "none", rp) is not None
                       and _wall(rows, mo, cell, rp)
                       < _wall(rows, mo, "none", rp))
            deltas[cell] = {"per_repeat": diffs,
                            "median": statistics.median(diffs) if diffs else None,
                            "wins_vs_none": wins,
                            "pairs": len(models) * len(reps)}

    per_model = {}
    for model in models:
        entry = {"cells": {}, "paired_delta_vs_none": {},
                 "presolve_shape": None}
        for cell in cells:
            mrows = [r for r in rows if r["model"] == model and r["cell"] == cell]
            entry["cells"][cell] = {
                "wall": bnf._spread([r["wall_seconds"] for r in mrows
                                     if r.get("wall_seconds") is not None]),
                "transform": bnf._spread([r["transform_seconds"]
                                          for r in mrows]),
                "inverse": bnf._spread([r["inverse_seconds"] for r in mrows]),
                "solve": bnf._spread([r["solve_seconds"] for r in mrows]),
                "statuses": sorted({str(r.get("status")) for r in mrows}),
                "certificate_max": bnf._spread(
                    [r["certificate_max"] for r in mrows
                     if r.get("certificate_max") is not None]),
                "rows_after": sorted({r["rows_after"] for r in mrows}),
                "cols_after": sorted({r["cols_after"] for r in mrows}),
            }
            if cell != "none":
                diffs = [(_wall(rows, model, cell, rp)
                          - _wall(rows, model, "none", rp))
                         for rp in reps
                         if _wall(rows, model, cell, rp) is not None
                         and _wall(rows, model, "none", rp) is not None]
                entry["paired_delta_vs_none"][cell] = {
                    **bnf._spread(diffs),
                    "wins": sum(1 for x in diffs if x < 0.0)}
        prow = next((r for r in rows if r["model"] == model
                     and r["cell"] == "presolve"), None)
        if prow is not None:
            entry["presolve_shape"] = {
                "rows_before": prow["rows_before"],
                "cols_before": prow["cols_before"],
                "rows_after": prow["rows_after"],
                "cols_after": prow["cols_after"],
                "reduced": prow["presolve_reduced"],
                "row_reduction": prow.get("row_reduction"),
                "col_reduction": prow.get("col_reduction"),
                "presolve_status": (prow.get("presolve") or {}).get(
                    "presolve_status"),
            }
        per_model[model] = entry

    guard_violation = any(
        r.get("certificate_max") is not None
        and r["certificate_max"] > exp23.TOL_FEAS
        and r.get("status") == "CERTIFIED" for r in rows)
    unmodified = all(r.get("lifted_pair_unmodified_by_gate", True)
                     for r in rows)

    summary = {
        "experiment": "2x2 presolve x equilibration for the native lane",
        "classification": "DIRECT MEASUREMENT; INTERLEAVED; PAIRED PER-MODEL "
                          "DELTAS; NO NORMALIZATION",
        "not_native_column": (
            "the 'presolve' and 'both' cells call HiGHS presolve/postsolve; "
            "they are a MEASUREMENT of whether presolve would help, NOT "
            "native-column results.  The standing project rule is that the "
            "native column uses no presolve.  'none' and 'equil' are "
            "self-contained."),
        "headline_definition": ("wall_seconds = transform_seconds + "
                                "solve_seconds + inverse_seconds; the "
                                "transform and its inverse are INSIDE the "
                                "timed region, as HiGHS's presolve time is "
                                "inside its own"),
        "verdict_authority": ("exp23.certificate_pair on the ORIGINAL raw "
                              "(B, b, d) with the lifted/unscaled pair; the "
                              "transformed-space certificate is recorded as "
                              "transformed_certificate_max and never sets the "
                              "verdict"),
        "configuration": {"route": ROUTE, "candidates": list(CANDIDATES)},
        "cells": {cell: {"presolve": CELL_USES_PRESOLVE[cell],
                         "equilibration": CELL_USES_EQUIL[cell]}
                  for cell in cells},
        "per_cell": per_cell,
        "paired_delta_vs_none": deltas,
        "per_model": per_model,
        "correctness": {
            "runs": len(rows),
            "certified": sum(r.get("status") == "CERTIFIED" for r in rows),
            "all_certified": all(r.get("status") == "CERTIFIED" for r in rows),
            "certified_by_cell": {c: per_cell[c]["certified"] for c in cells},
            "worst_certificate_max_by_cell": {
                c: max((r["certificate_max"] for r in rows
                        if r["cell"] == c
                        and r.get("certificate_max") is not None),
                       default=None) for c in cells},
            "tol_feas": exp23.TOL_FEAS,
            "guard_violation": guard_violation,
            "lifted_pair_is_the_certified_pair": unmodified,
        },
        "equilibration_identity_audit": identity_audit,
        "ruiz_rounds": args.ruiz_rounds,
        "blocks": blocks,
        "warmup_discarded": warmup,
        "models": models,
        "repeats": args.repeats,
        "provenance": bnf._provenance(),
        "harness_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "presolve_walker_sha256": hashlib.sha256(
            (HERE / "presolve_walker.py").read_bytes()).hexdigest(),
        "candidate_source_sha": qs.source_sha(),
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2,
                                       default=str) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("per_cell", "paired_delta_vs_none", "correctness")},
                     sort_keys=True, indent=2, default=str)[:4000])
    print("rows:    " + str(rows_path))
    print("summary: " + str(summary_path))


if __name__ == "__main__":
    main()
