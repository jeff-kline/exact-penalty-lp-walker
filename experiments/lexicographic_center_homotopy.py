"""Factor-preserving global perturbation for fixed-t Mangasarian face repair."""

from __future__ import annotations

import signal

import numpy as np

from B_homotopy_face import StageTimeout, certify_face
from path_piece_skeleton import PathPieceSkeletonFactor
from primal_face_pricing import priced_settle_escape


def _timeout(_signum, _frame):
    raise StageTimeout("lexicographic-center stage ceiling reached")


def priority_direction(b):
    """Return one fixed, injectively ranked global perturbation direction."""
    b = np.asarray(b, dtype=float)
    rank = (np.arange(b.size, dtype=float) + 1.0) / max(1, b.size)
    signs = np.where(np.arange(b.size) % 2 == 0, 1.0, -1.0)
    return signs * rank * (1.0 + np.abs(b))


def follow_center_ladder(
        B, b, d, W, t, deltas=(1e-6, 1e-8, 0.0), candidates=12,
        rounds=30, tol=1e-7, seconds_per_stage=8, return_state=False):
    """Follow one global center perturbation and accept only exact return."""
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    W = np.asarray(W, dtype=bool).copy()
    sigma = priority_direction(b)
    initial_W = W.copy()
    ever_changed = False
    records = []
    # B and d never change, so this factor and its row updates survive every
    # delta stage.  Only the stored projection-center vector is replaced.
    factor = PathPieceSkeletonFactor(B, b, d)
    use_alarm = seconds_per_stage is not None
    old_handler = (signal.signal(signal.SIGALRM, _timeout)
                   if use_alarm else None)
    try:
        for delta in deltas:
            bd = b + float(delta) * sigma
            factor.b = bd
            diagnostics = {}
            stats = {}
            if use_alarm:
                signal.alarm(int(seconds_per_stage))
            try:
                settled_W, coefficients, settled = priced_settle_escape(
                    B, bd, d, W, t, piece_solver=factor.solve,
                    max_candidates=candidates, rounds=rounds, tol=tol,
                    diagnostics=diagnostics, stats=stats)
            except StageTimeout as error:
                records.append({
                    "delta": float(delta), "status": "RESOURCE-LIMIT",
                    "detail": str(error),
                    "support_before": int(W.sum()),
                    "factor": factor.diagnostics(),
                })
                return {
                    "status": "RESOURCE-LIMIT", "records": records,
                    "returned_to_original": False,
                    "working_set_changed": ever_changed,
                }
            finally:
                if use_alarm:
                    signal.alarm(0)
            changes = int(np.sum(settled_W != W))
            ever_changed = ever_changed or changes > 0
            record = {
                "delta": float(delta), "settled": bool(settled),
                "diagnostics": diagnostics, "stats": stats,
                "support_before": int(W.sum()),
                "support_after": int(settled_W.sum()),
                "working_set_changes": changes,
                "factor": factor.diagnostics(),
            }
            if not settled or coefficients is None:
                record["status"] = "FAILED"
                records.append(record)
                return {
                    "status": "FAILED", "records": records,
                    "returned_to_original": False,
                    "working_set_changed": ever_changed,
                }
            certified, certificate = certify_face(
                B, bd, d, settled_W, t, coefficients, tol=tol)
            record["certificate"] = certificate
            record["status"] = "CERTIFIED" if certified else "FAILED"
            records.append(record)
            W = settled_W
            if not certified:
                return {
                    "status": "FAILED", "records": records,
                    "returned_to_original": False,
                    "working_set_changed": ever_changed,
                }
        returned = bool(float(deltas[-1]) == 0.0)
        result = {
            "status": "CERTIFIED", "records": records,
            "returned_to_original": returned,
            "working_set_changed": ever_changed,
            "final_diff_from_initial": int(np.sum(W != initial_W)),
            "final_support": int(W.sum()), "t": float(t),
            "factor": factor.diagnostics(),
        }
        if return_state:
            result["working_set"] = W.copy()
            result["coefficients"] = tuple(
                np.asarray(value, dtype=float).copy()
                for value in coefficients)
        return result
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
